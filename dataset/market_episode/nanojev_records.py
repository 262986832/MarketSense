"""NanoJev 训练记录生成：记录映射 + 状态序列化 + split 分配 + 原子落盘。

**记录契约**（对齐第三方 ``NanoJev/scripts/train_pipeline_decisions.py`` 的
``validate_training_row`` / ``read_training_records``；NanoJev 全程只读）：

* ``id`` = ``state_id`` = ``{segment_id}:{决策 K 线片段内序号}``（全局唯一，天然不跨 split）；
* ``family_id`` = ``metadata.source_group_id`` = ``segment_id``（一个片段 = 一个 episode，
  直接复用 NanoJev 的 state/source_group 跨 split 泄漏检查）；
* ``split`` = 片段的 ``split_role``；
* ``state`` = 确定性文本序列化的**相对比值**状态（决策 K 线完整数据的比值表达 +
  成交量/持仓量归一化 + 当前仓位 + 账户回撤幅度）；
* ``questions`` 恰好一个 choice 题 ``next_action``：空仓
  ``{open_long, open_short, stay_flat}`` / 持仓 ``{close, hold, reverse}``；
* ``gold`` = 规则真值动作 ID + ``gold_label_kind: "deterministic_truth"``
  （首轮不发 ``gold_probs``：NanoJev 校验器允许硬 gold 无概率，trainer 自动派生 one-hot）。

**状态序列化模板（实现阶段冻结项 2，本轮冻结；变更需走新版本标记）**

```text
marketsense.episode_state.v1
bar=<片段内 0 基序号>
px_ratio: o=<..> h=<..> l=<..> c=<..>
vol_ratio: v=<..> oi_open=<..> oi_close=<..>
position: flat | long entry=<..> stop=<..> | short entry=<..> stop=<..>
drawdown: <..>
```

* 比值分母 = **片段首根**（价格用首根开盘价，量/持仓量用首根同名列），小数位固定
  （默认 6）；分母 ≤ 0 时写 ``na``（不产生 ``inf``/绝对数）；
* **无历史窗口**：状态只含决策 K 线单根 + 仓位 + 回撤，绝对价格不进入模型输入；
* 输出按 ``run`` 目录隔离（``<output_dir>/<run_id>``，``run_id`` 由输入指纹确定性派生，
  同输入同目录同字节，不覆盖其它输入的产物）。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dataset.errors import DatasetError
from dataset.storage import file_sha256

from dataset.market_episode.audit import (
    build_audit_payload,
    check_audit_consistency,
    check_records,
    check_state_leakage,
)
from dataset.market_episode.labels import (
    ACTION_CLOSE,
    ACTION_HOLD,
    ACTION_OPEN_LONG,
    ACTION_OPEN_SHORT,
    ACTION_REVERSE,
    ACTION_STAY_FLAT,
    DecisionPoint,
    SegmentOutcome,
    evaluate_segment,
)
from dataset.market_episode.replay import (
    Bar,
    PositionSnapshot,
    format_ratio_value,
    load_segment_bars,
    px_ratio_line,
    vol_ratio_line,
)
from dataset.market_episode.segments import (
    SPLIT_ROLES,
    EpisodeParams,
    Segment,
    validate_segments,
)

#: 状态文本的 schema 版本标记（格式演进必须换标记）
STATE_SCHEMA = "marketsense.episode_state.v1"
#: 记录中的 choice 题目 ID
QUESTION_ID = "next_action"
#: 题目说明（确定性固定文案）
QUESTION_INSTRUCTIONS = "选择下一分钟要执行的动作（gold 为确定性规则真值）。"
#: 空仓候选（动作集合依仓位而变：模型必须知晓仓位）
FLAT_CRITERIA: Mapping[str, str] = {
    ACTION_OPEN_LONG: "开多（买入开仓，按决策 K 线最高 + 1 tick 成交）",
    ACTION_OPEN_SHORT: "开空（卖出开仓，按决策 K 线最低 − 1 tick 成交）",
    ACTION_STAY_FLAT: "继续空仓",
}
#: 持仓候选
HELD_CRITERIA: Mapping[str, str] = {
    ACTION_CLOSE: "平仓（多头按决策 K 线最低 − 1 tick 卖出，空头对称）",
    ACTION_HOLD: "继续持有",
    ACTION_REVERSE: "反手（同一决策点先平旧仓、再开反向 1 手）",
}
#: gold 的依据类型（冻结首轮形式）
GOLD_LABEL_KIND = "deterministic_truth"
#: 审计文件名
AUDIT_FILENAME = "audit.json"


@dataclass(frozen=True)
class GenerationResult:
    """一次生成的产物信息（供 CLI 打印与测试断言）。"""

    run_id: str
    run_dir: Path
    record_counts: Mapping[str, int]
    output_sha256: Mapping[str, str]
    audit: Mapping[str, Any]


def render_state(
    *,
    bar: Bar,
    reference_bar: Bar,
    position: PositionSnapshot | None,
    drawdown: float,
    price_precision: int,
) -> str:
    """确定性状态文本（模板见模块 docstring）。

    只做「决策 K 线单根 + 仓位 + 回撤」的序列化：函数签名决定它无法访问决策 K 线
    之后的任何 bar（无未来数据泄漏在构造层面成立）。
    """
    if position is None:
        position_line = "position: flat"
    else:
        position_line = (
            f"position: {position.direction} entry={format_ratio_value(position.entry_ratio, price_precision)}"
            f" stop={format_ratio_value(position.stop_ratio, price_precision)}"
        )
    return "\n".join(
        (
            STATE_SCHEMA,
            f"bar={bar.index}",
            px_ratio_line(bar, reference_bar, price_precision),
            vol_ratio_line(bar, reference_bar, price_precision),
            position_line,
            f"drawdown: {format_ratio_value(drawdown, price_precision)}",
        )
    )


def build_record(
    *,
    segment: Segment,
    bars: tuple[Bar, ...],
    point: DecisionPoint,
    price_precision: int,
) -> dict[str, Any]:
    """把一个入选决策点映射为 NanoJev 训练记录。"""
    bar = bars[point.bar_index]
    state = render_state(
        bar=bar,
        reference_bar=bars[0],
        position=point.position,
        drawdown=point.drawdown,
        price_precision=price_precision,
    )
    criteria = FLAT_CRITERIA if point.position is None else HELD_CRITERIA
    if point.action not in criteria:
        raise DatasetError(
            f"动作 {point.action!r} 与仓位候选集不匹配（{(segment.segment_id, point.bar_index)}）"
        )
    record_id = f"{segment.segment_id}:{point.bar_index}"
    return {
        "id": record_id,
        "state_id": record_id,
        "family_id": segment.segment_id,
        "split": segment.split_role,
        "state": state,
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": QUESTION_INSTRUCTIONS,
                "criteria": dict(criteria),
            }
        },
        "gold": {QUESTION_ID: point.action},
        "gold_label_kind": {QUESTION_ID: GOLD_LABEL_KIND},
        "metadata": {
            "segment_id": segment.segment_id,
            "source_group_id": segment.segment_id,
            "split_role": segment.split_role,
            "bar_index": point.bar_index,
        },
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dump_json(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def _write_files_atomically(run_dir: Path, files: Mapping[str, str]) -> dict[str, str]:
    """先写同父目录临时目录、再逐个 ``os.replace`` 落位（失败不留半成品）。"""
    parent = run_dir.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DatasetError(f"输出目录不可创建: {parent} ({type(exc).__name__}: {exc})") from None
    stage = Path(tempfile.mkdtemp(dir=parent, prefix=f".{run_dir.name}."))
    try:
        for name, text in files.items():
            (stage / name).write_text(text, encoding="utf-8", newline="")
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DatasetError(
                f"输出目录不可创建: {run_dir} ({type(exc).__name__}: {exc})"
            ) from None
        digests: dict[str, str] = {}
        for name in sorted(files):
            os.replace(stage / name, run_dir / name)
            digests[name] = file_sha256(run_dir / name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return digests


def generate_dataset(
    segments: tuple[Segment, ...] | list[Segment],
    *,
    symbols: Mapping[str, float],
    data_dir: str | Path,
    params: EpisodeParams,
    output_dir: str | Path,
) -> GenerationResult:
    """确定性地生成按 split 的 NanoJev JSONL 训练集与审计文件。

    流程：清单/数据校验 → 逐片段回放 + 标签 → 记录映射 → 自检（结构、隔离、泄漏、
    计数一致）→ 原子落盘到 ``<output_dir>/<run_id>``。

    :raises DatasetError: 校验失败或自检不通过（不写出任何产物）
    """
    segments = tuple(segments)
    validate_segments(segments, data_dir=data_dir, symbols=symbols)

    outcomes: dict[str, SegmentOutcome] = {}
    bars_by_segment: dict[str, tuple[Bar, ...]] = {}
    records_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ROLES}

    for segment in segments:
        bars, source_version = load_segment_bars(segment, data_dir=data_dir)
        outcome = evaluate_segment(
            segment,
            bars,
            tick_size=float(symbols[segment.symbol]),
            params=params,
            source_data_version=source_version,
        )
        outcomes[segment.segment_id] = outcome
        bars_by_segment[segment.segment_id] = bars
        for point in outcome.selected:
            records_by_split[segment.split_role].append(
                build_record(
                    segment=segment,
                    bars=bars,
                    point=point,
                    price_precision=params.price_precision,
                )
            )

    all_records = [
        record for split in SPLIT_ROLES for record in records_by_split[split]
    ]
    check_records(all_records)
    check_state_leakage(
        all_records, bars_by_segment=bars_by_segment, price_precision=params.price_precision
    )

    segments_text = _dump_json([segment.canonical() for segment in segments])
    symbols_text = _dump_json({symbol: float(value) for symbol, value in sorted(symbols.items())})
    params_text = _dump_json(params.as_dict())
    input_hashes = {
        "segment_count": len(segments),
        "segments_sha256": _sha256_text(segments_text),
        "symbols_sha256": _sha256_text(symbols_text),
        "params_sha256": _sha256_text(params_text),
    }
    run_id = "run-" + _sha256_text(
        "|".join((segments_text, symbols_text, params_text))
    )[:12]

    split_texts: dict[str, str] = {}
    for split in SPLIT_ROLES:
        rows = records_by_split[split]
        split_texts[f"{split}.jsonl"] = "".join(
            _dump_json(record) + "\n" for record in rows
        )
    split_digests = {name: _sha256_text(text) for name, text in split_texts.items()}

    payload = build_audit_payload(
        run_id=run_id,
        params=params,
        segments=segments,
        symbols=symbols,
        outcomes=outcomes,
        records_by_split=records_by_split,
        split_digests=split_digests,
        input_hashes=input_hashes,
    )
    check_audit_consistency(payload, records_by_split)

    files: dict[str, str] = dict(split_texts)
    files[AUDIT_FILENAME] = _dump_json(payload, indent=2) + "\n"

    run_dir = Path(output_dir) / run_id
    digests = _write_files_atomically(run_dir, files)
    return GenerationResult(
        run_id=run_id,
        run_dir=run_dir,
        record_counts={split: len(records_by_split[split]) for split in SPLIT_ROLES},
        output_sha256=digests,
        audit=payload,
    )


__all__ = [
    "AUDIT_FILENAME",
    "FLAT_CRITERIA",
    "GOLD_LABEL_KIND",
    "HELD_CRITERIA",
    "QUESTION_ID",
    "QUESTION_INSTRUCTIONS",
    "STATE_SCHEMA",
    "GenerationResult",
    "build_record",
    "generate_dataset",
    "render_state",
]

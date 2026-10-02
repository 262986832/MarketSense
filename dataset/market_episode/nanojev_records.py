"""NanoJev 训练记录生成：记录映射 + 状态序列化 + split 分配 + 原子落盘。

**记录契约**（对齐第三方 ``NanoJev/scripts/train_pipeline_decisions.py`` 的
``validate_training_row`` / ``read_training_records``；NanoJev 全程只读）：

* ``id`` = ``state_id`` = ``{segment_id}:{决策 K 线片段内序号}``（全局唯一，天然不跨 split）；
* ``family_id`` = ``metadata.source_group_id`` = ``segment_id``（一个片段 = 一个 episode，
  直接复用 NanoJev 的 state/source_group 跨 split 泄漏检查）；
* ``split`` = 片段的 ``split_role``；
* ``state`` = 确定性文本序列化的**相对比值**状态（v4 六部分：账户（仓位 + 回撤）/
  联动（量/持仓量）/ 日线（上一交易日高/低/收）/ 日内（``bar=`` 序号 + 今日高/低）/
  现价（决策 K 线 OHLC）/ 盘口（na 占位））；
* ``questions`` 恰好一个 choice 题 ``next_action``：空仓
  ``{open_long, open_short, stay_flat}`` / 持仓 ``{close, hold, reverse}``；
  候选文案只留动作语义（买入开仓/卖出开仓/继续空仓/平仓/继续持有/反手），
  成交价位由执行程序与滑点决定，不进模型输入（2026-10-02 用户拍板精简）；
* ``gold`` = 规则真值动作 ID + ``gold_label_kind: "deterministic_truth"``
  （首轮不发 ``gold_probs``：NanoJev 校验器允许硬 gold 无概率，trainer 自动派生 one-hot）。

**状态序列化模板（实现阶段冻结项 2；v2 起新增盘面状态行，2026-10-02 用户拍板方案 A；
v3 起行间换行符前后加空格，转义后的 JSON 文本更易读，同日拍板；v4 起行重排为宏观→微观
六部分（账户/联动/日线/日内/现价/盘口），持仓值中文化（空仓/持多/持空）+ 盘口 ``na``
占位，数值语义与 v3 逐项等价，同日用户拍板）**

```text
marketsense.episode_state.v4
账户: 持仓=空仓 回撤=<..>
联动: v=<..> oi_open=<..> oi_close=<..>
日线: prev_h=<..> prev_l=<..> prev_c=<..>
日内: bar=<片段内 0 基序号> today_h=<..> today_l=<..>
现价: o=<..> h=<..> l=<..> c=<..>
盘口: na
```

（行与行之间用 ``" \n "`` 连接，即换行符前后各一个空格。持仓非空时账户行为：
``账户: 持仓=持多|持空 entry=<..> stop=<..> 回撤=<..>``。）

* 比值分母 = **片段首根**（价格用首根开盘价，量/持仓量用首根同名列），小数位固定
  （默认 6）；分母 ≤ 0 时写 ``na``（不产生 ``inf``/绝对数）；
* **无历史窗口**：状态只含决策 K 线单根 + 仓位 + 回撤 + 盘面状态（v4 拆「日线」「日内」
  两行），绝对价格不进入模型输入；「盘口」行为 ``na`` 常量占位（真实 bid/ask 未接，非目标）；
* **账户行**（v4）：持仓值中文标签（空仓/持多/持空，未知方向抛 ``DatasetError`` 不静默）；
  持仓非空时 ``entry/stop`` 在 ``回撤`` 前（沿用 v3 的 position→drawdown 相对顺序，仅合并为一行）；
* **日线行**（原 v2 board_state 行 prev 部分，v4 拆出；与 ``dataset/board_state.py`` 同口径）：
  ``prev_h/prev_l/prev_c`` = **上一交易日**日线高/低/收（来源 ``{symbol}_1d.csv``，
  取日线文件中严格早于片段交易日的最后一行）÷ 片段首根开盘价；
* **日内行**（原 v2 board_state 行 today 部分 + ``bar=`` 序号，v4 拆出）：
  - ``today_h/today_l`` = 片段首根至决策 K 线（**含**）的 1m 高/低**累计极值** ÷ 片段首根开盘价
    （State(T) 只用 ≤ 决策 K 线的数据，无未来泄漏）；
  - 分母与现价行一致（全交易日片段下片段首根 = 交易日窗口首根 = 今日开盘，
    见 ``artifacts/nanojev-integration-alignment/01-requirement/requirement-report.md``）；
  - 今日开盘价不写（它是分母本身，比值恒为 1.000000，写入是纯冗余 token）；
  - 片段无上一交易日日线 → **跳过该片段**（不产出记录），记入审计与 stderr 告警（不静默）。
* 输出按 ``run`` 目录隔离（``<output_dir>/<run_id>``，``run_id`` 由输入指纹确定性派生，
  同输入同目录同字节，不覆盖其它输入的产物）。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from dataset.errors import DatasetError
from dataset.storage import file_sha256, load_ohlcv

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
    LONG,
    PositionSnapshot,
    SHORT,
    format_ratio_value,
    load_segment_bars,
    px_ratio_line,
    ratio_or_none,
    vol_ratio_line,
)
from dataset.market_episode.segments import (
    SPLIT_ROLES,
    EpisodeParams,
    Segment,
    validate_segments,
)

#: 状态文本的 schema 版本标记（格式演进必须换标记；v2 新增盘面状态行；v3 行间换行符
#: 前后加空格——转义后的 JSON 文本更易读；v4 行重排为宏观→微观六部分（账户/联动/日线/
#: 日内/现价/盘口）+ 持仓值中文化 + 盘口 na 占位，数值语义与 v3 逐项等价，2026-10-02 用户拍板）
STATE_SCHEMA = "marketsense.episode_state.v4"
#: questions 文本的 schema 版本标记（首次建立；候选文案演进必须换标记）
QUESTION_SCHEMA = "marketsense.episode_question.v1"
#: 记录中的 choice 题目 ID
QUESTION_ID = "next_action"
#: 题目说明（确定性固定文案）
QUESTION_INSTRUCTIONS = "选择下一分钟要执行的动作（gold 为确定性规则真值）。"
#: 空仓候选（动作集合依仓位而变：模型必须知晓仓位）。
#: 文案只留动作语义；成交价位由执行程序与滑点决定（2026-10-02 用户拍板精简）
FLAT_CRITERIA: Mapping[str, str] = {
    ACTION_OPEN_LONG: "买入开仓",
    ACTION_OPEN_SHORT: "卖出开仓",
    ACTION_STAY_FLAT: "继续空仓",
}
#: 持仓候选（文案同上：只留动作语义，价位/执行细节不进模型输入）
HELD_CRITERIA: Mapping[str, str] = {
    ACTION_CLOSE: "平仓",
    ACTION_HOLD: "继续持有",
    ACTION_REVERSE: "反手",
}
#: 账户行「持仓=」值映射（v4：flat（代码中为 ``None``）→空仓、long→持多、short→持空；
#: 未知方向抛 ``DatasetError``，保持确定性不静默）
POSITION_LABELS: Mapping[str | None, str] = {
    None: "空仓",
    LONG: "持多",
    SHORT: "持空",
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
    board_state_skipped: Mapping[str, str]


@dataclass(frozen=True)
class BoardStateValues:
    """盘面状态绝对值（序列化时再除以片段首根开盘价）。

    * ``prev_day_*``：上一交易日日线高/低/收（来源 ``{symbol}_1d.csv``，绝对价格层）；
    * ``today_*``：片段首根至决策 K 线（含）的 1m 高/低累计极值（State(T) 只用 ≤ 决策 K 线数据）。
    """

    prev_day_high: float
    prev_day_low: float
    prev_day_close: float
    today_high: float
    today_low: float


def _daily_line(
    board_state: BoardStateValues,
    reference_bar: Bar,
    price_precision: int,
) -> str:
    """日线行：上一交易日高/低/收的比值（分母 = 片段首根开盘价；分母 ≤ 0 时逐值写 ``na``）。"""
    return (
        "日线: "
        f"prev_h={format_ratio_value(ratio_or_none(board_state.prev_day_high, reference_bar.open), price_precision)}"
        f" prev_l={format_ratio_value(ratio_or_none(board_state.prev_day_low, reference_bar.open), price_precision)}"
        f" prev_c={format_ratio_value(ratio_or_none(board_state.prev_day_close, reference_bar.open), price_precision)}"
    )


def _intraday_line(
    bar: Bar,
    board_state: BoardStateValues,
    reference_bar: Bar,
    price_precision: int,
) -> str:
    """日内行：片段内 0 基序号 + 今日高/低累计极值的比值（State(T) 只用 ≤ 决策 K 线的数据）。"""
    return (
        f"日内: bar={bar.index}"
        f" today_h={format_ratio_value(ratio_or_none(board_state.today_high, reference_bar.open), price_precision)}"
        f" today_l={format_ratio_value(ratio_or_none(board_state.today_low, reference_bar.open), price_precision)}"
    )


def render_state(
    *,
    bar: Bar,
    reference_bar: Bar,
    position: PositionSnapshot | None,
    drawdown: float,
    price_precision: int,
    board_state: BoardStateValues,
) -> str:
    """确定性状态文本（v4 六部分：账户/联动/日线/日内/现价/盘口；模板见模块 docstring）。

    只做「决策 K 线单根 + 仓位 + 回撤 + 盘面状态」的序列化：函数签名决定它无法访问
    决策 K 线之后的任何 bar（``board_state`` 的今日值由调用方只用 ≤ 决策 K 线的数据算好
    传入，无未来数据泄漏在构造层面成立）。
    """
    if position is None:
        held_text = POSITION_LABELS[None]
    else:
        label = POSITION_LABELS.get(position.direction)
        if label is None:
            raise DatasetError(f"未知持仓方向，无法序列化账户行: {position.direction!r}")
        held_text = (
            f"{label} entry={format_ratio_value(position.entry_ratio, price_precision)}"
            f" stop={format_ratio_value(position.stop_ratio, price_precision)}"
        )
    return " \n ".join(
        (
            STATE_SCHEMA,
            f"账户: 持仓={held_text} 回撤={format_ratio_value(drawdown, price_precision)}",
            vol_ratio_line(bar, reference_bar, price_precision),
            _daily_line(board_state, reference_bar, price_precision),
            _intraday_line(bar, board_state, reference_bar, price_precision),
            px_ratio_line(bar, reference_bar, price_precision),
            "盘口: na",
        )
    )


def build_record(
    *,
    segment: Segment,
    bars: tuple[Bar, ...],
    point: DecisionPoint,
    price_precision: int,
    prev_day_ohlc: tuple[float, float, float],
) -> dict[str, Any]:
    """把一个入选决策点映射为 NanoJev 训练记录。

    ``prev_day_ohlc`` = 上一交易日日线 (高, 低, 收)（来源 1d 文件，绝对价格层）；
    ``today_high/today_low`` 只由 ``bars[: point.bar_index + 1]``（≤ 决策 K 线）累计，
    无未来数据泄漏。
    """
    bar = bars[point.bar_index]
    prefix = bars[: point.bar_index + 1]
    state = render_state(
        bar=bar,
        reference_bar=bars[0],
        position=point.position,
        drawdown=point.drawdown,
        price_precision=price_precision,
        board_state=BoardStateValues(
            prev_day_high=prev_day_ohlc[0],
            prev_day_low=prev_day_ohlc[1],
            prev_day_close=prev_day_ohlc[2],
            today_high=max(item.high for item in prefix),
            today_low=min(item.low for item in prefix),
        ),
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


def _load_daily_ohlc(data_dir: str | Path, symbol: str) -> Any:
    """读 ``{symbol}_1d.csv`` 返回 LoadedOHLCV（日线文件逐行一个交易日，升序）。

    :raises DataLoadError: 1d 文件缺失/损坏（硬错误，不静默跳过）
    """
    return load_ohlcv(symbol, "1d", data_dir=data_dir)


def _daily_rows(loaded: Any) -> tuple[tuple[Any, float, float, float], ...]:
    """LoadedOHLCV → 按文件序的 (交易日, 高, 低, 收) 元组（交易日 = 日线 timestamp 日历日）。"""
    return tuple(
        (pd.Timestamp(ts).date(), float(high), float(low), float(close))
        for ts, high, low, close in zip(
            loaded.df["timestamp"], loaded.df["high"], loaded.df["low"], loaded.df["close"]
        )
    )


def _prev_daily_ohlc(
    daily_rows: tuple[tuple[Any, float, float, float], ...],
    trade_date: Any,
) -> tuple[float, float, float] | None:
    """上一交易日日线 (高, 低, 收)：日线文件中严格早于 ``trade_date`` 的最后一行。

    无前日（数据起点）或日线值非有限 → 返回 ``None``（调用方跳过该片段，不静默）。
    """
    prev: tuple[float, float, float] | None = None
    for row_date, high, low, close in daily_rows:
        if row_date >= trade_date:
            break
        prev = (high, low, close)
    if prev is not None and not all(math.isfinite(value) for value in prev):
        return None
    return prev


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

    v2 起：每个片段需在 ``{symbol}_1d.csv`` 中找到严格早于片段交易日的上一交易日
    日线（供 board_state 行）；找不到的片段被跳过（记入审计与结果，不静默），
    全部片段被跳过则硬错误（不写出任何产物）。

    :raises DatasetError: 校验失败或自检不通过（不写出任何产物）
    """
    segments = tuple(segments)
    validate_segments(segments, data_dir=data_dir, symbols=symbols)

    outcomes: dict[str, SegmentOutcome] = {}
    bars_by_segment: dict[str, tuple[Bar, ...]] = {}
    records_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ROLES}
    daily_loaded_by_symbol: dict[str, Any] = {}
    prev_daily_by_segment: dict[str, tuple[float, float, float]] = {}
    board_state_skipped: dict[str, str] = {}

    for segment in segments:
        loaded = daily_loaded_by_symbol.get(segment.symbol)
        if loaded is None:
            loaded = _load_daily_ohlc(data_dir, segment.symbol)
            daily_loaded_by_symbol[segment.symbol] = loaded
        # 片段交易日 = 片段结束时间的日历日（窗口止于 14:59；清单与测试夹具均满足）
        prev = _prev_daily_ohlc(_daily_rows(loaded), segment.end.date())
        if prev is None:
            board_state_skipped[segment.segment_id] = (
                f"trade_date={segment.end.date()} 无上一交易日日线（或值非有限，1d 文件）"
            )
            continue
        prev_daily_by_segment[segment.segment_id] = prev
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
                    prev_day_ohlc=prev,
                )
            )

    if not prev_daily_by_segment:
        raise DatasetError(
            "所有片段都缺少上一交易日日线（1d 文件），无法生成 board_state 状态；"
            f"跳过明细: {dict(sorted(board_state_skipped.items()))}"
        )

    all_records = [
        record for split in SPLIT_ROLES for record in records_by_split[split]
    ]
    check_records(all_records)
    check_state_leakage(
        all_records,
        bars_by_segment=bars_by_segment,
        price_precision=params.price_precision,
        prev_daily_by_segment=prev_daily_by_segment,
    )

    segments_text = _dump_json([segment.canonical() for segment in segments])
    symbols_text = _dump_json({symbol: float(value) for symbol, value in sorted(symbols.items())})
    params_text = _dump_json(params.as_dict())
    input_hashes = {
        "segment_count": len(segments),
        "segments_sha256": _sha256_text(segments_text),
        "symbols_sha256": _sha256_text(symbols_text),
        "params_sha256": _sha256_text(params_text),
        "state_schema_sha256": _sha256_text(STATE_SCHEMA),
        "question_schema_sha256": _sha256_text(QUESTION_SCHEMA),
    }
    # run_id 把 STATE_SCHEMA / QUESTION_SCHEMA 纳入哈希输入：状态或候选文案变更产生
    # 新 run（旧 run 保留不覆盖）
    run_id = "run-" + _sha256_text(
        "|".join((segments_text, symbols_text, params_text, STATE_SCHEMA, QUESTION_SCHEMA))
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
        board_state_skipped=board_state_skipped,
        daily_source_versions={
            symbol: loaded.source_data_version
            for symbol, loaded in daily_loaded_by_symbol.items()
        },
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
        board_state_skipped=board_state_skipped,
    )


__all__ = [
    "AUDIT_FILENAME",
    "FLAT_CRITERIA",
    "GOLD_LABEL_KIND",
    "HELD_CRITERIA",
    "POSITION_LABELS",
    "QUESTION_ID",
    "QUESTION_INSTRUCTIONS",
    "QUESTION_SCHEMA",
    "STATE_SCHEMA",
    "BoardStateValues",
    "GenerationResult",
    "build_record",
    "generate_dataset",
    "render_state",
]

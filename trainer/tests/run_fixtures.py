"""trainer/tests 共享夹具：手写最小 run 目录（不 import dataset，纯 JSONL 构造）。

字段语义与数据侧生成产物对齐（id = ``{segment_id}:{bar_index}``、唯一问题
``next_action``、旁挂行 11 键），但由本模块独立构造——trainer 的数据契约
测试不依赖 MarketSense 数据侧代码（trainer 不 import dataset 的镜像约束）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

#: 与数据侧一致的决策问题候选（6 动作；键序 = 记录渲染顺序）
CRITERIA: dict[str, str] = {
    "open_long": "预期上行幅度足够：买入开多",
    "open_short": "预期下行幅度足够：卖出开空",
    "close": "到达离场条件：平仓",
    "reverse": "到达反向条件：反手",
    "hold": "已有持仓且未到离场条件：继续持有",
    "stay_flat": "无机会或信号不足：继续空仓",
}

_OUTCOME_KEYS = (
    "id",
    "segment_id",
    "bar_index",
    "split",
    "action",
    "direction",
    "outcome",
    "risk_ratio",
    "r_multiple",
    "exit_bar_index",
    "exit_reason",
)


def make_record(
    record_id: str,
    split: str,
    gold: str,
    *,
    state: str = "账户: 空仓 净值=100.000000\n现价: 开=1.000000 高=1.100000 低=0.998000 收=1.050000",
    criteria: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """一条最小合法训练记录（schema 必要子集）。"""
    segment_id, _, bar = record_id.rpartition(":")
    return {
        "id": record_id,
        "family_id": segment_id,
        "state": state,
        "split": split,
        "gold": {"next_action": gold},
        "questions": {
            "next_action": {
                "type": "choice",
                "instructions": "基于状态选择下一动作",
                "criteria": dict(criteria or CRITERIA),
            }
        },
    }


def make_outcome_row(
    record_id: str,
    split: str,
    action: str,
    *,
    outcome: float = 0.01,
    risk_ratio: float = 0.01,
    exit_bar_index: int | None = None,
    exit_reason: str = "reverse_close",
) -> dict[str, Any]:
    """一条最小合法旁挂行（r_multiple = outcome / risk_ratio）。"""
    segment_id, _, bar = record_id.rpartition(":")
    bar_index = int(bar)
    direction = "long" if action == "open_long" else "short"
    return {
        "id": record_id,
        "segment_id": segment_id,
        "bar_index": bar_index,
        "split": split,
        "action": action,
        "direction": direction,
        "outcome": outcome,
        "risk_ratio": risk_ratio,
        "r_multiple": outcome / risk_ratio,
        "exit_bar_index": bar_index if exit_bar_index is None else exit_bar_index,
        "exit_reason": exit_reason,
    }


def write_run(
    run_dir: str | Path,
    records: Iterable[Mapping[str, Any]],
    outcome_rows: Iterable[Mapping[str, Any]] = (),
) -> Path:
    """把记录/旁挂写成 run 目录布局（``{split}.jsonl`` + ``outcomes.jsonl``）。"""
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_split.setdefault(record["split"], []).append(record)
    for split, split_records in by_split.items():
        path = run / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_records),
            encoding="utf-8",
        )
    rows = list(outcome_rows)
    if rows:
        (run / "outcomes.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    return run


def default_run(tmp_path: Path) -> Path:
    """最小合法 run：train 3（2 开仓 + 1 hold）、dev 1（开仓）、test 1（hold）。"""
    records = [
        make_record("seg-a:0", "train", "open_long"),
        make_record("seg-a:3", "train", "hold"),
        make_record("seg-b:1", "train", "open_short"),
        make_record("seg-c:0", "dev", "open_long"),
        make_record("seg-d:2", "test", "stay_flat"),
    ]
    outcome_rows = [
        make_outcome_row("seg-a:0", "train", "open_long", outcome=0.02, risk_ratio=0.01),
        make_outcome_row("seg-b:1", "train", "open_short", outcome=-0.01, risk_ratio=0.01),
        make_outcome_row("seg-c:0", "dev", "open_long", outcome=0.05, risk_ratio=0.02),
    ]
    return write_run(tmp_path / "run", records, outcome_rows)

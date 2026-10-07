"""训练数据装载 / outcome 连接 / leaf 契约渲染 / 完整题打包（T6；纯 stdlib，不 import torch）。

数据源 = episode 生成 run 目录：``{split}.jsonl``（NanoJev 训练记录契约）+
``outcomes.jsonl``（outcome 旁挂，utility-trainer 任务 T1 产物）。

本模块是 ``trainer/`` 与 MarketSense 数据侧之间的**唯一契约点**：为了保持
trainer 独立（不 import ``dataset``、不 import NanoJev），记录/旁挂行的必要
schema 校验、leaf token 契约字符串、``pack_complete_questions`` 预算语义都在
这里按 NanoJev 只读参考实现**同语义重写**（来源见各函数 docstring；契约字符串
由 ``trainer/tests/test_data.py`` 逐字断言，防漂移）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from trainer.errors import TrainerError
from trainer.weights import UtilityWeightConfig, utility_weight

__all__ = [
    "OUTCOMES_FILENAME",
    "OPEN_ACTIONS",
    "QUESTION_ID",
    "SPLITS",
    "action_class",
    "build_examples",
    "join_outcomes",
    "load_outcome_rows",
    "load_run",
    "load_split_records",
    "pack_complete_questions",
    "pull_report",
    "validate_outcome_row",
    "validate_record",
]

#: NanoJev ``train_pipeline_decisions.SPLITS`` 的同语义重写（只读参考，不 import）
SPLITS = ("train", "dev", "calibration", "test", "ood")
#: outcome 旁挂文件名（与数据侧 ``dataset/market_episode/nanojev_records.py`` 同值）
OUTCOMES_FILENAME = "outcomes.jsonl"
#: 开仓动作（旁挂行的覆盖范围）
OPEN_ACTIONS = ("open_long", "open_short")
#: 决策问题 id（训练记录唯一问题）
QUESTION_ID = "next_action"
#: 反向候选映射（方向惩罚 v1：open_long ↔ open_short 互为反向）
_OPPOSITE_ACTION = {"open_long": "open_short", "open_short": "open_long"}
#: 动作 → 报告类（open/flat/hold/exit）
_ACTION_CLASSES = {
    "open_long": "open",
    "open_short": "open",
    "stay_flat": "flat",
    "hold": "hold",
    "close": "exit",
    "reverse": "exit",
}

#: 旁挂行必需键（与数据侧生成约定同步）
_OUTCOME_ROW_FIELDS = (
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


def _parse_state_id(record_id: str) -> tuple[str, int]:
    """``{segment_id}:{bar_index}`` → ``(segment_id, bar_index)``（不合法即错）。"""
    if not isinstance(record_id, str) or record_id.count(":") < 1:
        raise TrainerError(f"记录 id 必须形如 '{{segment_id}}:{{bar_index}}': {record_id!r}")
    segment_id, _, tail = record_id.rpartition(":")
    if not segment_id or not tail.isdigit():
        raise TrainerError(f"记录 id 必须形如 '{{segment_id}}:{{bar_index}}': {record_id!r}")
    return segment_id, int(tail)


def validate_record(record: Any, *, split: str, where: str) -> None:
    """训练记录 schema **必要子集**校验（镜像 NanoJev ``validate_training_row``
    的必要部分；来源 ``train_pipeline_decisions.py`` L178-226，只读参考）。
    """
    if not isinstance(record, dict):
        raise TrainerError(f"{where} 记录必须是 JSON 对象")
    for field in ("id", "family_id", "state", "split", "questions"):
        if field not in record:
            raise TrainerError(f"{where} 记录缺少必需字段 {field!r}")
    record_id = record["id"]
    if not isinstance(record_id, str) or not record_id.strip():
        raise TrainerError(f"{where} id 必须为非空字符串: {record_id!r}")
    if not isinstance(record["family_id"], str) or not record["family_id"].strip():
        raise TrainerError(f"{where} family_id 必须为非空字符串: {record['family_id']!r}")
    if not isinstance(record["state"], str) or not record["state"].strip():
        raise TrainerError(f"{where} state 必须为非空字符串: {record['state']!r}")
    if record["split"] != split:
        raise TrainerError(f"{where} 记录 split={record['split']!r} 与所在文件 {split!r} 不一致")
    _parse_state_id(record_id)
    questions = record["questions"]
    if not isinstance(questions, dict) or QUESTION_ID not in questions:
        raise TrainerError(f"{where} questions 缺少决策问题 {QUESTION_ID!r}")
    question = questions[QUESTION_ID]
    if not isinstance(question, dict):
        raise TrainerError(f"{where} 问题 {QUESTION_ID!r} 必须是 JSON 对象")
    if question.get("type") != "choice":
        raise TrainerError(
            f"{where} 问题 {QUESTION_ID!r} type 必须为 'choice': {question.get('type')!r}"
        )
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise TrainerError(f"{where} 问题 instructions 必须为非空字符串: {instructions!r}")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or len(criteria) < 2:
        raise TrainerError(f"{where} choice 问题 candidates 必须 ≥ 2: {criteria!r}")
    for key, text in criteria.items():
        if not isinstance(key, str) or not key or not isinstance(text, str) or not text.strip():
            raise TrainerError(f"{where} 候选键/文本必须为非空字符串: {key!r} → {text!r}")
    # gold 在记录级（镜像 NanoJev ``validate_training_row``：``row["gold"][qid]``，
    # 来源 ``train_pipeline_decisions.py`` L199-206；问题对象内无 gold 键）。
    gold_map = record.get("gold")
    if not isinstance(gold_map, dict) or set(gold_map) - set(questions):
        raise TrainerError(f"{where} gold 必须是问题 id → 目标的记录级映射: {gold_map!r}")
    gold = gold_map.get(QUESTION_ID)
    if not isinstance(gold, str) or gold not in criteria:
        raise TrainerError(f"{where} gold 必须是候选键之一: {gold!r}")


def load_split_records(run_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """读取一个 split 的训练记录（缺文件 → 空列表，镜像 NanoJev
    ``read_training_records``；逐行 schema 子集校验 + id 唯一）。
    """
    path = Path(run_dir) / f"{split}.jsonl"
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name}:{line_no}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise TrainerError(f"{where} 不是合法 JSON: {error}") from error
        validate_record(record, split=split, where=where)
        if record["id"] in seen:
            raise TrainerError(f"{where} 记录 id 重复: {record['id']!r}")
        seen.add(record["id"])
        records.append(record)
    return records


def validate_outcome_row(row: Any, where: str) -> None:
    """旁挂行 schema 校验（字段/类型/基本不变量；值级复算由数据侧审计负责）。"""
    if not isinstance(row, dict):
        raise TrainerError(f"{where} 旁挂行必须是 JSON 对象")
    for field in _OUTCOME_ROW_FIELDS:
        if field not in row:
            raise TrainerError(f"{where} 旁挂行缺少键 {field!r}")
    extra = set(row) - set(_OUTCOME_ROW_FIELDS)
    if extra:
        raise TrainerError(f"{where} 旁挂行出现未知键: {sorted(extra)}")
    row_id, segment_id = row["id"], row["segment_id"]
    bar_index, exit_bar_index = row["bar_index"], row["exit_bar_index"]
    for name, value in (("bar_index", bar_index), ("exit_bar_index", exit_bar_index)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TrainerError(f"{where} {name} 必须为非负整数: {value!r}")
    if not isinstance(segment_id, str) or not segment_id:
        raise TrainerError(f"{where} segment_id 必须为非空字符串: {segment_id!r}")
    if row_id != f"{segment_id}:{bar_index}":
        raise TrainerError(f"{where} id 与 segment_id/bar_index 不一致: {row_id!r}")
    if row["split"] not in SPLITS:
        raise TrainerError(f"{where} split 非法: {row['split']!r}")
    if row["action"] not in OPEN_ACTIONS:
        raise TrainerError(f"{where} action 必须为开仓动作: {row['action']!r}")
    expected_direction = "long" if row["action"] == "open_long" else "short"
    if row["direction"] != expected_direction:
        raise TrainerError(
            f"{where} action/direction 不一致: {row['action']!r}/{row['direction']!r}"
        )
    if exit_bar_index < bar_index:
        raise TrainerError(
            f"{where} exit_bar_index({exit_bar_index}) 不得早于 bar_index({bar_index})"
        )
    for name in ("outcome", "risk_ratio", "r_multiple"):
        value = row[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TrainerError(f"{where} {name} 必须为数值: {value!r}")
        if not math.isfinite(float(value)):
            raise TrainerError(f"{where} {name} 必须为有限数值: {value!r}")
    if float(row["risk_ratio"]) <= 0:
        raise TrainerError(f"{where} risk_ratio 必须为正数: {row['risk_ratio']!r}")


def load_outcome_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    """读取 outcome 旁挂（缺文件 → 空列表；逐行 schema 校验 + 行级 id 唯一）。"""
    path = Path(run_dir) / OUTCOMES_FILENAME
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name}:{line_no}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise TrainerError(f"{where} 不是合法 JSON: {error}") from error
        validate_outcome_row(row, where)
        if row["id"] in seen:
            raise TrainerError(f"{where} 旁挂行 id 重复: {row['id']!r}")
        seen.add(row["id"])
        rows.append(row)
    return rows


def is_open_record(record: Mapping[str, Any]) -> bool:
    """记录的 gold 是否为开仓动作（旁挂行的覆盖范围；记录级 gold 映射）。"""
    gold = record["gold"][QUESTION_ID]
    return gold in OPEN_ACTIONS


def join_outcomes(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    outcome_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """记录 ↔ 旁挂按 ``id`` 连接（**四类硬错误** + split 一致性，违规即 ``TrainerError``）。

    1. 开仓记录缺少旁挂行；
    2. 旁挂行引用未知记录 id；
    3. 旁挂行 id 重复；
    4. 非开仓记录带旁挂行（旁挂只覆盖 gold ∈ {open_long, open_short}）。
    另：旁挂行 ``split`` 与记录 ``split`` 不一致同样报错（覆盖一致性）。

    :return: ``record_id → 旁挂行`` 映射。
    """
    records: dict[str, Mapping[str, Any]] = {}
    for split, split_records in records_by_split.items():
        for record in split_records:
            records[record["id"]] = record
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in outcome_rows:
        row_id = row["id"]
        if row_id in by_id:
            raise TrainerError(f"outcome 旁挂行 id 重复: {row_id!r}")
        record = records.get(row_id)
        if record is None:
            raise TrainerError(f"outcome 旁挂行引用未知记录 id: {row_id!r}")
        if row["split"] != record["split"]:
            raise TrainerError(
                f"outcome 旁挂行 split 与记录不一致: {row_id!r} "
                f"旁挂 {row['split']!r} vs 记录 {record['split']!r}"
            )
        by_id[row_id] = row
    for record_id, record in records.items():
        has_row = record_id in by_id
        if is_open_record(record) and not has_row:
            raise TrainerError(f"开仓记录缺少 outcome 旁挂行: {record_id!r}")
        if not is_open_record(record) and has_row:
            raise TrainerError(f"非开仓记录不得有 outcome 旁挂行: {record_id!r}")
    return by_id


def action_class(action: str) -> str:
    """gold 动作 → 报告类（open/flat/hold/exit；未知动作报错）。"""
    if action not in _ACTION_CLASSES:
        raise TrainerError(f"未知 gold 动作: {action!r}")
    return _ACTION_CLASSES[action]


def pull_report(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    outcomes_by_id: Mapping[str, Mapping[str, Any]],
    config: UtilityWeightConfig | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """逐 split 逐类（open/flat/hold/exit）的计数 / 权重均值 / 总权重拉力报告。

    权重：open 类 = ``utility_weight(r_multiple)``；其余类 = 1.0（中性）。
    ``--validate-only`` 的拉力可见性输出（DESIGN D8）。
    """
    report: dict[str, dict[str, dict[str, Any]]] = {}
    for split in SPLITS:
        classes: dict[str, list[float]] = {name: [] for name in ("open", "flat", "hold", "exit")}
        for record in records_by_split.get(split, ()):
            gold = record["gold"][QUESTION_ID]
            class_name = action_class(gold)
            if class_name == "open":
                row = outcomes_by_id[record["id"]]
                weight = utility_weight(row["r_multiple"], config)
            else:
                weight = 1.0
            classes[class_name].append(weight)
        report[split] = {
            name: {
                "count": len(weights),
                "mean_weight": (round(sum(weights) / len(weights), 12) if weights else None),
                "total_weight": round(sum(weights), 12),
            }
            for name, weights in classes.items()
        }
    return report


def build_examples(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    outcomes_by_id: Mapping[str, Mapping[str, Any]],
    *,
    encode: Callable[[str], list[int]],
    eos_token_id: int,
    max_length: int,
    config: UtilityWeightConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """记录 → 模型示例（leaf token 契约**同语义重写**，来源 NanoJev
    ``predict_toy_decisions.py`` L104-112，只读参考）::

        State:\n{state}\n
        Question type: choice\nQuestion:\n{instructions}\n
        每候选: Candidate:\n{key}: {criteria[key]}\nDecision: + EOS

    其中 choice 候选文本 = ``f"{key}: {criteria[key]}"``。契约字符串由
    ``trainer/tests/test_data.py`` 逐字断言。

    :param encode: tokenizer 的无特殊 token 编码函数（``tokenizer.encode(t,
        add_special_tokens=False)``），本模块不依赖具体 tokenizer 实现。
    """
    if config is None:
        config = UtilityWeightConfig()
    result: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        examples: list[dict[str, Any]] = []
        for record in records_by_split.get(split, ()):
            question = record["questions"][QUESTION_ID]
            candidate_ids = list(question["criteria"])
            texts = [f"{key}: {question['criteria'][key]}" for key in candidate_ids]
            segments = [
                f"State:\n{record['state']}\n",
                f"Question type: {question['type']}\nQuestion:\n{question['instructions']}\n",
            ]
            prefix: list[int] = []
            for segment in segments:
                prefix.extend(encode(segment))
            leaves = [
                prefix + encode(f"Candidate:\n{text}\nDecision:") + [eos_token_id]
                for text in texts
            ]
            largest = max(map(len, leaves))
            if largest > max_length:
                raise TrainerError(
                    f"{record['id']}:{QUESTION_ID} 候选路径为 {largest} token，"
                    f"超过 max_length={max_length}；未截断输入"
                )
            # gold 在记录级（与 validate_record 同口径，问题对象内无 gold 键）
            gold = record["gold"][QUESTION_ID]
            gold_index = candidate_ids.index(gold)
            is_open = gold in OPEN_ACTIONS
            outcome_row = outcomes_by_id.get(record["id"])
            if is_open:
                if outcome_row is None:
                    raise TrainerError(f"开仓记录缺少 outcome 旁挂行: {record['id']!r}")
                weight = utility_weight(outcome_row["r_multiple"], config)
                opposite_key = _OPPOSITE_ACTION.get(gold)
                opposite_index = (
                    candidate_ids.index(opposite_key) if opposite_key in candidate_ids else None
                )
            else:
                weight = 1.0
                opposite_index = None
            examples.append(
                {
                    "id": f"{record['id']}:{QUESTION_ID}",
                    "state_id": record["id"],
                    "qid": QUESTION_ID,
                    "split": split,
                    "type": "choice",
                    "candidate_ids": candidate_ids,
                    "candidate_texts": texts,
                    "leaf_tokens": leaves,
                    "gold_index": gold_index,
                    "weight": weight,
                    "is_open": is_open,
                    "opposite_index": opposite_index,
                }
            )
        result[split] = examples
    return result


def pack_complete_questions(
    examples: Sequence[Mapping[str, Any]],
    max_questions: int,
    max_tokens: int,
) -> list[list[Mapping[str, Any]]]:
    """完整题预算打包（**同语义重写**，来源 NanoJev
    ``train_pipeline_decisions.py`` L237-255，只读参考）：

    预算 = 全部候选路径数 × 最长 padded 路径长；softmax 永不跨题拆分、
    候选永不截断；单题超预算 → 立即报错（要求提高显式预算，而非静默拆分）。
    """
    if max_questions <= 0 or max_tokens < 0:
        raise TrainerError("打包预算非法：max_questions 必须 > 0、max_tokens 必须 ≥ 0")
    result: list[list[Mapping[str, Any]]] = []
    group: list[Mapping[str, Any]] = []
    paths, width = 0, 0
    for example in examples:
        n = len(example["leaf_tokens"])
        size = max(map(len, example["leaf_tokens"]))
        if max_tokens and n * size > max_tokens:
            raise TrainerError(
                f"完整题 {example['id']} 需要 {n * size} 个 padded token，"
                f"超出预算 {max_tokens}；请提高显式预算——softmax 不得跨题拆分、候选不得截断"
            )
        prospective = (paths + n) * max(width, size)
        if group and (len(group) >= max_questions or (max_tokens and prospective > max_tokens)):
            result.append(group)
            group, paths, width = [], 0, 0
        group.append(example)
        paths += n
        width = max(width, size)
    if group:
        result.append(group)
    return result


def load_run(
    run_dir: str | Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    """装载 run 目录（记录 + 旁挂 + 连接）；跨 split id 唯一性在此强制。"""
    records_by_split = {split: load_split_records(run_dir, split) for split in SPLITS}
    outcome_rows = load_outcome_rows(run_dir)
    outcomes_by_id = join_outcomes(records_by_split, outcome_rows)
    seen: dict[str, str] = {}
    for split, records in records_by_split.items():
        for record in records:
            if record["id"] in seen:
                raise TrainerError(
                    f"记录 id 跨 split 重复: {record['id']!r} 同时出现在 "
                    f"{seen[record['id']]!r} 与 {split!r}"
                )
            seen[record["id"]] = split
    return records_by_split, outcomes_by_id, outcome_rows

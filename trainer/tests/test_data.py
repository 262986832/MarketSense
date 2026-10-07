"""T6 数据装载 / 连接 / leaf 契约 / 打包单测（torch 无关）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainer.data import (
    OUTCOMES_FILENAME,
    QUESTION_ID,
    SPLITS,
    action_class,
    build_examples,
    is_open_record,
    join_outcomes,
    load_outcome_rows,
    load_run,
    load_split_records,
    pack_complete_questions,
    pull_report,
    validate_outcome_row,
    validate_record,
)
from trainer.errors import TrainerError
from trainer.tests.run_fixtures import (
    CRITERIA,
    default_run,
    make_outcome_row,
    make_record,
    write_run,
)
from trainer.weights import UtilityWeightConfig


def _identity_encode(text: str) -> list[int]:
    """恒等 tokenizer：字 → ord（leaf 契约断言用，token 值不影响结构）。"""
    return [ord(char) % 1000 for char in text]


# ---------------------------------------------------------------- 记录校验


def test_validate_record_accepts_minimal_record() -> None:
    record = make_record("seg-a:0", "train", "open_long")
    validate_record(record, split="train", where="t:1")  # 不抛即通过


def test_validate_record_rejects_split_mismatch() -> None:
    record = make_record("seg-a:0", "dev", "open_long")
    with pytest.raises(TrainerError, match="split"):
        validate_record(record, split="train", where="t:1")


def test_validate_record_rejects_bad_id() -> None:
    record = make_record("seg-a:0", "train", "hold")
    record["id"] = "no-colon"
    with pytest.raises(TrainerError, match="id"):
        validate_record(record, split="train", where="t:1")


def test_validate_record_rejects_missing_question_or_gold() -> None:
    record = make_record("seg-a:0", "train", "hold")
    del record["questions"]["next_action"]
    with pytest.raises(TrainerError, match="next_action"):
        validate_record(record, split="train", where="t:1")
    record = make_record("seg-a:0", "train", "hold")
    record["gold"] = {"next_action": "open_up"}
    with pytest.raises(TrainerError, match="gold"):
        validate_record(record, split="train", where="t:1")


def test_validate_record_rejects_too_few_candidates() -> None:
    record = make_record("seg-a:0", "train", "hold")
    record["questions"]["next_action"]["criteria"] = {"hold": "继续持有"}
    with pytest.raises(TrainerError, match="candidates"):
        validate_record(record, split="train", where="t:1")


# ---------------------------------------------------------------- 旁挂行校验


def test_outcome_row_roundtrip_via_fixtures() -> None:
    row = make_outcome_row("seg-a:0", "train", "open_long")
    validate_outcome_row(row, "t:1")
    keys = tuple(row)
    assert keys == OUTCOMES_ROW_KEYS


def test_outcome_row_rejects_id_mismatch() -> None:
    row = make_outcome_row("seg-a:0", "train", "open_long")
    row["id"] = "seg-b:0"
    with pytest.raises(TrainerError, match="id 与 segment_id/bar_index 不一致"):
        validate_outcome_row(row, "t:1")


def test_outcome_row_rejects_action_direction_mismatch() -> None:
    row = make_outcome_row("seg-a:0", "train", "open_long")
    row["direction"] = "short"
    with pytest.raises(TrainerError, match="action/direction"):
        validate_outcome_row(row, "t:1")


def test_outcome_row_rejects_bad_exit_index() -> None:
    row = make_outcome_row("seg-a:0", "train", "open_long")
    row["exit_bar_index"] = row["bar_index"] - 1
    with pytest.raises(TrainerError, match="exit_bar_index"):
        validate_outcome_row(row, "t:1")


def test_outcome_row_rejects_unknown_keys_and_nonpositive_risk() -> None:
    row = make_outcome_row("seg-a:0", "train", "open_long")
    row["extra"] = 1
    with pytest.raises(TrainerError, match="未知键"):
        validate_outcome_row(row, "t:1")
    row = make_outcome_row("seg-a:0", "train", "open_long")
    row["risk_ratio"] = 0.0
    with pytest.raises(TrainerError, match="risk_ratio"):
        validate_outcome_row(row, "t:1")


def test_load_outcome_rows_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_outcome_rows(tmp_path) == []


# ---------------------------------------------------------------- 连接四类硬错误


def test_join_outcomes_happy_path(tmp_path: Path) -> None:
    run = default_run(tmp_path)
    records_by_split, outcomes_by_id, rows = load_run(run)
    assert set(records_by_split) == set(SPLITS)
    assert len(records_by_split["train"]) == 3
    assert set(outcomes_by_id) == {"seg-a:0", "seg-b:1", "seg-c:0"}
    assert len(rows) == 3


def test_join_outcomes_missing_row_for_open(tmp_path: Path) -> None:
    run = default_run(tmp_path)
    (run / OUTCOMES_FILENAME).write_text(
        json.dumps(make_outcome_row("seg-a:0", "train", "open_long")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TrainerError, match="缺少 outcome 旁挂行"):
        load_run(run)


def test_join_outcomes_unknown_record(tmp_path: Path) -> None:
    run = default_run(tmp_path)
    rows = (run / OUTCOMES_FILENAME).read_text(encoding="utf-8").splitlines()
    rows.append(json.dumps(make_outcome_row("seg-z:9", "train", "open_long")))
    (run / OUTCOMES_FILENAME).write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(TrainerError, match="未知记录 id"):
        load_run(run)


def test_join_outcomes_duplicate_id(tmp_path: Path) -> None:
    run = default_run(tmp_path)
    rows = (run / OUTCOMES_FILENAME).read_text(encoding="utf-8").splitlines()
    rows.append(rows[0])
    (run / OUTCOMES_FILENAME).write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(TrainerError, match="id 重复"):
        load_run(run)


def test_join_outcomes_row_on_non_open_record(tmp_path: Path) -> None:
    run = default_run(tmp_path)
    rows = (run / OUTCOMES_FILENAME).read_text(encoding="utf-8").splitlines()
    rows.append(json.dumps(make_outcome_row("seg-a:3", "train", "open_long")))
    (run / OUTCOMES_FILENAME).write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(TrainerError, match="非开仓记录"):
        load_run(run)


def test_join_outcomes_split_mismatch(tmp_path: Path) -> None:
    run = default_run(tmp_path)
    rows = (run / OUTCOMES_FILENAME).read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["split"] = "dev"
    rows[0] = json.dumps(row)
    (run / OUTCOMES_FILENAME).write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(TrainerError, match="split 与记录不一致"):
        load_run(run)


def test_join_outcomes_cross_split_duplicate_id(tmp_path: Path) -> None:
    run = default_run(tmp_path)
    records = (run / "train.jsonl").read_text(encoding="utf-8").splitlines()
    # 用无旁挂行的 hold 记录（seg-a:3），避免先触发旁挂 split 一致性检查
    row = json.loads(next(line for line in records if "seg-a:3" in line))
    row["split"] = "test"
    (run / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(TrainerError, match="跨 split 重复"):
        load_run(run)


def test_load_split_records_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_split_records(tmp_path, "train") == []


# ---------------------------------------------------------------- leaf 契约逐字


def test_build_examples_leaf_contract_literal_strings(tmp_path: Path) -> None:
    run = default_run(tmp_path)
    records_by_split, outcomes_by_id, _ = load_run(run)
    examples = build_examples(
        records_by_split,
        outcomes_by_id,
        encode=_identity_encode,
        eos_token_id=999,
        max_length=10000,
    )
    record = make_record("seg-a:0", "train", "open_long")
    expected_prefix = (
        f"State:\n{record['state']}\n"
        f"Question type: choice\nQuestion:\n{record['questions']['next_action']['instructions']}\n"
    )
    candidate_keys = list(CRITERIA)
    expected_leaves = [
        _identity_encode(expected_prefix)
        + _identity_encode(f"Candidate:\n{key}: {CRITERIA[key]}\nDecision:")
        + [999]
        for key in candidate_keys
    ]
    example = examples["train"][0]
    assert example["id"] == "seg-a:0:next_action"
    assert example["candidate_ids"] == candidate_keys
    assert example["leaf_tokens"] == expected_leaves
    assert example["gold_index"] == candidate_keys.index("open_long")
    assert example["is_open"] is True
    assert example["opposite_index"] == candidate_keys.index("open_short")
    assert example["weight"] == pytest.approx(2.0)  # r = 0.02/0.01 = 2 → 1 + 0.5·2 = 2.0


def test_build_examples_weights_and_neutral(tmp_path: Path) -> None:
    run = default_run(tmp_path)
    records_by_split, outcomes_by_id, _ = load_run(run)
    examples = build_examples(
        records_by_split,
        outcomes_by_id,
        encode=_identity_encode,
        eos_token_id=999,
        max_length=10000,
    )
    weights = {ex["id"]: ex["weight"] for split in SPLITS for ex in examples[split]}
    assert weights["seg-a:0:next_action"] == pytest.approx(2.0)  # r=2
    assert weights["seg-b:1:next_action"] == pytest.approx(0.5)  # r=-1
    assert weights["seg-c:0:next_action"] == pytest.approx(2.25)  # r=2.5
    assert weights["seg-a:3:next_action"] == 1.0  # hold → 中性
    assert weights["seg-d:2:next_action"] == 1.0  # stay_flat → 中性
    hold_example = examples["train"][1]
    assert hold_example["is_open"] is False
    assert hold_example["opposite_index"] is None


def test_build_examples_overlength_raises_without_truncation(tmp_path: Path) -> None:
    run = default_run(tmp_path)
    records_by_split, outcomes_by_id, _ = load_run(run)
    with pytest.raises(TrainerError, match="超过 max_length"):
        build_examples(
            records_by_split,
            outcomes_by_id,
            encode=_identity_encode,
            eos_token_id=999,
            max_length=10,
        )


def test_build_examples_weight_config_passthrough(tmp_path: Path) -> None:
    run = default_run(tmp_path)
    records_by_split, outcomes_by_id, _ = load_run(run)
    config = UtilityWeightConfig(utility_alpha=1.0, r_cap=2.0, weight_min=0.2, weight_max=4.0)
    examples = build_examples(
        records_by_split,
        outcomes_by_id,
        encode=_identity_encode,
        eos_token_id=999,
        max_length=10000,
        config=config,
    )
    assert examples["train"][0]["weight"] == pytest.approx(3.0)  # r=2 → 1+2 = 3


# ---------------------------------------------------------------- 打包


def _pack_example(question_id: str, paths: int, width: int) -> dict:
    return {
        "id": question_id,
        "type": "choice",
        "candidate_ids": [f"c{i}" for i in range(paths)],
        "leaf_tokens": [[1] * width] * paths,
    }


def test_pack_groups_by_question_count() -> None:
    examples = [_pack_example(f"q{i}", 3, 4) for i in range(5)]
    groups = pack_complete_questions(examples, 2, 0)
    assert [len(group) for group in groups] == [2, 2, 1]


def test_pack_respects_token_budget() -> None:
    examples = [_pack_example(f"q{i}", 2, 5) for i in range(4)]
    groups = pack_complete_questions(examples, 10, 20)  # 每题 2×5=10，最多 2 题/组
    assert [len(group) for group in groups] == [2, 2]


def test_pack_single_oversize_question_raises() -> None:
    with pytest.raises(TrainerError, match="请提高显式预算"):
        pack_complete_questions([_pack_example("big", 4, 10)], 4, 10)


def test_pack_invalid_budget_raises() -> None:
    with pytest.raises(TrainerError, match="预算非法"):
        pack_complete_questions([_pack_example("q", 2, 2)], 0, 0)


# ---------------------------------------------------------------- 拉力报告


def test_pull_report_counts_and_weights(tmp_path: Path) -> None:
    run = default_run(tmp_path)
    records_by_split, outcomes_by_id, _ = load_run(run)
    report = pull_report(records_by_split, outcomes_by_id)
    train = report["train"]
    assert train["open"]["count"] == 2
    assert train["open"]["total_weight"] == pytest.approx(2.5)  # 2.0 + 0.5
    assert train["open"]["mean_weight"] == pytest.approx(1.25)
    assert train["hold"]["count"] == 1
    assert train["hold"]["total_weight"] == pytest.approx(1.0)
    assert train["flat"]["count"] == 0
    assert train["flat"]["mean_weight"] is None
    assert train["exit"]["count"] == 0
    assert report["test"]["flat"]["count"] == 1
    assert report["dev"]["open"]["count"] == 1


def test_action_class_mapping() -> None:
    assert action_class("open_long") == "open"
    assert action_class("open_short") == "open"
    assert action_class("stay_flat") == "flat"
    assert action_class("hold") == "hold"
    assert action_class("close") == "exit"
    assert action_class("reverse") == "exit"
    with pytest.raises(TrainerError, match="未知 gold 动作"):
        action_class("open_up")


def test_is_open_record() -> None:
    assert is_open_record(make_record("s:0", "train", "open_short"))
    assert not is_open_record(make_record("s:0", "train", "close"))


# ---------------------------------------------------------------- 夹具自洽

OUTCOMES_ROW_KEYS = (
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


def test_fixture_row_keys_match_contract() -> None:
    row = make_outcome_row("seg-a:0", "train", "open_long")
    assert tuple(row) == OUTCOMES_ROW_KEYS

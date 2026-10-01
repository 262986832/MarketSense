"""T5（审计）：确定性双跑、跨 split 隔离、状态泄漏抽查、计数汇总一致性。

审计自检必须能**检出人为构造的坏数据**（注入跨 split / 下一根状态 / 绝对价格）。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from dataset.errors import DatasetError
from dataset.market_episode.audit import (
    check_audit_consistency,
    check_no_absolute_values,
    check_records,
    check_split_isolation,
    check_state_leakage,
    parse_state_id,
    verify_determinism,
)
from dataset.market_episode.nanojev_records import generate_dataset, render_state
from dataset.market_episode.replay import load_segment_bars
from dataset.market_episode.segments import (
    SPLIT_ROLES,
    EpisodeParams,
    load_segments,
    load_symbols_config,
)
from dataset.storage import save_ohlcv
from dataset.tests.market_episode_fixtures import (
    SYMBOL,
    bars,
    build_workspace,
    frame,
)

#: 一段 = 开多 → 持有 → 程序止损离场 → 空仓（每段 4 根）；重复 3 段覆盖 train/dev/test
_PATTERN = [
    (1000, 1000.5, 999.5, 1000),
    (1000, 1010, 1000, 1009),
    (1009, 1012, 990, 992),
    (992, 993, 988, 991),
]
_ROWS = _PATTERN * 3
#: 每段到片段末仍有持仓（用于验证片段末强平审计）
_HOLDING_PATTERN = [
    (100, 100.2, 99.8, 100),
    (100, 110, 100, 109),
]
_HOLDING_ROWS = _HOLDING_PATTERN * 3


def _workspace_and_records(tmp_path: Path):
    workspace = build_workspace(tmp_path, _ROWS)
    symbols = load_symbols_config(workspace.symbols_path)
    segments = load_segments(workspace.manifest)
    params = EpisodeParams()
    result = generate_dataset(
        segments,
        symbols=symbols,
        data_dir=workspace.data_dir,
        params=params,
        output_dir=tmp_path / "out",
    )
    records = [
        json.loads(line)
        for split in SPLIT_ROLES
        for line in (result.run_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    bars_by_segment = {
        segment.segment_id: load_segment_bars(segment, data_dir=workspace.data_dir)[0]
        for segment in segments
    }
    return workspace, segments, symbols, params, result, records, bars_by_segment


def test_parse_state_id_round_trip_and_rejects_bad_format() -> None:
    assert parse_state_id("seg-1:12") == ("seg-1", 12)

    with pytest.raises(DatasetError, match="state_id 格式非法"):
        parse_state_id("seg-1")
    with pytest.raises(DatasetError, match="决策 K 线序号非法"):
        parse_state_id("seg-1:abc")
    with pytest.raises(DatasetError, match="必须 ≥ 0"):
        parse_state_id("seg-1:-1")


def test_generated_records_pass_all_audit_checks(tmp_path: Path) -> None:
    _, _, _, params, result, records, bars_by_segment = _workspace_and_records(tmp_path)

    check_records(records)
    check_state_leakage(
        records, bars_by_segment=bars_by_segment, price_precision=params.price_precision
    )
    check_audit_consistency(result.audit, _records_by_split(result))
    assert all(row["split"] in SPLIT_ROLES for row in records)


def _records_by_split(result) -> dict[str, list[dict]]:
    return {
        split: [
            json.loads(line)
            for line in (result.run_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for split in SPLIT_ROLES
    }


def test_split_isolation_detects_injected_cross_split_record(tmp_path: Path) -> None:
    _, _, _, _, _, records, _ = _workspace_and_records(tmp_path)
    check_split_isolation(records)

    tampered = [copy.deepcopy(record) for record in records]
    tampered[-1]["split"] = "ood" if tampered[-1]["split"] != "ood" else "train"

    with pytest.raises(DatasetError, match="跨 split"):
        check_split_isolation(tampered)


def test_split_isolation_detects_injected_source_group_split(tmp_path: Path) -> None:
    _, _, _, _, _, records, _ = _workspace_and_records(tmp_path)
    tampered = [copy.deepcopy(record) for record in records]
    tampered[-1]["metadata"]["source_group_id"] = "shared-group"
    tampered[0]["metadata"]["source_group_id"] = "shared-group"

    with pytest.raises(DatasetError, match="跨 split"):
        check_split_isolation(tampered)


def test_state_leakage_detects_state_built_from_next_bar(tmp_path: Path) -> None:
    _, _, _, params, _, records, bars_by_segment = _workspace_and_records(tmp_path)
    check_state_leakage(
        records, bars_by_segment=bars_by_segment, price_precision=params.price_precision
    )

    # 人为把某条记录的状态换成分片内"下一根"的比值行 → 必须被检出
    target = next(record for record in records if parse_state_id(record["state_id"])[1] == 0)
    segment_id, bar_index = parse_state_id(target["state_id"])
    segment_bars = bars_by_segment[segment_id]
    assert bar_index + 1 < len(segment_bars)
    tampered = [copy.deepcopy(record) for record in records]
    index = records.index(target)
    tampered[index]["state"] = render_state(
        bar=segment_bars[bar_index + 1],
        reference_bar=segment_bars[0],
        position=None,
        drawdown=0.0,
        price_precision=params.price_precision,
    )

    with pytest.raises(DatasetError, match="与决策 K 线不一致"):
        check_state_leakage(
            tampered, bars_by_segment=bars_by_segment, price_precision=params.price_precision
        )


def test_state_leakage_detects_absolute_price_in_state(tmp_path: Path) -> None:
    _, _, _, params, _, records, bars_by_segment = _workspace_and_records(tmp_path)
    segment_id, bar_index = parse_state_id(records[0]["state_id"])
    segment_bars = bars_by_segment[segment_id]
    bar = segment_bars[bar_index]

    with pytest.raises(DatasetError, match="绝对价格"):
        check_no_absolute_values(
            records[0]["state"] + f"\npx_raw: {bar.close:.6f}",
            bar,
            precision=params.price_precision,
        )

    # 决策 K 线之后的邻根同样不得出现绝对价格
    tampered = [copy.deepcopy(record) for record in records]
    next_bar = segment_bars[bar_index + 1] if bar_index + 1 < len(segment_bars) else bar
    tampered[0]["state"] = tampered[0]["state"] + f"\nnext_close: {next_bar.close:.6f}"
    with pytest.raises(DatasetError, match="绝对价格"):
        check_state_leakage(
            tampered, bars_by_segment=bars_by_segment, price_precision=params.price_precision
        )


def test_check_records_detects_duplicate_id_and_bad_gold(tmp_path: Path) -> None:
    _, _, _, _, _, records, _ = _workspace_and_records(tmp_path)

    duplicated = [copy.deepcopy(record) for record in records] + [copy.deepcopy(records[0])]
    with pytest.raises(DatasetError, match="记录 ID 重复"):
        check_records(duplicated)

    bad_gold = [copy.deepcopy(record) for record in records]
    bad_gold[0]["gold"] = {"next_action": "open_twice"}
    with pytest.raises(DatasetError, match="必须是已提供的候选 ID"):
        check_records(bad_gold)


def test_audit_payload_totals_match_scenario(tmp_path: Path) -> None:
    _, _, _, _, result, records, _ = _workspace_and_records(tmp_path)

    totals = result.audit["totals"]
    assert totals["segments"] == 3
    assert totals["selected"] == len(records) == 6
    assert totals["stop_exits"] == 3
    assert totals["deaths"] == 0
    assert totals["decision_points"] == totals["selected"] + totals["excluded_flat"]
    assert sum(1 for row in records if row["gold"]["next_action"] == "open_long") == 3
    assert sum(1 for row in records if row["gold"]["next_action"] == "hold") == 3
    assert result.audit["params"]["flat_sample_band_minutes"] == 2
    assert result.audit["frozen_decisions"]["reference_price"] == "segment_first_bar_open"
    assert set(result.audit["outputs"]["split_files"]) == {
        f"{split}.jsonl" for split in SPLIT_ROLES
    }


def test_check_audit_consistency_detects_tampered_counts(tmp_path: Path) -> None:
    _, _, _, _, result, _, _ = _workspace_and_records(tmp_path)
    payload = copy.deepcopy(result.audit)
    payload["per_split"]["train"]["records"] += 1

    with pytest.raises(DatasetError, match="不一致"):
        check_audit_consistency(payload, _records_by_split(result))


def test_check_audit_consistency_detects_question_count_mismatch() -> None:
    payload = {
        "per_split": {split: {"records": 0, "questions": 0} for split in SPLIT_ROLES},
        "per_segment": [],
        "totals": {"selected": 0},
    }
    payload["per_split"]["train"]["questions"] = 1

    with pytest.raises(DatasetError, match="不一致"):
        check_audit_consistency(payload, {split: [] for split in SPLIT_ROLES})


def test_verify_determinism_reports_identical_sha256(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, _ROWS)
    segments = load_segments(workspace.manifest)
    symbols = load_symbols_config(workspace.symbols_path)
    params = EpisodeParams()

    digests = verify_determinism(
        segments,
        symbols=symbols,
        data_dir=workspace.data_dir,
        params=params,
        work_root=tmp_path / "determinism",
    )

    direct = generate_dataset(
        segments,
        symbols=symbols,
        data_dir=workspace.data_dir,
        params=params,
        output_dir=tmp_path / "direct",
    )
    assert digests == dict(direct.output_sha256)
    assert "train.jsonl" in digests and "audit.json" in digests


def test_build_audit_payload_marks_forced_closes(tmp_path: Path) -> None:
    """片段末强平必须出现在审计里（否则复算无法对齐）。"""
    workspace = build_workspace(tmp_path, _HOLDING_ROWS)
    segments = load_segments(workspace.manifest)
    symbols = load_symbols_config(workspace.symbols_path)
    params = EpisodeParams()
    result = generate_dataset(
        segments,
        symbols=symbols,
        data_dir=workspace.data_dir,
        params=params,
        output_dir=tmp_path / "out",
    )

    payload = result.audit
    assert payload["schema"] == "marketsense.episode_audit.v1"
    # 审计中的 bar 序号均为**片段内**序号（与 state_id 的决策 K 线标识同口径）
    assert [entry["segment_end_forced_close"] for entry in payload["per_segment"]] == [1, 1, 1]
    assert payload["totals"]["segment_end_forced_closes"] == 3
    assert payload["input"]["segments_sha256"]
    rendered = (result.run_dir / "audit.json").read_text(encoding="utf-8")
    assert json.loads(rendered) == payload


def test_build_audit_payload_has_no_wall_clock_or_environment_values(tmp_path: Path) -> None:
    """审计文件不含墙钟时间/绝对路径，保证双跑逐字节一致（输入指纹只含内容 hash）。"""
    _, _, _, _, result, _, _ = _workspace_and_records(tmp_path)

    rendered = json.dumps(result.audit, ensure_ascii=False, sort_keys=True)
    assert str(tmp_path) not in rendered
    assert "timestamp" not in result.audit
    assert "generated_at" not in result.audit


def test_state_of_flat_minute_uses_bars_up_to_its_own_index() -> None:
    """复核审计泄漏检查用的独立重算口径（分母 = 片段首根，分子 = 决策 K 线）。"""
    bar_list = bars(_ROWS[:3])
    state = render_state(
        bar=bar_list[2],
        reference_bar=bar_list[0],
        position=None,
        drawdown=0.0,
        price_precision=6,
    )

    assert "px_ratio: o=1.009000 h=1.012000 l=0.990000 c=0.992000" in state


def test_audit_records_source_data_version_per_segment(tmp_path: Path) -> None:
    """TEST 补充：每条片段的原始数据版本必须写进审计（否则产物无法追溯到底层行情）。

    背景：REVIEW P3-1 记录 ``run_id`` 只由清单 + 参数派生，**不含**行情数据指纹，
    因此数据更新会原地覆盖同一 run 目录。本用例只断言“数据身份仍可从审计追溯”
    这一不变式（更换 CSV 后 ``source_data_version`` 必须变化），不断言 run_id 稳定性。
    """
    rows_b = [(1000.0, 1002.0, 998.0, 1001.0)] * len(_ROWS)
    workspace = build_workspace(tmp_path, _ROWS)
    segments = load_segments(workspace.manifest)
    symbols = load_symbols_config(workspace.symbols_path)
    params = EpisodeParams()

    def generate(name: str):
        return generate_dataset(
            segments,
            symbols=symbols,
            data_dir=workspace.data_dir,
            params=params,
            output_dir=tmp_path / name,
        )

    first = generate("out-a")
    before = {
        entry["segment_id"]: entry["source_data_version"]
        for entry in first.audit["per_segment"]
    }
    assert before and all(value and "sha256=" in value for value in before.values())

    # 只替换底层行情 CSV（清单与参数不变）
    save_ohlcv(frame(rows_b), symbol=SYMBOL, period="1m", output_dir=workspace.data_dir)
    second = generate("out-b")
    after = {
        entry["segment_id"]: entry["source_data_version"]
        for entry in second.audit["per_segment"]
    }

    assert set(after) == set(before)
    assert after != before  # 数据身份变化必须反映在审计里
    assert second.audit["input"]["segments_sha256"] == first.audit["input"]["segments_sha256"]

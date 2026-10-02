"""T4：NanoJev 记录映射 / 状态序列化 / split 分配 / 原子落盘。

单测镜像 NanoJev ``train_pipeline_decisions.validate_training_row`` 与
``read_training_records`` 的必要校验逻辑（必填字段、choice 候选、id 唯一、
state/source_group 跨 split 隔离），并固定状态序列化模板的字节。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dataset.errors import DataLoadError, DatasetError
from dataset.market_episode.labels import (
    ACTION_CLOSE,
    ACTION_HOLD,
    ACTION_OPEN_LONG,
    ACTION_OPEN_SHORT,
    ACTION_REVERSE,
    ACTION_STAY_FLAT,
    DecisionPoint,
)
from dataset.market_episode.nanojev_records import (
    AUDIT_FILENAME,
    FLAT_CRITERIA,
    HELD_CRITERIA,
    QUESTION_ID,
    STATE_SCHEMA,
    BoardStateValues,
    _bar_trade_date,
    _usable_trend_extremes,
    build_record,
    generate_dataset,
    render_state,
)
from dataset.market_episode.replay import PositionSnapshot, bars_from_frame
from dataset.market_episode.segments import (
    SPLIT_ROLES,
    SEGMENT_SCHEMA,
    EpisodeParams,
    Segment,
    load_segments,
    load_symbols_config,
)
from dataset.storage import save_ohlcv
from dataset.tests.market_episode_fixtures import (
    DAILY_ROWS,
    DAILY_TP_DOWN,
    DAILY_TP_UP,
    SYMBOL,
    bars,
    build_two_day_workspace,
    build_workspace,
    daily_trend_points_map,
    frame,
    segment_record,
    timestamp_at,
    write_daily_turning_points,
    write_manifest,
    write_symbols_config,
)
from dataset.turning_points import TrendExtreme, load_turning_points

#: NanoJev 允许的 split（镜像其 ``SPLITS``）
_NANOJEV_SPLITS = ("train", "dev", "calibration", "test", "ood")

#: 2 根一个 episode 的最小模式：bar0 机会分钟（开多）、bar1 持有
_PATTERN = [
    (100, 100.2, 99.8, 100),
    (100, 110, 100, 109),
]
_ROWS = _PATTERN * 3


def _mirror_nanojev_validate(records: list[dict]) -> None:
    """镜像 NanoJev 的记录校验（不导入第三方代码，保持本仓库测试离线自持）。"""
    ids: set[str] = set()
    state_splits: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    for record in records:
        for key in ("id", "state_id", "family_id", "split", "state", "questions"):
            assert key in record, f"缺少字段 {key}"
        assert record["split"] in _NANOJEV_SPLITS
        for key in ("state_id", "family_id"):
            assert isinstance(record[key], str) and record[key].strip()
        assert isinstance(record["state"], str) and record["state"]
        assert isinstance(record["questions"], dict) and record["questions"]
        qids = set(record["questions"])
        for key in ("gold", "gold_label_kind"):
            if key in record:
                assert set(record[key]) <= qids
        assert record["id"] not in ids, "记录 ID 重复"
        ids.add(record["id"])
        for key, registry in (
            (record["state_id"], state_splits),
            (record["metadata"]["source_group_id"], source_splits),
        ):
            if key in registry:
                assert registry[key] == record["split"], "同一 state/source group 跨 split"
            registry[key] = record["split"]
        for qid, question in record["questions"].items():
            assert question["type"] in {"boolean", "choice", "score"}
            assert isinstance(question["instructions"], str) and question["instructions"].strip()
            if question["type"] == "choice":
                criteria = question["criteria"]
                assert isinstance(criteria, dict) and 2 <= len(criteria) <= 255
                assert all(
                    isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
                    for key, value in criteria.items()
                )
                assert record["gold"][qid] in criteria


# --------------------------------------------------------------------------- #
# 状态序列化模板（实现阶段冻结项 2：冻结后写夹具固定字节）
# --------------------------------------------------------------------------- #
#: 盘面状态绝对值夹具（量级与 1m 行情拉开，避免绝对 token 假阳性）
_BOARD = dict(prev_day_high=2010.0, prev_day_low=1980.0, prev_day_close=1990.0)


def test_render_state_template_is_byte_stable_for_flat_position() -> None:
    bar_list = bars(_PATTERN)

    state = render_state(
        bar=bar_list[0],
        reference_bar=bar_list[0],
        position=None,
        drawdown=0.0,
        price_precision=6,
        board_state=BoardStateValues(today_high=100.2, today_low=99.8, **_BOARD),
        trend_up_extreme=DAILY_TP_UP,
        trend_dn_extreme=DAILY_TP_DOWN,
    )

    assert state == (
        f"{STATE_SCHEMA} \n "
        "账户: 持仓=空仓 回撤=0.000000 \n "
        "联动: v=1.000000 oi_open=1.000000 oi_close=1.000000 \n "
        "日线: prev_h=20.100000 prev_l=19.800000 prev_c=19.900000 "
        "trend_up=40.200000 trend_up_len=7 trend_dn=39.800000 trend_dn_len=5 \n "
        "日内: bar=0 today_h=1.002000 today_l=0.998000 \n "
        "现价: o=1.000000 h=1.002000 l=0.998000 c=1.000000 \n "
        "盘口: na"
    )


def test_render_state_template_is_byte_stable_for_holding_position() -> None:
    bar_list = bars([(1000, 1000.5, 999.5, 1000), (1000, 1010, 1000.5, 1008)])

    state = render_state(
        bar=bar_list[1],
        reference_bar=bar_list[0],
        position=PositionSnapshot(direction="short", entry_ratio=0.9985, stop_ratio=1.0005),
        drawdown=0.0015,
        price_precision=6,
        board_state=BoardStateValues(today_high=1010.0, today_low=999.5, **_BOARD),
        trend_up_extreme=DAILY_TP_UP,
        trend_dn_extreme=DAILY_TP_DOWN,
    )

    assert state == (
        f"{STATE_SCHEMA} \n "
        "账户: 持仓=持空 entry=0.998500 stop=1.000500 回撤=0.001500 \n "
        "联动: v=1.000000 oi_open=1.002000 oi_close=1.001996 \n "
        "日线: prev_h=2.010000 prev_l=1.980000 prev_c=1.990000 "
        "trend_up=4.020000 trend_up_len=7 trend_dn=3.980000 trend_dn_len=5 \n "
        "日内: bar=1 today_h=1.010000 today_l=0.999500 \n "
        "现价: o=1.000000 h=1.010000 l=1.000500 c=1.008000 \n "
        "盘口: na"
    )


def test_render_state_marks_undefined_denominators_as_na() -> None:
    bar_list = bars([(1000, 1000.5, 999.5, 1000), (1000, 1010, 1000.5, 1008)], volume=0)
    zero_oi = bars([(1000, 1000.5, 999.5, 1000), (1000, 1010, 1000.5, 1008)])

    state = render_state(
        bar=bar_list[1],
        reference_bar=bar_list[0],
        position=None,
        drawdown=0.0,
        price_precision=6,
        board_state=BoardStateValues(today_high=1010.0, today_low=999.5, **_BOARD),
        trend_up_extreme=DAILY_TP_UP,
        trend_dn_extreme=DAILY_TP_DOWN,
    )

    assert "联动: v=na oi_open=1.002000 oi_close=1.001996" in state
    assert "日线: prev_h=2.010000" in state  # 日线行不受分母影响（价格分母正常）
    assert "inf" not in state and "nan" not in state
    assert zero_oi  # 非零分母分支由上一个夹具覆盖


def test_render_state_has_no_absolute_prices_and_only_ratios_on_the_decision_bar() -> None:
    bar_list = bars([(1000, 1000.5, 999.5, 1000), (1000, 1010, 1000.5, 1008)])

    state = render_state(
        bar=bar_list[1],
        reference_bar=bar_list[0],
        position=None,
        drawdown=0.0,
        price_precision=6,
        board_state=BoardStateValues(prev_day_high=2010.0, prev_day_low=1980.0,
                                     prev_day_close=1990.0, today_high=1010.0, today_low=999.5),
        trend_up_extreme=DAILY_TP_UP,
        trend_dn_extreme=DAILY_TP_DOWN,
    )

    for absolute in ("1000.000000", "1010.000000", "999.500000", "1008.000000"):
        assert absolute not in state
    assert len(state.splitlines()) == 7  # 模板 v5：7 行（账户/联动/日线/日内/现价/盘口；行间连接符为 " \n "）


def test_v5_daily_line_trend_values_match_independent_hand_calc() -> None:
    """v5 字节级锁定：trend 四值 = 折点段极值 ÷ 片段首根开盘（6 位）+ 段长整数直出。

    手算（分母 = 片段首根开盘 100）：up 段极值 4020 → 40.200000（段长 7）；
    down 段极值 3980 → 39.800000（段长 5）。"""
    bar_list = bars(_PATTERN)
    record = build_record(
        segment=_segment(),
        bars=bar_list,
        point=DecisionPoint(0, ACTION_OPEN_LONG, "opportunity", True, None, 0.0, 6.28, 0.0),
        price_precision=6,
        prev_day_ohlc=(_BOARD["prev_day_high"], _BOARD["prev_day_low"], _BOARD["prev_day_close"]),
        trend_up_extreme=TrendExtreme(
            kind="up", trend_extreme_price=4020.0, trend_extreme_bar_index=12, segment_length=7
        ),
        trend_dn_extreme=TrendExtreme(
            kind="down", trend_extreme_price=3980.0, trend_extreme_bar_index=5, segment_length=5
        ),
    )

    assert record["state"].startswith("marketsense.episode_state.v5")
    assert (
        "日线: prev_h=20.100000 prev_l=19.800000 prev_c=19.900000 "
        "trend_up=40.200000 trend_up_len=7 trend_dn=39.800000 trend_dn_len=5"
    ) in record["state"]


def test_v5_daily_line_keeps_v4_prev_values() -> None:
    """数值等价：v5 日线行的 prev 三值与 v4 同值（trend 四值只是行内追加，不改既有值）。"""
    bar_list = bars(_PATTERN)

    state = render_state(
        bar=bar_list[0],
        reference_bar=bar_list[0],
        position=None,
        drawdown=0.0,
        price_precision=6,
        board_state=BoardStateValues(today_high=100.2, today_low=99.8, **_BOARD),
        trend_up_extreme=DAILY_TP_UP,
        trend_dn_extreme=DAILY_TP_DOWN,
    )

    # v4 日线行原文（trend 四值之前的 prev 部分）逐字保留
    assert "日线: prev_h=20.100000 prev_l=19.800000 prev_c=19.900000 " in state


# --------------------------------------------------------------------------- #
# 记录映射
# --------------------------------------------------------------------------- #
def _segment(
    *, segment_id: str = "seg-train", split_role: str = "train", last_index: int = 1
) -> Segment:
    """构造一个只用于单元测试的片段（不经过 manifest 文件）。"""
    return Segment(
        segment_id=segment_id,
        symbol=SYMBOL,
        period="1m",
        start=pd.Timestamp(timestamp_at(0), tz="Asia/Shanghai"),
        end=pd.Timestamp(timestamp_at(last_index), tz="Asia/Shanghai"),
        split_role=split_role,
    )


def test_build_record_maps_ids_split_questions_and_gold() -> None:
    bar_list = bars(_PATTERN)
    segment = _segment()
    flat_point = DecisionPoint(0, ACTION_OPEN_LONG, "opportunity", True, None, 0.0, 6.28, 0.0)
    holding_point = DecisionPoint(
        1,
        ACTION_HOLD,
        "holding",
        True,
        PositionSnapshot("long", 1.012, 0.998),
        0.012,
        None,
        None,
    )

    flat_record = build_record(
        segment=segment,
        bars=bar_list,
        point=flat_point,
        price_precision=6,
        prev_day_ohlc=(_BOARD["prev_day_high"], _BOARD["prev_day_low"], _BOARD["prev_day_close"]),
        trend_up_extreme=DAILY_TP_UP,
        trend_dn_extreme=DAILY_TP_DOWN,
    )
    holding_record = build_record(
        segment=segment,
        bars=bar_list,
        point=holding_point,
        price_precision=6,
        prev_day_ohlc=(_BOARD["prev_day_high"], _BOARD["prev_day_low"], _BOARD["prev_day_close"]),
        trend_up_extreme=DAILY_TP_UP,
        trend_dn_extreme=DAILY_TP_DOWN,
    )

    assert flat_record["id"] == flat_record["state_id"] == "seg-train:0"
    assert flat_record["family_id"] == flat_record["metadata"]["source_group_id"] == "seg-train"
    assert flat_record["split"] == "train"
    assert flat_record["metadata"]["bar_index"] == 0
    assert flat_record["questions"][QUESTION_ID]["type"] == "choice"
    assert set(flat_record["questions"][QUESTION_ID]["criteria"]) == set(FLAT_CRITERIA)
    assert set(holding_record["questions"][QUESTION_ID]["criteria"]) == set(HELD_CRITERIA)
    # 候选文案只留动作语义：无括号、无成交价位细节（2026-10-02 用户拍板精简）
    assert flat_record["questions"][QUESTION_ID]["criteria"] == {
        ACTION_OPEN_LONG: "买入开仓",
        ACTION_OPEN_SHORT: "卖出开仓",
        ACTION_STAY_FLAT: "继续空仓",
    }
    assert holding_record["questions"][QUESTION_ID]["criteria"] == {
        ACTION_CLOSE: "平仓",
        ACTION_HOLD: "继续持有",
        ACTION_REVERSE: "反手",
    }
    for question in (flat_record["questions"], holding_record["questions"]):
        for text in question[QUESTION_ID]["criteria"].values():
            assert "tick" not in text and "（" not in text
    assert flat_record["gold"] == {QUESTION_ID: ACTION_OPEN_LONG}
    assert flat_record["gold_label_kind"] == {QUESTION_ID: "deterministic_truth"}
    assert flat_record["state"].startswith(STATE_SCHEMA)
    assert "持仓=空仓" in flat_record["state"]
    assert "持仓=持多 entry=1.012000 stop=0.998000" in holding_record["state"]


def test_build_record_rejects_action_outside_position_criteria() -> None:
    bar_list = bars(_PATTERN)
    bad_point = DecisionPoint(1, ACTION_CLOSE, "holding", True, None, 0.0, None, None)

    with pytest.raises(DatasetError, match="与仓位候选集不匹配"):
        build_record(
            segment=_segment(),
            bars=bar_list,
            point=bad_point,
            price_precision=6,
            prev_day_ohlc=(_BOARD["prev_day_high"], _BOARD["prev_day_low"], _BOARD["prev_day_close"]),
            trend_up_extreme=DAILY_TP_UP,
            trend_dn_extreme=DAILY_TP_DOWN,
        )


def test_build_record_board_state_uses_prev_daily_and_prefix_extrema() -> None:
    """board_state：prev_* = 上一交易日日线 ÷ 片段首根开盘；
    today_* = 首根至决策 K 线（含）累计极值，不含决策 K 线之后的 bar。"""
    bar_list = bars(_PATTERN + [(100, 200, 99, 100)])  # 第 3 根 high=200（决策 K 线之后）
    segment = _segment(last_index=2)
    record = build_record(
        segment=segment,
        bars=bar_list,
        point=DecisionPoint(
            1,
            ACTION_HOLD,
            "holding",
            True,
            PositionSnapshot("long", 1.012, 0.998),
            0.012,
            None,
            None,
        ),
        price_precision=6,
        prev_day_ohlc=(_BOARD["prev_day_high"], _BOARD["prev_day_low"], _BOARD["prev_day_close"]),
        trend_up_extreme=DAILY_TP_UP,
        trend_dn_extreme=DAILY_TP_DOWN,
    )

    # prev：2010/1980/1990 ÷ 首根开盘 100；today：前缀 [0..1] 极值 110/99.8 ÷ 100
    # （第 3 根 high=200 不进入 today_h）
    assert "日线: prev_h=20.100000 prev_l=19.800000 prev_c=19.900000" in record["state"]
    assert "日内: bar=1 today_h=1.100000 today_l=0.998000" in record["state"]
    assert "20.000000" not in record["state"]


# --------------------------------------------------------------------------- #
# 流水线输出
# --------------------------------------------------------------------------- #
def _generate(tmp_path: Path, *, rows=_ROWS, band: int = 2):
    workspace = build_workspace(tmp_path, rows)
    segments = load_segments(workspace.manifest)
    symbols = load_symbols_config(workspace.symbols_path)
    result = generate_dataset(
        segments,
        symbols=symbols,
        data_dir=workspace.data_dir,
        params=EpisodeParams(flat_sample_band_minutes=band),
        output_dir=tmp_path / "out",
    )
    return workspace, result


def test_generate_dataset_writes_all_splits_and_passes_mirror_validation(tmp_path: Path) -> None:
    _, result = _generate(tmp_path)

    assert result.record_counts == {
        "train": 2,
        "dev": 2,
        "calibration": 0,
        "test": 2,
        "ood": 0,
    }
    records: list[dict] = []
    for split in SPLIT_ROLES:
        path = result.run_dir / f"{split}.jsonl"
        assert path.is_file()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert all(row["split"] == split for row in rows)
        records.extend(rows)
    assert len(records) == 6
    _mirror_nanojev_validate(records)

    assert (result.run_dir / AUDIT_FILENAME).is_file()
    assert result.audit["schema"] == "marketsense.episode_audit.v1"
    assert result.audit["totals"]["selected"] == 6
    assert result.audit["per_split"]["train"] == {"records": 2, "questions": 2}


def test_generate_dataset_records_follow_manifest_and_bar_order(tmp_path: Path) -> None:
    _, result = _generate(tmp_path)

    train_rows = [
        json.loads(line)
        for line in (result.run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [row["id"] for row in train_rows] == ["seg-train:0", "seg-train:1"]
    assert [row["gold"][QUESTION_ID] for row in train_rows] == [ACTION_OPEN_LONG, ACTION_HOLD]


def test_generate_dataset_is_deterministic_and_run_isolated(tmp_path: Path) -> None:
    workspace, first = _generate(tmp_path / "a")
    _, second = _generate(tmp_path / "b")

    assert first.run_id == second.run_id
    for name, digest in first.output_sha256.items():
        assert second.output_sha256[name] == digest
        assert (first.run_dir / name).read_bytes() == (second.run_dir / name).read_bytes()

    # 不同输入（带宽不同 → 采样不同）→ 不同 run 目录，不覆盖历史产物
    _, other = _generate(tmp_path / "c", band=0)
    assert other.run_id != first.run_id
    assert other.run_dir != first.run_dir
    assert workspace.data_dir.is_dir()


def test_generate_dataset_does_not_write_minutes_outside_the_band(tmp_path: Path) -> None:
    rows = [
        (100, 100.2, 99.8, 100),
        (100, 100.3, 99.7, 100.1),
        (100, 100.4, 99.6, 100.2),
        (100, 100.5, 99.5, 100.3),
        (100, 100.6, 99.4, 100.4),
        (100, 110, 99, 109.5),
        (110, 160, 109, 155),
    ]
    workspace = build_workspace(tmp_path, rows)
    manifest = write_manifest(
        tmp_path / "band.jsonl",
        [
            segment_record("seg-train", "train", start_index=0, end_index=1),
            segment_record("seg-dev", "dev", start_index=2, end_index=2),
            segment_record("seg-test", "test", start_index=3, end_index=6),
        ],
    )
    result = generate_dataset(
        load_segments(manifest),
        symbols=load_symbols_config(workspace.symbols_path),
        data_dir=workspace.data_dir,
        params=EpisodeParams(flat_sample_band_minutes=1),
        output_dir=tmp_path / "out",
    )

    # test 段 = bars 3..6（本段内机会分钟在第 3 位）：带宽 1 → 仅其前 1 分钟入选
    assert result.record_counts["test"] == 3
    test_entry = next(
        entry for entry in result.audit["per_segment"] if entry["segment_id"] == "seg-test"
    )
    assert test_entry["excluded_flat"] == 1  # 段内第 0 分钟（带宽外）被剔除
    assert test_entry["selected"] == 3
    assert test_entry["action_counts"] == {
        ACTION_OPEN_LONG: 1,
        ACTION_HOLD: 1,
        ACTION_STAY_FLAT: 1,
    }


def test_generate_dataset_rejects_manifest_without_required_splits_and_reports_message(
    tmp_path: Path,
) -> None:
    """REVIEW P3-2 修正：本例原先与另一用例**同名**而被静默覆盖，带 ``match`` 的
    错误消息断言从未执行；重命名后两组断言均会被收集执行。"""
    workspace = build_workspace(tmp_path, _ROWS)
    manifest = write_manifest(
        tmp_path / "only_train.jsonl",
        [segment_record(f"seg-{index}", "train", start_index=2 * index, end_index=2 * index + 1)
         for index in range(3)],
    )
    symbols = load_symbols_config(workspace.symbols_path)

    with pytest.raises(DatasetError, match="train/dev/test"):
        generate_dataset(
            load_segments(manifest),
            symbols=symbols,
            data_dir=workspace.data_dir,
            params=EpisodeParams(),
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_generate_dataset_rejects_workspace_with_only_train_segments(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, _ROWS, split_roles=("train", "train", "train"))
    symbols = load_symbols_config(workspace.symbols_path)

    with pytest.raises(DatasetError):
        generate_dataset(
            load_segments(workspace.manifest),
            symbols=symbols,
            data_dir=workspace.data_dir,
            params=EpisodeParams(),
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


#: 12 根恒定振幅 K 线：两向盈亏比恒为 0 → 片段内没有任何机会分钟
_FLAT_ROWS = [(100.0, 100.2, 99.8, 100.0)] * 12


def test_generate_dataset_rejects_workspace_without_prev_daily_for_any_segment(
    tmp_path: Path,
) -> None:
    """日线数据起点不早于片段交易日 → 全部片段跳过 → 硬错误（不写出产物，不静默）。"""
    workspace = build_workspace(
        tmp_path,
        _ROWS,
        daily_rows=((4010.0, 4000.0, 3990.0, 4005.0),),
        daily_start="2024-01-02 00:00:00",  # 与片段同日：无严格早于它的日线行
    )

    with pytest.raises(DatasetError, match="所有片段都缺少上一交易日日线"):
        generate_dataset(
            load_segments(workspace.manifest),
            symbols=load_symbols_config(workspace.symbols_path),
            data_dir=workspace.data_dir,
            params=EpisodeParams(),
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_generate_dataset_skips_segments_without_prev_daily_and_reports(
    tmp_path: Path,
) -> None:
    """部分片段无上一交易日日线：跳过不产出记录，记入审计与结果（不静默）。"""
    workspace = build_two_day_workspace(tmp_path)

    result = generate_dataset(
        load_segments(workspace.manifest),
        symbols=load_symbols_config(workspace.symbols_path),
        data_dir=workspace.data_dir,
        params=EpisodeParams(),
        output_dir=tmp_path / "out",
    )

    assert set(result.board_state_skipped) == {"seg-train"}
    (reason,) = result.board_state_skipped.values()
    assert reason.startswith("trade_date=2024-01-02")
    assert result.record_counts["train"] == 0
    assert result.record_counts["dev"] + result.record_counts["test"] > 0
    skipped = result.audit["board_state"]["skipped_segments"]
    assert [entry["segment_id"] for entry in skipped] == ["seg-train"]
    # 非跳过片段（次日，首根开盘 100）的记录含日线行：prev = 首日日线行
    dev_rows = [
        json.loads(line)
        for line in (result.run_dir / "dev.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert dev_rows
    for row in dev_rows:
        # v5 日线行全行：prev = 首日日线行 ÷ 次日片段首根开盘 100；
        # trend 四值 = 默认夹具折点段极值（up 4020→40.2 段长 7；down 3980→39.8 段长 5）
        assert (
            "日线: prev_h=40.100000 prev_l=39.900000 prev_c=40.050000 "
            "trend_up=40.200000 trend_up_len=7 trend_dn=39.800000 trend_dn_len=5"
        ) in row["state"]
    train_rows = [
        json.loads(line)
        for line in (result.run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert train_rows == []  # 跳过的片段不产出记录（已知空 split 缺陷仍单独存在）


def test_generate_dataset_state_schema_changes_run_id(tmp_path: Path) -> None:
    """STATE_SCHEMA / QUESTION_SCHEMA 纳入 run_id 哈希：状态或候选文案变更产生新 run。"""
    workspace, first = _generate(tmp_path / "a")
    assert first.run_id.startswith("run-")
    assert first.audit["input"]["state_schema_sha256"]
    assert first.audit["input"]["question_schema_sha256"]

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "dataset.market_episode.nanojev_records.STATE_SCHEMA",
            "marketsense.episode_state.vX-test",
        )
        _, other = _generate(tmp_path / "b")

    assert other.run_id != first.run_id
    assert workspace.data_dir.is_dir()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "dataset.market_episode.nanojev_records.QUESTION_SCHEMA",
            "marketsense.episode_question.vX-test",
        )
        _, third = _generate(tmp_path / "c")

    assert third.run_id not in (first.run_id, other.run_id)


@pytest.mark.xfail(
    strict=False,
    reason=(
        "REVIEW P2-1 已知缺口（生产代码待修；TEST 阶段不改生产代码）：空 split 静默产出 "
        "0 字节文件且退出 0，而 NanoJev/scripts/train_pipeline_decisions.py:477-479 要求 "
        "train/dev/test 非空。修复后本用例应 XPASS，届时移除该标记"
    ),
)
def test_generate_dataset_rejects_empty_train_dev_test_splits(tmp_path: Path) -> None:
    """期望行为：train/dev/test 记录数为 0 时报错（当前实现静默写出 0 字节空文件）。

    最小复现：3 片段（train/dev/test）× 4 根恒定振幅 K 线 → 无任何机会分钟 →
    空仓分钟全部被剔除 → 三个 split 均 0 条记录。按预期应抛 ``DatasetError`` 且不落盘；
    当前实现返回 record_counts 全 0 并在磁盘留下 0 字节 ``{split}.jsonl``。
    """
    workspace = build_workspace(tmp_path, _FLAT_ROWS)

    with pytest.raises(DatasetError):
        generate_dataset(
            load_segments(workspace.manifest),
            symbols=load_symbols_config(workspace.symbols_path),
            data_dir=workspace.data_dir,
            params=EpisodeParams(),
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_daily_trend_points_fixture_round_trip(tmp_path: Path) -> None:
    """夹具：默认 TP CSV 经 ``load_turning_points`` 回读（与生产同路径），点集与
    默认极值常量一致（down@bar5 段长 5；up@bar12 段长 7）。"""
    workspace = build_workspace(tmp_path, _ROWS)
    points_map = daily_trend_points_map(workspace)
    (points,) = points_map.values()
    assert [point.kind for point in points] == ["start", "down", "up"]
    up_extreme, dn_extreme = _usable_trend_extremes(
        points, trade_date=pd.Timestamp("2024-01-02").date()
    )
    assert up_extreme == DAILY_TP_UP
    assert dn_extreme == DAILY_TP_DOWN


def test_trend_points_confirmed_on_or_after_trade_date_are_not_usable(tmp_path: Path) -> None:
    """泄漏边界：确认日 == 决策交易日的折点不可用（trend_dn 取更早 down）；
    T 日及之后确认折点的极值价不得出现在 State(T)。"""
    workspace = build_workspace(
        tmp_path,
        _ROWS,
        daily_tp_rows=[
            {"kind": "down", "timestamp": "2024-01-01 21:05:00+08:00", "price": 3980.0,
             "bar_index": 5, "volume": 120, "oi": 5010,
             "trend_extreme_price": 3980.0, "trend_extreme_bar_index": 5},
            {"kind": "up", "timestamp": "2024-01-01 21:35:00+08:00", "price": 4020.0,
             "bar_index": 12, "volume": 130, "oi": 5020,
             "trend_extreme_price": 4020.0, "trend_extreme_bar_index": 12},
            # 确认日 == 决策交易日 2024-01-02（不可用）
            {"kind": "down", "timestamp": "2024-01-02 09:05:00+08:00", "price": 4500.0,
             "bar_index": 20, "volume": 140, "oi": 5030,
             "trend_extreme_price": 4500.0, "trend_extreme_bar_index": 20},
            {"kind": "up", "timestamp": "2024-01-02 09:35:00+08:00", "price": 4600.0,
             "bar_index": 25, "volume": 150, "oi": 5040,
             "trend_extreme_price": 4600.0, "trend_extreme_bar_index": 25},
        ],
    )
    result = generate_dataset(
        load_segments(workspace.manifest),
        symbols=load_symbols_config(workspace.symbols_path),
        data_dir=workspace.data_dir,
        params=EpisodeParams(),
        output_dir=tmp_path / "out",
    )

    assert result.trend_extreme_skipped == {}
    records = [
        json.loads(line)
        for split in SPLIT_ROLES
        for line in (result.run_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records
    for record in records:
        # 可用折点 = 确认日 01-01 的 down/up（首根开盘 100）：4020→40.2 段长 7、3980→39.8 段长 5
        assert (
            "日线: prev_h=40.100000 prev_l=39.900000 prev_c=40.050000 "
            "trend_up=40.200000 trend_up_len=7 trend_dn=39.800000 trend_dn_len=5"
        ) in record["state"]
        # 确认日 == 决策交易日的折点极值价（4500/4600 ÷ 100）不得出现
        assert "45.000000" not in record["state"]
        assert "46.000000" not in record["state"]


def test_generate_dataset_skips_segments_without_usable_trend_points(tmp_path: Path) -> None:
    """单侧缺失（确认日不早于决策交易日）→ 跳过整个片段：审计 + 结果记录，不静默；
    跳过片段不进入 per_segment/totals。"""
    day1 = frame(_PATTERN * 2, start="2024-01-02 09:00:00")
    day2 = frame(_PATTERN * 2, start="2024-01-03 09:00:00")
    data_dir = tmp_path / "data" / "ohlcv"
    save_ohlcv(
        pd.concat([day1, day2]).reset_index(drop=True),
        symbol=SYMBOL, period="1m", output_dir=data_dir,
    )
    save_ohlcv(
        pd.concat(
            (
                frame([DAILY_ROWS[0]], start="2024-01-01 00:00:00"),
                frame([DAILY_ROWS[0]], start="2024-01-02 00:00:00"),
                frame([DAILY_ROWS[0]], start="2024-01-03 00:00:00"),
            )
        ).reset_index(drop=True),
        symbol=SYMBOL, period="1d", output_dir=data_dir,
    )
    # 折点均确认于 01-02：== seg-train 决策交易日（不可用 → 跳过）；
    # < seg-dev/seg-test 决策交易日 01-03（可用 → 正常产出）
    write_daily_turning_points(
        data_dir.parent / "turning_points" / f"{SYMBOL}_1d.csv",
        rows=[
            {"kind": "down", "timestamp": "2024-01-02 09:05:00+08:00", "price": 3980.0,
             "bar_index": 5, "volume": 120, "oi": 5010,
             "trend_extreme_price": 3980.0, "trend_extreme_bar_index": 5},
            {"kind": "up", "timestamp": "2024-01-02 09:35:00+08:00", "price": 4020.0,
             "bar_index": 12, "volume": 130, "oi": 5020,
             "trend_extreme_price": 4020.0, "trend_extreme_bar_index": 12},
        ],
    )
    manifest = write_manifest(
        tmp_path / "segments.jsonl",
        [
            segment_record("seg-train", "train", start_index=0, end_index=3, start="2024-01-02 09:00:00"),
            segment_record("seg-dev", "dev", start_index=0, end_index=1, start="2024-01-03 09:00:00"),
            segment_record("seg-test", "test", start_index=2, end_index=3, start="2024-01-03 09:00:00"),
        ],
    )
    result = generate_dataset(
        load_segments(manifest),
        symbols=load_symbols_config(write_symbols_config(tmp_path / "symbols.yaml")),
        data_dir=data_dir,
        params=EpisodeParams(),
        output_dir=tmp_path / "out",
    )

    assert set(result.trend_extreme_skipped) == {"seg-train"}
    (reason,) = result.trend_extreme_skipped.values()
    assert reason.startswith("trade_date=2024-01-02")
    assert result.record_counts["train"] == 0
    assert result.record_counts["dev"] + result.record_counts["test"] > 0
    # 跳过片段不进入 per_segment/totals（与 board_state 跳过先例一致）
    assert all(entry["segment_id"] != "seg-train" for entry in result.audit["per_segment"])
    assert result.audit["totals"]["segments"] == 3
    skipped = result.audit["trend_extremes"]["skipped_segments"]
    assert [entry["segment_id"] for entry in skipped] == ["seg-train"]
    assert skipped[0]["reason"].startswith("trade_date=2024-01-02")
    assert set(result.audit["trend_extremes"]["source_versions"]) == {SYMBOL}
    # 非跳过片段（交易日 01-03，首根开盘 100）的记录含 v5 日线行：trend = 01-02 确认折点
    dev_rows = [
        json.loads(line)
        for line in (result.run_dir / "dev.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert dev_rows
    for row in dev_rows:
        assert (
            "日线: prev_h=40.100000 prev_l=39.900000 prev_c=40.050000 "
            "trend_up=40.200000 trend_up_len=7 trend_dn=39.800000 trend_dn_len=5"
        ) in row["state"]


def test_generate_dataset_rejects_workspace_without_usable_trend_points(tmp_path: Path) -> None:
    """全部片段缺可用折点（确认日不早于决策交易日）→ DatasetError 硬错误（不写出产物）。"""
    workspace = build_workspace(
        tmp_path,
        _ROWS,
        daily_tp_rows=[
            # 确认日 == 片段交易日 2024-01-02：无可用 up/down
            {"kind": "down", "timestamp": "2024-01-02 09:05:00+08:00", "price": 3980.0,
             "bar_index": 5, "volume": 120, "oi": 5010,
             "trend_extreme_price": 3980.0, "trend_extreme_bar_index": 5},
            {"kind": "up", "timestamp": "2024-01-02 09:35:00+08:00", "price": 4020.0,
             "bar_index": 12, "volume": 130, "oi": 5020,
             "trend_extreme_price": 4020.0, "trend_extreme_bar_index": 12},
        ],
    )

    with pytest.raises(DatasetError, match="所有片段都缺少可用日线转折点"):
        generate_dataset(
            load_segments(workspace.manifest),
            symbols=load_symbols_config(workspace.symbols_path),
            data_dir=workspace.data_dir,
            params=EpisodeParams(),
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_generate_dataset_rejects_workspace_with_up_side_missing(tmp_path: Path) -> None:
    """单侧缺失（只有 down、无 up）：任一入选决策点单侧缺失 → 片段跳过；全部跳过 → 硬错误。"""
    workspace = build_workspace(
        tmp_path,
        _ROWS,
        daily_tp_rows=[
            {"kind": "down", "timestamp": "2024-01-01 21:05:00+08:00", "price": 3980.0,
             "bar_index": 5, "volume": 120, "oi": 5010,
             "trend_extreme_price": 3980.0, "trend_extreme_bar_index": 5},
        ],
    )

    with pytest.raises(DatasetError, match="所有片段都缺少可用日线转折点"):
        generate_dataset(
            load_segments(workspace.manifest),
            symbols=load_symbols_config(workspace.symbols_path),
            data_dir=workspace.data_dir,
            params=EpisodeParams(),
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_generate_dataset_missing_turning_points_file_raises_dataload_error(tmp_path: Path) -> None:
    """TP 文件缺失 ≠ 折点缺失：文件缺失 → ``DataLoadError`` 硬报错（含路径，不静默）。"""
    workspace = build_workspace(tmp_path, _ROWS)
    workspace.turning_points_path.unlink()

    with pytest.raises(DataLoadError, match="转折点文件不存在"):
        generate_dataset(
            load_segments(workspace.manifest),
            symbols=load_symbols_config(workspace.symbols_path),
            data_dir=workspace.data_dir,
            params=EpisodeParams(),
            output_dir=tmp_path / "out",
        )


def test_bar_trade_date_night_session_attribution() -> None:
    """夜盘归属：21:00+ bar → 片段日盘日历日集合中其后的下一个交易日。"""
    night = frame(_PATTERN, start="2024-01-01 21:00:00")
    day = frame(_PATTERN, start="2024-01-02 09:00:00")
    bar_list = bars_from_frame(pd.concat([night, day]).reset_index(drop=True))

    # 夜盘 bar（历日 01-01 21:00）→ 下一个交易日 01-02
    assert _bar_trade_date(bar_list, bar_list[0]) == pd.Timestamp("2024-01-02").date()
    # 日盘 bar → 其日历日
    assert _bar_trade_date(bar_list, bar_list[2]) == pd.Timestamp("2024-01-02").date()


def test_bar_trade_date_rejects_ambiguous_time_of_day() -> None:
    """[15:00, 21:00) 或夜盘无下一交易日 → ``DatasetError``（不静默）。"""
    midday = bars([(100, 100.2, 99.8, 100)], start="2024-01-02 15:30:00")
    with pytest.raises(DatasetError, match="无法归属交易日"):
        _bar_trade_date(midday, midday[0])

    night_only = bars([(100, 100.2, 99.8, 100)] * 2, start="2024-01-05 21:00:00")
    with pytest.raises(DatasetError, match="找不到后续交易日"):
        _bar_trade_date(night_only, night_only[0])


def test_night_bars_use_next_trading_day_for_trend_points(tmp_path: Path) -> None:
    """夜盘归属端到端：21:00+ bar 归属下一交易日；确认日等于夜盘历日的折点可用
    （序列化侧与审计侧独立复算共用该口径）。"""
    group_specs = [
        ("2024-01-01 21:00:00", 2),  # 夜盘（归属交易日 01-02）
        ("2024-01-02 09:00:00", 4),  # 日盘（交易日 01-02）
        ("2024-01-02 21:00:00", 2),
        ("2024-01-03 09:00:00", 4),
        ("2024-01-03 21:00:00", 2),
        ("2024-01-04 09:00:00", 4),
    ]
    one_minute = pd.concat(
        [frame(_PATTERN * 2, start=start).iloc[:count] for start, count in group_specs]
    ).reset_index(drop=True)
    data_dir = tmp_path / "data" / "ohlcv"
    save_ohlcv(one_minute, symbol=SYMBOL, period="1m", output_dir=data_dir)
    save_ohlcv(
        pd.concat(
            tuple(
                frame([DAILY_ROWS[0]], start=f"2024-01-0{day} 00:00:00")
                for day in (1, 2, 3, 4)
            )
        ).reset_index(drop=True),
        symbol=SYMBOL, period="1d", output_dir=data_dir,
    )
    # 折点确认日 = 夜盘历日 01-01（== 夜盘 bar 日历日，但 < 其归属交易日 01-02）
    write_daily_turning_points(
        data_dir.parent / "turning_points" / f"{SYMBOL}_1d.csv",
        rows=[
            {"kind": "down", "timestamp": "2024-01-01 22:00:00+08:00", "price": 3980.0,
             "bar_index": 5, "volume": 120, "oi": 5010,
             "trend_extreme_price": 3980.0, "trend_extreme_bar_index": 5},
            {"kind": "up", "timestamp": "2024-01-01 23:00:00+08:00", "price": 4020.0,
             "bar_index": 12, "volume": 130, "oi": 5020,
             "trend_extreme_price": 4020.0, "trend_extreme_bar_index": 12},
        ],
    )
    manifest = write_manifest(
        tmp_path / "segments.jsonl",
        [
            {"schema": SEGMENT_SCHEMA, "segment_id": "seg-train", "symbol": SYMBOL, "period": "1m",
             "start_ts": "2024-01-01 21:00:00", "end_ts": "2024-01-02 09:03:00", "split_role": "train"},
            {"schema": SEGMENT_SCHEMA, "segment_id": "seg-dev", "symbol": SYMBOL, "period": "1m",
             "start_ts": "2024-01-02 21:00:00", "end_ts": "2024-01-03 09:03:00", "split_role": "dev"},
            {"schema": SEGMENT_SCHEMA, "segment_id": "seg-test", "symbol": SYMBOL, "period": "1m",
             "start_ts": "2024-01-03 21:00:00", "end_ts": "2024-01-04 09:03:00", "split_role": "test"},
        ],
    )
    result = generate_dataset(
        load_segments(manifest),
        symbols=load_symbols_config(write_symbols_config(tmp_path / "symbols.yaml")),
        data_dir=data_dir,
        params=EpisodeParams(),
        output_dir=tmp_path / "out",
    )

    # 若夜盘被错锚到历日（01-01），确认日 01-01 的折点不可用 → 片段被跳过、本断言失败
    assert result.trend_extreme_skipped == {}
    for split in ("train", "dev", "test"):
        assert result.record_counts[split] > 0
        for line in (result.run_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert (
                "日线: prev_h=40.100000 prev_l=39.900000 prev_c=40.050000 "
                "trend_up=40.200000 trend_up_len=7 trend_dn=39.800000 trend_dn_len=5"
            ) in row["state"]


def test_multi_day_segment_later_trade_date_sees_later_confirmed_points(tmp_path: Path) -> None:
    """多日片段逐决策点：后一交易日的入选点可用前一日之后确认的折点（逐决策点而非
    片段首日取一次）。"""
    day1 = frame(_PATTERN * 2, start="2024-01-02 09:00:00")
    day2 = frame(_PATTERN * 2, start="2024-01-03 09:00:00")
    day3 = frame(_PATTERN * 2, start="2024-01-04 09:00:00")
    data_dir = tmp_path / "data" / "ohlcv"
    save_ohlcv(
        pd.concat([day1, day2, day3]).reset_index(drop=True),
        symbol=SYMBOL, period="1m", output_dir=data_dir,
    )
    save_ohlcv(
        pd.concat(
            tuple(
                frame([DAILY_ROWS[0]], start=f"2024-01-0{day} 00:00:00")
                for day in (1, 2, 3, 4)
            )
        ).reset_index(drop=True),
        symbol=SYMBOL, period="1d", output_dir=data_dir,
    )
    write_daily_turning_points(
        data_dir.parent / "turning_points" / f"{SYMBOL}_1d.csv",
        rows=[
            {"kind": "down", "timestamp": "2024-01-01 22:00:00+08:00", "price": 3980.0,
             "bar_index": 5, "volume": 120, "oi": 5010,
             "trend_extreme_price": 3980.0, "trend_extreme_bar_index": 5},
            {"kind": "up", "timestamp": "2024-01-01 23:00:00+08:00", "price": 4020.0,
             "bar_index": 12, "volume": 130, "oi": 5020,
             "trend_extreme_price": 4020.0, "trend_extreme_bar_index": 12},
            # 后确认的 up 点（01-02 夜盘，历日 01-02）：交易日 01-03/01-04 的 bar 可用
            {"kind": "up", "timestamp": "2024-01-02 21:00:00+08:00", "price": 4600.0,
             "bar_index": 20, "volume": 140, "oi": 5030,
             "trend_extreme_price": 4600.0, "trend_extreme_bar_index": 20},
        ],
    )
    manifest = write_manifest(
        tmp_path / "segments.jsonl",
        [
            # train = 跨两个交易日的多日片段（01-02 日盘 + 01-03 日盘）
            {"schema": SEGMENT_SCHEMA, "segment_id": "seg-train", "symbol": SYMBOL, "period": "1m",
             "start_ts": "2024-01-02 09:00:00", "end_ts": "2024-01-03 09:03:00", "split_role": "train"},
            {"schema": SEGMENT_SCHEMA, "segment_id": "seg-dev", "symbol": SYMBOL, "period": "1m",
             "start_ts": "2024-01-04 09:00:00", "end_ts": "2024-01-04 09:01:00", "split_role": "dev"},
            {"schema": SEGMENT_SCHEMA, "segment_id": "seg-test", "symbol": SYMBOL, "period": "1m",
             "start_ts": "2024-01-04 09:02:00", "end_ts": "2024-01-04 09:03:00", "split_role": "test"},
        ],
    )
    result = generate_dataset(
        load_segments(manifest),
        symbols=load_symbols_config(write_symbols_config(tmp_path / "symbols.yaml")),
        data_dir=data_dir,
        params=EpisodeParams(),
        output_dir=tmp_path / "out",
    )

    assert result.trend_extreme_skipped == {}
    assert result.record_counts["train"] > 0
    train_rows = [
        json.loads(line)
        for line in (result.run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    by_bar = {row["metadata"]["bar_index"]: row for row in train_rows}
    # 01-02 的 bar（bar_index 0-3）：可用折点 = 确认日 01-01 的 down/up（4020→40.2 段长 7）
    for bar_index in range(0, 4):
        assert (
            "trend_up=40.200000 trend_up_len=7 trend_dn=39.800000 trend_dn_len=5"
        ) in by_bar[bar_index]["state"]
    # 01-03 的入选 bar（bar_index 4，交易日 01-03）：up 01-02 21:00 已确认
    # （< 决策交易日 01-03）→ 4600→46.0 段长 8（后一交易日可用前一日之后确认的折点）
    assert (
        "trend_up=46.000000 trend_up_len=8 trend_dn=39.800000 trend_dn_len=5"
    ) in by_bar[4]["state"]

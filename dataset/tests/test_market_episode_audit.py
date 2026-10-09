"""T5（审计）：确定性双跑、跨 split 隔离、状态泄漏抽查、计数汇总一致性。

审计自检必须能**检出人为构造的坏数据**（注入跨 split / 下一根状态 / 绝对价格）。
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from dataset.errors import DatasetError
from dataset.market_episode.audit import (
    NET_VALUE_BASE,
    AccountReplayInputs,
    check_audit_consistency,
    check_no_absolute_values,
    check_outcome_sidecar,
    check_records,
    check_split_isolation,
    check_state_leakage,
    parse_state_id,
    verify_determinism,
)
from dataset.market_episode.labels import evaluate_segment
from dataset.market_episode.nanojev_records import (
    OUTCOMES_FILENAME,
    BoardStateValues,
    generate_dataset,
    render_state,
)
from dataset.market_episode.replay import bars_from_frame, load_segment_bars
from dataset.market_episode.segments import (
    SPLIT_ROLES,
    EpisodeParams,
    load_segments,
    load_symbols_config,
)
from dataset.periods import resolve_duration_seconds
from dataset.storage import save_ohlcv
from dataset.tests.market_episode_fixtures import (
    DAILY_ROWS,
    DAILY_T_ROW,
    RENDER_TREND_CONTEXT,
    SYMBOL,
    bars,
    build_workspace,
    daily_rows_map,
    daily_trend_points_map,
    frame,
    prev_daily_map,
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


def _workspace_and_records(tmp_path: Path, *, linkage_symbols: tuple[str, ...] = ()):
    workspace = build_workspace(tmp_path, _ROWS)
    if linkage_symbols:
        save_ohlcv(
            frame(_ROWS), symbol=linkage_symbols[0], period="1m", output_dir=workspace.data_dir
        )
        save_ohlcv(
            pd.concat([
                frame([DAILY_ROWS[0]], start="2024-01-01 00:00:00"),
                frame([DAILY_T_ROW], start="2024-01-02 00:00:00"),
            ]).reset_index(drop=True),
            symbol=linkage_symbols[0], period="1d", output_dir=workspace.data_dir,
        )
    symbols = load_symbols_config(workspace.symbols_path)
    segments = load_segments(workspace.manifest)
    params = EpisodeParams(linkage_symbols=linkage_symbols)
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


def _symbols_by_segment(workspace) -> dict[str, str]:
    """segment_id → symbol（``check_state_leakage`` 新必参；从片段清单确定性派生）。"""
    return {segment.segment_id: segment.symbol for segment in load_segments(workspace.manifest)}


def _bars_by_symbol_for_linkage(workspace, bars_by_segment, symbols) -> dict[str, dict[str, tuple]]:
    """联动品种审计入参：联动 symbol → segment_id → 片段序列。"""
    return {symbol: dict(bars_by_segment) for symbol in symbols}


def _bars_by_symbol(
    workspace, bars_by_segment
) -> dict[str, dict[str, tuple]]:
    """symbol → segment_id → 片段完整 1m 序列（v10 联动行独立复算入参，两级容器）。

    构造方式与生成侧同构（``generate_dataset``：
    ``bars_by_symbol.setdefault(segment.symbol, {})[segment.segment_id] = bars``）。
    """
    mapping: dict[str, dict[str, tuple]] = {}
    for segment_id, symbol in _symbols_by_segment(workspace).items():
        mapping.setdefault(symbol, {})[segment_id] = bars_by_segment[segment_id]
    return mapping


def _account_inputs_by_segment(
    segments, symbols, params, bars_by_segment
) -> dict[str, AccountReplayInputs]:
    """按时间序串行重放 ``evaluate_segment``（与 ``generate_dataset`` 处理顺序同口径）

    重建逐片段账户复算入参（事件轨迹 + 处理 bar 数 + 链起点净值/峰值快照；
    首片段起点 = ``NET_VALUE_BASE``，逐片段 = 上一片段末结算净值/峰值，
    与生成侧净值链 carry 同源同值）。
    """
    carry_net = NET_VALUE_BASE
    carry_peak = NET_VALUE_BASE
    inputs: dict[str, AccountReplayInputs] = {}
    for segment in sorted(segments, key=lambda s: (s.start, s.end, s.segment_id)):
        outcome = evaluate_segment(
            segment,
            bars_by_segment[segment.segment_id],
            tick_size=float(symbols[segment.symbol]),
            params=params,
            initial_equity=carry_net / NET_VALUE_BASE,
            initial_peak=carry_peak / NET_VALUE_BASE,
        )
        inputs[segment.segment_id] = AccountReplayInputs(
            events=outcome.events,
            processed_bar_count=outcome.processed_bar_count,
            initial_net_value=carry_net,
            initial_peak_net_value=carry_peak,
        )
        carry_net = NET_VALUE_BASE * outcome.final_equity
        carry_peak = NET_VALUE_BASE * outcome.final_peak
    return inputs


def test_parse_state_id_round_trip_and_rejects_bad_format() -> None:
    assert parse_state_id("seg-1:12") == ("seg-1", 12)

    with pytest.raises(DatasetError, match="state_id 格式非法"):
        parse_state_id("seg-1")
    with pytest.raises(DatasetError, match="决策 K 线序号非法"):
        parse_state_id("seg-1:abc")
    with pytest.raises(DatasetError, match="必须 ≥ 0"):
        parse_state_id("seg-1:-1")


def test_generated_records_pass_all_audit_checks(tmp_path: Path) -> None:
    workspace, _, _, params, result, records, bars_by_segment = _workspace_and_records(tmp_path)

    check_records(records)
    check_state_leakage(
        records,
        bars_by_segment=bars_by_segment,
        price_precision=params.price_precision,
        prev_daily_by_segment=prev_daily_map(workspace),
        trend_points_by_symbol=daily_trend_points_map(workspace),
        daily_rows_by_symbol=daily_rows_map(workspace),
        symbols_by_segment=_symbols_by_segment(workspace),
        bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
        breakthrough_window=params.breakthrough_window,
        breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
        linkage_bars_by_symbol=(
            _bars_by_symbol_for_linkage(workspace, bars_by_segment, params.linkage_symbols)
            if params.linkage_symbols
            else None
        ),
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


def test_linkage_audit_accepts_fixed_labels_and_detects_line_tampering(tmp_path: Path) -> None:
    workspace, _, _, params, _, records, bars_by_segment = _workspace_and_records(
        tmp_path, linkage_symbols=("INE.sc2611",)
    )
    linkage_bars = _bars_by_symbol_for_linkage(workspace, bars_by_segment, params.linkage_symbols)
    audit_args = dict(
        bars_by_segment=bars_by_segment,
        price_precision=params.price_precision,
        prev_daily_by_segment=prev_daily_map(workspace),
        trend_points_by_symbol=daily_trend_points_map(workspace),
        daily_rows_by_symbol=daily_rows_map(workspace),
        symbols_by_segment=_symbols_by_segment(workspace),
        bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
        breakthrough_window=params.breakthrough_window,
        breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
        linkage_bars_by_symbol=linkage_bars,
        linkage_daily_rows_by_symbol=daily_rows_map(workspace),
    )
    check_state_leakage(records, **audit_args)
    row = records[0]["state"].splitlines()[4].strip()
    assert row.startswith("联动: 1dK相关度=")
    assert "1mK相关度=" in row
    assert "参考（突破=" not in row
    tampered = copy.deepcopy(records)
    tampered[0]["state"] = tampered[0]["state"].replace("1dK相关度=na", "1dK相关度=0.999999", 1)
    with pytest.raises(DatasetError, match="联动行"):
        check_state_leakage(tampered, **audit_args)

    tampered = copy.deepcopy(records)
    tampered[0]["state"] = tampered[0]["state"].replace("1mK相关度=na", "1mK相关度=0.999999", 1)
    with pytest.raises(DatasetError, match="联动行"):
        check_state_leakage(tampered, **audit_args)

    tampered = copy.deepcopy(records)
    tampered[0]["state"] = tampered[0]["state"].replace("相关度=na", "相关度=0.000000", 1)
    with pytest.raises(DatasetError, match="联动行"):
        check_state_leakage(tampered, **audit_args)


def test_independent_1m_linkage_correlation_truncates_before_intersection() -> None:
    """超过窗口的片段中，窗口前信号不能改变 1m Pearson 相关度。"""
    from dataset.market_episode.audit import _independent_linkage_1m_correlation

    primary_signals = (1, -1, -1, 1, 1)
    secondary_signals = (-1, 1, -1, 1, 1)

    def rows_for_signals(signals: tuple[int, ...]) -> list[tuple[float, float, float, float]]:
        rows = [(100.0, 100.0, 100.0, 100.0)]
        for signal in signals:
            previous_open, previous_high, previous_low, previous_close = rows[-1]
            if signal > 0:
                rows.append((previous_close, previous_high + 1, previous_low, previous_close + 1))
            else:
                rows.append((previous_close, previous_high, previous_low - 1, previous_close - 1))
        return rows

    primary_bars = bars(rows_for_signals(primary_signals))
    secondary_bars = bars(rows_for_signals(secondary_signals))
    window = 4  # Four bars yield three in-window signals; two earlier signals are excluded.

    assert len(primary_bars) > window
    window_r = _independent_linkage_1m_correlation(primary_bars, secondary_bars, window)
    full_prefix_r = _independent_linkage_1m_correlation(
        primary_bars, secondary_bars, len(primary_bars)
    )
    assert window_r == 1.0
    assert full_prefix_r != window_r


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
    workspace, _, _, params, _, records, bars_by_segment = _workspace_and_records(tmp_path)
    check_state_leakage(
        records,
        bars_by_segment=bars_by_segment,
        price_precision=params.price_precision,
        prev_daily_by_segment=prev_daily_map(workspace),
        trend_points_by_symbol=daily_trend_points_map(workspace),
        daily_rows_by_symbol=daily_rows_map(workspace),
        symbols_by_segment=_symbols_by_segment(workspace),
        bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
        breakthrough_window=params.breakthrough_window,
        breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
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
        net_value=100.0,
        today_pnl=0.0,
        price_precision=params.price_precision,
        board_state=BoardStateValues(
            prev_day_high=2010.0,
            prev_day_low=1980.0,
            prev_day_close=1990.0,
            today_high=max(bar.high for bar in segment_bars[: bar_index + 2]),
            today_low=min(bar.low for bar in segment_bars[: bar_index + 2]),
        ),
        trend_context=RENDER_TREND_CONTEXT,
    )

    with pytest.raises(DatasetError, match="与决策 K 线不一致"):
        check_state_leakage(
            tampered,
            bars_by_segment=bars_by_segment,
            price_precision=params.price_precision,
            prev_daily_by_segment=prev_daily_map(workspace),
            trend_points_by_symbol=daily_trend_points_map(workspace),
            daily_rows_by_symbol=daily_rows_map(workspace),
            symbols_by_segment=_symbols_by_segment(workspace),
            bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
            breakthrough_window=params.breakthrough_window,
            breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
        )


def test_state_leakage_detects_absolute_price_in_state(tmp_path: Path) -> None:
    workspace, _, _, params, _, records, bars_by_segment = _workspace_and_records(tmp_path)
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
            tampered,
            bars_by_segment=bars_by_segment,
            price_precision=params.price_precision,
            prev_daily_by_segment=prev_daily_map(workspace),
            trend_points_by_symbol=daily_trend_points_map(workspace),
            daily_rows_by_symbol=daily_rows_map(workspace),
            symbols_by_segment=_symbols_by_segment(workspace),
            bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
            breakthrough_window=params.breakthrough_window,
            breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
        )


def test_state_leakage_detects_wrong_board_state_line(tmp_path: Path) -> None:
    """日内行与独立重算不一致（今高篡改）→ 必须被检出。"""
    workspace, _, _, params, _, records, bars_by_segment = _workspace_and_records(tmp_path)
    check_state_leakage(
        records,
        bars_by_segment=bars_by_segment,
        price_precision=params.price_precision,
        prev_daily_by_segment=prev_daily_map(workspace),
        trend_points_by_symbol=daily_trend_points_map(workspace),
        daily_rows_by_symbol=daily_rows_map(workspace),
        symbols_by_segment=_symbols_by_segment(workspace),
        bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
        breakthrough_window=params.breakthrough_window,
        breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
    )

    tampered = [copy.deepcopy(record) for record in records]
    assert "今高=" in tampered[0]["state"]
    # 人为把日内行的今高换成片段首根以外的值
    tampered[0]["state"] = re.sub(
        r"今高=[\d.]+", "今高=9.999999", tampered[0]["state"], count=1
    )

    with pytest.raises(DatasetError, match="日内行与决策 K 线不一致"):
        check_state_leakage(
            tampered,
            bars_by_segment=bars_by_segment,
            price_precision=params.price_precision,
            prev_daily_by_segment=prev_daily_map(workspace),
            trend_points_by_symbol=daily_trend_points_map(workspace),
            daily_rows_by_symbol=daily_rows_map(workspace),
            symbols_by_segment=_symbols_by_segment(workspace),
            bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
            breakthrough_window=params.breakthrough_window,
            breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
        )


def test_state_leakage_detects_prev_daily_absolute_price(tmp_path: Path) -> None:
    """上一交易日日线绝对价格不得出现在状态文本（只能以比值出现）。"""
    workspace, _, _, params, _, records, bars_by_segment = _workspace_and_records(tmp_path)
    prev_daily = prev_daily_map(workspace)[parse_state_id(records[0]["state_id"])[0]]

    tampered = [copy.deepcopy(record) for record in records]
    tampered[0]["state"] = (
        tampered[0]["state"] + f"\nprev_close_raw: {prev_daily[2]:.{params.price_precision}f}"
    )
    with pytest.raises(DatasetError, match="上一交易日绝对价格"):
        check_state_leakage(
            tampered,
            bars_by_segment=bars_by_segment,
            price_precision=params.price_precision,
            prev_daily_by_segment=prev_daily_map(workspace),
            trend_points_by_symbol=daily_trend_points_map(workspace),
            daily_rows_by_symbol=daily_rows_map(workspace),
            symbols_by_segment=_symbols_by_segment(workspace),
            bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
            breakthrough_window=params.breakthrough_window,
            breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
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
    # train/dev = 事件 + 前看（每段只利 bar0 开仓事件，bar1 hold → excluded_holding、
    # bar3 空仓 → excluded_flat）；test = 平带（bar0 + bar1 入选，bar3 剩除）
    assert totals["selected"] == len(records) == 4
    assert totals["stop_exits"] == 3
    assert totals["deaths"] == 0
    assert totals["decision_points"] == 9
    assert (
        totals["decision_points"]
        == totals["selected"] + totals["excluded_flat"] + totals["excluded_holding"]
    )
    assert sum(1 for row in records if row["gold"]["next_action"] == "open_long") == 3
    assert sum(1 for row in records if row["gold"]["next_action"] == "hold") == 1
    assert result.audit["params"]["flat_sample_band_minutes"] == 2
    assert result.audit["params"]["event_lookforward_open"] == 2
    assert result.audit["params"]["event_lookforward_exit"] == 10
    assert result.audit["frozen_decisions"]["reference_price"] == "segment_first_bar_open"
    assert set(result.audit["outputs"]["split_files"]) == {
        f"{split}.jsonl" for split in SPLIT_ROLES
    }


def test_audit_per_segment_records_excluded_holding(tmp_path: Path) -> None:
    """逐片段审计含 excluded_holding，且恒等式在片段级成立；train 与 test
    同型片段在双机制下产出不同计数（train 1 条、test 2 条）。"""
    _, _, _, _, result, _, _ = _workspace_and_records(tmp_path)

    per_segment = {entry["segment_id"]: entry for entry in result.audit["per_segment"]}
    assert per_segment["seg-train"]["selected"] == 1
    assert per_segment["seg-train"]["excluded_holding"] == 1
    assert per_segment["seg-train"]["excluded_flat"] == 1
    assert per_segment["seg-test"]["selected"] == 2
    assert per_segment["seg-test"]["excluded_holding"] == 0
    for entry in result.audit["per_segment"]:
        assert entry["decision_points"] == (
            entry["selected"] + entry["excluded_flat"] + entry["excluded_holding"]
        )


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
        "totals": {"selected": 0, "decision_points": 0, "excluded_flat": 0,
                   "excluded_holding": 0},
    }
    payload["per_split"]["train"]["questions"] = 1

    with pytest.raises(DatasetError, match="不一致"):
        check_audit_consistency(payload, {split: [] for split in SPLIT_ROLES})


def test_check_audit_consistency_detects_identity_violation() -> None:
    """恒等式硬门（两层级）：decision_points ≠ selected + excluded_flat +
    excluded_holding → DatasetError（per_segment 求和与 totals 各查一道）。"""
    base_totals = {"selected": 2, "decision_points": 3, "excluded_flat": 1,
                   "excluded_holding": 0}

    def _payload(totals: dict) -> dict:
        return {
            "per_split": {split: {"records": totals["selected"], "questions": totals["selected"]}
                          for split in SPLIT_ROLES},
            "per_segment": [],
            "totals": totals,
        }

    # totals 级：holding 计数被篡改 → 恒等式破裂
    tampered = dict(base_totals, excluded_holding=1)
    with pytest.raises(DatasetError, match="不一致"):
        check_audit_consistency(_payload(tampered), {split: [] for split in SPLIT_ROLES})

    # per_segment 级：totals 一致但片段级恒等式破裂 → 同样拒绝
    per_segment = [
        {"segment_id": f"seg-{i}", "decision_points": 3, "selected": 2,
         "excluded_flat": 1, "excluded_holding": 1}  # 3 ≠ 2 + 1 + 1
        for i in range(3)
    ]
    payload = {
        "per_split": {split: {"records": 6, "questions": 6} for split in SPLIT_ROLES},
        "per_segment": per_segment,
        "totals": {"selected": 6, "decision_points": 9, "excluded_flat": 3,
                   "excluded_holding": 0},
    }
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
    # v8 净值链（时间序串行回放）：三片段同型亏损使净值加性累计 100 → 97.8 → 95.6，
    # 第 3 片段 bar1 盯市权益 94.4 → 回撤 (100 − 94.4)/100 = 5.6% ≥ 阈值 5% → 死亡
    # 吸收态（检出当根强平、episode 结束、当根不产决策样本）→ 无片段末强平（None），
    # 死亡强平记入 deaths 而非 segment_end_forced_close
    assert [entry["segment_end_forced_close"] for entry in payload["per_segment"]] == [1, 1, None]
    assert payload["totals"]["segment_end_forced_closes"] == 2
    assert payload["totals"]["deaths"] == 1
    assert [entry["deaths"] for entry in payload["per_segment"]] == [[], [], [1]]
    assert [entry["death_forced_close"] for entry in payload["per_segment"]] == [[], [], [True]]
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
        net_value=100.0,
        today_pnl=0.0,
        price_precision=6,
        board_state=BoardStateValues(prev_day_high=2010.0, prev_day_low=1980.0,
                                     prev_day_close=1990.0, today_high=1012.0, today_low=990.0),
        trend_context=RENDER_TREND_CONTEXT,
    )

    assert (
        "现价: bar=2 价=0.992000 成交量比=1.000000 持仓量比=1.003992 1mK突破=na"
    ) in state


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


def test_state_leakage_detects_wrong_trend_line(tmp_path: Path) -> None:
    """v9 日线行趋势项篡改（涨势 -1 段极值比值）→ 独立重算不一致（日线行与决策
    K 线不一致）→ 必须被检出。"""
    import re as _re

    workspace, _, _, params, _, records, bars_by_segment = _workspace_and_records(tmp_path)
    check_state_leakage(
        records,
        bars_by_segment=bars_by_segment,
        price_precision=params.price_precision,
        prev_daily_by_segment=prev_daily_map(workspace),
        trend_points_by_symbol=daily_trend_points_map(workspace),
        daily_rows_by_symbol=daily_rows_map(workspace),
        symbols_by_segment=_symbols_by_segment(workspace),
        bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
        breakthrough_window=params.breakthrough_window,
        breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
    )

    tampered = [copy.deepcopy(record) for record in records]
    assert "涨势(-1, 最高=" in tampered[0]["state"]
    tampered[0]["state"] = _re.sub(
        r"涨势\(-1, 最高=[\d.]+", "涨势(-1, 最高=9.999999", tampered[0]["state"], count=1
    )

    with pytest.raises(DatasetError, match="日线行与决策 K 线不一致"):
        check_state_leakage(
            tampered,
            bars_by_segment=bars_by_segment,
            price_precision=params.price_precision,
            prev_daily_by_segment=prev_daily_map(workspace),
            trend_points_by_symbol=daily_trend_points_map(workspace),
            daily_rows_by_symbol=daily_rows_map(workspace),
            symbols_by_segment=_symbols_by_segment(workspace),
            bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
            breakthrough_window=params.breakthrough_window,
            breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
        )

    tampered_order = [copy.deepcopy(record) for record in records]
    parts = tampered_order[0]["state"].split(" \n ")
    daily_items = parts[2].split(" ")
    daily_items[1], daily_items[2] = daily_items[2], daily_items[1]
    parts[2] = " ".join(daily_items)
    tampered_order[0]["state"] = " \n ".join(parts)
    with pytest.raises(DatasetError, match="日线行与决策 K 线不一致"):
        check_state_leakage(
            tampered_order,
            bars_by_segment=bars_by_segment,
            price_precision=params.price_precision,
            prev_daily_by_segment=prev_daily_map(workspace),
            trend_points_by_symbol=daily_trend_points_map(workspace),
            daily_rows_by_symbol=daily_rows_map(workspace),
            symbols_by_segment=_symbols_by_segment(workspace),
            bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
            breakthrough_window=params.breakthrough_window,
            breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
        )

    tampered_direction = [copy.deepcopy(record) for record in records]
    original_direction = (
        "当前为跌势" if "当前为跌势" in tampered_direction[0]["state"] else "当前为涨势"
    )
    tampered_direction[0]["state"] = tampered_direction[0]["state"].replace(
        original_direction,
        "当前为涨势" if original_direction == "当前为跌势" else "当前为跌势",
        1,
    )
    with pytest.raises(DatasetError, match="日线行与决策 K 线不一致"):
        check_state_leakage(
            tampered_direction,
            bars_by_segment=bars_by_segment,
            price_precision=params.price_precision,
            prev_daily_by_segment=prev_daily_map(workspace),
            trend_points_by_symbol=daily_trend_points_map(workspace),
            daily_rows_by_symbol=daily_rows_map(workspace),
            symbols_by_segment=_symbols_by_segment(workspace),
            bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
            breakthrough_window=params.breakthrough_window,
            breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
        )


def test_state_leakage_detects_trend_extreme_absolute_price(tmp_path: Path) -> None:
    """可用折点的绝对极值价不得出现在状态文本（与 prev_daily 绝对价同型硬门）。"""
    workspace, _, _, params, _, records, bars_by_segment = _workspace_and_records(tmp_path)

    tampered = [copy.deepcopy(record) for record in records]
    tampered[0]["state"] = tampered[0]["state"] + "\ntrend_raw: 4020.000000"
    with pytest.raises(DatasetError, match="可用折点绝对极值价"):
        check_state_leakage(
            tampered,
            bars_by_segment=bars_by_segment,
            price_precision=params.price_precision,
            prev_daily_by_segment=prev_daily_map(workspace),
            trend_points_by_symbol=daily_trend_points_map(workspace),
            daily_rows_by_symbol=daily_rows_map(workspace),
            symbols_by_segment=_symbols_by_segment(workspace),
            bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
            breakthrough_window=params.breakthrough_window,
            breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
        )


def test_audit_payload_has_trend_extremes_section(tmp_path: Path) -> None:
    """审计新增 trend_extremes 段：skipped_segments / source_versions（sha256 指纹）。"""
    from dataset.market_episode.audit import FROZEN_DECISIONS

    _, _, _, _, result, _, _ = _workspace_and_records(tmp_path)
    section = result.audit["trend_extremes"]
    assert section["skipped_segments"] == []
    assert set(section["source_versions"]) == {SYMBOL}
    assert section["source_versions"][SYMBOL].startswith("sha256=")
    assert (
        result.audit["frozen_decisions"]["state_template"]
        == FROZEN_DECISIONS["state_template"]
    )
    assert (
        result.audit["frozen_decisions"]["daily_trend_extremes"]
        == FROZEN_DECISIONS["daily_trend_extremes"]
    )


def test_independent_bar_trade_date_night_attribution() -> None:
    """审计侧独立交易日归属与序列化侧同口径：21:00+ bar → 下一交易日。"""
    from dataset.market_episode.audit import _independent_bar_trade_date

    night = frame(_HOLDING_PATTERN, start="2024-01-01 21:00:00")
    day = frame(_HOLDING_PATTERN, start="2024-01-02 09:00:00")
    bar_list = bars_from_frame(pd.concat([night, day]).reset_index(drop=True))

    # 夜盘 bar（历日 01-01 21:00）→ 下一个交易日 01-02
    assert _independent_bar_trade_date(bar_list, bar_list[0]) == pd.Timestamp("2024-01-02").date()
    # 日盘 bar → 其日历日
    assert _independent_bar_trade_date(bar_list, bar_list[2]) == pd.Timestamp("2024-01-02").date()


def test_state_leakage_account_line_independent_recompute(tmp_path: Path) -> None:
    """T6b 端到端冒烟：``generate_dataset`` 产出后，按时间序串行重放重建的账户入参
    （含净值链 carry，逐片段起点 100 → 97.8 → 95.6）独立复算账户行三值 →
    正常记录通过（无报错）；审计 ``account_chain`` 节在位。"""
    workspace, segments, symbols, params, result, records, bars_by_segment = (
        _workspace_and_records(tmp_path)
    )
    check_state_leakage(
        records,
        bars_by_segment=bars_by_segment,
        price_precision=params.price_precision,
        prev_daily_by_segment=prev_daily_map(workspace),
        trend_points_by_symbol=daily_trend_points_map(workspace),
        daily_rows_by_symbol=daily_rows_map(workspace),
        symbols_by_segment=_symbols_by_segment(workspace),
        bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
        breakthrough_window=params.breakthrough_window,
        breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
        account_inputs_by_segment=_account_inputs_by_segment(
            segments, symbols, params, bars_by_segment
        ),
    )
    # generate_dataset 已在内部通过同型检查（否则不落盘）；审计 account_chain 节在位
    for entry in result.audit["per_segment"]:
        assert entry["account_chain"]["initial_net_value"] > 0
        assert entry["account_chain"]["final_net_value"] > 0


def test_state_leakage_detects_tampered_account_line(tmp_path: Path) -> None:
    """T6b tamper 防护：账户行净值/今日/回撤任一值被篡改 → 必须报 DatasetError；
    正常记录（未篡改）通过；参数缺省（None）保持旧行为（账户行不参与比对，不报错）。"""
    workspace, segments, symbols, params, _, records, bars_by_segment = (
        _workspace_and_records(tmp_path)
    )
    account_inputs_by_segment = _account_inputs_by_segment(
        segments, symbols, params, bars_by_segment
    )
    kwargs = dict(
        bars_by_segment=bars_by_segment,
        price_precision=params.price_precision,
        prev_daily_by_segment=prev_daily_map(workspace),
        trend_points_by_symbol=daily_trend_points_map(workspace),
        daily_rows_by_symbol=daily_rows_map(workspace),
        symbols_by_segment=_symbols_by_segment(workspace),
        bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
        breakthrough_window=params.breakthrough_window,
        breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
    )

    # 正常记录（未篡改）→ 通过
    check_state_leakage(
        records, account_inputs_by_segment=account_inputs_by_segment, **kwargs
    )

    # 篡改账户行净值/今日/回撤任一值 → 独立复算不一致 → DatasetError
    for key in ("净值", "今日", "回撤"):
        tampered = [copy.deepcopy(record) for record in records]
        pattern = rf"{key}=[\d.]+"
        assert re.search(pattern, tampered[0]["state"])
        tampered[0]["state"] = re.sub(
            pattern, f"{key}=9.999999", tampered[0]["state"], count=1
        )
        with pytest.raises(DatasetError, match="账户行净值/今日/回撤与独立复算不一致"):
            check_state_leakage(
                tampered, account_inputs_by_segment=account_inputs_by_segment, **kwargs
            )

    # 参数缺省（None）→ 旧行为不变：被篡改的账户行不参与任何比对（不报错）
    tampered = [copy.deepcopy(record) for record in records]
    tampered[0]["state"] = re.sub(
        r"净值=[\d.]+", "净值=9.999999", tampered[0]["state"], count=1
    )
    check_state_leakage(tampered, **kwargs)


def test_state_leakage_linkage_line_independent_recompute(tmp_path: Path) -> None:
    """v11 无联动品种：主段 ``<主显示名>（突破=<主值>）`` 独立复算全部通过。

    期望值按夹具 K 线手算（_PATTERN 每段前 2 根：
    bar0=(1000, 1000.5, 999.5, 1000)、bar1=(1000, 1010, 1000, 1009)；主显示名 =
    夹具 symbol ``TEST.sym`` 去交易所前缀 → ``sym``）：
    - 片段首根（bar0）：可用根数 1 < 2 → ``突破=na``；
    - bar1：bar1.high=1010 > bar0.high=1000.5 且 bar1.close=1009 > bar0.close=1000
      → 信号 +1；n=2 → 单信号权重 1 → momentum = 1·(+1)/1 = 1.0 → ``突破=1.000000``。
    """
    workspace, _, _, params, _, records, bars_by_segment = _workspace_and_records(tmp_path)
    check_state_leakage(
        records,
        bars_by_segment=bars_by_segment,
        price_precision=params.price_precision,
        prev_daily_by_segment=prev_daily_map(workspace),
        trend_points_by_symbol=daily_trend_points_map(workspace),
        daily_rows_by_symbol=daily_rows_map(workspace),
        symbols_by_segment=_symbols_by_segment(workspace),
        bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
        breakthrough_window=params.breakthrough_window,
        breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
    )

    first = next(record for record in records if parse_state_id(record["state_id"])[1] == 0)
    second = next(record for record in records if parse_state_id(record["state_id"])[1] == 1)
    assert "联动: 1dK相关度=na 1mK相关度=na" in first["state"]
    assert "联动: 1dK相关度=na 1mK相关度=na" in second["state"]


def test_state_leakage_detects_tampered_linkage_line(tmp_path: Path) -> None:
    """v14 联动行篡改防护：相关度值/na 占位任一被篡改均拒绝。

    另锁定 ``bars_by_symbol`` 缺省 None 的 v11 行为：主品种突破值复算源回退
    ``bars_by_segment`` 的片段序列（生成侧同源同值）→ v11 格式记录仍通过；
    v9 旧格式（``联动: na`` 常量占位）不再兼容 → 报错。
    """
    workspace, _, _, params, _, records, bars_by_segment = _workspace_and_records(tmp_path)
    kwargs = dict(
        bars_by_segment=bars_by_segment,
        price_precision=params.price_precision,
        prev_daily_by_segment=prev_daily_map(workspace),
        trend_points_by_symbol=daily_trend_points_map(workspace),
        daily_rows_by_symbol=daily_rows_map(workspace),
        symbols_by_segment=_symbols_by_segment(workspace),
        bars_by_symbol=_bars_by_symbol(workspace, bars_by_segment),
        breakthrough_window=params.breakthrough_window,
        breakthrough_duration_seconds=resolve_duration_seconds(params.breakthrough_period),
    )

    # 篡改实算突破动量（bar1，期望 1.000000）→ 独立复算不一致 → DatasetError
    tampered = [copy.deepcopy(record) for record in records]
    second = next(record for record in tampered if parse_state_id(record["state_id"])[1] == 1)
    assert "1mK突破=1.000000" in second["state"]
    second["state"] = re.sub(r"1mK突破=[\d.]+", "1mK突破=9.999999", second["state"], count=1)
    with pytest.raises(DatasetError, match="现价行与决策 K 线不一致"):
        check_state_leakage(tampered, **kwargs)

    # 篡改片段首根 na 占位（bar0，期望 ``sym（突破=na）``）→ 期望 na 不再匹配 → DatasetError
    tampered = [copy.deepcopy(record) for record in records]
    first = next(record for record in tampered if parse_state_id(record["state_id"])[1] == 0)
    first["state"] = first["state"].replace("联动: 1dK相关度=na 1mK相关度=na", "联动: 1dK相关度=0.999999 1mK相关度=na", 1)
    with pytest.raises(DatasetError, match="联动行与决策 K 线不一致"):
        check_state_leakage(tampered, **kwargs)

    # v11：缺省 bars_by_symbol=None → 主品种突破值复算源回退 bars_by_segment 的
    # 片段序列（与生成侧同源同值）→ 原始 v11 格式记录在该缺省下仍通过
    check_state_leakage(records, **dict(kwargs, bars_by_symbol=None))

    # 同一批记录改回 v9 旧格式（``联动: na`` 常量占位）→ v11 期望主段
    # ``<主显示名>（突破=<值>）`` 不再匹配 → DatasetError（v9 格式兼容已随 v11 移除）
    v9_style = [copy.deepcopy(record) for record in records]
    for record in v9_style:
        record["state"] = re.sub(r"联动: \S+", "联动: na", record["state"], count=1)
    with pytest.raises(DatasetError, match="联动行与决策 K 线不一致"):
        check_state_leakage(v9_style, **dict(kwargs, bars_by_symbol=None))


# --------------------------------------------------------------------------- #
# utility-trainer（T2）：outcome 旁挂硬校验（独立复算 / 恒等式 / 覆盖）
# --------------------------------------------------------------------------- #

#: 与 audit 既有用例一致的最小模式：开多 → 持有 → 程序止损离场 → 空仓（每段 4 根）
_SIDE_PATTERN = [
    (100, 100.2, 99.8, 100),
    (100, 110, 100, 109),
    (100, 100.2, 99.8, 100),
    (100, 110, 100, 109),
]


def _generate_with_sidecar(tmp_path: Path):
    """生成含 outcome 旁挂的最小工作区（开仓记录 → 旁挂行 1:1）。"""
    workspace = build_workspace(tmp_path, _SIDE_PATTERN)
    segments = load_segments(workspace.manifest)
    symbols = load_symbols_config(workspace.symbols_path)
    result = generate_dataset(
        segments,
        symbols=symbols,
        data_dir=workspace.data_dir,
        params=EpisodeParams(flat_sample_band_minutes=2),
        output_dir=tmp_path / "out",
    )
    sidecar_rows = [
        json.loads(line)
        for line in (result.run_dir / OUTCOMES_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    records_by_split = {}
    for split in SPLIT_ROLES:
        path = result.run_dir / f"{split}.jsonl"
        records_by_split[split] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    return result, records_by_split, sidecar_rows, workspace


def _sidecar_check_inputs(workspace):
    """从工作区重建 check_outcome_sidecar 的独立输入（事件流 / K 线 / tick）。"""
    segments = load_segments(workspace.manifest)
    bars_by_segment = {
        segment.segment_id: load_segment_bars(segment, data_dir=workspace.data_dir)[0]
        for segment in segments
    }
    outcomes_by_segment = {
        segment.segment_id: evaluate_segment(
            segment,
            bars_by_segment[segment.segment_id],
            tick_size=1.0,
            params=EpisodeParams(flat_sample_band_minutes=2),
        )
        for segment in segments
    }
    return {
        "outcomes_by_segment": outcomes_by_segment,
        "bars_by_segment": bars_by_segment,
        "tick_size_by_segment": {segment_id: 1.0 for segment_id in outcomes_by_segment},
    }


def test_check_outcome_sidecar_happy_path_returns_node(tmp_path: Path) -> None:
    result, records_by_split, sidecar_rows, workspace = _generate_with_sidecar(tmp_path)
    assert sidecar_rows, "最小工作区应产生至少一条开仓旁挂行"

    node = check_outcome_sidecar(
        records_by_split=records_by_split,
        sidecar_rows=sidecar_rows,
        **_sidecar_check_inputs(workspace),
    )

    assert set(node) == {"by_split", "per_segment"}
    assert node["by_split"] == result.outcome_row_counts
    assert sum(entry["open_records"] for entry in node["per_segment"]) == len(sidecar_rows)
    for entry in node["per_segment"]:
        identity = entry["outcome_sum"] + entry["reverse_open_pnl_sum"]
        assert identity == pytest.approx(entry["realized_pnl_ratio"], abs=1e-9)


def test_check_outcome_sidecar_detects_tamper_and_coverage_break(tmp_path: Path) -> None:
    result, records_by_split, sidecar_rows, workspace = _generate_with_sidecar(tmp_path)
    kwargs = dict(
        records_by_split=records_by_split,
        **_sidecar_check_inputs(workspace),
    )

    # 篡改 outcome 值 → 独立复算逐值比对检出
    tampered = [dict(row) for row in sidecar_rows]
    tampered[0] = {**tampered[0], "outcome": tampered[0]["outcome"] * 2}
    with pytest.raises(DatasetError, match="outcome 与独立复算不一致"):
        check_outcome_sidecar(sidecar_rows=tampered, **kwargs)

    # 篡改 exit_reason → 精确比对检出
    tampered = [dict(row) for row in sidecar_rows]
    tampered[0] = {**tampered[0], "exit_reason": "death"}
    with pytest.raises(DatasetError, match="exit_reason 与独立复算不一致"):
        check_outcome_sidecar(sidecar_rows=tampered, **kwargs)

    # 丢弃一行 → 事件流 decision 开仓缺旁挂行（双向覆盖）检出
    with pytest.raises(DatasetError, match="缺少旁挂行"):
        check_outcome_sidecar(sidecar_rows=sidecar_rows[1:], **kwargs)

    # 伪造未知 bar 行 → 事件流中无对应开仓事件检出
    fake = {**sidecar_rows[0], "id": f"{sidecar_rows[0]['segment_id']}:999", "bar_index": 999}
    with pytest.raises(DatasetError, match="无对应 decision 开仓事件"):
        check_outcome_sidecar(sidecar_rows=[*sidecar_rows, fake], **kwargs)

    # id 与 segment_id/bar_index 不一致 → schema 检出
    broken = [dict(row) for row in sidecar_rows]
    broken[0] = {**broken[0], "id": "nope:0"}
    with pytest.raises(DatasetError, match="id 与 segment_id/bar_index 不一致"):
        check_outcome_sidecar(sidecar_rows=broken, **kwargs)


def test_check_audit_consistency_validates_outcomes_node(tmp_path: Path) -> None:
    result, records_by_split, _, _ = _generate_with_sidecar(tmp_path)
    payload = result.audit
    # 一致的 outcomes 节 → 通过
    check_audit_consistency(payload, records_by_split)
    # by_split 与开仓记录数不一致 → 检出
    bad = copy.deepcopy(payload)
    split = next(split for split in SPLIT_ROLES if bad["outcomes"]["by_split"][split] > 0)
    bad["outcomes"]["by_split"][split] += 1
    with pytest.raises(DatasetError, match="outcomes.by_split"):
        check_audit_consistency(bad, records_by_split)
    # per_segment 求和与 by_split 求和不一致 → 检出
    bad = copy.deepcopy(payload)
    bad["outcomes"]["per_segment"][0]["open_records"] += 1
    with pytest.raises(DatasetError, match="per_segment.open_records 求和"):
        check_audit_consistency(bad, records_by_split)

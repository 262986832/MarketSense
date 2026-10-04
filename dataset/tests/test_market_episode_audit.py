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
    check_records,
    check_split_isolation,
    check_state_leakage,
    parse_state_id,
    verify_determinism,
)
from dataset.market_episode.labels import evaluate_segment
from dataset.market_episode.nanojev_records import (
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
from dataset.storage import save_ohlcv
from dataset.tests.market_episode_fixtures import (
    DAILY_TP_DOWN,
    DAILY_TP_UP,
    SYMBOL,
    bars,
    build_workspace,
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


def _symbols_by_segment(workspace) -> dict[str, str]:
    """segment_id → symbol（``check_state_leakage`` 新必参；从片段清单确定性派生）。"""
    return {segment.segment_id: segment.symbol for segment in load_segments(workspace.manifest)}


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
        symbols_by_segment=_symbols_by_segment(workspace),
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
    workspace, _, _, params, _, records, bars_by_segment = _workspace_and_records(tmp_path)
    check_state_leakage(
        records,
        bars_by_segment=bars_by_segment,
        price_precision=params.price_precision,
        prev_daily_by_segment=prev_daily_map(workspace),
        trend_points_by_symbol=daily_trend_points_map(workspace),
        symbols_by_segment=_symbols_by_segment(workspace),
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
        trend_up_extreme=DAILY_TP_UP,
        trend_dn_extreme=DAILY_TP_DOWN,
    )

    with pytest.raises(DatasetError, match="与决策 K 线不一致"):
        check_state_leakage(
            tampered,
            bars_by_segment=bars_by_segment,
            price_precision=params.price_precision,
            prev_daily_by_segment=prev_daily_map(workspace),
            trend_points_by_symbol=daily_trend_points_map(workspace),
            symbols_by_segment=_symbols_by_segment(workspace),
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
            symbols_by_segment=_symbols_by_segment(workspace),
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
        symbols_by_segment=_symbols_by_segment(workspace),
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
            symbols_by_segment=_symbols_by_segment(workspace),
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
            symbols_by_segment=_symbols_by_segment(workspace),
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
        trend_up_extreme=DAILY_TP_UP,
        trend_dn_extreme=DAILY_TP_DOWN,
    )

    assert (
        "现价: 开=1.009000 高=1.012000 低=0.990000 收=0.992000 bar=2 成交量比=1.000000 持仓量比=1.003992"
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
    """v5 日线行 trend 篡改 → 独立重算不一致（日线行与决策 K 线不一致）→ 必须被检出。"""
    import re as _re

    workspace, _, _, params, _, records, bars_by_segment = _workspace_and_records(tmp_path)
    check_state_leakage(
        records,
        bars_by_segment=bars_by_segment,
        price_precision=params.price_precision,
        prev_daily_by_segment=prev_daily_map(workspace),
        trend_points_by_symbol=daily_trend_points_map(workspace),
        symbols_by_segment=_symbols_by_segment(workspace),
    )

    tampered = [copy.deepcopy(record) for record in records]
    assert "trend_up=" in tampered[0]["state"]
    tampered[0]["state"] = _re.sub(
        r"trend_up=[\d.]+", "trend_up=9.999999", tampered[0]["state"], count=1
    )

    with pytest.raises(DatasetError, match="日线行与决策 K 线不一致"):
        check_state_leakage(
            tampered,
            bars_by_segment=bars_by_segment,
            price_precision=params.price_precision,
            prev_daily_by_segment=prev_daily_map(workspace),
            trend_points_by_symbol=daily_trend_points_map(workspace),
            symbols_by_segment=_symbols_by_segment(workspace),
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
            symbols_by_segment=_symbols_by_segment(workspace),
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
        symbols_by_segment=_symbols_by_segment(workspace),
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
        symbols_by_segment=_symbols_by_segment(workspace),
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

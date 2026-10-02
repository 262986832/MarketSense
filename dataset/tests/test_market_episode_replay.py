"""T2：确定性回放账户引擎（成交模型、止损锚定、盯市/死亡、比值表达、无泄漏）。

手工推演的小场景夹具逐条对照冻结执行语义（tick = 1 便于核对）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataset.errors import DatasetError
from dataset.market_episode.nanojev_records import BoardStateValues, render_state
from dataset.market_episode.replay import (
    LONG,
    REASON_DEATH,
    REASON_REVERSE_CLOSE,
    REASON_REVERSE_OPEN,
    REASON_SEGMENT_END,
    REASON_STOP,
    SHORT,
    ReplayAccount,
    load_segment_bars,
)
from dataset.market_episode.segments import load_segments
from dataset.storage import save_ohlcv
from dataset.tests.market_episode_fixtures import (
    DAILY_TP_DOWN,
    DAILY_TP_UP,
    SYMBOL,
    bars,
    build_workspace,
    frame,
    write_manifest,
    segment_record,
)

#: 手工场景：小振幅（相对价格 1000）便于核对比值，tick = 1
_ROWS = [
    (1000, 1000.5, 999.5, 1000),
    (1000, 1010, 1000.5, 1008),
    (1008, 1009, 964, 966),
]


def _account(rows=_ROWS) -> ReplayAccount:
    return ReplayAccount(bars(rows), tick_size=1.0)


def test_fill_prices_are_decision_bar_extremes_plus_minus_one_tick() -> None:
    account = _account()

    assert account.reference_price == 1000.0
    assert account.buy_price(account.bars[0]) == 1001.5  # 最高 + 1 tick
    assert account.sell_price(account.bars[0]) == 998.5  # 最低 − 1 tick
    assert account.buy_price(account.bars[2]) == 1010.0
    assert account.sell_price(account.bars[2]) == 963.0


def test_long_entry_and_stop_anchor_to_opening_decision_bar() -> None:
    account = _account()

    event = account.open_position(LONG, account.bars[0])

    position = account.position
    assert position is not None
    assert event.price == 1001.5
    assert event.price_ratio == pytest.approx(1.0015)
    assert position.stop_price == 999.5  # 多：开仓 K 线最低
    assert position.entry_ratio == pytest.approx(1.0015)
    assert position.stop_ratio == pytest.approx(0.9995)


def test_short_entry_and_stop_anchor_to_opening_decision_bar() -> None:
    account = _account()

    account.open_position(SHORT, account.bars[1])

    position = account.position
    assert position is not None
    assert position.entry_price == 1000.5 - 1.0  # 最低 − 1 tick
    assert position.stop_price == 1010.0  # 空：开仓 K 线最高
    assert position.stop_ratio == pytest.approx(1.01)


def test_close_settles_in_ratio_units() -> None:
    account = _account()
    account.open_position(LONG, account.bars[0])

    event = account.close_position(account.bars[1], reason=REASON_SEGMENT_END)

    assert event.price == 1000.5 - 1.0
    assert event.pnl_ratio == pytest.approx(0.9995 - 1.0015)
    assert account.realized_pnl == pytest.approx(-0.002)
    assert account.position is None


def test_reverse_is_two_fills_at_the_same_decision_bar() -> None:
    account = _account()
    account.open_position(LONG, account.bars[0])

    close_event, open_event = account.reverse_position(SHORT, account.bars[1])

    assert close_event.reason == REASON_REVERSE_CLOSE
    assert open_event.reason == REASON_REVERSE_OPEN
    assert close_event.bar_index == open_event.bar_index == 1
    # 多头反手为空头：两笔都是"卖出"，成交价相同（同一决策 K 线最低 − 1 tick）
    assert close_event.side == open_event.side == "sell"
    assert close_event.price == open_event.price == 999.5
    position = account.position
    assert position is not None and position.direction == SHORT
    assert position.stop_price == 1010.0  # 新持仓止损锚定反手当根最高
    assert [event.reason for event in account.events] == [
        "decision",
        REASON_REVERSE_CLOSE,
        REASON_REVERSE_OPEN,
    ]


def test_stop_touched_uses_opposite_extreme() -> None:
    account = _account()
    account.open_position(LONG, account.bars[0])

    assert account.stop_touched(account.bars[1]) is False  # low 1000.5 > stop 999.5
    assert account.stop_touched(account.bars[2]) is True  # low 964 <= stop 999.5

    short_account = _account()
    short_account.open_position(SHORT, account.bars[0])
    assert short_account.stop_touched(account.bars[1]) is True  # high 1010 >= stop 1010


def test_mark_to_market_uses_previous_bar_close() -> None:
    account = _account()

    flat_report = account.mark_to_market(0)
    assert flat_report.mark_price is None
    assert flat_report.equity == pytest.approx(1.0)
    assert flat_report.drawdown == 0.0

    account.open_position(LONG, account.bars[0])
    report = account.mark_to_market(1)

    assert report.mark_price == 1000.0  # t−1 数据：决策 K 线 1 用 close[0]
    assert report.equity == pytest.approx(1.0 + (1.0 - 1.0015))
    assert report.drawdown == pytest.approx(0.0015)
    assert report.peak_equity == pytest.approx(1.0)


def test_mark_to_market_ignores_current_bar_close_and_updates_peak() -> None:
    rows = [
        (1000, 1000.5, 999.5, 1000),
        (1000, 1020, 999.6, 1019),  # 有利波动：峰值抬高
        (1019, 1021, 1018, 1020),
    ]
    account = _account(rows)
    account.open_position(LONG, account.bars[0])

    first = account.mark_to_market(1)
    assert first.equity == pytest.approx(1.0 + (1.0 - 1.0015))
    second = account.mark_to_market(2)

    assert second.mark_price == 1019.0  # close[1]，与当前根 close 无关
    assert second.equity == pytest.approx(1.0 + (1.019 - 1.0015))
    assert second.peak_equity == pytest.approx(1.0 + (1.019 - 1.0015))
    assert second.drawdown == pytest.approx(0.0)


def test_death_detected_at_decision_bar_by_drawdown_threshold() -> None:
    rows = [(100, 105, 95, 100), (100, 101, 99, 100)]
    account = ReplayAccount(bars(rows), tick_size=1.0)
    account.open_position(LONG, account.bars[0])  # entry = 106

    report = account.mark_to_market(1)

    assert report.mark_price == 100.0
    assert report.equity == pytest.approx(0.94)
    assert report.drawdown == pytest.approx(0.06)
    assert report.is_dead(0.05) is True
    assert report.is_dead(0.07) is False


def test_force_close_at_segment_end_and_stop_exit_use_same_fill_model() -> None:
    account = _account()
    account.open_position(LONG, account.bars[0])

    stop_event = account.close_position(account.bars[2], reason=REASON_STOP)
    assert stop_event.price == 964.0 - 1.0  # 触发当根最低 − 1 tick
    assert account.position is None

    other = _account()
    other.open_position(SHORT, other.bars[0])
    end_event = other.close_position(other.bars[-1], reason=REASON_SEGMENT_END)
    assert end_event.price == 1009.0 + 1.0
    assert end_event.pnl_ratio == pytest.approx(0.9985 - 1.01)

    death_account = _account()
    death_account.open_position(SHORT, death_account.bars[0])
    death_event = death_account.close_position(death_account.bars[-1], reason=REASON_DEATH)
    assert death_event.reason == REASON_DEATH


@pytest.mark.parametrize(
    "action, message",
    [
        ("open_position", "已有持仓，不能重复开仓"),
        ("close_position", "无持仓，不能平仓"),
        ("reverse_position", "无持仓，不能反手"),
    ],
)
def test_account_rejects_illegal_transitions(action: str, message: str) -> None:
    account = _account()
    if action == "open_position":
        account.open_position(LONG, account.bars[0])
        with pytest.raises(DatasetError, match="已有持仓"):
            account.open_position(LONG, account.bars[1])
        return
    if action == "reverse_position":
        with pytest.raises(DatasetError, match=message):
            account.reverse_position(SHORT, account.bars[0])
        return
    with pytest.raises(DatasetError, match=message):
        account.close_position(account.bars[0], reason=REASON_STOP)


def test_account_rejects_reverse_to_same_direction() -> None:
    account = _account()
    account.open_position(LONG, account.bars[0])

    with pytest.raises(DatasetError, match="反手方向非法"):
        account.reverse_position(LONG, account.bars[1])
    with pytest.raises(DatasetError, match="未知持仓方向"):
        account.open_position("sideways", account.bars[1])


def test_account_rejects_non_positive_reference_price() -> None:
    with pytest.raises(DatasetError, match="首根开盘价必须为正数"):
        ReplayAccount(bars([(0, 0, 0, 0)]), tick_size=1.0)


def test_replay_is_deterministic() -> None:
    def run() -> tuple:
        account = _account()
        account.open_position(LONG, account.bars[0])
        account.mark_to_market(1)
        account.reverse_position(SHORT, account.bars[1])
        account.mark_to_market(2)
        account.close_position(account.bars[2], reason=REASON_STOP)
        return account.events, account.realized_pnl, account.peak_equity

    assert run() == run()


def test_load_segment_bars_filters_inclusively_and_keeps_source_version(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, [(100, 101, 99, 100)] * 6)
    segment = load_segments(workspace.manifest)[0]

    loaded, source_version = load_segment_bars(segment, data_dir=workspace.data_dir)

    assert len(loaded) == 2  # seg-train 覆盖 6 根中的前 2 根
    assert [bar.index for bar in loaded] == [0, 1]
    assert loaded[0].timestamp == str(segment.start)
    assert source_version is not None and "sha256=" in source_version


def test_load_segment_bars_rejects_unsorted_or_duplicated_timestamps(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "ohlcv"
    rows = [(100, 101, 99, 100)] * 4
    frame_data = frame(rows)
    reversed_frame = frame_data.iloc[::-1].reset_index(drop=True)
    save_ohlcv(reversed_frame, symbol=SYMBOL, period="1m", output_dir=data_dir)
    manifest = write_manifest(
        tmp_path / "segments.jsonl",
        [
            segment_record("seg-train", "train", start_index=0, end_index=1),
            segment_record("seg-dev", "dev", start_index=2, end_index=2),
            segment_record("seg-test", "test", start_index=3, end_index=3),
        ],
    )
    segment = load_segments(manifest)[0]

    with pytest.raises(DatasetError, match="严格升序且唯一"):
        load_segment_bars(segment, data_dir=data_dir)


def test_visible_state_depends_only_on_bars_up_to_decision_bar() -> None:
    """防泄漏：决策点状态只由 ≤ 决策 K 线的 bar 计算（未来 bar 变化不影响状态）。"""
    base_rows = [(1000, 1000.5, 999.5, 1000), (1000, 1010, 1000.5, 1008), (1008, 1009, 964, 966)]
    index = 1
    reference_bar = bars(base_rows)[0]
    board = BoardStateValues(prev_day_high=2000.0, prev_day_low=1980.0, prev_day_close=1990.0,
                             today_high=1010.0, today_low=999.5)
    state_before = render_state(
        bar=bars(base_rows)[index],
        reference_bar=reference_bar,
        position=None,
        drawdown=0.0,
        price_precision=6,
        board_state=board,
        trend_up_extreme=DAILY_TP_UP,
        trend_dn_extreme=DAILY_TP_DOWN,
    )

    mutated = bars(base_rows + [(966, 1200, 900, 1100)])[index]  # 决策 K 线之后的 bar 改变
    state_after = render_state(
        bar=mutated,
        reference_bar=reference_bar,
        position=None,
        drawdown=0.0,
        price_precision=6,
        board_state=board,
        trend_up_extreme=DAILY_TP_UP,
        trend_dn_extreme=DAILY_TP_DOWN,
    )

    assert state_before == state_after
    assert "1200" not in state_before and "1100" not in state_before


def test_visible_state_contains_no_absolute_prices() -> None:
    rows = [(1000, 1000.5, 999.5, 1000), (1000, 1010, 1000.5, 1008)]
    bar_list = bars(rows)
    state = render_state(
        bar=bar_list[1],
        reference_bar=bar_list[0],
        position=None,
        drawdown=0.0,
        price_precision=6,
        board_state=BoardStateValues(prev_day_high=2000.0, prev_day_low=1980.0,
                                     prev_day_close=1990.0, today_high=1010.0, today_low=999.5),
        trend_up_extreme=DAILY_TP_UP,
        trend_dn_extreme=DAILY_TP_DOWN,
    )

    for price in (1000, 1000.5, 999.5, 1010, 1008):
        assert f"{price:.6f}" not in state
    assert "现价" in state and "联动" in state and "回撤" in state

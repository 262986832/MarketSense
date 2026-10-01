"""T3：盈亏比规则真值标签 + (b) 采样（手工推演逐条对照冻结语义）。

场景均为手工推演的小 K 线序列（tick = 1）：止损先触 → 不开仓；盈亏比严格大于阈值 →
开仓；反转两条件 → 平仓/反手；持有；程序止损离场不产生模型决策样本；死亡后不再产生
样本；带宽内"不做"分钟入选、带宽外剔除。
"""

from __future__ import annotations

import pytest

import pandas as pd

from dataset.market_episode.labels import (
    ACTION_CLOSE,
    ACTION_HOLD,
    ACTION_OPEN_LONG,
    ACTION_OPEN_SHORT,
    ACTION_REVERSE,
    ACTION_STAY_FLAT,
    SELECTION_EXCLUDED_FLAT,
    SELECTION_FLAT_BAND,
    SELECTION_HOLDING,
    SELECTION_OPPORTUNITY,
    DecisionPoint,
    apply_flat_band,
    evaluate_segment,
    holding_action,
    reversal_conditions,
    reward_risk,
    select_open_action,
)
from dataset.market_episode.segments import EpisodeParams, Segment
from dataset.tests.market_episode_fixtures import SYMBOL, bars, timestamp_at

#: 场景 A：小振幅前置空仓 + 一根大阳线机会分钟（含带宽采样）
_BAND_ROWS = [
    (100, 100.2, 99.8, 100),
    (100, 100.3, 99.7, 100.1),
    (100, 100.4, 99.6, 100.2),
    (100, 100.5, 99.5, 100.3),
    (100, 100.6, 99.4, 100.4),
    (100, 110, 99, 109.5),
    (110, 160, 109, 155),
]

#: 场景 B：开多 → 反转（反向盈亏比 > 阈值）→ 反手 → 片段末强平
_REVERSE_ROWS = [
    (1000, 1000.5, 999.5, 1000),
    (1000, 1010, 1000.5, 1008),
    (1008, 1009, 964, 966),
]

#: 场景 C：开多 → 持有 → 程序止损离场（不产生样本）→ 空仓
_STOP_ROWS = [
    (1000, 1000.5, 999.5, 1000),
    (1000, 1010, 1000, 1009),
    (1009, 1012, 990, 992),
    (992, 993, 988, 991),
]

#: 场景 D：开多 → 持有 → 程序止损离场 → 空仓死亡（已空仓，无强平）
_DEATH_FLAT_ROWS = [
    (1000, 1010, 950, 1000),
    (1000, 1200, 1005, 1180),
    (1180, 1200, 940, 945),
    (985, 990, 980, 988),
    (988, 990, 980, 985),  # 死亡之后：不处理
]

#: 场景 E：持仓时回撤触发死亡 → 检出当根强平
_DEATH_HOLDING_ROWS = [
    (100, 105, 95, 100),
    (100, 140, 100, 139),
    (139, 140, 90, 91),
]

#: 场景 F：开空 → 反转反手做多 → 片段末强平（空头路径全程对称性；手工推演见用例）
_SHORT_REVERSE_ROWS = [
    (1000, 1000.5, 999.5, 1000),
    (1000, 1000.4, 990, 991),
    (991, 1050, 992, 1048),
]


def make_segment(bar_count: int, *, split_role: str = "train", segment_id: str = "seg-1") -> Segment:
    return Segment(
        segment_id=segment_id,
        symbol=SYMBOL,
        period="1m",
        start=pd.Timestamp(timestamp_at(0), tz="Asia/Shanghai"),
        end=pd.Timestamp(timestamp_at(bar_count - 1), tz="Asia/Shanghai"),
        split_role=split_role,
    )


def run(rows, *, params: EpisodeParams | None = None, tick_size: float = 1.0):
    bar_list = bars(rows)
    outcome = evaluate_segment(
        make_segment(len(rows)),
        bar_list,
        tick_size=tick_size,
        params=params or EpisodeParams(),
    )
    return outcome


# --------------------------------------------------------------------------- #
# reward_risk：《标准》语义逐条
# --------------------------------------------------------------------------- #
def test_reward_risk_is_zero_when_stop_touched_first() -> None:
    rows = [(1000, 1000.5, 999.5, 1000), (1000, 1001, 999, 999.5)]
    bar_list = bars(rows)

    # 多：入场 1001.5（最高 + 1tick）、止损 999.5；下一根最低 999 先触止损 → MFE 0
    assert reward_risk(bar_list, 0, "long", 1.0) == 0.0
    # 空：入场 998.5、止损 1000.5；下一根最高 1001 >= 止损 → MFE 0
    assert reward_risk(bar_list, 0, "short", 1.0) == 0.0


def test_reward_risk_counts_favorable_move_before_stop_touch() -> None:
    rows = [
        (1000, 1000.5, 999.5, 1000),
        (1000, 1010, 1000.5, 1008),
        (1008, 1009, 990, 992),  # 止损被触 → MFE 截断
    ]
    bar_list = bars(rows)

    # MFE = 1010 − 1001.5 = 8.5；止损距离 = (1000.5 − 999.5) + 1 = 2 → 4.25
    assert reward_risk(bar_list, 0, "long", 1.0) == pytest.approx(4.25)


def test_reward_risk_uses_remaining_bars_when_stop_never_touched() -> None:
    rows = [(1000, 1000.5, 999.5, 1000), (1000, 1010, 1000.5, 1008)]
    bar_list = bars(rows)

    assert reward_risk(bar_list, 0, "long", 1.0) == pytest.approx(8.5 / 2)
    # 最后一根之后没有数据 → 无法形成有利波动
    assert reward_risk(bar_list, 1, "long", 1.0) == 0.0


def test_reward_risk_treats_adverse_side_first_within_a_bar() -> None:
    """同根同时触及止损与新高 → 保守约定先算不利侧（该根有利波动不计入）。"""
    rows = [(1000, 1000.5, 999.5, 1000), (1000, 1010, 999, 1005)]
    bar_list = bars(rows)

    assert reward_risk(bar_list, 0, "long", 1.0) == 0.0


# --------------------------------------------------------------------------- #
# select_open_action：严格大于 / 更优一侧 / 均不达标
# --------------------------------------------------------------------------- #
def test_select_open_action_requires_strictly_greater_than_threshold() -> None:
    assert select_open_action(3.0, 0.0, 3.0) is None
    assert select_open_action(3.0000001, 0.0, 3.0) == ACTION_OPEN_LONG
    assert select_open_action(0.0, 3.5, 3.0) == "open_short"
    assert select_open_action(1.0, 2.0, 3.0) is None


def test_select_open_action_picks_better_side_and_is_deterministic_on_tie() -> None:
    assert select_open_action(4.0, 3.5, 3.0) == ACTION_OPEN_LONG
    assert select_open_action(3.5, 4.0, 3.0) == "open_short"
    # 冻结「无并列问题」；仍给出确定性并列口径（相等取多），保证实现可复现
    assert select_open_action(4.0, 4.0, 3.0) == ACTION_OPEN_LONG


def test_select_open_action_two_sided_qualification_is_unreachable_in_real_bars() -> None:
    """冻结成交模型下两向不可能同时有正盈亏比（实现注释中的证明的回归保护）。"""
    import itertools

    candidates = [1000.0, 1005.0, 1010.0, 990.0, 995.0]
    for high, low in itertools.product(candidates, repeat=2):
        if low > high:
            continue
        rows = [(1000, 1000.5, 999.5, 1000), (1000, high, low, (high + low) / 2)]
        bar_list = bars(rows)
        assert not (
            reward_risk(bar_list, 0, "long", 1.0) > 0
            and reward_risk(bar_list, 0, "short", 1.0) > 0
        )


# --------------------------------------------------------------------------- #
# reversal_conditions / holding_action
# --------------------------------------------------------------------------- #
def test_reversal_requires_next_bar_strictly_lower_and_no_new_high() -> None:
    long_stop = 999.5
    # 条件①成立 + 条件②成立（先到止损价）
    bars_ok = bars(
        [(1000, 1000.5, 999.5, 1000), (1000, 1010, 1000.5, 1008), (1008, 1009, 964, 966)]
    )
    assert reversal_conditions(bars_ok, 1, "long", long_stop) is True

    # 条件①不成立（t+1 最高不低于当前最高）
    bars_cond1_fail = bars(
        [(1000, 1000.5, 999.5, 1000), (1000, 1010, 1000.5, 1008), (1008, 1012, 990, 992)]
    )
    assert reversal_conditions(bars_cond1_fail, 1, "long", long_stop) is False

    # 条件①成立但条件②不成立（先创新高）
    bars_cond2_fail = bars(
        [
            (1000, 1000.5, 999.5, 1000),
            (1000, 1010, 1000.5, 1008),
            (1008, 1011, 1006, 1009),
        ]
    )
    assert reversal_conditions(bars_cond2_fail, 1, "long", long_stop) is False

    # 条件①成立 + 扫到片段末既未超当前 K 线也未触止损 → 冻结项 6：同样成立
    bars_scan_end = bars(
        [(1000, 1000.5, 999.5, 1000), (1000, 1010, 1000.5, 1008), (1008, 1009, 1000, 1001)]
    )
    assert reversal_conditions(bars_scan_end, 1, "long", long_stop) is True

    # 最后一根没有 t+1 → 不成立
    assert reversal_conditions(bars_ok, 2, "long", long_stop) is False


def test_reversal_conditions_are_symmetric_for_short() -> None:
    short_stop = 1000.5
    bars_ok = bars(
        [(1000, 1000.5, 999.5, 1000), (1000, 999.5, 990, 991), (991, 1001, 992, 1000)]
    )
    assert reversal_conditions(bars_ok, 1, "short", short_stop) is True

    # 空头：条件①成立 + 扫到片段末既未创新低也未触止损 → 冻结项 6：同样成立
    bars_scan_end = bars(
        [(1000, 1000.5, 999.5, 1000), (1000, 999.5, 990, 991), (991, 1000, 995, 998)]
    )
    assert reversal_conditions(bars_scan_end, 1, "short", short_stop) is True

    # 空头：条件①成立但随后的 K 线创出新低（低于当前 K 线最低）→ 条件②不成立
    bars_new_low = bars(
        [
            (1000, 1000.5, 999.5, 1000),
            (1000, 999.5, 990, 991),
            (991, 1000, 995, 998),
            (998, 999, 985, 986),
        ]
    )
    assert reversal_conditions(bars_new_low, 1, "short", short_stop) is False


def test_holding_action_close_hold_reverse() -> None:
    close_bars = bars(
        [(1000, 1000.5, 999.5, 1000), (1000, 1010, 1000.5, 1008), (1008, 1009, 1000, 1001)]
    )
    assert (
        holding_action(
            close_bars, 1, direction="long", stop_price=999.5, tick_size=1.0, threshold=3.0
        )
        == ACTION_CLOSE
    )

    reverse_bars = bars(_REVERSE_ROWS)
    assert (
        holding_action(
            reverse_bars, 1, direction="long", stop_price=999.5, tick_size=1.0, threshold=3.0
        )
        == ACTION_REVERSE
    )

    hold_bars = bars(_STOP_ROWS)
    assert (
        holding_action(
            hold_bars, 1, direction="long", stop_price=999.5, tick_size=1.0, threshold=3.0
        )
        == ACTION_HOLD
    )


# --------------------------------------------------------------------------- #
# (b) 采样带宽
# --------------------------------------------------------------------------- #
def test_apply_flat_band_selects_neighbouring_flat_minutes_only() -> None:
    def point(index: int, selection: str, selected: bool) -> DecisionPoint:
        return DecisionPoint(
            bar_index=index,
            action=ACTION_OPEN_LONG if selection == SELECTION_OPPORTUNITY else ACTION_STAY_FLAT,
            selection=selection,
            selected=selected,
            position=None,
            drawdown=0.0,
            reward_risk_long=None,
            reward_risk_short=None,
        )

    points = [point(index, "flat", False) for index in range(6)]
    points[5] = point(5, SELECTION_OPPORTUNITY, True)

    banded = apply_flat_band(list(points), 2)

    assert [p.selection for p in banded] == [
        SELECTION_EXCLUDED_FLAT,
        SELECTION_EXCLUDED_FLAT,
        SELECTION_EXCLUDED_FLAT,
        SELECTION_FLAT_BAND,
        SELECTION_FLAT_BAND,
        SELECTION_OPPORTUNITY,
    ]
    assert [p.selected for p in banded] == [False, False, False, True, True, True]

    narrowed = apply_flat_band(list(points), 0)
    assert [p.selected for p in narrowed] == [False, False, False, False, False, True]


def test_apply_flat_band_excludes_flat_minutes_without_any_opportunity() -> None:
    points = [
        DecisionPoint(
            bar_index=index,
            action=ACTION_STAY_FLAT,
            selection="flat",
            selected=False,
            position=None,
            drawdown=0.0,
            reward_risk_long=None,
            reward_risk_short=None,
        )
        for index in range(3)
    ]

    banded = apply_flat_band(points, 2)

    assert all(point.selection == SELECTION_EXCLUDED_FLAT for point in banded)
    assert all(point.selected is False for point in banded)


def test_evaluate_segment_samples_opportunity_band_and_holding_minutes() -> None:
    outcome = run(_BAND_ROWS, params=EpisodeParams(flat_sample_band_minutes=2))

    assert [point.bar_index for point in outcome.selected] == [3, 4, 5, 6]
    assert [point.action for point in outcome.selected] == [
        ACTION_STAY_FLAT,
        ACTION_STAY_FLAT,
        ACTION_OPEN_LONG,
        ACTION_HOLD,
    ]
    assert [point.selection for point in outcome.decision_points] == [
        SELECTION_EXCLUDED_FLAT,
        SELECTION_EXCLUDED_FLAT,
        SELECTION_EXCLUDED_FLAT,
        SELECTION_FLAT_BAND,
        SELECTION_FLAT_BAND,
        SELECTION_OPPORTUNITY,
        SELECTION_HOLDING,
    ]
    assert outcome.excluded_flat == 3
    assert outcome.segment_end_forced_close == 6
    # 空仓点带盈亏比（该方向未达标故为 0），持仓点不带
    assert outcome.decision_points[3].reward_risk_long == 0.0
    assert outcome.decision_points[6].reward_risk_long is None


def test_evaluate_segment_band_width_is_configurable() -> None:
    outcome = run(_BAND_ROWS, params=EpisodeParams(flat_sample_band_minutes=0))

    assert [point.bar_index for point in outcome.selected] == [5, 6]
    assert outcome.excluded_flat == 5


# --------------------------------------------------------------------------- #
# 持仓语义：反手 / 程序止损离场 / 死亡
# --------------------------------------------------------------------------- #
def test_evaluate_segment_reverse_then_segment_end_forced_close() -> None:
    outcome = run(_REVERSE_ROWS)

    assert [point.action for point in outcome.selected] == [
        ACTION_OPEN_LONG,
        ACTION_REVERSE,
        ACTION_HOLD,
    ]
    assert [point.position.direction for point in outcome.selected if point.position] == [
        "long",
        "short",
    ]
    reasons = [event.reason for event in outcome.events]
    assert reasons == [
        "decision",
        "reverse_close",
        "reverse_open",
        "segment_end",
    ]
    reverse_close, reverse_open = outcome.events[1], outcome.events[2]
    assert reverse_close.price == reverse_open.price == 999.5
    assert outcome.segment_end_forced_close == 2
    # 多头亏 0.002，空头片段末强平亏 0.0105
    assert outcome.realized_pnl_ratio == pytest.approx(-0.0125)


def test_evaluate_segment_opens_short_and_reverses_to_long() -> None:
    """空头路径完整推演（对称性；tick = 1，默认阈值 3.0、回撤阈值 0.05）：

    * bar0：多向入场 1001.5 / 止损 999.5（风险 2），下一根最低 990 先触止损 → 多向
      盈亏比 0；空向入场 998.5 / 止损 1000.5（风险 2），bar1 最高 1000.4 未触止损、
      最低 990 → MFE 8.5 → 4.25 > 3 → 开空（成交价 = 最低 − 1 tick）；
    * bar1：空头止损 1000.5 未被触及（最高 1000.4），回撤 0.0015 < 0.05；反转条件①
      （bar2 最高 1050 > 1000.4 且最低 992 > 990）与②（先到止损价）均成立，反向
      盈亏比 = (1050 − 1001.4)/11.4 = 4.263 > 3 → 反手（两笔同价 = 最高 + 1 tick）；
    * bar2：多头止损 990 未触（最低 992），回撤 0.0133 < 0.05，无 t+1 → 持有；
      回放结束后按同一成交方法强平（= 最低 992 − 1 tick）。
    """
    outcome = run(_SHORT_REVERSE_ROWS)

    assert [point.action for point in outcome.selected] == [
        ACTION_OPEN_SHORT,
        ACTION_REVERSE,
        ACTION_HOLD,
    ]
    assert [point.position.direction for point in outcome.selected if point.position] == [
        "short",
        "long",
    ]
    assert [event.reason for event in outcome.events] == [
        "decision",
        "reverse_close",
        "reverse_open",
        "segment_end",
    ]
    assert outcome.events[0].price == 999.5 - 1.0  # 开空 = 最低 − 1 tick
    assert outcome.events[1].price == outcome.events[2].price == 1000.4 + 1.0  # 反手同价
    assert outcome.events[3].price == 992.0 - 1.0  # 片段末强平 = 最低 − 1 tick
    assert outcome.segment_end_forced_close == 2
    assert outcome.excluded_flat == 0
    # 空头 −0.0029、多头片段末强平 −0.0104
    assert outcome.realized_pnl_ratio == pytest.approx(-0.0133)


def test_evaluate_segment_program_stop_exit_produces_no_decision_sample() -> None:
    outcome = run(_STOP_ROWS)

    assert outcome.stop_exits == (2,)
    # bar 2（程序止损离场）没有任何决策样本
    assert 2 not in [point.bar_index for point in outcome.decision_points]
    assert [point.bar_index for point in outcome.selected] == [0, 1]
    assert outcome.segment_end_forced_close is None  # 离场后到片段末为空仓
    assert outcome.events[-1].reason == "stop"
    assert outcome.events[-1].price == 990.0 - 1.0


def test_evaluate_segment_death_when_flat_stops_episode() -> None:
    outcome = run(_DEATH_FLAT_ROWS)

    assert [(event.bar_index, event.forced_close) for event in outcome.deaths] == [(3, False)]
    assert outcome.processed_bar_count == 4  # bar 4 不再处理
    assert [point.bar_index for point in outcome.decision_points] == [0, 1]
    assert [point.action for point in outcome.selected] == [ACTION_OPEN_LONG, ACTION_HOLD]
    assert outcome.realized_pnl_ratio == pytest.approx((939.0 - 1011.0) / 1000.0)


def test_evaluate_segment_death_while_holding_forces_close_on_detection_bar() -> None:
    outcome = run(_DEATH_HOLDING_ROWS)

    assert [(event.bar_index, event.forced_close) for event in outcome.deaths] == [(1, True)]
    assert [point.bar_index for point in outcome.decision_points] == [0]
    assert outcome.processed_bar_count == 2
    assert outcome.events[-1].reason == "death"
    assert outcome.events[-1].price == 100.0 - 1.0  # 检出当根最低 − 1 tick
    assert outcome.segment_end_forced_close is None


# --------------------------------------------------------------------------- #
# 确定性
# --------------------------------------------------------------------------- #
def test_evaluate_segment_is_deterministic() -> None:
    first = run(_REVERSE_ROWS)
    second = run(_REVERSE_ROWS)

    assert first == second
    assert first.selected == second.selected


def test_evaluate_segment_requires_bars() -> None:
    from dataset.errors import DatasetError

    with pytest.raises(DatasetError, match="没有 K 线"):
        evaluate_segment(make_segment(1), (), tick_size=1.0, params=EpisodeParams())

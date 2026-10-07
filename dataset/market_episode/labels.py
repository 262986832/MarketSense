"""policy 层：盈亏比规则真值标签生成器 + (b) 采样（标签语义已冻结）。

**标签语义（DESIGN 已冻结，2026-09-30 用户确认实现阶段口径）**

空仓决策点（i = 决策 K 线序号，扫描从 ``i+1`` 开始；标签程序可见未来）：

* 多为例：入场 = ``high[i] + tick``，止损 = ``low[i]``，止损距离 = 当根振幅 + tick；
* 逐根前向扫描，**同根同时触及止损与新高按保守约定先算不利侧**：某根 ``low <= 止损``
  → 停止扫描（该根的有利波动不计入）；
* 盈亏比 = MFE（止损被触前各根最大有利波动）/ 止损距离；
* 盈亏比**严格大于**阈值的方向成为开仓候选；两向均达标 → 取更优一侧；均不达标 →
  继续空仓。
  性能说明：冻结成交模型下「两向均达标」在真实行情中**不可达**（任一根先越过
  某方向的入场价必然先触及另一方向的止损；证明见本文件 ``select_open_action`` 注释），
  但规则仍按要求实现并可直接单测。

持仓决策点：

* 止损被触（早于反转判定）→ **程序自动止损离场**（程序行为，不产生模型决策样本），
  离场后回到空仓；
* 当前 K 线满足反转两条件 → 标签 = 平仓；反向方向盈亏比 > 阈值 → 标签 = 反手；
* 否则 → 标签 = 持有。

反转两条件（多头，空头对称）：

1. ``t+1`` 整根严格低于当前 K 线：``high[t+1] < high[t] 且 low[t+1] < low[t]``；
2. 从 ``t+1`` 起扫描：先到止损价 → 成立（冻结项 6）；扫到片段末既未超当前 K 线
   也未触止损 → 同样成立（冻结项 6）。

(b) 采样（双机制，label-band-sampling 任务用户拍板，2026-10-06）：

* **test/calibration/ood**（维持原口径）：进训练集 = 机会分钟（gold = 开多/开空）
  + 机会分钟周边 ± 带宽内的空仓分钟（gold = 继续空仓）+ 持仓期间分钟
  （gold = 持有/平仓/反手）；其余空仓分钟剔除并计审计；
* **train/dev**：进训练集 = 事件决策记录（gold = 开多/开空 → open 类事件；
  gold = 平仓/反手 → exit 类事件；事件包含自身）+ 每个事件紧邻其前 N 条决策记录
  （open 类窗口 ``event_lookforward_open`` 默认 2、exit 类窗口
  ``event_lookforward_exit`` 默认 10；一条记录同时是多个事件的前导只保留一次，
  标签统一 ``event_lead``；窗口不跨片段）；未被任何事件窗口覆盖的持仓分钟剔除
  （``excluded_holding``），未被覆盖的空仓分钟剔除（``excluded_flat``）；
  程序止损离场/死亡分钟不产生决策记录，不计入 N。
采样只改变 selected/selection 标记，发生在 gold 回放与账户值计算之后（确定性、
无未来泄漏语义不变）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Mapping

from dataset.errors import DatasetError

from dataset.market_episode.replay import (
    LONG,
    REASON_DEATH,
    REASON_DECISION,
    REASON_SEGMENT_END,
    REASON_STOP,
    SHORT,
    Bar,
    PositionSnapshot,
    ReplayAccount,
    TradeEvent,
)
from dataset.market_episode.segments import EpisodeParams, Segment

#: 模型动作（choice 题候选 ID；空仓三选一 / 持仓三选一）
ACTION_OPEN_LONG = "open_long"
ACTION_OPEN_SHORT = "open_short"
ACTION_STAY_FLAT = "stay_flat"
ACTION_CLOSE = "close"
ACTION_HOLD = "hold"
ACTION_REVERSE = "reverse"

#: 采样分类
SELECTION_OPPORTUNITY = "opportunity"
SELECTION_FLAT_BAND = "flat_band"
SELECTION_EXCLUDED_FLAT = "excluded_flat"
SELECTION_EXCLUDED_HOLDING = "excluded_holding"
SELECTION_EVENT_LEAD = "event_lead"
SELECTION_HOLDING = "holding"
#: 空仓点（尚未定采样归属）的中间状态：待 :func:`apply_flat_band` 或
#: :func:`apply_event_lookforward` 决定进/出训练集
SELECTION_FLAT = "flat"

#: 事件前看采样生效的 split（label-band-sampling 拍板：train/dev 新机制；
#: test/calibration/ood 维持 ±2 平带，calibration/ood 为低风险默认，未单独拍板）
EVENT_LOOKFORWARD_SPLITS = frozenset({"train", "dev"})
#: 开仓类事件动作（窗口 = ``event_lookforward_open``）
_OPEN_EVENT_ACTIONS = frozenset({ACTION_OPEN_LONG, ACTION_OPEN_SHORT})
#: 离场类事件动作（窗口 = ``event_lookforward_exit``）
_EXIT_EVENT_ACTIONS = frozenset({ACTION_CLOSE, ACTION_REVERSE})


@dataclass(frozen=True)
class DecisionPoint:
    """一个决策点的真值标签与采样结论（选中的点会转成 NanoJev 记录）。"""

    bar_index: int
    action: str
    selection: str
    selected: bool
    position: PositionSnapshot | None
    drawdown: float
    reward_risk_long: float | None
    reward_risk_short: float | None
    #: 账户净值（净值尺度；v8 账户行「净值」键的数据源，由调用方（账户回放）
    #: 只用 ≤ 决策 K 线的数据算好传入；跨片段净值链（时间序串行回放，初值 100）
    #: 由后续任务接入，当前默认 0.0 仅为未接入的占位，非真实净值。
    #: 见 artifacts/account-service/02-design/tech-design.md）
    net_value: float = 0.0
    #: 片段内今日收益（净值尺度，每片段重置；v8 账户行「今日」键的数据源，同上由调用方算好传入）
    today_pnl: float = 0.0


@dataclass(frozen=True)
class DeathEvent:
    """死亡事件：检出当根序号 + 检出时是否仍有持仓（有则按同一成交方法强平）。"""

    bar_index: int
    forced_close: bool


@dataclass(frozen=True)
class SegmentOutcome:
    """一个片段（episode）的标签与账户轨迹汇总（供审计文件使用）。"""

    segment_id: str
    symbol: str
    split_role: str
    bar_count: int
    processed_bar_count: int
    decision_points: tuple[DecisionPoint, ...]
    stop_exits: tuple[int, ...]
    deaths: tuple[DeathEvent, ...]
    segment_end_forced_close: int | None
    events: tuple[TradeEvent, ...]
    realized_pnl_ratio: float
    peak_equity: float
    # 片段末结算净值/峰值（比值口径，默认参数下 final_equity 以 1.0 起算；
    # 跨片段传递时作为下一片段 evaluate_segment 的 initial_equity/initial_peak 来源）
    final_equity: float
    final_peak: float
    source_data_version: str | None

    @property
    def selected(self) -> tuple[DecisionPoint, ...]:
        return tuple(point for point in self.decision_points if point.selected)

    @property
    def excluded_flat(self) -> int:
        return sum(
            1 for point in self.decision_points if point.selection == SELECTION_EXCLUDED_FLAT
        )

    @property
    def excluded_holding(self) -> int:
        """事件前看采样（train/dev）下未被任何事件窗口覆盖的持仓分钟数。"""
        return sum(
            1
            for point in self.decision_points
            if point.selection == SELECTION_EXCLUDED_HOLDING
        )

    @property
    def action_counts(self) -> Mapping[str, int]:
        return dict(Counter(point.action for point in self.selected))


def reward_risk(bars: tuple[Bar, ...], bar_index: int, direction: str, tick_size: float) -> float:
    """某方向在决策 K 线上的盈亏比 = MFE（止损被触前最大有利波动）/ 止损距离。

    * 入场/止损锚定决策 K 线高低点（多：入场 = 最高 + tick、止损 = 最低；空对称）；
    * 前向扫描从 ``bar_index + 1`` 开始，**同根先算不利侧**（触及止损即停止扫描，
      该根的有利波动不计入）；
    * 扫描到片段末仍未触止损 → MFE 取剩余各根的最大有利波动（片段末强平由账户结算）。
    """
    if direction not in (LONG, SHORT):
        raise DatasetError(f"未知方向: {direction!r}")
    bar = bars[bar_index]
    if direction == LONG:
        entry = float(bar.high) + tick_size
        stop = float(bar.low)
        best = 0.0
        for offset in range(bar_index + 1, len(bars)):
            other = bars[offset]
            if float(other.low) <= stop:  # 保守：先算不利侧 → 停止扫描
                break
            excursion = float(other.high) - entry
            if excursion > best:
                best = excursion
        risk = entry - stop
    else:
        entry = float(bar.low) - tick_size
        stop = float(bar.high)
        best = 0.0
        for offset in range(bar_index + 1, len(bars)):
            other = bars[offset]
            if float(other.high) >= stop:  # 保守：先算不利侧 → 停止扫描
                break
            excursion = entry - float(other.low)
            if excursion > best:
                best = excursion
        risk = stop - entry
    if not risk > 0:  # pragma: no cover - tick_size > 0 保证风险为正
        raise DatasetError("止损距离必须为正")
    return best / risk


def reversal_conditions(
    bars: tuple[Bar, ...], bar_index: int, direction: str, stop_price: float
) -> bool:
    """反转 K 线两条件（多头见模块 docstring；空头对称）。"""
    if bar_index + 1 >= len(bars):
        return False
    current = bars[bar_index]
    nxt = bars[bar_index + 1]
    if direction == LONG:
        if not (float(nxt.high) < float(current.high) and float(nxt.low) < float(current.low)):
            return False
    else:
        if not (float(nxt.high) > float(current.high) and float(nxt.low) > float(current.low)):
            return False
    # 条件②：先到止损价 → 成立；扫到片段末既未超当前 K 线也未触止损 → 同样成立
    condition2 = True
    for offset in range(bar_index + 1, len(bars)):
        other = bars[offset]
        if direction == LONG:
            if float(other.low) <= stop_price:
                break
            if float(other.high) > float(current.high):
                condition2 = False
                break
        else:
            if float(other.high) >= stop_price:
                break
            if float(other.low) < float(current.low):
                condition2 = False
                break
    return condition2


def select_open_action(
    reward_risk_long: float, reward_risk_short: float, threshold: float
) -> str | None:
    """空仓开仓判定：返回模型动作（``open_long``/``open_short``）或 ``None``（继续空仓）。

    冻结规则：盈亏比**严格大于**阈值的方向为候选；两向均达标取更优一侧；均不达标 →
    继续空仓。冻结「无并列问题」，此处仍给出确定性并列口径（比值相等取多）以便实现
    可复现。

    .. note::
       在冻结成交模型下「两向均达标」**不可达**：设 ``j_L`` 为第一根 ``low <= low[i]``
       的 K 线、``j_S`` 为第一根 ``high >= high[i]`` 的 K 线。多向 MFE > 0 需要
       ``∃ j < j_L: high[j] > high[i] + tick``，而该根必然满足 ``high[j] >= high[i]``
       → ``j_S <= j``，于是空向扫描在 ``j_S`` 即停止且此前各根 ``low > low[i]``
       （否则 ``j_L <= j_S`` 与 ``j_L > j`` 矛盾）→ 空向 MFE = 0。即任一决策点最多
       一个方向能有正盈亏比。规则仍按冻结语义实现（可直接对函数单测）。
    """
    long_ok = reward_risk_long > threshold
    short_ok = reward_risk_short > threshold
    if long_ok and short_ok:
        return ACTION_OPEN_LONG if reward_risk_long >= reward_risk_short else ACTION_OPEN_SHORT
    if long_ok:
        return ACTION_OPEN_LONG
    if short_ok:
        return ACTION_OPEN_SHORT
    return None


def holding_action(
    bars: tuple[Bar, ...],
    bar_index: int,
    *,
    direction: str,
    stop_price: float,
    tick_size: float,
    threshold: float,
) -> str:
    """持仓判定：``close`` / ``reverse`` / ``hold``（止损离场由账户层先行处理）。"""
    if not reversal_conditions(bars, bar_index, direction, stop_price):
        return ACTION_HOLD
    reverse_direction = SHORT if direction == LONG else LONG
    if reward_risk(bars, bar_index, reverse_direction, tick_size) > threshold:
        return ACTION_REVERSE
    return ACTION_CLOSE


def apply_flat_band(points: list[DecisionPoint], band: int) -> list[DecisionPoint]:
    """(b) 采样：机会分钟周边 ± ``band`` 的空仓分钟进入训练集，其余空仓分钟剔除。"""
    if band < 0:
        raise DatasetError("采样带宽不能为负")
    opportunity_bars = [
        point.bar_index for point in points if point.selection == SELECTION_OPPORTUNITY
    ]
    if not opportunity_bars:
        # 该片段没有任何机会分钟 → 空仓分钟全部剔除（避免把"人一眼可判断不该做"的
        # 分钟当成空仓示例）
        return [
            replace(
                point,
                selection=SELECTION_EXCLUDED_FLAT,
                selected=False,
            )
            if point.selection == SELECTION_FLAT
            else point
            for point in points
        ]
    result: list[DecisionPoint] = []
    for point in points:
        if point.selection != SELECTION_FLAT:
            result.append(point)
            continue
        near = any(abs(point.bar_index - bar) <= band for bar in opportunity_bars)
        result.append(
            replace(
                point,
                selection=SELECTION_FLAT_BAND if near else SELECTION_EXCLUDED_FLAT,
                selected=near,
            )
        )
    return result


def apply_event_lookforward(
    points: list[DecisionPoint], *, open_lookforward: int, exit_lookforward: int
) -> list[DecisionPoint]:
    """(b) 采样（train/dev）：事件决策记录 + 紧邻其前 N 条决策记录入选。

    * 事件判定（按 gold 动作）：``open_long``/``open_short`` → open 类事件（窗口
      ``open_lookforward``）；``close``/``reverse`` → exit 类事件（窗口
      ``exit_lookforward``）；``hold``/``stay_flat`` 非事件（程序止损离场根本不在
      决策序列中，死亡分钟同理——两者均不计入 N）；
    * 对每个事件（按列表序），把同片段内**紧邻其前 N 条决策记录**（位置
      ``max(0, i-N) .. i-1``）标记为前导；窗口不跨片段（函数按片段点列表操作）;
      去重按 bar（一条记录同时是多个事件的前导只保留一次；同时是 open 与 exit
      窗口前导也只保留一次，标签统一 ``event_lead``）；
    * 终态重标：事件记录保留其分支选择（open 事件 = opportunity、exit 事件 =
      holding，``selected=True``——事件包含自身，前导身份不覆盖事件身份）；前导 →
      ``event_lead``、``selected=True``；其余 flat → ``excluded_flat``、
      其余 holding → ``excluded_holding``（均 ``selected=False``）；
    * ``bar_index``/action/position 等其余字段不动；纯列表操作，无墙钟/随机。

    :raises DatasetError: 两个 N 非负整数校验失败（0 合法 = 只保留事件本身）
    """
    for name, value in (
        ("open_lookforward", open_lookforward),
        ("exit_lookforward", exit_lookforward),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DatasetError(f"事件前看条数必须为非负整数: {name}={value!r}")

    lead_positions: set[int] = set()
    for position, point in enumerate(points):
        if point.action in _OPEN_EVENT_ACTIONS:
            window = open_lookforward
        elif point.action in _EXIT_EVENT_ACTIONS:
            window = exit_lookforward
        else:
            continue
        for lead in range(max(0, position - window), position):
            lead_positions.add(lead)

    result: list[DecisionPoint] = []
    for position, point in enumerate(points):
        if point.action in _OPEN_EVENT_ACTIONS or point.action in _EXIT_EVENT_ACTIONS:
            # 事件包含自身：保留分支选择（opportunity/holding），必入选
            result.append(point)
        elif position in lead_positions:
            result.append(replace(point, selection=SELECTION_EVENT_LEAD, selected=True))
        elif point.selection == SELECTION_FLAT:
            result.append(replace(point, selection=SELECTION_EXCLUDED_FLAT, selected=False))
        else:
            # 持仓分钟（hold，未入选）：显式剔除类别（与空仓剔除分开计数）
            result.append(replace(point, selection=SELECTION_EXCLUDED_HOLDING, selected=False))
    return result


def evaluate_segment(
    segment: Segment,
    bars: tuple[Bar, ...],
    *,
    tick_size: float,
    params: EpisodeParams,
    source_data_version: str | None = None,
    initial_equity: float = 1.0,
    initial_peak: float | None = None,
) -> SegmentOutcome:
    """在片段内逐根确定性推进账户并生成规则真值标签。

    推进顺序（固定，保证确定性与无未来泄漏）：

    1. t−1 盯市（``mark_to_market``）→ 空仓时死亡判定；
    2. 空仓 → 开仓评估（可见未来仅用于计算 gold，状态仍只由 ≤ 决策 K 线数据构造）；
    3. 持仓 → 先判程序止损离场（不产生样本），再判死亡，最后判反转/持有。

    跨片段传递：``initial_equity`` / ``initial_peak``（缺省 1.0 / ``None``）由上一
    片段末 ``SegmentOutcome.final_equity`` / ``final_peak`` 传入；缺省行为与
    不传时逐值一致（``initial_peak=None`` → 账户层取 ``initial_equity``）。
    """
    if not bars:
        raise DatasetError(f"片段 {segment.segment_id} 没有 K 线")
    account = ReplayAccount(
        bars,
        tick_size=tick_size,
        initial_equity=initial_equity,
        initial_peak=initial_peak,
    )
    points: list[DecisionPoint] = []
    stop_exits: list[int] = []
    deaths: list[DeathEvent] = []
    segment_end_forced_close: int | None = None
    processed = 0

    for index, bar in enumerate(bars):
        processed += 1
        report = account.mark_to_market(index)
        if account.position is None:
            if report.is_dead(params.drawdown_threshold):
                # 死亡（空仓时无可强平）：episode 结束，本根不再产生决策样本
                deaths.append(DeathEvent(bar_index=index, forced_close=False))
                break
            long_rr = reward_risk(bars, index, LONG, tick_size)
            short_rr = reward_risk(bars, index, SHORT, tick_size)
            action = select_open_action(long_rr, short_rr, params.reward_risk_threshold)
            points.append(
                DecisionPoint(
                    bar_index=index,
                    action=action or ACTION_STAY_FLAT,
                    selection=SELECTION_OPPORTUNITY if action else SELECTION_FLAT,
                    selected=bool(action),
                    position=None,
                    drawdown=report.drawdown,
                    reward_risk_long=long_rr,
                    reward_risk_short=short_rr,
                )
            )
            if action == ACTION_OPEN_LONG:
                account.open_position(LONG, bar, reason=REASON_DECISION)
            elif action == ACTION_OPEN_SHORT:
                account.open_position(SHORT, bar, reason=REASON_DECISION)
            continue

        position = account.position
        if account.stop_touched(bar):
            # 程序自动止损离场：程序行为，不产生模型决策样本（止损被触早于反转判定）
            stop_exits.append(index)
            account.close_position(bar, reason=REASON_STOP)
            continue
        if report.is_dead(params.drawdown_threshold):
            # 死亡：检出当根按同一成交方法强平，episode 结束（本根不产生决策样本）
            deaths.append(DeathEvent(bar_index=index, forced_close=True))
            account.close_position(bar, reason=REASON_DEATH)
            break

        action = holding_action(
            bars,
            index,
            direction=position.direction,
            stop_price=position.stop_price,
            tick_size=tick_size,
            threshold=params.reward_risk_threshold,
        )
        points.append(
            DecisionPoint(
                bar_index=index,
                action=action,
                selection=SELECTION_HOLDING,
                selected=True,
                position=account.snapshot,
                drawdown=report.drawdown,
                reward_risk_long=None,
                reward_risk_short=None,
            )
        )
        if action == ACTION_CLOSE:
            account.close_position(bar, reason=REASON_DECISION)
        elif action == ACTION_REVERSE:
            reverse_direction = SHORT if position.direction == LONG else LONG
            account.reverse_position(reverse_direction, bar)
    else:
        # 片段末：仍有持仓 → 最后一根按同一成交方法强平结算（冻结）
        if account.position is not None:
            segment_end_forced_close = len(bars) - 1
            account.close_position(bars[-1], reason=REASON_SEGMENT_END)

    if segment.split_role in EVENT_LOOKFORWARD_SPLITS:
        # train/dev：事件 + 紧邻其前 N 条决策记录（按标签类型，label-band-sampling）
        points = apply_event_lookforward(
            points,
            open_lookforward=params.event_lookforward_open,
            exit_lookforward=params.event_lookforward_exit,
        )
    else:
        # test/calibration/ood：维持机会分钟周边 ± 带宽平带（原口径不变）
        points = apply_flat_band(points, params.flat_sample_band_minutes)
    return SegmentOutcome(
        segment_id=segment.segment_id,
        symbol=segment.symbol,
        split_role=segment.split_role,
        bar_count=len(bars),
        processed_bar_count=processed,
        decision_points=tuple(points),
        stop_exits=tuple(stop_exits),
        deaths=tuple(deaths),
        segment_end_forced_close=segment_end_forced_close,
        events=account.events,
        realized_pnl_ratio=account.realized_pnl,
        peak_equity=account.peak_equity,
        # 片段末已全部平仓结算（各离场路径均先 close）：净值 = 初始净值 + 已实现盈亏
        final_equity=account.initial_equity + account.realized_pnl,
        final_peak=account.peak_equity,
        source_data_version=source_data_version,
    )


__all__ = [
    "ACTION_CLOSE",
    "ACTION_HOLD",
    "ACTION_OPEN_LONG",
    "ACTION_OPEN_SHORT",
    "ACTION_REVERSE",
    "ACTION_STAY_FLAT",
    "EVENT_LOOKFORWARD_SPLITS",
    "SELECTION_EXCLUDED_FLAT",
    "SELECTION_EXCLUDED_HOLDING",
    "SELECTION_EVENT_LEAD",
    "SELECTION_FLAT",
    "SELECTION_FLAT_BAND",
    "SELECTION_HOLDING",
    "SELECTION_OPPORTUNITY",
    "DecisionPoint",
    "DeathEvent",
    "SegmentOutcome",
    "apply_event_lookforward",
    "apply_flat_band",
    "evaluate_segment",
    "holding_action",
    "reversal_conditions",
    "reward_risk",
    "select_open_action",
]

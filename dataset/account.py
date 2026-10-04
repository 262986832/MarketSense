"""账户域：确定性回放账户（决策 K 线成交模型 + 盯市/死亡 + 比值化记账）。

账户域职责：维护回放账户的全部执行语义——持仓状态（空仓 / 持多 / 持空）、
固定 1 手、累计净值（展示口径 = ``NET_VALUE_BASE = 100`` × equity，初始 100，
由后续任务接入展示；本模块先完成 ``initial_equity`` / ``initial_peak`` 参数化）、
今日收益（片段内权益变化）、回撤与死亡判定，以及收盘强平契约：
**交易日收盘（片段末）仍有持仓 → 按同一成交方法强平结算**。

职责边界：本模块只提供**执行语义**（成交价、持仓、止损锚定、盯市、死亡），不包含
任何标签判定（标签属 ``dataset.market_episode.labels`` 的 policy 层）；这样「未来
评估/未来执行器」与「本轮监督标签」可以共用同一套执行口径，而标签语义可独立演化。
本模块**仅依赖** ``dataset.errors``，可脱离 ``market_episode`` 独立使用；
``dataset.market_episode.replay`` 对账户域名字做 re-export 兼容 shim。

**执行语义（DESIGN 已冻结，2026-09-30 用户确认）**

* 决策点：每根**已收盘** 1 分钟 K 线收盘后产生；决策 K 线 = 刚收盘那根；
* 成交模型：买入（开多/平空）= 决策 K 线**最高价 + 1 tick**；
  卖出（开空/平多）= 决策 K 线**最低价 − 1 tick**（不利方向最差成交，
  1 tick 不利滑点内嵌，不另计手续费）；
* 止损锚定开仓决策 K 线对侧极值：多 = 开仓 K 线最低，空 = 开仓 K 线最高；
* 反手 = 同一决策点的两笔成交（先平旧仓、再开反向 1 手，两笔均锚定同一决策 K 线）；
* 账户：固定 1 手、仅多/空两向、不金字塔加仓；盯市以 **t−1** 数据为依据
  （决策 K 线序号 i 用 ``close[i-1]`` 盯市）；权益峰值自初始权益起累计，
  在**盯市时与结算后**两处更新（结算后更新 = T4.5 设计拍板：平仓结算权益
  计入高水位，浮盈变现不使下一片段峰值起点偏低；跨片段传递见
  ``ReplayAccount.__init__`` 的 ``initial_equity`` / ``initial_peak``）；
  回撤比 = (peak − equity)/peak ≥ 阈值即死亡，按检出当根同一成交方法强平；
* 片段末仍有持仓 → 最后一根按同一成交方法强平结算；
* 可见价格一律为**比值**（分母 = 片段首根**开盘价**，6 位小数），绝对价格只留在
  ``data/ohlcv/`` 原始行情层。

记账口径（实现阶段冻结项 4）：**比值化记账**——初始权益 1.0，1 手 = 1 单位名义，
不计合约乘数 / 保证金 / 手续费；因此单笔盈亏 = 价格比值之差（无量纲）。
"""

from __future__ import annotations

from dataclasses import dataclass

from dataset.errors import DatasetError

#: 持仓方向
LONG = "long"
SHORT = "short"
DIRECTIONS: tuple[str, str] = (LONG, SHORT)

#: 事件 reason（审计用；不参与任何判定）
REASON_DECISION = "decision"
REASON_STOP = "stop"
REASON_DEATH = "death"
REASON_SEGMENT_END = "segment_end"
REASON_REVERSE_CLOSE = "reverse_close"
REASON_REVERSE_OPEN = "reverse_open"


@dataclass(frozen=True)
class Bar:
    """片段内的一根已收盘 K 线（片段内 0 基下标 = 决策 K 线标识）。"""

    index: int
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    open_oi: int
    close_oi: int


@dataclass(frozen=True)
class Position:
    """当前持仓（绝对价格与比值表达并存；比值用于状态构造）。"""

    direction: str
    entry_price: float
    entry_ratio: float
    stop_price: float
    stop_ratio: float
    entry_bar_index: int


@dataclass(frozen=True)
class PositionSnapshot:
    """决策时点的仓位快照（状态可见部分；不含绝对价格）。"""

    direction: str
    entry_ratio: float
    stop_ratio: float


@dataclass(frozen=True)
class TradeEvent:
    """一笔成交（确定性审计轨迹；绝对价格仅供复算，不进入模型输入）。"""

    bar_index: int
    side: str  # "buy" / "sell"
    direction: str  # 该笔成交建立/了结的持仓方向
    reason: str
    price: float
    price_ratio: float
    pnl_ratio: float | None  # 平仓笔的实现盈亏（比值口径）；开仓笔为 None


@dataclass(frozen=True)
class MarkReport:
    """决策时点的盯市报告（t−1 口径）。"""

    bar_index: int
    mark_price: float | None
    equity: float
    peak_equity: float
    drawdown: float

    def is_dead(self, threshold: float) -> bool:
        return self.drawdown >= threshold


def ratio_or_none(numerator: float, denominator: float) -> float | None:
    """比值；分母 ≤ 0 时无定义（返回 ``None``，绝不产生 ``inf``）。"""
    return None if denominator <= 0 else float(numerator) / float(denominator)


def format_ratio_value(value: float | None, precision: int) -> str:
    """比值 → 固定小数位文本（无定义写 ``na``；双跑逐字节一致）。"""
    return "na" if value is None else f"{float(value):.{int(precision)}f}"


class ReplayAccount:
    """确定性回放账户：固定 1 手，多/空两向，比值化记账。

    调用约定（由 policy 层驱动，顺序固定）：

    1. 决策时点先 ``mark_to_market(i)``（t−1 盯市，含峰值更新）；
    2. 持仓时先 ``stop_touched(bar)`` 判程序止损（早于反转判定）；
    3. 需要时调用 ``open_position`` / ``close_position``（结算后更新峰值） /
       ``reverse_position``（内含先平仓，同享结算后峰值更新）。
    """

    __slots__ = (
        "_bars",
        "_reference_price",
        "_tick_size",
        "_initial_equity",
        "_initial_peak",
        "_position",
        "_realized_pnl",
        "_peak_equity",
        "_events",
    )

    def __init__(
        self,
        bars: tuple[Bar, ...],
        *,
        tick_size: float,
        initial_equity: float = 1.0,
        initial_peak: float | None = None,
    ) -> None:
        """构造回放账户。

        :param initial_equity: 初始权益（比值化记账，默认 1.0；跨片段传递时由
            前一片段末结算净值传入）。
        :param initial_peak: 初始权益峰值（缺省 ``None`` → 取 ``initial_equity``；
            跨片段传递时由前一片段末峰值传入）。
        """
        if not bars:
            raise DatasetError("回放账户需要非空 K 线序列")
        if not float(tick_size) > 0:
            raise DatasetError("tick_size 必须为正数")
        if not float(initial_equity) > 0:
            raise DatasetError("initial_equity 必须为正数")
        if initial_peak is not None and not float(initial_peak) > 0:
            raise DatasetError("initial_peak 必须为正数")
        self._bars = bars
        # 冻结项 1：分母 = 片段首根参考价 = 首根开盘价（首根成交前第一个已知价格）
        self._reference_price = float(bars[0].open)
        if not self._reference_price > 0:
            raise DatasetError(
                f"片段首根开盘价必须为正数（比值表达的分母）: {bars[0].open!r}"
            )
        self._tick_size = float(tick_size)
        # 跨片段传递（拍板 5A）：净值/峰值同机制传递，缺省行为与迁移前一致
        # （initial_equity = 1.0，initial_peak = initial_equity）
        self._initial_equity = float(initial_equity)
        self._initial_peak = (
            float(initial_peak) if initial_peak is not None else self._initial_equity
        )
        self._position: Position | None = None
        self._realized_pnl = 0.0
        self._peak_equity = self._initial_peak
        self._events: list[TradeEvent] = []

    # ------------------------------------------------------------------ 只读视图
    @property
    def bars(self) -> tuple[Bar, ...]:
        return self._bars

    @property
    def reference_price(self) -> float:
        return self._reference_price

    @property
    def tick_size(self) -> float:
        return self._tick_size

    @property
    def initial_equity(self) -> float:
        return self._initial_equity

    @property
    def initial_peak(self) -> float:
        return self._initial_peak

    @property
    def position(self) -> Position | None:
        return self._position

    @property
    def snapshot(self) -> PositionSnapshot | None:
        """决策时点的仓位快照（比值口径，绝对价格不进入可见状态）。"""
        position = self._position
        if position is None:
            return None
        return PositionSnapshot(
            direction=position.direction,
            entry_ratio=position.entry_ratio,
            stop_ratio=position.stop_ratio,
        )

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def events(self) -> tuple[TradeEvent, ...]:
        return tuple(self._events)

    # ------------------------------------------------------------------ 价格口径
    def ratio(self, price: float) -> float:
        """绝对价格 → 可见比值（分母 = 片段首根开盘价）。"""
        return float(price) / self._reference_price

    def buy_price(self, bar: Bar) -> float:
        """买入成交价 = 决策 K 线最高 + 1 tick（不利方向最差成交）。"""
        return float(bar.high) + self._tick_size

    def sell_price(self, bar: Bar) -> float:
        """卖出成交价 = 决策 K 线最低 − 1 tick（不利方向最差成交）。"""
        return float(bar.low) - self._tick_size

    # ------------------------------------------------------------------ 盯市/死亡
    def mark_to_market(self, bar_index: int) -> MarkReport:
        """按 t−1 口径盯市并更新权益峰值（决策时点只使用 ≤ 决策 K 线的数据）。

        权益 = ``initial_equity`` + 已实现盈亏 + 持仓浮盈；持仓浮盈用
        ``close[bar_index - 1]`` 计算；``bar_index == 0`` 时账户尚未有持仓
        （初始空仓），故盯市价为空。
        """
        mark_price: float | None = None
        equity = self._initial_equity + self._realized_pnl
        if self._position is not None and bar_index >= 1:
            mark_price = float(self._bars[bar_index - 1].close)
            mark_ratio = self.ratio(mark_price)
            if self._position.direction == LONG:
                equity += mark_ratio - self._position.entry_ratio
            else:
                equity += self._position.entry_ratio - mark_ratio
        if equity > self._peak_equity:
            self._peak_equity = equity
        peak = self._peak_equity
        drawdown = 0.0 if peak <= 0 else (peak - equity) / peak
        return MarkReport(
            bar_index=bar_index,
            mark_price=mark_price,
            equity=equity,
            peak_equity=peak,
            drawdown=drawdown,
        )

    # ------------------------------------------------------------------ 成交
    def stop_touched(self, bar: Bar) -> bool:
        """当前持仓的止损价是否被该根 K 线触及（开仓当根不判定，由调用方保证）。"""
        position = self._position
        if position is None:
            return False
        if position.direction == LONG:
            return float(bar.low) <= position.stop_price
        return float(bar.high) >= position.stop_price

    def open_position(self, direction: str, bar: Bar, *, reason: str = REASON_DECISION) -> TradeEvent:
        """按决策 K 线成交模型开仓（止损锚定该 K 线对侧极值）。"""
        if direction not in DIRECTIONS:
            raise DatasetError(f"未知持仓方向: {direction!r}")
        if self._position is not None:
            raise DatasetError("已有持仓，不能重复开仓")
        if direction == LONG:
            entry_price = self.buy_price(bar)
            stop_price = float(bar.low)
        else:
            entry_price = self.sell_price(bar)
            stop_price = float(bar.high)
        position = Position(
            direction=direction,
            entry_price=entry_price,
            entry_ratio=self.ratio(entry_price),
            stop_price=stop_price,
            stop_ratio=self.ratio(stop_price),
            entry_bar_index=bar.index,
        )
        self._position = position
        event = TradeEvent(
            bar_index=bar.index,
            side="buy" if direction == LONG else "sell",
            direction=direction,
            reason=reason,
            price=entry_price,
            price_ratio=position.entry_ratio,
            pnl_ratio=None,
        )
        self._events.append(event)
        return event

    def close_position(self, bar: Bar, *, reason: str) -> TradeEvent:
        """按决策 K 线成交模型平掉当前持仓并结算（比值口径）。

        结算后更新权益峰值（T4.5，设计拍板 §2）：结算权益 =
        ``initial_equity`` + 已实现盈亏，峰值 = max(峰值, 结算权益)——
        片段末强平结算净值可能高于盯市峰值（浮盈变现），不更新则下一片段
        峰值起点偏低、首根盯市出现负回撤。各离场路径（决策平仓/止损/死亡/
        片段末强平/反手先平）均经本方法结算，峰值更新单点生效。
        """
        position = self._position
        if position is None:
            raise DatasetError("无持仓，不能平仓")
        exit_price = self.sell_price(bar) if position.direction == LONG else self.buy_price(bar)
        exit_ratio = self.ratio(exit_price)
        pnl_ratio = (
            exit_ratio - position.entry_ratio
            if position.direction == LONG
            else position.entry_ratio - exit_ratio
        )
        self._realized_pnl += pnl_ratio
        self._position = None
        # 结算后峰值更新（盯市更新保持不变，见 mark_to_market）
        settled_equity = self._initial_equity + self._realized_pnl
        if settled_equity > self._peak_equity:
            self._peak_equity = settled_equity
        event = TradeEvent(
            bar_index=bar.index,
            side="sell" if position.direction == LONG else "buy",
            direction=position.direction,
            reason=reason,
            price=exit_price,
            price_ratio=exit_ratio,
            pnl_ratio=pnl_ratio,
        )
        self._events.append(event)
        return event

    def reverse_position(
        self, direction: str, bar: Bar
    ) -> tuple[TradeEvent, TradeEvent]:
        """反手：同一决策点两笔成交（先平旧仓、再开反向 1 手，均锚定该决策 K 线）。"""
        position = self._position
        if position is None:
            raise DatasetError("无持仓，不能反手")
        if direction not in DIRECTIONS or direction == position.direction:
            raise DatasetError(f"反手方向非法: {direction!r}")
        close_event = self.close_position(bar, reason=REASON_REVERSE_CLOSE)
        open_event = self.open_position(direction, bar, reason=REASON_REVERSE_OPEN)
        return close_event, open_event


__all__ = [
    "DIRECTIONS",
    "LONG",
    "SHORT",
    "REASON_DEATH",
    "REASON_DECISION",
    "REASON_REVERSE_CLOSE",
    "REASON_REVERSE_OPEN",
    "REASON_SEGMENT_END",
    "REASON_STOP",
    "Bar",
    "MarkReport",
    "Position",
    "PositionSnapshot",
    "ReplayAccount",
    "TradeEvent",
    "format_ratio_value",
    "ratio_or_none",
]

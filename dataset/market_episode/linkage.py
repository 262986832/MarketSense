"""联动行突破动量（linkage-breakthrough）纯函数域。

三态突破信号（相邻两根 K 线，严格比较）::

    signal(prev, cur) = +1  若 cur.high > prev.high 且 cur.close > prev.close
                      = -1  若 cur.low  < prev.low  且 cur.close < prev.close
                      =  0  其余

动量 = 末尾 N 根（已收盘）K 线/桶的相邻对信号的线性加权平均，权重 1..N-1
（最近信号权重最大）。

**硬约束（AGENTS.md §7，本模块全程成立）**

* 无未来数据泄漏：只使用**已收盘**数据——
  - 1m 路径（``duration_seconds == 60``）：调用方传入**截至决策 K 线 T（刚收盘）**
    的 1m 序列，逐根天然已收盘，直接可用；
  - 非 1m 路径：T 所在 Nm 桶在 T 收盘时刻可能**未收满**（如 5m 桶 21:05 起、
    T=21:07 收盘时只含 3 根），未收满桶不可用——由 :func:`resample_bars` 的
    ``is_closed`` 判定排除，:func:`breakthrough_momentum` 只取已收满桶。
* 确定性：相同输入产生相同输出；无墙钟、无随机源、无外部 IO。
* 纯 Python 求和（不用 numpy），与审计侧独立复算可逐项对照。

本模块与状态渲染解耦（渲染接线属 ``nanojev_records``，STATE_SCHEMA v10 的后续任务）。

联动品种扩展（linkage-symbol，STATE_SCHEMA v11 准备）：:func:`align_bars_by_timestamp`
按时间戳交集对齐主/联动品种窗口，:func:`linkage_breakthrough_momentum` 与
:func:`pearson_correlation` 在交集序列上计算联动突破值与皮尔逊相关度（纯 Python，
口径同上）；:func:`signal_pairs_from_aligned` 是主品种对照信号的唯一入口
（保证与联动品种用同一交集序列，逐点对应）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from dataset.account import Bar
from dataset.errors import DatasetError

__all__ = [
    "ResampledBar",
    "align_bars_by_timestamp",
    "breakthrough_momentum",
    "breakthrough_signal",
    "daily_linkage_correlation",
    "linkage_breakthrough_momentum",
    "pearson_correlation",
    "resample_bars",
    "signal_pairs_from_aligned",
]


@dataclass(frozen=True)
class ResampledBar:
    """一根重采样 K 线（1m 桶聚合，时间戳 = epoch 秒）。

    ``start_ts`` / ``end_ts`` 为桶起止的 epoch 秒（``end_ts = start_ts + duration``）；
    ``is_closed`` 见 :func:`resample_bars`。
    """

    open: float
    high: float
    low: float
    close: float
    volume: int
    open_oi: int
    close_oi: int
    start_ts: int
    end_ts: int
    is_closed: bool


def _open_epoch_seconds(bar: Bar) -> int:
    """bar 开盘时刻的 epoch 秒（整数运算，避免浮点误差；输入时间戳为含时区 ISO 文本）。"""
    return int(pd.Timestamp(bar.timestamp).value) // 10**9


def breakthrough_signal(prev: Bar | ResampledBar, cur: Bar | ResampledBar) -> int:
    """相邻两根 K 线/桶的三态突破信号（严格比较，相等 → 0）。

    +1：``cur.high > prev.high`` 且 ``cur.close > prev.close``（涨势突破）；
    -1：``cur.low < prev.low`` 且 ``cur.close < prev.close``（跌势突破，对称）；
    其余（含任一侧相等）→ 0。

    只读 ``high``/``low``/``close`` 三个属性（鸭子类型）；``Bar`` 与
    ``ResampledBar`` 均满足，1m 与重采样路径共用同一判定。
    """
    if cur.high > prev.high and cur.close > prev.close:
        return 1
    if cur.low < prev.low and cur.close < prev.close:
        return -1
    return 0


def _weighted_signal_average(signals: Sequence[int]) -> float | None:
    """信号序列的线性加权平均（权重 1..n，时间升序、最近最大）；空序列 → ``None``。

    ``signals[i]`` 的权重 = ``i + 1``；``momentum = Σ(w_i · s_i) / Σ(w_i)``
    （纯 Python 求和）。「信号序列 → 加权」一步由 :func:`breakthrough_momentum`
    与 :func:`linkage_breakthrough_momentum` 共用（同一口径，不复制逻辑）。
    """
    if not signals:
        return None
    numerator = 0.0
    denominator = 0
    for position, signal in enumerate(signals):
        weight = position + 1
        numerator += weight * signal
        denominator += weight
    return numerator / denominator


def resample_bars(bars: Sequence[Bar], duration_seconds: int) -> tuple[ResampledBar, ...]:
    """1m K 线 → ``duration_seconds`` 桶（按 bar 开盘时刻对齐时间边界）。

    * 桶起点 = ``floor(open_ts / duration) * duration``（``open_ts`` = bar 开盘时刻的
      epoch 秒）；与 tqsdk 原生周期对齐语义同源；休市空洞不影响（休市后下一根按
      自身时间重新对齐桶边界，跨休市不拼接）；
    * 聚合：open = 首根 open；high = max；low = min；close = 末根 close；
      volume = sum；open_oi = 首根；close_oi = 末根；
    * ``is_closed``：下一根 1m bar（下一桶首根）的开盘时刻 ≥ 桶终点即收满；
      **无下一根 → False**（片段尾桶不收满不可用）。

    输入 bar 时间戳须升序（上游 ``load_segment_bars`` 已保证）；本函数不做重排序。
    """
    if duration_seconds < 1:
        raise DatasetError(f"duration_seconds 必须 ≥ 1，实际: {duration_seconds}")
    buckets: dict[int, list[Bar]] = {}
    for bar in bars:
        start = (_open_epoch_seconds(bar) // duration_seconds) * duration_seconds
        buckets.setdefault(start, []).append(bar)
    starts = sorted(buckets)
    resampled: list[ResampledBar] = []
    for position, start in enumerate(starts):
        group = buckets[start]
        end = start + duration_seconds
        if position + 1 < len(starts):
            next_open = _open_epoch_seconds(buckets[starts[position + 1]][0])
            is_closed = next_open >= end
        else:
            is_closed = False
        resampled.append(
            ResampledBar(
                open=group[0].open,
                high=max(bar.high for bar in group),
                low=min(bar.low for bar in group),
                close=group[-1].close,
                volume=sum(bar.volume for bar in group),
                open_oi=group[0].open_oi,
                close_oi=group[-1].close_oi,
                start_ts=start,
                end_ts=end,
                is_closed=is_closed,
            )
        )
    return tuple(resampled)


def breakthrough_momentum(
    bars: Sequence[Bar], window: int, duration_seconds: int
) -> float | None:
    """突破信号的线性加权动量 ∈ [-1, +1]；可用根数不足 2 → ``None``。

    :param bars: **截至决策 K 线 T（刚收盘）** 的 1m 序列（不跨片段延伸）；
    :param window: 最多取末尾 ``window`` 根已收盘 K 线/桶（窗口不足用可用根数）；
    :param duration_seconds: 60 → 直接用 ``bars``（1m 逐根天然收满）；否则重采样并
        只取 ``is_closed=True`` 的桶（T 所在未收满桶自动排除）。

    权重与信号数：``n = min(window, 可用根数)`` 根产生 **n-1 个相邻对信号**
    （首根无 prev），权重 1..n-1（最近信号权重 = n-1）；
    ``momentum = Σ(w_i · s_i) / Σ(w_i)``（纯 Python 求和）。
    ``n < 2``（含空序列与片段首根）→ ``None``（渲染层映射 ``突破=na``）。
    """
    if window < 1:
        raise DatasetError(f"window 必须 ≥ 1，实际: {window}")
    if duration_seconds == 60:
        closed: Sequence[Bar | ResampledBar] = bars
    else:
        closed = [
            resampled
            for resampled in resample_bars(bars, duration_seconds)
            if resampled.is_closed
        ]
    count = min(window, len(closed))
    if count < 2:
        return None
    sequence = closed[-count:]
    signals = [
        breakthrough_signal(sequence[position - 1], sequence[position])
        for position in range(1, count)
    ]
    return _weighted_signal_average(signals)


def align_bars_by_timestamp(
    primary_bars: Sequence[Bar], secondary_bars: Sequence[Bar], window: int
) -> tuple[tuple[Bar, Bar], ...]:
    """主/联动品种 K 线按时间戳交集对齐（保序对序列）。

    :param primary_bars: 主品种 1m 序列，**必须是截至决策 K 线 T（刚收盘）的切片**
        （调用方保证，与 :func:`breakthrough_momentum` 的输入口径一致）；本函数取其
        末尾 ``min(window, len)`` 根作为主品种窗口；
    :param secondary_bars: 联动品种 1m 序列；其中每根须为已收盘 K 线且收盘时刻
        ≤ T 收盘时刻——交集按主品种时间戳取值，secondary 时间戳晚于窗口末根者
        天然不会入选（上界由交集天然保证），其余约束由调用方保证，本函数不额外检查；
    :param window: 主品种窗口根数上限（≥ 1；不足用可用根数）。

    返回**保序**对序列 ``((p1, s1), (p2, s2), ...)``：只含双方都存在的时间戳
    （secondary 缺分钟 → 该主品种 K 线被剔除；secondary 独有分钟永不入选）。
    时间戳比较用 :class:`pd.Timestamp` 归一（``Bar.timestamp`` 为含时区 ISO 文本，
    同型比较，与 :func:`_open_epoch_seconds` 同一解析口径）。空窗口 → ``()``。

    :raises DatasetError: ``window < 1``
    """
    if window < 1:
        raise DatasetError(f"window 必须 ≥ 1，实际: {window}")
    count = min(window, len(primary_bars))
    if count < 1:
        return ()
    secondary_by_ts: dict[pd.Timestamp, Bar] = {
        pd.Timestamp(bar.timestamp): bar for bar in secondary_bars
    }
    pairs: list[tuple[Bar, Bar]] = []
    for bar in primary_bars[-count:]:
        secondary = secondary_by_ts.get(pd.Timestamp(bar.timestamp))
        if secondary is not None:
            pairs.append((bar, secondary))
    return tuple(pairs)


def linkage_breakthrough_momentum(
    aligned_pairs: Sequence[tuple[Bar, Bar]],
) -> float | None:
    """联动品种突破值：对齐对序列的 secondary 侧相邻对三态信号线性加权 ∈ [-1, +1]。

    口径与 :func:`breakthrough_momentum` 完全一致（复用 :func:`breakthrough_signal`
    与共用加权函数；窗口截断已在对齐时完成）：``n = len(aligned_pairs)`` 对产生
    **n-1 个相邻对信号**，权重 1..n-1（最近信号权重 = n-1）；
    ``n < 2``（含空对序列）→ ``None``（渲染层映射 ``突破=na``）。
    """
    secondary = [cur for _, cur in aligned_pairs]
    signals = [
        breakthrough_signal(secondary[position - 1], secondary[position])
        for position in range(1, len(secondary))
    ]
    return _weighted_signal_average(signals)


def signal_pairs_from_aligned(
    aligned_pairs: Sequence[tuple[Bar, Bar]],
) -> tuple[tuple[int, int], ...]:
    """对齐对序列的相邻对信号对 ``(primary_signal, secondary_signal)``（n-1 个）。

    相关度用的**主品种对照信号必须经本函数取得**：主/联动两侧用同一交集序列
    （相邻对跨过被剔除的分钟，保证逐点对应）；不得对主品种全窗口另行计算
    （交集剔除分钟时两者不同）。空/单对 → ``()``。
    """
    pairs: list[tuple[int, int]] = []
    for position in range(1, len(aligned_pairs)):
        prev_primary, prev_secondary = aligned_pairs[position - 1]
        cur_primary, cur_secondary = aligned_pairs[position]
        pairs.append(
            (
                breakthrough_signal(prev_primary, cur_primary),
                breakthrough_signal(prev_secondary, cur_secondary),
            )
        )
    return tuple(pairs)


def daily_linkage_correlation(
    primary_rows: Sequence[tuple[object, float, float, float]],
    secondary_rows: Sequence[tuple[object, float, float, float]],
    trade_date: object,
    window: int,
) -> float | None:
    """交易日前最近交集日线窗口的三态突破信号 Pearson r。

    Rows are (trade_date, high, low, close), in source order. The decision date
    itself and later rows are excluded before the stable date intersection.
    """
    if window < 1:
        raise DatasetError(f"window 必须 ≥ 1，实际: {window}")
    primary = {row[0]: row[1:] for row in primary_rows if row[0] < trade_date}
    secondary = {row[0]: row[1:] for row in secondary_rows if row[0] < trade_date}
    dates = sorted(primary.keys() & secondary.keys())[-window:]
    if len(dates) < 3:
        return None
    primary_signals: list[int] = []
    secondary_signals: list[int] = []
    for previous, current in zip(dates, dates[1:]):
        p0, p1 = primary[previous], primary[current]
        s0, s1 = secondary[previous], secondary[current]
        primary_signals.append(
            1 if p1[0] > p0[0] and p1[2] > p0[2]
            else -1 if p1[1] < p0[1] and p1[2] < p0[2] else 0
        )
        secondary_signals.append(
            1 if s1[0] > s0[0] and s1[2] > s0[2]
            else -1 if s1[1] < s0[1] and s1[2] < s0[2] else 0
        )
    return pearson_correlation(primary_signals, secondary_signals)


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """皮尔逊相关系数 r（纯 Python，无 numpy）；无定义边界 → ``None``。

    ``r = Σ((x-x̄)(y-ȳ)) / √(Σ(x-x̄)²·Σ(y-ȳ)²)``；``len(xs) < 2``（不足一对
    相邻信号）或任一序列**零方差**（Σ平方差 == 0）→ ``None``（渲染层映射
    ``相关度=na``，不静默取 0）。

    :raises DatasetError: 两序列长度不一致（逐点对应的信号序列长度由调用方
        保证一致；静默截断会掩盖口径错误，显式拒绝）。
    """
    if len(xs) != len(ys):
        raise DatasetError(f"皮尔逊相关度两序列长度必须一致: {len(xs)} != {len(ys)}")
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    if variance_x == 0 or variance_y == 0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)

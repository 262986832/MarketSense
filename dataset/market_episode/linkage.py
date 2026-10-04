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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from dataset.account import Bar
from dataset.errors import DatasetError

__all__ = [
    "ResampledBar",
    "breakthrough_momentum",
    "breakthrough_signal",
    "resample_bars",
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
    numerator = 0.0
    denominator = 0
    for weight in range(1, count):
        prev, cur = sequence[weight - 1], sequence[weight]
        numerator += weight * breakthrough_signal(prev, cur)
        denominator += weight
    return numerator / denominator

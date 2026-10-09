"""转折点提取与落盘。

检测规则（触发条件与状态机，沿用本项目早期版本的转折点脚本，
不做任何增删或阈值调整）：

```text
上涨状态:
    只要当前 K 线 Low >= 前一根 K 线 Low，就继续保持上涨；
    一旦当前 K 线 Low < 前一根 K 线 Low:
        在该根 K 线确认一个 ``up`` 点（上涨段终结），状态切换为下跌。
下跌状态:
    只要当前 K 线 High <= 前一根 K 线 High，就继续保持下跌；
    一旦当前 K 线 High > 前一根 K 线 High:
        在该根 K 线确认一个 ``down`` 点（下跌段终结），状态切换为上涨。
```

细则：比较运算符**严格**（``==`` 不转向）；每个状态每根 K 线只做**一次**比较
（触发转向的那根 K 线切换后不按新状态重判——同根同时创新高与创新低只发射一个
点，不级联）；末尾补最后一根 K 线收盘价（进行中的段尚未确认，不输出）。

移植自本项目早期版本的转折点脚本（新增落盘/读取
``save_turning_points`` / ``load_turning_points``）：**检测（触发序列）与早期版本
一致；发射字段按 2026-09-26 甲口径偏离**（见下文）。

点信息增补（2026-09-24 需求）
----------------------------

每个转折点除时间与价格外，还携带其**值根** K 线的：

* ``volume``：值根 K 线时间范围内的成交量合计；
* ``oi``：值根 K 线**结束时刻**的持仓量（天勤 ``close_oi`` 口径；与主流行情软件的
  「K 线持仓量」一致。``open_oi`` 口径可由 ``bar_index`` 关联 K 线 CSV 离线补算）。

值根由 kind 推导：``start`` / ``close`` 点为自身那根 K 线；``up`` / ``down`` 点为
确认根的前一根（``bar_index − 1``，可推导）。

并派生相邻点之间的**相对值**（由点序列确定性推导，无墙钟时间）：

* ``dt_minutes``：当前点与前一点 ``timestamp`` 之差，单位分钟（1 分钟）；
* ``price_ratio``：当前点价格 / 前一点价格（比值；减 1 即变化幅度）；
* ``volume_ratio``：当前点成交量 / 前一点成交量；
* ``oi_ratio``：当前点持仓量 / 前一点持仓量。

首个点无前一点 → 四个相对值均为空；前一点值为 0 时比值无定义（真实存在无成交
分钟），同样留空，不产生 ``inf``。检测（触发序列）不受增补影响。

发射口径（2026-09-26 甲口径）
-----------------------------

* ``kind`` 集合为 ``start`` / ``up`` / ``down`` / ``close``（2026-09-25 起反转类
  kind 由旧名 ``high``/``low`` 更名为 ``up``/``down``）。
* ``up`` / ``down`` 点在**触发根** ``t`` 确认：``timestamp`` / ``bar_index`` =
  ``t``；``price`` / ``volume`` / ``oi`` 取**前一根**（值根 ``t − 1``）：``up`` 点
  ``price`` = ``highs[t − 1]``，``down`` 点 ``price`` = ``lows[t − 1]``。
* 检测循环 ``t`` 从 1 开始，值根 ``t − 1`` 恒存在，无任何边界特判。
* ``start``（首根开盘）与 ``close``（末根收盘）口径不变。
* 相对 2026-09-25 旧口径，``up``/``down`` 点的字段含义变化；新旧落盘文件列集合
  相同，:func:`load_turning_points` 不会拒绝旧文件，但值含义不同，依赖方须以
  ``python -m dataset turning-points`` / ``prepare`` 重新生成（无兼容层）。

点信息增补（2026-10-02 当次趋势极值）
-------------------------------------

``up`` / ``down`` 点额外携带其**当次趋势段**的实际极值（检测触发序列不变，
只增不删）：

* 段范围 = 闭区间 ``[s, t − 1]``：``s`` 为上一个**已确认** up/down 点的触发根
  ``bar_index``（窗口内无前序反转点时初始段 ``s = 0``，即 start 点所在根）；
  触发根 ``t`` 不属于当次段——它属于下一段（up 触发根创新低，是下一段下跌腿的
  首根动力；down 对称）。段长 = ``t − s``（恒 ≥ 1；相邻两次触发真实存在，
  同根双条件后下一根立即反向时 ``t = s + 1``，段 = ``[s, s]`` 单根段非退化）。
* 极值语义（用户确认）：``up`` 点 = 高点，``trend_extreme_price`` = 段内实际最高
  ``max(high[s..t−1])``；``down`` 点 = 低点，段内实际最低 ``min(low[s..t−1])``。
* 极值所在 bar = 段内**首个**达到极值的 K 线（平局取最早，与
  ``max(range, key=…)`` 首次出现语义一致，确定性）。
* 不变式：up 点 ``trend_extreme_price ≥ price``（``price`` = 段末根高点）、
  down 点 ``≤ price``；相等当且仅当极值就在值根（段末根）。
* ``start`` / ``close`` 点及收尾进行中的段不产生极值（两字段为 ``None``，
  CSV 空单元格）；sidecar（窗口元信息）不变。
* 落盘列由 11 列扩为 13 列，新列 ``trend_extreme_price`` /
  ``trend_extreme_bar_index`` 插在 ``oi`` 之后、相对值之前（保持「绝对值在前，
  相对值在后」分组）。旧 11 列文件 :func:`load_turning_points` 按「缺少必需列」
  拒绝 → 必须用 ``python -m dataset turning-points`` 重新生成（无兼容层）。

数据层子函数 :func:`recent_trend_extremes`（详见函数 docstring）可从点序列取
最近 ``n`` 个高点与 ``n`` 个低点的段极值（:class:`TrendExtreme`，含段长；段长
由点序列确定性推导，不入 CSV）。

趋势状态分类（2026-10-03 v9）
-----------------------------

:func:`trend_state_direction` 对 :func:`recent_trend_extremes`（``n=2``）返回的
两个时间倒序元组做**纯比较**分类（v9 训练状态日线行「趋势=」键的三分支判据，见
``artifacts/trend-state-v9/02-design/tech-design.md``）：

* ``"up"``（涨势中）：高/低两方向最近项（``-1``）段极值**均严格大于**次近项（``-2``）；
* ``"down"``（跌势中）：均严格小于；
* ``"range"``（震荡）：其余，含任一方向相等（判据不含等号）。

纯比较、无选择/过滤逻辑（「审计侧独立实现不调用数据层选择函数」的边界不被破坏）；
:func:`recent_trend_extremes` 本体不变（v9 调用侧改传 ``n=2``）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from dataset.errors import DataLoadError, DatasetError
from dataset.ohlcv import TIMEZONE
from dataset.storage import (
    commit_data_with_sidecar,
    ensure_directory,
    read_csv_table,
    OUTPUT_FORMAT,
    SIDECAR_SUFFIX,
)

#: 初始方向模式
INITIAL_DIRECTION_MODES: Final[tuple[str, ...]] = ("auto", "up", "down")
#: 转折点类型
TURNING_POINT_KINDS: Final[tuple[str, ...]] = ("start", "up", "down", "close")
#: 落盘列顺序（数据契约）：绝对值在前，相对值（相邻点派生）在后
TURNING_POINT_COLUMNS: Final[tuple[str, ...]] = (
    "point_index",
    "kind",
    "timestamp",
    "price",
    "bar_index",
    "volume",
    "oi",
    "trend_extreme_price",
    "trend_extreme_bar_index",
    "dt_minutes",
    "price_ratio",
    "volume_ratio",
    "oi_ratio",
)

#: 相对值列（由 :func:`relative_metrics` 从点序列确定性派生）
RELATIVE_COLUMNS: Final[tuple[str, ...]] = (
    "dt_minutes",
    "price_ratio",
    "volume_ratio",
    "oi_ratio",
)


@dataclass(frozen=True)
class TurningPoint:
    """路径上的一个点。

    :param kind: ``start`` | ``up`` | ``down`` | ``close``（2026-09-25 起
        ``up``/``down`` 取代旧名 ``high``/``low``）
    :param timestamp: 确认根 K 线的开始时间（``up``/``down`` 点 = 触发根 ``t``；
        ``start``/``close`` = 首根/末根）
    :param price: 该点的价格（``up`` = 值根最高价、``down`` = 值根最低价、
        起点开盘价、终点收盘价；``up``/``down`` 的值根 = ``bar_index − 1``，可推导）
    :param bar_index: 在传入窗口中的 0 基下标（确认根；``up``/``down`` 点的值根
        = ``bar_index − 1``，可推导）
    :param volume: 值根 K 线的成交量（K 线时间范围内的成交量合计）
    :param oi: 值根 K 线结束时刻的持仓量（天勤 ``close_oi`` 口径）
    :param trend_extreme_price: 当次趋势段实际极值（up 点 = 段内最高、down 点 =
        段内最低；2026-10-02 起 up/down 点填充；start/close 为 ``None``）
    :param trend_extreme_bar_index: 极值所在 bar（窗口 0 基，平局取最早；
        start/close 为 ``None``）
    """

    kind: str
    timestamp: pd.Timestamp
    price: float
    bar_index: int
    volume: int
    oi: int
    trend_extreme_price: float | None = None
    trend_extreme_bar_index: int | None = None


@dataclass(frozen=True)
class TrendExtreme:
    """一个 up/down 点所在趋势段的当次极值（2026-10-02）。

    :param kind: ``"up"``（段内实际最高）| ``"down"``（段内实际最低）
    :param trend_extreme_price: 段内实际最高/最低价
    :param trend_extreme_bar_index: 极值所在 bar（窗口 0 基，平局取最早）
    :param segment_length: 段长（K 线数量，≥ 1；由点序列确定性推导）
    :param confirmation_bar_index: 确认根索引（``TurningPoint.bar_index``）
    :param confirmation_timestamp: 确认根时间戳；同索引下作为确定性顺序键
    """

    kind: str
    trend_extreme_price: float
    trend_extreme_bar_index: int
    segment_length: int
    confirmation_bar_index: int
    confirmation_timestamp: pd.Timestamp | None = field(default=None, compare=False)


@dataclass(frozen=True)
class RelativeMetrics:
    """相邻转折点之间的相对值（与点序列按位置对齐）。

    首个点无前一点 → 四个字段均为 ``None``；前一点值为 0 时比值无定义，也为
    ``None``（不产生 ``inf``）。全部由点序列确定性推导，无墙钟时间。

    :param dt_minutes: 当前点与前一点 ``timestamp`` 之差（单位分钟，1 分钟）
    :param price_ratio: 当前点价格 / 前一点价格（比值；减 1 即变化幅度）
    :param volume_ratio: 当前点成交量 / 前一点成交量
    :param oi_ratio: 当前点持仓量 / 前一点持仓量
    """

    dt_minutes: float | None
    price_ratio: float | None
    volume_ratio: float | None
    oi_ratio: float | None


@dataclass(frozen=True)
class WindowMeta:
    """转折点窗口元信息（写入 sidecar；不含墙钟时间）。"""

    initial_direction_requested: str
    initial_direction_resolved: str
    row_count: int
    window_first_timestamp: str
    window_last_timestamp: str
    input_source_data_version: str | None = None


@dataclass(frozen=True)
class LoadedTurningPoints:
    """转折点读取结果。"""

    points: tuple[TurningPoint, ...]
    relative: tuple[RelativeMetrics, ...]
    symbol: str
    period: str
    path: Path
    meta: dict[str, object]


def build_window_meta(
    df: pd.DataFrame,
    *,
    initial_direction_requested: str,
    initial_direction_resolved: str,
    input_source_data_version: str | None = None,
) -> WindowMeta:
    """由输入 OHLCV 窗口构造 :class:`WindowMeta`。"""
    if df is None or df.empty:
        raise DatasetError("OHLCV 为空，无法构造转折点窗口元信息")
    return WindowMeta(
        initial_direction_requested=initial_direction_requested,
        initial_direction_resolved=initial_direction_resolved,
        row_count=int(len(df)),
        window_first_timestamp=str(df["timestamp"].iloc[0]),
        window_last_timestamp=str(df["timestamp"].iloc[-1]),
        input_source_data_version=input_source_data_version,
    )


def resolve_initial_direction(df: pd.DataFrame, mode: str) -> str:
    """确定窗口起始方向（``up`` = 起始跟踪高点，``down`` = 起始跟踪低点）。

    * ``"up"`` / ``"down"``：人工强制；
    * ``"auto"``：``Low[1] < Low[0]`` 视为起始下跌，否则 ``High[1] > High[0]``
      视为起始上涨；两者都不成立时退回「首根实体方向」（``close >= open`` 为上涨）。
      单根窗口同样退回实体方向。
    """
    if mode not in INITIAL_DIRECTION_MODES:
        raise ValueError(
            f"initial_direction 必须是 {INITIAL_DIRECTION_MODES} 之一: {mode!r}"
        )

    if mode in ("up", "down"):
        return mode

    if len(df) >= 2:
        first_low, second_low = float(df["low"].iloc[0]), float(df["low"].iloc[1])
        first_high, second_high = float(df["high"].iloc[0]), float(df["high"].iloc[1])
        if second_low < first_low:
            return "down"
        if second_high > first_high:
            return "up"

    return "up" if float(df["close"].iloc[0]) >= float(df["open"].iloc[0]) else "down"


def _segment_extreme(
    values: Sequence[float], start: int, end: int, *, highest: bool
) -> tuple[float, int]:
    """闭区间 ``[start, end]``（含两端）内的实际极值及首个达到极值的 bar（0 基）。

    平局取最早 bar：只有**严格**更优才更新（与 ``max(range, key=…)`` 首次出现
    语义一致，确定性）。
    """
    best_price = values[start]
    best_index = start
    for index in range(start + 1, end + 1):
        better = values[index] > best_price if highest else values[index] < best_price
        if better:
            best_price = values[index]
            best_index = index
    return best_price, best_index


def find_turning_points(
    df: pd.DataFrame,
    *,
    initial_direction: str = "auto",
) -> list[TurningPoint]:
    """按人工规则提取转折点路径（检测触发序列与早期版本一致；发射字段按 2026-09-26
    甲口径，并自 2026-10-02 起 up/down 点携带当次趋势段极值，见模块 docstring）。

    :param df: 标准 OHLCV（含 ``timestamp/open/high/low/close``，按时间升序，
        且全部为**已收盘** K 线）；增补信息需要 ``volume`` 与 ``close_oi`` 列
        （标准落盘契约固定包含）
    :param initial_direction: 见 :func:`resolve_initial_direction`
    :return: 路径点序列（起点 → 已确认反转点… → 终点收盘价）
    """
    required = (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_oi",
    )
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"OHLCV 缺少列: {missing}")

    n = len(df)
    if n == 0:
        raise ValueError("OHLCV 为空，无法提取转折点")

    timestamps = [ts for ts in df["timestamp"]]
    opens = [float(x) for x in df["open"]]
    highs = [float(x) for x in df["high"]]
    lows = [float(x) for x in df["low"]]
    closes = [float(x) for x in df["close"]]
    volumes = [int(x) for x in df["volume"]]
    ois = [int(x) for x in df["close_oi"]]

    direction = resolve_initial_direction(df, initial_direction)
    points: list[TurningPoint] = [
        TurningPoint("start", timestamps[0], opens[0], 0, volumes[0], ois[0])
    ]
    if n == 1:
        points.append(
            TurningPoint("close", timestamps[0], closes[0], 0, volumes[0], ois[0])
        )
        return points

    # t 从 1 开始：up/down 点在触发根 t 确认，值根 t − 1 恒存在（无边界特判）
    # segment_start：当次趋势段起点（上一个已确认 up/down 点的触发根，初始 0）。
    # 段 = [segment_start, t − 1] 闭区间，触发根 t 归下一段（见模块 docstring
    # 2026-10-02 节）；up 点取段内实际最高、down 点取段内实际最低，平局取最早。
    segment_start = 0
    for t in range(1, n):
        if direction == "up":
            # 严格小于才转向；触发根当根不按新状态重判（单比较，不级联）
            if lows[t] < lows[t - 1]:
                extreme_price, extreme_index = _segment_extreme(
                    highs, segment_start, t - 1, highest=True
                )
                points.append(
                    TurningPoint(
                        "up", timestamps[t], highs[t - 1], t,
                        volumes[t - 1], ois[t - 1],
                        trend_extreme_price=extreme_price,
                        trend_extreme_bar_index=extreme_index,
                    )
                )
                direction = "down"
                segment_start = t
        else:
            if highs[t] > highs[t - 1]:
                extreme_price, extreme_index = _segment_extreme(
                    lows, segment_start, t - 1, highest=False
                )
                points.append(
                    TurningPoint(
                        "down", timestamps[t], lows[t - 1], t,
                        volumes[t - 1], ois[t - 1],
                        trend_extreme_price=extreme_price,
                        trend_extreme_bar_index=extreme_index,
                    )
                )
                direction = "up"
                segment_start = t

    # 路径收尾：进行中的段不输出（尚未确认），只补最后一根收盘价
    points.append(
        TurningPoint(
            "close", timestamps[n - 1], closes[n - 1], n - 1,
            volumes[n - 1], ois[n - 1],
        )
    )
    return points


def relative_metrics(points: Sequence[TurningPoint]) -> tuple[RelativeMetrics, ...]:
    """由点序列确定性派生相邻点相对值（与 ``points`` 按位置对齐）。

    规则见模块 docstring「点信息增补」：首点四值全 ``None``；前一点值为 0 时
    比值为 ``None``（不产生 ``inf``）。
    """
    if not points:
        return ()

    def _ratio(current: float, previous: float) -> float | None:
        return None if previous == 0 else current / previous

    metrics = [RelativeMetrics(None, None, None, None)]
    for previous, current in zip(points, points[1:]):
        metrics.append(
            RelativeMetrics(
                dt_minutes=(current.timestamp - previous.timestamp).total_seconds() / 60.0,
                price_ratio=_ratio(current.price, previous.price),
                volume_ratio=_ratio(current.volume, previous.volume),
                oi_ratio=_ratio(current.oi, previous.oi),
            )
        )
    return tuple(metrics)


def recent_trend_extremes(
    points: Sequence[TurningPoint], *, n: int = 1
) -> tuple[tuple[TrendExtreme, ...], tuple[TrendExtreme, ...]]:
    """取最近 ``n`` 个 up 点的段内最高与 ``n`` 个 down 点的段内最低（2026-10-02）。

    返回 ``(highs, lows)`` 二值 tuple，各为 :class:`TrendExtreme` 元组，
    **按时间倒序**（最近优先）。高点 = up 点的段内实际最高、低点 = down 点的
    段内实际最低（模块 docstring 2026-10-02 节语义）。

    段长由点序列**确定性推导**（不入 CSV）：up/down 点（触发根 ``t`` = 其
    ``bar_index``）的段起点 = 序列中上一个 up/down 点的 ``bar_index``
    （无前序 → 0），``segment_length = t − 段起点``。内存序列与
    :func:`load_turning_points` 回读序列同构（``bar_index`` 经 CSV 精确回读）。

    :param points: 转折点序列（须经 :func:`find_turning_points` 或
        :func:`load_turning_points` 构造，up/down 点已携带极值字段）
    :param n: 取最近的高点/低点数量（≥ 1）
    :return: ``(highs, lows)``，各为 ``n`` 个 :class:`TrendExtreme`（时间倒序）
    :raises DatasetError: ``n`` 非整数（含 bool）或 ``< 1``；up/down 点数量任一
        ``< n``；序列中 up/down 点缺极值字段
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise DatasetError(f"n 必须为整数（不含 bool）: {n!r}")
    if n < 1:
        raise DatasetError(f"n 必须 ≥ 1: {n}")

    ups = [p for p in points if p.kind == "up"]
    downs = [p for p in points if p.kind == "down"]
    if len(ups) < n:
        raise DatasetError(
            f"up 点数量不足: n={n}, 序列中实际 up 点 {len(ups)} 个"
        )
    if len(downs) < n:
        raise DatasetError(
            f"down 点数量不足: n={n}, 序列中实际 down 点 {len(downs)} 个"
        )

    highs: list[TrendExtreme] = []
    lows: list[TrendExtreme] = []
    previous_trigger: int | None = None
    for point in points:
        if point.kind not in ("up", "down"):
            continue
        if point.trend_extreme_price is None or point.trend_extreme_bar_index is None:
            raise DatasetError(
                "up/down 点缺少当次趋势极值字段"
                f"（须经 find_turning_points/load_turning_points 构造）: "
                f"kind={point.kind}, bar_index={point.bar_index}"
            )
        segment_start = 0 if previous_trigger is None else previous_trigger
        extreme = TrendExtreme(
            kind=point.kind,
            trend_extreme_price=point.trend_extreme_price,
            trend_extreme_bar_index=point.trend_extreme_bar_index,
            segment_length=point.bar_index - segment_start,
            confirmation_bar_index=point.bar_index,
            confirmation_timestamp=point.timestamp,
        )
        (highs if point.kind == "up" else lows).append(extreme)
        previous_trigger = point.bar_index

    # 时间倒序（最近优先）：取各自最近 n 个后反转
    return tuple(reversed(highs[-n:])), tuple(reversed(lows[-n:]))


def trend_state_direction(
    highs: tuple[TrendExtreme, ...], lows: tuple[TrendExtreme, ...]
) -> str:
    """对称三分支趋势状态分类（2026-10-03 v9；纯比较，无选择/过滤逻辑）。

    输入 :func:`recent_trend_extremes`（``n=2``）返回的两个时间倒序元组（**最近优先**：
    ``[0]`` = 编号 ``-1`` 最近项、``[1]`` = 编号 ``-2`` 次近项）；判据 = 各方向最近项
    （``-1``）与次近项（``-2``）的段极值价（``trend_extreme_price``）比较（在**绝对
    价格层**比较——同分母比值与绝对价序等价；分母 ≤ 0 时序列化侧整段退化为 ``na``，
    与本分类无关）：

    * ``"up"``（涨势中）：``-1``（最近）up 段极值 > ``-2``（次近）且 ``-1`` down
      段极值 > ``-2``（高抬高价升）；
    * ``"down"``（跌势中）：两方向 ``-1`` 段极值均 ``<`` ``-2``（高降低降）；
    * ``"range"``（震荡）：其余（含任一方向 ``-1 == -2`` 相等——判据不含等号）。

    :param highs: 最近 2 个 up 点段极值（时间倒序，``recent_trend_extremes`` 的
        首个返回值）
    :param lows: 最近 2 个 down 点段极值（时间倒序，``recent_trend_extremes`` 的
        第二个返回值）
    :return: ``"up"`` | ``"down"`` | ``"range"``
    :raises DatasetError: 任一元组长度 ``< 2``
    """
    if len(highs) < 2 or len(lows) < 2:
        raise DatasetError(
            "trend_state_direction 需要各方向 ≥2 个段极值"
            f"（recent_trend_extremes n=2 的返回值）: highs={len(highs)}, lows={len(lows)}"
        )
    # 时间倒序（最近优先）：[0] = 编号 -1（最近）、[1] = 编号 -2（次近）
    high_higher = highs[0].trend_extreme_price > highs[1].trend_extreme_price
    high_lower = highs[0].trend_extreme_price < highs[1].trend_extreme_price
    low_higher = lows[0].trend_extreme_price > lows[1].trend_extreme_price
    low_lower = lows[0].trend_extreme_price < lows[1].trend_extreme_price
    if high_higher and low_higher:
        return "up"
    if high_lower and low_lower:
        return "down"
    return "range"


def turning_points_filename(symbol: str, period: str) -> str:
    """转折点文件名：``{symbol}_{period}.csv``。"""
    return f"{symbol}_{period}.{OUTPUT_FORMAT}"


def _format_timestamp(value: pd.Timestamp) -> str:
    """时间序列化为 ISO8601 带时区偏移（与 OHLCV CSV 同一写法）。"""
    return pd.Timestamp(value).isoformat(sep=" ")


def _float_or_nan(value: float | None) -> float:
    """相对值写入 CSV 的表示：``None`` → ``NaN``（``to_csv`` 输出空单元格）。"""
    return float("nan") if value is None else float(value)


def _int_or_nan(value: int | None) -> float:
    """可空整数列写入 CSV 的表示：``None`` → ``NaN``（``to_csv`` 输出空单元格）。"""
    return float("nan") if value is None else int(value)


def save_turning_points(
    points: list[TurningPoint],
    *,
    symbol: str,
    period: str,
    output_dir: str | Path,
    window_meta: WindowMeta,
) -> Path:
    """把转折点路径落盘为 CSV，并写入窗口 sidecar。

    CSV 列固定 ``TURNING_POINT_COLUMNS``（绝对值 + 相邻点相对值）；相对值由
    :func:`relative_metrics` 从点序列确定性派生，首点四值为空单元格。sidecar 记录
    窗口元信息（不含墙钟时间，保证重复运行逐字节一致）。

    :return: 落盘数据文件路径
    :raises DatasetError: 空路径点 / symbol、period 非法
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise DatasetError("symbol 必须为非空字符串")
    if not isinstance(period, str) or not period.strip():
        raise DatasetError("period 必须为非空字符串")
    if not points:
        raise DatasetError("拒绝落盘空转折点序列")

    output_dir_path = Path(output_dir)
    ensure_directory(output_dir_path)
    path = output_dir_path / turning_points_filename(symbol, period)

    metrics = relative_metrics(points)
    frame = pd.DataFrame(
        [
            {
                "point_index": index,
                "kind": point.kind,
                "timestamp": _format_timestamp(point.timestamp),
                "price": float(point.price),
                "bar_index": int(point.bar_index),
                "volume": int(point.volume),
                "oi": int(point.oi),
                "trend_extreme_price": _float_or_nan(point.trend_extreme_price),
                "trend_extreme_bar_index": _int_or_nan(point.trend_extreme_bar_index),
                "dt_minutes": _float_or_nan(metric.dt_minutes),
                "price_ratio": _float_or_nan(metric.price_ratio),
                "volume_ratio": _float_or_nan(metric.volume_ratio),
                "oi_ratio": _float_or_nan(metric.oi_ratio),
            }
            for index, (point, metric) in enumerate(zip(points, metrics, strict=True))
        ],
        columns=list(TURNING_POINT_COLUMNS),
    )
    # 与 OHLCV 落盘一致：先写临时文件再 os.replace，CSV 与 sidecar 成对出现
    commit_data_with_sidecar(
        path,
        frame.to_csv(index=False),
        build_meta=lambda digest: {
            "symbol": symbol,
            "period": period,
            "row_count": int(window_meta.row_count),
            "window_first_timestamp": window_meta.window_first_timestamp,
            "window_last_timestamp": window_meta.window_last_timestamp,
            "initial_direction_requested": window_meta.initial_direction_requested,
            "initial_direction_resolved": window_meta.initial_direction_resolved,
            "input_source_data_version": window_meta.input_source_data_version,
            "file_sha256": digest,
        },
    )
    return path


def _read_sidecar(path: Path) -> dict[str, object]:
    """读取转折点 sidecar；不存在返回空映射，损坏抛 :class:`DataLoadError`。"""
    sidecar = path.with_suffix(SIDECAR_SUFFIX)
    if not sidecar.is_file():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise DataLoadError(f"转折点 sidecar 损坏: {sidecar} ({exc})") from exc
    if not isinstance(payload, dict):
        raise DataLoadError(f"转折点 sidecar 必须是 JSON 对象: {sidecar}")
    return payload


def load_turning_points(
    symbol: str, period: str, *, data_dir: str | Path
) -> LoadedTurningPoints:
    """读取 ``{symbol}_{period}.csv`` 转折点与 sidecar（可离线复现路径）。

    :raises DataLoadError: 文件不存在 / 缺列 / 类型无法收敛 / sidecar 损坏
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise DataLoadError("symbol 必须为非空字符串")
    if not isinstance(period, str) or not period.strip():
        raise DataLoadError("period 必须为非空字符串")
    path = Path(data_dir) / turning_points_filename(symbol, period)
    if not path.is_file():
        raise DataLoadError(f"转折点文件不存在: {path}")

    raw = read_csv_table(path)  # 任何读取失败 → DataLoadError（含路径与原因）

    missing = [col for col in TURNING_POINT_COLUMNS if col not in raw.columns]
    if missing:
        raise DataLoadError(f"转折点文件缺少必需列: {missing}")
    if raw.empty:
        raise DataLoadError(f"转折点文件为空: {path}")

    try:
        stamps = pd.to_datetime(raw["timestamp"], utc=True, format="ISO8601")
    except (TypeError, ValueError) as exc:
        raise DataLoadError(f"timestamp 无法解析为时间: {exc}") from exc
    stamps = stamps.dt.tz_convert(TIMEZONE)

    invalid_kinds = [
        kind for kind in raw["kind"].tolist() if kind not in TURNING_POINT_KINDS
    ]
    if invalid_kinds:
        raise DataLoadError(f"kind 取值非法: {invalid_kinds}")

    # 绝对值列：volume/oi 严格整数（与 OHLCV volume 同口径）
    counts = pd.to_numeric(raw["volume"], errors="coerce")
    ois = pd.to_numeric(raw["oi"], errors="coerce")
    if bool(counts.isna().any() | ois.isna().any()):
        raise DataLoadError("volume/oi 存在缺失或非数值")
    if bool((counts % 1 != 0).any() | (ois % 1 != 0).any()):
        raise DataLoadError("volume/oi 必须为整数，存在小数值")

    # 相对值列：空单元格（NaN）合法（首点/零分母），其余收敛为 float
    def _optional_floats(name: str) -> list[float | None]:
        values = pd.to_numeric(raw[name], errors="coerce")
        return [None if pd.isna(v) else float(v) for v in values]

    optional = {name: _optional_floats(name) for name in RELATIVE_COLUMNS}

    # 当次趋势极值列（2026-10-02）：空单元格合法（start/close 点不产生极值）；
    # 有值时 price 收敛 float、bar_index 收敛 int（非数值/小数 → DataLoadError，
    # 与 volume/oi 同等严格度）。空单元格（NaN）与非数值垃圾区分开，不静默。
    def _optional_extreme_floats(name: str) -> list[float | None]:
        raw_column = raw[name]
        values = pd.to_numeric(raw_column, errors="coerce")
        junk = values.isna() & ~raw_column.isna()
        if bool(junk.any()):
            raise DataLoadError(f"{name} 存在缺失或非数值")
        return [None if pd.isna(v) else float(v) for v in values]

    def _optional_extreme_ints(name: str) -> list[int | None]:
        raw_column = raw[name]
        values = pd.to_numeric(raw_column, errors="coerce")
        junk = values.isna() & ~raw_column.isna()
        if bool(junk.any()):
            raise DataLoadError(f"{name} 存在缺失或非数值")
        if bool((values.dropna() % 1 != 0).any()):
            raise DataLoadError(f"{name} 必须为整数，存在小数值")
        return [None if pd.isna(v) else int(v) for v in values]

    extreme_prices = _optional_extreme_floats("trend_extreme_price")
    extreme_bars = _optional_extreme_ints("trend_extreme_bar_index")

    points = tuple(
        TurningPoint(
            kind=str(row.kind),
            timestamp=stamp,
            price=float(row.price),
            bar_index=int(row.bar_index),
            volume=int(count),
            oi=int(oi_value),
            trend_extreme_price=extreme_prices[index],
            trend_extreme_bar_index=extreme_bars[index],
        )
        for index, (row, stamp, count, oi_value) in enumerate(
            zip(
                raw.itertuples(index=False), stamps, counts, ois, strict=True
            )
        )
    )
    relative = tuple(
        RelativeMetrics(
            dt_minutes=optional["dt_minutes"][i],
            price_ratio=optional["price_ratio"][i],
            volume_ratio=optional["volume_ratio"][i],
            oi_ratio=optional["oi_ratio"][i],
        )
        for i in range(len(points))
    )
    return LoadedTurningPoints(
        points=points,
        relative=relative,
        symbol=symbol,
        period=period,
        path=path,
        meta=_read_sidecar(path),
    )

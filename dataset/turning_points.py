"""转折点提取与落盘。

规则**逐条照搬** ``reference/perception/scripts/turning_points.py``（D-06），不做任何
增删或阈值调整：

```text
上涨状态（跟踪当前上涨段最高价 High）:
    只要当前 K 线 Low >= 前一根 K 线 Low，就继续保持上涨；
    一旦当前 K 线 Low < 前一根 K 线 Low:
        记录此前上涨段的最高价及其时间，状态切换为下跌。
下跌状态（跟踪当前下跌段最低价 Low）:
    只要当前 K 线 High <= 前一根 K 线 High，就继续保持下跌；
    一旦当前 K 线 High > 前一根 K 线 High:
        记录此前下跌段的最低价及其时间，状态切换为上涨。
```

细则（与参考实现一致）：比较运算符**严格**（``==`` 不转向）；极值时间取该极值所在
K 线起始 ``timestamp``；**转向那根 K 线也参与本段极值搜索**；极值并列保留**最早**
一根；末尾补最后一根 K 线收盘价（进行中的极值尚未确认，不输出）。

移植自 ``reference/perception/scripts/turning_points.py``（新增落盘/读取
``save_turning_points`` / ``load_turning_points``，算法部分保持等价）。

点信息增补（2026-09-24 需求）
----------------------------

每个转折点除时间与价格外，还携带其 ``bar_index`` 所指那根 K 线（锚点；``up``/``down``
点为极值前一根，见下文 2026-09-25 变更）的：

* ``volume``：该 K 线时间范围内的成交量合计；
* ``oi``：该 K 线**结束时刻**的持仓量（天勤 ``close_oi`` 口径；与主流行情软件的
  「K 线持仓量」一致。``open_oi`` 口径可由 ``bar_index`` 关联 K 线 CSV 离线补算）。

并派生相邻点之间的**相对值**（由点序列确定性推导，无墙钟时间）：

* ``dt_minutes``：当前点与前一点 ``timestamp`` 之差，单位分钟（1 分钟）；
* ``price_ratio``：当前点价格 / 前一点价格（比值；减 1 即变化幅度）；
* ``volume_ratio``：当前点成交量 / 前一点成交量；
* ``oi_ratio``：当前点持仓量 / 前一点持仓量。

首个点无前一点 → 四个相对值均为空；前一点值为 0 时比值无定义（真实存在无成交
分钟），同样留空，不产生 ``inf``。检测算法（哪些 K 线是转折点）与参考实现逐条等价，
不受增补影响。

发射口径变更（2026-09-25 用户决策 D1–D3）
-----------------------------------------

* kind 更名（D3）：极值类转折点 ``high`` → ``up``（上涨段极值高确认）、
  ``low`` → ``down``（下跌段极值低确认）；``start`` / ``close`` 不变。
* 锚点前移（D1「逻辑统一」，整点各字段同源）：``up`` / ``down`` 点的 ``timestamp`` /
  ``price`` / ``volume`` / ``oi`` 全部取**极值 K 线的前一根**（锚点
  ``bar_index = 极值 bar − 1``），``price`` = 锚点 bar 的最高价（up）或最低价（down）。
* bar 0 边界（D2）：极值落在 bar 0 时无前一根，整点留在 bar 0，``price`` 取 bar 0
  自身极值价（跟踪初值即 bar 0 极值价，按上式自动一致，无需特判）。
* 检测算法与参考实现仍逐条等价；仅发射的 kind 命名与字段锚定偏离参考实现。旧 kind
  （``high``/``low``）落盘文件不再能被 :func:`load_turning_points` 读取（D5：无兼容
  层，重新生成即可）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    :param timestamp: K 线开始时间（该点 ``bar_index`` 所指那根；``up``/``down`` 点
        为极值 K 线的前一根，极值在 bar 0 时为 bar 0 自身）
    :param price: 该点的价格（up = 锚点 bar 最高价、down = 锚点 bar 最低价、
        起点开盘价、终点收盘价）
    :param bar_index: 在传入窗口中的 0 基下标（``up``/``down`` = 极值 bar − 1，
        极值在 bar 0 时为 0）
    :param volume: 该点 ``bar_index`` 所指 K 线的成交量（K 线时间范围内的成交量合计）
    :param oi: 该点 ``bar_index`` 所指 K 线结束时刻的持仓量（天勤 ``close_oi`` 口径）
    """

    kind: str
    timestamp: pd.Timestamp
    price: float
    bar_index: int
    volume: int
    oi: int


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


def find_turning_points(
    df: pd.DataFrame,
    *,
    initial_direction: str = "auto",
) -> list[TurningPoint]:
    """按人工规则提取转折点路径（检测规则与参考实现逐条一致；发射口径见模块
    docstring「发射口径变更（2026-09-25 用户决策 D1–D3）」）。

    :param df: 标准 OHLCV（含 ``timestamp/open/high/low/close``，按时间升序，
        且全部为**已收盘** K 线）；增补信息需要 ``volume`` 与 ``close_oi`` 列
        （标准落盘契约固定包含）
    :param initial_direction: 见 :func:`resolve_initial_direction`
    :return: 路径点序列（起点 → 已确认极值… → 终点收盘价）
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

    # 当前段的极值（上涨跟踪 high，下跌跟踪 low）
    if direction == "up":
        extreme = highs[0]
    else:
        extreme = lows[0]
    extreme_ts = timestamps[0]
    extreme_idx = 0

    for t in range(1, n):
        if direction == "up":
            # 严格大于才更新 → 并列保留最早一根
            if highs[t] > extreme:
                extreme, extreme_ts, extreme_idx = highs[t], timestamps[t], t
            # 转向那根 K 线已参与上面的极值搜索（人工确认细则 3）
            if lows[t] < lows[t - 1]:
                # 发射锚点 = 极值前一根（D1）；极值在 bar 0 时整点留在 bar 0（D2）
                anchor = extreme_idx - 1 if extreme_idx > 0 else 0
                points.append(
                    TurningPoint(
                        "up", timestamps[anchor], highs[anchor], anchor,
                        volumes[anchor], ois[anchor],
                    )
                )
                direction = "down"
                extreme, extreme_ts, extreme_idx = lows[t], timestamps[t], t
        else:
            if lows[t] < extreme:
                extreme, extreme_ts, extreme_idx = lows[t], timestamps[t], t
            if highs[t] > highs[t - 1]:
                # 发射锚点 = 极值前一根（D1）；极值在 bar 0 时整点留在 bar 0（D2）
                anchor = extreme_idx - 1 if extreme_idx > 0 else 0
                points.append(
                    TurningPoint(
                        "down", timestamps[anchor], lows[anchor], anchor,
                        volumes[anchor], ois[anchor],
                    )
                )
                direction = "up"
                extreme, extreme_ts, extreme_idx = highs[t], timestamps[t], t

    # 路径收尾：进行中的极值不输出（尚未确认），只补最后一根收盘价
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


def turning_points_filename(symbol: str, period: str) -> str:
    """转折点文件名：``{symbol}_{period}.csv``。"""
    return f"{symbol}_{period}.{OUTPUT_FORMAT}"


def _format_timestamp(value: pd.Timestamp) -> str:
    """时间序列化为 ISO8601 带时区偏移（与 OHLCV CSV 同一写法）。"""
    return pd.Timestamp(value).isoformat(sep=" ")


def _float_or_nan(value: float | None) -> float:
    """相对值写入 CSV 的表示：``None`` → ``NaN``（``to_csv`` 输出空单元格）。"""
    return float("nan") if value is None else float(value)


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

    points = tuple(
        TurningPoint(
            kind=str(row.kind),
            timestamp=stamp,
            price=float(row.price),
            bar_index=int(row.bar_index),
            volume=int(count),
            oi=int(oi_value),
        )
        for row, stamp, count, oi_value in zip(
            raw.itertuples(index=False), stamps, counts, ois, strict=True
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

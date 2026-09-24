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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

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
TURNING_POINT_KINDS: Final[tuple[str, ...]] = ("start", "high", "low", "close")
#: 落盘列顺序（数据契约）
TURNING_POINT_COLUMNS: Final[tuple[str, ...]] = (
    "point_index",
    "kind",
    "timestamp",
    "price",
    "bar_index",
)


@dataclass(frozen=True)
class TurningPoint:
    """路径上的一个点。

    :param kind: ``start`` | ``high`` | ``low`` | ``close``
    :param timestamp: K 线开始时间（极值所在 / 起点 / 终点所在那根）
    :param price: 该点的价格（极值价、起点开盘价、终点收盘价）
    :param bar_index: 在传入窗口中的 0 基下标
    """

    kind: str
    timestamp: pd.Timestamp
    price: float
    bar_index: int


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
    """按人工规则提取转折点路径（规则与参考实现逐条一致）。

    :param df: 标准 OHLCV（含 ``timestamp/open/high/low/close``，按时间升序，
        且全部为**已收盘** K 线）
    :param initial_direction: 见 :func:`resolve_initial_direction`
    :return: 路径点序列（起点 → 已确认极值… → 终点收盘价）
    """
    required = ("timestamp", "open", "high", "low", "close")
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

    direction = resolve_initial_direction(df, initial_direction)
    points: list[TurningPoint] = [TurningPoint("start", timestamps[0], opens[0], 0)]
    if n == 1:
        points.append(TurningPoint("close", timestamps[0], closes[0], 0))
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
                points.append(TurningPoint("high", extreme_ts, extreme, extreme_idx))
                direction = "down"
                extreme, extreme_ts, extreme_idx = lows[t], timestamps[t], t
        else:
            if lows[t] < extreme:
                extreme, extreme_ts, extreme_idx = lows[t], timestamps[t], t
            if highs[t] > highs[t - 1]:
                points.append(TurningPoint("low", extreme_ts, extreme, extreme_idx))
                direction = "up"
                extreme, extreme_ts, extreme_idx = highs[t], timestamps[t], t

    # 路径收尾：进行中的极值不输出（尚未确认），只补最后一根收盘价
    points.append(TurningPoint("close", timestamps[n - 1], closes[n - 1], n - 1))
    return points


def turning_points_filename(symbol: str, period: str) -> str:
    """转折点文件名：``{symbol}_{period}.csv``。"""
    return f"{symbol}_{period}.{OUTPUT_FORMAT}"


def _format_timestamp(value: pd.Timestamp) -> str:
    """时间序列化为 ISO8601 带时区偏移（与 OHLCV CSV 同一写法）。"""
    return pd.Timestamp(value).isoformat(sep=" ")


def save_turning_points(
    points: list[TurningPoint],
    *,
    symbol: str,
    period: str,
    output_dir: str | Path,
    window_meta: WindowMeta,
) -> Path:
    """把转折点路径落盘为 CSV，并写入窗口 sidecar。

    CSV 列固定 ``point_index,kind,timestamp,price,bar_index``；sidecar 记录窗口元信息
    （不含墙钟时间，保证重复运行逐字节一致）。

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

    frame = pd.DataFrame(
        [
            {
                "point_index": index,
                "kind": point.kind,
                "timestamp": _format_timestamp(point.timestamp),
                "price": float(point.price),
                "bar_index": int(point.bar_index),
            }
            for index, point in enumerate(points)
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

    points = tuple(
        TurningPoint(
            kind=str(row.kind),
            timestamp=stamp,
            price=float(row.price),
            bar_index=int(row.bar_index),
        )
        for row, stamp in zip(raw.itertuples(index=False), stamps, strict=True)
    )
    return LoadedTurningPoints(
        points=points,
        symbol=symbol,
        period=period,
        path=path,
        meta=_read_sidecar(path),
    )

"""标准 OHLCV 列约定、时间归一化与天勤 K 线标准化。

职责边界（沿用参考实现）：

* 标准化 = 字段映射 + 时间归一化（Asia/Shanghai）+ 数值类型收敛 + 稳定升序；
* 持仓量（``open_oi``/``close_oi``）是标准契约的**固定列**：天勤每根 K 线自带
  「K 线起始/结束时刻的持仓量」，两条取数路径（``get_kline_data_series`` /
  ``get_kline_serial``）均返回，无需开关；
* 合法性校验（唯一、非空、OHLC 关系）属于 :mod:`dataset.validator`，本模块只对
  「无法收敛到标准类型」的脏数据报错；
* 防未来数据泄漏：``clean_serial_klines`` 剔除填充行与**未收盘** K 线。

移植自 ``reference/perception/src/marksense/data/ohlcv.py`` 与
``reference/perception/src/marksense/data/timeutils.py``（两个模块合并到本文件）。
"""

from __future__ import annotations

import datetime
from typing import Final

import pandas as pd

from dataset.errors import DatasetError

#: 交易所时区（时间归一化目标；保证可排序、无歧义）
TIMEZONE: Final[str] = "Asia/Shanghai"

#: 持仓量字段（天勤每根 K 线自带：K 线起始时刻 / 结束时刻的持仓量）
OI_COLUMNS: Final[tuple[str, ...]] = ("open_oi", "close_oi")

#: 标准 OHLCV 字段（顺序即落盘列顺序，数据契约固定不变；持仓量为固定列）
OHLCV_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
) + OI_COLUMNS

#: 天勤 ``get_kline_data_series`` / ``get_kline_serial`` 必需字段
#: （两条路径均固定返回 ``open_oi``/``close_oi``）
REQUIRED_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_oi",
    "close_oi",
)

#: 天勤 ``get_kline_serial`` 必需列（``id`` 用于剔除填充行）
REQUIRED_SERIAL_COLUMNS: Final[tuple[str, ...]] = ("id", "datetime")

_PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")


def epoch_ns_to_timestamp(values: pd.Series) -> pd.Series:
    """纳秒时间戳整数序列 → tz-aware（Asia/Shanghai）时间序列。

    调用方需先确认 ``values`` 为整数 dtype（见 :func:`standardize_klines`）。
    """
    try:
        ns = values.astype("int64")
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"datetime 无法按纳秒时间戳整数解析: {exc}") from exc
    return pd.to_datetime(ns, unit="ns", utc=True).dt.tz_convert(TIMEZONE)


def parse_date_bound(value: str, name: str) -> datetime.datetime:
    """解析 ISO 日期/时间边界字符串（``get_kline_data_series`` 的 start_dt/end_dt）。"""
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{name} 必须为非空的 ISO 日期/时间字符串")
    try:
        return datetime.datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise DatasetError(f"{name} 不是合法的 ISO 日期/时间: {value!r}") from exc


def _coerce_float(series: pd.Series, name: str) -> pd.Series:
    """价格列收敛为 float64；无法转换的脏数据报错。"""
    try:
        return series.astype("float64")
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"{name} 无法转换为浮点数: {exc}") from exc


def _coerce_integer(series: pd.Series, name: str) -> pd.Series:
    """整数列收敛为 int64；缺失、非数值、小数值一律报错。"""
    numeric = pd.to_numeric(series, errors="coerce")
    if bool(numeric.isna().any()):
        raise DatasetError(f"{name} 存在缺失或非数值")
    if bool((numeric % 1 != 0).any()):
        raise DatasetError(f"{name} 必须为整数，存在小数值")
    return numeric.astype("int64")


def standardize_klines(raw: pd.DataFrame) -> pd.DataFrame:
    """把天勤原始 K 线 DataFrame 标准化为标准 OHLCV（含持仓量固定列）。

    * 字段映射：``datetime → timestamp``；价格与成交量原名保留；
    * ``timestamp``：纳秒整数 → Asia/Shanghai tz-aware，按 timestamp 稳定升序；
    * ``open_oi``/``close_oi``：持仓量固定保留，收敛为 ``int64``；
    * 唯一性与 OHLC 合法性由 :func:`dataset.validator.validate_ohlcv` 校验。

    :raises DatasetError: 结果为空 / 缺字段 / datetime 非整数 / 类型无法收敛
    """
    if raw is None or raw.empty:
        raise DatasetError("天勤返回的 K 线为空")

    missing = [col for col in REQUIRED_SOURCE_COLUMNS if col not in raw.columns]
    if missing:
        raise DatasetError(f"天勤 K 线缺少必需字段: {missing}")

    dt = raw["datetime"]
    if not pd.api.types.is_integer_dtype(dt):
        raise DatasetError(f"datetime 必须为纳秒时间戳整数，实际 dtype={dt.dtype}")

    data: dict[str, pd.Series] = {"timestamp": epoch_ns_to_timestamp(dt)}
    for col in _PRICE_COLUMNS:
        data[col] = _coerce_float(raw[col], col)
    data["volume"] = _coerce_integer(raw["volume"], "volume")
    for col in OI_COLUMNS:
        data[col] = _coerce_integer(raw[col], col)

    standardized = pd.DataFrame(data)
    return standardized.sort_values("timestamp", kind="stable").reset_index(drop=True)


def clean_serial_klines(
    raw: pd.DataFrame,
    *,
    duration_seconds: int,
    as_of: pd.Timestamp | datetime.datetime | None = None,
) -> pd.DataFrame:
    """清洗 ``get_kline_serial`` 原始序列 → 标准 OHLCV（剔除未收盘 K 线）。

    * 固定 ``data_length`` 行；数据不足时前端以「负 id + datetime=0」行填充；
    * 全部列为 float64，整秒纳秒时间戳是 512 的倍数，可无损 ``round`` 还原 int64；
    * 形成中的 K 线拥有正 id，无法凭 id 符号区分是否收盘 → 以
      「起点 + 周期 <= as_of」判定已收盘（防泄漏红线）。

    :param duration_seconds: K 线周期（秒），用于计算收盘时刻
    :param as_of: 判定已收盘的基准时间（``Timestamp``/``datetime``/ISO 字符串）；
        缺省取当前北京时间（naive 视为北京时间）
    """
    if raw is None or raw.empty:
        raise DatasetError("天勤返回的 K 线序列为空")
    for col in REQUIRED_SERIAL_COLUMNS:
        if col not in raw.columns:
            raise DatasetError(f"K 线序列缺少必需列: {col}")

    if as_of is None:
        as_of = pd.Timestamp.now(tz=TIMEZONE)
    else:
        as_of = pd.Timestamp(as_of)  # 接受 datetime / 字符串 / Timestamp
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize(TIMEZONE)
    # 天勤 datetime 与 tz-aware Timestamp.value 同为真实 epoch 纳秒，直接同帧比较
    as_of_ns = int(as_of.value)

    ids = pd.to_numeric(raw["id"], errors="coerce")
    dts = pd.to_numeric(raw["datetime"], errors="coerce")
    keep = (ids >= 0) & dts.notna() & (dts > 0)  # 剔除填充行
    if not bool(keep.any()):
        raise DatasetError("K 线序列中无有效数据行")

    ns = dts[keep].round().astype("int64")  # float64 → int64 无损还原
    duration_ns = int(duration_seconds) * 1_000_000_000
    completed = ns + duration_ns <= as_of_ns  # 剔除未收盘 K 线（防泄漏）
    if not bool(completed.any()):
        raise DatasetError("K 线序列中无已收盘数据")
    ns = ns[completed]

    frame = raw.loc[ns.index].copy()
    frame["datetime"] = ns
    return standardize_klines(frame)

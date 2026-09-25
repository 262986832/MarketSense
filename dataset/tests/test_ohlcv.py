"""K 线标准化与未收盘剔除（AC-2、AC-5）。"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from dataset.errors import DatasetError
from dataset.ohlcv import (
    OHLCV_COLUMNS,
    TIMEZONE,
    clean_serial_klines,
    parse_date_bound,
    standardize_klines,
)
from dataset.tests.conftest import (
    ONE_MINUTE_SECONDS,
    RAW_ROWS_ASC,
    build_raw_klines,
    build_serial_klines,
)


def test_standardize_klines_columns_dtypes_and_order() -> None:
    df = standardize_klines(build_raw_klines())

    assert tuple(df.columns) == OHLCV_COLUMNS
    assert str(df["timestamp"].dt.tz) == TIMEZONE
    assert df["timestamp"].is_monotonic_increasing
    for col in ("open", "high", "low", "close"):
        assert str(df[col].dtype) == "float64"
    assert str(df["volume"].dtype) == "int64"
    assert str(df["timestamp"].iloc[0]) == RAW_ROWS_ASC[0][0] + "+08:00"


def test_standardize_klines_sorts_ascending() -> None:
    rows = [RAW_ROWS_ASC[2], RAW_ROWS_ASC[0], RAW_ROWS_ASC[1]]
    df = standardize_klines(build_raw_klines(rows))
    assert df["timestamp"].is_monotonic_increasing
    assert str(df["timestamp"].iloc[0]) == RAW_ROWS_ASC[0][0] + "+08:00"


def test_standardize_klines_always_keeps_open_interest_columns() -> None:
    """持仓量是标准契约固定列：无条件保留且收敛为 int64。"""
    df = standardize_klines(build_raw_klines())
    assert OHLCV_COLUMNS[-2:] == ("open_oi", "close_oi")
    assert tuple(df.columns) == OHLCV_COLUMNS
    assert str(df["open_oi"].dtype) == "int64"
    assert str(df["close_oi"].dtype) == "int64"
    assert df["open_oi"].tolist() == [r[6] for r in RAW_ROWS_ASC]
    assert df["close_oi"].tolist() == [r[7] for r in RAW_ROWS_ASC]


def test_standardize_klines_rejects_empty_and_missing_columns() -> None:
    with pytest.raises(DatasetError, match="为空"):
        standardize_klines(pd.DataFrame())
    with pytest.raises(DatasetError, match="缺少必需字段"):
        standardize_klines(build_raw_klines().drop(columns=["volume"]))
    # 持仓量是必需字段：缺失（含任一列）或含 NaN 均报错
    with pytest.raises(DatasetError, match="缺少必需字段"):
        standardize_klines(build_raw_klines().drop(columns=["close_oi"]))
    with_nan = build_raw_klines()
    with_nan.loc[0, "open_oi"] = float("nan")
    with pytest.raises(DatasetError, match="open_oi 存在缺失或非数值"):
        standardize_klines(with_nan)


def test_standardize_klines_rejects_non_integer_datetime() -> None:
    raw = build_raw_klines()
    raw["datetime"] = raw["datetime"].astype("float64")
    with pytest.raises(DatasetError, match="纳秒时间戳整数"):
        standardize_klines(raw)


def test_standardize_klines_rejects_fractional_volume() -> None:
    raw = build_raw_klines()
    raw["volume"] = raw["volume"].astype("float64")
    raw.loc[0, "volume"] = 100.5
    with pytest.raises(DatasetError, match="必须为整数"):
        standardize_klines(raw)


def test_clean_serial_klines_drops_padding_rows() -> None:
    raw = build_serial_klines(width=10)
    assert len(raw) == 10

    df = clean_serial_klines(
        raw,
        duration_seconds=ONE_MINUTE_SECONDS,
        as_of=pd.Timestamp("2024-01-02 09:10:00", tz=TIMEZONE),
    )

    assert len(df) == len(RAW_ROWS_ASC)
    assert str(df["timestamp"].iloc[0]) == RAW_ROWS_ASC[0][0] + "+08:00"
    assert df["volume"].tolist() == [r[5] for r in RAW_ROWS_ASC]


def test_clean_serial_klines_drops_last_unclosed_bar() -> None:
    rows = RAW_ROWS_ASC + [("2024-01-02 09:03:00", 68100.0, 68200.0, 68090.0, 68150.0, 90, 4900, 4850)]
    raw = build_serial_klines(rows, width=10)
    # as_of == 最后一根开始时间 → 该根尚未收盘（start + 60s > as_of）
    as_of = pd.Timestamp("2024-01-02 09:03:00", tz=TIMEZONE)

    df = clean_serial_klines(raw, duration_seconds=ONE_MINUTE_SECONDS, as_of=as_of)

    assert len(df) == len(RAW_ROWS_ASC)
    assert str(df["timestamp"].iloc[-1]) == RAW_ROWS_ASC[-1][0] + "+08:00"


def test_clean_serial_klines_keeps_bar_closed_exactly_at_as_of() -> None:
    raw = build_serial_klines(width=10)
    as_of = pd.Timestamp("2024-01-02 09:03:00", tz=TIMEZONE)  # 09:02 + 60s == as_of

    df = clean_serial_klines(raw, duration_seconds=ONE_MINUTE_SECONDS, as_of=as_of)

    assert len(df) == len(RAW_ROWS_ASC)


def test_clean_serial_klines_naive_as_of_treated_as_beijing() -> None:
    raw = build_serial_klines(width=10)
    df = clean_serial_klines(
        raw,
        duration_seconds=ONE_MINUTE_SECONDS,
        as_of=datetime.datetime(2024, 1, 2, 9, 10, 0),
    )
    assert len(df) == len(RAW_ROWS_ASC)


def test_clean_serial_klines_errors_when_nothing_closed() -> None:
    raw = build_serial_klines(width=10)
    # as_of == 首根开始时间 → 所有 K 线的 start + 60s 都晚于 as_of
    as_of = pd.Timestamp(RAW_ROWS_ASC[0][0], tz=TIMEZONE)
    with pytest.raises(DatasetError, match="无已收盘数据"):
        clean_serial_klines(raw, duration_seconds=ONE_MINUTE_SECONDS, as_of=as_of)


def test_clean_serial_klines_errors_on_missing_columns() -> None:
    raw = build_serial_klines(width=10).drop(columns=["id"])
    with pytest.raises(DatasetError, match="缺少必需列"):
        clean_serial_klines(raw, duration_seconds=ONE_MINUTE_SECONDS)


def test_parse_date_bound_accepts_iso_and_rejects_bad_values() -> None:
    assert parse_date_bound("2024-01-02", "start") == datetime.datetime(2024, 1, 2)
    with pytest.raises(DatasetError, match="非空"):
        parse_date_bound("", "start")
    with pytest.raises(DatasetError, match="不是合法的 ISO"):
        parse_date_bound("2024/01/02", "end")

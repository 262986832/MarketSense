"""落盘前校验规则（AC-2、AC-5）。"""

from __future__ import annotations

import pandas as pd
import pytest

from dataset.errors import ValidationError
from dataset.tests.conftest import build_ohlcv
from dataset.validator import validate_ohlcv

_VALID_ROWS = [
    (100.0, 101.0, 99.0, 100.5),
    (100.5, 102.0, 100.0, 101.0),
    (101.0, 103.0, 100.5, 102.0),
]


def _codes(report) -> set[str]:
    return {v.code for v in report.violations}


def test_valid_frame_passes() -> None:
    report = validate_ohlcv(build_ohlcv(_VALID_ROWS))
    assert report.ok
    assert report.violations == ()
    report.raise_if_invalid()  # 不抛异常


def test_zero_volume_is_valid() -> None:
    df = build_ohlcv(_VALID_ROWS)
    df.loc[1, "volume"] = 0
    assert validate_ohlcv(df).ok


def test_duplicate_timestamps_reported() -> None:
    df = build_ohlcv(_VALID_ROWS)
    df.loc[1, "timestamp"] = df["timestamp"].iloc[0]
    report = validate_ohlcv(df)
    assert "TIMESTAMP_DUPLICATE" in _codes(report)
    assert not report.ok


def test_not_ascending_reported() -> None:
    df = build_ohlcv(_VALID_ROWS).iloc[::-1].reset_index(drop=True)
    assert "TIMESTAMP_NOT_ASCENDING" in _codes(validate_ohlcv(df))


def test_naive_timestamp_reported_as_dtype_violation() -> None:
    df = build_ohlcv(_VALID_ROWS)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    assert "TIMESTAMP_DTYPE" in _codes(validate_ohlcv(df))


def test_missing_columns_short_circuits() -> None:
    df = build_ohlcv(_VALID_ROWS).drop(columns=["volume"])
    report = validate_ohlcv(df)
    assert [v.code for v in report.violations] == ["MISSING_COLUMNS"]


def test_empty_frame_reported() -> None:
    df = build_ohlcv(_VALID_ROWS).iloc[0:0]
    assert "EMPTY_DATA" in _codes(validate_ohlcv(df))


def test_missing_price_value_reported() -> None:
    df = build_ohlcv(_VALID_ROWS)
    df.loc[1, "open"] = float("nan")
    assert "OHLC_MISSING_VALUE" in _codes(validate_ohlcv(df))


def test_non_finite_price_reported() -> None:
    df = build_ohlcv(_VALID_ROWS)
    df.loc[1, "high"] = float("inf")
    assert "OHLC_NOT_FINITE" in _codes(validate_ohlcv(df))


def test_negative_volume_reported() -> None:
    df = build_ohlcv(_VALID_ROWS)
    df.loc[1, "volume"] = -5
    assert "VOLUME_NEGATIVE" in _codes(validate_ohlcv(df))


def test_ohlc_relation_violations_reported() -> None:
    df = build_ohlcv(_VALID_ROWS)
    df.loc[0, "high"] = 99.5  # high < max(open, close)=100.5
    df.loc[1, "low"] = 101.5  # low > min(open, close)=100.5
    report = validate_ohlcv(df)
    assert "HIGH_BELOW_MAX_OPEN_CLOSE" in _codes(report)
    assert "LOW_ABOVE_MIN_OPEN_CLOSE" in _codes(report)


def test_multiple_violations_collected_at_once() -> None:
    df = build_ohlcv(_VALID_ROWS)
    df.loc[1, "timestamp"] = df["timestamp"].iloc[0]
    df.loc[1, "volume"] = -1
    report = validate_ohlcv(df)
    assert {"TIMESTAMP_DUPLICATE", "VOLUME_NEGATIVE"} <= _codes(report)


def test_raise_if_invalid_includes_all_codes() -> None:
    df = build_ohlcv(_VALID_ROWS)
    df.loc[0, "volume"] = -1
    with pytest.raises(ValidationError) as excinfo:
        validate_ohlcv(df).raise_if_invalid()
    assert "[VOLUME_NEGATIVE]" in str(excinfo.value)

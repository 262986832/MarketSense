"""周期映射与校验（AC-3）。"""

from __future__ import annotations

import pytest

from dataset.errors import ConfigError, UnknownPeriodError
from dataset.periods import (
    DURATION_SECONDS_TO_PERIOD,
    PERIOD_TO_DURATION_SECONDS,
    resolve_duration_seconds,
    supported_periods_text,
    validate_period_seconds,
)


def test_period_mapping_is_five_fixed_tiers() -> None:
    assert dict(PERIOD_TO_DURATION_SECONDS) == {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "1d": 86400,
    }
    assert DURATION_SECONDS_TO_PERIOD[86400] == "1d"


@pytest.mark.parametrize(
    ("period", "seconds"),
    [("1m", 60), ("5m", 300), ("15m", 900), ("1h", 3600), ("1d", 86400)],
)
def test_resolve_duration_seconds_known_periods(period: str, seconds: int) -> None:
    assert resolve_duration_seconds(period) == seconds


def test_unknown_period_error_lists_supported_periods() -> None:
    with pytest.raises(UnknownPeriodError) as excinfo:
        resolve_duration_seconds("7m")
    assert "1m, 5m, 15m, 1h, 1d" in str(excinfo.value)
    assert supported_periods_text() == "1m, 5m, 15m, 1h, 1d"


def test_validate_period_seconds_accepts_mapping_values_only() -> None:
    assert validate_period_seconds(300) == 300
    with pytest.raises(ConfigError) as excinfo:
        validate_period_seconds(420)
    assert "支持的取值" in str(excinfo.value)

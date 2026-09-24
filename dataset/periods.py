"""周期映射与校验。

周期名 → tqsdk ``duration_seconds``；首批支持五档。使用哪个周期由配置/CLI
注入，不硬编码（参数配置化）。

移植自 ``reference/perception/src/marksense/data/periods.py``。
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from dataset.errors import ConfigError, UnknownPeriodError

#: MarketSense 周期名 → duration_seconds（docs/10 §6 的五档，语义不变）
PERIOD_TO_DURATION_SECONDS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "1d": 86400,
    }
)

DURATION_SECONDS_TO_PERIOD: Final[Mapping[int, str]] = MappingProxyType(
    {duration: period for period, duration in PERIOD_TO_DURATION_SECONDS.items()}
)


def supported_periods_text() -> str:
    """支持的周期列表文本（按映射表顺序，即周期由小到大），供 CLI/错误消息使用。"""
    return ", ".join(PERIOD_TO_DURATION_SECONDS)


def resolve_duration_seconds(period: str) -> int:
    """周期名 → ``duration_seconds``；未知周期抛 :class:`UnknownPeriodError`。

    错误消息包含支持周期列表，供 CLI 直接输出（AC-3）。
    """
    try:
        return PERIOD_TO_DURATION_SECONDS[period]
    except KeyError:
        raise UnknownPeriodError(
            f"未知周期 {period!r}，支持的周期: {supported_periods_text()}"
        ) from None


def validate_period_seconds(period_seconds: int) -> int:
    """校验配置中的 ``duration_seconds`` 是否在映射表内。"""
    if period_seconds not in DURATION_SECONDS_TO_PERIOD:
        supported = ", ".join(str(d) for d in sorted(DURATION_SECONDS_TO_PERIOD))
        raise ConfigError(
            f"未知 period_seconds {period_seconds!r}，支持的取值: {supported}"
        )
    return period_seconds

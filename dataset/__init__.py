"""dataset 子应用：天勤 K 线落盘 + 转折点提取（公开接口面）。

用法：``python -m dataset {fetch,turning-points,prepare} ...``，详见 ``dataset/README.md``。
"""

from __future__ import annotations

from dataset.cli import main
from dataset.config import DatasetConfig, load_dataset_config, require_credentials
from dataset.errors import (
    ConfigError,
    DataLoadError,
    DatasetError,
    ProviderNotConnectedError,
    UnknownPeriodError,
    ValidationError,
)
from dataset.ohlcv import (
    OHLCV_COLUMNS,
    TIMEZONE,
    clean_serial_klines,
    parse_date_bound,
    standardize_klines,
)
from dataset.periods import (
    PERIOD_TO_DURATION_SECONDS,
    resolve_duration_seconds,
    supported_periods_text,
)
from dataset.provider import DataProvider, TianQinProvider, validate_data_length
from dataset.storage import LoadedOHLCV, load_ohlcv, save_ohlcv
from dataset.turning_points import (
    INITIAL_DIRECTION_MODES,
    TURNING_POINT_COLUMNS,
    LoadedTurningPoints,
    TurningPoint,
    WindowMeta,
    build_window_meta,
    find_turning_points,
    load_turning_points,
    resolve_initial_direction,
    save_turning_points,
)
from dataset.validator import ValidationReport, Violation, validate_ohlcv

__all__ = [
    # CLI
    "main",
    # 配置
    "DatasetConfig",
    "load_dataset_config",
    "require_credentials",
    # 异常
    "DatasetError",
    "ConfigError",
    "UnknownPeriodError",
    "ProviderNotConnectedError",
    "DataLoadError",
    "ValidationError",
    # OHLCV
    "OHLCV_COLUMNS",
    "TIMEZONE",
    "standardize_klines",
    "clean_serial_klines",
    "parse_date_bound",
    # 周期
    "PERIOD_TO_DURATION_SECONDS",
    "resolve_duration_seconds",
    "supported_periods_text",
    # 校验
    "ValidationReport",
    "Violation",
    "validate_ohlcv",
    # 落盘/读取
    "LoadedOHLCV",
    "save_ohlcv",
    "load_ohlcv",
    # 转折点
    "INITIAL_DIRECTION_MODES",
    "TURNING_POINT_COLUMNS",
    "TurningPoint",
    "WindowMeta",
    "LoadedTurningPoints",
    "build_window_meta",
    "resolve_initial_direction",
    "find_turning_points",
    "save_turning_points",
    "load_turning_points",
    # Provider
    "DataProvider",
    "TianQinProvider",
    "validate_data_length",
]

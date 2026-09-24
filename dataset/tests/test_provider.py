"""天勤 Provider（api_factory 注入、两条取数路径与边界）（AC-1、AC-3、AC-5）。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dataset import config as config_module
from dataset.config import DatasetConfig, load_dataset_config
from dataset.errors import DatasetError, ProviderNotConnectedError
from dataset.ohlcv import TIMEZONE
from dataset.provider import (
    SERIAL_MAX_DATA_LENGTH,
    TianQinProvider,
    validate_data_length,
)
from dataset.tests.conftest import (
    FakeTqApi,
    RAW_ROWS_ASC,
    build_raw_klines,
    build_serial_klines,
)

_SERIAL_ROWS = [
    ("2024-01-02 09:00:00", 67950.0, 68020.0, 67900.0, 68000.0, 100, 5000, 5000),
    ("2024-01-02 09:01:00", 68000.0, 68100.0, 67900.0, 68050.0, 120, 5000, 4980),
    ("2024-01-02 09:02:00", 68050.0, 68120.0, 68010.0, 68100.0, 150, 4980, 4900),
    ("2024-01-02 09:03:00", 68100.0, 68200.0, 68090.0, 68150.0, 90, 4900, 4850),
    ("2024-01-02 09:04:00", 68150.0, 68250.0, 68140.0, 68200.0, 80, 4850, 4800),
]


def _config(tmp_path: Path, **overrides) -> DatasetConfig:
    values = {
        "provider_type": "tianqin",
        "account": "user",
        "password": "pass",
        "output_dir": tmp_path / "data",
        "output_format": "csv",
        "include_oi": False,
        "period": "1m",
        "initial_direction": "auto",
        "config_path": None,
    }
    values.update(overrides)
    return DatasetConfig(**values)


def _provider(tmp_path: Path, api: FakeTqApi, **overrides) -> TianQinProvider:
    return TianQinProvider(_config(tmp_path, **overrides), api_factory=lambda _cfg: api)


def test_not_connected_raises_for_both_paths(tmp_path: Path) -> None:
    provider = TianQinProvider(_config(tmp_path), api_factory=lambda _cfg: FakeTqApi())

    with pytest.raises(ProviderNotConnectedError, match="fetch_history"):
        provider.fetch_history("DCE.v2701", "1m", "2024-01-01", "2024-01-02")
    with pytest.raises(ProviderNotConnectedError, match="fetch_recent"):
        provider.fetch_recent("DCE.v2701", "1m", 3)


def test_connect_is_idempotent_and_close_resets(tmp_path: Path) -> None:
    created: list[FakeTqApi] = []

    def factory(_cfg: DatasetConfig) -> FakeTqApi:
        api = FakeTqApi()
        created.append(api)
        return api

    provider = TianQinProvider(_config(tmp_path), api_factory=factory)
    provider.connect()
    provider.connect()
    assert len(created) == 1

    provider.close()
    provider.close()
    assert created[0].close_count == 1
    with pytest.raises(ProviderNotConnectedError):
        provider.fetch_recent("DCE.v2701", "1m", 3)


def test_connect_failure_wrapped(tmp_path: Path) -> None:
    def failing(_cfg: DatasetConfig):
        raise RuntimeError("network down")

    provider = TianQinProvider(_config(tmp_path), api_factory=failing)
    with pytest.raises(DatasetError, match="天勤连接/认证失败"):
        provider.connect()


def test_default_factory_without_credentials_never_echoes_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(config_module.ENV_ACCOUNT, raising=False)
    monkeypatch.delenv(config_module.ENV_CONFIG, raising=False)
    monkeypatch.delenv(config_module.ENV_DATA_DIR, raising=False)
    sentinel = "S3CRET-PASSWORD"
    monkeypatch.setenv(config_module.ENV_PASSWORD, sentinel)
    monkeypatch.setattr(
        config_module, "DEFAULT_CONFIG_PATH", tmp_path / "missing.yaml"
    )
    config = load_dataset_config()
    provider = TianQinProvider(config)  # api_factory 缺省 = 生产工厂

    with pytest.raises(DatasetError) as excinfo:
        provider.connect()

    assert sentinel not in str(excinfo.value)
    assert "account" in str(excinfo.value)


def test_fetch_history_standardizes_and_passes_datetime_bounds(tmp_path: Path) -> None:
    api = FakeTqApi()
    provider = _provider(tmp_path, api)
    provider.connect()

    df = provider.fetch_history("DCE.v2701", "5m", "2024-01-01", "2024-01-02")

    call = api.calls[-1]
    assert call["method"] == "get_kline_data_series"
    assert call["symbol"] == "DCE.v2701"
    assert call["duration_seconds"] == 300
    assert call["start_dt"].isoformat() == "2024-01-01T00:00:00"
    assert call["end_dt"].isoformat() == "2024-01-02T00:00:00"
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert str(df["timestamp"].dt.tz) == TIMEZONE
    assert len(df) == len(RAW_ROWS_ASC)


def test_fetch_history_rejects_bad_date_bounds(tmp_path: Path) -> None:
    provider = _provider(tmp_path, FakeTqApi())
    provider.connect()
    with pytest.raises(DatasetError, match="不是合法的 ISO"):
        provider.fetch_history("DCE.v2701", "1m", "2024/01/01", "2024-01-02")


def test_fetch_recent_requests_one_extra_bar_and_returns_n(tmp_path: Path) -> None:
    api = FakeTqApi(serial=build_serial_klines(_SERIAL_ROWS, width=8))
    provider = _provider(tmp_path, api)
    provider.connect()
    as_of = pd.Timestamp("2024-01-02 09:10:00", tz=TIMEZONE)

    df = provider.fetch_recent("DCE.v2701", "1m", 4, as_of=as_of)

    assert api.calls[-1]["method"] == "get_kline_serial"
    assert api.calls[-1]["data_length"] == 5  # 多取 1 根
    assert api.calls[-1]["duration_seconds"] == 60
    assert len(df) == 4
    assert list(df.index) == [0, 1, 2, 3]
    assert str(df["timestamp"].iloc[-1]) == _SERIAL_ROWS[-1][0] + "+08:00"


def test_fetch_recent_drops_unclosed_bar_before_truncating(tmp_path: Path) -> None:
    api = FakeTqApi(serial=build_serial_klines(_SERIAL_ROWS, width=8))
    provider = _provider(tmp_path, api)
    provider.connect()
    # as_of = 最后一根开始时间 → 最后一根未收盘，被 clean_serial_klines 剔除
    as_of = pd.Timestamp(_SERIAL_ROWS[-1][0], tz=TIMEZONE)

    df = provider.fetch_recent("DCE.v2701", "1m", 4, as_of=as_of)

    assert len(df) == 4
    assert str(df["timestamp"].iloc[-1]) == _SERIAL_ROWS[-2][0] + "+08:00"


def test_fetch_recent_at_max_length_does_not_overflow(tmp_path: Path) -> None:
    api = FakeTqApi(serial=build_serial_klines(_SERIAL_ROWS, width=8))
    provider = _provider(tmp_path, api)
    provider.connect()

    provider.fetch_recent(
        "DCE.v2701",
        "1m",
        SERIAL_MAX_DATA_LENGTH,
        as_of=pd.Timestamp("2024-01-02 09:10:00", tz=TIMEZONE),
    )

    assert api.calls[-1]["data_length"] == SERIAL_MAX_DATA_LENGTH


def _serial_rows(count: int) -> list[tuple]:
    """生成 ``count`` 根连续 1m K 线（间隔 1 分钟，升序）。"""
    base = pd.Timestamp("2024-01-02 09:00:00", tz=TIMEZONE)
    rows = []
    for index in range(count):
        stamp = base + pd.Timedelta(minutes=index)
        price = 68000.0 + (index % 50)
        rows.append(
            (
                stamp.strftime("%Y-%m-%d %H:%M:%S"),
                price,
                price + 10.0,
                price - 10.0,
                price + 1.0,
                100,
                5000,
                5000,
            )
        )
    return rows


def test_fetch_recent_at_max_length_returns_at_most_n_minus_one(tmp_path: Path) -> None:
    """P2-3：上限 8964 时无法多取 1 根，末根未收盘被剔除 → 实际 8963 根（已知边界）。

    本测试把该边界固定为事实（不静默）：CLI 层在返回根数少于请求根数时输出告警，
    见 ``test_cli.py::test_bars_shortfall_is_reported_on_stderr``。
    """
    rows = _serial_rows(SERIAL_MAX_DATA_LENGTH)
    api = FakeTqApi(serial=build_serial_klines(rows, width=SERIAL_MAX_DATA_LENGTH))
    provider = _provider(tmp_path, api)
    provider.connect()
    as_of = pd.Timestamp(rows[-1][0], tz=TIMEZONE)  # 末根未收盘

    df = provider.fetch_recent("DCE.v2701", "1m", SERIAL_MAX_DATA_LENGTH, as_of=as_of)

    assert api.calls[-1]["data_length"] == SERIAL_MAX_DATA_LENGTH
    assert len(df) == SERIAL_MAX_DATA_LENGTH - 1
    assert str(df["timestamp"].iloc[-1]) == rows[-2][0] + "+08:00"


@pytest.mark.parametrize("bad_length", [0, -1, 8965, True, 1.5])
def test_data_length_bounds_rejected(tmp_path: Path, bad_length) -> None:
    provider = _provider(tmp_path, FakeTqApi())
    provider.connect()
    with pytest.raises(DatasetError, match=f"1~{SERIAL_MAX_DATA_LENGTH}"):
        provider.fetch_recent("DCE.v2701", "1m", bad_length)
    with pytest.raises(DatasetError, match=f"1~{SERIAL_MAX_DATA_LENGTH}"):
        validate_data_length(bad_length)


def test_fetch_recent_times_out_when_serial_never_ready(tmp_path: Path) -> None:
    """序列始终全为填充行时，等待就绪超时 → 明确报错，不挂起。"""
    api = FakeTqApi(serial=build_serial_klines([], width=3), serial_ready=False)
    provider = _provider(tmp_path, api)
    provider.connect()

    with pytest.raises(DatasetError, match="就绪超时"):
        provider.fetch_recent("DCE.v2701", "1m", 3, wait_timeout_seconds=0.05)


def test_fetch_recent_without_wait_support_reports_no_valid_rows(tmp_path: Path) -> None:
    class NoWaitApi(FakeTqApi):
        wait_update = None  # 桩不支持等待：直接报「无有效数据行」

    api = NoWaitApi(serial=build_serial_klines([], width=3))
    provider = _provider(tmp_path, api)
    provider.connect()

    with pytest.raises(DatasetError, match="无有效数据行"):
        provider.fetch_recent("DCE.v2701", "1m", 3)


def test_include_oi_propagates_to_both_paths(tmp_path: Path) -> None:
    api = FakeTqApi(serial=build_serial_klines(_SERIAL_ROWS, width=8))
    provider = _provider(tmp_path, api, include_oi=True)
    provider.connect()

    history = provider.fetch_history("DCE.v2701", "1m", "2024-01-01", "2024-01-02")
    recent = provider.fetch_recent(
        "DCE.v2701", "1m", 3, as_of=pd.Timestamp("2024-01-02 09:10:00", tz=TIMEZONE)
    )

    assert "open_oi" in history.columns and "close_oi" in history.columns
    assert "open_oi" in recent.columns and "close_oi" in recent.columns


def test_fetch_and_save_writes_csv_and_sidecar(tmp_path: Path) -> None:
    provider = _provider(tmp_path, FakeTqApi())
    provider.connect()

    path = provider.fetch_and_save("DCE.v2701", "1m", "2024-01-01", "2024-01-02")

    assert path == tmp_path / "data" / "ohlcv" / "DCE.v2701_1m.csv"
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["start_dt"] == "2024-01-01"
    assert meta["end_dt"] == "2024-01-02"
    assert meta["row_count"] == len(RAW_ROWS_ASC)


def test_fetch_recent_and_save_uses_actual_window_bounds(tmp_path: Path) -> None:
    api = FakeTqApi(serial=build_serial_klines(_SERIAL_ROWS, width=8))
    provider = _provider(tmp_path, api)
    provider.connect()

    path = provider.fetch_recent_and_save(
        "DCE.v2701",
        "1m",
        3,
        as_of=pd.Timestamp("2024-01-02 09:10:00", tz=TIMEZONE),
    )

    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["row_count"] == 3
    assert meta["first_timestamp"] == _SERIAL_ROWS[-3][0] + "+08:00"
    assert meta["last_timestamp"] == _SERIAL_ROWS[-1][0] + "+08:00"
    assert meta["start_dt"] == meta["first_timestamp"]
    assert meta["end_dt"] == meta["last_timestamp"]


def test_fetch_recent_waits_until_serial_ready(tmp_path: Path) -> None:
    """序列就绪（存在 id>=0 行）时不进入等待，直接返回数据。"""
    api = FakeTqApi(serial=build_serial_klines(_SERIAL_ROWS, width=8), serial_ready=False)
    provider = _provider(tmp_path, api)
    provider.connect()

    df = provider.fetch_recent(
        "DCE.v2701",
        "1m",
        3,
        as_of=pd.Timestamp("2024-01-02 09:10:00", tz=TIMEZONE),
        wait_timeout_seconds=0.01,
    )

    assert len(df) == 3


def test_history_path_does_not_require_serial_columns(tmp_path: Path) -> None:
    api = FakeTqApi(raw=build_raw_klines())
    provider = _provider(tmp_path, api)
    provider.connect()

    df = provider.fetch_history("DCE.v2701", "1d", "2024-01-01", "2024-01-03")

    assert api.calls[-1]["duration_seconds"] == 86400
    assert len(df) == len(RAW_ROWS_ASC)

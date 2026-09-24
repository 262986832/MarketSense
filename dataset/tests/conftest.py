"""dataset 测试公共夹具（数据构造桩，测试不触网）。

原始 K 线与序列形态按天勤契约构造：

* ``get_kline_data_series`` 的 ``datetime`` 为纳秒时间戳整数（北京时间）；
* ``get_kline_serial`` 固定 width 行，数据不足时前端以「负 id + datetime=0」填充，
  所有列为 float64，形成中的 K 线拥有正 id。

移植自 ``reference/perception/tests/data/conftest.py``（配置桩改为
``dataset`` / ``tianqin`` 两段式 ``build_config_payload``；新增 ``build_ohlcv``
标准 OHLCV 构造器与 ``FakeSerialApi`` 保留）。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

#: 已按时间升序的三根已收盘 K 线（北京时间）
RAW_ROWS_ASC: list[tuple[str, float, float, float, float, int, int, int]] = [
    ("2024-01-02 09:00:00", 67950.0, 68020.0, 67900.0, 68000.0, 100, 5000, 5000),
    ("2024-01-02 09:01:00", 68000.0, 68100.0, 67900.0, 68050.0, 120, 5000, 4980),
    ("2024-01-02 09:02:00", 68050.0, 68120.0, 68010.0, 68100.0, 150, 4980, 4900),
]

#: 1 分钟周期秒数（构造未收盘判定用）
ONE_MINUTE_SECONDS: int = 60


def _beijing_ns(moment: str) -> int:
    """把北京时间字符串转换为纳秒时间戳整数（天勤 datetime 契约）。"""
    return int(pd.Timestamp(moment, tz="Asia/Shanghai").value)


def build_raw_klines(rows: list[tuple] | None = None) -> pd.DataFrame:
    """构造天勤 ``get_kline_data_series`` 形态的原始 K 线 DataFrame。"""
    rows = RAW_ROWS_ASC if rows is None else rows
    return pd.DataFrame(
        {
            "datetime": [_beijing_ns(r[0]) for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [r[5] for r in rows],
            "open_oi": [r[6] for r in rows],
            "close_oi": [r[7] for r in rows],
        }
    )


def build_serial_klines(rows: list[tuple] | None = None, width: int = 10) -> pd.DataFrame:
    """构造 ``get_kline_serial`` 形态的原始 DataFrame（含填充行、float64）。"""
    rows = RAW_ROWS_ASC if rows is None else rows
    n = len(rows)
    pad = max(width - n, 0)
    valid = [
        {
            "id": float(i),
            "datetime": float(_beijing_ns(r[0])),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
            "open_oi": float(r[6]),
            "close_oi": float(r[7]),
        }
        for i, r in enumerate(rows)
    ]
    padded = [
        {
            "id": float(-pad + i),
            "datetime": 0.0,
            "open": float("nan"),
            "high": float("nan"),
            "low": float("nan"),
            "close": float("nan"),
            "volume": float("nan"),
            "open_oi": float("nan"),
            "close_oi": float("nan"),
        }
        for i in range(pad)
    ]
    return pd.DataFrame(padded + valid)


def build_ohlcv(
    rows: list[tuple[float, float, float, float]],
    *,
    start: str = "2026-09-23 09:00:00",
    minutes: int = 1,
) -> pd.DataFrame:
    """``(open, high, low, close)`` 列表 → 标准 OHLCV（升序、tz-aware）。"""
    base = pd.Timestamp(start, tz="Asia/Shanghai")
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(minutes=minutes * i) for i in range(len(rows))],
            "open": [float(r[0]) for r in rows],
            "high": [float(r[1]) for r in rows],
            "low": [float(r[2]) for r in rows],
            "close": [float(r[3]) for r in rows],
            "volume": [100 for _ in rows],
        }
    )


#: 转折点规则固定向量（由 ``reference/perception/tests/test_turning_points.py`` 的 8 条
#: 用例整理而来，作为移植后的固定期望值，同时供 parity 测试对比参考实现）。
#: 每条向量：``rows`` = (open, high, low, close) 序列；``initial_direction`` 为输入模式；
#: ``expect_*`` 为预期输出（kind / price / bar_index 序列）。
TURNING_POINT_VECTORS: list[dict[str, Any]] = [
    {
        "name": "up_then_down_records_high_including_turning_bar",
        "rows": [
            (100, 101, 99, 100),
            (100, 110, 100, 109),
            (109, 120, 105, 119),
            (119, 121, 104, 110),
            (110, 112, 100, 101),
            (101, 113, 99, 112),
        ],
        "initial_direction": "up",
        "expect_kinds": ["start", "high", "low", "close"],
        "expect_prices": [100, 121, 99, 112],
        "expect_bar_index": [0, 3, 5, 5],
    },
    {
        "name": "equal_low_does_not_break_uptrend",
        "rows": [(60, 100, 50, 60), (60, 105, 50, 70), (70, 110, 50, 80), (80, 110, 49, 70)],
        "initial_direction": "up",
        "expect_kinds": ["start", "high", "close"],
        "expect_prices": [60, 110, 70],
        "expect_bar_index": [0, 2, 3],
    },
    {
        "name": "equal_high_does_not_break_downtrend",
        "rows": [(80, 100, 50, 60), (60, 100, 45, 50), (50, 99, 40, 45), (45, 101, 42, 95)],
        "initial_direction": "down",
        "expect_kinds": ["start", "low", "close"],
        "expect_prices": [80, 40, 95],
        "expect_bar_index": [0, 2, 3],
    },
    {
        "name": "extreme_tie_keeps_earliest_bar",
        "rows": [(100, 100, 90, 95), (95, 100, 91, 96), (96, 100, 92, 97), (97, 100, 89, 90)],
        "initial_direction": "up",
        "expect_kinds": ["start", "high", "close"],
        "expect_prices": [100, 100, 90],
        "expect_bar_index": [0, 0, 3],
    },
    {
        "name": "single_bar_returns_start_and_close",
        "rows": [(100, 105, 95, 102)],
        "initial_direction": "auto",
        "expect_kinds": ["start", "close"],
        "expect_prices": [100, 102],
        "expect_bar_index": [0, 0],
    },
    {
        "name": "close_always_appended_even_if_equal_to_last_extreme",
        "rows": [(100, 100, 90, 95), (95, 100, 91, 100), (100, 100, 89, 100)],
        "initial_direction": "up",
        "expect_kinds": ["start", "high", "close"],
        "expect_prices": [100, 100, 100],
        "expect_bar_index": [0, 0, 2],
    },
]


class FakeTqApi:
    """替代 ``TqApi`` 的最小测试桩：区间路径 + 序列路径 + 就绪等待。"""

    def __init__(
        self,
        raw: pd.DataFrame | None = None,
        serial: pd.DataFrame | None = None,
        *,
        serial_ready: bool = True,
    ) -> None:
        self._raw = build_raw_klines() if raw is None else raw
        self._serial = build_serial_klines() if serial is None else serial
        self._serial_ready = serial_ready
        self.calls: list[dict[str, Any]] = []
        self.close_count = 0

    def get_kline_data_series(
        self,
        *,
        symbol: str,
        duration_seconds: int,
        start_dt: Any,
        end_dt: Any,
        adj_type: str | None = None,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "method": "get_kline_data_series",
                "symbol": symbol,
                "duration_seconds": duration_seconds,
                "start_dt": start_dt,
                "end_dt": end_dt,
            }
        )
        return self._raw

    def get_kline_serial(
        self, *, symbol: str, duration_seconds: int, data_length: int
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "method": "get_kline_serial",
                "symbol": symbol,
                "duration_seconds": duration_seconds,
                "data_length": data_length,
            }
        )
        return self._serial

    def wait_update(self, deadline: float | None = None) -> bool:
        return self._serial_ready

    def is_changing(self, obj: Any, key: Any = None) -> bool:
        return True

    def close(self) -> None:
        self.close_count += 1


def build_config_payload(**overrides: Any) -> dict[str, Any]:
    """构造 ``dataset`` / ``tianqin`` 两段式配置映射（写入 YAML 用）。"""
    dataset_section: dict[str, Any] = {
        "type": "tianqin",
        "output_dir": "data",
        "output_format": "csv",
        "include_oi": False,
        "period": "1m",
        "initial_direction": "auto",
    }
    tianqin_section: dict[str, Any] = {"account": "user", "password": "pass"}
    dataset_section.update(overrides.pop("dataset", {}))
    tianqin_section.update(overrides.pop("tianqin", {}))
    payload: dict[str, Any] = {"dataset": dataset_section, "tianqin": tianqin_section}
    payload.update(overrides)
    return payload


@pytest.fixture
def fake_api() -> FakeTqApi:
    """默认桩：区间路径 3 根 K 线、序列路径 10 行（含 7 行填充）。"""
    return FakeTqApi()

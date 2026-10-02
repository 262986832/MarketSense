"""盘面状态（board state）测试：夜盘归属、固定/动态状态、相对价、无未来泄漏、确定性。

语义口径见 ``dataset/board_state.py`` 模块 docstring（2026-10-01 冻结）。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from dataset.board_state import (
    BOARD_STATE_SCHEMA,
    PRECISION,
    STATE_COLUMNS,
    BoardStateTracker,
    attribute_windows,
    build_board_states,
    board_state_filename,
    render_board_state_csv,
    save_board_states,
)
from dataset.cli import main
from dataset.errors import DatasetError
from dataset.storage import file_sha256, save_ohlcv

SYMBOL = "DCE.v2701"


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """``(timestamp 文本, open, high, low, close)`` → 标准 OHLCV（tz-aware +08:00）。"""
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(r[0], tz="Asia/Shanghai") for r in rows],
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [100] * len(rows),
            "open_oi": [1000] * len(rows),
            "close_oi": [1000] * len(rows),
        }
    )


#: 场景：9-04（周五）日盘 + 夜盘（夜盘属 9-07 周一）；9-07（周一）日盘
SCENARIO_MINUTE: list[tuple[str, float, float, float, float]] = [
    ("2026-09-04 09:00:00+08:00", 100.0, 102.0, 99.0, 101.0),
    ("2026-09-04 09:01:00+08:00", 101.0, 105.0, 100.0, 103.0),  # 周五日盘高点 105
    ("2026-09-04 21:00:00+08:00", 103.0, 104.0, 101.0, 102.0),  # 周五夜盘 → 周一窗口首根
    ("2026-09-04 21:01:00+08:00", 102.0, 103.0, 98.0, 102.0),
    ("2026-09-07 09:00:00+08:00", 102.0, 106.0, 101.0, 105.0),
    ("2026-09-07 09:01:00+08:00", 105.0, 110.0, 104.0, 109.0),  # 周一窗口高点 110
    ("2026-09-07 09:02:00+08:00", 109.0, 109.0, 97.0, 108.0),  # 周一窗口低点 97
]

SCENARIO_DAILY: list[tuple[str, float, float, float, float]] = [
    ("2026-09-04 00:00:00+08:00", 100.0, 105.0, 98.0, 103.0),
    ("2026-09-07 00:00:00+08:00", 103.0, 110.0, 97.0, 108.0),
]


# ---------- 夜盘归属 ----------


def test_attribute_windows_night_belongs_to_next_trading_day() -> None:
    df = _bars(SCENARIO_MINUTE)
    windows, unassigned = attribute_windows(df)
    # 9-04（周五）：仅自身日盘 2 根；9-07（周一）：周五夜盘 2 根 + 周一日盘 3 根
    assert windows[dt.date(2026, 9, 4)] == [0, 1]
    assert windows[dt.date(2026, 9, 7)] == [2, 3, 4, 5, 6]
    assert unassigned == 0


def test_attribute_windows_day_without_night() -> None:
    df = _bars([("2026-09-08 09:00:00+08:00", 100, 101, 99, 100)])
    windows, unassigned = attribute_windows(df)
    assert windows[dt.date(2026, 9, 8)] == [0]
    assert unassigned == 0


def test_attribute_windows_unassigned_1530_bar() -> None:
    df = _bars(
        [
            ("2026-09-08 09:00:00+08:00", 100, 101, 99, 100),
            ("2026-09-08 15:30:00+08:00", 100, 101, 99, 100),  # 15:00–20:59：不属任何窗口
        ]
    )
    windows, unassigned = attribute_windows(df)
    assert windows[dt.date(2026, 9, 8)] == [0]
    assert unassigned == 1


# ---------- 跟踪器：固定/动态/相对价 ----------


def test_tracker_fixed_fields_and_relative_translation() -> None:
    t = BoardStateTracker(
        prev_day_high=105.0,
        prev_day_low=98.0,
        prev_day_close=103.0,
        today_open=100.0,
    )
    s = t.update(bar_high=102.0, bar_low=99.0)
    assert s == {
        "prev_day_high": 1.05,
        "prev_day_low": 0.98,
        "prev_day_close": 1.03,
        "today_open": 1.0,
        "today_high": 1.02,
        "today_low": 0.99,
    }


def test_tracker_dynamic_running_extremes() -> None:
    t = BoardStateTracker(
        prev_day_high=105.0,
        prev_day_low=98.0,
        prev_day_close=103.0,
        today_open=100.0,
    )
    t.update(bar_high=102.0, bar_low=99.0)
    s = t.update(bar_high=110.0, bar_low=97.0)
    assert s["today_high"] == 1.10 and s["today_low"] == 0.97
    # 新 K 线不再创新高/新低 → 保持累计极值
    s = t.update(bar_high=108.0, bar_low=99.0)
    assert s["today_high"] == 1.10 and s["today_low"] == 0.97
    # 固定字段不随 K 线变化
    assert s["prev_day_high"] == 1.05 and s["prev_day_low"] == 0.98
    assert s["prev_day_close"] == 1.03 and s["today_open"] == 1.0


def test_tracker_no_future_leakage_prefix_property() -> None:
    highs = [r[2] for r in SCENARIO_MINUTE]
    lows = [r[3] for r in SCENARIO_MINUTE]

    def _feed(tracker: BoardStateTracker, hs: list[float], ls: list[float]) -> list[dict]:
        return [tracker.update(bar_high=h, bar_low=l) for h, l in zip(hs, ls)]

    tracker = BoardStateTracker(
        prev_day_high=105.0, prev_day_low=98.0, prev_day_close=103.0, today_open=100.0
    )
    full = _feed(tracker, highs, lows)
    for k in range(1, len(highs) + 1):
        prefix_tracker = BoardStateTracker(
            prev_day_high=105.0, prev_day_low=98.0, prev_day_close=103.0, today_open=100.0
        )
        prefix = _feed(prefix_tracker, highs[:k], lows[:k])
        # State(T) 只依赖 <= T 的 K 线：截断后续 K 线不改变已有快照
        assert prefix == full[:k]


def test_tracker_rejects_nonpositive_base() -> None:
    with pytest.raises(ValueError):
        BoardStateTracker(
            prev_day_high=105.0, prev_day_low=98.0, prev_day_close=103.0, today_open=0.0
        )


# ---------- 构建：固定值来自日线、逐行动态更新 ----------


def _scenario_result(**kwargs) -> object:
    minute = _bars(SCENARIO_MINUTE)
    daily = _bars(SCENARIO_DAILY)
    return build_board_states(minute, daily, symbol=SYMBOL, **kwargs)


def test_build_prev_day_values_from_daily_file() -> None:
    result = _scenario_result()
    monday = result.df[result.df["trade_date"] == "2026-09-07"]
    assert len(monday) == 5  # 周五夜盘 2 根 + 周一日盘 3 根
    first = monday.iloc[0]
    assert first["timestamp"] == "2026-09-04 21:00:00+08:00"  # 窗口首根 = 周五夜盘
    base = 103.0  # 窗口首根开盘价
    assert first["today_open"] == 1.0
    assert first["prev_day_high"] == round(105.0 / base, PRECISION)
    assert first["prev_day_low"] == round(98.0 / base, PRECISION)
    assert first["prev_day_close"] == round(103.0 / base, PRECISION)


def test_build_dynamic_state_updates_per_bar() -> None:
    result = _scenario_result()
    monday = result.df[result.df["trade_date"] == "2026-09-07"].reset_index(drop=True)
    highs = [float(r["high"]) for _, r in
             _bars(SCENARIO_MINUTE).iloc[2:7].iterrows()]
    lows = [float(r["low"]) for _, r in _bars(SCENARIO_MINUTE).iloc[2:7].iterrows()]
    for i, row in monday.iterrows():
        assert row["today_high"] == round(max(highs[: i + 1]) / 103.0, PRECISION)
        assert row["today_low"] == round(min(lows[: i + 1]) / 103.0, PRECISION)
    # 逐根单调：high 不降、low 不升
    assert (monday["today_high"].diff().dropna() >= 0).all()
    assert (monday["today_low"].diff().dropna() <= 0).all()


def test_build_skips_first_trading_day_explicitly() -> None:
    result = _scenario_result()
    assert result.trade_days == ["2026-09-07"]
    assert len(result.skipped_days) == 1
    assert "2026-09-04" in result.skipped_days[0]
    assert "无前置交易日" in result.skipped_days[0]
    assert set(result.df["trade_date"]) == {"2026-09-07"}


def test_build_errors_when_daily_missing_prev_day() -> None:
    minute = _bars(SCENARIO_MINUTE)
    daily = _bars([SCENARIO_DAILY[1]])  # 只落了 9-07，缺 9-04
    with pytest.raises(DatasetError, match="2026-09-04"):
        build_board_states(minute, daily, symbol=SYMBOL)


def test_build_start_end_filter() -> None:
    result = _scenario_result(start=dt.date(2026, 9, 7), end=dt.date(2026, 9, 7))
    assert result.trade_days == ["2026-09-07"]
    assert len(result.df) == 5
    # 只请求首日 → 该日被跳过（无前置），df 为空
    empty = _scenario_result(start=dt.date(2026, 9, 4), end=dt.date(2026, 9, 4))
    assert empty.df.empty and empty.trade_days == []


# ---------- 渲染与落盘：确定性、sidecar ----------


def test_render_csv_format_is_deterministic() -> None:
    result = _scenario_result()
    text1 = render_board_state_csv(result.df)
    text2 = render_board_state_csv(result.df)
    assert text1 == text2
    lines = text1.splitlines()
    assert lines[0] == ",".join(STATE_COLUMNS)
    row = lines[1].split(",")
    assert row[0] == "2026-09-07" and row[2] == "0"
    assert row[3] == "1.019417"  # round(105/103, 6) 固定 6 位小数
    assert row[6] == "1.000000"  # today_open 恒为 1
    assert text1.endswith("\n")


def test_save_board_states_sidecar_contents(tmp_path: Path) -> None:
    minute = _bars(SCENARIO_MINUTE)
    daily = _bars(SCENARIO_DAILY)
    minute_path = save_ohlcv(
        minute, symbol=SYMBOL, period="1m", output_dir=tmp_path / "ohlcv"
    )
    daily_path = save_ohlcv(
        daily, symbol=SYMBOL, period="1d", output_dir=tmp_path / "ohlcv"
    )
    result = build_board_states(minute, daily, symbol=SYMBOL)
    path = save_board_states(
        result,
        symbol=SYMBOL,
        period="1m",
        output_dir=tmp_path,
        minute_path=minute_path,
        daily_path=daily_path,
    )
    assert path == tmp_path / "board_state" / board_state_filename(SYMBOL)
    assert path.is_file()
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["schema"] == BOARD_STATE_SCHEMA
    assert sidecar["symbol"] == SYMBOL and sidecar["period"] == "1m"
    assert sidecar["precision"] == PRECISION
    assert sidecar["trade_days"] == ["2026-09-07"]
    assert len(sidecar["skipped_days"]) == 1
    assert sidecar["row_count"] == len(result.df)
    assert sidecar["source_minute"]["file_sha256"] == file_sha256(minute_path)
    assert sidecar["source_daily"]["file_sha256"] == file_sha256(daily_path)
    assert sidecar["file_sha256"] == file_sha256(path)
    # 语义描述齐备（窗口/固定/动态/相对价/前日来源）
    assert set(sidecar["semantics"]) == {
        "window",
        "fixed",
        "dynamic",
        "dynamic_rule",
        "relative",
        "prev_day_source",
    }


# ---------- CLI：离线、可复现、参数校验 ----------


def _config_file(tmp_path: Path) -> Path:
    path = tmp_path / "tianqin.test.yaml"
    path.write_text(
        "dataset:\n"
        "  type: tianqin\n"
        f"  output_dir: {tmp_path / 'data'}\n"
        "  output_format: csv\n"
        "  period: 1m\n"
        "tianqin:\n"
        "  account: user\n"
        "  password: pass\n",
        encoding="utf-8",
    )
    return path


def _seed_scenario(tmp_path: Path) -> None:
    ohlcv_dir = tmp_path / "data" / "ohlcv"
    save_ohlcv(_bars(SCENARIO_MINUTE), symbol=SYMBOL, period="1m", output_dir=ohlcv_dir)
    save_ohlcv(_bars(SCENARIO_DAILY), symbol=SYMBOL, period="1d", output_dir=ohlcv_dir)


def _forbidden_factory():
    def _forbidden(_cfg):  # pragma: no cover - 触发即为失败
        pytest.fail("board-state 子命令不得联网")

    return _forbidden


def test_board_state_cli_offline_and_reproducible(tmp_path: Path, capsys) -> None:
    _seed_scenario(tmp_path)
    config = _config_file(tmp_path)
    argv = ["board-state", "--symbol", SYMBOL, "--config", str(config)]

    assert main(argv, api_factory=_forbidden_factory()) == 0
    path = tmp_path / "data" / "board_state" / board_state_filename(SYMBOL)
    assert path.is_file()
    first = (path.read_bytes(), path.with_suffix(".json").read_bytes())

    assert main(argv, api_factory=_forbidden_factory()) == 0
    assert first == (path.read_bytes(), path.with_suffix(".json").read_bytes())

    df = pd.read_csv(path)
    assert len(df) == 5
    assert set(df["trade_date"]) == {"2026-09-07"}
    assert df["today_open"].eq(1.0).all()
    out = capsys.readouterr()
    assert "跳过交易日 2026-09-04" in out.err  # 首日跳过显式告警，不静默


def test_board_state_cli_start_end_filter(tmp_path: Path) -> None:
    _seed_scenario(tmp_path)
    config = _config_file(tmp_path)
    assert (
        main(
            [
                "board-state",
                "--symbol",
                SYMBOL,
                "--start",
                "2026-09-07",
                "--end",
                "2026-09-07",
                "--config",
                str(config),
            ],
            api_factory=_forbidden_factory(),
        )
        == 0
    )
    path = tmp_path / "data" / "board_state" / board_state_filename(SYMBOL)
    df = pd.read_csv(path)
    assert set(df["trade_date"]) == {"2026-09-07"}


def test_board_state_cli_rejects_non_1m_period(tmp_path: Path) -> None:
    _seed_scenario(tmp_path)
    config = _config_file(tmp_path)
    assert (
        main(
            ["board-state", "--symbol", SYMBOL, "--period", "5m", "--config", str(config)],
            api_factory=_forbidden_factory(),
        )
        == 2
    )


def test_board_state_cli_start_end_must_pair(tmp_path: Path) -> None:
    _seed_scenario(tmp_path)
    config = _config_file(tmp_path)
    assert (
        main(
            ["board-state", "--symbol", SYMBOL, "--start", "2026-09-07", "--config", str(config)],
            api_factory=_forbidden_factory(),
        )
        == 2
    )


def test_board_state_cli_missing_daily_errors(tmp_path: Path, capsys) -> None:
    ohlcv_dir = tmp_path / "data" / "ohlcv"
    save_ohlcv(_bars(SCENARIO_MINUTE), symbol=SYMBOL, period="1m", output_dir=ohlcv_dir)
    config = _config_file(tmp_path)
    assert (
        main(["board-state", "--symbol", SYMBOL, "--config", str(config)], api_factory=_forbidden_factory())
        == 1
    )
    out = capsys.readouterr()
    assert "日线" in out.err

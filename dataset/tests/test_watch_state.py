"""``dataset/watch_state`` 单测：显示层裁剪 + 边界/时间戳头 + 循环驱动。"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from dataset import watch_state as watch_state_module
from dataset.errors import DatasetError
from dataset.market_episode.nanojev_records import STATE_SCHEMA
from dataset.watch_state import (
    MARKET_LINE_PREFIXES,
    EXIT_INTERRUPTED,
    market_lines,
    next_boundary,
    run_watch,
    timestamp_header,
)


def _sample_state_text() -> str:
    """与 ``render_state`` 同构的七段样本（schema/账户/行情五部分，``" \\n "`` 连接）。"""
    return " \n ".join(
        (
            STATE_SCHEMA,
            "账户: 持空 净值=100.000000 今日=0.000000 回撤=0.000000",
            "日线: 昨日高=1.010000 昨日低=0.990000 昨日收=1.000000",
            "日内: bar=7 今高=1.020000 今低=0.995000",
            "联动: sym（突破=0.250000）",
            "现价: 开=1.000000 高=1.005000 低=0.998000 收=1.003000",
            "盘口: na",
        )
    )


class TestMarketLines:
    def test_keeps_five_market_lines_byte_exact(self) -> None:
        lines = market_lines(_sample_state_text())
        assert lines == [
            "日线: 昨日高=1.010000 昨日低=0.990000 昨日收=1.000000",
            "日内: bar=7 今高=1.020000 今低=0.995000",
            "联动: sym（突破=0.250000）",
            "现价: 开=1.000000 高=1.005000 低=0.998000 收=1.003000",
            "盘口: na",
        ]

    def test_excludes_schema_and_account_lines(self) -> None:
        lines = market_lines(_sample_state_text())
        assert len(lines) == 5
        assert all(not line.startswith(STATE_SCHEMA) for line in lines)
        assert all(not line.startswith("账户") for line in lines)

    def test_prefix_must_be_exact(self) -> None:
        text = " \n ".join(("日线x: 不应命中", "日线: 命中"))
        assert market_lines(text) == ["日线: 命中"]

    def test_order_preserved_no_reordering(self) -> None:
        text = " \n ".join(("现价: a", "盘口: na", "日线: b", "日内: c", "联动: d"))
        assert market_lines(text) == [
            "现价: a",
            "盘口: na",
            "日线: b",
            "日内: c",
            "联动: d",
        ]

    def test_prefixes_constant(self) -> None:
        assert MARKET_LINE_PREFIXES == ("日线", "日内", "联动", "现价", "盘口")


class TestNextBoundary:
    """绝对 epoch 纳秒上取整（设计 §3.3）：周期网格、夜盘、跨零点、tz 保留。"""

    def test_1m_next_minute(self) -> None:
        now = pd.Timestamp("2024-01-02 09:04:30+08:00")
        assert next_boundary(now, 60) == pd.Timestamp("2024-01-02 09:05:00+08:00")

    def test_exact_boundary_advances_one_period(self) -> None:
        now = pd.Timestamp("2024-01-02 09:05:00+08:00")
        assert next_boundary(now, 60) == pd.Timestamp("2024-01-02 09:06:00+08:00")

    def test_5m_night_session_grid(self) -> None:
        now = pd.Timestamp("2024-01-01 23:07:31+08:00")
        assert next_boundary(now, 300) == pd.Timestamp("2024-01-01 23:10:00+08:00")

    def test_crosses_midnight_without_date_assumption(self) -> None:
        now = pd.Timestamp("2024-01-01 23:59:59.5+08:00")
        assert next_boundary(now, 60) == pd.Timestamp("2024-01-02 00:00:00+08:00")

    def test_15m_and_1h_grid(self) -> None:
        now = pd.Timestamp("2024-01-02 09:47:00+08:00")
        assert next_boundary(now, 900) == pd.Timestamp("2024-01-02 10:00:00+08:00")
        assert next_boundary(now, 3600) == pd.Timestamp("2024-01-02 10:00:00+08:00")

    def test_timezone_preserved(self) -> None:
        aware = pd.Timestamp("2024-01-02 09:04:30+08:00")
        assert next_boundary(aware, 60).tz is not None
        naive = pd.Timestamp("2024-01-02 09:04:30")
        assert next_boundary(naive, 60).tz is None


class TestTimestampHeader:
    """``[YYYY-MM-DD HH:MM symbol period]``；时间 = 决策 K 线起点 + 60s（北京）。"""

    def test_format_beijing_close_time(self) -> None:
        ts = pd.Timestamp("2024-01-02 09:04:00+08:00")
        assert timestamp_header("TEST.sym", "1m", ts) == "[2024-01-02 09:05 TEST.sym 1m]"

    def test_period_field_is_trigger_period(self) -> None:
        """period 字段 = watch 触发周期；时间仍由 1m 决策 K 线收盘时刻决定。"""
        ts = pd.Timestamp("2024-01-02 09:04:00+08:00")
        assert timestamp_header("TEST.sym", "5m", ts) == "[2024-01-02 09:05 TEST.sym 5m]"

    def test_naive_treated_as_beijing(self) -> None:
        ts = pd.Timestamp("2024-01-02 09:04:00")
        assert timestamp_header("TEST.sym", "1m", ts) == "[2024-01-02 09:05 TEST.sym 1m]"

    def test_other_tz_converted_to_beijing(self) -> None:
        ts = pd.Timestamp("2024-01-02 01:04:00+00:00")
        assert timestamp_header("TEST.sym", "1m", ts) == "[2024-01-02 09:05 TEST.sym 1m]"


def _frame_1m(stamps: list[str]) -> pd.DataFrame:
    """最小 1m 帧（判重键 = 末行 timestamp）。"""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(stamps).tz_localize("Asia/Shanghai"),
            "close": [100.0 + index for index in range(len(stamps))],
        }
    )


class _WatchHarness:
    """run_watch 注入面收集器：假取数队列/假时钟/假等待/渲染与告警记录。"""

    def __init__(
        self,
        frames: list[pd.DataFrame | BaseException],
        *,
        rest: tuple[pd.DataFrame, dict[str, Any | None]] | None = None,
        now: pd.Timestamp | None = None,
        interrupt_on_wait_call: int | None = None,
    ) -> None:
        self.frames = list(frames)
        self.rest = rest or (pd.DataFrame(), {})
        self.now = now or pd.Timestamp("2024-01-02 09:04:30+08:00")
        self.interrupt_on_wait_call = interrupt_on_wait_call
        self.rendered: list[tuple[Any, pd.Timestamp]] = []
        self.warnings: list[str] = []
        self.deadlines: list[float] = []
        self.rest_calls = 0
        self.build_kwargs: list[dict[str, Any]] = []

    def fetch_1m(self) -> pd.DataFrame:
        item = self.frames.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def fetch_rest(self) -> tuple[pd.DataFrame, dict[str, Any | None]]:
        self.rest_calls += 1
        return self.rest

    def render(self, snapshot: Any, key: pd.Timestamp) -> None:
        self.rendered.append((snapshot, key))

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    def now_fn(self) -> pd.Timestamp:
        return self.now

    def wait_fn(self, deadline: float) -> None:
        self.deadlines.append(deadline)
        self.now = pd.Timestamp(deadline, unit="s", tz="Asia/Shanghai")
        if (
            self.interrupt_on_wait_call is not None
            and len(self.deadlines) >= self.interrupt_on_wait_call
        ):
            raise KeyboardInterrupt  # 桩等待：到指定次数即停（让 run_watch 可终止）

    def run(self, **kwargs: Any) -> int:
        defaults: dict[str, Any] = {
            "symbol": "TEST.sym",
            "period": "1m",
            "bars": 10,
            "daily_bars": 2,
            "params": None,
            "turning_points": (),
            "price_precision": 6,
            "full": False,
            "fetch_1m": self.fetch_1m,
            "fetch_rest": self.fetch_rest,
            "render": self.render,
            "warn": self.warn,
            "now_fn": self.now_fn,
            "wait_fn": self.wait_fn,
        }
        defaults.update(kwargs)
        return run_watch(**defaults)


@pytest.fixture
def fake_build(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """monkeypatch ``watch_state.build_live_state``：记录 kwargs、返回哨兵快照。

    builder 本身的集成由 CLI 桩测试（真实 builder）覆盖；此处专注循环编排。
    """
    calls: list[dict[str, Any]] = []

    def _fake_build(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(watch_state_module, "build_live_state", _fake_build)
    return calls


class TestRunWatch:
    """循环编排：首轮立即输出、判重三态、失败退出、Ctrl+C 130。"""

    def test_first_cycle_immediate_then_interrupt(
        self, fake_build: list[dict[str, Any]]
    ) -> None:
        frame = _frame_1m(["2024-01-02 09:03:00", "2024-01-02 09:04:00"])
        harness = _WatchHarness([frame], interrupt_on_wait_call=1)

        assert harness.run() == EXIT_INTERRUPTED
        assert len(harness.rendered) == 1  # 首轮立即输出，不等边界
        assert harness.deadlines == [
            pd.Timestamp("2024-01-02 09:05:00", tz="Asia/Shanghai").timestamp()
        ]  # 首轮输出后才进入首次边界等待（假时钟 09:04:30 → 09:05）
        assert harness.warnings == []

    def test_same_key_silent_new_key_output_once(
        self, fake_build: list[dict[str, Any]]
    ) -> None:
        frame_a = _frame_1m(["2024-01-02 09:03:00", "2024-01-02 09:04:00"])
        frame_b = _frame_1m(
            ["2024-01-02 09:03:00", "2024-01-02 09:04:00", "2024-01-02 09:05:00"]
        )
        harness = _WatchHarness([frame_a, frame_a, frame_b], interrupt_on_wait_call=3)

        assert harness.run() == EXIT_INTERRUPTED
        assert len(harness.rendered) == 2  # 首轮 + 新根一次；同根静默
        assert harness.rest_calls == 2  # 判重门：同根不取 1d/联动
        assert len(fake_build) == 2

    def test_data_delay_next_round_supplements(
        self, fake_build: list[dict[str, Any]]
    ) -> None:
        """边界醒来无新根 → 静默回等，下一轮判重补出（不阻塞、不重复）。"""
        frame_a = _frame_1m(["2024-01-02 09:03:00", "2024-01-02 09:04:00"])
        frame_b = _frame_1m(
            ["2024-01-02 09:03:00", "2024-01-02 09:04:00", "2024-01-02 09:05:00"]
        )
        harness = _WatchHarness(
            [frame_a, frame_a, frame_a, frame_b], interrupt_on_wait_call=4
        )

        assert harness.run() == EXIT_INTERRUPTED
        assert [key for _, key in harness.rendered] == [
            pd.Timestamp("2024-01-02 09:04:00", tz="Asia/Shanghai"),
            pd.Timestamp("2024-01-02 09:05:00", tz="Asia/Shanghai"),
        ]
        assert len(harness.deadlines) == 4
        # 边界 = 假时钟上取整到 1m 网格（首轮后从 09:04:30 → 09:05 起）
        assert harness.deadlines[0] == pd.Timestamp(
            "2024-01-02 09:05:00", tz="Asia/Shanghai"
        ).timestamp()

    def test_build_receives_decision_index_and_linkage(
        self, fake_build: list[dict[str, Any]]
    ) -> None:
        frame = _frame_1m(["2024-01-02 09:03:00", "2024-01-02 09:04:00"])
        daily = pd.DataFrame()
        linkage: dict[str, Any | None] = {"TEST.lnk": None}
        harness = _WatchHarness(
            [frame], rest=(daily, linkage), interrupt_on_wait_call=1
        )

        harness.run()

        assert fake_build[0]["decision_index"] == 1
        assert fake_build[0]["bars_1m"] is frame
        assert fake_build[0]["daily_rows"] is daily
        assert fake_build[0]["linkage_bars"] is linkage

    def test_empty_1m_frame_counts_as_failure(
        self, fake_build: list[dict[str, Any]]
    ) -> None:
        frame = _frame_1m(["2024-01-02 09:04:00"])
        empty = frame.iloc[0:0]
        harness = _WatchHarness([frame, empty, empty, empty])

        assert harness.run() == 1  # 连续 3 次失败（含空帧）→ 终止
        assert len(harness.warnings) == 3
        assert "连续第 3 次" in harness.warnings[-1]

    def test_three_consecutive_failures_exit_1(
        self, fake_build: list[dict[str, Any]]
    ) -> None:
        harness = _WatchHarness([DatasetError("断"), DatasetError("断"), DatasetError("断")])

        assert harness.run() == 1
        assert [
            f"连续第 {n} 次" in warning
            for n, warning in enumerate(harness.warnings, 1)
        ] == [True, True, True]

    def test_two_failures_then_success_continues(
        self, fake_build: list[dict[str, Any]]
    ) -> None:
        frame_a = _frame_1m(["2024-01-02 09:04:00"])
        frame_b = _frame_1m(["2024-01-02 09:04:00", "2024-01-02 09:05:00"])
        harness = _WatchHarness(
            [DatasetError("抖动"), DatasetError("抖动"), frame_a, frame_b, frame_b],
            interrupt_on_wait_call=4,
        )

        assert harness.run() == EXIT_INTERRUPTED
        assert len(harness.warnings) == 2  # 失败告警两次后成功重置
        assert len(harness.rendered) == 2  # 成功后继续：新根照常输出
        assert harness.rest_calls == 2

    def test_keyboard_interrupt_in_fetch_returns_130(
        self, fake_build: list[dict[str, Any]]
    ) -> None:
        harness = _WatchHarness([KeyboardInterrupt()])

        assert harness.run() == EXIT_INTERRUPTED
        assert harness.warnings == []

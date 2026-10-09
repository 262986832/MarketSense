"""``dataset/live_state`` + ``dataset state-now`` 测试（state-now-view 任务 T4）。

覆盖（对应 ``artifacts/state-now-view/02-design/tech-design.md`` 验证策略 T1–T3/T6–T9
与协调者拍板 A 的三情形要求）：

* 冻结路径纯增量别名 = 同一对象（零行为改动的结构性证据）；
* builder：形状/键序/双跑一致/无未来泄漏（切片上界）/夜盘归属/截断判定（A3）/
  T 日线行号锚定三情形（拍板 A：窗口起点==折点窗口起点恒等、更深起点时长不变、
  锚缺失硬错误）/折点不足/参数透传/联动 na 降级与 metadata；
* CLI：离线（合成 + 本机真实落盘数据，缺文件 skip）与在线桩（按品种注入帧、
  联动失败降级、主品种失败硬错误、参数校验、凭证缺失不回显）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from dataset import config as config_module
from dataset.cli import main
from dataset.errors import DatasetError
from dataset.live_state import DENOMINATOR_NOTE, LiveStateSnapshot, build_live_state
from dataset.market_episode import nanojev_records
from dataset.market_episode.nanojev_records import (
    FLAT_CRITERIA,
    QUESTION_ID,
    QUESTION_INSTRUCTIONS,
    STATE_SCHEMA,
)
from dataset.market_episode.segments import EpisodeParams
from dataset.storage import load_ohlcv, save_ohlcv
from dataset.tests.conftest import build_serial_klines
from dataset.tests.market_episode_fixtures import (
    DAILY_TP_ROWS,
    SYMBOL,
    frame,
    write_daily_turning_points,
)
from dataset.turning_points import load_turning_points
from dataset.watch_state import market_lines

#: 今日（trade_date = 2024-01-02）窗口：夜盘（日历 2024-01-01 21:00 起）+ 日盘
#: （2024-01-02 09:00 起），各 5 根；决策 K 线 = 末根 09:04
NIGHT_ROWS: tuple[tuple[float, float, float, float], ...] = (
    (100, 101, 99, 100),
    (100, 103, 100, 102),
    (102, 102, 98, 99),
    (99, 104, 99, 103),
    (103, 103, 101, 102),
)
DAY_ROWS: tuple[tuple[float, float, float, float], ...] = (
    (102, 105, 101, 104),
    (104, 106, 103, 105),
    (105, 105, 102, 103),
    (103, 104, 100, 101),
    (101, 103, 101, 102),
)
#: 上一交易日日线（01-01）与决策交易日日线（01-02，v9 行号口径）
DAILY_PREV: tuple[float, float, float, float] = (4000.0, 4010.0, 3990.0, 4005.0)
DAILY_T: tuple[float, float, float, float] = (4010.0, 4020.0, 4000.0, 4015.0)

EXPECTED_METADATA_KEYS = [
    "symbol",
    "decision_bar_timestamp",
    "trade_date",
    "reference_bar_timestamp",
    "reference_open",
    "price_precision",
    "bar_index_in_today",
    "today_window",
    "daily_rows",
    "prev_day_date",
    "breakthrough",
    "linkage_symbols",
    "na_details",
    "denominator_note",
]


def daily_frame(
    rows: list[tuple[str, tuple[float, float, float, float]]],
) -> pd.DataFrame:
    """``[(日期, (开, 高, 低, 收)), ...]`` → 1d 标准 OHLCV（逐日独立 frame 再拼接）。"""
    parts = [frame([ohlc], start=f"{date} 00:00:00") for date, ohlc in rows]
    return pd.concat(parts).reset_index(drop=True)


def today_window_frame() -> pd.DataFrame:
    """默认 1m 帧：夜盘 5 根（01-01 21:00~21:04）+ 日盘 5 根（01-02 09:00~09:04）。"""
    night = frame(list(NIGHT_ROWS), start="2024-01-01 21:00:00")
    day = frame(list(DAY_ROWS), start="2024-01-02 09:00:00")
    return pd.concat([night, day]).reset_index(drop=True)


def build_default(
    daily_rows: pd.DataFrame | None = None,
    params: EpisodeParams | None = None,
    linkage_bars: dict[str, pd.DataFrame | None] | None = None,
) -> LiveStateSnapshot:
    """默认形态的快照：T = 日盘末根、daily = [01-01, 01-02]、TP = 默认点集。"""
    return build_live_state(
        symbol=SYMBOL,
        bars_1m=today_window_frame(),
        decision_index=9,
        daily_rows=(
            daily_frame([("2024-01-01", DAILY_PREV), ("2024-01-02", DAILY_T)])
            if daily_rows is None
            else daily_rows
        ),
        turning_points=_default_points(),
        linkage_bars=linkage_bars,
        params=EpisodeParams() if params is None else params,
    )


def _default_points() -> tuple:
    """默认折点（从 DAILY_TP_ROWS 的 CSV 语义直接构造元组，不走文件）。

    与 ``load_turning_points`` 回读同构：up/down 点必须携带当次趋势极值字段
    （``recent_trend_extremes`` 对缺失字段硬报错，生产 CSV 回读恒携带）。
    """
    from dataset.turning_points import TurningPoint  # 局部导入避免夹具循环

    return tuple(
        TurningPoint(
            kind=entry["kind"],
            timestamp=pd.Timestamp(entry["timestamp"]),
            price=entry["price"],
            bar_index=entry["bar_index"],
            volume=entry.get("volume", 0),
            oi=entry.get("oi", 0),
            trend_extreme_price=entry.get("trend_extreme_price"),
            trend_extreme_bar_index=entry.get("trend_extreme_bar_index"),
        )
        for entry in DAILY_TP_ROWS
    )


class TestFrozenAlias:
    """T1：冻结路径纯增量导出的结构性证据（零行为改动）。"""

    def test_alias_is_same_object(self) -> None:
        assert nanojev_records.build_usable_trend_context is (
            nanojev_records._usable_trend_context
        )

    def test_alias_in_all(self) -> None:
        assert "build_usable_trend_context" in nanojev_records.__all__


class TestSnapshotShape:
    """builder 输出形状：state 七段 / question / candidates / metadata 键序。"""

    def test_state_text_seven_segments(self) -> None:
        snapshot = build_default()
        parts = snapshot.state_text.split(" \n ")
        assert len(parts) == 7
        assert parts[0] == STATE_SCHEMA
        assert "持仓=空仓" in parts[1]
        assert "净值=100.000000" in parts[1]
        assert "今日=0.000000" in parts[1]
        assert "回撤=0.000000" in parts[1]
        # v9 日线行趋势项：涨势/跌势各 2 项 + 三分支状态（默认点集 → HH+HL → 涨势中），
        # 状态时长 = 拍板 A 锚定行号（1）− 最近可用折点确认根（0）= 1
        # 趋势极值为比值（分母 = 今日首根开盘 100）→ 4040.0 渲染为 40.400000
        assert "涨势(-1, 最高=40.400000" in parts[2]
        assert "整体为涨势中" in parts[2]
        assert "当前为跌势" in parts[2]
        assert "时长=1根" in parts[2]
        assert parts[6] == "盘口: na"

    def test_question_and_candidates(self) -> None:
        snapshot = build_default()
        assert snapshot.question == {
            QUESTION_ID: {
                "type": "choice",
                "instructions": QUESTION_INSTRUCTIONS,
                "criteria": dict(FLAT_CRITERIA),
            }
        }
        assert snapshot.candidates == dict(FLAT_CRITERIA)

    def test_metadata_key_order_and_values(self) -> None:
        snapshot = build_default()
        assert list(snapshot.metadata) == EXPECTED_METADATA_KEYS
        meta = snapshot.metadata
        assert meta["symbol"] == SYMBOL
        assert meta["decision_bar_timestamp"] == "2024-01-02T09:04:00+08:00"
        assert meta["trade_date"] == "2024-01-02"
        assert meta["reference_bar_timestamp"] == "2024-01-01T21:00:00+08:00"
        assert meta["reference_open"] == 100.0
        assert meta["price_precision"] == 6
        assert meta["bar_index_in_today"] == 9
        assert meta["today_window"] == {
            "first": "2024-01-01T21:00:00+08:00",
            "last": "2024-01-02T09:04:00+08:00",
            "bars": 10,
            "truncated": False,
        }
        assert list(meta["today_window"]) == ["first", "last", "bars", "truncated"]
        assert meta["daily_rows"] == {
            "first": "2024-01-01",
            "last": "2024-01-02",
            "rows": 2,
            "daily_row_derived": False,
        }
        assert meta["prev_day_date"] == "2024-01-01"
        assert meta["breakthrough"] == {
            "window": 20,
            "period": "1m",
            "duration_seconds": 60,
        }
        assert meta["linkage_symbols"] == []
        assert meta["na_details"] == []
        assert meta["denominator_note"] == DENOMINATOR_NOTE
        # JSON 可序列化（预测程序消费边界）
        json.dumps(snapshot.metadata, ensure_ascii=False)

    def test_price_line_bar_index_in_today(self) -> None:
        snapshot = build_default()
        price_line = snapshot.state_text.split(" \n ")[5]
        assert price_line.startswith("现价: 开=1.010000")
        assert "bar=9" in price_line

    def test_no_linkage_means_no_correlation_segment(self) -> None:
        snapshot = build_default()
        linkage_line = snapshot.state_text.split(" \n ")[4]
        assert linkage_line.startswith("联动: sym（突破=")
        assert "相关度" not in linkage_line


class TestDeterminismAndLeak:
    """AC1：双跑一致 + 喂入未来 bar 输出不变（切片上界构造层保证）。"""

    def test_double_run_identical(self) -> None:
        assert build_default() == build_default()

    def test_future_rows_beyond_decision_index_ignored(self) -> None:
        base = build_default()
        future = frame(
            [(102, 104, 101, 103)] * 3, start="2024-01-02 09:05:00"
        )
        extended = pd.concat([today_window_frame(), future]).reset_index(drop=True)
        leaked = build_live_state(
            symbol=SYMBOL,
            bars_1m=extended,
            decision_index=9,
            daily_rows=daily_frame(
                [("2024-01-01", DAILY_PREV), ("2024-01-02", DAILY_T)]
            ),
            turning_points=_default_points(),
            params=EpisodeParams(),
        )
        assert leaked == base


class TestDailyIndexAnchoring:
    """拍板 A：T 日线行号锚定三情形 + A2 推导恒等。"""

    def test_direct_when_window_starts_at_tp_window(self) -> None:
        """daily 首日 == 折点窗口首日（01-01）→ 锚 pos=0 → 与设计字面口径逐值相同。"""
        snapshot = build_default()
        assert snapshot.metadata["daily_rows"]["daily_row_derived"] is False
        assert "时长=1根" in snapshot.state_text.split(" \n ")[2]

    def test_deeper_window_keeps_duration(self) -> None:
        """1d 窗口起点深于折点窗口（前置 2023-12-29 行）→ 锚定修正，时长不变。

        若无锚定（直接取 pos(T)=2），时长会虚高为 2 → 本用例锁定回归。
        """
        deeper = daily_frame(
            [
                ("2023-12-29", (3990.0, 4000.0, 3980.0, 3995.0)),
                ("2024-01-01", DAILY_PREV),
                ("2024-01-02", DAILY_T),
            ]
        )
        snapshot = build_default(daily_rows=deeper)
        base = build_default()
        assert snapshot.state_text == base.state_text
        assert snapshot.metadata["daily_rows"]["rows"] == 3
        assert snapshot.metadata["daily_rows"]["daily_row_derived"] is False

    def test_derived_equals_missing_t_row(self) -> None:
        """在线 1d 无 T 行 → 行号推导（prev+1）与手工补 T 行在 state 上恒等。"""
        derived = build_default(
            daily_rows=daily_frame([("2024-01-01", DAILY_PREV)])
        )
        explicit = build_default()
        assert derived.state_text == explicit.state_text
        assert derived.metadata["daily_rows"]["daily_row_derived"] is True
        assert explicit.metadata["daily_rows"]["daily_row_derived"] is False
        assert derived.metadata["daily_rows"]["last"] == "2024-01-01"

    def test_anchor_date_missing_hard_error(self) -> None:
        """锚日期（折点首点）不在 1d 窗口内 → 硬错误并指引增大 --daily-bars。"""
        with pytest.raises(DatasetError, match="增大 --daily-bars"):
            build_default(
                daily_rows=daily_frame(
                    [
                        ("2023-12-29", (3990.0, 4000.0, 3980.0, 3995.0)),
                        ("2024-01-02", DAILY_T),
                    ]
                )
            )

    def test_anchor_invariant_violation_hard_error(self) -> None:
        """折点首点不是 start/bar_index=0 → 同源不变式破坏，硬错误。"""
        from dataset.turning_points import TurningPoint

        broken = (
            TurningPoint(
                kind="start",
                timestamp=pd.Timestamp("2024-01-01 00:00:00+08:00"),
                price=4000.0,
                bar_index=3,
                volume=100,
                oi=5000,
            ),
        ) + _default_points()[1:]
        with pytest.raises(DatasetError, match="同源"):
            build_live_state(
                symbol=SYMBOL,
                bars_1m=today_window_frame(),
                decision_index=9,
                daily_rows=daily_frame(
                    [("2024-01-01", DAILY_PREV), ("2024-01-02", DAILY_T)]
                ),
                turning_points=broken,
                params=EpisodeParams(),
            )

    def test_stale_daily_window_hard_error(self) -> None:
        """1d 末行晚于 trade_date（错窗）→ 硬错误。"""
        with pytest.raises(DatasetError, match="错窗"):
            build_default(
                daily_rows=daily_frame(
                    [
                        ("2024-01-01", DAILY_PREV),
                        ("2024-01-02", DAILY_T),
                        ("2024-01-03", DAILY_T),
                    ]
                )
            )

    def test_insufficient_turning_points_hard_error(self) -> None:
        """可用 up/down 各 <2（训练侧是跳过，快照侧硬报错不静默）。"""
        from dataset.turning_points import TurningPoint

        thin = (
            TurningPoint(
                kind="start",
                timestamp=pd.Timestamp("2024-01-01 00:00:00+08:00"),
                price=4000.0,
                bar_index=0,
                volume=100,
                oi=5000,
            ),
            TurningPoint(
                kind="down",
                timestamp=pd.Timestamp("2024-01-01 21:05:00+08:00"),
                price=3980.0,
                bar_index=0,
                volume=120,
                oi=5010,
            ),
            TurningPoint(
                kind="up",
                timestamp=pd.Timestamp("2024-01-01 21:35:00+08:00"),
                price=4020.0,
                bar_index=0,
                volume=130,
                oi=5020,
            ),
        )
        with pytest.raises(DatasetError, match="turning-points"):
            build_live_state(
                symbol=SYMBOL,
                bars_1m=today_window_frame(),
                decision_index=9,
                daily_rows=daily_frame(
                    [("2024-01-01", DAILY_PREV), ("2024-01-02", DAILY_T)]
                ),
                turning_points=thin,
                params=EpisodeParams(),
            )


class TestWindowAttribution:
    """夜盘归属（E5）+ 截断判定（A3）+ 未入窗硬错误 + 越界。"""

    def test_night_session_belongs_to_next_trading_day(self) -> None:
        """T = 01-01 21:04（夜盘）→ trade_date = 01-02；1d 无 T 行 → 推导路径。"""
        night_only = frame(list(NIGHT_ROWS), start="2024-01-01 21:00:00")
        snapshot = build_live_state(
            symbol=SYMBOL,
            bars_1m=night_only,
            decision_index=4,
            daily_rows=daily_frame([("2024-01-01", DAILY_PREV)]),
            turning_points=_default_points(),
            params=EpisodeParams(),
        )
        assert snapshot.metadata["trade_date"] == "2024-01-02"
        assert snapshot.metadata["decision_bar_timestamp"] == (
            "2024-01-01T21:04:00+08:00"
        )
        assert snapshot.metadata["bar_index_in_today"] == 4
        assert snapshot.metadata["daily_rows"]["daily_row_derived"] is True
        assert snapshot.metadata["today_window"]["truncated"] is False

    def test_night_session_attribution_via_daily_rows(self) -> None:
        """夜盘快照 + 1d 已含 T 行（离线历史数据）→ 归属键由日线日期揭示，直接取行号。

        归属修复的补集情形：``_explicit_trading_days`` 以日线日期为已知交易日全集，
        夜盘根归属 = 严格大于其日历日的最近已知交易日（此处 01-02 ∈ 日线）→
        无需 A2 推导；与上一用例（隐含次日 + 推导）共同锁住夜盘快照两条归属路径。
        """
        night_only = frame(list(NIGHT_ROWS), start="2024-01-01 21:00:00")
        snapshot = build_live_state(
            symbol=SYMBOL,
            bars_1m=night_only,
            decision_index=4,
            daily_rows=daily_frame(
                [("2024-01-01", DAILY_PREV), ("2024-01-02", DAILY_T)]
            ),
            turning_points=_default_points(),
            params=EpisodeParams(),
        )
        assert snapshot.metadata["trade_date"] == "2024-01-02"
        assert snapshot.metadata["daily_rows"]["daily_row_derived"] is False
        assert snapshot.metadata["bar_index_in_today"] == 4

    def test_not_truncated_when_first_bar_exactly_21(self) -> None:
        snapshot = build_default()
        assert snapshot.metadata["today_window"]["truncated"] is False

    def test_truncated_when_night_start_not_covered(self) -> None:
        """帧首根 21:02 仍属今日窗口且晚于 21:00 → truncated（夜盘起点未覆盖）。"""
        truncated_frame = frame(
            list(NIGHT_ROWS[2:]), start="2024-01-01 21:02:00"
        )
        snapshot = build_live_state(
            symbol=SYMBOL,
            bars_1m=truncated_frame,
            decision_index=2,
            daily_rows=daily_frame(
                [("2024-01-01", DAILY_PREV), ("2024-01-02", DAILY_T)]
            ),
            turning_points=_default_points(),
            params=EpisodeParams(),
        )
        assert snapshot.metadata["today_window"]["truncated"] is True

    def test_not_truncated_when_window_start_inside_frame(self) -> None:
        """今日窗口首根不在帧首（前置 20:59 未入窗根）→ 起点已被覆盖，不截断。"""
        prefix = frame([(90, 91, 89, 90)], start="2024-01-01 20:59:00")
        frame_with_prefix = pd.concat(
            [prefix, frame(list(NIGHT_ROWS[:2]), start="2024-01-01 21:00:00")]
        ).reset_index(drop=True)
        snapshot = build_live_state(
            symbol=SYMBOL,
            bars_1m=frame_with_prefix,
            decision_index=2,
            daily_rows=daily_frame(
                [("2024-01-01", DAILY_PREV), ("2024-01-02", DAILY_T)]
            ),
            turning_points=_default_points(),
            params=EpisodeParams(),
        )
        assert snapshot.metadata["today_window"]["truncated"] is False
        assert snapshot.metadata["bar_index_in_today"] == 1

    def test_unassigned_time_hard_error(self) -> None:
        """T 在 15:00–20:59 → 不归属任何交易日，硬错误（不静默）。"""
        mixed = pd.concat(
            [
                frame([(100, 101, 99, 100)], start="2024-01-01 21:00:00"),
                frame([(101, 102, 100, 101)], start="2024-01-02 15:30:00"),
            ]
        ).reset_index(drop=True)
        with pytest.raises(DatasetError, match="15:00–20:59"):
            build_live_state(
                symbol=SYMBOL,
                bars_1m=mixed,
                decision_index=1,
                daily_rows=daily_frame([("2024-01-01", DAILY_PREV)]),
                turning_points=_default_points(),
                params=EpisodeParams(),
            )

    def test_decision_index_out_of_bounds(self) -> None:
        with pytest.raises(DatasetError, match="decision_index 越界"):
            build_live_state(
                symbol=SYMBOL,
                bars_1m=today_window_frame(),
                decision_index=10,
                daily_rows=daily_frame([("2024-01-01", DAILY_PREV)]),
                turning_points=_default_points(),
                params=EpisodeParams(),
            )


class TestLinkageAndParams:
    """联动品种 meta/na 降级（D4/E9）与参数透传。"""

    def test_linkage_same_timestamps_correlation_defined(self) -> None:
        """联动帧与主品种同时戳 → 突破/相关度可算，na_details 为空。"""
        params = EpisodeParams(linkage_symbols=("TEST.lnk",))
        snapshot = build_default(
            params=params,
            linkage_bars={"TEST.lnk": today_window_frame()},
        )
        meta = snapshot.metadata
        assert [entry["symbol"] for entry in meta["linkage_symbols"]] == ["TEST.lnk"]
        entry = meta["linkage_symbols"][0]
        assert entry["available"] is True
        assert entry["first_timestamp"] == "2024-01-01T21:00:00+08:00"
        assert meta["na_details"] == []
        linkage_line = snapshot.state_text.split(" \n ")[4]
        assert "lnk（突破=" in linkage_line
        assert "相关度=" in linkage_line

    def test_linkage_mismatched_timestamps_na(self) -> None:
        """联动帧时间戳与主品种无交集 → 突破/相关度 na + 成因入 na_details。"""
        params = EpisodeParams(linkage_symbols=("TEST.lnk",))
        shifted = frame(
            list(NIGHT_ROWS), start="2024-01-01 21:30:00"
        )
        snapshot = build_default(params=params, linkage_bars={"TEST.lnk": shifted})
        assert snapshot.metadata["na_details"] == [
            "lnk 突破=na：交集对 0 <2",
            "lnk 相关度=na：有效信号对 0 <2 或零方差",
        ]

    def test_linkage_missing_data_degrades_to_na(self) -> None:
        """联动品种缺数据（None）→ 空序列 na 降级 + available=false（不中断快照）。"""
        params = EpisodeParams(linkage_symbols=("TEST.lnk",))
        snapshot = build_default(params=params, linkage_bars={"TEST.lnk": None})
        entry = snapshot.metadata["linkage_symbols"][0]
        assert entry["available"] is False
        assert entry["first_timestamp"] is None
        assert entry["last_timestamp"] is None
        assert snapshot.metadata["na_details"] == [
            "lnk 突破=na：交集对 0 <2",
            "lnk 相关度=na：有效信号对 0 <2 或零方差",
        ]

    def test_breakthrough_window_short_na_detail(self) -> None:
        """1m 可用根数 <2（片段首根）→ 主突破 na + 成因（可用根数 1 <2）。

        ``breakthrough_momentum`` 的 na 条件 = ``min(window, 可用根数) <2``
        （window 不足用可用根数补齐，2 根即可产生 1 个信号），非「根数 < 窗口」。
        """
        one_bar = frame(list(NIGHT_ROWS[:1]), start="2024-01-01 21:00:00")
        snapshot = build_live_state(
            symbol=SYMBOL,
            bars_1m=one_bar,
            decision_index=0,
            daily_rows=daily_frame([("2024-01-01", DAILY_PREV)]),
            turning_points=_default_points(),
            params=EpisodeParams(),
        )
        assert snapshot.metadata["na_details"] == ["主突破=na：1m 可用根数 1 <2"]

    def test_params_passthrough_to_metadata(self) -> None:
        params = EpisodeParams(breakthrough_window=5, breakthrough_period="1m")
        snapshot = build_default(params=params)
        assert snapshot.metadata["breakthrough"] == {
            "window": 5,
            "period": "1m",
            "duration_seconds": 60,
        }
        assert snapshot.metadata["price_precision"] == 6


class _PerSymbolFakeApi:
    """按 ``(symbol, duration_seconds)`` 注入序列帧的 ``TqApi`` 最小桩。"""

    def __init__(self, frames: dict[tuple[str, int], pd.DataFrame]) -> None:
        self._frames = frames
        self.calls: list[dict[str, Any]] = []
        self.close_count = 0

    def get_kline_serial(
        self, *, symbol: str, duration_seconds: int, data_length: int
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "symbol": symbol,
                "duration_seconds": duration_seconds,
                "data_length": data_length,
            }
        )
        frame_ = self._frames.get((symbol, duration_seconds))
        if frame_ is None:
            raise DatasetError(f"桩无该品种/周期数据: {symbol} @{duration_seconds}")
        return frame_

    def wait_update(self, deadline: float | None = None) -> bool:
        return True

    def is_changing(self, obj: Any, key: Any = None) -> bool:
        return True

    def close(self) -> None:
        self.close_count += 1


def _serial_rows(
    ohlc_rows: list[tuple[float, float, float, float]],
    timestamps: list[str],
) -> list[tuple]:
    """``(开, 高, 低, 收)`` + 时间戳 → ``build_serial_klines`` 的 8 元组行。"""
    return [
        (
            ts,
            ohlc[0],
            ohlc[1],
            ohlc[2],
            ohlc[3],
            100,
            5000 + 10 * index,
            5010 + 10 * index,
        )
        for index, (ts, ohlc) in enumerate(zip(timestamps, ohlc_rows))
    ]


def _one_minute_timestamps() -> list[str]:
    stamps = [f"2024-01-01 21:0{minute}:00" for minute in range(5)]
    stamps += [f"2024-01-02 09:0{minute}:00" for minute in range(5)]
    return stamps


def _primary_serial_frames() -> dict[tuple[str, int], pd.DataFrame]:
    ohlc = [*NIGHT_ROWS, *DAY_ROWS]
    return {
        ("TEST.sym", 60): build_serial_klines(
            _serial_rows(list(ohlc), _one_minute_timestamps()), width=12
        ),
        ("TEST.sym", 86400): build_serial_klines(
            _serial_rows(
                [DAILY_PREV, DAILY_T],
                ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
            ),
            width=3,
        ),
    }


def _write_cli_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    """CLI 测试工作区：1m/1d CSV + 折点 CSV + episode 参数 YAML（均只写 tmp）。"""
    data_dir = tmp_path / "data" / "ohlcv"
    save_ohlcv(today_window_frame(), symbol=SYMBOL, period="1m", output_dir=data_dir)
    save_ohlcv(
        daily_frame([("2024-01-01", DAILY_PREV), ("2024-01-02", DAILY_T)]),
        symbol=SYMBOL,
        period="1d",
        output_dir=data_dir,
    )
    tp_dir = tmp_path / "data" / "turning_points"
    write_daily_turning_points(tp_dir / f"{SYMBOL}_1d.csv")
    episode_yaml = tmp_path / "episode.yaml"
    episode_yaml.write_text(
        yaml.safe_dump({"episode": {}}, allow_unicode=True), encoding="utf-8"
    )
    return data_dir, tp_dir, episode_yaml


def _parse_cli_stdout(out: str) -> tuple[str, dict[str, Any]]:
    """stdout → (state_text, metadata dict)（metadata JSON 块 = 末注行之前）。"""
    state = out.split("=== state ===\n", 1)[1].split("\n=== question ===", 1)[0]
    metadata_block = out.split("=== metadata ===\n", 1)[1]
    metadata_text = metadata_block.split("\n注：", 1)[0]
    return state, json.loads(metadata_text)


class TestCliOffline:
    """``state-now --offline``：合成数据 + 确定性 stdout。"""

    def test_offline_prints_snapshot(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data_dir, tp_dir, episode_yaml = _write_cli_workspace(tmp_path)
        code = main(
            [
                "state-now",
                "--symbol",
                SYMBOL,
                "--offline",
                "--once",
                "--full",
                "--data-dir",
                str(data_dir),
                "--turning-points-dir",
                str(tp_dir),
                "--episode-config",
                str(episode_yaml),
            ]
        )
        captured = capsys.readouterr()
        assert code == 0
        state, metadata = _parse_cli_stdout(captured.out)
        assert state.startswith(STATE_SCHEMA)
        assert metadata["data_source"] == "offline"
        assert metadata["symbol"] == SYMBOL
        assert metadata["trade_date"] == "2024-01-02"
        assert metadata["daily_rows"]["daily_row_derived"] is False
        assert metadata["denominator_note"] == DENOMINATOR_NOTE
        assert captured.out.rstrip("\n").endswith(f"注：{DENOMINATOR_NOTE}")
        # question / candidates 节
        assert "next_action: type=choice" in captured.out
        assert (
            "open_long=买入开仓 | open_short=卖出开仓 | stay_flat=继续空仓"
            in captured.out
        )

    def test_offline_once_preserves_complete_layout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """离线单次默认保留完整布局，schema/账户与问题元数据均在场。"""
        data_dir, tp_dir, episode_yaml = _write_cli_workspace(tmp_path)
        code = main(
            [
                "state-now",
                "--symbol",
                SYMBOL,
                "--offline",
                "--once",
                "--data-dir",
                str(data_dir),
                "--turning-points-dir",
                str(tp_dir),
                "--episode-config",
                str(episode_yaml),
            ]
        )
        captured = capsys.readouterr()
        assert code == 0
        params = EpisodeParams()
        loaded_1m = load_ohlcv(SYMBOL, "1m", data_dir=data_dir)
        loaded_1d = load_ohlcv(SYMBOL, "1d", data_dir=data_dir)
        loaded_tp = load_turning_points(SYMBOL, "1d", data_dir=tp_dir)
        snapshot = build_live_state(
            symbol=SYMBOL,
            bars_1m=loaded_1m.df,
            decision_index=len(loaded_1m.df) - 1,
            daily_rows=loaded_1d.df,
            turning_points=loaded_tp.points,
            linkage_bars=None,
            params=params,
            price_precision=params.price_precision,
        )
        assert captured.out.startswith("=== state ===\n")
        assert snapshot.state_text in captured.out
        assert "=== question ===" in captured.out
        assert "=== candidates ===" in captured.out
        assert "=== metadata ===" in captured.out
        assert "注：" in captured.out

    def test_offline_once_matches_default_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--once`` 保留完整旧布局、无头；--full 在 once 模式是 no-op。"""
        data_dir, tp_dir, episode_yaml = _write_cli_workspace(tmp_path)
        argv_base = [
            "state-now",
            "--symbol",
            SYMBOL,
            "--offline",
            "--once",
            "--data-dir",
            str(data_dir),
            "--turning-points-dir",
            str(tp_dir),
            "--episode-config",
            str(episode_yaml),
        ]
        assert main(argv_base) == 0
        default_out = capsys.readouterr().out
        assert main([*argv_base, "--once"]) == 0
        once_out = capsys.readouterr().out
        assert once_out == default_out
        assert once_out.startswith("=== state ===\n")
        assert "=== question ===" in once_out
        assert "=== candidates ===" in once_out
        assert "=== metadata ===" in once_out
        assert "注：" in once_out
        assert not once_out.startswith("[")  # --once 无时间戳头
        # --full 在 --once 模式为 no-op
        assert main([*argv_base, "--once", "--full"]) == 0
        once_full_out = capsys.readouterr().out
        assert main([*argv_base, "--full"]) == 0
        full_out = capsys.readouterr().out
        assert once_full_out == once_out == full_out

    def test_offline_full_prints_accepted_layout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--full`` 完整布局：schema/账户/question/candidates/metadata 全在场。"""
        data_dir, tp_dir, episode_yaml = _write_cli_workspace(tmp_path)
        code = main(
            [
                "state-now",
                "--symbol",
                SYMBOL,
                "--offline",
                "--once",
                "--full",
                "--data-dir",
                str(data_dir),
                "--turning-points-dir",
                str(tp_dir),
                "--episode-config",
                str(episode_yaml),
            ]
        )
        captured = capsys.readouterr()
        assert code == 0
        out = captured.out
        assert out.startswith("=== state ===\n")
        assert STATE_SCHEMA in out
        assert "账户" in out
        assert "=== question ===" in out
        assert "next_action: type=choice" in out
        assert "=== candidates ===" in out
        assert (
            "open_long=买入开仓 | open_short=卖出开仓 | stay_flat=继续空仓"
            in out
        )
        assert "=== metadata ===" in out
        assert out.rstrip("\n").endswith(f"注：{DENOMINATOR_NOTE}")

    def test_offline_linkage_missing_csv_degrades(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data_dir, tp_dir, _ = _write_cli_workspace(tmp_path)
        episode_yaml = tmp_path / "episode_link.yaml"
        episode_yaml.write_text(
            yaml.safe_dump(
                {"episode": {"linkage_symbols": ["TEST.lnk"]}}, allow_unicode=True
            ),
            encoding="utf-8",
        )
        code = main(
            [
                "state-now",
                "--symbol",
                SYMBOL,
                "--offline",
                "--once",
                "--full",
                "--data-dir",
                str(data_dir),
                "--turning-points-dir",
                str(tp_dir),
                "--episode-config",
                str(episode_yaml),
            ]
        )
        captured = capsys.readouterr()
        assert code == 0
        assert "警告：联动品种 TEST.lnk 离线读取失败，按 na 降级" in captured.err
        _, metadata = _parse_cli_stdout(captured.out)
        assert metadata["linkage_symbols"][0]["available"] is False
        assert metadata["na_details"]


class TestCliOnlineStub:
    """``state-now`` 在线桩：按品种注入帧（FakeTqApi 同模式的按品种扩展）。"""

    def test_online_snapshot_via_stub(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _isolate_config(tmp_path, monkeypatch)
        data_dir, tp_dir, episode_yaml = _write_cli_workspace(tmp_path)
        fake = _PerSymbolFakeApi(_primary_serial_frames())

        code = main(
            [
                "state-now",
                "--full",
                "--once",
                "--symbol",
                "TEST.sym",
                "--once",
                "--bars",
                "10",
                "--daily-bars",
                "2",
                "--turning-points-dir",
                str(tp_dir),
                "--episode-config",
                str(episode_yaml),
            ],
            api_factory=lambda _config: fake,
        )
        captured = capsys.readouterr()
        assert code == 0
        state, metadata = _parse_cli_stdout(captured.out)
        assert state.startswith(STATE_SCHEMA)
        assert metadata["data_source"] == "online"
        assert metadata["trade_date"] == "2024-01-02"
        assert metadata["decision_bar_timestamp"] == "2024-01-02T09:04:00+08:00"
        requested = {(call["symbol"], call["duration_seconds"]) for call in fake.calls}
        assert ("TEST.sym", 60) in requested
        assert ("TEST.sym", 86400) in requested
        assert fake.close_count == 1
        _ = data_dir  # 离线目录不参与在线路径

    @pytest.mark.parametrize("full", [False, True])
    def test_online_watch_emits_header_and_closes_on_interrupt(
        self,
        full: bool,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """watch CLI 首轮即输出；假等待 Ctrl+C 返回 130 并释放连接。"""
        _isolate_config(tmp_path, monkeypatch)
        _, tp_dir, episode_yaml = _write_cli_workspace(tmp_path)
        fake = _PerSymbolFakeApi(_primary_serial_frames())

        def interrupt_wait(_deadline: float) -> None:
            raise KeyboardInterrupt

        argv = [
            "state-now",
            "--symbol",
            "TEST.sym",
            "--bars",
            "10",
            "--daily-bars",
            "2",
            "--turning-points-dir",
            str(tp_dir),
            "--episode-config",
            str(episode_yaml),
        ]
        if full:
            argv.append("--full")
        code = main(
            argv,
            api_factory=lambda _config: fake,
            now_fn=lambda: pd.Timestamp("2024-01-02 09:04:30", tz="Asia/Shanghai"),
            wait_fn=interrupt_wait,
        )

        captured = capsys.readouterr()
        assert code == 130
        assert captured.out.startswith(
            "[2024-01-02 09:05 TEST.sym 1m]\n"
        )
        if full:
            assert "=== state ===" in captured.out
            assert "=== metadata ===" in captured.out
        else:
            lines = captured.out.splitlines()
            assert len(lines) == 6  # timestamp header + exactly five market lines
            assert [line.split(":", 1)[0] for line in lines[1:]] == [
                "日线",
                "日内",
                "联动",
                "现价",
                "盘口",
            ]
            assert "=== state ===" not in captured.out
        assert fake.close_count == 1

    def test_online_linkage_failure_degrades(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _isolate_config(tmp_path, monkeypatch)
        _, tp_dir, _ = _write_cli_workspace(tmp_path)
        episode_yaml = tmp_path / "episode_link.yaml"
        episode_yaml.write_text(
            yaml.safe_dump(
                {"episode": {"linkage_symbols": ["TEST.lnk"]}}, allow_unicode=True
            ),
            encoding="utf-8",
        )
        fake = _PerSymbolFakeApi(_primary_serial_frames())  # 无 TEST.lnk 帧 → 失败
        code = main(
            [
                "state-now",
                "--full",
                "--once",
                "--symbol",
                "TEST.sym",
                "--bars",
                "10",
                "--daily-bars",
                "2",
                "--turning-points-dir",
                str(tp_dir),
                "--episode-config",
                str(episode_yaml),
            ],
            api_factory=lambda _config: fake,
        )
        captured = capsys.readouterr()
        assert code == 0
        assert "警告：联动品种 TEST.lnk 在线取数失败，按 na 降级" in captured.err
        _, metadata = _parse_cli_stdout(captured.out)
        assert metadata["linkage_symbols"][0]["available"] is False
        assert metadata["na_details"]

    def test_online_primary_failure_hard_error(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _isolate_config(tmp_path, monkeypatch)
        _, tp_dir, _ = _write_cli_workspace(tmp_path)
        frames = _primary_serial_frames()
        del frames[("TEST.sym", 60)]  # 主品种 1m 失败 → 硬错误
        fake = _PerSymbolFakeApi(frames)
        code = main(
            [
                "state-now",
                "--once",
                "--symbol",
                "TEST.sym",
                "--bars",
                "10",
                "--daily-bars",
                "2",
                "--turning-points-dir",
                str(tp_dir),
            ],
            api_factory=lambda _config: fake,
        )
        captured = capsys.readouterr()
        assert code == 1
        assert "错误：" in captured.err


def _isolate_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """测试环境隔离：无本地凭证文件、无环境覆盖（沿用 test_provider 模式）。"""
    monkeypatch.delenv(config_module.ENV_ACCOUNT, raising=False)
    monkeypatch.delenv(config_module.ENV_PASSWORD, raising=False)
    monkeypatch.delenv(config_module.ENV_CONFIG, raising=False)
    monkeypatch.delenv(config_module.ENV_DATA_DIR, raising=False)
    monkeypatch.setattr(
        config_module, "DEFAULT_CONFIG_PATH", tmp_path / "missing.yaml"
    )


class TestCliValidation:
    """参数校验（先于配置加载/联网，失败退出码 2）。"""

    @pytest.mark.parametrize(
        "argv_tail",
        [
            ["--bars", "1"],
            ["--bars", "0"],
            ["--bars", "-5"],
            ["--bars", "8965"],
            ["--daily-bars", "0"],
            ["--daily-bars", "8965"],
        ],
    )
    def test_depth_bounds_rejected(self, argv_tail: list[str]) -> None:
        code = main(["state-now", "--symbol", SYMBOL, *argv_tail])
        assert code == 2

    def test_missing_symbol_rejected(self) -> None:
        assert main(["state-now"]) == 2


class TestCliCredentials:
    """凭证缺失：清晰报错且永不回显（E6/R6，沿用 test_provider 模式）。"""

    def test_missing_credentials_never_echo_password(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(config_module.ENV_ACCOUNT, raising=False)
        monkeypatch.delenv(config_module.ENV_CONFIG, raising=False)
        monkeypatch.delenv(config_module.ENV_DATA_DIR, raising=False)
        sentinel = "S3CRET-PASSWORD"
        monkeypatch.setenv(config_module.ENV_PASSWORD, sentinel)
        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH", tmp_path / "missing.yaml"
        )
        code = main(["state-now", "--symbol", SYMBOL, "--once"])
        captured = capsys.readouterr()
        assert code == 1
        # 既有 require_credentials 文案（config.py 冻结路径）：明确缺凭证且不含值
        assert "缺少天勤凭证" in captured.err
        assert "account" in captured.err
        assert sentinel not in captured.out
        assert sentinel not in captured.err


#: 本机真实落盘数据（离线真数据验证；缺文件 skip，不阻塞 CI/其它机器）
REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_1M = REPO_ROOT / "data" / "ohlcv" / "DCE.v2701_1m.csv"
REAL_1D = REPO_ROOT / "data" / "ohlcv" / "DCE.v2701_1d.csv"
REAL_TP = REPO_ROOT / "data" / "turning_points" / "DCE.v2701_1d.csv"
REAL_SC_1M = REPO_ROOT / "data" / "ohlcv" / "INE.sc2611_1m.csv"


@pytest.mark.skipif(
    not (REAL_1M.is_file() and REAL_1D.is_file() and REAL_TP.is_file()),
    reason="本机无 DCE.v2701 真实落盘数据（1m/1d/折点 CSV）",
)
class TestCliOfflineRealData:
    """AC3：离线真实落盘数据端到端（T = 1m CSV 末行）。"""

    def test_real_data_snapshot(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        episode_yaml = tmp_path / "episode.yaml"
        episode_yaml.write_text(
            yaml.safe_dump(
                {"episode": {"linkage_symbols": ["INE.sc2611"]}},
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        code = main(
            [
                "state-now",
                "--symbol",
                "DCE.v2701",
                "--offline",
                "--once",
                "--full",
                "--data-dir",
                str(REAL_1M.parent),
                "--turning-points-dir",
                str(REAL_TP.parent),
                "--episode-config",
                str(episode_yaml),
            ]
        )
        captured = capsys.readouterr()
        assert code == 0
        state, metadata = _parse_cli_stdout(captured.out)
        assert state.startswith(STATE_SCHEMA)
        assert metadata["data_source"] == "offline"
        assert metadata["symbol"] == "DCE.v2701"
        assert metadata["trade_date"] == "2026-09-30"
        assert metadata["today_window"]["bars"] > 0
        assert metadata["linkage_symbols"][0]["available"] is REAL_SC_1M.is_file()
        assert captured.out.rstrip("\n").endswith(f"注：{DENOMINATOR_NOTE}")

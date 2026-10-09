"""T2：联动行突破动量纯函数（linkage.py）+ T1：breakthrough 配置参数。

覆盖：三态突破信号 h/c 边界（严格比较）；动量加权锁定值（手算期望）；
首根/空序列 → None；窗口不足用可用根数；5m 重采样锁定值（聚合规则 +
未收满桶不可用 + 跨休市对齐）；breakthrough_window/period 校验与 YAML 透传。

防泄漏口径与确定性见 :mod:`dataset.market_episode.linkage` 模块 docstring。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dataset.account import Bar
from dataset.errors import ConfigError, DatasetError
from dataset.market_episode.linkage import (
    ResampledBar,
    align_bars_by_timestamp,
    breakthrough_momentum,
    breakthrough_signal,
    daily_linkage_correlation,
    linkage_breakthrough_momentum,
    pearson_correlation,
    resample_bars,
    signal_pairs_from_aligned,
)
from dataset.market_episode.segments import (
    EpisodeParams,
    load_episode_config,
    load_episode_params,
)
from dataset.tests.market_episode_fixtures import build_workspace

_ROWS = [(100, 101, 99, 100)] * 9


def _bar(
    index: int,
    timestamp: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 1,
    open_oi: int = 100,
    close_oi: int = 200,
) -> Bar:
    return Bar(
        index=index,
        timestamp=timestamp,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        open_oi=open_oi,
        close_oi=close_oi,
    )


def _minute_bars(
    timestamps: list[str],
    rows: list[tuple[float, float, float, float]],
    volumes: list[int] | None = None,
) -> tuple[Bar, ...]:
    """按 (open, high, low, close) 行构造同序 Bar 序列（时间戳与行一一对应）。"""
    return tuple(
        _bar(
            index=offset,
            timestamp=timestamp,
            open_=row[0],
            high=row[1],
            low=row[2],
            close=row[3],
            volume=1 if volumes is None else volumes[offset],
        )
        for offset, (timestamp, row) in enumerate(zip(timestamps, rows))
    )


def _night_timestamps(count: int) -> list[str]:
    base = pd.Timestamp("2024-01-02 21:00:00+08:00")
    return [
        (base + pd.Timedelta(minutes=offset)).isoformat() for offset in range(count)
    ]


#: 夜盘 21:00–21:10 的 11 根 1m 价路径（→ 3 个 5m 桶，桶 C 未收满）与逐根成交量
_NIGHT_ROWS: list[tuple[float, float, float, float]] = [
    (10, 10, 10, 10),
    (10, 10.5, 10, 10.5),
    (10.5, 11, 10.5, 11),
    (11, 11, 11, 11),
    (11, 11, 11, 11),
    (11, 12, 11, 12),
    (12, 12, 11.5, 11.5),
    (11.5, 11.5, 11.5, 11.5),
    (11.5, 11.5, 11.5, 11.5),
    (11.5, 11.5, 11.5, 11.5),
    (11.5, 11.5, 11.5, 11.5),
]
_NIGHT_VOLUMES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


# --------------------------------------------------------------------------- #
# 三态突破信号：h/c 边界（严格比较）
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "row, expected",
    [
        ((10, 11, 10, 11), 1),  # 高↑ 且 收↑ → +1（涨势突破）
        ((10, 11, 10, 9), 0),  # 高↑ 但 收↓（低不↓ → 不构成 -1）
        ((10, 11, 10, 10), 0),  # 高↑ 但 收=（收必须严格 >）
        ((10, 10, 10, 11), 0),  # 高= 且 收↑（高必须严格 >）
        ((9, 9, 9, 9), -1),  # 低↓ 且 收↓ → -1（跌势突破）
        ((10, 10, 9, 10), 0),  # 低↓ 但 收=
        ((10, 10, 10, 9), 0),  # 低= 且 收↓（低必须严格 <）
        ((10, 10, 10, 10), 0),  # 全等
    ],
)
def test_breakthrough_signal_three_states(row: tuple, expected: int) -> None:
    prev = _bar(0, "2024-01-02 21:00:00+08:00", 10, 10, 10, 10)
    cur = _bar(1, "2024-01-02 21:01:00+08:00", *row)
    assert breakthrough_signal(prev, cur) == expected


def test_breakthrough_signal_low_breakout_uses_dedicated_prev() -> None:
    # 专属 prev=(10,12,8,11)：低↓ 且 收↓ 的对称边界（高不↑ → 不触发 +1）
    prev = _bar(0, "2024-01-02 21:00:00+08:00", 10, 12, 8, 11)
    cur = _bar(1, "2024-01-02 21:01:00+08:00", 11, 11.5, 7.5, 10.5)
    assert breakthrough_signal(prev, cur) == -1


# --------------------------------------------------------------------------- #
# 动量：加权锁定值（手算）
# --------------------------------------------------------------------------- #


def _three_bars() -> tuple[Bar, ...]:
    """三根 1m：相邻对信号 = (+1, 0)。"""
    return _minute_bars(
        _night_timestamps(3),
        [(10, 10, 10, 10), (10, 11, 10, 11), (11, 11, 10, 11)],
    )


def test_momentum_weighted_locked_value() -> None:
    # 权重 1..n-1（最近 = n-1）：(1·1 + 2·0) / (1+2) = 1/3
    assert breakthrough_momentum(
        _three_bars(), window=20, duration_seconds=60
    ) == pytest.approx(1.0 / 3.0)


def test_momentum_symmetric_signals_locked_value() -> None:
    # 信号 = (-1, +1)：(1·(-1) + 2·(+1)) / 3 = 1/3
    bars = _minute_bars(
        _night_timestamps(3),
        [(10, 10, 10, 10), (9, 9, 9, 9), (10, 10, 9, 10)],
    )
    assert breakthrough_momentum(
        bars, window=20, duration_seconds=60
    ) == pytest.approx(1.0 / 3.0)


def test_momentum_four_bars_locked_value() -> None:
    # 信号 = (+1, 0, -1)：(1·1 + 2·0 + 3·(-1)) / 6 = -1/3
    bars = _minute_bars(
        _night_timestamps(4),
        [
            (10, 10, 10, 10),
            (10, 11, 10, 11),  # +1
            (11, 11, 10, 11),  # 0
            (10.5, 11, 9.5, 10),  # vs b2: 低↓ 收↓ → -1
        ],
    )
    assert breakthrough_momentum(
        bars, window=20, duration_seconds=60
    ) == pytest.approx(-1.0 / 3.0)


@pytest.mark.parametrize("window", [20, 2])
def test_momentum_returns_none_without_pair(window: int) -> None:
    single = _minute_bars(_night_timestamps(1), [(10, 10, 10, 10)])
    assert breakthrough_momentum((), window=window, duration_seconds=60) is None
    assert breakthrough_momentum(single, window=window, duration_seconds=60) is None


def test_momentum_window_one_truncates_to_single_bar() -> None:
    assert breakthrough_momentum(
        _three_bars(), window=1, duration_seconds=60
    ) is None


def test_momentum_insufficient_window_uses_available_bars() -> None:
    # 3 根 < window=20：用可用根数（拍板 A），结果与全窗口一致 = 1/3
    assert breakthrough_momentum(
        _three_bars(), window=20, duration_seconds=60
    ) == pytest.approx(1.0 / 3.0)
    # window=2 截掉最早一根 → 只剩 b1→b2 的单信号 = 0/1 = 0
    assert breakthrough_momentum(_three_bars(), window=2, duration_seconds=60) == 0.0


def test_momentum_rejects_invalid_window() -> None:
    with pytest.raises(DatasetError, match="window 必须 ≥ 1"):
        breakthrough_momentum(_three_bars(), window=0, duration_seconds=60)


# --------------------------------------------------------------------------- #
# 重采样：聚合规则 + 收满判定 + 跨休市
# --------------------------------------------------------------------------- #


def _night_bars(count: int = 11) -> tuple[Bar, ...]:
    return _minute_bars(
        _night_timestamps(count),
        _NIGHT_ROWS[:count],
        volumes=_NIGHT_VOLUMES[:count],
    )


def test_resample_bars_aggregation_and_closed_flags() -> None:
    resampled = resample_bars(_night_bars(), 300)
    assert len(resampled) == 3

    first = resampled[0]  # 21:00–21:05（21:00..21:04，下一根 21:05 ≥ 桶终点 → 收满）
    assert (first.open, first.high, first.low, first.close) == (10, 11, 10, 11)
    assert first.volume == 15
    assert (first.start_ts, first.end_ts) == (1704200400, 1704200700)
    assert first.is_closed is True

    second = resampled[1]  # 21:05–21:10（21:05..21:09，下一根 21:10 ≥ 桶终点 → 收满）
    assert (second.open, second.high, second.low, second.close) == (11, 12, 11, 11.5)
    assert second.volume == 40
    assert (second.start_ts, second.end_ts) == (1704200700, 1704201000)
    assert second.is_closed is True

    third = resampled[2]  # 21:10–21:15（仅 21:10 一根，无下一根 → 不收满）
    assert (third.open, third.high, third.low, third.close) == (11.5, 11.5, 11.5, 11.5)
    assert third.volume == 11
    assert (third.start_ts, third.end_ts) == (1704201000, 1704201300)
    assert third.is_closed is False


def test_resample_bars_oi_takes_first_and_last() -> None:
    bars = _night_bars()
    assert bars[0].open_oi == 100
    resampled = resample_bars(bars, 300)
    # 逐根持仓量可区分：桶 open_oi = 首根、close_oi = 末根（桶 A：index 0/4；B：5/9）
    assert (resampled[0].open_oi, resampled[0].close_oi) == (
        bars[0].open_oi,
        bars[4].close_oi,
    )
    assert (resampled[1].open_oi, resampled[1].close_oi) == (
        bars[5].open_oi,
        bars[9].close_oi,
    )
    assert (resampled[2].open_oi, resampled[2].close_oi) == (
        bars[10].open_oi,
        bars[10].close_oi,
    )


def test_resample_bars_gap_alignment_across_session_break() -> None:
    # 日盘尾（14:58/14:59）→ 夜盘（21:00/21:01）：休市空洞不拼接，各自对齐桶边界
    bars = _minute_bars(
        [
            "2024-01-02 14:58:00+08:00",
            "2024-01-02 14:59:00+08:00",
            "2024-01-02 21:00:00+08:00",
            "2024-01-02 21:01:00+08:00",
        ],
        [
            (10, 10.5, 10, 10.5),
            (10.5, 11, 10, 11),
            (9, 9.5, 9, 9.5),
            (9.5, 9.5, 9, 9),
        ],
        volumes=[3, 4, 1, 2],
    )
    resampled = resample_bars(bars, 300)
    assert len(resampled) == 2
    day_bucket, night_bucket = resampled
    # 日盘桶（14:55–15:00）被夜盘首根（21:00 ≥ 15:00）确认收满
    assert (day_bucket.start_ts, day_bucket.end_ts) == (1704178500, 1704178800)
    assert day_bucket.is_closed is True
    assert (day_bucket.open, day_bucket.high, day_bucket.low, day_bucket.close) == (
        10,
        11,
        10,
        11,
    )
    assert day_bucket.volume == 7
    # 夜盘桶从 21:00 重新对齐，且为片段尾桶 → 不收满
    assert (night_bucket.start_ts, night_bucket.end_ts) == (1704200400, 1704200700)
    assert night_bucket.is_closed is False


def test_resample_bars_rejects_nonpositive_duration() -> None:
    with pytest.raises(DatasetError, match="duration_seconds 必须 ≥ 1"):
        resample_bars(_three_bars(), 0)


def test_momentum_resampled_locked_value() -> None:
    bars = _night_bars()
    # 已收满桶 = A（收 11）、B（收 11.5）；桶 C 未收满不可用。
    # signal(A, B) = +1（高↑ 收↑）→ 单信号权重 1 → momentum = 1.0
    assert breakthrough_momentum(bars, window=20, duration_seconds=300) == 1.0


def test_momentum_resampled_partial_bucket_unavailable() -> None:
    # 21:00..21:07：桶 A 收满，桶 B（21:05..21:07）无下一根 → 未收满不可用 → 无相邻对
    bars = _night_bars(8)
    resampled = resample_bars(bars, 300)
    assert [bucket.is_closed for bucket in resampled] == [True, False]
    assert breakthrough_momentum(bars, window=20, duration_seconds=300) is None


def test_momentum_resampled_window_truncates_to_one_bucket() -> None:
    # window=1 → 只取末 1 个已收满桶 → 无相邻对 → None
    assert breakthrough_momentum(_night_bars(), window=1, duration_seconds=300) is None


def test_daily_linkage_correlation_intersection_cutoff_window_and_na() -> None:
    from datetime import date, timedelta

    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(6)]
    values = (10.0, 11.0, 99.0, 10.0, 11.0, 999.0)
    primary = tuple(
        (day, values[index], values[index] - 1.0, values[index])
        for index, day in enumerate(dates)
    )
    secondary = tuple(
        (day, values[index], values[index] - 1.0, values[index])
        for index, day in enumerate(dates)
        if index != 2
    )
    assert daily_linkage_correlation(primary, secondary, dates[5], window=4) == 1.0
    assert daily_linkage_correlation(primary, secondary, dates[5], window=2) is None
    assert daily_linkage_correlation(primary, secondary, dates[3], window=20) is None

    with pytest.raises(DatasetError, match="window 必须 ≥ 1"):
        daily_linkage_correlation(primary, secondary, dates[5], window=0)


def test_daily_linkage_correlation_zero_variance_is_undefined() -> None:
    from datetime import date, timedelta

    dates = [date(2024, 2, 1) + timedelta(days=index) for index in range(4)]
    primary = tuple((day, 10.0 + index, 9.0, 10.0 + index) for index, day in enumerate(dates))
    secondary = tuple((day, 20.0, 20.0, 20.0) for day in dates)
    assert daily_linkage_correlation(primary, secondary, dates[-1], window=4) is None


# --------------------------------------------------------------------------- #
# T1：EpisodeParams 配置校验 + YAML 透传
# --------------------------------------------------------------------------- #


class TestBreakthroughParams:
    def test_defaults_and_as_dict(self) -> None:
        params = EpisodeParams()
        assert params.breakthrough_window == 20
        assert params.breakthrough_period == "1m"
        exported = params.as_dict()
        assert exported["breakthrough_window"] == 20
        assert exported["breakthrough_period"] == "1m"

    def test_window_zero_rejected(self) -> None:
        with pytest.raises(DatasetError, match="breakthrough_window 必须 ≥ 1"):
            load_episode_params({"breakthrough_window": 0}, where="测试配置段")

    @pytest.mark.parametrize("period", ["1d", "2m"])
    def test_period_rejected_with_legal_list(self, period: str) -> None:
        with pytest.raises(DatasetError, match="1m, 5m, 15m, 1h"):
            load_episode_params({"breakthrough_period": period}, where="测试配置段")

    def test_valid_override(self) -> None:
        params = load_episode_params(
            {"breakthrough_window": 5, "breakthrough_period": "15m"},
            where="测试配置段",
        )
        assert params.breakthrough_window == 5
        assert params.breakthrough_period == "15m"

    def test_yaml_passthrough_via_load_episode_config(self, tmp_path: Path) -> None:
        workspace = build_workspace(
            tmp_path,
            _ROWS,
            episode={"breakthrough_window": 5, "breakthrough_period": "5m"},
        )
        config = load_episode_config(workspace.config_path)
        assert config.params.breakthrough_window == 5
        assert config.params.breakthrough_period == "5m"

    def test_yaml_defaults_when_absent(self, tmp_path: Path) -> None:
        workspace = build_workspace(tmp_path, _ROWS)
        config = load_episode_config(workspace.config_path)
        assert config.params.breakthrough_window == 20
        assert config.params.breakthrough_period == "1m"

    def test_invalid_yaml_period_propagates(self, tmp_path: Path) -> None:
        workspace = build_workspace(
            tmp_path, _ROWS, episode={"breakthrough_period": "1d"}
        )
        with pytest.raises(ConfigError, match="1m, 5m, 15m, 1h"):
            load_episode_config(workspace.config_path)


# --------------------------------------------------------------------------- #
# T3：时间戳交集对齐 + 联动突破值 + 皮尔逊相关度 + 对照信号对
# --------------------------------------------------------------------------- #

_BASE_TS = pd.Timestamp("2024-01-02 21:00:00+08:00")


def _ts(offset: int) -> str:
    return (_BASE_TS + pd.Timedelta(minutes=offset)).isoformat()


#: 主品种 6 根价路径：全窗口相邻对信号 = [+1, 0, -1, 0, +1]
_PRIMARY_ROWS: list[tuple[float, float, float, float]] = [
    (10, 10, 10, 10),  # t0 (21:00)
    (10, 11, 10, 11),  # t1 → sig(t0,t1)=+1（高↑ 收↑）
    (11, 11, 11, 11),  # t2 → sig(t1,t2)=0（高=）
    (11, 12, 10.5, 10.5),  # t3 → sig(t2,t3)=-1（低↓ 收↓；高↑ 但 收↓）
    (10.5, 10.5, 10.5, 10.5),  # t4 → sig(t3,t4)=0（低=）
    (10.5, 11.5, 10.5, 11.5),  # t5 → sig(t4,t5)=+1
]

#: 联动品种：缺 t2 分钟 + 独有 t9 分钟（永不入选）；交集序列 t0,t1,t3,t4,t5
_SECONDARY_ROWS: list[tuple[float, float, float, float]] = [
    (20, 20, 20, 20),  # t0
    (20, 21, 20, 21),  # t1 → sig(t0,t1)=+1
    (20.5, 21, 19.5, 19.5),  # t3 → sig(t1,t3)=-1（低 19.5<20 且收 19.5<21）
    (19.5, 19.5, 19.5, 19.5),  # t4 → sig(t3,t4)=0（低=）
    (19.5, 20.5, 19.5, 20.5),  # t5 → sig(t4,t5)=+1
    (20.5, 21, 20.5, 21),  # t9（secondary 独有分钟）
]


def _aligned_bars(window: int = 6) -> tuple[tuple[Bar, Bar], ...]:
    primary = _minute_bars([_ts(offset) for offset in range(6)], _PRIMARY_ROWS)
    secondary = _minute_bars([_ts(offset) for offset in (0, 1, 3, 4, 5, 9)], _SECONDARY_ROWS)
    return align_bars_by_timestamp(primary, secondary, window)


def test_primary_rows_signal_baseline() -> None:
    """主品种价路径自检：全窗口动量 = (1·1+2·0+3·(−1)+4·0+5·1)/15 = 0.2。"""
    bars = _minute_bars([_ts(offset) for offset in range(6)], _PRIMARY_ROWS)
    assert breakthrough_momentum(bars, window=20, duration_seconds=60) == pytest.approx(0.2)


def test_align_keeps_intersection_in_order() -> None:
    aligned = _aligned_bars()
    assert len(aligned) == 5  # 缺 t2 被剔除；secondary 独有 t9 永不入选
    for primary, secondary in aligned:
        assert primary.timestamp == secondary.timestamp
    assert [pair[0].timestamp for pair in aligned] == [_ts(offset) for offset in (0, 1, 3, 4, 5)]


def test_align_window_truncates_to_primary_tail() -> None:
    aligned = _aligned_bars(window=3)
    assert [pair[0].timestamp for pair in aligned] == [_ts(offset) for offset in (3, 4, 5)]
    assert all(p.timestamp == s.timestamp for p, s in aligned)

    single = _aligned_bars(window=2)
    assert [pair[0].timestamp for pair in single] == [_ts(offset) for offset in (4, 5)]


def test_align_empty_primary_returns_empty() -> None:
    assert align_bars_by_timestamp([], [], window=5) == ()


def test_align_rejects_invalid_window() -> None:
    bars = _minute_bars([_ts(0)], [(10, 10, 10, 10)])
    with pytest.raises(DatasetError, match="window 必须 ≥ 1"):
        align_bars_by_timestamp(bars, bars, window=0)


def test_linkage_momentum_locked_value() -> None:
    """3 对 → 2 个信号手算：sig(s0,s1)=+1、sig(s1,s2)=−1 → (1·1+2·(−1))/3 = −1/3。"""
    times = [_ts(offset) for offset in (0, 1, 2)]
    secondary = _minute_bars(
        times, [(10, 10, 10, 10), (10, 11, 10, 11), (10, 10.8, 9.5, 9.5)]
    )
    primary = _minute_bars(times, [(1, 1, 1, 1)] * 3)
    aligned = tuple(zip(primary, secondary))
    assert linkage_breakthrough_momentum(aligned) == pytest.approx(-1 / 3)


def test_linkage_momentum_on_default_intersection() -> None:
    """默认交集（5 对 → 4 信号）：(1·1+2·(−1)+3·0+4·1)/10 = 0.3。"""
    assert linkage_breakthrough_momentum(_aligned_bars()) == pytest.approx(0.3)


@pytest.mark.parametrize("aligned", [(), _aligned_bars(window=1)], ids=["empty", "single"])
def test_linkage_momentum_returns_none_without_pair(aligned: tuple) -> None:
    assert linkage_breakthrough_momentum(aligned) is None


def test_signal_pairs_locked_value_and_intersection_consistency() -> None:
    pairs = signal_pairs_from_aligned(_aligned_bars())
    # 主品种侧 (t1,t3) 跨过被剔除的 t2 → 交集口径为 0（全窗口序列在同位置邻对为 −1）
    assert pairs == ((1, 1), (0, -1), (0, 0), (1, 1))
    # 与主品种独立信号一致性：primary 侧信号 = 交集序列上直接计算的相邻对信号
    primary_intersection = [p for p, _ in _aligned_bars()]
    independent = [
        breakthrough_signal(primary_intersection[i - 1], primary_intersection[i])
        for i in range(1, len(primary_intersection))
    ]
    assert [p for p, _ in pairs] == independent
    assert signal_pairs_from_aligned(_aligned_bars(window=1)) == ()
    assert signal_pairs_from_aligned(()) == ()


def test_pearson_locked_values() -> None:
    assert pearson_correlation([1, 0, -1], [1, 0, -1]) == pytest.approx(1.0)
    assert pearson_correlation([1, 0, -1], [-1, 0, 1]) == pytest.approx(-1.0)
    assert pearson_correlation([1, -1, 1, -1], [1, 1, -1, -1]) == pytest.approx(0.0)
    # 交集实例：primary [1,0,0,1] vs secondary [1,-1,0,1] → r = 1.5/√2.75
    aligned = _aligned_bars()
    xs = [p for p, _ in signal_pairs_from_aligned(aligned)]
    ys = [s for _, s in signal_pairs_from_aligned(aligned)]
    assert pearson_correlation(xs, ys) == pytest.approx(0.904534, abs=1e-6)


def test_pearson_undefined_returns_none() -> None:
    assert pearson_correlation([], []) is None  # 无相邻信号对
    assert pearson_correlation([1], [2]) is None  # 单元素
    assert pearson_correlation([5, 5, 5], [1, 2, 3]) is None  # x 零方差
    assert pearson_correlation([1, 2, 3], [7, 7, 7]) is None  # y 零方差


def test_pearson_length_mismatch_rejected() -> None:
    with pytest.raises(DatasetError, match="长度必须一致"):
        pearson_correlation([1, 2], [1])


# --------------------------------------------------------------------------- #
# T2：linkage_symbols 配置校验 + YAML 透传
# --------------------------------------------------------------------------- #


class TestLinkageSymbolsParams:
    def test_default_empty_tuple_and_as_dict_list(self) -> None:
        params = EpisodeParams()
        assert params.linkage_symbols == ()
        assert params.as_dict()["linkage_symbols"] == []

    def test_valid_override_preserves_order(self) -> None:
        params = load_episode_params(
            {"linkage_symbols": ["INE.sc2611"]}, where="测试配置段"
        )
        assert params.linkage_symbols == ("INE.sc2611",)
        assert params.as_dict()["linkage_symbols"] == ["INE.sc2611"]

    @pytest.mark.parametrize(
        "raw, message",
        [
            ("INE.sc2611", "必须为字符串列表"),  # 非列表（裸字符串）
            (123, "必须为字符串列表"),  # 非列表
            (["INEsc2611"], "交易所.合约"),  # 无点
            (["INE.sc.2611"], "交易所.合约"),  # 多点
            ([""], "交易所.合约"),  # 空段
            (["INE."], "交易所.合约"),  # 合约端空
            ([".sc2611"], "交易所.合约"),  # 交易所端空
            (["INE.sc2611", 123], "必须为字符串"),  # 元素非字符串
            (["INE.sc2611", "DCE.v2701"], "允许 0 或 1"),  # 多项拒绝
        ],
    )
    def test_invalid_rejected(self, raw: object, message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            load_episode_params({"linkage_symbols": raw}, where="测试配置段")

    def test_duplicate_symbol_rejected(self) -> None:
        with pytest.raises(ConfigError, match="linkage_symbols.*重复元素"):
            load_episode_params(
                {"linkage_symbols": ["INE.sc2611", "INE.sc2611"]},
                where="测试配置段",
            )

    def test_yaml_passthrough(self, tmp_path: Path) -> None:
        workspace = build_workspace(
            tmp_path, _ROWS, episode={"linkage_symbols": ["INE.sc2611"]}
        )
        config = load_episode_config(workspace.config_path)
        assert config.params.linkage_symbols == ("INE.sc2611",)

    def test_yaml_rejects_multiple_linkage_symbols(self, tmp_path: Path) -> None:
        workspace = build_workspace(
            tmp_path, _ROWS, episode={"linkage_symbols": ["INE.sc2611", "DCE.v2701"]}
        )
        with pytest.raises(ConfigError, match="linkage_symbols.*允许 0 或 1"):
            load_episode_config(workspace.config_path)

    def test_yaml_defaults_when_absent(self, tmp_path: Path) -> None:
        workspace = build_workspace(tmp_path, _ROWS)
        config = load_episode_config(workspace.config_path)
        assert config.params.linkage_symbols == ()

"""转折点规则（含 reference 8 条向量）、落盘与读回复现（AC-6、AC-7）。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dataset.errors import DataLoadError, DatasetError
from dataset.tests.conftest import TURNING_POINT_VECTORS, build_ohlcv
from dataset.errors import DatasetError
from dataset.turning_points import (
    INITIAL_DIRECTION_MODES,
    RELATIVE_COLUMNS,
    TURNING_POINT_COLUMNS,
    TURNING_POINT_KINDS,
    RelativeMetrics,
    TurningPoint,
    WindowMeta,
    build_window_meta,
    find_turning_points,
    load_turning_points,
    relative_metrics,
    resolve_initial_direction,
    save_turning_points,
)


def _kinds(points) -> list[str]:
    return [p.kind for p in points]


@pytest.mark.parametrize("vector", TURNING_POINT_VECTORS, ids=lambda v: v["name"])
def test_turning_point_vectors_match_fixed_expectations(vector: dict) -> None:
    df = build_ohlcv(vector["rows"])
    points = find_turning_points(df, initial_direction=vector["initial_direction"])

    assert _kinds(points) == vector["expect_kinds"]
    assert [p.price for p in points] == vector["expect_prices"]
    assert [p.bar_index for p in points] == vector["expect_bar_index"]
    # 时间戳必须落在 bar_index（锚点）所指 K 线上
    for point in points:
        assert point.timestamp == df["timestamp"].iloc[point.bar_index]
    # 增补信息：volume/oi 取自该点 bar_index 所指（锚点）K 线（默认夹具逐根递增）
    assert [p.volume for p in points] == [100] * len(points)
    assert [p.oi for p in points] == [
        5000 + 10 * (i + 1) for i in vector["expect_bar_index"]
    ]


def test_turning_point_kinds_contract() -> None:
    """kind 集合契约（AC-1）：2026-09-25 起为 start/up/down/close。"""
    assert TURNING_POINT_KINDS == ("start", "up", "down", "close")


def test_up_then_down_extreme_includes_turning_bar() -> None:
    rows = [
        (100, 101, 99, 100),
        (100, 110, 100, 109),
        (109, 120, 105, 119),
        (119, 121, 104, 110),  # low 104 < 105 转向；同根 high 121 创新高 → 极值在转向根
        (110, 112, 100, 101),
        (101, 113, 99, 112),  # high 113 > 112 转向；同根 low 99 创新低 → 极值在转向根
    ]
    df = build_ohlcv(rows)
    points = find_turning_points(df, initial_direction="up")

    # up 点：极值高在 bar 3（转向根参与极值搜索）→ 锚定其前一根 bar 2（D1）
    assert points[1].kind == "up"
    assert points[1].bar_index == 2
    assert points[1].timestamp == df["timestamp"].iloc[2]
    assert points[1].price == df["high"].iloc[2]
    # down 点：极值低在 bar 5 → 锚定 bar 4
    assert points[2].kind == "down"
    assert points[2].bar_index == 4
    assert points[2].timestamp == df["timestamp"].iloc[4]
    assert points[2].price == df["low"].iloc[4]


def test_bar0_extreme_down_point_stays_at_bar0() -> None:
    """AC-3 bar 0 边界（D2）：初始 down、第 1 根向上突破（high[1] > high[0]）→
    极值低在 bar 0（low[1] 未创新低），down 点整点留在 bar 0，price 取 bar 0 自身极值价。"""
    df = build_ohlcv([(100, 100, 90, 95), (95, 105, 90, 100)])
    points = find_turning_points(df, initial_direction="down")

    assert _kinds(points) == ["start", "down", "close"]
    down = points[1]
    assert down.bar_index == 0
    assert down.price == df["low"].iloc[0] == 90.0
    assert down.timestamp == df["timestamp"].iloc[0]
    assert down.volume == df["volume"].iloc[0]
    assert down.oi == df["close_oi"].iloc[0]


def test_bar0_extreme_up_point_stays_at_bar0() -> None:
    """AC-3 bar 0 边界（D2）：初始 up、第 1 根向下跌破（low[1] < low[0]）→
    极值高在 bar 0（high[1] 未创新高，并列保留最早），up 点整点留在 bar 0。"""
    df = build_ohlcv([(100, 100, 90, 95), (95, 100, 85, 90)])
    points = find_turning_points(df, initial_direction="up")

    assert _kinds(points) == ["start", "up", "close"]
    up = points[1]
    assert up.bar_index == 0
    assert up.price == df["high"].iloc[0] == 100.0
    assert up.timestamp == df["timestamp"].iloc[0]
    assert up.volume == df["volume"].iloc[0]
    assert up.oi == df["close_oi"].iloc[0]


def test_resolve_initial_direction_auto_rules() -> None:
    # 第二根 low 更低 → 起始下跌
    down = build_ohlcv([(100, 101, 99, 100), (100, 101, 98, 99)])
    assert resolve_initial_direction(down, "auto") == "down"

    # 第二根 high 更高且 low 未更低 → 起始上涨
    up = build_ohlcv([(100, 101, 99, 100), (100, 102, 99, 101)])
    assert resolve_initial_direction(up, "auto") == "up"

    # 两者都不成立 → 退回首根实体方向（close >= open 为上涨）
    flat_up = build_ohlcv([(100, 101, 99, 101), (100, 101, 99, 100)])
    assert resolve_initial_direction(flat_up, "auto") == "up"
    flat_down = build_ohlcv([(101, 101, 99, 100), (100, 101, 99, 100)])
    assert resolve_initial_direction(flat_down, "auto") == "down"

    # 强制模式（单根窗口也适用）
    single = build_ohlcv([(100, 101, 99, 100)])
    assert resolve_initial_direction(single, "up") == "up"
    assert resolve_initial_direction(single, "down") == "down"


def test_invalid_initial_direction_mode_raises() -> None:
    df = build_ohlcv([(100, 101, 99, 100)])
    with pytest.raises(ValueError, match="initial_direction"):
        resolve_initial_direction(df, "sideways")
    assert INITIAL_DIRECTION_MODES == ("auto", "up", "down")


def test_missing_columns_and_empty_frame_raise() -> None:
    df = build_ohlcv([(100, 101, 99, 100)]).drop(columns=["high"])
    with pytest.raises(ValueError, match="缺少列"):
        find_turning_points(df)
    with pytest.raises(ValueError, match="为空"):
        find_turning_points(build_ohlcv([]))


def test_missing_volume_or_close_oi_column_raises() -> None:
    """增补信息依赖 volume 与 close_oi（标准契约固定包含），缺失即明确报错。"""
    with pytest.raises(ValueError, match="缺少列"):
        find_turning_points(build_ohlcv([(100, 101, 99, 100)]).drop(columns=["volume"]))
    with pytest.raises(ValueError, match="缺少列"):
        find_turning_points(build_ohlcv([(100, 101, 99, 100)]).drop(columns=["close_oi"]))


def test_save_turning_points_writes_csv_columns_and_sidecar(tmp_path: Path) -> None:
    df = build_ohlcv(TURNING_POINT_VECTORS[0]["rows"])
    points = find_turning_points(df, initial_direction="up")
    resolved = resolve_initial_direction(df, "auto")
    meta = build_window_meta(
        df,
        initial_direction_requested="auto",
        initial_direction_resolved=resolved,
        input_source_data_version="tianqin|DCE.v2701|1m|-|-|rows=6|a|b|sha256=deadbeef",
    )

    path = save_turning_points(
        points,
        symbol="DCE.v2701",
        period="1m",
        output_dir=tmp_path,
        window_meta=meta,
    )

    assert path == tmp_path / "DCE.v2701_1m.csv"
    frame = pd.read_csv(path)
    assert list(frame.columns) == list(TURNING_POINT_COLUMNS)
    assert frame["point_index"].tolist() == list(range(len(points)))
    assert frame["kind"].tolist() == _kinds(points)
    assert frame["price"].tolist() == [p.price for p in points]
    assert frame["bar_index"].tolist() == [p.bar_index for p in points]
    assert frame["volume"].tolist() == [p.volume for p in points]
    assert frame["oi"].tolist() == [p.oi for p in points]

    # 相对值：首点四个字段为空单元格；其余与 relative_metrics 逐值一致
    for name in RELATIVE_COLUMNS:
        assert bool(frame[name].isna().iloc[0])
    expected = relative_metrics(points)
    for row, metric in zip(
        list(frame.itertuples(index=False))[1:], expected[1:], strict=True
    ):
        assert row.dt_minutes == pytest.approx(metric.dt_minutes)
        assert row.price_ratio == pytest.approx(metric.price_ratio)
        assert row.volume_ratio == pytest.approx(metric.volume_ratio)
        assert row.oi_ratio == pytest.approx(metric.oi_ratio)

    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(sidecar) == {
        "symbol",
        "period",
        "row_count",
        "window_first_timestamp",
        "window_last_timestamp",
        "initial_direction_requested",
        "initial_direction_resolved",
        "input_source_data_version",
        "file_sha256",
    }
    assert sidecar["row_count"] == len(df)
    assert sidecar["initial_direction_requested"] == "auto"
    assert sidecar["initial_direction_resolved"] in ("up", "down")
    assert sidecar["window_first_timestamp"] == str(df["timestamp"].iloc[0])
    assert sidecar["window_last_timestamp"] == str(df["timestamp"].iloc[-1])


def test_save_turning_points_reports_unusable_output_dir(tmp_path: Path) -> None:
    """P2-N2 回归：转折点落盘目录不可创建 → :class:`DatasetError`（不抛裸 ``OSError``）。"""
    df = build_ohlcv(TURNING_POINT_VECTORS[0]["rows"])
    points = find_turning_points(df, initial_direction="up")
    meta = build_window_meta(
        df,
        initial_direction_requested="up",
        initial_direction_resolved="up",
        input_source_data_version=None,
    )
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    with pytest.raises(DatasetError, match="输出目录不可用") as excinfo:
        save_turning_points(
            points,
            symbol="DCE.v2701",
            period="1m",
            output_dir=blocker / "sub",
            window_meta=meta,
        )

    assert "NotADirectoryError" in str(excinfo.value)
    assert str(blocker / "sub") in str(excinfo.value)


def test_turning_points_output_is_deterministic(tmp_path: Path) -> None:
    df = build_ohlcv(TURNING_POINT_VECTORS[0]["rows"])
    points = find_turning_points(df, initial_direction="up")
    meta = build_window_meta(
        df,
        initial_direction_requested="up",
        initial_direction_resolved="up",
        input_source_data_version=None,
    )

    first = save_turning_points(
        points, symbol="DCE.v2701", period="1m", output_dir=tmp_path / "a", window_meta=meta
    )
    second = save_turning_points(
        points, symbol="DCE.v2701", period="1m", output_dir=tmp_path / "b", window_meta=meta
    )

    assert first.read_bytes() == second.read_bytes()
    assert first.with_suffix(".json").read_bytes() == second.with_suffix(".json").read_bytes()


def test_turning_points_round_trip(tmp_path: Path) -> None:
    df = build_ohlcv(TURNING_POINT_VECTORS[0]["rows"])
    points = find_turning_points(df, initial_direction="up")
    meta = build_window_meta(
        df,
        initial_direction_requested="up",
        initial_direction_resolved="up",
        input_source_data_version="fingerprint",
    )
    save_turning_points(
        points, symbol="DCE.v2701", period="1m", output_dir=tmp_path, window_meta=meta
    )

    loaded = load_turning_points("DCE.v2701", "1m", data_dir=tmp_path)

    assert [p.kind for p in loaded.points] == _kinds(points)
    assert [p.price for p in loaded.points] == [p.price for p in points]
    assert [p.bar_index for p in loaded.points] == [p.bar_index for p in points]
    assert [p.volume for p in loaded.points] == [p.volume for p in points]
    assert [p.oi for p in loaded.points] == [p.oi for p in points]
    assert [str(p.timestamp) for p in loaded.points] == [
        str(p.timestamp) for p in points
    ]
    # 相对值读回与确定性派生完全一致
    assert loaded.relative == relative_metrics(points)
    assert loaded.meta["input_source_data_version"] == "fingerprint"
    assert loaded.meta["symbol"] == "DCE.v2701"


def test_save_turning_points_rejects_empty_sequence(tmp_path: Path) -> None:
    df = build_ohlcv([(100, 101, 99, 100)])
    meta = build_window_meta(
        df, initial_direction_requested="auto", initial_direction_resolved="up"
    )
    with pytest.raises(DatasetError, match="拒绝落盘空转折点序列"):
        save_turning_points(
            [], symbol="DCE.v2701", period="1m", output_dir=tmp_path, window_meta=meta
        )


def test_build_window_meta_rejects_empty_frame() -> None:
    with pytest.raises(DatasetError, match="无法构造转折点窗口元信息"):
        build_window_meta(
            build_ohlcv([]),
            initial_direction_requested="auto",
            initial_direction_resolved="up",
        )


def test_load_turning_points_wraps_read_failures(tmp_path: Path) -> None:
    """P2-1：空文件/异编码不符读取失败收敛为 DataLoadError（含文件路径）。"""
    path = tmp_path / "DCE.v2701_1m.csv"

    path.write_text("", encoding="utf-8")
    with pytest.raises(DataLoadError) as excinfo:
        load_turning_points("DCE.v2701", "1m", data_dir=tmp_path)
    assert "文件读取失败" in str(excinfo.value)
    assert str(path) in str(excinfo.value)

    path.write_bytes(b"\xff\xfe\x00\x01binary")
    with pytest.raises(DataLoadError) as excinfo:
        load_turning_points("DCE.v2701", "1m", data_dir=tmp_path)
    assert str(path) in str(excinfo.value)


def test_save_turning_points_is_atomic(tmp_path: Path) -> None:
    """P2-2：转折点 CSV 与 sidecar 同样先写临时文件再提交，无残留。"""
    df = build_ohlcv(TURNING_POINT_VECTORS[0]["rows"])
    save_turning_points(
        find_turning_points(df, initial_direction="up"),
        symbol="DCE.v2701",
        period="1m",
        output_dir=tmp_path,
        window_meta=build_window_meta(
            df,
            initial_direction_requested="auto",
            initial_direction_resolved=resolve_initial_direction(df, "auto"),
            input_source_data_version=None,
        ),
    )

    assert list(tmp_path.glob(".*.tmp")) == []
    assert (tmp_path / "DCE.v2701_1m.csv").is_file()
    assert (tmp_path / "DCE.v2701_1m.json").is_file()


def test_load_turning_points_errors(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="不存在"):
        load_turning_points("DCE.v2701", "1m", data_dir=tmp_path)

    path = tmp_path / "DCE.v2701_1m.csv"
    path.write_text("point_index,kind,timestamp\n0,start,2026-09-23 09:00:00+08:00\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="缺少必需列"):
        load_turning_points("DCE.v2701", "1m", data_dir=tmp_path)

    path.write_text(
        f"{','.join(TURNING_POINT_COLUMNS)}\n"
        "0,open,2026-09-23 09:00:00+08:00,100,0,100,5000,1.0,1.0,1.0,1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(DataLoadError, match="kind 取值非法"):
        load_turning_points("DCE.v2701", "1m", data_dir=tmp_path)

    # 2026-09-25 起旧 kind 不再合法（AC-1）：load 必须拒绝 high/low
    for legacy_kind in ("high", "low"):
        path.write_text(
            f"{','.join(TURNING_POINT_COLUMNS)}\n"
            f"0,{legacy_kind},2026-09-23 09:00:00+08:00,100,0,100,5000,1.0,1.0,1.0,1.0\n",
            encoding="utf-8",
        )
        with pytest.raises(DataLoadError, match="kind 取值非法"):
            load_turning_points("DCE.v2701", "1m", data_dir=tmp_path)

    path.write_text(
        f"{','.join(TURNING_POINT_COLUMNS)}\n"
        "0,start,not-a-time,100,0,100,5000,,,,\n",
        encoding="utf-8",
    )
    with pytest.raises(DataLoadError, match="无法解析为时间"):
        load_turning_points("DCE.v2701", "1m", data_dir=tmp_path)

    header = ",".join(TURNING_POINT_COLUMNS)
    path.write_text(
        f"{header}\n0,start,2026-09-23 09:00:00+08:00,100,0,,5000,,,,\n",
        encoding="utf-8",
    )
    with pytest.raises(DataLoadError, match="volume/oi 存在缺失"):
        load_turning_points("DCE.v2701", "1m", data_dir=tmp_path)


def test_window_meta_dataclass_is_frozen() -> None:
    meta = WindowMeta(
        initial_direction_requested="auto",
        initial_direction_resolved="up",
        row_count=3,
        window_first_timestamp="a",
        window_last_timestamp="b",
    )
    with pytest.raises(Exception):
        meta.row_count = 4  # type: ignore[misc]


# ---- 相对值规则（确定性派生）----


def _tp(
    kind: str,
    minute: int,
    price: float,
    volume: int,
    oi: int,
    *,
    step_minutes: int = 5,
) -> TurningPoint:
    base = pd.Timestamp("2026-09-23 09:00:00", tz="Asia/Shanghai")
    return TurningPoint(
        kind=kind,
        timestamp=base + pd.Timedelta(minutes=step_minutes * minute),
        price=price,
        bar_index=minute,
        volume=volume,
        oi=oi,
    )


def test_relative_metrics_first_point_is_all_none() -> None:
    points = [_tp("start", 0, 100.0, 100, 5000), _tp("close", 0, 102.0, 100, 5010)]
    metrics = relative_metrics(points)
    assert metrics[0] == RelativeMetrics(None, None, None, None)
    # 同一根 K 线上的 close 点：时间差为 0，比值为 1
    assert metrics[1].dt_minutes == 0.0
    assert metrics[1].price_ratio == pytest.approx(1.02)
    assert metrics[1].volume_ratio == pytest.approx(1.0)
    assert metrics[1].oi_ratio == pytest.approx(5010 / 5000)


def test_relative_metrics_empty_sequence() -> None:
    assert relative_metrics([]) == ()


def test_relative_metrics_zero_denominator_is_none() -> None:
    """前一点值为 0（真实存在无成交分钟）→ 比值无定义，为 None 而非 inf。"""
    points = [
        _tp("start", 0, 100.0, 100, 5000),
        _tp("up", 1, 110.0, 0, 5050),  # 前一点 volume=100 → 0.0；oi 1.01
        _tp("down", 2, 99.0, 50, 0),  # 前一点 volume=0 → None；oi 0/5050=0.0
        _tp("close", 2, 99.5, 60, 30),  # 前一点 oi=0 → None
    ]
    metrics = relative_metrics(points)
    assert metrics[1].volume_ratio == 0.0
    assert metrics[1].oi_ratio == pytest.approx(1.01)
    assert metrics[2].dt_minutes == 5.0
    assert metrics[2].volume_ratio is None
    assert metrics[2].oi_ratio == 0.0
    assert metrics[3].dt_minutes == 0.0
    assert metrics[3].volume_ratio == pytest.approx(1.2)
    assert metrics[3].oi_ratio is None


def test_relative_metrics_dt_minutes_across_gap() -> None:
    """时间差按 K 线起始时间计算，单位分钟（含跨缺口）。"""
    points = [_tp("start", 0, 100.0, 100, 5000), _tp("up", 3, 110.0, 100, 5010)]
    metrics = relative_metrics(points)
    assert metrics[1].dt_minutes == 15.0  # 3 × 5 分钟步长

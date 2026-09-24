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
    TURNING_POINT_COLUMNS,
    WindowMeta,
    build_window_meta,
    find_turning_points,
    load_turning_points,
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
    # 极值时间必须落在对应 bar_index 所在 K 线的 timestamp 上
    for point in points:
        assert point.timestamp == df["timestamp"].iloc[point.bar_index]


def test_up_then_down_records_high_including_turning_bar() -> None:
    rows = [
        (100, 101, 99, 100),
        (100, 110, 100, 109),
        (109, 120, 105, 119),
        (119, 121, 104, 110),  # low 104 < 105 转向；同根 high 121 创新高 → 记为高点
        (110, 112, 100, 101),
        (101, 113, 99, 112),  # high 113 > 112 转向；同根 low 99 创新低 → 记为低点
    ]
    df = build_ohlcv(rows)
    points = find_turning_points(df, initial_direction="up")

    assert points[1].bar_index == 3  # 高点落在转向根
    assert points[2].bar_index == 5
    assert points[1].timestamp == df["timestamp"].iloc[3]


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
    assert [str(p.timestamp) for p in loaded.points] == [
        str(p.timestamp) for p in points
    ]
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
        "point_index,kind,timestamp,price,bar_index\n"
        "0,open,2026-09-23 09:00:00+08:00,100,0\n",
        encoding="utf-8",
    )
    with pytest.raises(DataLoadError, match="kind 取值非法"):
        load_turning_points("DCE.v2701", "1m", data_dir=tmp_path)

    path.write_text(
        "point_index,kind,timestamp,price,bar_index\n"
        "0,start,not-a-time,100,0\n",
        encoding="utf-8",
    )
    with pytest.raises(DataLoadError, match="无法解析为时间"):
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

"""OHLCV 落盘、来源指纹与读取复现（AC-2、AC-4、AC-7）。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dataset.errors import DataLoadError, DatasetError
from dataset.storage import (
    build_source_data_version,
    load_ohlcv,
    ohlcv_filename,
    save_ohlcv,
)
from dataset.tests.conftest import build_ohlcv

_ROWS = [
    (100.0, 101.0, 99.0, 100.5),
    (100.5, 102.0, 100.0, 101.0),
    (101.0, 103.0, 100.5, 102.0),
]

_EXPECTED_SIDECAR_KEYS = {
    "provider",
    "symbol",
    "period",
    "output_format",
    "start_dt",
    "end_dt",
    "row_count",
    "first_timestamp",
    "last_timestamp",
    "file_sha256",
    "source_data_version",
}


def _df() -> pd.DataFrame:
    return build_ohlcv(_ROWS)


def test_save_ohlcv_writes_csv_and_sidecar(tmp_path: Path) -> None:
    path = save_ohlcv(_df(), symbol="DCE.v2701", period="1m", output_dir=tmp_path)

    assert path == tmp_path / "DCE.v2701_1m.csv"
    assert path.is_file()
    sidecar = path.with_suffix(".json")
    assert sidecar.is_file()

    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert set(meta) == _EXPECTED_SIDECAR_KEYS
    assert meta["provider"] == "tianqin"
    assert meta["symbol"] == "DCE.v2701"
    assert meta["period"] == "1m"
    assert meta["output_format"] == "csv"
    assert meta["row_count"] == len(_ROWS)
    assert meta["first_timestamp"] == str(_df()["timestamp"].iloc[0])
    assert meta["last_timestamp"] == str(_df()["timestamp"].iloc[-1])


def test_sidecar_has_no_wall_clock_field(tmp_path: Path) -> None:
    path = save_ohlcv(_df(), symbol="DCE.v2701", period="1m", output_dir=tmp_path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    forbidden = {"created_at", "saved_at", "generated_at", "timestamp", "run_time"}
    assert forbidden.isdisjoint(meta)


def test_source_data_version_contains_fingerprint_parts(tmp_path: Path) -> None:
    path = save_ohlcv(
        _df(),
        symbol="DCE.v2701",
        period="1m",
        output_dir=tmp_path,
        start="2026-09-23",
        end="2026-09-24",
    )
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    expected = build_source_data_version(
        symbol="DCE.v2701",
        period="1m",
        start="2026-09-23",
        end="2026-09-24",
        row_count=len(_ROWS),
        first_timestamp=str(_df()["timestamp"].iloc[0]),
        last_timestamp=str(_df()["timestamp"].iloc[-1]),
        file_sha256=meta["file_sha256"],
    )
    assert meta["source_data_version"] == expected
    assert meta["source_data_version"].count("|") == 8
    assert "rows=3" in meta["source_data_version"]
    assert meta["source_data_version"].endswith(f"sha256={meta['file_sha256'][:16]}")


def test_repeated_save_is_byte_identical(tmp_path: Path) -> None:
    df = _df()
    first = save_ohlcv(df, symbol="DCE.v2701", period="1m", output_dir=tmp_path / "a")
    second = save_ohlcv(df, symbol="DCE.v2701", period="1m", output_dir=tmp_path / "b")

    assert first.read_bytes() == second.read_bytes()
    assert (
        first.with_suffix(".json").read_bytes()
        == second.with_suffix(".json").read_bytes()
    )


def test_save_ohlcv_rejects_empty_and_bad_identifiers(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="拒绝落盘空 OHLCV"):
        save_ohlcv(_df().iloc[0:0], symbol="DCE.v2701", period="1m", output_dir=tmp_path)
    with pytest.raises(DatasetError, match="symbol 必须为非空字符串"):
        save_ohlcv(_df(), symbol="", period="1m", output_dir=tmp_path)


def test_load_ohlcv_round_trips_values_and_version(tmp_path: Path) -> None:
    saved = save_ohlcv(
        _df(),
        symbol="DCE.v2701",
        period="1m",
        output_dir=tmp_path,
        start="2026-09-23",
        end="2026-09-24",
    )
    meta = json.loads(saved.with_suffix(".json").read_text(encoding="utf-8"))

    loaded = load_ohlcv("DCE.v2701", "1m", data_dir=tmp_path)

    assert loaded.symbol == "DCE.v2701"
    assert loaded.period == "1m"
    assert loaded.path == saved
    assert loaded.source_data_version == meta["source_data_version"]
    assert loaded.row_count_declared == len(_ROWS)
    assert list(loaded.df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert str(loaded.df["timestamp"].iloc[0]) == str(_df()["timestamp"].iloc[0])
    assert loaded.df["close"].tolist() == pytest.approx([r[3] for r in _ROWS])
    assert loaded.df["volume"].tolist() == [100, 100, 100]


def test_load_ohlcv_preserves_file_order(tmp_path: Path) -> None:
    df = _df()
    save_ohlcv(df, symbol="DCE.v2701", period="1m", output_dir=tmp_path)
    path = tmp_path / ohlcv_filename("DCE.v2701", "1m")

    reversed_df = df.iloc[::-1].reset_index(drop=True)
    reversed_df.to_csv(path, index=False)

    loaded = load_ohlcv("DCE.v2701", "1m", data_dir=tmp_path)
    assert str(loaded.df["timestamp"].iloc[0]) == str(df["timestamp"].iloc[-1])


def test_load_ohlcv_wraps_non_oserror_read_failures(tmp_path: Path) -> None:
    """P2-1：空文件（EmptyDataError）/异编码（UnicodeDecodeError）也收敛为 DataLoadError。"""
    path = tmp_path / ohlcv_filename("DCE.v2701", "1m")

    path.write_text("", encoding="utf-8")
    with pytest.raises(DataLoadError) as excinfo:
        load_ohlcv("DCE.v2701", "1m", data_dir=tmp_path)
    assert "文件读取失败" in str(excinfo.value)
    assert str(path) in str(excinfo.value)  # 消息含文件路径

    path.write_bytes(b"\xff\xfe\x00\x01binary")
    with pytest.raises(DataLoadError) as excinfo:
        load_ohlcv("DCE.v2701", "1m", data_dir=tmp_path)
    assert "文件读取失败" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_load_ohlcv_rejects_row_count_mismatch(tmp_path: Path) -> None:
    """P2-2：sidecar 声明的 row_count 与实际行数不一致 → DataLoadError。"""
    df = _df()
    saved = save_ohlcv(df, symbol="DCE.v2701", period="1m", output_dir=tmp_path)
    df.iloc[:-1].to_csv(saved, index=False)  # 截断一行，sidecar 未同步

    with pytest.raises(DataLoadError, match="行数与来源指纹不一致"):
        load_ohlcv("DCE.v2701", "1m", data_dir=tmp_path)


def test_load_ohlcv_optional_sha256_verification(tmp_path: Path) -> None:
    """P2-2：可选 file_sha256 重算校验（默认关闭，开启后能拦住同长度的内容篡改）。"""
    df = _df()
    saved = save_ohlcv(df, symbol="DCE.v2701", period="1m", output_dir=tmp_path)
    tampered = df.copy()
    tampered.loc[0, "close"] = 999.0
    tampered.to_csv(saved, index=False)  # 行数不变，内容被改

    load_ohlcv("DCE.v2701", "1m", data_dir=tmp_path)  # 默认不重算摘要

    with pytest.raises(DataLoadError, match="文件摘要与来源指纹不一致"):
        load_ohlcv("DCE.v2701", "1m", data_dir=tmp_path, verify_sha256=True)

    saved.with_suffix(".json").unlink()  # 无 sidecar 时显式校验 → 明确报错，不静默通过
    with pytest.raises(DataLoadError, match="缺少 file_sha256"):
        load_ohlcv("DCE.v2701", "1m", data_dir=tmp_path, verify_sha256=True)


def test_save_ohlcv_is_atomic_and_pairs_csv_with_sidecar(tmp_path: Path) -> None:
    """P2-2：先写临时文件 + os.replace；成功后无残留，失败不产生半成品（成对落盘）。"""
    df = _df()
    save_ohlcv(df, symbol="DCE.v2701", period="1m", output_dir=tmp_path)
    assert list(tmp_path.glob(".*.tmp")) == []

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    csv_path = blocked / ohlcv_filename("DCE.v2701", "1m")
    csv_path.with_suffix(".json").mkdir()  # sidecar 目标被目录占用 → 提交失败

    with pytest.raises(DatasetError, match="落盘失败"):
        save_ohlcv(df, symbol="DCE.v2701", period="1m", output_dir=blocked)

    assert not csv_path.exists()  # 不产生「有 CSV 无 sidecar」的半成品
    assert list(blocked.glob(".*.tmp")) == []


def test_save_ohlcv_reports_unusable_output_dir(tmp_path: Path) -> None:
    """P2-N2 回归：``output_dir`` 不可创建 → :class:`DatasetError`，不抛裸 ``OSError``。"""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    with pytest.raises(DatasetError, match="输出目录不可用") as excinfo:
        save_ohlcv(_df(), symbol="DCE.v2701", period="1m", output_dir=blocker / "sub")

    assert "NotADirectoryError" in str(excinfo.value)
    assert str(blocker / "sub") in str(excinfo.value)
    assert not (blocker / "sub").exists()


def test_load_ohlcv_errors(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="不存在"):
        load_ohlcv("DCE.v2701", "1m", data_dir=tmp_path)

    df = _df()
    save_ohlcv(df, symbol="DCE.v2701", period="1m", output_dir=tmp_path)
    path = tmp_path / ohlcv_filename("DCE.v2701", "1m")

    path.with_suffix(".json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DataLoadError, match="来源指纹文件损坏"):
        load_ohlcv("DCE.v2701", "1m", data_dir=tmp_path)

    path.with_suffix(".json").unlink()
    df.drop(columns=["volume"]).to_csv(path, index=False)
    with pytest.raises(DataLoadError, match="缺少必需列"):
        load_ohlcv("DCE.v2701", "1m", data_dir=tmp_path)

    df.to_csv(path, index=False)
    loaded = load_ohlcv("DCE.v2701", "1m", data_dir=tmp_path)
    assert loaded.source_data_version is None  # 无 sidecar 时不伪装可追溯

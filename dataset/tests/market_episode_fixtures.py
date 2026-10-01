"""``dataset/market_episode`` 测试的公共构造器（离线、只写 ``tmp_path``）。

放在独立模块而不是 ``conftest.py``：只是新增测试的私有夹具，避免改动既有共享夹具文件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import yaml

from dataset.market_episode.replay import Bar, bars_from_frame
from dataset.market_episode.segments import SEGMENT_SCHEMA, SYMBOLS_SCHEMA
from dataset.storage import save_ohlcv
from dataset.tests.conftest import build_ohlcv

#: 测试用合约与 tick（tick = 1 便于手工推演）
SYMBOL = "TEST.sym"
TICK_SIZE = 1.0
#: 测试用起始时间（北京时间）
START = "2024-01-02 09:00:00"


def frame(
    rows: Sequence[tuple[float, float, float, float]],
    *,
    start: str = START,
    volume: int = 100,
    oi_start: int = 5000,
    oi_step: int = 10,
) -> pd.DataFrame:
    """``(open, high, low, close)`` 列表 → 标准 OHLCV（成交量恒定、持仓量逐根递增）。"""
    count = len(rows)
    return build_ohlcv(
        list(rows),
        start=start,
        volume=volume,
        open_oi=[oi_start + oi_step * i for i in range(count)],
        close_oi=[oi_start + oi_step * (i + 1) for i in range(count)],
    )


def bars(
    rows: Sequence[tuple[float, float, float, float]], *, start: str = START, **kwargs: object
) -> tuple[Bar, ...]:
    """构造片段内 ``Bar`` 序列（片段内 0 基下标 = 决策 K 线标识）。"""
    return bars_from_frame(frame(rows, start=start, **kwargs).reset_index(drop=True))


def timestamp_at(index: int, *, start: str = START, minutes: int = 1) -> str:
    return str(pd.Timestamp(start, tz="Asia/Shanghai") + pd.Timedelta(minutes=minutes * index))


def segment_record(
    segment_id: str,
    split_role: str,
    *,
    start_index: int,
    end_index: int,
    start: str = START,
    symbol: str = SYMBOL,
    notes: str | None = None,
) -> dict[str, Any]:
    """一条清单记录（端点含；默认 1 分钟步长）。"""
    record: dict[str, Any] = {
        "schema": SEGMENT_SCHEMA,
        "segment_id": segment_id,
        "symbol": symbol,
        "period": "1m",
        "start_ts": timestamp_at(start_index, start=start),
        "end_ts": timestamp_at(end_index, start=start),
        "split_role": split_role,
    }
    if notes is not None:
        record["notes"] = notes
    return record


def write_manifest(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def write_symbols_config(
    path: Path, *, symbol: str = SYMBOL, tick_size: float = TICK_SIZE
) -> Path:
    path.write_text(
        yaml.safe_dump(
            {"schema": SYMBOLS_SCHEMA, "symbols": {symbol: {"tick_size": tick_size}}},
            allow_unicode=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def write_episode_config(
    path: Path,
    *,
    data_dir: Path,
    symbols_path: Path,
    episode: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {"dataset": {"output_dir": str(data_dir.parent)}}
    section: dict[str, Any] = {"symbols": str(symbols_path)}
    section.update(episode or {})
    payload["episode"] = section
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=True), encoding="utf-8")
    return path


@dataclass(frozen=True)
class Workspace:
    """一个自洽的临时工作区（K 线 + 清单 + 品种配置 + 配置文件）。"""

    root: Path
    data_dir: Path
    manifest: Path
    symbols_path: Path
    config_path: Path
    rows: tuple[tuple[float, float, float, float], ...]

    def records(self) -> list[dict[str, Any]]:
        return [
            json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines()
        ]


def build_workspace(
    tmp_path: Path,
    rows: Sequence[tuple[float, float, float, float]],
    *,
    split_roles: Sequence[str] = ("train", "dev", "test"),
    tick_size: float = TICK_SIZE,
    symbol: str = SYMBOL,
    episode: dict[str, Any] | None = None,
) -> Workspace:
    """把 ``rows`` 均分成 ``split_roles`` 个互不重叠的片段（一个片段 = 一个 episode）。"""
    if len(rows) < len(split_roles):
        raise ValueError("K 线数量必须不少于片段数量")
    data_dir = tmp_path / "data" / "ohlcv"
    save_ohlcv(frame(rows), symbol=symbol, period="1m", output_dir=data_dir)

    boundaries: list[tuple[int, int]] = []
    size = len(rows) // len(split_roles)
    for position in range(len(split_roles)):
        first = position * size
        last = first + size - 1 if position < len(split_roles) - 1 else len(rows) - 1
        boundaries.append((first, last))
    records = [
        segment_record(f"seg-{role}", role, start_index=first, end_index=last, symbol=symbol)
        for role, (first, last) in zip(split_roles, boundaries)
    ]
    manifest = write_manifest(tmp_path / "segments.jsonl", records)
    symbols_path = write_symbols_config(tmp_path / "symbols.yaml", symbol=symbol, tick_size=tick_size)
    config_path = write_episode_config(
        tmp_path / "episode.yaml", data_dir=data_dir, symbols_path=symbols_path, episode=episode
    )
    return Workspace(
        root=tmp_path,
        data_dir=data_dir,
        manifest=manifest,
        symbols_path=symbols_path,
        config_path=config_path,
        rows=tuple(rows),
    )


__all__ = [
    "START",
    "SYMBOL",
    "TICK_SIZE",
    "Workspace",
    "bars",
    "build_workspace",
    "frame",
    "segment_record",
    "timestamp_at",
    "write_episode_config",
    "write_manifest",
    "write_symbols_config",
]

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
#: 测试用上一交易日日线（供 board_state prev_* 值；量级与 1m 行情拉开，避免绝对 token 假阳性）
DAILY_START = "2024-01-01 00:00:00"
DAILY_ROWS: tuple[tuple[float, float, float, float], ...] = (
    (4000.0, 4010.0, 3990.0, 4005.0),
)


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
    """一个自洽的临时工作区（1m K 线 + 上一交易日日线 + 清单 + 品种配置 + 配置文件）。"""

    root: Path
    data_dir: Path
    manifest: Path
    symbols_path: Path
    config_path: Path
    rows: tuple[tuple[float, float, float, float], ...]
    daily_path: Path
    prev_daily: tuple[float, float, float]

    def records(self) -> list[dict[str, Any]]:
        return [
            json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines()
        ]


def prev_daily_map(workspace: Workspace) -> dict[str, tuple[float, float, float]]:
    """片段 ID → 上一交易日日线 (高, 低, 收)（供 check_state_leakage 独立重算入参）。"""
    return {
        record["segment_id"]: workspace.prev_daily for record in workspace.records()
    }


def build_workspace(
    tmp_path: Path,
    rows: Sequence[tuple[float, float, float, float]],
    *,
    split_roles: Sequence[str] = ("train", "dev", "test"),
    tick_size: float = TICK_SIZE,
    symbol: str = SYMBOL,
    episode: dict[str, Any] | None = None,
    daily_rows: Sequence[tuple[float, float, float, float]] | None = None,
    daily_start: str = DAILY_START,
) -> Workspace:
    """把 ``rows`` 均分成 ``split_roles`` 个互不重叠的片段（一个片段 = 一个 episode）。

    ``daily_rows``/``daily_start`` 可覆盖日线夹具（默认 DAILY_ROWS/DAILY_START：
    上一交易日 = 片段交易日前一天）。"""
    if len(rows) < len(split_roles):
        raise ValueError("K 线数量必须不少于片段数量")
    data_dir = tmp_path / "data" / "ohlcv"
    save_ohlcv(frame(rows), symbol=symbol, period="1m", output_dir=data_dir)
    daily_path = save_ohlcv(
        frame(list(daily_rows if daily_rows is not None else DAILY_ROWS), start=daily_start),
        symbol=symbol,
        period="1d",
        output_dir=data_dir,
    )

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
        daily_path=daily_path,
        prev_daily=DAILY_ROWS[0][1:],
    )


#: 两交易日模式（每 2 根一个机会模式：bar0 盘整、bar1 机会分钟）
TWO_DAY_PATTERN = [
    (100, 100.2, 99.8, 100),
    (100, 110, 100, 109),
]


def build_two_day_workspace(tmp_path: Path) -> Workspace:
    """两个交易日的 1m 数据（首日 4 根 + 次日 4 根）；日线只有首日/次日两行。

    首日片段（seg-train）无上一交易日日线 → 生成时被跳过；
    次日片段（seg-dev/seg-test）的 prev = 首日日线行（prev_daily）。"""
    day1 = frame(TWO_DAY_PATTERN * 2, start="2024-01-02 09:00:00")
    day2 = frame(TWO_DAY_PATTERN * 2, start="2024-01-03 09:00:00")
    data_dir = tmp_path / "data" / "ohlcv"
    save_ohlcv(
        pd.concat([day1, day2]).reset_index(drop=True),
        symbol=SYMBOL,
        period="1m",
        output_dir=data_dir,
    )
    daily_path = save_ohlcv(
        pd.concat(
            (
                frame([DAILY_ROWS[0]], start="2024-01-02 00:00:00"),
                frame([(4010.0, 4020.0, 4000.0, 4015.0)], start="2024-01-03 00:00:00"),
            )
        ).reset_index(drop=True),
        symbol=SYMBOL,
        period="1d",
        output_dir=data_dir,
    )
    records = [
        segment_record("seg-train", "train", start_index=0, end_index=3, start="2024-01-02 09:00:00"),
        segment_record("seg-dev", "dev", start_index=0, end_index=1, start="2024-01-03 09:00:00"),
        segment_record("seg-test", "test", start_index=2, end_index=3, start="2024-01-03 09:00:00"),
    ]
    manifest = write_manifest(tmp_path / "segments.jsonl", records)
    symbols_path = write_symbols_config(tmp_path / "symbols.yaml")
    config_path = write_episode_config(
        tmp_path / "episode.yaml", data_dir=data_dir, symbols_path=symbols_path
    )
    return Workspace(
        root=tmp_path,
        data_dir=data_dir,
        manifest=manifest,
        symbols_path=symbols_path,
        config_path=config_path,
        rows=TWO_DAY_PATTERN * 4,
        daily_path=daily_path,
        prev_daily=DAILY_ROWS[0][1:],
    )


__all__ = [
    "DAILY_ROWS",
    "DAILY_START",
    "START",
    "SYMBOL",
    "TICK_SIZE",
    "TWO_DAY_PATTERN",
    "Workspace",
    "bars",
    "build_two_day_workspace",
    "build_workspace",
    "frame",
    "prev_daily_map",
    "segment_record",
    "timestamp_at",
    "write_episode_config",
    "write_manifest",
    "write_symbols_config",
]

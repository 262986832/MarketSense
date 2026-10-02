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
from dataset.turning_points import (
    TURNING_POINT_COLUMNS,
    TrendExtreme,
    TurningPoint,
    load_turning_points,
)

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
#: 测试用日线转折点默认点集（v5 日线行 trend 四值数据源；确认日 2024-01-01 严格早于
#: 片段交易日 2024-01-02/01-03，up/down 各 ≥1；kind 含 start 点以覆盖
#: 「up/down 之外不参与选择」的过滤行为；量级与 1m 行情（~100）拉开，避免绝对 token
#: 假阳性；timestamp 带时区偏移与生产 CSV 写法一致——``load_turning_points`` 按
#: ISO8601+utc 解析，naive 串会被当作 UTC 而平移日历日）
DAILY_TP_ROWS: tuple[dict[str, Any], ...] = (
    {
        "kind": "start",
        "timestamp": "2024-01-01 00:00:00+08:00",
        "price": 4000.0,
        "bar_index": 0,
        "volume": 100,
        "oi": 5000,
    },
    {
        "kind": "down",
        "timestamp": "2024-01-01 21:05:00+08:00",
        "price": 3980.0,
        "bar_index": 5,
        "volume": 120,
        "oi": 5010,
        "trend_extreme_price": 3980.0,
        "trend_extreme_bar_index": 5,
    },
    {
        "kind": "up",
        "timestamp": "2024-01-01 21:35:00+08:00",
        "price": 4020.0,
        "bar_index": 12,
        "volume": 130,
        "oi": 5020,
        "trend_extreme_price": 4020.0,
        "trend_extreme_bar_index": 12,
    },
)
#: 默认点集的可用 trend 极值（手工推导，与 recent_trend_extremes/_independent_trend_extremes
#: 的段长推导一致：down@bar5 段长 5−0=5；up@bar12 段长 12−5=7）
DAILY_TP_UP = TrendExtreme(
    kind="up", trend_extreme_price=4020.0, trend_extreme_bar_index=12, segment_length=7
)
DAILY_TP_DOWN = TrendExtreme(
    kind="down", trend_extreme_price=3980.0, trend_extreme_bar_index=5, segment_length=5
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


def write_daily_turning_points(
    path: Path,
    rows: Sequence[dict[str, Any]] | None = None,
) -> Path:
    """按 ``TURNING_POINT_COLUMNS`` 13 列写日线转折点 CSV（回读走 ``load_turning_points``
    与生产同路径，读取结果可在测试中独立核对）。

    ``rows`` 缺省用 ``DAILY_TP_ROWS``（确认日早于片段交易日、up/down 各 ≥1）。
    自定义点集供边界用例构造单侧缺失/同日确认/多日泛化等定制 TP 文件；缺失键写
    空单元格（相对值/极值列的空单元格合法，与生产 CSV 表示一致）。
    """
    entries = list(DAILY_TP_ROWS if rows is None else rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(TURNING_POINT_COLUMNS)]
    for entry in entries:
        fields = []
        for column in TURNING_POINT_COLUMNS:
            value = entry.get(column)
            fields.append("" if value is None else str(value))
        lines.append(",".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def daily_trend_points_map(workspace: Workspace) -> dict[str, tuple[TurningPoint, ...]]:
    """symbol → 日线转折点元组（供 ``check_state_leakage`` 新必参的独立重算入参）。

    读取走 ``load_turning_points``（与生产同路径；symbol 由文件名 ``{symbol}_1d.csv``
    确定性反解）。
    """
    name = workspace.turning_points_path.name
    suffix = "_1d.csv"
    symbol = name[: -len(suffix)]
    loaded = load_turning_points(symbol, "1d", data_dir=workspace.turning_points_path.parent)
    return {symbol: loaded.points}


@dataclass(frozen=True)
class Workspace:
    """一个自洽的临时工作区（1m K 线 + 上一交易日日线 + 清单 + 品种配置 + 配置文件
    + 日线转折点 CSV）。

    ``turning_points_path`` = v5 日线行 trend 四值数据源（与 ``generate_dataset`` 默认
    推导目录 ``<data_dir>/../turning_points`` 一致，既有调用点零修改即可获得文件）。
    """

    root: Path
    data_dir: Path
    manifest: Path
    symbols_path: Path
    config_path: Path
    rows: tuple[tuple[float, float, float, float], ...]
    daily_path: Path
    prev_daily: tuple[float, float, float]
    turning_points_path: Path

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
    daily_tp_rows: Sequence[dict[str, Any]] | None = None,
) -> Workspace:
    """把 ``rows`` 均分成 ``split_roles`` 个互不重叠的片段（一个片段 = 一个 episode）。

    ``daily_rows``/``daily_start`` 可覆盖日线夹具（默认 DAILY_ROWS/DAILY_START：
    上一交易日 = 片段交易日前一天）；``daily_tp_rows`` 可覆盖日线转折点夹具
    （默认 DAILY_TP_ROWS：确认日严格早于片段交易日、up/down 各 ≥1，满足 v5 生成
    要求，既有调用点零修改即可获得文件）。"""
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
    turning_points_path = write_daily_turning_points(
        data_dir.parent / "turning_points" / f"{symbol}_1d.csv", rows=daily_tp_rows
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
        turning_points_path=turning_points_path,
    )


#: 两交易日模式（每 2 根一个机会模式：bar0 盘整、bar1 机会分钟）
TWO_DAY_PATTERN = [
    (100, 100.2, 99.8, 100),
    (100, 110, 100, 109),
]


def build_two_day_workspace(
    tmp_path: Path, *, daily_tp_rows: Sequence[dict[str, Any]] | None = None
) -> Workspace:
    """两个交易日的 1m 数据（首日 4 根 + 次日 4 根）；日线只有首日/次日两行。

    首日片段（seg-train）无上一交易日日线 → 生成时被跳过（在日线折点检查之前，
    故其不参与 v5 TP 检查）；次日片段（seg-dev/seg-test）的 prev = 首日日线行
    （prev_daily）。默认 TP 夹具（确认日 2024-01-01）对次日片段（交易日 2024-01-03）
    两侧齐全。"""
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
    turning_points_path = write_daily_turning_points(
        data_dir.parent / "turning_points" / f"{SYMBOL}_1d.csv", rows=daily_tp_rows
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
        turning_points_path=turning_points_path,
    )


__all__ = [
    "DAILY_ROWS",
    "DAILY_START",
    "DAILY_TP_DOWN",
    "DAILY_TP_ROWS",
    "DAILY_TP_UP",
    "START",
    "SYMBOL",
    "TICK_SIZE",
    "TWO_DAY_PATTERN",
    "Workspace",
    "bars",
    "build_two_day_workspace",
    "build_workspace",
    "daily_trend_points_map",
    "frame",
    "prev_daily_map",
    "segment_record",
    "timestamp_at",
    "write_daily_turning_points",
    "write_episode_config",
    "write_manifest",
    "write_symbols_config",
]

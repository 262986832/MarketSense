"""episode mechanics 层：数据装载 + 状态行构造 + 账户域 re-export 兼容 shim。

账户域（持仓/成交/盯市/死亡执行语义）已迁移至 :mod:`dataset.account`（仅依赖
``dataset.errors``，可脱离 ``market_episode`` 独立使用）；本模块对账户域名字做
**re-export 兼容 shim**——既有 ``from dataset.market_episode.replay import Bar,
ReplayAccount, ...`` 的全部调用点（labels/nanojev_records/audit/tests）继续可用。

本模块保留 episode 特有逻辑：

* ``bars_from_frame`` / ``load_segment_bars`` —— 已落盘 K 线 → ``Bar`` 元组的数据装载
  （片段 = episode；时间列必须严格升序且唯一）；
* ``px_ratio_line`` —— v4 状态行构造（现价；v7 并入 ``bar`` 序号与量/持仓量比值，
  「联动」行变为 ``na`` 常量占位）。

账户执行语义契约（决策 K 线成交模型、止损锚定、t-1 盯市、死亡/片段末强平）见
:mod:`dataset.account` 的 docstring。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset.account import (
    DIRECTIONS,
    SHORT,
    LONG,
    REASON_DEATH,
    REASON_DECISION,
    REASON_REVERSE_CLOSE,
    REASON_REVERSE_OPEN,
    REASON_SEGMENT_END,
    REASON_STOP,
    Bar,
    MarkReport,
    Position,
    PositionSnapshot,
    ReplayAccount,
    TradeEvent,
    format_ratio_value,
    ratio_or_none,
)
from dataset.errors import DatasetError
from dataset.ohlcv import OHLCV_COLUMNS
from dataset.storage import load_ohlcv

from dataset.market_episode.segments import Segment

__all__ = [
    "DIRECTIONS",
    "LONG",
    "SHORT",
    "REASON_DEATH",
    "REASON_DECISION",
    "REASON_REVERSE_CLOSE",
    "REASON_REVERSE_OPEN",
    "REASON_SEGMENT_END",
    "REASON_STOP",
    "Bar",
    "MarkReport",
    "Position",
    "PositionSnapshot",
    "ReplayAccount",
    "TradeEvent",
    "bars_from_frame",
    "format_ratio_value",
    "load_segment_bars",
    "px_ratio_line",
    "ratio_or_none",
]


def bars_from_frame(df: pd.DataFrame, *, start_index: int = 0) -> tuple[Bar, ...]:
    """标准 OHLCV DataFrame → ``Bar`` 元组（保持文件原序）。

    :raises DatasetError: 缺列 / 空表
    """
    if df is None or df.empty:
        raise DatasetError("K 线为空，无法构造 episode")
    missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise DatasetError(f"K 线缺少必需列: {missing}")
    rows = zip(
        df["timestamp"],
        df["open"],
        df["high"],
        df["low"],
        df["close"],
        df["volume"],
        df["open_oi"],
        df["close_oi"],
    )
    return tuple(
        Bar(
            index=start_index + offset,
            timestamp=str(timestamp),
            open=float(open_),
            high=float(high),
            low=float(low),
            close=float(close),
            volume=int(volume),
            open_oi=int(open_oi),
            close_oi=int(close_oi),
        )
        for offset, (
            timestamp,
            open_,
            high,
            low,
            close,
            volume,
            open_oi,
            close_oi,
        ) in enumerate(rows)
    )


def load_segment_bars(
    segment: Segment, *, data_dir: str | Path
) -> tuple[tuple[Bar, ...], str | None]:
    """读取片段区间内的已落盘 K 线（端点含），返回 ``(bars, source_data_version)``。

    只使用片段区间内的数据；区间按 ``start <= t <= end`` 过滤（冻结：片段 = episode）。
    时间列必须是严格升序且唯一：episode 的因果推进依赖行序，混序数据必须报错而不是
    静默排序（避免与 ``data/ohlcv`` 原文件口径不一致）。

    :raises DatasetError: 片段内无 K 线 / 时间列非严格升序或重复
    """
    loaded = load_ohlcv(segment.symbol, segment.period, data_dir=data_dir)
    frame = loaded.df
    stamps = frame["timestamp"]
    selected = frame.loc[(stamps >= segment.start) & (stamps <= segment.end)]
    if selected.empty:
        raise DatasetError(f"片段 {segment.segment_id} 区间内没有任何 K 线")
    ordered = selected["timestamp"]
    if not bool(ordered.is_monotonic_increasing) or not bool(ordered.is_unique):
        raise DatasetError(
            f"片段 {segment.segment_id} 的 K 线时间列必须严格升序且唯一（拒绝静默排序）"
        )
    return bars_from_frame(selected.reset_index(drop=True)), loaded.source_data_version


def px_ratio_line(
    bar: Bar, reference_bar: Bar, precision: int, *, breakthrough_label: str, breakthrough: float | None
) -> str:
    """v14 现价行；价格仅保留决策收盘价比值，另含 bar、量比及主突破值。"""
    return (
        "现价: "
        f"bar={bar.index}"
        f" 价={format_ratio_value(ratio_or_none(bar.close, reference_bar.open), precision)}"
        f" 成交量比={format_ratio_value(ratio_or_none(bar.volume, reference_bar.volume), precision)}"
        f" 持仓量比={format_ratio_value(ratio_or_none(bar.close_oi, reference_bar.close_oi), precision)}"
        f" {breakthrough_label}={format_ratio_value(breakthrough, precision)}"
    )

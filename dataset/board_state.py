"""盘面状态数据（board state）：读取并逐根维护「固定的前日高/低/收 + 今日开盘，动态的今日最高/最低」。

研究 building block（2026-10-01 首个状态读取实现；**非最终模型输入格式**，后续演进需另行讨论）。

语义（确定性、无未来数据泄漏，均为本实现冻结口径）：

- **窗口**：交易日 ``T`` 的窗口 = 归属 ``T`` 的夜盘 K 线 + ``T`` 的日盘 K 线；
  夜盘 K 线（time-of-day ≥ 21:00）归属其日历日之后的**下一个交易日**（周五夜盘 → 周一）；
  交易日 = 存在日盘 K 线（time-of-day < 15:00）的日历日。与片段清单（``sep2026.jsonl``）同口径。
- **固定状态**（窗口内不变）：
  - ``prev_day_high`` / ``prev_day_low`` / ``prev_day_close``：**上一交易日**的日线 OHLC
    （来源 1d 落盘文件；日线与 1m 夜盘归属同口径，2026-09 真实数据 26/27 交叉核对一致）；
  - ``today_open``：窗口首根 K 线开盘价（= 日线开盘口径）。
- **动态状态**（逐根更新）：``today_high`` = 已处理 K 线高点的累计最大；
  ``today_low`` = 已处理 K 线低点的累计最小。``State(T)`` 只用 ``≤ T`` 的 K 线（无未来泄漏）；
  初始（未处理任何 K 线前）两者 = ``today_open``（平盘先验；因 high ≥ open ≥ low，
  与「只对 K 线高低点取累计极值」等价）。
- **相对价**：六个价格字段一律 = 值 / ``today_open``（6 位小数）；
  ``today_open`` 恒为 ``1.000000``。绝对 OHLC 只留在 ``data/ohlcv/`` 原始层。

输出（``board_state/{symbol}_board_state.csv`` + sidecar）：

```text
trade_date, timestamp, bar_index, prev_day_high, prev_day_low, prev_day_close, today_open, today_high, today_low
```

无墙钟因素；同输入 → 同字节输出。CLI 见 ``dataset/cli.py`` 的 ``board-state`` 子命令。
"""

from __future__ import annotations

import bisect
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Mapping

import pandas as pd

from dataset.errors import DatasetError
from dataset.storage import commit_data_with_sidecar, file_sha256

#: 模型可见比值小数位（与 episode 流水线冻结默认 price_precision=6 一致）
PRECISION = 6

#: 夜盘起始 / 日盘结束（bar open 时间口径；DCE 夜盘 21:00-23:00、日盘 09:00-15:00）
NIGHT_START = dt.time(21, 0)
DAY_END = dt.time(15, 0)

#: 输出列（顺序固定；六个价格列均为相对价）
STATE_COLUMNS = (
    "trade_date",
    "timestamp",
    "bar_index",
    "prev_day_high",
    "prev_day_low",
    "prev_day_close",
    "today_open",
    "today_high",
    "today_low",
)


def attribute_windows(
    minute_df: pd.DataFrame,
    *,
    trading_days: Collection[dt.date] | None = None,
) -> tuple[dict[dt.date, list[int]], int]:
    """把 1 分钟 K 线按夜盘归属规则分窗。

    :param minute_df: 标准 OHLCV（``timestamp`` tz-aware，升序）
    :param trading_days: 显式交易日全集（可选；**纯增量参数**，缺省 ``None`` = 由
        帧内日盘根日历日推导，行为与既有调用完全一致）。夜盘根归属 = 集合中
        严格大于其日历日的最近交易日。在线夜盘快照场景（``dataset/live_state``）
        决策交易日的日盘根尚未产生、帧内推导必然缺失归属键，由调用方传入
        完整全集（日线日期 ∪ 帧内日盘日历日 ∪ 尾段夜盘隐含次日）。
    :return: (交易日 → 按时间升序的行索引列表, 未入窗 K 线数)
        未入窗 = time-of-day 在 ``[15:00, 21:00)`` 的 K 线（DCE 本数据不存在；
        若出现则说明数据含未知时段，调用方应显式告警不静默）。
    """
    ts = minute_df["timestamp"]
    dates: list[dt.date] = list(ts.dt.date)
    tods: list[dt.time] = [t.time() for t in ts]

    if trading_days is None:
        trading_days = {d for d, tod in zip(dates, tods) if tod < DAY_END}
    trading_days = sorted(set(trading_days))

    def _next_trading_day(d: dt.date) -> dt.date | None:
        i = bisect.bisect_right(trading_days, d)
        return trading_days[i] if i < len(trading_days) else None

    windows: dict[dt.date, list[int]] = {t: [] for t in trading_days}
    unassigned = 0
    for idx, (d, tod) in enumerate(zip(dates, tods)):
        if tod >= NIGHT_START:
            owner = _next_trading_day(d)  # 夜盘 → 之后的下一个交易日
        elif tod < DAY_END:
            owner = d  # 日盘 → 自身日历日（含此类 K 线的日历日即交易日）
        else:
            owner = None  # 15:00–20:59：不属于任何窗口
            unassigned += 1
        if owner is not None:
            windows[owner].append(idx)
    return windows, unassigned


class BoardStateTracker:
    """单交易日窗口内的盘面状态跟踪器（“运行过程中动态更新”语义）。

    固定字段构造时一次给定；动态字段随 ``update()`` 逐根累计。
    无未来数据：``State(T)`` 只依赖 ``≤ T`` 的 K 线。
    """

    def __init__(
        self,
        *,
        prev_day_high: float,
        prev_day_low: float,
        prev_day_close: float,
        today_open: float,
        precision: int = PRECISION,
    ) -> None:
        base = float(today_open)
        if not base > 0:
            raise ValueError(f"today_open 必须为正数（相对价分母）: {today_open!r}")
        self._base = base
        self._precision = precision
        self._prev = {
            "prev_day_high": float(prev_day_high),
            "prev_day_low": float(prev_day_low),
            "prev_day_close": float(prev_day_close),
        }
        # 平盘先验（见模块 docstring）：未处理 K 线前，今日高/低 = today_open
        self.today_high = base
        self.today_low = base

    def update(self, *, bar_high: float, bar_low: float) -> dict[str, float]:
        """吸收一根 K 线，返回更新后的状态快照（相对价，已按精度取整）。"""
        if bar_high > 0 and bar_high > self.today_high:
            self.today_high = float(bar_high)
        if bar_low > 0 and bar_low < self.today_low:
            self.today_low = float(bar_low)
        return self.snapshot()

    def snapshot(self) -> dict[str, float]:
        """当前状态（相对价：值 / today_open，6 位小数；today_open 恒为 1.0）。"""
        p = self._precision
        out = {k: round(v / self._base, p) for k, v in self._prev.items()}
        out["today_open"] = 1.0
        out["today_high"] = round(self.today_high / self._base, p)
        out["today_low"] = round(self.today_low / self._base, p)
        return out


@dataclass
class BoardStateResult:
    """一次盘面状态构建的结果（纯数据，无墙钟）。"""

    df: pd.DataFrame
    symbol: str
    trade_days: list[str] = field(default_factory=list)
    skipped_days: list[str] = field(default_factory=list)
    unassigned_bars: int = 0
    minute_path: Path | None = None
    daily_path: Path | None = None


def build_board_states(
    minute_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    symbol: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
    precision: int = PRECISION,
) -> BoardStateResult:
    """构建盘面状态序列（每根 K 线一行，动态字段逐根更新）。

    :param minute_df: 1 分钟 K 线（标准 OHLCV）
    :param daily_df: 日线 K 线（标准 OHLCV；timestamp 为交易日 00:00）
    :param start/end: 可选，只生成 ``start ≤ 交易日 ≤ end`` 的状态（含边界）
    :raises DatasetError: 上一交易日在 1m 序列中存在、但日线文件缺该交易日（数据不一致）
    """
    windows, unassigned = attribute_windows(minute_df)
    trading_days = sorted(windows)

    daily: dict[dt.date, dict[str, float]] = {}
    if not daily_df.empty:
        dts = daily_df["timestamp"]
        for i, d in enumerate(dts.dt.date):
            daily[d] = {
                "high": float(daily_df["high"].iloc[i]),
                "low": float(daily_df["low"].iloc[i]),
                "close": float(daily_df["close"].iloc[i]),
            }

    rows: list[dict[str, Any]] = []
    trade_days: list[str] = []
    skipped: list[str] = []
    for t in trading_days:
        if (start is not None and t < start) or (end is not None and t > end):
            continue
        t_str = t.isoformat()
        pos = trading_days.index(t)
        if pos == 0:
            # 1m 数据起点（或其前置不在数据内）：无前置交易日 → 跳过（显式记录，不静默）
            skipped.append(f"{t_str}: 无前置交易日（数据起点），前日高/低/收不可得")
            continue
        prev_day = trading_days[pos - 1]
        if prev_day not in daily:
            raise DatasetError(
                f"日线数据缺少上一交易日 {prev_day.isoformat()}（交易日 {t_str} 的"
                "前日高/低/收不可得；1m 与 1d 数据交易日不一致，请重新落盘日线）"
            )
        prev = daily[prev_day]
        idxs = windows[t]
        first = idxs[0]
        tracker = BoardStateTracker(
            prev_day_high=prev["high"],
            prev_day_low=prev["low"],
            prev_day_close=prev["close"],
            today_open=float(minute_df["open"].iloc[first]),
            precision=precision,
        )
        ts_col = minute_df["timestamp"]
        for bar_index, i in enumerate(idxs):
            snap = tracker.update(
                bar_high=float(minute_df["high"].iloc[i]),
                bar_low=float(minute_df["low"].iloc[i]),
            )
            rows.append(
                {
                    "trade_date": t_str,
                    "timestamp": ts_col.iloc[i].isoformat(sep=" "),
                    "bar_index": bar_index,
                    **snap,
                }
            )
        trade_days.append(t_str)

    df = pd.DataFrame(rows, columns=list(STATE_COLUMNS))
    return BoardStateResult(
        df=df,
        symbol=symbol,
        trade_days=trade_days,
        skipped_days=skipped,
        unassigned_bars=unassigned,
    )


BOARD_STATE_SUBDIR = "board_state"

#: sidecar schema 名
BOARD_STATE_SCHEMA = "marketsense.board_state.v1"

#: 语义说明（写入 sidecar；与模块 docstring 同口径）
_BOARD_STATE_SEMANTICS = {
    "window": "交易日窗口=归属该交易日的夜盘K线+日盘K线（夜盘=前一历日21:00起，周五夜盘→周一）",
    "fixed": ["prev_day_high", "prev_day_low", "prev_day_close", "today_open"],
    "dynamic": ["today_high", "today_low"],
    "dynamic_rule": "已处理K线高/低点累计max/min；State(T)只用<=T的K线（无未来泄漏）",
    "relative": "全部价格字段=值/today_open（6位小数）；today_open=窗口首根开盘价（=日线开盘口径）",
    "prev_day_source": "上一交易日日线OHLC（1d落盘文件，夜盘归属与1m同口径）",
}


def board_state_filename(symbol: str) -> str:
    return f"{symbol}_board_state.csv"


def render_board_state_csv(df: pd.DataFrame) -> str:
    """渲染为确定性 CSV 文本（六个价格列固定 6 位小数；无墙钟因素）。"""
    lines = [",".join(STATE_COLUMNS)]
    for row in df.itertuples(index=False):
        lines.append(
            f"{row.trade_date},{row.timestamp},{row.bar_index},"
            f"{row.prev_day_high:.6f},{row.prev_day_low:.6f},{row.prev_day_close:.6f},"
            f"{row.today_open:.6f},{row.today_high:.6f},{row.today_low:.6f}"
        )
    return "\n".join(lines) + "\n"


def save_board_states(
    result: BoardStateResult,
    *,
    symbol: str,
    period: str,
    output_dir: str | Path,
    minute_path: Path,
    daily_path: Path,
) -> Path:
    """落盘「盘面状态 CSV + sidecar」（原子成对提交；同输入 → 同字节输出）。"""
    out_dir = Path(output_dir) / BOARD_STATE_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / board_state_filename(symbol)
    data_text = render_board_state_csv(result.df)

    def _source(p: Path) -> dict[str, Any]:
        return {
            "path": str(p),
            "file_sha256": file_sha256(p),
        }

    def build_meta(digest: str) -> Mapping[str, Any]:
        return {
            "schema": BOARD_STATE_SCHEMA,
            "symbol": symbol,
            "period": period,
            "semantics": _BOARD_STATE_SEMANTICS,
            "precision": PRECISION,
            "source_minute": _source(minute_path),
            "source_daily": _source(daily_path),
            "trade_days": result.trade_days,
            "skipped_days": result.skipped_days,
            "unassigned_bars": result.unassigned_bars,
            "row_count": int(len(result.df)),
            "file_sha256": digest,
        }

    commit_data_with_sidecar(path, data_text, build_meta=build_meta)
    return path


__all__ = [
    "BOARD_STATE_SCHEMA",
    "BOARD_STATE_SUBDIR",
    "PRECISION",
    "STATE_COLUMNS",
    "BoardStateResult",
    "BoardStateTracker",
    "attribute_windows",
    "board_state_filename",
    "build_board_states",
    "render_board_state_csv",
    "save_board_states",
]

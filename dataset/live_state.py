"""``dataset state-now`` 快照 builder（state-now-view 任务，设计见
``artifacts/state-now-view/02-design/tech-design.md``）。

**性质**：纯函数——同输入逐字节同输出；不读文件、不碰网络、无墙钟、无随机。
所有数据由调用方喂入（在线 ``provider.fetch_recent`` / 离线 ``load_ohlcv``），
**防泄漏上界 = ``decision_index`` 切片**：本模块对 ``bars_1m.iloc[: decision_index+1]``
之后的内容不读取（AC1「喂入未来 bar 输出不变」在构造层成立）。

**与训练路径的关系**（复用而非重写）：

* 状态文本 = :func:`dataset.market_episode.nanojev_records.render_state`(v12 模板）
  原样复用；参数透传方式与 ``build_record`` 逐项同构，仅两处替换——
  ``reference_bar`` = **今日首根**（快照分母，训练片段 = 片段首根），账户初值
  ``position=None / net_value=100 / today_pnl=0 / drawdown=0``（空仓起始态）。
* 趋势上下文 = ``build_usable_trend_context``（v9 ``_usable_trend_context`` 的
  纯增量公开别名）。
* question/candidates = E3 同构（``FLAT_CRITERIA`` 空仓候选，不新增文案，
  ``QUESTION_SCHEMA`` 不变）。
* 窗口归属 = ``dataset.board_state.attribute_windows``（夜盘归属同口径）。

**快照口径差异（有意为之，写入 metadata ``denominator_note``）**：分母 = 今日首根
开盘价（训练片段分母 = 片段首根开盘价）；两者最终统一是预测程序任务的研究决策，
本组件不锁死（D2）。

**T 日线行号锚定（A2 + 协调者拍板 A）**：折点 CSV 的 ``bar_index`` 是**生成时**
1d 文件的 0 基行号；快照喂入的 1d 窗口（在线 ``--daily-bars``）起点可能深于折点
生成窗口 → 直接用 ``bar_index`` 会整体偏移。同源锚定公式::

    daily_index_t = (pos(T) - pos(锚)) + 锚.bar_index        # T 行存在
    daily_index_t = (pos(prev) - pos(锚)) + 锚.bar_index + 1 # T 行缺失（A2 推导）

锚 = 折点序列首点（``kind="start"``、``bar_index=0`` 恒成立，守卫不满足即硬错误）；
``daily_rows`` 首日 == 折点生成窗口首日时 ``pos(锚)=0``，公式退化为设计字面口径
（直接取行号）。``pos`` = 行在**本次喂入** ``daily_rows`` 中的 0 基位置，
锚日期不在窗口内 → ``DatasetError``（指引增大 ``--daily-bars``）。

**夜盘快照的交易日归属（协调者拍板的未决点 2）**：``attribute_windows`` 缺省从
帧内日盘根推导交易日；在线夜盘快照（日历日 D 21:00–23:00，归属次一交易日 T）时
T 的日盘根必然不在前缀内 → 缺省推导下 T 永远无法入窗。本模块向其显式传入完整
交易日全集 = ``daily_rows`` 日期 ∪ 帧内日盘根日历日 ∪ 尾段夜盘隐含次日（仅当某夜
盘日历日在全集内无严格后续交易日时补其后一日历日）。全部数值输出对该隐含键不
敏感（T 行号走 A2 推导、prev/趋势过滤只用严格早于该键的数据）；当真实次一交易
日无法由数据确定（节假日/周末前的夜盘且日线不揭示后继行）时，隐含键可能为非
交易日历日，仅影响 ``metadata.trade_date`` 标签，语义边界记录于
``artifacts/state-now-view/03-code/change-report.md`` 设计缺口处置节。

**消费边界（D3）**：预测程序只应 import :func:`build_live_state`，喂流式
1m/1d + ``decision_index``；数值语义在 ``metadata`` 中，不应解析 ``state_text``
反推数值。``metadata`` 不含 ``data_source``（online/offline 由 CLI 层打印时插入）。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from typing import Any, Mapping

import pandas as pd

from dataset.board_state import DAY_END, NIGHT_START, PRECISION, attribute_windows
from dataset.errors import DatasetError
from dataset.market_episode.linkage import (
    align_bars_by_timestamp,
    breakthrough_momentum,
    linkage_breakthrough_momentum,
    pearson_correlation,
    signal_pairs_from_aligned,
    resample_bars,
)
from dataset.market_episode.nanojev_records import (
    FLAT_CRITERIA,
    QUESTION_ID,
    QUESTION_INSTRUCTIONS,
    BoardStateValues,
    build_usable_trend_context,
    render_state,
)
from dataset.market_episode.replay import Bar, bars_from_frame
from dataset.market_episode.segments import EpisodeParams
from dataset.periods import resolve_duration_seconds
from dataset.turning_points import TurningPoint

__all__ = [
    "DENOMINATOR_NOTE",
    "LiveStateSnapshot",
    "build_live_state",
]

#: 快照分母口径提示（metadata + CLI stdout 尾行共用；D2 拍板文案，不新增措辞）
DENOMINATOR_NOTE = "快照口径（分母=今日首根开盘）≠ 训练片段口径（分母=片段首根开盘）"

#: 账户初值（净值尺度 100 基，与训练展示口径同；D4）
_NET_VALUE_INITIAL = 100.0


@dataclass(frozen=True)
class LiveStateSnapshot:
    """state-now 快照（frozen；同输入双跑全等）。"""

    #: v12 状态文本（七段原文，``STATE_SCHEMA = marketsense.episode_state.v12``）
    state_text: str
    #: choice 题（结构同训练记录 ``questions``，单题 ``next_action``）
    question: dict[str, Any]
    #: 空仓候选 ``动作 → 文案``（``FLAT_CRITERIA`` 拷贝）
    candidates: dict[str, str]
    #: 结构化元数据（JSON 可序列化，键序固定，见模块 docstring 与设计 §3.5）
    metadata: dict[str, Any]


def _linkage_display_name(symbol: str) -> str:
    """联动品种显示名 = 去交易所前缀原样保留（与 ``_linkage_display_name`` 同规则）。"""
    return symbol.split(".", 1)[1] if "." in symbol else symbol


def _iso(bar: Bar) -> str:
    return pd.Timestamp(bar.timestamp).isoformat()


def _daily_by_date(
    daily_rows: pd.DataFrame,
) -> dict[dt.date, tuple[int, float, float, float]]:
    """1d 窗口 → ``日期 → (行位置, high, low, close)``（文件原序，board_state 同口径）。"""
    table: dict[dt.date, tuple[int, float, float, float]] = {}
    for pos, (day, high, low, close) in enumerate(
        zip(
            daily_rows["timestamp"].dt.date,
            daily_rows["high"],
            daily_rows["low"],
            daily_rows["close"],
        )
    ):
        table[day] = (pos, float(high), float(low), float(close))
    return table


def _explicit_trading_days(prefix_df: pd.DataFrame, daily_rows: pd.DataFrame) -> set[dt.date]:
    """夜盘快照的显式交易日全集（未决点 2，见模块 docstring）。

    = ``daily_rows`` 日期（已知真实交易日）∪ 帧内日盘根日历日（真实交易日，
    防 1d 窗口浅于 1m 前缀时的归属键缺失）∪ 尾段夜盘隐含次日：对每个夜盘日历日
    ``D``，若全集内不存在严格大于 ``D`` 的交易日，则补 ``D + 1 日历日``（保证末段
    夜盘根必有归属窗口；降序处理保证最后一段夜盘优先获得隐含键）。隐含键按构造
    与已知交易日不相交且彼此互异，不会污染既有窗口的归属。
    """
    ts = prefix_df["timestamp"]
    dates: list[dt.date] = list(ts.dt.date)
    tods = [t.time() for t in ts]
    trading_days: set[dt.date] = set(daily_rows["timestamp"].dt.date)
    trading_days.update(d for d, tod in zip(dates, tods) if tod < DAY_END)
    night_dates = sorted(
        {d for d, tod in zip(dates, tods) if tod >= NIGHT_START}, reverse=True
    )
    for day in night_dates:
        if not any(existing > day for existing in trading_days):
            trading_days.add(day + dt.timedelta(days=1))
    return trading_days


def _prev_day_row(
    daily_by_date: dict[dt.date, tuple[int, float, float, float]],
    trade_date: dt.date,
) -> tuple[dt.date, float, float, float]:
    """严格早于 ``trade_date`` 的最近日线行 (high, low, close)；缺失 → ``DatasetError``。"""
    earlier = [day for day in daily_by_date if day < trade_date]
    if not earlier:
        raise DatasetError(
            f"1d 窗口缺少 {trade_date} 之前的交易日行（prev_day_* 不可得）；"
            "请执行 `python -m dataset fetch --period 1d` 补足日线数据"
        )
    prev = max(earlier)
    _, high, low, close = daily_by_date[prev]
    return prev, high, low, close


def _daily_index_t(
    daily_by_date: dict[dt.date, tuple[int, float, float, float]],
    trade_date: dt.date,
    anchor: TurningPoint,
) -> tuple[int, bool]:
    """T 的 1d 行号（拍板 A 锚定公式）+ 是否为 A2 推导（T 行缺失）。

    :raises DatasetError: ``daily_rows`` 末行晚于 trade_date（错窗）；
        锚日期不在 1d 窗口内（指引增大 ``--daily-bars``）
    """
    last_date = max(daily_by_date)
    if last_date > trade_date:
        raise DatasetError(
            f"1d 窗口末行 {last_date} 晚于快照交易日 {trade_date}（数据陈旧/错窗）；"
            "请核对 1d 与 1m 数据的时间范围"
        )
    anchor_pos = daily_by_date.get(anchor.timestamp.date())
    if anchor_pos is None:
        raise DatasetError(
            f"折点首点日期 {anchor.timestamp.date().isoformat()} 不在 1d 窗口内"
            f"（窗口 {min(daily_by_date)}~{last_date}）；折点 CSV 与 1d 窗口必须同源，"
            "请增大 --daily-bars 或重生成折点 CSV（`python -m dataset turning-points "
            "--period 1d`）"
        )
    row = daily_by_date.get(trade_date)
    if row is not None:
        # T 行存在：锚定公式退化为「直接行号」当且仅当锚即窗口首行（pos=0）
        return (row[0] - anchor_pos[0]) + anchor.bar_index, False
    # T 行缺失（在线 1d 只含已收盘日线）：推导 = 前一交易日行号 + 1（T 开盘即知，无未来）
    earlier = [day for day in daily_by_date if day < trade_date]
    if not earlier:
        raise DatasetError(
            f"1d 窗口缺少 {trade_date} 之前的交易日行（无法推导 T 行号）；"
            "请执行 `python -m dataset fetch --period 1d` 补足日线数据"
        )
    prev_pos = daily_by_date[max(earlier)][0]
    derived = (prev_pos - anchor_pos[0]) + anchor.bar_index + 1
    return derived, True


def build_live_state(
    *,
    symbol: str,
    bars_1m: pd.DataFrame,
    decision_index: int,
    daily_rows: pd.DataFrame,
    turning_points: tuple[TurningPoint, ...],
    linkage_bars: Mapping[str, pd.DataFrame | None] | None = None,
    params: EpisodeParams = EpisodeParams(),
    price_precision: int = PRECISION,
) -> LiveStateSnapshot:
    """构造「此刻」的 state/question/candidates/metadata 快照（纯函数）。

    :param symbol: 主品种（``交易所.合约``，显示名去前缀）。
    :param bars_1m: 标准 OHLCV 1m 帧（升序、tz-aware；可含 T 之后的行，**不被读取**）。
    :param decision_index: T（刚收盘那根 1m）在 ``bars_1m`` 中的 0 基位置；
        显式切片防泄漏——此后内容不读取。在线调用方传 ``len(frame)-1``。
    :param daily_rows: 标准 OHLCV 1d 帧（升序；在线只含已收盘日线 → T 行可能缺失）。
    :param turning_points: 日线折点序列（``load_turning_points(symbol, "1d")``）；
        首点必须为 ``kind="start"``、``bar_index=0``（锚定前提，守卫不满足硬错误）。
    :param linkage_bars: 联动品种 → 1m 帧（缺数据/取数失败传 ``None``）；
        未配置联动品种时传 ``None``。
    :param params: episode 参数（只用 ``breakthrough_window/breakthrough_period/
        linkage_symbols``）。
    :param price_precision: 模型可见比值小数位（缺省 6，与训练同口径 E11；
        CLI 传 ``params.price_precision`` 以透传配置）。
    :returns: :class:`LiveStateSnapshot`。
    :raises DatasetError: 越界 / T 未入窗（15:00–20:59）/ 日线缺前置行 /
        折点与 1d 不同源 / 折点不足（up/down 各 <2）/ 联动品种帧非法。
    """
    if not 0 <= decision_index < len(bars_1m):
        raise DatasetError(
            f"decision_index 越界: {decision_index}（bars_1m 共 {len(bars_1m)} 根，"
            "要求 0 <= decision_index < 根数）"
        )

    # ① 防泄漏切片：此后只看前缀（>T 的行不被读取，AC1）
    prefix_df = bars_1m.iloc[: decision_index + 1]
    bars = bars_from_frame(prefix_df)

    # ② 交易日窗口归属（夜盘归属与 board_state 同口径；15:00–20:59 未入窗 → 硬错误）。
    # 在线夜盘快照 T 的日盘根不在前缀内 → 显式传入完整交易日全集（未决点 2）
    windows, _ = attribute_windows(
        prefix_df, trading_days=sorted(_explicit_trading_days(prefix_df, daily_rows))
    )
    trade_date = next(
        (day for day, positions in windows.items() if decision_index in positions),
        None,
    )
    if trade_date is None:
        raise DatasetError(
            f"决策 K 线 time-of-day 在 15:00–20:59，无法归属交易日: "
            f"{bars[-1].timestamp!r}（快照不产出，不静默）"
        )
    today_positions = windows[trade_date]
    bar_index_in_today = len(today_positions) - 1

    # ③ 分母基准（D2）：今日首根；T 重定基到今日窗口内序号（render 只用 index/timestamp/值）
    reference_bar = bars[today_positions[0]]
    bar_t = replace(bars[-1], index=bar_index_in_today)

    # 截断判定（A3）：帧首根仍属 T 窗口且时刻 > 21:00 → 夜盘起点未被覆盖
    first_in_window_at_frame_start = 0 in today_positions
    first_tod = pd.Timestamp(bars[0].timestamp).time()
    truncated = first_in_window_at_frame_start and first_tod > NIGHT_START

    # ④⑤ 日线层：prev_day_* + T 行号（拍板 A 锚定）；排序防御（升序输入下位置不变）
    daily_rows = daily_rows.sort_values("timestamp").reset_index(drop=True)
    if daily_rows.empty:
        raise DatasetError(
            "1d 窗口为空（prev_day_* 与趋势时长不可得）；"
            "请执行 `python -m dataset fetch --period 1d`"
        )
    daily_by_date = _daily_by_date(daily_rows)
    prev_date, prev_high, prev_low, prev_close = _prev_day_row(
        daily_by_date, trade_date
    )

    if not turning_points:
        raise DatasetError(
            "日线折点序列为空；请生成折点 CSV（`python -m dataset turning-points "
            "--period 1d`）"
        )
    anchor = turning_points[0]
    if anchor.kind != "start" or anchor.bar_index != 0:
        raise DatasetError(
            f"折点首点不是 start/bar_index=0（kind={anchor.kind!r}, "
            f"bar_index={anchor.bar_index}）；折点 CSV 与 1d 文件必须同源，"
            "请重生成折点 CSV"
        )
    daily_index_t, daily_row_derived = _daily_index_t(daily_by_date, trade_date, anchor)

    # ⑥ 趋势上下文（v9 语义同源复用）；None = 折点不足 → 快照无法跳过，硬报错
    trend_context = build_usable_trend_context(
        turning_points, trade_date, daily_index_t
    )
    if trend_context is None:
        raise DatasetError(
            f"日线折点趋势上下文不可用（可用 up/down 折点各需 ≥2 且确认日严格早于 "
            f"{trade_date.isoformat()}）；请检查/重生成折点 CSV（`python -m dataset "
            "turning-points --period 1d`）或增大 --daily-bars 后重试"
        )

    # ⑦ 盘面状态绝对值（今日极值只由 ≤T 的今日窗口根累计，天然无未来）
    board_state = BoardStateValues(
        prev_day_high=prev_high,
        prev_day_low=prev_low,
        prev_day_close=prev_close,
        today_high=max(bars[i].high for i in today_positions),
        today_low=min(bars[i].low for i in today_positions),
    )

    # ⑦' 联动品种（D4）：缺数据 → 空元组（render 内 na 语义原样生效），meta 记录可用性
    links = dict(linkage_bars) if linkage_bars else {}
    linkage_symbol_bars: dict[str, tuple[Bar, ...]] = {}
    linkage_meta: list[dict[str, Any]] = []
    for link_symbol in params.linkage_symbols:
        frame = links.get(link_symbol)
        link_bars = bars_from_frame(frame) if frame is not None and len(frame) else ()
        linkage_symbol_bars[link_symbol] = link_bars
        linkage_meta.append(
            {
                "symbol": link_symbol,
                "available": len(link_bars) > 0,
                "first_timestamp": _iso(link_bars[0]) if link_bars else None,
                "last_timestamp": _iso(link_bars[-1]) if link_bars else None,
            }
        )

    # ⑧⑨ 复算 na 成因（只复算成因，不重写序列化；公开辅助与 render_state 同源）
    duration_seconds = resolve_duration_seconds(params.breakthrough_period)
    na_details: list[str] = []
    closed_count = (
        len(bars)
        if duration_seconds == 60
        else len(
            [
                bucket
                for bucket in resample_bars(bars, duration_seconds)
                if bucket.is_closed
            ]
        )
    )
    if breakthrough_momentum(bars, params.breakthrough_window, duration_seconds) is None:
        na_details.append(
            f"主突破=na：1m 可用根数 {min(params.breakthrough_window, closed_count)} <2"
        )
    if linkage_symbol_bars:
        first_link = params.linkage_symbols[0]
        aligned = align_bars_by_timestamp(
            bars, linkage_symbol_bars[first_link], params.breakthrough_window
        )
        if linkage_breakthrough_momentum(aligned) is None:
            na_details.append(
                f"{_linkage_display_name(first_link)} 突破=na：交集对 {len(aligned)} <2"
            )
        signal_pairs = signal_pairs_from_aligned(aligned)
        correlation = pearson_correlation(
            [primary for primary, _ in signal_pairs],
            [secondary for _, secondary in signal_pairs],
        )
        if correlation is None:
            na_details.append(
                f"{_linkage_display_name(first_link)} 相关度=na："
                f"有效信号对 {len(signal_pairs)} <2 或零方差"
            )

    # ⑩ 状态文本：参数透传与 build_record 逐项同构（reference_bar/账户值按 D2/D4 替换）
    state_text = render_state(
        bar=bar_t,
        reference_bar=reference_bar,
        position=None,
        drawdown=0.0,
        net_value=_NET_VALUE_INITIAL,
        today_pnl=0.0,
        price_precision=price_precision,
        board_state=board_state,
        trend_context=trend_context,
        breakthrough_bars=bars,
        breakthrough_window=params.breakthrough_window,
        breakthrough_duration_seconds=duration_seconds,
        primary_symbol=symbol,
        linkage_symbol_bars=linkage_symbol_bars,
    )

    # ⑪ metadata（键序固定，设计 §3.5；全部由输入与中间量填充，无墙钟无随机）
    metadata: dict[str, Any] = {
        "symbol": symbol,
        "decision_bar_timestamp": _iso(bars[-1]),
        "trade_date": trade_date.isoformat(),
        "reference_bar_timestamp": _iso(reference_bar),
        "reference_open": reference_bar.open,
        "price_precision": price_precision,
        "bar_index_in_today": bar_index_in_today,
        "today_window": {
            "first": _iso(bars[today_positions[0]]),
            "last": _iso(bars[today_positions[-1]]),
            "bars": len(today_positions),
            "truncated": truncated,
        },
        "daily_rows": {
            "first": min(daily_by_date).isoformat(),
            "last": max(daily_by_date).isoformat(),
            "rows": len(daily_by_date),
            "daily_row_derived": daily_row_derived,
        },
        "prev_day_date": prev_date.isoformat(),
        "breakthrough": {
            "window": params.breakthrough_window,
            "period": params.breakthrough_period,
            "duration_seconds": duration_seconds,
        },
        "linkage_symbols": linkage_meta,
        "na_details": na_details,
        "denominator_note": DENOMINATOR_NOTE,
    }
    return LiveStateSnapshot(
        state_text=state_text,
        question={
            QUESTION_ID: {
                "type": "choice",
                "instructions": QUESTION_INSTRUCTIONS,
                "criteria": dict(FLAT_CRITERIA),
            }
        },
        candidates=dict(FLAT_CRITERIA),
        metadata=metadata,
    )

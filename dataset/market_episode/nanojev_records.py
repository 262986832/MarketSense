"""NanoJev 训练记录生成：记录映射 + 状态序列化 + split 分配 + 原子落盘。

**记录契约**（对齐第三方 ``NanoJev/scripts/train_pipeline_decisions.py`` 的
``validate_training_row`` / ``read_training_records``；NanoJev 全程只读）：

* ``id`` = ``state_id`` = ``{segment_id}:{决策 K 线片段内序号}``（全局唯一，天然不跨 split）；
* ``family_id`` = ``metadata.source_group_id`` = ``segment_id``（一个片段 = 一个 episode，
  直接复用 NanoJev 的 state/source_group 跨 split 泄漏检查）；
* ``split`` = 片段的 ``split_role``；
* ``state`` = 确定性文本序列化的**相对比值**状态（v8 六部分：账户（六键：持仓 /
  [开仓价/止损价] / 净值 / 今日 / 回撤）/ 日线（上一交易日高/低/收 + 日线折点趋势极值）/
  日内（今日高/低）/
  联动（na 占位）/ 现价（决策 K 线 OHLC + bar 序号 + 量/持仓量比值）/ 盘口（na 占位））；
* ``questions`` 恰好一个 choice 题 ``next_action``：空仓
  ``{open_long, open_short, stay_flat}`` / 持仓 ``{close, hold, reverse}``；
  候选文案只留动作语义（买入开仓/卖出开仓/继续空仓/平仓/继续持有/反手），
  成交价位由执行程序与滑点决定，不进模型输入（2026-10-02 用户拍板精简）；
* ``gold`` = 规则真值动作 ID + ``gold_label_kind: "deterministic_truth"``
  （首轮不发 ``gold_probs``：NanoJev 校验器允许硬 gold 无概率，trainer 自动派生 one-hot）。

**状态序列化模板（实现阶段冻结项 2；v2 起新增盘面状态行，2026-10-02 用户拍板方案 A；
v3 起行间换行符前后加空格，转义后的 JSON 文本更易读，同日拍板；v4 起行重排为宏观→微观
六部分（账户/联动/日线/日内/现价/盘口），持仓值中文化（空仓/持多/持空）+ 盘口 ``na``
占位，数值语义与 v3 逐项等价，同日用户拍板；v5 起「日线」行新增 4 个 trend 键
（最近可确认 up/down 折点的段极值 + 段长，同日拍板）；v6 起行重排为
账户/日线/日内/联动/现价/盘口（联动行移至日内行之后），
行内键与数值语义与 v5 逐项等价，同日用户拍板；v7 起：①训练 state 文本键名中文化
（昨日高/昨日低/昨日收/今高/今低/开/高/低/收，持仓时账户行 entry/stop → 开仓价/止损价），
②日内行 ``bar`` 序号与联动行量/持仓量比值并入现价行（持仓量只保留收盘时刻，
开盘时刻丢弃），③联动行变为 ``na`` 常量占位（同盘口行约定），
trend 四键本次不改（后续随折点信息补充一并调整），2026-10-02 用户拍板；
v8 起账户行升六键（持仓/[开仓价/止损价]/净值/今日/回撤；净值/今日为净值尺度新值、
由调用方只用 ≤ 决策 K 线的数据算好传入，跨片段净值链由 ``generate_dataset``
时间序串行回放接入，2026-10-03 用户拍板，
见 ``artifacts/account-service/02-design/tech-design.md``）**

```text
marketsense.episode_state.v8
账户: 持仓=空仓 净值=<..> 今日=<..> 回撤=<..>
日线: 昨日高=<..> 昨日低=<..> 昨日收=<..> trend_up=<..> trend_up_len=<n> trend_dn=<..> trend_dn_len=<n>
日内: 今高=<..> 今低=<..>
联动: na
现价: 开=<..> 高=<..> 低=<..> 收=<..> bar=<片段内 0 基序号> 成交量比=<..> 持仓量比=<..>
盘口: na
```

（行与行之间用 ``" \n "`` 连接，即换行符前后各一个空格。持仓非空时账户行为：
``账户: 持仓=持多|持空 开仓价=<..> 止损价=<..> 净值=<..> 今日=<..> 回撤=<..>``。）

* 比值分母 = **片段首根**（价格用首根开盘价，量/持仓量用首根同名列），小数位固定
  （默认 6）；分母 ≤ 0 时写 ``na``（不产生 ``inf``/绝对数）；
* **无历史窗口**：状态只含决策 K 线单根 + 仓位 + 净值 + 今日 + 回撤 + 盘面状态
  （v4 拆「日线」「日内」两行，v5 在日线行追加折点趋势项，v7 键名中文化并将量/持仓量比
  与 bar 序号并入现价行，v8 账户行升六键加净值/今日），绝对价格不进入模型输入；
  「盘口」与「联动」行为 ``na`` 常量占位
  （真实 bid/ask 未接，非目标；v7 起量/持仓量比已并入现价行，联动行不再承载实际值）；
* **账户行**（v4，v8 升六键）：持仓值中文标签（空仓/持多/持空，未知方向抛
  ``DatasetError`` 不静默）；v8 起键序固定为 持仓/[开仓价/止损价]/净值/今日/回撤
  （磁盘键名沿用 v7 的 ``开仓价/止损价``，内部字段 entry/stop；空仓时无开仓价/止损价
  两键，沿用 v7 先例）；``净值/今日``（render_state 参数 net_value/today_pnl，
  ``DecisionPoint`` 同名字段）为净值尺度新值（初值 100、今日收益每片段重置、比值化记账），
  ``回撤`` 公式不变；三值均由调用方只用 ≤ 决策 K 线的数据算好传入
  （跨片段净值链由 ``generate_dataset`` 时间序串行回放接入，见
  ``artifacts/account-service/02-design/tech-design.md``）；
* **日线行**（原 v2 board_state 行 prev 部分，v4 拆出，v5 追加 trend 四值，v7 键名中文化；
  prev 与 ``dataset/board_state.py`` 同口径）：
  - ``昨日高/昨日低/昨日收``（内部字段 prev_h/prev_l/prev_c）= **上一交易日**日线高/低/收
    （来源 ``{symbol}_1d.csv``，
    取日线文件中严格早于片段交易日的最后一行）÷ 片段首根开盘价；
  - ``trend_up``/``trend_dn`` = 最近**可确认** up/down 折点的段内实际最高/最低价
    （``trend_extreme_price``，来源 ``data/turning_points/{symbol}_1d.csv``）÷ 片段首根开盘价；
    ``trend_up_len``/``trend_dn_len`` = 对应趋势段长（整数 K 线数量，由点序列确定性推导）；
  - 「可确认」口径：折点确认根日期**严格早于**该 bar 的**决策交易日**（State(T) 不得引用
    T 日及之后确认的折点，无未来泄漏）；决策交易日归属与 ``board_state.attribute_windows``
    夜盘规则同口径（日盘 bar → 其日历日；夜盘 bar（≥ 21:00）→ 其后下一个交易日；
    ``[15:00, 21:00)`` 或夜盘无下一交易日 → ``DatasetError`` 不静默）；
  - 逐决策点各自取其决策交易日的可用最近折点（多日片段的后段决策点能看到后确认的折点）；
  - 任一入选决策点的可用 up/down 折点单侧缺失 → **跳过整个片段**（不产出记录），记入审计
    ``trend_extremes.skipped_segments`` 与 stderr 告警（不静默）；全部片段被跳过 → 硬错误；
* **日内行**（原 v2 board_state 行 today 部分，v4 拆出；v7 起 ``bar=`` 序号移入现价行，
  键名中文化）：
  - ``今高/今低``（内部字段 today_h/today_l）= 片段首根至决策 K 线（**含**）的 1m 高/低
    **累计极值** ÷ 片段首根开盘价
    （State(T) 只用 ≤ 决策 K 线的数据，无未来泄漏）；
  - 分母与现价行一致（全交易日片段下片段首根 = 交易日窗口首根 = 今日开盘，
    见 ``artifacts/nanojev-integration-alignment/01-requirement/requirement-report.md``）；
  - 今日开盘价不写（它是分母本身，比值恒为 1.000000，写入是纯冗余 token）；
  - 片段无上一交易日日线 → **跳过该片段**（不产出记录），记入审计与 stderr 告警（不静默）。
* **输入目录**：日线折点默认取 ``<data_dir>/../turning_points``（与 turning-points 子命令
  落盘布局一致；``--daily-turning-points-dir`` 可覆盖）；文件缺失/损坏 → ``DataLoadError``
  硬报错（文件缺失 ≠ 折点缺失：文件在但无可用折点 → 跳过片段）。
* 输出按 ``run`` 目录隔离（``<output_dir>/<run_id>``，``run_id`` 由输入指纹确定性派生，
  同输入同目录同字节，不覆盖其它输入的产物）。
"""

from __future__ import annotations

import bisect
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from dataset.board_state import DAY_END, NIGHT_START
from dataset.errors import DatasetError
from dataset.storage import file_sha256, load_ohlcv
from dataset.turning_points import (
    TrendExtreme,
    load_turning_points,
    recent_trend_extremes,
)

from dataset.market_episode.audit import (
    AccountReplayInputs,
    NET_VALUE_BASE,
    build_audit_payload,
    check_account_chain,
    check_audit_consistency,
    check_records,
    check_state_leakage,
)
from dataset.market_episode.labels import (
    ACTION_CLOSE,
    ACTION_HOLD,
    ACTION_OPEN_LONG,
    ACTION_OPEN_SHORT,
    ACTION_REVERSE,
    ACTION_STAY_FLAT,
    DecisionPoint,
    SegmentOutcome,
    evaluate_segment,
)
from dataset.market_episode.replay import (
    Bar,
    LONG,
    PositionSnapshot,
    SHORT,
    ReplayAccount,
    TradeEvent,
    format_ratio_value,
    load_segment_bars,
    px_ratio_line,
    ratio_or_none,
)
from dataset.market_episode.segments import (
    SPLIT_ROLES,
    EpisodeParams,
    Segment,
    validate_segments,
)

#: 状态文本的 schema 版本标记（格式演进必须换标记；v2 新增盘面状态行；v3 行间换行符
#: 前后加空格——转义后的 JSON 文本更易读；v4 行重排为宏观→微观六部分（账户/联动/日线/
#: 日内/现价/盘口）+ 持仓值中文化 + 盘口 na 占位，数值语义与 v3 逐项等价，2026-10-02 用户拍板；
#: v5 日线行新增日线折点趋势极值项 trend_up/trend_up_len/trend_dn/trend_dn_len
#: （确认根日期严格早于决策交易日的最近 up/down 折点段极值 + 段长，2026-10-02 用户拍板）；
#: v6 行重排（账户/日线/日内/联动/现价/盘口）——联动行移至日内行之后，行内键与
#: 数值语义与 v5 逐项等价，2026-10-02 用户拍板；
#: v7 训练 state 文本键名中文化（昨日高/昨日低/昨日收/今高/今低/开/高/低/收，持仓时
#: entry/stop → 开仓价/止损价），日内行 bar 序号与联动行量/持仓量比值并入现价行
#: （持仓量只保留收盘时刻），联动行变为 na 常量占位，trend 四键本次不改，
#: 2026-10-02 用户拍板；
#: v8 账户行升六键（持仓/[开仓价/止损价]/净值/今日/回撤；键序固定，磁盘键名沿用 v7 的
#: 开仓价/止损价；净值/今日为净值尺度新值（初值 100、今日收益每片段重置），回撤公式
#: 不变；三值由调用方只用 ≤ 决策 K 线的数据算好传入，跨片段净值链由 generate_dataset
#: 时间序串行回放接入；v8 在 v7-chinese-keys 基础上扩展，2026-10-03 用户拍板，
#: 见 artifacts/account-service/02-design/tech-design.md）
STATE_SCHEMA = "marketsense.episode_state.v8"
#: questions 文本的 schema 版本标记（首次建立；候选文案演进必须换标记）
QUESTION_SCHEMA = "marketsense.episode_question.v1"
#: 记录中的 choice 题目 ID
QUESTION_ID = "next_action"
#: 题目说明（确定性固定文案）
QUESTION_INSTRUCTIONS = "选择下一分钟要执行的动作（gold 为确定性规则真值）。"
#: 空仓候选（动作集合依仓位而变：模型必须知晓仓位）。
#: 文案只留动作语义；成交价位由执行程序与滑点决定（2026-10-02 用户拍板精简）
FLAT_CRITERIA: Mapping[str, str] = {
    ACTION_OPEN_LONG: "买入开仓",
    ACTION_OPEN_SHORT: "卖出开仓",
    ACTION_STAY_FLAT: "继续空仓",
}
#: 持仓候选（文案同上：只留动作语义，价位/执行细节不进模型输入）
HELD_CRITERIA: Mapping[str, str] = {
    ACTION_CLOSE: "平仓",
    ACTION_HOLD: "继续持有",
    ACTION_REVERSE: "反手",
}
#: 账户行「持仓=」值映射（v4：flat（代码中为 ``None``）→空仓、long→持多、short→持空；
#: 未知方向抛 ``DatasetError``，保持确定性不静默）
POSITION_LABELS: Mapping[str | None, str] = {
    None: "空仓",
    LONG: "持多",
    SHORT: "持空",
}
#: gold 的依据类型（冻结首轮形式）
GOLD_LABEL_KIND = "deterministic_truth"
#: 审计文件名
AUDIT_FILENAME = "audit.json"


@dataclass(frozen=True)
class GenerationResult:
    """一次生成的产物信息（供 CLI 打印与测试断言）。"""

    run_id: str
    run_dir: Path
    record_counts: Mapping[str, int]
    output_sha256: Mapping[str, str]
    audit: Mapping[str, Any]
    board_state_skipped: Mapping[str, str]
    trend_extreme_skipped: Mapping[str, str]


@dataclass(frozen=True)
class BoardStateValues:
    """盘面状态绝对值（序列化时再除以片段首根开盘价）。

    * ``prev_day_*``：上一交易日日线高/低/收（来源 ``{symbol}_1d.csv``，绝对价格层）；
    * ``today_*``：片段首根至决策 K 线（含）的 1m 高/低累计极值（State(T) 只用 ≤ 决策 K 线数据）。
    """

    prev_day_high: float
    prev_day_low: float
    prev_day_close: float
    today_high: float
    today_low: float


def _bar_trade_date(bars: tuple[Bar, ...], bar: Bar) -> dt.date:
    """决策 K 线的**决策交易日**归属（与 ``board_state.attribute_windows`` 夜盘归属同口径）。

    * 日盘 bar（time-of-day < 15:00）→ 其日历日；
    * 夜盘 bar（time-of-day ≥ 21:00）→ 片段交易日集合（片段内日盘 bar 的日历日，升序）
      中该日历日之后的**下一个交易日**；
    * ``[15:00, 21:00)`` 或夜盘无下一交易日 → 片段窗口内不应出现，出现即清单/数据异常 →
      ``DatasetError``（不静默，与「混序数据必须报错」同风格）。
    """
    stamp = pd.Timestamp(bar.timestamp)
    calendar_day = stamp.date()
    time_of_day = stamp.time()
    if time_of_day < DAY_END:
        return calendar_day
    if time_of_day < NIGHT_START:
        raise DatasetError(
            f"决策 K 线 time-of-day 在 15:00–20:59，无法归属交易日: {bar.timestamp!r}"
        )
    trading_days = sorted(
        {
            pd.Timestamp(item.timestamp).date()
            for item in bars
            if pd.Timestamp(item.timestamp).time() < DAY_END
        }
    )
    position = bisect.bisect_right(trading_days, calendar_day)
    if position >= len(trading_days):
        raise DatasetError(
            f"夜盘决策 K 线在片段内找不到后续交易日，无法归属交易日: {bar.timestamp!r}"
        )
    return trading_days[position]


def _usable_trend_extremes(
    points: tuple[TurningPoint, ...], trade_date: dt.date
) -> tuple[TrendExtreme, TrendExtreme] | None:
    """决策交易日 ``trade_date`` 可用的最近 1 个 up + 1 个 down 折点段极值。

    「可用」= 确认根日期（``timestamp`` 日历日）**严格早于** ``trade_date``（State(T)
    不得引用 T 日及之后确认的折点，无未来泄漏）。先过滤后检查：up/down 任一侧缺失
    是**正常边界**（调用方走跳过路径），返回 ``None`` 而不抛异常；两侧齐全时复用
    :func:`recent_trend_extremes`（语义同源：段闭区间、平局取最早、时间倒序、段长
    由点序列推导）。过滤后序列为 up/down 前缀，段长推导与全序列一致。

    :raises DatasetError: 可用 up/down 点缺极值字段（文件损坏，硬错误不静默）
    """
    usable = tuple(
        point
        for point in points
        if point.kind in ("up", "down") and point.timestamp.date() < trade_date
    )
    if not any(point.kind == "up" for point in usable):
        return None
    if not any(point.kind == "down" for point in usable):
        return None
    highs, lows = recent_trend_extremes(usable, n=1)
    return highs[0], lows[0]


def _default_daily_turning_points_dir(data_dir: str | Path) -> Path:
    """日线折点默认目录：``<data_dir>/../turning_points``。

    与 ``cli.turning_points_dir`` 落盘布局一致（episode ``data_dir`` 默认
    ``<base_output>/ohlcv``，父目录即 base_output）；延迟导入避免模块级循环依赖。
    """
    from dataset.cli import turning_points_dir  # 延迟导入：cli 在模块层依赖本包

    return turning_points_dir(Path(data_dir).parent)


def _daily_line(
    board_state: BoardStateValues,
    reference_bar: Bar,
    price_precision: int,
    trend_up_extreme: TrendExtreme,
    trend_dn_extreme: TrendExtreme,
) -> str:
    """日线行（v5/v7）：上一交易日高/低/收（v7 起中文名昨日高/低/收）+ 日线折点趋势极值的比值
    （分母 = 片段首根开盘价；分母 ≤ 0 时逐值写 ``na``；段长为整数，不入比值口径）。"""
    return (
        "日线: "
        f"昨日高={format_ratio_value(ratio_or_none(board_state.prev_day_high, reference_bar.open), price_precision)}"
        f" 昨日低={format_ratio_value(ratio_or_none(board_state.prev_day_low, reference_bar.open), price_precision)}"
        f" 昨日收={format_ratio_value(ratio_or_none(board_state.prev_day_close, reference_bar.open), price_precision)}"
        f" trend_up={format_ratio_value(ratio_or_none(trend_up_extreme.trend_extreme_price, reference_bar.open), price_precision)}"
        f" trend_up_len={trend_up_extreme.segment_length}"
        f" trend_dn={format_ratio_value(ratio_or_none(trend_dn_extreme.trend_extreme_price, reference_bar.open), price_precision)}"
        f" trend_dn_len={trend_dn_extreme.segment_length}"
    )


def _intraday_line(
    board_state: BoardStateValues,
    reference_bar: Bar,
    price_precision: int,
) -> str:
    """日内行（v7）：今日高/低累计极值的比值（State(T) 只用 ≤ 决策 K 线的数据；
    ``bar`` 序号自 v7 起移入现价行，不再出现在本行）。"""
    return (
        f"日内: 今高={format_ratio_value(ratio_or_none(board_state.today_high, reference_bar.open), price_precision)}"
        f" 今低={format_ratio_value(ratio_or_none(board_state.today_low, reference_bar.open), price_precision)}"
    )


def render_state(
    *,
    bar: Bar,
    reference_bar: Bar,
    position: PositionSnapshot | None,
    drawdown: float,
    net_value: float,
    today_pnl: float,
    price_precision: int,
    board_state: BoardStateValues,
    trend_up_extreme: TrendExtreme,
    trend_dn_extreme: TrendExtreme,
) -> str:
    """确定性状态文本（v8 六部分：账户/日线/日内/联动/现价/盘口；模板见模块 docstring）。

    只做「决策 K 线单根 + 仓位 + 净值 + 今日 + 回撤 + 盘面状态 + 日线折点极值」的序列化：
    函数签名决定它无法访问决策 K 线之后的任何 bar（``board_state`` 的今日值与
    ``net_value``/``today_pnl`` 均由调用方只用 ≤ 决策 K 线的数据算好传入；
    折点极值由调用方按确认日 < 决策交易日过滤后传入，无未来数据泄漏在构造层面成立）。

    v7：联动行变为 ``na`` 常量占位（量/持仓量比值移入现价行）；``bar`` 序号移入现价行。
    v8：账户行升六键（持仓/[开仓价/止损价]/净值/今日/回撤，键序固定）。
    """
    account_values = (
        f"净值={format_ratio_value(net_value, price_precision)}"
        f" 今日={format_ratio_value(today_pnl, price_precision)}"
        f" 回撤={format_ratio_value(drawdown, price_precision)}"
    )
    if position is None:
        held_text = POSITION_LABELS[None]
    else:
        label = POSITION_LABELS.get(position.direction)
        if label is None:
            raise DatasetError(f"未知持仓方向，无法序列化账户行: {position.direction!r}")
        held_text = (
            f"{label} 开仓价={format_ratio_value(position.entry_ratio, price_precision)}"
            f" 止损价={format_ratio_value(position.stop_ratio, price_precision)}"
        )
    return " \n ".join(
        (
            STATE_SCHEMA,
            f"账户: 持仓={held_text} {account_values}",
            _daily_line(
                board_state,
                reference_bar,
                price_precision,
                trend_up_extreme,
                trend_dn_extreme,
            ),
            _intraday_line(board_state, reference_bar, price_precision),
            "联动: na",
            px_ratio_line(bar, reference_bar, price_precision),
            "盘口: na",
        )
    )


def build_record(
    *,
    segment: Segment,
    bars: tuple[Bar, ...],
    point: DecisionPoint,
    price_precision: int,
    prev_day_ohlc: tuple[float, float, float],
    trend_up_extreme: TrendExtreme,
    trend_dn_extreme: TrendExtreme,
) -> dict[str, Any]:
    """把一个入选决策点映射为 NanoJev 训练记录。

    ``prev_day_ohlc`` = 上一交易日日线 (高, 低, 收)（来源 1d 文件，绝对价格层）；
    ``trend_up_extreme``/``trend_dn_extreme`` = 该决策交易日可用的最近 up/down 折点段极值
    （调用方按确认日 < 决策交易日过滤后传入，无未来数据泄漏）；
    ``today_high/today_low`` 只由 ``bars[: point.bar_index + 1]``（≤ 决策 K 线）累计，
    无未来数据泄漏；``point.net_value/point.today_pnl``（v8 账户行「净值/今日」）同样由
    调用方（账户回放）只用 ≤ 决策 K 线的数据算好透传（跨片段净值链由后续任务接入）。
    """
    bar = bars[point.bar_index]
    prefix = bars[: point.bar_index + 1]
    state = render_state(
        bar=bar,
        reference_bar=bars[0],
        position=point.position,
        drawdown=point.drawdown,
        net_value=point.net_value,
        today_pnl=point.today_pnl,
        price_precision=price_precision,
        board_state=BoardStateValues(
            prev_day_high=prev_day_ohlc[0],
            prev_day_low=prev_day_ohlc[1],
            prev_day_close=prev_day_ohlc[2],
            today_high=max(item.high for item in prefix),
            today_low=min(item.low for item in prefix),
        ),
        trend_up_extreme=trend_up_extreme,
        trend_dn_extreme=trend_dn_extreme,
    )
    criteria = FLAT_CRITERIA if point.position is None else HELD_CRITERIA
    if point.action not in criteria:
        raise DatasetError(
            f"动作 {point.action!r} 与仓位候选集不匹配（{(segment.segment_id, point.bar_index)}）"
        )
    record_id = f"{segment.segment_id}:{point.bar_index}"
    return {
        "id": record_id,
        "state_id": record_id,
        "family_id": segment.segment_id,
        "split": segment.split_role,
        "state": state,
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": QUESTION_INSTRUCTIONS,
                "criteria": dict(criteria),
            }
        },
        "gold": {QUESTION_ID: point.action},
        "gold_label_kind": {QUESTION_ID: GOLD_LABEL_KIND},
        "metadata": {
            "segment_id": segment.segment_id,
            "source_group_id": segment.segment_id,
            "split_role": segment.split_role,
            "bar_index": point.bar_index,
        },
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dump_json(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def _write_files_atomically(run_dir: Path, files: Mapping[str, str]) -> dict[str, str]:
    """先写同父目录临时目录、再逐个 ``os.replace`` 落位（失败不留半成品）。"""
    parent = run_dir.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DatasetError(f"输出目录不可创建: {parent} ({type(exc).__name__}: {exc})") from None
    stage = Path(tempfile.mkdtemp(dir=parent, prefix=f".{run_dir.name}."))
    try:
        for name, text in files.items():
            (stage / name).write_text(text, encoding="utf-8", newline="")
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DatasetError(
                f"输出目录不可创建: {run_dir} ({type(exc).__name__}: {exc})"
            ) from None
        digests: dict[str, str] = {}
        for name in sorted(files):
            os.replace(stage / name, run_dir / name)
            digests[name] = file_sha256(run_dir / name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return digests


def _load_daily_ohlc(data_dir: str | Path, symbol: str) -> Any:
    """读 ``{symbol}_1d.csv`` 返回 LoadedOHLCV（日线文件逐行一个交易日，升序）。

    :raises DataLoadError: 1d 文件缺失/损坏（硬错误，不静默跳过）
    """
    return load_ohlcv(symbol, "1d", data_dir=data_dir)


def _daily_rows(loaded: Any) -> tuple[tuple[Any, float, float, float], ...]:
    """LoadedOHLCV → 按文件序的 (交易日, 高, 低, 收) 元组（交易日 = 日线 timestamp 日历日）。"""
    return tuple(
        (pd.Timestamp(ts).date(), float(high), float(low), float(close))
        for ts, high, low, close in zip(
            loaded.df["timestamp"], loaded.df["high"], loaded.df["low"], loaded.df["close"]
        )
    )


def _prev_daily_ohlc(
    daily_rows: tuple[tuple[Any, float, float, float], ...],
    trade_date: Any,
) -> tuple[float, float, float] | None:
    """上一交易日日线 (高, 低, 收)：日线文件中严格早于 ``trade_date`` 的最后一行。

    无前日（数据起点）或日线值非有限 → 返回 ``None``（调用方跳过该片段，不静默）。
    """
    prev: tuple[float, float, float] | None = None
    for row_date, high, low, close in daily_rows:
        if row_date >= trade_date:
            break
        prev = (high, low, close)
    if prev is not None and not all(math.isfinite(value) for value in prev):
        return None
    return prev


def _replay_account_values(
    bars: tuple[Bar, ...],
    events: tuple[TradeEvent, ...],
    *,
    tick_size: float,
    processed_bar_count: int,
    initial_net_value: float,
    initial_peak_net_value: float,
) -> dict[int, tuple[float, float, float]]:
    """由 TradeEvent 轨迹重驱 :class:`ReplayAccount`，复算逐处理 bar 的（净值, 今日, 回撤）。

    生成侧填 ``DecisionPoint`` 净值/今日的真实值来源（替换 T5a 的 0.0 占位）：三值全部由
    :class:`ReplayAccount` 只用 ≤ 决策 K 线的数据算出（与 ``evaluate_segment`` 的
    point 构造路径同源同 mark 报告；先记号后成交，调用约定一致）；决策点三值 =
    该根成交**前**的盯市值（同根事件流顺序即执行顺序——反手 = 先平后开两笔）。
    净值 = ``NET_VALUE_BASE × equity``；今日 = 净值 − 片段起点净值（``initial_net_value``，
    净值尺度，每片段重置）；回撤取 mark 报告值。审计侧独立复算
    （``audit._independent_account_values``）已由 ``check_state_leakage`` 接线
    （T6b，入参 = 本函数同款链起点快照），与本函数重放结果逐值比对——两者不共享
    实现（防自证：任一侧漂移/被篡改即 ``DatasetError``）。

    :raises DatasetError: processed_bar_count 越界 / 存在未消费事件 /
        重驱成交轨迹与事件流不一致（均不静默）
    """
    if not 0 < processed_bar_count <= len(bars):
        raise DatasetError(
            f"账户回放 processed_bar_count 越界: {processed_bar_count}"
            f"（片段共 {len(bars)} 根）"
        )
    account = ReplayAccount(
        bars,
        tick_size=tick_size,
        initial_equity=float(initial_net_value) / NET_VALUE_BASE,
        initial_peak=float(initial_peak_net_value) / NET_VALUE_BASE,
    )
    values: dict[int, tuple[float, float, float]] = {}
    cursor = 0
    for index in range(processed_bar_count):
        # 先记号：决策点三值为该根成交前的盯市值（与 evaluate_segment 推进顺序一致）
        report = account.mark_to_market(index)
        net_value = NET_VALUE_BASE * report.equity
        values[index] = (net_value, net_value - float(initial_net_value), report.drawdown)
        # 后成交：事件流顺序即执行顺序（反手 = 同根先平后开两笔）
        while cursor < len(events) and events[cursor].bar_index == index:
            event = events[cursor]
            if event.pnl_ratio is None:
                account.open_position(event.direction, bars[index], reason=event.reason)
            else:
                account.close_position(bars[index], reason=event.reason)
            cursor += 1
    if cursor != len(events):
        raise DatasetError(
            "账户回放存在未消费的成交事件（事件数与处理 bar 范围不一致）: "
            f"{cursor}/{len(events)}"
        )
    if list(account.events) != list(events):
        # 重驱为确定性回放：成交价/盈亏全部由同一 bars + tick 重算，事件流必须逐项一致；
        # 任何不一致说明记账回放漂移 → 硬错误（不静默）
        raise DatasetError("账户回放成交轨迹与事件流不一致（不静默）")
    return values


def generate_dataset(
    segments: tuple[Segment, ...] | list[Segment],
    *,
    symbols: Mapping[str, float],
    data_dir: str | Path,
    params: EpisodeParams,
    output_dir: str | Path,
    daily_turning_points_dir: str | Path | None = None,
) -> GenerationResult:
    """确定性地生成按 split 的 NanoJev JSONL 训练集与审计文件。

    流程：清单/数据校验 → 逐片段回放 + 标签 → 逐入选决策点取可用日线折点极值（v5）
    → 记录映射 → 自检（结构、隔离、泄漏、计数一致）→ 原子落盘到
    ``<output_dir>/<run_id>``。

    v2 起：每个片段需在 ``{symbol}_1d.csv`` 中找到严格早于片段交易日的上一交易日
    日线（供 board_state 行）；找不到的片段被跳过（记入审计与结果，不静默），
    全部片段被跳过则硬错误（不写出任何产物）。

    v5 起：每个片段还需在日线转折点文件（``{symbol}_1d.csv``，默认
    ``<data_dir>/../turning_points``，``daily_turning_points_dir`` 可覆盖）中找到
    确认日严格早于各入选决策交易日的 up/down 折点各 ≥1；文件缺失 → ``DataLoadError``
    硬报错；任一入选决策点折点单侧/双侧缺失 → 跳过整个片段（记入审计与结果，
    不静默）；全部片段被跳过则硬错误（不写出任何产物）。

    v8 起（T5b 接线）：处理顺序 = ``(start, end, segment_id)`` 时间升序稳定排序——
    净值/峰值跨片段按**全局时间序**串行传递（多 symbol 清单跨 symbol 同样串行传递，
    设计披露 ③）；输入指纹（``canonical()`` 列表）保持清单原序不变（只改处理顺序，
    不改指纹）。净值链：首评估片段起点 = 100.0（``NET_VALUE_BASE``×1.0，峰值起点
    同值）；每评估片段起点 = 上一评估片段末结算净值/峰值（跳过片段不产出记录，
    链冻结穿过，carry 不变）；每入选决策点的净值/今日由事件流重驱账户的 mark 报告
    复算（今日 = 净值 − 片段起点净值）；落盘前 ``check_account_chain`` 硬校验
    （违规不写出任何产物），per_segment 审计新增 ``account_chain`` 节（起末净值，
    跨片段可追溯）。``check_state_leakage`` 调用点传入逐片段账户复算入参
    （事件轨迹 + 处理 bar 数 + 链起点快照，T6b）：账户行净值/今日/回撤由审计侧
    独立复算防自证，不一致即不写出任何产物。

    :raises DatasetError: 校验失败或自检不通过（不写出任何产物）
    :raises DataLoadError: 日线/日线折点文件缺失或损坏（硬错误，不静默）
    """
    segments = tuple(segments)
    validate_segments(segments, data_dir=data_dir, symbols=symbols)

    # v8（T5b）：处理顺序 = (start, end, segment_id) 时间升序稳定排序（净值链按全局
    # 时间序串行传递，设计披露 ③：多 symbol 清单跨 symbol 同样串行传递）；输入指纹
    # （下方 segments_text 的 canonical() 列表）保持清单原序不变——只改处理顺序，
    # 不改指纹。
    ordered_segments = sorted(
        segments, key=lambda segment: (segment.start, segment.end, segment.segment_id)
    )

    tp_dir = (
        Path(daily_turning_points_dir)
        if daily_turning_points_dir is not None
        else _default_daily_turning_points_dir(data_dir)
    )

    outcomes: dict[str, SegmentOutcome] = {}
    bars_by_segment: dict[str, tuple[Bar, ...]] = {}
    records_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ROLES}
    daily_loaded_by_symbol: dict[str, Any] = {}
    prev_daily_by_segment: dict[str, tuple[float, float, float]] = {}
    board_state_skipped: dict[str, str] = {}
    trend_points_by_symbol: dict[str, tuple[TurningPoint, ...]] = {}
    trend_source_versions: dict[str, str] = {}
    trend_extreme_skipped: dict[str, str] = {}
    # v8（T5b）净值链（净值尺度，逐评估片段串行传递；跳过片段链冻结穿过——carry 不变）
    carry_net_value = NET_VALUE_BASE
    carry_peak_net_value = NET_VALUE_BASE
    chain_entries: list[dict[str, Any]] = []
    # v8（T6b）逐片段账户链起点快照（净值, 峰值），供审计侧账户行独立复算入参
    account_initial_values: dict[str, tuple[float, float]] = {}

    for segment in ordered_segments:
        loaded = daily_loaded_by_symbol.get(segment.symbol)
        if loaded is None:
            loaded = _load_daily_ohlc(data_dir, segment.symbol)
            daily_loaded_by_symbol[segment.symbol] = loaded
        # 片段交易日 = 片段结束时间的日历日（窗口止于 14:59；清单与测试夹具均满足）
        prev = _prev_daily_ohlc(_daily_rows(loaded), segment.end.date())
        if prev is None:
            board_state_skipped[segment.segment_id] = (
                f"trade_date={segment.end.date()} 无上一交易日日线（或值非有限，1d 文件）"
            )
            continue
        prev_daily_by_segment[segment.segment_id] = prev
        bars, source_version = load_segment_bars(segment, data_dir=data_dir)
        # v8（T5b）：净值链节传递——本片段起点 = 上一评估片段末结算净值/峰值
        # （净值尺度 ÷ NET_VALUE_BASE 换算为 equity 尺度；首片段 = 100/100）
        outcome = evaluate_segment(
            segment,
            bars,
            tick_size=float(symbols[segment.symbol]),
            params=params,
            source_data_version=source_version,
            initial_equity=carry_net_value / NET_VALUE_BASE,
            initial_peak=carry_peak_net_value / NET_VALUE_BASE,
        )
        # v8（T5b）：事件流重驱账户（mark 报告）填每决策点净值/今日（替换 T5a 的 0.0
        # 占位；净值 = NET_VALUE_BASE×equity，今日 = 净值 − 片段起点净值）
        account_values = _replay_account_values(
            bars,
            outcome.events,
            tick_size=float(symbols[segment.symbol]),
            processed_bar_count=outcome.processed_bar_count,
            initial_net_value=carry_net_value,
            initial_peak_net_value=carry_peak_net_value,
        )
        # v8（T6b）：记录本片段账户链起点快照（与上面传入 _replay_account_values 的
        # initial 同源同值；供 check_state_leakage 调用审计侧独立复算防自证比对）
        account_initial_values[segment.segment_id] = (carry_net_value, carry_peak_net_value)
        outcome = replace(
            outcome,
            decision_points=tuple(
                replace(
                    point,
                    net_value=account_values[point.bar_index][0],
                    today_pnl=account_values[point.bar_index][1],
                )
                for point in outcome.decision_points
            ),
        )
        # v5：逐入选决策点取其决策交易日的可用日线折点极值（确认日 < 决策交易日；
        # 片段内按 trade_date 缓存，单交易日片段只算一次）
        tp_points = trend_points_by_symbol.get(segment.symbol)
        if tp_points is None:
            loaded_points = load_turning_points(segment.symbol, "1d", data_dir=tp_dir)
            tp_points = loaded_points.points
            trend_points_by_symbol[segment.symbol] = tp_points
            trend_source_versions[segment.symbol] = f"sha256={file_sha256(loaded_points.path)}"
        extremes_by_trade_date: dict[dt.date, tuple[TrendExtreme, TrendExtreme]] = {}
        prepared: list[tuple[DecisionPoint, TrendExtreme, TrendExtreme]] = []
        missing_trade_date: dt.date | None = None
        for point in outcome.selected:
            trade_date = _bar_trade_date(bars, bars[point.bar_index])
            if trade_date not in extremes_by_trade_date:
                extremes = _usable_trend_extremes(tp_points, trade_date)
                if extremes is None:
                    missing_trade_date = trade_date
                    break
                extremes_by_trade_date[trade_date] = extremes
            trend_up, trend_dn = extremes_by_trade_date[trade_date]
            prepared.append((point, trend_up, trend_dn))
        if missing_trade_date is not None:
            trend_extreme_skipped[segment.segment_id] = (
                f"trade_date={missing_trade_date} 缺日线转折点"
                "（需确认日严格早于该交易日的 up/down 折点各 ≥1，1d 转折点文件）"
            )
            # 跳过片段：不产出记录，净值链冻结穿过（carry 不变，传给下一实际生成片段）
            continue
        # v8（T5b）：链条目（处理时间序）+ 传递给下一实际生成片段
        # （末结算净值 = NET_VALUE_BASE×final_equity，峰值同机制）
        final_net_value = NET_VALUE_BASE * outcome.final_equity
        chain_entries.append(
            {
                "segment_id": segment.segment_id,
                "initial_net_value": carry_net_value,
                "final_net_value": final_net_value,
            }
        )
        carry_net_value = final_net_value
        carry_peak_net_value = NET_VALUE_BASE * outcome.final_peak
        outcomes[segment.segment_id] = outcome
        bars_by_segment[segment.segment_id] = bars
        for point, trend_up, trend_dn in prepared:
            records_by_split[segment.split_role].append(
                build_record(
                    segment=segment,
                    bars=bars,
                    point=point,
                    price_precision=params.price_precision,
                    prev_day_ohlc=prev,
                    trend_up_extreme=trend_up,
                    trend_dn_extreme=trend_dn,
                )
            )

    if not prev_daily_by_segment:
        raise DatasetError(
            "所有片段都缺少上一交易日日线（1d 文件），无法生成 board_state 状态；"
            f"跳过明细: {dict(sorted(board_state_skipped.items()))}"
        )
    if not outcomes:
        raise DatasetError(
            "所有片段都缺少可用日线转折点（1d 转折点文件），无法生成 v5 日线行；"
            f"跳过明细: {dict(sorted(trend_extreme_skipped.items()))}"
        )

    all_records = [
        record for split in SPLIT_ROLES for record in records_by_split[split]
    ]
    check_records(all_records)
    # v8（T5b）硬门：跨片段净值链按处理时间序逐对核验（首评估片段起点净值 =
    # NET_VALUE_BASE、下一片段起点净值 == 上一片段末结算净值），违规 raise
    # DatasetError（不写出任何产物）；归一化链条目写入 per_segment 的 account_chain 节
    account_chain = check_account_chain(chain_entries)
    # v8（T6b）硬门：账户行三值（净值/今日/回撤）由审计侧 _independent_account_values
    # 独立复算（内联重放，不调用 ReplayAccount/序列化实现）并与 state 文本逐值比对
    # （防自证/tamper 防护）；入参 = 逐片段事件轨迹 + 处理 bar 数 + 链起点快照，
    # 不一致即 raise DatasetError（不写出任何产物）
    account_inputs_by_segment = {
        segment_id: AccountReplayInputs(
            events=outcome.events,
            processed_bar_count=outcome.processed_bar_count,
            initial_net_value=account_initial_values[segment_id][0],
            initial_peak_net_value=account_initial_values[segment_id][1],
        )
        for segment_id, outcome in outcomes.items()
    }
    check_state_leakage(
        all_records,
        bars_by_segment=bars_by_segment,
        price_precision=params.price_precision,
        prev_daily_by_segment=prev_daily_by_segment,
        trend_points_by_symbol=trend_points_by_symbol,
        symbols_by_segment={segment.segment_id: segment.symbol for segment in segments},
        account_inputs_by_segment=account_inputs_by_segment,
    )

    segments_text = _dump_json([segment.canonical() for segment in segments])
    symbols_text = _dump_json({symbol: float(value) for symbol, value in sorted(symbols.items())})
    params_text = _dump_json(params.as_dict())
    input_hashes = {
        "segment_count": len(segments),
        "segments_sha256": _sha256_text(segments_text),
        "symbols_sha256": _sha256_text(symbols_text),
        "params_sha256": _sha256_text(params_text),
        "state_schema_sha256": _sha256_text(STATE_SCHEMA),
        "question_schema_sha256": _sha256_text(QUESTION_SCHEMA),
    }
    # run_id 把 STATE_SCHEMA / QUESTION_SCHEMA 纳入哈希输入：状态或候选文案变更产生
    # 新 run（旧 run 保留不覆盖）
    run_id = "run-" + _sha256_text(
        "|".join((segments_text, symbols_text, params_text, STATE_SCHEMA, QUESTION_SCHEMA))
    )[:12]

    split_texts: dict[str, str] = {}
    for split in SPLIT_ROLES:
        rows = records_by_split[split]
        split_texts[f"{split}.jsonl"] = "".join(
            _dump_json(record) + "\n" for record in rows
        )
    split_digests = {name: _sha256_text(text) for name, text in split_texts.items()}

    payload = build_audit_payload(
        run_id=run_id,
        params=params,
        segments=segments,
        symbols=symbols,
        outcomes=outcomes,
        records_by_split=records_by_split,
        split_digests=split_digests,
        input_hashes=input_hashes,
        board_state_skipped=board_state_skipped,
        trend_extreme_skipped=trend_extreme_skipped,
        trend_source_versions=trend_source_versions,
        daily_source_versions={
            symbol: loaded.source_data_version
            for symbol, loaded in daily_loaded_by_symbol.items()
        },
        account_chain=account_chain,
    )
    check_audit_consistency(payload, records_by_split)

    files: dict[str, str] = dict(split_texts)
    files[AUDIT_FILENAME] = _dump_json(payload, indent=2) + "\n"

    run_dir = Path(output_dir) / run_id
    digests = _write_files_atomically(run_dir, files)
    return GenerationResult(
        run_id=run_id,
        run_dir=run_dir,
        record_counts={split: len(records_by_split[split]) for split in SPLIT_ROLES},
        output_sha256=digests,
        audit=payload,
        board_state_skipped=board_state_skipped,
        trend_extreme_skipped=trend_extreme_skipped,
    )


__all__ = [
    "AUDIT_FILENAME",
    "FLAT_CRITERIA",
    "GOLD_LABEL_KIND",
    "HELD_CRITERIA",
    "POSITION_LABELS",
    "QUESTION_ID",
    "QUESTION_INSTRUCTIONS",
    "QUESTION_SCHEMA",
    "STATE_SCHEMA",
    "BoardStateValues",
    "GenerationResult",
    "build_record",
    "generate_dataset",
    "render_state",
]

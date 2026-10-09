"""NanoJev 训练记录生成：记录映射 + 状态序列化 + split 分配 + 原子落盘。

**记录契约**（对齐第三方 ``NanoJev/scripts/train_pipeline_decisions.py`` 的
``validate_training_row`` / ``read_training_records``；NanoJev 全程只读）：

* ``id`` = ``state_id`` = ``{segment_id}:{决策 K 线片段内序号}``（全局唯一，天然不跨 split）；
* ``family_id`` = ``metadata.source_group_id`` = ``segment_id``（一个片段 = 一个 episode，
  直接复用 NanoJev 的 state/source_group 跨 split 泄漏检查）；
* ``split`` = 片段的 ``split_role``；
* ``state`` = 确定性文本序列化的**相对比值**状态（v12 六部分：账户（六键：持仓 /
  [开仓价/止损价] / 净值 / 今日 / 回撤）/ 日线（按确认时间交错的涨势/跌势各最近 2 个折点段极值与段长、最近确认后的当前方向与时长、极值比较整体分类、行尾昨日高/低/收）/
  日内（今日高/低）/ 联动（品种化突破与相关度）/ 现价（决策 K 线 OHLC + bar 序号 + 量/持仓量比值）/ 盘口（na 占位））；
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
见 ``artifacts/account-service/02-design/tech-design.md``；
v9 起日线行趋势项升级为涨势/跌势各最近 2 个折点（编号时间倒序 -1/-2：段极值比值 +
段长「时长=n根」）+ 对称三分支趋势状态（趋势=涨势中|跌势中|震荡 + 状态时长
「时长=n根」，时长 = 最近可用折点确认根 → 决策交易日 T 的 1d 行号差；跳过判据收紧为
up/down 折点各 ≥2 + 新增缺决策交易日日线行跳过原因），其余行与 v8 逐字节同构，
2026-10-03 用户拍板，见 ``artifacts/trend-state-v9/02-design/tech-design.md``）**

```text
marketsense.episode_state.v12
账户: 持仓=空仓 净值=<..> 今日=<..> 回撤=<..>
日线: <按确认时间升序交错的涨势/跌势折点项，各方向 -1 最近、-2 次近> 当前为<涨势|跌势> 时长=<n>根 整体为<涨势中|跌势中|震荡> 昨日高=<..> 昨日低=<..> 昨日收=<..>
日内: 今高=<..> 今低=<..>
联动: <主显示名>（突破=<momentum|na>）[，<联动显示名>（突破=<..>）][ 相关度=<r|na>]
现价: 开=<..> 高=<..> 低=<..> 收=<..> bar=<片段内 0 基序号> 成交量比=<..> 持仓量比=<..>
盘口: na
```

（行与行之间用 ``" \n "`` 连接，即换行符前后各一个空格。持仓非空时账户行为：
``账户: 持仓=持多|持空 开仓价=<..> 止损价=<..> 净值=<..> 今日=<..> 回撤=<..>``。）

v10：联动行升为突破动量（``联动: 突破=<momentum|na>``；momentum ∈ [-1, +1]，
三态判据 + 权重 1..N-1 线性加权，只用已收盘 K 线/桶；首根/可用根数 < 2 → ``na``；
窗口/周期 = ``EpisodeParams.breakthrough_window/breakthrough_period``，见
``artifacts/linkage-breakthrough/02-design/tech-design.md``）。

v11：联动行升为品种化联动（2026-10-04 拍板设计，见
``artifacts/linkage-symbol/02-design/tech-design.md``）：无联动品种 =
``联动: <主显示名>（突破=<主值>）``（主值 = v10 连续窗口突破动量口径不变）；
有联动品种 = 每联动品种一段 ``，<联动显示名>（突破=<联动值>）``（全角逗号连接）+
首个联动品种 `` 相关度=<r>``（同一交集序列上主/联动相邻对三态信号的皮尔逊 r；
有效信号对 < 2 或任一序列零方差 → ``na``）。显示名 = 去交易所前缀**原样保留**
（``DCE.v2701`` → ``v2701``、``INE.sc2611`` → ``sc2611``，大小写与配置一致，
不引入大小写改写；任务决定，change-report 披露）。三段 na 语义独立判定。

v12：日线折点项按确认根索引升序展示（同索引以确认时间戳作次序键），保持各方向 -1/-2 编号；最近确认折点决定当前方向，段极值比较结果单独作为整体分类，昨日高低收移至日线行尾。

* 比值分母 = **片段首根**（价格用首根开盘价，量/持仓量用首根同名列），小数位固定
  （默认 6）；分母 ≤ 0 时写 ``na``（不产生 ``inf``/绝对数）；
* **无历史窗口**：状态只含决策 K 线单根 + 仓位 + 净值 + 今日 + 回撤 + 盘面状态
  （v4 拆「日线」「日内」两行，v5 在日线行追加折点趋势项，v7 键名中文化并将量/持仓量比
  与 bar 序号并入现价行，v8 账户行升六键加净值/今日，v9 日线行趋势项升为各 2 项 +
  三分支趋势状态），绝对价格不进入模型输入；
  「盘口」行为 ``na`` 常量占位
  （真实 bid/ask 未接，非目标；v7 起量/持仓量比已并入现价行；联动行 v10 起为突破动量，
  见上，不再是常量占位）；
* **账户行**（v4，v8 升六键）：持仓值中文标签（空仓/持多/持空，未知方向抛
  ``DatasetError`` 不静默）；v8 起键序固定为 持仓/[开仓价/止损价]/净值/今日/回撤
  （磁盘键名沿用 v7 的 ``开仓价/止损价``，内部字段 entry/stop；空仓时无开仓价/止损价
  两键，沿用 v7 先例）；``净值/今日``（render_state 参数 net_value/today_pnl，
  ``DecisionPoint`` 同名字段）为净值尺度新值（初值 100、今日收益每片段重置、比值化记账），
  ``回撤`` 公式不变；三值均由调用方只用 ≤ 决策 K 线的数据算好传入
  （跨片段净值链由 ``generate_dataset`` 时间序串行回放接入，见
  ``artifacts/account-service/02-design/tech-design.md``）；
* **日线行**（原 v2 board_state 行 prev 部分，v4 拆出，v5 追加 trend 四值，v7 键名中文化，
  v9 升级为涨势/跌势各 2 项 + 对称三分支趋势状态；prev 与 ``dataset/board_state.py`` 同口径）：
  - ``昨日高/昨日低/昨日收``（内部字段 prev_h/prev_l/prev_c）= **上一交易日**日线高/低/收
    （来源 ``{symbol}_1d.csv``，
    取日线文件中严格早于片段交易日的最后一行）÷ 片段首根开盘价；
  - ``涨势(-1/-2, 最高=.., 时长=n根)``/``跌势(-1/-2, 最低=.., 时长=n根)`` =
    最近**可确认** up/down 折点各 **2 个**（编号时间倒序：-1 最近、-2 次近）的段内实际
    最高/最低价（``trend_extreme_price``，来源 ``data/turning_points/{symbol}_1d.csv``）
    ÷ 片段首根开盘价；``时长`` = 对应趋势段长（整数日线根数，由点序列确定性推导）；
  - ``趋势=涨势中|跌势中|震荡``（对称三分支判据，比较 = 各方向 -1 与 -2 的段极值价，
    见 :func:`dataset.turning_points.trend_state_direction`）：高/低两方向 -1 均严格
    大于 -2 → 涨势中；均严格小于 → 跌势中；其余（含任一方向相等）→ 震荡；
    之后的 ``时长`` = 最近一个可用折点（不分方向）确认根在 1d 文件中的行号 → 决策
    交易日 T 在 1d 文件中的行号之差（即最近折点确认次日起至 T 开盘经历的日线根数；
    确认于上一交易日 → 1 根）：只读取 T 的**行号**（T 是交易日这件事在 T 开盘时已知），
    不读取 T 日线 OHLC 值，无未来信息；
  - **同源不变式**：折点 CSV 的 ``bar_index`` 与 1d 文件行号必须同源（同一 1d 内容
    生成；合并 1d 与重生成折点绑为同一步保证这一点）；不同源 → 状态时长错位
    （审计侧独立重算可检出）；
  - 「可确认」口径：折点确认根日期**严格早于**该 bar 的**决策交易日**（State(T) 不得引用
    T 日及之后确认的折点，无未来泄漏）；决策交易日归属与 ``board_state.attribute_windows``
    夜盘规则同口径（日盘 bar → 其日历日；夜盘 bar（≥ 21:00）→ 其后下一个交易日；
    ``[15:00, 21:00)`` 或夜盘无下一交易日 → ``DatasetError`` 不静默）；
  - 逐决策点各自取其决策交易日的可用折点上下文（多日片段的后段决策点能看到后确认的
    折点）；
  - 任一入选决策交易日的可用 up/down 折点**各 < 2**，或 1d 文件缺该决策交易日行 →
    **跳过整个片段**（不产出记录），记入审计
    ``trend_extremes.skipped_segments`` 与 stderr 告警（两类原因文案区分，不静默）；
    全部片段被跳过 → 硬错误；
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
from typing import Any, Mapping, Sequence

import pandas as pd

from dataset.board_state import DAY_END, NIGHT_START
from dataset.errors import DataLoadError, DatasetError
from dataset.periods import resolve_duration_seconds
from dataset.storage import file_sha256, load_ohlcv
from dataset.turning_points import (
    TrendExtreme,
    TurningPoint,
    load_turning_points,
    recent_trend_extremes,
    trend_state_direction,
)

from dataset.market_episode.audit import (
    AccountReplayInputs,
    NET_VALUE_BASE,
    build_audit_payload,
    check_account_chain,
    check_audit_consistency,
    check_outcome_sidecar,
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
from dataset.market_episode.linkage import (
    align_bars_by_timestamp,
    breakthrough_momentum,
    linkage_breakthrough_momentum,
    pearson_correlation,
    signal_pairs_from_aligned,
)
from dataset.market_episode.replay import (
    Bar,
    LONG,
    PositionSnapshot,
    REASON_DECISION,
    REASON_REVERSE_OPEN,
    SHORT,
    ReplayAccount,
    TradeEvent,
    bars_from_frame,
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
#: v9 日线行趋势项升级为涨势/跌势各最近 2 个折点（编号时间倒序 -1/-2：段极值比值 +
#: 段长「时长=n根」）+ 对称三分支趋势状态（趋势=涨势中|跌势中|震荡 + 状态时长
#: 「时长=n根」；时长 = 最近可用折点确认根 → 决策交易日 T 的 1d 行号差，只读 T 行号
#: 不读 T 日线 OHLC；折点 bar_index 与 1d 行号同源不变式见模块 docstring）；
#: 跳过判据收紧为 up/down 折点各 ≥2 + 新增「缺决策交易日日线行」跳过原因；
#: 其余行与 v8 逐字节同构，2026-10-03 用户拍板，
#: 见 artifacts/trend-state-v9/02-design/tech-design.md）
#: v10 联动行升为突破动量 ``联动: 突破=<momentum|na>``（三态判据 + 权重 1..N-1 线性
#: 加权 + 只用已收盘 K 线/桶；首根/可用根数 < 2 → ``na``；窗口/周期 =
#: EpisodeParams.breakthrough_window/breakthrough_period，其余五行与 v9 逐字节同构，
#: 见 artifacts/linkage-breakthrough/02-design/tech-design.md）
#: v11 联动行升为品种化联动：无联动品种 = ``联动: <主显示名>（突破=<主值>）``
#: （主值 = v10 连续窗口口径不变）；有联动品种 = 每联动品种
#: ``，<联动显示名>（突破=<联动值>）`` 段（全角逗号连接）+ 首个联动品种
#: `` 相关度=<r>``（同一交集序列上主/联动相邻对三态信号的皮尔逊 r，无定义 → na）；
#: 显示名 = 去交易所前缀原样保留（大小写与配置一致）；三段 na 独立判定，
#: 2026-10-04 拍板，见 artifacts/linkage-symbol/02-design/tech-design.md）
STATE_SCHEMA = "marketsense.episode_state.v12"
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
#: outcome 旁挂文件名（utility-trainer 任务：每条 gold ∈ {open_long, open_short} 决策
#: 记录一行其交易最终结果；随 {split}.jsonl / audit.json 经同一原子写落盘；
#: ``{split}.jsonl`` 字节不变，run_id 哈希输入不含 sidecar（派生数据不入指纹））
OUTCOMES_FILENAME = "outcomes.jsonl"
#: sidecar 行的固定键集合（落盘 ``sort_keys``；缺失/多余键属生成侧实现错误）
OUTCOME_ROW_FIELDS: Mapping[str, type] = {
    "id": str,
    "segment_id": str,
    "bar_index": int,
    "split": str,
    "action": str,
    "direction": str,
    "outcome": float,
    "risk_ratio": float,
    "r_multiple": float,
    "exit_bar_index": int,
    "exit_reason": str,
}


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
    #: outcome 旁挂行数（按 split；utility-trainer 任务 T1，仅开仓记录）
    outcome_row_counts: Mapping[str, int]


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


@dataclass(frozen=True)
class UsableTrendContext:
    """某决策交易日可用的日线折点趋势上下文（v9 日线行序列化输入）。

    * ``highs``/``lows``：最近 2 个 up/down 折点段极值（:class:`TrendExtreme`，
      **时间倒序**，``recent_trend_extremes(usable, n=2)`` 返回值）；
    * ``state_duration``：趋势状态时长（日线根数）= 决策交易日 T 在 1d 文件中的
      0 基行号 − 最近可用折点（不分方向）确认根 ``bar_index``（恒 ≥ 1）。

    全部字段由调用方只用决策交易日 T 开盘即知的数据构造（折点确认日严格早于 T、
    T 行号在 T 开盘即知），无未来数据泄漏在构造层面成立。
    """

    highs: tuple[TrendExtreme, TrendExtreme]
    lows: tuple[TrendExtreme, TrendExtreme]
    state_duration: int
    latest_confirmation_kind: str


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


def _trade_date_daily_index(
    daily_rows: tuple[tuple[Any, float, float, float], ...],
    trade_date: Any,
) -> int | None:
    """决策交易日 ``trade_date`` 在日线文件中的 0 基行号；缺行 → ``None``
    （调用方按跳过片段处理，不静默）。

    v9 趋势状态时长口径：只用 T 的**行号**（T 是交易日这件事在 T 开盘时已知），
    不读取 T 日线 OHLC 值，无未来信息；行号与折点 CSV 的 ``bar_index`` 必须同源
    （同一 1d 内容生成，见模块 docstring 同源不变式）。
    """
    for index, (row_date, _high, _low, _close) in enumerate(daily_rows):
        if row_date == trade_date:
            return index
    return None


def _usable_trend_context(
    points: tuple[TurningPoint, ...],
    trade_date: dt.date,
    trade_date_daily_index: int,
) -> UsableTrendContext | None:
    """决策交易日 ``trade_date`` 可用的日线折点趋势上下文（v9，取代 v5/v8 的
    ``_usable_trend_extremes``：n=1 单极值 → n=2 双极值 + 状态时长）。

    「可用」= 确认根日期（``timestamp`` 日历日）**严格早于** ``trade_date``（State(T)
    不得引用 T 日及之后确认的折点，无未来泄漏；过滤逻辑与 v5/v8 完全一致）。先过滤后
    检查：可用 up/down 点任一侧 ``< 2`` 是**正常边界**（调用方走跳过路径），返回
    ``None`` 而不抛异常；两侧各 ≥2 时复用 :func:`recent_trend_extremes`（``n=2``，
    语义同源：段闭区间、平局取最早、时间倒序、段长由点序列推导；过滤后序列为
    up/down 前缀，段长推导与全序列一致）。

    ``state_duration`` = ``trade_date_daily_index``（T 的 1d 行号）−
    ``usable[-1].bar_index``（最近可用折点确认根，``usable`` 为时间序前缀、末元素即
    最近折点）：确认日严格早于 T 且折点与 1d 同源 ⇒ 恒 ≥ 1，违反即同源不变式被破坏
    → ``DatasetError``（不静默）。

    :raises DatasetError: 可用 up/down 点缺极值字段（文件损坏，硬错误不静默）；
        状态时长 ``< 1``（折点与 1d 行号不同源）
    """
    usable = tuple(sorted(
        (
            point
            for point in points
            if point.kind in ("up", "down") and point.timestamp.date() < trade_date
        ),
        key=lambda point: (point.bar_index, point.timestamp),
    ))
    up_count = sum(1 for point in usable if point.kind == "up")
    down_count = len(usable) - up_count
    if up_count < 2 or down_count < 2:
        return None
    highs, lows = recent_trend_extremes(usable, n=2)
    state_duration = trade_date_daily_index - usable[-1].bar_index
    if state_duration < 1:
        raise DatasetError(
            f"趋势状态时长 < 1（trade_date={trade_date}，"
            f"daily_index={trade_date_daily_index}，"
            f"最近可用折点 bar_index={usable[-1].bar_index}）；"
            "折点 CSV 的 bar_index 与 1d 文件行号必须同源（同一 1d 内容生成）"
        )
    return UsableTrendContext(
        highs=highs,
        lows=lows,
        state_duration=state_duration,
        latest_confirmation_kind=usable[-1].kind,
    )


#: 纯增量公开别名（state-now 快照 builder 复用；与 ``_usable_trend_context`` 同一对象，
#: 不改变任何既有函数/常量/序列化输出，见 artifacts/state-now-view/02-design/tech-design.md §3.4）
build_usable_trend_context = _usable_trend_context


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
    *,
    trend_context: UsableTrendContext,
) -> str:
    """日线行（v12）：折点按确认根索引升序交错输出；最近确认折点后的当前方向与时长、
    保持独立的极值比较整体分类，昨日高/低/收置于行尾。价格比值分母为片段首根开盘价；
    段长/状态时长为整数。"""
    open_price = reference_bar.open
    direction = trend_state_direction(trend_context.highs, trend_context.lows)
    trend_label = {"up": "涨势中", "down": "跌势中", "range": "震荡"}.get(direction)
    if trend_label is None:
        raise DatasetError(f"未知趋势状态分类: {direction!r}")
    indexed_extremes = sorted(
        (
            (item.confirmation_bar_index, item.confirmation_timestamp, index, item)
            for kind_extremes in (trend_context.highs, trend_context.lows)
            for index, item in enumerate(kind_extremes)
        ),
        key=lambda value: (value[0], -1 if value[1] is None else value[1].value),
    )
    trend_items = []
    for _, _, index, item in indexed_extremes:
        label = "涨势" if item.kind == "up" else "跌势"
        field = "最高" if item.kind == "up" else "最低"
        number = -1 - index
        trend_items.append(
            f"{label}({number}, {field}={format_ratio_value(ratio_or_none(item.trend_extreme_price, open_price), price_precision)}, 时长={item.segment_length}根)"
        )
    current_direction = "跌势" if trend_context.latest_confirmation_kind == "up" else "涨势"
    return (
        "日线: "
        + " ".join(trend_items)
        + f" 当前为{current_direction} 时长={trend_context.state_duration}根"
        + f" 整体为{trend_label}"
        + f" 昨日高={format_ratio_value(ratio_or_none(board_state.prev_day_high, open_price), price_precision)}"
        + f" 昨日低={format_ratio_value(ratio_or_none(board_state.prev_day_low, open_price), price_precision)}"
        + f" 昨日收={format_ratio_value(ratio_or_none(board_state.prev_day_close, open_price), price_precision)}"
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


def _linkage_display_name(symbol: str) -> str:
    """联动行品种显示名：去交易所前缀后**原样保留**（``DCE.v2701`` → ``v2701``、
    ``INE.sc2611`` → ``sc2611``，大小写与配置一致，不引入大小写改写；
    无 ``.`` 前缀的 symbol 原样返回）。任务决定（2026-10-04）：忠实配置原文，
    若需 ``SC2611`` 大写显示后续一行改（change-report 披露）。"""
    return symbol.split(".", 1)[1] if "." in symbol else symbol


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
    trend_context: UsableTrendContext,
    breakthrough_bars: Sequence[Bar] = (),
    breakthrough_window: int = 20,
    breakthrough_duration_seconds: int = 60,
    primary_symbol: str = "",
    linkage_symbol_bars: Mapping[str, Sequence[Bar]] | None = None,
) -> str:
    """确定性状态文本（v12 六部分：账户/日线/日内/联动/现价/盘口；模板见模块 docstring）。

    只做「决策 K 线单根 + 仓位 + 净值 + 今日 + 回撤 + 盘面状态 + 日线折点趋势上下文」
    的序列化：函数签名决定它无法访问决策 K 线之后的任何 bar（``board_state`` 的今日值与
    ``net_value``/``today_pnl`` 均由调用方只用 ≤ 决策 K 线的数据算好传入；
    ``trend_context`` 由调用方按确认日 < 决策交易日过滤 + T 行号构造，无未来数据泄漏
    在构造层面成立）。

    v7：联动行变为 ``na`` 常量占位（量/持仓量比值移入现价行）；``bar`` 序号移入现价行。
    v8：账户行升六键（持仓/[开仓价/止损价]/净值/今日/回撤，键序固定）。
    v9：日线行趋势项升为涨势/跌势各 2 项（编号倒序 -1/-2）+ 三分支趋势状态与时长。
    v10：联动行升为突破动量 ``联动: 突破=<momentum|na>``——对 ``breakthrough_bars``
    （调用方传入**截至决策 K 线 T（含）**的片段 1m 序列，防泄漏上界由调用方切片保证）
    按 ``breakthrough_window``/``breakthrough_duration_seconds`` 计算
    （``dataset.market_episode.linkage.breakthrough_momentum``，函数内只用已收盘数据：
    1m 逐根天然收盘，非 1m 由重采样 ``is_closed`` 排除未收满桶）；
    首根/可用根数 < 2 → ``na``（缺省空序列同）。缺省窗口/周期 = EpisodeParams 默认口径，
    生产调用方（build_record）必须显式透传 params 派生值。

    v11：联动行升为品种化联动——``primary_symbol``（主品种 symbol，显示名去交易所
    前缀原样保留）与 ``linkage_symbol_bars``（symbol → 联动品种**片段窗口内**的
    完整 1m 序列；不要求调用方预切片到 T——交集对齐以主品种窗口（已截至 T）
    时间戳为准，联动品种仅取相同时间戳，天然上界 ≤ T，无未来泄漏）：
    无联动品种 = ``联动: <主显示名>（突破=<主值>）``；有联动品种 = 每联动品种
    ``，<联动显示名>（突破=<联动值>）`` 段（交集对齐 = 主品种窗口（同 v10 窗口）与
    联动品种按时间戳保序交集；联动值 = 交集序列 secondary 侧三态加权；
    ``dataset.market_episode.linkage``）+ 首个联动品种 `` 相关度=<r>``
    （同一交集序列上主/联动相邻对三态信号的皮尔逊 r）。三段 na 语义独立判定：
    主值 na（首根/可用根数不足）、联动值 na（交集对 < 2）、相关度 na
    （有效信号对 < 2 或任一序列零方差）。多联动品种：相关度只对首个联动品种
    （单品种任务；多品种格式设计留白）。
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
    # v10/v11 联动行：主值 = 突破动量（momentum ∈ [-1, +1]，非比值；首根/可用根数
    # 不足 → na）。breakthrough_bars 必须已截至决策 K 线 T（含）：build_record 传
    # bars[: bar_index + 1]，防泄漏上界由调用方切片保证；momentum 函数内只用已收盘
    # 数据（1m 逐根天然收盘；非 1m 由 resample_bars 的 is_closed 排除未收满桶），
    # 正负号照常（负值自然带 - 号）。
    # v11 品种化联动：主段 ``<主显示名>（突破=<主值>）``（v10 连续窗口口径不变）；
    # 有联动品种时每品种一段 ``，<联动显示名>（突破=<联动值>）``（全角逗号连接；
    # 联动值 = 交集序列 secondary 侧三态加权，交集按主品种时间戳天然上界）+
    # 首个联动品种 `` 相关度=<r>``（同一交集序列上主/联动相邻对信号的皮尔逊 r）。
    # 三段 na 独立判定（format_ratio_value(None) → na）。
    momentum = breakthrough_momentum(
        breakthrough_bars, breakthrough_window, breakthrough_duration_seconds
    )
    linkage_parts = [
        f"{_linkage_display_name(primary_symbol)}"
        f"（突破={format_ratio_value(momentum, price_precision)}）"
    ]
    correlation: float | None = None
    if linkage_symbol_bars:
        for link_position, (link_symbol, link_bars) in enumerate(
            linkage_symbol_bars.items()
        ):
            aligned = align_bars_by_timestamp(
                breakthrough_bars, link_bars, breakthrough_window
            )
            link_value = linkage_breakthrough_momentum(aligned)
            linkage_parts.append(
                f"{_linkage_display_name(link_symbol)}"
                f"（突破={format_ratio_value(link_value, price_precision)}）"
            )
            if link_position == 0:
                signal_pairs = signal_pairs_from_aligned(aligned)
                correlation = pearson_correlation(
                    [primary for primary, _ in signal_pairs],
                    [secondary for _, secondary in signal_pairs],
                )
    linkage_line = "联动: " + "，".join(linkage_parts)
    if linkage_symbol_bars:
        linkage_line += f" 相关度={format_ratio_value(correlation, price_precision)}"
    return " \n ".join(
        (
            STATE_SCHEMA,
            f"账户: 持仓={held_text} {account_values}",
            _daily_line(
                board_state,
                reference_bar,
                price_precision,
                trend_context=trend_context,
            ),
            _intraday_line(board_state, reference_bar, price_precision),
            linkage_line,
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
    trend_context: UsableTrendContext,
    breakthrough_window: int = 20,
    breakthrough_duration_seconds: int = 60,
    linkage_symbol_bars: Mapping[str, Sequence[Bar]] | None = None,
) -> dict[str, Any]:
    """把一个入选决策点映射为 NanoJev 训练记录。

    ``prev_day_ohlc`` = 上一交易日日线 (高, 低, 收)（来源 1d 文件，绝对价格层）；
    ``trend_context`` = 该决策交易日可用的日线折点趋势上下文（涨势/跌势各最近 2 个
    折点段极值 + 状态时长；调用方按确认日 < 决策交易日过滤 + T 行号构造，
    无未来数据泄漏）；
    ``today_high/today_low`` 只由 ``bars[: point.bar_index + 1]``（≤ 决策 K 线）累计，
    无未来数据泄漏；``point.net_value/point.today_pnl``（v8 账户行「净值/今日」）同样由
    调用方（账户回放）只用 ≤ 决策 K 线的数据算好透传（跨片段净值链由后续任务接入）；
    ``breakthrough_window/breakthrough_duration_seconds``（v10 联动行）＝突破动量的窗口与
    周期秒数（生产调用方由 ``params.breakthrough_window`` /
    ``resolve_duration_seconds(params.breakthrough_period)`` 传入；缺省 = EpisodeParams
    默认口径 20/60；联动行动量对 ``bars[: point.bar_index + 1]``（≤ 决策 K 线）计算，
    防泄漏上界同上）；
    ``linkage_symbol_bars``（v11）＝联动品种 → 该片段**窗口内**的完整 1m 序列
    （生产调用方由 generate_dataset 按片段 start~end 切片传入；交集对齐以主品种
    截至 T 的窗口时间戳为上界，防泄漏在 render_state 内成立；缺省 None =
    无联动品种格式）。
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
        trend_context=trend_context,
        breakthrough_bars=prefix,
        breakthrough_window=breakthrough_window,
        breakthrough_duration_seconds=breakthrough_duration_seconds,
        primary_symbol=segment.symbol,
        linkage_symbol_bars=linkage_symbol_bars,
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


def pair_open_close_events(
    events: Sequence[TradeEvent],
) -> list[tuple[TradeEvent, TradeEvent]]:
    """开仓→平仓事件**单栈配对**（纯函数；utility-trainer 任务 T1）。

    账户固定 1 手、单仓位 ⇒ 事件流中开仓/平仓严格交替（开仓入栈、平仓弹栈，
    栈深恒 ≤ 1），每笔平仓事件恰好了结栈顶那笔开仓 → 配对确定性。

    :return: ``(开仓事件, 其配对平仓事件)`` 列表（事件流出现序）。
    :raises DatasetError: 平仓事件无栈可弹 / 开仓事件叠加（栈深 > 1）/
        事件流末尾仍有未配对开仓（均不静默）
    """
    stack: list[TradeEvent] = []
    pairs: list[tuple[TradeEvent, TradeEvent]] = []
    for event in events:
        if event.pnl_ratio is None:
            if stack:
                raise DatasetError(
                    "开仓事件叠加（栈深 > 1，账户单仓位约定被破坏）: "
                    f"bar {event.bar_index} reason={event.reason!r}"
                )
            stack.append(event)
        else:
            if not stack:
                raise DatasetError(
                    f"平仓事件无配对开仓事件（栈空）: bar {event.bar_index} "
                    f"reason={event.reason!r}"
                )
            pairs.append((stack.pop(), event))
    if stack:
        raise DatasetError(
            f"存在未配对的开仓事件: bar {stack[0].bar_index} reason={stack[0].reason!r}"
        )
    return pairs


def build_outcome_rows(
    *,
    segment: Segment,
    bars: tuple[Bar, ...],
    events: Sequence[TradeEvent],
    tick_size: float,
    decision_points: Sequence[DecisionPoint] = (),
) -> list[dict[str, Any]]:
    """构建一个片段的 outcome 旁挂行（utility-trainer 任务 T1；纯函数）。

    仅覆盖 ``gold ∈ {open_long, open_short}`` 的入选决策记录：开仓事件中
    ``reason == "decision"`` 者与开仓决策点一一对应（``evaluate_segment`` 仅开仓
    分支以该 reason 开仓）；``reverse_open`` 开仓（反手）不产生记录、不产出旁挂行
    （其平仓盈亏由审计恒等式单独核算）。行值语义：

    * ``outcome`` = 配对平仓事件的 ``pnl_ratio``（D6 冻结口径：比值记账，分母 =
      片段首根开盘价）；
    * ``risk_ratio`` = (决策 K 线高 − 低 + tick) / 片段首根开盘（决策 K 线当日可知，
      无未来数据）；
    * ``r_multiple`` = outcome / risk_ratio（训练权重输入，R 计价与标签体系同口径）；
    * ``exit_bar_index`` / ``exit_reason`` = 配对平仓事件归属。

    :param decision_points: 生产路径传入 ``SegmentOutcome.decision_points``，
        逐开仓事件交叉核验「同 bar 存在 selected 的开仓决策点且方向一致」
        （一一对应防漂移；缺省空序列 = 不做该核验）。
    :raises DatasetError: 决策 K 线越界 / risk_ratio 非有限正值 / outcome 非有限 /
        开仓事件与决策点不对应（均不静默）
    """
    reference_open = float(bars[0].open)
    if not reference_open > 0:
        raise DatasetError(
            f"片段首根开盘价必须为正数（比值表达的分母）: {bars[0].open!r}"
        )
    tick = float(tick_size)
    if not tick > 0:
        raise DatasetError(f"tick_size 必须为正数: {tick_size!r}")
    points_by_bar = {point.bar_index: point for point in decision_points}
    rows: list[dict[str, Any]] = []
    for open_event, close_event in pair_open_close_events(events):
        if open_event.reason != REASON_DECISION:
            # 反手开仓（reverse_open）：无决策记录，不产出旁挂行
            continue
        if not 0 <= open_event.bar_index < len(bars):
            raise DatasetError(
                f"开仓事件决策 K 线序号越界: {open_event.bar_index}"
                f"（片段 {segment.segment_id} 共 {len(bars)} 根）"
            )
        if decision_points:
            point = points_by_bar.get(open_event.bar_index)
            expected_action = (
                ACTION_OPEN_LONG if open_event.direction == LONG else ACTION_OPEN_SHORT
            )
            if (
                point is None
                or not point.selected
                or point.action != expected_action
            ):
                raise DatasetError(
                    f"开仓事件与开仓决策点不一一对应: "
                    f"{(segment.segment_id, open_event.bar_index)} "
                    f"reason={open_event.reason!r} direction={open_event.direction!r}"
                )
        bar = bars[open_event.bar_index]
        outcome = float(close_event.pnl_ratio)
        risk_ratio = (float(bar.high) - float(bar.low) + tick) / reference_open
        if not math.isfinite(risk_ratio) or risk_ratio <= 0:
            raise DatasetError(
                f"risk_ratio 必须为有限正数: {(segment.segment_id, open_event.bar_index)} "
                f"risk_ratio={risk_ratio!r}"
            )
        if not math.isfinite(outcome):
            raise DatasetError(
                f"outcome 必须为有限数值: {(segment.segment_id, open_event.bar_index)} "
                f"outcome={outcome!r}"
            )
        r_multiple = outcome / risk_ratio
        if not math.isfinite(r_multiple):
            raise DatasetError(
                f"r_multiple 必须为有限数值: {(segment.segment_id, open_event.bar_index)} "
                f"r_multiple={r_multiple!r}"
            )
        rows.append(
            {
                "id": f"{segment.segment_id}:{open_event.bar_index}",
                "segment_id": segment.segment_id,
                "bar_index": open_event.bar_index,
                "split": segment.split_role,
                "action": (
                    ACTION_OPEN_LONG
                    if open_event.direction == LONG
                    else ACTION_OPEN_SHORT
                ),
                "direction": open_event.direction,
                "outcome": outcome,
                "risk_ratio": risk_ratio,
                "r_multiple": r_multiple,
                "exit_bar_index": close_event.bar_index,
                "exit_reason": close_event.reason,
            }
        )
    rows.sort(key=lambda row: row["bar_index"])
    return rows


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

    流程：清单/数据校验 → 逐片段回放 + 标签 → 逐入选决策点取可用日线折点趋势上下文（v9）
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

    v9 起：折点要求收紧为 up/down **各 ≥2**（``_usable_trend_context`` →
    ``UsableTrendContext``：最近 2 up + 2 down 段极值 + 对称三分支趋势状态），
    新增第二类跳过原因「1d 文件缺决策交易日行」（状态时长口径需要 T 行号，
    ``_trade_date_daily_index``）；两类原因文案区分，均记入既有
    ``trend_extreme_skipped`` 桶与 stderr 告警；跳过片段仍在净值链 carry 更新前
    ``continue``（冻结穿过语义不变，v8 既有）。

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
    # v10：联动行突破动量的周期秒数（1m→60 直接用决策序列；5m/15m/1h 重采样；
    # 1d/未知周期已由 EpisodeParams 校验拒绝）
    breakthrough_duration_seconds = resolve_duration_seconds(params.breakthrough_period)

    # v11：联动品种 1m LoadedOHLCV 按 symbol 缓存（启动时一次性加载；CSV 缺失 → 硬错误）
    linkage_frames_by_symbol: dict[str, Any] = {}
    # v11：联动品种 1m 数据启动时一次性加载（配置声明了联动品种就必须有数据，
    # 不静默降级；CSV 不存在 → DatasetError（含 fetch 命令提示））
    for link_symbol in params.linkage_symbols:
        try:
            linkage_frames_by_symbol[link_symbol] = load_ohlcv(
                link_symbol, "1m", data_dir=data_dir
            )
        except DataLoadError as error:
            raise DatasetError(
                f"联动品种 {link_symbol!r} 的 1m K 线 CSV 不存在或不可读：请先执行 "
                f"`python -m dataset fetch --symbol {link_symbol} --period 1m "
                f"--start <start> --end <end>` 落盘（data_dir={data_dir}）"
            ) from error

    outcomes: dict[str, SegmentOutcome] = {}
    bars_by_segment: dict[str, tuple[Bar, ...]] = {}
    # utility-trainer（T1）：outcome 旁挂行（逐片段构建，按 split 累积；仅开仓记录）
    outcome_rows_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLIT_ROLES
    }
    outcome_rows_by_segment: dict[str, list[dict[str, Any]]] = {}
    # v10：1m K 线按 symbol → segment_id 两级缓存（同 symbol 多片段各自独立序列，
    # 不跨片段延伸），供 check_state_leakage 审计侧联动行独立复算
    bars_by_symbol: dict[str, dict[str, tuple[Bar, ...]]] = {}
    # v11：联动行审计独立复算入参（联动 symbol → segment_id → 片段完整 1m 序列，
    # 同 bars_by_symbol 模式；片段窗口切片同主品种规则，窗口内无数据 → 空序列 → na）
    linkage_bars_by_symbol: dict[str, dict[str, tuple[Bar, ...]]] = {}
    records_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ROLES}
    daily_loaded_by_symbol: dict[str, Any] = {}
    # v9：日线行元组按 symbol 缓存（上一交易日查找与决策交易日行号查找共用同源行序）
    daily_rows_by_symbol: dict[str, tuple[tuple[Any, float, float, float], ...]] = {}
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
            daily_rows_by_symbol[segment.symbol] = _daily_rows(loaded)
        daily_rows = daily_rows_by_symbol[segment.symbol]
        # 片段交易日 = 片段结束时间的日历日（窗口止于 14:59；清单与测试夹具均满足）
        prev = _prev_daily_ohlc(daily_rows, segment.end.date())
        if prev is None:
            board_state_skipped[segment.segment_id] = (
                f"trade_date={segment.end.date()} 无上一交易日日线（或值非有限，1d 文件）"
            )
            continue
        prev_daily_by_segment[segment.segment_id] = prev
        bars, source_version = load_segment_bars(segment, data_dir=data_dir)
        bars_by_symbol.setdefault(segment.symbol, {})[segment.segment_id] = bars
        # v11：联动品种片段窗口切片（同主品种规则 start <= t <= end）；窗口内无数据
        # → 空序列（渲染 na，如 sc2611 数据起点晚于片段）；审计入参两级容器同步填充；
        # 时间列严格升序且唯一校验同主品种口径（拒绝静默排序）
        segment_linkage_bars: dict[str, tuple[Bar, ...]] = {}
        for link_symbol, loaded_link in linkage_frames_by_symbol.items():
            link_frame = loaded_link.df
            link_stamps = link_frame["timestamp"]
            link_selected = link_frame.loc[
                (link_stamps >= segment.start) & (link_stamps <= segment.end)
            ]
            if link_selected.empty:
                link_bars: tuple[Bar, ...] = ()
            else:
                link_ordered = link_selected["timestamp"]
                if not bool(link_ordered.is_monotonic_increasing) or not bool(
                    link_ordered.is_unique
                ):
                    raise DatasetError(
                        f"联动品种 {link_symbol} 片段 {segment.segment_id} 的 K 线时间列"
                        "必须严格升序且唯一（拒绝静默排序）"
                    )
                link_bars = bars_from_frame(link_selected.reset_index(drop=True))
            segment_linkage_bars[link_symbol] = link_bars
            linkage_bars_by_symbol.setdefault(link_symbol, {})[segment.segment_id] = (
                link_bars
            )
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
        # v9：逐入选决策点取其决策交易日的可用折点趋势上下文（确认日 < 决策交易日；
        # 片段内按 trade_date 缓存，单交易日片段只算一次；date→行号映射由
        # _trade_date_daily_index 从同一日线行序解析，折点与 1d 行号同源不变式见
        # 模块 docstring）。两类跳过原因文案区分：① 1d 文件缺决策交易日行；
        # ② 可用 up/down 折点各 < 2
        contexts_by_trade_date: dict[dt.date, UsableTrendContext] = {}
        prepared: list[tuple[DecisionPoint, UsableTrendContext]] = []
        skip_reason: str | None = None
        for point in outcome.selected:
            trade_date = _bar_trade_date(bars, bars[point.bar_index])
            if trade_date not in contexts_by_trade_date:
                daily_index = _trade_date_daily_index(daily_rows, trade_date)
                if daily_index is None:
                    skip_reason = (
                        f"trade_date={trade_date} 缺决策交易日日线行"
                        "（v9 趋势状态时长口径需要 1d 文件含该交易日行）"
                    )
                    break
                context = _usable_trend_context(tp_points, trade_date, daily_index)
                if context is None:
                    skip_reason = (
                        f"trade_date={trade_date} 缺日线转折点"
                        "（需确认日严格早于该交易日的 up/down 折点各 ≥2，1d 转折点文件）"
                    )
                    break
                contexts_by_trade_date[trade_date] = context
            prepared.append((point, contexts_by_trade_date[trade_date]))
        if skip_reason is not None:
            trend_extreme_skipped[segment.segment_id] = skip_reason
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
        # utility-trainer（T1）：本片段 outcome 旁挂行（事件流单栈配对；仅开仓记录）
        segment_outcome_rows = build_outcome_rows(
            segment=segment,
            bars=bars,
            events=outcome.events,
            tick_size=float(symbols[segment.symbol]),
            decision_points=outcome.decision_points,
        )
        outcome_rows_by_segment[segment.segment_id] = segment_outcome_rows
        outcome_rows_by_split[segment.split_role].extend(segment_outcome_rows)
        for point, context in prepared:
            records_by_split[segment.split_role].append(
                build_record(
                    segment=segment,
                    bars=bars,
                    point=point,
                    price_precision=params.price_precision,
                    prev_day_ohlc=prev,
                    trend_context=context,
                    breakthrough_window=params.breakthrough_window,
                    breakthrough_duration_seconds=breakthrough_duration_seconds,
                    linkage_symbol_bars=segment_linkage_bars or None,
                )
            )

    if not prev_daily_by_segment:
        raise DatasetError(
            "所有片段都缺少上一交易日日线（1d 文件），无法生成 board_state 状态；"
            f"跳过明细: {dict(sorted(board_state_skipped.items()))}"
        )
    if not outcomes:
        raise DatasetError(
            "所有片段都被跳过（1d 文件缺决策交易日行，或可用 up/down 日线折点各 < 2），"
            "无法生成 v9 日线行；"
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
    # v12（T4）硬门：日线行全内容（按确认时序的涨势/跌势各 2 项编号极值 + 当前方向 + 整体分类 + 状态时长 + 行尾昨日 OHLC）
    # 由审计侧 _independent_trend_context/_independent_daily_line_v12 从折点 CSV +
    # 日线行元组独立内联重算（不调用生成侧选择/序列化实现），不一致即不写出任何产物
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
        daily_rows_by_symbol=daily_rows_by_symbol,
        # v10：联动行突破动量独立复算入参（symbol → segment_id → 片段完整 1m 序列）
        # + 窗口/周期秒数（生成侧同源透传，审计侧内联重写判据防自证）
        bars_by_symbol=bars_by_symbol,
        breakthrough_window=params.breakthrough_window,
        breakthrough_duration_seconds=breakthrough_duration_seconds,
        # v11：联动行独立复算入参（联动 symbol → segment_id → 片段完整 1m 序列；
        # 无联动品种配置 → None = 无联动品种格式比对）
        linkage_bars_by_symbol=linkage_bars_by_symbol or None,
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

    # utility-trainer（T1/T2）：outcome 旁挂文本 + 落盘前硬校验（审计侧独立复算 /
    # 逐片段恒等式 / 覆盖一致；违规 raise 不写出任何产物），校验返回审计 outcomes 节
    outcome_rows = [
        row for split in SPLIT_ROLES for row in outcome_rows_by_split[split]
    ]
    outcomes_text = "".join(_dump_json(row) + "\n" for row in outcome_rows)
    outcomes_section = check_outcome_sidecar(
        records_by_split=records_by_split,
        sidecar_rows=outcome_rows,
        outcomes_by_segment=outcomes,
        bars_by_segment=bars_by_segment,
        tick_size_by_segment={
            segment_id: float(symbols[outcome.symbol])
            for segment_id, outcome in outcomes.items()
        },
    )

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
        outcomes_section=outcomes_section,
        outcome_sidecar_sha256=_sha256_text(outcomes_text),
    )
    check_audit_consistency(payload, records_by_split)

    files: dict[str, str] = dict(split_texts)
    files[AUDIT_FILENAME] = _dump_json(payload, indent=2) + "\n"
    files[OUTCOMES_FILENAME] = outcomes_text

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
        outcome_row_counts={
            split: len(outcome_rows_by_split[split]) for split in SPLIT_ROLES
        },
    )


__all__ = [
    "AUDIT_FILENAME",
    "FLAT_CRITERIA",
    "GOLD_LABEL_KIND",
    "HELD_CRITERIA",
    "OUTCOMES_FILENAME",
    "POSITION_LABELS",
    "QUESTION_ID",
    "QUESTION_INSTRUCTIONS",
    "QUESTION_SCHEMA",
    "STATE_SCHEMA",
    "BoardStateValues",
    "GenerationResult",
    "build_outcome_rows",
    "build_record",
    "build_usable_trend_context",
    "generate_dataset",
    "pair_open_close_events",
    "render_state",
]

# dataset — 数据准备子应用

从仓库根以普通 Python 包方式运行（无打包元数据、无需安装）：

```bash
/opt/anaconda3/envs/marketsense/bin/python -m dataset --help
/opt/anaconda3/envs/marketsense/bin/python -m pytest dataset/tests -q
```

## 运行前提

**必须在仓库根目录运行**（用 `python -m dataset` 调用；`python -m` 会把当前目录加入
`sys.path`，无需安装、无打包元数据）。**conda 在本机需手动初始化**：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marketsense
cd /Users/jiangdianjing/agentspace/MarketSense
```

不要用系统自带的 `/usr/local/bin/python3`（其 `numpy` 已损坏）。

用途：把「天勤 K 线取数 → 标准化落盘（CSV + 来源指纹）→ 转折点提取 → 转折点落盘」
做成自持、可复现、可测试的能力，并提供参数化命令行入口。

- **确定性**：同名输入 → 同一字节输出；sidecar **不含墙钟时间**。
- **无未来数据泄漏**：在线取数经 `clean_serial_klines` 剔除 `id<0`/`datetime<=0`
  填充行与「起点 + 周期 > 当前时间」的**未收盘** K 线；`as_of` 可注入。
- **凭证不入库**：凭证仅在内存中传递，错误消息**永不**回显凭证值。
- 转折点输出是**中间数据**，不是最终模型输入格式（D-07）。

## CLI 用法

```text
python -m dataset fetch          --symbol S [--symbol S2 ...] --period P (--bars N | --start ISO --end ISO) [--output-dir DIR] [--config FILE]
python -m dataset turning-points --symbol S [--symbol S2 ...] --period P [--initial-direction auto|up|down] [--data-dir DIR] [--output-dir DIR] [--config FILE]
python -m dataset prepare        --symbol S [--symbol S2 ...] --period P (--bars N | --start ISO --end ISO) [--initial-direction ...] [--output-dir DIR] [--config FILE]
python -m dataset episode-generate --segments FILE [--daily-turning-points-dir DIR] [--output-dir DIR] [--config FILE]
python -m dataset board-state    --symbol S [--symbol S2 ...] [--period 1m] [--start DATE --end DATE] [--data-dir DIR] [--output-dir DIR] [--config FILE]
```

- `--period` 必填，仅支持 `1m, 5m, 15m, 1h, 1d`（`board-state` 例外：可选，仅支持 `1m`）；非法值 stderr 输出支持列表并非 0 退出。
- `--bars` 与 `--start/--end` **互斥**，且必须给其一；`--bars` 取值范围 `1..8964`。
  实现上会向天勤多取 1 根以凑满 N 根**已收盘** K 线；取上限 `8964` 时无法多取，
  末根未收盘被剔除后实际至多返回 `8963` 根 —— 此时 CLI 会在 stderr 输出告警（不静默）。
  请求根数未被满足时一律告警，并**区分归因**（已达 `--bars` 上限 vs 数据源提供的已收盘
  K 线不足）；需要更早数据请改用 `--start/--end`。
- `--symbol` 可重复（一次命令处理多个品种）。
- 退出码：`0` 成功；`1` 运行期失败（配置/凭证/取数/校验/读取；含 `output_dir` 不可创建）；
  `2` 用法错误。

示例：

```bash
# 最近 3 根已收盘 1 分钟 K 线 → data/ohlcv/DCE.v2701_1m.csv + .json
python -m dataset fetch --symbol DCE.v2701 --period 1m --bars 3

# 离线：读取已落盘 K 线 → data/turning_points/DCE.v2701_1m.csv + .json（不联网）
python -m dataset turning-points --symbol DCE.v2701 --period 1m
python -m dataset turning-points --symbol DCE.v2701 --period 1m --initial-direction up

# 一步：fetch + 转折点
python -m dataset prepare --symbol DCE.v2701 --period 1d --bars 200

# episode 训练数据生成（离线；需片段清单 + symbols.local.yaml 的 tick_size）
python -m dataset episode-generate --segments data/segments/my_segments.jsonl
```

# episode 训练数据生成（离线；需片段清单 + symbols.local.yaml 的 tick_size）
python -m dataset episode-generate --segments data/segments/my_segments.jsonl

# 盘面状态读取（离线；需已落盘 1m + 1d K 线）
python -m dataset board-state --symbol DCE.v2701 --start 2026-09-01 --end 2026-09-30
```

### `board-state`：盘面状态读取

离线读取已落盘 1 分钟 + 日线 K 线，按交易日窗口逐根维护盘面状态（研究 building block，
**非最终模型输入格式**）：

- **窗口**：交易日 `T` 的窗口 = 归属 `T` 的夜盘 K 线 + `T` 的日盘 K 线；夜盘 K 线
  （time-of-day ≥ 21:00）归属其日历日之后的**下一个交易日**（周五夜盘 → 周一）；
  交易日 = 存在日盘 K 线（< 15:00）的日历日。与片段清单（`sep2026.jsonl`）同口径。
- **固定状态**：`prev_day_high` / `prev_day_low` / `prev_day_close`（上一交易日日线 OHLC，
  来源 1d 落盘文件）与 `today_open`（窗口首根开盘价 = 日线开盘口径）。
- **动态状态**：`today_high` / `today_low` = 已处理 K 线高/低点的累计 max/min
  （`State(T)` 只用 ≤ T 的 K 线，无未来泄漏）。
- **相对价**：六个价格字段一律 = 值 / `today_open`（6 位小数），`today_open` 恒为 `1.000000`；
  绝对 OHLC 只留在 `data/ohlcv/` 原始层。

输出：`data/board_state/{symbol}_board_state.csv` + sidecar
（列：`trade_date, timestamp, bar_index, prev_day_high, prev_day_low, prev_day_close, today_open, today_high, today_low`；
每根 K 线一行，sidecar 含语义说明与 1m/1d 来源指纹，无墙钟，同输入 → 同字节输出）。

前置与边界：

- 需先 `fetch --period 1m` 与 `fetch --period 1d` 落盘；缺日线报错（exit 1）。
- 1m 数据起点的首个交易日无前置交易日 → 跳过 → stderr 告警（不静默）。
- 上一交易日在 1m 序列中存在但日线缺该日 → 报错（1m/1d 数据不一致，请重新落盘日线）。
- 2026-10-01 实测：DCE.v2701 2026-09 全月 21 个交易日 7125 行；每日末行
  `today_high`/`today_low` 与当日日线高/低（比值）一致、`prev_day_*` 与上一交易日日线一致、
  `today_open` 恒为 1；两次运行 CSV+sidecar 字节一致。

### `episode-generate`：episode 训练数据生成

输入用户指定的震荡片段清单（每片段 = 一个 episode，`dataset/config/segments.example.jsonl`
为模板）与已落盘 1 分钟 K 线及 1 分钟日线（日线供 board_state 行的上一交易日值），确定性地输出
对齐 NanoJev 训练契约的按 split JSONL + 审计文件。不联网、不取数、不训练模型、不修改 `NanoJev/`；
模型可见价格一律用比值（分母 = 片段首根开盘价，6 位小数）。模块与语义分层：
`dataset/account.py`（账户域：确定性回放账户执行语义——持仓/成交模型/止损锚定/t−1 盯市
回撤与死亡/收盘强平契约 + 比值化记账 + 跨片段净值链参数 `initial_equity`/`initial_peak`；
仅依赖 `dataset.errors`，可脱离 `market_episode` 独立使用；契约见其 docstring）、
`dataset/market_episode/replay.py`（数据装载 + 状态行构造 + 账户域 re-export 兼容 shim——
既有 `from dataset.market_episode.replay import Bar, ReplayAccount, ...` 调用点继续可用）与
`labels.py`（policy：盈亏比规则真值标签 + (b) 采样），另有 `segments.py`
（清单/品种配置/参数）、`nanojev_records.py`（记录映射与落盘）、`audit.py`（确定性双跑、
跨 split 隔离、泄漏抽查 + 账户行独立复算、跨片段账户净值链硬校验、计数汇总）。

前置与用法：先 `dataset fetch --period 1m` 与 `dataset fetch --period 1d` 落盘 K 线（缺日线
硬报错不静默），并 `dataset turning-points --period 1d` 落盘日线折点（v5 日线行 trend 四值
数据源；默认取 `<data_dir>/../turning_points`，`--daily-turning-points-dir` 可覆盖；文件
缺失/损坏 → `DataLoadError` 硬报错 exit 1，不静默），再把 `dataset/config/symbols.example.yaml`
复制为 `symbols.local.yaml` 并填 `tick_size`，然后用 `--segments` 指向片段清单
（详细步骤与配置项见 `README.md` §7.2 小节 7）：

```bash
python -m dataset episode-generate --segments data/segments/my_segments.jsonl
```

输出（默认取配置 `episode.output_dir`，否则 `data/nanojev_dataset`）：

```text
data/nanojev_dataset/<run_id>/{train,dev,...}.jsonl   # 按 split 的记录（5 个文件都会写出；空 split 为 0 字节）
data/nanojev_dataset/<run_id>/audit.json              # 计数/指纹/冻结项/生成参数
```

关键约束与已知限制（实测）：

- 清单校验：字段完整非空、`period` 必须 `1m`、symbol+period 已有落盘 K 线、时间段落在数据范围内、
  同 symbol+period 不同 split 时间不重叠、清单覆盖 `train`/`dev`/`test`、每个 symbol 均有正数 `tick_size`。
- **状态文本模板（v8）**：`marketsense.episode_state.v8`，7 行：`账户:`、`日线:`、
  `日内:`、`联动:`、`现价:`、`盘口:`（na 占位）；
  v7 拍板：训练 state 文本键名中文化；日内行 `bar=` 序号与联动行量/持仓量比值并入现价行
  （持仓量只保留收盘时刻，开盘时刻丢弃）；联动行变为 `na` 常量占位；trend 四键本次不改。
  v8 拍板（2026-10-03，account-service 任务）：账户行升六键加净值/今日（跨片段净值链同步接入）。
  行间用 `" \n "` 连接（换行符前后各一个空格，v3 起生效，转义后的 JSON 文本更易读）。
  - `账户: 持仓=<空仓|持多|持空>[ 开仓价=<..> 止损价=<..>] 净值=<..> 今日=<..> 回撤=<..>`：
    v8 键序固定为 持仓/[开仓价/止损价]/净值/今日/回撤（磁盘键名沿用 v7 的 `开仓价/止损价`，
    内部字段 entry/stop；空仓时无开仓价/止损价两键，沿用 v7 先例；持仓值中文标签
    空/持多/持空，未知方向报错不静默）。`净值`（净值尺度）= 100×(1 + 跨片段累计已实现盈亏
    + 当前浮盈)（初值 100，展示口径 `NET_VALUE_BASE × equity`，equity 为比值化记账权益、
    盯市价 = `close[i-1]` t−1 口径不变）；`今日`（净值尺度）= 净值 − 片段起点净值
    （片段内权益变化，每片段重置，片段首个决策点 = 0.000000，可为负）；`回撤` 公式不变
    = (峰值 − 权益)/峰值，但峰值跨片段延续（见下方净值链）→ 数值不再每片段从 0 起算、
    与 v7 不同；三值均由调用方（账户回放）只用 ≤ 决策 K 线的数据算好传入，无未来泄漏；
  - `联动: na`：v7 起常量占位（量/持仓量比自 v7 起在现价行）；
  - `日线: 昨日高=<..> 昨日低=<..> 昨日收=<..> trend_up=<..> trend_up_len=<n> trend_dn=<..> trend_dn_len=<n>`
    （v4 起由原 board_state 行拆出，v5 起追加 trend 四键，v7 起键名中文化）：
    `昨日高/昨日低/昨日收`（内部字段 prev_h/prev_l/prev_c）= **上一交易日**日线高/低/收
    （来源 `{symbol}_1d.csv`，取日线文件中严格早于片段
    交易日的最后一行）÷ 片段首根开盘价；
    `trend_up`/`trend_dn` = 最近**可确认** up/down 折点的段内实际最高/最低价
    （`trend_extreme_price`，来源日线折点 CSV `data/turning_points/{symbol}_1d.csv`）÷
    片段首根开盘价，`trend_up_len`/`trend_dn_len` = 对应趋势段长（整数，由点序列确定性推导）；
    「可确认」口径：折点确认根日期**严格早于**该 bar 的**决策交易日**（State(T) 不得引用
    T 日及之后确认的折点，无未来泄漏）；决策交易日归属与 `dataset board-state` 夜盘规则
    同口径（日盘 bar → 其日历日；夜盘 bar ≥ 21:00 → 其后下一个交易日）；逐决策点各自取
    其决策交易日的可用最近折点（多日片段的后段决策点能看到后确认的折点）；
  - `日内: 今高=<..> 今低=<..>`（v4 起由原 board_state 行拆出；v7 起 `bar=` 序号移入现价行，
    键名中文化）：
    `今高/今低`（内部字段 today_h/today_l）= 片段首根至决策 K 线（含）的 1m 高/低**累计极值**
    ÷ 片段首根开盘价
    （State(T) 只用 ≤ 决策 K 线的数据，无未来泄漏）；
  - `现价: 开=<..> 高=<..> 低=<..> 收=<..> bar=<片段内 0 基序号> 成交量比=<..> 持仓量比=<..>`：
    决策 K 线 OHLC 比值（v7 键名中文化）；价格分母 = 片段首根开盘价
    （全交易日片段下片段首根 = 交易日窗口首根 = 今日开盘，与 `dataset board-state` 同口径）；
    今日开盘价不写（比值恒为 1.000000，纯冗余 token）；`bar=` = 片段内 0 基序号（v7 自日内行
    移入）；`成交量比` = 决策 K 线成交量 ÷ 首根成交量、`持仓量比` = 决策 K 线收盘时刻持仓量 ÷
    首根收盘时刻持仓量（v7 自联动行移入，开盘时刻持仓量丢弃）；分母 ≤ 0 时逐值写 `na`（不产生 `inf`）；
  - 片段无上一交易日日线，或任一入选决策点的可用 up/down 折点单侧缺失 → **跳过该片段**
    （不产出记录），记入审计（`board_state.skipped_segments` / `trend_extremes.skipped_segments`）
    与 stderr 告警（不静默）；全部片段被跳过则硬报错不写出产物。
- **跨片段净值链（v8 拍板，2026-10-03）**：片段按 `(start, end, segment_id)` 全局时间升序
  稳定排序后**串行回放**（输入指纹保持清单原序，`run_id` 不受排序影响；多 symbol 清单跨
  symbol 同样串行传递）；首评估片段起点净值 = 100.0（峰值起点同值）；每评估片段起点
  = 上一评估片段末结算净值/峰值（净值与峰值同机制传递，经 `evaluate_segment` 的
  `initial_equity`/`initial_peak` 参数接入，最终落到 `ReplayAccount` 同名参数）；
  跳过片段（缺上一交易日日线/缺可用折点）不产出记录，链**冻结穿过**（carry 不变）；
  split 起点边界（dev 起点 = train 末片段结算净值、test 起点 = dev 末）由时间序处理 +
  split 时间连续性自然成立；sep2026 实测链：train 末 115.957034 → dev 末 120.915886 →
  test 末 127.404954（audit `account_chain` 逐片段可追溯）。死亡级联披露：死亡后净值/峰值
  冻结延续，若某片段死亡（回撤 ≥ 阈值），后续片段首根盯市回撤仍 ≥ 阈值 → 级联死亡
  （后续片段 0 决策点/0 记录）；sep2026 实测 deaths=0 未触发。
- **审计 account_chain 与账户行独立复算（v8）**：`check_account_chain` 在落盘前硬校验
  跨片段净值链（首评估片段起点净值 = 100、逐对「下一片段起点净值 = 上一片段末结算净值」
  精确比对；违规不写出任何产物）；校验通过的起末净值写入 `audit.json` per_segment 条目
  `account_chain` 节（跨片段可追溯）；账户行三值（净值/今日/回撤）由审计侧
  `_independent_account_values` 内联重放独立复算（由 TradeEvent 轨迹 + K 线 + 链起点
  (净值, 峰值) 重放 t−1 盯市规则，**不调用** `ReplayAccount`/序列化实现，防自证），
  经 `check_state_leakage` 与 state 文本逐值比对（不一致不写出任何产物）；
  `check_no_absolute_values` 对「账户」行**豁免**邻根绝对价 token 扫描（六键均为净值尺度
  账户值，与绝对价不同域，不构成泄漏），其余行扫描语义不变；`AUDIT_SCHEMA` 保持 v1
  （`frozen_decisions` 新增 `account_carry`/`today_pnl` 两键，payload 纯增量）。
- **questions/candidates 文案（2026-10-02 用户拍板精简）**：唯一 choice 题 `next_action`，候选文案只留动作语义——空仓 `open_long=买入开仓 / open_short=卖出开仓 / stay_flat=继续空仓`，持仓 `close=平仓 / hold=继续持有 / reverse=反手`；成交价位由执行程序与滑点决定，不进模型输入。questions 文本由 `QUESTION_SCHEMA = marketsense.episode_question.v1` 标记（首次建立），纳入审计 `input.question_schema_sha256` 与 `run_id` 哈希；文案再演进必须换标记（产生新 run）。
- 确定性：无墙钟/随机；同输入双跑输出 sha256 一致。
- `run_id` = sha256(片段清单 + 品种配置 + 参数 + 状态 schema 标记 + questions schema 标记) 的前 12 位；**状态/候选文案变更产生新 run**（旧 run 保留不覆盖）；
  仍**不含行情数据指纹**，因此同清单/同参数下换数据重跑会
  **覆盖**同目录产物（README 及变更报告早期“不覆盖历史产物”的表述仅在清单/参数/schema 变化时成立）。
- **空 split 不报错**：某 split 为空时对应 `.jsonl` 为 0 字节且 CLI 仍以 0 退出；NanoJev trainer
  要求 `train`/`dev`/`test` 非空（`NanoJev/scripts/train_pipeline_decisions.py:477-479`），
  其 `--validate-only` 也不会拦空目录（`:423-424`）——生成后需自行确认非空。
- 本流水线已在真实片段上端到端运行：首轮（2026-10-01，无 board_state，模板 v1）
  `run-c1cb097177a3`；v2（含 board_state 行，2026-10-02）`run-13aff088b982`；
  同日候选文案精简 `run-b25cfd1ff370`；同日状态模板 v3（换行符前后加空格）
  `run-36b037252a62`；同日状态模板 v4（六部分中文标签重排：账户/联动/日线/日内/现价/盘口，
  `bar=` 归入日内行，盘口 `na` 占位）`run-3db1bf63afc2`；同日状态模板 v5（「日线」行新增
  `trend_up/trend_up_len/trend_dn/trend_dn_len` 4 键，数据源 = 日线折点 CSV；9 个 train 片段
  9-1~9-11 因缺 up 折点跳过并告警）`run-7cbd6516d46f`；同日状态模板 v6（行重排）
  `run-6ae3e38289a3`；同日状态模板 v7（键名中文化）`run-4f1cd33a3cfe`；
  v8（账户行六键 + 跨片段净值链，2026-10-03）**当前产物** `run-c364ea4a30ac`
  （DCE.v2701 2026-09 全月，train 1392 / dev 886 / test 655 全部非空，NanoJev `--validate-only`
  通过；9 个 train 片段 9-1~9-11 因缺 up 折点跳过并告警；净值链 100 → train 末
  115.957034 → dev 末 120.915886 → test 末 127.404954，deaths=0；与 v7 run 逐值比对：
  gold/持仓/开仓价/止损价全部一致，回撤因峰值跨片段延续数值不同、公式不变；
  真实数据双跑 sha256 一致；见主 README §5「首轮真实数据」）。
  v2/v3/v4 run 的 5729 条记录 board_state 值均与 `dataset board-state` CSV 全量交叉核对一致。
  磁盘现存 4 个 run（上述 v5/v6/v7/v8；更早的 v1~v4 run 已由用户会话清理）。

契约硬门（只读执行第三方脚本）：

```bash
python3 NanoJev/scripts/train_pipeline_decisions.py --validate-only --input data/nanojev_dataset/<run_id>
```

## 数据契约

**K 线 CSV**（`data/ohlcv/{symbol}_{period}.csv`，列顺序固定）：

```text
timestamp,open,high,low,close,volume,open_oi,close_oi
```

`timestamp` 为 ISO8601 带 `+08:00`、唯一、严格升序（`Asia/Shanghai`）；OHLC `float64`；
`volume int64`；持仓量为固定列：`open_oi`/`close_oi` 分别是天勤该根 K 线**起始时刻** / **结束时刻**的持仓量
（`int64`，两条取数路径均返回，无需配置开关）。
同名 `.json` sidecar 记录：

```text
provider, symbol, period, output_format, start_dt, end_dt, row_count,
first_timestamp, last_timestamp, file_sha256, source_data_version
```

`source_data_version = tianqin|{symbol}|{period}|{start}|{end}|rows=N|{first}|{last}|sha256={16hex}`，
**不含墙钟时间**。`--bars` 模式下 `start_dt/end_dt` 取实际窗口首末时间。

**转折点 CSV**（`data/turning_points/{symbol}_{period}.csv`）：

```text
point_index,kind,timestamp,price,bar_index,volume,oi,trend_extreme_price,trend_extreme_bar_index,dt_minutes,price_ratio,volume_ratio,oi_ratio
```

`kind ∈ {start, up, down, close}`（2026-09-25 起反转类 kind 由 `high`/`low` 更名为
`up`/`down`）；`bar_index` 为窗口内 0 基下标（可回溯关联 K 线 CSV）。`up`/`down` 点在
**触发根**确认（up 态命中 `low[t] < low[t-1]`、down 态命中 `high[t] > high[t-1]` 的
那根）：`timestamp`/`bar_index` = 触发根 `t`；`price` = 前一根 high（up）或 low（down）；
`volume`/`oi` = 前一根，值根 = `bar_index − 1`（可推导，无需额外列）。`start`（首根
开盘）与 `close`（末根收盘）取自身根。每个点携带其值根 K 线的：`volume`（该 K 线
成交量合计）、`oi`（该 K 线**结束时刻**持仓量，天勤 `close_oi` 口径；`open_oi` 口径
可由 `bar_index` 关联 K 线 CSV 补算）。
`trend_extreme_price`/`trend_extreme_bar_index`（2026-10-02 起）为 up/down 点的
**当次趋势段**实际极值：段 = 闭区间 `[上一已确认 up/down 点触发根, 触发根 − 1]`
（初始段起点 = bar 0，触发根归下一段，段长 = 触发根 − 段起点）；up 点 = 段内实际
最高、down 点 = 段内实际最低，平局取最早 bar；`start`/`close` 为空单元格。不变式：
up 点 `trend_extreme_price` ≥ `price`、down 点 ≤ `price`（相等当且仅当极值就在
值根）。段长不入 CSV（可由 `bar_index` + 点序列确定性推导；数据层子函数
`recent_trend_extremes` 返回段长）。
后四列为相邻点**相对值**，由点序列确定性派生：`dt_minutes` = 当前点与前一点
`timestamp` 之差（单位分钟）；`price_ratio`/`volume_ratio`/`oi_ratio` = 当前点值 /
前一点值（比值，减 1 即变化幅度）。首点无前一点 → 四列均为空单元格；前一点值为 0
（真实存在无成交分钟）时比值无定义，同样留空，不产生 `inf`。
同名 sidecar 记录
`symbol/period/row_count/window_first_timestamp/window_last_timestamp/`
`initial_direction_requested/initial_direction_resolved/input_source_data_version/file_sha256`。

## 配置与凭证安全

配置模板：`dataset/config/tianqin.example.yaml`（入库，凭证留空）。本地真实凭证写入
`dataset/config/tianqin.local.yaml`（已被 `.gitignore` 排除，禁止提交）：

```bash
cp dataset/config/tianqin.example.yaml dataset/config/tianqin.local.yaml
```

解析优先级（高 → 低）：**CLI 参数 > 环境变量 > 配置文件 > 内置默认值**。
环境变量：

```text
MARKETSENSE_TQ_ACCOUNT     # 凭证 account 覆盖
MARKETSENSE_TQ_PASSWORD    # 凭证 password 覆盖
MARKETSENSE_DATASET_CONFIG # 配置文件路径（未显式 --config 时使用）
MARKETSENSE_DATA_DIR       # output_dir 覆盖
```

- 凭证缺失时给出明确错误（说明缺哪个字段、可用的配置方式），**不打印凭证值**、不打印配置对象。
- YAML 解析失败时的错误消息**只含固定文案 + 文件路径 + 行列号**（如
  `配置文件 YAML 解析失败: <path>（YAML 文档组装失败；第 5 行，第 13 列）`）：
  不拼接 PyYAML 的 `str(exc)`/`problem`/`context`，因为其中可能回显出错行原文、
  未定义别名/锚点名或标签名（均可能来自凭证行）；`ReaderError` 由字符偏移换算行列号。
- 兜底脱敏（防凭证值被误填到其它字段）不会改写白名单文案（支持周期/格式列表）。
- 不读取任何内置凭证文件；配置由 `--config` / 环境变量注入。
- 默认 `output_dir: data`，已被根 `.gitignore` 忽略（`data/`、`*.csv` 双保险）。

## Python 接口（供导入复用）

```python
from dataset import (
    load_dataset_config, DatasetConfig,
    resolve_duration_seconds, supported_periods_text,
    standardize_klines, clean_serial_klines,
    validate_ohlcv, ValidationReport,
    save_ohlcv, load_ohlcv, LoadedOHLCV,
    TianQinProvider, DataProvider,
    TurningPoint, TrendExtreme, find_turning_points, resolve_initial_direction,
    save_turning_points, load_turning_points, WindowMeta,
    RelativeMetrics, relative_metrics, recent_trend_extremes,
)

# 最近 n 个高/低点的当次趋势段极值（时间倒序，含段长；n 超额/非法 → DatasetError）
loaded = load_turning_points("DCE.v2701", "1d", data_dir="data/turning_points")
highs, lows = recent_trend_extremes(loaded.points, n=1)
```

## 测试

```bash
/opt/anaconda3/envs/marketsense/bin/python -m pytest dataset/tests -q        # 全量离线测试
/opt/anaconda3/envs/marketsense/bin/python -m pytest dataset/tests -v        # 逐用例输出
```

- 全部测试**不触网**：天勤通过 `api_factory` 注入 `FakeTqApi` 桩。
- 2026-09-26 实测：`168 passed`（两次运行 9.72s / 10.80s）。
- 2026-09-30 实测（含 episode 流水线测试）：`277 passed`（11.64s）。
- 2026-10-02 实测（含当次趋势极值用例）：`320 passed, 1 xfailed`（11.95s；转折点用例 45 个）。
- 2026-10-02 实测（含日线折点极值用例，v5）：`338 passed, 1 xfailed`（15.43s；转折点用例 45 个）。
- 2026-10-03 实测（account-service：账户域迁移 + v8 账户行 + 跨片段净值链用例）：
  `344 passed, 1 xfailed`（14.07s）。

## 已知边界（未验证项）

- 历史区间模式（`--start/--end`）依赖 `get_kline_data_series`，需天勤专业版权限；
  本机权限**未验证**（`U-1`）。无权限时应改用 `--bars`。
- 本包不做实时订阅、不做模型/决策，也不定义最终模型输入格式（非目标）；
  `board-state`（盘面状态读取）是首个状态 building block（研究 building block，非最终格式）。
- episode 训练数据生成已在真实片段上端到端运行（2026-10-01 首轮 `run-c1cb097177a3`；
  当前产物 `run-c364ea4a30ac`（v8 状态：账户行六键 + 跨片段净值链）；
  详细结果见主 README §5「首轮真实数据」）。
- K 线契约于 2026-09-24 扩展：新增固定持仓量列（`open_oi`/`close_oi`），转折点 CSV
  新增 `volume/oi/相对值` 列；旧格式落盘文件需重新 `fetch` + `turning-points` 再生成。
- 转折点发射语义于 2026-09-26 变更（甲口径：`up`/`down` 点 `timestamp`/`bar_index` =
  触发根，`price`/`volume`/`oi` 取前一根）。旧落盘文件与新文件同 schema（列集合相同），
  `load_turning_points` 不会拒绝旧文件，但值含义不同；须重新 `turning-points` /
  `prepare` 再生成（无兼容层）。
- 转折点 CSV 于 2026-10-02 扩展（11 → 13 列）：新增 `trend_extreme_price` /
  `trend_extreme_bar_index`（up/down 点的当次趋势段实际极值，检测触发序列不变），
  并新增数据层子函数 `recent_trend_extremes`（最近 n 个高/低点段极值，含段长）。
  旧 11 列文件 `load_turning_points` 按「缺少必需列」拒绝，须重新
  `python -m dataset turning-points` 再生成（无兼容层）。

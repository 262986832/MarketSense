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
python -m dataset episode-generate --segments FILE [--output-dir DIR] [--config FILE]
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
为模板）与已落盘 1 分钟 K 线，确定性地输出对齐 NanoJev 训练契约的按 split JSONL + 审计文件。
不联网、不取数、不训练模型、不修改 `NanoJev/`；模型可见价格一律用比值（分母 = 片段首根开盘价，
6 位小数）。模块与语义分层：`dataset/market_episode/replay.py`（mechanics：决策 K 线成交
模型、止损锚定、t-1 盯市回撤与死亡、片段末强平）与 `labels.py`（policy：盈亏比规则真值
标签 + (b) 采样），另有 `segments.py`（清单/品种配置/参数）、`nanojev_records.py`
（记录映射与落盘）、`audit.py`（确定性双跑、跨 split 隔离、泄漏抽查、计数汇总）。

前置与用法：先 `dataset fetch --period 1m` 落盘 K 线，再把
`dataset/config/symbols.example.yaml` 复制为 `symbols.local.yaml` 并填 `tick_size`，
然后用 `--segments` 指向片段清单（详细步骤与配置项见 `README.md` §7.2 小节 7）：

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
- 确定性：无墙钟/随机；同输入双跑输出 sha256 一致。
- `run_id` = sha256(片段清单 + 品种配置 + 参数) 的前 12 位；**不含行情数据指纹**，
  因此同清单/同参数下换数据重跑会**覆盖**同目录产物（README 及变更报告早期“不覆盖历史产物”的
  表述仅在清单/参数变化时成立）。
- **空 split 不报错**：某 split 为空时对应 `.jsonl` 为 0 字节且 CLI 仍以 0 退出；NanoJev trainer
  要求 `train`/`dev`/`test` 非空（`NanoJev/scripts/train_pipeline_decisions.py:477-479`），
  其 `--validate-only` 也不会拦空目录（`:423-424`）——生成后需自行确认非空。
- 本流水线已在真实片段上端到端运行（2026-10-01，DCE.v2701 2026-09 全月，
  `run-c1cb097177a3`，train 4188 / dev 886 / test 655 全部非空，NanoJev `--validate-only` 通过；
  见主 README §5「首轮真实数据」）。

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
point_index,kind,timestamp,price,bar_index,volume,oi,dt_minutes,price_ratio,volume_ratio,oi_ratio
```

`kind ∈ {start, up, down, close}`（2026-09-25 起反转类 kind 由 `high`/`low` 更名为
`up`/`down`）；`bar_index` 为窗口内 0 基下标（可回溯关联 K 线 CSV）。`up`/`down` 点在
**触发根**确认（up 态命中 `low[t] < low[t-1]`、down 态命中 `high[t] > high[t-1]` 的
那根）：`timestamp`/`bar_index` = 触发根 `t`；`price` = 前一根 high（up）或 low（down）；
`volume`/`oi` = 前一根，值根 = `bar_index − 1`（可推导，无需额外列）。`start`（首根
开盘）与 `close`（末根收盘）取自身根。每个点携带其值根 K 线的：`volume`（该 K 线
成交量合计）、`oi`（该 K 线**结束时刻**持仓量，天勤 `close_oi` 口径；`open_oi` 口径
可由 `bar_index` 关联 K 线 CSV 补算）。
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
    TurningPoint, find_turning_points, resolve_initial_direction,
    save_turning_points, load_turning_points, WindowMeta,
    RelativeMetrics, relative_metrics,
)
```

## 测试

```bash
/opt/anaconda3/envs/marketsense/bin/python -m pytest dataset/tests -q        # 全量离线测试
/opt/anaconda3/envs/marketsense/bin/python -m pytest dataset/tests -v        # 逐用例输出
```

- 全部测试**不触网**：天勤通过 `api_factory` 注入 `FakeTqApi` 桩。
- 2026-09-26 实测：`168 passed`（两次运行 9.72s / 10.80s）。
- 2026-09-30 实测（含 episode 流水线测试）：`277 passed`（11.64s）。

## 已知边界（未验证项）

- 历史区间模式（`--start/--end`）依赖 `get_kline_data_series`，需天勤专业版权限；
  本机权限**未验证**（`U-1`）。无权限时应改用 `--bars`。
- 本包不做实时订阅、不做模型/决策，也不定义最终模型输入格式（非目标）；
  `board-state`（盘面状态读取）是首个状态 building block（研究 building block，非最终格式）。
- episode 训练数据生成已在真实片段上端到端运行（2026-10-01，`run-c1cb097177a3`；
  详细结果见主 README §5「首轮真实数据」）。
- K 线契约于 2026-09-24 扩展：新增固定持仓量列（`open_oi`/`close_oi`），转折点 CSV
  新增 `volume/oi/相对值` 列；旧格式落盘文件需重新 `fetch` + `turning-points` 再生成。
- 转折点发射语义于 2026-09-26 变更（甲口径：`up`/`down` 点 `timestamp`/`bar_index` =
  触发根，`price`/`volume`/`oi` 取前一根）。旧落盘文件与新文件同 schema（列集合相同），
  `load_turning_points` 不会拒绝旧文件，但值含义不同；须重新 `turning-points` /
  `prepare` 再生成（无兼容层）。

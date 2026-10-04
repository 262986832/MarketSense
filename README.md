# MarketSense

> **机器盘感与概率决策研究项目**
>
> 状态：**训练数据准备阶段（为 NanoJev 准备训练数据）** · 最后更新：2026-09-26
>
> 本 README 是对当前仓库**实际内容**的说明，不是对未来的架构承诺。
> 正文内容区分四级状态：**Implemented（已实现，当前仓库）**、
> **Research Direction（研究方向）**、**Future Work（后续工作）**、
> **Open Questions（未决问题）**。

---

## 1. MarketSense 是什么

MarketSense 是一个面向量化交易的研究项目，试图回答一个具体的研究问题：

> **能否让模型学习市场行情的结构、状态，以及状态随时间的演化，
> 并输出具有统计意义的概率信息？**

它明确**不做**的事情是简化版的行情预测：

```text
K 线  →  涨 / 跌        ← MarketSense 不采用这条路线
```

它倾向的研究路线是：

```text
K 线
  ↓
市场感知 / Perception
  ↓
客观事实
  ↓
市场状态
  ↓
状态序列
  ↓
模型（当前候选：NanoJev，见 §5）
  ↓
未来结果概率
```

> ⚠️ 上面的链路是**研究假设**，不是已冻结的架构。见 §4–§6。

---

## 2. 项目想解决什么问题

1. **把 K 线转成机器可理解的、结构化的市场描述**——而不是把 K 线图直接丢给模型。
2. **让"事实"与"判断"分离**：可计算、可复现的客观事实由确定性代码产出；
   概率与状态判断交给模型。
3. **从"单点状态"走向"状态序列"**：不仅描述此刻的市场，还要研究状态如何随时间演化。
4. **输出概率而非结论**：在当前市场状态下，不同未来结果出现的概率分布，
   而不是"一定涨/一定跌"。

---

## 3. 当前仓库的实际结构

```text
MarketSense/
├── .devflow/            # DevFlow 安装状态
├── .pi/                 # pi Agent 技能（devflow 等）
├── .gitignore           # 忽略数据、模型产物、密钥等
├── README.md            # 本文件
├── AGENTS.md            # Agent 工作规范
├── dataset/             # ▶ 已实现：数据准备子应用（天勤 K 线 + 转折点 + episode 训练数据 + 盘面状态），用法见 §7.2
├── scripts/             # ▶ 已实现：辅助脚本（转折点价格折线图），用法见 §7.3
├── artifacts/           # DevFlow 流程产物（见 §8）
└── NanoJev/             # 第三方决策模型项目（含独立 .git），性质见下
```

- **`dataset/` 是 MarketSense 自身的第一块已实现代码**：数据准备子应用
  （天勤 K 线取数 → 标准化落盘 → 转折点提取 → episode 训练数据生成 → 盘面状态读取；
  `dataset/market_episode/` 为 episode 流水线，含机制/标签分层）；
  300 个离线测试通过 + 1 个 xfailed（已知缺陷最小复现），2026-10-01 实测，**详细用法见 §7.2**。
- 除 `dataset/` 与 `scripts/` 外，MarketSense 自身的特征 / 状态 / 市场描述 /
  模型训练与评估等**尚未建立**。
- `NanoJev/` 是**第三方决策模型项目**（0.6B 并行决策模型，Qwen3-0.6B 主干 + 决策头，
  输入 state/question/candidates、直接输出概率分布），**含独立 `.git`**，
  **不属于本仓库源码**：不修改其内部内容、不提交其内容。
  把它用作 MarketSense 的决策模型是**当前研究假设**（见 §5）。
- 上一版 bootstrap 的根级脚手架（`AGENT.md`、`PROJECT_STATUS.md`、
  `docs/PROJECT_CONSTITUTION.md`、`docs/REFERENCE_POLICY.md`、`specs/`、`research/`、
  `experiments/`、`tasks/`）已被**有意删除**（不要恢复）。

---

## 4. 关于"市场描述"与"概率"（Research Direction）

### 4.1 市场描述

MarketSense 想研究"K 线能否被转成机器可理解的结构化市场描述"。
**可能**包含（当前仅是研究方向，**不是最终规范**）：

```text
价格位置 · 波动状态 · 趋势状态 · 区间 · 突破 · 回撤
转折点 · 结构关系 · 时间关系 · 其他客观市场事实
```

> 现在**不要**把这些定义成最终 Schema。早期版本曾有一版观察语言（v3），
> 可作为起点参考而非既定终点。

### 4.2 概率

目标不是"未来一定涨还是一定跌"，而是：

```text
当前状态  →  未来可能路径  →  概率分布
```

具体**预测什么、如何定义 Label**——均未最终确定。
注意：概率输出是 MarketSense 要自己解决的核心问题——候选模型 NanoJev 虽以
概率分布为输出形态（见 §5），但**市场场景的概率目标如何定义**仍需自行研究。

---

## 5. 当前阶段：为 NanoJev 准备训练数据

**决策模型候选已选定 NanoJev**（研究假设，非最终承诺）：一个 0.6B 的并行决策模型
（Qwen3-0.6B 主干 + 决策头），输入**状态 + 问题 + 候选**，直接输出**概率分布**
（Choice / Boolean / Score 三种问题类型，零输出 token 解码；详见 `NanoJev/README.md`）。
它"输入状态与问题、输出概率"的形态与 MarketSense 的研究目标一致；
**但它面向游戏任务（Maze / Snake / ViZDoom），能否迁移到行情场景是本研究阶段要回答的问题。**

下一阶段的核心任务是**为 NanoJev 准备训练数据**。

**已具备的基础材料（已实现，用法见 §7）**：

| 材料 | 来源 | 状态 |
|---|---|---|
| K 线数据 | `dataset fetch`（天勤取数 → CSV + 来源指纹） | 已实现 |
| 转折点数据 | `dataset turning-points`（离线提取 → CSV） | 已实现 |
| 价格折线数据 | 转折点 `price` 序列（`data/turning_points/*.csv`）及其折线图（`scripts/plot_price_line.py` → PNG） | 已实现 |
| episode 训练数据（首轮） | `dataset episode-generate`（用户片段清单 → 确定性回放 → 盈亏比规则真值标签 → 按 split 的 NanoJev JSONL + 审计；用法见 §7.2） | 已实现，并已在真实片段上端到端生成通过契约校验（首轮 2026-10-01；状态模板 v2~v7 + 候选文案精简 2026-10-02；v8 账户行六键 + 跨片段净值链 2026-10-03；同日 v9 日线行趋势项升为涨势/跌势各 2 项 + 三分支趋势状态；2026-10-04 v10 联动行升突破动量，
见下方「首轮真实数据」） |
| 盘面状态数据 | `dataset board-state`（离线：固定的前日高/低/收 + 今日开盘，动态的今日最高/最低逐根更新，全部以今日开盘价为基准的相对价；用法见 §7.2） | 已实现，并已在真实数据上运行与交叉核对（2026-10-01） |

首轮 episode 训练数据流水线的语义已在 `artifacts/nanojev-training-data/01-requirement/requirement-report.md`
（需求）与 `artifacts/nanojev-training-data/02-design/tech-design.md`（设计）中冻结：
每分钟一个决策、成交与止损锚定"刚收盘那根 K 线"、固定 1 手、无机械止盈止损、
模型可见价格一律用比值表达、开仓/平仓/反手标签由盈亏比与反转 K 线规则给出。
2026-10-02 用户拍板方案 A：状态模板升为 **v2**（`render_state` 新增 board_state 行，
昨日高/低/收与今日高/低随状态文本进入训练记录）。

**首轮真实数据（2026-10-01 生成；2026-10-02 v2~v7 状态模板 + 候选文案精简；
2026-10-03 v8 账户行六键 + 跨片段净值链；同日 v9 日线行趋势项升级；
2026-10-04 v10 联动行升突破动量，均通过契约校验）**：
DCE.v2701（PVC）2026-09 全月 21 个交易日片段（`data/segments/sep2026.jsonl`：
train 9-1~9-18 / dev 9-21~9-24 / test 9-28~9-30，按时间顺序切分，夜盘归属其交易所交易日）→
v9 产物 `data/nanojev_dataset/run-de0026683ce6/`（模板 v9：日线行趋势项升为
涨势/跌势各最近 2 个折点（编号时间倒序 -1/-2：段极值比值 + 时长）+ 对称三分支趋势状态
`趋势=涨势中|跌势中|震荡 时长=n根`；账户行六键与 v8 同构；1d 数据窗口扩至
**2026-01-19~09-30 共 171 行**（备份 1d ∪ 原 1d 合并，重叠段 24 根逐值核实一致，
日线折点 CSV 重生成）；折点门槛收紧为 up/down 各 ≥2，**21/21 片段全保留、0 告警**
（v8 中因缺折点跳过的 9 个 train 片段回参），**train 4188 / dev 886 / test 655**（5729 条
记录）全部非空；净值链 100 → train 末 155.041756 → dev 末 160.000608 → test 末
166.489676（deaths=0）；与 v8 run 逐值比对：共享 12 片段 2933 条记录
gold/持仓/开仓价/止损价/昨日高/低/收全部一致，共享 12 片段净值增量 12/12 一致
（链入口因 train 9 片段回参而前移，绝对净值不同）；dev+test 1541/1541 条日线行趋势内容
与折点 CSV + 1d CSV 独立重算一致；确定性双跑 sha256 一致，
NanoJev 原生 `--validate-only` 契约硬门通过；
2026-10-04 v10（联动行升突破动量）生成**当前产物** `data/nanojev_dataset/run-d14277ebcf70/`
（模板 v10：联动行 `联动: 突破=<momentum|na>`（momentum ∈ [-1, +1]，首根 → `na`），
其余五行与 v9 逐字节同构；**train 4188 / dev 886 / test 655**（5729 条）全部非空；
与 v9 run 逐值比对**仅联动行变化**：`突破=na` 6 条（= bar0 有记录的片段），非 na 值域
[-1,1] mean 0.0337；净值链 100 → 155.041756 → 160.000608 → 166.489676 与 v9 一致；
三轮生成 sha256 幂等，NanoJev `--validate-only` 通过）；
磁盘现存 6 个 run：当前 run-d14277ebcf70（v10：联动行突破动量）+ run-de0026683ce6（v9）
+ run-c364ea4a30ac（v8：账户行六键
`持仓/开仓价/止损价/净值/今日/回撤` + 跨片段净值链，train 1392 / dev 886 / test 655）+
历史 `run-4f1cd33a3cfe/`（模板 v7：
键名中文化）、`run-6ae3e38289a3/`（模板 v6：行重排账户/日线/日内/联动/现价/盘口）、
`run-7cbd6516d46f/`（模板 v5：「日线」行新增 trend 四键）
——更早的 v1~v4 run（`run-3db1bf63afc2` 等）已由用户会话清理，不再在盘。
六个 run 的 NanoJev 原生 `--validate-only` 契约硬门均通过。
2026-10-02 用户拍板：候选文案精简为动作语义（买入开仓/卖出开仓/继续空仓/平仓/继续持有/反手），
成交价位由执行程序与滑点决定，不进模型输入（`QUESTION_SCHEMA = marketsense.episode_question.v1`）。
`tick_size = 5` 为**用户确认值**
（公开资料记载最小变动价位 1 元/吨，按用户确认执行，已记录于 `dataset/config/symbols.local.yaml` 注释）；
切分设计（每交易所交易日一个 episode + 时间顺序三分割）为执行时确定性默认，
调整只需编辑片段清单重跑（`run_id` 随清单/参数/schema 变化，旧产物保留）。

**后续任务（Future Work）**：

1. **补充其它材料与数据（未设计、未实现）**：在价格折线之外，补充训练所需的其它输入材料
   （如市场状态描述、候选构造、问题模板等——具体形式**未决**）。
2. **在真实片段上端到端生成并校验（已完成，2026-10-01）**：首个真实数据集
   `data/nanojev_dataset/run-c1cb097177a3/` 已生成并通过 NanoJev 原生
   `--validate-only` 契约校验（见上方「首轮真实数据」）。
3. **训练与评估（未设计）**：NanoJev 在行情场景的适用性仍未验证；训练、评估与
   趋势行情的 OOD 安全评估均未设计。**训练/推理入口硬性要求 CUDA**（见 §7.4）。
4. **推理期范围（未决）**：是否只允许在用户指定的震荡片段内决策、是否需识别非震荡
   并停止/拒绝交易，尚未决定。

> ⚠️ 以上后续任务只是**方向列表**，不是设计承诺；进入具体任务前需另行讨论确认。

---

## 6. Open Questions（未决问题）

- **训练数据的形式**：哪些材料进入 NanoJev 的输入？question / candidates 如何构造？
- **标签**：预测哪些未来结果？时间窗口多长？如何标注、如何避免特征/标签窗口交叉？
- **对接契约**：市场状态如何映射为 NanoJev 的 state/question/candidates 格式？
- Perception 的最终表示形式是什么？早期观察语言（v3）的思路是否沿用？
- 原始 K 线是否仍需**作为并行输入**（与结构化描述并存）？
- 转折点的价值有多大？如何整合？最佳参数是否重要？
- 模型应消费什么：文本描述、数值状态、折线图像、还是组合？
- NanoJev 面向游戏任务设计，**迁移到行情场景**需要什么改动？它是否为最佳方案？
- **评估体系**：概率预测如何评估才算有统计意义？

---

## 7. 环境、运行与使用

### 7.1 Python 环境

本项目使用独立 conda 环境：

```text
/opt/anaconda3/envs/marketsense/bin/python      # Python 3.11.13
```

> **conda 在本机需要手动启动**，Agent 与开发者都应显式初始化后再使用：
>
> ```bash
> source /opt/anaconda3/etc/profile.d/conda.sh
> conda activate marketsense
> ```
>
> 或直接调用绝对路径的解释器，避免依赖 shell 初始化：
>
> ```bash
> /opt/anaconda3/envs/marketsense/bin/python -m pytest
> ```
>
> 注意：系统自带的 `/usr/local/bin/python3` 的 `numpy` 已损坏，**不要**用它运行本项目。

### 7.2 `dataset/` 子应用：详细使用方法（数据准备）

`dataset/` 是 MarketSense **已实现**的数据准备子应用：把「天勤 K 线取数 → 标准化落盘
（CSV + 来源指纹）→ 转折点提取 → 转折点落盘」做成自持、可复现、可测试的能力。
本节是使用摘要；完整细节（模块出处、全部字段、Python 接口）见 `dataset/README.md`。

#### 前置

```bash
# conda 需手动初始化（或直接使用绝对路径解释器）
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marketsense
cd /Users/jiangdianjing/agentspace/MarketSense      # 必须在仓库根运行
```

无需安装：`dataset/` 是普通包，用 `python -m dataset` 从仓库根调用
（`python -m` 会把当前目录加入 `sys.path`）。不要用系统自带的 `/usr/local/bin/python3`。

#### 1. 配置天勤凭证（本地、不入库）

```bash
cp dataset/config/tianqin.example.yaml dataset/config/tianqin.local.yaml
# 编辑该文件，填入 tianqin.account / tianqin.password
```

`dataset/config/*.local.yaml` 已在 `.gitignore` 中排除，**不要提交**。

解析优先级（高 → 低）：**CLI 参数 > 环境变量 > 配置文件 > 内置默认值**。

| 环境变量 | 作用 |
|---|---|
| `MARKETSENSE_TQ_ACCOUNT` | 覆盖账号 |
| `MARKETSENSE_TQ_PASSWORD` | 覆盖密码 |
| `MARKETSENSE_DATASET_CONFIG` | 指定配置文件路径（未给 `--config` 时） |
| `MARKETSENSE_DATA_DIR` | 覆盖输出目录 |

凭证仅在内存中传递；错误消息**不会回显**账号/密码。

#### 2. 命令总览

```text
python -m dataset fetch          --symbol S [--symbol S2 ...] --period P (--bars N | --start ISO --end ISO) [--output-dir DIR] [--config FILE]
python -m dataset turning-points --symbol S [--symbol S2 ...] --period P [--initial-direction auto|up|down] [--data-dir DIR] [--output-dir DIR] [--config FILE]
python -m dataset prepare        --symbol S [--symbol S2 ...] --period P (--bars N | --start ISO --end ISO) [--initial-direction ...] [--output-dir DIR] [--config FILE]
python -m dataset episode-generate --segments FILE [--daily-turning-points-dir DIR] [--output-dir DIR] [--config FILE]
python -m dataset board-state    --symbol S [--symbol S2 ...] [--period 1m] [--start DATE --end DATE] [--data-dir DIR] [--output-dir DIR] [--config FILE]
```

| 子命令 | 是否联网 | 作用 |
|---|---|---|
| `fetch` | 是 | 取 K 线 → 校验 → 落盘 |
| `turning-points` | 否 | 读取已落盘 K 线 → 提取并落盘转折点 |
| `prepare` | 是 | `fetch` + 转折点，一步完成 |
| `episode-generate` | 否 | 片段清单 + 已落盘 1 分钟 K 线、日线及日线折点 → 按 split 的 NanoJev JSONL + 审计 |
| `board-state` | 否 | 读取已落盘 1m/1d K 线 → 盘面状态 CSV + sidecar |

参数要点：

- `--period` 必填，仅支持 `1m, 5m, 15m, 1h, 1d`；非法值在 stderr 打印支持列表并以非 0 退出。
- `--bars N` 与 `--start/--end` **互斥且必须给其一**；`--bars` 取值范围 `1..8964`。
- `--symbol` 可重复（一次命令处理多个品种）。
- `--initial-direction`：`auto`（默认）/ `up` / `down`。
- 退出码：`0` 成功；`1` 运行期失败（配置/凭证/取数/校验/读取）；`2` 用法错误。

#### 3. 示例

```bash
# (a) 最近 200 根已收盘 1 分钟 K 线 → data/ohlcv/DCE.v2701_1m.csv + .json
python -m dataset fetch --symbol DCE.v2701 --period 1m --bars 200

# (b) 指定历史区间（可能需天勤专业版权限，见下方「已知边界」）
python -m dataset fetch --symbol DCE.v2701 --period 1d --start 2024-01-01 --end 2025-12-31

# (c) 多品种
python -m dataset fetch --symbol DCE.v2701 --symbol SHFE.cu2612 --period 1d --bars 500

# (d) 离线提取转折点（不联网，可反复运行）
python -m dataset turning-points --symbol DCE.v2701 --period 1m
python -m dataset turning-points --symbol DCE.v2701 --period 1m --initial-direction up

# (e) 一步到位：取数 + 转折点
python -m dataset prepare --symbol DCE.v2701 --period 1d --bars 500

# (f) 指定输出目录与配置文件
python -m dataset fetch --symbol DCE.v2701 --period 1d --bars 100 \
  --output-dir /tmp/msdata --config dataset/config/tianqin.local.yaml
```

#### 4. 输出与数据契约

默认落地位置：

```text
data/ohlcv/{symbol}_{period}.csv           # K 线
data/ohlcv/{symbol}_{period}.json          # 来源指纹 sidecar
data/turning_points/{symbol}_{period}.csv  # 转折点
data/turning_points/{symbol}_{period}.json # 窗口 sidecar
```

K 线 CSV 列（顺序固定）：

```text
timestamp,open,high,low,close,volume,open_oi,close_oi
```

`timestamp` 为 ISO8601 带 `+08:00`、唯一、严格升序；OHLC 为 `float64`，`volume` 为 `int64`；
持仓量为固定列：`open_oi`/`close_oi` 分别是天勤该根 K 线**起始/结束时刻**的持仓量
（`int64`，两条取数路径均返回）。
sidecar 记录 `provider/symbol/period/row_count/first_timestamp/last_timestamp/`
`file_sha256/source_data_version` 等，**不含墙钟时间**（保证「同输入 → 同字节输出」）。

转折点 CSV 列：

```text
point_index,kind,timestamp,price,bar_index,volume,oi,trend_extreme_price,trend_extreme_bar_index,dt_minutes,price_ratio,volume_ratio,oi_ratio
```

`kind ∈ {start, up, down, close}`。`up`/`down` 点（原 `high`/`low`，2026-09-25 起更名）的
`timestamp/price/volume/oi` 取**极值 K 线的前一根**（锚点 `bar_index = 极值 bar − 1`，
`price` = 前一根 high/low）；极值落在 bar 0 时整点留在 bar 0、`price` 取 bar 0 自身
极值价。每个点携带其 `bar_index` 所指那根 K 线的 `volume`（成交量合计）
与 `oi`（该 K 线结束时刻持仓量，天勤 `close_oi` 口径）。`trend_extreme_price`/
`trend_extreme_bar_index`（2026-10-02 起）为 up/down 点的**当次趋势段**实际极值
（段 = 闭区间 `[上一已确认 up/down 点触发根, 触发根 − 1]`，初始段起点 = bar 0；
up 点 = 段内实际最高、down 点 = 段内实际最低，平局取最早 bar；`start`/`close` 为
空单元格；up 点极值 ≥ `price`、down 点 ≤ `price`）。后四列为相邻点相对值
（由点序列确定性派生）：`dt_minutes` = 与前一点时间差（单位分钟）；
`price_ratio`/`volume_ratio`/`oi_ratio` = 当前点值 / 前一点值（比值）。
首点无前一点 → 四列为空；前一点值为 0 时比值无定义，同样留空（不产生 `inf`）。
转折点输出是**中间数据**，**不是**最终模型输入格式。

#### 5. 测试

```bash
/opt/anaconda3/envs/marketsense/bin/python -m pytest dataset/tests -q
```

全部测试**离线、不触网**（天勤以 `FakeTqApi` 桩注入），且不修改生产代码。
2026-09-26 实测：`168 passed`（9.72s / 10.80s）；2026-09-30 加入 episode 流水线测试后
实测：`277 passed`（11.64s）。

#### 6. 已知边界与注意事项

- **历史区间模式**（`--start/--end`）依赖天勤 `get_kline_data_series`，**需专业版权限**；
  2026-10-01 实测本机为**免费版**，区间模式不可用（`U-1` 已解决）。任意历史区间取数
  请改用 `--bars`（1..8964）。
- `--bars 8964`（平台上限）时无法多取 1 根凑整，末根未收盘被剔除后实际至多返回
  `8963` 根；CLI 会给出告警（不静默）。
- 凭证与数据**不要提交**：`data/`、`*.csv`、`dataset/config/*.local.yaml` 已被 `.gitignore` 覆盖。
- 不做实时订阅；不实现 MarketState / 特征 / 描述 / 决策（非目标）。
- K 线/转折点契约于 2026-09-24 扩展（新增持仓量列与转折点增补信息/相对值列）；
  旧格式落盘文件需重新 `fetch` + `turning-points` 再生成。
- 转折点契约于 2026-09-25 变更：极值类 kind 更名（`high`/`low` → `up`/`down`）且
  `up`/`down` 点锚点前移至极值 K 线的前一根；旧 kind 落盘文件不再可读，需重新
  `turning-points` 再生成。
- 转折点 CSV 于 2026-10-02 扩展（11 → 13 列）：新增 `trend_extreme_price` /
  `trend_extreme_bar_index`（up/down 点的当次趋势段实际极值，检测触发序列不变），
  并新增数据层子函数 `recent_trend_extremes`（最近 n 个高/低点段极值，含段长）。
  旧 11 列文件 `load_turning_points` 按「缺少必需列」拒绝，需重新
  `python -m dataset turning-points` 再生成（无兼容层）。

#### 7. `episode-generate`：episode 训练数据生成（首轮流水线）

把用户指定的震荡片段清单与已落盘的 1 分钟 K 线，确定性生成对齐 NanoJev 训练契约的
按 split JSONL 训练集与审计文件。**不联网**、不训练模型、不修改 `NanoJev/`。

**前置（四步）**：

```bash
# 1) 1 分钟 K 线与日线必须已落盘（日线供 board_state 行的上一交易日值；流水线只读消费，不取数）
python -m dataset fetch --symbol DCE.v2701 --period 1m --bars 800
python -m dataset fetch --symbol DCE.v2701 --period 1d
#    日线折点 CSV 也必须已落盘（v5 日线行 trend 四值数据源）
python -m dataset turning-points --symbol DCE.v2701 --period 1d

# 2) 品种配置：复制模板并填 tick_size（symbols.local.yaml 已被 .gitignore 覆盖）
cp dataset/config/symbols.example.yaml dataset/config/symbols.local.yaml

# 3) 片段清单：按 dataset/config/segments.example.jsonl 的格式写入 data/segments/ 下
```

**运行**：

```bash
python -m dataset episode-generate --segments data/segments/my_segments.jsonl
# 若需改研究阈值/目录：自建一个 YAML（无内置模板）并用 --config 指定
python -m dataset episode-generate --segments data/segments/my_segments.jsonl \
  --output-dir data/nanojev_dataset --config /path/to/my_episode.yaml
```

| 参数 | 说明 |
|---|---|
| `--segments FILE` | 必填。片段清单 JSONL（每行一个片段 = 一个 episode） |
| `--daily-turning-points-dir DIR` | 日线折点输入目录（v5 日线行 trend 四值数据源；默认 `<data_dir>/../turning_points`，与 `turning-points` 子命令落盘布局一致；目录内缺 `{symbol}_1d.csv` 报错退出 1） |
| `--output-dir DIR` | 产物根目录（默认取配置 `episode.output_dir`，否则 `data/nanojev_dataset`） |
| `--config FILE` | 配置文件（YAML；未给时依次取 `MARKETSENSE_DATASET_CONFIG`、`dataset/config/tianqin.local.yaml`，再退回内置默认值） |

片段清单字段（模板 `dataset/config/segments.example.jsonl`）：

```text
schema(marketsense.segment.v1), segment_id, symbol, period(必须 1m),
start_ts, end_ts, split_role(train|dev|calibration|test|ood), notes(可选)
```

可配置项（配置文件 YAML，优先级：CLI 参数 > 环境变量 > 配置文件 > 内置默认值）：

```yaml
episode:
  data_dir: data/ohlcv                # K 线目录（1m + 1d；默认取 dataset.output_dir 的 ohlcv/ 子目录）
  output_dir: data/nanojev_dataset    # 产物根目录
  symbols: dataset/config/symbols.local.yaml   # 品种 tick 配置（默认与配置文件同目录的 symbols.local.yaml）
  reward_risk_threshold: 3            # 开仓/反手盈亏比阈值（严格大于；冻结默认 3）
  drawdown_threshold: 0.05            # 账户回撤死亡阈值（冻结默认 5%）
  price_precision: 6                  # 模型可见比值小数位（冻结默认 6）
  flat_sample_band_minutes: 2         # (b) 采样“不做”带宽度（分钟）
  breakthrough_window: 20             # 联动行突破动量窗口（最近 N 根已收盘 K 线/桶，v10）
  breakthrough_period: "1m"           # 联动行突破动量粒度（1m/5m/15m/1h；1d 拒绝；v10）
```

环境变量：`MARKETSENSE_DATASET_CONFIG`（配置文件路径）、`MARKETSENSE_DATA_DIR`（覆盖产物根目录）。

**校验规则**（不满足则报错退出）：字段完整且非空；`period` 必须 `1m`；symbol+period 已有落盘 K 线；
时间段落在数据范围内；同一 symbol+period 的不同 split 时间不重叠；清单必须覆盖 `train`/`dev`/`test`；
清单涉及的每个 symbol 都必须在品种配置里有正数 `tick_size`。

**输出**：

```text
<output_dir>/<run_id>/{train,dev,calibration,test,ood}.jsonl   # 按 split 的记录（5 个文件都会写出；空 split 为 0 字节）
<output_dir>/<run_id>/audit.json                              # 计数/指纹/冻结项/生成参数
```

- 模型可见价格一律为**比值**（分母 = 片段首根开盘价，6 位小数）；绝对 OHLC 只留在 `data/ohlcv/` 原始层。
- **状态文本（模板 v10）**：7 行（`账户:`/`日线:`/`日内:`/`联动:`/`现价:`/`盘口:`，行间连接符 `" \n "）；
  v7 拍板：训练 state 文本键名中文化；日内行 `bar=` 序号与联动行量/持仓量比值并入现价行
  （持仓量只保留收盘时刻，开盘时刻丢弃）；联动行变为 `na` 常量占位；trend 四键不改；
  v8 拍板（2026-10-03，account-service 任务）：账户行升六键加净值/今日（跨片段净值链同步接入，见下）；
  v9 拍板（2026-10-03，trend-state-v9 任务）：日线行趋势项升级（见下），其余行与 v8 逐字节同构；
  v10 拍板（2026-10-04，linkage-breakthrough 任务）：联动行升突破动量（见下），其余五行与 v9 逐字节同构；
  `账户: 持仓=<空仓|持多|持空>[ 开仓价=<..> 止损价=<..>] 净值=<..> 今日=<..> 回撤=<..>`
  （v8 键序固定为 持仓/[开仓价/止损价]/净值/今日/回撤，空仓时无开仓价/止损价两键；
  持仓值中文标签，持仓非空时开仓价/止损价在 净值 前；`净值`（净值尺度）=
  100×(1 + 跨片段累计已实现盈亏 + 当前浮盈)，初值 100；`今日` = 净值 − 片段起点净值
  （每片段重置，可为负）；`回撤` 公式不变 = (峰值 − 权益)/峰值，峰值跨片段延续）；
  **日线行（v4 起由原 board_state 行拆出；v5 起追加 trend 四键；v7 键名中文化；v9 起趋势项
  升为涨势/跌势各 2 项 + 三分支趋势状态）**：
  `昨日高/昨日低/昨日收`（键 `prev_h/prev_l/prev_c` 的 v7 中文名）=
  上一交易日日线高/低/收（来源 `{symbol}_1d.csv`，取严格早于片段交易日的最后一行）÷ 片段首根
  开盘价；`涨势(-1/-2, 最高=.., 时长=n根)` = 最近 **2 个**可用 up 折点（编号**时间倒序**：
  -1 最近、-2 次近）的段内实际最高价（`trend_extreme_price`，来源日线折点 CSV
  `data/turning_points/{symbol}_1d.csv`）÷ 片段首根开盘价 + 对应段长；
  `跌势(-1/-2, 最低=.., 时长=n根)` = 最近 2 个可用 down 折点的段内实际最低价 ÷ 片段首根
  开盘价 + 段长；「可确认」口径 = 折点确认根日期
  **严格早于**该 bar 的**决策交易日**（State(T) 不引用 T 日及之后确认的折点，无未来泄漏）；
  逐决策点各自取其决策交易日的可用折点；决策交易日归属与 `dataset board-state` 夜盘规则
  同口径（日盘 bar → 日历日；夜盘 bar ≥ 21:00 → 下一交易日）；
  `趋势=<涨势中|跌势中|震荡>` = 对称三分支分类：两方向 -1 段极值均严格大于 -2（HH+HL）→
  涨势中、均严格小于 -2（LL+LH）→ 跌势中，其余（含任一方向相等）→ 震荡；其后的
  `时长=n根` = 状态时长 = 最近可用折点（**不分方向**）确认根 → 决策交易日日线开盘的日线根数
  （= T 的 1d 行号 − 最近可用折点确认根 `bar_index`，恒 ≥ 1；只用 T 行号、不读 T 日线 OHLC，
  无未来泄漏）；
  **日内行（同上拆出；v7 起 `bar=` 序号移入现价行）**：`今高/今低`（键 `today_h/today_l` 的
  v7 中文名）= 片段首根至决策 K 线
  （含）的 1m 高/低累计极值 ÷ 片段首根开盘价（State(T) 只用 ≤ 决策 K 线数据，无未来泄漏）；
  **联动行（v10）**：`联动: 突破=<momentum|na>`（v7~v9 为 `na` 常量占位）= 突破动量
  （momentum ∈ [-1, +1]，非比值）——对截至决策 K 线 T（含）的片段 1m 序列：末尾
  min(N, 可用根数) 根**已收盘** K 线/桶（N = `breakthrough_window`，默认 20）相邻对三态信号
  （`cur.high>prev.high` 且 `cur.close>prev.close` → +1；对称跌突破 → -1；其余 0，严格比较）
  线性加权 `Σ(w·s)/Σw`（权重 1..n-1，最近信号权重最大）；粒度 = `breakthrough_period`
  （默认 `1m`，可配 5m/15m/1h；非 1m 从 1m 按整分钟桶重采样，收满判定 = 下一根 1m 开盘 ≥
  桶终点、片段尾桶未收满不可用；1d/未知周期配置即拒绝）；首根/可用根数 < 2 → `突破=na`；
  **现价行（v7）**：`开/高/低/收`（键 `o/h/l/c` 的 v7 中文名）= 决策 K 线 OHLC ÷ 片段首根
  开盘价；`bar=` = 片段内 0 基序号（v7 自日内行移入）；`成交量比=` = 决策 K 线成交量 ÷ 首根
  成交量、`持仓量比=` = 决策 K 线**收盘时刻**持仓量 ÷ 首根收盘时刻持仓量（v7 自联动行移入，
  开盘时刻持仓量丢弃）；`盘口: na` 为占位（真实 bid/ask 未接，非目标），联动行 v10 起为突破动量（见上）；
  片段无上一交易日日线 → 跳过该片段（`board_state.skipped_segments`）；v9 起：任一入选
  决策交易日在 1d 文件缺该交易日行，或可用 up/down 折点任一侧 < 2 → 跳过该片段
  （`trend_extremes.skipped_segments`，两类原因文案区分）+ stderr 告警，
  全部片段被跳过则硬报错；与 `dataset board-state` 同口径。
- **跨片段净值链（v8，2026-10-03）**：片段按 `(start, end, segment_id)` 全局时间升序稳定排序后
  **串行回放**（输入指纹保持清单原序，`run_id` 不受排序影响）；首评估片段起点净值 = 100.0
  （峰值起点同值）；每评估片段起点 = 上一评估片段末结算净值/峰值；跳过片段链**冻结穿过**
  （carry 不变）；dev 起点 = train 末片段结算净值、test 起点 = dev 末（split 时间连续时自然成立）；
  sep2026 实测链（v9/v10 run 同值）：train 末 155.041756 → dev 末 160.000608 → test 末 166.489676
  （v8 run 为 115.957034 → 120.915886 → 127.404954；
  audit per_segment `account_chain` 节逐片段可追溯；落盘前 `check_account_chain` 硬校验，
  账户行三值另由审计侧独立重放复算并与 state 文本逐值比对；详见 `dataset/README.md`）。
- **questions/candidates 文案（2026-10-02 拍板精简）**：唯一 choice 题 `next_action`，
  候选文案只留动作语义——空仓 `open_long=买入开仓 / open_short=卖出开仓 / stay_flat=继续空仓`，
  持仓 `close=平仓 / hold=继续持有 / reverse=反手`；成交价位由执行程序与滑点决定，不进模型输入。
  questions 文本由 `QUESTION_SCHEMA = marketsense.episode_question.v1` 标记（首次建立），
  纳入审计 `input.question_schema_sha256` 与 `run_id` 哈希；文案再演进必须换标记（产生新 run）。
- `tick_size` 按品种配置：成交价 = 决策 K 线最高 + 1 tick / 最低 − 1 tick；止损距离 = 当根振幅 + 1 tick。
- **确定性**：无墙钟/随机；同输入两次生成输出 sha256 一致（`audit.json` 记录各文件指纹与双跑比对依据）。
- **`run_id` 组成**：`run-<12 位十六进制>` = sha256(片段清单 + 品种配置 + 参数 + 状态 schema 标记
  + questions schema 标记)。**状态/候选文案变更产生新 run**（旧产物保留）；行情数据本身变化
  仍不会改变 `run_id`：同清单/同参数下换数据重跑会覆盖同目录产物（若要保留旧产物，
  请改 `--output-dir` 或用不同清单）。
- 生成后可用 NanoJev 原生校验器作契约硬门（只读执行第三方脚本）：
  `python3 NanoJev/scripts/train_pipeline_decisions.py --validate-only --input <run_dir>`。

**已知限制（实测）**：

- **空 split 不报错**：若筛选结果使某个 split 为空，对应 `.jsonl` 为 0 字节，CLI 仍以 0 退出；
  而 NanoJev trainer 要求 `train`/`dev`/`test` 非空（`NanoJev/scripts/train_pipeline_decisions.py:477-479`），
  且其 `--validate-only` 对空目录也不会拦截（`:423-424`）。生成后请自行确认非空。
- 2026-10-01 已在真实片段上端到端运行（DCE.v2701，2026-09 全月，首轮 `run-c1cb097177a3`）；
  2026-10-02 模板 v2（含 board_state 行）重生成 `run-13aff088b982`；同日候选文案精简
  生成 `run-b25cfd1ff370`；同日状态模板 v3（行间换行符前后加空格）生成 `run-36b037252a62`；
  同日状态模板 v4（六部分中文标签重排：账户/联动/日线/日内/现价/盘口，`bar=` 归入日内行，
  盘口 `na` 占位）生成 `run-3db1bf63afc2`（train 4188 / dev 886 / test 655）；
  同日状态模板 v5（「日线」行追加 `trend_up/trend_up_len/trend_dn/trend_dn_len` 4 键，
  数据源 = 日线折点 CSV，确认根日期 < 决策交易日；9 个 train 片段 9-1~9-11 因缺 up 折点
  跳过并告警）生成 `run-7cbd6516d46f`，train 1392 / dev 886 / test 655
  全部非空，NanoJev `--validate-only` 通过；
  同日状态模板 v6（行重排：账户/日线/日内/联动/现价/盘口，联动行移至日内行之后）
  生成 `run-6ae3e38289a3`；
  同日状态模板 v7（键名中文化 + bar 序号/量/持仓量比值并入现价行 + 联动行 na 占位，
  trend 四键不改）生成 `run-4f1cd33a3cfe`；
  2026-10-03 状态模板 v8（账户行升六键 `持仓/开仓价/止损价/净值/今日/回撤` + 跨片段净值链，
  账户域迁至 `dataset/account.py`）生成 `run-c364ea4a30ac`，
  train 1392 / dev 886 / test 655 全部非空，净值链 100 → train 末 115.957034 → dev 末
  120.915886 → test 末 127.404954（deaths=0，死亡/止损级联未触发），与 v7 run 逐值比对：
  gold/持仓/开仓价/止损价全部一致（回撤因峰值跨片段延续数值不同、公式不变），
  确定性双跑 sha256 一致，NanoJev `--validate-only` 通过；
  同日 1d 数据窗口扩至 2026-01-19~09-30（备份 1d ∪ 原 1d 合并，重叠段 24 根逐值一致，
  日线折点 CSV 重生成），状态模板 v9（日线行趋势项升为涨势/跌势各 2 项 + 三分支趋势状态
  + 状态时长；折点门槛 up/down 各 ≥2）生成**当前产物** `run-de0026683ce6`：
  **21/21 片段全保留、0 告警，train 4188 / dev 886 / test 655** 全部非空，净值链
  100 → train 末 155.041756 → dev 末 160.000608 → test 末 166.489676（deaths=0），
  与 v8 run 共享 12 片段逐值比对一致（gold/持仓/开仓价/止损价/昨日高/低/收；
  净值链入口因 train 9 片段回参而前移，绝对净值不同），
  dev+test 1541/1541 条日线行趋势内容独立重算一致，确定性双跑 sha256 一致，
  NanoJev `--validate-only` 通过；
  2026-10-04 状态模板 v10（联动行升突破动量 `联动: 突破=<momentum|na>`；首根 → na；
  窗口默认 20/粒度默认 1m 均可配）生成**当前产物** `run-d14277ebcf70`：
  train 4188 / dev 886 / test 655，与 v9 run 逐值比对仅联动行变化（`突破=na` 6 条 =
  bar0 有记录的片段；非 na 值域 [-1,1] mean 0.0337），净值链与 v9 一致，
  三轮 sha256 幂等，NanoJev `--validate-only` 通过；
  磁盘现存 6 个 run（v5/v6/v7/v8/v9/v10；更早的 v1~v4 run 已由用户会话清理）；
  v2/v3/v4 run 的 board_state 值与 `dataset board-state` CSV 全量交叉核对一致；
  「生成后自行确认非空」的提醒仍适用于任何新清单。

> 语义细节（冻结的标签与执行规则、实现阶段冻结项及默认值）见
> `artifacts/nanojev-training-data/02-design/tech-design.md`；完整契约见 `dataset/README.md`。

#### 8. `board-state`：盘面状态读取

离线读取已落盘 1 分钟 + 日线 K 线，按交易日窗口逐根维护盘面状态（2026-10-01 新增，
研究 building block，**非最终模型输入格式**）：

```bash
# 前置：1m 与 1d K 线均已落盘
python -m dataset board-state --symbol DCE.v2701 --start 2026-09-01 --end 2026-09-30
# 全部交易日（不限区间）
python -m dataset board-state --symbol DCE.v2701
```

语义（确定性、无未来数据泄漏）：

- **窗口**：交易日 `T` 的窗口 = 归属 `T` 的夜盘 K 线（前一历日 21:00 起，周五夜盘 → 周一）
  + `T` 的日盘 K 线（≤ 14:59），与片段清单同口径；交易日 = 存在日盘 K 线的日历日。
- **固定状态**：`prev_day_high` / `prev_day_low` / `prev_day_close`（上一交易日日线 OHLC，
  来源 1d 落盘）、`today_open`（窗口首根开盘价 = 日线开盘口径）。
- **动态状态**：`today_high` / `today_low` = 已处理 K 线高/低点累计 max/min
  （`State(T)` 只用 ≤ T 的 K 线）；初始（未处理前）= `today_open`（平盘先验）。
- **相对价**：六列一律 = 值 / `today_open`（6 位小数），`today_open` 恒为 `1.000000`；
  绝对 OHLC 只留在 `data/ohlcv/` 原始层。

输出：`data/board_state/{symbol}_board_state.csv` + sidecar（每根 K 线一行；列
`trade_date, timestamp, bar_index, prev_day_high, prev_day_low, prev_day_close, today_open, today_high, today_low`；
sidecar 含语义说明、1m/1d 来源指纹、交易日与跳过日清单；无墙钟，同输入 → 同字节输出）。

已知边界：需先 `fetch --period 1m`/`1d` 落盘（缺日线 exit 1）；1m 数据起点首个交易日无前置
交易日 → 跳过并 stderr 告警；上一交易日在 1m 序列中存在但日线缺该日 → 报错（数据不一致）。

2026-10-01 实测（DCE.v2701，2026-09 全月）：21 个交易日 7125 行；每日末行 `today_high`/`today_low`
与当日日线高/低（比值）一致、`prev_day_*` 与上一交易日日线一致、`today_open` 恒为 1；
窗口首根抽查（9-1 = 8-31 21:00 夜盘、9-21 = 9-18 周五夜盘、9-28 = 09:00 中秋无夜盘）符合归属规则；
两次运行 CSV+sidecar 字节一致。

### 7.3 `scripts/plot_price_line.py`：转折点价格折线图（快速可视化）

`scripts/plot_price_line.py` 是一个独立小脚本：把转折点 CSV
（`data/turning_points/{symbol}_{period}.csv`，即 §7.2 `turning-points` 的产物）中的
`price` 画成折线图，用于直观查看价格走势。

- 横坐标 = **逐行累加的 `dt_minutes`**（首点 start 为 0）：水平间距与原始数据的
  时间比例一致（`dt_minutes` 为自然日分钟、含周末），不是按行号均匀排布；
- 纵坐标 = 原始 `price`（不缩放、不归一化）；
- 末行 `close` 的 `dt_minutes=0`，与前一转折点共用横坐标（忠实于数据）；
- 最简呈现：单色折线 + 数据点标记，无 kind（up/down/start/close）着色与标注。

#### 用法

```bash
# (a) 默认：读取 data/turning_points/DCE.v2701_1d.csv，
#     输出到同目录 DCE.v2701_1d_price.png
/opt/anaconda3/bin/python scripts/plot_price_line.py

# (b) 指定其他转折点 CSV
/opt/anaconda3/bin/python scripts/plot_price_line.py data/turning_points/DCE.v2701_1m.csv

# (c) 指定输出路径
/opt/anaconda3/bin/python scripts/plot_price_line.py -o /tmp/v2701_1d_price.png
```

> **环境注意**：脚本仅依赖标准库 + matplotlib。项目 `marketsense` 环境
> **当前未安装 matplotlib**，上述示例使用 base conda 的 `/opt/anaconda3/bin/python`
> （含 matplotlib 3.10.0，2026-09-25 实测）。若要在 `marketsense` 环境运行，
> 需先向该环境安装 matplotlib（引入新依赖，由使用者自行决策）。

输出：终端打印点数、x/y 范围与保存路径；PNG 为 12×5 英寸、150 dpi 的折线图。

### 7.4 NanoJev 训练/推理环境要求与 CPU 冒烟验证（2026-10-01 实测）

`NanoJev/` 的训练与推理入口**硬性要求 CUDA**，与本机内存无关：

```text
NanoJev/scripts/train_pipeline_decisions.py（训练入口，约 455 行）：
  if not torch.cuda.is_available(): raise RuntimeError("A usable CUDA device with the requested precision is required")
NanoJev/scripts/predict_toy_decisions.py（推理入口）：
  「此原型推理入口需要可用CUDA设备；本命令未启用CPU或远程回退」
```

**Mac Intel 16G（无 CUDA）上的三级 CPU 冒烟验证（全部通过，数据通路无问题）**：

1. `--self-check`：真实训练器代码在 CPU 上跑 schema / complete-question 预算 /
   数值梯度检查（装 torch 即可，不需要 CUDA、不下载模型）；
2. tokenize 级：真实记录经训练器自身 `validate_training_row` + `prepare_examples`
   （与训练入口同参调用）处理，候选路径全部合法（EOS 结尾、id 合法、≤ max_length）；
3. 前向级：Qwen3-0.6B 底座 fp32 + `DecisionModel(backbone, "attention")`（与训练入口
   同构造方式），CPU 前向 6 问题约 18s：logits 有限、概率归一
   （近均匀 = 零初始化设计预期，argmax 未命中属预期）。

CPU 冒烟环境（仓库外、可随时清理，**未污染任何现有环境**：marketsense 及其他 conda
环境、`NanoJev/`、仓库代码零改动）：

```bash
/opt/anaconda3/envs/trader_test/bin/python -m venv ~/.venvs/nanojev-smoke
# 注意：默认 PyPI 在本机下载 torch 会超时（>600s 实测），用清华镜像：
~/.venvs/nanojev-smoke/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  torch==2.2.2 "transformers==4.51.3" safetensors "numpy<2"
```

**模型缓存（已下载，可复用）**：`Qwen/Qwen3-0.6B`（~1.2GB，snapshot `c1899de2`）位于
`~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/`。再次运行 `--self-check`、
tokenize/前向冒烟或未来训练（CUDA 机）时 `from_pretrained` 直接复用，无需重新下载；
清理方式 = 删除该目录。

**冒烟过程中记录的两个坑（均非数据问题；以后写 tokenizer/模型加载脚本前先读）**：

1. **词表语义**：`tokenizer.vocab_size`（Qwen3 = 151643）**不含 added special tokens**；
   Qwen3 的 `eos_token_id = 151645`（`<|im_end|>`）超出基础词表属正常。校验 token id
   合法性必须用 `AutoConfig` 的 `config.vocab_size`（151936）；用 `tokenizer.vocab_size`
   会误报「非法 token id」（2026-10-01 冒烟实际踩过并修正）。
2. **transformers 版本 API 差异**：NanoJev 按其固定规格 transformers 5.17 编写，
   `AutoModel.from_pretrained(..., dtype=torch.float32)` 的 `dtype=` 是 5.x 才有的别名；
   transformers 4.51 会报
   `Qwen3Model.__init__() got an unexpected keyword argument 'dtype'`，须改用
   `torch_dtype=torch.float32`（2026-10-01 冒烟实际踩过并适配）。

注意版本差异：冒烟环境（torch 2.2.2 / transformers 4.51.3）≠ `NanoJev/requirements-toy.txt`
固定规格（torch==2.14.0 / transformers==5.17.0，来自其 A100 开发机）。
**真实训练机（CUDA）应按 `requirements-toy.txt` 安装固定版本**；数据产物无需任何改动：

```bash
python3 NanoJev/scripts/train_pipeline_decisions.py --input data/nanojev_dataset/run-3db1bf63afc2
```

---

## 8. 文档索引（核心）

| 文档 | 位置 | 说明 |
|---|---|---|
| Agent 工作规范 | `AGENTS.md` | 未来 Agent 必读 |
| 决策模型参考项目 | `NanoJev/README.md` | 第三方 NanoJev 自述：0.6B 并行决策模型的输入契约与用法（以该项目文档为准） |
| **`dataset/` 使用说明** | `dataset/README.md` | 已实现数据准备子应用的完整用法（本文 §7.2 为其摘要） |
| 流程产物（认知建立） | `artifacts/project-bootstrap/` | 项目认知与 README/AGENTS 建立的 DevFlow 产物 |
| 流程产物（数据子应用） | `artifacts/training-data-app/` | `dataset/` 子应用的需求 / 设计 / 审查 / 测试报告 |
| 流程产物（转折点契约变更） | `artifacts/turning-point-updown-price/` | 2026-09-25 转折点契约变更的 DevFlow 产物 |
| 流程产物（价格折线图脚本） | `artifacts/plot-turning-points-price/` | `scripts/plot_price_line.py` 的 DevFlow 产物（用法见 §7.3） |
| 流程产物（全量取数） | `artifacts/fetch-v2701-max-klines/` | DCE.v2701 全量 1m/1d 取数（2026-10-01；含区间模式需专业版的实测记录） |
| 流程产物（episode 数据） | `artifacts/nanojev-episode-sep2026/` | 2026-09 真实片段 episode 数据生成与契约校验（首轮 run-c1cb097177a3；v2/v3 含盘面状态见 §5「首轮真实数据」） |
| 流程产物（CPU 冒烟） | `artifacts/nanojev-smoke-intel-mac/` | Mac Intel 数据通路三级冒烟验证与 CUDA 硬约束结论 |


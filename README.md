# MarketSense

> **机器盘感与概率决策研究项目**
>
> 状态：**认知建立阶段（Bootstrap v2）** · 最后更新：2026-09-25
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
模型
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
├── .gitignore           # 忽略 reference/、数据、模型产物、密钥等
├── README.md            # 本文件
├── AGENTS.md            # Agent 工作规范
├── dataset/             # ▶ 已实现：数据准备子应用（天勤 K 线 + 转折点），用法见 §7.2
├── scripts/             # ▶ 已实现：辅助脚本（转折点价格折线图），用法见 §7.3
├── artifacts/           # DevFlow 流程产物（见 §8）
└── reference/           # 参考内容，不入库
```

- **`dataset/` 是 MarketSense 自身的第一块已实现代码**：数据准备子应用
  （天勤 K 线取数 → 标准化落盘 → 转折点提取；179 个离线测试通过，2026-09-25 实测），
  **详细用法见 §7.2**。其中部分模块移植自参考版本的数据层
  （模块出处对照见 `dataset/README.md`），**运行时不 import `reference/`**。
- 除 `dataset/` 外，MarketSense 自身的模型 / 特征 / 状态 / 描述 / 决策等**尚未建立**。
- `reference/` 是**参考内容**（第三方项目），已加入 `.gitignore`，
  **不纳入版本管理**；阅读它得到的是参考信息，不等于 MarketSense 的既定设计。
  定位、资产盘点与验证基线见**附录 A**（凭证安全提醒见 R6）。
- 上一版 bootstrap 的根级脚手架（`AGENT.md`、`PROJECT_STATUS.md`、
  `docs/PROJECT_CONSTITUTION.md`、`docs/REFERENCE_POLICY.md`、`specs/`、`research/`、
  `experiments/`、`tasks/`）已被**有意删除**（未提交，保持原样、不恢复）：
  commit `50d89d8` 的 bootstrap 自述未深入阅读 `reference/`，本次 README/AGENTS 是对该缺口的修正。

---

## 4. 关于"市场描述"与"概率"（Research Direction）

### 4.1 市场描述

MarketSense 想研究"K 线能否被转成机器可理解的结构化市场描述"。
**可能**包含（当前仅是研究方向，**不是最终规范**）：

```text
价格位置 · 波动状态 · 趋势状态 · 区间 · 突破 · 回撤
转折点 · 结构关系 · 时间关系 · 其他客观市场事实
```

> 现在**不要**把这些定义成最终 Schema。参考版本已有一版观察语言（v3），
> 是**可用的起点与参考**而非不可改的终点（见 R2）。

### 4.2 概率

目标不是"未来一定涨还是一定跌"，而是：

```text
当前状态  →  未来可能路径  →  概率分布
```

具体**预测什么、如何定义 Label、用什么模型**——均未最终确定。
注意：参考版本**没有任何概率输出能力**（见 R1），概率是 MarketSense 后续要做的事。

---

## 5. Future Work（后续大致方向）

1. **研究 Perception 输出的可用性**：参考版本的 `MarketState` + 观察语言（R2）
   能否作为状态表示？需要补什么、砍什么？
2. **让转折点进入状态理解**：转折点提取已在 `dataset/` 落地（§7.2），
   但它尚未接入任何状态/描述表示（参考版本中同样未接入，见 R4）。
3. **从状态到状态序列**：现有可用能力是"单点状态"，时间演化/序列建模是新的研究点。
4. **概率目标与 Label 定义**：定义"未来结果"是什么、如何标注、如何避免数据泄漏。
5. **模型选择**：Transformer 或小模型（`reference/minimind` 可作参考，见 R5），
   以及模型如何消费结构化描述。
6. **评估体系**：概率预测如何评估才算有统计意义。

---

## 6. Open Questions（未决问题）

- Perception 的最终表示形式是什么？现有观察语言（R2）是否需要重构？
- 原始 K 线是否仍需**作为并行输入**（与结构化描述并存）？
- 转折点的价值有多大？如何整合？最佳参数是否重要？
- 模型应消费什么：观察语言文本、状态数值、还是两者？
- Transformer 是否为最佳方案？
- 概率目标如何定义：预测哪些未来结果？时间窗口多长？
- **复用范围**：参考版本中哪些逻辑值得复用、以什么形式复用
  （迁移 / 派生 / 重写）——**待需求讨论完成后再定**（见 R1）。

---

## 7. 环境、运行与使用

### 7.1 Python 环境

本项目使用独立 conda 环境（`dataset/` 与参考版本共用）：

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
```

| 子命令 | 是否联网 | 作用 |
|---|---|---|
| `fetch` | 是 | 取 K 线 → 校验 → 落盘 |
| `turning-points` | 否 | 读取已落盘 K 线 → 提取并落盘转折点 |
| `prepare` | 是 | `fetch` + 转折点，一步完成 |

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
point_index,kind,timestamp,price,bar_index,volume,oi,dt_minutes,price_ratio,volume_ratio,oi_ratio
```

`kind ∈ {start, up, down, close}`。`up`/`down` 点（原 `high`/`low`，2026-09-25 起更名）的
`timestamp/price/volume/oi` 取**极值 K 线的前一根**（锚点 `bar_index = 极值 bar − 1`，
`price` = 前一根 high/low）；极值落在 bar 0 时整点留在 bar 0、`price` 取 bar 0 自身
极值价。每个点携带其 `bar_index` 所指那根 K 线的 `volume`（成交量合计）
与 `oi`（该 K 线结束时刻持仓量，天勤 `close_oi` 口径）。后四列为相邻点相对值
（由点序列确定性派生）：`dt_minutes` = 与前一点时间差（单位分钟）；
`price_ratio`/`volume_ratio`/`oi_ratio` = 当前点值 / 前一点值（比值）。
首点无前一点 → 四列为空；前一点值为 0 时比值无定义，同样留空（不产生 `inf`）。
转折点输出是**中间数据**，**不是**最终模型输入格式。

#### 5. 测试

```bash
/opt/anaconda3/envs/marketsense/bin/python -m pytest dataset/tests -q
```

179 个测试全部**离线、不触网**（天勤以 `FakeTqApi` 桩注入），且不修改生产代码
（2026-09-25 实测：`179 passed in 10.91s`）。

#### 6. 已知边界与注意事项

- **历史区间模式**（`--start/--end`）依赖天勤 `get_kline_data_series`，**可能需专业版权限**；
  本机权限**未验证**（`U-1`）。无权限时请改用 `--bars`。
- `--bars 8964`（平台上限）时无法多取 1 根凑整，末根未收盘被剔除后实际至多返回
  `8963` 根；CLI 会给出告警（不静默）。
- 凭证与数据**不要提交**：`data/`、`*.csv`、`dataset/config/*.local.yaml` 已被 `.gitignore` 覆盖。
- 不做实时订阅；不实现 MarketState / 特征 / 描述 / 决策（非目标）。
- K 线/转折点契约于 2026-09-24 扩展（新增持仓量列与转折点增补信息/相对值列）；
  旧格式落盘文件需重新 `fetch` + `turning-points` 再生成。
- 转折点契约于 2026-09-25 变更：极值类 kind 更名（`high`/`low` → `up`/`down`）且
  `up`/`down` 点锚点前移至极值 K 线的前一根；旧 kind 落盘文件不再可读，需重新
  `turning-points` 再生成。

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

---

## 8. 文档索引（核心）

| 文档 | 位置 | 说明 |
|---|---|---|
| Agent 工作规范 | `AGENTS.md` | 未来 Agent 必读 |
| **`dataset/` 使用说明** | `dataset/README.md` | 已实现数据准备子应用的完整用法（本文 §7.2 为其摘要；含模块出处对照） |
| 流程产物（认知建立） | `artifacts/project-bootstrap/` | 项目认知与 README/AGENTS 建立的 DevFlow 产物 |
| 流程产物（数据子应用） | `artifacts/training-data-app/` | `dataset/` 子应用的需求 / 设计 / 审查 / 测试报告 |
| 流程产物（转折点契约变更） | `artifacts/turning-point-updown-price/` | 2026-09-25 转折点契约变更的 DevFlow 产物 |
| 流程产物（价格折线图脚本） | `artifacts/plot-turning-points-price/` | `scripts/plot_price_line.py` 的 DevFlow 产物（用法见 §7.3） |


# MarketSense

> **机器盘感与概率决策研究项目**
>
> 状态：**认知建立阶段（Bootstrap v2）** · 最后更新：2026-09-24
>
> 本 README 是对当前仓库**实际内容**的说明，不是对未来的架构承诺。
> 文档中区分四级状态：**Implemented in Reference（参考版本已实现）**、
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

> ⚠️ 上面的链路是**研究假设**，不是已冻结的架构。见 §7 与 §10。

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
├── artifacts/           # DevFlow 流程产物（如本次 project-bootstrap）
└── reference/           # 参考内容，不入库（详见 §4）
    ├── perception/      # 早期版本（已终止）的完整实现，仅作参考（package: marksense）
    └── minimind/        # 第三方开源小语言模型项目（含独立 .git）
```

- **MarketSense 自身的源代码尚未建立**；仓库根目录目前只有文档与流程产物。
- `reference/` 是**参考内容**，已加入 `.gitignore`，**不纳入本仓库版本管理**：
  其中 `perception/` 是**之前做过、现已终止的一个版本**（保留作参考与资产来源），
  `minimind/` 是第三方项目（自带独立 `.git`，原样放置）。
- 阅读 `reference/` 得到的是**参考信息**，不等于 MarketSense 的既定设计。

---

## 4. `reference/` 的性质与内容

`reference/` 里**不是同一种东西**，需要区分对待。

> **定位（已确认）**：`reference/perception/` 是**之前做过的一个版本，已经终止**，
> 现在**仅作参考**——我们只会复用其中的**一部分逻辑**，具体复用范围**待需求讨论完成后再定**（见 §8）。
> 因此它的状态、缺陷与内部文档都**不需要维护或修复**。

### 4.1 `reference/perception/` — 已终止的早期版本（参考）

- 它的 Python 包名是 **`marksense`**（`pyproject.toml`: `name = "marksense"`），
  它自己的 `README.md` 标题即 **"MarketSense 机器盘感与概率决策系统"**
  → 它是本项目**早期自己做过的一个版本**，整体移入 `reference/` 作为参考保留。
- 它含有真实可运行的代码、设计文档（`docs/01`~`docs/10`）、测试与真实行情数据，
  以及自己的工作纪律文件（`AGENTS.md`、`PROJECT_STATUS.md`、`TASKS.md`、`CHANGELOG.md`）。
- **它的实现不等于 MarketSense 的既定设计**；读它是为了判断"哪些逻辑值得复用"。

> **必须注意**：`reference/perception/README.md` 描述的是一套比"市场描述"更大的系统
> （`OHLCV → MarketState → Canonical Description → NanoJev → EvidenceVector → Decision`），
> 但其中 **NanoJev / Evidence / Decision 尚未实现**（见 §5）。

### 4.2 `reference/minimind/` — 第三方开源小语言模型项目

- 上游为开源项目 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)（Apache 2.0）。
- 内容：约 64M 参数的超小语言模型的**极简实现与完整训练链路**——
  预训练、SFT、LoRA、DPO、PPO/GRPO、工具调用、蒸馏等
  （`model/`、`trainer/`、`dataset/`、`scripts/`）。
- 与 MarketSense 的关系：**外部参考**，用来研究"小模型 + 可复现训练链路"这条路是否适用。
  **它不包含任何行情/市场逻辑。**

### 4.3 其他

- 本次检查未发现独立存放的论文、算法库等其他类别材料；
  `reference/` 顶层只有 `perception/` 与 `minimind/` 两个目录（截至 2026-09-24）。

---

## 5. 参考版本中已经实现的部分（可复用资产盘点）

> 这些是 `reference/perception/`（已终止的早期版本）中**已经存在、可供考察复用**的资产，
> **不是 MarketSense 当前的正式能力**，也不代表会被整体采用。
> 每一项都经**阅读源码 + 运行测试**核实，而非依据文档宣称。

### 5.1 已实现的链路

```text
OHLCV
  ↓  Phase 1  数据层
TianQinProvider / MarketDataLoader / DataValidator
  ↓  Phase 2  特征
ATR / Range / Rolling High-Low / Volume Ratio / Volatility
  ↓  Phase 3  市场状态
Range / Breakout / Pullback / Re-entry / Follow-through / Time-Context / Location / Structure
  ↓
MarketState（结构化状态数据类）
  ↓  Phase 4  标准化描述
Canonical Description = Market Observation Language v3
  ↓  Phase 5  训练数据
Question Dataset（Q001~Q005）/ FutureOutcome / Leakage 检查 / Dataset Validator
```

对应源码目录（`reference/perception/src/marksense/`）：

| 模块 | 目录 | 实现的职责 |
|---|---|---|
| 数据层 | `data/` | 天勤取数（tqsdk）、标准化、落盘、加载、校验、周期/时间工具 |
| 特征 | `features/` | `atr.py`、`range.py`、`rolling.py`、`volume.py`、`volatility.py` |
| 状态 | `state/` | `range_detector`、`breakout_detector`、`pullback_detector`、`reentry_detector`、`follow_through_detector`、`time_context`、`location`、`structure` → `market_state.py` |
| 描述 | `description/` | `canonical.py`（固定七行模板，v3） |
| 数据集 | `dataset/` | `question_dataset.py`、`future_outcome.py`、`leakage.py`、`validator.py` |

**MarketState 的核心字段**（`state/market_state.py`）——同时包含事件语义字段与
归一化数值字段：

```text
time_context, location, structure
range_age, range_width_atr
position_in_range, distance_to_upper_atr, distance_to_lower_atr
breakout_direction, breakout_distance_atr
pullback, pullback_depth_atr, reenter_range, follow_through
volume_state, volume_ratio
volatility_ratio, volatility_state
confirmation_lag
```

**Market Observation Language v3 的关键性质**（`description/canonical.py`、
`docs/market-observation.md`）：

- 输出为**固定七行模板**，每行为"有语义、无主观判断"的客观观察。
- **归一化 / 尺度不变**：用 ATR 倍数、区间相对位置、量比、波动比表达，
  使不同品种、不同价格尺度下同一行为产生同一语言。
- **绝对价格不进入观察语言**（保留在 Raw / 执行层）。
- 由 `tests/description/test_cross_instrument.py`（跨品种等价 + 平移不变性）强制。

### 5.2 验证结果（本次实际运行）

```bash
cd reference/perception
/opt/anaconda3/envs/marksense/bin/python -m pytest
# 结果：437 passed in 9.58s        （2026-09-24 实测）
```

> 说明：`reference/perception/PROJECT_STATUS.md` 中记录的是 `429 passed`，
> **状态文件滞后于实际代码**（见 §9 差异清单）。以实测 437 为准。

### 5.3 人工验证脚本（`reference/perception/scripts/`）

这些是**验证/分析用工具**，不属于核心链路：

| 脚本 | 作用 |
|---|---|
| `describe_ohlcv.py` | 离线回放：对已落盘 OHLCV 逐根输出市场描述，产物见 `docs/verification/` |
| `realtime_describe.py` | 在线实时：接天勤实时流，每根**新收盘** K 线打印一次描述 |
| `turning_points.py` | 转折点提取（见 §6） |

### 5.4 `turning_points.py` 当前的真实角色

`reference/perception/scripts/turning_points.py`：

- 按人工给定的「相邻 K 线破位」规则，把一段 OHLCV 压缩成一条**转折点路径**
  （`开盘价 → 转折极值… → 收盘价`）。
- 文件头注释明确写着它是 **「人工分析工具，不属于核心业务链路」**。
- 经核实：**`src/` 中没有任何代码引用它**（`grep` 无命中），它也没有接入
  MarketState 或 Canonical Description。
- 它拥有独立测试：`tests/test_turning_points.py`（8 passed）。

> **结论（含差异）**：`turning_points.py` 目前是一个**独立的离线分析脚本**，
> 而不是已接入的特征来源。把它作为"后续行情理解的重要特征来源"是
> **下一步的研究方向**，不是现状（见 §9.3）。

---

## 6. 关于"市场描述"与"概率"

### 6.1 市场描述（Research Direction）

MarketSense 想研究"K 线能否被转成机器可理解的结构化市场描述"。
**可能**包含（当前仅是研究方向，**不是最终规范**）：

```text
价格位置 · 波动状态 · 趋势状态 · 区间 · 突破 · 回撤
转折点 · 结构关系 · 时间关系 · 其他客观市场事实
```

> 现在**不要**把这些定义成最终 Schema。`reference/perception/` 已有的一版
> 观察语言（v3）是**可用的起点与参考**，而非不可改的终点。

### 6.2 概率（Research Direction）

目标不是"未来一定涨还是一定跌"，而是：

```text
当前状态  →  未来可能路径  →  概率分布
```

具体**预测什么、如何定义 Label、用什么模型**——均未最终确定。

---

## 7. Future Work（后续大致方向）

1. **研究 Perception 输出的可用性**：现有 `MarketState` + 观察语言能否作为状态表示？
   需要补什么、砍什么？
2. **整合转折点**：`turning_points.py` 的 Turning Points 如何成为状态理解的特征之一。
3. **从状态到状态序列**：现有实现是"单点状态"，时间演化/序列建模是新的研究点。
4. **概率目标与 Label 定义**：定义"未来结果"是什么、如何标注、如何避免数据泄漏。
5. **模型选择**：Transformer 或小模型（`reference/minimind` 可作参考），
   以及模型如何消费结构化描述。
6. **评估体系**：概率预测如何评估才算有统计意义。

---

## 8. Open Questions（未决问题）

- Perception 的最终表示形式是什么？现有观察语言是否需要重构？
- 原始 K 线是否仍需**作为并行输入**（与结构化描述并存）？
- turning points 的价值有多大？如何整合？最佳参数是否重要？
- 模型应消费什么：观察语言文本、MarketState 数值、还是两者？
- Transformer 是否为最佳方案？
- 概率目标如何定义：预测哪些未来结果？时间窗口多长？
- MarketSense 自己的源码结构如何组织（当前代码都在 `reference/perception/`）？
- **复用范围**：`reference/perception/` 中哪些逻辑值得复用、以什么形式复用
  （迁移 / 派生 / 重写）—— **待需求讨论完成后再定**。

---

## 9. 认知校正：实际代码与既有描述的差异

> 依据"以实际代码为准"的原则，明确记录以下差异。

### 9.1 Perception 的范围比"K 线 → 市场描述"更大

"把 K 线转成市场描述"是它的**产出之一**；它实际还包含
**数据接入与落盘、特征计算、状态机、训练数据集生成**。
它的 `README.md` 甚至描述了 `EvidenceVector` 与 `DecisionEngine`。

### 9.2 Perception 只实现了 Phase 1–5

- **已实现**：数据层、特征、MarketState、Canonical Description、Dataset。
- **未实现**：`NanoJev`（Phase 6）、`Evidence`（Phase 7）、`Decision`（Phase 8）、
  Full Pipeline / Replay（Phase 9）。
- 核实方式：`src/marksense/` 下**只有** `data/`、`features/`、`state/`、
  `description/`、`dataset/`，**没有** `nanojev/`、`evidence/`、`decision/`。
- 因此：Perception 目前**没有"概率输出"能力**——概率是 MarketSense 后续要做的事。

### 9.3 `turning_points.py` 是独立脚本，不是已接入特征

见 §5.4。文件自述"不属于核心业务链路"，且 `src/` 无引用。

### 9.4 命名冲突：有两个东西叫 "MarketSense"

`reference/perception/README.md` 的标题就是 "MarketSense"。
阅读时必须区分：**仓库根（本项目）** vs **`reference/perception/`（已终止的早期版本）**。

### 9.5 状态文件内部不一致

`reference/perception/PROJECT_STATUS.md` 中：

- §2 的阶段表把 Phase 3 / Phase 4 标为 `NOT_STARTED`，
  但 §1 的修订记录与实际代码都显示它们**已完成**（Canonical Description 已到 v3）。
- 记录的测试数 `429 passed` 与实测 `437 passed` 不符。

→ 该文件是**历史状态快照，已滞后**，不应作为唯一事实源；以代码与实测为准。

### 9.6 `config/market.yaml` 含明文凭证（不处理，仅提醒）

`reference/perception/config/market.yaml` 内含明文快期账号与密码。

- `reference/` 已在 `.gitignore` 中排除，**未进入版本库**；按"参考内容不处理"的约定，
  **本次不修改**该文件。
- 唯一需要遵守的一点：**不要把其中的凭证复制**到其他文件、文档、日志或 git 中。

### 9.7 历史脚手架已被有意移除

`git status` 显示以下文件已被删除（未提交）——这是一次**有意的重组**：
`AGENT.md`、`PROJECT_STATUS.md`、`docs/PROJECT_CONSTITUTION.md`、
`docs/REFERENCE_POLICY.md`、`specs/`、`research/`、`experiments/`、`tasks/`。

上一版 bootstrap（commit `50d89d8`）明确写着"Bootstrap 阶段未深入阅读其内容"，
即它**并未真正读过 `reference/`**。本次 README/AGENTS 是对该缺口的修正。
这些删除**保持原样**，不需要恢复。

---

## 10. 环境与运行

### 10.1 Python 环境

`reference/perception/`（参考版本）使用独立 conda 环境：

```text
/opt/anaconda3/envs/marketsense/bin/python      # Python 3.11.13
```

依赖（`reference/perception/pyproject.toml`）：`pandas`、`numpy`、`pyyaml`、
`tqsdk`、`pyarrow`。

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

### 10.2 运行参考版本的测试

```bash
cd reference/perception
/opt/anaconda3/envs/marketsense/bin/python -m pytest
```

---

## 11. 文档索引

> `reference/perception/` 下的文档均属于**已终止的早期版本**，只作参考，
> 不代表 MarketSense 的现行设计。

| 文档 | 位置 | 说明 |
|---|---|---|
| Agent 工作规范 | `AGENTS.md` | 未来 Agent 必读 |
| 参考版本总览 | `reference/perception/README.md` | 已终止早期版本的完整说明 |
| Perception 需求/设计 | `reference/perception/docs/01`~`08` | 需求、概要、详细设计、数据字典、任务书、验收 |
| 观察语言词表 | `reference/perception/docs/market-observation.md` | 现行 v3 词表 |
| 数据源设计 | `reference/perception/docs/10_天勤量化数据源设计.md` | 天勤接入设计 |
| 端到端验证样例 | `reference/perception/docs/verification/*.md` | 逐根描述的真实产物 |
| 流程产物 | `artifacts/project-bootstrap/` | 本次认知建立的 DevFlow 产物 |

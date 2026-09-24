# Architecture Map v0 — 第一版架构领域地图

> 状态：`Proposed`（v0，Bootstrap 阶段产出）
> 最后更新：2026-09-24 · 产生方式：Bootstrap 推导，**未经研究验证**

---

## 0. 重要声明

1. 本文件只是 **architectural domains（架构领域）的第一版草图**，用于划定讨论边界；
2. **不代表最终模块已经确定**，不代表实现结构、文件布局或类设计；
3. 所有具体设计（字段、类、Tensor shape、算法、层数、hidden size 等）留待 Architecture / Research 阶段的正式 Spec 与 ADR 决定；
4. 修改本文件属于架构决策：重大调整须走 ADR / 人工决策。

## 1. 架构领域（Domains）

| Domain | 职责（方向性描述，均 TBD） | 状态 |
| --- | --- | --- |
| **Market Perception** | 摄取并结构化原始市场数据，形成可用输入 | Not Decided |
| **Market State / DSL** | 将感知结果表示为某种中间状态/语言 | Not Decided |
| **Dataset** | 从 Market State 构建可训练数据集（含标签生成策略） | Not Decided |
| **Representation / Tokenization** | 将状态表示转换为模型输入序列（是否需要 tokenizer 待定） | Not Decided |
| **Model** | 序列建模主干（候选方向之一为 Transformer 类结构，未承诺） | Not Decided |
| **Probability Output** | 模型输出的概率语义与解码（何种条件分布待定） | Not Decided |
| **Training** | 训练目标、loss、优化流程 | Not Decided |
| **Evaluation** | 评估协议、指标、baseline、防泄漏规则 | Not Decided |
| **Experiment** | 实验编排、记录与复现支撑 | Not Decided（机制已建于 experiments/） |

## 2. 概念数据流（Conceptual Data Flow）

```text
Market Data
    ↓
Market Perception
    ↓
Market State
    ↓
Dataset
    ↓
Model
    ↓
Probability Output
    ↓
Evaluation
```

> 这是**概念级**数据流，仅表达信息流向：
> - 不确定具体字段、类、Tensor shape 或算法；
> - 不确定各环节是否合并/拆分；
> - 不承诺 Evaluation 结果会回溯影响上游（如在线学习闭环）——该问题列入 Open Questions。

## 3. 初步边界原则（待 Spec 验证）

- Perception 与 Model 之间的边界是当前最关键的未决问题之一（见 PROJECT_STATUS.md Open Questions）；
- 跨领域通信必须经过显式 Contract（specs/interfaces/），禁止隐式耦合；
- 任何"哪些信息允许进入模型 / 必须禁止进入模型"的规则属于 Contract 与 Evaluation 共同约束的范围，须显式定义以避免泄漏/前视偏差。

## 4. 关联

- Open Questions：[PROJECT_STATUS.md](../../PROJECT_STATUS.md)
- ADR 机制：[specs/decisions/README.md](../decisions/README.md)
- Reference 使用边界：[docs/REFERENCE_POLICY.md](../../docs/REFERENCE_POLICY.md)

# Project Constitution — MarketSense 项目宪法

> 本文件定义 MarketSense 的长期原则。所有 Agent 与未来的所有 Spec、代码、实验都必须遵守。
> 修改本文件属于**重大架构变化**，必须经过 Project Owner 人工决策（见 AGENT.md §13）。

---

## 1. Research First

重大设计必须有研究依据。进入高成本实现前，先做文献研究、技术调查或小规模验证实验，并将过程与结论记录到 `research/`。允许"直觉驱动"的探索，但不允许直觉直接变成不可追溯的核心架构。

## 2. Evidence over Assumption

不要把假设写成事实。Spec、文档、README 中必须区分：

- **已验证的结论**（给出 research/ 中的证据链接）
- **假设 / 草案**（标记 Proposed）
- **未决定事项**（标记 TBD / Not Decided，登记到 PROJECT_STATUS.md 的 Open Questions）

## 3. Specification before Implementation

核心模块先形成 Spec，再进入实现。**没有 Spec 的核心开发任务原则上不能开始**（流程与例外路径见 [specs/README.md](../specs/README.md)）。实现中发现 Spec 有问题，走 Proposal → Decision → 更新 Spec → 修改代码，不得静默偏离设计。

## 4. Reproducibility

实验、训练和评估必须尽可能可重复：

- 实验按 [experiments/README.md](../experiments/README.md) 模板记录（数据、配置、随机种子、环境、指标、结果）；
- 测试不得依赖网络、私有数据或不可复现的随机状态；
- 无法完全复现的历史结果必须如实标注其复现条件与局限。

## 5. Separation of Reference and Source

`reference/` 是参考资料（Research Reference / External Knowledge），**不是 MarketSense 源码**：

- 不自动成为本项目代码的一部分；
- 默认禁止修改；
- 不因其存在而预先约束 MarketSense 的架构选择。

## 6. On-demand Reference Research

Agent 不为"启动"而阅读 reference，只为"任务"而阅读 reference：

```text
Task → 确定需要什么知识 → 检索 reference → 研究相关部分
    → 记录研究结论到 research/ → 用于 Spec / Implementation
```

完整规则见 [REFERENCE_POLICY.md](REFERENCE_POLICY.md)。

## 7. Modular Architecture

模块之间通过**明确 Contract**连接。跨模块调用必须依赖已声明的接口（[specs/interfaces/](../specs/interfaces/README.md)），禁止隐式耦合（读对方内部状态、依赖未声明的实现细节）。

## 8. Agent Boundary

每个 Agent 必须拥有明确职责和修改范围：

- 职责 / 输入 / 输出 / 允许修改范围 / 禁止修改范围，缺一不可；
- 禁止多个 Agent 无边界修改同一核心模块；
- 并行开发通过 Contract + Interface + Ownership + Dependency 解耦。

## 9. Human Decision Points

以下事项不得由单个 Agent 静默决定，必须上报 Project Owner / 人工决策：

- 重大架构变化（模型范式、核心数据表示、模块边界重划）；
- 核心研究结论与既有 Spec 冲突；
- 训练目标 / Prediction Target 语义变化；
- 修改 `reference/`、引入新外部依赖或外部数据源；
- 涉及密钥、凭证、外部付费资源；
- 修改本宪法。

具体停止条件见 [AGENT.md §13](../AGENT.md)。

---

## 附：行为优先级

```text
正确建立项目基础 > 清晰定义工作协议 > 建立可持续协作机制
> 保持研究可追溯 > 代码数量
```

不要因为拥有 reference，就急于使用 reference；
不要因为能够写代码，就急于写代码。

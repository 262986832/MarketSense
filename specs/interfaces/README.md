# specs/interfaces/ — Contract / Interface Spec

> 模块之间通过**明确 Contract** 连接（项目宪法 §7）。跨模块通信必须依赖此处声明的接口，禁止隐式耦合。

## 状态

**当前为空 —— 尚无任何 Contract。** Contract 将在模块边界确定后，随模块 Spec 一并产生。

## Contract Spec 编写要求

- 命名：`interfaces/<producer>-to-<consumer>.md` 或按概念命名（如 `market-state-contract.md`）；
- 内容至少包括：
  - **Producer / Consumer**：谁产出、谁消费；
  - **数据语义**：每个字段的含义与单位（不预设具体类型/Tensor shape，除非已决策）；
  - **约束与不变量**：合法性条件、禁止事项（如信息泄漏/前视偏差规则）；
  - **版本与兼容**：破坏性变更如何演进（配合 ADR）；
  - **测试方式**：Contract 如何被验证。
- Contract 一经 `Accepted` 即对两侧模块构成约束；修改走 [specs/README.md §4](../README.md) 变更机制。

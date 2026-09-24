# specs/modules/ — 模块级 Spec

> 每个核心模块一份 Spec，命名：`modules/<domain-or-module>.md`（与 ARCHITECTURE_MAP 中的领域对应）。

## 状态

**当前为空 —— 尚无任何模块 Spec。** 这是刻意的：模块划分本身待 Architecture 阶段确定后，才产生模块 Spec。

## 编写要求

- 顶部标注：状态（Draft/Proposed/Accepted）、Owner、最后更新、关联的 architecture spec 与 ADR；
- 内容至少覆盖：问题定义 / 职责边界 / 输入输出（引用 interfaces/ Contract）/ 非目标 / 验收标准；
- 区分事实、假设（Proposed）、未决定（TBD）；
- 超出 Spec 范围的实现发现，走 [specs/README.md §4](../README.md) 的变更机制。

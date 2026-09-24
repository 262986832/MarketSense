# specs/ — Spec-Driven Development (SDD)

> MarketSense 采用 Spec-Driven Development：**没有 Spec 的核心开发任务，原则上不能直接开始。**

---

## 1. 目录结构

```text
specs/
├── README.md            # 本文件：SDD 工作流与变更机制
├── architecture/        # 架构级 Spec（当前：ARCHITECTURE_MAP.md v0）
├── modules/             # 模块级 Spec（每个核心模块一份；当前为空）
├── interfaces/          # 跨模块 Contract / Interface Spec（当前为空）
└── decisions/           # Architecture Decision Records（ADR；当前机制已建，尚无 ADR）
```

## 2. SDD 工作流

```text
Research
   ↓            # 文献 / reference / 技术调查，结论记录到 research/
Problem Definition
   ↓            # 明确问题、边界、非目标
Specification
   ↓            # 产出 specs/ 下的 Spec（architecture / module / interface）
Task
   ↓            # 在 tasks/ 建任务：范围、验收标准、依赖（见 tasks/README.md）
Implementation
   ↓            # 只实现 Spec 覆盖的内容；超范围先走变更机制
Test
   ↓            # 可运行、可重复的测试
Review
   ↓            # 自查 + Review Agent / 人工审查
Integration
   ↓            # 合入主分支，确认不破坏其他模块 Contract
Update Status
                # 更新 PROJECT_STATUS.md 与任务状态
```

## 3. Spec 状态词汇

| 状态 | 含义 |
| --- | --- |
| `Draft` | 起草中，不约束实现 |
| `Proposed` | 已成型，待评审/决策 |
| `Accepted` | 已接受，约束实现 |
| `Deprecated` | 已废弃（保留供追溯） |
| `Superseded by <link>` | 已被更新版本取代 |

## 4. 变更机制（发现 Spec 有问题时）

```text
发现问题
 ↓
Proposal（在 specs/decisions/ 建 ADR 或在 tasks/ 建变更任务，写明：动机、影响面、备选方案）
 ↓
Discussion（相关模块 Owner + 必要时人工）
 ↓
Decision（普通变更可由相关 Owner 批准；重大变更必须人工决策，见 AGENT.md §13）
 ↓
更新 Spec（先改 Spec，再改代码）
 ↓
修改代码（commit 中注明对应 Spec / ADR）
```

**禁止**：直接修改代码后遗忘设计原因；静默偏离 Accepted Spec。

## 5. 编写约定

- 每个 Spec 顶部标注：状态、负责人、最后更新日期、关联研究记录（research/）与上游决策（ADR）；
- 明确区分"已验证事实 / 假设（Proposed）/ 未决定（TBD）"（见项目宪法 §2）；
- Spec 中不写实现代码细节（伪代码可用于澄清语义）；
- 一份 Spec 一个主题，避免"上帝文档"。

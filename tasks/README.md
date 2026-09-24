# tasks/ — 任务机制

> Task 是 SDD 工作流中 Spec 与 Implementation 之间的执行单元。

## 状态

当前为空 —— 尚无任何 Task。Bootstrap 之后的第一批任务将产生于 Architecture / Specification 阶段。

## Task 文件

- 命名：`TASK-<编号>-<short-name>.md`（编号从 0001 递增）；
- 一个 Task 一个文件；完成后文件保留（作为历史），仅更新 Status。

## 必填字段

| 字段 | 说明 |
| --- | --- |
| Task ID | TASK-XXXX |
| Title | 一句话标题 |
| Purpose | 为什么做（关联问题/目标） |
| Related Spec | 关联的 specs/ 文件（核心开发任务必须非空） |
| Owner | 认领 Agent（认领时登记，避免重复工作） |
| Dependencies | 依赖的其他 Task / Spec / 决策 |
| Scope | 允许修改的文件/模块范围（Ownership 边界） |
| Acceptance Criteria | 可验证的完成标准 |
| Tests | 需要哪些测试 / 如何验证 |
| Status | `Open` → `Claimed` → `In Progress` → `In Review` → `Done` / `Blocked` / `Cancelled` |

## 规则

1. **核心开发任务没有 Related Spec 时不得开工**，先回 SDD 流程补 Spec（specs/README.md）；
2. 认领：将 Owner 改为自己、Status 改为 `Claimed`；放弃时恢复 `Open` 并注明原因；
3. 超出 Scope 的改动必须先扩 Task（说明理由）或另立 Task，禁止静默越界；
4. Blocked 超过需要人工决策的范畴时，同步登记到 PROJECT_STATUS.md 并按 AGENT.md §13 处理；
5. 完成时：Status → `Done`，并按 AGENT.md §11 更新 PROJECT_STATUS.md。

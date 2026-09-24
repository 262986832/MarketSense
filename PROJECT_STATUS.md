# PROJECT_STATUS.md — MarketSense 项目状态（Single Source of Truth）

> 任何 Agent 在开始工作前必须先读本文件；完成有项目级影响的工作后必须更新本文件。
> 最后更新：**2026-09-24**

---

## Current Phase

**M0 Bootstrap（已完成）** — 项目"操作系统"（规范 / SDD / 状态管理 / Agent 协作机制）已建立。

下一步阶段：**Architecture / Specification**（等待 Project Owner 指令，Agent 不得自行进入）。

## Current Milestone

- **M0 Bootstrap**：✅ 完成（2026-09-24）
- **M1 Architecture / Specification**：未开始 — 产出第一版架构 Spec、核心 Open Questions 的研究计划与初步答案
- M2+ （模块实现、实验体系等）：未规划，待 M1 产出后定义

## Completed

- [x] Git 仓库状态确认（master 分支，Bootstrap 前无历史 commit）
- [x] 工作区与 `reference/` 目录级结构确认（`minimind/`、`perception/`，未深入阅读）
- [x] README.md（项目说明，未知状态用 TBD 标记）
- [x] AGENT.md（Agent 工作规范与入口）
- [x] docs/PROJECT_CONSTITUTION.md（项目宪法）
- [x] docs/REFERENCE_POLICY.md（Reference Policy）
- [x] PROJECT_STATUS.md（本文件）
- [x] specs/ SDD 骨架（workflow、architecture map v0、modules / interfaces / decisions(ADR) 机制）
- [x] research/ 记录规则（含失败实验保留原则）
- [x] experiments/ 记录规则与模板
- [x] tasks/ 任务机制
- [x] .gitignore（含训练数据 / checkpoint / 缓存 / 日志 / 密钥防护；`reference/` 不入库）
- [x] Bootstrap 初始化 commit

## In Progress

（无）

## Blocked

（无）

## Next Actions

1. **等待 Project Owner 确认进入 Architecture / Specification 阶段**（人工决策点）；
2. 在该阶段中：对 Open Questions（见下）排优先级，形成研究计划；
3. 产出第一版 Architecture Spec（Market State 表示、Perception/Model 边界、Prediction Target 语义等），逐步转化为 ADR；
4. 视需要设立第一批职能 Agent（Perception / Model / Evaluation 等）并定义其 Ownership；
5. 建立测试基础设施与源码目录结构（由 Spec 决定，不预建）。

## Architecture Decisions

> ADR 登记处：[specs/decisions/](specs/decisions/README.md)。**当前尚无任何已接受（Accepted）的 ADR** —— Bootstrap 阶段只建立了 ADR 机制，未产生架构决策。

| ADR | 标题 | 状态 |
| --- | --- | --- |
| （暂无） | — | — |

## Open Questions

> 以下问题**全部未回答**，留给 Architecture / Research 阶段。Agent 不得在此阶段自行拍板。

- [ ] Market State 的最终表示形式是什么？
- [ ] Market DSL 是否需要？若需要，是否需要 tokenizer？
- [ ] Transformer（或最终模型）的输入是什么？
- [ ] Prediction Target 如何定义？
- [ ] 模型输出的 Probability 的语义是什么（何种条件分布 / 置信度）？
- [ ] 训练标签如何自动生成？
- [ ] Perception 与 Model 的边界在哪里？
- [ ] 哪些信息允许进入模型？
- [ ] 哪些信息必须禁止进入模型（防泄漏 / 前视偏差）？
- [ ] 评估体系与 baseline 如何建立？（新增于 Bootstrap）
- [ ] 源码目录结构与语言/工具链选型？（新增于 Bootstrap）

## Active Agents

| Agent | 职责 | 状态 |
| --- | --- | --- |
| Bootstrap Agent | 建立项目基础设施 | ✅ 已完成（2026-09-24），退出 |
| （暂无其他在任 Agent） | — | — |

> 职能 Agent（Perception / Market DSL / Model / Training / Evaluation / Integration / Review 等）尚未设立；设立时必须登记其职责、输入、输出、允许/禁止修改范围（见 AGENT.md §10）。

## Last Updated

**2026-09-24** — Bootstrap Agent 完成 M0 Bootstrap 并生成初始化 commit。

# AGENT.md — Pi Agent 工作规范

> **本文是所有进入 MarketSense 的 Pi Agent 的第一入口。**
> 开始任何工作前，你必须完整阅读本文，并查看 [PROJECT_STATUS.md](PROJECT_STATUS.md) 确认当前状态。

---

## 1. MarketSense 是什么？

MarketSense 是一个长期演进的研究型工程项目，目标是研究如何将市场数据转化为市场状态表示，并训练输出概率形式的预测，最终通过可复现的评估体系验证。（详见 [README.md](README.md)；最终架构**尚未确定**。）

## 2. 当前项目阶段

以 [PROJECT_STATUS.md](PROJECT_STATUS.md) 为唯一事实来源（Single Source of Truth）。Agent 不得基于过期的记忆或假设开展工作；若你发现状态文件与现实冲突，先更新状态文件。

## 3. Agent 启动检查清单

1. 读 [AGENT.md](AGENT.md)（本文）
2. 读 [PROJECT_STATUS.md](PROJECT_STATUS.md) — 当前 Phase / Next Actions / Open Questions
3. 读 [docs/PROJECT_CONSTITUTION.md](docs/PROJECT_CONSTITUTION.md) — 项目宪法
4. 若任务涉及核心模块：读对应的 [specs/](specs/) Spec
5. 从 [tasks/](tasks/) 认领任务，或确认你的任务指令来源
6. **不要**主动通读 `reference/`（见第 7 节）

## 4. SDD 工作方式（Spec-Driven Development）

核心原则：**没有 Spec 的核心开发任务，原则上不能直接开始。**

```text
Research → Problem Definition → Specification → Task
        → Implementation → Test → Review → Integration → Update Status
```

完整流程见 [specs/README.md](specs/README.md)。实现中发现 Spec 有问题时，走 Proposal → Decision → 更新 Spec → 修改代码的路径，**不得**静默改代码后遗忘设计原因。

## 5. 目录结构

```text
AGENT.md              # 本文
README.md             # 项目说明
PROJECT_STATUS.md     # 项目状态 Single Source of Truth
docs/                 # 政策文档（宪法、Reference Policy）
specs/                # Spec：architecture/ modules/ interfaces/ decisions/(ADR)
research/             # 研究记录（含失败实验）
experiments/          # 实验记录
tasks/                # 任务管理
reference/            # ⚠️ 外部参考资料（见第 7 节）
<源码目录>             # 尚未建立，由 Architecture 阶段的 Spec 决定
```

## 6. 代码修改规则

- 只修改你的 Ownership 范围内的文件（见第 10 节）；
- 核心模块的修改必须有对应 Spec；非核心修改也应在 Task 中说明目的；
- 不修改 `reference/` 中任何内容；
- 不修改其他 Agent 正在负责的模块；如需跨模块改动，通过 Proposal / 与 Owner Agent 协调；
- 每次修改应能通过测试（见第 8 节）并在 commit 中说明动机。

## 7. Reference 使用规则（重要）

> **Agent 不需要在启动时阅读整个 reference。只有任务需要时才进入 reference 进行定向研究。**

- `reference/` 是外部研究知识库（当前含 `minimind/`、`perception/`），**不是 MarketSense 源码**；
- 默认**禁止修改** `reference/` 中任何内容；
- 按需研究流程：Task → 确定需要什么知识 → 检索 reference → 研究相关部分 → 将结论记录到 `research/` → 用于 Spec / Implementation；
- 不得因阅读了某个参考项目，就默认 MarketSense 必须采用其架构。

详见 [docs/REFERENCE_POLICY.md](docs/REFERENCE_POLICY.md)。

## 8. 测试规则

- 核心模块合入前必须有可运行、可重复的测试（测试基础设施由后续 Spec 建立）；
- 测试不得依赖网络、外部私有数据或不可复现的随机状态；
- 实验类工作必须按 [experiments/README.md](experiments/README.md) 记录，保证可复现。

## 9. Git 规则

- 分支策略由 Project Owner / 后续 Spec 确定；当前默认在主分支上以清晰粒度提交；
- Commit message 使用祈使句前缀（`feat:` / `fix:` / `docs:` / `chore:` / `spec:` / `research:`）；
- 禁止提交：训练数据、模型 checkpoint、缓存、日志、临时文件、本地环境文件、密钥（`.gitignore` 已设置防护）；
- 提交前 `git status` 检查，不提交无关文件。

## 10. 任务认领规则与 Agent 边界

- Task 机制见 [tasks/README.md](tasks/README.md)；认领任务时在 Task 文件中登记 Owner，避免重复工作；
- **每个 Agent 必须有明确的：职责、输入、输出、允许修改范围、禁止修改范围**；
- Agent 职能（如 Perception Agent / Model Agent / Training Agent / Evaluation Agent 等）将由后续阶段按需设立，当前不存在任何在任 Agent；
- 并行开发通过 **Contract + Interface + Ownership + Dependency** 解耦；禁止多个 Agent 无边界修改同一核心模块。

## 11. 状态更新规则

完成任何有项目级影响的工作（Spec 定稿、模块合入、阶段推进、遇到阻塞）后，Agent **必须**同步更新 [PROJECT_STATUS.md](PROJECT_STATUS.md) 的对应条目（Completed / In Progress / Blocked / Next Actions / Last Updated）。状态更新是任务的一部分，不是可选项。

## 12. 研究记录规则

- 任何研究活动（文献调查、reference 分析、技术验证、实验）都应记录到 [research/](research/README.md)；
- **失败实验也是项目知识，不得简单删除**；
- 研究结论若影响架构，应转化为 Spec 或 ADR。

## 13. 何时必须停止并请求人工决策

遇到以下情况，Agent 必须停止并将问题记录到 PROJECT_STATUS.md 的 Open Questions，等待 Project Owner / 人工决策：

- 重大架构变化（模型结构范式、核心数据表示、模块边界重划）；
- 核心研究结论与既有 Spec 冲突；
- 训练目标 / prediction target 的语义变化；
- 需要修改 `reference/` 或引入新的外部依赖 / 外部数据源；
- 涉及密钥、凭证、外部付费资源；
- 任务指令与项目宪法冲突；
- 任何"你不确定是否越权"的情形——**宁可停下询问，不要静默拍板**。

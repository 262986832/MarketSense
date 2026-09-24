# MarketSense

> 状态：**Bootstrap 完成 → 待进入 Architecture / Specification**
> 项目状态 Single Source of Truth：[PROJECT_STATUS.md](PROJECT_STATUS.md)

## MarketSense 是什么？

MarketSense 是一个**长期演进的研究型软件工程项目**，研究方向（草案，未最终确定）：

> 将市场数据（Market Data）转化为结构化的市场状态表示（Market State），并以此训练模型输出**概率形式**的预测，再通过严格、可复现的评估体系验证其有效性。

项目目前处于最早期阶段。核心模型结构、数据表示、训练目标、评估指标等**均未确定**，将在 Architecture / Research 阶段以 Spec-Driven Development（SDD）方式逐步确定，并用 TBD / Proposed / Researching / Not Decided 明确标记未知状态。

## 当前阶段

```text
Phase:     M0 Bootstrap（已完成）
Milestone: 建立项目"操作系统"（规范 / SDD / 状态管理 / Agent 协作机制）
Next:      Architecture / Specification（等待 Project Owner 指令）
```

**本阶段禁止开始任何模型、训练、数据生成等核心实现。** 工作方式见 [AGENT.md](AGENT.md)。

## 项目结构

```text
MarketSense/
├── AGENT.md              # Agent 入口与协作规范（所有 Agent 必读）
├── README.md             # 本文件
├── PROJECT_STATUS.md     # 项目状态 Single Source of Truth
├── docs/                 # 项目宪法、Reference Policy 等政策文档
├── specs/                # SDD：架构 / 模块 / 接口 / 决策（ADR）
├── research/             # 研究记录（含失败实验，均为项目知识）
├── experiments/          # 实验记录规则与模板
├── tasks/                # 任务管理
└── reference/            # ⚠️ 外部参考资料，不是 MarketSense 源码
    ├── minimind/         # 参考项目（按需定向研究，勿全量阅读）
    └── perception/       # 参考项目（按需定向研究，勿全量阅读）
```

真正的源码目录（model / training / dataset 等）**尚未建立**，将由 Architecture 阶段的 Spec 决定，Bootstrap 阶段刻意不预建。

## SDD 如何工作？

核心原则：**没有 Spec 的核心开发任务，原则上不能直接开始。**

```text
Research → Problem Definition → Specification → Task
        → Implementation → Test → Review → Integration → Update Status
```

完整工作流、Spec 状态词汇与变更（Proposal）机制见 [specs/README.md](specs/README.md)。

## reference 是什么？

`reference/` 是**外部研究知识库**（Research Reference / External Knowledge），性质上：

- **不是** MarketSense 源码，默认**禁止修改**；
- **不要求** Agent 启动时全量阅读，只在任务需要时定向检索研究；
- 研究结论应沉淀到 `research/`。

详见 [docs/REFERENCE_POLICY.md](docs/REFERENCE_POLICY.md)。

## 如何开始开发？

1. 通读 [AGENT.md](AGENT.md)（Agent 行为规范）与 [docs/PROJECT_CONSTITUTION.md](docs/PROJECT_CONSTITUTION.md)（项目宪法）；
2. 查看 [PROJECT_STATUS.md](PROJECT_STATUS.md) 了解当前阶段与 Next Actions；
3. 从 [tasks/](tasks/) 认领任务；若任务属于核心开发而尚无 Spec，先按 SDD 流程产出 Spec。

# Reference Policy — 参考资料使用规则

> `reference/` 的性质：**Research Reference / External Knowledge（外部研究知识库）**，不是 MarketSense 源码。
> 本政策长期有效，修改需经 Project Owner 决策。

---

## 1. 当前 reference 清单

```text
reference/
├── minimind/      # 外部参考项目（含独立 .git；一级子目录：dataset/ model/ trainer/ scripts/ 等）
└── perception/    # 外部参考项目（一级子目录：config/ data/ docs/ scripts/ src/ tests/ 等）
```

> 以上仅为目录级确认（Bootstrap 阶段未深入阅读其内容）。未来结构可能演变为：
> `papers/`、`libraries/`、`algorithms/`、`experiments/`、`other/` 等，由项目演进决定。

## 2. 核心规则

### 2.1 Reference 不是源码

- `reference/` 中的项目**不自动成为 MarketSense 源码**；
- 不得将其代码复制进核心目录（如需引用思想/片段，必须以研究记录 + 自行实现的方式进入本项目，并遵守上游许可证——许可证问题不明时上报）；
- 阅读了某个参考项目，**不意味着** MarketSense 必须采用其架构。

### 2.2 Reference 不要求全量阅读

- Agent **不**为了启动任务而扫描整个 `reference/`；
- Bootstrap / 日常开发均不要求"读完 minimind / perception"。

### 2.3 按需研究

只有当具体任务需要时：

```text
Task
 ↓
确定需要什么知识
 ↓
检索 reference（定向、最小范围）
 ↓
研究相关部分
 ↓
记录研究结果到 research/
 ↓
用于 Spec / Implementation
```

示例：
- Model Agent 研究 Transformer → 定向进入 `reference/minimind/` 阅读相关模块 → 形成 Research / Design → 实现 MarketSense Model；
- Perception/Data Agent 研究现有市场感知 → 定向进入 `reference/perception/` 研究其输入/输出/数据结构 → 形成 Contract；
- Evaluation Agent 研究 baseline → 在 reference 中查找相关算法/论文 → 建立 baseline。

### 2.4 不直接修改

- **默认禁止修改 `reference/` 中的任何内容**；
- 若未来确有必要（如修复参考实现以复现实验）：
  - 必须经过明确的 Project Owner 决策或单独立项任务；
  - 禁止在 reference 内部原地改（污染上游），优先以"外部补丁 / 派生副本"方式处理。

## 3. 版本管理

- `reference/` 已加入 `.gitignore`，**不纳入 MarketSense 仓库版本管理**：
  - `minimind/` 含独立 `.git`，直接 `git add` 会形成嵌入仓库（gitlink），语义混乱；
  - reference 属外部资产，由工作区管理者按需放置/更新；
- 引用某个参考结论时，在 `research/` 记录中注明：参考项目、路径、commit/版本（可获得时）与查看日期，保证可追溯。

## 4. 违规处理

Agent 若发现自身或他方违反本政策（如已修改 reference、已复制参考代码入库），应立即停止、在 PROJECT_STATUS.md 的 Blocked/Open Questions 中登记，并上报 Project Owner。

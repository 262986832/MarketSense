# research/ — 研究记录

> Research First / Evidence over Assumption（项目宪法 §1、§2）：任何研究活动的过程与结论都沉淀在这里，使项目知识**可追溯**。

## 记录范围

- 研究问题（Research Question）与假设
- 文献研究、技术调查
- `reference/` 中参考项目的定向分析（含：项目、路径、commit/版本、查看日期）
- 技术验证（小规模 spike / prototype）
- **实验与失败实验**（失败实验也是项目知识，**不得简单删除**）
- 研究结论（若影响架构 → 转化为 Spec / ADR，并在此注明链接）

## 文件组织

- 命名：`YYYY-MM-DD-<topic>.md`（可按主题聚合为目录，如 `2026-10-01-tokenizer-survey/`）
- 一份记录一个主题，避免"上帝文档"

## 记录模板

```markdown
# <主题>

- 日期 / 记录人：
- 关联 Task / Spec / ADR：
- 状态：Ongoing | Concluded | Failed | Superseded

## 研究问题
## 假设
## 方法 / 调查范围
## 证据与发现（引用来源：论文 / reference 路径 / 实验数据）
## 结论（区分：已验证 / 仍是假设 / 已证伪）
## 对项目的影响（是否产生 Spec / ADR / 新 Open Question）
```

## 规则

1. 结论必须区分"已验证 / 假设 / 已证伪"，不得把假设写成事实；
2. 引用 `reference/` 结论时注明来源路径与版本（见 docs/REFERENCE_POLICY.md §3）；
3. 研究记录只增不删；推翻旧结论用新记录 + 状态标记（Superseded / Failed）。

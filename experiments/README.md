# experiments/ — 实验规则

> 实验必须**尽可能可重复**（项目宪法 §4）。本目录存放实验记录文件；实验产物（数据、checkpoint、运行日志）不入库（.gitignore 已防护）。

## 状态

当前为空 —— 尚无任何实验。实验由后续 Research / Architecture 阶段按需创建。

## 规则

1. 每个实验一份记录文件：`EXP-<编号>-<short-name>.md`（如 `EXP-0001-baseline-perception.md`）；
2. 记录至少包含下方模板全部字段；缺失字段写 `N/A` 并说明原因，**不得留空**；
3. 实验失败照常记录（Hypothesis / Result / Interpretation 照写，Conclusion 写证伪内容）——**失败实验也是项目知识**；
4. 随机性必须受控：记录随机种子、环境（OS / Python / 关键库版本 / 硬件）；
5. 指标必须附带计算口径，避免"同名不同义"。

## 实验记录模板

```markdown
# EXP-XXXX: <实验名称>

- **Experiment ID**: EXP-XXXX
- **日期 / 执行人**：
- **状态**: Planned | Running | Completed | Failed | Superseded
- **关联**: （Task / Spec / research/ 记录）

## Hypothesis
（本实验检验什么假设？预期结果是什么？）

## Dataset
（数据来源、版本、划分方式、规模；数据本身不入库，记录获取方式）

## Configuration
（配置、超参数、随机种子、环境信息：OS / Python / 库版本 / 硬件）

## Method
（方法与流程，达到他人可复现的粒度）

## Baseline
（对照基线是什么？为什么选它？）

## Metrics
（指标定义与计算口径）

## Result
（原始结果：表格 / 图表引用 / 数值）

## Interpretation
（结果如何解读？与假设一致吗？）

## Conclusion
（结论：假设被验证 / 证伪 / 不确定；对项目的含义）

## Reproducibility Information
（复现步骤、脚本位置、种子、数据获取方式；已知不可复现之处及原因）
```

# AGENTS.md — MarketSense Agent 工作规范

> 本文件面向所有参与 MarketSense 的 Agent。开始任何工作前请先读完本文件。
> 项目入口见 [`README.md`](README.md)。
>
> 原则：**少而必要**。这里只写会实际影响判断的规则，不做流程表演。
> 若本文件与仓库实际代码冲突，**以实际代码为准，并报告冲突**。

---

## 1. MarketSense 是什么

MarketSense 是一个**研究型**项目，研究问题：

> 能否让模型学习市场行情的结构、状态，以及状态随时间的演化，
> 并输出具有统计意义的**概率**信息？

核心目标是"**事实与判断分离**"：

```text
确定性代码  →  客观事实（可计算、可复现）
模型        →  概率与状态判断
```

不是 `K 线 → 涨/跌预测`。

当前**决策模型候选**是 `NanoJev/`（第三方 0.6B 并行决策模型，见 §3）——
这是**当前研究假设**，不是最终承诺。

---

## 2. 当前阶段：为 NanoJev 准备训练数据

- **已实现的基础**（用法见 `README.md` §7）：
  1. `dataset/` 数据准备子应用：天勤 K 线取数 → 标准化落盘 → 转折点提取
     —— 可生成 **K 线数据、转折点数据**；
  2. `scripts/plot_price_line.py`：转折点 **价格折线数据**（PNG）；
  3. `dataset episode-generate`：**首轮 episode 训练数据流水线**（已实现，并已在真实片段上端到端运行，见下方执行记录）
     —— 输入用户指定的震荡片段清单 + 1 分钟 K 线及日线 + 品种 `tick_size` 配置，
     确定性输出对齐 NanoJev 训练契约的按 split JSONL + 审计文件。最小用法：

     ```bash
     # 前置：①1m 与 1d K 线已落盘（dataset fetch --period 1m / 1d；日线供 board_state 行）
     #      ②symbols.local.yaml 填 tick_size
     #      ③片段清单写入 data/segments/（模板 dataset/config/segments.example.jsonl）
     python -m dataset episode-generate --segments data/segments/my_segments.jsonl
     # 契约硬门（只读执行第三方脚本；train/dev/test 必须非空）
     python3 NanoJev/scripts/train_pipeline_decisions.py --validate-only --input data/nanojev_dataset/<run_id>
     ```

     完整参数、配置项与输出布局见 `README.md` §7.2 小节 7。
  4. `dataset board-state`：**盘面状态读取**（2026-10-01 新增，研究 building block，
     非最终模型输入格式）——离线读取已落盘 1m + 1d K 线，按交易日窗口（夜盘前一历日
     21:00 起 → 当日 14:59，夜盘归属其交易所交易日）逐根维护：固定状态
     前日高/低/收 + 今日开盘，动态状态今日最高/最低（已处理 K 线累计极值），
     全部以今日开盘价为基准的相对价（今日开盘 = 1，6 位小数）。最小用法：

     ```bash
     # 前置：1m 与 1d K 线均已落盘（dataset fetch --period 1m / 1d）
     python -m dataset board-state --symbol DCE.v2701 --start 2026-09-01 --end 2026-09-30
     ```

     语义与边界见 `README.md` §7.2 小节 8；已在真实数据上运行与交叉核对（2026-10-01）。
- **上一轮（research-task: nanojev-training-data）已冻结的口径**：
  每分钟一个决策；成交与止损锚定“刚收盘那根 K 线”（决策 K 线）高低点 ± 1 tick；
  固定 1 手、无动态仓位、无机械止盈止损；开仓/平仓/反手标签由**盈亏比规则真值**给出
  （严格大于阈值、多空取更优侧；反转 K 线两条件；反手 = 同根两个决策）；
  模型可见价格一律用**比值**（分母 = 片段首根开盘价，6 位小数），绝对 OHLC 不得进模型输入。
  2026-10-02 用户拍板方案 A：状态模板升 v2（`render_state` 新增 board_state 行，
  prev 日线高/低/收与今日高/低随状态文本进训练记录；prev 真值源 = 1d 日线文件，
  分母 = 片段首根开盘；缺上一交易日日线的片段跳过并告警）；同日拍板候选文案精简：
  只留动作语义（买入开仓/卖出开仓/继续空仓/平仓/继续持有/反手），
  成交价位由执行程序与滑点决定，不进模型输入（`QUESTION_SCHEMA = marketsense.episode_question.v1`）；
  同日拍板状态模板升 v3：行间换行符前后加空格（`" \n "` 连接，转义后的 JSON 文本更易读）。
  2026-10-02 用户对话确认（协调者转录）+ 协调者拍板（DESIGN 面询）状态模板升 v4：
  行重排为宏观→微观六部分（账户/联动/日线/日内(行首含 `bar=`)/现价/盘口，`盘口: na` 占位），
  持仓值中文化（空仓/持多/持空），数值语义与 v3 逐项等价
  （`STATE_SCHEMA = marketsense.episode_state.v4`；完整语义见
  `artifacts/state-template-four-parts/02-design/tech-design.md`）。
  同日拍板状态模板升 v5（daily-extreme-state 任务）：「日线」行新增
  `trend_up/trend_up_len/trend_dn/trend_dn_len` 4 键（最近可确认 up/down 折点的段内实际
  极值价 ÷ 片段首根开盘价 + 对应段长；数据源 = 日线折点 CSV `data/turning_points/{symbol}_1d.csv`；
  口径 = 折点确认根日期严格早于决策交易日、逐决策点各自取可用最近折点，决策交易日归属与
  board_state 夜盘规则同口径；缺折点片段跳过并审计 + stderr 告警；`episode-generate` 新增
  `--daily-turning-points-dir` 参数，默认 `<data_dir>/../turning_points`；
  `STATE_SCHEMA = marketsense.episode_state.v5`；完整语义见
  `artifacts/daily-extreme-state/02-design/tech-design.md`）。
  完整语义与实现阶段冻结项见 `artifacts/nanojev-training-data/02-design/tech-design.md`。
- **首轮真实数据（2026-10-01 生成；2026-10-02 v2~v5 状态模板 + 候选文案精简，均通过契约校验）**：
  DCE.v2701（PVC）2026-09 全月 21 个交易日片段（`data/segments/sep2026.jsonl`：
  train 9-1~9-18 / dev 9-21~9-24 / test 9-28~9-30，按时间顺序切分，夜盘归属其交易所交易日）→
  当前产物 `data/nanojev_dataset/run-7cbd6516d46f/`（模板 v5 + 精简候选文案，
  **train 1392 / dev 886 / test 655**，全部非空；state 文本为 v5 六部分中文标签结构
  （账户/联动/日线/日内/现价/盘口），「日线」行含 trend 四键（数据源 = 日线折点 CSV，
  确认根日期 < 决策交易日）；9 个 train 片段（9-1~9-11）因缺 up 折点跳过并告警；
  board_state 值已与 board-state CSV 全量交叉核对一致，trend 四值与日线折点 CSV
  独立重算一致；当前 run 双跑 sha256 一致）；
  历史 `run-3db1bf63afc2/`（模板 v4，train 4188）、`run-36b037252a62/`（模板 v3）、
  `run-b25cfd1ff370/`（v2 状态 + 精简文案）、
  `run-13aff088b982/`（v2 状态、候选文案含价位括号）与首轮 `run-c1cb097177a3/`（模板 v1）保留。
  六者 NanoJev 原生 `--validate-only` 契约硬门均通过。`tick_size = 5` 为**用户确认值**
  （公开资料记载最小变动价位 1 元/吨，按用户确认执行，见 `dataset/config/symbols.local.yaml`）。
  Mac Intel 16G（无 CUDA）已完成数据通路三级 CPU 冒烟（`--self-check` / tokenize / 前向，
  全部通过）；**训练与推理入口硬性要求 CUDA，本机不可训练**（详见 `README.md` §7.4）。
- **下一阶段核心任务**：
  1. 在真实片段上端到端生成并校验（**已完成，2026-10-01；2026-10-02 v2/v3 + 候选文案精简**，
     见上方「首轮真实数据」；新片段清单仍按 `dataset/config/segments.example.jsonl` 格式提供）；
  2. 价格折线/episode 数据之外，**后续还要补充其它材料与数据**（具体形式**未决**，
     进入具体任务前先讨论确认，见 §8）；
  3. 训练与评估、趋势行情的 OOD 安全评估（未设计）。

> 已知未修复缺陷：空 split 静默通过（三个 `{split}.jsonl` 全 0 字节仍退出 0）——
> 生成后必须自行确认 `train`/`dev`/`test` 非空，否则 NanoJev trainer 会拒绝
> （2026-10-01 九月运行已自查非空：4188/886/655；该提醒仍适用于任何新清单）。
> 详见 `artifacts/nanojev-training-data/04-test/test-report.md`。

---

## 3. `NanoJev/` 的性质

- `NanoJev/` 是**第三方决策模型项目**的克隆（0.6B 并行决策模型：
  Qwen3-0.6B 主干 + 决策头；输入 state / question / candidates，
  直接输出概率分布，零输出 token 解码），**含独立 `.git`**，
  **不属于本仓库源码**，不纳入 MarketSense 的版本管理。
- **不要修改** `NanoJev/` 内部的任何代码、文档、配置；不要提交其内容。
- 它面向游戏任务（Maze / Snake / ViZDoom）。**把它用于行情场景是研究假设**；
  从它身上学到的东西 ≠ MarketSense 的设计决定。
- 其输入契约（state/question/candidates → Choice/Boolean/Score 概率）
  以 `NanoJev/README.md` 与 `NanoJev/docs/` 为准。
- 其训练与推理入口**硬性要求 CUDA**（`train_pipeline_decisions.py` 约 455 行、
  `predict_toy_decisions.py` 推理入口）。无 CUDA 机器（如本机 Mac Intel）可跑
  `--validate-only`、`--self-check` 与 CPU 冒烟验证数据通路，不可真实训练
  （详见 `README.md` §7.4）。
- 写涉及 tokenizer / 模型加载的脚本前，先读 `README.md` §7.4 的两个已记录坑
  （`tokenizer.vocab_size` 不含 added special tokens；transformers 4.51/5.x 的
  `torch_dtype=`/`dtype=` API 差异）与冒烟环境、模型缓存位置。

---

## 4. 不要把研究假设当成事实

当前路线 `K线 → Perception → 客观事实 → 市场状态 → 状态序列 → 模型(NanoJev) → 概率`
是**研究假设**，不是不可改变的架构承诺。

未来完全可能：perception 需要修改、turning points 不够好、
原始 K 线仍需作为输入、描述需要新表达、NanoJev 不合适、概率目标需要重定义。

因此：

- 写文档/注释时，明确区分 **已实现 / 研究假设 / 未决**；
- 不要把自己的推断写成既成事实；
- 不要把提示词或设想中的描述当作已验证的结论。

---

## 5. 先理解上下文，再动代码

- 开始工作前，先读相关**代码 + 文档 + 测试**；不要在不理解的情况下重写已有模块。
- 遇到文档之间、文档与代码之间的矛盾：**报告，不要自行选一个**。

---

## 6. 结论必须有依据；不得美化验证

- 只报告**实际执行**的命令与**实际观察**到的结果。
- 严格区分 **passed / failed / blocked / not-run**；没有证据不得宣布完成。
- **禁止为了让结果好看而修改验证方式、放宽断言、删除失败用例或挑选性报告**；
  失败与异常是研究信息，应如实保留并记录。
- 引用外部结论时注明来源（文件路径、必要时含 commit/版本与查看日期）。

---

## 7. 保持确定性、无未来数据泄漏（硬约束）

训练数据生成同样适用，且是最容易被无意破坏的地方：

- **无未来数据泄漏**：`State(T)` 只能使用 `data <= T` 的数据；未来数据只用于生成标签，
  且特征窗口与标签窗口**严禁交叉**。
- **确定性**：相同输入产生相同输出；避免引入墙钟时间等不确定因素。
- **参数配置化**：不硬编码研究阈值。
- 首轮 episode 流水线（`dataset/market_episode/`）已按上述约束实现：可见状态只由
  ≤ 决策 K 线的数据构造，而标签程序可见未来（仅用于 gold 判定）——这是有意分开的两件事，
  改它时必须同时保持两者（状态不得引入未来信息，标签不得引入不可复现因素）。

---

## 8. 不要过早设计

除非明确进入相应研究任务，否则**不要**：

- 提前冻结训练数据格式、Schema、Label 定义或数据切分方式；
- 开始训练模型、设计最终模型结构或参数；
- 为"项目完整"而提前实现大量代码。

注意区分两个层次：

- **首轮 episode 流水线的口径已由用户明确决策冻结**（见 §2 与 `02-design/tech-design.md`）——
  这是上一次研究任务的已批准产物，**不要擅自改动**（改口径需先改需求/设计并获用户确认）；
- 这**不等于**最终格式/标签/切分已冻结：后续补充材料时仍适用上面的"不要过早设计"约束。

---

## 9. 环境：conda 需要手动启动

本项目使用独立环境：

```text
/opt/anaconda3/envs/marketsense/bin/python     # Python 3.11.13
```

**conda 在本机不会自动生效，必须手动初始化**：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marketsense
```

或直接用绝对路径解释器（推荐在 Agent/脚本中）：

```bash
/opt/anaconda3/envs/marketsense/bin/python -m pytest
```

⚠️ 系统自带的 `/usr/local/bin/python3` 的 `numpy` 已损坏，**不要**用。

---

## 10. 数据与凭证安全

- **禁止**提交训练数据、模型 checkpoint、缓存、日志、密钥
  （`data/`、`*.csv`、`dataset/config/*.local.yaml` 等已被 `.gitignore` 覆盖）。
- `NanoJev/` 自带的模型 / 数据产物同样**不要**提交到本仓库。
- 需要凭证时，从本地环境/环境变量注入。

---

## 11. 何时停下来问人

出现以下情况时**停止并请求人工决策**，不要自行推进：

- 发现文档与代码、或文档之间的实质性冲突；
- 任务范围超出"已确认要做的事"，或需要引入新依赖/新架构；
- 无法验证某个关键结论，而它会影响后续方向。

报告时请给出：**事实、证据（路径/命令/结果）、影响、可选方案与建议**。

---

## 12. 最小必要范围

- 只改目标所需的内容，不做顺带重构，不扩大授权。
- **不自动** `commit` / `push` / 部署 / 执行其他未授权的外部写操作。
- 保护用户已有改动：仓库中存在**有意删除**的内容（旧根级脚手架等），
  **勿擅自恢复或覆盖**。

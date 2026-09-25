# dataset — 数据准备子应用

从仓库根以普通 Python 包方式运行（无打包元数据、无需安装）：

```bash
/opt/anaconda3/envs/marketsense/bin/python -m dataset --help
/opt/anaconda3/envs/marketsense/bin/python -m pytest dataset/tests -q
```

## 运行前提

**必须在仓库根目录运行**（用 `python -m dataset` 调用；`python -m` 会把当前目录加入
`sys.path`，无需安装、无打包元数据）。**conda 在本机需手动初始化**：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marketsense
cd /Users/jiangdianjing/agentspace/MarketSense
```

不要用系统自带的 `/usr/local/bin/python3`（其 `numpy` 已损坏）。

用途：把「天勤 K 线取数 → 标准化落盘（CSV + 来源指纹）→ 转折点提取 → 转折点落盘」
做成自持、可复现、可测试的能力，并提供参数化命令行入口。

- **确定性**：同名输入 → 同一字节输出；sidecar **不含墙钟时间**。
- **无未来数据泄漏**：在线取数经 `clean_serial_klines` 剔除 `id<0`/`datetime<=0`
  填充行与「起点 + 周期 > 当前时间」的**未收盘** K 线；`as_of` 可注入。
- **凭证不入库**：凭证仅在内存中传递，错误消息**永不**回显凭证值。
- 转折点输出是**中间数据**，不是最终模型输入格式（D-07）。

## 模块出处（移植自 `reference/perception/`，只读参考，不 import）

| 本包模块 | 来源（`reference/perception/...`） |
|---|---|
| `dataset/errors.py` | `src/marksense/data/errors.py`（基类更名 `DatasetError`） |
| `dataset/periods.py` | `src/marksense/data/periods.py` |
| `dataset/ohlcv.py` | `src/marksense/data/ohlcv.py` + `src/marksense/data/timeutils.py`（合并） |
| `dataset/validator.py` | `src/marksense/data/validator.py`（去掉 `DataValidator` 包装类） |
| `dataset/storage.py` | `src/marksense/data/storage.py` + `src/marksense/data/loader.py`（合并，固定 CSV） |
| `dataset/provider.py` | `src/marksense/data/provider.py`（不移植 `subscribe`/实时订阅） |
| `dataset/config.py` | `src/marksense/data/config.py`（改为 `dataset` + `tianqin` 两段 + 环境变量覆盖） |
| `dataset/turning_points.py` | `scripts/turning_points.py`（检测算法与参考实现等价；点信息锚定与 kind 命名按 2026-09-25 用户决策偏离参考实现；新增落盘/读取） |
| `dataset/tests/conftest.py` | `tests/data/conftest.py`（`FakeTqApi`/`build_raw_klines`/`build_serial_klines`） |
| `dataset/tests/test_turning_points.py` | `tests/test_turning_points.py`（8 条向量作为固定期望值） |
| `dataset/tests/test_parity_turning_points.py` | 新增：与参考脚本直接对比（参考缺失时 skip） |

参考目录 `reference/` 已被 `.gitignore` 排除，本包**不 import** 它、不复制其中的明文凭证。

## CLI 用法

```text
python -m dataset fetch          --symbol S [--symbol S2 ...] --period P (--bars N | --start ISO --end ISO) [--output-dir DIR] [--config FILE]
python -m dataset turning-points --symbol S [--symbol S2 ...] --period P [--initial-direction auto|up|down] [--data-dir DIR] [--output-dir DIR] [--config FILE]
python -m dataset prepare        --symbol S [--symbol S2 ...] --period P (--bars N | --start ISO --end ISO) [--initial-direction ...] [--output-dir DIR] [--config FILE]
```

- `--period` 必填，仅支持 `1m, 5m, 15m, 1h, 1d`；非法值 stderr 输出支持列表并非 0 退出。
- `--bars` 与 `--start/--end` **互斥**，且必须给其一；`--bars` 取值范围 `1..8964`。
  实现上会向天勤多取 1 根以凑满 N 根**已收盘** K 线；取上限 `8964` 时无法多取，
  末根未收盘被剔除后实际至多返回 `8963` 根 —— 此时 CLI 会在 stderr 输出告警（不静默）。
  请求根数未被满足时一律告警，并**区分归因**（已达 `--bars` 上限 vs 数据源提供的已收盘
  K 线不足）；需要更早数据请改用 `--start/--end`。
- `--symbol` 可重复（一次命令处理多个品种）。
- 退出码：`0` 成功；`1` 运行期失败（配置/凭证/取数/校验/读取；含 `output_dir` 不可创建）；
  `2` 用法错误。

示例：

```bash
# 最近 3 根已收盘 1 分钟 K 线 → data/ohlcv/DCE.v2701_1m.csv + .json
python -m dataset fetch --symbol DCE.v2701 --period 1m --bars 3

# 离线：读取已落盘 K 线 → data/turning_points/DCE.v2701_1m.csv + .json（不联网）
python -m dataset turning-points --symbol DCE.v2701 --period 1m
python -m dataset turning-points --symbol DCE.v2701 --period 1m --initial-direction up

# 一步：fetch + 转折点
python -m dataset prepare --symbol DCE.v2701 --period 1d --bars 200
```

## 数据契约

**K 线 CSV**（`data/ohlcv/{symbol}_{period}.csv`，列顺序固定）：

```text
timestamp,open,high,low,close,volume,open_oi,close_oi
```

`timestamp` 为 ISO8601 带 `+08:00`、唯一、严格升序（`Asia/Shanghai`）；OHLC `float64`；
`volume int64`；持仓量为固定列：`open_oi`/`close_oi` 分别是天勤该根 K 线**起始时刻** / **结束时刻**的持仓量
（`int64`，两条取数路径均返回，无需配置开关）。
同名 `.json` sidecar 记录：

```text
provider, symbol, period, output_format, start_dt, end_dt, row_count,
first_timestamp, last_timestamp, file_sha256, source_data_version
```

`source_data_version = tianqin|{symbol}|{period}|{start}|{end}|rows=N|{first}|{last}|sha256={16hex}`，
**不含墙钟时间**。`--bars` 模式下 `start_dt/end_dt` 取实际窗口首末时间。

**转折点 CSV**（`data/turning_points/{symbol}_{period}.csv`）：

```text
point_index,kind,timestamp,price,bar_index,volume,oi,dt_minutes,price_ratio,volume_ratio,oi_ratio
```

`kind ∈ {start, up, down, close}`（2026-09-25 起：极值类 kind 由 `high`/`low` 更名为
`up`/`down`）；`bar_index` 为窗口内 0 基下标（可回溯关联 K 线 CSV）。`up`/`down` 点的
`timestamp/price/volume/oi` 取**极值 K 线的前一根**（锚点 `bar_index = 极值 bar − 1`；
`price` = 前一根 high/low）；极值落在 bar 0 时整点留在 bar 0，`price` 取 bar 0 自身
极值价。每个点携带其 `bar_index` 所指（锚点）那根 K 线的：`volume`（该 K 线成交量合计）、
`oi`（该 K 线**结束时刻**持仓量，天勤 `close_oi` 口径；`open_oi` 口径可由 `bar_index`
关联 K 线 CSV 补算）。
后四列为相邻点**相对值**，由点序列确定性派生：`dt_minutes` = 当前点与前一点
`timestamp` 之差（单位分钟）；`price_ratio`/`volume_ratio`/`oi_ratio` = 当前点值 /
前一点值（比值，减 1 即变化幅度）。首点无前一点 → 四列均为空单元格；前一点值为 0
（真实存在无成交分钟）时比值无定义，同样留空，不产生 `inf`。
同名 sidecar 记录
`symbol/period/row_count/window_first_timestamp/window_last_timestamp/`
`initial_direction_requested/initial_direction_resolved/input_source_data_version/file_sha256`。

## 配置与凭证安全

配置模板：`dataset/config/tianqin.example.yaml`（入库，凭证留空）。本地真实凭证写入
`dataset/config/tianqin.local.yaml`（已被 `.gitignore` 排除，禁止提交）：

```bash
cp dataset/config/tianqin.example.yaml dataset/config/tianqin.local.yaml
```

解析优先级（高 → 低）：**CLI 参数 > 环境变量 > 配置文件 > 内置默认值**。
环境变量：

```text
MARKETSENSE_TQ_ACCOUNT     # 凭证 account 覆盖
MARKETSENSE_TQ_PASSWORD    # 凭证 password 覆盖
MARKETSENSE_DATASET_CONFIG # 配置文件路径（未显式 --config 时使用）
MARKETSENSE_DATA_DIR       # output_dir 覆盖
```

- 凭证缺失时给出明确错误（说明缺哪个字段、可用的配置方式），**不打印凭证值**、不打印配置对象。
- YAML 解析失败时的错误消息**只含固定文案 + 文件路径 + 行列号**（如
  `配置文件 YAML 解析失败: <path>（YAML 文档组装失败；第 5 行，第 13 列）`）：
  不拼接 PyYAML 的 `str(exc)`/`problem`/`context`，因为其中可能回显出错行原文、
  未定义别名/锚点名或标签名（均可能来自凭证行）；`ReaderError` 由字符偏移换算行列号。
- 兜底脱敏（防凭证值被误填到其它字段）不会改写白名单文案（支持周期/格式列表）。
- 不使用 `reference/perception/config/market.yaml`（含明文凭证）。
- 默认 `output_dir: data`，已被根 `.gitignore` 忽略（`data/`、`*.csv` 双保险）。

## Python 接口（供导入复用）

```python
from dataset import (
    load_dataset_config, DatasetConfig,
    resolve_duration_seconds, supported_periods_text,
    standardize_klines, clean_serial_klines,
    validate_ohlcv, ValidationReport,
    save_ohlcv, load_ohlcv, LoadedOHLCV,
    TianQinProvider, DataProvider,
    TurningPoint, find_turning_points, resolve_initial_direction,
    save_turning_points, load_turning_points, WindowMeta,
    RelativeMetrics, relative_metrics,
)
```

## 测试

```bash
/opt/anaconda3/envs/marketsense/bin/python -m pytest dataset/tests -q        # 全量离线测试
/opt/anaconda3/envs/marketsense/bin/python -m pytest dataset/tests -v        # 含 parity 用例
```

- 全部测试**不触网**：天勤通过 `api_factory` 注入 `FakeTqApi` 桩。
- `dataset/tests/test_parity_turning_points.py` 以 `importlib` 加载
  `reference/perception/scripts/turning_points.py` 直接对比输出；参考目录缺失时自动 skip。

## 已知边界（未验证项）

- 历史区间模式（`--start/--end`）依赖 `get_kline_data_series`，需天勤专业版权限；
  本机权限**未验证**（`U-1`）。无权限时应改用 `--bars`。
- 本包不做实时订阅、不做模型/特征/状态/决策，也不定义最终模型输入格式（非目标）。
- K 线契约于 2026-09-24 扩展：新增固定持仓量列（`open_oi`/`close_oi`），转折点 CSV
  新增 `volume/oi/相对值` 列；旧格式落盘文件需重新 `fetch` + `turning-points` 再生成。

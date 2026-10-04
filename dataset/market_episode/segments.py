"""片段清单（manifest）、品种 tick 配置与 episode 参数：加载 + 校验。

本模块只做确定性的「读入 + 校验」：不推进账户、不生成标签、不写产物。

**实现阶段冻结项（用户已确认，可回退；来源
``artifacts/nanojev-training-data/02-design/tech-design.md`` §实现阶段冻结项 1–6）**
在本轮实现中的取值：

1. 片段首根参考价 = **片段首根开盘价**（由 :mod:`dataset.market_episode.replay`
   使用）；
2. 状态序列化模板 = 决策 K 线单根（**无历史窗口**）+ 仓位 + 回撤，见
   :mod:`dataset.market_episode.nanojev_records`；
3. 止损自动离场成交价 = 触发当根决策 K 线对侧极值 ± 1 tick；MFE 取止损被触前各根
   最高/最低价的最大有利波动，同根同时触及止损与新高按保守约定**先算不利侧**
   （见 :mod:`dataset.market_episode.labels`）；
4. 权益/回撤记账 = **比值化记账**：初始权益 1.0（= 1 单位名义），1 手 = 1 单位名义，
   不计合约乘数 / 保证金 / 手续费（1 tick 不利滑点已内嵌于成交模型）；
5. (b) 采样带宽默认 **2 分钟**（``flat_sample_band_minutes``，可配置）；片段首根即
   首个决策 K 线，账户初始空仓、权益峰值 = 初始权益；
6. 反转条件②在前向扫描中的定义：**先到止损价 → 成立**；扫描到片段末既未超当前
   K 线也未触止损 → **同样成立**（持仓按片段末强平结算）。

清单 schema 属本次 DESIGN 冻结范围：字段为最小集，演进仅允许新增可选字段
（当前实现按 v1 严格拒绝未知字段，以免把拼写错误当成新字段静默忽略）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Final, Mapping

import pandas as pd
import yaml

from dataset.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    ENV_CONFIG,
    ENV_DATA_DIR,
)
from dataset.errors import ConfigError, DatasetError, UnknownPeriodError
from dataset.ohlcv import TIMEZONE
from dataset.periods import resolve_duration_seconds
from dataset.storage import load_ohlcv

#: 片段清单每行的显式 schema 标记
SEGMENT_SCHEMA: Final[str] = "marketsense.segment.v1"
#: 品种 tick 配置的显式 schema 标记
SYMBOLS_SCHEMA: Final[str] = "marketsense.symbols.v1"
#: 本次流水线只接受 1 分钟周期（决策频率 = 每分钟一个决策点）
EPISODE_PERIOD: Final[str] = "1m"
#: split 角色（与 NanoJev ``train_pipeline_decisions.SPLITS`` 对齐）
SPLIT_ROLES: Final[tuple[str, ...]] = ("train", "dev", "calibration", "test", "ood")
#: NanoJev 训练硬约束：清单必须覆盖这三个 split
REQUIRED_SPLIT_ROLES: Final[tuple[str, ...]] = ("train", "dev", "test")

#: 清单必需字段（``schema`` 为显式 schema 标记）
_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "schema",
    "segment_id",
    "symbol",
    "period",
    "start_ts",
    "end_ts",
    "split_role",
)
#: 清单可选字段（最小集；演进只允许在此追加）
_OPTIONAL_FIELDS: Final[tuple[str, ...]] = ("notes",)

#: episode 输出目录默认值（相对仓库根）
DEFAULT_EPISODE_OUTPUT_DIR: Final[str] = "data/nanojev_dataset"
#: 本地品种 tick 配置文件名（位于配置文件同目录；已被 .gitignore 覆盖）
DEFAULT_SYMBOLS_FILENAME: Final[str] = "symbols.local.yaml"
#: 默认 K 线目录名（相对 dataset.output_dir）
DEFAULT_OHLCV_SUBDIR: Final[str] = "ohlcv"
#: ``episode`` 配置段允许的键
_EPISODE_KEYS: Final[tuple[str, ...]] = (
    "data_dir",
    "output_dir",
    "symbols",
    "drawdown_threshold",
    "reward_risk_threshold",
    "price_precision",
    "flat_sample_band_minutes",
    "breakthrough_window",
    "breakthrough_period",
)
#: episode 参数在 ``episode`` 段中的键名
_PARAM_KEYS: Final[tuple[str, ...]] = (
    "drawdown_threshold",
    "reward_risk_threshold",
    "price_precision",
    "flat_sample_band_minutes",
    "breakthrough_window",
    "breakthrough_period",
)

#: 联动行突破动量允许的粒度（``1d`` 显式拒绝：片段窗口 < 1 个交易日，
#: 聚合至多 1 桶，相邻对统计无意义；来源 linkage-breakthrough tech-design §方案 3）
BREAKTHROUGH_PERIODS: Final[tuple[str, ...]] = ("1m", "5m", "15m", "1h")


@dataclass(frozen=True)
class Segment:
    """一个用户指定的行情片段 = 一个 episode（起止时间界定，含首末端点）。

    :param segment_id: 片段唯一标识（= episode id = NanoJev ``family_id``）
    :param symbol: 合约代码（如 ``DCE.v2701``）
    :param period: 周期名（本次固定 ``1m``）
    :param start: 片段起始时间（含，Asia/Shanghai）
    :param end: 片段结束时间（含，Asia/Shanghai）
    :param split_role: ``train``/``dev``/``calibration``/``test``/``ood``
    :param notes: 可选备注
    """

    segment_id: str
    symbol: str
    period: str
    start: pd.Timestamp
    end: pd.Timestamp
    split_role: str
    notes: str | None = None

    @property
    def split_key(self) -> tuple[str, str]:
        """跨 split 隔离的检查键：同一 symbol + period 的不同 split 片段时间不重叠。"""
        return (self.symbol, self.period)

    def overlaps(self, other: "Segment") -> bool:
        """两个片段的时间区间是否相交（同 symbol + period 时才可能重叠）。"""
        if self.split_key != other.split_key:
            return False
        return bool(self.start <= other.end and other.start <= self.end)

    def canonical(self) -> dict[str, Any]:
        """确定性可序列化形式（用于输入指纹；不含文件路径等环境相关值）。"""
        return {
            "segment_id": self.segment_id,
            "symbol": self.symbol,
            "period": self.period,
            "start_ts": _isoformat(self.start),
            "end_ts": _isoformat(self.end),
            "split_role": self.split_role,
            "notes": self.notes,
        }


def _isoformat(ts: pd.Timestamp) -> str:
    """时间戳的确定性文本形式（秒精度 ISO8601，含时区偏移）。"""
    return ts.isoformat()


@dataclass(frozen=True)
class EpisodeParams:
    """episode 流水线的研究阈值/口径（全部参数化，不硬编码在算法里）。"""

    #: 账户死亡阈值：回撤比 ≥ 该值即 episode 结束（默认 5%）
    drawdown_threshold: float = 0.05
    #: 开仓/反手盈亏比阈值：**严格大于**该值才成立（默认 3）
    reward_risk_threshold: float = 3.0
    #: 模型可见比值的小数位（冻结：6 位）
    price_precision: int = 6
    #: (b) 采样带宽：机会分钟周边 ± N 个空仓分钟的"不做"样本
    flat_sample_band_minutes: int = 2
    #: 联动行突破动量窗口（最近 N 根已收盘 K 线/桶的相邻对加权，默认 20）
    breakthrough_window: int = 20
    #: 联动行突破动量的 K 线粒度（1m 直接用决策序列；5m/15m/1h 重采样；1d 拒绝）
    breakthrough_period: str = "1m"

    def as_dict(self) -> dict[str, Any]:
        return {
            "drawdown_threshold": self.drawdown_threshold,
            "reward_risk_threshold": self.reward_risk_threshold,
            "price_precision": self.price_precision,
            "flat_sample_band_minutes": self.flat_sample_band_minutes,
            "breakthrough_window": self.breakthrough_window,
            "breakthrough_period": self.breakthrough_period,
        }


@dataclass(frozen=True)
class EpisodeConfig:
    """``episode-generate`` 的运行配置（YAML + 环境变量，复用 dataset 配置模式）。"""

    data_dir: Path
    output_dir: Path
    symbols_path: Path
    params: EpisodeParams
    config_path: Path | None = None


# --------------------------------------------------------------------------- #
# YAML 读取（不回显原文：配置文件可能同时含天勤凭证）
# --------------------------------------------------------------------------- #
def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    """读取 YAML 映射；错误消息**不含**原文片段（配置文件可能含凭证）。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"配置文件读取失败: {path} ({type(exc).__name__}: {exc})") from None
    except UnicodeDecodeError:
        raise ConfigError(f"配置文件不是有效 UTF-8，无法解析: {path}") from None
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError:
        # 不回显 PyYAML 的 message（可能含出错行原文，而该行可能是凭证行）
        raise ConfigError(f"配置文件 YAML 解析失败（不回显原文）: {path}") from None
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ConfigError(f"配置文件必须是 YAML 映射: {path}")
    return payload


def _optional_section(payload: Mapping[str, Any], name: str, path: Path) -> Mapping[str, Any]:
    """取可选配置段；缺省返回空映射，存在但不是映射则报错。"""
    if name not in payload:
        return {}
    section = payload[name]
    if not isinstance(section, Mapping):
        raise ConfigError(f"配置段 {name} 必须是映射: {path}")
    return section


def _nonempty_str(section: Mapping[str, Any], key: str, *, where: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} 的 {key} 必须为非空字符串")
    return value.strip()


def _positive_number(section: Mapping[str, Any], key: str, *, where: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} 的 {key} 必须为数值")
    number = float(value)
    if not number > 0:
        raise ConfigError(f"{where} 的 {key} 必须为正数，实际: {value!r}")
    return number


def _nonnegative_int(section: Mapping[str, Any], key: str, *, where: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{where} 的 {key} 必须为非负整数")
    return value


def load_symbols_config(path: str | Path) -> Mapping[str, float]:
    """加载品种配置（symbol → tick_size）。

    :raises ConfigError: 文件缺失 / 缺 schema 标记 / 非法 tick_size / 未知字段
    """
    symbols_path = Path(path)
    if not symbols_path.is_file():
        raise ConfigError(f"品种配置不存在: {symbols_path}")
    payload = _read_yaml_mapping(symbols_path)
    if payload.get("schema") != SYMBOLS_SCHEMA:
        raise ConfigError(f"品种配置缺少 schema 标记 {SYMBOLS_SCHEMA!r}: {symbols_path}")
    unknown = sorted(set(payload) - {"schema", "symbols"})
    if unknown:
        raise ConfigError(f"品种配置含未知字段 {unknown}: {symbols_path}")
    raw = payload.get("symbols")
    if not isinstance(raw, Mapping) or not raw:
        raise ConfigError(f"品种配置的 symbols 必须为非空映射: {symbols_path}")
    table: dict[str, float] = {}
    for symbol, entry in raw.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise ConfigError(f"品种配置的合约代码必须为非空字符串: {symbols_path}")
        if not isinstance(entry, Mapping):
            raise ConfigError(f"品种配置 {symbol} 必须为映射: {symbols_path}")
        unknown_entry = sorted(set(entry) - {"tick_size"})
        if unknown_entry:
            raise ConfigError(f"品种配置 {symbol} 含未知字段 {unknown_entry}: {symbols_path}")
        tick = entry.get("tick_size")
        if isinstance(tick, bool) or not isinstance(tick, (int, float)) or not float(tick) > 0:
            raise ConfigError(f"品种配置 {symbol}.tick_size 必须为正数: {symbols_path}")
        table[symbol] = float(tick)
    return table


def load_segments(path: str | Path) -> tuple[Segment, ...]:
    """读取 JSONL 片段清单并做逐行字段校验（顺序保留，作为确定性输出顺序）。

    :raises DatasetError: 文件缺失 / 非法 JSON / 字段缺失或非法 / segment_id 重复
    """
    manifest = Path(path)
    if not manifest.is_file():
        raise DatasetError(f"片段清单不存在: {manifest}")
    try:
        text = manifest.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"片段清单读取失败: {manifest} ({type(exc).__name__})") from None

    segments: list[Segment] = []
    seen_ids: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line, object_pairs_hook=_unique_object)
        except ValueError as exc:
            raise DatasetError(f"片段清单 {manifest}:{lineno} JSON 解析失败: {exc}") from None
        segment = _segment_from_record(raw, where=f"{manifest}:{lineno}")
        if segment.segment_id in seen_ids:
            raise DatasetError(
                f"片段清单 {manifest}:{lineno} segment_id 重复: {segment.segment_id}"
            )
        seen_ids.add(segment.segment_id)
        segments.append(segment)

    if not segments:
        raise DatasetError(f"片段清单为空: {manifest}")
    return tuple(segments)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"重复键 {key}")
        payload[key] = value
    return payload


def _segment_from_record(raw: Any, *, where: str) -> Segment:
    if not isinstance(raw, Mapping):
        raise DatasetError(f"{where} 每行必须为 JSON 对象")
    missing = [key for key in _REQUIRED_FIELDS if key not in raw]
    if missing:
        raise DatasetError(f"{where} 缺少必需字段: {missing}")
    unknown = sorted(set(raw) - set(_REQUIRED_FIELDS) - set(_OPTIONAL_FIELDS))
    if unknown:
        raise DatasetError(f"{where} 含未知字段: {unknown}")

    if raw["schema"] != SEGMENT_SCHEMA:
        raise DatasetError(f"{where} schema 标记必须为 {SEGMENT_SCHEMA!r}")
    segment_id = raw["segment_id"]
    symbol = raw["symbol"]
    period = raw["period"]
    split_role = raw["split_role"]
    for name, value in (
        ("segment_id", segment_id),
        ("symbol", symbol),
        ("period", period),
        ("split_role", split_role),
    ):
        if not isinstance(value, str) or not value.strip():
            raise DatasetError(f"{where} {name} 必须为非空字符串")
    if period != EPISODE_PERIOD:
        raise DatasetError(f"{where} period 必须为 {EPISODE_PERIOD!r}（本次流水线口径）")
    if split_role not in SPLIT_ROLES:
        raise DatasetError(f"{where} split_role 必须为 {list(SPLIT_ROLES)} 之一")

    start = _parse_timestamp(raw["start_ts"], field="start_ts", where=where)
    end = _parse_timestamp(raw["end_ts"], field="end_ts", where=where)
    # 端点含：``start == end`` 表示只含 1 根 K 线的片段；仅拒绝 ``start > end``
    if start > end:
        raise DatasetError(f"{where} start_ts 必须不晚于 end_ts")

    notes = raw.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise DatasetError(f"{where} notes 必须为字符串")

    return Segment(
        segment_id=segment_id.strip(),
        symbol=symbol.strip(),
        period=period,
        start=start,
        end=end,
        split_role=split_role,
        notes=notes,
    )


def _parse_timestamp(value: Any, *, field: str, where: str) -> pd.Timestamp:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{where} {field} 必须为非空 ISO8601 字符串")
    try:
        stamp = pd.Timestamp(value.strip())
    except (ValueError, TypeError):
        raise DatasetError(f"{where} {field} 不是合法时间: {value!r}") from None
    if stamp is pd.NaT or pd.isna(stamp):
        raise DatasetError(f"{where} {field} 不是合法时间: {value!r}")
    # 无时区按交易所时区（Asia/Shanghai）解释，与 K 线接收口径一致
    return stamp.tz_localize(TIMEZONE) if stamp.tzinfo is None else stamp.tz_convert(TIMEZONE)


def validate_segments(
    segments: tuple[Segment, ...] | list[Segment],
    *,
    data_dir: str | Path,
    symbols: Mapping[str, float],
) -> None:
    """校验片段清单与已落盘数据的一致性（不修改任何状态，不读未来数据）。

    检查项（DESIGN 冻结）：

    * 清单非空且 symbol 都在品种配置中（tick size 按品种写入配置）；
    * 清单覆盖 train/dev/test（NanoJev 训练硬约束）；
    * 同一 symbol + period 的**不同 split_role** 片段时间不重叠（跨 split 时间隔离）；
    * 每个片段都能在 ``data_dir`` 找到已落盘 CSV，且片段时间段落在数据范围内、
      区间内至少 1 根 K 线。

    :raises DatasetError: 任一检查失败（消息给出可操作的定位信息）
    """
    if not segments:
        raise DatasetError("片段清单为空")

    missing_symbols = sorted({s.symbol for s in segments} - set(symbols))
    if missing_symbols:
        raise DatasetError(
            f"品种配置缺少 tick_size: {missing_symbols}（tick size 必须按品种写入配置）"
        )
    for symbol in missing_symbols:
        if float(symbols.get(symbol, 0.0)) <= 0:  # pragma: no cover - 防御性
            raise DatasetError(f"品种配置 {symbol}.tick_size 必须为正数")

    roles = {s.split_role for s in segments}
    missing_roles = [role for role in REQUIRED_SPLIT_ROLES if role not in roles]
    if missing_roles:
        raise DatasetError(
            f"片段清单必须覆盖 train/dev/test（NanoJev 训练硬约束），缺少: {missing_roles}"
        )

    for left, right in combinations(segments, 2):
        if left.split_role != right.split_role and left.overlaps(right):
            raise DatasetError(
                "同一 symbol+period 的不同 split 片段时间重叠（跨 split 时间隔离要求不重叠）: "
                f"{left.segment_id}({left.split_role}) 与 "
                f"{right.segment_id}({right.split_role})"
            )

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for segment in segments:
        key = segment.split_key
        if key not in frames:
            try:
                frames[key] = load_ohlcv(segment.symbol, segment.period, data_dir=data_dir).df
            except DatasetError as exc:
                raise DatasetError(
                    f"片段 {segment.segment_id} 读取 K 线失败（{segment.symbol}/{segment.period}）: {exc}"
                ) from None
        stamps = frames[key]["timestamp"]
        first, last = stamps.min(), stamps.max()
        if segment.start < first or segment.end > last:
            raise DatasetError(
                f"片段 {segment.segment_id} 时间段 [{_isoformat(segment.start)}, "
                f"{_isoformat(segment.end)}] 超出数据范围 "
                f"[{_isoformat(first)}, {_isoformat(last)}]"
            )
        inside = int(((stamps >= segment.start) & (stamps <= segment.end)).sum())
        if inside < 1:
            raise DatasetError(f"片段 {segment.segment_id} 区间内没有任何 K 线")


def load_episode_params(section: Mapping[str, Any], *, where: str) -> EpisodeParams:
    """从 ``episode`` 配置段构造参数（未知键报错，避免拼写错误被静默忽略）。"""
    defaults = EpisodeParams()
    unknown = sorted(set(section) - set(_EPISODE_KEYS))
    if unknown:
        raise ConfigError(f"{where} 含未知配置项: {unknown}")

    drawdown = defaults.drawdown_threshold
    if "drawdown_threshold" in section:
        drawdown = _positive_number(section, "drawdown_threshold", where=where)
        if not drawdown < 1.0:
            raise ConfigError(f"{where} 的 drawdown_threshold 必须小于 1")
    reward_risk = defaults.reward_risk_threshold
    if "reward_risk_threshold" in section:
        reward_risk = _positive_number(section, "reward_risk_threshold", where=where)
    precision = defaults.price_precision
    if "price_precision" in section:
        precision = _nonnegative_int(section, "price_precision", where=where)
        if precision > 12:
            raise ConfigError(f"{where} 的 price_precision 必须 ≤ 12")
    band = defaults.flat_sample_band_minutes
    if "flat_sample_band_minutes" in section:
        band = _nonnegative_int(section, "flat_sample_band_minutes", where=where)
    window = defaults.breakthrough_window
    if "breakthrough_window" in section:
        window = _nonnegative_int(section, "breakthrough_window", where=where)
        if window < 1:
            raise ConfigError(f"{where} 的 breakthrough_window 必须 ≥ 1")
    period = defaults.breakthrough_period
    if "breakthrough_period" in section:
        period = _nonempty_str(section, "breakthrough_period", where=where)
    allowed = ", ".join(BREAKTHROUGH_PERIODS)
    if period not in BREAKTHROUGH_PERIODS:
        raise ConfigError(
            f"{where} 的 breakthrough_period 必须为 {allowed} 之一，实际: {period!r}"
            f"（经 resolve_duration_seconds 校验；1d/未知周期不支持）"
        )
    try:
        resolve_duration_seconds(period)
    except UnknownPeriodError:
        # 与白名单双保险：映射表未来若调整，这里兜底拒绝未登记周期
        raise ConfigError(
            f"{where} 的 breakthrough_period 必须为 {allowed} 之一，实际: {period!r}"
        ) from None
    return EpisodeParams(
        drawdown_threshold=float(drawdown),
        reward_risk_threshold=float(reward_risk),
        price_precision=int(precision),
        flat_sample_band_minutes=int(band),
        breakthrough_window=int(window),
        breakthrough_period=str(period),
    )


def load_episode_config(
    path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> EpisodeConfig:
    """加载 episode 运行配置（复用 dataset 的 YAML + 环境变量模式）。

    解析优先级（高 → 低）：

    * 本函数参数（来自 CLI ``--output-dir`` 等）；
    * 环境变量 ``MARKETSENSE_DATA_DIR``（dataset 既有语义：产物根目录，
      episode 的 K 线目录默认取其 ``ohlcv/`` 子目录）；
    * 配置文件 ``episode`` 段（``data_dir``/``output_dir``/``symbols`` + 研究阈值）；
    * 内置默认值（``data/ohlcv``、``data/nanojev_dataset``、
      ``dataset/config/symbols.local.yaml``）。

    :raises ConfigError: 配置文件非法 / 品种配置缺失
    """
    explicit = Path(path) if path is not None else None
    if explicit is not None and not explicit.is_file():
        raise ConfigError(f"配置文件不存在: {explicit}")
    if explicit is None:
        env_path = os.environ.get(ENV_CONFIG)
        if env_path:
            candidate = Path(env_path)
            if not candidate.is_file():
                raise ConfigError(f"配置文件不存在（来自 {ENV_CONFIG}）: {candidate}")
            explicit = candidate
        elif DEFAULT_CONFIG_PATH.is_file():
            explicit = DEFAULT_CONFIG_PATH

    payload = _read_yaml_mapping(explicit) if explicit is not None else {}
    episode_section = _optional_section(payload, "episode", explicit or Path("<memory>"))
    dataset_section = _optional_section(payload, "dataset", explicit or Path("<memory>"))
    where = "episode 配置段"

    base_output = dataset_section.get("output_dir", DEFAULT_OUTPUT_DIR)
    if not isinstance(base_output, str) or not base_output.strip():
        raise ConfigError("配置项 dataset.output_dir 必须为非空字符串")
    env_data_dir = os.environ.get(ENV_DATA_DIR)
    if env_data_dir:
        if not env_data_dir.strip():
            raise ConfigError(f"环境变量 {ENV_DATA_DIR} 不能为空")
        base_output = env_data_dir

    resolved_data_dir = data_dir
    if resolved_data_dir is None:
        raw_data_dir = episode_section.get("data_dir")
        if raw_data_dir is not None:
            if not isinstance(raw_data_dir, str) or not raw_data_dir.strip():
                raise ConfigError(f"{where} 的 data_dir 必须为非空字符串")
            resolved_data_dir = raw_data_dir
        else:
            resolved_data_dir = str(Path(base_output) / DEFAULT_OHLCV_SUBDIR)

    resolved_output_dir: str | Path = output_dir
    if resolved_output_dir is None:
        raw_output_dir = episode_section.get("output_dir", DEFAULT_EPISODE_OUTPUT_DIR)
        if not isinstance(raw_output_dir, str) or not raw_output_dir.strip():
            raise ConfigError(f"{where} 的 output_dir 必须为非空字符串")
        resolved_output_dir = raw_output_dir

    raw_symbols = episode_section.get("symbols")
    if raw_symbols is not None:
        if not isinstance(raw_symbols, str) or not raw_symbols.strip():
            raise ConfigError(f"{where} 的 symbols 必须为非空字符串")
        symbols_path = Path(raw_symbols)
    else:
        symbols_path = (
            (explicit.parent if explicit is not None else Path(DEFAULT_CONFIG_PATH.parent))
            / DEFAULT_SYMBOLS_FILENAME
        )
    if not symbols_path.is_file():
        raise ConfigError(
            f"品种配置不存在: {symbols_path}（复制 dataset/config/symbols.example.yaml "
            f"并按品种填写 tick_size）"
        )

    return EpisodeConfig(
        data_dir=Path(resolved_data_dir),
        output_dir=Path(resolved_output_dir),
        symbols_path=symbols_path,
        params=load_episode_params(episode_section, where=where),
        config_path=explicit,
    )


__all__ = [
    "SEGMENT_SCHEMA",
    "SYMBOLS_SCHEMA",
    "EPISODE_PERIOD",
    "SPLIT_ROLES",
    "REQUIRED_SPLIT_ROLES",
    "DEFAULT_EPISODE_OUTPUT_DIR",
    "EpisodeConfig",
    "EpisodeParams",
    "Segment",
    "load_episode_config",
    "load_episode_params",
    "load_segments",
    "load_symbols_config",
    "validate_segments",
]

"""dataset 配置：YAML 文件 + 环境变量覆盖 + 凭证安全。

解析优先级（高 → 低）：CLI 参数（在 ``cli.py`` 中应用） > 环境变量 >
配置文件 > 内置默认值。

环境变量（tqsdk 自身不读账号环境变量，故由本模块实现）：

* ``MARKETSENSE_TQ_ACCOUNT`` / ``MARKETSENSE_TQ_PASSWORD``：凭证覆盖；
* ``MARKETSENSE_DATASET_CONFIG``：配置文件路径（未显式传 ``path`` 时使用）；
* ``MARKETSENSE_DATA_DIR``：``output_dir`` 覆盖。

凭证安全：错误消息只说明「缺少账户/密码」及其来源，**永不**回显凭证值；
不读取任何内置凭证文件。

配置形状移植自早期版本数据层的 config 模块（由单段
``data_source`` 改为 ``dataset`` + ``tianqin`` 两段，并新增环境变量覆盖）。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

import yaml

from dataset.errors import ConfigError, DatasetError
from dataset.periods import resolve_duration_seconds, supported_periods_text
from dataset.turning_points import INITIAL_DIRECTION_MODES

DATASET_SECTION: Final[str] = "dataset"
TIANQIN_SECTION: Final[str] = "tianqin"
SUPPORTED_PROVIDER_TYPES: Final[tuple[str, ...]] = ("tianqin",)
SUPPORTED_OUTPUT_FORMATS: Final[tuple[str, ...]] = ("csv",)
DEFAULT_OUTPUT_DIR: Final[str] = "data"
DEFAULT_PERIOD: Final[str] = "1m"
DEFAULT_INITIAL_DIRECTION: Final[str] = "auto"

#: 解析器异常类名 → **固定**中文标签（白名单）。表外的类名一律用兜底标签，
#: 保证错误消息里不出现任何解析器生成的、可变的文本（见 ``_yaml_error_detail``）。
_YAML_ERROR_LABELS: Final[Mapping[str, str]] = {
    "ScannerError": "扫描 YAML 标记失败",
    "ParserError": "YAML 语法结构解析失败",
    "ComposerError": "YAML 文档组装失败",
    "ConstructorError": "YAML 类型构造失败",
    "ReaderError": "YAML 字符流读取失败",
    "MarkedYAMLError": "YAML 语法错误",
}
#: 白名单之外的解析器异常类名使用的固定标签
_YAML_ERROR_LABEL_FALLBACK: Final[str] = "YAML 解析失败（未识别的解析器错误）"

#: credential 行的值提取（**不依赖 YAML 解析**，仅用于脱敏兜底）
_CREDENTIAL_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[ \t]*(?:account|password)[ \t]*:[ \t]*(?P<value>.*)$",
    re.IGNORECASE | re.MULTILINE,
)
#: 兜底脱敏时保留的片段最小长度（短片段会误伤正常文案）
_MIN_CREDENTIAL_FRAGMENT_LENGTH: Final[int] = 3

ENV_ACCOUNT: Final[str] = "MARKETSENSE_TQ_ACCOUNT"
ENV_PASSWORD: Final[str] = "MARKETSENSE_TQ_PASSWORD"
ENV_CONFIG: Final[str] = "MARKETSENSE_DATASET_CONFIG"
ENV_DATA_DIR: Final[str] = "MARKETSENSE_DATA_DIR"

#: 本地真实凭证文件（已被 .gitignore 排除；不存在时用默认值 + 环境变量）
DEFAULT_CONFIG_PATH: Final[Path] = (
    Path(__file__).resolve().parent / "config" / "tianqin.local.yaml"
)
#: 入库模板（凭证为空）
EXAMPLE_CONFIG_PATH: Final[Path] = (
    Path(__file__).resolve().parent / "config" / "tianqin.example.yaml"
)


@dataclass(frozen=True)
class DatasetConfig:
    """数据准备配置。

    :param provider_type: 数据源类型，当前仅 ``tianqin``
    :param account/password: 凭证（允许为空，连接时强制校验）
    :param output_dir: 数据产物根目录（OHLCV 落 ``ohlcv/``、转折点落 ``turning_points/``）
    :param output_format: 落盘格式，当前仅 ``csv``（D-02）
    :param period: 默认周期（五档之一）
    :param initial_direction: 默认初始方向模式（``auto``/``up``/``down``）
    :param config_path: 实际读取的配置文件路径（无文件时为 ``None``）
    """

    provider_type: str
    account: str
    password: str
    output_dir: Path
    output_format: str
    period: str
    initial_direction: str
    config_path: Path | None = None

    @property
    def missing_credentials(self) -> tuple[str, ...]:
        """缺失的凭证字段名（不返回值本身）。"""
        missing = []
        if not self.account:
            missing.append("account")
        if not self.password:
            missing.append("password")
        return tuple(missing)


def require_credentials(config: DatasetConfig) -> None:
    """凭证缺失时抛明确错误；消息**不含**凭证值。"""
    missing = config.missing_credentials
    if missing:
        raise DatasetError(
            f"缺少天勤凭证（缺 {'/'.join(missing)}）：请在配置文件 {TIANQIN_SECTION} 段设置 "
            f"{TIANQIN_SECTION}.account / {TIANQIN_SECTION}.password（模板见 "
            f"{EXAMPLE_CONFIG_PATH.name}），或通过环境变量 {ENV_ACCOUNT} / {ENV_PASSWORD} 提供"
            "（凭证值不会出现在错误消息中）"
        )


def _resolve_config_path(path: str | Path | None) -> Path | None:
    """确定配置文件路径；显式给出但不存在时报错，未给出且默认文件不存在时返回 None。"""
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise ConfigError(f"配置文件不存在: {candidate}")
        return candidate

    env_path = os.environ.get(ENV_CONFIG)
    if env_path:
        candidate = Path(env_path)
        if not candidate.is_file():
            raise ConfigError(f"配置文件不存在（来自 {ENV_CONFIG}）: {candidate}")
        return candidate

    return DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.is_file() else None


def _credential_fragments(text: str) -> tuple[str, ...]:
    """从配置原文里尽力提取 credential 行的值片段（仅用于兜底脱敏）。

    不依赖 YAML 解析（解析已失败），只按行匹配 ``account``/``password`` 键后取值，
    再切成不含引号/分隔符的 token；同时产出「去掉 YAML 指示符前缀（``*``/``&``/``!``）」
    的形态，使别名/锚点/标签写法下的值也能被兜底脱敏命中。提取不到时返回空元组。
    """
    fragments: list[str] = []
    for match in _CREDENTIAL_LINE_RE.finditer(text):
        for token in re.findall(r"[^\s'\"#:,]+", match.group("value")):
            for candidate in (token, token.lstrip("*&!")):
                if len(candidate) >= _MIN_CREDENTIAL_FRAGMENT_LENGTH:
                    fragments.append(candidate)
    return tuple(dict.fromkeys(fragments))


def _protected_texts() -> tuple[str, ...]:
    """兜底脱敏**不得改写**的白名单文案（本代码生成的固定列表，非用户可控）。

    否则形如 ``password: 15m`` / ``password: csv`` 的巧合会把「支持周期/格式列表」这类
    可操作的错误文案削弱成 ``***``（P2-N1）。
    """
    return (supported_periods_text(), repr(SUPPORTED_OUTPUT_FORMATS))


def _redact(message: str, fragments: tuple[str, ...]) -> str:
    """把消息中**独立出现**的凭证片段替换为 ``***``（兜底，正常路径不应命中）。

    按词边界匹配，避免短凭证值（如 ``pass``）误伤文案中的 ``password`` 等字段名；
    消息里的白名单文案（支持周期/格式列表）先占位保护、脱敏后还原（P2-N1）。
    """
    if not fragments:
        return message
    protected: list[tuple[str, str]] = []
    working = message
    for index, text in enumerate(_protected_texts()):
        if text and text in working:
            placeholder = f"\x00{index}\x00"
            working = working.replace(text, placeholder)
            protected.append((placeholder, text))
    for fragment in fragments:
        working = re.sub(rf"(?<!\w){re.escape(fragment)}(?!\w)", "***", working)
    for placeholder, text in protected:
        working = working.replace(placeholder, text)
    return working


def _env_credential_fragments() -> tuple[str, ...]:
    """环境变量来源的凭证值（仅用于兜底脱敏，值本身不写入任何消息）。"""
    fragments = [
        value
        for name in (ENV_ACCOUNT, ENV_PASSWORD)
        if (value := os.environ.get(name))
        and len(value) >= _MIN_CREDENTIAL_FRAGMENT_LENGTH
    ]
    return tuple(dict.fromkeys(fragments))


def _read_text_quietly(path: Path | None) -> str:
    """尽力读取配置原文（仅用于错误路径的脱敏；失败返回空串）。"""
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _yaml_error_location(exc: yaml.YAMLError, source_text: str) -> tuple[int, int] | None:
    """取解析器给出的**位置**（1 基行列号）；取不到返回 ``None``（不编造）。

    只读 ``problem_mark`` / ``context_mark`` 的行列号，以及 ``ReaderError.position``
    在原文中的偏移；**不读** ``problem`` / ``context`` 文本（它们可能含原文 token）。
    """
    for attribute in ("problem_mark", "context_mark"):
        mark = getattr(exc, attribute, None)
        line = getattr(mark, "line", None)
        column = getattr(mark, "column", None)
        if isinstance(line, int) and isinstance(column, int) and line >= 0 and column >= 0:
            return line + 1, column + 1
    # ``ReaderError``（如含不可打印控制字符）没有 mark，只有 ``position`` 偏移
    position = getattr(exc, "position", None)
    if isinstance(position, int) and 0 <= position <= len(source_text):
        prefix = source_text[:position]
        return prefix.count("\n") + 1, position - (prefix.rfind("\n") + 1) + 1
    return None


def _yaml_error_detail(exc: yaml.YAMLError, source_text: str) -> str:
    """构造**不含任何解析器生成文本**的 YAML 错误摘要。

    PyYAML 的 ``str(exc)``/``problem``/``context`` 都可能回显配置原文 token（出错行片段、
    未定义别名/锚点名、标签名等），出错行恰为凭证行时会把密码/账户明文带进错误消息
    （AC-8、AGENTS.md §12；别名写法 ``password: *<哨兵>`` 即为一例）。因此这里只输出：

    * 白名单化的**固定**中文标签（未知类名走兜底标签）；
    * ``problem_mark`` / ``context_mark`` 的行列号（``ReaderError`` 由 ``position`` 换算），
      位置不可得时省略，不编造。

    任何原文片段都不参与拼接，因此也不需要「先回显再脱敏」。
    """
    label = _YAML_ERROR_LABELS.get(type(exc).__name__, _YAML_ERROR_LABEL_FALLBACK)
    location = _yaml_error_location(exc, source_text)
    if location is None:
        return label
    line, column = location
    return f"{label}；第 {line} 行，第 {column} 列"


def _read_payload(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"配置文件读取失败: {path} ({type(exc).__name__}: {exc})") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"配置文件不是有效 UTF-8，无法解析: {path} ({type(exc).__name__})"
        ) from None

    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        # ``from None``：不暴露 PyYAML 原始异常（其消息含出错行原文，可能含凭证）
        raise ConfigError(
            f"配置文件 YAML 解析失败: {path}（{_yaml_error_detail(exc, text)}）"
        ) from None
    if not isinstance(payload, Mapping):
        raise ConfigError(f"配置文件必须是 YAML 映射: {path}")
    return payload


def _section(payload: Mapping[str, Any], name: str, path: Path) -> Mapping[str, Any]:
    if name not in payload:
        raise ConfigError(f"配置文件缺少 {name} 配置段: {path}")
    section = payload[name]
    if not isinstance(section, Mapping):
        raise ConfigError(f"配置段 {name} 必须是映射: {path}")
    return section


def _optional_str(section: Mapping[str, Any], key: str, default: str, *, path: Path) -> str:
    value = section.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"配置项 {key} 必须为字符串: {path}")
    return value


def _load_from_file(path: Path) -> DatasetConfig:
    payload = _read_payload(path)
    dataset = _section(payload, DATASET_SECTION, path)
    tianqin = _section(payload, TIANQIN_SECTION, path)

    for key in ("account", "password"):
        if key not in tianqin:
            raise ConfigError(f"{TIANQIN_SECTION} 缺少必需键: {key} ({path})")
        if not isinstance(tianqin[key], str):
            raise ConfigError(f"配置项 {TIANQIN_SECTION}.{key} 必须为字符串: {path}")

    return DatasetConfig(
        provider_type=_optional_str(dataset, "type", "tianqin", path=path),
        account=tianqin["account"],
        password=tianqin["password"],
        output_dir=Path(_optional_str(dataset, "output_dir", DEFAULT_OUTPUT_DIR, path=path)),
        output_format=_optional_str(dataset, "output_format", "csv", path=path),
        period=_optional_str(dataset, "period", DEFAULT_PERIOD, path=path),
        initial_direction=_optional_str(
            dataset, "initial_direction", DEFAULT_INITIAL_DIRECTION, path=path
        ),
        config_path=path,
    )


def _defaults() -> DatasetConfig:
    return DatasetConfig(
        provider_type="tianqin",
        account="",
        password="",
        output_dir=Path(DEFAULT_OUTPUT_DIR),
        output_format="csv",
        period=DEFAULT_PERIOD,
        initial_direction=DEFAULT_INITIAL_DIRECTION,
        config_path=None,
    )


def _apply_environment(config: DatasetConfig) -> DatasetConfig:
    """环境变量覆盖配置文件/默认值（凭证来源标注，不回显值）。"""
    updates: dict[str, Any] = {}
    if (account := os.environ.get(ENV_ACCOUNT)) is not None:
        updates["account"] = account
    if (password := os.environ.get(ENV_PASSWORD)) is not None:
        updates["password"] = password
    if (data_dir := os.environ.get(ENV_DATA_DIR)) is not None:
        if not data_dir.strip():
            raise ConfigError(f"环境变量 {ENV_DATA_DIR} 不能为空")
        updates["output_dir"] = Path(data_dir)
    return config if not updates else _replace(config, **updates)


def _replace(config: DatasetConfig, **updates: Any) -> DatasetConfig:
    values = {
        "provider_type": config.provider_type,
        "account": config.account,
        "password": config.password,
        "output_dir": config.output_dir,
        "output_format": config.output_format,
        "period": config.period,
        "initial_direction": config.initial_direction,
        "config_path": config.config_path,
    }
    values.update(updates)
    return DatasetConfig(**values)


def _validate(config: DatasetConfig) -> DatasetConfig:
    if config.provider_type not in SUPPORTED_PROVIDER_TYPES:
        raise ConfigError(
            f"不支持的数据源类型 {config.provider_type!r}，当前仅支持: {SUPPORTED_PROVIDER_TYPES}"
        )
    if config.output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ConfigError(
            f"未知 output_format {config.output_format!r}，支持: {SUPPORTED_OUTPUT_FORMATS}"
        )
    if not str(config.output_dir).strip():
        raise ConfigError("output_dir 必须为非空路径")
    resolve_duration_seconds(config.period)  # 未知周期抛 UnknownPeriodError（含支持列表）
    if config.initial_direction not in INITIAL_DIRECTION_MODES:
        raise ConfigError(
            f"未知 initial_direction {config.initial_direction!r}，"
            f"支持: {INITIAL_DIRECTION_MODES}"
        )
    return config


def load_dataset_config(path: str | Path | None = None) -> DatasetConfig:
    """加载配置：文件 → 环境变量覆盖 → 校验。

    :param path: 显式配置文件路径（CLI ``--config``）；``None`` 时依次尝试
        ``MARKETSENSE_DATASET_CONFIG`` 与 ``dataset/config/tianqin.local.yaml``
    :raises ConfigError: 文件不存在 / 缺段缺键 / 非法取值
    """
    resolved = _resolve_config_path(path)
    try:
        base = _load_from_file(resolved) if resolved is not None else _defaults()
        return _validate(_apply_environment(base))
    except DatasetError as exc:
        # 统一兜底：任何配置错误消息都不得包含凭证值（文件与环境的凭证值均参与脱敏）
        fragments = _credential_fragments(_read_text_quietly(resolved)) + _env_credential_fragments()
        redacted = _redact(str(exc), fragments)
        if redacted == str(exc):
            raise
        raise type(exc)(redacted) from None

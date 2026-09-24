"""配置解析与凭证安全（AC-8）。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from dataset import config as config_module
from dataset.config import (
    ENV_ACCOUNT,
    ENV_CONFIG,
    ENV_DATA_DIR,
    ENV_PASSWORD,
    DatasetConfig,
    load_dataset_config,
    require_credentials,
)
from dataset.errors import ConfigError, DatasetError, UnknownPeriodError
from dataset.tests.conftest import build_config_payload


def _write_config(tmp_path: Path, **overrides) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(build_config_payload(**overrides), allow_unicode=True),
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离外部环境变量，并让默认本地凭证文件不参与测试。"""
    for name in (ENV_ACCOUNT, ENV_PASSWORD, ENV_CONFIG, ENV_DATA_DIR):
        monkeypatch.delenv(name, raising=False)


def test_load_from_file(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        dataset={"output_dir": str(tmp_path / "out"), "period": "5m", "include_oi": True},
        tianqin={"account": "file-user", "password": "file-pass"},
    )

    cfg = load_dataset_config(path)

    assert cfg.provider_type == "tianqin"
    assert cfg.account == "file-user"
    assert cfg.password == "file-pass"
    assert cfg.output_dir == tmp_path / "out"
    assert cfg.output_format == "csv"
    assert cfg.include_oi is True
    assert cfg.period == "5m"
    assert cfg.initial_direction == "auto"
    assert cfg.config_path == path
    assert cfg.missing_credentials == ()


def test_environment_overrides_file_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_config(
        tmp_path,
        dataset={"output_dir": str(tmp_path / "file-out")},
        tianqin={"account": "file-user", "password": "file-pass"},
    )
    monkeypatch.setenv(ENV_ACCOUNT, "env-user")
    monkeypatch.setenv(ENV_PASSWORD, "env-pass")
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "env-out"))

    cfg = load_dataset_config(path)

    assert (cfg.account, cfg.password) == ("env-user", "env-pass")
    assert cfg.output_dir == tmp_path / "env-out"


def test_env_config_path_used_when_no_explicit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_config(tmp_path, tianqin={"account": "env-cfg", "password": "p"})
    monkeypatch.setenv(ENV_CONFIG, str(path))

    cfg = load_dataset_config()

    assert cfg.account == "env-cfg"
    assert cfg.config_path == path


def test_defaults_used_when_no_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "missing.yaml")

    cfg = load_dataset_config()

    assert cfg.provider_type == "tianqin"
    assert cfg.output_dir == Path("data")
    assert cfg.period == "1m"
    assert cfg.initial_direction == "auto"
    assert cfg.config_path is None
    assert set(cfg.missing_credentials) == {"account", "password"}


def test_default_path_picked_up_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_config(tmp_path, tianqin={"account": "local", "password": "local-pass"})
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", path)

    cfg = load_dataset_config()

    assert cfg.account == "local"


def test_missing_explicit_path_reports_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="配置文件不存在"):
        load_dataset_config(tmp_path / "nope.yaml")


def test_missing_env_config_path_reports_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_CONFIG, str(tmp_path / "nope.yaml"))
    with pytest.raises(ConfigError, match=ENV_CONFIG):
        load_dataset_config()


def test_missing_sections_and_keys_reported(tmp_path: Path) -> None:
    blank = tmp_path / "blank.yaml"
    blank.write_text("foo: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="缺少 dataset 配置段"):
        load_dataset_config(blank)

    no_tianqin = tmp_path / "no_tianqin.yaml"
    no_tianqin.write_text("dataset:\n  type: tianqin\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="缺少 tianqin 配置段"):
        load_dataset_config(no_tianqin)

    missing_key = tmp_path / "missing_key.yaml"
    missing_key.write_text(
        "dataset:\n  type: tianqin\ntianqin:\n  account: u\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="tianqin 缺少必需键: password"):
        load_dataset_config(missing_key)


#: 出错行恰为凭证行的畸形 YAML。PyYAML 原始异常（``str(exc)``/``problem``/``context``）
#: 会回显该行原文或其 token（未定义别名/锚点名、标签名），因此这些输入是「错误消息
#: 不得含凭证明文」的回归向量。``{pw}`` 为哨兵密码占位；用例名不含 ``line``/``column``
#: 等词，避免哨兵断言被路径字符串偶然满足。
_HEAD = 'dataset:\n  output_dir: data\ntianqin:\n  account: "acct"\n'
_MALFORMED_CREDENTIAL_YAML: list[tuple[str, str]] = [
    (
        "value_extra_colon",
        'dataset:\n  output_dir: data\ntianqin:\n  account: "acct"\n  password: "{pw}" : oops\n',
    ),
    (
        "unclosed_quote",
        'dataset:\n  output_dir: data\ntianqin:\n  account: "acct"\n  password: "{pw}\n',
    ),
    (
        "tab_before_value",
        'dataset:\n  output_dir: data\ntianqin:\n  account: "acct"\n  password:\t{pw}\n',
    ),
    (
        "account_field_extra_colon",
        'dataset:\n  output_dir: data\ntianqin:\n  account: {pw}: oops\n  password: "x"\n',
    ),
    # P1-2：别名/锚点/标签写法下，解析器的 problem/context 会回显 token 原文
    ("undefined_alias_password", _HEAD + "  password: *{pw}\n"),
    (
        "undefined_alias_account",
        'dataset:\n  output_dir: data\ntianqin:\n  account: *{pw}\n  password: "x"\n',
    ),
    (
        "duplicate_anchor",
        _HEAD + "  password: &{pw} first\n  password: &{pw} second\n",
    ),
    ("undefined_tag_password", _HEAD + "  password: !{pw} value\n"),
    (
        "undefined_tag_account",
        'dataset:\n  output_dir: data\ntianqin:\n  account: !{pw}\n  password: "x"\n',
    ),
    ("tab_indented_field", _HEAD + "\tpassword: {pw}\n"),
    # 更多畸形形态（P1-2 回归扩展：合并键 / 多文档 / flow 括号 / 保留指示符 / NUL）
    ("merge_key_alias", _HEAD + "  password: {pw}\n  <<: *{pw}\n"),
    ("multi_document", _HEAD + '  password: "{pw}"\n---\nfoo: 1\n'),
    ("flow_map_unclosed", _HEAD + "  password: {{{pw}\n"),
    ("reserved_indicator_at", _HEAD + "  password: @{pw}\n"),
    ("nul_byte_password", _HEAD + '  password: "{pw}\x00"\n'),
    # 控制字符：触发 ReaderError（无 problem/mark，只靠 position 定位）
    ("control_char_password", _HEAD + '  password: "{pw}\x01"\n'),
    (
        "control_char_account",
        'dataset:\n  output_dir: data\ntianqin:\n  account: "{pw}\x02"\n  password: "x"\n',
    ),
]


def test_invalid_yaml_and_non_mapping_payload_reported(tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("dataset: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML 解析失败"):
        load_dataset_config(broken)

    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("just-a-string\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="必须是 YAML 映射"):
        load_dataset_config(scalar)


@pytest.mark.parametrize(
    "name, template",
    _MALFORMED_CREDENTIAL_YAML,
    ids=[name for name, _ in _MALFORMED_CREDENTIAL_YAML],
)
def test_yaml_error_never_echoes_credential_values(
    tmp_path: Path, name: str, template: str
) -> None:
    """P1-1/P1-2 回归：凭证行 YAML 语法错误时，错误消息不得回显凭证原文。

    同时做结构性断言：消息去掉路径后，必须**逐字**等于「固定文案 + 行列号」，
    即不携带任何解析器生成的文本（``problem``/``context``/别名/锚点名/标签名/原文行）。
    """
    sentinel = "SENTINEL_PW_2f9a"
    config = tmp_path / f"{name}.yaml"
    config.write_text(template.format(pw=sentinel), encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        load_dataset_config(config)

    message = str(excinfo.value)
    assert "YAML 解析失败" in message
    assert sentinel not in message
    assert str(config) in message  # 仍能定位到文件
    # 仍保留行列号（1 基）；ReaderError 由 position 换算，不编造
    assert re.search(r"第 \d+ 行，第 \d+ 列", message), message
    # 结构性：去掉路径后只允许「固定文案 + 行列号」
    without_path = message.replace(str(config), "")
    assert re.fullmatch(
        r"配置文件 YAML 解析失败: （[^（）]*；第 \d+ 行，第 \d+ 列）", without_path
    ), without_path
    # 不保留 PyYAML 原始异常，也不在 traceback 中回显（其消息含出错行原文）
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True


def test_config_error_paths_never_echo_credential_values(tmp_path: Path) -> None:
    """P1-1 回归：其余配置错误路径同样不回显 account/password 的值。"""
    sentinel = "SENTINEL_PW_2f9a"

    wrong_type = _write_config(tmp_path, tianqin={"account": sentinel, "password": 123})
    with pytest.raises(ConfigError) as excinfo:
        load_dataset_config(wrong_type)
    assert sentinel not in str(excinfo.value)

    missing_key = tmp_path / "missing_key.yaml"
    missing_key.write_text(
        f'dataset:\n  output_dir: data\ntianqin:\n  account: "{sentinel}"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as excinfo:
        load_dataset_config(missing_key)
    assert sentinel not in str(excinfo.value)

    not_a_mapping = tmp_path / "not_a_mapping.yaml"
    not_a_mapping.write_text(f"{sentinel}: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_dataset_config(not_a_mapping)
    assert sentinel not in str(excinfo.value)


def test_any_config_error_message_is_scrubbed_for_credential_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-1 兜底：凭证值即使被误填到其它字段，也不得出现在错误消息中。"""
    sentinel = "SENTINEL_PW_2f9a"

    # (a) 文件来源：period 误填为密码值（period 错误消息会回显该值）
    mistaken = _write_config(
        tmp_path,
        dataset={"period": sentinel},
        tianqin={"account": "u", "password": sentinel},
    )
    with pytest.raises(UnknownPeriodError) as excinfo:
        load_dataset_config(mistaken)
    assert sentinel not in str(excinfo.value)
    assert "1m, 5m, 15m, 1h, 1d" in str(excinfo.value)  # 消息仍可操作

    # (b) 环境变量来源：密码来自 MARKETSENSE_TQ_PASSWORD，同时在 period 出现
    monkeypatch.setenv(ENV_PASSWORD, sentinel)
    env_case = _write_config(
        tmp_path,
        dataset={"period": sentinel},
        tianqin={"account": "u", "password": ""},
    )
    with pytest.raises(UnknownPeriodError) as excinfo:
        load_dataset_config(env_case)
    assert sentinel not in str(excinfo.value)


def test_short_credential_value_keeps_supported_lists_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2-N1 回归：口令恰为 ``15m``/``csv`` 时，白名单文案（支持周期/格式列表）完整可见。

    兜底脱敏按「凭证值 token」匹配，若不做白名单保护，会把 ``1m, 5m, 15m, 1h, 1d``
    或 ``('csv',)`` 里的同形 token 换成 ``***``，削弱 AC-3 要求的可操作错误信息。
    """
    for sentinel_password in ("15m", "csv"):
        wrong_period = _write_config(
            tmp_path / f"file-{sentinel_password}",
            dataset={"period": "7m"},
            tianqin={"account": "u", "password": sentinel_password},
        )
        with pytest.raises(UnknownPeriodError) as excinfo:
            load_dataset_config(wrong_period)
        assert "1m, 5m, 15m, 1h, 1d" in str(excinfo.value)

        wrong_format = _write_config(
            tmp_path / f"fmt-{sentinel_password}",
            dataset={"output_format": "parquet"},
            tianqin={"account": "u", "password": sentinel_password},
        )
        with pytest.raises(ConfigError) as excinfo:
            load_dataset_config(wrong_format)
        assert "('csv',)" in str(excinfo.value)

        # 环境变量来源同样不得削弱白名单文案
        monkeypatch.setenv(ENV_PASSWORD, sentinel_password)
        env_case = _write_config(
            tmp_path / f"env-{sentinel_password}",
            dataset={"period": "7m"},
            tianqin={"account": "u", "password": ""},
        )
        with pytest.raises(UnknownPeriodError) as excinfo:
            load_dataset_config(env_case)
        assert "1m, 5m, 15m, 1h, 1d" in str(excinfo.value)
        monkeypatch.delenv(ENV_PASSWORD, raising=False)


def test_short_credential_value_still_redacted_outside_whitelisted_text(tmp_path: Path) -> None:
    """P2-N1 对照：白名单之外的同名 token 仍按凭证脱敏（保护有范围，不是关闭脱敏）。"""
    wrong_type = _write_config(
        tmp_path,
        dataset={"type": "15m"},
        tianqin={"account": "u", "password": "15m"},
    )

    with pytest.raises(ConfigError) as excinfo:
        load_dataset_config(wrong_type)

    message = str(excinfo.value)
    assert "不支持的数据源类型" in message
    assert "'15m'" not in message
    assert "'***'" in message


def test_redact_uses_word_boundaries() -> None:
    """兜底脱敏按词边界匹配：短凭证值（如 ``pass``）不得误伤 ``password`` 等文案。"""
    assert config_module._redact("配置项 tianqin.password 必须为字符串", ("pass",)) == (
        "配置项 tianqin.password 必须为字符串"
    )
    assert config_module._redact("未知周期 'pass'，支持的周期: 1m", ("pass",)) == (
        "未知周期 '***'，支持的周期: 1m"
    )


def test_non_utf8_config_reports_clean_error(tmp_path: Path) -> None:
    """非 UTF-8 配置：报错不崩溃、不回显文件内容。"""
    config = tmp_path / "latin1.yaml"
    config.write_bytes(b'dataset:\n  output_dir: "\xe9"\n')

    with pytest.raises(ConfigError, match="不是有效 UTF-8"):
        load_dataset_config(config)


def test_invalid_values_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="不支持的数据源类型"):
        load_dataset_config(_write_config(tmp_path, dataset={"type": "other"}))

    with pytest.raises(ConfigError, match="未知 output_format"):
        load_dataset_config(_write_config(tmp_path, dataset={"output_format": "parquet"}))

    with pytest.raises(ConfigError, match="include_oi 必须为布尔值"):
        load_dataset_config(_write_config(tmp_path, dataset={"include_oi": "yes"}))

    with pytest.raises(UnknownPeriodError, match="1m, 5m, 15m, 1h, 1d"):
        load_dataset_config(_write_config(tmp_path, dataset={"period": "7m"}))

    with pytest.raises(ConfigError, match="未知 initial_direction"):
        load_dataset_config(_write_config(tmp_path, dataset={"initial_direction": "sideways"}))

    with pytest.raises(ConfigError, match="output_dir 必须为非空路径"):
        load_dataset_config(_write_config(tmp_path, dataset={"output_dir": "  "}))

    with pytest.raises(ConfigError, match="必须为字符串"):
        load_dataset_config(
            _write_config(tmp_path, tianqin={"account": 123, "password": "p"})
        )


def test_empty_env_data_dir_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_DATA_DIR, "   ")
    with pytest.raises(ConfigError, match=ENV_DATA_DIR):
        load_dataset_config()


def test_missing_credentials_error_never_echoes_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "S3CRET-PASSWORD-VALUE"
    monkeypatch.setenv(ENV_PASSWORD, sentinel)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "missing.yaml")

    cfg = load_dataset_config()
    assert cfg.missing_credentials == ("account",)

    with pytest.raises(DatasetError) as excinfo:
        require_credentials(cfg)

    message = str(excinfo.value)
    assert "缺少天勤凭证（缺 account）" in message
    assert sentinel not in message
    assert cfg.password not in message


def test_require_credentials_passes_when_both_present() -> None:
    cfg = DatasetConfig(
        provider_type="tianqin",
        account="u",
        password="p",
        output_dir=Path("data"),
        output_format="csv",
        include_oi=False,
        period="1m",
        initial_direction="auto",
    )
    require_credentials(cfg)  # 不抛异常


def test_example_config_template_is_valid_and_has_empty_credentials() -> None:
    cfg = load_dataset_config(config_module.EXAMPLE_CONFIG_PATH)

    assert cfg.account == ""
    assert cfg.password == ""
    assert cfg.output_dir == Path("data")
    assert cfg.output_format == "csv"
    assert cfg.period == "1m"
    assert set(cfg.missing_credentials) == {"account", "password"}

"""CLI 入口与离线端到端集成（AC-1、AC-3、AC-7）。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dataset import config as config_module
from dataset.cli import main
from dataset.provider import SERIAL_MAX_DATA_LENGTH
from dataset.storage import load_ohlcv
from dataset.tests.conftest import FakeTqApi, build_config_payload, build_serial_klines
from dataset.turning_points import load_turning_points

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_SERIAL_ROWS = [
    ("2024-01-02 09:00:00", 67950.0, 68020.0, 67900.0, 68000.0, 100, 5000, 5000),
    ("2024-01-02 09:01:00", 68000.0, 68100.0, 67900.0, 68050.0, 120, 5000, 4980),
    ("2024-01-02 09:02:00", 68050.0, 68120.0, 68010.0, 68100.0, 150, 4980, 4900),
    ("2024-01-02 09:03:00", 68100.0, 68200.0, 68090.0, 68150.0, 90, 4900, 4850),
    ("2024-01-02 09:04:00", 68150.0, 68250.0, 68140.0, 68200.0, 80, 4850, 4800),
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离外部环境变量，保证 CLI 行为只由参数与配置决定。"""
    for name in (
        config_module.ENV_ACCOUNT,
        config_module.ENV_PASSWORD,
        config_module.ENV_CONFIG,
        config_module.ENV_DATA_DIR,
    ):
        monkeypatch.delenv(name, raising=False)


def _config_file(tmp_path: Path, **overrides) -> Path:
    path = tmp_path / "tianqin.test.yaml"
    payload = build_config_payload(
        dataset={"output_dir": str(tmp_path / "data")}, **overrides
    )
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


def _fake_api() -> FakeTqApi:
    return FakeTqApi(serial=build_serial_klines(_SERIAL_ROWS, width=8))


def _factory(api: FakeTqApi):
    return lambda _cfg: api


def _csv_path(tmp_path: Path, symbol: str = "DCE.v2701") -> Path:
    return tmp_path / "data" / "ohlcv" / f"{symbol}_1m.csv"


def _tp_path(tmp_path: Path, symbol: str = "DCE.v2701") -> Path:
    return tmp_path / "data" / "turning_points" / f"{symbol}_1m.csv"


def test_fetch_bars_writes_ohlcv_csv_and_sidecar(tmp_path: Path, capsys) -> None:
    config = _config_file(tmp_path)

    code = main(
        ["fetch", "--symbol", "DCE.v2701", "--period", "1m", "--bars", "3", "--config", str(config)],
        api_factory=_factory(_fake_api()),
    )

    assert code == 0
    path = _csv_path(tmp_path)
    assert path.is_file()
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    frame = pd.read_csv(path)
    assert list(frame.columns) == [
        "timestamp", "open", "high", "low", "close", "volume", "open_oi", "close_oi",
    ]
    assert len(frame) == 3
    assert sidecar["row_count"] == 3
    assert sidecar["symbol"] == "DCE.v2701"
    assert sidecar["start_dt"] == sidecar["first_timestamp"]
    captured = capsys.readouterr()
    assert "已落盘 K 线" in captured.out
    assert captured.err == ""  # 请求根数被满足时不告警


def test_fetch_is_byte_identical_across_runs(tmp_path: Path) -> None:
    config = _config_file(tmp_path)
    argv = ["fetch", "--symbol", "DCE.v2701", "--period", "1m", "--bars", "3", "--config", str(config)]

    assert main(argv, api_factory=_factory(_fake_api())) == 0
    first = _csv_path(tmp_path)
    first_bytes = (first.read_bytes(), first.with_suffix(".json").read_bytes())

    assert main(argv, api_factory=_factory(_fake_api())) == 0
    second_bytes = (first.read_bytes(), first.with_suffix(".json").read_bytes())

    assert first_bytes == second_bytes


def test_fetch_range_mode_uses_history_path(tmp_path: Path) -> None:
    config = _config_file(tmp_path)
    api = _fake_api()

    code = main(
        [
            "fetch",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-02",
            "--config",
            str(config),
        ],
        api_factory=_factory(api),
    )

    assert code == 0
    assert api.calls[-1]["method"] == "get_kline_data_series"
    sidecar = json.loads(_csv_path(tmp_path).with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["start_dt"] == "2024-01-01"
    assert sidecar["end_dt"] == "2024-01-02"


def test_fetch_accepts_multiple_symbols(tmp_path: Path) -> None:
    config = _config_file(tmp_path)

    code = main(
        [
            "fetch",
            "--symbol",
            "DCE.v2701",
            "--symbol",
            "SHFE.cu2607",
            "--period",
            "1m",
            "--bars",
            "3",
            "--config",
            str(config),
        ],
        api_factory=_factory(_fake_api()),
    )

    assert code == 0
    assert _csv_path(tmp_path, "DCE.v2701").is_file()
    assert _csv_path(tmp_path, "SHFE.cu2607").is_file()


def test_turning_points_is_offline_and_reproducible(tmp_path: Path) -> None:
    config = _config_file(tmp_path)
    assert (
        main(
            ["fetch", "--symbol", "DCE.v2701", "--period", "1m", "--bars", "3", "--config", str(config)],
            api_factory=_factory(_fake_api()),
        )
        == 0
    )

    def _forbidden(_cfg):  # pragma: no cover - 触发即为失败
        pytest.fail("turning-points 子命令不得联网")

    argv = ["turning-points", "--symbol", "DCE.v2701", "--period", "1m", "--config", str(config)]
    assert main(argv, api_factory=_forbidden) == 0

    path = _tp_path(tmp_path)
    assert path.is_file()
    first_bytes = (path.read_bytes(), path.with_suffix(".json").read_bytes())

    assert main(argv, api_factory=_forbidden) == 0
    assert first_bytes == (path.read_bytes(), path.with_suffix(".json").read_bytes())

    loaded = load_turning_points("DCE.v2701", "1m", data_dir=tmp_path / "data" / "turning_points")
    ohlcv = load_ohlcv("DCE.v2701", "1m", data_dir=tmp_path / "data" / "ohlcv")
    assert [p.kind for p in loaded.points][0] == "start"
    assert [p.kind for p in loaded.points][-1] == "close"
    assert loaded.meta["input_source_data_version"] == ohlcv.source_data_version
    assert loaded.meta["row_count"] == len(ohlcv.df)


def test_turning_points_can_read_from_explicit_data_dir(tmp_path: Path) -> None:
    config = _config_file(tmp_path)
    other_dir = tmp_path / "elsewhere"
    assert (
        main(
            [
                "fetch",
                "--symbol",
                "DCE.v2701",
                "--period",
                "1m",
                "--bars",
                "3",
                "--output-dir",
                str(other_dir),
                "--config",
                str(config),
            ],
            api_factory=_factory(_fake_api()),
        )
        == 0
    )

    code = main(
        [
            "turning-points",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--data-dir",
            str(other_dir / "ohlcv"),
            "--output-dir",
            str(other_dir),
            "--config",
            str(config),
        ]
    )

    assert code == 0
    assert (other_dir / "turning_points" / "DCE.v2701_1m.csv").is_file()


def test_prepare_produces_both_artifacts(tmp_path: Path) -> None:
    config = _config_file(tmp_path)

    code = main(
        [
            "prepare",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--bars",
            "4",
            "--initial-direction",
            "up",
            "--config",
            str(config),
        ],
        api_factory=_factory(_fake_api()),
    )

    assert code == 0
    assert _csv_path(tmp_path).is_file()
    assert _tp_path(tmp_path).is_file()
    sidecar = json.loads(_tp_path(tmp_path).with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["initial_direction_requested"] == "up"
    assert sidecar["initial_direction_resolved"] == "up"


def test_bars_shortfall_is_reported_on_stderr(tmp_path: Path, capsys) -> None:
    """P2-3/P2-N4：返回根数少于请求根数时显式告警，且按真实原因归因（不归给 8964）。"""
    config = _config_file(tmp_path)
    api = _fake_api()  # 序列中只有 5 根已收盘 K 线

    code = main(
        [
            "fetch",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--bars",
            "6",
            "--config",
            str(config),
        ],
        api_factory=_factory(api),
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "警告" in captured.err
    assert "请求 6 根" in captured.err
    assert "实际返回 5 根" in captured.err
    assert "与 --bars 上限无关" in captured.err  # P2-N4：真实原因归因
    assert "8964" not in captured.err  # 不得一律归因于上限
    assert "已落盘 K 线" in captured.out
    assert len(pd.read_csv(_csv_path(tmp_path))) == 5


def test_bars_shortfall_at_upper_limit_blames_the_limit(tmp_path: Path, capsys) -> None:
    """P2-N4 对照：请求恰为 ``--bars 8964`` 上限时，告警应指出上限原因。

    末两行分别是「当前分钟」与「下一分钟」的 K 线，均属未收盘必被剔除，
    因此实际返回为 ``8964 - 2``（与时钟无关：两行永远在 as_of 之后）。
    """
    config = _config_file(tmp_path)
    base = pd.Timestamp.now(tz="Asia/Shanghai").floor("min") + pd.Timedelta(minutes=1)
    expected_rows = SERIAL_MAX_DATA_LENGTH - 2
    rows = [
        (
            (base - pd.Timedelta(minutes=index)).strftime("%Y-%m-%d %H:%M:%S"),
            68000.0,
            68100.0,
            67900.0,
            68050.0,
            100,
            5000,
            5000,
        )
        for index in range(SERIAL_MAX_DATA_LENGTH - 1, -1, -1)
    ]
    api = FakeTqApi(serial=build_serial_klines(rows, width=SERIAL_MAX_DATA_LENGTH))

    code = main(
        [
            "fetch",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--bars",
            str(SERIAL_MAX_DATA_LENGTH),
            "--config",
            str(config),
        ],
        api_factory=_factory(api),
    )

    captured = capsys.readouterr()
    assert code == 0
    assert f"请求 {SERIAL_MAX_DATA_LENGTH} 根" in captured.err
    assert f"实际返回 {expected_rows} 根" in captured.err
    assert f"--bars 上限 {SERIAL_MAX_DATA_LENGTH}" in captured.err
    assert len(pd.read_csv(_csv_path(tmp_path))) == expected_rows


def test_output_dir_creation_failure_is_a_clean_error(tmp_path: Path, capsys) -> None:
    """P2-N2 回归：``output_dir`` 不可创建时给明确错误 + 退出码 1，不抛裸 OSError。"""
    config = _config_file(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    code = main(
        [
            "fetch",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--bars",
            "3",
            "--output-dir",
            str(blocker / "sub"),
            "--config",
            str(config),
        ],
        api_factory=_factory(_fake_api()),
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "错误：" in captured.err
    assert "输出目录不可用" in captured.err
    assert "NotADirectoryError" in captured.err
    assert "Traceback" not in captured.err
    assert str(blocker / "sub") in captured.err  # 可操作：指出路径


def test_malformed_config_does_not_leak_credentials(tmp_path: Path, capsys) -> None:
    """P1-1 回归：凭证行 YAML 语法错误 → CLI stderr 不含凭证明文，退出码 1。"""
    sentinel = "SENTINEL_PW_2f9a"
    config = tmp_path / "bad.yaml"
    config.write_text(
        "dataset:\n  output_dir: data\ntianqin:\n"
        f'  account: "acct"\n  password: "{sentinel}" : oops\n',
        encoding="utf-8",
    )

    code = main(
        [
            "fetch",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--bars",
            "3",
            "--config",
            str(config),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "YAML 解析失败" in captured.err
    assert sentinel not in captured.err
    assert sentinel not in captured.out
    assert not _csv_path(tmp_path).exists()  # 失败不产生半成品


def test_invalid_period_exits_nonzero_with_supported_list(capsys) -> None:
    code = main(["turning-points", "--symbol", "DCE.v2701", "--period", "7m"])

    captured = capsys.readouterr()
    assert code != 0
    assert "1m, 5m, 15m, 1h, 1d" in captured.err


def test_window_arguments_are_validated(tmp_path: Path, capsys) -> None:
    config = _config_file(tmp_path)
    base = ["fetch", "--symbol", "DCE.v2701", "--period", "1m", "--config", str(config)]

    assert main(base, api_factory=_factory(_fake_api())) == 2
    assert "必须提供 --bars" in capsys.readouterr().err

    assert (
        main(
            base + ["--bars", "3", "--start", "2024-01-01", "--end", "2024-01-02"],
            api_factory=_factory(_fake_api()),
        )
        == 2
    )
    assert "互斥" in capsys.readouterr().err

    assert main(base + ["--start", "2024-01-01"], api_factory=_factory(_fake_api())) == 2
    assert "必须提供 --bars" in capsys.readouterr().err

    assert main(base + ["--bars", "0"], api_factory=_factory(_fake_api())) == 2
    assert "1~8964" in capsys.readouterr().err

    assert main(base + ["--bars", "3", "--start", "2024/01/01", "--end", "2024-01-02"]) == 2
    assert "互斥" in capsys.readouterr().err


def test_missing_input_file_reports_failure(tmp_path: Path, capsys) -> None:
    config = _config_file(tmp_path)
    code = main(
        ["turning-points", "--symbol", "DCE.v2701", "--period", "1m", "--config", str(config)]
    )

    assert code == 1
    assert "OHLCV 文件不存在" in capsys.readouterr().err


def test_missing_credentials_message_does_not_leak(tmp_path: Path, capsys) -> None:
    sentinel = "S3CRET-PASSWORD-VALUE"
    config = _config_file(
        tmp_path, tianqin={"account": "", "password": sentinel}
    )

    code = main(
        [
            "fetch",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--bars",
            "3",
            "--config",
            str(config),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "缺少天勤凭证" in captured.err
    assert sentinel not in captured.err
    assert not _csv_path(tmp_path).exists()  # 失败不留半成品


def test_invalid_config_path_reports_failure(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "fetch",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--bars",
            "3",
            "--config",
            str(tmp_path / "missing.yaml"),
        ]
    )

    assert code == 1
    assert "配置文件不存在" in capsys.readouterr().err


def test_help_exits_zero_in_process(capsys) -> None:
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in ("fetch", "turning-points", "prepare"):
        assert name in out


def _run_subprocess(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MARKETSENSE_")}
    return subprocess.run(
        [sys.executable, "-m", "dataset", *args],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_subprocess_help_smoke() -> None:
    result = _run_subprocess("--help")

    assert result.returncode == 0
    for name in ("fetch", "turning-points", "prepare"):
        assert name in result.stdout


def test_subprocess_invalid_period_smoke() -> None:
    result = _run_subprocess("turning-points", "--symbol", "DCE.v2701", "--period", "7m")

    assert result.returncode != 0
    assert "1m, 5m, 15m, 1h, 1d" in result.stderr


def test_subprocess_malformed_config_does_not_leak_credentials(tmp_path: Path) -> None:
    """P1-1/P1-2 端到端：真实子进程 stderr 中不得出现哨兵密码（各类畸形形态）。"""
    sentinel = "SENTINEL_PW_2f9a"
    head = "dataset:\n  output_dir: data\ntianqin:\n  account: acct\n"
    vectors = {
        "extra_colon": head + f'  password: "{sentinel}" : oops\n',
        "undefined_alias": head + f"  password: *{sentinel}\n",
        "undefined_alias_account": (
            "dataset:\n  output_dir: data\ntianqin:\n"
            f"  account: *{sentinel}\n  password: \"x\"\n"
        ),
        "undefined_tag": head + f"  password: !{sentinel} value\n",
        "control_char": head + f'  password: "{sentinel}\x01"\n',
        # P1-2 回归扩展：合并键 / 多文档 / 保留指示符 / NUL 控制字符
        "merge_key_alias": head + f"  password: {sentinel}\n  <<: *{sentinel}\n",
        "multi_document": head + f'  password: "{sentinel}"\n---\nfoo: 1\n',
        "reserved_indicator": head + f"  password: @{sentinel}\n",
        "nul_byte": head + f'  password: "{sentinel}\x00"\n',
    }

    for name, text in vectors.items():
        config = tmp_path / f"bad_{name}.yaml"
        config.write_text(text, encoding="utf-8")

        result = _run_subprocess(
            "fetch",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--bars",
            "3",
            "--config",
            str(config),
        )

        assert result.returncode == 1, (name, result.stderr)
        assert "YAML 解析失败" in result.stderr, (name, result.stderr)
        assert sentinel not in result.stderr, (name, result.stderr)
        assert sentinel not in result.stdout, (name, result.stdout)
        assert str(config) in result.stderr, (name, result.stderr)


def test_deeply_nested_config_subprocess_does_not_leak_credentials(tmp_path: Path) -> None:
    """极端畸形输入（深嵌套 YAML，P2-N5）：真实子进程仍不得回显凭证值。

    本用例**只**锁定安全不变量（无泄露、不崩溃为 0 退出）。P2-N5 的契约缺口
    （``RecursionError`` 逃出 ``yaml.YAMLError`` → stderr 为 traceback、无 ``错误：``
    前缀）属已知非阻断缺陷，此处**不**把该行为固化为期望值，详见 04-test 报告的未覆盖项。
    """
    sentinel = "SENTINEL_PW_2f9a"
    config = tmp_path / "deep.yaml"
    config.write_text(
        "dataset:\n  output_dir: data\ntianqin:\n  account: acct\n"
        f'  password: "{sentinel}"\n  extra: ' + "[" * 600 + "]" * 600 + "\n",
        encoding="utf-8",
    )

    result = _run_subprocess(
        "fetch",
        "--symbol",
        "DCE.v2701",
        "--period",
        "1m",
        "--bars",
        "3",
        "--config",
        str(config),
    )

    assert result.returncode == 1, result.stderr
    assert sentinel not in result.stderr
    assert sentinel not in result.stdout


def test_fetch_with_bars_one_returns_a_single_closed_bar(tmp_path: Path, capsys) -> None:
    """``--bars`` 下边界 1：取 1 根已收盘 K 线，落盘正确且不告警。"""
    config = _config_file(tmp_path)

    code = main(
        [
            "fetch",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--bars",
            "1",
            "--config",
            str(config),
        ],
        api_factory=_factory(_fake_api()),
    )

    captured = capsys.readouterr()
    assert code == 0
    frame = pd.read_csv(_csv_path(tmp_path))
    assert len(frame) == 1
    sidecar = json.loads(
        _csv_path(tmp_path).with_suffix(".json").read_text(encoding="utf-8")
    )
    assert sidecar["row_count"] == 1
    assert captured.err == ""  # 请求根数被满足 → 不告警


def test_persisted_timestamps_are_ascending_unique_and_beijing(tmp_path: Path) -> None:
    """AC-2：落盘 CSV 的 timestamp 为 Asia/Shanghai（+08:00）、严格升序、无重复。"""
    config = _config_file(tmp_path)
    assert (
        main(
            [
                "fetch",
                "--symbol",
                "DCE.v2701",
                "--period",
                "1m",
                "--bars",
                "5",
                "--config",
                str(config),
            ],
            api_factory=_factory(_fake_api()),
        )
        == 0
    )

    raw = pd.read_csv(_csv_path(tmp_path))["timestamp"].tolist()
    assert all(str(value).endswith("+08:00") for value in raw)
    stamps = pd.to_datetime(pd.Series(raw))
    assert stamps.is_monotonic_increasing
    assert stamps.is_unique
    assert stamps.iloc[0].utcoffset() == pd.Timedelta(hours=8)


def test_turning_points_on_empty_input_file_reports_clean_error(
    tmp_path: Path, capsys
) -> None:
    """输入边界（空数据）：空 OHLCV 文件 → 明确错误 + 退出码 1 + 不留半成品。"""
    config = _config_file(tmp_path)
    input_dir = tmp_path / "data" / "ohlcv"
    input_dir.mkdir(parents=True)
    (input_dir / "DCE.v2701_1m.csv").write_text("", encoding="utf-8")

    code = main(
        [
            "turning-points",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--config",
            str(config),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "错误：" in captured.err
    assert "文件读取失败" in captured.err
    assert str(input_dir / "DCE.v2701_1m.csv") in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "data" / "turning_points").exists()


def test_duplicate_timestamps_are_rejected_without_partial_output(
    tmp_path: Path, capsys
) -> None:
    """输入边界（重复时间戳）：校验层拦截 → 退出码 1，CSV/sidecar 均不落盘。"""
    config = _config_file(tmp_path)
    rows = list(_SERIAL_ROWS) + [list(_SERIAL_ROWS)[-1]]  # 末两行时间戳重复
    api = FakeTqApi(serial=build_serial_klines(rows, width=8))

    code = main(
        [
            "fetch",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--bars",
            "6",
            "--config",
            str(config),
        ],
        api_factory=_factory(api),
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "TIMESTAMP_DUPLICATE" in captured.err
    assert "Traceback" not in captured.err
    assert not _csv_path(tmp_path).exists()
    assert not _csv_path(tmp_path).with_suffix(".json").exists()


def test_dependency_failure_reports_clean_error_without_leaking_credentials(
    tmp_path: Path, capsys
) -> None:
    """依赖失败（认证/网络）：明确错误 + 退出码 1 + 无 traceback + 不泄露配置凭证。"""
    sentinel = "S3CRET-CONN-PW"
    config = _config_file(tmp_path, tianqin={"account": "acct", "password": sentinel})

    def failing_factory(_cfg):
        raise ConnectionError("connection refused")

    code = main(
        [
            "fetch",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--bars",
            "3",
            "--config",
            str(config),
        ],
        api_factory=failing_factory,
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "天勤连接/认证失败" in captured.err
    assert "connection refused" in captured.err  # 仍给出可操作的原因
    assert "Traceback" not in captured.err
    assert sentinel not in captured.err
    assert sentinel not in captured.out
    assert not _csv_path(tmp_path).exists()


def test_bars_upper_limit_with_insufficient_source_writes_actual_rows(
    tmp_path: Path, capsys
) -> None:
    """P2-N6 边界：``--bars 8964`` 且数据源本身只有 5 根 → 不崩溃、行数与告警一致。

    已知非阻断（P2-N6）：此情形告警仍把原因归给 ``--bars`` 上限（文案精度问题），
    本用例只锁定「不静默、数量与退出码正确」，不把该归因固化为期望值。
    """
    config = _config_file(tmp_path)

    code = main(
        [
            "fetch",
            "--symbol",
            "DCE.v2701",
            "--period",
            "1m",
            "--bars",
            str(SERIAL_MAX_DATA_LENGTH),
            "--config",
            str(config),
        ],
        api_factory=_factory(_fake_api()),  # 桩源仅 5 根已收盘 K 线
    )

    captured = capsys.readouterr()
    assert code == 0
    assert f"请求 {SERIAL_MAX_DATA_LENGTH} 根" in captured.err
    assert "实际返回 5 根" in captured.err
    assert len(pd.read_csv(_csv_path(tmp_path))) == 5

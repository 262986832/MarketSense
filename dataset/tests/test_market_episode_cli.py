"""T5（CLI）：``dataset episode-generate`` 接入与既有子命令零改动。

覆盖：``--help`` 可运行、正常生成、缺失清单/品种配置的失败路径与退出码、
既有子命令的参数面未变。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dataset import config as config_module
from dataset.cli import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, build_parser, main
from dataset.market_episode.nanojev_records import AUDIT_FILENAME
from dataset.market_episode.segments import SPLIT_ROLES
from dataset.storage import save_ohlcv
from dataset.tests.market_episode_fixtures import (
    SYMBOL,
    build_two_day_workspace,
    build_workspace,
    frame,
    write_daily_turning_points,
    write_episode_config,
    write_manifest,
    write_symbols_config,
    segment_record,
)

_ROWS = [
    (100, 100.2, 99.8, 100),
    (100, 110, 100, 109),
] * 3


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


def test_build_parser_registers_episode_generate_without_changing_existing_commands() -> None:
    parser = build_parser()

    help_text = parser.format_help()
    for command in ("fetch", "turning-points", "prepare", "episode-generate"):
        assert command in help_text

    args = parser.parse_args(
        ["episode-generate", "--segments", "s.jsonl", "--output-dir", "out", "--config", "c.yaml"]
    )
    assert args.command == "episode-generate"
    assert args.segments == "s.jsonl"
    assert args.output_dir == "out"
    assert args.config == "c.yaml"

    # T5 新增参数：--daily-turning-points-dir（显式覆盖默认推导目录，既有参数面不变；
    # 该参数注册在 episode-generate 子 parser，顶层 --help 不展示）
    assert "--daily-turning-points-dir" not in parser.format_help()
    subparser = next(
        action.choices["episode-generate"]
        for action in parser._actions
        if action.choices and "episode-generate" in getattr(action, "choices", {})
    )
    assert "--daily-turning-points-dir" in subparser.format_help()
    tp_args = parser.parse_args(
        [
            "episode-generate",
            "--segments",
            "s.jsonl",
            "--output-dir",
            "out",
            "--config",
            "c.yaml",
            "--daily-turning-points-dir",
            "tp",
        ]
    )
    assert tp_args.daily_turning_points_dir == "tp"

    fetch_args = parser.parse_args(["fetch", "--symbol", "S", "--period", "1m", "--bars", "3"])
    assert fetch_args.command == "fetch"


def test_episode_generate_help_exits_ok(capsys) -> None:
    code = main(["episode-generate", "--help"])

    assert code == EXIT_OK
    assert "episode-generate" in capsys.readouterr().out


def test_episode_generate_missing_segments_is_usage_error(capsys) -> None:
    code = main(["episode-generate"])

    assert code == EXIT_USAGE
    assert "--segments" in capsys.readouterr().err


def test_episode_generate_writes_split_files_and_reports_path(tmp_path: Path, capsys) -> None:
    workspace = build_workspace(tmp_path, _ROWS)
    output_dir = tmp_path / "out"

    code = main(
        [
            "episode-generate",
            "--segments",
            str(workspace.manifest),
            "--config",
            str(workspace.config_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == EXIT_OK, captured.err
    assert "已生成 episode 训练数据" in captured.out
    run_dirs = [path for path in output_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert run_dir.name.startswith("run-")
    for split in SPLIT_ROLES:
        assert (run_dir / f"{split}.jsonl").is_file()
    assert (run_dir / AUDIT_FILENAME).is_file()
    assert str(run_dir) in captured.out

    train_rows = [
        json.loads(line)
        for line in (run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["split"] for row in train_rows] == ["train"] * len(train_rows)
    assert train_rows


def test_episode_generate_is_idempotent_for_same_input(tmp_path: Path, capsys) -> None:
    workspace = build_workspace(tmp_path, _ROWS)
    output_dir = tmp_path / "out"
    argv = [
        "episode-generate",
        "--segments",
        str(workspace.manifest),
        "--config",
        str(workspace.config_path),
        "--output-dir",
        str(output_dir),
    ]

    assert main(argv) == EXIT_OK
    first = {
        path.name: path.read_bytes()
        for path in next(output_dir.iterdir()).iterdir()
        if path.is_file()
    }
    assert main(argv) == EXIT_OK
    capsys.readouterr()
    second = {
        path.name: path.read_bytes()
        for path in next(output_dir.iterdir()).iterdir()
        if path.is_file()
    }

    assert first == second


def test_episode_generate_missing_manifest_file_fails(tmp_path: Path, capsys) -> None:
    workspace = build_workspace(tmp_path, _ROWS)

    code = main(
        [
            "episode-generate",
            "--segments",
            str(tmp_path / "nope.jsonl"),
            "--config",
            str(workspace.config_path),
        ]
    )

    assert code == EXIT_FAILURE
    assert "片段清单不存在" in capsys.readouterr().err


def test_episode_generate_missing_symbols_config_fails(tmp_path: Path, capsys) -> None:
    workspace = build_workspace(tmp_path, _ROWS)
    workspace.symbols_path.unlink()

    code = main(
        [
            "episode-generate",
            "--segments",
            str(workspace.manifest),
            "--config",
            str(workspace.config_path),
        ]
    )

    assert code == EXIT_FAILURE
    assert "品种配置不存在" in capsys.readouterr().err


def test_episode_generate_rejects_manifest_without_required_splits(tmp_path: Path, capsys) -> None:
    workspace = build_workspace(tmp_path, _ROWS)
    manifest = write_manifest(
        tmp_path / "only_train.jsonl",
        [
            segment_record("seg-a", "train", start_index=0, end_index=1),
            segment_record("seg-b", "train", start_index=2, end_index=3),
        ],
    )

    code = main(
        [
            "episode-generate",
            "--segments",
            str(manifest),
            "--config",
            str(workspace.config_path),
        ]
    )

    assert code == EXIT_FAILURE
    assert "train/dev/test" in capsys.readouterr().err


def test_episode_generate_reports_skipped_segments_on_stderr(tmp_path: Path, capsys) -> None:
    """无上一交易日日线的片段被跳过并打 stderr 告警（不静默）。"""
    workspace = build_two_day_workspace(tmp_path)
    output_dir = tmp_path / "out"

    code = main(
        [
            "episode-generate",
            "--segments",
            str(workspace.manifest),
            "--config",
            str(workspace.config_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == EXIT_OK, captured.err
    assert "[board-state 跳过片段] seg-train" in captured.err
    assert "trade_date=2024-01-02" in captured.err
    assert "共跳过 1 个片段" in captured.err


def test_episode_generate_uses_configured_output_dir_by_default(tmp_path: Path, capsys) -> None:
    workspace = build_workspace(
        tmp_path, _ROWS, episode={"output_dir": str(tmp_path / "configured")}
    )

    code = main(
        [
            "episode-generate",
            "--segments",
            str(workspace.manifest),
            "--config",
            str(workspace.config_path),
        ]
    )

    assert code == EXIT_OK, capsys.readouterr().err
    assert list((tmp_path / "configured").iterdir())


def test_episode_generate_reports_trend_extreme_skips_on_stderr(tmp_path: Path, capsys) -> None:
    """T5（d）：折点单侧缺失片段 → 逐片段 + 汇总告警写 stderr，退出码 0；
    跳过片段不产生记录，其余片段正常产出。"""
    day1 = frame(_ROWS[:4], start="2024-01-02 09:00:00")
    day2 = frame(_ROWS[:4], start="2024-01-03 09:00:00")
    data_dir = tmp_path / "data" / "ohlcv"
    save_ohlcv(pd.concat([day1, day2]).reset_index(drop=True), symbol=SYMBOL, period="1m", output_dir=data_dir)
    save_ohlcv(
        pd.concat(
            (
                frame([(4000.0, 4010.0, 3990.0, 4005.0)], start="2024-01-01 00:00:00"),
                frame([(4000.0, 4010.0, 3990.0, 4005.0)], start="2024-01-02 00:00:00"),
                frame([(4000.0, 4010.0, 3990.0, 4005.0)], start="2024-01-03 00:00:00"),
            )
        ).reset_index(drop=True),
        symbol=SYMBOL, period="1d", output_dir=data_dir,
    )
    # 折点均确认于 01-02：== seg-train 决策交易日（不可用 → 跳过）；
    # < seg-dev/seg-test 决策交易日 01-03 且 up/down 各 ≥2（可用 → 正常产出）；
    # v9 同源不变式：bar_index = 1d 文件行号（01-02 = 行 1）
    write_daily_turning_points(
        data_dir.parent / "turning_points" / f"{SYMBOL}_1d.csv",
        rows=[
            {"kind": "down", "timestamp": "2024-01-02 09:05:00+08:00", "price": 3980.0,
             "bar_index": 1, "volume": 120, "oi": 5010,
             "trend_extreme_price": 3980.0, "trend_extreme_bar_index": 1},
            {"kind": "up", "timestamp": "2024-01-02 09:35:00+08:00", "price": 4020.0,
             "bar_index": 1, "volume": 130, "oi": 5020,
             "trend_extreme_price": 4020.0, "trend_extreme_bar_index": 1},
            {"kind": "down", "timestamp": "2024-01-02 10:05:00+08:00", "price": 3990.0,
             "bar_index": 1, "volume": 140, "oi": 5030,
             "trend_extreme_price": 3990.0, "trend_extreme_bar_index": 1},
            {"kind": "up", "timestamp": "2024-01-02 10:35:00+08:00", "price": 4040.0,
             "bar_index": 1, "volume": 150, "oi": 5040,
             "trend_extreme_price": 4040.0, "trend_extreme_bar_index": 1},
        ],
    )
    manifest = write_manifest(
        tmp_path / "segments.jsonl",
        [
            segment_record("seg-train", "train", start_index=0, end_index=3, start="2024-01-02 09:00:00"),
            segment_record("seg-dev", "dev", start_index=0, end_index=1, start="2024-01-03 09:00:00"),
            segment_record("seg-test", "test", start_index=2, end_index=3, start="2024-01-03 09:00:00"),
        ],
    )
    symbols_path = write_symbols_config(tmp_path / "symbols.yaml")
    config_path = write_episode_config(
        tmp_path / "episode.yaml", data_dir=data_dir, symbols_path=symbols_path
    )

    code = main(
        [
            "episode-generate",
            "--segments",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    assert code == EXIT_OK
    # 逐片段 + 汇总告警（seg-train 因折点不可用被跳过）
    assert "[trend-extreme 跳过片段] seg-train" in captured.err
    assert "trade_date=2024-01-02 缺日线转折点" in captured.err
    assert "共跳过 1 个片段（缺可用日线转折点，详见 stderr）" in captured.err
    run_dirs = sorted((tmp_path / "out").glob("run-*"))
    assert len(run_dirs) == 1
    train_lines = (run_dirs[0] / "train.jsonl").read_text(encoding="utf-8").splitlines()
    dev_lines = (run_dirs[0] / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    assert train_lines == []  # 跳过片段不产生记录
    assert dev_lines  # 非跳过片段正常产出
    assert run_dirs[0].joinpath(AUDIT_FILENAME).is_file()


def test_episode_generate_missing_turning_points_file_fails(tmp_path: Path, capsys) -> None:
    """T5（f）：TP 文件缺失 → stderr 硬报错（含路径），退出码 1。"""
    workspace = build_workspace(tmp_path, _ROWS)
    workspace.turning_points_path.unlink()

    code = main(
        [
            "episode-generate",
            "--segments",
            str(workspace.manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--config",
            str(workspace.config_path),
        ]
    )

    captured = capsys.readouterr()
    assert code == EXIT_FAILURE
    assert "转折点文件不存在" in captured.err
    assert str(workspace.turning_points_path) in captured.err
    assert not (tmp_path / "out").exists()

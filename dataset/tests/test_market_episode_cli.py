"""T5（CLI）：``dataset episode-generate`` 接入与既有子命令零改动。

覆盖：``--help`` 可运行、正常生成、缺失清单/品种配置的失败路径与退出码、
既有子命令的参数面未变。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset import config as config_module
from dataset.cli import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, build_parser, main
from dataset.market_episode.nanojev_records import AUDIT_FILENAME
from dataset.market_episode.segments import SPLIT_ROLES
from dataset.tests.market_episode_fixtures import build_workspace, write_manifest, segment_record

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

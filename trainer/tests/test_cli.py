"""T6/T7 CLI 单测：validate-only（纯 stdlib）/ self-check（CPU）/ train CUDA 硬门。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainer.cli import main
from trainer.tests.run_fixtures import default_run, write_run, make_record

# ---------------------------------------------------------------- validate-only（torch 无关）


def test_validate_only_reports_pull(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    run = default_run(tmp_path)
    code = main(["--data", str(run), "--validate-only"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"] == str(run)
    assert payload["splits"]["train"] == {"records": 3, "questions": 3}
    assert payload["splits"]["test"] == {"records": 1, "questions": 1}
    train_pull = payload["pull"]["train"]
    assert train_pull["open"]["count"] == 2
    assert train_pull["open"]["total_weight"] == pytest.approx(2.5)  # 2.0 + 0.5
    assert train_pull["hold"]["count"] == 1
    assert payload["pull"]["dev"]["open"]["count"] == 1
    assert payload["utility_weight"]["utility_alpha"] == 0.5
    assert payload["utility_weight"]["direction_penalty_lambda"] == 0.25


def test_validate_only_requires_train_dev_test_nonempty(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    run = default_run(tmp_path)
    (run / "test.jsonl").unlink()
    assert main(["--data", str(run), "--validate-only"]) == 1
    assert "test 必须非空" in capsys.readouterr().err


def test_validate_only_surfaces_join_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    run = default_run(tmp_path)
    (run / "outcomes.jsonl").write_text("", encoding="utf-8")  # 旁挂空 → 开仓记录缺行
    assert main(["--data", str(run), "--validate-only"]) == 1
    assert "缺少 outcome 旁挂行" in capsys.readouterr().err


def test_validate_only_weight_validation_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    run = default_run(tmp_path)
    assert main(["--data", str(run), "--validate-only", "--weight-min", "1.5"]) == 1
    assert "weight_min" in capsys.readouterr().err


def test_validate_only_invalid_lambda(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    run = default_run(tmp_path)
    assert main(["--data", str(run), "--validate-only", "--direction-penalty-lambda", "-1"]) == 1
    assert "direction_penalty_lambda" in capsys.readouterr().err


def test_missing_data_arg() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--validate-only"])
    assert excinfo.value.code == 2


def test_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--validate-only", "--self-check"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------- self-check（torch）


def test_self_check_passes(capsys: pytest.CaptureFixture) -> None:
    pytest.importorskip("torch")
    assert main(["--self-check"]) == 0
    out = capsys.readouterr().out
    items = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert items[-1] == {"check": "self_check", "status": "ok"}
    checks = {item["check"] for item in items}
    assert {
        "weight_boundaries",
        "weighted_ce_value",
        "penalty_gradient_signs",
        "numerical_gradient",
        "tiny_backbone_step",
        "pack_behavior",
    } <= checks


# ---------------------------------------------------------------- train CUDA 硬门


def test_train_requires_cuda(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():  # 本机无 CUDA；有 CUDA 的机器跳过此门测试
        pytest.skip("CUDA 可用，无法在本机复现 CUDA 硬门")
    run = default_run(tmp_path)
    code = main(["--data", str(run), "--train", "--output-dir", str(tmp_path / "out")])
    assert code == 2
    err = capsys.readouterr().err
    assert "CUDA" in err
    assert "README §7.4" in err


# ---------------------------------------------------------------- 夹具自洽


def test_fixture_run_has_expected_layout(tmp_path: Path) -> None:
    run = default_run(tmp_path)
    assert (run / "train.jsonl").is_file()
    assert (run / "dev.jsonl").is_file()
    assert (run / "test.jsonl").is_file()
    assert (run / "outcomes.jsonl").is_file()
    assert not (run / "calibration.jsonl").exists()
    assert make_record("seg:0", "train", "open_long")["id"] == "seg:0"
    assert write_run(tmp_path / "w", []) == tmp_path / "w"

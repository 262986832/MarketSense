"""T7 训练循环单测（torch）：确定性双跑 / 日志契约 / checkpoint / 错误路径。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from trainer.errors import TrainerError  # noqa: E402
from trainer.losses import question_ce_values  # noqa: E402
from trainer.model import DecisionModel, build_tiny_backbone  # noqa: E402
from trainer.train import (  # noqa: E402
    TrainingConfig,
    evaluate_dev_ce,
    run_training,
    save_checkpoint,
)

_CONFIG = TrainingConfig(
    steps=3,
    head_steps=2,
    batch_questions=4,
    microbatch_questions=2,
    max_microbatch_tokens=16384,
    eval_every=1,
    seed=17,
)


def _example(question_id: str, *, weight: float = 1.0, is_open: bool = False) -> dict:
    return {
        "id": question_id,
        "type": "choice",
        "candidate_ids": ["c0", "c1", "c2"],
        "leaf_tokens": [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
        "gold_index": 1,
        "weight": weight,
        "is_open": is_open,
        "opposite_index": 2 if is_open else None,
    }


def _examples() -> dict[str, list[dict]]:
    train = [
        _example(f"t{i:02d}:0:next_action", weight=2.5 if i % 3 == 0 else 1.0, is_open=i % 3 == 0)
        for i in range(30)
    ]
    dev = [_example(f"d{i}:0:next_action", weight=1.0) for i in range(6)]
    return {"train": train, "dev": dev}


def _fresh_model() -> DecisionModel:
    torch.manual_seed(11)
    return DecisionModel(build_tiny_backbone(), "attention")


def _train_once(out_dir: Path) -> dict:
    lines: list[str] = []
    save_checkpoint(
        _fresh_model(),
        out_dir,
        set_head="attention",
        max_length=512,
        data_sha256={},
        hyperparams={},
    )
    summary = run_training(
        _fresh_model(),
        _examples(),
        pad_token_id=0,
        config=_CONFIG,
        out_dir=out_dir,
        direction_penalty_lambda=0.25,
        log=lines.append,
    )
    return {"summary": summary, "lines": lines}


def test_training_is_deterministic_double_run(tmp_path: Path) -> None:
    """同参数双跑：日志逐行一致 + best.safetensors 逐字节一致。"""
    first = _train_once(tmp_path / "first")
    second = _train_once(tmp_path / "second")
    assert first["lines"] == second["lines"]
    assert first["summary"]["best_step"] == second["summary"]["best_step"]
    assert first["summary"]["best_dev_ce"] == pytest.approx(second["summary"]["best_dev_ce"])
    best_bytes = [
        hashlib.sha256((tmp_path / name / "best.safetensors").read_bytes()).hexdigest()
        for name in ("first", "second")
    ]
    assert best_bytes[0] == best_bytes[1]


def test_log_lines_have_contract_fields_without_wallclock(tmp_path: Path) -> None:
    """日志键 = 契约集合；无任何墙钟时间字段（AGENTS.md §7）。"""
    lines = _train_once(tmp_path / "run")["lines"]
    parsed = [json.loads(line) for line in lines]
    assert len(parsed) == _CONFIG.head_steps + _CONFIG.steps
    for item in parsed:
        assert set(item) <= {"step", "phase", "loss", "questions", "microbatches",
                             "batch_question_ids_sha256", "dev_ce"}
        assert "elapsed_seconds" not in item
        assert item["phase"] in ("head", "full")
        assert item["questions"] == _CONFIG.batch_questions
    assert all("dev_ce" in item for item in parsed if item["phase"] == "full")


def test_single_denominator_across_microbatches(tmp_path: Path) -> None:
    """单一优化步单一分母：微批数 = 批题数 / 微批题数；sha 字段为 64 位十六进制。"""
    lines = _train_once(tmp_path / "run")["lines"]
    first = json.loads(lines[0])
    assert first["microbatches"] == 2  # 4 题 / 微批 2
    assert isinstance(first["batch_question_ids_sha256"], str)
    assert len(first["batch_question_ids_sha256"]) == 64


def test_checkpoint_skeleton_and_refusal_to_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "ckpt"
    save_checkpoint(
        _fresh_model(),
        out,
        set_head="attention",
        max_length=512,
        data_sha256={"train.jsonl": "aa", "outcomes.jsonl": "bb"},
        hyperparams={"steps": 3, "model": "tiny"},
    )
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["schema_version"] == "marketsense-utility-trainer-v1"
    assert config["set_head"] == "attention"
    assert config["max_length"] == 512
    assert config["data_sha256"] == {"outcomes.jsonl": "bb", "train.jsonl": "aa"}
    assert set(config["deps"]) == {"torch", "transformers", "safetensors"}
    assert config["utility_weight"]["utility_alpha"] == 0.5
    assert (out / "backbone_config" / "config.json").is_file()
    assert not (out / "tokenizer").exists()  # tokenizer=None → 不落 tokenizer/
    with pytest.raises(TrainerError, match="不覆盖"):
        save_checkpoint(
            _fresh_model(), out, set_head="attention", max_length=512,
            data_sha256={}, hyperparams={},
        )


def test_run_training_writes_best_and_train_log(tmp_path: Path) -> None:
    out = tmp_path / "ckpt"
    save_checkpoint(
        _fresh_model(), out, set_head="attention", max_length=512,
        data_sha256={}, hyperparams={},
    )
    lines: list[str] = []
    summary = run_training(
        _fresh_model(), _examples(), pad_token_id=0, config=_CONFIG,
        out_dir=out, direction_penalty_lambda=0.25, log=lines.append,
    )
    assert summary["best_step"] >= 1
    assert 0.0 < summary["best_dev_ce"] < float("inf")
    assert (out / "best.safetensors").is_file()
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    assert len(log) == _CONFIG.head_steps + _CONFIG.steps
    with pytest.raises(TrainerError, match="config.json"):
        run_training(
            _fresh_model(), _examples(), pad_token_id=0, config=_CONFIG,
            out_dir=tmp_path / "missing",
        )


def test_run_training_requires_nonempty_train_dev(tmp_path: Path) -> None:
    out = tmp_path / "ckpt"
    save_checkpoint(
        _fresh_model(), out, set_head="attention", max_length=512,
        data_sha256={}, hyperparams={},
    )
    examples = _examples()
    with pytest.raises(TrainerError, match="train split 必须非空"):
        run_training(
            _fresh_model(), {"train": [], "dev": examples["dev"]}, pad_token_id=0,
            config=_CONFIG, out_dir=out,
        )
    with pytest.raises(TrainerError, match="dev split 必须非空"):
        run_training(
            _fresh_model(), {"train": examples["train"]}, pad_token_id=0,
            config=_CONFIG, out_dir=out,
        )
    with pytest.raises(TrainerError, match="少于一个有效批"):
        run_training(
            _fresh_model(),
            {"train": examples["train"][:2], "dev": examples["dev"]},
            pad_token_id=0, config=_CONFIG, out_dir=out,
        )


def test_training_config_rejects_invalid_values() -> None:
    with pytest.raises(TrainerError, match="必须 > 0"):
        TrainingConfig(steps=0)
    with pytest.raises(TrainerError, match="head_steps"):
        TrainingConfig(head_steps=-1)
    with pytest.raises(TrainerError, match="precision"):
        TrainingConfig(precision="fp16")
    with pytest.raises(TrainerError, match="grad_clip"):
        TrainingConfig(grad_clip=0.0)


def test_evaluate_dev_ce_unequal_microbatch_groups() -> None:
    """不等大微批组回归（REVIEW P1-2 回修）：题均 CE 与打包分组方式无关。

    3 道 dev 题、microbatch_questions=2 → 组 [2, 1]（现有训练夹具
    dev=6/微批=2 恰为等大组——等大组恰好掩盖了此缺陷，故成此前盲区）。缺陷实现（逐微批累加
    「题均 mean」再除以总题数）会偏离真实题均；修复后三种口径必须一致：
    不等大组、单组全量、以及不经打包的直接全批前向真值。
    """
    dev = [_example(f"d{i}:0:next_action", weight=1.0) for i in range(3)]
    model = _fresh_model()
    model.eval()
    value_unequal = evaluate_dev_ce(
        model, dev, pad_token_id=0,
        config=TrainingConfig(microbatch_questions=2),  # 组 [2, 1]
    )
    value_single_group = evaluate_dev_ce(
        model, dev, pad_token_id=0,
        config=TrainingConfig(microbatch_questions=8),  # 单组 [3]
    )
    with torch.inference_mode():
        logits, _ = model(dev, 0)
        expected = float(question_ce_values(logits, dev).mean())
    # 容差 rel=1e-4：跨分组前向的 fp32 噪声（padding 宽度不同的归约顺序）
    # 实测 ~1e-7 量级；缺陷实现的偏离为 O(30%)（组 [2,1] 时 ≈ −33%），
    # 相距容差 3 个数量级以上，不会漏报。
    assert value_unequal == pytest.approx(expected, rel=1e-4)
    assert value_single_group == pytest.approx(expected, rel=1e-4)

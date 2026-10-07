"""训练循环与 checkpoint 落盘（utility-trainer 任务 T7；torch）。

训练机制与 NanoJev ``train_pipeline_decisions.py`` L460-533 同语义（只读参考，
不 import 其代码），差异点均为 DESIGN 明确要求：

* 损失 = 效用加权 CE + 方向惩罚（:mod:`trainer.losses`），微批累积采用
  「单个优化步单一分母」语义（``loss_sum / len(batch)``，与微批大小/K 无关）；
* checkpoint 选择 = **未加权 dev 题均 CE**（评估与目标分离）；
* 训练日志**不含墙钟时间**（NanoJev 日志含 ``elapsed_seconds``，本实现剔除；
  AGENTS.md §7 确定性要求）；
* checkpoint 文件集与键兼容 NanoJev 推理脚本只读加载（``config.json`` +
  ``backbone_config/`` + ``tokenizer/`` + ``best.safetensors``）。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from trainer.errors import TrainerError
from trainer.losses import question_ce_values, question_weighted_loss_sum
from trainer.weights import (
    TRAINER_CHECKPOINT_SCHEMA,
    UTILITY_WEIGHT_SCHEMA,
    UtilityWeightConfig,
)

__all__ = [
    "TrainingConfig",
    "run_training",
    "save_checkpoint",
    "seed_everything",
]

#: 训练日志 JSON 行的固定键（无墙钟时间字段）
_LOG_FIELDS = ("step", "phase", "loss", "questions", "microbatches", "batch_question_ids_sha256")


@dataclass(frozen=True)
class TrainingConfig:
    """训练超参（默认值镜像 NanoJev 新训练路径；非默认值由 CLI 传入）。"""

    steps: int = 300
    head_steps: int = 12
    batch_questions: int = 12
    microbatch_questions: int = 4
    max_microbatch_tokens: int = 16384
    eval_every: int = 50
    seed: int = 17
    backbone_lr: float = 2e-5
    head_lr: float = 2e-4
    head_warmup_lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if min(self.steps, self.batch_questions, self.microbatch_questions, self.eval_every) <= 0:
            raise TrainerError("steps/batch_questions/microbatch_questions/eval_every 必须 > 0")
        if self.head_steps < 0:
            raise TrainerError("head_steps 必须 ≥ 0")
        if self.max_microbatch_tokens < 0:
            raise TrainerError("max_microbatch_tokens 必须 ≥ 0")
        for name in ("backbone_lr", "head_lr", "head_warmup_lr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise TrainerError(f"{name} 必须为有限正数: {value!r}")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise TrainerError(f"weight_decay 必须为非负有限数: {self.weight_decay!r}")
        if not math.isfinite(self.grad_clip) or self.grad_clip <= 0:
            raise TrainerError(f"grad_clip 必须为有限正数: {self.grad_clip!r}")
        if self.precision not in ("fp32", "bf16"):
            raise TrainerError(f"precision 必须为 fp32 或 bf16: {self.precision!r}")


def seed_everything(seed: int) -> None:
    """固定 python/torch 随机源（CUDA 可用时含 CUDA；确定性要求，AGENTS.md §7）。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_question_ids_sha256(batch: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256("\n".join(ex["id"] for ex in batch).encode()).hexdigest()


def save_checkpoint(
    model: torch.nn.Module,
    out_dir: str | Path,
    *,
    set_head: str,
    max_length: int,
    data_sha256: Mapping[str, str],
    hyperparams: Mapping[str, Any],
    weight_config: UtilityWeightConfig | None = None,
    tokenizer: Any = None,
) -> Path:
    """写入 checkpoint 目录（文件集与键兼容 NanoJev ``local_checkpoint_files``；
    已有 checkpoint 不覆盖，镜像 NanoJev 约定）。

    文件集：``config.json``（含 ``set_head``/``max_length``/``schema_version``/
    ``data_sha256``/``deps``/超参）+ ``backbone_config/`` + ``tokenizer/``（提供
    tokenizer 时）+ ``best.safetensors``（由训练过程按 dev 改善写入）。
    """
    out = Path(out_dir)
    if (out / "config.json").exists() or (out / "best.safetensors").exists():
        raise TrainerError(f"输出目录已有 checkpoint，不覆盖（请用新目录）: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if weight_config is None:
        weight_config = UtilityWeightConfig()
    config = {
        "schema_version": TRAINER_CHECKPOINT_SCHEMA,
        "set_head": set_head,
        "max_length": max_length,
        "data_sha256": dict(sorted(data_sha256.items())),
        "deps": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "safetensors")
        },
        "utility_weight": {
            "schema": UTILITY_WEIGHT_SCHEMA,
            "utility_alpha": weight_config.utility_alpha,
            "r_cap": weight_config.r_cap,
            "weight_min": weight_config.weight_min,
            "weight_max": weight_config.weight_max,
            "direction_penalty_lambda": weight_config.direction_penalty_lambda,
        },
        "parameter_count": sum(tensor.numel() for tensor in model.parameters()),
        "selection": "minimum dev unweighted question-mean CE",
        **dict(hyperparams),
    }
    (out / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if tokenizer is not None:
        tokenizer.save_pretrained(out / "tokenizer")
    model.backbone.config.save_pretrained(out / "backbone_config")
    return out


def _packed_microbatches(examples, config: TrainingConfig):
    from trainer.data import pack_complete_questions

    return pack_complete_questions(
        examples, config.microbatch_questions, config.max_microbatch_tokens
    )


def _forward_loss(
    model: torch.nn.Module,
    group: Sequence[Mapping[str, Any]],
    pad_token_id: int,
    config: TrainingConfig,
    direction_penalty_lambda: float,
) -> torch.Tensor:
    """单微批加权损失**和**（分母由调用方按整个优化步统一处理）。"""
    if config.precision == "bf16":
        device_type = model.scalar.weight.device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True):
            logits, _ = model(group, pad_token_id)
            loss = question_weighted_loss_sum(
                logits, group, direction_penalty_lambda=direction_penalty_lambda
            )
        return loss
    logits, _ = model(group, pad_token_id)
    return question_weighted_loss_sum(
        logits, group, direction_penalty_lambda=direction_penalty_lambda
    )


def evaluate_dev_ce(
    model: torch.nn.Module,
    dev_examples: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    config: TrainingConfig,
) -> float:
    """未加权 dev 题均 CE（checkpoint 选择口径）。

    逐微批累加**逐题 CE 之和**、最后除以总题数（``Σ_ce / N``）：题均指标必须
    与打包分组方式无关——若对「逐微批题均 mean」直接跨组累加再除以总题数，
    不等大组时小组题目会获得 ``n_full/n_small`` 倍的相对权重（REVIEW P1-2
    回修；回归测试 ``test_evaluate_dev_ce_unequal_microbatch_groups``）。
    """
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for group in _packed_microbatches(list(dev_examples), config):
            logits, _ = model(group, pad_token_id)
            total += float(question_ce_values(logits, group).sum())
            count += len(group)
    if count == 0:
        raise TrainerError("dev 评估没有任何题目")
    return total / count


def run_training(
    model: torch.nn.Module,
    examples_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    pad_token_id: int,
    config: TrainingConfig,
    out_dir: str | Path,
    direction_penalty_lambda: float = 0.25,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """确定性训练循环（head warmup → full 微调；dev CE 选 best checkpoint）。

    :param examples_by_split: 至少含非空 ``train``/``dev``（``build_examples`` 产物，
        每题带 ``gold_index``/``weight``/``is_open``/``opposite_index``）。
    :param log: 逐行 JSON 日志输出回调（键 = ``_LOG_FIELDS``；无墙钟时间）。
    :raises TrainerError: train/dev 为空、训练题不足一个批、任一完整题超微批预算
        （训练前预检）、无有限 dev checkpoint 可选。
    """
    from trainer.data import pack_complete_questions

    out = Path(out_dir)
    if not (out / "config.json").is_file():
        raise TrainerError(
            f"输出目录缺少 config.json（请先调用 save_checkpoint 落盘 checkpoint 骨架）: {out}"
        )
    train_examples = list(examples_by_split.get("train", ()))
    dev_examples = list(examples_by_split.get("dev", ()))
    if not train_examples:
        raise TrainerError("train split 必须非空")
    if not dev_examples:
        raise TrainerError("dev split 必须非空")
    if len(train_examples) < config.batch_questions:
        raise TrainerError(
            f"训练题数（{len(train_examples)}）少于一个有效批（{config.batch_questions}）"
        )
    # 训练前预检：任何完整题都不能超声明预算（失败快，镜像 NanoJev L507-509）
    for split, examples in examples_by_split.items():
        pack_complete_questions(
            list(examples), config.microbatch_questions, config.max_microbatch_tokens
        )
    if config.precision == "bf16" and model.scalar.weight.device.type != "cuda":
        raise TrainerError("bf16 精度要求 CUDA 设备（本机不可用时请用 fp32）")

    seed_everything(config.seed)

    head_params = [
        param
        for name, param in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    body_params = list(model.backbone.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": body_params, "lr": config.backbone_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )

    best = float("inf")
    best_step: int | None = None
    logs: list[dict[str, Any]] = []
    total_steps = config.head_steps + config.steps
    for step in range(total_steps):
        warm = step < config.head_steps
        for param in body_params:
            param.requires_grad_(not warm)
        optimizer.param_groups[1]["lr"] = config.head_warmup_lr if warm else config.head_lr
        batch = random.sample(train_examples, config.batch_questions)
        groups = pack_complete_questions(
            batch, config.microbatch_questions, config.max_microbatch_tokens
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # 单一优化步单一分母（len(batch)），与微批大小/K 无关（镜像 NanoJev 累积语义）
        loss_sum = 0.0
        for group in groups:
            loss = (
                _forward_loss(model, group, pad_token_id, config, direction_penalty_lambda)
                / len(batch)
            )
            if not torch.isfinite(loss):
                raise TrainerError("损失非有限，训练中止")
            loss.backward()
            loss_sum += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        item = {
            "step": step + 1,
            "phase": "head" if warm else "full",
            "loss": loss_sum,
            "questions": len(batch),
            "microbatches": len(groups),
            "batch_question_ids_sha256": _batch_question_ids_sha256(batch),
        }
        if not warm and (
            (step + 1 - config.head_steps) % config.eval_every == 0 or step + 1 == total_steps
        ):
            dev_ce = evaluate_dev_ce(
                model, dev_examples, pad_token_id=pad_token_id, config=config
            )
            item["dev_ce"] = dev_ce
            if math.isfinite(dev_ce) and dev_ce < best:
                best, best_step = dev_ce, step + 1
                from safetensors.torch import save_file

                save_file(
                    {
                        key: value.detach().cpu().contiguous().clone()
                        for key, value in model.state_dict().items()
                    },
                    out / "best.safetensors",
                )
        logs.append(item)
        log(json.dumps(item, ensure_ascii=False, sort_keys=True))
    if best_step is None:
        raise TrainerError("没有任何有限 dev checkpoint 被选中")
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(out / "best.safetensors")))
    (out / "train_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return {"best_step": best_step, "best_dev_ce": best, "logs": logs}

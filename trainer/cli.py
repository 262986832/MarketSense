"""trainer 命令行入口（utility-trainer 任务 T6/T7）。

用法（详见 README §7）::

    python -m trainer --data RUN_DIR --validate-only          # 纯 stdlib，不 import torch
    python -m trainer --self-check                            # CPU 自检（tiny backbone，不下载模型）
    python -m trainer --data RUN_DIR --train --output-dir DIR # 真实训练（CUDA 硬门）

三种模式互斥；``--validate-only`` 输出逐类权重拉力报告（JSON）；
真实训练有 CUDA 硬门（本机无 CUDA 时报错并指引 README §7.4）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from trainer.data import (
    OUTCOMES_FILENAME,
    SPLITS,
    build_examples,
    load_run,
    pull_report,
)
from trainer.errors import TrainerError
from trainer.weights import UTILITY_WEIGHT_SCHEMA, UtilityWeightConfig

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trainer",
        description="MarketSense 效用训练器（独立于 NanoJev 代码，模型与其 DecisionModel 架构同构）",
    )
    parser.add_argument(
        "--data",
        metavar="DIR",
        help="episode 生成 run 目录（{split}.jsonl + outcomes.jsonl）",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="纯 stdlib 校验 + 逐类权重拉力报告（不 import torch）",
    )
    mode.add_argument(
        "--self-check",
        action="store_true",
        help="CPU 自检（权重/梯度数值断言 + tiny backbone 一步前向反向；不下载模型）",
    )
    mode.add_argument("--train", action="store_true", help="真实训练（CUDA 硬门）")
    parser.add_argument("--output-dir", metavar="DIR", help="训练 checkpoint 输出目录")
    # 效用权重配置（DESIGN D8 默认值）
    parser.add_argument("--utility-alpha", type=float, default=0.5)
    parser.add_argument("--r-cap", type=float, default=3.0)
    parser.add_argument("--weight-min", type=float, default=0.5)
    parser.add_argument("--weight-max", type=float, default=2.5)
    parser.add_argument("--direction-penalty-lambda", type=float, default=0.25)
    # 训练超参（默认镜像 NanoJev 新训练路径）
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="backbone HF 模型名/路径")
    parser.add_argument("--set-head", choices=["none", "attention"], default="attention")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--head-steps", type=int, default=12)
    parser.add_argument("--batch-questions", type=int, default=12)
    parser.add_argument("--microbatch-questions", type=int, default=4)
    parser.add_argument("--max-microbatch-tokens", type=int, default=16384)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--head-warmup-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="fp32")
    return parser


def _weight_config_from_args(args: argparse.Namespace) -> UtilityWeightConfig:
    try:
        return UtilityWeightConfig(
            utility_alpha=args.utility_alpha,
            r_cap=args.r_cap,
            weight_min=args.weight_min,
            weight_max=args.weight_max,
            direction_penalty_lambda=args.direction_penalty_lambda,
        )
    except ValueError as error:
        raise TrainerError(str(error)) from error


def _validate_only(args: argparse.Namespace) -> int:
    """纯 stdlib：schema 子集校验 + 连接校验 + train/dev/test 非空 + 拉力报告。"""
    config = _weight_config_from_args(args)
    records_by_split, outcomes_by_id, _ = load_run(args.data)
    for split in ("train", "dev", "test"):
        if not records_by_split[split]:
            # 镜像 NanoJev L477-479（只读参考）：train/dev/test 必须非空
            raise TrainerError(f"{split} 必须非空")
    report = {
        "data": str(args.data),
        "splits": {
            split: {
                "records": len(records_by_split[split]),
                "questions": len(records_by_split[split]),
            }
            for split in SPLITS
        },
        "pull": pull_report(records_by_split, outcomes_by_id, config),
        "utility_weight": {
            "schema": UTILITY_WEIGHT_SCHEMA,
            "utility_alpha": config.utility_alpha,
            "r_cap": config.r_cap,
            "weight_min": config.weight_min,
            "weight_max": config.weight_max,
            "direction_penalty_lambda": config.direction_penalty_lambda,
        },
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def _data_sha256(run_dir: str | Path) -> dict[str, str]:
    """输入指纹：现有 split 文件 + outcome 旁挂（存在才计）。"""
    run = Path(run_dir)
    digests: dict[str, str] = {}
    for split in SPLITS:
        path = run / f"{split}.jsonl"
        if path.is_file():
            digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = run / OUTCOMES_FILENAME
    if sidecar.is_file():
        digests[OUTCOMES_FILENAME] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    return digests


def _train(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from trainer.model import DecisionModel, from_pretrained_kwargs
    from trainer.train import TrainingConfig, run_training, save_checkpoint

    if not torch.cuda.is_available():
        print(
            "trainer: 真实训练需要 CUDA（训练与推理入口硬性要求 CUDA，本机不可训练）；"
            "CPU 冒烟请用 --self-check / --validate-only，环境与坑位见 README §7.4",
            file=sys.stderr,
        )
        return 2
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        print("trainer: 当前 CUDA 设备不支持 bf16，请用 --precision fp32", file=sys.stderr)
        return 2
    if not args.output_dir:
        print("trainer: --train 需要 --output-dir", file=sys.stderr)
        return 2

    config = TrainingConfig(
        steps=args.steps,
        head_steps=args.head_steps,
        batch_questions=args.batch_questions,
        microbatch_questions=args.microbatch_questions,
        max_microbatch_tokens=args.max_microbatch_tokens,
        eval_every=args.eval_every,
        seed=args.seed,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        head_warmup_lr=args.head_warmup_lr,
        weight_decay=args.weight_decay,
        precision=args.precision,
    )
    weight_config = _weight_config_from_args(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    backbone = AutoModel.from_pretrained(
        args.model, attn_implementation="sdpa", trust_remote_code=False,
        **from_pretrained_kwargs(torch.float32),
    )
    model = DecisionModel(backbone, args.set_head)
    model.backbone.config.use_cache = False
    max_position = getattr(
        model.backbone.config, "max_position_embeddings", args.max_length
    )
    if args.max_length > max_position:
        raise TrainerError(
            f"max_length={args.max_length} 超过 backbone 上下文上限 {max_position}"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    records_by_split, outcomes_by_id, _ = load_run(args.data)
    examples_by_split = build_examples(
        records_by_split,
        outcomes_by_id,
        encode=encode,
        eos_token_id=tokenizer.eos_token_id,
        max_length=args.max_length,
        config=weight_config,
    )
    if not examples_by_split["train"]:
        raise TrainerError("train split 必须非空")
    if not examples_by_split["dev"]:
        raise TrainerError("dev split 必须非空")
    hyperparams = {
        "model": args.model,
        "data_dir": str(args.data),
        "steps": config.steps,
        "head_steps": config.head_steps,
        "batch_questions": config.batch_questions,
        "microbatch_questions": config.microbatch_questions,
        "max_microbatch_tokens": config.max_microbatch_tokens,
        "eval_every": config.eval_every,
        "seed": config.seed,
        "backbone_lr": config.backbone_lr,
        "head_lr": config.head_lr,
        "head_warmup_lr": config.head_warmup_lr,
        "weight_decay": config.weight_decay,
        "grad_clip": config.grad_clip,
        "precision": config.precision,
        "parameter_storage": "float32",
    }
    save_checkpoint(
        model,
        args.output_dir,
        set_head=args.set_head,
        max_length=args.max_length,
        data_sha256=_data_sha256(args.data),
        hyperparams=hyperparams,
        weight_config=weight_config,
        tokenizer=tokenizer,
    )
    summary = run_training(
        model,
        examples_by_split,
        pad_token_id=tokenizer.pad_token_id,
        config=config,
        out_dir=args.output_dir,
        direction_penalty_lambda=weight_config.direction_penalty_lambda,
    )
    summary_payload = {
        "best_step": summary["best_step"],
        "best_dev_ce": summary["best_dev_ce"],
        "selected_on": "dev unweighted question-mean CE",
        "output_dir": str(args.output_dir),
    }
    (Path(args.output_dir) / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"done": str(args.output_dir), **summary_payload}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.self_check:
            from trainer.selfcheck import run_self_check

            return run_self_check()
        if not args.data:
            parser.error("--data is required")
        if args.validate_only:
            return _validate_only(args)
        return _train(args)
    except TrainerError as error:
        print(f"trainer: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"trainer: {error}", file=sys.stderr)
        return 1

"""MarketSense 效用训练器（utility-trainer 任务，2026-10-06 拍板 D1–D8）。

独立顶层包：**不 import ``dataset``、不 import NanoJev**（D1）。模型与 NanoJev
DecisionModel 架构同构、checkpoint 可被其推理脚本只读加载（D3）；损失 = 效用
加权 CE + 方向惩罚 λ（D2/D8）；数据源 = episode run 目录（``{split}.jsonl`` +
``outcomes.jsonl``，与 NanoJev 契约对齐）。

入口：``python -m trainer``（``--validate-only`` 纯 stdlib / ``--self-check`` CPU /
``--train`` CUDA 硬门），详见 :mod:`trainer.cli` 与 README §7。
"""

from __future__ import annotations

from trainer.errors import TrainerError
from trainer.weights import (
    TRAINER_CHECKPOINT_SCHEMA,
    UTILITY_WEIGHT_SCHEMA,
    UtilityWeightConfig,
    utility_weight,
)

__all__ = [
    "TRAINER_CHECKPOINT_SCHEMA",
    "UTILITY_WEIGHT_SCHEMA",
    "TrainerError",
    "UtilityWeightConfig",
    "utility_weight",
]

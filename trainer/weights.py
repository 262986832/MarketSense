"""效用权重配置与纯函数（utility-trainer 任务 T5；torch 无关，纯 Python）。

权重形状（DESIGN 冻结：`artifacts/utility-trainer/02-design/tech-design.md`，
2026-10-06 用户拍板 D8）::

    w(r) = clip(1 + α·clip(r, −1, r_cap), w_min, w_max)
    r    = r_multiple（无 outcome 的记录 = 中性，w = 1.0）

默认 α=0.5、r_cap=3.0、w∈[0.5, 2.5]；r=−1 → 0.5、r=0 → 1.0、r≥3 → 2.5。
方向惩罚系数 λ=0.25 仅作用于开仓记录、**不乘记录权重**（与效用权重正交，
λ 语义不随 outcome 分布漂移）；λ=0 关闭。

r_multiple 口径与数据侧 `outcomes.jsonl` 同源（= outcome / risk_ratio，
比值记账、分母 = 片段首根开盘价，见 dataset/market_episode 生成侧）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "UTILITY_WEIGHT_SCHEMA",
    "TRAINER_CHECKPOINT_SCHEMA",
    "UtilityWeightConfig",
    "utility_weight",
]

#: 权重配置的 schema 标记（写进训练 config.json，便于后续追溯）
UTILITY_WEIGHT_SCHEMA = "marketsense.utility_weight.v1"

#: 本训练器 checkpoint 的 schema 标记（与 NanoJev 的
#: ``openjev-decision-pipeline-v1`` 区分，语义见 tech-design D3）
TRAINER_CHECKPOINT_SCHEMA = "marketsense-utility-trainer-v1"


@dataclass(frozen=True)
class UtilityWeightConfig:
    """效用权重 + 方向惩罚系数配置（非法值在构造时即拒绝，不静默）。"""

    utility_alpha: float = 0.5
    r_cap: float = 3.0
    weight_min: float = 0.5
    weight_max: float = 2.5
    direction_penalty_lambda: float = 0.25

    def __post_init__(self) -> None:
        values = {
            "utility_alpha": self.utility_alpha,
            "r_cap": self.r_cap,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "direction_penalty_lambda": self.direction_penalty_lambda,
        }
        for name, value in values.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} 必须是数值: {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} 必须为有限数值: {value!r}")
        if self.utility_alpha < 0:
            raise ValueError(f"utility_alpha 必须 ≥ 0: {self.utility_alpha!r}")
        if self.r_cap <= 0:
            raise ValueError(f"r_cap 必须为正数: {self.r_cap!r}")
        if not 0 < self.weight_min <= 1 <= self.weight_max:
            raise ValueError(
                "必须满足 0 < weight_min ≤ 1 ≤ weight_max（否则中性记录 w=1.0 "
                f"落在可达区间之外）: got [{self.weight_min!r}, {self.weight_max!r}]"
            )
        if self.direction_penalty_lambda < 0:
            raise ValueError(
                f"direction_penalty_lambda 必须 ≥ 0: {self.direction_penalty_lambda!r}"
            )


def utility_weight(
    r_multiple: float | None, config: UtilityWeightConfig | None = None
) -> float:
    """单个开仓记录的效用权重（纯函数；无 outcome → 1.0）。

    ``w(r) = clip(1 + α·clip(r, −1, r_cap), w_min, w_max)``；
    ``r = None``（非开仓记录 / 缺 outcome）→ 中性 1.0。
    """
    if config is None:
        config = UtilityWeightConfig()
    if r_multiple is None:
        return 1.0
    r = float(r_multiple)
    if not math.isfinite(r):
        raise ValueError(f"r_multiple 必须为有限数值: {r_multiple!r}")
    r_eff = min(max(r, -1.0), config.r_cap)
    weight = 1.0 + config.utility_alpha * r_eff
    return min(max(weight, config.weight_min), config.weight_max)

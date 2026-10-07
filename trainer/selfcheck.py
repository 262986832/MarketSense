"""CPU 自检（``--self-check``；utility-trainer 任务 T7，验收 #2）。

检查项（全部 CPU、fp32、不下载任何模型，风格镜像 NanoJev ``self_check``）：

1. 权重边界值（r=−1/0/+3/+10 → 0.5/1.0/2.5/2.5；无 outcome → 1.0）；
2. 加权 CE 解析值比对（手算 softmax/NLL，容差 1e-6）；
3. 方向惩罚梯度符号（autograd：∂Pen/∂z_反向 > 0、∂Pen/∂z_其余 < 0）；
4. 加权损失数值梯度 vs 解析梯度（中心差分，容差 1e-4）；
5. tiny 随机 backbone 一步前向反向（loss 有限、参数更新非零）；
6. 完整题打包行为（超预算报错、预算内正确分组）。
"""

from __future__ import annotations

import json
import math
from typing import Any

import torch

from trainer.data import pack_complete_questions
from trainer.errors import TrainerError
from trainer.losses import question_penalty_values, question_weighted_loss_sum
from trainer.model import DecisionModel, build_tiny_backbone
from trainer.weights import UtilityWeightConfig, utility_weight

__all__ = ["run_self_check", "self_check_results"]


def _example(
    question_id: str,
    *,
    gold_index: int,
    candidate_count: int = 3,
    weight: float = 1.0,
    is_open: bool = False,
    opposite_index: int | None = None,
) -> dict[str, Any]:
    return {
        "id": question_id,
        "type": "choice",
        "candidate_ids": [f"c{i}" for i in range(candidate_count)],
        "leaf_tokens": [[1, 2, 3]] * candidate_count,
        "gold_index": gold_index,
        "weight": weight,
        "is_open": is_open,
        "opposite_index": opposite_index,
    }


def self_check_results() -> list[dict[str, Any]]:
    """执行全部自检项，返回逐项结果（不打印）；任一失败抛 ``TrainerError``。"""
    results: list[dict[str, Any]] = []

    # 1. 权重边界值
    config = UtilityWeightConfig()
    checks = {
        "r=-1": (utility_weight(-1.0, config), config.weight_min),
        "r=0": (utility_weight(0.0, config), 1.0),
        "r=+3": (utility_weight(3.0, config), config.weight_max),
        "r=+10": (utility_weight(10.0, config), config.weight_max),
        "r=None": (utility_weight(None, config), 1.0),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise TrainerError(f"自检权重边界失败 {name}: {actual!r} ≠ {expected!r}")
    results.append({"check": "weight_boundaries", "status": "ok", "cases": len(checks)})

    # 2. 加权 CE 解析值比对
    examples = [
        _example("q1", gold_index=0, weight=2.5, is_open=True, opposite_index=1),
        _example("q2", gold_index=2, weight=1.0),
    ]
    logits = torch.log(torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]))
    lambda_value = 0.25
    total = question_weighted_loss_sum(
        logits, examples, direction_penalty_lambda=lambda_value
    )
    ce1 = -(math.log(0.2))
    ce2 = -(math.log(0.7))
    pen1 = -math.log1p(-0.3)
    expected_total = 2.5 * ce1 + lambda_value * pen1 + 1.0 * ce2
    if not abs(float(total) - expected_total) <= 1e-6:
        raise TrainerError(
            f"自检加权 CE 失败: {float(total)!r} ≠ {expected_total!r}"
        )
    results.append({"check": "weighted_ce_value", "status": "ok"})

    # 3. 方向惩罚梯度符号
    penalty_logits = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    penalty_examples = [
        _example("q", gold_index=2, is_open=True, opposite_index=0)
    ]
    penalty = question_penalty_values(penalty_logits, penalty_examples)
    penalty.sum().backward()
    grad = penalty_logits.grad[0]
    if not grad[0] > 0:
        raise TrainerError(f"自检方向惩罚梯度符号失败: ∂Pen/∂z_反向 = {grad[0]!r} 应 > 0")
    if not (grad[1] < 0 and grad[2] < 0):
        raise TrainerError(
            f"自检方向惩罚梯度符号失败: ∂Pen/∂z_其余 应 < 0: {grad[1:].tolist()}"
        )
    results.append({"check": "penalty_gradient_signs", "status": "ok"})

    # 4. 数值梯度 vs 解析梯度（中心差分；float64 比较，fp32 差分精度不足）
    numeric_logits = torch.tensor(
        [[0.1, -0.2, 0.3], [0.5, 0.0, -0.4]], dtype=torch.float64, requires_grad=True
    )
    numeric_examples = [
        _example("a", gold_index=0, weight=2.5, is_open=True, opposite_index=1),
        _example("b", gold_index=2, weight=0.8),
    ]
    loss = question_weighted_loss_sum(
        numeric_logits, numeric_examples, direction_penalty_lambda=0.25
    )
    loss.backward()
    analytic = numeric_logits.grad.clone()
    epsilon = 1e-4
    for row in range(numeric_logits.shape[0]):
        for column in range(numeric_logits.shape[1]):
            plus = numeric_logits.detach().clone()
            minus = numeric_logits.detach().clone()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            loss_plus = question_weighted_loss_sum(
                plus, numeric_examples, direction_penalty_lambda=0.25
            )
            loss_minus = question_weighted_loss_sum(
                minus, numeric_examples, direction_penalty_lambda=0.25
            )
            numeric = float((loss_plus - loss_minus) / (2 * epsilon))
            if not abs(numeric - float(analytic[row, column])) <= 1e-4:
                raise TrainerError(
                    f"自检数值梯度失败: [{row},{column}] numeric={numeric!r} "
                    f"analytic={float(analytic[row, column])!r}"
                )
    results.append({"check": "numerical_gradient", "status": "ok"})

    # 5. tiny 随机 backbone 一步前向反向
    torch.manual_seed(7)
    model = DecisionModel(build_tiny_backbone(), "attention")
    tiny_examples = [
        {
            "id": "t1",
            "type": "choice",
            "candidate_ids": ["c0", "c1", "c2"],
            "leaf_tokens": [[5, 6, 7, 8], [5, 6], [7, 8, 9, 10, 11]],
            "gold_index": 1,
            "weight": 2.0,
            "is_open": True,
            "opposite_index": 0,
        }
    ]
    step_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name in ("scalar.weight", "set_output.weight")
    }
    step_logits, _ = model(tiny_examples, pad_token=0)
    step_loss = question_weighted_loss_sum(
        step_logits, tiny_examples, direction_penalty_lambda=0.25
    )
    if not torch.isfinite(step_loss):
        raise TrainerError(f"自检 tiny 前向失败: loss 非有限 {float(step_loss)!r}")
    step_loss.backward()
    step_optimizer.step()
    for name, parameter in model.named_parameters():
        if name in before and bool(torch.equal(before[name], parameter.detach())):
            raise TrainerError(f"自检 tiny 反向失败: 参数 {name} 未更新")
    results.append({"check": "tiny_backbone_step", "status": "ok"})

    # 6. 完整题打包行为
    packed = pack_complete_questions(tiny_examples * 5, 2, 0)
    if [len(group) for group in packed] != [2, 2, 1]:
        raise TrainerError(f"自检打包行为失败: 分组 {[len(g) for g in packed]} ≠ [2, 2, 1]")
    oversized = {
        "id": "big",
        "type": "choice",
        "candidate_ids": ["c0", "c1"],
        "leaf_tokens": [[1] * 10] * 2,
    }
    try:
        pack_complete_questions([oversized], 4, 10)
    except TrainerError:
        results.append({"check": "pack_behavior", "status": "ok"})
    else:
        raise TrainerError("自检打包行为失败: 超预算题未报错")
    return results


def run_self_check(log=print) -> int:
    """执行全部自检并逐行打印 JSON 结果；全部通过返回 0。"""
    for item in self_check_results():
        log(json.dumps(item, ensure_ascii=False, sort_keys=True))
    log(json.dumps({"check": "self_check", "status": "ok"}, ensure_ascii=False))
    return 0

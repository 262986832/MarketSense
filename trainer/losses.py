"""损失函数（utility-trainer 任务 T5；纯 torch，无 backbone 依赖）。

损失 v1（DESIGN D2/D8 冻结）::

    L = (1/B) · Σ_i [ w_i·CE_i + 1{i 开仓}·λ·Pen_i ]

* ``CE_i = −log softmax(z_i[:k])[gold_i]``（完整候选 softmax、padding 不进 softmax）；
* ``Pen_i = −log(1 − p(反向候选))``，仅开仓记录、**不乘记录权重**（与效用权重
  正交）；``p`` 先 clamp ≤ 1−1e-6 再 ``log1p`` 保证数值稳定；
* 题均归一（除以 B = 有效批题数），镜像 NanoJev ``grouped_target_loss``
  （``train_pipeline_decisions.py`` L228-234，只读参考）的语义。

梯度符号（解析，由 ``trainer/tests`` 与 ``--self-check`` 数值断言）：
``∂Pen/∂z_反向 > 0``、``∂Pen/∂z_其余候选 < 0``——即惩罚项把概率质量从反向
候选推向其余候选（含 gold），与交叉熵方向一致、幅度正交于记录权重。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from trainer.errors import TrainerError

__all__ = [
    "question_ce_values",
    "question_penalty_values",
    "question_weighted_loss_sum",
    "question_unweighted_ce_mean",
]

#: 概率上界 clamp（1 − 1e-6）：保证 1 − p ≥ 1e-6，log1p 数值稳定
_PROBABILITY_MAX = 1.0 - 1e-6


def question_ce_values(logits: torch.Tensor, examples: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    """逐题 gold 交叉熵（完整候选 softmax；``logits`` 已按候选 padding 掩码）。"""
    if len(logits) != len(examples):
        raise TrainerError(
            f"logits 行数与题数不一致: {len(logits)} vs {len(examples)}"
        )
    values = []
    for row, example in enumerate(examples):
        k = len(example["candidate_ids"])
        gold = example["gold_index"]
        if not 0 <= gold < k:
            raise TrainerError(f"{example['id']} gold_index 越界: {gold} (k={k})")
        log_probs = F.log_softmax(logits[row, :k], dim=-1)
        values.append(-log_probs[gold])
    return torch.stack(values)


def question_penalty_values(
    logits: torch.Tensor, examples: Sequence[Mapping[str, Any]]
) -> torch.Tensor:
    """逐题方向惩罚 ``−log1p(−p(反向候选))``（仅开仓题有定义）。

    ``p = softmax(z[:k])[反向候选索引]``，clamp ≤ 1−1e-6 后 ``log1p``。
    """
    values = []
    for row, example in enumerate(examples):
        opposite = example.get("opposite_index")
        if opposite is None:
            values.append(torch.zeros((), dtype=logits.dtype, device=logits.device))
            continue
        k = len(example["candidate_ids"])
        probs = torch.softmax(logits[row, :k], dim=-1)
        p_opposite = probs[opposite].clamp(max=_PROBABILITY_MAX)
        values.append(-torch.log1p(-p_opposite))
    return torch.stack(values)


def question_weighted_loss_sum(
    logits: torch.Tensor,
    examples: Sequence[Mapping[str, Any]],
    *,
    direction_penalty_lambda: float,
) -> torch.Tensor:
    """一批题的加权损失**总和**（除以整个优化步的批题数由调用方负责，
    镜像 NanoJev「单个优化步单一分母，与微批大小/K 无关」的累积语义）。

    ``λ = 0`` 时短路跳过惩罚项计算（保证精确为 0，且不产生额外图节点）。
    """
    if not examples:
        raise TrainerError("损失计算需要至少一道完整题")
    ce = question_ce_values(logits, examples)
    terms = [logits.new_tensor(0.0)] * len(examples)
    for index, example in enumerate(examples):
        terms[index] = float(example.get("weight", 1.0)) * ce[index]
    if direction_penalty_lambda > 0:
        penalty = question_penalty_values(logits, examples)
        for index, example in enumerate(examples):
            if example.get("is_open"):
                terms[index] = terms[index] + direction_penalty_lambda * penalty[index]
    return torch.stack(terms).sum()


def question_unweighted_ce_mean(
    logits: torch.Tensor, examples: Sequence[Mapping[str, Any]]
) -> torch.Tensor:
    """单组题的未加权题均 CE（标量）。

    注意：跨微批累加须用 :func:`question_ce_values` 逐题求和（``sum()``）
    后再除以总题数——对「题均 mean」直接跨组累加再除以总题数，不等大组时
    会给小组题目不成比例的权重（REVIEW P1-2 回修的缺陷模式，勿复用）。
    """
    return question_ce_values(logits, examples).mean()

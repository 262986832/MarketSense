"""T5 损失函数单测（torch）：数值公式 / padding 屏蔽 / λ=0 短路 / 梯度符号。"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from trainer.errors import TrainerError  # noqa: E402
from trainer.losses import (  # noqa: E402
    question_ce_values,
    question_penalty_values,
    question_unweighted_ce_mean,
    question_weighted_loss_sum,
)
from trainer.weights import utility_weight  # noqa: E402


def _example(
    question_id: str,
    gold_index: int,
    *,
    weight: float = 1.0,
    is_open: bool = False,
    opposite_index: int | None = None,
    candidate_count: int = 3,
) -> dict:
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


def test_ce_values_match_manual_computation() -> None:
    examples = [_example("a", 0), _example("b", 2)]
    logits = torch.log(torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]))
    values = question_ce_values(logits, examples)
    assert values[0] == pytest.approx(-math.log(0.2), rel=1e-6)
    assert values[1] == pytest.approx(-math.log(0.7), rel=1e-6)


def test_ce_values_ignore_padded_candidates() -> None:
    """kmax > k 时，padding 列（置 -1e9）不得进入 softmax。"""
    examples = [_example("a", 1, candidate_count=2)]
    logits = torch.tensor([[0.0, 0.0, -1e9]])
    values = question_ce_values(logits, examples)
    assert values[0] == pytest.approx(math.log(2.0), rel=1e-6)


def test_ce_values_reject_out_of_range_gold() -> None:
    examples = [_example("a", 5)]
    with pytest.raises(TrainerError, match="越界"):
        question_ce_values(torch.zeros(1, 3), examples)


def test_weighted_loss_sum_matches_formula() -> None:
    examples = [
        _example("a", 0, weight=2.5, is_open=True, opposite_index=1),
        _example("b", 2, weight=1.0),
    ]
    logits = torch.log(torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]))
    lambda_value = 0.25
    total = question_weighted_loss_sum(logits, examples, direction_penalty_lambda=lambda_value)
    ce_a, ce_b = -math.log(0.2), -math.log(0.7)
    penalty_a = -math.log1p(-0.3)
    expected = 2.5 * ce_a + lambda_value * penalty_a + 1.0 * ce_b
    assert float(total) == pytest.approx(expected, rel=1e-6)


def test_penalty_not_multiplied_by_weight() -> None:
    """权重变化不得影响惩罚项贡献（DESIGN D2：惩罚与效用权重正交）。"""
    lambda_value = 0.5
    base = [_example("a", 0, weight=1.0, is_open=True, opposite_index=2)]
    heavy = [_example("a", 0, weight=4.0, is_open=True, opposite_index=2)]
    logits = torch.log(torch.tensor([[0.2, 0.3, 0.5]]))
    loss_base = question_weighted_loss_sum(logits, base, direction_penalty_lambda=lambda_value)
    loss_heavy = question_weighted_loss_sum(logits, heavy, direction_penalty_lambda=lambda_value)
    ce = -math.log(0.2)
    penalty = -math.log1p(-0.5)
    assert float(loss_base) == pytest.approx(1.0 * ce + 0.5 * penalty, rel=1e-6)
    assert float(loss_heavy) == pytest.approx(4.0 * ce + 0.5 * penalty, rel=1e-6)


def test_lambda_zero_skips_penalty_exactly() -> None:
    examples = [_example("a", 0, weight=2.0, is_open=True, opposite_index=1)]
    logits = torch.log(torch.tensor([[0.2, 0.3, 0.5]]))
    loss = question_weighted_loss_sum(logits, examples, direction_penalty_lambda=0.0)
    assert float(loss) == pytest.approx(2.0 * -math.log(0.2), rel=1e-6)  # fp32 精度


def test_lambda_zero_contributes_no_grad_to_penalty() -> None:
    examples = [_example("a", 0, weight=1.0, is_open=True, opposite_index=1)]
    logits = torch.zeros(1, 3, requires_grad=True)
    question_weighted_loss_sum(logits, examples, direction_penalty_lambda=0.0).backward()
    # 纯 CE 的梯度 = p − onehot；opposite_index=1 的梯度应为 softmax 概率 1/3
    expected = torch.tensor([1 / 3 - 1, 1 / 3, 1 / 3])
    assert torch.allclose(logits.grad[0], expected, atol=1e-6)


def test_penalty_gradient_signs() -> None:
    examples = [_example("a", 2, is_open=True, opposite_index=0)]
    logits = torch.zeros(1, 3, requires_grad=True)
    penalty = question_penalty_values(logits, examples)
    penalty.sum().backward()
    grad = logits.grad[0]
    assert grad[0] > 0  # 反向候选：升概率受罚
    assert grad[1] < 0 and grad[2] < 0  # 其余候选（含 gold）：降概率受罚


def test_penalty_value_formula() -> None:
    examples = [_example("a", 1, is_open=True, opposite_index=2)]
    logits = torch.log(torch.tensor([[0.2, 0.3, 0.5]]))
    penalty = question_penalty_values(logits, examples)
    assert float(penalty[0]) == pytest.approx(-math.log1p(-0.5), rel=1e-6)


def test_non_open_examples_have_zero_penalty() -> None:
    examples = [_example("a", 0), _example("b", 1, is_open=False, opposite_index=None)]
    logits = torch.zeros(2, 3)
    penalty = question_penalty_values(logits, examples)
    assert float(penalty.abs().sum()) == 0.0


def test_unweighted_ce_mean() -> None:
    examples = [_example("a", 0), _example("b", 2)]
    logits = torch.log(torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]))
    mean = question_unweighted_ce_mean(logits, examples)
    assert float(mean) == pytest.approx((-math.log(0.2) + -math.log(0.7)) / 2, rel=1e-6)


def test_weighted_loss_sum_requires_examples() -> None:
    with pytest.raises(TrainerError, match="至少一道"):
        question_weighted_loss_sum(torch.zeros(0, 3), [], direction_penalty_lambda=0.25)


#: 可分玩具集的真实 6 动作候选（与数据侧决策问题同键序）
_TOY_ACTIONS = ["open_long", "open_short", "close", "reverse", "hold", "stay_flat"]
_TOY_STAY_FLAT = _TOY_ACTIONS.index("stay_flat")
_TOY_OPPOSITE = {"open_long": "open_short", "open_short": "open_long"}


def _toy_example(question_id: str, gold_action: str, *, r_multiple: float | None) -> dict:
    """可分玩具集的一道完整题（真实 6 动作候选；权重由 outcome 分布经
    ``utility_weight`` 给出：开仓题 r≥r_cap → w=2.5，中性题 w=1.0）。"""
    is_open = gold_action in _TOY_OPPOSITE
    return {
        "id": question_id,
        "type": "choice",
        "candidate_ids": list(_TOY_ACTIONS),
        "leaf_tokens": [[1, 2, 3]] * len(_TOY_ACTIONS),
        "gold_index": _TOY_ACTIONS.index(gold_action),
        "weight": utility_weight(r_multiple),
        "is_open": is_open,
        "opposite_index": (
            _TOY_ACTIONS.index(_TOY_OPPOSITE[gold_action]) if is_open else None
        ),
    }


def test_separable_toy_lazy_solution_loss_far_exceeds_converged() -> None:
    """可分集行为断言（执行计划 T5 / 需求验收 3；REVIEW P1-1 回修）。

    构造可分特征 + 已知 outcome 分布的玩具集：4 道开仓题
    （gold = open_long/open_short、r_multiple = 4.0 ≥ r_cap → w = 2.5）
    + 6 道空仓题（gold = stay_flat、w = 1.0）。比较两个解在
    ``question_weighted_loss_sum`` 下的加权总损失：

    * 懒惰解（永远预测不开仓）：每题都把概率压在 stay_flat 上——空仓题
      恰好全对，但每道开仓题付出完整 margin 的 CE（且被 w=2.5 放大）；
    * 收敛解：每题 gold 候选概率 → 1。

    断言：① 懒惰解加权总损失显著高于收敛解（≥ 10×）；② 效用加权后
    懒惰解的代价高于等权（outcome 分布驱动的再加权确实抬升摆烂解成本）。
    """
    examples = [
        _toy_example(f"open{i}", "open_long" if i % 2 == 0 else "open_short",
                     r_multiple=4.0)
        for i in range(4)
    ] + [_toy_example(f"flat{i}", "stay_flat", r_multiple=None) for i in range(6)]
    margin = 8.0
    lazy_logits = torch.zeros(len(examples), len(_TOY_ACTIONS))
    converged_logits = torch.zeros(len(examples), len(_TOY_ACTIONS))
    for row, example in enumerate(examples):
        lazy_logits[row, _TOY_STAY_FLAT] = margin  # 懒惰解：永远押 stay_flat
        converged_logits[row, example["gold_index"]] = margin  # 收敛解：押 gold
    lambda_value = 0.25
    lazy_loss = float(question_weighted_loss_sum(
        lazy_logits, examples, direction_penalty_lambda=lambda_value
    ))
    converged_loss = float(question_weighted_loss_sum(
        converged_logits, examples, direction_penalty_lambda=lambda_value
    ))
    assert math.isfinite(lazy_loss) and lazy_loss > 0
    assert math.isfinite(converged_loss) and converged_loss > 0
    # ① 懒惰解显著更差（本构造实测比值 ≈ 3×10³；门禁 ≥ 10× 保守充分）
    assert lazy_loss > 10.0 * converged_loss
    # ② 等权对照：把权重全部抹成 1.0 后懒惰解的代价更小——再加权本身
    #    （而非题目数量）抬升了摆烂解成本，这正是效用权重的动机命题
    equal_weight = [{**example, "weight": 1.0} for example in examples]
    lazy_loss_equal = float(question_weighted_loss_sum(
        lazy_logits, equal_weight, direction_penalty_lambda=lambda_value
    ))
    assert lazy_loss > lazy_loss_equal

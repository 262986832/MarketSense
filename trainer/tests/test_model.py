"""T4 决策模型单测（torch）：state_dict 键同构 / 初始化 / padding 掩码 / 前向形状。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from trainer.model import DecisionModel, build_tiny_backbone, from_pretrained_kwargs  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model() -> DecisionModel:
    torch.manual_seed(3)
    return DecisionModel(build_tiny_backbone(), "attention")


def _example(question_id: str, paths: list[list[int]], gold: int, **extra) -> dict:
    base = {
        "id": question_id,
        "type": "choice",
        "candidate_ids": [f"c{i}" for i in range(len(paths))],
        "leaf_tokens": paths,
        "gold_index": gold,
    }
    base.update(extra)
    return base


def test_state_dict_keys_isomorphic_to_nanojev(tiny_model: DecisionModel) -> None:
    """非 backbone 键与 NanoJev DecisionModel 逐键一致（同构来源 L86-141）。"""
    keys = {name for name, _ in tiny_model.named_parameters()}
    expected = {
        "norm.weight",
        "norm.bias",
        "scalar.weight",
        "scalar.bias",
        "set_project.weight",
        "set_project.bias",
        "set_attention.in_proj_weight",
        "set_attention.in_proj_bias",
        "set_attention.out_proj.weight",
        "set_attention.out_proj.bias",
        "set_output.weight",
        "set_output.bias",
    }
    assert expected <= keys
    assert all(key == "backbone" or key.startswith("backbone.") or key in expected for key in keys)


def test_output_head_zero_initialized(tiny_model: DecisionModel) -> None:
    """只有最终残差投影从零开始；上游层非零。"""
    assert torch.count_nonzero(tiny_model.set_output.weight) == 0
    assert torch.count_nonzero(tiny_model.set_output.bias) == 0
    assert torch.count_nonzero(tiny_model.set_project.weight) > 0
    assert torch.count_nonzero(tiny_model.scalar.weight) > 0


def test_forward_output_shape_and_masking(tiny_model: DecisionModel) -> None:
    logits, mask = tiny_model(
        [_example("a", [[1, 2], [3, 4], [5, 6]], 0)],
        pad_token=0,
    )
    assert logits.shape == (1, 3)
    assert mask.shape == (1, 3)
    assert bool(mask.all())
    # 全组等大小：无 padding，掩码内值均有限
    assert bool(torch.isfinite(logits).all())


def test_forward_masks_padded_candidates(tiny_model: DecisionModel) -> None:
    """同批不同候选数时 padding 候选必须被 -1e9 掩码。"""
    logits, mask = tiny_model(
        [
            _example("a", [[1, 2], [3, 4], [5, 6]], 0),
            _example("b", [[7, 8], [9, 10]], 1),
        ],
        pad_token=0,
    )
    assert logits.shape == (2, 3)
    assert bool(mask[0].all()) and bool(mask[1, :2].all()) and not bool(mask[1, 2])
    assert bool(torch.isfinite(logits[1, :2].max()))
    assert float(logits[1, 2]) == -1e9


def test_forward_is_deterministic(tiny_model: DecisionModel) -> None:
    tiny_model.eval()
    examples = [_example("a", [[1, 2], [3, 4], [5, 6]], 0)]
    with torch.inference_mode():
        first, _ = tiny_model(examples, pad_token=0)
        second, _ = tiny_model(examples, pad_token=0)
    assert torch.equal(first, second)


def test_invalid_set_head_rejected() -> None:
    with pytest.raises(ValueError, match="set_head"):
        DecisionModel(build_tiny_backbone(), "mlp")


def test_tiny_backbone_is_qwen3_and_small() -> None:
    backbone = build_tiny_backbone()
    assert backbone.config.model_type == "qwen3"
    assert backbone.config.hidden_size == 64
    assert sum(p.numel() for p in backbone.parameters()) < 2_000_000


def test_from_pretrained_kwargs_matches_transformers_version() -> None:
    import transformers

    kwargs = from_pretrained_kwargs(torch.float32)
    major = int(transformers.__version__.split(".", 1)[0])
    assert kwargs == {("dtype" if major >= 5 else "torch_dtype"): torch.float32}

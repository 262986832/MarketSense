"""T5 权重纯函数单测（torch 无关）：边界值 / 中性 / 配置校验。"""

from __future__ import annotations

import math

import pytest

from trainer.weights import UtilityWeightConfig, utility_weight


def test_weight_boundary_values() -> None:
    config = UtilityWeightConfig()
    assert utility_weight(-1.0, config) == pytest.approx(0.5)
    assert utility_weight(0.0, config) == pytest.approx(1.0)
    assert utility_weight(3.0, config) == pytest.approx(2.5)
    assert utility_weight(10.0, config) == pytest.approx(2.5)
    assert utility_weight(-100.0, config) == pytest.approx(0.5)


def test_weight_without_outcome_is_neutral() -> None:
    assert utility_weight(None) == 1.0


def test_weight_defaults_satisfy_design_shape() -> None:
    config = UtilityWeightConfig()
    assert config.utility_alpha == 0.5
    assert config.r_cap == 3.0
    assert config.weight_min == 0.5
    assert config.weight_max == 2.5
    assert config.direction_penalty_lambda == 0.25


def test_weight_custom_config_shape() -> None:
    config = UtilityWeightConfig(
        utility_alpha=1.0, r_cap=2.0, weight_min=0.2, weight_max=3.0
    )
    assert utility_weight(-1.0, config) == pytest.approx(0.2)
    assert utility_weight(0.5, config) == pytest.approx(1.5)
    assert utility_weight(2.0, config) == pytest.approx(3.0)


def test_weight_is_monotone_in_r() -> None:
    config = UtilityWeightConfig()
    previous = utility_weight(-1.0, config)
    for step in range(1, 40):
        current = utility_weight(-1.0 + step * 0.2, config)
        assert current >= previous
        previous = current


def test_weight_rejects_nonfinite_r() -> None:
    with pytest.raises(ValueError, match="有限"):
        utility_weight(math.nan)
    with pytest.raises(ValueError, match="有限"):
        utility_weight(math.inf)


@pytest.mark.parametrize(
    "kwargs, pattern",
    [
        ({"utility_alpha": -0.1}, "utility_alpha"),
        ({"r_cap": 0.0}, "r_cap"),
        ({"r_cap": -1.0}, "r_cap"),
        ({"weight_min": 0.0}, "weight_min"),
        ({"weight_min": 1.5}, "weight_min"),
        ({"weight_max": 0.8}, "weight_max"),
        ({"weight_min": 2.0, "weight_max": 1.0}, "weight_min"),
        ({"direction_penalty_lambda": -0.5}, "direction_penalty_lambda"),
    ],
)
def test_config_rejects_invalid_values(kwargs: dict, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        UtilityWeightConfig(**kwargs)


def test_config_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="有限"):
        UtilityWeightConfig(utility_alpha=math.nan)

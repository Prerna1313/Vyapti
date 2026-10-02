import pytest

from vyapti_simulator.system_b.tsrd.evaluation_statistics import (
    paired_world_bootstrap,
    summarize_training_seeds,
)


def test_paired_bootstrap_resamples_world_differences_deterministically():
    first = paired_world_bootstrap(
        {"w1": 0.8, "w2": 0.4, "w3": 0.6},
        {"w1": 0.5, "w2": 0.5, "w3": 0.2},
        samples=1000,
        seed=7,
    )
    second = paired_world_bootstrap(
        {"w1": 0.8, "w2": 0.4, "w3": 0.6},
        {"w1": 0.5, "w2": 0.5, "w3": 0.2},
        samples=1000,
        seed=7,
    )
    assert first == second
    assert first["mean_difference"] == pytest.approx(0.2)
    assert first["world_count"] == 3
    assert first["bootstrap_method"] == "paired_percentile_world_resampling"


def test_paired_bootstrap_rejects_mismatched_worlds():
    with pytest.raises(ValueError, match="same nonempty world IDs"):
        paired_world_bootstrap({"w1": 0.5}, {"w2": 0.5})


def test_training_seed_variability_is_reported_separately():
    summary = summarize_training_seeds([0.5, 0.7, 0.6], seeds=[11, 22, 33])
    assert summary["mean"] == pytest.approx(0.6)
    assert summary["standard_deviation"] == pytest.approx(0.1)
    assert summary["training_seeds"] == [11, 22, 33]

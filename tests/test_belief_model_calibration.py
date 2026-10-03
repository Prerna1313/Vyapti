from pathlib import Path

import numpy as np

from scripts.training.calibrate_train250_hmm import fit_transition_parameters
from vyapti_simulator.system_b.tsrd.algorithm_interface import algorithm_provenance
from vyapti_simulator.system_b.tsrd.training_setup import resolve_setup


def test_transition_estimator_uses_pooled_adjacent_observed_cells():
    grid = np.zeros((36, 600), dtype=bool)
    grid[0, 1:3] = True  # 0->1, 1->1, 1->0
    result = fit_transition_parameters([grid])
    counts = result["counts"]
    assert counts["01"] == 1
    assert counts["11"] == 1
    assert counts["10"] == 1
    assert counts["00"] == 36 * 599 - 3
    assert 0.0 < result["inactive_to_active_probability"] < 1.0
    assert 0.0 < result["active_to_active_probability"] < 1.0


def test_hmm_policies_resolve_the_same_train250_calibration():
    project = Path(__file__).parents[1]
    algorithms = (
        "ppo_lstm", "discrete_sac", "discrete_sac_belief_periodic", "belief_ucb",
    )
    parameters = []
    for name in algorithms:
        setup = resolve_setup(
            project / "training_setup/environments/train250_composed.json",
            project / f"training_setup/algorithms/{name}.json",
            seed=5, episodes=0, checkpoint_every=1, execution_mode="online_baseline",
        )
        settings = setup["algorithm"]["settings"]
        parameters.append(tuple(settings[key] for key in (
            "prior_active_probability", "inactive_to_active_probability",
            "active_to_active_probability",
        )))
        assert setup["belief_model_calibration"]["source"]["dataset_split"] == "TRAIN only"
        assert "belief_model" in setup["setup_sources"]
        assert algorithm_provenance(setup)["belief_model_calibration"]["calibration_id"] == (
            "train250_observed_band_occupancy_50ms_v1"
        )
    assert len(set(parameters)) == 1
    assert parameters[0] == (
        0.3426563254340138, 0.022195484398616423, 0.9575318615765981,
    )

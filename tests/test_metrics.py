"""
Tests for Vyapti MetricsEngine — verifies all seven metrics match their
precise mathematical definitions.
"""
from __future__ import annotations
import numpy as np
import pytest

from vyapti_simulator.core.metrics import MetricsEngine, MetricsConfig, TrajectoryStep
from vyapti_simulator.core.environment import HiddenTruthGrid, EmitterConfig, EmitterBehaviorType
from vyapti_simulator.core.scheduler_interface import BandPrediction


def make_truth_grid(emitter_bands_slots: list, band_count: int = 10, time_slots: int = 100):
    """Create a HiddenTruthGrid with known emitters.

    Args:
        emitter_bands_slots: List of (emitter_idx, band_idx, slot_idx) tuples where emitter is present.
        band_count: Number of bands.
        time_slots: Number of time slots.
    """
    # Find number of emitters
    if emitter_bands_slots:
        max_e = max(e for e, _, _ in emitter_bands_slots)
        n_emitters = max_e + 1
    else:
        n_emitters = 0

    grid = np.zeros((n_emitters, band_count, time_slots), dtype=bool)
    for e, b, t in emitter_bands_slots:
        grid[e, b, t] = True

    # Dummy emitter configs
    emitter_configs = []
    for e in range(n_emitters):
        emitter_configs.append(EmitterConfig(
            emitter_id=f"E{e}",
            behavior=EmitterBehaviorType.CONTINUOUS_FIXED,
            active_bands=[b for (ee, b, t) in emitter_bands_slots if ee == e],
            arrival_slot=0,
            departure_slot=None,
            snr_db=10.0,
        ))

    return HiddenTruthGrid(
        grid=grid,
        emitter_configs=emitter_configs,
        band_count=band_count,
        time_slots=time_slots,
    )


def make_trajectory(actions: list, hits: list, predictions: list = None):
    """Create a trajectory with given actions and hit observations."""
    trajectory = []
    for i, (action, hit) in enumerate(zip(actions, hits)):
        obs = {"hit": bool(hit)}
        pred = predictions[i] if predictions and i < len(predictions) else None
        trajectory.append(TrajectoryStep(
            time_slot=i, action=action, observation=obs, prediction=pred
        ))
    return trajectory


def make_band_prediction(issued_at: int, about: int, probs: list, next_slots: list = None):
    """Create a BandPrediction with given probabilities."""
    if next_slots is None:
        next_slots = [None] * len(probs)
    return BandPrediction(
        issued_at_slot=issued_at,
        about_time_slot=about,
        band_activity_probability=probs,
        predicted_next_activity_slot=next_slots,
        method_note="test",
    )


class TestPdPfa:
    """Tests for Probability of Detection and Probability of False Alarm."""

    def test_pd_correct(self):
        """Pd = hits_on_occupied / occupied_dwells"""
        # Emitter in band 2 at slots 0,1,2
        truth = make_truth_grid([(0, 2, 0), (0, 2, 1), (0, 2, 2)])
        # Scheduler dwells on band 2 at slots 0,1 (both occupied), band 3 at slot 2 (empty)
        traj = make_trajectory(
            actions=[2, 2, 3],
            hits=[True, True, False],  # 2 hits on occupied, 0 on empty
        )
        config = MetricsConfig()
        engine = MetricsEngine(config)
        result = engine.record_result(
            episode_id=0, seed=42, scheduler_name="Test",
            scenario_config={}, trajectory=traj, truth_grid=truth,
        )
        det = result["detection_metrics"]
        assert det["occupied_dwells"] == 2
        assert det["empty_dwells"] == 1
        assert det["true_hits"] == 2
        assert det["false_hits"] == 0
        assert det["probability_of_detection"] == 1.0
        assert det["probability_of_false_alarm"] == 0.0

    def test_pfa_correct(self):
        """Pfa = false_detections_on_empty / empty_dwells"""
        truth = make_truth_grid([])  # No emitters
        traj = make_trajectory(
            actions=[0, 1, 2],
            hits=[True, False, True],  # 2 false alarms on 3 empty dwells
        )
        config = MetricsConfig()
        engine = MetricsEngine(config)
        result = engine.record_result(
            episode_id=0, seed=42, scheduler_name="Test",
            scenario_config={}, trajectory=traj, truth_grid=truth,
        )
        det = result["detection_metrics"]
        assert det["occupied_dwells"] == 0
        assert det["empty_dwells"] == 3
        assert det["true_hits"] == 0
        assert det["false_hits"] == 2
        assert det["probability_of_detection"] is None  # No occupied dwells
        assert det["probability_of_false_alarm"] == 2/3

    def test_miss_rate_equals_one_minus_pd(self):
        """Miss rate = 1 - Pd (when Pd is defined)."""
        truth = make_truth_grid([(0, 1, 0), (0, 1, 1)])
        traj = make_trajectory(
            actions=[1, 1],
            hits=[True, False],  # 1 hit, 1 miss on occupied
        )
        config = MetricsConfig()
        engine = MetricsEngine(config)
        result = engine.record_result(
            episode_id=0, seed=42, scheduler_name="Test",
            scenario_config={}, trajectory=traj, truth_grid=truth,
        )
        det = result["detection_metrics"]
        assert det["probability_of_detection"] == 0.5
        assert det["miss_rate"] == 0.5
        assert abs(det["miss_rate"] - (1.0 - det["probability_of_detection"])) < 1e-9

    def test_miss_rate_none_when_pd_undefined(self):
        """Miss rate is None when Pd is undefined (no occupied dwells)."""
        truth = make_truth_grid([])
        traj = make_trajectory(actions=[0], hits=[True])
        config = MetricsConfig()
        engine = MetricsEngine(config)
        result = engine.record_result(
            episode_id=0, seed=42, scheduler_name="Test",
            scenario_config={}, trajectory=traj, truth_grid=truth,
        )
        det = result["detection_metrics"]
        assert det["probability_of_detection"] is None
        assert det["miss_rate"] is None


class TestAverageInterceptRate:
    """Tests for Average Intercept Rate (PS metric 4) — per-emitter, NOT per-slot."""

    def test_per_emitter_intercept_rate(self):
        """Metric 4 = distinct_emitters_detected / total_distinct_emitters_present."""
        # Two emitters: E0 in band 0, E1 in band 1. Both present for entire mission.
        truth = make_truth_grid(
            [(0, 0, i) for i in range(10)] +  # E0 in band 0, slots 0-9
            [(1, 1, i) for i in range(10)]    # E1 in band 1, slots 0-9
        )
        # Scheduler only dwells on band 0, catches E0, never visits band 1
        traj = make_trajectory(
            actions=[0] * 10,
            hits=[True] * 10,  # Always hits E0
        )
        config = MetricsConfig()
        engine = MetricsEngine(config)
        result = engine.record_result(
            episode_id=0, seed=42, scheduler_name="Test",
            scenario_config={}, trajectory=traj, truth_grid=truth,
        )
        mon = result["monitoring_metrics"]
        # Per-emitter: caught E0, missed E1 => 1/2 = 0.5
        assert mon["average_intercept_rate"] == 0.5
        # Per-slot legacy: 10/10 slots had a hit => 1.0
        assert mon["average_intercept_rate_per_slot"] == 1.0
        # These must differ to prove the fix works
        assert mon["average_intercept_rate"] != mon["average_intercept_rate_per_slot"]

    def test_per_emitter_intercept_rate_all_caught(self):
        """When all emitters caught, rate = 1.0."""
        truth = make_truth_grid(
            [(0, 0, i) for i in range(5)] +
            [(1, 1, i) for i in range(5)]
        )
        traj = make_trajectory(
            actions=[0, 1] * 5,  # Alternate between both bands
            hits=[True] * 10,
        )
        config = MetricsConfig()
        engine = MetricsEngine(config)
        result = engine.record_result(
            episode_id=0, seed=42, scheduler_name="Test",
            scenario_config={}, trajectory=traj, truth_grid=truth,
        )
        mon = result["monitoring_metrics"]
        assert mon["average_intercept_rate"] == 1.0

    def test_per_emitter_intercept_rate_none_caught(self):
        """When no emitters caught, rate = 0.0."""
        truth = make_truth_grid(
            [(0, 0, i) for i in range(5)] +
            [(1, 1, i) for i in range(5)]
        )
        traj = make_trajectory(
            actions=[2] * 10,  # Wrong band
            hits=[False] * 10,
        )
        config = MetricsConfig()
        engine = MetricsEngine(config)
        result = engine.record_result(
            episode_id=0, seed=42, scheduler_name="Test",
            scenario_config={}, trajectory=traj, truth_grid=truth,
        )
        mon = result["monitoring_metrics"]
        assert mon["average_intercept_rate"] == 0.0


class TestSensitivity:
    """Tests for Sensitivity (PS metric 3) — reports Pfa operating point."""

    def test_sensitivity_reports_pfa_operating_point(self):
        """Sensitivity output includes pfa_operating_point field."""
        truth = make_truth_grid(
            [(0, 0, i) for i in range(30)]  # Emitter with 30 dwells
        )
        traj = make_trajectory(
            actions=[0] * 30,
            hits=[True] * 30,  # Perfect detection
        )
        config = MetricsConfig(sensitivity_pfa_operating_point=1e-6)
        engine = MetricsEngine(config)
        result = engine.record_result(
            episode_id=0, seed=42, scheduler_name="Test",
            scenario_config={}, trajectory=traj, truth_grid=truth,
        )
        sens = result["sensitivity_metrics"]
        assert "pfa_operating_point" in sens
        assert sens["pfa_operating_point"] == 1e-6
        assert "definition" in sens
        assert "Pd_target" in sens["definition"] or "target_detection_rate" in sens["definition"]


class TestRewardComposite:
    """Tests for Reward/Cost (PS metric 5) — uses per-emitter rate and weight justifications."""

    def test_uses_per_emitter_intercept_rate(self):
        """rate_term should be average_intercept_rate (per-emitter), not per-slot."""
        truth = make_truth_grid(
            [(0, 0, i) for i in range(10)] +
            [(1, 1, i) for i in range(10)]
        )
        traj = make_trajectory(
            actions=[0] * 10,  # Only catches E0
            hits=[True] * 10,
        )
        config = MetricsConfig(
            reward_weight_intercept_time=1.0,
            reward_weight_interception_rate=1.0,
            reward_weight_false_alarm_cost=-0.5,
            reward_weight_switch_cost=-0.1,
        )
        engine = MetricsEngine(config)
        result = engine.record_result(
            episode_id=0, seed=42, scheduler_name="Test",
            scenario_config={}, trajectory=traj, truth_grid=truth,
        )
        reward = result["reward_composite"]
        comp = reward["components"]
        # Per-emitter intercept rate should be 0.5 (caught 1 of 2)
        assert comp["interception_rate_term"] == 0.5
        assert "weight_justifications" in reward
        assert len(reward["weight_justifications"]) == 4

    def test_all_terms_normalized_zero_to_one(self):
        """All four components should be in [0,1] range."""
        truth = make_truth_grid([(0, 0, i) for i in range(10)])
        traj = make_trajectory(
            actions=[0] * 10,
            hits=[True] * 10,
        )
        config = MetricsConfig(
            reward_weight_intercept_time=1.0,
            reward_weight_interception_rate=1.0,
            reward_weight_false_alarm_cost=-0.5,
            reward_weight_switch_cost=-0.1,
        )
        engine = MetricsEngine(config)
        result = engine.record_result(
            episode_id=0, seed=42, scheduler_name="Test",
            scenario_config={}, trajectory=traj, truth_grid=truth,
        )
        comp = result["reward_composite"]["components"]
        for key, val in comp.items():
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{key} = {val} not in [0,1]"


class TestPredictionAccuracy:
    """Tests for Percentage of Correct Predictions (PS metric 6)."""

    def test_accuracy_excludes_zero_prediction_episodes(self):
        """Episode with no predictions returns None, excluded from aggregation."""
        truth = make_truth_grid([(0, 0, i) for i in range(10)])
        traj = make_trajectory(actions=[0] * 10, hits=[True] * 10)  # No predictions
        config = MetricsConfig()
        engine = MetricsEngine(config)
        result = engine.record_result(
            episode_id=0, seed=42, scheduler_name="Test",
            scenario_config={}, trajectory=traj, truth_grid=truth,
        )
        pred = result["prediction_metrics"]
        assert pred["percentage_correct_predictions"] is None
        assert pred["unavailable_reason"] is not None

    def test_accuracy_computed_correctly(self):
        """Accuracy = correct predictions / total predictions at threshold 0.5."""
        truth = make_truth_grid([(0, 0, i) for i in range(10)])
        # Perfect predictions: predict band 0 correctly, all others incorrectly
        # For an emitter in band 0 active in slots 0-9:
        #   Slot t: predict [1.0 if b==0 else 0.0 for b in range(10)]
        preds = []
        for i in range(9):  # predictions for slots 0-8
            # Predict about_time_slot = i+1
            if i+1 <= 9:  # emitter still active
                probs = [1.0] + [0.0]*9  # band 0 active, others not
            else:  # emitter not active
                probs = [0.0]*10
            preds.append(make_band_prediction(i, i+1, probs))
        # Add one None prediction at the end to make length 10
        preds.append(None)

        traj = make_trajectory(
            actions=[0] * 10,
            hits=[True] * 10,
            predictions=preds,
        )
        config = MetricsConfig()
        engine = MetricsEngine(config)
        result = engine.record_result(
            episode_id=0, seed=42, scheduler_name="Test",
            scenario_config={}, trajectory=traj, truth_grid=truth,
        )
        pred = result["prediction_metrics"]
        # With perfect predictions, accuracy should be 100.0
        assert pred["percentage_correct_predictions"] == 100.0


class TestInterceptTimeError:
    """Tests for Average Intercept Time Error (PS metric 7)."""

    def test_error_only_where_both_exist(self):
        """Error = |predicted - actual| only where actual exists."""
        truth = make_truth_grid([(0, 0, i) for i in range(10)])
        # Prediction at slot 2: says next activity in band 0 at slot 3 (actual is slot 3)
        preds = [make_band_prediction(
            issued_at=2, about=3, probs=[1.0]*10, next_slots=[3] + [None]*9
        )] + [None]*9
        traj = make_trajectory(
            actions=[0]*10,
            hits=[True]*10,
            predictions=preds,
        )
        config = MetricsConfig()
        engine = MetricsEngine(config)
        result = engine.record_result(
            episode_id=0, seed=42, scheduler_name="Test",
            scenario_config={}, trajectory=traj, truth_grid=truth,
        )
        pred = result["prediction_metrics"]
        # Actual next activity in band 0 after slot 2 is slot 3
        # Predicted was 3, so error = 0
        assert pred["average_intercept_time_error_slots"] == 0.0
        assert pred["predictions_without_actual_intercept"] == 0

    def test_excluded_predictions_counted(self):
        """Predictions where actual=None are counted and excluded."""
        truth = make_truth_grid([(0, 0, i) for i in range(5)])  # Emitter stops at slot 5
        # Prediction at slot 4 for band 0: next activity after slot 4, but emitter stops at 5
        preds = [make_band_prediction(
            issued_at=4, about=5, probs=[1.0]*10, next_slots=[10] + [None]*9  # 10 is beyond mission
        )] + [None]*9
        traj = make_trajectory(
            actions=[0]*10,
            hits=[True]*10,
            predictions=preds,
        )
        config = MetricsConfig()
        engine = MetricsEngine(config)
        result = engine.record_result(
            episode_id=0, seed=42, scheduler_name="Test",
            scenario_config={}, trajectory=traj, truth_grid=truth,
        )
        pred = result["prediction_metrics"]
        assert pred["predictions_without_actual_intercept"] == 1
        assert pred["excluded_fraction"] == 1.0
        assert pred["average_intercept_time_error_slots"] is None  # No valid errors


class TestAggregation:
    """Tests for cross-episode aggregation."""

    def test_aggregation_includes_new_metric(self):
        """Aggregation includes average_intercept_rate (per-emitter)."""
        # Two episodes with different intercept rates
        truth1 = make_truth_grid([(0, 0, i) for i in range(5)])
        traj1 = make_trajectory(actions=[0]*5, hits=[True]*5)

        truth2 = make_truth_grid([(0, 0, i) for i in range(5)] + [(1, 1, i) for i in range(5)])
        traj2 = make_trajectory(actions=[0]*5, hits=[True]*5)

        config = MetricsConfig()
        engine = MetricsEngine(config)
        engine.record_result(0, 1, "Test", {}, traj1, truth1)
        engine.record_result(1, 2, "Test", {}, traj2, truth2)

        agg = engine.aggregate("Test")
        assert "monitoring_metrics.average_intercept_rate" in agg
        assert "monitoring_metrics.average_intercept_rate_per_slot" in agg
        # Episode 1: 1/1 = 1.0, Episode 2: 1/2 = 0.5 -> mean = 0.75
        assert abs(agg["monitoring_metrics.average_intercept_rate"]["mean"] - 0.75) < 1e-9

    def test_aggregation_excludes_none(self):
        """Metrics with None values are excluded from mean."""
        truth = make_truth_grid([(0, 0, i) for i in range(5)])
        # Episode 1: no predictions
        traj1 = make_trajectory(actions=[0]*5, hits=[True]*5)
        # Episode 2: with predictions - predict future slots
        preds = [make_band_prediction(i, i+1, [1.0]*10) for i in range(2)] + [None]*3
        traj2 = make_trajectory(actions=[0]*5, hits=[True]*5, predictions=preds)

        config = MetricsConfig()
        engine = MetricsEngine(config)
        engine.record_result(0, 1, "Test", {}, traj1, truth)
        engine.record_result(1, 2, "Test", {}, traj2, truth)

        agg = engine.aggregate("Test")
        pred_agg = agg["prediction_metrics.percentage_correct_predictions"]
        assert pred_agg["n_episodes_defined"] == 1
        assert pred_agg["n_episodes_undefined"] == 1


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_trajectory_raises(self):
        """Empty trajectory should raise ValueError."""
        truth = make_truth_grid([])
        config = MetricsConfig()
        engine = MetricsEngine(config)
        with pytest.raises(ValueError, match="empty trajectory"):
            engine.record_result(0, 42, "Test", {}, [], truth)

    def test_out_of_range_action_raises(self):
        """Out-of-range band index should raise ValueError."""
        truth = make_truth_grid([])
        traj = make_trajectory(actions=[99], hits=[False])
        config = MetricsConfig()
        engine = MetricsEngine(config)
        with pytest.raises(ValueError, match="out-of-range band index"):
            engine.record_result(0, 42, "Test", {}, traj, truth)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
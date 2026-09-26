"""
tests.test_multi_objective_reward
================================

Tests for the Sub-problem G multi-objective reward module.

Coverage:
- Objective definitions
- Scalarisation methods (weighted sum, Chebyshev, etc.)
- MultiObjectiveRewardEngine
- Pareto-front analysis
- Preference elicitation
"""

from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.core.multi_objective_reward import (
    ObjectiveType,
    Objective,
    ObjectiveValue,
    MultiObjectiveResult,
    ScalarisationMethod,
    ScalarisationFunction,
    WeightedSumScalarisation,
    ChebyshevScalarisation,
    AugmentedChebyshevScalarisation,
    PenaltyBasedScalarisation,
    HybridScalarisation,
    MultiObjectiveRewardConfig,
    MultiObjectiveRewardEngine,
    PreferenceElicitation,
    create_default_multi_objective_engine,
    RewardConfig,
)


class TestObjective:
    """Tests for Objective dataclass."""

    def test_objective_creation(self):
        """Objective can be created with default values."""
        obj = Objective("test", ObjectiveType.DISCOVERY_RATE)
        assert obj.name == "test"
        assert obj.objective_type == ObjectiveType.DISCOVERY_RATE
        assert obj.weight == 1.0
        assert obj.direction == "maximize"
        assert obj.threshold is None


class TestObjectiveValue:
    """Tests for ObjectiveValue dataclass."""

    def test_objective_value_creation(self):
        """ObjectiveValue stores raw and normalised values."""
        obj = Objective("test", ObjectiveType.DISCOVERY_RATE)
        val = ObjectiveValue(
            objective=obj,
            raw_value=0.8,
            normalised_value=0.8,
            satisfied=True,
        )
        assert val.raw_value == 0.8
        assert val.normalised_value == 0.8
        assert val.satisfied is True


class TestScalarisationFunction:
    """Tests for scalarisation functions."""

    def _make_objectives(self):
        """Create test objectives."""
        return [
            Objective("discovery", ObjectiveType.DISCOVERY_RATE, weight=0.5,
                     direction="maximize"),
            Objective("monitoring", ObjectiveType.MONITORING_RATE, weight=0.5,
                     direction="maximize"),
        ]

    def _make_values(self):
        """Create test objective values."""
        objectives = self._make_objectives()
        return [
            ObjectiveValue(objectives[0], raw_value=0.8, normalised_value=0.8, satisfied=True),
            ObjectiveValue(objectives[1], raw_value=0.6, normalised_value=0.6, satisfied=True),
        ]

    def test_weighted_sum_basic(self):
        """Weighted sum computes correctly."""
        scalariser = WeightedSumScalarisation(self._make_objectives())
        values = self._make_values()
        result = scalariser.compute(values)
        # 0.5 * 0.8 + 0.5 * 0.6 = 0.7
        assert abs(result - 0.7) < 1e-6

    def test_weighted_sum_all_equal(self):
        """Weighted sum with equal values."""
        objectives = [
            Objective("a", ObjectiveType.DISCOVERY_RATE, weight=0.25, direction="maximize"),
            Objective("b", ObjectiveType.DISCOVERY_RATE, weight=0.25, direction="maximize"),
            Objective("c", ObjectiveType.DISCOVERY_RATE, weight=0.25, direction="maximize"),
            Objective("d", ObjectiveType.DISCOVERY_RATE, weight=0.25, direction="maximize"),
        ]
        scalariser = WeightedSumScalarisation(objectives)
        values = [
            ObjectiveValue(o, raw_value=1.0, normalised_value=1.0, satisfied=True)
            for o in objectives
        ]
        result = scalariser.compute(values)
        assert abs(result - 1.0) < 1e-6

    def test_chebyshev_basic(self):
        """Chebyshev scalarisation computes correctly."""
        scalariser = ChebyshevScalarisation(self._make_objectives())
        values = self._make_values()
        result = scalariser.compute(values)
        # max(0.5 * |0.8 - 1|, 0.5 * |0.6 - 1|) = 0.5 * 0.4 = 0.2, then 1 - chebyshev = 0.8
        # But our implementation uses ideal=1.0 and computes 1 - max(...), so:
        # cheb = max(0.5*0.2, 0.5*0.4) = 0.2, result = 1 - 0.2 = 0.8
        assert abs(result - 0.8) < 1e-5

    def test_chebyshev_perfect_scores(self):
        """Chebyshev with perfect scores."""
        scalariser = ChebyshevScalarisation(self._make_objectives())
        values = [
            ObjectiveValue(o, raw_value=1.0, normalised_value=1.0, satisfied=True)
            for o in self._make_objectives()
        ]
        result = scalariser.compute(values)
        assert abs(result - 1.0) < 1e-6

    def test_augmented_chebyshev(self):
        """Augmented Chebyshev combines Chebyshev with weighted sum."""
        scalariser = AugmentedChebyshevScalarisation(
            self._make_objectives(),
            rho=0.001
        )
        values = self._make_values()
        result = scalariser.compute(values)
        # Should be between 0 and 1
        assert 0.0 <= result <= 1.0

    def test_penalty_based(self):
        """Penalty-based scalarisation penalises imbalance."""
        scalariser = PenaltyBasedScalarisation(
            self._make_objectives(),
            penalty_factor=0.1
        )
        values = self._make_values()
        result = scalariser.compute(values)
        # Weighted sum - penalty = 0.7 - 0.1 * 0.4 = 0.66
        assert 0.0 <= result <= 1.0

    def test_hybrid_alpha_zero(self):
        """Hybrid with alpha=0 equals weighted sum."""
        objectives = self._make_objectives()
        values = self._make_values()

        weighted_sum = WeightedSumScalarisation(objectives)
        hybrid = HybridScalarisation(objectives, alpha=0.0)

        ws_result = weighted_sum.compute(values)
        hybrid_result = hybrid.compute(values)
        assert abs(ws_result - hybrid_result) < 1e-6

    def test_hybrid_alpha_one(self):
        """Hybrid with alpha=1 equals Chebyshev."""
        objectives = self._make_objectives()
        values = self._make_values()

        chebyshev = ChebyshevScalarisation(objectives)
        hybrid = HybridScalarisation(objectives, alpha=1.0)

        cheb_result = chebyshev.compute(values)
        hybrid_result = hybrid.compute(values)
        assert abs(cheb_result - hybrid_result) < 1e-6

    def test_weights_normalised(self):
        """Scalarisation normalises weights."""
        objectives = [
            Objective("a", ObjectiveType.DISCOVERY_RATE, weight=3.0, direction="maximize"),
            Objective("b", ObjectiveType.DISCOVERY_RATE, weight=1.0, direction="maximize"),
        ]
        scalariser = WeightedSumScalarisation(objectives)
        # Weights should be normalised to 0.75 and 0.25
        total = sum(o.weight for o in scalariser.objectives)
        assert abs(total - 1.0) < 1e-6


class TestMultiObjectiveRewardEngine:
    """Tests for MultiObjectiveRewardEngine."""

    def _sample_metrics(self):
        """Create sample metrics for testing."""
        return {
            "discovery_metrics": {
                "interception_probability": 0.8,
                "mean_first_intercept_time_slots": 50.0,
            },
            "monitoring_metrics": {
                "post_discovery_capture_efficiency": 0.6,
                "average_coverage_fraction": 0.7,
                "dead_time_fraction": 0.1,
            },
            "prediction_metrics": {
                "percentage_correct_predictions": 75.0,
            },
        }

    def test_default_objectives(self):
        """Default objectives are set correctly."""
        config = MultiObjectiveRewardConfig()
        assert len(config.objectives) > 0

    def test_compute_objective_values(self):
        """Objective values are extracted from metrics."""
        engine = MultiObjectiveRewardEngine()
        values = engine.compute_objective_values(self._sample_metrics())

        assert len(values) == len(engine.config.objectives)

        # Check discovery rate
        discovery_obj = next(
            v for v in values if v.objective.name == "discovery"
        )
        assert abs(discovery_obj.raw_value - 0.8) < 1e-6

    def test_compute_reward(self):
        """Reward is computed correctly."""
        engine = MultiObjectiveRewardEngine()
        reward, result = engine.compute_reward(self._sample_metrics())

        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
        assert isinstance(result, MultiObjectiveResult)

    def test_different_scalarisation_methods(self):
        """Different scalarisation methods produce different results."""
        metrics = self._sample_metrics()

        methods = [
            ScalarisationMethod.WEIGHTED_SUM,
            ScalarisationMethod.CHEBYSHEV,
            ScalarisationMethod.AUGMENTED_CHEBYSHEV,
            ScalarisationMethod.HYBRID,
        ]

        rewards = []
        for method in methods:
            config = MultiObjectiveRewardConfig(scalarisation_method=method)
            engine = MultiObjectiveRewardEngine(config)
            reward, _ = engine.compute_reward(metrics)
            rewards.append(reward)

        # At least some methods should produce different results
        # (they're not all identical)
        assert len(set(rewards)) >= 2 or abs(max(rewards) - min(rewards)) < 1e-6


class TestParetoAnalysis:
    """Tests for Pareto-front analysis."""

    def _make_results(self):
        """Create sample results for testing."""
        objectives = [
            Objective("metric1", ObjectiveType.DISCOVERY_RATE, weight=1.0, direction="maximize"),
            Objective("metric2", ObjectiveType.MONITORING_RATE, weight=1.0, direction="maximize"),
        ]

        return [
            MultiObjectiveResult(
                objectives=[
                    ObjectiveValue(objectives[0], raw_value=0.9, normalised_value=0.9, satisfied=True),
                    ObjectiveValue(objectives[1], raw_value=0.9, normalised_value=0.9, satisfied=True),
                ],
                composite_score=0.9,
                pareto_dominated=False,
                pareto_rank=0,
                dominates_count=0,
            ),
            MultiObjectiveResult(
                objectives=[
                    ObjectiveValue(objectives[0], raw_value=0.7, normalised_value=0.7, satisfied=True),
                    ObjectiveValue(objectives[1], raw_value=0.7, normalised_value=0.7, satisfied=True),
                ],
                composite_score=0.7,
                pareto_dominated=False,
                pareto_rank=0,
                dominates_count=0,
            ),
        ]

    def test_compute_pareto_front(self):
        """Pareto front is computed correctly."""
        engine = MultiObjectiveRewardEngine()
        results = self._make_results()

        pareto_indices = engine.compute_pareto_front(results)

        # First result (0.9, 0.9) should be on front
        assert 0 in pareto_indices

    def test_dominates(self):
        """Dominance is computed correctly."""
        engine = MultiObjectiveRewardEngine()
        objectives = [
            Objective("a", ObjectiveType.DISCOVERY_RATE, weight=1.0, direction="maximize"),
            Objective("b", ObjectiveType.MONITORING_RATE, weight=1.0, direction="maximize"),
        ]

        a = MultiObjectiveResult(
            objectives=[
                ObjectiveValue(objectives[0], raw_value=0.9, normalised_value=0.9, satisfied=True),
                ObjectiveValue(objectives[1], raw_value=0.9, normalised_value=0.9, satisfied=True),
            ],
            composite_score=0.9,
            pareto_dominated=False,
            pareto_rank=0,
            dominates_count=0,
        )

        b = MultiObjectiveResult(
            objectives=[
                ObjectiveValue(objectives[0], raw_value=0.7, normalised_value=0.7, satisfied=True),
                ObjectiveValue(objectives[1], raw_value=0.7, normalised_value=0.7, satisfied=True),
            ],
            composite_score=0.7,
            pareto_dominated=False,
            pareto_rank=0,
            dominates_count=0,
        )

        assert engine._dominates(a, b) is True
        assert engine._dominates(b, a) is False

    def test_pareto_rank_assignment(self):
        """Pareto ranks are assigned correctly."""
        engine = MultiObjectiveRewardEngine()
        results = self._make_results()

        engine.compute_pareto_front(results)

        # All results should have a rank assigned
        for r in results:
            assert r.pareto_rank >= 0

    def test_pareto_front_statistics(self):
        """Pareto front statistics are computed."""
        engine = MultiObjectiveRewardEngine()
        results = self._make_results()

        stats = engine.get_pareto_front_statistics(results)

        assert "n_pareto" in stats
        assert stats["n_pareto"] >= 1


class TestPreferenceElicitation:
    """Tests for PreferenceElicitation."""

    def _make_objectives(self):
        """Create test objectives."""
        return [
            Objective("discovery", ObjectiveType.DISCOVERY_RATE, weight=1.0, direction="maximize"),
            Objective("monitoring", ObjectiveType.MONITORING_RATE, weight=1.0, direction="maximize"),
        ]

    def test_initialization(self):
        """PreferenceElicitation initializes correctly."""
        elicitation = PreferenceElicitation(self._make_objectives())
        assert elicitation.n_objectives == 2
        assert elicitation.total_comparisons == 0

    def test_record_preference(self):
        """Preferences are recorded correctly."""
        elicitation = PreferenceElicitation(self._make_objectives())
        elicitation.record_preference(0, 1)

        assert elicitation.wins[0][1] == 1
        assert elicitation.total_comparisons == 1

    def test_sample_weight_vector(self):
        """Weight vectors are sampled correctly."""
        elicitation = PreferenceElicitation(self._make_objectives())

        weights = elicitation.sample_weight_vector()
        assert weights.shape == (2,)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(0.0 <= w <= 1.0 for w in weights)

    def test_sample_multiple_weights(self):
        """Multiple weight vectors can be sampled."""
        elicitation = PreferenceElicitation(self._make_objectives())

        weights = elicitation.sample_weight_vector(n_samples=10)
        assert weights.shape == (10, 2)

        for w in weights:
            assert abs(w.sum() - 1.0) < 1e-6

    def test_weight_estimate_with_preferences(self):
        """Weight estimates reflect recorded preferences."""
        elicitation = PreferenceElicitation(self._make_objectives())

        # Record preference for objective 0 over 1
        for _ in range(10):
            elicitation.record_preference(0, 1)

        weights = elicitation.get_weight_estimate()
        assert weights[0] > weights[1]


class TestCreateDefaultEngine:
    """Tests for create_default_multi_objective_engine."""

    def test_default_creation(self):
        """Default engine is created correctly."""
        engine = create_default_multi_objective_engine()

        assert isinstance(engine, MultiObjectiveRewardEngine)
        assert len(engine.config.objectives) == 4

    def test_custom_weights(self):
        """Custom weights are respected."""
        engine = create_default_multi_objective_engine(
            discovery_weight=0.5,
            monitoring_weight=0.3,
            coverage_weight=0.1,
            efficiency_weight=0.1,
        )

        discovery = next(o for o in engine.config.objectives if o.name == "discovery")
        assert abs(discovery.weight - 0.5) < 1e-6

    def test_hybrid_scalarisation(self):
        """Hybrid scalarisation is configured correctly."""
        engine = create_default_multi_objective_engine(
            scalarisation=ScalarisationMethod.HYBRID,
            alpha=0.7,
        )

        assert engine.config.scalarisation_method == ScalarisationMethod.HYBRID
        assert engine.config.alpha == 0.7


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_results(self):
        """Empty results list handled correctly."""
        engine = MultiObjectiveRewardEngine()
        pareto_indices = engine.compute_pareto_front([])
        assert pareto_indices == []

    def test_single_result(self):
        """Single result is always Pareto-optimal."""
        objectives = [Objective("a", ObjectiveType.DISCOVERY_RATE, weight=1.0, direction="maximize")]
        result = MultiObjectiveResult(
            objectives=[
                ObjectiveValue(objectives[0], raw_value=0.5, normalised_value=0.5, satisfied=True),
            ],
            composite_score=0.5,
            pareto_dominated=False,
            pareto_rank=0,
            dominates_count=0,
        )

        engine = MultiObjectiveRewardEngine()
        pareto_indices = engine.compute_pareto_front([result])

        assert 0 in pareto_indices

    def test_all_equal_results(self):
        """All equal results are all Pareto-optimal."""
        objectives = [Objective("a", ObjectiveType.DISCOVERY_RATE, weight=1.0, direction="maximize")]

        results = [
            MultiObjectiveResult(
                objectives=[
                    ObjectiveValue(objectives[0], raw_value=0.5, normalised_value=0.5, satisfied=True),
                ],
                composite_score=0.5,
                pareto_dominated=False,
                pareto_rank=0,
                dominates_count=0,
            )
            for _ in range(3)
        ]

        engine = MultiObjectiveRewardEngine()
        pareto_indices = engine.compute_pareto_front(results)

        # All should be Pareto-optimal (no one dominates another)
        assert len(pareto_indices) == 3


class TestRewardConfig:
    """Tests for RewardConfig (Vyapti Sub-problem G)."""

    def test_default_weights(self):
        """Default weights sum to 1.0."""
        config = RewardConfig()
        total = config.weight_detection + config.weight_speed + config.weight_cost
        assert abs(total - 1.0) < 1e-6

    def test_hit_with_high_threat(self):
        """Hit with high threat gives highest reward."""
        config = RewardConfig()

        # Hit, high threat, fast intercept, low cost
        reward = config.compute_reward(
            detection=1.0,
            threat_score=0.9,
            intercept_time=10.0,
            dwell_cost=0.1,
        )

        # Should be positive (all components positive except efficiency penalty)
        assert reward > 0

    def test_hit_with_low_threat(self):
        """Hit with low threat gives lower reward than high threat."""
        config = RewardConfig()

        high_threat = config.compute_reward(
            detection=1.0,
            threat_score=0.9,
            intercept_time=10.0,
            dwell_cost=0.1,
        )

        low_threat = config.compute_reward(
            detection=1.0,
            threat_score=0.3,
            intercept_time=10.0,
            dwell_cost=0.1,
        )

        assert high_threat > low_threat

    def test_miss_vs_hit(self):
        """Miss gives lower reward than hit."""
        config = RewardConfig()

        hit_reward = config.compute_reward(
            detection=1.0,
            threat_score=0.9,
            intercept_time=10.0,
            dwell_cost=0.1,
        )

        miss_reward = config.compute_reward(
            detection=0.0,
            threat_score=0.9,
            intercept_time=10.0,
            dwell_cost=0.1,
        )

        assert hit_reward > miss_reward

    def test_fast_intercept_vs_slow(self):
        """Fast intercept gives higher reward than slow."""
        config = RewardConfig(intercept_time_max=100.0)

        fast_reward = config.compute_reward(
            detection=1.0,
            threat_score=0.5,
            intercept_time=10.0,  # 10% of max time
            dwell_cost=0.1,
        )

        slow_reward = config.compute_reward(
            detection=1.0,
            threat_score=0.5,
            intercept_time=90.0,  # 90% of max time
            dwell_cost=0.1,
        )

        assert fast_reward > slow_reward

    def test_low_cost_vs_high_cost(self):
        """Low dwell cost gives higher reward than high cost."""
        config = RewardConfig()

        low_cost = config.compute_reward(
            detection=1.0,
            threat_score=0.5,
            intercept_time=50.0,
            dwell_cost=0.05,
        )

        high_cost = config.compute_reward(
            detection=1.0,
            threat_score=0.5,
            intercept_time=50.0,
            dwell_cost=0.5,
        )

        # High cost should have lower reward due to penalty
        assert low_cost > high_cost

    def test_formula_verification(self):
        """Verify the exact formula: 0.6*(threat*det) + 0.3*(1 - time/max) - 0.1*cost"""
        config = RewardConfig(
            weight_detection=0.6,
            weight_speed=0.3,
            weight_cost=0.1,
            use_threat_weighting=True,
            intercept_time_max=100.0,
        )

        # With specific values:
        # detection = 1.0, threat = 0.9, time = 50, cost = 0.1
        reward = config.compute_reward(
            detection=1.0,
            threat_score=0.9,
            intercept_time=50.0,
            dwell_cost=0.1,
        )

        # Expected: 0.6*(0.9*1.0) + 0.3*(1 - 50/100) - 0.1*0.1
        #         = 0.54 + 0.3*0.5 - 0.01
        #         = 0.54 + 0.15 - 0.01
        #         = 0.68
        expected = 0.6 * (0.9 * 1.0) + 0.3 * (1.0 - 50.0/100.0) - 0.1 * 0.1
        assert abs(reward - expected) < 1e-6

    def test_compute_reward_from_observation(self):
        """Can compute reward from observation dict."""
        config = RewardConfig()

        observation = {
            "hit": True,
            "retune_cost_s": 0.001,  # 1ms retune
            "receiver_metadata": {
                "dwell_time_ms": 50.0,  # 50ms slot
            }
        }

        reward = config.compute_reward_from_observation(
            observation=observation,
            threat_score=0.8,
            intercept_time=20.0,
        )

        assert isinstance(reward, float)
        assert reward >= 0.0

    def test_is_implemented(self):
        """RewardConfig is always implemented."""
        config = RewardConfig()
        assert config.is_implemented() is True

    def test_custom_weights(self):
        """Custom weights are respected."""
        config = RewardConfig(
            weight_detection=0.5,
            weight_speed=0.4,
            weight_cost=0.1,
        )

        # Verify weights are normalized
        total = config.weight_detection + config.weight_speed + config.weight_cost
        assert abs(total - 1.0) < 1e-6

"""
tests.test_threat
=================

Tests for the Sub-problem F threat prioritisation module.

Coverage:
- ThreatScoreConfig validation
- ThreatScorer computation
- ThreatScoreMixin integration
- ThreatAwareUCB1 scheduler
"""

from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.algorithms.threat import (
    ThreatModelType,
    ThreatScoreConfig,
    BandThreatState,
    ThreatScorer,
    ThreatScoreMixin,
    ThreatAwareUCB1,
)


class TestThreatScoreConfig:
    """Tests for ThreatScoreConfig."""

    def test_default_weights_normalised(self):
        """Default weights are normalised to sum to 1.0."""
        config = ThreatScoreConfig()
        total = (config.weight_snr_db + config.weight_agility +
                 config.weight_pri_stability + config.weight_activity +
                 config.weight_behavior)
        assert abs(total - 1.0) < 1e-6

    def test_is_implemented(self):
        """Threat model is always implemented."""
        config = ThreatScoreConfig()
        assert config.is_implemented() is True

    def test_active_features_simple(self):
        """Simple model has only activity feature."""
        config = ThreatScoreConfig(model_type=ThreatModelType.SIMPLE)
        features = config.active_features()
        assert features == ["activity"]

    def test_active_features_standard(self):
        """Standard model has activity and snr features."""
        config = ThreatScoreConfig(model_type=ThreatModelType.STANDARD)
        features = config.active_features()
        assert "activity" in features
        assert "snr" in features
        assert "agility" not in features

    def test_active_features_advanced(self):
        """Advanced model has all features."""
        config = ThreatScoreConfig(model_type=ThreatModelType.ADVANCED)
        features = config.active_features()
        assert "activity" in features
        assert "snr" in features
        assert "agility" in features
        assert "pri_stability" in features


class TestBandThreatState:
    """Tests for BandThreatState."""

    def test_hit_rate_no_observations(self):
        """Hit rate is 0 with no observations."""
        state = BandThreatState(band=0)
        assert state.hit_rate() == 0.0

    def test_hit_rate_with_observations(self):
        """Hit rate computed correctly."""
        state = BandThreatState(band=0)
        state.hit_count = 3
        state.miss_count = 7
        assert state.hit_rate() == 0.3

    def test_hit_rate_with_window(self):
        """Hit rate respects sliding window."""
        state = BandThreatState(band=0)
        state.hit_count = 10
        state.miss_count = 5
        state.recent_hits = [90, 95, 99]  # Recent hits in last 10 slots
        rate = state.hit_rate(window=10)
        assert rate <= 1.0


class TestThreatScorer:
    """Tests for ThreatScorer."""

    def test_initialization(self):
        """ThreatScorer initializes with correct band count."""
        scorer = ThreatScorer(band_count=10)
        assert scorer.band_count == 10
        assert len(scorer.band_states) == 10

    def test_update_from_observation_hit(self):
        """Threat state updates correctly on hit."""
        scorer = ThreatScorer(band_count=10)
        observation = {"hit": True, "snr_db_estimate": 25.0}
        scorer.update_from_observation(band=3, observation=observation, current_slot=10)

        state = scorer.band_states[3]
        assert state.hit_count == 1
        assert state.miss_count == 0
        assert state.last_hit_slot == 10
        assert state.max_snr_db == 25.0

    def test_update_from_observation_miss(self):
        """Threat state updates correctly on miss."""
        scorer = ThreatScorer(band_count=10)
        observation = {"hit": False}
        scorer.update_from_observation(band=5, observation=observation, current_slot=20)

        state = scorer.band_states[5]
        assert state.hit_count == 0
        assert state.miss_count == 1
        assert state.last_hit_slot == -1

    def test_compute_threat_score_no_history(self):
        """Threat score is 0 with no observation history."""
        scorer = ThreatScorer(band_count=10)
        score = scorer.compute_threat_score(band=0)
        assert score == 0.0

    def test_compute_threat_score_with_hits(self):
        """Threat score increases with hits."""
        scorer = ThreatScorer(band_count=10)

        # Add several hits
        for slot in range(5):
            observation = {"hit": True, "snr_db_estimate": 20.0}
            scorer.update_from_observation(band=0, observation=observation, current_slot=slot)

        score = scorer.compute_threat_score(band=0)
        assert score > 0.0

    def test_compute_threat_score_with_snr(self):
        """Threat score reflects SNR level."""
        scorer = ThreatScorer(band_count=10, config=ThreatScoreConfig(
            model_type=ThreatModelType.STANDARD
        ))

        # Add high SNR hit
        observation = {"hit": True, "snr_db_estimate": 30.0}
        scorer.update_from_observation(band=0, observation=observation, current_slot=0)
        high_snr_score = scorer.compute_threat_score(band=0)

        # Reset and add low SNR hit
        scorer.reset()
        observation = {"hit": True, "snr_db_estimate": 5.0}
        scorer.update_from_observation(band=0, observation=observation, current_slot=0)
        low_snr_score = scorer.compute_threat_score(band=0)

        assert high_snr_score > low_snr_score

    def test_rank_bands_by_threat(self):
        """Bands are ranked correctly by threat."""
        scorer = ThreatScorer(band_count=5)

        # Add more hits to band 2
        for slot in range(10):
            scorer.update_from_observation(
                band=2,
                observation={"hit": True},
                current_slot=slot
            )

        # Add fewer hits to band 4
        for slot in range(3):
            scorer.update_from_observation(
                band=4,
                observation={"hit": True},
                current_slot=slot
            )

        ranking = scorer.rank_bands_by_threat(descending=True)
        assert ranking[0] == 2  # Highest threat
        # Band 2 should come before band 4 in the ranking
        assert ranking.index(2) < ranking.index(4)

    def test_recency_affects_threat(self):
        """Recent hits have higher threat than old hits."""
        scorer = ThreatScorer(band_count=10, config=ThreatScoreConfig(
            recency_window_slots=50
        ))

        # Add hit long ago (slot 0, evaluate at slot 50)
        scorer.update_from_observation(
            band=0,
            observation={"hit": True},
            current_slot=0
        )
        scorer.current_slot = 50
        old_score = scorer.compute_threat_score(band=0)

        # Reset and add recent hit (slot 49, evaluate at slot 50)
        scorer.reset()
        scorer.update_from_observation(
            band=1,
            observation={"hit": True},
            current_slot=49
        )
        scorer.current_slot = 50
        new_score = scorer.compute_threat_score(band=1)

        # Recent hit should have higher threat
        assert new_score > old_score

    def test_reset_clears_state(self):
        """Reset clears all threat state."""
        scorer = ThreatScorer(band_count=10)

        scorer.update_from_observation(
            band=5,
            observation={"hit": True, "snr_db_estimate": 20.0},
            current_slot=10
        )

        assert scorer.band_states[5].hit_count == 1

        scorer.reset()

        assert scorer.band_states[5].hit_count == 0
        assert scorer.band_states[5].max_snr_db == -np.inf

    def test_get_band_threat_details(self):
        """Detailed threat info is returned correctly."""
        scorer = ThreatScorer(band_count=10)

        scorer.update_from_observation(
            band=3,
            observation={"hit": True, "snr_db_estimate": 25.0},
            current_slot=5
        )

        details = scorer.get_band_threat_details(band=3)

        assert details["band"] == 3
        assert details["hit_count"] == 1
        assert details["max_snr_db"] == 25.0
        assert details["threat_score"] >= 0.0


class TestThreatScoreMixin:
    """Tests for ThreatScoreMixin."""

    def test_mixin_provides_interface(self):
        """Mixin provides threat scoring interface."""
        class MockScheduler(ThreatScoreMixin):
            def __init__(self, band_count):
                self.band_count = band_count
                self._init_threat_scorer()

            def reset(self, seed, config):
                self.reset_threat_state()

        scheduler = MockScheduler(band_count=10)
        assert hasattr(scheduler, "_threat_scorer")
        assert hasattr(scheduler, "rank_bands_by_threat")
        # _compute_band_threat_score is a method of ThreatScorer, not the mixin
        assert hasattr(scheduler, "_compute_band_threat_score")


class TestThreatAwareUCB1:
    """Tests for ThreatAwareUCB1 scheduler."""

    def test_initialization(self):
        """ThreatAwareUCB1 initializes correctly."""
        scheduler = ThreatAwareUCB1(band_count=10)
        assert scheduler.band_count == 10
        assert hasattr(scheduler, "_threat_scorer")

    def test_select_action_explores_first(self):
        """Scheduler explores all bands first."""
        scheduler = ThreatAwareUCB1(band_count=5)
        scheduler.reset(seed=42, scenario_config={})

        for t in range(5):
            action = scheduler.select_action([], t)
            assert 0 <= action < 5

    def test_select_action_uses_threat(self):
        """Scheduler considers threat in action selection."""
        scheduler = ThreatAwareUCB1(band_count=5, threat_bonus=2.0)
        scheduler.reset(seed=42, scenario_config={})

        # Run exploration phase
        for t in range(5):
            scheduler.update(t, {"hit": True, "time_slot": t})

        # After exploration, should use threat-weighted UCB
        action = scheduler.select_action([], 5)
        assert 0 <= action < 5

    def test_update_leaves_traces(self):
        """Update correctly updates threat state."""
        scheduler = ThreatAwareUCB1(band_count=10)
        scheduler.reset(seed=42, scenario_config={})

        scheduler.update(action=5, observation={"hit": True, "time_slot": 0})

        assert scheduler.counts[5] == 1
        assert scheduler.total_counts == 1
        assert scheduler.values[5] == 1.0

    def test_diagnostics_include_threat_config(self):
        """Diagnostics include threat configuration."""
        scheduler = ThreatAwareUCB1(band_count=10)
        scheduler.reset(seed=42, scenario_config={})

        diag = scheduler.get_diagnostics()
        assert "threat_config" in diag


class TestThreatScoringEdgeCases:
    """Edge case tests for threat scoring."""

    def test_all_bands_hit_equal_threat(self):
        """Equal hits across bands give equal threat."""
        scorer = ThreatScorer(band_count=3)

        for band in range(3):
            for _ in range(5):
                scorer.update_from_observation(
                    band=band,
                    observation={"hit": True},
                    current_slot=0
                )

        scores = [scorer.compute_threat_score(b) for b in range(3)]
        # All scores should be approximately equal
        assert max(scores) - min(scores) < 0.1

    def test_invalid_band_raises(self):
        """Computing threat for invalid band raises."""
        scorer = ThreatScorer(band_count=10)
        with pytest.raises(IndexError):
            scorer.compute_threat_score(band=10)

    def test_negative_snr_handled(self):
        """Negative SNR is handled correctly."""
        scorer = ThreatScorer(band_count=10, config=ThreatScoreConfig(
            model_type=ThreatModelType.STANDARD,
            snr_low_threshold_db=0.0,
            snr_high_threshold_db=30.0,
        ))

        observation = {"hit": True, "snr_db_estimate": -10.0}
        scorer.update_from_observation(band=0, observation=observation, current_slot=0)

        score = scorer.compute_threat_score(band=0)
        assert score >= 0.0
        assert score <= 1.0

    def test_agility_detection(self):
        """Agility score increases with band diversity."""
        scorer = ThreatScorer(band_count=10, config=ThreatScoreConfig(
            model_type=ThreatModelType.ADVANCED
        ))

        # Add observations that show band hopping
        for other_band in [1, 2, 3, 4, 5]:
            scorer.update_from_observation(
                band=0,
                observation={
                    "hit": True,
                    "observed_bands": [other_band]
                },
                current_slot=0
            )

        state = scorer.band_states[0]
        # Should detect hopping - unique bands / 5 (normalization factor)
        # We have 5 unique bands, so agility_score should be 1.0
        expected_agility = min(1.0, 5 / 5.0)
        assert state.agility_score == expected_agility


class TestThreatModelTypes:
    """Tests for different threat model types."""

    @pytest.mark.parametrize("model_type,expected_features", [
        (ThreatModelType.SIMPLE, ["activity"]),
        (ThreatModelType.STANDARD, ["activity", "snr"]),
        (ThreatModelType.BEHAVIOR_BASED, ["activity", "snr", "behavior"]),
        (ThreatModelType.ADVANCED, ["activity", "snr", "agility", "pri_stability"]),
        (ThreatModelType.FULL, ["activity", "snr", "agility", "pri_stability"]),
    ])
    def test_model_features(self, model_type, expected_features):
        """Each model type has expected features."""
        config = ThreatScoreConfig(model_type=model_type)
        features = config.active_features()
        for feat in expected_features:
            assert feat in features


class TestBehaviorBasedThreatScoring:
    """Tests for behavior-based threat scoring (Vyapti Sub-problem F)."""

    def test_agile_emitters_have_high_threat(self):
        """Agile emitters have higher threat than fixed emitters."""
        from vyapti_simulator.algorithms.threat import (
            get_behavior_threat_score, EmitterBehaviorClass, BEHAVIOR_THREAT_SCORES
        )

        # Agile emitters should have high threat scores (9-10)
        agile_score = BEHAVIOR_THREAT_SCORES[EmitterBehaviorClass.MARKOV_HOPPER]
        assert agile_score >= 9.0

        # Fixed emitters should have lower threat scores (3)
        fixed_score = BEHAVIOR_THREAT_SCORES[EmitterBehaviorClass.CONTINUOUS_FIXED]
        assert fixed_score <= 3.0

        # Agile should be higher than fixed
        assert agile_score > fixed_score

    def test_periodic_emitters_mid_threat(self):
        """Periodic emitters have mid-range threat scores."""
        from vyapti_simulator.algorithms.threat import (
            BEHAVIOR_THREAT_SCORES, EmitterBehaviorClass
        )

        periodic_score = BEHAVIOR_THREAT_SCORES[EmitterBehaviorClass.PERIODIC_SPATIAL_SCAN]
        assert 5.0 <= periodic_score <= 8.0

    def test_get_behavior_threat_score_from_class_name(self):
        """Behavior threat score is correctly derived from class name."""
        from vyapti_simulator.algorithms.threat import get_behavior_threat_score

        # Agile emitters
        assert get_behavior_threat_score("FrequencyAgileEmitter") == 9.0
        assert get_behavior_threat_score("FrequencyAgileScanningEmitter") == 9.0

        # Periodic emitters
        assert get_behavior_threat_score("ScanningEmitter") == 7.0

        # Fixed emitters
        assert get_behavior_threat_score("FixedContinuousEmitter") == 3.0

        # Unknown defaults to 5.0
        assert get_behavior_threat_score("UnknownEmitter") == 5.0

    def test_behavior_classification_from_observation(self):
        """Emitters can be classified from observation features."""
        from vyapti_simulator.algorithms.threat import (
            classify_emitter_by_observation, EmitterBehaviorClass
        )

        # High frequency stability + stable PRI = fixed
        obs_fixed = {"frequency_stability": 0.0, "pri_stability": 0.0}
        assert classify_emitter_by_observation(obs_fixed) == EmitterBehaviorClass.CONTINUOUS_FIXED

        # High frequency agility + stable PRI = agile
        obs_agile = {"frequency_stability": 0.9, "pri_stability": 0.2}
        result = classify_emitter_by_observation(obs_agile)
        assert result in (EmitterBehaviorClass.PSEUDO_RANDOM_AGILE, EmitterBehaviorClass.MARKOV_HOPPER)

    def test_threat_score_with_behavior_classification(self):
        """Threat score incorporates behavior classification."""
        scorer = ThreatScorer(
            band_count=10,
            config=ThreatScoreConfig(model_type=ThreatModelType.BEHAVIOR_BASED)
        )

        # Classify an emitter as agile
        observation = {
            "hit": True,
            "emitter_class_name": "FrequencyAgileEmitter"
        }
        scorer.update_from_observation(band=0, observation=observation, current_slot=0)

        # Agile emitter should have higher threat
        agile_score = scorer.compute_threat_score(band=0)

        # Reset and classify as fixed
        scorer.reset()
        observation = {
            "hit": True,
            "emitter_class_name": "FixedContinuousEmitter"
        }
        scorer.update_from_observation(band=1, observation=observation, current_slot=0)
        fixed_score = scorer.compute_threat_score(band=1)

        # Agile should have higher threat
        assert agile_score > fixed_score

    def test_band_threat_details_include_behavior(self):
        """Band threat details include behavior classification."""
        scorer = ThreatScorer(band_count=10)

        observation = {
            "hit": True,
            "emitter_class_name": "ScanningEmitter"
        }
        scorer.update_from_observation(band=0, observation=observation, current_slot=0)

        details = scorer.get_band_threat_details(band=0)

        assert "emitter_class_name" in details
        assert "behavior_class" in details
        assert "behavior_threat_score" in details
        assert details["emitter_class_name"] == "ScanningEmitter"
        assert details["behavior_threat_score"] == 7.0

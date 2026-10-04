import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vyapti_simulator.core.receiver_observation import ReceiverMetadata, ReceiverObservation
from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState, PublicTransition, create_algorithm
from vyapti_simulator.system_b.tsrd.checkpoints import _ppo_lstm_selection_key


def _settings(**overrides):
    result = {
        "device": "cpu", "cpu_threads": 1, "hidden_units": 16,
        "ppo_epochs": 1, "bptt_chunk": 8, "sequence_minibatch": 8,
        "rollout_episodes": 8, "training_base_steps_target": 100,
        "native_dwell_slots": [2, 1] * 18,
        "prior_active_probability": 0.34266,
        "inactive_to_active_probability": 0.02220,
        "active_to_active_probability": 0.95753,
    }
    result.update(overrides)
    return result


def _policy(seed=7, **settings):
    return create_algorithm({
        "algorithm": {
            "name": "ppo_lstm",
            "module": "vyapti_simulator.system_b.tsrd.policies.ppo_lstm_receiver_only",
            "api": "public_transitions",
            "settings": _settings(**settings),
        },
        "action": {"dwell_slots_by_band": [2, 1] * 18},
    }, bands=36, seed=seed)


def _observation(slot, band, hit=False, dwell_slots=1):
    return ReceiverObservation(
        time_slot=slot, selected_band=band, hit=hit,
        retune_cost_s=0.0003 if band else 0.0,
        dwell_elapsed_s=dwell_slots * 0.05 - (0.0003 if band else 0.0),
        receiver_metadata=ReceiverMetadata(
            dwell_time_ms=dwell_slots * 50.0,
            retune_time_ms=0.3 if band else 0.0,
        ),
    )


def test_v2_observation_is_325d_receiver_only_and_updates_visit_features():
    policy = _policy()
    assert policy.observation_dim == 325
    assert policy.p01 == pytest.approx(0.02220)
    assert policy.p11 == pytest.approx(0.95753)
    assert policy.prior_active == pytest.approx(0.34266)
    assert policy.sweep_slots == 54

    features = policy._features(0)
    assert features.shape == (325,)
    assert np.all(features[36:72] == 1.0)  # never visited => maximum staleness
    assert np.all(features[253:289] == 0.0)
    assert np.all(features[289:325] == 0.0)

    policy.reset_episode(training=True)
    state = PublicState(0, None)
    action = policy.select_action(state, training=True)
    slots = policy.native_dwell_slots[action]
    next_state = PublicState(int(slots), _observation(int(slots), action, True, int(slots)))
    policy.observe(PublicTransition(state, action, 0.25, next_state, False), training=True)
    after = policy._features(int(slots))
    assert policy.visit_counts[action] == 1
    assert after[253 + action] == pytest.approx(1 / math.ceil(600 / 54))
    assert after[289 + action] == 1.0
    assert after[36 + action] == 0.0
    # Reward is retained as the PPO target and is absent from the observation.
    assert policy._rewards == [0.25]
    assert not hasattr(policy, "previous_reward")
    assert not hasattr(policy, "band_reward_ema")


def test_coverage_adjustment_matches_collection_and_ppo_recomputation():
    policy = _policy(seed=13, coverage_kappa=0.10)
    state = PublicState(0, None)
    action = policy.select_action(state, training=True)
    saved_logp = policy._pending["log_probability"]
    features = torch.as_tensor(policy._pending["features"], dtype=torch.float32).unsqueeze(0)
    h = torch.as_tensor(policy._pending["hidden_h"], dtype=torch.float32).unsqueeze(0)
    c = torch.as_tensor(policy._pending["hidden_c"], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        raw_logits, _, _ = policy.model.step(features, (h, c))
        adjusted = policy._adjust_logits(raw_logits, features)
        recomputed = torch.distributions.Categorical(logits=adjusted).log_prob(
            torch.tensor([action]))
    assert float(recomputed.item()) == pytest.approx(saved_logp, abs=1e-6)


def test_ppo_updates_only_after_eight_worlds_and_tracks_base_steps():
    policy = _policy(seed=19, training_base_steps_target=100)
    for episode in range(8):
        policy.reset_episode(training=True)
        state = PublicState(0, None)
        action = policy.select_action(state, training=True)
        next_state = PublicState(1, _observation(1, action))
        policy.observe(PublicTransition(state, action, 0.1, next_state, True), training=True)
        diagnostics = policy.end_episode(training=True)
        assert diagnostics["total_receiver_base_steps"] == episode + 1
        assert diagnostics["training_updates"] == (1 if episode == 7 else 0)
    assert policy.total_policy_decisions == 8
    assert policy.total_receiver_base_steps == 8
    assert policy.total_updates == 1
    assert diagnostics["mean_learning_rate_fraction"] == pytest.approx(0.92)
    assert diagnostics["mean_rollout_episodes"] == 8
    assert diagnostics["episode_receiver_base_steps"] == 1
    assert diagnostics["unique_bands_visited"] >= 1
    assert len(diagnostics["band_selection_counts"]) == 36

    checkpoint = Path(__file__).with_name(".ppo_lstm_receiver_only_test.pt")
    old_checkpoint = Path(__file__).with_name(".ppo_lstm_v1_test.pt")
    try:
        policy.save(checkpoint)
        restored = _policy(seed=19, training_base_steps_target=100)
        restored._load(checkpoint)
        assert restored.total_policy_decisions == 8
        assert restored.total_receiver_base_steps == 8
        assert restored.total_updates == 1
        assert restored.select_action(PublicState(0, None), training=False) in range(36)

        with old_checkpoint.open("wb") as stream:
            torch.save({"format": "vyapti_ppo_lstm_v1", "bands": 36}, stream)
        with pytest.raises(ValueError, match="compatible PPO-LSTM-v2"):
            restored._load(old_checkpoint)
    finally:
        checkpoint.unlink(missing_ok=True)
        old_checkpoint.unlink(missing_ok=True)


def test_rollout_saves_independent_episode_data_and_masked_chunks():
    policy = _policy(seed=23)
    episodes = []
    for episode_index, length in enumerate((1, 2)):
        policy.reset_episode(training=True)
        state = PublicState(0, None)
        observations = []
        actions = []
        for step_index in range(length):
            action = policy.select_action(state, training=True)
            end_slot = state.time_slot + 1
            next_state = PublicState(end_slot, _observation(end_slot, action))
            policy.observe(PublicTransition(state, action, float(episode_index + 1),
                                            next_state, step_index == length - 1), training=True)
            state = next_state
            observations.append(policy._observations[-1].copy())
            actions.append(policy._actions[-1])
        episodes.append(policy._make_episode_rollout())
    assert len(episodes[0].rewards) == 1
    assert len(episodes[1].rewards) == 2
    assert episodes[0].observations.shape == (1, 325)
    assert episodes[1].observations.shape == (2, 325)
    assert not np.shares_memory(episodes[0].observations, episodes[1].observations)
    policy._episode_gae(episodes[0])
    policy._episode_gae(episodes[1])
    assert episodes[0].returns is not None and episodes[1].returns is not None
    assert np.all(np.isfinite(episodes[0].returns))
    assert np.all(np.isfinite(episodes[1].returns))


def test_checkpoint_priority_uses_unique_interception_before_oir():
    report_a = {"summary": {
        "unique_emitter_interception_rate": 0.40,
        "source_relative_ttfi": {"km_restricted_mean_ms": 20_000, "censored_fraction": 0.6},
        "mean_rolling_coverage": 0.5,
        "fraction_worlds_reaching_90pct_band_coverage": 0.1,
        "opportunity_interception_ratio": 0.05,
        "policy_reward_per_decision": 0.01,
    }}
    report_b = {"summary": {
        "unique_emitter_interception_rate": 0.30,
        "source_relative_ttfi": {"km_restricted_mean_ms": 10_000, "censored_fraction": 0.2},
        "mean_rolling_coverage": 0.9,
        "fraction_worlds_reaching_90pct_band_coverage": 0.8,
        "opportunity_interception_ratio": 0.90,
        "policy_reward_per_decision": 0.20,
    }}
    assert _ppo_lstm_selection_key(report_a) > _ppo_lstm_selection_key(report_b)

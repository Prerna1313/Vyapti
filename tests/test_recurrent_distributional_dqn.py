import json
import numpy as np
import pytest
from pathlib import Path

torch = pytest.importorskip("torch")

from vyapti_simulator.core.receiver_observation import ReceiverMetadata, ReceiverObservation
from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState, PublicTransition, create_algorithm
from vyapti_simulator.system_b.tsrd.checkpoints import (
    _recurrent_dqn_selection_key,
    checkpoint_selection_key,
)
from vyapti_simulator.system_b.tsrd.training_setup import resolve_setup
from vyapti_simulator.system_b.tsrd.policies.recurrent_distributional_dqn import (
    DuelingIQNRecurrentQ,
    Episode,
    OBS_DIM,
    OBSERVATION_SCHEMA,
    CHECKPOINT_FORMAT,
    TRAINING_BASE_STEP_TARGET,
    manifest,
    SequenceRef,
    SequencePrioritizedReplay,
)


def _settings():
    return {
        "device": "cpu",
        "cpu_threads": 1,
        "hidden_units": 8,
        "n_step": 2,
        "batch_size": 2,
        "replay_capacity_steps": 100,
        "min_replay_sequences": 1,
        "min_replay_episodes": 1,
        "updates_per_episode": 1,
        "burn_in": 2,
        "learning_sequence": 2,
        "sequence_stride": 1,
        "target_update_interval": 1,
        "n_quantiles": 4,
        "n_target_quantiles": 4,
        "quantile_embed_dim": 8,
        "warmup_actions": 100,
        "native_dwell_slots": [2] * 7 + [1] * 29,
        "training_base_step_target": 100,
    }


def test_iqn_fixed_taus_and_sequence_priorities_are_deterministic():
    model = DuelingIQNRecurrentQ(325, 36, hidden=8, n_quantiles=4, quantile_embed_dim=8)
    assert torch_equal(model.cos_basis, torch.arange(8, dtype=torch.float32) * np.pi)
    taus_a = model.fixed_taus(2, 3, 4, model.cos_basis.device)
    taus_b = model.fixed_taus(2, 3, 4, model.cos_basis.device)
    assert torch_equal(taus_a, taus_b)
    assert np.allclose(taus_a[0, 0].numpy(), [0.125, 0.375, 0.625, 0.875])

    replay = SequencePrioritizedReplay(100, seed=7, sequence_stride=2)
    # T=10 means the terminal decision at index 9 must also be a sample start.
    ep = Episode(
        obs=np.zeros((11, 325), dtype=np.float32),
        actions=np.zeros(10, dtype=np.int64),
        rewards=np.zeros(10, dtype=np.float32),
        gammas=np.ones(10, dtype=np.float32),
        dones=np.zeros(10, dtype=np.float32),
    )
    replay.add(ep)
    assert [ref.start for ref in replay.sequence_refs] == [0, 2, 4, 6, 8, 9]
    refs, weights = replay.sample(8, alpha=0.9, beta=0.6)
    assert len(refs) == 8 and weights.shape == (8,)
    replay.update_priorities(refs, np.arange(1, 9, dtype=np.float32))
    assert max(ref.priority for ref in replay.sequence_refs) > 1


def torch_equal(left, right):
    import torch
    return torch.equal(left, right)


def test_r3dqn_runs_a_training_update_and_receiver_only_eval():
    config = {
        "action": {"dwell_slots_by_band": [2] * 7 + [1] * 29},
        "receiver": {"base_slot_duration_ms": 50, "detection_probability": 0.9,
                     "false_alarm_probability": 0.05, "mission_duration_s": 30},
        "algorithm": {
            "name": "recurrent_distributional_dqn",
            "module": "vyapti_simulator.system_b.tsrd.policies.recurrent_distributional_dqn",
            "api": "public_transitions",
            "settings": _settings(),
        },
    }
    policy = create_algorithm(config, bands=36, seed=17)
    assert policy.native_dwell_slots.tolist() == [2] * 7 + [1] * 29
    assert policy.sweep_slots == 43
    assert policy.model.encoder[0].in_features == 325

    state = PublicState(0, None)
    for index in range(12):
        action = policy.select_action(state, training=True)
        slots = int(policy.native_dwell_slots[action])
        end_slot = state.time_slot + slots
        observation = ReceiverObservation(
            time_slot=end_slot,
            selected_band=action,
            hit=bool(index % 3 == 0),
            retune_cost_s=0.0003,
            dwell_elapsed_s=slots * 0.05 - 0.0003,
            receiver_metadata=ReceiverMetadata(dwell_time_ms=slots * 50,
                                               retune_time_ms=0.3),
        )
        next_state = PublicState(end_slot, observation)
        transition = PublicTransition(state, action, 0.1, next_state, index == 11)
        policy.observe(transition, training=True)
        state = next_state

    episode = policy._episode_from_buffers()
    assert episode is not None
    assert episode.obs.shape == (13, 325)  # T transitions and the terminal successor.
    assert episode.rewards[-1] == pytest.approx(0.1)
    assert episode.dones[-1] == 1.0

    boundary_refs = [SequenceRef(episode, 0), SequenceRef(episode, 11)]
    (prefixes, obs, actions, rewards, gammas, dones, mask) = policy._materialize_sequences(boundary_refs)
    assert [len(prefix) for prefix in prefixes] == [0, 2]
    assert mask.sum(axis=1).tolist() == [2.0, 1.0]
    assert obs.shape == (2, 4, 325)
    assert np.array_equal(obs[1, 1], episode.obs[-1])
    assert rewards[1, 0] == pytest.approx(episode.rewards[-1])
    assert dones[1, 0] == 1.0
    terminal_return, terminal_discount = policy._build_n_step_targets(
        torch.as_tensor(rewards[1:2]), torch.as_tensor(gammas[1:2]),
        torch.as_tensor(dones[1:2]), 0, 2, 2,
    )
    assert terminal_return[0, 0].item() == pytest.approx(episode.rewards[-1])
    assert terminal_discount[0, 0].item() == 0.0

    diagnostics = policy.end_episode(training=True)
    assert diagnostics["training_updates"] == 1
    assert np.isfinite(diagnostics["mean_loss"])
    assert diagnostics["replay_sequences"] >= 1
    assert diagnostics["total_policy_decisions"] == 12
    assert diagnostics["total_receiver_base_steps"] == sum(
        int(policy.native_dwell_slots[action]) for action in policy._episode_actions
    )
    assert diagnostics["episode_base_steps"] == diagnostics["total_receiver_base_steps"]
    assert diagnostics["replay_episodes"] == 1
    assert diagnostics["per_beta"] > policy.per_beta_start
    assert len(diagnostics["band_selection_counts"]) == 36
    assert diagnostics["unique_bands_visited"] > 0
    assert 0.0 <= diagnostics["top_1_action_concentration"] <= 1.0
    assert diagnostics["never_visited_bands"] < 36
    assert np.isfinite(diagnostics["mean_staleness_slots"])
    assert np.isfinite(diagnostics["mean_recent_hit_rate"])

    policy.reset_episode(training=False)
    first = policy.select_action(PublicState(0, None), training=False)
    policy.reset_episode(training=False)
    second = policy.select_action(PublicState(0, None), training=False)
    assert first == second


def test_r3dqn_is_registered_with_shared_training_setup():
    project = __import__("pathlib").Path(__file__).parents[1]
    setup = resolve_setup(
        project / "training_setup/environments/train250_composed.json",
        project / "training_setup/algorithms/recurrent_distributional_dqn.json",
        seed=29,
        episodes=2,
        checkpoint_every=1,
    )
    assert setup["algorithm"]["name"] == "recurrent_distributional_dqn"
    assert setup["algorithm"]["settings"]["prior_active_probability"] == pytest.approx(0.3426563254)
    assert setup["algorithm"]["settings"]["inactive_to_active_probability"] == pytest.approx(0.0221954844)
    policy = create_algorithm(setup, bands=36, seed=29)
    assert policy.native_dwell_slots.tolist() == setup["action"]["dwell_slots_by_band"]
    assert policy.observation_dim == OBS_DIM == 325
    assert policy.min_replay_episodes == 8
    assert policy.training_base_step_target == TRAINING_BASE_STEP_TARGET == 480_000


def test_recent_hit_rate_staleness_visits_and_base_step_beta_are_receiver_causal():
    config = {
        "action": {"dwell_slots_by_band": [2] * 7 + [1] * 29},
        "algorithm": {
            "name": "recurrent_distributional_dqn",
            "module": "vyapti_simulator.system_b.tsrd.policies.recurrent_distributional_dqn",
            "api": "public_transitions",
            "settings": _settings(),
        },
    }
    policy = create_algorithm(config, bands=36, seed=43)
    assert policy.sweep_slots == 43
    policy.reset_episode(training=True)
    first = policy._features(0)
    assert first.shape == (325,)
    assert np.all(first[36:72] == 1.0)
    assert np.all(first[253:289] == 0.0)
    assert np.all(first[289:325] == 0.0)

    # Feed eight actual receiver observations to one band; the ninth replaces
    # the oldest hit in the bounded, receiver-only history.
    for hit in (True, False, True, False, False, False, False, False):
        policy.band_selection_counts[0] += 1
        policy.episode_decisions += 1
        policy._commit_receiver_observation(0, hit, 1)
    feature = policy._features(policy.elapsed_slots)
    assert len(policy.recent_hit_history[0]) == 8
    assert feature[289] == pytest.approx(2 / 8)
    assert feature[253] == pytest.approx(1.0)
    assert feature[36] == pytest.approx(0.0)
    policy.last_hit_slot[0] = 0
    sweep_relative = policy._features(43)
    assert sweep_relative[217] == pytest.approx(0.25)
    assert np.all(sweep_relative[217 + 1:253] == 1.0)
    policy.total_actions = 100
    policy.total_receiver_base_steps = 50
    assert policy._current_per_beta() == pytest.approx(0.8)


def test_dqn_rejects_noncanonical_sweep_profile():
    config = {
        "action": {"dwell_slots_by_band": [1] * 36},
        "algorithm": {
            "name": "recurrent_distributional_dqn",
            "module": "vyapti_simulator.system_b.tsrd.policies.recurrent_distributional_dqn",
            "api": "public_transitions",
            "settings": {**_settings(), "native_dwell_slots": [1] * 36},
        },
    }
    with pytest.raises(ValueError, match="must sum to 43"):
        create_algorithm(config, bands=36, seed=44)


def test_dqn_waits_for_eight_eligible_replay_episodes():
    config = {
        "action": {"dwell_slots_by_band": [2] * 7 + [1] * 29},
        "algorithm": {
            "name": "recurrent_distributional_dqn",
            "module": "vyapti_simulator.system_b.tsrd.policies.recurrent_distributional_dqn",
            "api": "public_transitions",
            "settings": {
                **_settings(), "min_replay_episodes": 8, "min_replay_sequences": 1,
                "n_step": 2, "burn_in": 2, "learning_sequence": 2,
                "updates_per_episode": 1, "sequence_stride": 1,
            },
        },
    }
    policy = create_algorithm(config, bands=36, seed=47)
    for episode_index in range(8):
        policy.reset_episode(training=True)
        state = PublicState(0, None)
        for step_index in range(6):
            action = policy.select_action(state, training=True)
            next_slot = state.time_slot + 1
            observation = ReceiverObservation(
                time_slot=next_slot, selected_band=action, hit=False,
                retune_cost_s=0.0003,
                dwell_elapsed_s=0.05 - 0.0003,
                receiver_metadata=ReceiverMetadata(dwell_time_ms=50.0, retune_time_ms=0.3),
            )
            next_state = PublicState(next_slot, observation)
            policy.observe(PublicTransition(state, action, 0.1, next_state, step_index == 5),
                           training=True)
            state = next_state
        diagnostics = policy.end_episode(training=True)
        assert diagnostics["training_updates"] == (1 if episode_index == 7 else 0)
        assert diagnostics["replay_episodes"] == episode_index + 1
        # The runner writes these diagnostics with allow_nan=False. Metrics
        # before replay warmup must therefore be null, not NaN.
        json.dumps(diagnostics, allow_nan=False)
        if episode_index < 7:
            assert diagnostics["mean_loss"] is None
        else:
            assert np.isfinite(diagnostics["mean_loss"])
    assert policy.total_updates == 1


def test_manifest_and_checkpoint_identity_reject_old_253_schema():
    policy = create_algorithm({
        "action": {"dwell_slots_by_band": [2] * 7 + [1] * 29},
        "algorithm": {
            "name": "recurrent_distributional_dqn",
            "module": "vyapti_simulator.system_b.tsrd.policies.recurrent_distributional_dqn",
            "api": "public_transitions", "settings": _settings(),
        },
    }, bands=36, seed=53)
    details = manifest()
    assert details["obs_dim"] == 325
    assert details["min_replay_sequences"] == 128
    assert details["min_replay_episodes"] == 8
    assert details["recent_hit_window"] == 8
    assert details["training_base_step_target"] == 480_000
    assert details["observation_schema"] == OBSERVATION_SCHEMA
    assert details["reward_in_policy_observation"] is False

    checkpoint = Path(__file__).with_name(".recurrent_distributional_dqn_test.pt")
    old_checkpoint = Path(__file__).with_name(".recurrent_distributional_dqn_old_test.pt")
    try:
        policy.total_actions = 17
        policy.total_receiver_base_steps = 29
        policy.save(checkpoint)
        restored = create_algorithm({
            "action": {"dwell_slots_by_band": [2] * 7 + [1] * 29},
            "algorithm": {
                "name": "recurrent_distributional_dqn",
                "module": "vyapti_simulator.system_b.tsrd.policies.recurrent_distributional_dqn",
                "api": "public_transitions", "settings": _settings(),
            },
        }, bands=36, seed=53, checkpoint=checkpoint)
        assert restored.total_actions == 17
        assert restored.total_receiver_base_steps == 29

        import torch
        with old_checkpoint.open("wb") as stream:
            torch.save({
                "format": "vyapti_recurrent_distributional_dqn_v2",
                "bands": 36, "obs_dim": 253,
                "observation_schema": "receiver_belief_periodicity_dwell_hit_history_v2",
            }, stream)
        with pytest.raises(ValueError, match="Checkpoint format"):
            restored._load(old_checkpoint)
        assert CHECKPOINT_FORMAT == "vyapti_recurrent_distributional_dqn"
    finally:
        checkpoint.unlink(missing_ok=True)
        old_checkpoint.unlink(missing_ok=True)


def test_dqn_checkpoint_priority_uses_unique_capture_before_oir():
    higher_oir = {
        "summary": {
            "unique_emitter_interception_rate": 0.30,
            "source_relative_ttfi": {"km_restricted_mean_ms": 18000.0},
            "fraction_worlds_reaching_90pct_band_coverage": 0.2,
            "interception_by_deadline": {"interception_rate": {"5s": 0.1}},
            "opportunity_interception_ratio": 0.08,
            "policy_reward_per_decision": 0.4,
        }
    }
    higher_unique_capture = {
        "summary": {
            "unique_emitter_interception_rate": 0.31,
            "source_relative_ttfi": {"km_restricted_mean_ms": 19000.0},
            "fraction_worlds_reaching_90pct_band_coverage": 0.1,
            "interception_by_deadline": {"interception_rate": {"5s": 0.05}},
            "opportunity_interception_ratio": 0.07,
            "policy_reward_per_decision": 0.3,
        }
    }
    module = "vyapti_simulator.system_b.tsrd.policies.recurrent_distributional_dqn"
    assert _recurrent_dqn_selection_key(higher_unique_capture) > _recurrent_dqn_selection_key(higher_oir)
    assert checkpoint_selection_key(module, higher_unique_capture) == _recurrent_dqn_selection_key(
        higher_unique_capture
    )


def test_staleness_diagnostic_uses_unclipped_receiver_slot_age():
    policy = create_algorithm({
        "action": {"dwell_slots_by_band": [2] * 7 + [1] * 29},
        "algorithm": {
            "name": "recurrent_distributional_dqn",
            "module": "vyapti_simulator.system_b.tsrd.policies.recurrent_distributional_dqn",
            "api": "public_transitions", "settings": _settings(),
        },
    }, bands=36, seed=61)
    policy.reset_episode(training=False)
    # The actor feature remains clipped at 4 * sweep_slots, while diagnostics
    # retain the actual age from the beginning of this unseen band's episode.
    features = policy._features(600)
    assert np.all(features[36:72] == 1.0)
    assert np.all(policy._raw_staleness_slots(600) == 600)
    policy.elapsed_slots = 600
    diagnostics = policy.end_episode(training=False)
    assert diagnostics["maximum_staleness_slots"] == 600

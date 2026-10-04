import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vyapti_simulator.core.receiver_observation import ReceiverMetadata, ReceiverObservation
from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState, PublicTransition, create_algorithm
from vyapti_simulator.system_b.tsrd.training_setup import resolve_setup
from vyapti_simulator.system_b.tsrd.policies.recurrent_distributional_dqn import (
    DuelingIQNRecurrentQ,
    Episode,
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
        "updates_per_episode": 1,
        "burn_in": 2,
        "learning_sequence": 2,
        "sequence_stride": 1,
        "target_update_interval": 1,
        "n_quantiles": 4,
        "n_target_quantiles": 4,
        "quantile_embed_dim": 8,
        "warmup_actions": 100,
        "native_dwell_slots": [1, 2] * 18,
    }


def test_iqn_fixed_taus_and_sequence_priorities_are_deterministic():
    model = DuelingIQNRecurrentQ(253, 36, hidden=8, n_quantiles=4, quantile_embed_dim=8)
    assert torch_equal(model.cos_basis, torch.arange(1, 9, dtype=torch.float32) * np.pi)
    taus_a = model.fixed_taus(2, 3, 4, model.cos_basis.device)
    taus_b = model.fixed_taus(2, 3, 4, model.cos_basis.device)
    assert torch_equal(taus_a, taus_b)
    assert np.allclose(taus_a[0, 0].numpy(), [0.125, 0.375, 0.625, 0.875])

    replay = SequencePrioritizedReplay(100, seed=7, sequence_stride=2)
    # T=10 means the terminal decision at index 9 must also be a sample start.
    ep = Episode(
        obs=np.zeros((11, 253), dtype=np.float32),
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
        "action": {"dwell_slots_by_band": [1, 2] * 18},
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
    assert policy.native_dwell_slots.tolist() == [1, 2] * 18
    assert policy.model.encoder[0].in_features == 253

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
    assert episode.obs.shape == (13, 253)  # T transitions and the terminal successor.
    assert episode.rewards[-1] == pytest.approx(0.1)
    assert episode.dones[-1] == 1.0

    boundary_refs = [SequenceRef(episode, 0), SequenceRef(episode, 11)]
    (prefixes, obs, actions, rewards, gammas, dones, mask) = policy._materialize_sequences(boundary_refs)
    assert [len(prefix) for prefix in prefixes] == [0, 2]
    assert mask.sum(axis=1).tolist() == [2.0, 1.0]
    assert obs.shape == (2, 4, 253)
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

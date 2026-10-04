from __future__ import annotations

import numpy as np
import pytest
import torch
import json

from vyapti_simulator.system_b.tsrd.policies import residual_discrete_sac as residual


def test_entropy_temperature_update_moves_alpha_toward_target_entropy():
    # With entropy below target, alpha must rise to encourage more entropy;
    # with entropy above target, alpha must fall.
    for entropy, should_increase in ((0.5, True), (1.5, False)):
        log_alpha = torch.tensor(0.0, requires_grad=True)
        optimizer = torch.optim.SGD([log_alpha], lr=0.1)
        before = log_alpha.detach().exp().item()
        loss = residual.entropy_temperature_loss(
            log_alpha, torch.tensor([entropy]), target_entropy=1.0
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after = log_alpha.detach().exp().item()
        assert (after > before) is should_increase


def test_contextual_ts_feedback_is_receiver_hit_rate():
    assert residual.contextual_ts_feedback([]) == 0.0
    assert residual.contextual_ts_feedback([True]) == 1.0
    assert residual.contextual_ts_feedback([True, False]) == 0.5


def test_old_truth_reward_ts_posteriors_are_rejected():
    sampler = residual.ContextualThompsonSampler(residual.N_BANDS, seed=1)
    state = sampler.state_dict()
    assert state["feedback_schema"] == "receiver_hit_rate_v1"
    assert state["feedback_scale"] == 1.0

    stale_state = dict(state)
    stale_state.pop("feedback_schema")
    with pytest.raises(ValueError, match="different feedback schema"):
        sampler.load_state_dict(stale_state)


def test_contextual_ts_rejects_out_of_range_feedback():
    sampler = residual.ContextualThompsonSampler(residual.N_BANDS, seed=1)
    context = np.zeros(residual.TS_CONTEXT_DIM, dtype=np.float64)
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        sampler.update(0, context, 10.0)


def test_evaluation_ts_adapts_from_hits_not_mission_reward():
    class FakeWorld:
        def __init__(self):
            self.slots = 0

        def step_dwell_training(self, band, dwell, reward_mode):
            assert reward_mode == residual.REWARD_MODE
            looks = []
            for _ in range(dwell):
                looks.append({"hit": False, "time_slot": self.slots})
                self.slots += 1
            return looks, 10_000.0, self.slots >= 5

    class PriorOnlyAgent:
        def select_eval_action(self, observation, candidates):
            return 0, {"q_advantage": 0.0, "policy_prob": 1.0}

    dwell = np.ones(residual.N_BANDS, dtype=np.int32)
    dwell[:7] = 2
    ts = residual.ContextualThompsonSampler(residual.N_BANDS, seed=2)
    trajectory, diagnostics = residual.rollout_policy_on_world(
        FakeWorld(),
        PriorOnlyAgent(),
        residual.make_calibrated_transition(),
        np.full(residual.N_BANDS, residual.CAL_PRIOR_ACTIVE),
        dwell,
        ts,
    )

    assert diagnostics["decisions"] > 0
    assert int(ts.pulls.sum()) == diagnostics["decisions"]
    # Mission reward is deliberately large, but the observed outcome was MISS.
    # Therefore it must not appear in the TS posterior's response vector.
    np.testing.assert_array_equal(ts.b, np.zeros_like(ts.b))


@pytest.mark.parametrize("duplicate_mean", [False, True])
def test_candidate_quotas_preserve_underexplored_and_priority(monkeypatch, duplicate_mean):
    dwell = np.ones(residual.N_BANDS, dtype=np.int32)
    state = residual.ResidualCausalState(
        residual.make_calibrated_transition(),
        np.full(residual.N_BANDS, residual.CAL_PRIOR_ACTIVE), dwell,
    )
    state.visit_counts.fill(2)
    state.visit_counts[3:6] = 0
    state.total_decisions = int(state.visit_counts.sum())

    def peak(band):
        values = np.zeros(residual.N_BANDS, dtype=np.float32)
        values[band] = 1.0
        return values

    monkeypatch.setattr(state, "belief_vector", lambda: peak(7))
    monkeypatch.setattr(state, "periodicity_features", lambda: (peak(6), peak(6)))
    monkeypatch.setattr(state, "staleness_vector", lambda: peak(8))
    monkeypatch.setattr(state, "priority_vector", lambda: peak(9))
    ts = residual.ContextualThompsonSampler(residual.N_BANDS, seed=5)
    monkeypatch.setattr(ts, "sample_scores", lambda contexts: (
        peak(0), peak(0 if duplicate_mean else 1), peak(2),
    ))
    candidates = residual.ResidualCandidateGenerator(dwell, ts).build(state)
    assert candidates.bands[0] == 0
    assert len(set(candidates.bands)) == 10
    assert {2, 3, 4, 5, 6, 7, 8, 9}.issubset(set(candidates.bands))
    assert candidates.features.shape == (10, 18)
    np.testing.assert_array_equal(
        candidates.features[:, -1], (state.visit_counts[candidates.bands] == 0),
    )


def test_underexplored_count_ties_use_oldest_raw_visit():
    dwell = np.ones(residual.N_BANDS, dtype=np.int32)
    state = residual.ResidualCausalState(
        residual.make_calibrated_transition(),
        np.full(residual.N_BANDS, residual.CAL_PRIOR_ACTIVE), dwell,
    )
    state.visit_counts.fill(1)
    state.last_visit_end.fill(100)
    state.last_visit_end[33:36] = [1, 2, 3]
    state.total_decisions = residual.N_BANDS
    ts = residual.ContextualThompsonSampler(residual.N_BANDS, seed=5)
    generator = residual.ResidualCandidateGenerator(dwell, ts)
    candidates = generator.build(state)
    # TS proposals may themselves select an old band. The reserved slots must
    # still include all three oldest bands rather than prefer tied low indices.
    assert {33, 34, 35}.issubset(set(candidates.bands))


def test_eval_actor_argmax_survives_negative_critic_advantage(monkeypatch):
    agent = residual.ResidualDiscreteSAC(
        residual.OBS_DIM, residual.CANDIDATE_DIM, residual.CANDIDATE_COUNT, seed=3,
    )
    probs = torch.full((1, 10), 0.09, device=residual.DEVICE)
    probs[0, 3] = 0.19
    monkeypatch.setattr(agent, "policy_distribution", lambda *args, **kwargs: (
        probs, probs.log(),
    ))

    class FakeQ:
        def all_values(self, observation, candidates):
            values = torch.zeros((1, 10), device=residual.DEVICE)
            values[0, 0] = 10.0
            values[0, 3] = -10.0
            return values

    monkeypatch.setattr(agent, "q1", FakeQ())
    monkeypatch.setattr(agent, "q2", FakeQ())
    choice, info = agent.select_eval_action(
        np.zeros(residual.OBS_DIM), np.zeros((10, 18)),
    )
    assert choice == 3
    assert info["q_advantage"] < 0.0
    assert info["used_prior"] == 0.0


def test_new_candidate_replay_update_and_checkpoint_schema():
    dwell = np.ones(residual.N_BANDS, dtype=np.int32)
    state = residual.ResidualCausalState(
        residual.make_calibrated_transition(),
        np.full(residual.N_BANDS, residual.CAL_PRIOR_ACTIVE), dwell,
    )
    ts = residual.ContextualThompsonSampler(residual.N_BANDS, seed=5)
    generator = residual.ResidualCandidateGenerator(dwell, ts)
    agent = residual.ResidualDiscreteSAC(
        residual.OBS_DIM, generator.feature_dim, residual.CANDIDATE_COUNT, seed=3,
    )
    replay = residual.ReplayBuffer(8, residual.OBS_DIM, 10, 18, seed=4)
    before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    for action in range(4):
        obs, candidates = state.observation(), generator.build(state)
        band = int(candidates.bands[action])
        state.step(band, [bool(action % 2)])
        replay.add(obs, candidates.features, action, 0.01 * action,
                   state.observation(), generator.build(state).features,
                   action == 3, residual.GAMMA_BASE)
    stats = agent.update(replay.sample(4, residual.DEVICE), prior_bias=0.25)
    assert all(np.isfinite(value) for value in stats.values())
    assert any(not torch.equal(old, new) for old, new in zip(before, agent.actor.parameters()))
    payload = agent.state_dict(step=4)
    agent.load_state_dict(payload)
    stale = dict(payload, candidate_dim=17)
    stale.pop("candidate_schema")
    with pytest.raises(ValueError, match="candidate schema"):
        agent.load_state_dict(stale)


def test_actual_trainer_crosses_replay_warmup_and_saves_loadable_checkpoint(monkeypatch, tmp_path):
    # Exercise train() itself: manual construction of an 18-field agent did
    # not catch the old hard-coded 17-field network in the training entry point.
    class FakeWorld:
        def __init__(self):
            self.native_dwell_slots = np.ones(residual.N_BANDS, dtype=np.int32)
            self.native_dwell_slots[:7] = 2
            self.slots = 0

        def dwell_slots_for_band(self, band):
            return int(self.native_dwell_slots[band])

        def reset(self, seed):
            self.slots = 0

        def step_dwell_training(self, band, dwell, reward_mode):
            assert reward_mode == residual.REWARD_MODE
            looks = [{"hit": bool((self.slots + offset) % 2),
                      "time_slot": self.slots + offset} for offset in range(dwell)]
            self.slots += dwell
            return looks, 0.01, self.slots >= 6

    class FakeFactory:
        def make_train_world(self, seed):
            return FakeWorld()

    monkeypatch.setattr(residual, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(residual, "REPLAY_CAPACITY", 16)
    monkeypatch.setattr(residual, "MIN_REPLAY_SIZE", 2)
    monkeypatch.setattr(residual, "WARMUP_ACTIONS", 2)
    monkeypatch.setattr(residual, "BATCH_SIZE", 4)
    residual.train(
        FakeFactory(), tmp_path, seed=3,
        transition=residual.make_calibrated_transition(),
        prior_active=np.full(residual.N_BANDS, residual.CAL_PRIOR_ACTIVE),
        max_actions=8,
    )
    checkpoint = tmp_path / "checkpoints" / "residual_sac_final.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["step"] == 8
    assert payload["candidate_dim"] == residual.CANDIDATE_DIM == 18
    assert payload["train_stats"]["replay_size"] == 8
    assert "actor_loss" in payload["train_stats"]
    assert all(np.isfinite(value) for value in payload["train_stats"].values())
    agent, transition, prior, dwell, ts = residual.load_agent(checkpoint)
    state = residual.ResidualCausalState(transition, prior, dwell)
    candidates = residual.ResidualCandidateGenerator(dwell, ts).build(state)
    choice, _ = agent.select_eval_action(state.observation(), candidates.features)
    assert 0 <= choice < residual.CANDIDATE_COUNT
    manifest = json.loads((tmp_path / "experiment_config.json").read_text())
    assert manifest["residual_contract"]["candidate_dim"] == 18

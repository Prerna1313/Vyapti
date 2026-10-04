from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.system_b.tsrd.policies import residual_discrete_sac as residual


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

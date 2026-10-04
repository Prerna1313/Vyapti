from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluation import evaluate_residual_discrete_sac as evaluator


def test_sampled_actor_reproducible_and_independent_of_torch_rng(monkeypatch):
    monkeypatch.setattr(evaluator.learner, "DEVICE", torch.device("cpu"))

    class Q:
        def all_values(self, obs, candidates):
            return torch.tensor([[10.0, -10.0] + [0.0] * 8])

    agent = SimpleNamespace(
        policy_distribution=lambda *args, **kwargs: (
            torch.tensor([[0.6, 0.4] + [0.0] * 8]), None,
        ), q1=Q(), q2=Q(),
    )
    observation, candidates = np.zeros(325), np.zeros((10, 18))
    first = evaluator.DiagnosticPolicy(agent, "actor_sampled", 420000)
    second = evaluator.DiagnosticPolicy(agent, "actor_sampled", 420000)
    actions, advantages = [], []
    for _ in range(40):
        choice, info = first.select_eval_action(observation, candidates)
        torch.rand(7)  # Unrelated randomness cannot change the actor's stream.
        assert second.select_eval_action(observation, candidates)[0] == choice
        actions.append(choice)
        advantages.append(info["q_advantage"])
    assert set(actions) == {0, 1}
    assert min(advantages) < 0.0  # No critic fallback in sampled mode.
    ts_only = evaluator.DiagnosticPolicy(None, "ts_only", 420000)
    assert ts_only.select_eval_action(observation, candidates)[0] == 0
    agent.select_eval_action = lambda *args: (7, {"q_advantage": -1.0, "policy_prob": 0.4})
    greedy = evaluator.DiagnosticPolicy(agent, "actor_argmax", 420000)
    assert greedy.select_eval_action(observation, candidates)[0] == 7


@pytest.mark.parametrize("mode", ["actor_argmax", "ts_only", "actor_sampled"])
def test_ablation_pins_receiver_seeds_and_resets_ts_per_world(monkeypatch, mode):
    learner = evaluator.learner
    monkeypatch.setattr(learner, "DEVICE", torch.device("cpu"))
    trained_ts = learner.ContextualThompsonSampler(36, seed=7)
    initial = trained_ts.b.copy()
    monkeypatch.setattr(learner, "load_agent", lambda checkpoint: (
        object(), None, None, None, trained_ts,
    ))
    seen = []

    def rollout(env, policy, transition, prior, dwell, ts):
        assert policy.mode == mode
        np.testing.assert_array_equal(ts.b, initial)
        context = np.ones(learner.TS_CONTEXT_DIM)
        ts.update(0, context, 1.0)
        seen.append((env.receiver_seed, float(ts.rng.random()), policy.rng.random()))
        return [], {"decisions": 1, "ts_prior_decisions": 1, "residual_decisions": 0}

    monkeypatch.setattr(learner, "rollout_policy_on_world", rollout)

    class World:
        def __init__(self, recipe):
            self.recipe = recipe

        def reset(self, seed):
            self.receiver_seed = seed

    class Factory:
        def make_val_world(self, recipe):
            return World(recipe)

        def score_episode(self, env, trajectory):
            return {"cell_level": {}, "frozen_world_identity_sha256": env.recipe["identity"]}

    recipes = [{"world_id": i, "identity": str(i), "receiver_seed": 90 + i} for i in range(2)]
    result = evaluator.evaluate_ablation(Factory(), "checkpoint", recipes, mode, 420000)
    assert [row["receiver_seed"] for row in result["rows"]] == [90, 91]
    assert [row["policy_eval_seed"] for row in result["rows"]] == [420000, 420001]
    assert [row["frozen_world_identity_sha256"] for row in result["rows"]] == ["0", "1"]
    np.testing.assert_array_equal(trained_ts.b, initial)
    original = seen.copy()
    seen.clear()
    evaluator.evaluate_ablation(Factory(), "checkpoint", recipes, mode, 420000)
    assert seen == original


def test_test_split_rejects_diagnostic_modes(monkeypatch):
    monkeypatch.setattr("sys.argv", ["evaluate", "--run", "unused", "--checkpoint", "unused.pt",
                                    "--split", "test", "--final", "--policy-mode", "ts_only"])
    with pytest.raises(SystemExit) as error:
        evaluator.main()
    assert error.value.code == 2

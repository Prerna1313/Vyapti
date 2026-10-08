from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluation import evaluate_residual_discrete_sac as evaluator


@pytest.mark.parametrize("mode", ["actor_sampled", "uniform_candidates"])
@pytest.mark.parametrize("mixed_dwell", [False, True])
def test_decision_trace_preserves_rollout_and_rewards(tmp_path, monkeypatch, mode, mixed_dwell):
    import json
    learner = evaluator.learner
    monkeypatch.setattr(learner, "DEVICE", torch.device("cpu"))

    class Q:
        def __init__(self, offset):
            self.offset = offset

        def all_values(self, obs, candidates):
            return torch.arange(10, dtype=torch.float32)[None] + self.offset

    probs = torch.softmax(torch.arange(10, dtype=torch.float32), dim=0)[None]
    agent = SimpleNamespace(policy_distribution=lambda *args, **kwargs: (probs, probs.log()),
                            q1=Q(0), q2=Q(0.5))

    class World:
        def __init__(self):
            self.slot = 0
            self.actions = []

        @property
        def done(self):
            return self.slot >= 20

        def step_dwell_training(self, band, dwell, reward_mode):
            self.actions.append(band)
            looks = [{"time_slot": slot, "hit": bool((slot + band) % 3 == 0)}
                     for slot in range(self.slot, min(self.slot + dwell, 20))]
            self.slot += len(looks)
            self.last_reward_components = {"new_intercept_utility": 0.1,
                "elapsed_cost": 0.02, "false_alarm_cost": 0.01, "mission_reward": 0.07}
            return looks, 0.07, self.done

    transition = np.tile(np.array([[0.9, 0.1], [0.1, 0.9]]), (36, 1, 1))
    prior = np.full(36, 0.5)
    dwell = np.ones(36, dtype=int)
    if mixed_dwell:
        dwell[[0, 1, 6, 7, 17, 18, 19]] = 2
    worlds = [World(), World()]
    trajectories = []
    diagnostics = []
    path = tmp_path / "decisions.jsonl"
    for traced, env in zip((False, True), worlds):
        policy = evaluator.DiagnosticPolicy(agent, mode, 420000)
        ts = learner.ContextualThompsonSampler(36, seed=420000)
        if traced:
            trajectory, diag = evaluator.traced_rollout(env, policy, transition, prior,
                                                       dwell, ts, path, "world_0")
        else:
            trajectory, diag = learner.rollout_policy_on_world(env, policy, transition,
                                                               prior, dwell, ts)
        trajectories.append([(s.action, s.time_slot, s.observation) for s in trajectory])
        diagnostics.append(diag)
    assert worlds[0].actions == worlds[1].actions
    assert trajectories[0] == trajectories[1]
    for key in ("q_advantages", "policy_probs", "decisions"):
        assert diagnostics[0][key] == diagnostics[1][key]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len(worlds[1].actions)
    assert len(trajectories[1]) == 20
    assert rows[-1]["done"] is True
    assert diagnostics[1]["mission_reward_components"]["mission_reward"] == pytest.approx(0.07 * len(rows))
    for row in rows:
        assert row["q_spread"] == 9
        assert row["mean_critic_disagreement"] == 0.5
        assert sum(row["selection_probabilities"]) == pytest.approx(1)


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
    seen.clear()
    alternate = evaluator.evaluate_ablation(Factory(), "checkpoint", recipes, mode, 420000,
                                           action_seed_base=12345)
    assert [entry[:2] for entry in seen] == [entry[:2] for entry in original]
    assert [entry[2] for entry in seen] != [entry[2] for entry in original]
    assert [row["receiver_seed"] for row in alternate["rows"]] == [90, 91]


def test_test_split_rejects_diagnostic_modes(monkeypatch):
    monkeypatch.setattr("sys.argv", ["evaluate", "--run", "unused", "--checkpoint", "unused.pt",
                                    "--split", "test", "--final", "--policy-mode", "ts_only"])
    with pytest.raises(SystemExit) as error:
        evaluator.main()
    assert error.value.code == 2


def test_temperature_changes_probabilities_and_uniform_ignores_actor(monkeypatch):
    monkeypatch.setattr(evaluator.learner, "DEVICE", torch.device("cpu"))

    class Q:
        def all_values(self, *args):
            return torch.zeros((1, 10))

    probs = torch.tensor([[0.6, 0.4] + [0.0] * 8], dtype=torch.float64)
    agent = SimpleNamespace(policy_distribution=lambda *args, **kwargs: (probs, probs.log()),
                            q1=Q(), q2=Q())
    observation, candidates = np.zeros(325), np.zeros((10, 18))

    class Recorder:
        def choice(self, count, p):
            self.probabilities = p
            return 0

    probabilities = []
    for temperature in (0.8, 1.0, 1.2):
        policy = evaluator.DiagnosticPolicy(agent, "actor_sampled", 1, temperature)
        policy.rng = Recorder()
        policy.select_eval_action(observation, candidates)
        expected = np.array([0.6, 0.4]) ** (1 / temperature)
        expected /= expected.sum()
        np.testing.assert_allclose(policy.rng.probabilities[:2], expected)
        probabilities.append(policy.rng.probabilities[0])
    assert probabilities[0] > probabilities[1] > probabilities[2]
    agent.policy_distribution = lambda *args, **kwargs: pytest.fail("Uniform baseline must ignore actor probabilities")
    policy = evaluator.DiagnosticPolicy(agent, "uniform_candidates", 1)
    policy.rng = Recorder()
    policy.select_eval_action(observation, candidates)
    np.testing.assert_allclose(policy.rng.probabilities, np.full(10, 0.1))


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperatures_rejected(temperature):
    with pytest.raises(ValueError, match="finite and positive"):
        evaluator.DiagnosticPolicy(None, "actor_sampled", 1, temperature)


def test_calibration_output_paths_separate_settings_and_preserve_original():
    original = evaluator.evaluation_destination("run", "val", "normal", "final.pt", "actor_sampled", True)
    assert original.as_posix() == "run/eval/val/normal/ablations/actor_sampled/checkpoints/final"
    paths = {evaluator.evaluation_destination("run", "val", "normal", "final.pt", mode,
                                             True, temperature, seed)
             for mode in ("actor_sampled", "uniform_candidates")
             for temperature in (0.8, 1.0, 1.2) for seed in (420000, 420001)}
    assert len(paths) == 12
    assert original not in paths


def test_reused_calibration_reports_require_worlds_seeds_and_reset(monkeypatch):
    import json
    from pathlib import Path
    from scripts.evaluation.calibrate_residual_discrete_sac import checked_report

    rows = [{"world_id": i, "frozen_world_identity_sha256": str(i), "receiver_seed": 90 + i}
            for i in range(50)]
    report = {"policy_mode": "actor_sampled", "split": "val", "condition": "normal",
              "checkpoint_sha256": "checkpoint", "frozen_world_catalog_sha256": "catalog",
              "ts_episode_reset_protocol": "independent deep copy of trained posterior for each world",
              "eval_seed_base": 420000, "summary": {"worlds": 50}, "rows": rows}
    monkeypatch.setattr(Path, "read_text", lambda path, **kwargs:
                        json.dumps(report) if path.name == "summary.json" else
                        "\n".join(json.dumps(row) for row in rows))
    expected = {str(i): 90 + i for i in range(50)}
    args = (Path("summary.json"), "actor_sampled", 1.0, 420000, 420000,
            "checkpoint", "catalog", expected)
    assert checked_report(*args) == report  # Accept the existing temperature-1 report.
    with pytest.raises(ValueError, match="calibration settings"):
        checked_report(Path("summary.json"), "actor_sampled", 1.0, 420001, 420000,
                       "checkpoint", "catalog", expected)
    report.pop("ts_episode_reset_protocol")
    with pytest.raises(ValueError, match="calibration settings"):
        checked_report(*args)

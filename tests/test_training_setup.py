"""Shared specs, public transitions, and checkpoint lifecycle integration."""
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tests.test_train250_cache import _small_pool
from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState, PublicTransition
from vyapti_simulator.system_b.tsrd.checkpoints import freeze_selection
from vyapti_simulator.system_b.tsrd.experiment import (
    _strongest_emitter_in_selected_window, _emitters_with_prior_opportunity,
    evaluate, load_contract, train,
)
from vyapti_simulator.system_b.tsrd.training_setup import resolve_setup


ROOT = Path(__file__).parents[1] / "training_setup"


def test_environment_has_no_algorithm_and_heldout_recipes_are_shared():
    environment = json.loads((ROOT / "environments/train250_composed.json").read_text())
    assert not set(environment).intersection({"policy_module", "algorithm", "train_episodes", "seed", "neural_policy"})
    config = resolve_setup(ROOT / "environments/train250_composed.json",
                           ROOT / "algorithms/ucb1.json", seed=20261002,
                           episodes=100, checkpoint_every=25)
    assert config["algorithm"]["name"] == "ucb1"
    assert len(config["evaluation"]["val_composed_worlds"]) == 50
    assert len(config["evaluation"]["test_composed_worlds"]) == 50
    for split in ("val", "test"):
        assert all(r["source_config_ids"] == config["evaluation"][f"{split}_config_ids"]
                   for r in config["evaluation"][f"{split}_composed_worlds"])
    assert config["checkpointing"]["interval_episodes"] == 25
    assert config["setup_sources"]["evaluation"]["sha256"]
    load_contract(config)


def test_public_state_cannot_carry_untyped_truth():
    with pytest.raises(TypeError, match="closed ReceiverObservation"):
        PublicState(0, {"truth": [1], "world_seed": 3})
    with pytest.raises(TypeError):
        PublicTransition(PublicState(0, None), 0, 0.0, PublicState(1, None), False, truth=True)


def test_binary_cell_hit_truth_credit_selects_only_strongest_emitter():
    world = SimpleNamespace(
        recorded_pulses=(
            np.asarray([[1_000.0, 105.0, 1.0, 0.0, -30.0],
                        [2_000.0, 108.0, 1.0, 0.0, -12.0],
                        [3_000.0, 104.0, 1.0, 0.0, -20.0]]),
            np.asarray([11, 22, 33]),
        ),
        config=SimpleNamespace(slot_duration_s=lambda: 0.05),
        receiver_geometry=(np.asarray([100.0]), 10.0),
    )
    assert _strongest_emitter_in_selected_window(world, 0, 0, 0.0) == 22


def test_unresolved_elapsed_cost_excludes_emitters_without_prior_opportunity():
    first_slots = {10: 0, 20: 3, 30: 8}
    assert _emitters_with_prior_opportunity(first_slots, 3) == {10}
    assert _emitters_with_prior_opportunity(first_slots, 4) == {10, 20}


def test_off_policy_adapter_gets_transitions_and_checkpoint_selection_is_frozen(tmp_path, monkeypatch):
    module = ModuleType("test_replay_algorithm")
    instances = []

    class ReplayAlgorithm:
        def __init__(self, bands, settings, checkpoint):
            self.bands, self.settings = bands, settings
            self.steps = json.loads(checkpoint.read_text())["steps"] if checkpoint else 0
            self.replay = []
            self.updates = 0
            self.terminals = 0
            instances.append(self)

        def reset_episode(self, *, training):
            pass

        def select_action(self, state, *, training):
            assert type(state) is PublicState
            assert state.previous_observation is None or state.previous_observation.time_slot == state.time_slot - 1
            return state.time_slot % self.bands

        def observe(self, transition, *, training):
            assert type(transition) is PublicTransition
            assert transition.next_state.time_slot - transition.state.time_slot in (1, 2)
            assert isinstance(transition.reward, float)
            if training:
                self.replay.append(transition)
                self.steps += 1
                self.terminals += transition.terminated
                # An off-policy adapter owns replay and update cadence.
                if self.steps % self.settings["update_every"] == 0:
                    self.updates += 1

        def end_episode(self, *, training):
            return {"updates": self.updates, "replay_size": len(self.replay)} if training else None

        def save(self, path):
            path.write_text(json.dumps({"steps": self.steps}))

    module.create = lambda *, bands, seed, settings, checkpoint=None: ReplayAlgorithm(bands, settings, checkpoint)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    data, cache = _small_pool(tmp_path)
    fixture = Path(__file__).parent / "fixtures/tsrd/config_0_stare.h5"
    for split in ("val", "test"):
        directory = data / "stare" / f"{split}_stare"
        directory.mkdir(parents=True)
        shutil.copyfile(fixture, directory / "config_9.h5")
    config = resolve_setup(ROOT / "environments/train250_composed.json",
                           ROOT / "algorithms/ucb1.json", seed=20261002,
                           episodes=100, checkpoint_every=25)
    config.update(data_root=str(data), cache_root=str(cache), expected_train_configs=2, train_episodes=3)
    config["checkpointing"]["interval_episodes"] = 1
    config["algorithm"] = {"name": "replay_fixture", "module": module.__name__, "api": "public_transitions", "settings": {"update_every": 7}}
    for split in ("val", "test"):
        config["evaluation"][f"{split}_config_ids"] = ["config_9"]
        config["evaluation"][f"expected_{split}_configs"] = 1
        config["evaluation"][f"{split}_composed_worlds"] = [
            {"id": f"{split}_world_000", "source_config_ids": ["config_9"], "world_seed": 17, "emitter_count": 1}]
    run = tmp_path / "run"
    train(config, run)
    decisions = [json.loads(line) for line in (run / "train/steps.jsonl").read_text().splitlines()
                 if "decision_reward" in json.loads(line)]
    assert instances[0].steps == len(decisions)
    assert 900 <= instances[0].steps <= 1800
    assert instances[0].terminals == 3
    assert instances[0].updates == instances[0].steps // 7
    previous_band_by_episode = {}
    for item in decisions:
        episode = item["episode"]
        band = item["action_band"]
        observation = item["observation"]
        previous_band = previous_band_by_episode.get(episode)
        expected_retune = 0.0003 if previous_band is not None and previous_band != band else 0.0
        assert observation["retune_cost_s"] == expected_retune
        previous_band_by_episode[episode] = band
        reward = item["decision_reward"]
        expected_utility = (reward["new_first_intercepts"] / reward["eligible_emitters"]
                            if reward["eligible_emitters"] else 0.0)
        elapsed_seconds = item["dwell_slots"] * 0.05 + expected_retune
        expected_elapsed = (reward["unresolved_emitters_before_action"]
                            / reward["eligible_emitters"] * elapsed_seconds / 30.0
                            if reward["eligible_emitters"] else 0.0)
        expected_false_alarm_cost = (
            reward["false_alarms"] / reward["empty_opportunities"] * elapsed_seconds / 30.0
            if reward["empty_opportunities"] else 0.0)
        assert reward["new_intercept_utility"] == pytest.approx(expected_utility)
        assert reward["elapsed_cost"] == pytest.approx(expected_elapsed)
        assert reward["false_alarm_cost"] == pytest.approx(expected_false_alarm_cost)
        assert reward["reward"] == pytest.approx(
            expected_utility - expected_elapsed - expected_false_alarm_cost)
    index = json.loads((run / "checkpoints/index.json").read_text())["checkpoints"]
    assert [r["file"] for r in index] == ["episode_000001.json", "episode_000002.json", "final.json"]
    assert [r["receiver_steps"] for r in index] == [600, 1200, 1800]
    assert all(r["sha256"] for r in index)
    assert len((run / "train/algorithm_updates.jsonl").read_text().splitlines()) == 3
    evaluate(run, "val", checkpoint="episode_000001.json")
    evaluate(run, "val")
    assert all(not instance.replay for instance in instances[1:])
    with pytest.raises(ValueError, match="VAL"):
        evaluate(run, "test", final=True)
    selected = freeze_selection(run, "episode_000001.json")
    assert selected["checkpoint_file"] == "episode_000001.json"
    with pytest.raises(FileExistsError):
        freeze_selection(run, "final.json")
    with pytest.raises(ValueError, match="frozen"):
        evaluate(run, "test", final=True, checkpoint="final.json")
    report = evaluate(run, "test", final=True)
    assert report["checkpoint_file"] == "episode_000001.json"
    assert (run / "eval/test/checkpoints/episode_000001/normal/summary.json").is_file()
    (run / "checkpoints/episode_000002.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        evaluate(run, "val", checkpoint="episode_000002.json")


def test_run_budget_and_checkpoint_interval_must_be_explicit():
    with pytest.raises(ValueError, match="positive"):
        resolve_setup(ROOT / "environments/train250_composed.json", ROOT / "algorithms/ucb1.json",
                      seed=42, episodes=100, checkpoint_every=0)

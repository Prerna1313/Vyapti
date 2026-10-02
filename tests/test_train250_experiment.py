"""End-to-end contract checks on a tiny TRAIN cache and held-out recording."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.test_train250_cache import _small_pool
from vyapti_simulator.tsrd.experiment import _evaluation_ids, evaluate, plot_run, train
from vyapti_simulator.tsrd.train250_cache import compose_heldout_world
from vyapti_simulator.tsrd.tsrd_adapter import stare_pulse_occupancy
from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment


def test_training_and_fixed_validation_are_reproducible(tmp_path):
    data, cache = _small_pool(tmp_path)
    fixture = Path(__file__).parent / "fixtures/tsrd/config_0_stare.h5"
    val = data / "stare/val_stare/config_9.h5"
    val.parent.mkdir(parents=True)
    shutil.copyfile(fixture, val)
    with pytest.raises(ValueError, match="Expected 50 fixed val configs"):
        _evaluation_ids(data, "val", ["config_9"], expected_count=50)
    template = Path(__file__).parents[1] / "experiments/configs/train250_binary_v1.json"
    config = json.loads(template.read_text(encoding="utf-8"))
    config.update(data_root=str(data), cache_root=str(cache), expected_train_configs=2,
                  train_episodes=1)
    config["evaluation"]["val_config_ids"] = ["config_9"]
    config["evaluation"]["expected_val_configs"] = 1
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    run = tmp_path / "runs/trial"
    checkpoint = train(config_path, run)
    assert checkpoint.is_file()
    episode = json.loads((run / "train/episodes.jsonl").read_text().splitlines()[0])
    assert episode["sources"][0]["source_file"].startswith("stare/train_stare/")
    with pytest.raises(ValueError, match="explicit"):
        evaluate(run, "test")
    report = evaluate(run, "val")
    assert report["summary"]["worlds"] == 1
    row = json.loads((run / "eval/val/normal/per_world.jsonl").read_text().splitlines()[0])
    assert row["config_id"] == "config_9"
    assert row["source_sha256"]
    assert row["scorecard"]["cell_level"]["occupied_cell_count"] >= 0
    steps = [json.loads(line) for line in (run / "eval/val/normal/steps.jsonl").read_text().splitlines()]
    assert len(steps) == 600
    assert all("truth" not in step["observation"] for step in steps)
    assert (run / "plots/train_reward.png") in plot_run(run)
    assert (run / "plots/val_metrics.png").is_file()
    second = tmp_path / "runs/trial-2"
    train(config_path, second)
    repeated = evaluate(second, "val")
    assert repeated["summary"] == report["summary"]
    assert (second / "eval/val/normal/worlds.json").read_text() == (
        run / "eval/val/normal/worlds.json").read_text()
    with pytest.raises(FileExistsError):
        evaluate(run, "val")


def test_composed_heldout_world_is_deterministic_and_split_bound(tmp_path):
    data, _ = _small_pool(tmp_path)
    fixture = Path(__file__).parent / "fixtures/tsrd/config_0_stare.h5"
    directory = data / "stare/val_stare"
    directory.mkdir(parents=True)
    for name in ("config_7", "config_9"):
        shutil.copyfile(fixture, directory / f"{name}.h5")
    args = (data, "val", ["config_7", "config_9"])
    first, sources = compose_heldout_world(*args, world_seed=4, emitter_count=2,
                                           receiver_seed=12)
    second, again = compose_heldout_world(*args, world_seed=4, emitter_count=2,
                                          receiver_seed=12)
    assert sources == again
    assert first.replay_signature() == second.replay_signature()
    assert [first.step(t % 36)[0] for t in range(20)] == [
        second.step(t % 36)[0] for t in range(20)]
    with pytest.raises(FileNotFoundError):
        compose_heldout_world(data, "test", ["config_7"], world_seed=4,
                              emitter_count=1, receiver_seed=12)


def test_all_val_and_test_conditions_write_isolated_audits(tmp_path):
    data, cache = _small_pool(tmp_path)
    fixture = Path(__file__).parent / "fixtures/tsrd/config_0_stare.h5"
    for split in ("val", "test"):
        directory = data / "stare" / f"{split}_stare"
        directory.mkdir(parents=True)
        shutil.copyfile(fixture, directory / "config_9.h5")
    config = json.loads((Path(__file__).parents[1] /
                         "experiments/configs/train250_binary_v1.json").read_text(encoding="utf-8"))
    config.update(data_root=str(data), cache_root=str(cache), expected_train_configs=2,
                  train_episodes=1)
    for split in ("val", "test"):
        config["evaluation"][f"{split}_config_ids"] = ["config_9"]
        config["evaluation"][f"expected_{split}_configs"] = 1
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    run = tmp_path / "run"
    train(path, run)
    # No model-selection access to even the beam TEST variants without final.
    with pytest.raises(ValueError, match="explicit"):
        evaluate(run, "test", condition="all")
    for split in ("val", "test"):
        reports = evaluate(run, split, final=split == "test", condition="all")
        assert set(reports["conditions"]) == {"normal", "beam_periodic", "beam_stochastic"}
        for condition, report in reports["conditions"].items():
            assert report["label"] == f"{split.upper()}_{condition.upper()}"
            assert report["summary"]["worlds"] == 1
            directory = run / "eval" / split / condition
            audit = json.loads((directory / "world_logs/config_9.json").read_text())
            assert audit["condition"] == condition
            assert audit["emitters"][0]["hidden_illumination_intervals"]
            for line in (directory / "steps.jsonl").read_text().splitlines():
                obs = json.loads(line)["observation"]
                assert not set(obs).intersection({"source_label", "emitter_id", "illumination_seed", "beam_state"})
    train_episode = json.loads((run / "train/episodes.jsonl").read_text().splitlines()[0])
    assert "condition" not in train_episode
    assert not (run / "train/world_logs").exists()


def test_slot_and_retune_boundaries_are_synchronized(tmp_path):
    data, cache = _small_pool(tmp_path)
    from vyapti_simulator.tsrd.train250_cache import build_train_pool_from_cache
    pool, _ = build_train_pool_from_cache(data, cache, expected_configs=2)
    cfg = replace(pool.config, detection_probability=1.0,
                  false_alarm_probability=0.0, retune_time_ms=1.0)
    centres = pool.centres_mhz

    def world(toa_us):
        pdws = np.array([[t, centres[0], 1.0, 0.0, -50.0] for t in toa_us])
        occupancy = stare_pulse_occupancy(pdws, centres, pool.halfwidth_mhz, cfg.time_slots)
        result = TSRDStareEnvironment(
            cfg.band_count, cfg.time_slots, occupancy, None, cfg,
            stare_data=pdws, stare_labels=np.zeros(len(pdws), dtype=int),
            band_centres_mhz=centres, passband_halfwidth_mhz=pool.halfwidth_mhz,
        )
        result.reset(seed=8)
        result.step(1)  # First slot on a different band forces a 1 ms retune.
        return result

    before = world([50_999.0])
    assert not before.eligible_recorded_pulse(0, 1, 0.001)
    assert not before.step(0)[0]["hit"]
    at = world([51_000.0])
    assert at.eligible_recorded_pulse(0, 1, 0.001)
    assert at.step(0)[0]["hit"]
    next_slot = world([100_000.0])
    assert not next_slot.eligible_recorded_pulse(0, 1, 0.001)
    assert not next_slot.step(0)[0]["hit"]
    assert next_slot.eligible_recorded_pulse(0, 2, 0.0)

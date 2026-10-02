"""Scientific contract and complete seeded-comparison checks on small recordings."""
import itertools
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest

from tests.test_train250_cache import _small_pool
from vyapti_simulator.system_b.tsrd.algorithm_comparison import run_comparison, finalize_test, paired_report
from vyapti_simulator.system_b.tsrd.frequency_agility import trajectory_features, fit_train_reference
from vyapti_simulator.system_b.tsrd.privileged_scheduler import optimal_expected_oir_schedule, ttfi_distribution
from vyapti_simulator.system_b.tsrd.training_setup import resolve_setup
from vyapti_simulator.system_b.tsrd.experiment import train, evaluate, plot_run

ROOT = Path(__file__).parents[1]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _setup(tmp_path):
    data, cache = _small_pool(tmp_path)
    index_path = cache / "emitter_index.json"
    index = json.loads(index_path.read_text())
    for i, entry in enumerate(index["entries"]):
        label = entry["source_label"]
        toa = np.arange(6, dtype=float) * 50000 + 1000
        frequencies = np.full(6, 1250.0) if i == 0 else np.array([1250, 8250] * 3, dtype=float)
        pulse = np.column_stack([toa, frequencies, np.full(6, 1.0), np.zeros(6), np.full(6, -70)])
        with h5py.File(data / entry["stare_file"], "r+") as source:
            del source["data"], source["labels"]
            source.create_dataset("data", data=pulse)
            source.create_dataset("labels", data=np.full(6, label, dtype=np.int64))
            source["metadata"].attrs["collection_time_s"] = 0.3
            source["metadata/receiver"].attrs["collection_time_s"] = 0.3
        entry["recorded_pulse_count"] = 6
        np.savez_compressed(cache / entry["npz_file"], emitter_labels=np.array([label]),
            **{f"pdw_toa_us_{label}": toa, f"pdw_frequency_mhz_{label}": frequencies,
               f"pdw_pulse_width_{label}": pulse[:, 2], f"pdw_aoa_deg_{label}": pulse[:, 3],
               f"pdw_amplitude_dbm_{label}": pulse[:, 4], f"total_pulses_{label}": 6})
    _write(index_path, index)
    for split in ("val", "test"):
        target = data / "stare" / f"{split}_stare" / "config_9.h5"
        target.parent.mkdir(parents=True)
        shutil.copyfile(data / index["entries"][1]["stare_file"], target)
    environment = json.loads((ROOT / "training_setup/environments/train250_composed.json").read_text())
    environment.update(data_root=str(data), cache_root=str(cache), expected_train_configs=2,
                       evaluation_spec="heldout.json")
    environment["world"]["time_offset_us"] = 0
    evaluation = json.loads((ROOT / "training_setup/evaluation/heldout_worlds.json").read_text())
    evaluation["analysis_protocol_spec"] = str(ROOT / "training_setup/evaluation/mode_b_protocol.json")
    for split in ("val", "test"):
        evaluation[f"expected_{split}_configs"] = 1
        evaluation[f"{split}_config_ids"] = ["config_9"]
        evaluation[f"{split}_composed_worlds"] = [
            {"id": f"{split}_world_000", "source_config_ids": ["config_9"], "emitter_count": 1, "world_seed": 17}]
    env_path = tmp_path / "environment.json"
    _write(env_path, environment)
    _write(tmp_path / "heldout.json", evaluation)
    algorithm = json.loads((ROOT / "training_setup/algorithms/ucb_prior.json").read_text())
    paths = []
    for name in ("first", "second"):
        path = tmp_path / (name + ".json")
        _write(path, dict(algorithm, name=name))
        paths.append(path)
    return env_path, paths, cache


def test_agility_ignores_in_channel_jitter_and_simultaneous_pulse_order():
    stable = trajectory_features([0, 1000, 2000], [1250.1, 1250.2, 1249.9])
    assert stable["observed_transition_count"] == 0
    a = trajectory_features([0, 0, 1000, 1000], [1250, 8250, 8250, 1250])
    assert a["observed_transition_count"] == 0
    assert a["distinct_channels_visited"] == 2
    assert a["simultaneous_states"] == 2


def test_oracle_dynamic_program_matches_exhaustive_retune_constrained_optimum():
    stay = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=bool)
    switched = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=bool)
    world = SimpleNamespace(n_bands=2, n_slots=4, receiver_profile="binary_v1",
        config=SimpleNamespace(retune_time_ms=1.0, detection_probability=0.8),
        eligible_recorded_pulse=lambda band, slot, cost: bool((switched if cost else stay)[band, slot]))
    schedule, expected = optimal_expected_oir_schedule(world)
    def value(actions):
        return sum(0.8 * (stay if t == 0 or actions[t - 1] == b else switched)[b, t] for t, b in enumerate(actions))
    optimum = max(value(actions) for actions in itertools.product(range(2), repeat=4))
    assert expected == pytest.approx(optimum)
    assert value(schedule) == pytest.approx(optimum)


def test_ttfi_preserves_censored_emitters_and_zero_time_intercepts():
    result = ttfi_distribution({"mission_duration_s": 1, "per_emitter": {
        "detected": {"ttfi_ms": 0, "ttfi_censored": False, "followup_ms": 1000},
        "dark": {"ttfi_ms": 1000, "ttfi_censored": True, "followup_ms": 1000}}})
    assert result["emitters"] == 2
    assert result["censoring_fraction"] == 0.5
    assert result["km_cdf"][0]["cdf"] == 0.5
    assert result["rmst_ms"] == 500


def test_optional_predictions_and_train_only_agility_are_integrated(tmp_path, monkeypatch):
    environment, paths, cache = _setup(tmp_path)
    reference = fit_train_reference(cache, expected_configs=2)
    assert reference["fit_split"] == "train"
    assert reference["selected_train_config_ids"] == ["config_0", "config_1"]
    # Hostile held-out trajectories do not affect the TRAIN-fitted boundaries.
    val = tmp_path / "Data/stare/val_stare/config_9.h5"
    with h5py.File(val, "r+") as source:
        source["data"][:, 1] = 1250
    assert fit_train_reference(cache, expected_configs=2) == reference
    module = ModuleType("predicting_fixture")
    class PredictingAlgorithm:
        def reset_episode(self, *, training): pass
        def select_action(self, state, *, training): return 2
        def predict_band_activity(self, state):
            assert not hasattr(state, "truth")
            assert state.previous_observation.time_slot == state.time_slot - 1
            return [0.25] * 36
        def predict_next_intercept_slot(self, state): return state.time_slot + 1
        def observe(self, transition, *, training): pass
        def end_episode(self, *, training): return None
        def save(self, path): path.write_text("{}")
    module.create = lambda **kwargs: PredictingAlgorithm()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = tmp_path / "predicting.json"
    _write(spec, {"schema": "vyapti_algorithm_spec_v1", "name": "predicting", "module": module.__name__,
                  "api": "public_transitions", "settings": {}, "checkpoint_extension": ".json"})
    config = resolve_setup(environment, spec, seed=11, episodes=1, checkpoint_every=1)
    run = tmp_path / "predicting_run"
    train(config, run)
    reports = evaluate(run, "val", condition="all")
    regimes = []
    for condition, report in reports["conditions"].items():
        row = json.loads((run / "eval/val" / condition / "per_world.jsonl").read_text().splitlines()[0])
        regimes.append(row["agility"]["regime"])
        assert row["prediction_metrics"]["band_activity"]["available"]
        assert row["prediction_metrics"]["band_activity"]["prediction_slots"] == 5
        assert report["analysis"]["paired_oracle_oir_gap"] is None or report["analysis"]["paired_oracle_oir_gap"]["world_count"] == 1
        assert (run / "eval/val" / condition / "oracle_logs/val_world_000.json").exists()
    assert len(set(regimes)) == 1
    assert not (run / "eval/test").exists()
    figures = plot_run(run)
    assert len(figures) == 10
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures)
    assert len([path for path in figures if path.name.endswith("_ttfi_cdf.png")]) == 3
    assert len([path for path in figures if path.name.endswith("_agility.png")]) == 3


def test_pilot_runs_three_seeds_pairs_worlds_and_keeps_test_sealed(tmp_path):
    environment, paths, _ = _setup(tmp_path)
    output = tmp_path / "pilot"
    report = run_comparison(environment, paths, output, stage="pilot", episodes=1, checkpoint_every=1)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["training_seeds"] == [11, 22, 33]
    assert len(manifest["entries"]) == 6
    comparison = report["conditions"]["normal"]["paired_comparisons"][0]
    assert comparison["paired_ci"]["world_count"] == 1
    assert comparison["paired_ci"]["mean_difference"] == 0
    assert not any((Path(e["run"]) / "eval/test").exists() for e in manifest["entries"])
    with pytest.raises(ValueError, match="frozen final"):
        finalize_test(output)
    first = manifest["entries"][0]
    config = json.loads((Path(first["run"]) / "config.json").read_text())
    from vyapti_simulator.system_b.tsrd.checkpoints import evaluation_directory
    path = evaluation_directory(first["run"], config, first["checkpoint"], "val", "normal") / "per_world.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    row["source_sha256"] = {"tampered": "different"}
    path.write_text(json.dumps(row) + "\n")
    from vyapti_simulator.system_b.tsrd.experiment import _hash
    summary_path = path.parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["per_world_sha256"] = _hash(path)
    _write(summary_path, summary)
    with pytest.raises(ValueError, match="changed|identical worlds"):
        paired_report(manifest["entries"], manifest["analysis_protocol"], split="val")


def test_final_runs_five_seeds_and_reveals_frozen_test_only_once(tmp_path):
    environment, paths, _ = _setup(tmp_path)
    output = tmp_path / "final"
    run_comparison(environment, paths[:1], output, stage="final", episodes=1, checkpoint_every=1)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["entries"]) == 5
    assert not manifest["test_revealed"]
    entry = manifest["entries"][0]
    checkpoint = Path(entry["run"]) / "checkpoints" / entry["checkpoint"]
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Selected run changed"):
        finalize_test(output)
    assert not json.loads((output / "manifest.json").read_text())["test_revealed"]
    checkpoint.write_bytes(original)
    test = finalize_test(output)
    assert test["split"] == "test"
    assert list(test["conditions"]["normal"]["algorithms"]) == ["first"]
    assert json.loads((output / "manifest.json").read_text())["test_revealed"]
    with pytest.raises(ValueError, match="not revealed TEST"):
        finalize_test(output)


def test_frozen_train_reference_rejects_npz_changes(tmp_path):
    environment, paths, cache = _setup(tmp_path)
    reference = fit_train_reference(cache, expected_configs=2)
    npz = cache / "configs/config_0.npz"
    npz.write_bytes(npz.read_bytes() + b"trailing-change")
    config = resolve_setup(environment, paths[0], seed=11, episodes=1, checkpoint_every=1)
    with pytest.raises(ValueError, match="NPZ changed"):
        train(config, tmp_path / "stale_run", agility_reference=reference)

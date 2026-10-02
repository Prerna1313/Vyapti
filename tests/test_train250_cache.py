"""Integration checks for the NPZ-backed TRAIN pool and its receiver boundary."""

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from vyapti_simulator.tsrd.train250_cache import (
    CachedTSRDTrain250Pool,
    build_stare_evaluation_world,
    load_train250_cache,
    sample_fresh_training_world,
    write_runtime_manifest,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _small_pool(tmp_path):
    data_root = tmp_path / "Data"
    cache_root = tmp_path / "cache"
    fixture = Path(__file__).parent / "fixtures/tsrd/config_0_stare.h5"
    selected = ["config_0", "config_1"]
    entries = []
    for config_id in selected:
        source = data_root / "stare/train_stare" / f"{config_id}.h5"
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture, source)
        with h5py.File(source) as h:
            data = h["data"][:]
            labels = h["labels"][:].reshape(-1)
        label = int(np.unique(labels)[0])
        data = data[labels == label]
        config_npz = cache_root / "configs" / f"{config_id}.npz"
        config_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config_npz,
            emitter_labels=np.array([label]),
            **{
                f"pdw_toa_us_{label}": data[:, 0],
                f"pdw_frequency_mhz_{label}": data[:, 1],
                f"pdw_pulse_width_{label}": data[:, 2],
                f"pdw_aoa_deg_{label}": data[:, 3],
                f"pdw_amplitude_dbm_{label}": data[:, 4],
                f"total_pulses_{label}": len(data),
            },
        )
        entries.append({
            "uid": f"{config_id}::emitter::{label}",
            "config_id": config_id,
            "source_label": label,
            "npz_file": f"configs/{config_id}.npz",
            "stare_file": f"stare/train_stare/{config_id}.h5",
            "recorded_pulse_count": len(data),
        })
    _write_json(cache_root / "manifest.json", {
        "cache_schema": "vyapti_tsrd_emitter_pool_v1",
        "pool_name": "train_250",
        "selected_train_configs": 2,
        "selected_config_ids": selected,
        "selected_emitter_contributions": 2,
        "source_pool_fingerprint": "fixture",
        "receiver_contract": {
            "receiver_profile": "binary_v1",
            "detection_probability": 0.9,
            "false_alarm_probability": 0.05,
            "retune_time_ms": 1.0,
        },
    })
    _write_json(cache_root / "selected_config_ids.json", {"selected_config_ids": selected})
    _write_json(cache_root / "emitter_index.json", {
        "schema": "vyapti_emitter_index_v1", "entries": entries,
    })
    _write_json(cache_root / "config_summaries.json", {
        "configs": [{"config_id": config_id} for config_id in selected],
    })
    return data_root, cache_root


def test_npz_world_preserves_all_source_pdws_and_causal_receiver(tmp_path):
    data_root, cache_root = _small_pool(tmp_path)
    cache = load_train250_cache(cache_root, expected_configs=2)
    pool = CachedTSRDTrain250Pool(data_root, cache)
    first, sources = pool.sample_world(seed=42, emitter_count=2)
    second, again = pool.sample_world(seed=42, emitter_count=2)
    assert sources == again
    assert first.replay_signature() == second.replay_signature()
    world_data, world_labels = first.recorded_pulses
    for source in sources:
        with h5py.File(data_root / source["source_file"]) as h:
            raw = h["data"][:]
            labels = h["labels"][:].reshape(-1)
        expected = raw[labels == source["source_label"]]
        actual = world_data[world_labels == source["world_emitter_id"]]
        np.testing.assert_array_equal(actual, expected)
    first.reset(seed=7)
    second.reset(seed=7)
    observations = [first.step(t % 36)[0] for t in range(12)]
    assert observations == [second.step(t % 36)[0] for t in range(12)]
    assert "pool_id" not in observations[0]
    assert "source_config_id" not in observations[0]
    sampled, _, _, emitter_count = sample_fresh_training_world(
        pool, np.random.default_rng(8)
    )
    assert emitter_count == 1
    assert sampled.n_slots == 600


def test_cache_rejects_stale_source_identity(tmp_path):
    _, cache_root = _small_pool(tmp_path)
    path = cache_root / "emitter_index.json"
    index = json.loads(path.read_text())
    index["entries"][0]["stare_file"] = "stare/train_stare_250/config_0.h5"
    _write_json(path, index)
    with pytest.raises(ValueError, match="Stale or cross-split"):
        load_train250_cache(cache_root, expected_configs=2)


def test_held_out_replay_reads_h5_without_cache(tmp_path):
    data_root, cache_root = _small_pool(tmp_path)
    fixture = Path(__file__).parent / "fixtures/tsrd/config_0_stare.h5"
    val = data_root / "stare/val_stare/config_9.h5"
    val.parent.mkdir(parents=True)
    shutil.copyfile(fixture, val)
    shutil.rmtree(cache_root)
    first = build_stare_evaluation_world(data_root, "val", "config_9", receiver_seed=9)
    second = build_stare_evaluation_world(data_root, "val", "config_9", receiver_seed=9)
    assert first.replay_signature() == second.replay_signature()
    assert [first.step(t % 36)[0] for t in range(12)] == [
        second.step(t % 36)[0] for t in range(12)
    ]
    with pytest.raises(ValueError, match="val or test"):
        build_stare_evaluation_world(data_root, "train", "config_0", receiver_seed=9)


def test_runtime_manifest_records_cache_provenance_without_overwriting(tmp_path):
    _, cache_root = _small_pool(tmp_path)
    cache = load_train250_cache(cache_root, expected_configs=2)
    target = tmp_path / "runs" / "trial" / "runtime_manifest.json"
    write_runtime_manifest(
        cache, target, receiver_profile="binary_v1",
        detection_probability=0.9, false_alarm_probability=0.05,
        retune_time_ms=1.0, training_seed=123,
    )
    content = json.loads(target.read_text(encoding="utf-8"))
    assert content["selected_train_configs"] == 2
    assert content["emitter_contributions"] == 2
    assert content["manifest_sha256"] == cache["manifest_sha256"]
    assert content["emitter_index_sha256"] == cache["emitter_index_sha256"]
    assert content["training_seed"] == 123
    with pytest.raises(FileExistsError):
        write_runtime_manifest(
            cache, target, receiver_profile="binary_v1",
            detection_probability=0.9, false_alarm_probability=0.05,
            retune_time_ms=1.0,
        )

from __future__ import annotations

"""Loader/adapter for the user's already-created TSRD TRAIN-500 cache.

Important:
- DO NOT rerun HDF5 preprocessing from 2,500 files.
- The existing cache is authoritative for the 500-config selection.
- The current TSRDTrainWorldPool API exposes sample_world(), but no cache-backed
  constructor. To preserve its exact world-generation semantics, this adapter
  creates a 500-file symlink corpus and instantiates TSRDTrainWorldPool on ONLY
  those selected 500 configs.
- The saved emitter-contribution cache is used for selection/provenance checks;
  the simulator remains responsible for exact world construction.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

N_BANDS = 36
BAND_CENTRES_MHZ = 250.0 + 500.0 * np.arange(N_BANDS, dtype=np.float64)
RECEIVER_PROFILE = "binary_v1"
AMPLITUDE_MIDPOINT_DB = -90.0
AMPLITUDE_SCALE_DB = 5.0
MAX_OBSERVED_PDWS = 32
RECEIVER_DETECTION_PROBABILITY = 0.90
RECEIVER_FALSE_ALARM_PROBABILITY = 0.05
RECEIVER_RETUNE_TIME_MS = 1.0

CACHE_SCHEMA = "vyapti_tsrd_emitter_pool_v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def config_sort_key(config_id: str) -> tuple[int, str]:
    try:
        return int(str(config_id).split("_")[-1]), str(config_id)
    except ValueError:
        return 10**9, str(config_id)


def fingerprint_cache(manifest: dict[str, Any], selected_config_ids: list[str], emitter_index: dict[str, Any]) -> str:
    payload = {
        "cache_schema": manifest.get("cache_schema"),
        "pool_name": manifest.get("pool_name"),
        "selection_method": manifest.get("selection_method"),
        "selection_seed": manifest.get("selection_seed"),
        "selected_config_ids": selected_config_ids,
        "emitter_uids": [row["uid"] for row in emitter_index["entries"]],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_train500_cache(cache_root: Path) -> dict[str, Any]:
    cache_root = Path(cache_root)
    manifest = load_json(cache_root / "manifest.json")
    if manifest.get("cache_schema") != CACHE_SCHEMA:
        raise RuntimeError(f"Unexpected cache_schema={manifest.get('cache_schema')!r}")
    if int(manifest.get("source_train_configs", -1)) != 2500:
        raise RuntimeError("Cache does not describe the full 2,500-config TRAIN corpus")
    if int(manifest.get("selected_train_configs", -1)) != 500:
        raise RuntimeError("Cache does not describe exactly 500 selected TRAIN configs")

    selected_payload = load_json(cache_root / "selected_config_ids.json")
    selected = sorted(
        [str(x) for x in selected_payload["selected_config_ids"]],
        key=config_sort_key,
    )
    if len(selected) != 500 or len(set(selected)) != 500:
        raise RuntimeError("selected_config_ids.json does not contain exactly 500 unique configs")

    emitter_index = load_json(cache_root / "emitter_index.json")
    entries = list(emitter_index.get("entries", []))
    if len(entries) == 0:
        raise RuntimeError("emitter_index.json contains no emitter contributions")
    uids = [str(row["uid"]) for row in entries]
    if len(uids) != len(set(uids)):
        raise RuntimeError("Duplicate emitter contribution UIDs in cached emitter index")
    entry_configs = {str(row["config_id"]) for row in entries}
    if entry_configs != set(selected):
        missing = sorted(set(selected) - entry_configs, key=config_sort_key)
        extra = sorted(entry_configs - set(selected), key=config_sort_key)
        raise RuntimeError(f"Emitter index/config selection mismatch; missing={missing[:5]}, extra={extra[:5]}")

    config_summaries = load_json(cache_root / "config_summaries.json")
    summaries = {str(x["config_id"]): x for x in config_summaries.get("configs", [])}
    if set(summaries) != set(selected):
        raise RuntimeError("config_summaries.json does not exactly cover the selected 500 configs")

    fingerprint = fingerprint_cache(manifest, selected, emitter_index)
    runtime_pool_root = cache_root / "selected_pool_corpus"
    return {
        "cache_root": cache_root,
        "manifest": manifest,
        "selected_config_ids": selected,
        "emitter_index": entries,
        "config_summaries": summaries,
        "fingerprint": fingerprint,
        "runtime_pool_root": runtime_pool_root,
        "emitter_count_distribution": np.asarray(
            list(Counter(str(row["config_id"]) for row in entries).values()),
            dtype=np.int64,
        ),
    }


def ensure_selected_pool_corpus(corpus_root: Path, cache: dict[str, Any]) -> Path:
    """Build symlinks only; no HDF5 bytes are copied."""
    corpus_root = Path(corpus_root)
    source_train = corpus_root / "stare" / "train_stare"
    target_train = Path(cache["runtime_pool_root"]) / "stare" / "train_stare"
    target_train.mkdir(parents=True, exist_ok=True)

    for config_id in cache["selected_config_ids"]:
        src = source_train / f"{config_id}.h5"
        if not src.exists():
            raise FileNotFoundError(f"Selected source file missing: {src}")
        dst = target_train / src.name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src.resolve())
        except OSError as exc:
            raise RuntimeError(
                "The runtime environment cannot create symlinks. "
                "Use Colab/Kaggle/Linux or create the 500-file selected corpus manually."
            ) from exc

    files = sorted(target_train.glob("config_*.h5"), key=lambda p: int(p.stem.split("_")[-1]))
    if len(files) != 500:
        raise RuntimeError(f"Selected runtime corpus has {len(files)} HDF5 files; expected 500")
    return target_train.parent.parent


def build_train_pool_from_cache(corpus_root: Path, cache_root: Path):
    cache = load_train500_cache(Path(cache_root))
    selected_pool_root = ensure_selected_pool_corpus(Path(corpus_root), cache)

    from vyapti_simulator.tsrd.world_pool import TSRDTrainWorldPool

    pool = TSRDTrainWorldPool(
        selected_pool_root,
        band_centres_mhz=BAND_CENTRES_MHZ,
        receiver_profile=RECEIVER_PROFILE,
        amplitude_midpoint_db=AMPLITUDE_MIDPOINT_DB,
        amplitude_scale_db=AMPLITUDE_SCALE_DB,
        max_observed_pdws=MAX_OBSERVED_PDWS,
        detection_probability=RECEIVER_DETECTION_PROBABILITY,
        false_alarm_probability=RECEIVER_FALSE_ALARM_PROBABILITY,
        retune_time_ms=RECEIVER_RETUNE_TIME_MS,
    )

    actual = {Path(str(c.stare_file)).stem for c in pool.contributions}
    expected = set(cache["selected_config_ids"])
    if actual != expected:
        missing = sorted(expected - actual, key=config_sort_key)
        extra = sorted(actual - expected, key=config_sort_key)
        raise RuntimeError(f"TSRDTrainWorldPool selected-config mismatch; missing={missing[:5]}, extra={extra[:5]}")

    cache_counts = Counter(str(x["config_id"]) for x in cache["emitter_index"])
    pool_counts = Counter(Path(str(c.stare_file)).stem for c in pool.contributions)
    if cache_counts != pool_counts:
        raise RuntimeError("Cached emitter contribution counts do not match TSRDTrainWorldPool contributions")

    cache["pool"] = pool
    return pool, cache


def emitter_count_distribution_from_cache(cache: dict[str, Any]) -> np.ndarray:
    counts = Counter(str(row["config_id"]) for row in cache["emitter_index"])
    values = np.asarray(list(counts.values()), dtype=np.int64)
    if values.size != 500:
        raise RuntimeError(f"Expected 500 per-config emitter counts, got {values.size}")
    return values


def sample_fresh_training_world(pool: Any, rng: np.random.Generator, distribution: np.ndarray | None = None):
    if distribution is None:
        counts = Counter(Path(str(c.stare_file)).stem for c in pool.contributions)
        distribution = np.asarray(list(counts.values()), dtype=np.int64)
    distribution = np.asarray(distribution, dtype=np.int64)
    if distribution.size != 500 or np.any(distribution <= 0):
        raise RuntimeError(f"Expected 500 positive per-config emitter counts, got shape={distribution.shape}")
    emitter_count = int(rng.choice(distribution))
    emitter_count = max(1, min(emitter_count, len(pool.contributions)))
    world_seed = int(rng.integers(1, np.iinfo(np.int32).max))
    env, sources = pool.sample_world(world_seed, emitter_count)
    if len(sources) != emitter_count:
        raise RuntimeError(f"sample_world returned {len(sources)} sources, expected {emitter_count}")
    return env, sources, world_seed, emitter_count


def write_runtime_manifest(cache: dict[str, Any]) -> None:
    save_json(
        Path(cache["cache_root"]) / "runtime_pool_manifest.json",
        {
            "schema": "vyapti_runtime_selected_pool_v1",
            "source_pool_fingerprint": cache["fingerprint"],
            "selected_configs": 500,
            "selected_config_ids": cache["selected_config_ids"],
            "selected_emitter_contributions": len(cache["emitter_index"]),
            "runtime_pool_root": str(cache["runtime_pool_root"]),
            "note": "500 selected HDF5 files are represented by symlinks; no source bytes are duplicated.",
        },
    )

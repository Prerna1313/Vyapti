"""TRAIN-fitted, receiver-scale agility of unmodified TSRD PDW trajectories.

Gunn et al., arXiv:2602.03856v2, describe frequency jitter and simultaneous
frequencies. We measure channel-set transitions, not nominal physical hops.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np

from .train250_cache import load_train250_cache

PAPER = "https://arxiv.org/html/2602.03856v2"
SCHEMA = "tsrd_native_agility_v1"


def validate_train_reference(reference, cache):
    if (reference.get("schema") != SCHEMA or reference.get("fit_split") != "train"
            or reference.get("cache_manifest_sha256") != cache["manifest_sha256"]
            or reference.get("emitter_index_sha256") != cache["emitter_index_sha256"]
            or reference.get("selected_train_config_ids") != cache["selected_config_ids"]):
        raise ValueError("Agility reference must match this TRAIN cache")
    rows = reference.get("train_source_worlds", [])
    if len(rows) != len(cache["selected_config_ids"]) or {r["source_config_id"] for r in rows} != set(cache["selected_config_ids"]):
        raise ValueError("Agility source inventory differs from TRAIN")
    for row in rows:
        path = Path(cache["cache_root"]) / "configs" / f"{row['source_config_id']}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["npz_sha256"]:
            raise ValueError("TRAIN NPZ changed after agility fitting")


def trajectory_features(toa_us, frequency_mhz, *, channel_width_mhz=500.0):
    toa = np.asarray(toa_us, dtype=float)
    frequency = np.asarray(frequency_mhz, dtype=float)
    if (toa.ndim != 1 or frequency.shape != toa.shape or not len(toa)
            or not np.all(np.isfinite(toa)) or not np.all(np.isfinite(frequency))
            or np.any(np.diff(toa) < 0) or not np.isfinite(channel_width_mhz)
            or channel_width_mhz <= 0):
        raise ValueError("Agility requires finite, sorted, nonempty aligned PDWs")
    channels = np.floor(frequency / channel_width_mhz).astype(np.int64)
    if np.any((channels < 0) | (channels >= 64)):
        raise ValueError("Analysis channel IDs must fit the 64-channel bitset")
    # All exactly simultaneous frequencies form one state. Never count a
    # transition solely because multitone pulses occur in another row order.
    starts = np.r_[0, np.flatnonzero(np.diff(toa) > 0) + 1]
    states = np.bitwise_or.reduceat(np.left_shift(np.uint64(1), channels.astype(np.uint64)), starts)
    times_s = toa[starts] * 1e-6
    changes = states[1:] != states[:-1]
    elapsed_s = float(times_s[-1] - times_s[0])
    dt = np.diff(times_s)
    change_times = times_s[1:][changes]
    complete_dwells = np.diff(change_times)  # endpoint runs are censored
    return {
        "observed_channel_transition_rate_hz": int(changes.sum()) / elapsed_s if elapsed_s > 0 else None,
        "observed_transition_count": int(changes.sum()),
        "observed_trace_duration_s": elapsed_s,
        "median_complete_channel_dwell_s": float(np.median(complete_dwells)) if len(complete_dwells) else None,
        "frequency_span_mhz": float(np.ptp(frequency)),
        "distinct_observed_frequencies": int(len(np.unique(frequency))),
        "distinct_channels_visited": int(len(np.unique(channels))),
        "transition_interval_time_fraction": float(dt[changes].sum() / elapsed_s) if elapsed_s > 0 else None,
        "recorded_pulses": len(toa),
        "simultaneous_states": len(starts),
    }


def world_features(data, labels, *, channel_width_mhz=500.0):
    records = {}
    for label in np.unique(labels):
        trace = data[labels == label]
        records[str(int(label))] = trajectory_features(trace[:, 0], trace[:, 1], channel_width_mhz=channel_width_mhz)
    rates = [row["observed_channel_transition_rate_hz"] for row in records.values()
             if row["observed_channel_transition_rate_hz"] is not None]
    return {"emitters": records, "mean_emitter_channel_transition_rate_hz": float(np.mean(rates)) if rates else None}


def classify_rate(value, thresholds):
    if value is None:
        return "unavailable"
    # Ties stay together; empty strata are honestly reported rather than
    # splitting identical-rate worlds to manufacture balanced regimes.
    low, high = thresholds["world_rate_quantiles_hz"]
    return "low" if value <= low else "medium" if value <= high else "high"


def fit_train_reference(cache_root, *, expected_configs=250, channel_width_mhz=500.0):
    cache = load_train250_cache(cache_root, expected_configs=expected_configs)
    by_config = defaultdict(list)
    for row in cache["emitter_index"]:
        by_config[row["config_id"]].append(row)
    rows = []
    for config_id in cache["selected_config_ids"]:
        path = Path(cache["cache_root"]) / "configs" / f"{config_id}.npz"
        metrics = {}
        with np.load(path, allow_pickle=False) as source:
            for row in by_config[config_id]:
                label = str(int(row["source_label"]))
                metrics[label] = trajectory_features(source["pdw_toa_us_" + label],
                    source["pdw_frequency_mhz_" + label], channel_width_mhz=channel_width_mhz)
        rates = [value["observed_channel_transition_rate_hz"] for value in metrics.values()
                 if value["observed_channel_transition_rate_hz"] is not None]
        rows.append({"source_config_id": config_id, "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "mean_emitter_channel_transition_rate_hz": float(np.mean(rates)) if rates else None,
                     "emitters": metrics})
    rates = [row["mean_emitter_channel_transition_rate_hz"] for row in rows
             if row["mean_emitter_channel_transition_rate_hz"] is not None]
    if not rates:
        raise ValueError("TRAIN contains no measurable agility trajectories")
    return {
        "schema": SCHEMA, "fit_split": "train", "dataset_paper": PAPER,
        "local_dataset_revision": cache["manifest"].get("TSRD_revision", "unknown"),
        "channel_width_mhz": channel_width_mhz, "channel_origin_mhz": 0.0,
        "feature": "mean_emitter_channel_transition_rate_hz",
        "quantiles": [1 / 3, 2 / 3],
        "world_rate_quantiles_hz": np.quantile(rates, [1 / 3, 2 / 3]).tolist(),
        "cache_manifest_sha256": cache["manifest_sha256"],
        "emitter_index_sha256": cache["emitter_index_sha256"],
        "source_pool_fingerprint": cache["source_pool_fingerprint"],
        "selected_train_config_ids": cache["selected_config_ids"], "train_source_worlds": rows,
        "limitations": ["receiver-scale channel changes, not nominal physical hop truth",
            "frequency jitter can cross a channel boundary", "missing PDWs can hide transitions",
            "world quantiles fit native TRAIN source-world means; compositions may shift that distribution"],
    }

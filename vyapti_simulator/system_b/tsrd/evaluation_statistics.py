"""Paired world-level and across-seed summaries for Mode-B comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def paired_world_bootstrap(
    metric_a: Mapping[str, float],
    metric_b: Mapping[str, float],
    *,
    name_a: str = "A",
    name_b: str = "B",
    samples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap paired per-world differences ``A - B`` with replacement.

    Inputs must contain exactly the same world IDs. The estimate is the mean
    of per-world differences, so each frozen world has equal weight. Pass
    world-level OIR values here; pooled count ratios have different weighting.
    """
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if not metric_a or set(metric_a) != set(metric_b):
        raise ValueError("paired metrics must have the same nonempty world IDs")

    world_ids = sorted(metric_a)
    a = np.asarray([metric_a[key] for key in world_ids], dtype=np.float64)
    b = np.asarray([metric_b[key] for key in world_ids], dtype=np.float64)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("paired metrics must be finite")

    differences = a - b
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(samples, len(differences)))
    estimates = differences[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return {
        "estimand": "mean_paired_world_difference_A_minus_B",
        "metric_a": name_a,
        "metric_b": name_b,
        "world_count": len(world_ids),
        "world_ids": world_ids,
        "mean_difference": float(differences.mean()),
        "confidence": confidence,
        "confidence_interval": [float(low), float(high)],
        "bootstrap_method": "paired_percentile_world_resampling",
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def summarize_training_seeds(values: Sequence[float], *, seeds: Sequence[int]) -> dict:
    """Report between-training-seed variation separately from world CIs."""
    if len(values) != len(seeds) or not values:
        raise ValueError("provide one metric value per nonempty training-seed list")
    if len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("training seeds must be unique nonnegative integers")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("seed metric values must be finite")
    return {
        "training_seeds": list(seeds),
        "values": array.tolist(),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "seed_count": len(array),
    }

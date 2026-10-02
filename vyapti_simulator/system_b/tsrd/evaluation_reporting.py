"""Stratified world reports and paired uncertainty for frozen replay results."""
from __future__ import annotations

import numpy as np

from .evaluation_statistics import paired_world_bootstrap
from .privileged_scheduler import ttfi_distribution


def analysis_summary(rows, protocol, summarize_scores):
    strata = {}
    for regime in ("low", "medium", "high", "unavailable"):
        selected = [row for row in rows if row["agility"]["regime"] == regime]
        strata[regime] = {"world_ids": [r["config_id"] for r in selected],
                          **summarize_scores([r["scorecard"] for r in selected])}
    oracle = {r["config_id"]: r["oracle_comparison"]["privileged_oir"] for r in rows
              if r["oracle_comparison"]["oir_gap"] is not None}
    policy = {r["config_id"]: r["oracle_comparison"]["policy_oir"] for r in rows if r["config_id"] in oracle}
    settings = protocol["uncertainty"]
    gap = paired_world_bootstrap(oracle, policy, name_a="privileged", name_b="policy",
        samples=settings["resamples"], seed=settings.get("bootstrap_seed", 20261002),
        confidence=settings["confidence"]) if oracle else None
    distributions = {}
    for side in ("policy", "privileged"):
        records = {f"{r['config_id']}:{emitter}": record for r in rows
                   for emitter, record in r["oracle_comparison"][side + "_ttfi"]["records"].items()}
        score = {"per_emitter": records, "mission_duration_s": 0.0}
        distributions[side] = ttfi_distribution(score)
    ratios = [r["oracle_comparison"]["ttfi_rmst_ratio_policy_over_privileged"] for r in rows
              if r["oracle_comparison"]["ttfi_rmst_ratio_policy_over_privileged"] is not None]
    feature_mean = lambda r, key: float(np.mean(values)) if (values := [emitter[key] for emitter in
        r["agility"]["emitters"].values() if emitter[key] is not None]) else None
    points = [{"world_id": r["config_id"], "agility_regime": r["agility"]["regime"],
               "mean_emitter_channel_transition_rate_hz": r["agility"]["mean_emitter_channel_transition_rate_hz"],
               "mean_frequency_span_mhz": feature_mean(r, "frequency_span_mhz"),
               "mean_complete_channel_dwell_s": feature_mean(r, "median_complete_channel_dwell_s"),
               "mean_distinct_channels_visited": feature_mean(r, "distinct_channels_visited"),
               "emitter_count": len(r["agility"]["emitters"]),
               "oir": r["scorecard"]["cell_level"]["occupied_cell_detection_ratio"],
               "policy_ttfi_rmst_ms": r["oracle_comparison"]["policy_ttfi"]["rmst_ms"],
               "oir_gap": r["oracle_comparison"]["oir_gap"]} for r in rows]
    heads = [r["prediction_metrics"]["band_activity"] for r in rows
             if r["prediction_metrics"]["band_activity"]["available"]]
    prediction = {"available": False, "reason": "algorithms emitted no scored band predictions"}
    if heads:
        weights = [h["prediction_slots"] * h["band_count"] for h in heads]
        calibration = []
        for i in range(len(heads[0]["calibration"])):
            bins = [h["calibration"][i] for h in heads]
            count = sum(b["count"] for b in bins)
            calibration.append({"lower": bins[0]["lower"], "upper": bins[0]["upper"], "count": count,
                "mean_probability": sum(b["count"] * (b["mean_probability"] or 0) for b in bins) / count if count else None,
                "observed_activity_rate": sum(b["count"] * (b["observed_activity_rate"] or 0) for b in bins) / count if count else None})
        prediction = {"available": True, "worlds_with_predictions": len(heads),
            "brier_score": float(np.average([h["brier_score"] for h in heads], weights=weights)),
            "log_loss": float(np.average([h["log_loss"] for h in heads], weights=weights)), "calibration": calibration}
    return {"agility_strata": strata, "agility_and_density_points": points,
            "band_prediction_metrics": prediction,
            "paired_oracle_oir_gap": gap, "oracle_oir_gap_unavailable_worlds": len(rows) - len(oracle),
            "ttfi_distributions": distributions, "mean_world_ttfi_rmst_ratio": float(np.mean(ratios)) if ratios else None,
            "worlds_with_defined_ttfi_ratio": len(ratios)}

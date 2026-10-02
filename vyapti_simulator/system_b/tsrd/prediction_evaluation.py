"""Optional predictions after observing slot t, targeting t+1; no truth input."""
from __future__ import annotations

import numpy as np


def collect_prediction(policy, state, *, bands):
    result = {"decision_slot": state.time_slot - 1, "target_slot": state.time_slot}
    hook = getattr(policy, "predict_band_activity", None)
    probabilities = hook(state) if callable(hook) else None
    if probabilities is not None:
        array = np.asarray(probabilities, dtype=float)
        if array.shape != (bands,) or not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
            raise ValueError("Prediction head must emit one finite probability per band")
        result["band_activity_probability"] = array.tolist()
    hook = getattr(policy, "predict_next_intercept_slot", None)
    intercept = hook(state) if callable(hook) else None
    if intercept is not None:
        if not np.isfinite(intercept) or intercept < result["target_slot"]:
            raise ValueError("Intercept prediction must be a finite future absolute slot")
        result["next_intercept_slot"] = float(intercept)
    return result if len(result) > 2 else None


def score_predictions(rows, world, trajectory, *, calibration_bins=10):
    available = [r for r in rows if "band_activity_probability" in r and r["target_slot"] < world.n_slots]
    result = {"band_activity": {"available": False, "reason": "no explicit in-horizon band predictions"},
              "intercept_time": {"available": False, "reason": "no explicit uncensored intercept predictions"}}
    if available:
        probabilities = np.asarray([r["band_activity_probability"] for r in available])
        truth = world.occupancy_grid[:, [r["target_slot"] for r in available]].T.astype(float)
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        decisions = probabilities >= 0.5
        tp = ((truth == 1) & decisions).sum(axis=0)
        fp = ((truth == 0) & decisions).sum(axis=0)
        fn = ((truth == 1) & ~decisions).sum(axis=0)
        divide = lambda x, y: [float(a / b) if b else None for a, b in zip(x, y)]
        calibration = []
        for i in range(calibration_bins):
            mask = (probabilities >= i / calibration_bins) & (probabilities < (i + 1) / calibration_bins)
            if i == calibration_bins - 1:
                mask |= probabilities == 1
            calibration.append({"lower": i / calibration_bins, "upper": (i + 1) / calibration_bins,
                "count": int(mask.sum()), "mean_probability": float(probabilities[mask].mean()) if mask.any() else None,
                "observed_activity_rate": float(truth[mask].mean()) if mask.any() else None})
        result["band_activity"] = {"available": True, "target": "next_slot_visible_recorded_band_activity",
            "prediction_slots": len(available), "band_count": world.n_bands,
            "brier_score": float(np.mean((probabilities - truth) ** 2)),
            "log_loss": float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped))),
            "calibration": calibration, "decision_threshold": 0.5,
            "per_band_precision": divide(tp, tp + fp), "per_band_recall": divide(tp, tp + fn),
            "per_band_f1": divide(2 * tp, 2 * tp + fp + fn)}
    hits = [step.time_slot for step in trajectory if step.observation["hit"] and
            world.eligible_recorded_pulse(step.action, step.time_slot, step.observation["retune_cost_s"])]
    errors, censored = [], 0
    supplied = [r for r in rows if "next_intercept_slot" in r]
    for row in supplied:
        index = np.searchsorted(hits, row["target_slot"])
        if index == len(hits):
            censored += 1
        else:
            errors.append(abs(row["next_intercept_slot"] - hits[index]))
    if supplied:
        result["intercept_time"] = {"available": bool(errors), "predictions": len(supplied),
            "uncensored_scored": len(errors), "censored_unscored": censored,
            "target": "next_true_intercept_of_this_policy_at_or_after_target_slot",
            "mean_absolute_error_slots": float(np.mean(errors)) if errors else None,
            "median_absolute_error_slots": float(np.median(errors)) if errors else None,
            "p95_absolute_error_slots": float(np.percentile(errors, 95)) if errors else None}
    return result

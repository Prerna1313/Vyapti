"""Evaluator-only truth-aware scheduling and distinct OIR/TTFI comparisons."""
from __future__ import annotations

import numpy as np

from vyapti_simulator.core.metrics import TrajectoryStep
from .replay_scorecard import score_recorded_replay
from .evaluation_illumination import annotate_illumination_score


def optimal_expected_oir_schedule(world):
    """Finite-horizon DP for the active binary receiver, with retune loss.

    The objective is expected true positive band-slot looks. Only recorded
    pulse truth is privileged; future noise draws are never inspected. This
    objective does not optimize emitter discovery or TTFI.
    """
    if world.receiver_profile != "binary_v1":
        raise ValueError("Expected-OIR oracle currently requires binary_v1")
    stay = np.zeros((world.n_bands, world.n_slots), dtype=bool)
    switch = np.zeros_like(stay)
    retune_s = world.config.retune_time_ms / 1000.0
    for slot in range(world.n_slots):
        for band in range(world.n_bands):
            stay[band, slot] = world.eligible_recorded_pulse(band, slot, 0.0)
            switch[band, slot] = world.eligible_recorded_pulse(band, slot, retune_s)
    pd = float(world.config.detection_probability)
    future = np.zeros(world.n_bands)
    choices = np.zeros((world.n_slots, world.n_bands), dtype=int)
    for slot in range(world.n_slots - 1, -1, -1):
        switching = pd * switch[:, slot] + future
        values = np.broadcast_to(switching, (world.n_bands, world.n_bands)).copy()
        continuing = pd * stay[:, slot] + future
        np.fill_diagonal(values, continuing)
        choice = values.argmax(axis=1)
        # Prefer staying when equally good; argmax otherwise uses lowest band.
        previous = np.arange(world.n_bands)
        choice[continuing >= values[previous, choice]] = previous[continuing >= values[previous, choice]]
        choices[slot] = choice
        if slot == 0:
            first_values = pd * stay[:, 0] + future  # first dwell has no retune
        future = values[previous, choice]
    first = int(np.argmax(first_values))
    schedule = [first]
    for slot in range(1, world.n_slots):
        schedule.append(int(choices[slot, schedule[-1]]))
    return schedule, float(first_values[first])


def ttfi_distribution(score, *, restriction_ms=None):
    """Kaplan-Meier CDF and RMST without dropping never-intercepted emitters."""
    records = score.get("illumination_stress", {}).get("source_relative_ttfi", score["per_emitter"])
    durations = np.asarray([r["ttfi_ms"] for r in records.values()], dtype=float)
    events = np.asarray([not r["ttfi_censored"] for r in records.values()], dtype=bool)
    followups = [r.get("followup_ms", score["mission_duration_s"] * 1000 - r.get("emitter_first_emission_ms", 0))
                 for r in records.values()]
    tau = float(min(followups)) if followups else None
    if restriction_ms is not None:
        tau = min(tau, float(restriction_ms)) if tau is not None else None
    survival, previous, area = 1.0, 0.0, 0.0
    cdf = []
    for time in np.unique(durations):
        risk = int(np.sum(durations >= time))
        failures = int(np.sum((durations == time) & events))
        censored = int(np.sum((durations == time) & ~events))
        if tau is not None and previous < tau:
            area += survival * (min(float(time), tau) - previous)
            previous = min(float(time), tau)
        survival *= 1.0 - failures / risk
        cdf.append({"time_ms": float(time), "cdf": float(1 - survival),
                    "at_risk": risk, "intercepts": failures, "censored": censored})
    if tau is not None and previous < tau:
        area += survival * (tau - previous)
    return {"records": records, "km_cdf": cdf, "emitters": len(records),
            "censoring_fraction": float(np.mean(~events)) if len(events) else None,
            "rmst_ms": float(area) if tau is not None else None, "restriction_ms": tau}


def evaluate_privileged(world, *, receiver_seed, illumination):
    schedule, expected_hits = optimal_expected_oir_schedule(world)
    world.reset(seed=receiver_seed)
    trajectory = []
    for slot, band in enumerate(schedule):
        observation, _ = world.step(band)
        trajectory.append(TrajectoryStep(action=band, time_slot=slot, observation=observation))
    score = score_recorded_replay(world, trajectory)
    annotate_illumination_score(score, illumination, world.config.slot_duration_s())
    return {"label": "privileged_recorded_truth_expected_oir_scheduler",
            "method": "finite_horizon_dynamic_program_with_retune_loss",
            "optimality_scope": "expected band-slot OIR under this binary_v1 replay receiver only",
            "absolute_physical_optimum": False, "future_receiver_noise_visible": False,
            "expected_true_detections": expected_hits, "actions": schedule, "scorecard": score}


def compare_privileged(policy_score, oracle):
    oracle_score = oracle["scorecard"]
    policy_oir = policy_score["cell_level"]["occupied_cell_detection_ratio"]
    oracle_oir = oracle_score["cell_level"]["occupied_cell_detection_ratio"]
    policy_ttfi = ttfi_distribution(policy_score)
    oracle_ttfi = ttfi_distribution(oracle_score)
    taus = [r["restriction_ms"] for r in (policy_ttfi, oracle_ttfi) if r["restriction_ms"] is not None]
    tau = min(taus) if len(taus) == 2 else None
    if tau is not None:
        policy_ttfi = ttfi_distribution(policy_score, restriction_ms=tau)
        oracle_ttfi = ttfi_distribution(oracle_score, restriction_ms=tau)
    denominator = oracle_ttfi["rmst_ms"]
    return {
        "oir_gap": oracle_oir - policy_oir if policy_oir is not None and oracle_oir is not None else None,
        "policy_oir": policy_oir, "privileged_oir": oracle_oir,
        "oir_gap_can_be_negative_due_to_receiver_noise": True,
        "ttfi_rmst_ratio_policy_over_privileged": policy_ttfi["rmst_ms"] / denominator
            if tau is not None and denominator is not None and denominator > 0 else None,
        "ttfi_ratio_unavailable_reason": None if tau is not None and denominator is not None and denominator > 0
            else "no common follow-up or privileged RMST is zero",
        "policy_ttfi": policy_ttfi, "privileged_ttfi": oracle_ttfi,
        "ttfi_objective": "secondary; expected-OIR oracle does not optimize TTFI",
    }

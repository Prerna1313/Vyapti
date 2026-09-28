"""Offline, label-aware scorecard for TSRD recorded-pulse receiver replays."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _km_quantile(durations: list[float], events: list[bool], quantile: float) -> float | None:
    if not durations:
        return None
    times = np.asarray(durations, dtype=float)
    occurred = np.asarray(events, dtype=bool)
    survival = 1.0
    for time in np.unique(times):
        at_risk = int(np.sum(times >= time))
        failures = int(np.sum((times == time) & occurred))
        if failures:
            survival *= 1.0 - failures / at_risk
            if 1.0 - survival >= quantile - 1e-12:
                return float(time)
    return None


def _km_restricted_mean(durations: list[float], events: list[bool]) -> tuple[float | None, float | None]:
    """KM restricted mean through the longest observed follow-up, in ms."""
    if not durations:
        return None, None
    times = np.asarray(durations, dtype=float)
    occurred = np.asarray(events, dtype=bool)
    horizon = float(np.max(times))
    survival = 1.0
    previous = 0.0
    area = 0.0
    for time in np.unique(times):
        area += survival * (float(time) - previous)
        at_risk = int(np.sum(times >= time))
        failures = int(np.sum((times == time) & occurred))
        survival *= 1.0 - failures / at_risk
        previous = float(time)
    return float(area), horizon


def score_recorded_replay(
    env: Any, trajectory: Sequence[Any],
    decision_visits: Sequence[tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    """Score one full episode; no result from this function is scheduler-visible."""
    if len(trajectory) != env.n_slots:
        raise ValueError("TSRD scorecard requires a full, untruncated episode")
    data, labels = env.recorded_pulses
    if data is None or labels is None:
        raise ValueError("Emitter scorecard requires labelled recorded PDWs")

    grid = env.hidden_truth.grid
    emitter_ids = [int(c.emitter_id) for c in env.hidden_truth.emitter_configs]
    if len(set(emitter_ids)) != len(emitter_ids):
        raise ValueError("Emitter IDs must be unique within a replay world")
    n_emitters = len(emitter_ids)
    slot_s = env.config.slot_duration_s()
    mission_s = env.n_slots * slot_s
    emitter_slots = grid.any(axis=1)
    discovered = np.zeros((n_emitters, env.n_slots), dtype=bool)
    eligible_emitter_slots = np.zeros_like(discovered)
    occupied_cells = grid.any(axis=0)
    eligible_cells = 0
    true_hits = 0
    false_alarms = 0
    covered_recorded_pulses = 0
    captured_recorded_pulses = 0
    pulse_resolved = getattr(env, "receiver_profile", "binary_v1") == "pdw_v2"
    true_hit_slots = np.zeros(env.n_slots, dtype=bool)
    eligible_slots = np.zeros(env.n_slots, dtype=bool)
    first_intercept = np.full(n_emitters, -1, dtype=int)
    toa = data[:, 0].astype(float, copy=False)
    freq = data[:, 1].astype(float, copy=False)
    in_mission = np.isfinite(toa) & (toa >= 0) & (toa < mission_s * 1e6)
    in_mission_pulses = int(np.count_nonzero(in_mission))
    centres, halfwidth = env.receiver_geometry
    label_indices = {label: i for i, label in enumerate(emitter_ids)}
    first_emission_us = np.full(n_emitters, np.inf)
    if len(labels):
        pulse_indices = np.fromiter(
            (label_indices[int(label)] for label in labels), dtype=np.intp, count=len(labels)
        )
        np.minimum.at(first_emission_us, pulse_indices, toa)

    for expected_slot, step in enumerate(trajectory):
        band = int(step.action)
        slot = int(step.time_slot)
        if not (0 <= band < env.n_bands and slot == expected_slot
                and int(step.observation["time_slot"]) == expected_slot):
            raise ValueError("TSRD trajectory must cover each slot exactly once in order")
        retune_s = float(step.observation.get("retune_cost_s", 0.0))
        start_us = (slot * slot_s + retune_s) * 1e6
        end_us = (slot + 1) * slot_s * 1e6
        lo = int(np.searchsorted(toa, start_us, side="left"))
        hi = int(np.searchsorted(toa, end_us, side="left"))
        matching = np.abs(freq[lo:hi] - centres[band]) <= halfwidth
        covered_recorded_pulses += int(matching.sum())
        present_ids = np.unique(labels[lo:hi][matching])
        if pulse_resolved:
            captured = env.recorded_capture_indices(slot)
            captured_recorded_pulses += len(captured)
            detected_ids = np.unique(labels[captured])
        else:
            detected_ids = present_ids
        for emitter_id in present_ids:
            eligible_emitter_slots[label_indices[int(emitter_id)], slot] = True
        eligible = bool(len(present_ids))
        eligible_cells += int(eligible)
        eligible_slots[slot] = eligible
        if not step.observation["hit"]:
            continue
        if eligible and len(detected_ids):
            true_hits += 1
            true_hit_slots[slot] = True
            for emitter_id in detected_ids:
                index = label_indices[int(emitter_id)]
                discovered[index, slot] = True
                if first_intercept[index] < 0:
                    first_intercept[index] = slot
        else:
            false_alarms += 1

    records: dict[str, dict[str, Any]] = {}
    durations: list[float] = []
    events: list[bool] = []
    observed_durations: list[float] = []
    first_discoveries = 0
    for i, emitter_id in enumerate(emitter_ids):
        opportunities = np.flatnonzero(emitter_slots[i])
        if not len(opportunities):
            continue
        first_slot = int(opportunities[0])
        first_emission_ms = float(first_emission_us[i] / 1000.0)
        intercepted = first_intercept[i] >= 0
        first_ms = float(first_intercept[i] * slot_s * 1000.0) if intercepted else None
        duration_ms = (
            float((first_intercept[i] - first_slot) * slot_s * 1000.0)
            if intercepted else float((env.n_slots - first_slot) * slot_s * 1000.0)
        )
        durations.append(duration_ms)
        events.append(bool(intercepted))
        if intercepted:
            first_discoveries += 1
            observed_durations.append(duration_ms)
        records[str(emitter_id)] = {
            "emitter_first_emission_ms": first_emission_ms,
            "emitter_first_opportunity_slot": first_slot,
            "emitter_first_intercept_slot": int(first_intercept[i]) if intercepted else None,
            "emitter_first_intercept_ms": first_ms,
            "emitter_intercepted": bool(intercepted),
            "ttfi_ms": duration_ms,
            "ttfi_censored": not bool(intercepted),
        }

    n_eligible_emitters = len(records)
    emitter_slot_count = int(emitter_slots.sum())
    detected_emitter_slots = int(discovered.sum())
    occupied_cell_count = int(occupied_cells.sum())
    empty_selected = len(trajectory) - eligible_cells
    restricted_mean_ms, restriction_ms = _km_restricted_mean(durations, events)
    if decision_visits is None:
        decision_visits = [(t, 1, int(step.action)) for t, step in enumerate(trajectory)]
    cursor = 0
    for start, slots, band in decision_visits:
        if start != cursor or slots not in (1, 2) or not 0 <= band < env.n_bands:
            raise ValueError("Decision visits must partition the base-slot timeline")
        if any(int(trajectory[t].action) != band for t in range(start, min(start + slots, env.n_slots))):
            raise ValueError("Decision visit band disagrees with receiver looks")
        cursor += slots
    if cursor != env.n_slots:
        raise ValueError("Decision visits must cover the full mission")
    actions = np.asarray([band for _, _, band in decision_visits], dtype=int)
    starts = np.asarray([start for start, _, _ in decision_visits], dtype=int)
    revisit_intervals_s = np.concatenate([
        np.diff(starts[actions == band]) * slot_s
        for band in range(env.n_bands)
        if np.count_nonzero(actions == band) >= 2
    ]) if any(np.count_nonzero(actions == band) >= 2 for band in range(env.n_bands)) else np.empty(0)
    true_hit_dwells = sum(bool(np.any(true_hit_slots[start:start + slots])) for start, slots, _ in decision_visits)
    empty_dwells = sum(not bool(np.any(eligible_slots[start:start + slots])) for start, slots, _ in decision_visits)
    per_emitter_pd = {
        str(emitter_id): _rate(int(discovered[i].sum()), int(eligible_emitter_slots[i].sum()))
        for i, emitter_id in enumerate(emitter_ids)
    }
    attribution = ("captured_recorded_pdw_labels" if pulse_resolved
                   else "upper_bound_from_band_positive")
    return {
        "metric_contract_version": (
            "tsrd_recorded_pulse_emitter_v4" if pulse_resolved
            else "tsrd_recorded_pulse_emitter_v3"
        ),
        "truth_basis": (
            "recorded_stare_pdws; exact captured-PDW attribution"
            if pulse_resolved else
            "recorded_stare_pdws; simultaneous labels optimistically credited"
        ),
        "mission_duration_s": mission_s,
        "per_emitter": records,
        "ttfi": {
            "mean_observed_ms": float(np.mean(observed_durations)) if observed_durations else None,
            "median_observed_ms": float(np.median(observed_durations)) if observed_durations else None,
            "p90_observed_ms": float(np.percentile(observed_durations, 90)) if observed_durations else None,
            "km_median_ms": _km_quantile(durations, events, 0.5),
            "km_p90_ms": _km_quantile(durations, events, 0.9),
            "censored_mean_ttfi_s": restricted_mean_ms / 1000.0 if restricted_mean_ms is not None else None,
            "censored_median_ttfi_s": (
                value / 1000.0 if (value := _km_quantile(durations, events, 0.5)) is not None else None
            ),
            "censored_p95_ttfi_s": (
                value / 1000.0 if (value := _km_quantile(durations, events, 0.95)) is not None else None
            ),
            "censored_mean_restriction_s": restriction_ms / 1000.0 if restriction_ms is not None else None,
            "censoring_fraction": _rate(n_eligible_emitters - first_discoveries, n_eligible_emitters),
        },
        "emitter_interception": {
            "eligible_emitters": n_eligible_emitters,
            "intercepted_emitters": first_discoveries,
            "unique_emitter_interception_rate": _rate(first_discoveries, n_eligible_emitters),
            "unique_emitter_discoveries_per_s": _rate(first_discoveries, mission_s),
            "unique_emitter_discoveries_per_sec": _rate(first_discoveries, mission_s),
            "per_emitter_opportunity_pd": per_emitter_pd,
            "pooled_per_emitter_opportunity_pd": _rate(
                int(discovered.sum()), int(eligible_emitter_slots.sum())
            ),
            "eligible_observed_emitter_slot_count": int(eligible_emitter_slots.sum()),
            "attribution_limit": attribution,
        },
        "illumination": {
            "emitter_slot_opportunity_count": emitter_slot_count,
            "detected_emitter_slot_opportunity_count": detected_emitter_slots,
            "illumination_interception_ratio": _rate(detected_emitter_slots, emitter_slot_count),
        },
        "cell_level": {
            "occupied_cell_count": occupied_cell_count,
            "detected_occupied_cell_count": true_hits,
            "occupied_cell_detection_ratio": _rate(true_hits, occupied_cell_count),
            "conditional_pd": _rate(true_hits, eligible_cells),
            "true_pfa": _rate(false_alarms, empty_selected),
            "eligible_selected_cells": eligible_cells,
            "empty_selected_cells": empty_selected,
        },
        "event_level": {
            "repeated_true_detections_per_s": _rate(detected_emitter_slots - first_discoveries, mission_s),
            "repeated_emitter_slot_detections": detected_emitter_slots - first_discoveries,
            "false_alarms_per_s": _rate(false_alarms, mission_s),
            "true_detections": true_hits,
            "false_alarms": false_alarms,
            "repeated_true_detection_events_per_sec": _rate(
                detected_emitter_slots - first_discoveries, mission_s
            ),
            "attribution_limit": attribution,
        },
        "pulse_level": {
            "time_window_basis": "assumed_replay_toa_window_[0,mission_duration)",
            "time_window_start_us": 0.0,
            "time_window_end_us": mission_s * 1e6,
            "physical_pulse_interception_ratio": None,
            "physical_pulse_interception_ratio_unavailable_reason": (
                "TSRD stare PDWs omit some emitted pulses; physical emission truth is unavailable"
            ),
            "offered_pulse_capture_ratio": None,
            "offered_pulse_capture_ratio_unavailable_reason": (
                "TSRD stare PDWs omit some emitted pulses; the offered physical pulse count is unknown"
                if pulse_resolved else
                "A band-level detector positive does not identify captured individual pulses"
            ),
            "recorded_pulse_coverage_ratio": _rate(covered_recorded_pulses, in_mission_pulses),
            "recorded_pulse_detected_ratio": (
                _rate(captured_recorded_pulses, in_mission_pulses) if pulse_resolved else None
            ),
            "captured_recorded_pulses": (
                captured_recorded_pulses if pulse_resolved else None
            ),
            "recorded_pulses": len(data),
            "recorded_pulses_in_mission": in_mission_pulses,
            "recorded_pulses_outside_mission": len(data) - in_mission_pulses,
            "covered_recorded_pulses": covered_recorded_pulses,
        },
        "revisit": {
            "visit_counts": np.bincount(actions, minlength=env.n_bands).astype(int).tolist(),
            "mean_revisit_s": float(np.mean(revisit_intervals_s)) if len(revisit_intervals_s) else None,
            "p95_revisit_s": float(np.percentile(revisit_intervals_s, 95)) if len(revisit_intervals_s) else None,
            "max_revisit_s": float(np.max(revisit_intervals_s)) if len(revisit_intervals_s) else None,
            "switch_rate": _rate(int(np.sum(actions[1:] != actions[:-1])), len(actions) - 1),
            "switch_count": int(np.sum(actions[1:] != actions[:-1])),
            "revisit_interval_count": int(len(revisit_intervals_s)),
            "intervals_s": revisit_intervals_s.tolist(),
            "unvisited_bands_excluded": int(np.sum([not np.any(actions == band) for band in range(env.n_bands)])),
        },
        "decision_level": {
            "total_dwells": len(decision_visits),
            "base_slots_consumed": env.n_slots,
            "true_hit_dwells": true_hit_dwells,
            "empty_dwells": empty_dwells,
            "true_detection_yield_per_dwell": _rate(true_hit_dwells, len(decision_visits)),
            "empty_scan_fraction": _rate(empty_dwells, len(decision_visits)),
        },
        "reward": {
            "detector_positive_total": true_hits + false_alarms,
            "false_alarm_aware_total": true_hits - false_alarms,
            "detector_positive_per_decision": _rate(true_hits + false_alarms, len(trajectory)),
            "false_alarm_aware_per_decision": _rate(true_hits - false_alarms, len(trajectory)),
        },
    }

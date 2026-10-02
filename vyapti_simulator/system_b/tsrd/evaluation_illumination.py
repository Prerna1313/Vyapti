"""Hidden, continuous-time illumination stress tests for VAL and TEST only.

These gates add controlled visibility assumptions to received STARE PDWs.
They do not reconstruct emitted pulses or infer antenna state from the data.
Frequency, ToA, PW, AoA and amplitude of retained pulses are unchanged.

Periodic and two-state CTMC schedules follow the model families discussed by
Clarkson, IET RSN (2019), DOI 10.1049/iet-rsn.2018.5668. Joint frequency and
spatial stress follows Teissier et al. (2026), DOI 10.1109/TAES.2026.3660987.
The numerical defaults below are stress assumptions, not fitted paper values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .tsrd_adapter import stare_pulse_occupancy
from .tsrd_environment import TSRDStareEnvironment


@dataclass(frozen=True, slots=True)
class IlluminationSettings:
    mode: str = "normal"
    period_s: float = 2.0
    duty_cycle: float = 0.1
    mean_on_s: float = 0.2
    mean_off_s: float = 1.8

    def __post_init__(self):
        if self.mode not in {"normal", "beam_periodic", "beam_stochastic"}:
            raise ValueError("Unknown held-out illumination condition")
        values = (self.period_s, self.duty_cycle, self.mean_on_s, self.mean_off_s)
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("Illumination settings must be finite numbers")
        if (self.period_s <= 0 or not 0 < self.duty_cycle <= 1
                or self.mean_on_s <= 0 or self.mean_off_s <= 0):
            raise ValueError("Invalid illumination period, duty cycle, or residence time")


def _emitter_rng(seed: int, label: int):
    key = hashlib.sha256(f"illumination:{seed}:{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "little"))


def _intervals(settings: IlluminationSettings, horizon_s: float, rng):
    """Return a complete [0,horizon) partition with explicit hidden state."""
    if settings.mode == "normal":
        return [{"start_s": 0.0, "end_s": horizon_s, "illuminated": True}], {}
    if settings.mode == "beam_periodic":
        phase_s = float(rng.uniform(0, settings.period_s))
        on_s = settings.period_s * settings.duty_cycle
        intervals = []
        cursor = 0.0
        # Include the preceding rotation because its illuminated window may
        # cross mission start. Half-open windows define transition boundaries.
        rotations = int(math.ceil(horizon_s / settings.period_s)) + 2
        if rotations > 1_000_000:
            raise ValueError("Illumination schedule exceeds transition budget")
        for cycle in range(-1, rotations):
            start = max(0.0, phase_s + cycle * settings.period_s)
            end = min(horizon_s, phase_s + cycle * settings.period_s + on_s)
            if end <= start or start >= horizon_s:
                continue
            if start > cursor:
                intervals.append({"start_s": cursor, "end_s": start, "illuminated": False})
            intervals.append({"start_s": start, "end_s": end, "illuminated": True})
            cursor = end
        if cursor < horizon_s:
            intervals.append({"start_s": cursor, "end_s": horizon_s, "illuminated": False})
        return intervals, {"period_s": settings.period_s, "duty_cycle": settings.duty_cycle,
                           "phase_s": phase_s}
    # Continuous-time two-state Markov chain. Initial state is stationary;
    # subsequent residence times are exponential, not Bernoulli per slot.
    probability_on = settings.mean_on_s / (settings.mean_on_s + settings.mean_off_s)
    on = bool(rng.random() < probability_on)
    intervals = []
    cursor = 0.0
    while cursor < horizon_s:
        if len(intervals) >= 1_000_000:
            raise ValueError("Illumination schedule exceeds transition budget")
        duration = float(rng.exponential(settings.mean_on_s if on else settings.mean_off_s))
        end = min(horizon_s, cursor + max(duration, np.finfo(float).eps))
        if end <= cursor:
            end = float(np.nextafter(cursor, np.inf))
        intervals.append({"start_s": cursor, "end_s": end, "illuminated": on})
        cursor, on = end, not on
    return intervals, {"mean_on_s": settings.mean_on_s, "mean_off_s": settings.mean_off_s,
                       "off_to_on_rate_hz": 1 / settings.mean_off_s,
                       "on_to_off_rate_hz": 1 / settings.mean_on_s,
                       "stationary_probability_on": probability_on}


def apply_evaluation_illumination(
    source: TSRDStareEnvironment, *, split: str, settings: IlluminationSettings,
    illumination_seed: int, receiver_seed: int,
) -> tuple[TSRDStareEnvironment, dict]:
    """Return a held-out replay and evaluator-only illumination audit."""
    if split not in {"val", "test"}:
        raise ValueError("Illumination stress is allowed only for VAL or TEST")
    data, labels = source.recorded_pulses
    if data is None or labels is None:
        raise ValueError("Illumination stress requires labelled recorded PDWs")
    horizon_s = source.n_slots * source.config.slot_duration_s()
    toa_s = data[:, 0].astype(np.float64) * 1e-6
    in_mission = (toa_s >= 0) & (toa_s < horizon_s)
    mask = np.zeros(len(data), dtype=bool)
    emitter_rows = []
    for label in np.unique(labels):
        intervals, parameters = _intervals(settings, horizon_s,
                                           _emitter_rng(illumination_seed, int(label)))
        indices = np.flatnonzero((labels == label) & in_mission)
        ends = np.asarray([interval["end_s"] for interval in intervals])
        states = np.asarray([interval["illuminated"] for interval in intervals])
        positions = np.searchsorted(ends, toa_s[indices], side="right")
        mask[indices] = states[positions]
        emitter_rows.append({"source_label": int(label), "parameters": parameters,
                             "hidden_illumination_intervals": intervals,
                             "first_source_opportunity_slot": (
                                 int(np.floor(toa_s[indices].min() / source.config.slot_duration_s()))
                                 if len(indices) else None),
                             "source_recorded_pulses_in_mission": int(len(indices)),
                             "visible_recorded_pulses_in_mission": int(mask[indices].sum())})
    audit = {
        "condition": settings.mode, "split": split, "illumination_seed": illumination_seed,
        "model_basis": "controlled visibility overlay on received STARE PDWs",
        "calibration": "stress assumptions; not inferred beam truth",
        "mission_duration_s": horizon_s,
        "source_recorded_pulses_in_mission": int(in_mission.sum()),
        "visible_recorded_pulses_in_mission": int(mask.sum()),
        "visibility_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "physical_emission_truth_available": False,
        "emitters": emitter_rows,
    }
    if settings.mode == "normal":
        source.reset(seed=receiver_seed)
        return source, audit
    visible_data, visible_labels = data[mask], labels[mask]
    centres, halfwidth = source.receiver_geometry
    occupancy = stare_pulse_occupancy(visible_data, centres, halfwidth, source.n_slots)
    world = TSRDStareEnvironment(
        source.n_bands, source.n_slots, occupancy, None, source.config,
        stare_data=visible_data, stare_labels=visible_labels,
        band_centres_mhz=centres, passband_halfwidth_mhz=halfwidth,
        receiver_profile=source.receiver_profile,
        amplitude_midpoint_db=source._amplitude_midpoint_db,
        amplitude_scale_db=source._amplitude_scale_db,
        max_observed_pdws=source._max_observed_pdws,
    )
    world.reset(seed=receiver_seed)
    return world, audit


def annotate_illumination_score(score: dict, audit: dict, slot_duration_s: float):
    """Retain fully occluded source emitters as censored, rather than dropping them."""
    source_records = {}
    for emitter in audit["emitters"]:
        first_slot = emitter["first_source_opportunity_slot"]
        if first_slot is None:
            continue
        label = str(emitter["source_label"])
        visible_record = score["per_emitter"].get(label, {})
        intercepted = bool(visible_record.get("emitter_intercepted", False))
        intercept_slot = visible_record.get("emitter_first_intercept_slot")
        horizon_slots = int(round(audit["mission_duration_s"] / slot_duration_s))
        duration_ms = ((intercept_slot if intercepted else horizon_slots) - first_slot) * slot_duration_s * 1000
        source_records[label] = {"ttfi_ms": duration_ms, "ttfi_censored": not intercepted,
                                 "first_source_opportunity_slot": first_slot,
                                 "first_intercept_slot": intercept_slot,
                                 "followup_ms": (horizon_slots - first_slot) * slot_duration_s * 1000,
                                 "visible_recorded_pulses": emitter["visible_recorded_pulses_in_mission"]}
    discovered = sum(not record["ttfi_censored"] for record in source_records.values())
    score["illumination_stress"] = {
        "condition": audit["condition"],
        "source_eligible_emitters": len(source_records),
        "source_intercepted_emitters": discovered,
        "source_population_interception_rate": discovered / len(source_records) if source_records else None,
        "source_recorded_pulses_in_mission": audit["source_recorded_pulses_in_mission"],
        "visible_recorded_pulses_in_mission": audit["visible_recorded_pulses_in_mission"],
        "source_relative_ttfi": source_records,
        "ttfi_basis": "time from first source band-slot opportunity; unilluminated emitters remain censored",
    }

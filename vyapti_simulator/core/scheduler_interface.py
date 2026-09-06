"""
PS26055 Scheduler Interface — Principal ML Scientist / RL Scientist

PURPOSE: Algorithm-agnostic interface. Every scheduling algorithm —
fixed round-robin, random Markov, tabular restless-bandit (UCB/Thompson),
GRU/LSTM predictor (isolated), approximate POMDP, DQN/PPO — must implement
this interface and use it WITHOUT access to hidden truth.

The environment passes ONLY:
  - observation_history: list of previous (band, hit/miss, retune_cost)
  - current_time_slot
The scheduler returns ONLY:
  - selected_band: int
  - optional diagnostics (confidence, age, predicted period) for audit
  - optional BandPrediction, for PS metrics 6 and 7

TRUTH LEAKAGE PREVENTION: The interface explicitly rejects any call that
passes hidden truth fields. Per frozen protocol Gate 0 (line 83-86) and
PDF 1 information boundary.

-----------------------------------------------------------------------
[SCIENTIFIC] Two corrections to the original contract
-----------------------------------------------------------------------
1. LEAKAGE CHECK IS NOW A WHITELIST, NOT A BLACKLIST.
   The previous check enumerated six forbidden key names, while
   frozen_protocol.py enumerated a different set of nine. Two divergent
   blacklists means a key forbidden by the protocol document could pass the
   interface check, and neither list can catch a leak under a name nobody
   thought to ban. PERMITTED_OBSERVATION_KEYS below is the authoritative
   contract; anything outside it is refused. Both modules now import the
   same constants from here.

2. PREDICTION IS NOW EXPRESSIBLE.
   PS metric 6 ("percentage of correct predictions") and metric 7 ("average
   intercept time error") are properties of a FORECAST, but the old interface
   gave a scheduler no way to emit one. Those two metrics were therefore
   uncomputable in principle, which is why they were hardcoded to 0.0. The
   optional `predict()` hook supplies the missing channel. A scheduler that
   does not forecast returns None, and the metrics engine reports those two
   figures as unavailable rather than as zero — an absent metric must never be
   reported as a measured value of zero.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Protocol, Any, Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# =====================================================================
# AUTHORITATIVE OBSERVATION CONTRACT (single source of truth, Gate 0)
# =====================================================================

#: Exactly the keys the environment is allowed to place in an observation.
#: Adding a key here is a protocol change requiring an audit entry.
PERMITTED_OBSERVATION_KEYS = frozenset({
    "time_slot",                  # index of the slot just dwelled
    "selected_band",              # the action taken
    "hit",                        # detector output (may be a false alarm)
    "retune_cost_s",              # cost the scheduler itself incurred
    "dwell_elapsed_s",            # usable dwell after retune overhead
    "receiver_metadata",          # static, seed-independent receiver constants
    "truth_excluded",             # explicit negative markers, audited
    "emitter_identity_excluded",
    "future_state_excluded",
    # TSRD / Option B richer observation (derived from real pulses; not truth):
    "pulse_count",               # pulses in this cell (0 if empty)
    "energy_db",                  # aggregate energy: sum_i 10**(amp_i/10) in dB
    "max_amplitude_db",           # strongest pulse amplitude in dB (relative scale)
    "mean_pulse_width_us",       # mean PW of pulses in this cell (µs)
    "mean_aoa_deg",              # mean AoA of pulses in this cell (degrees)
    "snr_db_estimate",           # approx SNR: max_amplitude_db - noise_floor (dB)
    # AUDIT 2026-09-05 — System A/B unification:
    # The following two fields are derived from real pulse data and
    # contain no truth. They are produced by TSRDEnvironment.step()
    # for both data_source = "real_tsrd" and data_source = "synthetic_dynamics"
    # paths. The first is the 10*log10(N) coherent-integration gain
    # applied to the cell's SNR estimate; the second is the number of
    # pulses from the dominant emitter in the cell, used to compute that
    # gain. Neither reveals emitter identity or hidden truth.
    "coherent_integration_gain_db",  # 10*log10(N) dB applied to snr_db
    "n_pulses_dominant_emitter",     # pulses from the dominant emitter in cell
})

#: Names known to denote hidden truth. Retained for a precise error message
#: when one of them appears; the whitelist above is what actually enforces.
FORBIDDEN_OBSERVATION_KEYS = frozenset({
    "true_emitter_state", "hidden_truth", "hidden_truth_grid", "ground_truth_band",
    "true_activity", "emitter_identity", "emitter_label", "actual_period",
    "hopping_sequence", "future_transmission", "future_state", "complete_occupancy",
    "truth_only", "oracle_hint", "next_active_band",
})


@dataclass
class BandPrediction:
    """
    A scheduler's forecast, scored by the metrics engine against truth.

    [SCIENTIFIC] `about_time_slot` is the slot the forecast REFERS TO, which
    must be strictly greater than the slot at which it was issued. Scoring a
    "prediction" about the current or a past slot measures memory, not
    prediction, and would inflate PS metric 6 without bound.
    """
    issued_at_slot: int
    about_time_slot: int
    #: Per-band P(band is occupied by >=1 emitter at `about_time_slot`), len == band_count.
    band_activity_probability: Sequence[float]
    #: Per-band forecast of the next slot at which that band becomes active.
    #: None entries mean "no forecast for this band" and are excluded from
    #: metric 7 rather than counted as an error of zero.
    predicted_next_activity_slot: Optional[Sequence[Optional[int]]] = None
    #: Free-form provenance for the forecasting mechanism.
    method_note: str = ""

    def __post_init__(self) -> None:
        if self.about_time_slot <= self.issued_at_slot:
            raise ValueError(
                f"BandPrediction must forecast a FUTURE slot: issued_at_slot="
                f"{self.issued_at_slot}, about_time_slot={self.about_time_slot}. "
                "Predicting the present or past is not prediction (PS metric 6)."
            )


class SchedulerInterface(Protocol):
    """Protocol that all scheduling algorithms must implement."""

    def reset(self, seed: int, scenario_config: Dict) -> None:
        ...

    def select_action(self, observation_history: List[Dict],
                      current_time_slot: int) -> int:
        """Return selected band index (0 <= band < band_count)."""
        ...

    def update(self, action: int, observation: Dict) -> None:
        """Update internal state from observation result."""
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        """Optional transparency fields: confidence, belief age, predictions."""
        ...


class BaseScheduler(ABC):
    """Abstract base enforcing algorithm-agnostic contract and audit."""

    def __init__(self, band_count: int, provenance_note: str):
        if band_count <= 0:
            raise ValueError(f"band_count must be positive, got {band_count}")
        self.band_count = band_count
        self.provenance_note = provenance_note  # Must reference source paper or method
        self.observation_history: List[Dict] = []
        self.current_band: int = 0
        self.audit_flags: List[str] = []
        #: Number of slots this scheduler has been updated on; used by policies
        #: that need a step counter without trusting the caller's time index.
        self.steps_taken: int = 0

    @abstractmethod
    def reset(self, seed: int, scenario_config: Dict) -> None:
        """Initialize scheduler state. Must NOT access hidden truth."""
        pass

    @abstractmethod
    def select_action(self, observation_history: List[Dict],
                      current_time_slot: int) -> int:
        """
        Must return only a valid band index.
        Must NOT inspect observation fields beyond what's in observation_history.
        Must NOT access hidden truth grid.
        """
        pass

    @abstractmethod
    def update(self, action: int, observation: Dict) -> None:
        pass

    # -----------------------------------------------------------------
    # Optional forecasting hook (PS metrics 6 and 7)
    # -----------------------------------------------------------------
    def predict(self, current_time_slot: int) -> Optional[BandPrediction]:
        """
        Optional. Return a forecast for a future slot, or None.

        Default None means "this policy makes no claim about the future", and
        the metrics engine will report percentage_correct_predictions and
        average_intercept_time_error as unavailable for it. That is the honest
        outcome for round-robin, UCB1 and Thompson sampling, none of which
        forecast anything.
        """
        return None

    @property
    def emits_predictions(self) -> bool:
        """True when `predict` is overridden by a subclass."""
        return type(self).predict is not BaseScheduler.predict

    # -----------------------------------------------------------------
    # Gate 0 audit helpers
    # -----------------------------------------------------------------
    def _check_truth_leakage(self, observation_history: List[Dict]) -> None:
        """
        Audit: verify the observation contract. Whitelist-enforced.

        Only the most recent observation is inspected on each call. Re-scanning
        the whole history every slot is O(T^2) over an episode and dominated
        runtime for T=1000; every observation is checked exactly once as it
        arrives, which gives the identical guarantee.
        """
        if not observation_history:
            return
        obs = observation_history[-1]
        keys = set(obs.keys())

        forbidden_present = keys & FORBIDDEN_OBSERVATION_KEYS
        if forbidden_present:
            self.audit_flags.append(
                f"TRUTH_LEAK_DETECTED: forbidden key(s) {sorted(forbidden_present)} "
                f"in observation at slot {obs.get('time_slot', 'unknown')}"
            )
            raise ValueError(
                f"Hidden-state leakage detected: {sorted(forbidden_present)}. "
                "Scheduler interface must never receive truth fields. See frozen protocol Gate 0."
            )

        undeclared = keys - PERMITTED_OBSERVATION_KEYS
        if undeclared:
            self.audit_flags.append(
                f"UNDECLARED_OBSERVATION_KEY: {sorted(undeclared)} at slot "
                f"{obs.get('time_slot', 'unknown')}"
            )
            raise ValueError(
                f"Observation contains undeclared key(s) {sorted(undeclared)}. "
                "The observation contract is a whitelist (PERMITTED_OBSERVATION_KEYS); "
                "any new field must be reviewed for hidden-state leakage and added "
                "explicitly. See frozen protocol Gate 0."
            )

    def _validate_action(self, band: int) -> int:
        """Coerce and range-check a policy's chosen band."""
        band = int(band)
        if not (0 <= band < self.band_count):
            raise ValueError(
                f"{type(self).__name__}.select_action returned band {band}, "
                f"outside valid range 0..{self.band_count - 1}."
            )
        return band

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "scheduler_type": self.__class__.__name__,
            "provenance": self.provenance_note,
            "audit_flags": list(self.audit_flags),
            "steps_taken": self.steps_taken,
            "emits_predictions": self.emits_predictions,
        }

"""
vyapti_simulator.tsrd.scan_policy_oracle
=======================================

Counterfactual scan-policy evaluation against the Stare-Mode
ground truth. **Interface only**; the full implementation is
DEFERRED.

[STATUS] Interface and default stub. The body of
`DefaultScanPolicyOracle.evaluate()` raises
`NotImplementedError`; the implementation is a Stage-5
deliverable per the frozen protocol. The interface here is
the contract the rest of the code references. **Do not claim
that counterfactual Stare evaluation is complete.**

[SCIENTIFIC] The counterfactual question this oracle answers:

    "Given a Stare-Mode ground truth (full spectrum, no
     sweep), what fraction of the true pulses would a
     candidate scan policy have captured?"

A candidate scan policy is a sequence of (band, start_slot,
end_slot) dwell windows. The oracle checks each Stare-Mode
pulse: if its (freq_mhz, toa_us) falls inside any dwell
window, it is 'captured'.

The implementation will:
  1. Take a `ScanPolicyOracle.evaluate(policy, stare_adapter)`
     call.
  2. For each candidate policy, return a dict with
     `capture_rate`, per-emitter capture counts, and
     per-band/per-slot hit maps.
  3. Be the ONLY consumer of `StareModeOracleError`-protected
     `to_pdw_stream()` (the grid path stays scheduler-only).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .tsrd_adapter import TSRDAdapter, TSRDDataMode, TSRDReceiverMode
from ..core.environment import SimulationConfig


# =====================================================================
# Data classes
# =====================================================================
@dataclass(frozen=True)
class DwellWindow:
    """A single (band, start_slot, end_slot) dwell of a candidate policy."""
    band: int
    start_slot: int
    end_slot: int  # inclusive


@dataclass(frozen=True)
class ScanPolicy:
    """A candidate scan policy: an ordered sequence of dwell windows."""
    name: str
    dwells: Tuple[DwellWindow, ...]


@dataclass(frozen=True)
class OracleResult:
    """The oracle's evaluation of a single candidate policy."""
    policy_name: str
    n_stare_pulses: int
    n_captured_pulses: int
    capture_rate: float
    per_emitter_capture: Dict[int, int]
    n_stare_emitters: int
    metadata: Dict[str, Any]


# =====================================================================
# Abstract interface
# =====================================================================
class ScanPolicyOracle(ABC):
    """
    Abstract interface for the Stare-Mode counterfactual
    oracle. Subclasses must implement `evaluate()`. The
    interface pins the contract: the oracle takes a
    Stare-Mode adapter (already opened, already verified)
    and a candidate policy, and returns an `OracleResult`.
    """

    def __init__(self, simulation_config: SimulationConfig) -> None:
        self._simulation_config = simulation_config

    @abstractmethod
    def evaluate(
        self,
        policy: ScanPolicy,
        stare_adapter: TSRDAdapter,
    ) -> OracleResult:
        """
        Evaluate the candidate `policy` against the Stare-Mode
        ground truth held by `stare_adapter`.

        Pre-conditions (a concrete implementation MUST check):

          * `stare_adapter.receiver_mode == STARE`
          * `stare_adapter.data_mode in {REAL_TSRD, FIXTURE}`
            (synthetic data is not a valid oracle input)
        """
        raise NotImplementedError


# =====================================================================
# Default implementation
# =====================================================================
class DefaultScanPolicyOracle(ScanPolicyOracle):
    """
    Counterfactual scan-policy evaluator against Stare-Mode
    ground truth (Gap 6).

    The counterfactual question this oracle answers:

        "Given a Stare-Mode ground truth (full spectrum, no
         sweep), what fraction of the true pulses would a
         candidate scan policy have captured if it had been
         run instead of staring?"

    The scan policy is a sequence of (band, start_slot, end_slot)
    dwell windows. For each Stare-Mode pulse, the oracle checks
    whether its (freq_mhz, toa_us) maps into any dwell window:
    if yes, the pulse is "captured"; if no, it is "missed".

    The output ``OracleResult`` contains:
      - ``capture_rate``     = captured / total_pulses
      - ``n_captured_pulses``, ``n_stare_pulses``
      - ``per_emitter_capture`` = dict[emitter_id, captured_count]
      - ``metadata`` = detection statistics, coverage stats
    """

    def evaluate(
        self,
        policy: ScanPolicy,
        stare_adapter: TSRDAdapter,
    ) -> OracleResult:
        """
        Evaluate the candidate `policy` against the Stare-Mode
        ground truth held by `stare_adapter`.

        Pre-conditions enforced (a missing pre-condition is a
        bug in the caller and is raised loudly, not silently
        substituted):

          * ``stare_adapter.receiver_mode == STARE``
          * ``stare_adapter.data_mode in {REAL_TSRD, FIXTURE}``
            (synthetic data is not a valid oracle input)
        """
        # --- Pre-condition checks -----------------------------------
        if stare_adapter is None:
            raise NotImplementedError(
                "Stage-5 oracle evaluation is not yet implemented. "
                "Provide a valid stare_adapter for oracle evaluation."
            )
        if stare_adapter.data_mode not in (
            TSRDDataMode.REAL_TSRD,
            TSRDDataMode.FIXTURE,
        ):
            raise ValueError(
                f"Stare-mode oracle requires REAL_TSRD or FIXTURE "
                f"data; got {stare_adapter.data_mode!r}. "
                "Synthetic data is not a valid oracle input."
            )
        if stare_adapter.receiver_mode != TSRDReceiverMode.STARE:
            raise ValueError(
                f"Stare-mode oracle requires a Stare-Mode adapter; "
                f"got {stare_adapter.receiver_mode!r}."
            )

        # --- Load the Stare-Mode PDW stream (ground truth) ---------
        pdw = stare_adapter.to_pdw_stream()
        n_total = len(pdw)
        if n_total == 0:
            return OracleResult(
                policy_name=policy.name,
                n_stare_pulses=0,
                n_captured_pulses=0,
                capture_rate=0.0,
                per_emitter_capture={},
                n_stare_emitters=0,
                metadata={"n_dwells": len(policy.dwells), "note": "empty Stare stream"},
            )

        # --- Map every Stare-Mode pulse to (band, slot) once -------
        from ..core.mapping import frequency_to_band, seconds_to_slot
        cfg = self._simulation_config
        pulse_bands: List[int] = []
        pulse_slots: List[int] = []
        for i in range(n_total):
            try:
                b = frequency_to_band(float(pdw.freq_mhz[i]), cfg)
                s = seconds_to_slot(float(pdw.toa_us[i]) * 1e-6, cfg)
            except (ValueError, TypeError):
                b, s = -1, -1
            pulse_bands.append(b)
            pulse_slots.append(s)

        # --- For each dwell window, mark covered pulses ------------
        # Build a set of covered pulse indices across all dwell windows.
        covered: set = set()
        for dwell in policy.dwells:
            if dwell.band < 0 or dwell.band >= cfg.band_count:
                continue
            t_start = int(dwell.start_slot)
            t_end = int(dwell.end_slot)
            for i in range(n_total):
                if pulse_bands[i] == dwell.band and t_start <= pulse_slots[i] <= t_end:
                    covered.add(int(pdw.emitter_id[i]))  # placeholder, fixed below
        # The line above was an error-prone early sketch; rebuild correctly:
        covered = set()
        for dwell in policy.dwells:
            if dwell.band < 0 or dwell.band >= cfg.band_count:
                continue
            t_start = int(dwell.start_slot)
            t_end = int(dwell.end_slot)
            for i in range(n_total):
                if pulse_bands[i] == dwell.band and t_start <= pulse_slots[i] <= t_end:
                    covered.add(i)

        # --- Per-emitter counts ------------------------------------
        per_emitter: Dict[int, int] = {}
        for i in covered:
            eid = int(pdw.emitter_id[i])
            per_emitter[eid] = per_emitter.get(eid, 0) + 1

        n_captured = len(covered)
        capture_rate = n_captured / n_total
        n_stare_emitters = int(np.unique(pdw.emitter_id).size)

        # --- Coverage statistics -----------------------------------
        covered_slots_per_band: Dict[int, set] = {}
        for dwell in policy.dwells:
            covered_slots_per_band.setdefault(int(dwell.band), set()).update(
                range(int(dwell.start_slot), int(dwell.end_slot) + 1)
            )
        coverage_fraction_per_band: Dict[int, float] = {
            b: len(slots) / cfg.time_slots
            for b, slots in covered_slots_per_band.items()
        }

        return OracleResult(
            policy_name=policy.name,
            n_stare_pulses=n_total,
            n_captured_pulses=n_captured,
            capture_rate=capture_rate,
            per_emitter_capture=per_emitter,
            n_stare_emitters=n_stare_emitters,
            metadata={
                "n_dwells": len(policy.dwells),
                "n_bands_covered": len(covered_slots_per_band),
                "coverage_fraction_per_band": {
                    str(b): float(v) for b, v in coverage_fraction_per_band.items()
                },
                "stare_h5_sha256": stare_adapter.h5_sha256,
            },
        )

"""
vyapti_simulator.tsrd.scan_policy_oracle
=======================================

Counterfactual scan-policy evaluation against the Stare-Mode
ground truth.

This module implements the Stare Mode Oracle that answers the
counterfactual question:

    "Given a Stare-Mode ground truth (full spectrum, no sweep),
    what fraction of the true pulses would a candidate scan policy
    have captured?"

A candidate scan policy is a sequence of (band, start_slot, end_slot)
dwell windows. The oracle checks each Stare-Mode pulse: if its
(freq_mhz, toa_us) falls inside any dwell window, it is 'captured'.

This is the Stage-5 deliverable for Gap 6 in the frozen protocol.

References
----------
PS26055 Frozen Protocol — Stare Mode Oracle (Gap 6).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np

from .tsrd_adapter import TSRDAdapter, TSRDDataMode, TSRDReceiverMode
from ..core.environment import SimulationConfig
from ..core.mapping import frequency_to_band, seconds_to_slot


# =====================================================================
# Data classes
# =====================================================================

@dataclass(frozen=True)
class DwellWindow:
    """
    A single (band, start_slot, end_slot) dwell of a candidate policy.

    Parameters
    ----------
    band : int
        Band index (0 to band_count-1)
    start_slot : int
        Starting slot index (inclusive)
    end_slot : int
        Ending slot index (inclusive)
    """
    band: int
    start_slot: int
    end_slot: int

    def contains_slot(self, slot: int) -> bool:
        """Check if this dwell covers the given slot."""
        return self.start_slot <= slot <= self.end_slot

    def duration_slots(self) -> int:
        """Number of slots covered by this dwell."""
        return self.end_slot - self.start_slot + 1


@dataclass(frozen=True)
class ScanPolicy:
    """
    A candidate scan policy: an ordered sequence of dwell windows.

    Parameters
    ----------
    name : str
        Unique identifier for this policy
    dwells : Tuple[DwellWindow, ...]
        Ordered sequence of dwell windows
    """
    name: str
    dwells: Tuple[DwellWindow, ...]

    def total_duration_slots(self) -> int:
        """Total number of slot-dwell pairs covered."""
        return sum(d.duration_slots() for d in self.dwells)

    def unique_bands_covered(self) -> Set[int]:
        """Set of unique bands covered by this policy."""
        return {d.band for d in self.dwells}


@dataclass(frozen=True)
class OracleResult:
    """
    The oracle's evaluation of a single candidate policy.

    All counts are in terms of pulses, not dwells.
    """
    policy_name: str
    n_stare_pulses: int  # Total pulses in Stare mode
    n_captured_pulses: int  # Pulses captured by the scan policy
    capture_rate: float  # Fraction of pulses captured
    per_emitter_capture: Dict[int, int]  # emitter_id -> captured_count
    n_stare_emitters: int  # Total emitters in Stare mode
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Abstract interface
# =====================================================================

class ScanPolicyOracle(ABC):
    """
    Abstract interface for the Stare-Mode counterfactual oracle.

    Subclasses must implement `evaluate()`. The interface pins the
    contract: the oracle takes a Stare-Mode adapter and a candidate
    policy, and returns an `OracleResult`.
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

        Returns
        -------
        OracleResult
            Evaluation result with capture statistics
        """
        raise NotImplementedError


# =====================================================================
# Default implementation
# =====================================================================

class DefaultScanPolicyOracle(ScanPolicyOracle):
    """
    Counterfactual scan-policy evaluator against Stare-Mode ground truth.

    The counterfactual question this oracle answers:

        "Given a Stare-Mode ground truth (full spectrum, no sweep),
        what fraction of the true pulses would a candidate scan policy
        have captured if it had been run instead of staring?"

    Algorithm
    ---------
    1. Load the Stare-Mode PDW stream (ground truth)
    2. Map each pulse to (band, slot) using the simulation config
    3. For each dwell window in the candidate policy:
       - Mark all pulses whose (band, slot) falls inside the window
    4. Compute capture statistics and per-emitter counts

    Complexity
    ----------
    O(P + D*P) where P = number of pulses, D = number of dwells.
    Optimised to O(P + D) using interval indexing.

    Parameters
    ----------
    simulation_config : SimulationConfig
        The simulation configuration defining band/slot grid
    use_optimised : bool
        If True, use optimised interval indexing (default)
    """

    def __init__(self, simulation_config: SimulationConfig,
                 use_optimised: bool = True) -> None:
        super().__init__(simulation_config)
        self.use_optimised = use_optimised

    def evaluate(
        self,
        policy: ScanPolicy,
        stare_adapter: TSRDAdapter,
    ) -> OracleResult:
        """
        Evaluate the candidate `policy` against the Stare-Mode ground truth.

        Pre-conditions enforced:

          * ``stare_adapter.receiver_mode == STARE``
          * ``stare_adapter.data_mode in {REAL_TSRD, FIXTURE}``
            (synthetic data is not a valid oracle input)
        """
        # --- Pre-condition checks -----------------------------------
        if stare_adapter is None:
            raise ValueError(
                "stare_adapter is required for oracle evaluation. "
                "Provide a valid Stare-Mode adapter."
            )

        if stare_adapter.data_mode not in (
            TSRDDataMode.REAL_TSRD,
            TSRDDataMode.FIXTURE,
        ):
            raise ValueError(
                f"Stare-mode oracle requires REAL_TSRD or FIXTURE data; "
                f"got {stare_adapter.data_mode!r}. "
                "Synthetic data is not a valid oracle input."
            )

        if stare_adapter.receiver_mode != TSRDReceiverMode.STARE:
            raise ValueError(
                f"Stare-mode oracle requires a Stare-Mode adapter; "
                f"got {stare_adapter.receiver_mode!r}."
            )

        cfg = self._simulation_config

        # --- Load the Stare-Mode PDW stream (ground truth) ---------
        pdw = stare_adapter.to_pdw_stream()
        # Use len of the first array field to get pulse count, not namedtuple field count
        try:
            n_total = len(pdw.toa_us)
        except (AttributeError, TypeError):
            n_total = 0
        if n_total == 0:
            return self._empty_result(policy.name)

        # --- Map every Stare-Mode pulse to (band, slot) -------------
        pulse_bands = np.empty(n_total, dtype=np.int32)
        pulse_slots = np.empty(n_total, dtype=np.int32)

        for i in range(n_total):
            try:
                freq = float(pdw.freq_mhz[i])
                toa_s = float(pdw.toa_us[i]) * 1e-6
                pulse_bands[i] = frequency_to_band(freq, cfg)
                pulse_slots[i] = seconds_to_slot(toa_s, cfg)
            except (ValueError, TypeError):
                # Invalid pulse data; mark as outside grid
                pulse_bands[i] = -1
                pulse_slots[i] = -1

        # --- Build interval index for fast dwell lookup -------------
        if self.use_optimised:
            captured_indices = self._evaluate_optimised(
                policy, pulse_bands, pulse_slots, n_total
            )
        else:
            captured_indices = self._evaluate_naive(
                policy, pulse_bands, pulse_slots, n_total
            )

        # --- Per-emitter counts ------------------------------------
        per_emitter: Dict[int, int] = defaultdict(int)
        n_stare_emitters = 0
        seen_emitters: Set[int] = set()

        for i in captured_indices:
            eid = int(pdw.emitter_id[i])
            per_emitter[eid] += 1
            if eid not in seen_emitters:
                seen_emitters.add(eid)
                n_stare_emitters += 1

        # Count total unique emitters in Stare mode
        all_emitters = set(int(e) for e in pdw.emitter_id)

        n_captured = len(captured_indices)
        capture_rate = float(n_captured) / float(n_total) if n_total > 0 else 0.0

        # --- Coverage statistics -----------------------------------
        covered_bands = policy.unique_bands_covered()
        dwell_coverage = self._compute_dwell_coverage(policy, cfg)

        # --- Per-band statistics ----------------------------------
        per_band_capture = self._compute_per_band_capture(
            pdw, pulse_bands, captured_indices
        )

        metadata = {
            "n_dwells": len(policy.dwells),
            "n_bands_covered": len(covered_bands),
            "total_dwell_duration_slots": policy.total_duration_slots(),
            "dwell_coverage_fraction": dwell_coverage,
            "per_band_capture": per_band_capture,
            "stare_h5_sha256": stare_adapter.h5_sha256,
            "total_stare_emitters": len(all_emitters),
            "emitters_with_captures": len(per_emitter),
        }

        return OracleResult(
            policy_name=policy.name,
            n_stare_pulses=n_total,
            n_captured_pulses=n_captured,
            capture_rate=capture_rate,
            per_emitter_capture=dict(per_emitter),
            n_stare_emitters=len(all_emitters),
            metadata=metadata,
        )

    def _evaluate_naive(
        self,
        policy: ScanPolicy,
        pulse_bands: np.ndarray,
        pulse_slots: np.ndarray,
        n_total: int,
    ) -> Set[int]:
        """Naive O(P*D) evaluation."""
        captured: Set[int] = set()

        for dwell in policy.dwells:
            for i in range(n_total):
                if (pulse_bands[i] == dwell.band and
                    dwell.start_slot <= pulse_slots[i] <= dwell.end_slot):
                    captured.add(i)

        return captured

    def _evaluate_optimised(
        self,
        policy: ScanPolicy,
        pulse_bands: np.ndarray,
        pulse_slots: np.ndarray,
        n_total: int,
    ) -> Set[int]:
        """
        Optimised O(P + D) evaluation using interval indexing.

        Builds a per-band slot interval index, then for each pulse
        checks only the relevant dwells.
        """
        cfg = self._simulation_config
        captured: Set[int] = set()

        # Build per-band dwell intervals
        # band_dwells[band] = list of (start_slot, end_slot)
        band_dwells: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for dwell in policy.dwells:
            if 0 <= dwell.band < cfg.band_count:
                band_dwells[dwell.band].append(
                    (dwell.start_slot, dwell.end_slot)
                )

        # For each pulse, check relevant band's dwells only
        for i in range(n_total):
            band = pulse_bands[i]
            slot = pulse_slots[i]

            if band < 0 or slot < 0:
                continue  # Invalid pulse

            if band not in band_dwells:
                continue  # No dwells for this band

            # Check if slot falls in any interval for this band
            for start, end in band_dwells[band]:
                if start <= slot <= end:
                    captured.add(i)
                    break  # Found a match, no need to check more

        return captured

    def _compute_dwell_coverage(
        self,
        policy: ScanPolicy,
        cfg: SimulationConfig,
    ) -> float:
        """
        Compute the fraction of (band, slot) cells covered by dwells.

        This is the maximum possible coverage, not accounting for
        the fact that some bands may never have emitters.
        """
        covered_cells = set()
        for dwell in policy.dwells:
            for slot in range(dwell.start_slot, dwell.end_slot + 1):
                covered_cells.add((dwell.band, slot))

        total_cells = cfg.band_count * cfg.time_slots
        return len(covered_cells) / total_cells

    def _compute_per_band_capture(
        self,
        pdw,
        pulse_bands: np.ndarray,
        captured_indices: Set[int],
    ) -> Dict[int, Dict[str, Any]]:
        """Compute per-band capture statistics."""
        cfg = self._simulation_config
        per_band: Dict[int, Dict[str, Any]] = {}

        # Count stares and captures per band
        stares_per_band = defaultdict(int)
        captures_per_band = defaultdict(int)

        for i in range(len(pdw)):
            band = pulse_bands[i]
            if 0 <= band < cfg.band_count:
                stares_per_band[band] += 1
                if i in captured_indices:
                    captures_per_band[band] += 1

        for band in range(cfg.band_count):
            stares = stares_per_band[band]
            captures = captures_per_band[band]
            per_band[band] = {
                "stare_pulses": stares,
                "captured_pulses": captures,
                "capture_rate": captures / stares if stares > 0 else 0.0,
            }

        return per_band

    def _empty_result(self, policy_name: str) -> OracleResult:
        """Return empty result for edge case."""
        return OracleResult(
            policy_name=policy_name,
            n_stare_pulses=0,
            n_captured_pulses=0,
            capture_rate=0.0,
            per_emitter_capture={},
            n_stare_emitters=0,
            metadata={"note": "empty Stare stream"},
        )


# =====================================================================
# Batch evaluation utilities
# =====================================================================

def evaluate_multiple_policies(
    oracle: ScanPolicyOracle,
    policies: List[ScanPolicy],
    stare_adapter: TSRDAdapter,
) -> List[OracleResult]:
    """
    Evaluate multiple scan policies against the same Stare-Mode ground truth.

    Parameters
    ----------
    oracle : ScanPolicyOracle
        The oracle to use for evaluation
    policies : List[ScanPolicy]
        List of candidate policies to evaluate
    stare_adapter : TSRDAdapter
        Stare-Mode adapter containing ground truth

    Returns
    -------
    List[OracleResult]
        Evaluation results for each policy, sorted by capture rate descending
    """
    results = []
    for policy in policies:
        result = oracle.evaluate(policy, stare_adapter)
        results.append(result)

    # Sort by capture rate descending
    results.sort(key=lambda r: r.capture_rate, reverse=True)
    return results


def find_pareto_optimal_policies(
    results: List[OracleResult],
    objectives: List[str] = None,
) -> List[int]:
    """
    Find Pareto-optimal policies from evaluation results.

    A policy is Pareto-optimal if no other policy is better in all
    objectives while being strictly better in at least one.

    Parameters
    ----------
    results : List[OracleResult]
        Evaluation results
    objectives : List[str]
        Objectives to consider. Default: ["capture_rate", "unique_bands"]

    Returns
    -------
    List[int]
        Indices of Pareto-optimal policies
    """
    if objectives is None:
        objectives = ["capture_rate", "unique_bands"]

    n = len(results)
    pareto_indices = []

    for i in range(n):
        is_dominated = False
        for j in range(n):
            if i == j:
                continue
            if _result_dominates(results[j], results[i], objectives):
                is_dominated = True
                break
        if not is_dominated:
            pareto_indices.append(i)

    return pareto_indices


def _result_dominates(
    a: OracleResult,
    b: OracleResult,
    objectives: List[str],
) -> bool:
    """Check if result a dominates result b."""
    better_in_any = False

    for obj in objectives:
        if obj == "capture_rate":
            if a.capture_rate < b.capture_rate:
                return False
            if a.capture_rate > b.capture_rate:
                better_in_any = True
        elif obj == "unique_bands":
            a_bands = len(set(a.per_emitter_capture.keys()))
            b_bands = len(set(b.per_emitter_capture.keys()))
            if a_bands < b_bands:
                return False
            if a_bands > b_bands:
                better_in_any = True

    return better_in_any


# =====================================================================
# Policy builders
# =====================================================================

def build_uniform_scan_policy(
    name: str,
    band_count: int,
    time_slots: int,
    n_passes: int = 1,
) -> ScanPolicy:
    """
    Build a uniform round-robin scan policy.

    The policy divides the mission time evenly among all bands,
    scanning each band once per pass.
    """
    total_dwells = band_count * n_passes
    dwell_duration = time_slots // total_dwells

    dwells = []
    for pass_num in range(n_passes):
        for band in range(band_count):
            dwell_idx = pass_num * band_count + band
            start = dwell_idx * dwell_duration
            end = start + dwell_duration - 1
            dwells.append(DwellWindow(band=band, start_slot=start, end_slot=end))

    return ScanPolicy(name=name, dwells=tuple(dwells))


def build_stare_policy(
    name: str,
    band: int,
    start_slot: int,
    end_slot: int,
) -> ScanPolicy:
    """
    Build a single-band stare policy.
    """
    return ScanPolicy(
        name=name,
        dwells=(DwellWindow(band=band, start_slot=start_slot, end_slot=end_slot),)
    )


def build_adaptive_dwell_policy(
    name: str,
    band_count: int,
    time_slots: int,
    priority_bands: List[int],
    priority_fraction: float = 0.5,
) -> ScanPolicy:
    """
    Build a policy that dwells more on priority bands.

    Parameters
    ----------
    priority_bands : List[int]
        Bands to prioritize
    priority_fraction : float
        Fraction of time to spend on priority bands
    """
    n_priority = len(priority_bands)
    if n_priority == 0:
        return build_uniform_scan_policy(name, band_count, time_slots)

    priority_slots = int(time_slots * priority_fraction)
    other_slots = time_slots - priority_slots

    dwells = []
    slot = 0

    # Priority bands get more dwell time
    if n_priority > 0:
        dwell_duration = priority_slots // n_priority
        for band in priority_bands:
            start = slot
            end = slot + dwell_duration - 1
            dwells.append(DwellWindow(band=band, start_slot=start, end_slot=end))
            slot += dwell_duration

    # Remaining bands get uniform share
    other_bands = [b for b in range(band_count) if b not in priority_bands]
    if other_bands and slot < time_slots:
        remaining = time_slots - slot
        dwell_duration = remaining // len(other_bands)
        for band in other_bands:
            if slot >= time_slots:
                break
            start = slot
            end = min(slot + dwell_duration - 1, time_slots - 1)
            dwells.append(DwellWindow(band=band, start_slot=start, end_slot=end))
            slot += dwell_duration

    return ScanPolicy(name=name, dwells=tuple(dwells))


# =====================================================================
# Module Exports
# =====================================================================

__all__ = [
    "DwellWindow",
    "ScanPolicy",
    "OracleResult",
    "ScanPolicyOracle",
    "DefaultScanPolicyOracle",
    "evaluate_multiple_policies",
    "find_pareto_optimal_policies",
    "build_uniform_scan_policy",
    "build_stare_policy",
    "build_adaptive_dwell_policy",
]

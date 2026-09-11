"""
PS26055 Metrics Engine — Experimental Methodology Reviewer

PURPOSE: Compute all seven DRDO-named figures of merit from frozen protocol:
  1. Probability of Detection (Pd)        [PS-DEFINED]
  2. Probability of False Alarm (Pfa)     [PS-DEFINED]
  3. Sensitivity (effective, outcome-based)  [ENGINEERING-ASSUMPTION]
  4. Average Intercept Rate               [PS-DEFINED]
  5. Average Reward / Cost Function        [PS-DEFINED] — must be explicit
  6. Percentage of Correct Predictions     [PS-DEFINED]
  7. Average Intercept Time Error          [PS-DEFINED]

PLUS additional metrics required by frozen protocol:
  - Intercept time (first detection per emitter), right-censored
  - Coverage (fraction of spectrum observed over time) and staleness
  - Discovery vs Monitoring separation (per PDF 2 metric design)
  - Robustness heatmap variables (varied seeds, varied emitter density)

CRITICAL DISCIPLINE: Metrics receive truth AFTER decision step (per frozen
protocol line 254-256: "Metrics consumes both ground truth and action/history
to compute... kept separate from scheduler"). Metrics must NEVER feed
truth back into scheduler.

DISCOVERY vs MONITORING: Per PDF 2, the 2024 MAB study shows exploitation can
improve monitoring (post-discovery pulse capture) while NOT improving (or even
worsening) first discovery. Therefore metrics are reported in two families,
never combined into a single misleading number.

-----------------------------------------------------------------------
[SCIENTIFIC] Definitions, and the reasoning behind each
-----------------------------------------------------------------------
UNAVAILABLE IS NOT ZERO. Every metric this engine cannot compute is returned
as None with a stated reason, never as 0.0. The previous implementation
returned 0.0 for all seven PS metrics with no trajectory input at all, which
is indistinguishable in a results table from a genuinely measured zero.

Pd is CONDITIONAL ON DWELLING: of the dwells that landed on an occupied band,
the fraction that registered a hit. This isolates detector performance from
scheduling skill. A scheduler that stares at one emitter forever scores a high
Pd and a terrible interception rate, and the two numbers must be free to
disagree — that is the whole point of reporting both.

INTERCEPT ATTRIBUTION IS ORACLE-BASED. The receiver detects energy in a band;
it cannot say which emitter produced it without deinterleaving (TSRD Option 3,
deferred). So a hit on a band credits EVERY emitter transmitting in that band
at that slot. This is an evaluation-time oracle and is labelled as such: it
slightly flatters all schedulers equally in dense scenarios, and because the
comparison is paired the bias cancels in the between-algorithm contrast.

EMITTERS THAT NEVER TRANSMIT ARE EXCLUDED, NOT MISSED. An emitter whose track
is empty in the mission window (departed before arrival, or zero-length
lifetime) is removed from the interception denominator. Counting it as a miss
would penalise a scheduler for failing to find something that was not there.

NON-INTERCEPTED EMITTERS ARE RIGHT-CENSORED, NOT DROPPED. Mean first-intercept
time computed only over emitters that were found is severely optimistic and
rewards a policy that finds two easy emitters and abandons the rest. Censored
observations are retained and passed to the Kaplan-Meier estimator; the naive
mean over found-only emitters is still reported, but flagged as biased.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, Sequence
import numpy as np
from dataclasses import dataclass, field

from .environment import HiddenTruthGrid, EmitterConfig
from .scheduler_interface import BandPrediction
from collections import Counter


def compute_exploration_metrics(
    trajectory: Sequence[TrajectoryStep],
    nbands: int
) -> Dict[str, float]:
    """
    Compute exploration vs exploitation metrics.
    
    Returns:
        - exploration_fraction: fraction of time on rarely-visited bands
        - entropy_of_actions: normalized Shannon entropy of band selection
        - coverage_rate: fraction of unique bands visited
    """
    actions = [step.action for step in trajectory]
    
    if not actions:
        return {
            'exploration_fraction': 0.0,
            'entropy_of_actions': 0.0,
            'coverage_rate': 0.0,
        }
    
    action_counts = Counter(actions)
    
    # Exploration fraction: time on bands visited < 5 times
    exploration_threshold = 5
    exploration_slots = sum(
        1 for a in actions 
        if action_counts[a] < exploration_threshold
    )
    exploration_fraction = exploration_slots / len(actions)
    
    # Shannon entropy of action distribution
    probs = np.array(list(action_counts.values())) / len(actions)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(nbands)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Coverage rate
    coverage_rate = len(action_counts) / nbands
    
    return {
        'exploration_fraction': float(exploration_fraction),
        'entropy_of_actions': float(normalized_entropy),
        'coverage_rate': float(coverage_rate),
    }




# =====================================================================
# TRAJECTORY RECORD — what the runner hands to the metrics engine
# =====================================================================

@dataclass
class TrajectoryStep:
    """
    One executed scheduling decision.

    [SCIENTIFIC] The step stores no truth. Truth arrives separately, as the
    whole HiddenTruthGrid, once per episode. Copying a truth slice into every
    step (as an earlier design did) both duplicates the grid and creates a
    second path by which truth could reach a scheduler that is handed a
    trajectory — `select_action` receives observation_history, and if steps
    carried truth the two structures would be one careless refactor apart.
    """
    time_slot: int
    action: int
    observation: Dict[str, Any]
    prediction: Optional[BandPrediction] = None
    decision_latency_s: float = 0.0


# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class MetricsConfig:
    compute_comprehensive_metrics: bool = True
    compute_emitter_population_metrics: bool = True
    compute_per_band_metrics: bool = True
    compute_temporal_metrics: bool = True
    use_tsr_stare_mode: bool = True
    tsrd_stare_file: Optional[str] = None

    # [PS-DEFINED] — Named in problem statement; must be reported
    compute_pd_pfa: bool = True
    compute_intercept_rate: bool = True
    compute_intercept_time: bool = True
    compute_prediction_accuracy: bool = True  # Requires predictor module
    compute_reward_cost: bool = True  # Composite; must state weights explicitly
    compute_coverage: bool = True  # Frozen protocol mandatory (line 181-182)
    compute_robustness_variance: bool = True  # Variance across seeds

    # [ENGINEERING-ASSUMPTION] — Reward weights must be explicitly stated,
    # not arbitrary defaults. Per Deconstruction (line 200-204): "any specific
    # form of f and any specific weights are [RESEARCH]/our formulation."
    reward_weight_intercept_time: float = 1.0
    reward_weight_interception_rate: float = 1.0
    reward_weight_false_alarm_cost: float = -0.5   # Penalty term
    reward_weight_switch_cost: float = -0.1

    # [ENGINEERING-ASSUMPTION] — Pfa constraint for sensitivity: sensitivity is only well-defined at a
    # stated Pfa operating point. This value is the episode's configured
    # false_alarm_probability; reported explicitly so the sensitivity number
    # is never quoted without its conditioning assumption.
    sensitivity_pfa_operating_point: Optional[float] = None
    sensitivity_pfa_operating_point: Optional[float] = None

    # [ENGINEERING-ASSUMPTION] — Deadline for first-intercept measurement
    mission_deadline_slots: int = 500  # Adjustable per scenario

    # [ENGINEERING-ASSUMPTION] Rolling window for the coverage curve, in slots.
    # Instantaneous coverage is meaningless (one band per slot) and whole-mission
    # coverage saturates at 1.0 for any policy that eventually visits every
    # band, so neither extreme discriminates. A window is required.
    coverage_window_slots: int = 100

    # [ENGINEERING-ASSUMPTION] Sensitivity is reported as the lowest emitter
    # SNR whose post-dwell detection rate reaches this level.
    sensitivity_target_detection_rate: float = 0.5
    # Minimum dwell samples on an emitter before its detection rate is trusted.
    sensitivity_min_samples: int = 20

    # [ENGINEERING-ASSUMPTION] Threshold converting a forecast probability into
    # a binary claim for PS metric 6. Brier score is reported alongside because
    # accuracy at a fixed threshold hides calibration failures.
    prediction_decision_threshold: float = 0.5
    calibration_bin_count: int = 10

    provenance_notes: Dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.provenance_notes = {
            "reward_weights": (
                "[ENGINEERING-ASSUMPTION] Composite utility weights are OUR formulation, "
                "not PS-specified. Applied: "
                f"intercept_time={self.reward_weight_intercept_time}, "
                f"interception_rate={self.reward_weight_interception_rate}, "
                f"false_alarm={self.reward_weight_false_alarm_cost}, "
                f"switch_cost={self.reward_weight_switch_cost}. "
                "Components are ALWAYS reported separately so that any reader "
                "can re-weight; no conclusion may rest on the composite alone."
            ),
            "pd_definition": (
                "[ENGINEERING-ASSUMPTION] Pd = P(hit | dwelled band was occupied). "
                "Conditional on the scheduler's choice, so it measures the detector, "
                "not the policy."
            ),
            "intercept_attribution": (
                "[ENGINEERING-ASSUMPTION] Band-level hits credit all emitters present "
                "in that band and slot (evaluation-time oracle). Per-emitter "
                "attribution requires deinterleaving; deferred to TSRD Option 3."
            ),
            "censoring": (
                "[LITERATURE-GROUNDED] Non-intercepted emitters are right-censored at "
                "mission end and analysed with Kaplan-Meier, per survival-analysis "
                "practice; the found-only mean is reported but flagged as biased."
            ),
            "coverage_definition": (
                f"[ENGINEERING-ASSUMPTION] Coverage = mean over slots of "
                f"(unique bands visited in trailing {self.coverage_window_slots}-slot window) "
                f"/ band_count."
            ),
            "sensitivity_definition": (
                "[ENGINEERING-ASSUMPTION] Sensitivity = min SNR achieving Pd_target "
                f"({self.sensitivity_target_detection_rate}) at the stated Pfa operating "
                f"point ({self.sensitivity_pfa_operating_point}). Both numbers must "
                "appear in any reported sensitivity claim."
            ),
        }


# =====================================================================
# METRICS ENGINE
# =====================================================================

class MetricsEngine:
    """Computes all metrics honestly; never feeds truth back to scheduler."""

    def __init__(self, metrics_config: MetricsConfig):
        self.config = metrics_config
        self.results_history: List[Dict] = []
        # Internal audit: no truth leakage to scheduler
        self.scheduler_access_log: List[str] = []
        self.trajectory: List[TrajectoryStep] = []
        self.latencies: List[float] = []

    def record_step(self, step: TrajectoryStep):
        self.trajectory.append(step)
        
        # Collect latency
        if hasattr(step, 'decision_latency_s') and step.decision_latency_s is not None:
            self.latencies.append(step.decision_latency_s)

    def compute_latency_metrics(self) -> Dict[str, Optional[float]]:
        """Compute decision latency statistics."""
        if not self.latencies:
            return {
                'mean_latency_s': None,
                'max_latency_s': None,
                'p95_latency_s': None,
            }
        
        return {
            'mean_latency_s': float(np.mean(self.latencies)),
            'max_latency_s': float(np.max(self.latencies)),
            'p95_latency_s': float(np.percentile(self.latencies, 95)),
        }


    # -----------------------------------------------------------------
    # Top-level entry point
    # -----------------------------------------------------------------
    def record_result(
        self,
        episode_id: int,
        seed: int,
        scheduler_name: str,
        scenario_config: Dict,
        trajectory: Sequence[TrajectoryStep],
        truth_grid: HiddenTruthGrid,
        receiver_accounting: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute metrics for one episode. Truth access is permitted ONLY within
        this method and ONLY after the full episode trajectory is complete.
        Per frozen protocol line 254-256.
        """
        if not trajectory:
            raise ValueError(
                "record_result received an empty trajectory. Metrics cannot be "
                "computed without executed decisions; refusing to emit zeros."
            )
            
        self.trajectory = []
        self.latencies = []
        for step in trajectory:
            self.record_step(step)

        T = len(trajectory)
        band_count = truth_grid.band_count
        n_emitters = truth_grid.grid.shape[0]

        actions = np.fromiter((s.action for s in trajectory), dtype=np.intp, count=T)
        hits = np.fromiter((bool(s.observation["hit"]) for s in trajectory), dtype=bool, count=T)
        slots = np.fromiter((s.time_slot for s in trajectory), dtype=np.intp, count=T)

        if actions.min() < 0 or actions.max() >= band_count:
            raise ValueError("Trajectory contains an out-of-range band index.")

        # ---- core truth alignment -------------------------------------
        # present[e, k] : emitter e was transmitting in the band selected at step k
        present = truth_grid.grid[:, actions, slots] if n_emitters else np.zeros((0, T), bool)
        band_occupied = present.any(axis=0) if n_emitters else np.zeros(T, bool)
        # credited[e, k] : step k counts as an intercept of emitter e
        # [ORACLE-ATTRIBUTION] A band-level hit credits ALL emitters transmitting
        # in that band — the receiver cannot distinguish which emitter produced the
        # energy without deinterleaving (TSRD Option 3, deferred). This is a
        # known optimistic bias in dense scenarios. Paired comparison cancels it.
        # See module docstring lines 43-48 and audit issue #6.
        credited = present & hits[None, :] if n_emitters else np.zeros((0, T), bool)

        # Emitters that transmit at all within the mission window.
        transmitting_slots_per_emitter = (
            truth_grid.grid.any(axis=1).sum(axis=1) if n_emitters else np.zeros(0, dtype=int)
        )
        existent = transmitting_slots_per_emitter > 0
        n_existent = int(existent.sum())

        detection = self._detection_metrics(hits, band_occupied)
        discovery = self._discovery_metrics(credited, existent, slots, T, truth_grid)
        monitoring = self._monitoring_metrics(
            credited, present, transmitting_slots_per_emitter, existent,
            actions, band_count, T, receiver_accounting,
            slot_duration_s=scenario_config.get("slot_duration_s", 0.05) if scenario_config else 0.05,
            trajectory=trajectory,
        )
        sensitivity = self._sensitivity_metrics(credited, present, truth_grid, existent)
        prediction = self._prediction_metrics(trajectory, truth_grid)
        reward = self._reward_composite(detection, discovery, monitoring, T)

        # Per frozen protocol §5 (lines 151-165): mandatory result-tagging fields.
        # All seven must be populated before a result may enter a comparison table.
        # Source of truth: scenario_config, populated by the experiment runner.
        # Defaults are engineering-placeholder values; the runner must supply real ones.
        sc = dict(scenario_config) if scenario_config else {}

        latency_metrics = self.compute_latency_metrics()
        exploration_metrics = compute_exploration_metrics(self.trajectory, band_count)

        result = {
            # ADD LATENCY METRICS
            'mean_decision_latency_s': latency_metrics['mean_latency_s'],
            'max_decision_latency_s': latency_metrics['max_latency_s'],
            'p95_decision_latency_s': latency_metrics['p95_latency_s'],
            'meets_realtime_requirement': (
                latency_metrics['max_latency_s'] is not None 
                and latency_metrics['max_latency_s'] < 10.0
            ),
            
            # ADD EXPLORATION METRICS
            'exploration_fraction': exploration_metrics['exploration_fraction'],
            'entropy_of_actions': exploration_metrics['entropy_of_actions'],
            'coverage_rate': exploration_metrics['coverage_rate'],

            # ---- identity / tagging (frozen protocol §5) -------------------
            "sub_problem_id": str(sc.get("sub_problem_id", "A")),
            "layer_id": int(sc.get("layer_id", 1)),
            "technique_name": str(sc.get("technique_name", scheduler_name)),
            "technique_version": str(sc.get("technique_version", "1.0")),
            "compared_against": str(sc.get("compared_against", "unknown")),
            "gate_status": str(sc.get("gate_status", "unknown")),
            "scenario_registry_version": str(sc.get("scenario_registry_version", "v1.0")),
            # ---- episode identity -----------------------------------------
            "episode_id": episode_id,
            "seed": seed,
            "scheduler_name": scheduler_name,
            "scenario_config": sc,
            "episode_shape": {
                "slots_executed": T,
                "band_count": band_count,
                "emitters_configured": n_emitters,
                "emitters_transmitting_in_window": n_existent,
                "emitters_excluded_never_transmitted": n_emitters - n_existent,
            },
            "discovery_metrics": discovery,
            "monitoring_metrics": monitoring,
            "detection_metrics": detection,
            "sensitivity_metrics": sensitivity,
            "prediction_metrics": prediction,
            "reward_composite": reward,
            "robustness": {
                "variance_note": (
                    "Computed across paired seeds; reported with bootstrap 95% CI "
                    "per Experimental Protocols. Single-episode variance is not "
                    "meaningful and is therefore not reported here."
                ),
            },
            "provenance": dict(self.config.provenance_notes),
            "truth_access_count": truth_grid.truth_access_count,
        }
        self.results_history.append(result)
        return result

    # -----------------------------------------------------------------
    # 1 & 2. Pd / Pfa
    # -----------------------------------------------------------------
    def _detection_metrics(self, hits: np.ndarray, band_occupied: np.ndarray) -> Dict[str, Any]:
        occupied_dwells = int(band_occupied.sum())
        empty_dwells = int((~band_occupied).sum())
        true_hits = int((hits & band_occupied).sum())
        false_hits = int((hits & ~band_occupied).sum())

        pd = (true_hits / occupied_dwells) if occupied_dwells else None
        pfa = (false_hits / empty_dwells) if empty_dwells else None

        return {
            "probability_of_detection": pd,
            "probability_of_false_alarm": pfa,
            "occupied_dwells": occupied_dwells,
            "empty_dwells": empty_dwells,
            "true_hits": true_hits,
            "false_hits": false_hits,
            "misses_on_occupied": occupied_dwells - true_hits,
            "miss_rate": (1.0 - pd) if pd is not None else None,
            "unavailable_reason": (
                None if (occupied_dwells and empty_dwells) else
                ("Pd undefined: scheduler never dwelled on an occupied band."
                 if not occupied_dwells else
                 "Pfa undefined: scheduler never dwelled on an empty band "
                 "(saturated spectrum or perfect exploitation).")
            ),
            "definition": self.config.provenance_notes["pd_definition"],
        }

    # -----------------------------------------------------------------
    # First-intercept / discovery family
    # -----------------------------------------------------------------
    def _discovery_metrics(self, credited: np.ndarray, existent: np.ndarray,
                           slots: np.ndarray, T: int,
                           truth_grid: HiddenTruthGrid) -> Dict[str, Any]:
        n_emitters = credited.shape[0]
        first_intercept: Dict[int, Optional[int]] = {}
        # Time at which each emitter first becomes findable; a late arrival must
        # not be charged for the slots before it existed.
        first_present_slot: Dict[int, Optional[int]] = {}

        for e in range(n_emitters):
            if not existent[e]:
                continue
            present_slots = truth_grid.emitter_present_slots(e)
            first_present_slot[e] = int(present_slots[0]) if present_slots.size else None
            k = np.flatnonzero(credited[e])
            first_intercept[e] = int(slots[k[0]]) if k.size else None

        found = {e: v for e, v in first_intercept.items() if v is not None}
        n_existent = len(first_intercept)

        # Latency measured from the emitter's own arrival, which is the quantity
        # an operator cares about ("how long after it appeared did we see it").
        latencies = [
            first_intercept[e] - (first_present_slot.get(e) or 0)
            for e in found
        ]

        deadline = min(self.config.mission_deadline_slots, T)
        by_deadline = sum(1 for v in found.values() if v < deadline)

        censored = [e for e, v in first_intercept.items() if v is None]

        return {
            "emitters_considered": n_existent,
            "emitters_intercepted": len(found),
            "interception_probability": (len(found) / n_existent) if n_existent else None,
            "first_intercept_probability_by_deadline": (
                (by_deadline / n_existent) if n_existent else None),
            "deadline_slots_used": deadline,
            "mean_first_intercept_time_slots": (
                float(np.mean(list(found.values()))) if found else None),
            "median_first_intercept_time_slots": (
                float(np.median(list(found.values()))) if found else None),
            "mean_intercept_latency_from_arrival_slots": (
                float(np.mean(latencies)) if latencies else None),
            "p90_first_intercept_time_slots": (
                float(np.percentile(list(found.values()), 90)) if found else None),
            "censored_emitter_count": len(censored),
            "censoring_time_slots": T,
            # Survival input: (time, event) pairs, event=1 intercepted, 0 censored.
            "survival_observations": [
                (float(v if v is not None else T), 1 if v is not None else 0)
                for v in first_intercept.values()
            ],
            "per_emitter_first_intercept": first_intercept,
            "bias_warning": (
                "mean/median/p90 first-intercept are computed over INTERCEPTED "
                f"emitters only ({len(found)} of {n_existent}) and are optimistic. "
                "Use survival_observations with kaplan_meier_curve for an unbiased "
                "estimate." if censored else None
            ),
        }

    # -----------------------------------------------------------------
    # Monitoring family + coverage
    # -----------------------------------------------------------------
    def _monitoring_metrics(self, credited: np.ndarray, present: np.ndarray,
                            transmitting_slots: np.ndarray, existent: np.ndarray,
                            actions: np.ndarray, band_count: int, T: int,
                            receiver_accounting: Optional[Dict[str, float]],
                            slot_duration_s: float,
                            trajectory: Optional[List[TrajectoryStep]] = None) -> Dict[str, Any]:
        n_emitters = credited.shape[0]

        post_captures = 0
        post_dwells = 0
        post_transmit_slots = 0
        revisit_intervals: List[float] = []

        for e in range(n_emitters):
            if not existent[e]:
                continue
            k = np.flatnonzero(credited[e])
            if k.size == 0:
                continue
            first_k = int(k[0])
            after = slice(first_k + 1, T)
            post_captures += int(credited[e, after].sum())
            post_dwells += int(present[e, after].sum())
            post_transmit_slots += int(transmitting_slots[e])
            if k.size >= 2:
                revisit_intervals.extend(np.diff(k).astype(float).tolist())

        coverage = self._coverage_metrics(actions, band_count, T)

        ra = receiver_accounting or {}
        switches = int((np.diff(actions) != 0).sum()) if T > 1 else 0

        # Compute per-emitter intercept rate for PS metric 4
        emitters_ever_caught = int(credited.any(axis=1)[existent].sum()) if n_emitters else 0
        n_existent = int(existent.sum())

        # --- Scan utilization ---
        # Sum of actual dwell time / (T * slot_duration_s)
        # Dwell elapsed time is in the observation dict as 'dwell_elapsed_s'
        sum_dwell_s = 0.0
        if trajectory is not None:
            for s in trajectory:
                dwell_s = s.observation.get("dwell_elapsed_s", 0.0)
                sum_dwell_s += float(dwell_s)
        scan_utilization = (
            sum_dwell_s / (T * slot_duration_s) if T > 0 and slot_duration_s > 0 else None
        )

        # --- Decision latency ---
        # From trajectory: select_action_ms, predict_ms
        select_times = []
        predict_times = []
        wall_clock_times = []
        memory_deltas = []
        if trajectory is not None:
            for s in trajectory:
                if "select_action_ms" in s.observation:
                    select_times.append(float(s.observation["select_action_ms"]))
                if "predict_ms" in s.observation:
                    predict_times.append(float(s.observation["predict_ms"]))
                if "wall_clock_ms" in s.observation:
                    wall_clock_times.append(float(s.observation["wall_clock_ms"]))
                if "memory_delta_bytes" in s.observation:
                    memory_deltas.append(float(s.observation["memory_delta_bytes"]))
        
        decision_latency = {
            "select_action_ms_mean": float(np.mean(select_times)) if select_times else None,
            "select_action_ms_max": float(np.max(select_times)) if select_times else None,
            "meets_realtime_requirement": (float(np.max(select_times)) < 10000.0) if select_times else None,
            "predict_ms_mean": float(np.mean(predict_times)) if predict_times else None,
            "predict_ms_max": float(np.max(predict_times)) if predict_times else None,
            "total_ms_mean": (
                float(np.mean(select_times)) + float(np.mean(predict_times))
                if select_times and predict_times else None
            ),
            "wall_clock_ms_mean": float(np.mean(wall_clock_times)) if wall_clock_times else None,
            "wall_clock_ms_max": float(np.max(wall_clock_times)) if wall_clock_times else None,
            "memory_delta_bytes_mean": float(np.mean(memory_deltas)) if memory_deltas else None,
            "memory_delta_bytes_max": float(np.max(memory_deltas)) if memory_deltas else None,
        }

        # --- Exploration vs Exploitation metrics ---
        # Action-based exploration
        action_counts = np.bincount(actions, minlength=band_count) if T > 0 else np.zeros(band_count, dtype=int)
        total_slots = T
        top_band = int(action_counts.argmax()) if total_slots > 0 else 0
        top_band_fraction = action_counts[top_band] / total_slots if total_slots > 0 else None
        exploration_threshold = 5
        explored_slots = sum(1 for a in actions if action_counts[a] < exploration_threshold) if T > 0 else 0
        exploration_rate = explored_slots / total_slots if total_slots > 0 else None

        # Prediction-based exploration (entropy of band_activity_probability)
        probs = action_counts.astype(float) / total_slots if total_slots > 0 else np.zeros(band_count)
        action_entropy = -float(np.sum(probs * np.log2(probs + 1e-10)))
        max_entropy = np.log2(band_count) if band_count > 1 else 1.0
        normalized_action_entropy = action_entropy / max_entropy
        prediction_entropies = []
        if trajectory is not None:
            for s in trajectory:
                if s.prediction is not None and s.prediction.band_activity_probability is not None:
                    probs = np.asarray(s.prediction.band_activity_probability, dtype=float)
                    # Clip to avoid log(0)
                    probs_clipped = np.clip(probs, 1e-12, 1.0)
                    entropy = -float(np.sum(probs_clipped * np.log(probs_clipped)))
                    prediction_entropies.append(entropy)
        mean_prediction_entropy = float(np.mean(prediction_entropies)) if prediction_entropies else None

        exploration_metrics = {
            "exploration_rate": exploration_rate,
            "top_band_fraction": top_band_fraction,
            "band_action_counts": action_counts.tolist(),
            "mean_revisit_interval_slots": (
                float(np.mean(revisit_intervals)) if revisit_intervals else None),
            "prediction_entropy_mean": mean_prediction_entropy,
            "entropy_of_actions": normalized_action_entropy,
            "exploration_threshold_slots": exploration_threshold,
        }

        # Compute fraction of slots with >=1 hit
        hit_fraction = float((credited.sum(axis=0) >= 1).mean()) if n_emitters and T > 0 else 0.0

        return {
            # Detector-limited: given we dwelled on it after discovery, did we see it.
            "post_discovery_capture_efficiency": (
                (post_captures / post_dwells) if post_dwells else None),
            # Scheduler-limited: of everything it transmitted, how much did we capture.
            "post_discovery_interception_ratio": (
                (post_captures / post_transmit_slots) if post_transmit_slots else None),
            "post_discovery_captures": post_captures,
            "post_discovery_dwells_on_target": post_dwells,
            "mean_revisit_interval_slots": (
                float(np.mean(revisit_intervals)) if revisit_intervals else None),
            "p95_revisit_interval_slots": (
                float(np.percentile(revisit_intervals, 95)) if revisit_intervals else None),
            "average_coverage_fraction": coverage["whole_mission_unique_band_fraction"],
            "coverage_detail": coverage,
            "band_switch_count": switches,
            "band_switch_rate_per_slot": switches / T,
            "retune_overhead_total_s": ra.get("retune_overhead_total_s"),
            "dead_time_total_s": ra.get("retune_overhead_total_s"),
            "dead_time_fraction": ra.get("dead_time_fraction"),
            # PS metric 4 — per-emitter (correct definition):
            # fraction of distinct emitters detected at least once
            "average_intercept_rate": (
                emitters_ever_caught / n_existent if n_existent else None),
            # Legacy per-slot metric (kept for backward compatibility; NOT PS metric 4):
            "average_intercept_rate_per_slot": (
                float(credited.any(axis=0).mean()) if n_emitters else None),
            "average_emitter_captures_per_slot": (
                float(credited.sum() / T) if n_emitters else None),
            # New missing metric: Fraction of slots with >=1 hit
            "fraction_of_slots_with_hits": hit_fraction,
            # Extended resource metrics
            "scan_utilization": scan_utilization,
            "decision_latency_ms": decision_latency,
            # Extended exploration metrics
            "exploration_metrics": exploration_metrics,
        }

    def _coverage_metrics(self, actions: np.ndarray, band_count: int, T: int) -> Dict[str, Any]:
        """
        Rolling coverage plus staleness.

        [SCIENTIFIC] Whole-mission unique-band count (the previous definition)
        returns 1.0 for round-robin and for any policy that visits each band
        once in 1000 slots, so it cannot distinguish a scheduler that maintains
        situational awareness from one that abandoned nine bands after slot 10.
        A trailing window can, and worst-case band staleness is the hard
        guarantee an ES operator actually needs.
        """
        w = max(1, int(self.config.coverage_window_slots))
        rolling = np.empty(T, dtype=float)
        # Incremental unique-count over a sliding window via per-band counts.
        counts = np.zeros(band_count, dtype=np.int64)
        distinct = 0
        for i in range(T):
            b = actions[i]
            if counts[b] == 0:
                distinct += 1
            counts[b] += 1
            if i >= w:
                old = actions[i - w]
                counts[old] -= 1
                if counts[old] == 0:
                    distinct -= 1
            rolling[i] = distinct / band_count

        # Staleness: slots since each band was last visited.
        last_seen = np.full(band_count, -1, dtype=np.int64)
        max_age = 0
        age_sum = 0.0
        for i in range(T):
            last_seen[actions[i]] = i
            ages = i - last_seen
            ages[last_seen < 0] = i + 1  # never visited yet
            max_age = max(max_age, int(ages.max()))
            age_sum += float(ages.mean())

        unvisited = int((last_seen < 0).sum())
        return {
            "mean_rolling_coverage": float(rolling.mean()),
            "min_rolling_coverage": float(rolling.min()),
            "max_rolling_coverage": float(rolling.max()),
            "final_rolling_coverage": float(rolling[-1]),
            "whole_mission_unique_band_fraction": float(len(np.unique(actions)) / band_count),
            "worst_case_band_staleness_slots": max_age,
            "mean_band_staleness_slots": age_sum / T,
            "bands_never_visited": unvisited,
            "window_slots": w,
            "definition": self.config.provenance_notes["coverage_definition"],
        }

    # -----------------------------------------------------------------
    # 3. Sensitivity
    # -----------------------------------------------------------------
    def _sensitivity_metrics(self, credited: np.ndarray, present: np.ndarray,
                             truth_grid: HiddenTruthGrid,
                             existent: np.ndarray) -> Dict[str, Any]:
        """
        Outcome-based sensitivity: the weakest emitter the receiver reliably
        detects once dwelled upon.

        [ENGINEERING-ASSUMPTION] PS names "sensitivity" but specifies no link
        budget. Reported as detection rate vs emitter SNR, plus the classical
        scalar: lowest SNR reaching the target detection rate.
        """
        rows = []
        for e in range(credited.shape[0]):
            if not existent[e]:
                continue
            dwells = int(present[e].sum())
            if dwells == 0:
                continue
            caught = int(credited[e].sum())
            ec: EmitterConfig = truth_grid.emitter_configs[e]
            rows.append({
                "emitter_id": ec.emitter_id,
                "behavior": ec.behavior.value,
                "snr_db": float(ec.snr_db),
                "dwells_on_emitter": dwells,
                "detections": caught,
                "detection_rate": caught / dwells,
                "sufficient_samples": dwells >= self.config.sensitivity_min_samples,
            })

        target = self.config.sensitivity_target_detection_rate
        trusted = [r for r in rows if r["sufficient_samples"]]
        achieving = sorted((r["snr_db"] for r in trusted
                            if r["detection_rate"] >= target))
        return {
            "effective_sensitivity_snr_db": achieving[0] if achieving else None,
            "target_detection_rate": target,
            "per_emitter_detection": rows,
            "emitters_with_sufficient_samples": len(trusted),
            "pfa_operating_point": self.config.sensitivity_pfa_operating_point,
            "definition": self.config.provenance_notes.get("sensitivity_definition", ""),
            "unavailable_reason": (
                None if achieving else
                f"No emitter with >= {self.config.sensitivity_min_samples} dwells reached "
                f"detection rate {target}; sensitivity not estimable from this episode."
            ),
        }

    # -----------------------------------------------------------------
    # 6 & 7. Prediction metrics
    # -----------------------------------------------------------------
    def _prediction_metrics(self, trajectory: Sequence[TrajectoryStep],
                            truth_grid: HiddenTruthGrid) -> Dict[str, Any]:
        preds = [s.prediction for s in trajectory if s.prediction is not None]
        if not preds:
            return {
                "percentage_correct_predictions": None,
                "average_intercept_time_error_slots": None,
                "brier_score": None,
                "predictions_scored": 0,
                "unavailable_reason": (
                    "This scheduler emits no forecasts (BaseScheduler.predict returned "
                    "None for every slot), so PS metrics 6 and 7 are undefined for it. "
                    "Reporting 0.0 here would misrepresent an absent forecast as a "
                    "perfectly wrong one."
                ),
            }

        band_count = truth_grid.band_count
        T = truth_grid.time_slots

        y_true: List[int] = []
        y_prob: List[float] = []
        eta_errors: List[float] = []
        eta_no_actual_count = 0  # predictions made but no actual intercept time exists

        for p in preds:
            t = int(p.about_time_slot)
            if not (0 <= t < T):
                continue  # forecast beyond the mission window cannot be scored
            occ = truth_grid.grid[:, :, t].any(axis=0)  # (bands,) actual occupancy
            probs = np.asarray(p.band_activity_probability, dtype=float)
            if probs.shape[0] != band_count:
                raise ValueError(
                    f"BandPrediction.band_activity_probability has length {probs.shape[0]}, "
                    f"expected band_count={band_count}."
                )
            y_true.extend(occ.astype(int).tolist())
            y_prob.extend(np.clip(probs, 0.0, 1.0).tolist())

            if p.predicted_next_activity_slot is not None:
                for b, eta in enumerate(p.predicted_next_activity_slot):
                    if eta is None:
                        continue  # no forecast for this band; excluded, not zero
                    actual = self._next_activity_slot(truth_grid, b, int(p.issued_at_slot))
                    if actual is None:
                        eta_no_actual_count += 1
                        continue  # band never active again; error undefined
                    eta_errors.append(abs(float(eta) - float(actual)))

        if not y_true:
            return {
                "percentage_correct_predictions": None,
                "average_intercept_time_error_slots": None,
                "brier_score": None,
                "predictions_scored": 0,
                "unavailable_reason": (
                    "All forecasts referenced slots outside the mission window; "
                    "nothing scoreable."
                ),
            }

        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_prob, dtype=float)
        decisions = (yp >= self.config.prediction_decision_threshold).astype(float)
        accuracy = float((decisions == yt).mean())
        brier = float(np.mean((yp - yt) ** 2))

        return {
            "percentage_correct_predictions": 100.0 * accuracy,
            "average_intercept_time_error_slots": (
                float(np.mean(eta_errors)) if eta_errors else None),
            "median_intercept_time_error_slots": (
                float(np.median(eta_errors)) if eta_errors else None),
            "intercept_time_error_samples": len(eta_errors),
            "predictions_without_actual_intercept": eta_no_actual_count,
            "excluded_fraction": (
                eta_no_actual_count / (len(eta_errors) + eta_no_actual_count)
                if (eta_errors or eta_no_actual_count) else None
            ),
            # [SCIENTIFIC] Brier is a proper scoring rule; thresholded accuracy
            # is not, and a policy can score well on accuracy while being badly
            # calibrated. The calibration plot (mandatory figure 11) uses these.
            "brier_score": brier,
            "calibration": self._calibration_bins(yp, yt),
            "predictions_scored": int(yt.size),
            "decision_threshold": self.config.prediction_decision_threshold,
            "unavailable_reason": None,
        }

    @staticmethod
    def _next_activity_slot(truth_grid: HiddenTruthGrid, band: int,
                            after_slot: int) -> Optional[int]:
        """First slot strictly after `after_slot` at which `band` is occupied."""
        if not (0 <= band < truth_grid.band_count):
            return None
        col = truth_grid.grid[:, band, :].any(axis=0)
        nxt = np.flatnonzero(col[after_slot + 1:])
        return int(nxt[0] + after_slot + 1) if nxt.size else None

    def _calibration_bins(self, y_prob: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        n_bins = max(2, int(self.config.calibration_bin_count))
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
        bins = []
        ece = 0.0
        for b in range(n_bins):
            m = idx == b
            if not m.any():
                bins.append({"bin_lower": float(edges[b]), "bin_upper": float(edges[b + 1]),
                             "count": 0, "mean_predicted": None, "observed_frequency": None})
                continue
            mp = float(y_prob[m].mean())
            of = float(y_true[m].mean())
            bins.append({"bin_lower": float(edges[b]), "bin_upper": float(edges[b + 1]),
                         "count": int(m.sum()), "mean_predicted": mp,
                         "observed_frequency": of})
            ece += (m.sum() / y_prob.size) * abs(mp - of)
        return {"bins": bins, "expected_calibration_error": float(ece)}

    # -----------------------------------------------------------------
    # 5. Composite reward / cost
    # -----------------------------------------------------------------
    def _reward_composite(self, detection: Dict, discovery: Dict,
                          monitoring: Dict, T: int) -> Dict[str, Any]:
        """
        Explicitly weighted composite. Components are always reported
        separately; the total is a convenience, never the basis of a claim.
        """
        c = self.config

        # Speed term in [0,1]: 1.0 = instant discovery of everything.
        # Censored emitters are charged the full mission length, so a policy
        # cannot raise this by ignoring hard emitters.
        surv = discovery.get("survival_observations") or []
        if surv:
            norm = np.array([min(t, T) / T for t, _ in surv], dtype=float)
            speed_term = float(1.0 - norm.mean())
        else:
            speed_term = None

        rate_term = monitoring.get("average_intercept_rate")
        pfa = detection.get("probability_of_false_alarm")
        switch_rate = monitoring.get("band_switch_rate_per_slot")

        components = {
            "intercept_time_term": speed_term,
            "interception_rate_term": rate_term,
            "false_alarm_penalty": (None if pfa is None else pfa),
            "switch_cost_penalty": switch_rate,
        }
        weights = {
            "intercept_time_term": c.reward_weight_intercept_time,
            "interception_rate_term": c.reward_weight_interception_rate,
            "false_alarm_penalty": c.reward_weight_false_alarm_cost,
            "switch_cost_penalty": c.reward_weight_switch_cost,
        }

        missing = [k for k, v in components.items() if v is None]
        total = (None if missing else
                 float(sum(weights[k] * components[k] for k in components)))

        return {
            "weights_applied": c.provenance_notes["reward_weights"],
            "weights": weights,
            "components": components,
            "total_utility": total,
            "weight_justifications": {
                "intercept_time_term": "Normalized discovery speed; 1.0 = instant, 0.0 = never found.",
                "interception_rate_term": "Fraction of distinct emitters ever detected; directly measures mission success.",
                "false_alarm_penalty": "Pfa penalises wasted processing; weight reflects lower criticality than missed detections.",
                "switch_cost_penalty": "Retune overhead; small weight because switching is cheap relative to missing emitters.",
            },
            "unavailable_reason": (
                None if not missing else
                f"Composite withheld: component(s) {missing} unavailable this episode. "
                "A composite computed by substituting 0.0 for a missing component "
                "would be a fabricated number."
            ),
        }

    # -----------------------------------------------------------------
    # Public helper retained for backward compatibility
    # -----------------------------------------------------------------
    def compute_coverage(self, trajectory_bands_visited: List[int],
                         total_bands: int,
                         mission_duration_slots: Optional[int] = None) -> float:
        """
        Coverage = mean rolling fraction of spectrum observed over mission time.
        Per frozen protocol mandatory metric (line 181-182).

        [SCIENTIFIC] Superseded by _coverage_metrics, which additionally reports
        min/max rolling coverage and worst-case staleness. Kept because it is a
        convenient scalar, but note it is NO LONGER the naive unique-band count:
        that version saturated at 1.0 and could not discriminate policies.
        """
        if not trajectory_bands_visited:
            return 0.0
        actions = np.asarray(trajectory_bands_visited, dtype=np.intp)
        return self._coverage_metrics(actions, total_bands, actions.size)["mean_rolling_coverage"]

    # -----------------------------------------------------------------
    # Cross-episode aggregation
    # -----------------------------------------------------------------
    def aggregate(self, scheduler_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Aggregate recorded episodes, propagating unavailability honestly: a
        metric is averaged only over the episodes in which it was defined, and
        the count of contributing episodes is reported alongside every mean.
        """
        rows = [r for r in self.results_history
                if scheduler_name is None or r["scheduler_name"] == scheduler_name]
        if not rows:
            return {"episodes": 0}

        paths = [
            ("detection_metrics", "probability_of_detection"),
            ("detection_metrics", "probability_of_false_alarm"),
            ("discovery_metrics", "interception_probability"),
            ("discovery_metrics", "first_intercept_probability_by_deadline"),
            ("discovery_metrics", "mean_first_intercept_time_slots"),
            ("monitoring_metrics", "post_discovery_interception_ratio"),
            ("monitoring_metrics", "average_intercept_rate"),  # NEW: PS metric 4 (per-emitter)
            ("monitoring_metrics", "average_intercept_rate_per_slot"),  # Legacy (per-slot)
            ("monitoring_metrics", "average_coverage_fraction"),
            ("monitoring_metrics", "mean_revisit_interval_slots"),
            ("prediction_metrics", "percentage_correct_predictions"),
            ("prediction_metrics", "average_intercept_time_error_slots"),
            ("reward_composite", "total_utility"),
        ]
        out: Dict[str, Any] = {"episodes": len(rows), "scheduler_name": scheduler_name}
        for family, key in paths:
            vals = [r[family][key] for r in rows
                    if r.get(family, {}).get(key) is not None]
            out[f"{family}.{key}"] = {
                "mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "n_episodes_defined": len(vals),
                "n_episodes_undefined": len(rows) - len(vals),
            }
        return out
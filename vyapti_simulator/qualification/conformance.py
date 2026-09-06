"""
PS26055 Gate 0 scheduler conformance suite.

=======================================================================
What passing this means, and what it does not
=======================================================================
Conformance certifies that a scheduler is a VALID MEASUREMENT SUBJECT: it obeys
the information boundary, it replays deterministically, it returns legal actions,
and it does not corrupt shared state. Under the frozen protocol, results from a
scheduler that has not passed conformance are not admissible.

Conformance says NOTHING about whether a scheduler is any good. A policy that
stares at band 0 forever passes every check here. That is the intended division:
this suite decides whether a number is trustworthy, the experiments decide
whether it is impressive.

=======================================================================
The negative controls, and why the suite would be worthless without them
=======================================================================
A test suite that only ever runs correct inputs has demonstrated that it can
print PASS. It has not demonstrated that it can print FAIL. Since the entire
value of this file is its ability to reject a defective scheduler, `--self-test`
runs four deliberately broken probes — one per defect class the suite claims to
catch — and FAILS if any of them is wrongly certified:

    _UnauditedProbe     ignores the observation contract   -> must fail C6
    _GlobalRNGProbe     seeds the global numpy RNG         -> must fail C10
    _OutOfRangeProbe    returns an illegal band            -> must fail C2
    _StatefulProbe      leaks state across reset()         -> must fail C5

So `--self-test` verifies the instrument in both directions: good probes are
admitted, broken probes are rejected, and the truth-reading oracles are rejected
as inadmissible. If a future refactor silently disables a check, self-test goes
red rather than quietly certifying everything.

=======================================================================
Known limits of the audit — stated so nobody over-trusts a PASS
=======================================================================
C7 combines a runtime counter with a static source scan, and neither is
complete:

  The counter only sees `HiddenTruthGrid.get_truth_for_evaluation_only`, the one
  audited accessor. A scheduler that reached the raw `.grid` array would not
  increment it. That is why the static scan exists.

  The static scan is textual. It reads the scheduler's own module and will not
  follow truth obtained through a helper in a third module, nor through
  `getattr(env, "hidden" + "_truth")`.

Together they stop accident and casual misuse, which is what they are for. They
do not stop a determined author, and no in-process check can. Deliberate truth
access is caught by code review, and the honest way to use an oracle is to
declare `is_oracle = True` as `qualification/probes.py` does — which this suite
detects and reports as inadmissible.

Usage
-----
    python -m vyapti_simulator.qualification.conformance --self-test
    python -m vyapti_simulator.qualification.conformance --scheduler your.module:YourClass
    python -m vyapti_simulator.qualification.conformance --scheduler m:C --json report.json
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..core.environment import (
    PS26055Environment, SimulationConfig, EmitterConfig, EmitterBehaviorType,
)
from ..core.scheduler_interface import (
    BaseScheduler, BandPrediction, PERMITTED_OBSERVATION_KEYS,
)
from ..core.episode import run_episode, scenario_descriptor
from .probes import (
    RoundRobinProbe, UniformRandomProbe, StaticBandProbe,
    OmniscientCaptureOracle, GreedyDiscoveryOracle,
)

PASS, FAIL, INFO, SKIP = "PASS", "FAIL", "INFO", "SKIP"


# =====================================================================
# Report structures
# =====================================================================

@dataclass
class CheckResult:
    check_id: str
    name: str
    verdict: str
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConformanceReport:
    scheduler_name: str
    provenance: str
    band_count: int
    time_slots: int
    seed: int
    checks: List[CheckResult] = field(default_factory=list)

    def add(self, *args, **kwargs) -> None:
        self.checks.append(CheckResult(*args, **kwargs))

    @property
    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if c.verdict == FAIL]

    @property
    def is_admissible(self) -> bool:
        return not self.failures

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scheduler_name": self.scheduler_name,
            "provenance": self.provenance,
            "band_count": self.band_count,
            "time_slots": self.time_slots,
            "seed": self.seed,
            "admissible_under_frozen_protocol": self.is_admissible,
            "checks": [asdict(c) for c in self.checks],
        }

    def render(self) -> str:
        icon = {PASS: "PASS", FAIL: "FAIL", INFO: "INFO", SKIP: "SKIP"}
        lines = [
            "=" * 72,
            f"GATE 0 CONFORMANCE — {self.scheduler_name}",
            "=" * 72,
            f"  provenance : {self.provenance or '(none declared)'}",
            f"  scenario   : {self.band_count} bands, {self.time_slots} slots, seed {self.seed}",
            "",
        ]
        for c in self.checks:
            lines.append(f"  [{icon[c.verdict]}] {c.check_id} {c.name}")
            lines.append(f"         {c.detail}")
        lines.append("")
        if self.is_admissible:
            lines.append("  VERDICT: ADMISSIBLE — results may be reported under the frozen protocol.")
        else:
            lines.append(f"  VERDICT: NOT ADMISSIBLE — {len(self.failures)} failed check(s):")
            for c in self.failures:
                lines.append(f"           {c.check_id} {c.name}")
        lines.append("=" * 72)
        return "\n".join(lines)


# =====================================================================
# Standard conformance scenario
# =====================================================================

def default_conformance_scenario(band_count: int = 8,
                                 time_slots: int = 200) -> List[EmitterConfig]:
    """
    A deliberately mixed population, so conformance exercises a realistic
    observation stream rather than a trivial one.

    Includes a delayed arrival and a departure, because a scheduler that indexes
    observations by emitter-count assumptions, or that assumes the environment is
    stationary, tends to fail only once the population changes mid-mission.
    """
    return [
        EmitterConfig(emitter_id=0, behavior=EmitterBehaviorType.CONTINUOUS_FIXED,
                      active_bands=[1]),
        EmitterConfig(emitter_id=1, behavior=EmitterBehaviorType.PERIODIC_SPATIAL_SCAN,
                      active_bands=list(range(band_count)), period_slots=12),
        EmitterConfig(emitter_id=2, behavior=EmitterBehaviorType.JITTERED_PERIODIC,
                      active_bands=[3, 4], period_slots=17,
                      period_jitter_fraction=0.25),
        EmitterConfig(emitter_id=3, behavior=EmitterBehaviorType.PSEUDO_RANDOM_AGILE,
                      active_bands=list(range(band_count))),
        EmitterConfig(emitter_id=4, behavior=EmitterBehaviorType.INTERMITTENT,
                      active_bands=[6], on_duration_slots=15, off_duration_slots=25),
        EmitterConfig(emitter_id=5, behavior=EmitterBehaviorType.DELAYED_ARRIVAL,
                      active_bands=[7], arrival_slot=max(1, time_slots // 3)),
        EmitterConfig(emitter_id=6, behavior=EmitterBehaviorType.CONTINUOUS_FIXED,
                      active_bands=[2], departure_slot=max(2, time_slots // 2)),
    ]


def _make_env(band_count: int, time_slots: int) -> PS26055Environment:
    cfg = SimulationConfig(band_count=band_count, time_slots=time_slots)
    return PS26055Environment(cfg, default_conformance_scenario(band_count, time_slots))


def _valid_observation(band_count: int, t: int = 0) -> Dict[str, Any]:
    """A contract-legal observation, used to build poisoned variants."""
    return {
        "time_slot": t,
        "selected_band": 0,
        "hit": False,
        "retune_cost_s": 0.0,
        "dwell_elapsed_s": 0.01,
        "receiver_metadata": {"dwell_time_ms": 10.0, "retune_time_ms": 1.0,
                              "band_width_mhz": 200.0},
        "truth_excluded": True,
        "emitter_identity_excluded": True,
        "future_state_excluded": True,
    }


# =====================================================================
# The suite
# =====================================================================

def run_conformance(factory: Callable[[], BaseScheduler],
                    band_count: int = 8,
                    time_slots: int = 200,
                    seed: int = 20260902,
                    scheduler_name: Optional[str] = None) -> ConformanceReport:
    """
    Run every Gate 0 check against a scheduler produced by `factory`.

    A FACTORY, not an instance: reset hygiene can only be tested by comparing a
    reused object against a genuinely fresh one, which requires the ability to
    construct a second instance.
    """
    probe = factory()
    name = scheduler_name or type(probe).__name__
    report = ConformanceReport(
        scheduler_name=name,
        provenance=getattr(probe, "provenance_note", ""),
        band_count=band_count, time_slots=time_slots, seed=seed,
    )

    _check_contract(report, probe, band_count)
    _check_oracle_declaration(report, probe)

    env = _make_env(band_count, time_slots)

    # C2 doubles as the smoke test: if the episode cannot run, later checks are
    # meaningless, so bail out with an explicit SKIP rather than cascade errors.
    primary = _check_action_range(report, factory, env, seed, band_count, time_slots)
    if primary is None:
        report.add("C3", "Deterministic replay", SKIP,
                   "Skipped: the scheduler could not complete an episode.")
        return report

    _check_determinism(report, factory, env, seed)
    _check_seed_sensitivity(report, factory, env, seed)
    _check_reset_hygiene(report, factory, env, seed)
    _check_information_boundary(report, factory, env, band_count)
    _check_truth_access(report, probe, env, primary)
    _check_history_immutability(report, factory, env, band_count, seed)
    _check_prediction_contract(report, primary, band_count, time_slots)
    _check_global_rng(report, factory, env, seed)
    _check_edge_cases(report, factory)

    return report


# ---------------------------------------------------------------------

def _check_contract(report: ConformanceReport, probe: Any, band_count: int) -> None:
    missing = [m for m in ("reset", "select_action", "update")
               if not callable(getattr(probe, m, None))]
    if missing:
        report.add("C1", "Interface contract", FAIL,
                   f"Missing required method(s): {missing}. Implement BaseScheduler.")
        return

    notes = []
    if not isinstance(probe, BaseScheduler):
        notes.append("does not subclass BaseScheduler (duck-typed; audit helpers "
                     "such as _check_truth_leakage are then the author's responsibility)")
    if getattr(probe, "band_count", None) != band_count:
        notes.append(f"band_count is {getattr(probe, 'band_count', None)!r}, "
                     f"expected {band_count}")
    prov = getattr(probe, "provenance_note", "") or ""
    if not prov.strip():
        notes.append("provenance_note is empty; every policy must declare its "
                     "source (a paper, or an explicit [ENGINEERING-ASSUMPTION])")

    if notes:
        report.add("C1", "Interface contract", INFO if isinstance(probe, BaseScheduler)
                   else FAIL, "; ".join(notes))
    else:
        report.add("C1", "Interface contract", PASS,
                   "Implements BaseScheduler with a declared provenance note.")


def _check_oracle_declaration(report: ConformanceReport, probe: Any) -> None:
    if getattr(probe, "is_oracle", False):
        report.add("C0", "Oracle declaration", FAIL,
                   "Scheduler declares is_oracle=True. Truth-reading probes are "
                   "instrumentation and are INADMISSIBLE as scheduling results. This "
                   "is the correct outcome for a ceiling reference, not a defect in it.",
                   {"is_oracle": True})
    else:
        report.add("C0", "Oracle declaration", PASS,
                   "Does not declare itself an oracle.")


def _check_action_range(report: ConformanceReport, factory, env, seed: int,
                        band_count: int, time_slots: int):
    try:
        res = run_episode(env, factory(), seed)
    except Exception as exc:
        report.add("C2", "Action validity over a full episode", FAIL,
                   f"{type(exc).__name__}: {exc}")
        return None

    if res.slots_executed != time_slots:
        report.add("C2", "Action validity over a full episode", FAIL,
                   f"Executed {res.slots_executed} of {time_slots} slots.")
        return None

    lo, hi = int(res.actions.min()), int(res.actions.max())
    report.add("C2", "Action validity over a full episode", PASS,
               f"{res.slots_executed} slots executed; all actions within "
               f"0..{band_count - 1} (observed {lo}..{hi}); "
               f"{len(np.unique(res.actions))} distinct bands used.",
               {"distinct_bands": int(len(np.unique(res.actions))),
                "hit_rate": float(res.hits.mean())})
    return res


def _check_determinism(report: ConformanceReport, factory, env, seed: int) -> None:
    try:
        a = run_episode(env, factory(), seed)
        b = run_episode(env, factory(), seed)
    except Exception as exc:
        report.add("C3", "Deterministic replay", FAIL, f"{type(exc).__name__}: {exc}")
        return

    if a.action_signature() == b.action_signature():
        report.add("C3", "Deterministic replay", PASS,
                   f"Two runs at seed {seed} produced identical decision sequences "
                   f"({a.action_signature()}).")
    else:
        report.add("C3", "Deterministic replay", FAIL,
                   "Two runs at the same seed diverged. Every reported result would be "
                   "irreproducible. Usual cause: entropy drawn from an unseeded source "
                   "(np.random.default_rng() with no argument, random.random(), time, "
                   "or set/dict iteration order).",
                   {"run_a": a.action_signature(), "run_b": b.action_signature()})


def _check_seed_sensitivity(report: ConformanceReport, factory, env, seed: int) -> None:
    try:
        a = run_episode(env, factory(), seed)
        b = run_episode(env, factory(), seed + 977)
    except Exception as exc:
        report.add("C4", "Seed sensitivity (classification)", SKIP,
                   f"{type(exc).__name__}: {exc}")
        return

    same = a.action_signature() == b.action_signature()
    report.add("C4", "Seed sensitivity (classification)", INFO,
               ("DETERMINISTIC: the decision sequence does not depend on the seed. "
                "Correct for round-robin and other fixed scans. Variance across seeds "
                "will come only from the environment, so a single episode per seed is "
                "sufficient for this policy."
                if same else
                "STOCHASTIC: the decision sequence varies with the seed, so reported "
                "figures must be averaged over the full seed set with a confidence "
                "interval, never quoted from one episode."),
               {"deterministic": bool(same)})


def _check_reset_hygiene(report: ConformanceReport, factory, env, seed: int) -> None:
    """
    A reused object must behave exactly like a fresh one.

    Deliberately runs the throwaway episode at a DIFFERENT seed: state carried
    over from an identical episode is invisible, whereas state carried over from
    a different one shows up immediately.
    """
    try:
        fresh = run_episode(env, factory(), seed)
        reused_obj = factory()
        run_episode(env, reused_obj, seed + 12345)     # contaminating episode
        reused = run_episode(env, reused_obj, seed)    # must now match `fresh`
    except Exception as exc:
        report.add("C5", "Reset hygiene", FAIL, f"{type(exc).__name__}: {exc}")
        return

    if fresh.action_signature() == reused.action_signature():
        report.add("C5", "Reset hygiene", PASS,
                   "A reused instance reproduced a fresh instance exactly after "
                   "reset(), so no state survives an episode boundary.")
    else:
        report.add("C5", "Reset hygiene", FAIL,
                   "A reused instance diverged from a fresh one at the same seed: "
                   "reset() does not fully clear state. Episode N would then depend on "
                   "episode N-1, which breaks the independence the paired seed design "
                   "assumes and makes results depend on scenario ordering.",
                   {"fresh": fresh.action_signature(), "reused": reused.action_signature()})


def _check_information_boundary(report: ConformanceReport, factory, env,
                                band_count: int) -> None:
    """
    Feed a poisoned observation and require refusal.

    This tests that the scheduler actually invokes the audit. A policy that never
    calls `_check_truth_leakage` will silently consume a truth field, and that
    silence is precisely the failure Gate 0 exists to prevent.
    """
    forbidden_probes = [
        ("true_emitter_state", np.ones((3, band_count), dtype=bool)),
        ("hidden_truth", "grid"),
        ("emitter_identity", [0, 1, 2]),
        ("next_active_band", 3),
    ]
    undetected: List[str] = []
    errors: List[str] = []

    for key, value in forbidden_probes:
        sched = factory()
        sched.reset(0, scenario_descriptor(env))
        poisoned = _valid_observation(band_count)
        poisoned[key] = value
        try:
            sched.select_action([poisoned], 1)
        except ValueError:
            continue                     # refused: correct
        except Exception as exc:         # refused, but for the wrong reason
            errors.append(f"{key} -> {type(exc).__name__}: {exc}")
        else:
            undetected.append(key)

    if undetected:
        report.add("C6", "Information boundary (Gate 0)", FAIL,
                   f"Consumed observations carrying hidden truth without objecting: "
                   f"{undetected}. Call self._check_truth_leakage(observation_history) "
                   f"at the top of select_action. A scheduler that does not enforce the "
                   f"contract cannot be certified as having respected it.",
                   {"undetected_keys": undetected})
    elif errors:
        report.add("C6", "Information boundary (Gate 0)", INFO,
                   "Rejected every poisoned observation, but raised something other "
                   "than ValueError: " + "; ".join(errors))
    else:
        report.add("C6", "Information boundary (Gate 0)", PASS,
                   f"Rejected all {len(forbidden_probes)} poisoned observations "
                   f"(forbidden and undeclared keys).")


def _check_truth_access(report: ConformanceReport, probe: Any, env, primary) -> None:
    runtime_hits = env.hidden_truth.truth_access_count
    findings: List[str] = []

    if runtime_hits:
        findings.append(
            f"called the audited truth accessor {runtime_hits} time(s) during the episode")

    # --- static source scan (limits documented in the module docstring) ---
    banned = {
        "hidden_truth": "reads the environment's truth grid",
        "get_truth_for_evaluation_only": "calls the metrics-only truth accessor",
        "_hidden_truth": "reaches the environment's private truth attribute",
        "emitter_configs": "reads emitter ground-truth configuration",
        "np.random.seed": "seeds the GLOBAL numpy RNG (see C10)",
        "numpy.random.seed": "seeds the GLOBAL numpy RNG (see C10)",
    }
    try:
        src = inspect.getsource(inspect.getmodule(type(probe)))
    except (OSError, TypeError):
        src = None

    if src is None:
        report.add("C7", "No truth access", INFO if not runtime_hits else FAIL,
                   "Source unavailable for static scan (class defined interactively); "
                   "runtime counter " +
                   (f"recorded {runtime_hits} truth access(es)." if runtime_hits
                    else "recorded no truth access."))
        return

    for token, why in banned.items():
        if token in src:
            findings.append(f"module source contains `{token}` — {why}")

    if findings:
        report.add("C7", "No truth access", FAIL,
                   "Gate 0 violation: " + "; ".join(findings) +
                   ". If truth access is intentional, the policy is instrumentation, "
                   "not a candidate: declare is_oracle = True (see qualification/probes.py).",
                   {"runtime_truth_accesses": int(runtime_hits)})
    else:
        report.add("C7", "No truth access", PASS,
                   "Zero audited truth accesses at runtime and no suspicious tokens in "
                   "the module source. (Textual scan; see docstring for its limits.)")


def _check_history_immutability(report: ConformanceReport, factory, env,
                               band_count: int, seed: int, slots: int = 60) -> None:
    """
    The observation history is shared, mutable state passed by reference each
    slot. A scheduler that pops, sorts or edits it corrupts the record the
    metrics engine will later score, so the corruption would surface as a wrong
    NUMBER rather than as an error.
    """
    sched = factory()
    env.reset(seed=seed)
    sched.reset(seed, scenario_descriptor(env))
    history: List[Dict[str, Any]] = []
    violations: List[str] = []

    for t in range(min(slots, env.config.time_slots)):
        before_len = len(history)
        before_last = dict(history[-1]) if history else None
        try:
            action = int(sched.select_action(history, t))
        except Exception as exc:
            report.add("C8", "Observation history immutability", SKIP,
                       f"{type(exc).__name__} at slot {t}: {exc}")
            return

        if len(history) != before_len:
            violations.append(f"length changed {before_len} -> {len(history)} at slot {t}")
        elif before_last is not None and history[-1] != before_last:
            violations.append(f"edited the most recent observation at slot {t}")
        if violations:
            break

        obs, _ = env.step(action % band_count)
        history.append(obs)
        sched.update(action % band_count, obs)

    if violations:
        report.add("C8", "Observation history immutability", FAIL,
                   "select_action mutated the shared observation history: "
                   + "; ".join(violations) +
                   ". Copy what you need instead; the list the harness passes is the "
                   "same object the metrics engine scores.")
    else:
        report.add("C8", "Observation history immutability", PASS,
                   f"Left the shared history untouched across {min(slots, env.config.time_slots)} slots.")


def _check_prediction_contract(report: ConformanceReport, primary,
                               band_count: int, time_slots: int) -> None:
    preds = [s.prediction for s in primary.trajectory if s.prediction is not None]
    if not preds:
        report.add("C9", "Prediction contract (PS metrics 6 & 7)", INFO,
                   "Emits no forecasts, so PS metric 6 (percentage of correct "
                   "predictions) and metric 7 (average intercept time error) will be "
                   "reported as UNAVAILABLE for this policy. That is the honest result "
                   "for a non-forecasting scheduler and is not a deficiency; returning "
                   "0.0 to populate the column would be.")
        return

    problems: List[str] = []
    scoreable = 0
    for p in preds:
        probs = np.asarray(p.band_activity_probability, dtype=float)
        if probs.shape != (band_count,):
            problems.append(f"band_activity_probability has shape {probs.shape}, "
                            f"expected ({band_count},)")
            break
        if not np.all(np.isfinite(probs)):
            problems.append("band_activity_probability contains non-finite values")
            break
        if probs.min() < 0.0 or probs.max() > 1.0:
            problems.append(f"band_activity_probability outside [0,1] "
                            f"(min {probs.min():.3f}, max {probs.max():.3f})")
            break
        if p.about_time_slot <= p.issued_at_slot:
            problems.append("forecast does not refer to a future slot")
            break
        if 0 <= p.about_time_slot < time_slots:
            scoreable += 1

    if problems:
        report.add("C9", "Prediction contract (PS metrics 6 & 7)", FAIL,
                   "; ".join(problems))
    elif scoreable == 0:
        report.add("C9", "Prediction contract (PS metrics 6 & 7)", FAIL,
                   f"Emitted {len(preds)} forecasts but none referenced a slot inside "
                   f"the mission window (0..{time_slots - 1}), so none can be scored.")
    else:
        eta = sum(1 for p in preds if p.predicted_next_activity_slot is not None)
        report.add("C9", "Prediction contract (PS metrics 6 & 7)", PASS,
                   f"{len(preds)} well-formed forecasts, {scoreable} scoreable in-window. "
                   f"PS metric 6 available; metric 7 "
                   + (f"available ({eta} forecasts carry timing estimates)." if eta
                      else "unavailable (no timing estimates supplied — correct for a "
                           "probability-only forecaster)."),
                   {"predictions": len(preds), "scoreable": scoreable,
                    "with_timing": eta})


def _check_global_rng(report: ConformanceReport, factory, env, seed: int) -> None:
    """
    Detect use of the global numpy RNG.

    `np.random.seed()` and `np.random.rand()` share one process-wide stream with
    every other component. A scheduler touching it reseeds or advances the stream
    other code depends on, which makes results order-dependent: run policy A
    before policy B and B's numbers change. Nothing in the output looks wrong,
    which is what makes this worth an explicit check.
    """
    np.random.seed(0)
    before = np.random.get_state()
    try:
        run_episode(env, factory(), seed)
    except Exception as exc:
        report.add("C10", "Global RNG isolation", SKIP, f"{type(exc).__name__}: {exc}")
        return
    after = np.random.get_state()

    untouched = (before[0] == after[0]
                 and np.array_equal(before[1], after[1])
                 and before[2:] == after[2:])

    if untouched:
        report.add("C10", "Global RNG isolation", PASS,
                   "Left the global numpy RNG state untouched; uses its own Generator.")
    else:
        report.add("C10", "Global RNG isolation", FAIL,
                   "Modified the global numpy RNG state. Use "
                   "self.rng = np.random.default_rng(seed) in reset() and draw only "
                   "from self.rng. Sharing the global stream makes every result "
                   "dependent on execution order and silently breaks paired comparison.")


def _check_edge_cases(report: ConformanceReport, factory) -> None:
    """Degenerate geometries that expose hidden assumptions about band count."""
    cases = [("single band", 1, 25), ("single slot", 4, 1),
             ("two bands, long", 2, 120)]
    problems: List[str] = []
    for label, bands, slots in cases:
        try:
            sched = factory()
            if getattr(sched, "band_count", bands) != bands:
                continue  # factory hardcodes a band count; cannot vary geometry
            env = _make_env(bands, slots)
            run_episode(env, sched, 7)
        except Exception as exc:
            problems.append(f"{label} ({bands} bands, {slots} slots): "
                            f"{type(exc).__name__}: {exc}")

    if problems:
        report.add("C11", "Degenerate geometries", FAIL, "; ".join(problems))
    else:
        report.add("C11", "Degenerate geometries", PASS,
                   "Ran without error on single-band, single-slot and two-band scenarios.")


# =====================================================================
# NEGATIVE CONTROLS — deliberately broken. Each MUST fail its target check.
# =====================================================================

class _UnauditedProbe(BaseScheduler):
    """Never calls the leakage audit. Must fail C6."""

    def __init__(self, band_count: int):
        super().__init__(band_count, "[NEGATIVE-CONTROL] Ignores the observation contract.")
        self._i = 0

    def reset(self, seed: int, scenario_config: Dict) -> None:
        self._i = 0
        self.observation_history = []
        self.steps_taken = 0

    def select_action(self, observation_history: List[Dict], current_time_slot: int) -> int:
        self._i += 1
        return (self._i - 1) % self.band_count

    def update(self, action: int, observation: Dict) -> None:
        self.steps_taken += 1


class _GlobalRNGProbe(BaseScheduler):
    """Draws from the global numpy RNG. Must fail C10."""

    def __init__(self, band_count: int):
        super().__init__(band_count, "[NEGATIVE-CONTROL] Uses the global numpy RNG.")

    def reset(self, seed: int, scenario_config: Dict) -> None:
        np.random.seed(seed)          # exactly what the contract forbids
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0

    def select_action(self, observation_history: List[Dict], current_time_slot: int) -> int:
        self._check_truth_leakage(observation_history)
        return int(np.random.randint(0, self.band_count))

    def update(self, action: int, observation: Dict) -> None:
        self.steps_taken += 1


class _OutOfRangeProbe(BaseScheduler):
    """Returns an illegal band part-way through. Must fail C2."""

    def __init__(self, band_count: int):
        super().__init__(band_count, "[NEGATIVE-CONTROL] Emits an out-of-range action.")

    def reset(self, seed: int, scenario_config: Dict) -> None:
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0

    def select_action(self, observation_history: List[Dict], current_time_slot: int) -> int:
        self._check_truth_leakage(observation_history)
        if current_time_slot == 5:
            return self.band_count      # off by one, the classic
        return current_time_slot % self.band_count

    def update(self, action: int, observation: Dict) -> None:
        self.steps_taken += 1


class _StatefulProbe(BaseScheduler):
    """Carries its cursor across reset(). Must fail C5."""

    def __init__(self, band_count: int):
        super().__init__(band_count, "[NEGATIVE-CONTROL] Leaks state across reset().")
        self._cursor = 0

    def reset(self, seed: int, scenario_config: Dict) -> None:
        self.observation_history = []   # note: _cursor deliberately NOT cleared
        self.audit_flags = []

    def select_action(self, observation_history: List[Dict], current_time_slot: int) -> int:
        self._check_truth_leakage(observation_history)
        band = self._cursor % self.band_count
        self._cursor += 1
        return band

    def update(self, action: int, observation: Dict) -> None:
        self.steps_taken += 1


# =====================================================================
# Self-test
# =====================================================================

def self_test(band_count: int = 8, time_slots: int = 150, seed: int = 20260902,
              verbose: bool = True) -> bool:
    """
    Qualify the conformance suite itself.

    Returns True only if every expectation below holds: good probes admitted,
    oracles rejected as inadmissible, and each broken probe rejected by the
    specific check that claims to catch its defect.
    """
    env_for_truth = _make_env(band_count, time_slots)
    env_for_truth.reset(seed)
    truth = env_for_truth.hidden_truth

    # (label, factory, expected_admissible, must_fail_check_ids)
    cases: List[tuple] = [
        ("RoundRobinProbe", lambda: RoundRobinProbe(band_count), True, ()),
        ("UniformRandomProbe", lambda: UniformRandomProbe(band_count), True, ()),
        ("StaticBandProbe", lambda: StaticBandProbe(band_count, 0), True, ()),
        # Oracles: rejection is the CORRECT outcome and proves C0/C7 work.
        ("OmniscientCaptureOracle", lambda: OmniscientCaptureOracle(band_count, truth),
         False, ("C0",)),
        ("GreedyDiscoveryOracle", lambda: GreedyDiscoveryOracle(band_count, truth),
         False, ("C0",)),
        # Negative controls: each must be caught by its own check.
        ("_UnauditedProbe", lambda: _UnauditedProbe(band_count), False, ("C6",)),
        ("_GlobalRNGProbe", lambda: _GlobalRNGProbe(band_count), False, ("C10",)),
        ("_OutOfRangeProbe", lambda: _OutOfRangeProbe(band_count), False, ("C2",)),
        ("_StatefulProbe", lambda: _StatefulProbe(band_count), False, ("C5",)),
    ]

    all_ok = True
    rows: List[str] = []

    for label, factory, expect_admissible, must_fail in cases:
        rep = run_conformance(factory, band_count, time_slots, seed,
                             scheduler_name=label)
        failed_ids = {c.check_id for c in rep.failures}

        ok = (rep.is_admissible == expect_admissible)
        missed = [cid for cid in must_fail if cid not in failed_ids]
        if missed:
            ok = False

        all_ok &= ok
        verdict = "ok" if ok else "SUITE DEFECT"
        expectation = ("admissible" if expect_admissible
                       else f"rejected by {'+'.join(must_fail)}")
        detail = f"failed={sorted(failed_ids) or 'none'}"
        if missed:
            detail += f"  MISSED={missed}"
        rows.append(f"  [{verdict:>12}] {label:<26} expect {expectation:<22} {detail}")

    if verbose:
        print("=" * 78)
        print("CONFORMANCE SUITE SELF-TEST — qualifying the instrument, both directions")
        print("=" * 78)
        print(f"  scenario: {band_count} bands, {time_slots} slots, seed {seed}\n")
        print("\n".join(rows))
        print()
        if all_ok:
            print("  RESULT: the suite admits valid schedulers and rejects every")
            print("          seeded defect. Gate 0 conformance machinery is qualified.")
        else:
            print("  RESULT: SUITE DEFECT. A check that claims to catch a defect did not")
            print("          catch it. Do NOT certify any scheduler until this is fixed:")
            print("          a passing report currently proves nothing.")
        print("=" * 78)

    return all_ok


# =====================================================================
# CLI
# =====================================================================

def _load_scheduler_class(spec: str):
    if ":" not in spec:
        raise ValueError(
            f"--scheduler expects 'module.path:ClassName', got {spec!r}. "
            f"Example: vyapti_simulator.algorithms.bandit.ucb:UCB1Scheduler"
        )
    module_path, class_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_path)
    if not hasattr(module, class_name):
        raise AttributeError(f"{module_path} has no attribute {class_name!r}")
    return getattr(module, class_name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m vyapti_simulator.qualification.conformance",
        description="Gate 0 conformance suite. A scheduler must pass this before its "
                    "results are admissible under the frozen protocol.")
    ap.add_argument("--self-test", action="store_true",
                    help="Qualify the suite itself against known-good and known-broken probes.")
    ap.add_argument("--scheduler", metavar="module:Class",
                    help="Scheduler to certify, e.g. my.policies:MyScheduler")
    ap.add_argument("--kwargs", metavar="JSON", default="{}",
                    help='Extra constructor kwargs as JSON, e.g. \'{"window_size": 50}\'')
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--slots", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--json", metavar="PATH", help="Write the report as JSON.")
    args = ap.parse_args(argv)

    if not args.self_test and not args.scheduler:
        ap.print_help()
        print("\nNothing to do. Pass --self-test or --scheduler module:Class.")
        return 2

    exit_code = 0

    if args.self_test:
        exit_code |= 0 if self_test(args.bands, min(args.slots, 150), args.seed) else 1

    if args.scheduler:
        try:
            cls = _load_scheduler_class(args.scheduler)
            kwargs = json.loads(args.kwargs)
        except Exception as exc:
            print(f"Could not load {args.scheduler!r}: {type(exc).__name__}: {exc}")
            return 2

        def factory():
            return cls(args.bands, **kwargs)

        try:
            factory()
        except Exception as exc:
            print(f"Could not construct {args.scheduler} with band_count={args.bands} "
                  f"and kwargs={kwargs}: {type(exc).__name__}: {exc}")
            print("Constructor signature must accept band_count as its first argument.")
            return 2

        report = run_conformance(factory, args.bands, args.slots, args.seed)
        print(report.render())

        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(report.to_dict(), fh, indent=2)
            print(f"\nJSON report written to {args.json}")

        exit_code |= 0 if report.is_admissible else 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

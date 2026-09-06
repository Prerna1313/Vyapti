"""
PS26055 canonical episode runner — the ONE loop.

=======================================================================
Why this file exists at all
=======================================================================
Before this module there were two ways to run an episode: a hardcoded
`band_choice = t % config.band_count` loop in `simulator.py` that never called a
scheduler, and whatever loop each experiment wrote for itself. That is the
failure mode a common simulator exists to prevent. If the conformance suite
certifies one loop and the experiments run another, the certificate is
worthless: a scheduler can pass conformance and still be driven incorrectly by
the harness that produces the published numbers.

`run_episode` is therefore the single execution path. `qualification/` calls it,
`experiments/` calls it, `simulator.py` calls it. There is no other loop.

=======================================================================
Decision order, and why it is this order
=======================================================================
Per slot t:

    action = scheduler.select_action(history, t)   # sees only past observations
    obs    = env.step(action)                      # truth consulted inside env
    history.append(obs)
    scheduler.update(action, obs)                  # learns from this slot
    pred   = scheduler.predict(t)                  # forecasts t+1 or later

`predict` runs AFTER `update` deliberately. A forecast issued before the current
slot's outcome is known is a different and strictly weaker quantity than one
issued after, and mixing the two across schedulers would make PS metric 6
incomparable. Standardising on "forecast with everything observed up to and
including t, about a slot strictly after t" makes it comparable, and
`BandPrediction.__post_init__` enforces the strictness.

=======================================================================
[SCIENTIFIC] scenario_config is an unaudited leakage channel — closed here
=======================================================================
`scheduler.reset(seed, scenario_config)` takes a dict, and nothing in the type
signature stops a caller from passing the emitter configuration into it. That
would hand every scheduler the periods, phases, hop sequences and arrival slots
of every emitter, i.e. the entire hidden truth, through a channel that the Gate 0
observation audit does not inspect — the audit checks observations, not the
reset payload. A scheduler could then pass conformance while knowing everything.

`scenario_descriptor()` below is consequently a WHITELIST of receiver and
mission constants that a real ES operator would genuinely know before the
mission (how many bands the receiver has, how long a dwell takes, how long the
mission runs). It contains no emitter information of any kind. Callers should
never assemble this dict by hand.

=======================================================================
[SCIENTIFIC] What "paired" means here, precisely
=======================================================================
Two schedulers run at the same seed face a bit-identical truth grid, because
truth generation is a pure function of (seed, emitter configs). That is exact
pairing of the environment, and it is where the variance reduction of the paired
design comes from.

Detector noise is seeded identically but CONSUMED decision-dependently: the
detector stream is drawn once per step, so two policies that dwell differently
reach different points in that stream. Pairing is therefore exact for truth and
only distributional for detector noise. Under the default configuration
(detection_probability=1.0, false_alarm_probability=0.0) the draws cannot change
any outcome, so pairing is exact end-to-end; it becomes approximate as soon as
`use_sensitivity_curve` or a sub-unity Pd is enabled. Any experiment reporting a
paired test under a stochastic detector must state this.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from .environment import PS26055Environment, EmitterConfig
from .scheduler_interface import BaseScheduler, BandPrediction
from .metrics import TrajectoryStep


# =====================================================================
# Scenario descriptor — whitelist, see module docstring
# =====================================================================

#: Receiver and mission constants a scheduler may legitimately know a priori.
#: Adding a key here is a Gate 0 protocol change and requires an audit entry.
PERMITTED_SCENARIO_KEYS = frozenset({
    "band_count", "time_slots", "dwell_time_ms", "retune_time_ms",
    "band_width_mhz", "total_spectrum_mhz", "receiver_ibw_mhz",
})


def scenario_descriptor(env: PS26055Environment) -> Dict[str, Any]:
    """
    Build the reset payload for a scheduler: receiver facts only, no truth.

    Deliberately does NOT include emitter_configs, emitter count, behaviour
    families, periods, phases, SNRs or arrival slots. A scheduler that needs to
    know how many emitters exist must estimate it from observations, which is
    the actual research problem.
    """
    c = env.config
    desc = {
        "band_count": c.band_count,
        "time_slots": c.time_slots,
        "dwell_time_ms": c.dwell_time_ms,
        "retune_time_ms": c.retune_time_ms,
        "band_width_mhz": c.band_width_mhz(),
        "total_spectrum_mhz": c.total_spectrum_mhz,
        "receiver_ibw_mhz": c.receiver_ibw_mhz,
    }
    leaked = set(desc) - PERMITTED_SCENARIO_KEYS
    if leaked:
        raise RuntimeError(
            f"scenario_descriptor built undeclared key(s) {sorted(leaked)}. "
            "The reset payload is a whitelist; see Gate 0."
        )
    return desc


# =====================================================================
# Result container
# =====================================================================

@dataclass
class EpisodeResult:
    """
    Everything one episode produced. Carries no truth grid: metrics are computed
    by handing this trajectory and the grid to MetricsEngine.record_result, which
    is the only component permitted to hold both at once.
    """
    scheduler_name: str
    seed: int
    slots_executed: int
    trajectory: List[TrajectoryStep]
    receiver_accounting: Dict[str, float]
    replay_signature: str
    diagnostics: Dict[str, Any]
    audit_flags: List[str] = field(default_factory=list)

    #: Compact arrays for replay comparison without walking the trajectory.
    actions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp))
    hits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))

    @property
    def predictions_emitted(self) -> int:
        return sum(1 for s in self.trajectory if s.prediction is not None)

    def action_signature(self) -> str:
        """Fingerprint of the decision sequence, for determinism checks."""
        import hashlib
        h = hashlib.sha256(np.ascontiguousarray(self.actions).tobytes()).hexdigest()
        return f"n={self.actions.size}:sha256={h[:32]}"


# =====================================================================
# The loop
# =====================================================================

def run_episode(
    env: PS26055Environment,
    scheduler: BaseScheduler,
    seed: int,
    emitter_family_config: Optional[Sequence[EmitterConfig]] = None,
    max_slots: Optional[int] = None,
    collect_predictions: bool = True,
    scheduler_name: Optional[str] = None,
) -> EpisodeResult:
    """
    Execute one full episode and return its trajectory.

    Parameters
    ----------
    max_slots
        Truncate the episode early. Intended for conformance smoke tests only;
        experiments must run the full mission, because a truncated episode
        right-censors every first-intercept time and silently biases discovery
        metrics toward whichever policy front-loads exploration.
    collect_predictions
        Call `predict()` each slot. Leave True: a scheduler that forecasts but is
        run with this off is reported as making no forecasts, and PS metrics 6
        and 7 then read "unavailable" for a policy that could in fact supply them.

    Notes
    -----
    The observation history handed to `select_action` is the live list, not a
    copy — copying it every slot is O(T^2). A scheduler that mutates it corrupts
    its own input; the conformance suite tests for exactly that rather than
    paying the copy cost on every run.
    """
    name = scheduler_name or type(scheduler).__name__

    env.reset(seed=seed, emitter_family_config=(
        list(emitter_family_config) if emitter_family_config is not None else None))
    scheduler.reset(seed, scenario_descriptor(env))

    band_count = env.config.band_count
    horizon = env.config.time_slots if max_slots is None else min(
        env.config.time_slots, int(max_slots))

    history: List[Dict[str, Any]] = []
    trajectory: List[TrajectoryStep] = []
    actions = np.empty(horizon, dtype=np.intp)
    hits = np.empty(horizon, dtype=bool)

    for t in range(horizon):
        if env.done:
            break

        action = scheduler.select_action(history, t)
        # Range-check here as well as in env.step: the runner can name the
        # offending scheduler, which env.step cannot.
        try:
            action = int(action)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{name}.select_action returned {action!r} at slot {t}, which is not "
                f"an integer band index."
            ) from exc
        if not (0 <= action < band_count):
            raise ValueError(
                f"{name}.select_action returned band {action} at slot {t}, outside "
                f"the valid range 0..{band_count - 1}. Use BaseScheduler."
                f"_validate_action() to catch this in the policy."
            )

        obs, leaked = env.step(action)
        if leaked:
            raise RuntimeError(
                f"Environment reported hidden-state leakage at slot {t}. This is an "
                f"environment defect, not a scheduler defect; Gate 0 cannot pass."
            )

        history.append(obs)
        scheduler.update(action, obs)

        prediction: Optional[BandPrediction] = None
        if collect_predictions:
            prediction = scheduler.predict(t)
            if prediction is not None and not isinstance(prediction, BandPrediction):
                raise TypeError(
                    f"{name}.predict returned {type(prediction).__name__}; it must "
                    f"return a BandPrediction or None."
                )

        trajectory.append(TrajectoryStep(
            time_slot=t, action=action, observation=obs, prediction=prediction))
        actions[t] = action
        hits[t] = bool(obs["hit"])

    executed = len(trajectory)
    return EpisodeResult(
        scheduler_name=name,
        seed=int(seed),
        slots_executed=executed,
        trajectory=trajectory,
        receiver_accounting=env.receiver_accounting(),
        replay_signature=env.replay_signature(),
        diagnostics=scheduler.get_diagnostics(),
        audit_flags=list(getattr(scheduler, "audit_flags", [])),
        actions=actions[:executed],
        hits=hits[:executed],
    )


def run_paired_episodes(
    env: PS26055Environment,
    schedulers: Dict[str, BaseScheduler],
    seed: int,
    emitter_family_config: Optional[Sequence[EmitterConfig]] = None,
    max_slots: Optional[int] = None,
) -> Dict[str, EpisodeResult]:
    """
    Run several schedulers against the identical truth grid at one seed.

    Each policy gets a fresh `env.reset(seed)`, which regenerates a bit-identical
    grid — verified here rather than assumed, because a silent divergence would
    invalidate every paired statistical test computed downstream while leaving
    the numbers looking entirely plausible.
    """
    results: Dict[str, EpisodeResult] = {}
    reference_signature: Optional[str] = None

    for name, sched in schedulers.items():
        res = run_episode(
            env, sched, seed,
            emitter_family_config=emitter_family_config,
            max_slots=max_slots,
            scheduler_name=name,
        )
        if reference_signature is None:
            reference_signature = res.replay_signature
        elif res.replay_signature != reference_signature:
            raise RuntimeError(
                "Paired comparison broken: truth grid differed between schedulers at "
                f"the same seed {seed}.\n  expected: {reference_signature}\n"
                f"  got ({name}): {res.replay_signature}\n"
                "Truth generation must be a pure function of (seed, emitter configs)."
            )
        results[name] = res

    return results


# =====================================================================
# Replay — single-seed debugging and determinism checks
# =====================================================================


@dataclass
class ReplayMismatch:
    """One slot where the replay diverged from the saved episode."""
    slot: int
    action: int
    saved_hit: bool
    replay_hit: bool
    saved_obs_keys: set
    replay_obs_keys: set
    changed_keys: dict  # key -> (saved_value, replay_value)


@dataclass
class ReplayResult:
    """
    Result of replaying a saved EpisodeResult against a fresh env.

    Determinism check: same (seed, emitter_family_config) + same actions →
    identical observations. Any divergence means either:
      (a) the env is non-deterministic (bug in env), or
      (b) the scheduler is non-deterministic (bug in scheduler), or
      (c) the env changed between runs (different parameters, different code)
    """
    original_result: EpisodeResult
    seed: int
    actions_replayed: int
    mismatches: List[ReplayMismatch]
    determinism_verified: bool

    @property
    def is_deterministic(self) -> bool:
        """True iff the replay produced bit-identical observations."""
        return len(self.mismatches) == 0

    def summary(self) -> str:
        if self.is_deterministic:
            return (
                f"ReplayResult(seed={self.seed}): "
                f"DETERMINISTIC — {self.actions_replayed} actions verified."
            )
        return (
            f"ReplayResult(seed={self.seed}): "
            f"NON-DETERMINISTIC — {len(self.mismatches)} mismatch(es), "
            f"first at slot {self.mismatches[0].slot}."
        )


def replay_episode(
    env: PS26055Environment,
    saved: EpisodeResult,
    *,
    emitter_family_config: Optional[Sequence[EmitterConfig]] = None,
) -> ReplayResult:
    """
    Replay a saved EpisodeResult's action sequence against a fresh env
    and verify the observations are identical.

    This is the single-seed debugging tool: when a scheduler fails conformance
    or produces unexpected results, call this with the same seed to reproduce
    exactly what happened. Pass the returned ``ReplayResult`` to
    ``format_replay_result()`` for a human-readable diff.

    Parameters
    ----------
    env : PS26055Environment
        A freshly constructed environment. ``env.reset()`` will be called
        with the same seed as the saved episode.
    saved : EpisodeResult
        The result from a prior ``run_episode()`` call. Its ``actions``
        compact array drives the replay.
    emitter_family_config : Sequence[EmitterConfig], optional
        Same config used in the original episode. Required for deterministic
        truth-grid reconstruction.

    Returns
    -------
    ReplayResult
        ``is_deterministic`` is True iff every replayed observation matched
        the saved one. ``mismatches`` lists all divergences.

    Example
    -------
    ::

        # Run once and save the result
        result = run_episode(env, scheduler, seed=42)

        # Later: replay to check determinism or debug a failure
        fresh_env = PS26055Environment(config=my_config)
        replay = replay_episode(fresh_env, result)
        print(replay.summary())
        if not replay.is_deterministic:
            for m in replay.mismatches:
                print(f"  Slot {m.slot}: hit {m.saved_hit} → {m.replay_hit}")
    """
    seed = saved.seed
    env.reset(seed=seed, emitter_family_config=(
        list(emitter_family_config) if emitter_family_config is not None else None))

    mismatches: List[ReplayMismatch] = []
    horizon = saved.actions.size

    for t in range(horizon):
        if env.done:
            break
        action = int(saved.actions[t])

        obs, leaked = env.step(action)
        if leaked:
            raise RuntimeError(
                f"Replay env.step({action}) reported hidden-state leakage at slot {t}. "
                "This is an env defect, not a scheduler defect."
            )

        saved_obs = saved.trajectory[t].observation
        # Compare only the fields that are in both observations
        common_keys = set(saved_obs.keys()) & set(obs.keys())
        changed_keys = {
            k: (saved_obs[k], obs[k])
            for k in common_keys
            if not _deep_equal(saved_obs[k], obs[k])
        }

        if changed_keys:
            mismatches.append(ReplayMismatch(
                slot=t,
                action=action,
                saved_hit=bool(saved_obs.get("hit", False)),
                replay_hit=bool(obs.get("hit", False)),
                saved_obs_keys=set(saved_obs.keys()),
                replay_obs_keys=set(obs.keys()),
                changed_keys=changed_keys,
            ))
            # Continue replaying even after a mismatch — capture all mismatches

    return ReplayResult(
        original_result=saved,
        seed=seed,
        actions_replayed=min(horizon, env.config.time_slots),
        mismatches=mismatches,
        determinism_verified=len(mismatches) == 0,
    )


def _deep_equal(a: Any, b: Any) -> bool:
    """Rough equality for comparison in replay."""
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=True)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    try:
        return bool(a == b)
    except Exception:
        return a is b


def format_replay_result(replay: ReplayResult, max_slots_shown: int = 10) -> str:
    """
    Human-readable summary of a ``ReplayResult``.

    Shows the first ``max_slots_shown`` mismatches with the
    key-by-key diff for each. Useful for pasting into a bug report.
    """
    lines = [replay.summary()]
    if replay.is_deterministic:
        return lines[0]
    lines.append(f"  First {min(len(replay.mismatches), max_slots_shown)} mismatch(es):")
    for m in replay.mismatches[:max_slots_shown]:
        lines.append(f"  Slot {m.slot} (action={m.action}):")
        for key, (saved_val, replay_val) in m.changed_keys.items():
            lines.append(f"    {key}: {saved_val!r} → {replay_val!r}")
    return "\n".join(lines)


__all__ = [
    "EpisodeResult", "run_episode", "run_paired_episodes",
    "scenario_descriptor", "PERMITTED_SCENARIO_KEYS",
    "ReplayResult", "ReplayMismatch",
    "replay_episode", "format_replay_result",
]

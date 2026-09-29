from __future__ import annotations

"""Shared causal state, timing, reward, registry, and evaluation utilities.

This module sits above the Vyapti TSRD RF environment (System B).
It handles Bayesian belief tracking, periodicity estimation, reward shaping,
and evaluation loop bookkeeping.

REAL API CONTRACT (verified against installed code)
----------------------------------------------------
TSRDTrainWorldPool:
    sample_world(seed: int, emitter_count: int)
        -> (TSRDStareEnvironment, list[dict])
    (NO sample_recipe method)

TSRDStareEnvironment:
    reset(seed=None)           -> None
    step(band)                 -> (obs_dict, False)         [2 values]
    step_training(band, mode)  -> (obs_dict, reward, done)  [3 values]
    step_dwell_training(band, dwell_slots, mode)
                               -> ([obs1, obs2], reward, done) [3 values]
    done                       -> bool (property)
    hidden_truth               -> occupancy object
    receiver_accounting()      -> dict

benchmark_protocol:
    score_recorded_replay(env, trajectory) -> dict   [use for metrics]
    _seed_for_file(stem, seed)             -> int
    _summarize(rows)                       -> dict
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vyapti_simulator.tsrd.benchmark_protocol import (
    score_recorded_replay,
    _summarize,
    _seed_for_file,
)
from vyapti_simulator.core.metrics import TrajectoryStep


# ============================================================
# BENCHMARK CONSTANTS
# ============================================================

N_BANDS = 36
MISSION_S = 30.0
BASE_SLOT_S = 0.050
N_BASE_SLOTS = 600
BAND_STEP_MHZ = 500.0
RECEIVER_BW_MHZ = 1000.0
OVERLAP_FRACTION = 0.50

PD = 0.90
PFA = 0.05

TRAIN_FILES = 2500
VAL_FILES = 100
TEST_FILES = 100
TRAIN_WORLDS_PER_EPOCH = 100
TRAIN_EPOCHS = 10
TOTAL_TRAIN_WORLDS = TRAIN_WORLDS_PER_EPOCH * TRAIN_EPOCHS
CHECKPOINT_STEPS = (100_000, 200_000, 300_000, 400_000)

DEFAULT_SEED = 20260928
TRAIN_RECIPE_SEED = DEFAULT_SEED + 10_000
VAL_REPLAY_SEED = DEFAULT_SEED + 20_000
TEST_REPLAY_SEED = DEFAULT_SEED + 30_000

EPS = 1e-12


# ============================================================
# ENVIRONMENT WRAPPERS  (corrected for real API)
# ============================================================

def safe_reset(env: Any, seed: int) -> None:
    """Reset the environment with a deterministic seed.

    env.reset(seed=...) returns None — the environment holds its
    own internal state. Callers should use state.pre_action_features()
    to build their observation vector after reset.
    """
    env.reset(seed=int(seed))


def safe_step(
    env: Any,
    action: int,
    reward_mode: str = "detector_positive",
) -> tuple[dict, float, bool]:
    """Single 50-ms step using the real step_training() API.

    Returns
    -------
    obs_dict : dict
        Raw observation from the environment (contains 'hit', 'time_slot', etc.)
    reward : float
    done : bool
    """
    obs_dict, reward, done = env.step_training(int(action), reward_mode)
    return obs_dict, float(reward), bool(done)


def safe_step_dwell(
    env: Any,
    action: int,
    dwell_slots: int,
    reward_mode: str = "detector_positive",
) -> tuple[list[dict], float, bool]:
    """Multi-slot (50ms or 100ms) step using the real step_dwell_training() API.

    Returns
    -------
    looks : list[dict]
        One obs_dict per base slot consumed.
    reward : float
        Summed reward over all slots in the dwell.
    done : bool
    """
    looks, reward, done = env.step_dwell_training(
        int(action), int(dwell_slots), reward_mode
    )
    return looks, float(reward), bool(done)


def get_dwell_slots(info_or_env: Any, default: int = 1) -> int:
    """Extract dwell slot count from an obs_dict or fallback to default."""
    if isinstance(info_or_env, dict):
        d = info_or_env.get("dwell_slots", None)
        if d is not None and int(d) > 0:
            return int(d)
    return int(default)


def detector_sequence(looks: list[dict]) -> list[bool]:
    """Extract per-slot hit booleans from a list of obs_dicts.

    Works correctly for both 50ms (1 look) and 100ms (2 looks) dwells.
    Never duplicates a single boolean — each slot has its own dict.
    """
    return [bool(look["hit"]) for look in looks]


def build_trajectory_step(action: int, obs_dict: dict) -> TrajectoryStep:
    """Wrap a single obs_dict into a TrajectoryStep for scorecard computation."""
    return TrajectoryStep(
        action=int(action),
        observation=obs_dict,
        time_slot=int(obs_dict["time_slot"]),
    )


# ============================================================
# TRAIN REGISTRY  (corrected for real pool API)
# ============================================================

def build_replay_registry(files: list[Path]) -> list[dict]:
    """Build replay registry directly from Stare file paths.

    Guarantees no leakage — val/test files are entirely independent
    of the training pool.
    """
    registry: list[dict] = []
    for i, path in enumerate(files):
        registry.append(
            {
                "world_id": i,
                "source_file": str(path),
                "config_id": path.stem,
                "replay": True,
            }
        )
    return registry


def build_train_recipes(pool: Any, *, seed: int = TRAIN_RECIPE_SEED) -> list[dict]:
    """Build deterministic training world registry using sample_world().

    Uses pool.sample_world(world_seed, emitter_count) — the only method
    available on TSRDTrainWorldPool. A deterministic world_seed is derived
    from the global world index so that every run produces identical worlds.
    """
    n_contributions = len(pool.contributions)
    rng = np.random.default_rng(int(seed))
    recipes: list[dict] = []

    for epoch in range(TRAIN_EPOCHS):
        for world_in_epoch in range(TRAIN_WORLDS_PER_EPOCH):
            global_world = len(recipes)

            # Sample emitter count reproducibly
            emitter_count = int(rng.integers(1, n_contributions + 1))

            # Fully deterministic world seed
            world_seed = int(seed + 1_000_003 * global_world)

            # Materialise world to get source provenance
            env, sources = pool.sample_world(world_seed, emitter_count)
            uids = [s["source_label"] for s in sources]
            source_config_ids = sorted(
                {Path(s["source_file"]).stem for s in sources}
            )

            recipes.append(
                {
                    "epoch": epoch + 1,
                    "world_in_epoch": world_in_epoch,
                    "global_world": global_world,
                    "n_emitters": emitter_count,
                    "world_seed": world_seed,
                    "uids": uids,
                    "source_config_ids": source_config_ids,
                }
            )
    return recipes


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


# ============================================================
# BAYESIAN BELIEF FILTER
# ============================================================

class BeliefFilter:
    """Two-state causal HMM filter over N_BANDS channels."""

    def __init__(
        self,
        transition: np.ndarray,
        pd: float = PD,
        pfa: float = PFA,
        prior_active: np.ndarray | None = None,
    ):
        transition = np.asarray(transition, dtype=np.float64)
        if transition.shape != (N_BANDS, 2, 2):
            raise ValueError(
                f"transition must have shape ({N_BANDS},2,2), got {transition.shape}"
            )
        self.transition = transition
        self.pd = float(pd)
        self.pfa = float(pfa)
        self.prior = (
            np.full(N_BANDS, 0.5, dtype=np.float64)
            if prior_active is None
            else np.asarray(prior_active, dtype=np.float64).copy()
        )
        self.belief = self.prior.copy()

    def reset(self, prior_active: np.ndarray | None = None) -> None:
        self.belief = (
            self.prior.copy()
            if prior_active is None
            else np.asarray(prior_active, dtype=np.float64).copy()
        )

    def _predict_one(self, band: int, p: float) -> float:
        p01 = self.transition[band, 0, 1]
        p11 = self.transition[band, 1, 1]
        return float(np.clip((1.0 - p) * p01 + p * p11, 1e-6, 1.0 - 1e-6))

    def _posterior(self, p: float, y: bool) -> float:
        py1 = self.pd if y else (1.0 - self.pd)
        py0 = self.pfa if y else (1.0 - self.pfa)
        den = p * py1 + (1.0 - p) * py0
        return float(np.clip(p * py1 / max(den, EPS), 1e-6, 1.0 - 1e-6))

    @staticmethod
    def entropy(p: np.ndarray | float) -> np.ndarray | float:
        x = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0 - 1e-8)
        h = -(x * np.log(x) + (1.0 - x) * np.log(1.0 - x))
        return h / math.log(2.0)

    def step_observations(self, band: int, ys: Iterable[bool]) -> list[float]:
        """Update belief given a sequence of per-slot hits. Returns pre-update beliefs."""
        band = int(band)
        pre: list[float] = []
        for y in ys:
            p_before = float(self.belief[band])
            pre.append(p_before)
            p_post = self._posterior(p_before, bool(y))
            next_belief = self.belief.copy()
            for b in range(N_BANDS):
                p0 = float(self.belief[b]) if b != band else p_post
                next_belief[b] = self._predict_one(b, p0)
            self.belief = next_belief
        return pre


# ============================================================
# PERIODICITY CUE
# ============================================================

class PeriodicityCue:
    """Causal dwell-level periodicity estimator.

    Timestamps individual positive observations within each dwell
    rather than using a single midpoint. This is required for
    correctness with 100ms (2-slot) dwells.
    """

    def __init__(self, max_events: int = 32, tolerance_fraction: float = 0.15):
        self.max_events = int(max_events)
        self.tolerance_fraction = float(tolerance_fraction)
        self.times: list[list[float]] = [[] for _ in range(N_BANDS)]

    def reset(self) -> None:
        self.times = [[] for _ in range(N_BANDS)]

    def observe_dwell(self, band: int, start_slot: int, positives: list[bool]) -> None:
        """Record per-slot hit timestamps. start_slot is the first slot of the dwell."""
        hist = self.times[int(band)]
        for i, pos in enumerate(positives):
            if pos:
                hist.append(float(start_slot + i))
        if len(hist) > self.max_events:
            del hist[: -self.max_events]

    def estimate(self, band: int, now_slot: int) -> tuple[float, float, float]:
        hist = self.times[int(band)]
        if len(hist) < 3:
            return 0.0, 0.0, 0.0
        gaps = np.diff(np.asarray(hist, dtype=np.float64))
        gaps = gaps[gaps > 0]
        if len(gaps) < 2:
            return 0.0, 0.0, 0.0
        period = float(np.median(gaps))
        if period <= 1.0:
            return 0.0, 0.0, 0.0
        cv = float(np.std(gaps) / max(np.mean(gaps), EPS))
        regularity = float(np.exp(-cv))
        last = float(hist[-1])
        elapsed = max(0.0, float(now_slot) - last)
        remainder = elapsed % period
        phase_error = min(remainder, period - remainder)
        tol = max(self.tolerance_fraction * period, 0.5)
        phase_score = float(np.exp(-phase_error / tol))
        confidence = float(np.clip(regularity * min(1.0, len(gaps) / 6.0), 0.0, 1.0))
        score = float(np.clip(phase_score * confidence, 0.0, 1.0))
        period_norm = float(np.clip(period / N_BASE_SLOTS, 0.0, 1.0))
        return score, confidence, period_norm

    def feature_matrix(self, now_slot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vals = [self.estimate(b, now_slot) for b in range(N_BANDS)]
        return (
            np.asarray([v[0] for v in vals], dtype=np.float32),
            np.asarray([v[1] for v in vals], dtype=np.float32),
            np.asarray([v[2] for v in vals], dtype=np.float32),
        )


# ============================================================
# CAUSAL SCHEDULER STATE
# ============================================================

@dataclass
class CausalSchedulerState:
    belief: BeliefFilter
    periodicity: PeriodicityCue
    last_visit: np.ndarray
    visit_counts: np.ndarray
    previous_action: int = -1
    previous_positive: bool = False
    elapsed_base_slots: int = 0

    @classmethod
    def create(
        cls,
        transition: np.ndarray,
        prior_active: np.ndarray | None = None,
    ) -> "CausalSchedulerState":
        return cls(
            belief=BeliefFilter(transition, prior_active=prior_active),
            periodicity=PeriodicityCue(),
            last_visit=np.full(N_BANDS, -1, dtype=np.int32),
            visit_counts=np.zeros(N_BANDS, dtype=np.int32),
        )

    def reset(self, prior_active: np.ndarray | None = None) -> None:
        self.belief.reset(prior_active)
        self.periodicity.reset()
        self.last_visit.fill(-1)
        self.visit_counts.fill(0)
        self.previous_action = -1
        self.previous_positive = False
        self.elapsed_base_slots = 0

    def step(self, action: int, dwell_slots: int, positives: list[bool]) -> None:
        """Update all causal state after one dwell.

        Call this AFTER extracting pre-action features and computing reward.
        Caches previous_action BEFORE overwriting it so repeat_penalty works.
        """
        self.periodicity.observe_dwell(action, self.elapsed_base_slots, positives)
        self.last_visit[int(action)] = self.elapsed_base_slots
        self.visit_counts[int(action)] += 1
        self.previous_action = int(action)
        self.previous_positive = any(positives)
        self.elapsed_base_slots += int(dwell_slots)

    def pre_action_features(self) -> dict[str, np.ndarray]:
        t = self.elapsed_base_slots
        staleness = np.where(
            self.last_visit >= 0, (t - self.last_visit) / N_BASE_SLOTS, 1.0
        )
        staleness = np.clip(staleness, 0.0, 1.0).astype(np.float32)
        periodic, periodic_conf, period_norm = self.periodicity.feature_matrix(t)
        belief = self.belief.belief.astype(np.float32)
        entropy = np.asarray(BeliefFilter.entropy(belief), dtype=np.float32)
        return {
            "belief": belief,
            "belief_entropy": entropy,
            "staleness": staleness,
            "periodicity": periodic,
            "periodicity_confidence": periodic_conf,
            "period_norm": period_norm,
            "visit_count_norm": np.clip(
                self.visit_counts / 32.0, 0.0, 1.0
            ).astype(np.float32),
        }


# ============================================================
# REWARD ENGINE
# ============================================================

class RewardEngine:
    VALID = {
        "detector_positive",
        "true_detection",
        "first_intercept",
        "causal_belief_hit",
        "causal_balanced",
    }

    def __init__(self, mode: str = "causal_belief_hit"):
        if mode not in self.VALID:
            raise ValueError(f"reward mode {mode!r} not in {sorted(self.VALID)}")
        self.mode = mode
        self.seen_emitters: set[int] = set()
        self.total_base_slots = 0
        self.visit_slots = np.zeros(N_BANDS, dtype=np.float64)

    def reset(self) -> None:
        self.seen_emitters.clear()
        self.total_base_slots = 0
        self.visit_slots.fill(0.0)

    def compute(
        self,
        *,
        action: int,
        dwell_slots: int,
        positives: list[bool],
        prebeliefs: np.ndarray,
        causal_state: CausalSchedulerState,
        previous_action: int,
    ) -> float:
        y = np.asarray(positives, dtype=np.float64)
        reward = 0.0

        if self.mode == "detector_positive":
            reward = float(y.sum())
        elif self.mode == "causal_belief_hit":
            reward = float(np.sum(y * np.asarray(prebeliefs, dtype=np.float64)))
        elif self.mode == "causal_balanced":
            belief_hit = float(np.sum(y * np.asarray(prebeliefs, dtype=np.float64)))
            periodic, _, _ = causal_state.periodicity.feature_matrix(
                causal_state.elapsed_base_slots
            )
            staleness = causal_state.pre_action_features()["staleness"][int(action)]
            fair = 1.0 / N_BANDS
            share = self.visit_slots[int(action)] / max(
                float(self.total_base_slots), 1.0
            )
            concentration_penalty = max(0.0, share - fair) / max(1.0 - fair, EPS)
            # previous_action must be cached BEFORE state.step() is called
            repeat_penalty = (
                0.0
                if previous_action < 0 or previous_action != int(action)
                else 0.002
            )
            reward = (
                belief_hit
                + 0.02 * float(staleness)
                + 0.01 * float(periodic[int(action)])
                - 0.02 * float(concentration_penalty) * float(dwell_slots)
                - repeat_penalty
            )

        self.visit_slots[int(action)] += float(dwell_slots)
        self.total_base_slots += int(dwell_slots)
        return float(reward)


# ============================================================
# AGGREGATE METRICS
# ============================================================

def aggregate_metrics(rows: list[dict]) -> dict[str, float]:
    """Aggregate per-episode scorecard rows using _summarize from benchmark_protocol."""
    if not rows:
        return {}
    return _summarize(rows)


# ============================================================
# EVALUATION LOOP  (corrected for real API)
# ============================================================

def evaluate_policy_on_registry(
    env_factory,
    registry: list[dict],
    policy_fn,
    transition: np.ndarray,
    prior_active: np.ndarray,
    dwell_slots: int = 1,
    reward_mode: str = "causal_belief_hit",
    deterministic_seed: int = DEFAULT_SEED,
) -> tuple[list[dict], dict[str, float]]:
    """Run policy_fn over all replays in registry and return per-episode rows + summary.

    policy_fn(state: CausalSchedulerState) -> int  (band index)

    Uses step_dwell_training() for multi-slot dwells. Collects a TrajectoryStep
    per base slot and calls score_recorded_replay() for the final DRDO scorecard.
    """
    rows: list[dict] = []

    for i, recipe in enumerate(registry):
        env = env_factory(recipe)
        state = CausalSchedulerState.create(transition, prior_active)
        state.reset(prior_active)
        reward_engine = RewardEngine(reward_mode)
        reward_engine.reset()

        file_stem = Path(recipe["source_file"]).stem
        episode_seed = _seed_for_file(file_stem, deterministic_seed + i)
        safe_reset(env, episode_seed)

        trajectory: list[TrajectoryStep] = []
        ppo_action_steps = 0

        while not env.done:
            action = int(policy_fn(state))

            # Cache previous_action BEFORE state.step() overwrites it
            prev_action = state.previous_action

            looks, _, done = safe_step_dwell(
                env, action, dwell_slots, "detector_positive"
            )
            positives = detector_sequence(looks)

            # Build trajectory for scorecard
            for look in looks:
                trajectory.append(build_trajectory_step(action, look))

            # Get pre-update beliefs for reward computation
            pre_seq = state.belief.step_observations(action, positives)

            reward_engine.compute(
                action=action,
                dwell_slots=dwell_slots,
                positives=positives,
                prebeliefs=np.asarray(pre_seq, dtype=np.float32),
                causal_state=state,
                previous_action=prev_action,
            )

            # Update all state AFTER reward
            state.step(action, dwell_slots, positives)
            ppo_action_steps += 1

        # Use score_recorded_replay() — the real metrics API
        metrics = score_recorded_replay(env, trajectory)
        metrics["world_id"] = int(recipe.get("world_id", i))
        metrics["source_file"] = recipe.get("source_file", "")
        metrics["ppo_action_steps"] = ppo_action_steps
        metrics["elapsed_base_slots"] = state.elapsed_base_slots
        rows.append(metrics)

    return rows, aggregate_metrics(rows)

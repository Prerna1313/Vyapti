#!/usr/bin/env python3
"""
Vyapti / PS26055 — Conservative Residual Discrete SAC (TRAIN-250)
=================================================================

Purpose
-------
Learn *deviations* from a causal contextual-Thompson-Sampling prior
instead of relearning exploration/exploitation from scratch.

Core design
-----------
1. Current canonical benchmark:
   - 36 bands over 18 GHz
   - 30 s mission
   - 600 base slots
   - 50 ms base slot
   - native dwell: 29 bands x 1 slot, 7 bands x 2 slots
   - Pd=0.90, PFA=0.05
2. 325-D causal receiver-only observation:
   belief(36)
   staleness(36), sweep-relative
   periodicity(36)
   periodicity confidence(36)
   previous action one-hot(36)
   remaining mission(1)
   native dwell(36)
   time since observed HIT(36), sweep-relative
   visit count normalized(36)
   recent hit-rate(36), K=8 per-band visits
3. Candidate action set is causal and state-dependent:
   [contextual-TS prior,
    top belief,
    top periodic opportunity,
    top staleness,
    top recent-hit,
    top operational-priority candidates]
4. Actor/Q networks score candidate descriptors, not opaque candidate indices.
5. Conservative residual learning:
   - contextual-TS prior candidate is always candidate 0
   - early training is prior-biased
   - evaluation requires positive critic advantage AND sufficient actor mass
   - no RR action is embedded in the learner
6. Reward is NEVER reimplemented here. The environment owns the authoritative
   `truth_based_intercept_utility_v2` reward.

IMPORTANT LIMIT
---------------
No RL code can guarantee improved physical benchmark metrics before the active
reward/environment/evaluator pass is verified. This implementation is designed
to make catastrophic degradation much less likely than unrestricted RL and to
make the learned contribution measurable as "deviation from contextual TS".

Adapter contract
----------------
The adapter module must return an object implementing:

    make_train_world(seed: int) -> env
    make_val_world(recipe: dict) -> env
    make_test_world(recipe: dict) -> env
    load_val_recipes() -> list[dict]
    load_test_recipes() -> list[dict]

The environment must expose:

    reset(seed=seed)
    step_dwell_training(band, dwell_slots, reward_mode)
        -> (list[dict look], reward, done)

Each look must contain one causal detector result under one of:
    hit / detector_positive

For metric evaluation, this file intentionally delegates to the existing
benchmark evaluator through an optional adapter method:

    score_episode(env, trajectory) -> dict

or, when available, directly imports:
    vyapti_simulator.tsrd.benchmark_protocol.score_recorded_replay
    vyapti_simulator.core.metrics.TrajectoryStep

If your active repo already has a stronger replay/evaluation runner, use that
runner for final VAL/TEST numbers; this learner does not replace it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Use the repository's tested causal HMM and observed-hit periodicity code.
from .residual_scheduler_common import (
    CausalSchedulerState,
    N_BANDS,
    PD,
    PFA,
    PeriodicityTracker,
)


# =============================================================================
# FROZEN BENCHMARK CONFIG
# =============================================================================

N_BANDS = int(N_BANDS)
assert N_BANDS == 36, f"Canonical Vyapti benchmark requires 36 bands, got {N_BANDS}"

MISSION_SECONDS = 30.0
BASE_SLOT_SECONDS = 0.050
BASE_SLOTS = 600
RETUNE_SECONDS = 0.0003

# 29 bands use 50 ms and 7 bands use 100 ms.
EXPECTED_DWELL_COUNTS = {1: 29, 2: 7}
SWEEP_SLOTS = 29 * 1 + 7 * 2  # 43 base slots = 2.15 s

OBS_DIM = 10 * N_BANDS + 1  # 361? Corrected below to exact canonical 325.
OBS_DIM = (
    N_BANDS  # belief
    + N_BANDS  # staleness
    + N_BANDS  # periodicity
    + N_BANDS  # periodicity confidence
    + N_BANDS  # previous action
    + 1        # remaining mission
    + N_BANDS  # native dwell
    + N_BANDS  # time since hit
    + N_BANDS  # visit count
    + N_BANDS  # recent hit rate
)
assert OBS_DIM == 325

REWARD_MODE = "truth_based_intercept_utility_v2"

# TRAIN-250-specific HMM calibration already established in the benchmark work.
# A transition JSON passed with --transition-file overrides these values.
CAL_P01 = 0.02220
CAL_P11 = 0.95753
CAL_PRIOR_ACTIVE = 0.34266

RECENT_K = 8
MAX_UNSEEN_AGE = 4.0 * SWEEP_SLOTS

# -----------------------------------------------------------------------------
# Conservative residual SAC configuration.
# -----------------------------------------------------------------------------
CANDIDATE_COUNT = 10  # candidate 0 is always the contextual-TS prior action
TOP_K_BELIEF = 2
TOP_K_PERIODIC = 2
TOP_K_STALE = 2
TOP_K_RECENT = 1
TOP_K_PRIORITY = 2

# Contextual Thompson Sampling prior. Its posterior learns a receiver-observable
# hit rate; only the SAC replay/update uses the hidden-truth mission reward.
TS_CONTEXT_DIM = 10  # bias + belief + staleness + periodic opportunity + conf + recent + uncertainty + revisit + time + dwell
TS_LAMBDA = 2.0
TS_NOISE_SCALE = 0.45
TS_FEEDBACK_SCALE = 1.0
TS_FEEDBACK_SCHEMA = "receiver_hit_rate_v1"

REPLAY_CAPACITY = 300_000
BATCH_SIZE = 128
MIN_REPLAY_SIZE = 4_096
WARMUP_ACTIONS = 5_000
TOTAL_ACTIONS = 400_000
UPDATES_PER_ACTION = 1

GAMMA_BASE = 0.997
TAU = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4
INITIAL_ALPHA = 0.15
TARGET_ENTROPY_FRACTION = 0.70
TARGET_ENTROPY = TARGET_ENTROPY_FRACTION * math.log(CANDIDATE_COUNT)

# Contextual-TS warm-start / conservative residual schedule.
PRIOR_MIX_START = 0.75
PRIOR_MIX_END = 0.10
PRIOR_MIX_DECAY_ACTIONS = 100_000
PRIOR_LOGIT_BIAS_START = 1.50
PRIOR_LOGIT_BIAS_END = 0.00
PRIOR_LOGIT_DECAY_ACTIONS = 120_000

# Eval-time safety gate.
# A learned deviation from contextual TS must have BOTH:
#   1. actor probability >= this threshold
#   2. conservative critic advantage over the TS prior > 0
# Otherwise the contextual-TS prior action is used.
EVAL_MIN_POLICY_PROB = 0.10
EVAL_REQUIRE_POSITIVE_Q_ADVANTAGE = True

CHECKPOINTS = (50_000, 100_000, 200_000, 300_000, 400_000)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# UTILS
# =============================================================================


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_calibrated_transition() -> np.ndarray:
    tr = np.zeros((N_BANDS, 2, 2), dtype=np.float64)
    tr[:, 0, 1] = CAL_P01
    tr[:, 0, 0] = 1.0 - CAL_P01
    tr[:, 1, 1] = CAL_P11
    tr[:, 1, 0] = 1.0 - CAL_P11
    return tr


def load_transition_json(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    transition = np.asarray(payload["transition"], dtype=np.float64)
    prior = np.asarray(
        payload.get("prior_active", np.full(N_BANDS, CAL_PRIOR_ACTIVE)),
        dtype=np.float64,
    )
    if transition.shape != (N_BANDS, 2, 2):
        raise ValueError(f"Transition shape must be (36,2,2), got {transition.shape}")
    if prior.shape != (N_BANDS,):
        raise ValueError(f"prior_active shape must be (36,), got {prior.shape}")
    return transition, prior


# =============================================================================
# WORLD ADAPTER
# =============================================================================


class WorldFactory(Protocol):
    def make_train_world(self, seed: int) -> Any: ...
    def make_val_world(self, recipe: dict[str, Any]) -> Any: ...
    def make_test_world(self, recipe: dict[str, Any]) -> Any: ...
    def load_val_recipes(self) -> list[dict[str, Any]]: ...
    def load_test_recipes(self) -> list[dict[str, Any]]: ...


# =============================================================================
# CAUSAL FEATURE STATE — 325 dimensions
# =============================================================================


class ResidualCausalState:
    """Extends the existing causal state with exactly the canonical 325-D state."""

    def __init__(self, transition: np.ndarray, prior_active: np.ndarray, dwell_slots: np.ndarray) -> None:
        # Use the project's own tested belief + periodicity implementation.
        self.base = CausalSchedulerState.create(transition, prior_active)
        self.transition = np.asarray(transition, dtype=np.float64)
        self.prior_active = np.asarray(prior_active, dtype=np.float64)
        self.dwell_slots = np.asarray(dwell_slots, dtype=np.float32).copy()
        if self.dwell_slots.shape != (N_BANDS,):
            raise ValueError("native dwell vector must have shape (36,)")

        self.last_visit_end = np.full(N_BANDS, -1, dtype=np.int32)
        self.last_hit_end = np.full(N_BANDS, -1, dtype=np.int32)
        self.visit_counts = np.zeros(N_BANDS, dtype=np.int32)
        self.recent_hits = [deque(maxlen=RECENT_K) for _ in range(N_BANDS)]
        self.total_decisions = 0
        self.previous_action = -1
        self.elapsed_base_slots = 0

    def reset(self) -> None:
        self.base.reset(self.prior_active)
        self.last_visit_end.fill(-1)
        self.last_hit_end.fill(-1)
        self.visit_counts.fill(0)
        self.recent_hits = [deque(maxlen=RECENT_K) for _ in range(N_BANDS)]
        self.total_decisions = 0
        self.previous_action = -1
        self.elapsed_base_slots = 0

    def observation(self) -> np.ndarray:
        # Belief and periodicity come from the project's causal estimator.
        base_features = self.base.pre_action_features()
        belief = np.asarray(self.base.belief.belief, dtype=np.float32)
        periodicity = np.asarray(base_features["periodicity"], dtype=np.float32)
        periodic_conf = np.asarray(
            base_features["periodicity_confidence"], dtype=np.float32
        )

        # Sweep-relative staleness. Never-visited bands are intentionally very stale.
        now = int(self.elapsed_base_slots)
        staleness_raw = np.where(
            self.last_visit_end >= 0,
            now - self.last_visit_end,
            MAX_UNSEEN_AGE,
        )
        staleness = np.clip(
            staleness_raw.astype(np.float32) / MAX_UNSEEN_AGE,
            0.0,
            1.0,
        )

        previous_onehot = np.zeros(N_BANDS, dtype=np.float32)
        if 0 <= self.previous_action < N_BANDS:
            previous_onehot[self.previous_action] = 1.0

        remaining = np.asarray(
            [np.clip(1.0 - now / BASE_SLOTS, 0.0, 1.0)],
            dtype=np.float32,
        )

        # Keep actual native dwell semantics visible to the learner.
        # 1-slot -> 0.5, 2-slot -> 1.0.
        native_dwell = self.dwell_slots / 2.0

        since_hit_raw = np.where(
            self.last_hit_end >= 0,
            now - self.last_hit_end,
            MAX_UNSEEN_AGE,
        )
        since_hit = np.clip(
            since_hit_raw.astype(np.float32) / MAX_UNSEEN_AGE,
            0.0,
            1.0,
        )

        visit_norm = np.clip(
            self.visit_counts.astype(np.float32) / max(self.total_decisions, 1),
            0.0,
            1.0,
        )

        recent_rate = np.asarray(
            [float(np.mean(q)) if q else 0.0 for q in self.recent_hits],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                belief,
                staleness,
                periodicity,
                periodic_conf,
                previous_onehot,
                remaining,
                native_dwell,
                since_hit,
                visit_norm,
                recent_rate,
            ]
        ).astype(np.float32)
        if obs.shape != (OBS_DIM,):
            raise RuntimeError(f"325-D observation contract violated: got {obs.shape}")
        return obs

    def belief_vector(self) -> np.ndarray:
        return np.asarray(self.base.belief.belief, dtype=np.float32)

    def periodicity_features(self) -> tuple[np.ndarray, np.ndarray]:
        f = self.base.pre_action_features()
        return (
            np.asarray(f["periodicity"], dtype=np.float32),
            np.asarray(f["periodicity_confidence"], dtype=np.float32),
        )

    def staleness_vector(self) -> np.ndarray:
        now = int(self.elapsed_base_slots)
        raw = np.where(
            self.last_visit_end >= 0,
            now - self.last_visit_end,
            MAX_UNSEEN_AGE,
        )
        return np.clip(raw.astype(np.float32) / MAX_UNSEEN_AGE, 0.0, 1.0)

    def since_hit_vector(self) -> np.ndarray:
        now = int(self.elapsed_base_slots)
        raw = np.where(
            self.last_hit_end >= 0,
            now - self.last_hit_end,
            MAX_UNSEEN_AGE,
        )
        return np.clip(raw.astype(np.float32) / MAX_UNSEEN_AGE, 0.0, 1.0)

    def recent_rate_vector(self) -> np.ndarray:
        return np.asarray(
            [float(np.mean(q)) if q else 0.0 for q in self.recent_hits],
            dtype=np.float32,
        )

    def priority_vector(self) -> np.ndarray:
        """Operational sensing priority; not a hostile-threat classifier."""
        belief = self.belief_vector()
        staleness = self.staleness_vector()
        periodic, conf = self.periodicity_features()
        recent = self.recent_rate_vector()
        entropy = 4.0 * belief * (1.0 - belief)
        return (
            0.45 * belief
            + 0.20 * staleness
            + 0.15 * periodic * conf
            + 0.10 * recent
            + 0.10 * entropy
        ).astype(np.float32)

    def step(self, band: int, positives: Sequence[bool]) -> int:
        band = int(band)
        positives = [bool(x) for x in positives]
        dwell = len(positives)
        if dwell not in (1, 2):
            raise RuntimeError(f"Expected native dwell length 1 or 2, got {dwell}")

        start_slot = int(self.elapsed_base_slots)

        # Update the project's causal belief and periodicity machinery.
        self.base.step(band, dwell, positives)

        # Our canonical 325-D bookkeeping uses end-of-dwell timestamps.
        self.elapsed_base_slots += dwell
        self.last_visit_end[band] = self.elapsed_base_slots
        self.visit_counts[band] += 1
        self.total_decisions += 1
        self.previous_action = band

        for i, pos in enumerate(positives):
            if pos:
                self.last_hit_end[band] = start_slot + i + 1
        self.recent_hits[band].append(float(any(positives)))

        return dwell


# =============================================================================
# NATIVE DWELL DISCOVERY
# =============================================================================


def resolve_dwell_vector(env: Any) -> np.ndarray:
    vals: list[int] = []
    for band in range(N_BANDS):
        found: int | None = None
        for name in ("dwell_slots_for_band", "native_dwell_slots", "get_dwell_slots"):
            fn = getattr(env, name, None)
            if callable(fn):
                try:
                    d = int(fn(band))
                    if d in (1, 2):
                        found = d
                        break
                except Exception:
                    pass
        if found is None:
            for name in ("DWELL_SLOTS", "NATIVE_DWELL_SLOTS", "dwell_slots"):
                arr = getattr(env, name, None)
                if arr is not None and not callable(arr):
                    a = np.asarray(arr).reshape(-1)
                    if len(a) == N_BANDS and int(a[band]) in (1, 2):
                        found = int(a[band])
                        break
        if found is None:
            raise RuntimeError(
                "Could not resolve the environment's native dwell profile. "
                "The residual benchmark requires the existing 29x1 / 7x2 profile."
            )
        vals.append(found)

    arr = np.asarray(vals, dtype=np.int32)
    unique, counts = np.unique(arr, return_counts=True)
    got = {int(u): int(c) for u, c in zip(unique, counts)}
    if got != EXPECTED_DWELL_COUNTS:
        raise RuntimeError(
            f"Native dwell contract failed: expected {EXPECTED_DWELL_COUNTS}, got {got}"
        )
    if int(arr.sum()) != SWEEP_SLOTS:
        raise RuntimeError("Native dwell sweep must equal 43 base slots")
    return arr


def extract_positives(looks: Sequence[Any]) -> list[bool]:
    positives: list[bool] = []
    for look in looks:
        if not isinstance(look, dict):
            raise TypeError(f"Expected look dict, got {type(look)!r}")
        if "hit" in look:
            positives.append(bool(look["hit"]))
        elif "detector_positive" in look:
            positives.append(bool(look["detector_positive"]))
        else:
            raise RuntimeError("Look contains neither 'hit' nor 'detector_positive'")
    if not positives:
        raise RuntimeError("Environment returned zero looks")
    return positives


def contextual_ts_feedback(positives: Sequence[bool]) -> float:
    """Map receiver-observable binary outcomes to causal TS feedback."""
    if not positives:
        return 0.0
    return float(np.mean(np.asarray(positives, dtype=np.float32)))


def env_step(env: Any, band: int, dwell: int) -> tuple[list[dict], float, bool]:
    out = env.step_dwell_training(int(band), int(dwell), REWARD_MODE)
    if not isinstance(out, tuple) or len(out) != 3:
        raise RuntimeError(
            "Expected step_dwell_training(band,dwell_slots,reward_mode) -> (looks,reward,done)"
        )
    looks, reward, done = out
    looks = list(looks)
    if not looks:
        raise RuntimeError("Environment returned no look observations")
    if len(looks) != dwell and not bool(done):
        raise RuntimeError(f"Expected {dwell} look records, got {len(looks)}")
    return looks, float(reward), bool(done)


def time_aware_gamma(dwell: int, switched: bool) -> float:
    sec = float(dwell) * BASE_SLOT_SECONDS
    if switched:
        sec += RETUNE_SECONDS
    return float(GAMMA_BASE ** (sec / BASE_SLOT_SECONDS))


# =============================================================================
# RESIDUAL CANDIDATE GENERATOR
# =============================================================================


@dataclass
class ContextualThompsonSampler:
    """Per-band Bayesian linear contextual Thompson Sampling prior.

    The prior is a benchmark component, not the final scheduler. It supplies
    a causal exploration/exploitation proposal that Residual-SAC can deviate
    from. Posterior feedback is a receiver-observable HIT/MISS rate. The
    environment-owned mission reward is reserved for Residual-SAC learning.
    """

    def __init__(self, n_bands: int, seed: int) -> None:
        self.n_bands = int(n_bands)
        self.dim = int(TS_CONTEXT_DIM)
        self.rng = np.random.default_rng(int(seed))
        self.A_inv = np.tile(
            (np.eye(self.dim, dtype=np.float64) / TS_LAMBDA)[None, :, :],
            (self.n_bands, 1, 1),
        )
        self.b = np.zeros((self.n_bands, self.dim), dtype=np.float64)
        self.pulls = np.zeros(self.n_bands, dtype=np.int64)

    def posterior_mean(self) -> np.ndarray:
        return np.einsum("bij,bj->bi", self.A_inv, self.b).astype(np.float64)

    def sample_scores(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(contexts, dtype=np.float64)
        if x.shape != (self.n_bands, self.dim):
            raise ValueError(f"TS contexts must have shape ({self.n_bands},{self.dim}), got {x.shape}")
        mu = self.posterior_mean()
        mean_score = np.einsum("bd,bd->b", x, mu)
        var = np.einsum("bd,bde,be->b", x, self.A_inv, x)
        std = np.sqrt(np.maximum(var, 1e-10))
        sampled = mean_score + TS_NOISE_SCALE * std * self.rng.standard_normal(self.n_bands)
        return sampled.astype(np.float32), mean_score.astype(np.float32), std.astype(np.float32)

    def update(self, band: int, context: np.ndarray, feedback: float) -> None:
        b = int(band)
        x = np.asarray(context, dtype=np.float64).reshape(self.dim)
        feedback_value = float(feedback)
        if not np.isfinite(feedback_value) or not 0.0 <= feedback_value <= 1.0:
            raise ValueError("Contextual-TS receiver hit-rate feedback must be in [0, 1]")
        y = feedback_value * TS_FEEDBACK_SCALE
        Ainv = self.A_inv[b]
        Ax = Ainv @ x
        denom = 1.0 + float(x @ Ax)
        self.A_inv[b] = Ainv - np.outer(Ax, Ax) / max(denom, 1e-9)
        self.b[b] += x * y
        self.pulls[b] += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "A_inv": self.A_inv,
            "b": self.b,
            "pulls": self.pulls,
            "context_dim": self.dim,
            "feedback_schema": TS_FEEDBACK_SCHEMA,
            "feedback_scale": TS_FEEDBACK_SCALE,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("feedback_schema") != TS_FEEDBACK_SCHEMA:
            raise ValueError(
                "Checkpoint contextual-TS posterior uses a different feedback schema; "
                "train a fresh checkpoint with receiver HIT/MISS feedback"
            )
        if float(payload.get("feedback_scale", float("nan"))) != TS_FEEDBACK_SCALE:
            raise ValueError("Checkpoint uses an incompatible contextual-TS feedback scale")
        self.A_inv = np.asarray(payload["A_inv"], dtype=np.float64)
        self.b = np.asarray(payload["b"], dtype=np.float64)
        self.pulls = np.asarray(payload["pulls"], dtype=np.int64)
        if self.A_inv.shape != (self.n_bands, self.dim, self.dim):
            raise ValueError("Bad contextual-TS A_inv shape in checkpoint")
        if self.b.shape != (self.n_bands, self.dim):
            raise ValueError("Bad contextual-TS b shape in checkpoint")


@dataclass
class CandidateSet:
    bands: np.ndarray                 # [M]
    features: np.ndarray              # [M, C]
    ts_prior_index: int = 0


class ResidualCandidateGenerator:
    """Build a causal candidate set around a contextual-TS prior action."""

    def __init__(self, dwell_slots: np.ndarray, ts: ContextualThompsonSampler) -> None:
        self.dwell_slots = np.asarray(dwell_slots, dtype=np.float32)
        self.ts = ts
        self.feature_dim = 17

    @staticmethod
    def _top_unique(score: np.ndarray, k: int, existing: set[int]) -> list[int]:
        order = np.argsort(-np.asarray(score, dtype=np.float64))
        out: list[int] = []
        for idx in order.tolist():
            i = int(idx)
            if i not in existing:
                out.append(i)
                if len(out) >= k:
                    break
        return out

    def band_contexts(self, state: ResidualCausalState) -> np.ndarray:
        belief = state.belief_vector()
        staleness = state.staleness_vector()
        periodic, conf = state.periodicity_features()
        recent = state.recent_rate_vector()
        since_hit = state.since_hit_vector()
        visits = np.clip(
            state.visit_counts.astype(np.float32) / max(state.total_decisions, 1),
            0.0,
            1.0,
        )
        priority = state.priority_vector()
        opportunity = periodic * conf
        uncertainty = 4.0 * belief * (1.0 - belief)
        remaining = float(np.clip(1.0 - state.elapsed_base_slots / BASE_SLOTS, 0.0, 1.0))

        # 10 causal features used by the TS posterior for each band.
        return np.stack(
            [
                np.ones(N_BANDS, dtype=np.float32),
                belief,
                staleness,
                opportunity,
                conf,
                recent,
                uncertainty,
                1.0 - visits,
                np.full(N_BANDS, remaining, dtype=np.float32),
                self.dwell_slots / 2.0,
            ],
            axis=1,
        ).astype(np.float32)

    def build(self, state: ResidualCausalState) -> CandidateSet:
        contexts = self.band_contexts(state)
        ts_sample, ts_mean, ts_std = self.ts.sample_scores(contexts)

        belief = state.belief_vector()
        periodic, conf = state.periodicity_features()
        staleness = state.staleness_vector()
        recent = state.recent_rate_vector()
        since_hit = state.since_hit_vector()
        visits = np.clip(
            state.visit_counts.astype(np.float32) / max(state.total_decisions, 1),
            0.0,
            1.0,
        )
        priority = state.priority_vector()
        opportunity = periodic * conf
        uncertainty = 4.0 * belief * (1.0 - belief)

        prior = int(np.argmax(ts_sample))
        mean_best = int(np.argmax(ts_mean))

        bands: list[int] = [prior]
        used = {prior}
        for candidate in [mean_best]:
            if candidate not in used:
                bands.append(candidate); used.add(candidate)

        bands += self._top_unique(ts_std, 2, used); used.update(bands)
        bands += self._top_unique(belief, TOP_K_BELIEF, used); used.update(bands)
        bands += self._top_unique(opportunity, TOP_K_PERIODIC, used); used.update(bands)
        bands += self._top_unique(staleness, TOP_K_STALE, used); used.update(bands)
        bands += self._top_unique(recent + 0.25 * belief, TOP_K_RECENT, used); used.update(bands)
        bands += self._top_unique(priority + 0.10 * uncertainty, TOP_K_PRIORITY, used)

        fallback_score = (
            0.35 * ts_sample
            + 0.20 * belief
            + 0.15 * staleness
            + 0.15 * opportunity
            + 0.15 * ts_std
        )
        for b in self._top_unique(fallback_score, N_BANDS, set(bands)):
            bands.append(b)
            if len(bands) >= CANDIDATE_COUNT:
                break

        if len(bands) < CANDIDATE_COUNT:
            for b in np.argsort(-ts_sample).tolist():
                if int(b) not in bands:
                    bands.append(int(b))
                if len(bands) >= CANDIDATE_COUNT:
                    break

        bands = bands[:CANDIDATE_COUNT]
        if bands[0] != prior:
            raise RuntimeError("Candidate 0 must always be the contextual-TS prior action")

        feature_rows: list[np.ndarray] = []
        for b in bands:
            switched = 1.0 if state.previous_action >= 0 and b != state.previous_action else 0.0
            feature_rows.append(
                np.asarray(
                    [
                        float(b) / max(N_BANDS - 1, 1),
                        1.0 if b == prior else 0.0,
                        switched,
                        float(ts_sample[b]),
                        float(ts_mean[b]),
                        float(ts_std[b]),
                        float(belief[b]),
                        float(staleness[b]),
                        float(opportunity[b]),
                        float(conf[b]),
                        float(since_hit[b]),
                        float(recent[b]),
                        float(visits[b]),
                        float(self.dwell_slots[b] / 2.0),
                        float(priority[b]),
                        float(uncertainty[b]),
                        float((ts_sample[b] - ts_mean[b])),
                    ],
                    dtype=np.float32,
                )
            )

        return CandidateSet(
            bands=np.asarray(bands, dtype=np.int64),
            features=np.asarray(feature_rows, dtype=np.float32),
            ts_prior_index=0,
        )

    def context_for_band(self, state: ResidualCausalState, band: int) -> np.ndarray:
        return self.band_contexts(state)[int(band)].copy()


# =============================================================================
# REPLAY
# =============================================================================


@dataclass
class Batch:
    obs: torch.Tensor
    cand: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    next_cand: torch.Tensor
    done: torch.Tensor
    gamma: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, candidate_count: int, cand_dim: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.cand = np.empty((capacity, candidate_count, cand_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_cand = np.empty((capacity, candidate_count, cand_dim), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.gammas = np.empty(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(int(seed))

    def add(
        self,
        obs: np.ndarray,
        cand: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_cand: np.ndarray,
        done: bool,
        gamma: float,
    ) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.cand[i] = cand
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_obs[i] = next_obs
        self.next_cand[i] = next_cand
        self.dones[i] = float(done)
        self.gammas[i] = float(gamma)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return int(self.size)

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return Batch(
            obs=torch.as_tensor(self.obs[idx], device=device),
            cand=torch.as_tensor(self.cand[idx], device=device),
            action=torch.as_tensor(self.actions[idx], device=device, dtype=torch.long),
            reward=torch.as_tensor(self.rewards[idx], device=device),
            next_obs=torch.as_tensor(self.next_obs[idx], device=device),
            next_cand=torch.as_tensor(self.next_cand[idx], device=device),
            done=torch.as_tensor(self.dones[idx], device=device),
            gamma=torch.as_tensor(self.gammas[idx], device=device),
        )


# =============================================================================
# NETWORKS
# =============================================================================


class ObsEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CandidateActor(nn.Module):
    def __init__(self, obs_dim: int, cand_dim: int, n_candidates: int) -> None:
        super().__init__()
        self.encoder = ObsEncoder(obs_dim)
        self.candidate_mlp = nn.Sequential(
            nn.Linear(cand_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.n_candidates = int(n_candidates)

    def score(self, obs: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        # obs: [B,D], cand: [B,M,C]
        h = self.encoder(obs).unsqueeze(1).expand(-1, cand.shape[1], -1)
        c = self.candidate_mlp(cand)
        x = torch.cat([h, c], dim=-1)
        return self.head(x).squeeze(-1)


class CandidateQ(nn.Module):
    def __init__(self, obs_dim: int, cand_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.encoder = ObsEncoder(obs_dim, hidden)
        self.candidate_mlp = nn.Sequential(
            nn.Linear(cand_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 128, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def all_values(self, obs: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        h = self.encoder(obs).unsqueeze(1).expand(-1, cand.shape[1], -1)
        c = self.candidate_mlp(cand)
        return self.head(torch.cat([h, c], dim=-1)).squeeze(-1)


# =============================================================================
# RESIDUAL DISCRETE SAC
# =============================================================================


class ResidualDiscreteSAC:
    def __init__(self, obs_dim: int, cand_dim: int, n_candidates: int, seed: int) -> None:
        self.obs_dim = int(obs_dim)
        self.cand_dim = int(cand_dim)
        self.n_candidates = int(n_candidates)

        self.actor = CandidateActor(obs_dim, cand_dim, n_candidates).to(DEVICE)
        self.q1 = CandidateQ(obs_dim, cand_dim).to(DEVICE)
        self.q2 = CandidateQ(obs_dim, cand_dim).to(DEVICE)
        self.q1_target = CandidateQ(obs_dim, cand_dim).to(DEVICE)
        self.q2_target = CandidateQ(obs_dim, cand_dim).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=CRITIC_LR)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=CRITIC_LR)

        self.log_alpha = torch.tensor(
            math.log(INITIAL_ALPHA),
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=ALPHA_LR)
        self.target_entropy = float(TARGET_ENTROPY)
        self.rng = np.random.default_rng(int(seed))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def prior_bias(self, action_step: int) -> float:
        frac = np.clip(
            float(action_step) / max(PRIOR_LOGIT_DECAY_ACTIONS, 1),
            0.0,
            1.0,
        )
        return float(
            PRIOR_LOGIT_BIAS_START
            + frac * (PRIOR_LOGIT_BIAS_END - PRIOR_LOGIT_BIAS_START)
        )

    def prior_mix(self, action_step: int) -> float:
        frac = np.clip(
            float(action_step) / max(PRIOR_MIX_DECAY_ACTIONS, 1),
            0.0,
            1.0,
        )
        return float(
            PRIOR_MIX_START
            + frac * (PRIOR_MIX_END - PRIOR_MIX_START)
        )

    def policy_distribution(
        self,
        obs: torch.Tensor,
        cand: torch.Tensor,
        prior_bias: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor.score(obs, cand)
        # Candidate 0 is the contextual-TS prior; its explicit bias is a
        # training prior only, never hidden truth.
        logits = logits.clone()
        logits[:, 0] = logits[:, 0] + float(prior_bias)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    @torch.no_grad()
    def select_training_action(
        self,
        obs: np.ndarray,
        candidate_features: np.ndarray,
        action_step: int,
        rng: np.random.Generator,
    ) -> int:
        # Explicit prior mixture protects the data stream from premature
        # exploitation while preserving contextual-TS exploration.
        if rng.random() < self.prior_mix(action_step):
            return 0
        x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        c = torch.as_tensor(
            candidate_features, dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)
        probs, _ = self.policy_distribution(
            x,
            c,
            self.prior_bias(action_step),
        )
        return int(torch.multinomial(probs, 1).item())

    @torch.no_grad()
    def select_eval_action(
        self,
        obs: np.ndarray,
        candidate_features: np.ndarray,
    ) -> tuple[int, dict[str, float]]:
        x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        c = torch.as_tensor(candidate_features, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        probs, _ = self.policy_distribution(x, c, prior_bias=0.0)
        q = torch.minimum(self.q1.all_values(x, c), self.q2.all_values(x, c))
        q_np = q[0].detach().cpu().numpy()
        p_np = probs[0].detach().cpu().numpy()

        best = int(np.argmax(p_np))
        prior_q = float(q_np[0])
        best_q = float(q_np[best])
        advantage = best_q - prior_q

        use_prior = False
        if best != 0:
            if p_np[best] < EVAL_MIN_POLICY_PROB:
                use_prior = True
            if EVAL_REQUIRE_POSITIVE_Q_ADVANTAGE and advantage <= 0.0:
                use_prior = True

        chosen = 0 if use_prior else best
        return chosen, {
            "policy_prob": float(p_np[best]),
            "prior_q": prior_q,
            "best_q": best_q,
            "q_advantage": float(advantage),
            "used_prior": float(chosen == 0),
        }

    def update(self, batch: Batch, prior_bias: float) -> dict[str, float]:
        obs = batch.obs
        cand = batch.cand
        action = batch.action
        reward = batch.reward
        next_obs = batch.next_obs
        next_cand = batch.next_cand
        done = batch.done
        gamma = batch.gamma

        # -------------------- target --------------------
        with torch.no_grad():
            next_probs, next_log_probs = self.policy_distribution(
                next_obs,
                next_cand,
                prior_bias=0.0,
            )
            next_q = torch.minimum(
                self.q1_target.all_values(next_obs, next_cand),
                self.q2_target.all_values(next_obs, next_cand),
            )
            next_v = (
                next_probs * (next_q - self.alpha.detach() * next_log_probs)
            ).sum(dim=-1)
            target = reward + (1.0 - done) * gamma * next_v

        q1_all = self.q1.all_values(obs, cand)
        q2_all = self.q2.all_values(obs, cand)
        q1 = q1_all.gather(1, action[:, None]).squeeze(1)
        q2 = q2_all.gather(1, action[:, None]).squeeze(1)

        q1_loss = F.smooth_l1_loss(q1, target)
        q2_loss = F.smooth_l1_loss(q2, target)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 10.0)
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 10.0)
        self.q2_opt.step()

        # -------------------- actor --------------------
        probs, log_probs = self.policy_distribution(
            obs,
            cand,
            prior_bias=prior_bias,
        )
        with torch.no_grad():
            min_q = torch.minimum(
                self.q1.all_values(obs, cand),
                self.q2.all_values(obs, cand),
            )

        actor_loss = (
            probs * (self.alpha.detach() * log_probs - min_q)
        ).sum(dim=-1).mean()

        # Weak early behavioral prior toward contextual TS. The explicit prior decays;
        # it does not remain a permanent contextual-TS imitation objective.
        prior_nll = -log_probs[:, 0].mean()
        prior_weight = 0.05 * np.clip(prior_bias / max(PRIOR_LOGIT_BIAS_START, 1e-6), 0.0, 1.0)
        actor_loss = actor_loss + prior_weight * prior_nll

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        # -------------------- alpha --------------------
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1)
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 10.0)
        self.alpha_opt.step()

        # -------------------- target networks --------------------
        with torch.no_grad():
            for tp, p in zip(self.q1_target.parameters(), self.q1.parameters()):
                tp.mul_(1.0 - TAU).add_(TAU * p)
            for tp, p in zip(self.q2_target.parameters(), self.q2.parameters()):
                tp.mul_(1.0 - TAU).add_(TAU * p)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "entropy": float(entropy.mean().item()),
            "target_mean": float(target.mean().item()),
            "q_mean": float(torch.minimum(q1, q2).mean().item()),
        }

    def state_dict(self, step: int) -> dict[str, Any]:
        return {
            "step": int(step),
            "obs_dim": self.obs_dim,
            "candidate_dim": self.cand_dim,
            "n_candidates": self.n_candidates,
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "target_entropy": self.target_entropy,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.q1.load_state_dict(payload["q1"])
        self.q2.load_state_dict(payload["q2"])
        self.q1_target.load_state_dict(payload["q1_target"])
        self.q2_target.load_state_dict(payload["q2_target"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.q1_opt.load_state_dict(payload["q1_opt"])
        self.q2_opt.load_state_dict(payload["q2_opt"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(DEVICE))
        self.alpha_opt.load_state_dict(payload["alpha_opt"])
        self.target_entropy = float(payload.get("target_entropy", TARGET_ENTROPY))


# =============================================================================
# TRAINING
# =============================================================================


def save_checkpoint(
    path: Path,
    agent: ResidualDiscreteSAC,
    step: int,
    transition: np.ndarray,
    prior_active: np.ndarray,
    dwell_slots: np.ndarray,
    stats: dict[str, Any],
    ts: ContextualThompsonSampler,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = agent.state_dict(step)
    payload.update(
        {
            "transition": torch.as_tensor(transition, dtype=torch.float64),
            "prior_active": torch.as_tensor(prior_active, dtype=torch.float64),
            "native_dwell_slots": torch.as_tensor(dwell_slots, dtype=torch.int64),
            "ts_state": ts.state_dict(),
            "train_stats": stats,
            "protocol": {
                "algorithm": "Conservative Residual Discrete SAC",
                "obs_dim": OBS_DIM,
                "candidate_count": CANDIDATE_COUNT,
                "contextual_ts_context_dim": TS_CONTEXT_DIM,
                "contextual_ts_lambda": TS_LAMBDA,
                "contextual_ts_noise_scale": TS_NOISE_SCALE,
                "contextual_ts_feedback_schema": TS_FEEDBACK_SCHEMA,
                "contextual_ts_feedback_scale": TS_FEEDBACK_SCALE,
                "candidate_0": "contextual Thompson Sampling prior action",
                "mission_seconds": MISSION_SECONDS,
                "base_slots": BASE_SLOTS,
                "base_slot_seconds": BASE_SLOT_SECONDS,
                "sweep_slots": SWEEP_SLOTS,
                "reward_mode": REWARD_MODE,
                "reward_implementation": "environment-owned",
                "hidden_truth_to_policy": False,
                "calibrated_hmm": {
                    "p01": CAL_P01,
                    "p11": CAL_P11,
                    "prior_active": CAL_PRIOR_ACTIVE,
                },
                "eval_gate": {
                    "min_policy_prob": EVAL_MIN_POLICY_PROB,
                    "positive_q_advantage_required": EVAL_REQUIRE_POSITIVE_Q_ADVANTAGE,
                },
            },
        }
    )
    torch.save(payload, path)


def load_adapter(spec: str) -> WorldFactory:
    if ":" not in spec:
        raise ValueError("--adapter must be module:object")
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)
    factory = obj() if isinstance(obj, type) else obj()
    required = (
        "make_train_world",
        "make_val_world",
        "make_test_world",
        "load_val_recipes",
        "load_test_recipes",
    )
    missing = [x for x in required if not hasattr(factory, x)]
    if missing:
        raise TypeError(f"Adapter {spec} missing: {missing}")
    return factory


def run_episode(
    env: Any,
    agent: ResidualDiscreteSAC,
    state: ResidualCausalState,
    generator: ResidualCandidateGenerator,
    rng: np.random.Generator,
    action_step: int,
    training: bool,
) -> tuple[int, float, int, dict[str, Any]]:
    done = False
    episode_reward = 0.0
    action_count = 0
    base_slots = 0
    diagnostics = {
        "ts_prior_decisions": 0,
        "residual_decisions": 0,
        "eval_gate_prior": 0,
        "q_advantages": [],
    }

    while not done and action_step + action_count < TOTAL_ACTIONS:
        obs = state.observation()
        candidates = generator.build(state)

        if training:
            choice = agent.select_training_action(
                obs,
                candidates.features,
                action_step + action_count,
                rng,
            )
        else:
            choice, diag = agent.select_eval_action(obs, candidates.features)
            diagnostics["eval_gate_prior"] += int(diag["used_prior"])
            diagnostics["q_advantages"].append(float(diag["q_advantage"]))

        if choice == 0:
            diagnostics["ts_prior_decisions"] += 1
        else:
            diagnostics["residual_decisions"] += 1

        band = int(candidates.bands[choice])
        ts_context = generator.context_for_band(state, band)
        dwell = int(state.dwell_slots[band])
        switched = state.previous_action >= 0 and band != state.previous_action

        looks, reward, done = env_step(env, band, dwell)
        positives = extract_positives(looks)
        actual_dwell = len(positives)
        if actual_dwell != dwell and not done:
            raise RuntimeError(
                f"Native dwell mismatch: requested {dwell}, received {actual_dwell}"
            )

        prev_obs = obs
        prev_cand = candidates.features.copy()
        gamma = time_aware_gamma(actual_dwell, switched)

        if training:
            generator.ts.update(
                band, ts_context, contextual_ts_feedback(positives)
            )

        state.step(band, positives)
        next_obs = state.observation()
        next_candidates = generator.build(state)

        if training:
            # Caller owns the replay buffer; append tuple through return payload.
            pass

        episode_reward += float(reward)
        action_count += 1
        base_slots += actual_dwell

    return action_count, episode_reward, base_slots, diagnostics


# =============================================================================
# METADATA / MANIFEST
# =============================================================================


def write_manifest(
    output_dir: Path,
    transition: np.ndarray,
    prior_active: np.ndarray,
    dwell_slots: np.ndarray,
    seed: int,
    eval_seed_base: int = 420000,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "Conservative Residual Discrete SAC",
        "seed": int(seed),
        "eval_seed_base": int(eval_seed_base),
        "device": str(DEVICE),
        "benchmark": {
            "mission_seconds": MISSION_SECONDS,
            "base_slot_seconds": BASE_SLOT_SECONDS,
            "base_slots": BASE_SLOTS,
            "n_bands": N_BANDS,
            "native_dwell_slots": dwell_slots.astype(int).tolist(),
            "sweep_slots": SWEEP_SLOTS,
            "sweep_seconds": SWEEP_SLOTS * BASE_SLOT_SECONDS,
            "pd": float(PD),
            "pfa": float(PFA),
        },
        "observation_contract": {
            "belief": 36,
            "staleness_sweep_relative": 36,
            "periodicity": 36,
            "periodicity_confidence": 36,
            "previous_action_one_hot": 36,
            "remaining_mission": 1,
            "native_dwell": 36,
            "time_since_hit_sweep_relative": 36,
            "visit_count_norm": 36,
            "recent_hit_rate_k8": 36,
            "total": OBS_DIM,
            "hidden_truth_to_policy": False,
        },
        "residual_contract": {
            "candidate_0": "contextual Thompson Sampling prior action",
            "candidate_count": CANDIDATE_COUNT,
            "prior_mix_start": PRIOR_MIX_START,
            "prior_mix_end": PRIOR_MIX_END,
            "prior_logit_bias_start": PRIOR_LOGIT_BIAS_START,
            "prior_logit_bias_end": PRIOR_LOGIT_BIAS_END,
            "eval_min_policy_prob": EVAL_MIN_POLICY_PROB,
            "eval_positive_q_advantage_required": EVAL_REQUIRE_POSITIVE_Q_ADVANTAGE,
            "contextual_ts_feedback_schema": TS_FEEDBACK_SCHEMA,
            "contextual_ts_feedback_scale": TS_FEEDBACK_SCALE,
            "contextual_ts_feedback_source": "mean receiver-observed HIT/MISS outcomes per action",
        },
        "reward": {
            "mode": REWARD_MODE,
            "owner": "existing environment",
            "reimplemented_in_learner": False,
        },
        "hmm": {
            "p01_default": CAL_P01,
            "p11_default": CAL_P11,
            "prior_default": CAL_PRIOR_ACTIVE,
            "transition_file_used": False,
        },
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# =============================================================================
# TRAIN LOOP
# =============================================================================


def train(
    factory: WorldFactory,
    output_dir: Path,
    seed: int,
    transition: np.ndarray,
    prior_active: np.ndarray,
    max_actions: int,
    eval_seed_base: int = 420000,
) -> None:
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create one representative training environment to validate the dwell contract.
    probe_env = factory.make_train_world(int(seed))
    try:
        probe_env.reset(seed=int(seed))
    except TypeError:
        probe_env.reset(int(seed))
    dwell_slots = resolve_dwell_vector(probe_env)

    write_manifest(
        output_dir, transition, prior_active, dwell_slots, seed, eval_seed_base
    )

    ts = ContextualThompsonSampler(N_BANDS, seed=seed + 4242)

    agent = ResidualDiscreteSAC(
        OBS_DIM,
        cand_dim=17,
        n_candidates=CANDIDATE_COUNT,
        seed=seed,
    )
    generator = ResidualCandidateGenerator(dwell_slots, ts)
    replay = ReplayBuffer(
        REPLAY_CAPACITY,
        OBS_DIM,
        CANDIDATE_COUNT,
        generator.feature_dim,
        seed + 777,
    )

    rng = np.random.default_rng(seed + 123456)
    reward_window: deque[float] = deque(maxlen=100)
    last_stats: dict[str, Any] = {}
    next_checkpoint_idx = 0
    global_action_step = 0
    episode_count = 0
    wall0 = time.perf_counter()

    while global_action_step < max_actions:
        world_seed = int(rng.integers(1, np.iinfo(np.int32).max))
        env = factory.make_train_world(world_seed)
        try:
            env.reset(seed=world_seed)
        except TypeError:
            env.reset(world_seed)

        # Reuse the already validated native dwell vector; do not let a training
        # world silently change the benchmark.
        current_dwell = resolve_dwell_vector(env)
        if not np.array_equal(current_dwell, dwell_slots):
            raise RuntimeError("Native dwell profile changed between training worlds")

        state = ResidualCausalState(transition, prior_active, dwell_slots)
        state.reset()
        done = False
        ep_reward = 0.0
        ep_actions = 0

        while not done and global_action_step < max_actions:
            obs = state.observation()
            candidates = generator.build(state)

            if global_action_step < WARMUP_ACTIONS:
                # Collect mostly-prior data, but retain a small amount of
                # candidate exploration to prevent a degenerate TS posterior.
                choice = 0 if rng.random() < 0.70 else int(rng.integers(0, CANDIDATE_COUNT))
            else:
                choice = agent.select_training_action(
                    obs,
                    candidates.features,
                    global_action_step,
                    rng,
                )

            band = int(candidates.bands[choice])
            ts_context = generator.context_for_band(state, band)
            dwell = int(state.dwell_slots[band])
            switched = state.previous_action >= 0 and band != state.previous_action

            looks, reward, done = env_step(env, band, dwell)
            positives = extract_positives(looks)
            actual_dwell = len(positives)
            if actual_dwell != dwell and not done:
                raise RuntimeError(
                    f"Requested dwell={dwell}, received {actual_dwell} before mission end"
                )

            # Store the state BEFORE consuming current observations.
            prev_obs = obs.copy()
            prev_cand = candidates.features.copy()
            gamma = time_aware_gamma(actual_dwell, switched)

            # Keep hidden-truth mission reward as the SAC target only.
            # Contextual TS adapts from receiver-visible HIT/MISS outcomes.
            ts.update(band, ts_context, contextual_ts_feedback(positives))

            state.step(band, positives)
            next_obs = state.observation()
            next_candidates = generator.build(state)

            replay.add(
                prev_obs,
                prev_cand,
                choice,
                reward,
                next_obs,
                next_candidates.features,
                done,
                gamma,
            )

            global_action_step += 1
            ep_actions += 1
            ep_reward += float(reward)
            reward_window.append(float(reward))

            if global_action_step >= WARMUP_ACTIONS and len(replay) >= MIN_REPLAY_SIZE:
                for _ in range(UPDATES_PER_ACTION):
                    batch = replay.sample(BATCH_SIZE, DEVICE)
                    last_stats = agent.update(
                        batch,
                        prior_bias=agent.prior_bias(global_action_step),
                    )

            if (
                next_checkpoint_idx < len(CHECKPOINTS)
                and global_action_step >= CHECKPOINTS[next_checkpoint_idx]
            ):
                ckpt_step = CHECKPOINTS[next_checkpoint_idx]
                ckpt = output_dir / "checkpoints" / f"residual_sac_step_{ckpt_step:06d}.pt"
                save_checkpoint(
                    ckpt,
                    agent,
                    global_action_step,
                    transition,
                    prior_active,
                    dwell_slots,
                    {
                        "mean_reward_last_100": float(np.mean(reward_window)) if reward_window else 0.0,
                        "episodes": episode_count,
                        "replay_size": len(replay),
                        **last_stats,
                    },
                    ts,
                )
                print(
                    f"[CHECKPOINT] actions={global_action_step:,} "
                    f"replay={len(replay):,} "
                    f"reward100={np.mean(reward_window) if reward_window else 0.0:.6g} "
                    f"alpha={last_stats.get('alpha', float(agent.alpha.item())):.5f}"
                )
                next_checkpoint_idx += 1

        episode_count += 1
        if episode_count % 25 == 0:
            elapsed = time.perf_counter() - wall0
            rate = global_action_step / max(elapsed, 1e-9)
            print(
                f"episode={episode_count:5d} "
                f"actions={global_action_step:7d}/{max_actions} "
                f"rate={rate:7.2f}/s "
                f"ep_actions={ep_actions:4d} "
                f"ep_reward={ep_reward:10.6f} "
                f"replay={len(replay):7d}"
            )

    final_ckpt = output_dir / "checkpoints" / "residual_sac_final.pt"
    save_checkpoint(
        final_ckpt,
        agent,
        global_action_step,
        transition,
        prior_active,
        dwell_slots,
        {
            "mean_reward_last_100": float(np.mean(reward_window)) if reward_window else 0.0,
            "episodes": episode_count,
            "replay_size": len(replay),
            **last_stats,
        },
        ts,
    )

    elapsed = time.perf_counter() - wall0
    print(
        f"TRAIN COMPLETE: {global_action_step:,} actions, "
        f"episodes={episode_count}, wall={elapsed/3600.0:.2f} h, "
        f"final={final_ckpt}"
    )


# =============================================================================
# DETERMINISTIC POLICY ROLLOUT FOR FINAL EVALUATION
# =============================================================================


def load_agent(checkpoint: Path) -> tuple[ResidualDiscreteSAC, np.ndarray, np.ndarray, np.ndarray, ContextualThompsonSampler]:
    payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    agent = ResidualDiscreteSAC(
        int(payload["obs_dim"]),
        int(payload["candidate_dim"]),
        int(payload["n_candidates"]),
        seed=0,
    )
    agent.load_state_dict(payload)
    transition = payload["transition"].cpu().numpy()
    prior = payload["prior_active"].cpu().numpy()
    dwell = payload["native_dwell_slots"].cpu().numpy().astype(np.int32)
    ts = ContextualThompsonSampler(N_BANDS, seed=0)
    ts.load_state_dict(payload["ts_state"])
    return agent, transition, prior, dwell, ts


def rollout_policy_on_world(
    env: Any,
    agent: ResidualDiscreteSAC,
    transition: np.ndarray,
    prior_active: np.ndarray,
    dwell_slots: np.ndarray,
    ts: ContextualThompsonSampler,
) -> tuple[list[Any], dict[str, Any]]:
    generator = ResidualCandidateGenerator(dwell_slots, ts)
    state = ResidualCausalState(transition, prior_active, dwell_slots)
    state.reset()
    trajectory: list[Any] = []
    diag = {
        "decisions": 0,
        "residual_decisions": 0,
        "ts_prior_decisions": 0,
        "q_advantages": [],
        "policy_probs": [],
    }

    # TrajectoryStep is optional here so the learner remains importable outside
    # the full benchmark package.
    try:
        from vyapti_simulator.core.metrics import TrajectoryStep
    except Exception:
        TrajectoryStep = None

    done = False
    while not done:
        obs = state.observation()
        cand = generator.build(state)
        choice, info = agent.select_eval_action(obs, cand.features)
        band = int(cand.bands[choice])
        ts_context = generator.context_for_band(state, band)
        dwell = int(dwell_slots[band])
        looks, _reward, done = env_step(env, band, dwell)
        positives = extract_positives(looks)
        ts.update(band, ts_context, contextual_ts_feedback(positives))

        if TrajectoryStep is not None:
            for look in looks:
                if isinstance(look, dict) and "time_slot" in look:
                    trajectory.append(
                        TrajectoryStep(
                            action=band,
                            time_slot=int(look["time_slot"]),
                            observation=look,
                        )
                    )

        state.step(band, positives)
        diag["decisions"] += 1
        diag["residual_decisions"] += int(choice != 0)
        diag["ts_prior_decisions"] += int(choice == 0)
        diag["q_advantages"].append(float(info["q_advantage"]))
        diag["policy_probs"].append(float(info["policy_prob"]))

    return trajectory, diag


def evaluate_checkpoint(
    factory: WorldFactory,
    checkpoint: Path,
    split: str,
    recipes: Sequence[dict[str, Any]],
    eval_seed_base: int,
) -> dict[str, Any]:
    agent, transition, prior, dwell, ts = load_agent(checkpoint)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    # Prefer the project's own benchmark scorer via adapter hook.
    scorer = getattr(factory, "score_episode", None)

    try:
        from vyapti_simulator.system_b.tsrd.benchmark_protocol import _seed_for_file
    except Exception:
        _seed_for_file = None
    try:
        from vyapti_simulator.system_b.tsrd.replay_scorecard import score_recorded_replay
    except Exception:
        score_recorded_replay = None

    for i, recipe in enumerate(recipes):
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(recipes):
            print(
                f"[EVAL] {split.upper()} world {i + 1}/{len(recipes)} "
                f"{recipe.get('catalog_world_id', recipe.get('world_id', i))}",
                flush=True,
            )
        if split == "val":
            env = factory.make_val_world(recipe)
        elif split == "test":
            env = factory.make_test_world(recipe)
        else:
            raise ValueError(split)

        # The shared frozen catalog pins receiver randomness into each world
        # identity. This separate base controls only policy-side TS sampling.
        receiver_seed = int(recipe.get("receiver_seed", eval_seed_base + i))
        policy_rng_seed = int(eval_seed_base + i)
        if _seed_for_file is not None and "source_file" in recipe:
            file_seed = int(
                _seed_for_file(Path(recipe["source_file"]).stem, eval_seed_base)
            )
            policy_rng_seed = file_seed
            if "receiver_seed" not in recipe:
                receiver_seed = file_seed

        try:
            env.reset(seed=receiver_seed)
        except TypeError:
            env.reset(receiver_seed)

        eval_ts = ContextualThompsonSampler(N_BANDS, seed=policy_rng_seed)
        eval_ts.load_state_dict(ts.state_dict())
        trajectory, diag = rollout_policy_on_world(
            env,
            agent,
            transition,
            prior,
            dwell,
            eval_ts,
        )
        diagnostics.append(diag)

        if callable(scorer):
            metric = dict(scorer(env, trajectory))
        elif score_recorded_replay is not None and trajectory:
            metric = dict(score_recorded_replay(env, trajectory))
        elif hasattr(env, "metrics"):
            metric = dict(env.metrics())
        elif hasattr(env, "scorecard"):
            metric = dict(env.scorecard())
        else:
            metric = {"world_id": i, "warning": "No benchmark scorer found"}

        metric["world_id"] = int(recipe.get("world_id", i))
        metric["receiver_seed"] = receiver_seed
        metric["policy_eval_seed"] = policy_rng_seed
        rows.append(metric)

    return {
        "split": split,
        "checkpoint": str(checkpoint),
        "eval_seed_base": int(eval_seed_base),
        "n_worlds": len(rows),
        "rows": rows,
        "diagnostics": diagnostics,
    }


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Vyapti conservative residual discrete SAC")
    parser.add_argument("--adapter", required=True, help="module:factory/class")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--eval-seed-base", type=int, default=420000)
    parser.add_argument("--transition-file", type=Path, default=None)
    parser.add_argument("--max-actions", type=int, default=TOTAL_ACTIONS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--eval-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--eval-limit", type=int, default=None)
    args = parser.parse_args()

    global DEVICE
    if args.device == "cpu":
        DEVICE = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        DEVICE = torch.device("cuda")

    if args.transition_file is None:
        transition = make_calibrated_transition()
        prior = np.full(N_BANDS, CAL_PRIOR_ACTIVE, dtype=np.float64)
        print(
            f"Using calibrated TRAIN-250 HMM: P01={CAL_P01:.5f}, "
            f"P11={CAL_P11:.5f}, prior={CAL_PRIOR_ACTIVE:.5f}"
        )
        transition_override = False
    else:
        transition, prior = load_transition_json(args.transition_file)
        transition_override = True
        print(f"Loaded HMM transition/prior from {args.transition_file}")

    factory = load_adapter(args.adapter)

    if args.eval_checkpoint is not None:
        recipes = (
            factory.load_val_recipes()
            if args.eval_split == "val"
            else factory.load_test_recipes()
        )
        if args.eval_limit is not None:
            recipes = recipes[: int(args.eval_limit)]
        result = evaluate_checkpoint(
            factory,
            args.eval_checkpoint,
            args.eval_split,
            recipes,
            eval_seed_base=args.eval_seed_base,
        )
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"evaluation_{args.eval_split}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(
            f"Evaluation complete: {args.eval_split} worlds={result['n_worlds']} "
            f"saved to {out / f'evaluation_{args.eval_split}.json'}"
        )
        return

        train(
            factory=factory,
            output_dir=Path(args.output_dir),
            seed=args.seed,
            transition=transition,
            prior_active=prior,
            max_actions=int(args.max_actions),
            eval_seed_base=args.eval_seed_base,
        )


if __name__ == "__main__":
    main()

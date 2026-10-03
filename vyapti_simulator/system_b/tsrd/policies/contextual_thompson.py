#!/usr/bin/env python3
"""
Vyapti / PS26055
Contextual Thompson Sampling + Bayesian Belief + Causal Periodicity

CURRENT BENCHMARK
-----------------
- Mode B reconstructed hidden worlds
- TRAIN-250 source pool
- 36 frequency bands
- 30 s mission
- 50 ms base slot
- 600 base slots
- native dwell = 50 ms or 100 ms depending on band
- 29 bands have 1 base slot dwell
- 7 bands have 2 base slot dwell

IMPORTANT INFORMATION BOUNDARY
-------------------------------
This policy never accesses:
    - hidden truth
    - emitter IDs
    - world occupancy grids
    - future activity
    - emitter configurations
    - generated world internals

It only consumes the shared runner's PublicTransition:
    transition.reward
    transition.action
    transition.state.time_slot
    transition.next_state.time_slot
    transition.next_state.previous_observation.hit
    transition.next_state.previous_observation.selected_band

REWARD
------
The environment/shared runner owns the authoritative reward.

This policy does NOT reconstruct reward.

The only reward used by Thompson Sampling is:

    reward = float(transition.reward)

Therefore SAC, PPO, UCB+Belief and this CTS policy can all be compared
under the exact same environment reward.

ARCHITECTURE
------------
For each candidate band a, construct a causal context:

    x(a,t) =
        [1,
         band_one_hot(a),
         current_belief(a),
         predicted_belief_after_dwell(a),
         aggregate_hit_probability(a),
         periodicity_score(a),
         periodicity_confidence(a),
         staleness(a),
         remaining_time,
         dwell_length(a),
         switch_indicator(a)]

A linear Gaussian contextual model is learned:

    E[r | x] = x^T theta

Bayesian posterior:

    A <- A + x x^T
    b <- b + x r

    theta_hat = A^{-1} b

At action selection:

    theta_t ~ N(theta_hat, v^2 A^{-1})

    action = argmax_a x(a,t)^T theta_t

This is a contextual-bandit baseline, not a full RL value function.
It deliberately has no Bellman target.

DWELL
-----
Dwell is treated as an action-context feature, not an added reward term.

For every candidate band:
    - native dwell duration is included in the context
    - predicted activity after that dwell is included
    - probability of at least one HIT during the dwell is included
    - switching is included as a contextual feature

This allows the model to learn whether a longer dwell is useful in a
given causal context without changing the environment reward.

HMM
---
Uses a factorised two-state HMM:

    state 0 = inactive
    state 1 = active

Observation model:

    P(HIT | active)   = Pd
    P(HIT | inactive) = Pfa

Current TRAIN-250 pooled calibration:
    P01 = 0.02220
    P11 = 0.95753
    prior_active = 0.34266

These are TRAIN-250 calibration values, not universal TSRD constants.

PERIODICITY
-----------
Periodicity is inferred only from previously observed receiver HIT events.

A HIT observed during a 2-slot dwell is timestamped at the dwell end because
the public scheduler observation does not expose the exact within-dwell
positive time.

The periodicity cue uses:
    - median HIT-to-HIT gap
    - coefficient of variation of gaps
    - phase proximity
    - confidence based on number of observed gaps

CHECKPOINTING
-------------
The checkpoint stores:
    - posterior A
    - posterior b
    - Cholesky factor
    - training action count
    - warmup visit counts
    - RNG state

No world is stored.

RUNNER API
----------
Matches the current shared algorithm interface:

    create(*, bands, seed, settings, checkpoint=None)

    reset_episode(training=...)
    select_action(public_state, training=...)
    observe(transition, training=...)
    end_episode(training=...)
    save(path)

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..algorithm_interface import PublicTransition


# =============================================================================
# FROZEN VYAPTI BENCHMARK
# =============================================================================

N_BANDS = 36

MISSION_SECONDS = 30.0
BASE_SLOT_SECONDS = 0.050
BASE_SLOTS = 600

PD = 0.90
PFA = 0.05

# TRAIN-250 pooled calibration supplied for the current benchmark.
TRAIN250_P01 = 0.02220
TRAIN250_P11 = 0.95753
TRAIN250_PRIOR_ACTIVE = 0.34266

EPS = 1e-12


# =============================================================================
# DEFAULT CTS SETTINGS
# =============================================================================

# Prior precision for Bayesian linear model.
#
# This is a numerical prior, NOT a Vyapti physical constant.
DEFAULT_PRIOR_PRECISION = 1.0

# Posterior sampling scale.
#
# This controls how strongly Thompson sampling explores.
# It is a tunable algorithm parameter, not a literature constant.
DEFAULT_TS_SCALE = 0.05

# Reward-noise standard deviation in the Bayesian linear model.
#
# IMPORTANT:
# This does not modify transition.reward.
# It only determines the assumed observation noise in the Bayesian posterior.
#
# Because the Vyapti reward is already normalized and relatively small, using
# sigma ~ 1 would make the posterior learn unnecessarily slowly.
DEFAULT_REWARD_NOISE_STD = 0.01

# Force complete band coverage early in training before relying entirely
# on Thompson sampling.
DEFAULT_WARMUP_ACTIONS = 72

# Periodicity history per band.
DEFAULT_PERIODICITY_HISTORY = 32

# Phase tolerance as a fraction of estimated period.
DEFAULT_PERIODICITY_TOLERANCE = 0.15


# Context:
#
# 1                   intercept
# 36                  band identity
# 1                   current active belief
# 1                   predicted active belief after dwell
# 1                   aggregate HIT probability
# 1                   periodicity score
# 1                   periodicity confidence
# 1                   staleness
# 1                   remaining mission fraction
# 1                   normalized dwell
# 1                   switch indicator
#
# total = 46
#
CONTEXT_DIM = 46


# =============================================================================
# UTILITY
# =============================================================================

def _as_band_array(value: Any, bands: int, name: str) -> np.ndarray:
    """
    Convert either:
        scalar
    or:
        length-bands sequence

    into a float64 array.

    This lets TRAIN-250 pooled calibration be replaced later by per-band
    calibration without changing the rest of the policy.
    """
    arr = np.asarray(value, dtype=np.float64)

    if arr.ndim == 0:
        return np.full(bands, float(arr), dtype=np.float64)

    if arr.shape != (bands,):
        raise ValueError(
            f"{name} must be a scalar or shape ({bands},), got {arr.shape}"
        )

    return arr.copy()


def _validate_probability_array(
    values: np.ndarray,
    name: str,
    lower: float = 0.0,
    upper: float = 1.0,
) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")

    if np.any(values <= lower) or np.any(values >= upper):
        raise ValueError(
            f"{name} must be strictly inside ({lower}, {upper})"
        )


# =============================================================================
# CAUSAL TWO-STATE HMM
# =============================================================================

class TwoStateHMMBelief:
    """
    Factorised two-state HMM with exact aggregate-dwell filtering.

    The scheduler observes one aggregate HIT/MISS outcome per completed dwell.
    For a 2-slot dwell the filter marginalises over the hidden states in both
    slots instead of pretending that one aggregate observation happened in
    only one base slot.
    """

    def __init__(
        self,
        transition: np.ndarray,
        pd: float = PD,
        pfa: float = PFA,
        prior_active: np.ndarray | None = None,
    ) -> None:

        transition = np.asarray(transition, dtype=np.float64)

        if transition.shape != (N_BANDS, 2, 2):
            raise ValueError(
                f"transition must have shape "
                f"({N_BANDS}, 2, 2), got {transition.shape}"
            )

        if np.any(transition < 0.0):
            raise ValueError("transition contains negative probabilities")

        if not np.allclose(
            transition.sum(axis=-1),
            1.0,
            atol=1e-8,
        ):
            raise ValueError("transition must be row-stochastic")

        self.transition = transition.copy()

        self.pd = float(pd)
        self.pfa = float(pfa)

        if not (0.0 < self.pfa < self.pd < 1.0):
            raise ValueError("Require 0 < PFA < Pd < 1")

        if prior_active is None:
            self.prior = np.full(
                N_BANDS,
                TRAIN250_PRIOR_ACTIVE,
                dtype=np.float64,
            )
        else:
            self.prior = np.asarray(
                prior_active,
                dtype=np.float64,
            ).copy()

        if self.prior.shape != (N_BANDS,):
            raise ValueError(
                f"prior_active must have shape ({N_BANDS},)"
            )

        self.prior = np.clip(
            self.prior,
            1e-6,
            1.0 - 1e-6,
        )

        self.reset()

    @classmethod
    def from_train250(
        cls,
        *,
        pd: float = PD,
        pfa: float = PFA,
        p01: float = TRAIN250_P01,
        p11: float = TRAIN250_P11,
        prior_active: float = TRAIN250_PRIOR_ACTIVE,
    ) -> "TwoStateHMMBelief":

        transition = np.zeros(
            (N_BANDS, 2, 2),
            dtype=np.float64,
        )

        transition[:, 0, 0] = 1.0 - p01
        transition[:, 0, 1] = p01

        transition[:, 1, 0] = 1.0 - p11
        transition[:, 1, 1] = p11

        prior = np.full(
            N_BANDS,
            prior_active,
            dtype=np.float64,
        )

        return cls(
            transition=transition,
            pd=pd,
            pfa=pfa,
            prior_active=prior,
        )

    def reset(self) -> None:
        self.belief = self.prior.copy()

    def copy_reset(self) -> "TwoStateHMMBelief":
        return TwoStateHMMBelief(
            transition=self.transition.copy(),
            pd=self.pd,
            pfa=self.pfa,
            prior_active=self.prior.copy(),
        )

    @staticmethod
    def _dist(active_probability: float) -> np.ndarray:
        p = float(
            np.clip(
                active_probability,
                1e-6,
                1.0 - 1e-6,
            )
        )

        return np.asarray(
            [1.0 - p, p],
            dtype=np.float64,
        )

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.clip(
            np.asarray(x, dtype=np.float64),
            0.0,
            None,
        )

        total = float(x.sum())

        if total <= EPS:
            return np.asarray(
                [0.5, 0.5],
                dtype=np.float64,
            )

        return x / total

    def predict_band(
        self,
        band: int,
        slots: int,
    ) -> float:

        band = int(band)
        slots = int(slots)

        if slots < 0:
            raise ValueError("slots must be >= 0")

        dist = self._dist(
            self.belief[band]
        )

        if slots > 0:
            dist = (
                dist
                @ np.linalg.matrix_power(
                    self.transition[band],
                    slots,
                )
            )

        return float(
            np.clip(
                dist[1],
                1e-6,
                1.0 - 1e-6,
            )
        )

    def predict_all(
        self,
        slots: int,
    ) -> np.ndarray:

        return np.asarray(
            [
                self.predict_band(
                    band,
                    int(slots),
                )
                for band in range(N_BANDS)
            ],
            dtype=np.float64,
        )

    def aggregate_hit_probability(
        self,
        band: int,
        dwell_slots: int,
    ) -> float:
        """
        P(at least one receiver HIT during this dwell
          | current causal belief)
        """

        band = int(band)
        dwell_slots = int(dwell_slots)

        if dwell_slots not in (1, 2):
            raise ValueError(
                "Current Vyapti native dwell must be 1 or 2 slots"
            )

        prior = self._dist(
            self.belief[band]
        )

        T = self.transition[band]

        # Distribution after d latent transitions.
        end_prior = (
            prior
            @ np.linalg.matrix_power(
                T,
                dwell_slots,
            )
        )

        # Probability mass associated with the complete path producing
        # NO detector-positive observation.
        no_hit_emission = np.diag(
            np.asarray(
                [
                    1.0 - self.pfa,
                    1.0 - self.pd,
                ],
                dtype=np.float64,
            )
        )

        joint_no_hit = prior.copy()

        for _ in range(dwell_slots):
            joint_no_hit = (
                joint_no_hit
                @ no_hit_emission
                @ T
            )

        p_no_hit = float(
            np.clip(
                joint_no_hit.sum(),
                0.0,
                1.0,
            )
        )

        return float(
            np.clip(
                1.0 - p_no_hit,
                0.0,
                1.0,
            )
        )

    def update_aggregate(
        self,
        band: int,
        aggregate_hit: bool,
        dwell_slots: int,
    ) -> None:
        """
        Exact Bayesian filtering for one completed dwell.
        """

        band = int(band)
        dwell_slots = int(dwell_slots)

        if dwell_slots not in (1, 2):
            raise ValueError(
                "Current Vyapti native dwell must be 1 or 2 slots"
            )

        # Predict every band through the elapsed dwell.
        predicted = self.predict_all(dwell_slots)

        prior = self._dist(
            self.belief[band]
        )

        T = self.transition[band]

        end_prior = (
            prior
            @ np.linalg.matrix_power(
                T,
                dwell_slots,
            )
        )

        no_hit_emission = np.diag(
            np.asarray(
                [
                    1.0 - self.pfa,
                    1.0 - self.pd,
                ],
                dtype=np.float64,
            )
        )

        joint_no_hit = prior.copy()

        for _ in range(dwell_slots):
            joint_no_hit = (
                joint_no_hit
                @ no_hit_emission
                @ T
            )

        p_no_hit = float(
            np.clip(
                joint_no_hit.sum(),
                0.0,
                1.0,
            )
        )

        if aggregate_hit:

            p_hit = max(
                1.0 - p_no_hit,
                EPS,
            )

            # P(end state | at least one HIT)
            posterior = (
                end_prior
                - joint_no_hit
            ) / p_hit

        else:

            # P(end state | no HIT)
            posterior = (
                joint_no_hit
                / max(
                    p_no_hit,
                    EPS,
                )
            )

        posterior = self._normalize(
            posterior
        )

        self.belief = predicted

        self.belief[band] = float(
            np.clip(
                posterior[1],
                1e-6,
                1.0 - 1e-6,
            )
        )


# =============================================================================
# CAUSAL PERIODICITY
# =============================================================================

class CausalPeriodicity:
    """
    Dwell-level causal periodicity estimator.

    Only receiver-observed positive events are used.

    No hidden transmission timestamps are available to the policy.

    Therefore a positive 2-slot dwell is represented at its dwell-end
    timestamp, exactly as in the current SAC implementation.
    """

    def __init__(
        self,
        max_events: int = DEFAULT_PERIODICITY_HISTORY,
        tolerance_fraction: float = DEFAULT_PERIODICITY_TOLERANCE,
    ) -> None:

        self.max_events = int(
            max_events
        )

        self.tolerance_fraction = float(
            tolerance_fraction
        )

        self.events: list[list[float]] = [
            []
            for _ in range(N_BANDS)
        ]

    def reset(self) -> None:
        self.events = [
            []
            for _ in range(N_BANDS)
        ]

    def observe(
        self,
        band: int,
        event_time_slot: float,
        aggregate_hit: bool,
    ) -> None:

        if not aggregate_hit:
            return

        band = int(band)
        t = float(event_time_slot)

        history = self.events[band]

        if history and t <= history[-1]:
            return

        history.append(t)

        if len(history) > self.max_events:

            del history[
                : len(history) - self.max_events
            ]

    def estimate(
        self,
        band: int,
        now_slot: int,
    ) -> tuple[float, float]:

        history = self.events[int(band)]

        # Need at least 3 events:
        # 2 gaps are the minimum for measuring regularity.
        if len(history) < 3:
            return 0.0, 0.0

        gaps = np.diff(
            np.asarray(
                history,
                dtype=np.float64,
            )
        )

        gaps = gaps[
            gaps >= 1.0
        ]

        if len(gaps) < 2:
            return 0.0, 0.0

        period = float(
            np.median(gaps)
        )

        if period <= 1.0:
            return 0.0, 0.0

        mean_gap = float(
            np.mean(gaps)
        )

        if mean_gap <= 0.0:
            return 0.0, 0.0

        cv = float(
            np.std(gaps)
            / mean_gap
        )

        regularity = float(
            np.exp(-cv)
        )

        elapsed = max(
            0.0,
            float(now_slot)
            - float(history[-1]),
        )

        remainder = (
            elapsed
            % period
        )

        phase_error = min(
            remainder,
            period - remainder,
        )

        tolerance = max(
            self.tolerance_fraction * period,
            0.5,
        )

        phase_score = float(
            np.exp(
                -phase_error / tolerance
            )
        )

        confidence = float(
            np.clip(
                regularity
                * min(
                    1.0,
                    len(gaps) / 6.0,
                ),
                0.0,
                1.0,
            )
        )

        score = float(
            np.clip(
                phase_score
                * confidence,
                0.0,
                1.0,
            )
        )

        return score, confidence

    def features(
        self,
        now_slot: int,
    ) -> tuple[np.ndarray, np.ndarray]:

        values = [
            self.estimate(
                band,
                now_slot,
            )
            for band in range(N_BANDS)
        ]

        scores = np.asarray(
            [
                value[0]
                for value in values
            ],
            dtype=np.float32,
        )

        confidence = np.asarray(
            [
                value[1]
                for value in values
            ],
            dtype=np.float32,
        )

        return scores, confidence


# =============================================================================
# CAUSAL EPISODE STATE
# =============================================================================

@dataclass
class CausalState:

    belief: TwoStateHMMBelief
    periodicity: CausalPeriodicity
    native_dwell_slots: np.ndarray

    last_visit_slot: np.ndarray
    elapsed_slots: int = 0
    previous_action: int = -1

    @classmethod
    def create(
        cls,
        belief_template: TwoStateHMMBelief,
        native_dwell_slots: np.ndarray,
        periodicity_history: int,
        periodicity_tolerance: float,
    ) -> "CausalState":

        return cls(
            belief=belief_template.copy_reset(),
            periodicity=CausalPeriodicity(
                max_events=periodicity_history,
                tolerance_fraction=periodicity_tolerance,
            ),
            native_dwell_slots=native_dwell_slots.copy(),
            last_visit_slot=np.full(
                N_BANDS,
                -1,
                dtype=np.int32,
            ),
        )

    def reset(self) -> None:

        self.belief.reset()
        self.periodicity.reset()

        self.last_visit_slot.fill(-1)

        self.elapsed_slots = 0
        self.previous_action = -1

    def observe_contexts(self) -> np.ndarray:
        """
        Construct one candidate feature vector per band.

        Shape:
            [36, 46]

        All quantities are causal.
        """

        now = int(
            self.elapsed_slots
        )

        period_score, period_conf = (
            self.periodicity.features(now)
        )

        remaining = float(
            np.clip(
                1.0
                - now / max(
                    BASE_SLOTS,
                    1,
                ),
                0.0,
                1.0,
            )
        )

        contexts = np.zeros(
            (N_BANDS, CONTEXT_DIM),
            dtype=np.float64,
        )

        for band in range(N_BANDS):

            dwell = min(
                int(self.native_dwell_slots[band]),
                BASE_SLOTS - now,
            )

            if dwell not in (1, 2):
                raise ValueError(
                    f"Invalid native dwell for band "
                    f"{band}: {dwell}"
                )

            # --------------------------------------------------------------
            # Candidate-specific causal belief
            # --------------------------------------------------------------

            current_belief = float(
                self.belief.belief[band]
            )

            predicted_belief = float(
                self.belief.predict_band(
                    band,
                    dwell,
                )
            )

            aggregate_hit_prob = float(
                self.belief.aggregate_hit_probability(
                    band,
                    dwell,
                )
            )

            # --------------------------------------------------------------
            # Causal staleness
            # --------------------------------------------------------------

            if self.last_visit_slot[band] < 0:

                staleness = 1.0

            else:

                staleness = float(
                    np.clip(
                        (
                            now
                            - self.last_visit_slot[band]
                        )
                        / max(
                            BASE_SLOTS,
                            1,
                        ),
                        0.0,
                        1.0,
                    )
                )

            # --------------------------------------------------------------
            # Switching
            # --------------------------------------------------------------

            switch_indicator = float(
                self.previous_action >= 0
                and self.previous_action != band
            )

            # --------------------------------------------------------------
            # 46-D context
            # --------------------------------------------------------------

            x = np.zeros(
                CONTEXT_DIM,
                dtype=np.float64,
            )

            cursor = 0

            # 1. Intercept
            x[cursor] = 1.0
            cursor += 1

            # 2. Band one-hot, 36 dimensions
            x[
                cursor + band
            ] = 1.0
            cursor += N_BANDS

            # 3. Current belief
            x[cursor] = current_belief
            cursor += 1

            # 4. Predicted belief after candidate dwell
            x[cursor] = predicted_belief
            cursor += 1

            # 5. Probability of at least one detector HIT during dwell
            x[cursor] = aggregate_hit_prob
            cursor += 1

            # 6. Periodicity score
            x[cursor] = float(
                period_score[band]
            )
            cursor += 1

            # 7. Periodicity confidence
            x[cursor] = float(
                period_conf[band]
            )
            cursor += 1

            # 8. Staleness
            x[cursor] = staleness
            cursor += 1

            # 9. Remaining mission fraction
            x[cursor] = remaining
            cursor += 1

            # 10. Candidate dwell duration
            #
            # 1 slot = 0.5
            # 2 slots = 1.0
            x[cursor] = float(
                dwell / 2.0
            )
            cursor += 1

            # 11. Candidate switch
            x[cursor] = switch_indicator
            cursor += 1

            if cursor != CONTEXT_DIM:
                raise RuntimeError(
                    f"Context construction error: "
                    f"cursor={cursor}, "
                    f"expected={CONTEXT_DIM}"
                )

            contexts[band] = x

        return contexts

    def commit_dwell(
        self,
        action: int,
        aggregate_hit: bool,
        dwell_slots: int,
    ) -> None:

        action = int(action)
        dwell_slots = int(dwell_slots)

        if dwell_slots not in (1, 2):
            raise ValueError(
                "dwell_slots must be 1 or 2"
            )

        expected_dwell = min(
            int(self.native_dwell_slots[action]),
            BASE_SLOTS - int(self.elapsed_slots),
        )

        if expected_dwell != dwell_slots:
            raise RuntimeError(
                f"Native dwell mismatch for band {action}: "
                f"configured={expected_dwell}, "
                f"transition={dwell_slots}"
            )

        start_slot = int(
            self.elapsed_slots
        )

        end_slot = (
            start_slot
            + dwell_slots
            - 1
        )

        # Bayesian update from observed receiver HIT/MISS.
        self.belief.update_aggregate(
            band=action,
            aggregate_hit=bool(aggregate_hit),
            dwell_slots=dwell_slots,
        )

        # Periodicity update only from observed HIT.
        #
        # Exact in-dwell positive time is not visible to the scheduler.
        self.periodicity.observe(
            band=action,
            event_time_slot=end_slot,
            aggregate_hit=bool(aggregate_hit),
        )

        self.last_visit_slot[action] = start_slot
        self.previous_action = action

        self.elapsed_slots += dwell_slots


# =============================================================================
# LINEAR-GAUSSIAN THOMPSON SAMPLER
# =============================================================================

class LinearGaussianThompsonSampler:
    """
    Bayesian linear contextual Thompson Sampling.

    Model:

        r = x^T theta + epsilon

        epsilon ~ N(0, sigma^2)

    Prior:

        theta ~ N(0, lambda^-1 I)

    Posterior:

        A = lambda I + sum(x x^T / sigma^2)

        b = sum(x r / sigma^2)

        theta_hat = A^-1 b

    To avoid repeatedly computing A^-1:

        - maintain Cholesky L with A = L L^T
        - use triangular solves
        - perform rank-one Cholesky updates
    """

    def __init__(
        self,
        dimension: int,
        prior_precision: float,
        sampling_scale: float,
        reward_noise_std: float,
    ) -> None:

        self.dimension = int(
            dimension
        )

        self.prior_precision = float(
            prior_precision
        )

        self.sampling_scale = float(
            sampling_scale
        )

        self.reward_noise_std = float(
            reward_noise_std
        )

        if self.dimension <= 0:
            raise ValueError(
                "dimension must be positive"
            )

        if self.prior_precision <= 0:
            raise ValueError(
                "prior_precision must be > 0"
            )

        if self.sampling_scale <= 0:
            raise ValueError(
                "sampling_scale must be > 0"
            )

        if self.reward_noise_std <= 0:
            raise ValueError(
                "reward_noise_std must be > 0"
            )

        self.noise_variance = (
            self.reward_noise_std ** 2
        )

        self.A = (
            self.prior_precision
            * np.eye(
                self.dimension,
                dtype=np.float64,
            )
        )

        self.b = np.zeros(
            self.dimension,
            dtype=np.float64,
        )

        self.L = np.linalg.cholesky(
            self.A
        )

    def posterior_mean(self) -> np.ndarray:
        """
        Solve A theta = b using Cholesky factors.
        """

        y = np.linalg.solve(
            self.L,
            self.b,
        )

        theta = np.linalg.solve(
            self.L.T,
            y,
        )

        return theta

    def sample_theta(
        self,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample:

            theta ~ N(theta_hat, v^2 A^-1)

        using A = L L^T.
        """

        mean = self.posterior_mean()

        z = rng.normal(
            0.0,
            1.0,
            size=self.dimension,
        )

        # If A = L L^T, then
        #
        #   L^-T z
        #
        # has covariance A^-1.
        perturbation = np.linalg.solve(
            self.L.T,
            z,
        )

        return (
            mean
            + self.sampling_scale
            * perturbation
        )

    def score(
        self,
        contexts: np.ndarray,
        rng: np.random.Generator,
        sample: bool = True,
    ) -> np.ndarray:

        contexts = np.asarray(
            contexts,
            dtype=np.float64,
        )

        if contexts.ndim != 2:
            raise ValueError(
                "contexts must be 2-D"
            )

        if contexts.shape[1] != self.dimension:
            raise ValueError(
                f"context dimension mismatch: "
                f"got {contexts.shape[1]}, "
                f"expected {self.dimension}"
            )

        if sample:

            theta = self.sample_theta(
                rng
            )

        else:

            theta = self.posterior_mean()

        return contexts @ theta

    @staticmethod
    def _chol_rank_one_update(
        L: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Lower-Cholesky rank-one update.

        Given:

            A = L L^T

        returns L_new such that:

            A + x x^T = L_new L_new^T
        """

        L = np.asarray(
            L,
            dtype=np.float64,
        ).copy()

        x = np.asarray(
            x,
            dtype=np.float64,
        ).copy()

        n = L.shape[0]

        if L.shape != (n, n):
            raise ValueError(
                "L must be square"
            )

        if x.shape != (n,):
            raise ValueError(
                f"x must have shape ({n},)"
            )

        for k in range(n):

            diag = float(
                L[k, k]
            )

            r = math.hypot(
                diag,
                float(x[k]),
            )

            c = r / max(
                diag,
                EPS,
            )

            s = x[k] / max(
                diag,
                EPS,
            )

            L[k, k] = r

            if k + 1 < n:

                old_col = L[
                    k + 1 :,
                    k
                ].copy()

                old_x = x[
                    k + 1 :
                ].copy()

                L[
                    k + 1 :,
                    k
                ] = (
                    old_col
                    + s * old_x
                ) / c

                x[
                    k + 1 :
                ] = (
                    c * old_x
                    - s * L[
                        k + 1 :,
                        k
                    ]
                )

        return L

    def update(
        self,
        context: np.ndarray,
        reward: float,
    ) -> None:
        """
        Bayesian posterior update.

        IMPORTANT:
        `reward` must be the environment's exact transition.reward.

        We do not reconstruct or reshape the reward.
        """

        x = np.asarray(
            context,
            dtype=np.float64,
        )

        if x.shape != (self.dimension,):
            raise ValueError(
                f"context must have shape "
                f"({self.dimension},), got {x.shape}"
            )

        r = float(
            reward
        )

        if not math.isfinite(r):
            raise ValueError(
                f"Reward must be finite, got {r}"
            )

        precision_weight = (
            1.0
            / self.noise_variance
        )

        weighted_x = (
            precision_weight
            * x
        )

        # Gaussian posterior:
        #
        # A <- A + xx^T / sigma^2
        # b <- b + xr / sigma^2
        self.A += (
            precision_weight
            * np.outer(
                x,
                x,
            )
        )

        self.b += (
            weighted_x
            * r
        )

        # Maintain Cholesky A = L L^T.
        self.L = (
            self._chol_rank_one_update(
                self.L,
                math.sqrt(
                    precision_weight
                )
                * x,
            )
        )


# =============================================================================
# POLICY
# =============================================================================

class ContextualThompsonBeliefPeriodicityPolicy:
    """
    Vyapti shared-runner Contextual Thompson Sampling policy.
    """

    FORMAT = (
        "vyapti_contextual_thompson_"
        "belief_periodic_v1"
    )

    ALGORITHM_NAME = (
        "contextual_thompson_belief_periodic"
    )

    def __init__(
        self,
        bands: int,
        seed: int,
        settings: dict[str, Any],
        checkpoint=None,
    ) -> None:

        if int(bands) != N_BANDS:
            raise ValueError(
                f"This frozen policy expects "
                f"{N_BANDS} bands, got {bands}"
            )

        self.bands = int(
            bands
        )

        self.settings = dict(
            settings
        )

        self.seed = int(
            seed
        )

        # --------------------------------------------------------------
        # RNG
        # --------------------------------------------------------------

        self.rng = np.random.default_rng(
            self.seed
        )

        # --------------------------------------------------------------
        # Detector model
        # --------------------------------------------------------------

        self.pd = float(
            self.settings.get(
                "detection_probability",
                PD,
            )
        )

        self.pfa = float(
            self.settings.get(
                "false_alarm_probability",
                PFA,
            )
        )

        if not (
            0.0
            < self.pfa
            < self.pd
            < 1.0
        ):
            raise ValueError(
                "Require 0 < PFA < Pd < 1"
            )

        # --------------------------------------------------------------
        # TRAIN-250 HMM calibration
        # --------------------------------------------------------------
        #
        # Scalars OR length-36 arrays are accepted.
        #
        p01_values = _as_band_array(
            self.settings.get(
                "inactive_to_active_probability",
                TRAIN250_P01,
            ),
            self.bands,
            "inactive_to_active_probability",
        )

        p11_values = _as_band_array(
            self.settings.get(
                "active_to_active_probability",
                TRAIN250_P11,
            ),
            self.bands,
            "active_to_active_probability",
        )

        prior_values = _as_band_array(
            self.settings.get(
                "prior_active_probability",
                TRAIN250_PRIOR_ACTIVE,
            ),
            self.bands,
            "prior_active_probability",
        )

        _validate_probability_array(
            p01_values,
            "P01",
        )

        _validate_probability_array(
            p11_values,
            "P11",
        )

        _validate_probability_array(
            prior_values,
            "prior_active",
        )

        # --------------------------------------------------------------
        # Native dwell profile
        # --------------------------------------------------------------
        #
        # We deliberately do NOT invent which 7 bands are 100 ms.
        #
        # The exact 36-entry vector must come from the existing runner/config.
        #
        # Example:
        #
        # native_dwell_slots = [
        #     1, 1, 1, ..., 2, ...
        # ]
        #
        dwell_setting = self.settings.get(
            "native_dwell_slots",
            None,
        )

        if dwell_setting is None:
            raise ValueError(
                "Contextual Thompson Sampling requires the exact "
                "36-entry `native_dwell_slots` setting so candidate "
                "contexts can include the real 50/100 ms dwell. "
                "Do not guess which 7 bands use 2 slots."
            )

        self.native_dwell_slots = np.asarray(
            dwell_setting,
            dtype=np.int64,
        )

        if self.native_dwell_slots.shape != (
            self.bands,
        ):
            raise ValueError(
                f"native_dwell_slots must have "
                f"shape ({self.bands},), got "
                f"{self.native_dwell_slots.shape}"
            )

        if not np.all(
            np.isin(
                self.native_dwell_slots,
                [1, 2],
            )
        ):
            raise ValueError(
                "native_dwell_slots may only contain 1 or 2"
            )

        if int(
            np.sum(
                self.native_dwell_slots == 1
            )
        ) != 29:
            raise ValueError(
                "Current frozen Vyapti contract expects "
                "exactly 29 one-slot dwells."
            )

        if int(
            np.sum(
                self.native_dwell_slots == 2
            )
        ) != 7:
            raise ValueError(
                "Current frozen Vyapti contract expects "
                "exactly 7 two-slot dwells."
            )

        # --------------------------------------------------------------
        # HMM transition matrices
        # --------------------------------------------------------------

        transition = np.zeros(
            (
                self.bands,
                2,
                2,
            ),
            dtype=np.float64,
        )

        transition[
            :,
            0,
            0,
        ] = (
            1.0
            - p01_values
        )

        transition[
            :,
            0,
            1,
        ] = p01_values

        transition[
            :,
            1,
            0,
        ] = (
            1.0
            - p11_values
        )

        transition[
            :,
            1,
            1,
        ] = p11_values

        self.belief_template = TwoStateHMMBelief(
            transition=transition,
            pd=self.pd,
            pfa=self.pfa,
            prior_active=prior_values,
        )

        # --------------------------------------------------------------
        # Periodicity configuration
        # --------------------------------------------------------------

        self.periodicity_history = int(
            self.settings.get(
                "periodicity_history",
                DEFAULT_PERIODICITY_HISTORY,
            )
        )

        self.periodicity_tolerance = float(
            self.settings.get(
                "periodicity_tolerance_fraction",
                DEFAULT_PERIODICITY_TOLERANCE,
            )
        )

        if self.periodicity_history < 3:
            raise ValueError(
                "periodicity_history must be >= 3"
            )

        # --------------------------------------------------------------
        # Thompson posterior
        # --------------------------------------------------------------

        prior_precision = float(
            self.settings.get(
                "prior_precision",
                DEFAULT_PRIOR_PRECISION,
            )
        )

        sampling_scale = float(
            self.settings.get(
                "thompson_sampling_scale",
                DEFAULT_TS_SCALE,
            )
        )

        reward_noise_std = float(
            self.settings.get(
                "reward_noise_std",
                DEFAULT_REWARD_NOISE_STD,
            )
        )

        self.ts = LinearGaussianThompsonSampler(
            dimension=CONTEXT_DIM,
            prior_precision=prior_precision,
            sampling_scale=sampling_scale,
            reward_noise_std=reward_noise_std,
        )

        # --------------------------------------------------------------
        # Warmup
        # --------------------------------------------------------------

        self.warmup_actions = int(
            self.settings.get(
                "warmup_actions",
                DEFAULT_WARMUP_ACTIONS,
            )
        )

        if self.warmup_actions < self.bands:
            self.warmup_actions = self.bands

        # Number of training observations obtained for each band.
        #
        # This is maintained across episodes and is only used for the early
        # coverage warmup. It does NOT become part of the public context.
        self.training_band_visits = np.zeros(
            self.bands,
            dtype=np.int64,
        )

        # --------------------------------------------------------------
        # Evaluation behavior
        # --------------------------------------------------------------

        self.sample_in_evaluation = bool(
            self.settings.get(
                "sample_in_evaluation",
                True,
            )
        )

        # --------------------------------------------------------------
        # Counters
        # --------------------------------------------------------------

        self.total_actions = 0

        self.total_updates = 0

        self.episode_decisions = 0

        self.episode_reward = 0.0

        self.band_counts = np.zeros(
            self.bands,
            dtype=np.int64,
        )

        self._pending = None

        self._last_scores = np.zeros(
            self.bands,
            dtype=np.float64,
        )

        self._last_sampled_theta = (
            np.zeros(
                CONTEXT_DIM,
                dtype=np.float64,
            )
        )

        self.state = None

        self.reset_episode(
            training=True
        )

        if checkpoint is not None:
            self._load(
                checkpoint
            )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset_episode(
        self,
        *,
        training: bool,
    ) -> None:

        self.state = CausalState.create(
            belief_template=self.belief_template,
            native_dwell_slots=self.native_dwell_slots,
            periodicity_history=self.periodicity_history,
            periodicity_tolerance=self.periodicity_tolerance,
        )

        self._pending = None

        self.episode_decisions = 0

        self.episode_reward = 0.0

        self.band_counts = np.zeros(
            self.bands,
            dtype=np.int64,
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        public_state,
        *,
        training: bool,
    ) -> int:

        if self._pending is not None:
            raise RuntimeError(
                "observe must follow every selected action"
            )

        if self.state is None:
            raise RuntimeError(
                "Policy episode state has not been initialized"
            )

        # Environment/public clock must remain identical to policy clock.
        if int(
            public_state.time_slot
        ) != int(
            self.state.elapsed_slots
        ):
            raise RuntimeError(
                "Policy causal clock diverged "
                "from environment clock"
            )

        # --------------------------------------------------------------
        # Candidate contexts
        # --------------------------------------------------------------

        contexts = (
            self.state.observe_contexts()
        )

        # --------------------------------------------------------------
        # Early training warmup
        # --------------------------------------------------------------
        #
        # First 36 training actions:
        # guarantee each band is sampled at least once.
        #
        # Remaining warmup:
        # continue selecting currently less-visited bands.
        #
        # This prevents an unlucky initial Thompson draw from leaving large
        # parts of the action space statistically unseen.
        # --------------------------------------------------------------

        if (
            training
            and self.total_actions
            < self.warmup_actions
        ):

            if self.total_actions < self.bands:

                action = int(
                    self.total_actions
                )

            else:

                visits = (
                    self.training_band_visits
                )

                min_visits = int(
                    visits.min()
                )

                candidates = np.flatnonzero(
                    visits == min_visits
                )

                action = int(
                    self.rng.choice(
                        candidates
                    )
                )

            self._last_scores = np.full(
                self.bands,
                -np.inf,
                dtype=np.float64,
            )

            self._last_scores[action] = 0.0

        else:

            sample_theta = (
                training
                or self.sample_in_evaluation
            )

            if sample_theta:

                theta = (
                    self.ts.sample_theta(
                        self.rng
                    )
                )

                self._last_sampled_theta = (
                    theta.copy()
                )

            else:

                theta = (
                    self.ts.posterior_mean()
                )

                self._last_sampled_theta = (
                    theta.copy()
                )

            scores = (
                contexts @ theta
            )

            self._last_scores = (
                scores.copy()
            )

            # Deterministic argmax after the TS parameter sample.
            action = int(
                np.argmax(
                    scores
                )
            )

        self.band_counts[action] += 1

        self._pending = {
            "action": action,
            "contexts": contexts,
            "selected_context": contexts[action].copy(),
            "start_slot": int(
                public_state.time_slot
            ),
        }

        return action

    # ------------------------------------------------------------------
    # Transition consumption
    # ------------------------------------------------------------------

    @staticmethod
    def _elapsed_slots(
        transition: PublicTransition,
    ) -> int:

        elapsed_slots = (
            int(
                transition.next_state.time_slot
            )
            - int(
                transition.state.time_slot
            )
        )

        if elapsed_slots not in (1, 2):
            raise ValueError(
                f"Expected a one- or two-slot dwell, "
                f"got {elapsed_slots}"
            )

        return elapsed_slots

    def observe(
        self,
        transition: PublicTransition,
        *,
        training: bool,
    ) -> None:

        if self._pending is None:
            raise RuntimeError(
                "select_action must precede observe"
            )

        # --------------------------------------------------------------
        # Public receiver observation
        # --------------------------------------------------------------

        observation = (
            transition
            .next_state
            .previous_observation
        )

        if observation is None:
            raise ValueError(
                "Contextual Thompson Sampling requires "
                "a receiver observation after every action"
            )

        action = int(
            transition.action
        )

        pending_action = int(
            self._pending["action"]
        )

        if action != pending_action:
            raise RuntimeError(
                "Transition action does not match "
                "the policy's pending action"
            )

        if int(
            observation.selected_band
        ) != action:
            raise ValueError(
                "Action and receiver-selected band disagree"
            )

        elapsed_slots = (
            self._elapsed_slots(
                transition
            )
        )

        # Policy causal clock must equal transition start.
        if int(
            self.state.elapsed_slots
        ) != int(
            transition.state.time_slot
        ):
            raise RuntimeError(
                "Policy causal clock diverged "
                "from environment transition clock"
            )

        expected_dwell = min(
            int(self.native_dwell_slots[action]),
            BASE_SLOTS - int(self.state.elapsed_slots),
        )

        if elapsed_slots != expected_dwell:
            raise RuntimeError(
                f"Observed dwell mismatch on band {action}: "
                f"expected {expected_dwell}, "
                f"transition={elapsed_slots}"
            )

        # --------------------------------------------------------------
        # AUTHORITATIVE ENVIRONMENT REWARD
        # --------------------------------------------------------------
        #
        # Do NOT reconstruct the reward here.
        #
        reward = float(
            transition.reward
        )

        if not math.isfinite(
            reward
        ):
            raise RuntimeError(
                f"Non-finite environment reward: {reward}"
            )

        # --------------------------------------------------------------
        # Receiver-only belief update
        # --------------------------------------------------------------

        aggregate_hit = bool(
            observation.hit
        )

        self.state.commit_dwell(
            action=action,
            aggregate_hit=aggregate_hit,
            dwell_slots=elapsed_slots,
        )

        # --------------------------------------------------------------
        # Thompson posterior update
        # --------------------------------------------------------------

        if training:

            selected_context = np.asarray(
                self._pending["selected_context"],
                dtype=np.float64,
            )

            # The posterior uses the SAME environment reward.
            #
            # There is no:
            #     reward += ...
            #     reward *= ...
            #     belief bonus
            #     periodicity bonus
            #     staleness bonus
            #
            self.ts.update(
                context=selected_context,
                reward=reward,
            )

            self.total_actions += 1

            self.total_updates += 1

            self.training_band_visits[
                action
            ] += 1

        self.episode_decisions += 1

        self.episode_reward += reward

        self._pending = None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def end_episode(
        self,
        *,
        training: bool,
    ) -> dict[str, Any]:

        if self._pending is not None:
            raise RuntimeError(
                "Cannot end episode before "
                "observing the final action"
            )

        posterior_mean = (
            self.ts.posterior_mean()
        )

        return {
            "algorithm": self.ALGORITHM_NAME,

            "episode_decisions": int(
                self.episode_decisions
            ),

            "episode_reward": float(
                self.episode_reward
            ),

            "total_training_actions": int(
                self.total_actions
            ),

            "posterior_updates": int(
                self.total_updates
            ),

            "band_selection_counts": (
                self.band_counts.tolist()
            ),

            "training_band_visits": (
                self.training_band_visits.tolist()
            ),

            "posterior_mean_norm": float(
                np.linalg.norm(
                    posterior_mean
                )
            ),

            "posterior_diag_min": float(
                np.min(
                    np.diag(
                        np.linalg.inv(
                            self.ts.A
                        )
                    )
                )
            ),

            "posterior_diag_max": float(
                np.max(
                    np.diag(
                        np.linalg.inv(
                            self.ts.A
                        )
                    )
                )
            ),

            "selected_belief": float(
                self.state.belief.belief[
                    max(
                        0,
                        self.state.previous_action,
                    )
                ]
                if self.state.previous_action >= 0
                else 0.0
            ),

            "elapsed_slots": int(
                self.state.elapsed_slots
            ),
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(
        self,
        path,
    ) -> None:

        payload = {
            "format": self.FORMAT,

            "bands": int(
                self.bands
            ),

            "context_dim": int(
                CONTEXT_DIM
            ),

            "settings": dict(
                self.settings
            ),

            "seed": int(
                self.seed
            ),

            "total_actions": int(
                self.total_actions
            ),

            "total_updates": int(
                self.total_updates
            ),

            "training_band_visits": (
                self.training_band_visits.copy()
            ),

            "A": self.ts.A.copy(),

            "b": self.ts.b.copy(),

            "L": self.ts.L.copy(),

            "prior_precision": float(
                self.ts.prior_precision
            ),

            "sampling_scale": float(
                self.ts.sampling_scale
            ),

            "reward_noise_std": float(
                self.ts.reward_noise_std
            ),

            "rng_state": (
                self.rng.bit_generator.state
            ),
        }

        torch.save(
            payload,
            Path(path),
        )

    def _load(
        self,
        path,
    ) -> None:

        payload = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=False,
        )

        if payload.get(
            "format"
        ) != self.FORMAT:
            raise ValueError(
                "Checkpoint format does not match "
                "Contextual Thompson Sampling policy"
            )

        if int(
            payload.get(
                "bands",
                -1,
            )
        ) != self.bands:
            raise ValueError(
                "Checkpoint action-space size "
                "does not match current policy"
            )

        if int(
            payload.get(
                "context_dim",
                -1,
            )
        ) != CONTEXT_DIM:
            raise ValueError(
                "Checkpoint context dimension "
                "does not match current policy"
            )

        A = np.asarray(
            payload["A"],
            dtype=np.float64,
        )

        b = np.asarray(
            payload["b"],
            dtype=np.float64,
        )

        L = np.asarray(
            payload["L"],
            dtype=np.float64,
        )

        if A.shape != (
            CONTEXT_DIM,
            CONTEXT_DIM,
        ):
            raise ValueError(
                "Checkpoint A has wrong shape"
            )

        if b.shape != (
            CONTEXT_DIM,
        ):
            raise ValueError(
                "Checkpoint b has wrong shape"
            )

        if L.shape != (
            CONTEXT_DIM,
            CONTEXT_DIM,
        ):
            raise ValueError(
                "Checkpoint L has wrong shape"
            )

        self.ts.A = A.copy()

        self.ts.b = b.copy()

        self.ts.L = L.copy()

        self.ts.prior_precision = float(
            payload.get(
                "prior_precision",
                self.ts.prior_precision,
            )
        )

        self.ts.sampling_scale = float(
            payload.get(
                "sampling_scale",
                self.ts.sampling_scale,
            )
        )

        self.ts.reward_noise_std = float(
            payload.get(
                "reward_noise_std",
                self.ts.reward_noise_std,
            )
        )

        self.ts.noise_variance = (
            self.ts.reward_noise_std ** 2
        )

        self.total_actions = int(
            payload.get(
                "total_actions",
                0,
            )
        )

        self.total_updates = int(
            payload.get(
                "total_updates",
                self.total_actions,
            )
        )

        visits = np.asarray(
            payload.get(
                "training_band_visits",
                np.zeros(
                    self.bands,
                    dtype=np.int64,
                ),
            ),
            dtype=np.int64,
        )

        if visits.shape != (
            self.bands,
        ):
            raise ValueError(
                "Checkpoint training_band_visits has "
                "the wrong shape"
            )

        self.training_band_visits = visits.copy()

        if "rng_state" in payload:
            self.rng.bit_generator.state = (
                payload["rng_state"]
            )


# =============================================================================
# SHARED RUNNER FACTORY
# =============================================================================

def create(
    *,
    bands: int,
    seed: int,
    settings: dict,
    checkpoint=None,
):
    """
    Required shared-runner factory.
    """

    return ContextualThompsonBeliefPeriodicityPolicy(
        bands=bands,
        seed=seed,
        settings=settings,
        checkpoint=checkpoint,
    )

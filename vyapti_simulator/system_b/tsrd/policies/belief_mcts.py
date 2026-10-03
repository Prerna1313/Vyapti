#!/usr/bin/env python3
"""
Vyapti / PS26055
Truth-hidden Belief-State Monte-Carlo Tree Search

Architecture
------------
    causal Bayesian belief
          +
    causal periodicity
          +
    Bayesian reward model
          +
    short-horizon MCTS
          =
    truth-hidden belief-MCTS scheduler


WHY THIS IMPLEMENTATION
-----------------------
The real Vyapti scheduler cannot see:

    - hidden Mode-B truth
    - emitter IDs
    - future emitter activity
    - occupancy truth grid
    - future transmission schedule

Therefore the planner is NOT allowed to clone or query the hidden world
during hypothetical planning.

Instead:

    Real environment
        |
        | PublicTransition
        v
    observed reward + HIT/MISS
        |
        +--> Bayesian reward model
        |
        +--> causal HMM belief
        |
        +--> causal periodicity
        |
        v
    MCTS hypothetical branches

The hypothetical branch simulator only samples:

    HIT / MISS

from the current causal belief.

Hypothetical reward is predicted by the Bayesian reward model.

When the real action is executed, the exact environment reward:

    transition.reward

is used to update the reward model.

The policy therefore never reconstructs or modifies the authoritative
environment reward.


CURRENT VYAPTI BENCHMARK
------------------------
- Mode-B reconstructed hidden worlds
- TRAIN-250 source pool
- 36 bands
- 30 second mission
- 50 ms base slot
- 600 base slots
- native dwell is either 50 ms or 100 ms
- exactly 29 bands use 1 slot
- exactly 7 bands use 2 slots
- Pd = 0.90
- Pfa = 0.05

TRAIN-250 causal HMM calibration:

    P01            = 0.02220
    P11            = 0.95753
    prior active   = 0.34266

These are TRAIN-250 calibration values.
They are not claimed as universal TSRD constants.


PLANNING MODEL
--------------
Each MCTS state contains only causal/public information:

    belief[36]
    staleness[36]
    periodicity score[36]
    periodicity confidence[36]
    previous action
    current time

Each candidate action has a 55-D context:

    [intercept,
     band one-hot (36),
     current belief,
     predicted belief after dwell,
     aggregate HIT probability,
     periodicity score,
     periodicity confidence,
     staleness,
     remaining mission fraction,
     normalized dwell,
     switch indicator,
     five candidate observation-history features,
     four global observation-history features]


REWARD MODEL
------------
A Bayesian linear model predicts the expected one-step environment reward:

    E[r | x] = x^T theta

The posterior is updated ONLY from observed:

    transition.reward

No belief bonus, periodicity bonus, staleness bonus, dwell bonus,
or handcrafted reward is added.

This keeps the actual environment reward identical to PPO/SAC/UCB/CTS.


MCTS
----
At each real decision:

    1. Start from current public causal state.
    2. Run N simulations.
    3. Selection using UCT.
    4. Expand an action.
    5. Sample HIT/MISS from the causal belief.
    6. Update belief, periodicity, staleness and time.
    7. Roll out using the Bayesian reward model.
    8. Backup discounted value.
    9. Select the root action with the largest visit count.

The planner uses short-horizon lookahead rather than the full 30-second
mission. This keeps planning latency bounded and follows the spirit of
MCTS-based radar resource scheduling.


IMPORTANT LIMITATION
--------------------
This is NOT an oracle POMCP implementation.

A true POMCP simulator would require a generative model capable of producing
both future observations AND future rewards from the hidden state.

That would be inappropriate here if implemented by directly querying the
Mode-B world.

Therefore this implementation uses:

    causal HMM -> hypothetical observations
    Bayesian reward model -> hypothetical rewards

while the real environment remains the final evaluator.

This is intentionally a truth-hidden model-based planner.

"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..algorithm_interface import PublicTransition


# =============================================================================
# FROZEN VYAPTI SETUP
# =============================================================================

N_BANDS = 36

MISSION_SECONDS = 30.0
BASE_SLOT_SECONDS = 0.050
BASE_SLOTS = 600

PD = 0.90
PFA = 0.05

# TRAIN-250 pooled calibration
TRAIN250_P01 = 0.02220
TRAIN250_P11 = 0.95753
TRAIN250_PRIOR_ACTIVE = 0.34266

EPS = 1e-12


# =============================================================================
# CONTEXT
# =============================================================================

# 1 intercept
# 36 band identity
# 1 current belief
# 1 predicted belief after candidate dwell
# 1 aggregate HIT probability
# 1 periodicity score
# 1 periodicity confidence
# 1 staleness
# 1 remaining time
# 1 normalized dwell
# 1 switch indicator
#
# 5 candidate-specific causal history features + 4 global causal history
# features are appended to the original 46 dimensions.
CONTEXT_DIM = 55


# =============================================================================
# DEFAULT PLANNING CONFIGURATION
# =============================================================================

# Number of MCTS simulations per actual scheduling decision.
#
# Reference search budget. It is not a real-time latency guarantee; measure
# planning time on the intended hardware before choosing a training budget.
DEFAULT_SIMULATIONS = 128

# Number of future scheduling decisions considered in each simulation.
#
# With 50/100 ms native dwells this represents roughly 200-400 ms.
DEFAULT_HORIZON = 4

# UCT exploration coefficient.
#
# Starting engineering value. This is not a physical constant.
DEFAULT_UCT_C = 1.25

# Bayesian reward-model prior precision.
DEFAULT_REWARD_PRIOR_PRECISION = 1.0

# Bayesian reward-model observation noise.
#
# Does NOT modify transition.reward.
DEFAULT_REWARD_NOISE_STD = 0.01

# Periodicity
DEFAULT_PERIODICITY_HISTORY = 32
DEFAULT_PERIODICITY_TOLERANCE = 0.15

# Initial random/coverage phase.
DEFAULT_WARMUP_ACTIONS = 72

# Default rollout is greedy under the current posterior mean.
# Small epsilon can be used experimentally, but 0 is the clean baseline.
DEFAULT_ROLLOUT_EPSILON = 0.0


# =============================================================================
# UTILITY
# =============================================================================

def _as_band_array(
    value: Any,
    bands: int,
    name: str,
) -> np.ndarray:

    arr = np.asarray(
        value,
        dtype=np.float64,
    )

    if arr.ndim == 0:

        return np.full(
            bands,
            float(arr),
            dtype=np.float64,
        )

    if arr.shape != (bands,):

        raise ValueError(
            f"{name} must be scalar or shape "
            f"({bands},), got {arr.shape}"
        )

    return arr.copy()


def _validate_probabilities(
    values: np.ndarray,
    name: str,
) -> None:

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            f"{name} contains non-finite values"
        )

    if np.any(values <= 0.0) or np.any(values >= 1.0):

        raise ValueError(
            f"{name} must lie strictly inside (0, 1)"
        )


# =============================================================================
# CAUSAL TWO-STATE HMM
# =============================================================================

class TwoStateHMMBelief:
    """
    Factorised two-state HMM.

        state 0 = inactive
        state 1 = active

    Observation:

        P(HIT | active)   = Pd
        P(HIT | inactive) = Pfa

    Aggregate dwell filtering is exact for one- and two-slot dwells.
    """

    def __init__(
        self,
        transition: np.ndarray,
        pd: float = PD,
        pfa: float = PFA,
        prior_active: np.ndarray | None = None,
    ) -> None:

        transition = np.asarray(
            transition,
            dtype=np.float64,
        )

        if transition.shape != (
            N_BANDS,
            2,
            2,
        ):
            raise ValueError(
                f"transition must have shape "
                f"({N_BANDS},2,2)"
            )

        if np.any(
            transition < 0.0
        ):
            raise ValueError(
                "transition contains negative probabilities"
            )

        if not np.allclose(
            transition.sum(axis=-1),
            1.0,
            atol=1e-8,
        ):
            raise ValueError(
                "transition must be row-stochastic"
            )

        self.transition = transition.copy()

        self.pd = float(pd)
        self.pfa = float(pfa)

        if not (
            0.0
            < self.pfa
            < self.pd
            < 1.0
        ):
            raise ValueError(
                "Require 0 < PFA < Pd < 1"
            )

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

        if self.prior.shape != (
            N_BANDS,
        ):
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
            (
                N_BANDS,
                2,
                2,
            ),
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

        self.belief = (
            self.prior.copy()
        )

    def copy_reset(
        self,
    ) -> "TwoStateHMMBelief":
        result = object.__new__(TwoStateHMMBelief)
        result.transition = self.transition.copy()
        result.pd = self.pd
        result.pfa = self.pfa
        result.prior = self.prior.copy()
        result.belief = self.prior.copy()
        return result

    def copy_current(self) -> "TwoStateHMMBelief":
        result = self.copy_reset()
        result.belief = self.belief.copy()
        return result

    @staticmethod
    def _dist(
        active_probability: float,
    ) -> np.ndarray:

        p = float(
            np.clip(
                active_probability,
                1e-6,
                1.0 - 1e-6,
            )
        )

        return np.asarray(
            [
                1.0 - p,
                p,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _normalize(
        x: np.ndarray,
    ) -> np.ndarray:

        x = np.clip(
            np.asarray(
                x,
                dtype=np.float64,
            ),
            0.0,
            None,
        )

        total = float(
            x.sum()
        )

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
        slots = int(slots)
        if slots < 0:
            raise ValueError("slots must be nonnegative")
        active = self.belief.copy()
        for _ in range(slots):
            inactive_to_active = self.transition[:, 0, 1]
            active_to_active = self.transition[:, 1, 1]
            active = (1.0 - active) * inactive_to_active + active * active_to_active
        return np.clip(active, 1e-6, 1.0 - 1e-6)

    def aggregate_hit_probability(
        self,
        band: int,
        dwell_slots: int,
    ) -> float:
        """
        P(at least one detector-positive outcome during the dwell | belief)
        """

        band = int(band)
        dwell_slots = int(
            dwell_slots
        )

        if dwell_slots not in (
            1,
            2,
        ):
            raise ValueError(
                "Native Vyapti dwell must be 1 or 2 slots"
            )

        prior = self._dist(
            self.belief[band]
        )

        T = self.transition[band]

        no_hit_emission = np.diag(
            np.asarray(
                [
                    1.0 - self.pfa,
                    1.0 - self.pd,
                ],
                dtype=np.float64,
            )
        )

        joint_no_hit = (
            prior.copy()
        )

        for _ in range(
            dwell_slots
        ):

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
        Exact Bayesian filter for a completed 1/2-slot receiver dwell.
        """

        band = int(band)
        dwell_slots = int(
            dwell_slots
        )

        if dwell_slots not in (
            1,
            2,
        ):
            raise ValueError(
                "dwell_slots must be 1 or 2"
            )

        predicted = (
            self.predict_all(
                dwell_slots
            )
        )

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

        joint_no_hit = (
            prior.copy()
        )

        for _ in range(
            dwell_slots
        ):

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

            posterior = (
                end_prior
                - joint_no_hit
            ) / p_hit

        else:

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

        self.events = [
            []
            for _ in range(
                N_BANDS
            )
        ]

    def reset(self) -> None:

        self.events = [
            []
            for _ in range(
                N_BANDS
            )
        ]

    def observe(
        self,
        band: int,
        event_time_slot: float,
        aggregate_hit: bool,
    ) -> None:

        if not aggregate_hit:
            return

        band = int(
            band
        )

        t = float(
            event_time_slot
        )

        history = self.events[
            band
        ]

        if history and (
            t <= history[-1]
        ):
            return

        history.append(t)

        if len(history) > (
            self.max_events
        ):

            del history[
                : len(history)
                - self.max_events
            ]

    def estimate(
        self,
        band: int,
        now_slot: int,
    ) -> tuple[float, float]:

        history = self.events[
            int(band)
        ]

        if len(history) < 3:
            return (
                0.0,
                0.0,
            )

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
            return (
                0.0,
                0.0,
            )

        period = float(
            np.median(gaps)
        )

        if period <= 1.0:
            return (
                0.0,
                0.0,
            )

        mean_gap = float(
            np.mean(gaps)
        )

        if mean_gap <= 0.0:
            return (
                0.0,
                0.0,
            )

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
            self.tolerance_fraction
            * period,
            0.5,
        )

        phase_score = float(
            np.exp(
                -phase_error
                / tolerance
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

        return (
            score,
            confidence,
        )

    def features(
        self,
        now_slot: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        values = [
            self.estimate(
                b,
                now_slot,
            )
            for b in range(
                N_BANDS
            )
        ]

        scores = np.asarray(
            [
                x[0]
                for x in values
            ],
            dtype=np.float64,
        )

        confidence = np.asarray(
            [
                x[1]
                for x in values
            ],
            dtype=np.float64,
        )

        return (
            scores,
            confidence,
        )


# =============================================================================
# PUBLIC / SIMULATION STATE
# =============================================================================

@dataclass
class BeliefState:

    belief: TwoStateHMMBelief
    periodicity: CausalPeriodicity
    native_dwell_slots: np.ndarray

    last_visit_slot: np.ndarray
    candidate_visit_count: np.ndarray = field(default_factory=lambda: np.zeros(N_BANDS, dtype=np.int64))
    candidate_hit_count: np.ndarray = field(default_factory=lambda: np.zeros(N_BANDS, dtype=np.int64))
    last_hit_slot: np.ndarray = field(default_factory=lambda: np.full(N_BANDS, -1, dtype=np.int32))
    total_observed_hits: int = 0
    cumulative_observed_reward: float = 0.0
    total_decisions: int = 0

    elapsed_slots: int = 0
    previous_action: int = -1

    def clone(self) -> "BeliefState":

        cloned_periodicity = (
            CausalPeriodicity(
                max_events=self.periodicity.max_events,
                tolerance_fraction=self.periodicity.tolerance_fraction,
            )
        )

        cloned_periodicity.events = [
            list(x)
            for x in self.periodicity.events
        ]

        return BeliefState(
            belief=self.belief.copy_current(),
            periodicity=cloned_periodicity,
            native_dwell_slots=(
                self.native_dwell_slots.copy()
            ),
            last_visit_slot=(
                self.last_visit_slot.copy()
            ),
            candidate_visit_count=self.candidate_visit_count.copy(),
            candidate_hit_count=self.candidate_hit_count.copy(),
            last_hit_slot=self.last_hit_slot.copy(),
            total_observed_hits=int(self.total_observed_hits),
            cumulative_observed_reward=float(self.cumulative_observed_reward),
            total_decisions=int(self.total_decisions),
            elapsed_slots=int(
                self.elapsed_slots
            ),
            previous_action=int(
                self.previous_action
            ),
        )

    def observe_contexts(self) -> np.ndarray:
        """
        Generate one 55-D candidate context per band from public history only.
        """

        contexts = np.zeros(
            (
                N_BANDS,
                CONTEXT_DIM,
            ),
            dtype=np.float64,
        )

        now = int(
            self.elapsed_slots
        )

        remaining = float(
            np.clip(
                1.0
                - now / BASE_SLOTS,
                0.0,
                1.0,
            )
        )

        bands = np.arange(N_BANDS)
        dwell = np.minimum(self.native_dwell_slots, BASE_SLOTS - now)
        current_belief = self.belief.belief
        p1 = self.belief.predict_all(1)
        p2 = self.belief.predict_all(2)
        predicted_belief = np.where(dwell == 1, p1, p2)

        prior = np.column_stack((1.0 - current_belief, current_belief))
        no_hit_likelihood = np.asarray([1.0 - self.belief.pfa, 1.0 - self.belief.pd])
        no_hit = prior * no_hit_likelihood
        no_hit = np.einsum("bi,bij->bj", no_hit, self.belief.transition)
        hit_probability_one = 1.0 - no_hit.sum(axis=1)
        no_hit_two = no_hit * no_hit_likelihood
        no_hit_two = np.einsum("bi,bij->bj", no_hit_two, self.belief.transition)
        hit_probability_two = 1.0 - no_hit_two.sum(axis=1)
        aggregate_hit_probability = np.where(dwell == 1, hit_probability_one, hit_probability_two)
        period_score, period_conf = self.periodicity.features(now)
        staleness = np.where(
            self.last_visit_slot < 0,
            1.0,
            np.clip((now - self.last_visit_slot) / BASE_SLOTS, 0.0, 1.0),
        )

        # Preserve the same 46 features and ordering as the original per-band
        # construction while evaluating the common HMM terms in vector form.
        contexts[:, 0] = 1.0
        contexts[bands, 1 + bands] = 1.0
        contexts[:, 37] = current_belief
        contexts[:, 38] = predicted_belief
        contexts[:, 39] = aggregate_hit_probability
        contexts[:, 40] = period_score
        contexts[:, 41] = period_conf
        contexts[:, 42] = staleness
        contexts[:, 43] = remaining
        contexts[:, 44] = dwell / 2.0
        contexts[:, 45] = float(self.previous_action >= 0) * (self.previous_action != bands)

        visits = self.candidate_visit_count.astype(np.float64)
        hits = self.candidate_hit_count.astype(np.float64)
        contexts[:, 46] = visits / max(self.total_decisions, 1)
        contexts[:, 47] = hits / np.maximum(visits, 1.0)
        contexts[:, 48] = hits / max(self.total_decisions, 1)
        since_hit = np.where(self.last_hit_slot < 0, BASE_SLOTS, now - self.last_hit_slot)
        since_visit = np.where(self.last_visit_slot < 0, BASE_SLOTS, now - self.last_visit_slot)
        contexts[:, 49] = np.clip(since_hit / BASE_SLOTS, 0.0, 1.0)
        contexts[:, 50] = np.clip(since_visit / BASE_SLOTS, 0.0, 1.0)
        contexts[:, 51] = self.total_decisions / BASE_SLOTS
        contexts[:, 52] = now / BASE_SLOTS
        contexts[:, 53] = self.cumulative_observed_reward / max(self.total_decisions, 1)
        contexts[:, 54] = self.total_observed_hits / max(self.total_decisions, 1)

        return contexts

    def commit_dwell(
        self,
        action: int,
        aggregate_hit: bool,
        dwell_slots: int,
        observed_reward: float = 0.0,
    ) -> None:

        action = int(
            action
        )

        dwell_slots = int(
            dwell_slots
        )

        expected = min(
            int(self.native_dwell_slots[action]),
            BASE_SLOTS - int(self.elapsed_slots),
        )

        if dwell_slots != expected:

            raise RuntimeError(
                f"Dwell mismatch for band {action}: "
                f"expected {expected}, "
                f"received {dwell_slots}"
            )

        start_slot = int(
            self.elapsed_slots
        )

        end_slot = (
            start_slot
            + dwell_slots
            - 1
        )

        self.belief.update_aggregate(
            band=action,
            aggregate_hit=aggregate_hit,
            dwell_slots=dwell_slots,
        )

        self.periodicity.observe(
            band=action,
            event_time_slot=end_slot,
            aggregate_hit=aggregate_hit,
        )

        self.last_visit_slot[
            action
        ] = start_slot

        self.candidate_visit_count[action] += 1
        if aggregate_hit:
            self.candidate_hit_count[action] += 1
            self.last_hit_slot[action] = end_slot
            self.total_observed_hits += 1
        self.total_decisions += 1
        self.cumulative_observed_reward += float(observed_reward)

        self.previous_action = action

        self.elapsed_slots += (
            dwell_slots
        )


# =============================================================================
# BAYESIAN LINEAR REWARD MODEL
# =============================================================================

class BayesianRewardModel:
    """
    Bayesian linear model for expected one-step environment reward.

        r = x^T theta + epsilon

        epsilon ~ N(0, sigma^2)

    This model never changes the true environment reward.

    Its purpose is solely to provide hypothetical rewards for MCTS branches.
    """

    def __init__(
        self,
        dimension: int,
        prior_precision: float = DEFAULT_REWARD_PRIOR_PRECISION,
        reward_noise_std: float = DEFAULT_REWARD_NOISE_STD,
    ) -> None:

        self.dimension = int(
            dimension
        )

        self.prior_precision = float(
            prior_precision
        )

        self.reward_noise_std = float(
            reward_noise_std
        )

        if self.prior_precision <= 0.0:
            raise ValueError(
                "prior_precision must be > 0"
            )

        if self.reward_noise_std <= 0.0:
            raise ValueError(
                "reward_noise_std must be > 0"
            )

        self.noise_variance = (
            self.reward_noise_std
            ** 2
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

    def posterior_mean(
        self,
    ) -> np.ndarray:

        y = np.linalg.solve(
            self.L,
            self.b,
        )

        return np.linalg.solve(
            self.L.T,
            y,
        )

    def predict(
        self,
        contexts: np.ndarray,
    ) -> np.ndarray:

        contexts = np.asarray(
            contexts,
            dtype=np.float64,
        )

        theta = (
            self.posterior_mean()
        )

        return contexts @ theta

    @staticmethod
    def _chol_rank_one_update(
        L: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:

        L = L.copy()
        x = x.copy()

        n = L.shape[0]

        for k in range(n):

            diag = float(
                L[k, k]
            )

            r = math.hypot(
                diag,
                float(x[k]),
            )

            c = (
                r
                / max(
                    diag,
                    EPS,
                )
            )

            s = (
                x[k]
                / max(
                    diag,
                    EPS,
                )
            )

            L[k, k] = r

            if k + 1 < n:

                old_col = L[
                    k + 1 :,
                    k,
                ].copy()

                old_x = x[
                    k + 1 :
                ].copy()

                L[
                    k + 1 :,
                    k,
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
                        k,
                    ]
                )

        return L

    def update(
        self,
        context: np.ndarray,
        reward: float,
    ) -> None:

        x = np.asarray(
            context,
            dtype=np.float64,
        )

        reward = float(
            reward
        )

        if x.shape != (
            self.dimension,
        ):
            raise ValueError(
                "Wrong reward-model context dimension"
            )

        if not math.isfinite(
            reward
        ):
            raise ValueError(
                "Environment reward is non-finite"
            )

        precision = (
            1.0
            / self.noise_variance
        )

        scaled_x = (
            math.sqrt(
                precision
            )
            * x
        )

        self.A += (
            precision
            * np.outer(
                x,
                x,
            )
        )

        self.b += (
            precision
            * x
            * reward
        )

        self.L = (
            self._chol_rank_one_update(
                self.L,
                scaled_x,
            )
        )


# =============================================================================
# MCTS TREE
# =============================================================================

@dataclass
class ActionEdge:

    action: int

    visits: int = 0
    value_sum: float = 0.0

    hit_child: "MCTSNode | None" = None
    miss_child: "MCTSNode | None" = None

    @property
    def value(self) -> float:

        if self.visits <= 0:
            return 0.0

        return (
            self.value_sum
            / self.visits
        )


@dataclass
class MCTSNode:

    state: BeliefState

    depth: int

    visits: int = 0

    children: dict[int, ActionEdge] = field(
        default_factory=dict
    )

    _context_cache: np.ndarray | None = field(
        default=None,
        repr=False,
    )

    def contexts(self) -> np.ndarray:
        if self._context_cache is None:
            self._context_cache = self.state.observe_contexts()
        return self._context_cache


# =============================================================================
# BELIEF MCTS
# =============================================================================

class BeliefMCTS:

    def __init__(
        self,
        reward_model: BayesianRewardModel,
        rng: np.random.Generator,
        simulations: int = DEFAULT_SIMULATIONS,
        horizon: int = DEFAULT_HORIZON,
        exploration_constant: float = DEFAULT_UCT_C,
        rollout_epsilon: float = DEFAULT_ROLLOUT_EPSILON,
    ) -> None:

        self.reward_model = (
            reward_model
        )

        self.rng = rng

        self.simulations = int(
            simulations
        )

        self.horizon = int(
            horizon
        )

        self.exploration_constant = float(
            exploration_constant
        )

        self.rollout_epsilon = float(
            rollout_epsilon
        )

        # The reward model stays fixed during planning; cache its posterior
        # to avoid solving the same linear system for every simulated action.
        self.reward_weights = self.reward_model.posterior_mean()

        if self.simulations < 1:
            raise ValueError(
                "simulations must be >= 1"
            )

        if self.horizon < 1:
            raise ValueError(
                "horizon must be >= 1"
            )

        if self.exploration_constant <= 0.0:
            raise ValueError(
                "exploration_constant must be > 0"
            )

        if not 0.0 <= self.rollout_epsilon <= 1.0:
            raise ValueError("rollout_epsilon must be in [0, 1]")

    # ------------------------------------------------------------------
    # Public planning entry point
    # ------------------------------------------------------------------

    def plan(
        self,
        root_state: BeliefState,
    ) -> tuple[int, dict[str, Any]]:

        root = MCTSNode(
            state=root_state.clone(),
            depth=0,
        )

        immediate_scores = self._predict(root.contexts())

        for _ in range(
            self.simulations
        ):

            self._simulate(
                node=root,
            )

        if not root.children:

            fallback = int(
                np.argmax(
                    immediate_scores
                )
            )

            return (
                fallback,
                {
                    "simulations": 0,
                    "root_visits": 0,
                },
            )

        # Standard robust MCTS decision:
        # choose root child with maximum visit count.
        best_action = max(
            root.children,
            key=lambda a: (
                root.children[a].visits,
                root.children[a].value,
            ),
        )

        root_values = {
            int(action): {
                "visits": int(
                    edge.visits
                ),
                "mean_value": float(
                    edge.value
                ),
            }
            for action, edge
            in root.children.items()
        }

        diagnostics = {
            "simulations": int(
                self.simulations
            ),
            "root_visits": int(
                root.visits
            ),
            "root_children": len(
                root.children
            ),
            "selected_action": int(
                best_action
            ),
            "root_action_stats": (
                root_values
            ),
        }

        return (
            int(best_action),
            diagnostics,
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate(
        self,
        node: MCTSNode,
    ) -> float:

        if (
            node.depth
            >= self.horizon
        ):

            return 0.0

        if (
            node.state.elapsed_slots
            >= BASE_SLOTS
        ):

            return 0.0

        # --------------------------------------------------------------
        # Action selection / expansion
        # --------------------------------------------------------------

        action = self._select_action(node)

        if action not in (
            node.children
        ):

            node.children[action] = (
                ActionEdge(
                    action=action
                )
            )

        edge = node.children[
            action
        ]

        # --------------------------------------------------------------
        # Hypothetical receiver observation
        #
        # This is the critical truth-hidden boundary.
        #
        # We sample HIT/MISS using causal belief only.
        # --------------------------------------------------------------

        dwell = min(
            int(node.state.native_dwell_slots[action]),
            BASE_SLOTS - int(node.state.elapsed_slots),
        )

        p_hit = (
            node.state
            .belief
            .aggregate_hit_probability(
                action,
                dwell,
            )
        )


        # --------------------------------------------------------------
        # Hypothetical immediate reward.
        #
        # IMPORTANT:
        # This is NOT transition.reward.
        #
        # It is a prediction from the Bayesian reward model because
        # transition.reward is unavailable for a hypothetical future action.
        # --------------------------------------------------------------

        contexts = node.contexts()

        predicted_reward = float(self._predict(contexts[action : action + 1])[0])

        # Enumerate both possible public observations and back up their
        # probability-weighted value. One branch is deepened per simulation;
        # the alternative branch gets a one-step causal rollout. This preserves
        # both outcome values without multiplying search cost at every depth.
        p_miss = 1.0 - p_hit
        sampled_hit = bool(self.rng.random() < p_hit)
        branch_values = {}
        for aggregate_hit, probability in ((True, p_hit), (False, p_miss)):
            child = edge.hit_child if aggregate_hit else edge.miss_child
            if child is None:
                child_state = node.state.clone()
                child_state.commit_dwell(
                    action=action,
                    aggregate_hit=aggregate_hit,
                    dwell_slots=dwell,
                    observed_reward=predicted_reward,
                )
                child = MCTSNode(state=child_state, depth=node.depth + 1)
                if aggregate_hit:
                    edge.hit_child = child
                else:
                    edge.miss_child = child
            if probability <= 0.0:
                branch_values[aggregate_hit] = 0.0
            elif aggregate_hit == sampled_hit:
                branch_values[aggregate_hit] = self._simulate(child)
            else:
                branch_values[aggregate_hit] = self._rollout(
                    child.state.clone(),
                    remaining_depth=min(1, self.horizon - child.depth),
                )
        future_value = p_hit * branch_values[True] + p_miss * branch_values[False]

        discount = (
            self._discount(
                dwell
            )
        )

        total = (
            predicted_reward
            + discount
            * future_value
        )

        # --------------------------------------------------------------
        # Backup
        # --------------------------------------------------------------

        edge.visits += 1
        edge.value_sum += total

        node.visits += 1

        return total

    # ------------------------------------------------------------------
    # UCT action selection
    # ------------------------------------------------------------------

    def _select_action(
        self,
        node: MCTSNode,
    ) -> int:

        # Expansion priority is local to this node's causal state.
        expansion_order = list(np.argsort(-self._predict(node.contexts())))

        # Expand previously unvisited actions first.
        for action in expansion_order:

            action = int(
                action
            )

            if action not in (
                node.children
            ):

                return action

        # All actions have been expanded:
        # use UCT.

        parent_visits = max(
            node.visits,
            1,
        )

        best_action = None
        best_score = -float(
            "inf"
        )

        for action, edge in (
            node.children.items()
        ):

            if edge.visits <= 0:

                score = float(
                    "inf"
                )

            else:

                exploration = (
                    self.exploration_constant
                    * math.sqrt(
                        math.log(
                            parent_visits
                            + 1.0
                        )
                        / edge.visits
                    )
                )

                score = (
                    edge.value
                    + exploration
                )

            if score > best_score:

                best_score = score
                best_action = int(
                    action
                )

        if best_action is None:
            raise RuntimeError(
                "MCTS failed to select an action"
            )

        return best_action

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout(
        self,
        state: BeliefState,
        remaining_depth: int,
    ) -> float:

        if (
            remaining_depth <= 0
            or state.elapsed_slots
            >= BASE_SLOTS
        ):
            return 0.0

        total = 0.0
        depth_discount = 1.0

        for _ in range(
            remaining_depth
        ):

            contexts = (
                state.observe_contexts()
            )

            predicted_rewards = self._predict(contexts)

            # Optional tiny stochastic rollout exploration.
            if (
                self.rollout_epsilon > 0.0
                and self.rng.random()
                < self.rollout_epsilon
            ):

                action = int(
                    self.rng.integers(
                        N_BANDS
                    )
                )

            else:

                action = int(
                    np.argmax(
                        predicted_rewards
                    )
                )

            dwell = min(
                int(state.native_dwell_slots[action]),
                BASE_SLOTS - int(state.elapsed_slots),
            )

            predicted_reward = float(
                predicted_rewards[
                    action
                ]
            )

            total += (
                depth_discount
                * predicted_reward
            )

            hit_probability = (
                state
                .belief
                .aggregate_hit_probability(
                    action,
                    dwell,
                )
            )

            aggregate_hit = (
                self.rng.random()
                < hit_probability
            )

            state.commit_dwell(
                action=action,
                aggregate_hit=aggregate_hit,
                dwell_slots=dwell,
                observed_reward=predicted_reward,
            )

            depth_discount *= (
                self._discount(
                    dwell
                )
            )

            if (
                state.elapsed_slots
                >= BASE_SLOTS
            ):

                break

        return float(
            total
        )

    @staticmethod
    def _discount(
        dwell_slots: int,
    ) -> float:

        # Same base-slot temporal convention as SAC:
        # each elapsed receiver slot advances the discount.
        gamma_base = 0.997

        return float(
            gamma_base
            ** int(dwell_slots)
        )

    def _predict(self, contexts: np.ndarray) -> np.ndarray:
        """Use the posterior snapshot fixed at the start of this plan."""
        return np.asarray(contexts, dtype=np.float64) @ self.reward_weights


# =============================================================================
# SHARED RUNNER POLICY
# =============================================================================

class VyaptiBeliefMCTSPolicy:

    FORMAT = "vyapti_belief_mcts_v2"

    ALGORITHM_NAME = (
        "belief_mcts"
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
                f"This policy expects {N_BANDS} bands"
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

        self.rng = np.random.default_rng(
            self.seed
        )

        # --------------------------------------------------------------
        # Receiver model
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

        p01 = _as_band_array(
            self.settings.get(
                "inactive_to_active_probability",
                TRAIN250_P01,
            ),
            self.bands,
            "P01",
        )

        p11 = _as_band_array(
            self.settings.get(
                "active_to_active_probability",
                TRAIN250_P11,
            ),
            self.bands,
            "P11",
        )

        prior = _as_band_array(
            self.settings.get(
                "prior_active_probability",
                TRAIN250_PRIOR_ACTIVE,
            ),
            self.bands,
            "prior_active",
        )

        _validate_probabilities(
            p01,
            "P01",
        )

        _validate_probabilities(
            p11,
            "P11",
        )

        _validate_probabilities(
            prior,
            "prior_active",
        )

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
        ] = 1.0 - p01

        transition[
            :,
            0,
            1,
        ] = p01

        transition[
            :,
            1,
            0,
        ] = 1.0 - p11

        transition[
            :,
            1,
            1,
        ] = p11

        self.belief_template = (
            TwoStateHMMBelief(
                transition=transition,
                pd=self.pd,
                pfa=self.pfa,
                prior_active=prior,
            )
        )

        # --------------------------------------------------------------
        # EXACT native dwell profile
        # --------------------------------------------------------------

        dwell_setting = self.settings.get(
            "native_dwell_slots",
            None,
        )

        if dwell_setting is None:

            raise ValueError(
                "native_dwell_slots is required. "
                "Pass the exact 36-entry vector from the current "
                "benchmark metadata. Do not guess the 7 two-slot bands."
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
                f"shape ({self.bands},)"
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
                "Expected exactly 29 one-slot bands"
            )

        if int(
            np.sum(
                self.native_dwell_slots == 2
            )
        ) != 7:

            raise ValueError(
                "Expected exactly 7 two-slot bands"
            )

        # --------------------------------------------------------------
        # Periodicity
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

        # --------------------------------------------------------------
        # Bayesian reward model
        # --------------------------------------------------------------

        self.reward_model = (
            BayesianRewardModel(
                dimension=CONTEXT_DIM,
                prior_precision=float(
                    self.settings.get(
                        "reward_prior_precision",
                        DEFAULT_REWARD_PRIOR_PRECISION,
                    )
                ),
                reward_noise_std=float(
                    self.settings.get(
                        "reward_noise_std",
                        DEFAULT_REWARD_NOISE_STD,
                    )
                ),
            )
        )

        # --------------------------------------------------------------
        # MCTS configuration
        # --------------------------------------------------------------

        self.planning_simulations = int(
            self.settings.get(
                "mcts_simulations",
                DEFAULT_SIMULATIONS,
            )
        )

        self.planning_horizon = int(
            self.settings.get(
                "mcts_horizon",
                DEFAULT_HORIZON,
            )
        )

        self.uct_c = float(
            self.settings.get(
                "mcts_uct_c",
                DEFAULT_UCT_C,
            )
        )

        self.rollout_epsilon = float(
            self.settings.get(
                "mcts_rollout_epsilon",
                DEFAULT_ROLLOUT_EPSILON,
            )
        )

        self.warmup_actions = int(
            self.settings.get(
                "warmup_actions",
                DEFAULT_WARMUP_ACTIONS,
            )
        )

        if self.warmup_actions < self.bands:

            self.warmup_actions = (
                self.bands
            )

        # --------------------------------------------------------------
        # Training state
        # --------------------------------------------------------------

        self.training_band_visits = np.zeros(
            self.bands,
            dtype=np.int64,
        )

        self.total_actions = 0
        self.total_reward_updates = 0

        # --------------------------------------------------------------
        # Episode state
        # --------------------------------------------------------------

        self.state: BeliefState | None = None

        self.episode_decisions = 0
        self.episode_reward = 0.0

        self.band_counts = np.zeros(
            self.bands,
            dtype=np.int64,
        )

        self._pending = None

        self._last_plan_info = {}
        self._planning_seconds_total = 0.0
        self._planning_calls = 0

        self.reset_episode(
            training=True
        )

        if checkpoint is not None:

            self._load(
                checkpoint
            )

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def reset_episode(
        self,
        *,
        training: bool,
    ) -> None:

        self.state = BeliefState(
            belief=self.belief_template.copy_reset(),
            periodicity=CausalPeriodicity(
                max_events=(
                    self.periodicity_history
                ),
                tolerance_fraction=(
                    self.periodicity_tolerance
                ),
            ),
            native_dwell_slots=(
                self.native_dwell_slots.copy()
            ),
            last_visit_slot=np.full(
                self.bands,
                -1,
                dtype=np.int32,
            ),
            candidate_visit_count=np.zeros(self.bands, dtype=np.int64),
            candidate_hit_count=np.zeros(self.bands, dtype=np.int64),
            last_hit_slot=np.full(self.bands, -1, dtype=np.int32),
        )

        self.episode_decisions = 0
        self.episode_reward = 0.0

        self.band_counts = np.zeros(
            self.bands,
            dtype=np.int64,
        )

        self._pending = None

        self._last_plan_info = {}
        self._planning_seconds_total = 0.0
        self._planning_calls = 0

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup_action(
        self,
    ) -> int:

        if self.total_actions < self.bands:

            return int(
                self.total_actions
            )

        visits = (
            self.training_band_visits
        )

        minimum = int(
            visits.min()
        )

        candidates = np.flatnonzero(
            visits == minimum
        )

        return int(
            self.rng.choice(
                candidates
            )
        )

    # ------------------------------------------------------------------
    # Real action selection
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
                "Episode state has not been initialized"
            )

        # --------------------------------------------------------------
        # Truth-hidden clock synchronization
        # --------------------------------------------------------------

        if int(
            public_state.time_slot
        ) != int(
            self.state.elapsed_slots
        ):

            raise RuntimeError(
                "MCTS policy clock diverged "
                "from environment clock"
            )

        # --------------------------------------------------------------
        # Initial coverage
        # --------------------------------------------------------------

        if (
            training
            and self.total_actions
            < self.warmup_actions
        ):

            action = (
                self._warmup_action()
            )

            self._last_plan_info = {
                "mode": "warmup",
                "simulations": 0,
            }

        else:

            planner = BeliefMCTS(
                reward_model=(
                    self.reward_model
                ),
                rng=self.rng,
                simulations=(
                    self.planning_simulations
                ),
                horizon=(
                    self.planning_horizon
                ),
                exploration_constant=(
                    self.uct_c
                ),
                rollout_epsilon=(
                    self.rollout_epsilon
                ),
            )

            plan_started = time.perf_counter()
            action, plan_info = planner.plan(self.state)
            plan_seconds = time.perf_counter() - plan_started
            self._planning_seconds_total += plan_seconds
            self._planning_calls += 1
            plan_info["planning_seconds"] = float(plan_seconds)

            self._last_plan_info = (
                plan_info
            )

        self.band_counts[
            action
        ] += 1

        # Store the exact context from the real state before executing.
        contexts = (
            self.state.observe_contexts()
        )

        self._pending = {
            "action": int(action),
            "context": contexts[
                action
            ].copy(),
            "start_slot": int(
                public_state.time_slot
            ),
        }

        return int(
            action
        )

    # ------------------------------------------------------------------
    # Real transition
    # ------------------------------------------------------------------

    @staticmethod
    def _elapsed_slots(
        transition: PublicTransition,
    ) -> int:

        elapsed = (
            int(
                transition.next_state.time_slot
            )
            - int(
                transition.state.time_slot
            )
        )

        if elapsed not in (
            1,
            2,
        ):

            raise ValueError(
                f"Expected one/two-slot dwell, got {elapsed}"
            )

        return elapsed

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

        observation = (
            transition
            .next_state
            .previous_observation
        )

        if observation is None:

            raise ValueError(
                "Belief-MCTS requires the public receiver observation"
            )

        action = int(
            transition.action
        )

        pending_action = int(
            self._pending[
                "action"
            ]
        )

        if action != pending_action:

            raise RuntimeError(
                "Transition action differs from pending MCTS action"
            )

        if int(
            observation.selected_band
        ) != action:

            raise ValueError(
                "Action and selected receiver band disagree"
            )

        elapsed_slots = (
            self._elapsed_slots(
                transition
            )
        )

        if int(
            self.state.elapsed_slots
        ) != int(
            transition.state.time_slot
        ):

            raise RuntimeError(
                "Policy causal clock diverged from transition clock"
            )

        expected_dwell = min(
            int(self.native_dwell_slots[action]),
            BASE_SLOTS - int(self.state.elapsed_slots),
        )

        if elapsed_slots != expected_dwell:

            raise RuntimeError(
                f"Native dwell mismatch: "
                f"band={action}, "
                f"expected={expected_dwell}, "
                f"transition={elapsed_slots}"
            )

        # ==============================================================
        # AUTHORITATIVE REWARD
        # ==============================================================

        reward = float(
            transition.reward
        )

        if not math.isfinite(
            reward
        ):

            raise RuntimeError(
                f"Non-finite transition reward: {reward}"
            )

        # ==============================================================
        # REAL OBSERVATION -> CAUSAL STATE
        # ==============================================================

        aggregate_hit = bool(
            observation.hit
        )

        self.state.commit_dwell(
            action=action,
            aggregate_hit=aggregate_hit,
            dwell_slots=elapsed_slots,
            observed_reward=reward,
        )

        # ==============================================================
        # UPDATE BAYESIAN REWARD MODEL
        # ==============================================================

        if training:

            selected_context = (
                self._pending[
                    "context"
                ]
            )

            # IMPORTANT:
            #
            # This is the only reward that enters the reward model.
            #
            # The MCTS planner itself never invents a real reward.
            self.reward_model.update(
                context=selected_context,
                reward=reward,
            )

            self.total_actions += 1

            self.total_reward_updates += 1

            self.training_band_visits[
                action
            ] += 1

        self.episode_decisions += 1

        self.episode_reward += reward

        self._pending = None

    # ------------------------------------------------------------------
    # Episode reporting
    # ------------------------------------------------------------------

    def end_episode(
        self,
        *,
        training: bool,
    ) -> dict[str, Any]:

        if self._pending is not None:

            raise RuntimeError(
                "Cannot end episode before final observation"
            )

        posterior_mean = (
            self.reward_model
            .posterior_mean()
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

            "reward_model_updates": int(
                self.total_reward_updates
            ),

            "band_selection_counts": (
                self.band_counts.tolist()
            ),

            "training_band_visits": (
                self.training_band_visits.tolist()
            ),

            "reward_model_parameter_norm": float(
                np.linalg.norm(
                    posterior_mean
                )
            ),

            "planning_simulations": int(
                self.planning_simulations
            ),

            "planning_horizon": int(
                self.planning_horizon
            ),

            "uct_c": float(
                self.uct_c
            ),

            "last_plan": (
                self._last_plan_info
            ),

            "planning_calls": int(self._planning_calls),

            "planning_seconds_total": float(self._planning_seconds_total),

            "planning_seconds_mean": float(
                self._planning_seconds_total / max(self._planning_calls, 1)
            ),
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(
        self,
        path,
    ) -> None:

        torch.save(
            {
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

                "total_reward_updates": int(
                    self.total_reward_updates
                ),

                "training_band_visits": (
                    self.training_band_visits.copy()
                ),

                "reward_model_A": (
                    self.reward_model.A.copy()
                ),

                "reward_model_b": (
                    self.reward_model.b.copy()
                ),

                "reward_model_L": (
                    self.reward_model.L.copy()
                ),

                "reward_prior_precision": (
                    self.reward_model.prior_precision
                ),

                "reward_noise_std": (
                    self.reward_model.reward_noise_std
                ),

                "rng_state": (
                    self.rng.bit_generator.state
                ),
            },
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
                "Checkpoint is not a Vyapti Belief-MCTS checkpoint"
            )

        if int(
            payload.get(
                "bands",
                -1,
            )
        ) != self.bands:

            raise ValueError(
                "Checkpoint action space does not match"
            )

        if int(
            payload.get(
                "context_dim",
                -1,
            )
        ) != CONTEXT_DIM:

            raise ValueError(
                "Checkpoint context dimension does not match"
            )

        A = np.asarray(
            payload[
                "reward_model_A"
            ],
            dtype=np.float64,
        )

        b = np.asarray(
            payload[
                "reward_model_b"
            ],
            dtype=np.float64,
        )

        L = np.asarray(
            payload[
                "reward_model_L"
            ],
            dtype=np.float64,
        )

        if A.shape != (
            CONTEXT_DIM,
            CONTEXT_DIM,
        ):

            raise ValueError(
                "Invalid reward-model A shape"
            )

        if b.shape != (
            CONTEXT_DIM,
        ):

            raise ValueError(
                "Invalid reward-model b shape"
            )

        if L.shape != (
            CONTEXT_DIM,
            CONTEXT_DIM,
        ):

            raise ValueError(
                "Invalid reward-model L shape"
            )

        self.reward_model.A = (
            A.copy()
        )

        self.reward_model.b = (
            b.copy()
        )

        self.reward_model.L = (
            L.copy()
        )

        self.reward_model.prior_precision = float(
            payload.get(
                "reward_prior_precision",
                self.reward_model.prior_precision,
            )
        )

        self.reward_model.reward_noise_std = float(
            payload.get(
                "reward_noise_std",
                self.reward_model.reward_noise_std,
            )
        )

        self.reward_model.noise_variance = (
            self.reward_model.reward_noise_std
            ** 2
        )

        self.total_actions = int(
            payload.get(
                "total_actions",
                0,
            )
        )

        self.total_reward_updates = int(
            payload.get(
                "total_reward_updates",
                0,
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
                "Invalid training_band_visits shape"
            )

        self.training_band_visits = (
            visits.copy()
        )

        if "rng_state" in payload:

            self.rng.bit_generator.state = (
                payload[
                    "rng_state"
                ]
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

    return VyaptiBeliefMCTSPolicy(
        bands=bands,
        seed=seed,
        settings=settings,
        checkpoint=checkpoint,
    )

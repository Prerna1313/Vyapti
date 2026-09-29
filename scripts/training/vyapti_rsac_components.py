from __future__ import annotations

"""
Vyapti Residual-SAC components.

Purpose
-------
Adds a causal scheduler layer above the existing TSRD/STARE environment.

Existing system retained:
    causal_harness.py
    vyapti_train500_cache.py
    TSRDStareEnvironment
    score_recorded_replay()

New capabilities:
    1. Population-level PO-RMAB / Whittle structured prior
    2. Causal UCB exploration feature
    3. Periodic / scan-window opportunity feature
    4. Online frequency-agility cue from observed positive-band transitions
    5. Operational priority feature
    6. Causal R1 utility reward
    7. Airtime-concentration cost
    8. Explicit retune accounting
    9. Residual-policy observation construction

IMPORTANT
---------
No hidden truth is used by the scheduler or reward.

Hidden truth remains evaluator-only.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import math

import numpy as np

from causal_harness import (
    N_BANDS,
    N_BASE_SLOTS,
    BASE_SLOT_S,
    PD,
    PFA,
    RETUNE_TIME_MS,
    CausalSchedulerState,
    BeliefFilter,
)


EPS = 1e-12


# ============================================================
# SMALL NUMERICAL UTILITIES
# ============================================================

def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-8:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mean) / std


def softmax_np(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / max(float(temperature), 1e-8)
    z = z - np.max(z)
    e = np.exp(z)
    return e / max(float(np.sum(e)), EPS)


def binary_entropy(p: float | np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return -(x * np.log2(x) + (1.0 - x) * np.log2(1.0 - x))


# ============================================================
# CAUSAL BELIEF SIMULATION FOR REWARD
# ============================================================

def simulate_belief_observations(
    state: CausalSchedulerState,
    band: int,
    positives: list[bool],
) -> tuple[list[float], list[float]]:
    """
    Compute the pre-observation belief and Bayesian posterior for
    every base slot in a dwell WITHOUT mutating the actual state.

    This mirrors BeliefFilter.step_observations() semantics:
        posterior update
        followed by one transition prediction step.

    Returns
    -------
    pre_beliefs
        Belief immediately before each observed slot.

    posteriors
        Observation-conditioned posterior immediately after each hit/miss,
        before the next transition prediction.
    """

    band = int(band)

    belief = np.asarray(
        state.belief.belief,
        dtype=np.float64,
    ).copy()

    pre_beliefs: list[float] = []
    posteriors: list[float] = []

    for y in positives:

        p_before = float(
            np.clip(
                belief[band],
                1e-8,
                1.0 - 1e-8,
            )
        )

        if bool(y):
            like_active = float(PD)
            like_inactive = float(PFA)
        else:
            like_active = float(1.0 - PD)
            like_inactive = float(1.0 - PFA)

        denom = (
            p_before * like_active
            + (1.0 - p_before) * like_inactive
        )

        p_post = float(
            np.clip(
                p_before
                * like_active
                / max(denom, EPS),
                1e-8,
                1.0 - 1e-8,
            )
        )

        pre_beliefs.append(p_before)
        posteriors.append(p_post)

        # Apply observation-conditioned posterior, then transition model.
        next_belief = belief.copy()

        for b in range(N_BANDS):

            if b == band:
                p = p_post
            else:
                p = float(belief[b])

            p01 = float(
                state.belief.transition[b, 0, 1]
            )

            p11 = float(
                state.belief.transition[b, 1, 1]
            )

            next_belief[b] = np.clip(
                (1.0 - p) * p01 + p * p11,
                1e-8,
                1.0 - 1e-8,
            )

        belief = next_belief

    return pre_beliefs, posteriors


# ============================================================
# ONLINE FREQUENCY-AGILITY CUE
# ============================================================

class FrequencyAgilityCue:
    """
    Causal online transition model over observed positive bands.

    It does NOT use emitter IDs or hidden truth.

    Whenever a positive is observed on band b, we record:
        previous_positive_band -> b

    The resulting transition distribution is an empirical cue for
    frequency-agile behavior.

    This is deliberately a lightweight baseline expert. Later it can
    be replaced by a learned temporal predictor without changing SAC.
    """

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = float(smoothing)
        self.counts = np.full(
            (N_BANDS, N_BANDS),
            self.smoothing,
            dtype=np.float64,
        )
        self.last_positive_band = -1

    def reset(self) -> None:
        self.counts.fill(self.smoothing)
        self.last_positive_band = -1

    def observe(
        self,
        band: int,
        positives: list[bool],
    ) -> None:

        band = int(band)

        if not any(positives):
            return

        if self.last_positive_band >= 0:
            self.counts[
                self.last_positive_band,
                band,
            ] += 1.0

        self.last_positive_band = band

    def distribution(self) -> np.ndarray:
        if self.last_positive_band < 0:
            return np.full(
                N_BANDS,
                1.0 / N_BANDS,
                dtype=np.float64,
            )

        row = self.counts[
            self.last_positive_band
        ]

        return row / max(
            float(row.sum()),
            EPS,
        )

    def next_band_score(self) -> np.ndarray:
        """
        P(next positive observation occurs on band b | causal history)
        """
        return self.distribution().astype(np.float32)


# ============================================================
# STRUCTURED PO-RMAB / WHITTLE PRIOR
# ============================================================

class StructuredPrior:
    """
    Population-level structured prior.

    This reads the already-generated PO-RMAB calibration artifacts.

    IMPORTANT:
        The prior is global training knowledge.
        It is not the hidden current-world truth.
    """

    def __init__(
        self,
        prior_dir: Path,
        expected_fingerprint: str | None = None,
    ):

        prior_dir = Path(prior_dir)

        transition_file = (
            prior_dir / "pormab_transition.json"
        )

        tables_file = (
            prior_dir / "pormab_index_tables.npy"
        )

        diagnostics_file = (
            prior_dir / "pormab_index_diagnostics.json"
        )

        if not transition_file.exists():
            raise FileNotFoundError(
                f"Missing {transition_file}"
            )

        if not tables_file.exists():
            raise FileNotFoundError(
                f"Missing {tables_file}"
            )

        if not diagnostics_file.exists():
            raise FileNotFoundError(
                f"Missing {diagnostics_file}"
            )

        payload = json.loads(
            transition_file.read_text(
                encoding="utf-8"
            )
        )

        if (
            expected_fingerprint is not None
            and payload.get("source_pool_fingerprint")
            not in (None, expected_fingerprint)
        ):
            raise RuntimeError(
                "PO-RMAB prior fingerprint does not match "
                "the current TRAIN-500 cache.\n"
                f"Prior: {payload.get('source_pool_fingerprint')}\n"
                f"Cache: {expected_fingerprint}"
            )

        self.transition = np.asarray(
            payload["transition"],
            dtype=np.float64,
        )

        self.prior_active = np.asarray(
            payload["prior_active"],
            dtype=np.float64,
        )

        diagnostics = json.loads(
            diagnostics_file.read_text(
                encoding="utf-8"
            )
        )

        self.belief_grid = np.asarray(
            diagnostics["belief_grid"],
            dtype=np.float64,
        )

        self.index_tables = np.load(
            tables_file
        )

        if self.transition.shape != (
            N_BANDS,
            2,
            2,
        ):
            raise RuntimeError(
                f"Unexpected transition shape: "
                f"{self.transition.shape}"
            )

        if self.prior_active.shape != (
            N_BANDS,
        ):
            raise RuntimeError(
                f"Unexpected prior shape: "
                f"{self.prior_active.shape}"
            )

        if self.index_tables.shape[0] != N_BANDS:
            raise RuntimeError(
                f"Unexpected Whittle table shape: "
                f"{self.index_tables.shape}"
            )

    def whittle_scores(
        self,
        beliefs: np.ndarray,
    ) -> np.ndarray:

        beliefs = np.asarray(
            beliefs,
            dtype=np.float64,
        )

        result = np.zeros(
            N_BANDS,
            dtype=np.float64,
        )

        for band in range(N_BANDS):
            result[band] = np.interp(
                np.clip(
                    beliefs[band],
                    0.0,
                    1.0,
                ),
                self.belief_grid,
                self.index_tables[band],
            )

        return result


# ============================================================
# RESIDUAL POLICY STATE
# ============================================================

@dataclass
class RSACState:
    causal: CausalSchedulerState
    agility: FrequencyAgilityCue

    @classmethod
    def create(
        cls,
        transition: np.ndarray,
        prior_active: np.ndarray,
    ) -> "RSACState":

        return cls(
            causal=CausalSchedulerState.create(
                transition,
                prior_active=prior_active,
            ),
            agility=FrequencyAgilityCue(),
        )

    def reset(
        self,
        prior_active: np.ndarray,
    ) -> None:

        self.causal.reset(
            prior_active
        )

        self.agility.reset()

    @property
    def previous_action(self) -> int:
        return int(
            self.causal.previous_action
        )

    @property
    def previous_positive(self) -> bool:
        return bool(
            self.causal.previous_positive
        )

    def pre_features(
        self,
        prior: StructuredPrior,
    ) -> dict[str, np.ndarray]:

        f = self.causal.pre_action_features()

        belief = np.asarray(
            f["belief"],
            dtype=np.float64,
        )

        entropy = np.asarray(
            f["belief_entropy"],
            dtype=np.float64,
        )

        staleness = np.asarray(
            f["staleness"],
            dtype=np.float64,
        )

        periodicity = np.asarray(
            f["periodicity"],
            dtype=np.float64,
        )

        periodicity_conf = np.asarray(
            f["periodicity_confidence"],
            dtype=np.float64,
        )

        visits = np.asarray(
            f["visit_count_norm"],
            dtype=np.float64,
        )

        agility = self.agility.next_band_score().astype(
            np.float64
        )

        # ----------------------------------------------------
        # D1/D2 proxy:
        # causal opportunity window inferred from periodicity.
        #
        # This is not a learned spatial tracker yet.
        # It is an explicit expert interface that can later be
        # replaced by the learned spatial predictor.
        # ----------------------------------------------------
        scan_window = (
            periodicity
            * periodicity_conf
        )

        # ----------------------------------------------------
        # F — operational priority
        #
        # This is NOT "hostile probability".
        # It is a causal sensing-priority heuristic.
        # ----------------------------------------------------
        threat_priority = (
            belief
            * (
                0.5
                + 0.5 * periodicity_conf
            )
            + 0.25 * entropy
        )

        # ----------------------------------------------------
        # B — structured population prior
        # ----------------------------------------------------
        whittle = prior.whittle_scores(
            belief
        )

        total_actions = max(
            int(round(
                self.causal.elapsed_base_slots
                / 1.0
            )),
            1,
        )

        counts = np.asarray(
            self.causal.visit_counts,
            dtype=np.float64,
        )

        ucb = np.sqrt(
            np.log(
                total_actions + 2.0
            )
            /
            np.maximum(
                counts + 1.0,
                1.0,
            )
        )

        base_score = (
            1.00 * standardize(whittle)
            + 0.35 * standardize(ucb)
            + 0.20 * standardize(scan_window)
            + 0.15 * standardize(staleness)
            + 0.20 * standardize(agility)
            + 0.25 * standardize(threat_priority)
        )

        base_logits = np.clip(
            base_score,
            -5.0,
            5.0,
        ).astype(np.float32)

        prev_onehot = np.zeros(
            N_BANDS,
            dtype=np.float32,
        )

        if 0 <= self.previous_action < N_BANDS:
            prev_onehot[
                self.previous_action
            ] = 1.0

        elapsed_norm = np.float32(
            np.clip(
                self.causal.elapsed_base_slots
                / max(N_BASE_SLOTS, 1),
                0.0,
                1.0,
            )
        )

        return {
            "belief": belief.astype(np.float32),
            "belief_entropy": entropy.astype(np.float32),
            "staleness": staleness.astype(np.float32),
            "visit_count_norm": visits.astype(np.float32),
            "periodicity": periodicity.astype(np.float32),
            "periodicity_confidence": periodicity_conf.astype(np.float32),
            "period_norm": np.asarray(
                f["period_norm"],
                dtype=np.float32,
            ),
            "agility_probability": agility.astype(np.float32),
            "scan_window": scan_window.astype(np.float32),
            "threat_priority": threat_priority.astype(np.float32),
            "ucb": ucb.astype(np.float32),
            "whittle": whittle.astype(np.float32),
            "base_logits": base_logits,
            "previous_action_onehot": prev_onehot,
            "elapsed_norm": np.asarray(
                [elapsed_norm],
                dtype=np.float32,
            ),
            "previous_positive": np.asarray(
                [float(self.previous_positive)],
                dtype=np.float32,
            ),
        }

    def observation(
        self,
        prior: StructuredPrior,
    ) -> tuple[np.ndarray, np.ndarray]:

        f = self.pre_features(prior)

        parts = [
            f["belief"],
            f["belief_entropy"],
            f["staleness"],
            f["visit_count_norm"],
            f["periodicity"],
            f["periodicity_confidence"],
            f["period_norm"],
            f["agility_probability"],
            f["scan_window"],
            f["threat_priority"],
            f["ucb"],
            f["whittle"],
            f["base_logits"],
            f["previous_action_onehot"],
            f["elapsed_norm"],
            f["previous_positive"],
        ]

        obs = np.concatenate(
            parts
        ).astype(np.float32)

        return (
            obs,
            f["base_logits"].copy(),
        )

    def apply_observation(
        self,
        action: int,
        positives: list[bool],
        dwell_slots: int,
    ) -> None:

        self.causal.belief.step_observations(
            action,
            positives,
        )

        self.agility.observe(
            action,
            positives,
        )

        self.causal.step(
            action,
            dwell_slots,
            positives,
        )


# ============================================================
# R1 CAUSAL REWARD
# ============================================================

@dataclass
class RewardBreakdown:
    reward: float
    detection_surprise: float
    information_gain: float
    coverage_utility: float
    scan_utility: float
    concentration_cost: float
    switch_cost: float
    retune_cost: float


class VyaptiRewardV1:
    """
    Causal scheduler utility.

    Utility:
        detection surprise
        information gain
        coverage / staleness
        periodic / scan-window opportunity

    Direct small cost:
        physical retune fraction

    Constraint signal:
        airtime concentration

    No hidden truth is used.
    """

    HIT_WEIGHT = 1.00
    INFO_WEIGHT = 0.40
    COVERAGE_WEIGHT = 0.15
    SCAN_WEIGHT = 0.10

    RETUNE_REWARD_WEIGHT = 0.10

    def compute(
        self,
        state: RSACState,
        prior: StructuredPrior,
        action: int,
        positives: list[bool],
        dwell_slots: int,
    ) -> RewardBreakdown:

        action = int(action)
        dwell_slots = int(dwell_slots)

        pre_beliefs, posteriors = (
            simulate_belief_observations(
                state.causal,
                action,
                positives,
            )
        )

        # ----------------------------------------------------
        # Detection-surprise utility
        # ----------------------------------------------------
        detection_surprise = 0.0

        for y, p in zip(
            positives,
            pre_beliefs,
        ):
            if bool(y):
                detection_surprise += (
                    0.5
                    + 0.5 * (1.0 - p)
                )

        # ----------------------------------------------------
        # Information gain
        # ----------------------------------------------------
        information_gain = 0.0

        for p_before, p_post in zip(
            pre_beliefs,
            posteriors,
        ):
            information_gain += float(
                binary_entropy(p_before)
                - binary_entropy(p_post)
            )

        # ----------------------------------------------------
        # Coverage / staleness
        # ----------------------------------------------------
        features = state.pre_features(
            prior
        )

        selected_staleness = float(
            features["staleness"][action]
        )

        selected_belief = float(
            features["belief"][action]
        )

        coverage_utility = (
            selected_staleness
            * (1.0 - selected_belief)
        )

        # ----------------------------------------------------
        # Periodic / scan opportunity
        # ----------------------------------------------------
        scan_utility = float(
            features["scan_window"][action]
        )

        # ----------------------------------------------------
        # Airtime concentration
        # ----------------------------------------------------
        previous_total_slots = float(
            state.causal.elapsed_base_slots
        )

        projected_total_slots = (
            previous_total_slots
            + float(dwell_slots)
        )

        projected_action_slots = (
            float(
                state.causal.visit_counts[action]
            )
            * float(dwell_slots)
            + float(dwell_slots)
        )

        share = (
            projected_action_slots
            /
            max(
                projected_total_slots,
                1.0,
            )
        )

        fair_share = 1.0 / N_BANDS

        concentration_cost = max(
            0.0,
            (
                share - fair_share
            )
            /
            max(
                1.0 - fair_share,
                EPS,
            ),
        )

        # ----------------------------------------------------
        # Switching + physical retune accounting
        # ----------------------------------------------------
        previous_action = (
            state.previous_action
        )

        switch_cost = float(
            previous_action >= 0
            and previous_action != action
        )

        dwell_seconds = (
            BASE_SLOT_S
            * max(dwell_slots, 1)
        )

        retune_fraction = (
            float(RETUNE_TIME_MS) / 1000.0
        ) / max(
            dwell_seconds,
            1e-9,
        )

        retune_cost = (
            switch_cost
            * retune_fraction
        )

        # ----------------------------------------------------
        # Utility
        # ----------------------------------------------------
        reward = (
            self.HIT_WEIGHT
            * detection_surprise
            + self.INFO_WEIGHT
            * information_gain
            + self.COVERAGE_WEIGHT
            * coverage_utility
            + self.SCAN_WEIGHT
            * scan_utility
            - self.RETUNE_REWARD_WEIGHT
            * retune_cost
        )

        return RewardBreakdown(
            reward=float(reward),
            detection_surprise=float(
                detection_surprise
            ),
            information_gain=float(
                information_gain
            ),
            coverage_utility=float(
                coverage_utility
            ),
            scan_utility=float(
                scan_utility
            ),
            concentration_cost=float(
                concentration_cost
            ),
            switch_cost=float(
                switch_cost
            ),
            retune_cost=float(
                retune_cost
            ),
        )


# ============================================================
# REPLAY ENVIRONMENT ADAPTER
# ============================================================

class VyaptiRSACEpisode:
    """
    Thin wrapper around the existing TSRD environment.

    It does not alter hidden truth or receiver physics.
    """

    def __init__(
        self,
        env: Any,
        state: RSACState,
        prior: StructuredPrior,
        reward_engine: VyaptiRewardV1,
        dwell_slots: int,
    ):
        self.env = env
        self.state = state
        self.prior = prior
        self.reward_engine = reward_engine
        self.dwell_slots = int(dwell_slots)

    def reset(
        self,
        seed: int,
        prior_active: np.ndarray,
    ) -> np.ndarray:

        self.env.reset(
            seed=int(seed)
        )

        self.state.reset(
            prior_active
        )

        obs, _ = self.state.observation(
            self.prior
        )

        return obs

    def action_space_size(self) -> int:
        return N_BANDS

    def current_observation(
        self
    ) -> np.ndarray:

        obs, _ = self.state.observation(
            self.prior
        )

        return obs

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        float,
        bool,
        dict[str, float],
    ]:

        action = int(action)

        looks, _env_reward, done = (
            self.env.step_dwell_training(
                action,
                self.dwell_slots,
                reward_mode="detector_positive",
            )
        )

        looks = list(looks)

        if len(looks) == 0:
            raise RuntimeError(
                "TSRD environment returned zero looks"
            )

        positives = [
            bool(
                look["hit"]
            )
            for look in looks
        ]

        breakdown = (
            self.reward_engine.compute(
                self.state,
                self.prior,
                action,
                positives,
                len(looks),
            )
        )

        self.state.apply_observation(
            action,
            positives,
            len(looks),
        )

        next_obs, _ = (
            self.state.observation(
                self.prior
            )
        )

        info = {
            "detection_surprise":
                breakdown.detection_surprise,
            "information_gain":
                breakdown.information_gain,
            "coverage_utility":
                breakdown.coverage_utility,
            "scan_utility":
                breakdown.scan_utility,
            "concentration_cost":
                breakdown.concentration_cost,
            "switch_cost":
                breakdown.switch_cost,
            "retune_cost":
                breakdown.retune_cost,
            "positive_count":
                float(sum(positives)),
        }

        return (
            next_obs,
            breakdown.reward,
            breakdown.concentration_cost,
            bool(done),
            info,
        )

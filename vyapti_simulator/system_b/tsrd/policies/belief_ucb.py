"""Receiver-belief UCB heuristic for the shared TSRD Mode-B runner.

Unlike classical UCB1, this policy exploits a factorized HMM prediction of
detector-hit probability and explores using per-band visit counts. It does not
learn from the environment's scalar reward.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..algorithm_interface import PublicTransition


EPS = 1e-12


@dataclass(frozen=True)
class BeliefUCBConfig:
    n_bands: int = 36
    pd: float = 0.90
    pfa: float = 0.05
    prior_active: float = 0.50
    inactive_to_active: float = 0.05
    active_to_active: float = 0.90
    exploration_c: float = 0.25
    warm_start_all_bands: bool = True


class AggregateBelief:
    """Factorized two-state HMM updated from one aggregate result per dwell."""

    def __init__(self, transition: np.ndarray, *, pd: float, pfa: float,
                 prior_active: float):
        transition = np.asarray(transition, dtype=np.float64)
        if transition.ndim != 3 or transition.shape[1:] != (2, 2):
            raise ValueError(f"transition must have shape (N, 2, 2), got {transition.shape}")
        if (not np.all(np.isfinite(transition)) or np.any(transition < 0.0)
                or not np.allclose(transition.sum(axis=-1), 1.0, atol=1e-8)):
            raise ValueError("Transition probabilities must be finite, nonnegative, and row-normalized")
        if not 0.0 < pfa < pd < 1.0 or not 0.0 < prior_active < 1.0:
            raise ValueError("Require 0 < Pfa < Pd < 1 and prior_active in (0, 1)")
        self.transition = transition
        self.n_bands = transition.shape[0]
        self.pd = float(pd)
        self.pfa = float(pfa)
        self.prior_active = np.full(self.n_bands, float(prior_active), dtype=np.float64)
        self.belief_active = self.prior_active.copy()

    def reset(self) -> None:
        self.belief_active = self.prior_active.copy()

    @staticmethod
    def _distribution(p_active: float) -> np.ndarray:
        p = float(np.clip(p_active, 1e-6, 1.0 - 1e-6))
        return np.asarray((1.0 - p, p), dtype=np.float64)

    def predict_all(self, dwell_slots: int) -> np.ndarray:
        return np.asarray([
            (self._distribution(self.belief_active[band])
             @ np.linalg.matrix_power(self.transition[band], dwell_slots))[1]
            for band in range(self.n_bands)
        ], dtype=np.float64)

    def hit_probability(self) -> np.ndarray:
        p = self.belief_active
        return np.clip(p * self.pd + (1.0 - p) * self.pfa, 0.0, 1.0)

    def aggregate_hit_probability(self, band: int, dwell_slots: int) -> float:
        """Probability of at least one receiver-positive cell in a dwell."""
        if not 0 <= band < self.n_bands:
            raise ValueError("Selected band is outside the belief action space")
        if dwell_slots not in (1, 2):
            raise ValueError("Belief-UCB supports the native one- or two-slot dwell")
        no_hit_emission = np.diag((1.0 - self.pfa, 1.0 - self.pd))
        no_hit = self._distribution(self.belief_active[band])
        transition = self.transition[band]
        for _ in range(dwell_slots):
            no_hit = (no_hit @ no_hit_emission) @ transition
        return float(np.clip(1.0 - no_hit.sum(), 0.0, 1.0))

    def update_after_dwell(self, band: int, aggregate_hit: bool, dwell_slots: int) -> None:
        if not 0 <= band < self.n_bands:
            raise ValueError("Selected band is outside the belief action space")
        if dwell_slots not in (1, 2):
            raise ValueError("Belief-UCB supports the native one- or two-slot dwell")

        predicted = self.predict_all(dwell_slots)
        prior = self._distribution(self.belief_active[band])
        matrix = self.transition[band]
        end_prior = prior @ np.linalg.matrix_power(matrix, dwell_slots)

        # The receiver exposes only whether any base-slot detector test hit.
        # Accumulate the joint no-hit probability over the whole dwell.
        no_hit_emission = np.diag((1.0 - self.pfa, 1.0 - self.pd))
        joint_no_hit = prior.copy()
        for _ in range(dwell_slots):
            joint_no_hit = (joint_no_hit @ no_hit_emission) @ matrix
        p_no_hit = float(joint_no_hit.sum())
        p_hit = max(1.0 - p_no_hit, 0.0)
        if aggregate_hit:
            posterior = end_prior if p_hit <= EPS else (end_prior - joint_no_hit) / p_hit
        else:
            posterior = end_prior if p_no_hit <= EPS else joint_no_hit / p_no_hit
        posterior = np.clip(posterior, 0.0, None)
        normalizer = float(posterior.sum())
        if normalizer <= EPS:
            posterior = np.asarray((0.5, 0.5), dtype=np.float64)
        else:
            posterior /= normalizer

        self.belief_active = predicted
        self.belief_active[band] = float(np.clip(posterior[1], 1e-6, 1.0 - 1e-6))


class BeliefUCBPolicy:
    """Greedy predicted hit probability plus visit-count exploration."""

    def __init__(self, bands: int, seed: int, settings: dict):
        if type(bands) is not int or bands < 1:
            raise ValueError("Belief-UCB needs a positive band count")
        self.bands = bands
        self.seed = int(seed)
        self.settings = dict(settings)
        self.exploration_c = float(self.settings.get("exploration_c", 0.25))
        self.warm_start = bool(self.settings.get("warm_start_all_bands", True))
        self.dwell_slots_by_band = tuple(
            int(value) for value in self.settings.get("dwell_slots_by_band", [1] * bands)
        )
        self.base_slot_seconds = float(self.settings.get("base_slot_seconds", 0.05))
        self.mission_duration_s = float(self.settings.get("mission_duration_s", 30.0))
        self.band_change_retune_time_s = float(self.settings.get("band_change_retune_time_s", 0.0003))
        p01 = float(self.settings.get("inactive_to_active_probability", 0.05))
        p11 = float(self.settings.get("active_to_active_probability", 0.90))
        if (not math.isfinite(self.exploration_c) or self.exploration_c < 0.0
                or not 0.0 < p01 < 1.0 or not 0.0 < p11 < 1.0
                or len(self.dwell_slots_by_band) != bands
                or any(value not in (1, 2) for value in self.dwell_slots_by_band)
                or not math.isfinite(self.base_slot_seconds) or self.base_slot_seconds <= 0.0
                or not math.isfinite(self.mission_duration_s) or self.mission_duration_s <= 0.0
                or not math.isfinite(self.band_change_retune_time_s)
                or self.band_change_retune_time_s < 0.0):
            raise ValueError("Invalid exploration coefficient or HMM transition probabilities")
        self.config = BeliefUCBConfig(
            n_bands=bands,
            pd=float(self.settings.get("detection_probability", 0.90)),
            pfa=float(self.settings.get("false_alarm_probability", 0.05)),
            prior_active=float(self.settings.get("prior_active_probability", 0.50)),
            inactive_to_active=p01,
            active_to_active=p11,
            exploration_c=self.exploration_c,
            warm_start_all_bands=self.warm_start,
        )
        transition = np.empty((bands, 2, 2), dtype=np.float64)
        transition[:, 0, :] = (1.0 - p01, p01)
        transition[:, 1, :] = (1.0 - p11, p11)
        self.belief = AggregateBelief(
            transition, pd=self.config.pd, pfa=self.config.pfa,
            prior_active=self.config.prior_active,
        )
        self.reset_episode(training=False)

    def reset_episode(self, *, training: bool) -> None:
        self.visits = np.zeros(self.bands, dtype=np.int64)
        self.total_decisions = 0
        self.last_action = -1
        self._awaiting_observation = False
        self.belief.reset()

    def _action_duration_seconds(self, band: int, state) -> tuple[int, float]:
        remaining_slots = max(
            1, int(math.ceil(self.mission_duration_s / self.base_slot_seconds)) - state.time_slot
        )
        dwell_slots = min(self.dwell_slots_by_band[band], remaining_slots)
        previous = state.previous_observation
        switches_band = previous is not None and previous.selected_band != band
        retune = self.band_change_retune_time_s if switches_band else 0.0
        return dwell_slots, dwell_slots * self.base_slot_seconds + retune

    def select_action(self, state, *, training: bool) -> int:
        if self._awaiting_observation:
            raise RuntimeError("observe must follow each Belief-UCB action")
        if self.warm_start and self.total_decisions < self.bands:
            action = self.total_decisions
        else:
            log_term = math.log1p(max(self.total_decisions, 1))
            scores = np.empty(self.bands, dtype=np.float64)
            for band in range(self.bands):
                dwell_slots, elapsed_seconds = self._action_duration_seconds(band, state)
                expected_hit = self.belief.aggregate_hit_probability(band, dwell_slots)
                exploit_rate = expected_hit / elapsed_seconds
                # Finite optimism lets warm_start_all_bands=False behave as
                # configured, while expressing exploration in the same
                # per-second units as the dwell-aware exploitation score.
                visits = max(int(self.visits[band]), 1)
                exploration_rate = (
                    self.exploration_c * math.sqrt(log_term / visits) / elapsed_seconds
                )
                scores[band] = exploit_rate + exploration_rate
            action = int(np.argmax(scores))
        self.last_action = action
        self._awaiting_observation = True
        return action

    def observe(self, transition: PublicTransition, *, training: bool) -> None:
        if not self._awaiting_observation:
            raise RuntimeError("select_action must precede observe")
        if transition.action != self.last_action:
            raise ValueError("Transition action differs from the selected Belief-UCB band")
        observation = transition.next_state.previous_observation
        if observation is None:
            raise ValueError("Belief-UCB requires a receiver observation after each action")
        slots = transition.next_state.time_slot - transition.state.time_slot
        self.belief.update_after_dwell(
            transition.action, bool(observation.hit), int(slots)
        )
        self.visits[transition.action] += 1
        self.total_decisions += 1
        self._awaiting_observation = False
        self.last_action = -1

    def end_episode(self, *, training: bool) -> dict:
        if self._awaiting_observation:
            raise RuntimeError("Cannot finish a world before observing its last action")
        return {
            "algorithm": "belief_ucb",
            "updates_during_evaluation": True,
            "adapts_from_receiver_observations": True,
            "updates_from_reward": False,
            "statistics_reset_each_world": True,
            "total_decisions": int(self.total_decisions),
            "band_selection_counts": self.visits.tolist(),
            "belief_active": self.belief.belief_active.tolist(),
            "belief_hit_probability": self.belief.hit_probability().tolist(),
            "exploration_c": self.exploration_c,
        }


def create(*, bands: int, seed: int, settings: dict, checkpoint=None):
    if checkpoint is not None:
        raise ValueError("Belief-UCB is an online heuristic and has no checkpoints")
    return BeliefUCBPolicy(bands=bands, seed=seed, settings=settings)

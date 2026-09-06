"""
PS26055 UCB-family bandit schedulers.

[BANDIT] [LITERATURE-GROUNDED] Auer, Cesa-Bianchi & Fischer (2002), "Finite-time
analysis of the multiarmed bandit problem" — UCB1.

-----------------------------------------------------------------------
Why UCB1 is included even though it is expected to underperform
-----------------------------------------------------------------------
UCB1 assumes each arm's reward is drawn i.i.d. from a FIXED distribution. The
PS26055 problem violates that assumption in two ways at once:

  - Rewards are non-stationary. A periodic emitter makes a band rewarding at
    t = 0..3 and barren at t = 4..9. There is no fixed mean to converge on.
  - Arms are restless. A band's state evolves whether or not it is observed,
    so information decays; UCB1's confidence radius shrinks monotonically with
    visit count and never re-widens, so it becomes progressively MORE certain
    about a quantity that is progressively MORE stale.

The second point is the sharper failure. UCB1's exploration term is
sqrt(2 ln t / n_b), which depends only on how often band b was sampled, never
on how long ago. After a few hundred slots it commits to whichever bands looked
best early and stops revisiting the rest, even though those bands may now hold
newly-arrived emitters.

UCB1 is therefore the mandatory *stationary* control in Gate 2: it shows that
generic bandit machinery is not sufficient, and that the non-stationary
machinery (sliding window, discounting, periodicity) earns its complexity.
Reporting UCB1 losing to a discounted variant is a positive scientific result.

UCB1 makes no forecast, so `predict()` is not overridden and PS metrics 6 and 7
are reported as unavailable for it rather than as zero.
"""

from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np

from ...core.scheduler_interface import BaseScheduler


class UCB1Scheduler(BaseScheduler):
    """
    Classic UCB1 over bands, reward = 1 if the dwell registered a hit.

    [ENGINEERING-ASSUMPTION] c defaults to sqrt(2), the value for which UCB1's
    regret bound is stated in Auer et al. (2002). It is NOT tuned against the
    evaluation scenarios: doing so would make the "stationary bandit is
    insufficient" conclusion unfalsifiable by construction.
    """

    def __init__(self, band_count: int, c: float = float(np.sqrt(2.0))):
        super().__init__(
            band_count,
            "[BANDIT] UCB1 (Auer, Cesa-Bianchi & Fischer 2002). Stationary-reward "
            "control condition; assumptions knowingly violated by restless, "
            "non-stationary emitter activity.",
        )
        self.c = float(c)
        self.counts = np.zeros(band_count, dtype=np.int64)
        self.values = np.zeros(band_count, dtype=float)
        self.total_count = 0

    def reset(self, seed: int, scenario_config: Dict) -> None:
        self.counts = np.zeros(self.band_count, dtype=np.int64)
        self.values = np.zeros(self.band_count, dtype=float)
        self.total_count = 0
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0

    def select_action(self, observation_history: List[Dict],
                      current_time_slot: int) -> int:
        self._check_truth_leakage(observation_history)

        # Initialisation phase: every arm once, in fixed order. Deterministic so
        # that the paired comparison is not perturbed by tie-break randomness.
        unvisited = np.flatnonzero(self.counts == 0)
        if unvisited.size:
            return self._validate_action(int(unvisited[0]))

        radius = self.c * np.sqrt(np.log(max(1, self.total_count)) / self.counts)
        return self._validate_action(int(np.argmax(self.values + radius)))

    def update(self, action: int, observation: Dict) -> None:
        self.observation_history.append(observation)
        reward = 1.0 if observation["hit"] else 0.0
        self.counts[action] += 1
        self.total_count += 1
        # Incremental sample mean.
        self.values[action] += (reward - self.values[action]) / self.counts[action]
        self.current_band = action
        self.steps_taken += 1

    def get_diagnostics(self) -> Dict:
        d = super().get_diagnostics()
        with np.errstate(divide="ignore", invalid="ignore"):
            radius = self.c * np.sqrt(np.log(max(1, self.total_count)) / np.maximum(self.counts, 1))
        d.update({
            "c": self.c,
            "visit_counts": self.counts.tolist(),
            "value_estimates": np.round(self.values, 4).tolist(),
            "confidence_radius": np.round(radius, 4).tolist(),
            # Evidence for the write-up: the radius collapses even though the
            # underlying quantity is still drifting.
            "mean_confidence_radius": float(radius.mean()),
            "bands_starved": int((self.counts < max(1, self.total_count // (4 * self.band_count))).sum()),
        })
        return d


class SlidingWindowUCBScheduler(BaseScheduler):
    """
    SW-UCB: UCB computed over a trailing window of outcomes.

    [BANDIT] [LITERATURE-GROUNDED] Garivier & Moulines (2011), "On upper-
    confidence bound policies for switching bandit problems". The minimal,
    principled repair of UCB1 for non-stationarity: forget observations older
    than `window`, so both the mean and the confidence radius track drift.

    Included so that the comparison against the periodicity-aware method is
    fair. If a strong non-stationary bandit already matches the proposed
    scheduler, the extra periodicity machinery has not earned its complexity —
    which is exactly what Gate 3 is designed to detect.
    """

    def __init__(self, band_count: int, window: int = 100,
                 c: float = float(np.sqrt(2.0))):
        super().__init__(
            band_count,
            "[BANDIT] Sliding-Window UCB (Garivier & Moulines 2011). "
            "Non-stationary control; strong baseline for Gate 3.",
        )
        self.window = int(window)
        self.c = float(c)
        self._actions: List[int] = []
        self._rewards: List[float] = []

    def reset(self, seed: int, scenario_config: Dict) -> None:
        self._actions = []
        self._rewards = []
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0

    def _windowed(self) -> tuple:
        a = np.asarray(self._actions[-self.window:], dtype=np.intp)
        r = np.asarray(self._rewards[-self.window:], dtype=float)
        counts = np.bincount(a, minlength=self.band_count).astype(np.int64) if a.size else \
            np.zeros(self.band_count, dtype=np.int64)
        sums = np.bincount(a, weights=r, minlength=self.band_count) if a.size else \
            np.zeros(self.band_count, dtype=float)
        return counts, sums

    def select_action(self, observation_history: List[Dict],
                      current_time_slot: int) -> int:
        self._check_truth_leakage(observation_history)
        counts, sums = self._windowed()
        unvisited = np.flatnonzero(counts == 0)
        if unvisited.size:
            return self._validate_action(int(unvisited[0]))
        means = sums / counts
        n_eff = float(counts.sum())
        radius = self.c * np.sqrt(np.log(max(1.0, n_eff)) / counts)
        return self._validate_action(int(np.argmax(means + radius)))

    def update(self, action: int, observation: Dict) -> None:
        self.observation_history.append(observation)
        self._actions.append(int(action))
        self._rewards.append(1.0 if observation["hit"] else 0.0)
        # Bound memory: nothing outside the window can ever be read again.
        if len(self._actions) > self.window:
            del self._actions[:-self.window]
            del self._rewards[:-self.window]
        self.current_band = action
        self.steps_taken += 1

    def get_diagnostics(self) -> Dict:
        d = super().get_diagnostics()
        counts, sums = self._windowed()
        d.update({
            "window": self.window,
            "windowed_visit_counts": counts.tolist(),
            "windowed_means": np.round(np.divide(sums, np.maximum(counts, 1)), 4).tolist(),
        })
        return d

"""Classical UCB1 for the Mode-B scalar-reward interface.

UCB1 is an online baseline here, not a pretrained policy. Counts reset for
each independent world. Within a world it updates from the environment's v2
scalar reward, affinely scaled from the declared bounded interval to [0, 1].

The stationarity and equal-duration assumptions of classical UCB1 do not hold
for restless emitters and band-dependent dwell times. This implementation is
kept as a transparent stationary-bandit control, not as a claim that UCB1 is
the best-matched algorithm for this environment.
"""
from __future__ import annotations

import math

import numpy as np

from ..algorithm_interface import PublicTransition


class ClassicalUCB1:
    """UCB1 with deterministic first-visit and tie-breaking order."""

    def __init__(self, bands: int, seed: int, *, reward_bounds=(-0.01, 1.0),
                 exploration_coefficient: float = math.sqrt(2.0)):
        if type(bands) is not int or bands <= 0:
            raise ValueError("bands must be a positive integer")
        if len(reward_bounds) != 2:
            raise ValueError("reward_bounds must contain [minimum, maximum]")
        self.bands = bands
        self.reward_low, self.reward_high = map(float, reward_bounds)
        if (not math.isfinite(self.reward_low) or not math.isfinite(self.reward_high)
                or self.reward_low >= self.reward_high):
            raise ValueError("reward_bounds must be finite and increasing")
        self.c = float(exploration_coefficient)
        if not math.isfinite(self.c) or self.c <= 0:
            raise ValueError("exploration_coefficient must be finite and positive")
        self.seed = int(seed)  # retained in provenance; tie-breaking is deterministic
        self.reset_episode(training=False)

    def reset_episode(self, *, training: bool) -> None:
        # UCB1 learns online within every world, including evaluation worlds.
        # It never carries statistics from one world to another.
        self.counts = np.zeros(self.bands, dtype=np.int64)
        self.reward_sums = np.zeros(self.bands, dtype=np.float64)
        self.raw_reward_sums = np.zeros(self.bands, dtype=np.float64)
        self.total_pulls = 0

    def select_action(self, state, *, training: bool) -> int:
        unvisited = np.flatnonzero(self.counts == 0)
        if unvisited.size:
            return int(unvisited[0])
        t = self.total_pulls + 1
        means = self.reward_sums / self.counts
        bonus = self.c * np.sqrt(np.log(float(t)) / self.counts)
        # numpy.argmax resolves ties at the smallest band index.
        return int(np.argmax(means + bonus))

    def observe(self, transition: PublicTransition, *, training: bool) -> None:
        action = transition.action
        if not 0 <= action < self.bands:
            raise ValueError(f"Invalid action {action}; expected 0..{self.bands - 1}")
        raw = float(transition.reward)
        if not math.isfinite(raw) or raw < self.reward_low or raw > self.reward_high:
            raise ValueError(
                "Environment reward is outside the frozen UCB1 scaling interval "
                f"[{self.reward_low}, {self.reward_high}]: {raw}"
            )
        scaled = (raw - self.reward_low) / (self.reward_high - self.reward_low)
        self.counts[action] += 1
        self.reward_sums[action] += scaled
        self.raw_reward_sums[action] += raw
        self.total_pulls += 1

    def end_episode(self, *, training: bool) -> dict:
        means = np.divide(self.reward_sums, self.counts,
                          out=np.zeros(self.bands, dtype=float), where=self.counts > 0)
        raw_means = np.divide(self.raw_reward_sums, self.counts,
                              out=np.zeros(self.bands, dtype=float), where=self.counts > 0)
        if self.total_pulls:
            bonuses = np.zeros(self.bands, dtype=float)
            visited = self.counts > 0
            bonuses[visited] = self.c * np.sqrt(
                np.log(float(self.total_pulls + 1)) / self.counts[visited]
            )
            indices = means + bonuses
        else:
            indices = np.zeros(self.bands, dtype=float)
        return {
            "algorithm": "ucb1",
            "updates_during_evaluation": True,
            "statistics_reset_each_world": True,
            "total_decisions": int(self.total_pulls),
            "band_selection_counts": self.counts.tolist(),
            "normalized_reward_means": means.tolist(),
            "raw_reward_means": raw_means.tolist(),
            "final_ucb_indices": indices.tolist(),
            "exploration_coefficient": self.c,
            "reward_bounds": [self.reward_low, self.reward_high],
        }

def create(*, bands: int, seed: int, settings: dict, checkpoint=None):
    if checkpoint is not None:
        raise ValueError("UCB1 is an online baseline and does not load checkpoints")
    bounds = settings.get("reward_bounds", [-0.01, 1.0])
    coefficient = settings.get("exploration_coefficient", math.sqrt(2.0))
    policy = ClassicalUCB1(bands, seed, reward_bounds=bounds,
                          exploration_coefficient=coefficient)
    return policy

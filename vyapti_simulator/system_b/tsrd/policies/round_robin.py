"""Deterministic cyclic band-selection baseline for the Mode-B runner."""
from __future__ import annotations

from ..algorithm_interface import PublicTransition


class RoundRobinPolicy:
    """Visit bands in order, wrapping to band zero after the last band."""

    def __init__(self, bands: int, seed: int, *, start_band: int = 0):
        if type(bands) is not int or bands < 1:
            raise ValueError("bands must be a positive integer")
        if type(start_band) is not int or not 0 <= start_band < bands:
            raise ValueError("start_band must be a valid band index")
        self.bands = bands
        self.seed = int(seed)  # recorded for provenance; the sequence is deterministic
        self.start_band = start_band
        self.reset_episode(training=False)

    def reset_episode(self, *, training: bool) -> None:
        self.next_band = self.start_band
        self.band_selection_counts = [0] * self.bands
        self.total_decisions = 0
        self.total_reward = 0.0
        self._awaiting_observation = False
        self._last_action = None

    def select_action(self, state, *, training: bool) -> int:
        if self._awaiting_observation:
            raise RuntimeError("observe must follow each Round Robin action")
        action = self.next_band
        self.next_band = (self.next_band + 1) % self.bands
        self.band_selection_counts[action] += 1
        self.total_decisions += 1
        self._awaiting_observation = True
        self._last_action = action
        return action

    def observe(self, transition: PublicTransition, *, training: bool) -> None:
        if not self._awaiting_observation:
            raise RuntimeError("select_action must precede observe for each transition")
        if transition.action != self._last_action:
            raise ValueError("Transition action does not match the last Round Robin selection")
        self.total_reward += float(transition.reward)
        self._awaiting_observation = False
        self._last_action = None

    def end_episode(self, *, training: bool) -> dict:
        if self._awaiting_observation:
            raise RuntimeError("Cannot finish a world before observing its last action")
        return {
            "algorithm": "round_robin",
            "updates_during_evaluation": False,
            "statistics_reset_each_world": True,
            "total_decisions": self.total_decisions,
            "band_selection_counts": list(self.band_selection_counts),
            "total_reward": self.total_reward,
            "start_band": self.start_band,
        }


def create(*, bands: int, seed: int, settings: dict, checkpoint=None):
    if checkpoint is not None:
        raise ValueError("Round Robin is a fixed online baseline and has no checkpoints")
    return RoundRobinPolicy(bands, seed, start_band=settings.get("start_band", 0))

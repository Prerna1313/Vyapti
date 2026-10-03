"""Small trainable adapter used only by generic runner integration tests."""
from __future__ import annotations

import json
from pathlib import Path


class RunnerFixture:
    def __init__(self, bands: int, checkpoint=None):
        self.bands = bands
        self.steps = 0
        if checkpoint is not None:
            self.steps = json.loads(Path(checkpoint).read_text(encoding="utf-8"))["steps"]

    def reset_episode(self, *, training: bool):
        self.episode_counts = [0] * self.bands
        self.decisions = 0

    def select_action(self, state, *, training: bool) -> int:
        return self.decisions % self.bands

    def observe(self, transition, *, training: bool):
        self.steps += 1
        self.episode_counts[transition.action] += 1
        self.decisions += 1

    def end_episode(self, *, training: bool):
        return {"band_selection_counts": list(self.episode_counts)}

    def save(self, path: Path):
        Path(path).write_text(json.dumps({"steps": self.steps}), encoding="utf-8")


def create(*, bands: int, seed: int, settings: dict, checkpoint=None):
    return RunnerFixture(bands, checkpoint)

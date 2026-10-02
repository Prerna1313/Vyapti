"""Small trainable baseline for the recorded-PDW experiment interface.

The persisted counts are learned only from receiver hits in TRAIN worlds.
Each evaluation world gets its own transient online counts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class Policy:
    def __init__(self, bands: int, seed: int, prior_hits=None, prior_looks=None):
        self.bands = bands
        self.rng = np.random.default_rng(seed)
        self.prior_hits = np.zeros(bands) if prior_hits is None else np.asarray(prior_hits, dtype=float)
        self.prior_looks = np.zeros(bands) if prior_looks is None else np.asarray(prior_looks, dtype=float)
        if self.prior_hits.shape != (bands,) or self.prior_looks.shape != (bands,):
            raise ValueError("Checkpoint band count differs from receiver")
        self.reset_episode()

    def reset_episode(self):
        self.hits = np.zeros(self.bands)
        self.looks = np.zeros(self.bands)

    def select_action(self, history, time_slot):
        looks = self.looks + self.prior_looks
        hits = self.hits + self.prior_hits
        # A unit beta prior prevents a frozen arm from dominating forever.
        total = max(2.0, float(looks.sum()))
        score = (hits + 1) / (looks + 2) + np.sqrt(2 * math.log(total) / (looks + 1))
        candidates = np.flatnonzero(score == score.max())
        return int(self.rng.choice(candidates))

    def observe(self, observation):
        band = int(observation["selected_band"])
        self.looks[band] += 1
        self.hits[band] += bool(observation["hit"])

    def finish_training_episode(self):
        self.prior_looks += self.looks
        self.prior_hits += self.hits
        self.reset_episode()

    def save(self, path: Path):
        path.write_text(json.dumps({
            "schema": "ucb_prior_v1", "bands": self.bands,
            "prior_hits": self.prior_hits.tolist(),
            "prior_looks": self.prior_looks.tolist(),
        }, indent=2), encoding="utf-8")


def create(bands: int, seed: int, checkpoint: Path | None = None):
    if checkpoint is None:
        return Policy(bands, seed)
    payload = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
    if payload.get("schema") != "ucb_prior_v1" or payload.get("bands") != bands:
        raise ValueError("Incompatible UCB checkpoint")
    return Policy(bands, seed, payload["prior_hits"], payload["prior_looks"])

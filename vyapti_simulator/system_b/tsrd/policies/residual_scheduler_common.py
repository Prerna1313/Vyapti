"""Compatibility state used by the imported residual-SAC learner.

This module only supplies the causal belief/periodicity interface expected by
the learner.  It uses the existing TRAIN-250 HMM and observed-HIT recurrence
implementation; it does not expose hidden world state.
"""
from __future__ import annotations

import numpy as np

from .contextual_thompson import CausalPeriodicity, TwoStateHMMBelief


class _CausalBase:
    def __init__(self, transition: np.ndarray, prior_active: np.ndarray):
        self.belief = TwoStateHMMBelief(
            transition=transition,
            pd=0.90,
            pfa=0.05,
            prior_active=prior_active,
        )
        self.periodicity = CausalPeriodicity(max_events=32, tolerance_fraction=0.15)
        self.elapsed_slots = 0

    def reset(self, prior_active: np.ndarray) -> None:
        self.belief.prior = np.asarray(prior_active, dtype=np.float64).copy()
        self.belief.reset()
        self.periodicity.reset()
        self.elapsed_slots = 0

    def pre_action_features(self) -> dict[str, np.ndarray]:
        score, confidence = self.periodicity.features(self.elapsed_slots)
        return {"periodicity": score, "periodicity_confidence": confidence}

    def step(self, band: int, dwell: int, positives) -> None:
        if len(positives) != dwell or dwell not in (1, 2):
            raise ValueError("Causal state needs one or two receiver look outcomes")
        for positive in positives:
            # Advance every band by one base slot, then condition the selected
            # band's posterior on its actual per-slot detector result.
            self.belief.update_aggregate(int(band), bool(positive), 1)
            self.elapsed_slots += 1
            self.periodicity.observe(int(band), self.elapsed_slots, bool(positive))


class CausalSchedulerState:
    """Factory matching the uploaded learner's expected causal-state API."""

    @staticmethod
    def create(transition: np.ndarray, prior_active: np.ndarray) -> _CausalBase:
        return _CausalBase(transition, prior_active)


N_BANDS = 36
PD = 0.90
PFA = 0.05
PeriodicityTracker = CausalPeriodicity

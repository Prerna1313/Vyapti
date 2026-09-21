#!/usr/bin/env python3
"""
Vyapti Hybrid-Aware Meta-UCB Bandit Combiner (Local Version).

This is a standalone hybrid module for local use. It does NOT overwrite or
modify the frozen vyapti_meta_ucb_combiner_frozen.py baseline artifact.

It preserves the empirical Meta-UCB/Cutkosky-style expert combiner structure
and its V3 empirical-parameter requirement, while adding one context-aware
expert. The hybrid context is supplied by HybridMetaScheduler:
- belief: p_active, uncertainty, staleness, reward_mean
- periodic_opportunity
- markov_next_probs
- volatility
- governor_scores

Important: This remains an empirical adaptation. It must not be described as a
proof that Cutkosky et al.'s theoretical conditions hold for the custom experts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from vyapti_simulator.core.scheduler_interface import BaseScheduler


# Re-export constants so the evaluation script can import from here.
ALPHA_USED = 0.5
DELTA = 0.05
R_EMPIRICAL_SCALE = 1.0
BAND_COUNT = 36
HORIZON = 600


class HybridContextExpert(BaseScheduler):
    """Context-aware expert used as an additional arm of Hybrid Meta-UCB."""

    def __init__(
        self,
        band_count: int,
        belief_weight: float = 0.35,
        periodic_weight: float = 0.25,
        markov_weight: float = 0.20,
        uncertainty_weight: float = 0.10,
        governor_weight: float = 0.10,
        volatility_penalty: float = 0.08,
    ):
        super().__init__(band_count, provenance_note="Hybrid Context Expert")
        self.band_count = band_count
        self.belief_weight = belief_weight
        self.periodic_weight = periodic_weight
        self.markov_weight = markov_weight
        self.uncertainty_weight = uncertainty_weight
        self.governor_weight = governor_weight
        self.volatility_penalty = volatility_penalty
        self.counts = np.zeros(band_count, dtype=np.int64)
        self.rewards = np.zeros(band_count, dtype=float)
        self.local_t = 0
        self.rng = np.random.default_rng()

    def reset(self, seed: int, scenario_config: Optional[dict] = None):
        self.rng = np.random.default_rng(seed)
        self.counts.fill(0)
        self.rewards.fill(0.0)
        self.local_t = 0

    def _context(self, observation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not observation_history:
            return {}
        last = observation_history[-1]
        return last.get("hybrid_context", {}) if isinstance(last, dict) else {}

    def select_action(
        self,
        observation_history: List[Dict[str, Any]],
        current_time_slot: int,
    ) -> int:
        unvisited = np.flatnonzero(self.counts == 0)
        if len(unvisited):
            return int(unvisited[0])

        ctx = self._context(observation_history)
        belief = ctx.get("belief", {})

        p_active = np.asarray(
            belief.get("p_active", np.full(self.band_count, 0.5)), dtype=float
        )
        uncertainty = np.asarray(
            belief.get("uncertainty", np.full(self.band_count, 0.5)), dtype=float
        )
        periodic = np.asarray(
            ctx.get("periodic_opportunity", np.zeros(self.band_count)), dtype=float
        )
        markov = np.asarray(
            ctx.get("markov_next_probs", np.full(self.band_count, 1.0 / self.band_count)),
            dtype=float,
        )
        volatility = np.asarray(
            ctx.get("volatility", np.zeros(self.band_count)), dtype=float
        )
        governor = np.asarray(
            ctx.get("governor_scores", np.zeros(self.band_count)), dtype=float
        )

        # Defensive shape handling: a malformed context must not break an episode.
        def vector(x: np.ndarray, default: float) -> np.ndarray:
            if x.shape != (self.band_count,):
                return np.full(self.band_count, default, dtype=float)
            return np.nan_to_num(x, nan=default, posinf=default, neginf=default)

        p_active = vector(p_active, 0.5)
        uncertainty = vector(uncertainty, 0.5)
        periodic = vector(periodic, 0.0)
        markov = vector(markov, 1.0 / self.band_count)
        volatility = vector(volatility, 0.0)
        governor = vector(governor, 0.0)

        empirical_mean = self.rewards / np.maximum(self.counts, 1)
        exploration = np.sqrt(2.0 * np.log(self.local_t + 1.0) / self.counts)

        scores = (
            empirical_mean
            + exploration
            + self.belief_weight * p_active
            + self.periodic_weight * periodic
            + self.markov_weight * markov
            + self.uncertainty_weight * uncertainty
            + self.governor_weight * governor
            - self.volatility_penalty * volatility
        )
        scores += self.rng.uniform(0.0, 1e-12, self.band_count)
        return int(np.argmax(scores))

    def update(self, action: int, observation: Dict[str, Any]):
        self.counts[action] += 1
        self.rewards[action] += float(bool(observation.get("hit", False)))
        self.local_t += 1


class HybridMetaUCBBanditCombiner(BaseScheduler):
    """
    Meta-UCB combiner with the frozen baseline experts plus Hybrid-Context.

    The V3 file was originally produced for four frozen experts. Therefore this
    class uses the existing V3 parameter for ARP as an explicitly declared
    conservative proxy for Hybrid-Context's empirical C(alpha). The output JSON
    records this proxy decision so it is auditable.
    """

    EXPERT_CLASSES = {
        "UCB1": "UCB1",
        "Whittle-Inspired": "Whittle-Inspired",
        "RLessUCB": "RLessUCB",
        "ARP": "ARP",
        "Hybrid-Context": HybridContextExpert,
    }

    def __init__(
        self,
        band_count: int,
        expert_params: Dict[str, Any],
        horizon: int,
        alpha_used: float = ALPHA_USED,
        delta: float = DELTA,
        r_empirical_scale: float = R_EMPIRICAL_SCALE,
        hybrid_c_proxy_expert: str = "ARP",
    ):
        super().__init__(band_count, provenance_note="Hybrid Meta-UCB Empirical Combiner")
        self.band_count = band_count
        self.horizon = horizon
        self.alpha_used = alpha_used
        self.delta = delta
        self.r_empirical_scale = r_empirical_scale
        self.hybrid_c_proxy_expert = hybrid_c_proxy_expert
        self.empirical_params = expert_params

        # Import frozen baseline experts dynamically to avoid circular imports.
        from vyapti_meta_ucb_combiner_frozen import (
            UCB1Scheduler,
            WhittleInspiredScheduler,
            RLessUCBScheduler,
            ARPScheduler,
        )

        frozen_experts = {
            "UCB1": UCB1Scheduler,
            "Whittle-Inspired": WhittleInspiredScheduler,
            "RLessUCB": RLessUCBScheduler,
            "ARP": ARPScheduler,
        }

        self.expert_names = list(self.EXPERT_CLASSES.keys())
        self.experts: Dict[str, BaseScheduler] = {}
        for name in self.expert_names:
            klass = self.EXPERT_CLASSES[name]
            if isinstance(klass, str):
                klass = frozen_experts[klass]
            self.experts[name] = klass(band_count=band_count)

        self.meta_counts = {name: 0 for name in self.expert_names}
        self.meta_rewards = {name: 0.0 for name in self.expert_names}
        self.deviation_sums = {name: 0.0 for name in self.expert_names}
        self.prior_means = {name: 0.0 for name in self.expert_names}
        self.eliminated = {name: False for name in self.expert_names}
        self.elimination_history: List[Dict[str, Any]] = []
        self.selected_expert: Optional[str] = None
        self.local_t = 0
        self.rng = np.random.default_rng()
        self.log_term = float(np.log(self.horizon ** 3 * len(self.expert_names) / self.delta))

        # Validate all frozen-expert entries are present at the chosen alpha.
        for name in ("UCB1", "Whittle-Inspired", "RLessUCB", "ARP"):
            _ = self.get_empirical_c(name)
        # Validate declared proxy exists as well.
        _ = self.get_empirical_c(self.hybrid_c_proxy_expert)

        self.R = {
            name: self.r_empirical_scale * self.get_empirical_c(name) * (self.horizon ** self.alpha_used)
            for name in self.expert_names
        }

    def reset(self, seed: int, scenario_config: Optional[dict] = None):
        self.rng = np.random.default_rng(seed)
        for name in self.expert_names:
            self.experts[name].reset(seed=seed, scenario_config=scenario_config)
            self.meta_counts[name] = 0
            self.meta_rewards[name] = 0.0
            self.deviation_sums[name] = 0.0
            self.prior_means[name] = 0.0
            self.eliminated[name] = False
        self.elimination_history = []
        self.selected_expert = None
        self.local_t = 0

    def get_empirical_c(self, expert_name: str) -> float:
        lookup_name = self.hybrid_c_proxy_expert if expert_name == "Hybrid-Context" else expert_name
        expert_data = self.empirical_params.get(lookup_name, {})
        candidates = expert_data.get("envelope_candidates", [])
        matches = [
            candidate for candidate in candidates
            if np.isclose(float(candidate.get("alpha", np.nan)), self.alpha_used)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one empirical C(alpha={self.alpha_used}) for {lookup_name}; "
                f"found {len(matches)}"
            )
        c_value = float(matches[0].get("empirical_C", np.nan))
        if not np.isfinite(c_value) or c_value <= 0:
            raise ValueError(f"Invalid empirical C for {lookup_name}: {c_value}")
        return c_value

    def compute_meta_score(self, expert_name: str) -> float:
        n = self.meta_counts[expert_name]
        if n == 0:
            return float("inf")
        mean_reward = self.meta_rewards[expert_name] / n
        c_value = self.get_empirical_c(expert_name)
        confidence = c_value * (n ** (self.alpha_used - 1.0)) * np.sqrt(8.0 * self.log_term / n)
        confidence = min(1.0, float(confidence))
        return float(mean_reward + confidence - self.R[expert_name] / self.horizon)

    def select_action(
        self,
        observation_history: List[Dict[str, Any]],
        current_time_slot: int,
    ) -> int:
        active = [name for name in self.expert_names if not self.eliminated[name]]
        if not active:
            raise RuntimeError("Hybrid Meta-UCB eliminated all experts.")
        if len(active) == 1:
            selected = active[0]
        else:
            scores = {name: self.compute_meta_score(name) for name in active}
            selected = max(scores, key=scores.get)
        self.selected_expert = selected
        return int(self.experts[selected].select_action(observation_history, current_time_slot))

    def update(self, action: int, observation: Dict[str, Any]):
        if self.selected_expert is None:
            raise RuntimeError("update() called before select_action().")
        selected = self.selected_expert
        reward = float(bool(observation.get("hit", False)))
        n_before = self.meta_counts[selected]
        prior_mean = self.meta_rewards[selected] / n_before if n_before > 0 else 0.0

        self.experts[selected].update(action, observation)
        self.deviation_sums[selected] += prior_mean - reward
        self.meta_counts[selected] += 1
        self.meta_rewards[selected] += reward
        n = self.meta_counts[selected]
        self.prior_means[selected] = self.meta_rewards[selected] / n

        c_value = self.get_empirical_c(selected)
        threshold = c_value * (n ** self.alpha_used) * 3.0 * np.sqrt(self.log_term / n)
        if self.deviation_sums[selected] > threshold and not self.eliminated[selected]:
            active_count = sum(not value for value in self.eliminated.values())
            if active_count > 1:
                self.eliminated[selected] = True
                self.elimination_history.append({
                    "expert_name": selected,
                    "global_round": int(self.local_t),
                    "meta_count": int(n),
                    "deviation_sum": float(self.deviation_sums[selected]),
                    "threshold": float(threshold),
                })
        self.local_t += 1
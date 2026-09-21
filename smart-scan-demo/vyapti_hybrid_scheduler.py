#!/usr/bin/env python3
"""
Vyapti Hybrid Scheduler — Meta-UCB core with belief, predictors, context, and soft governor.

This implements the final hybrid architecture:
- Layer 1: Bayesian belief state (P(active), uncertainty, staleness, reward history)
- Layer 2: Periodicity predictor (with next-arrival opportunity), Markov frequency predictor (emitter transitions), Volatility detector
- Layer 3: Behaviour context classifier (diagnostic only; not used in decision path)
- Layer 4: Hybrid-aware Meta-UCB bandit combiner (core action selector), now receiving hybrid context
- Layer 5: Soft action governor (preferences only, no hard overrides)

Interface:
- Implements BaseScheduler: select_action(observation_history, current_time_slot)
- Compatible with run_paired_episodes and TSRDMetricsEngine.record_result
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from vyapti_simulator.core.scheduler_interface import BaseScheduler
from vyapti_hybrid_meta_ucb_combiner import HybridMetaUCBBanditCombiner, ALPHA_USED, DELTA, R_EMPIRICAL_SCALE, BAND_COUNT, HORIZON


# =============================================================================
# Layer 1: Belief state
# =============================================================================

@dataclass
class BandBelief:
    p_active: float = 0.5
    uncertainty: float = 0.5
    staleness: int = 0
    reward_mean: float = 0.0


class BeliefState:
    def __init__(self, band_count: int):
        self.band_count = band_count
        self.bands = [BandBelief() for _ in range(band_count)]

    def reset(self):
        for b in self.bands:
            b.p_active = 0.5
            b.uncertainty = 0.5
            b.staleness = 0
            b.reward_mean = 0.0

    def update_after_step(self, band: int, hit: bool):
        for idx, b in enumerate(self.bands):
            b.staleness += 0 if idx == band else 1

        pd_assumed, pfa_assumed = 0.9, 0.01
        b = self.bands[band]
        prior = b.p_active

        if hit:
            lh1, lh0 = pd_assumed, pfa_assumed
        else:
            lh1, lh0 = 1.0 - pd_assumed, 1.0 - pfa_assumed

        num = lh1 * prior
        den = num + lh0 * (1.0 - prior)
        posterior = num / den if den > 1e-12 else prior
        b.p_active = float(np.clip(posterior, 0.01, 0.99))

        r = 1.0 if hit else 0.0
        b.reward_mean = 0.1 * r + 0.9 * b.reward_mean
        b.uncertainty = float(1.0 - abs(b.p_active - 0.5) * 2.0)


# =============================================================================
# Layer 2: Predictors
# =============================================================================

class PeriodicityPredictor:
    def __init__(self, band_count: int):
        self.band_count = band_count
        self.hit_times: List[List[int]] = [[] for _ in range(band_count)]

    def reset(self):
        self.hit_times = [[] for _ in range(self.band_count)]

    def update(self, band: int, t: int, hit: bool):
        if hit:
            self.hit_times[band].append(t)

    def periodic_confidence(self, band: int) -> float:
        times = self.hit_times[band]
        if len(times) < 3:
            return 0.0
        intervals = np.diff(times)
        if intervals.size == 0:
            return 0.0
        std = float(np.std(intervals))
        mean = float(np.mean(intervals))
        if mean <= 0.0:
            return 0.0
        score = max(0.0, 1.0 - std / (mean + 1e-6))
        return float(np.clip(score, 0.0, 1.0))

    def estimate_period(self, band: int) -> Optional[float]:
        times = self.hit_times[band]
        if len(times) < 3:
            return None
        intervals = np.diff(times)
        return float(np.mean(intervals))

    def predict_next_arrival(self, band: int, current_time: int) -> Optional[int]:
        times = self.hit_times[band]
        if len(times) < 2:
            return None
        period = self.estimate_period(band)
        if period is None or period <= 0:
            return None
        last_hit = times[-1]

        if current_time < last_hit:
            next_hit = last_hit
        else:
            k = int(np.ceil((current_time - last_hit) / period))
            next_hit = int(last_hit + k * period)

        return next_hit

    def periodic_opportunity(self, band: int, current_time: int) -> float:
        times = self.hit_times[band]
        if len(times) < 2:
            return 0.0
        period = self.estimate_period(band)
        if period is None or period <= 0:
            return 0.0
        conf = self.periodic_confidence(band)
        next_hit = self.predict_next_arrival(band, current_time)
        if next_hit is None:
            return 0.0
        dt = abs(next_hit - current_time)
        sigma = max(1.0, period * 0.2)
        opp = float(np.exp(-0.5 * (dt / sigma) ** 2))
        return float(conf * opp)


class MarkovFrequencyPredictor:
    def __init__(self, band_count: int, alpha: float = 1.0):
        self.band_count = band_count
        self.alpha = alpha
        self.counts = np.zeros((band_count, band_count), dtype=np.int64)
        self.last_observed_emitter_band: Optional[int] = None

    def reset(self):
        self.counts.fill(0)
        self.last_observed_emitter_band = None

    def update_observed_emitter_band(self, band: int):
        if self.last_observed_emitter_band is not None:
            i = self.last_observed_emitter_band
            self.counts[i, band] += 1
        self.last_observed_emitter_band = band

    def next_band_probabilities(self) -> np.ndarray:
        if self.last_observed_emitter_band is None:
            return np.full(self.band_count, 1.0 / self.band_count)

        i = self.last_observed_emitter_band
        row = self.counts[i].astype(float)
        row_smooth = row + self.alpha
        total_smooth = row_smooth.sum()
        if total_smooth <= 0:
            return np.full(self.band_count, 1.0 / self.band_count)
        return row_smooth / total_smooth

    def transition_confidence(self, band: int) -> float:
        row = self.counts[band]
        total = int(row.sum())
        if total == 0:
            return 0.0
        probs = row / float(total)
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        max_entropy = np.log(self.band_count)
        return float(1.0 - entropy / (max_entropy + 1e-6))


class VolatilityDetector:
    def __init__(self, band_count: int):
        self.band_count = band_count
        self.hit_history = [[] for _ in range(band_count)]

    def reset(self):
        self.hit_history = [[] for _ in range(self.band_count)]

    def update(self, band: int, hit: bool):
        h = self.hit_history[band]
        h.append(1.0 if hit else 0.0)
        if len(h) > 100:
            del h[: len(h) - 100]

    def volatility(self, band: int) -> float:
        h = self.hit_history[band]
        if len(h) < 10:
            return 0.0
        recent = np.array(h[-20:], dtype=float)
        return float(np.clip(np.std(recent), 0.0, 1.0))


# =============================================================================
# Layer 3: Behaviour context classifier (diagnostic only)
# =============================================================================

@dataclass
class BehaviourContext:
    band_type: str
    periodic_conf: float
    transition_conf: float
    volatility: float


class BehaviourClassifier:
    def classify(self, band: int, bel: BandBelief,
                 periodic: PeriodicityPredictor,
                 markov: MarkovFrequencyPredictor,
                 vol: VolatilityDetector) -> BehaviourContext:
        pc = periodic.periodic_confidence(band)
        tc = markov.transition_confidence(band)
        v = vol.volatility(band)

        if pc > 0.7 and v < 0.3:
            band_type = "periodic"
        elif tc > 0.5 and v > 0.4:
            band_type = "agile"
        elif bel.p_active > 0.6 and bel.uncertainty < 0.4 and v < 0.3:
            band_type = "static"
        elif v > 0.5 and tc < 0.4:
            band_type = "mixed"
        else:
            band_type = "unknown"

        return BehaviourContext(
            band_type=band_type,
            periodic_conf=pc,
            transition_conf=tc,
            volatility=v,
        )


# =============================================================================
# Layer 5: Soft action governor (preferences only)
# =============================================================================

class SoftActionGovernor:
    def __init__(self, band_count: int, max_staleness: int = 200):
        self.band_count = band_count
        self.max_staleness = max_staleness

    def governor_scores(self, belief: BeliefState) -> np.ndarray:
        stalenesses = np.array([b.staleness for b in belief.bands], dtype=float)
        p_active = np.array([b.p_active for b in belief.bands], dtype=float)
        score = stalenesses * (p_active > 0.6).astype(float)
        if score.max() > score.min():
            score = (score - score.min()) / (score.max() - score.min() + 1e-12)
        return score


# =============================================================================
# Hybrid scheduler
# =============================================================================

class HybridMetaScheduler(BaseScheduler):
    def __init__(self, band_count: int,
                 expert_params: Dict[str, Any],
                 horizon: int = HORIZON):
        super().__init__(band_count, provenance_note="Hybrid Meta-UCB Scheduler")

        self.band_count = band_count
        self.horizon = horizon

        self.meta = HybridMetaUCBBanditCombiner(
            band_count=band_count,
            expert_params=expert_params,
            horizon=horizon,
            alpha_used=ALPHA_USED,
            delta=DELTA,
            r_empirical_scale=R_EMPIRICAL_SCALE,
            hybrid_c_proxy_expert="ARP",
        )

        self.belief = BeliefState(band_count)
        self.periodic = PeriodicityPredictor(band_count)
        self.markov = MarkovFrequencyPredictor(band_count)
        self.vol = VolatilityDetector(band_count)
        self.behaviour_classifier = BehaviourClassifier()
        self.governor = SoftActionGovernor(band_count)

    def reset(self, seed: int, scenario_config: Optional[Dict[str, Any]] = None):
        rng = np.random.default_rng(seed)
        _ = rng

        self.meta.reset(seed=seed, scenario_config=scenario_config)
        self.belief.reset()
        self.periodic.reset()
        self.markov.reset()
        self.vol.reset()
        self.governor = SoftActionGovernor(self.band_count)

    def _build_hybrid_context(self, t: int) -> Dict[str, Any]:
        periodic_opportunity = np.array([
            self.periodic.periodic_opportunity(b, t) for b in range(self.band_count)
        ], dtype=float)

        markov_next_probs = self.markov.next_band_probabilities()
        vol_scores = np.array([
            self.vol.volatility(b) for b in range(self.band_count)
        ], dtype=float)

        belief_dict = {
            "p_active": np.array([b.p_active for b in self.belief.bands], dtype=float),
            "uncertainty": np.array([b.uncertainty for b in self.belief.bands], dtype=float),
            "staleness": np.array([b.staleness for b in self.belief.bands], dtype=int),
            "reward_mean": np.array([b.reward_mean for b in self.belief.bands], dtype=float),
        }

        governor_scores = self.governor.governor_scores(self.belief)

        return {
            "belief": belief_dict,
            "periodic_opportunity": periodic_opportunity,
            "markov_next_probs": markov_next_probs,
            "volatility": vol_scores,
            "governor_scores": governor_scores,
            "time_slot": t,
        }

    def select_action(self,
                      observation_history: List[Dict[str, Any]],
                      current_time_slot: int) -> int:

        ctx = self._build_hybrid_context(current_time_slot)
        current_obs_stub = {"global_t": current_time_slot, "hybrid_context": ctx}
        observation_history.append(current_obs_stub)

        try:
            raw_band = self.meta.select_action(
                observation_history=observation_history,
                current_time_slot=current_time_slot,
            )
        finally:
            observation_history.pop()

        return raw_band

    def update(self, action: int, observation: Dict[str, Any]):
        hit = bool(observation.get("hit", False))
        t = int(observation.get("global_t", 0))

        self.belief.update_after_step(action, hit)

        if hit:
            self.markov.update_observed_emitter_band(action)

        self.periodic.update(action, t, hit)
        self.vol.update(action, hit)

        self.meta.update(action, observation)
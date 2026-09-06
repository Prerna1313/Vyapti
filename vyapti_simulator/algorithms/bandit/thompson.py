"""
PS26055 Thompson-sampling bandit schedulers for non-stationary, restless arms.

[BANDIT] [LITERATURE-GROUNDED]
  - Thompson (1933); Agrawal & Goyal (2012) — Beta-Bernoulli Thompson sampling.
  - Trovo, Paladino, Restelli & Gatti (2020), "Sliding-window Thompson sampling
    for non-stationary settings" — SW-TS.
  - Raj & Kalyani (2017), "Taming non-stationary bandits: a Bayesian approach"
    — Discounted Thompson sampling.

These are the 2024-2025-current non-stationary baselines the algorithm audit
requires. Both are strong; the proposed periodicity-aware scheduler must beat
them on the DISCOVERY family specifically, not merely on aggregate reward, or
Gate 3 fails and that must be reported.

-----------------------------------------------------------------------
[SCIENTIFIC] Two implementation points that change the science
-----------------------------------------------------------------------
1. THE SLIDING WINDOW IS OVER TIME, NOT OVER PER-ARM OBSERVATIONS.
   A tempting implementation keeps, for each band, a list of that band's last W
   outcomes. That is NOT a sliding window: a band visited five times in 1000
   slots retains evidence from slot 3 forever, so its posterior never widens
   and the policy never re-explores it. The whole purpose of the window — the
   forgetting — is lost precisely for the rarely-visited bands where forgetting
   matters most. This implementation windows over GLOBAL slots, per Trovo et
   al.: evidence older than W slots is discarded regardless of which band it
   concerns, so a band unvisited for W slots reverts to the uniform prior and
   is guaranteed to be sampled again.

2. DISCOUNTED TS DECAYS EVERY ARM, NOT ONLY THE PLAYED ARM.
   Per Raj & Kalyani, S_b and F_b are multiplied by gamma each round for all b,
   while only the played arm accumulates new evidence. The consequence is the
   property UCB1 structurally lacks: an unobserved band's posterior widens back
   toward Beta(1,1) as its evidence ages, producing automatic revisiting driven
   by staleness rather than by visit count. For restless arms — where a band's
   true state changes whether or not you look — that is the correct inductive
   bias, and it is why discounted TS is expected to beat UCB1 here.

Neither policy forecasts, so `predict()` is not overridden and PS metrics 6 and
7 are reported unavailable for them.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np

from ...core.scheduler_interface import BaseScheduler


class SlidingWindowThompsonSampling(BaseScheduler):
    """
    SW-TS: Beta-Bernoulli Thompson sampling over a trailing window of slots.

    [ENGINEERING-ASSUMPTION] window=100 slots is one order of magnitude above
    the scenario registry's longest emitter period (20 slots), so the window
    spans several full emitter cycles and the posterior reflects current
    behaviour rather than a single phase. Fixed a priori; swept in the
    robustness study rather than tuned on the evaluation scenarios.
    """

    def __init__(self, band_count: int, window: int = 100,
                 prior_alpha: float = 1.0, prior_beta: float = 1.0):
        super().__init__(
            band_count,
            "[BANDIT] Sliding-Window Thompson Sampling (Trovo et al. 2020). "
            "Non-stationary baseline; window is over global slots so unvisited "
            "bands revert to the prior and are re-explored.",
        )
        self.window = int(window)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.rng: Optional[np.random.Generator] = None
        # Ring buffer of (band, reward) for the trailing window.
        self._actions: List[int] = []
        self._rewards: List[float] = []

    def reset(self, seed: int, scenario_config: Dict) -> None:
        self.rng = np.random.default_rng(seed)
        self._actions = []
        self._rewards = []
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0

    def _posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        a = np.asarray(self._actions, dtype=np.intp)
        r = np.asarray(self._rewards, dtype=float)
        if a.size == 0:
            return (np.full(self.band_count, self.prior_alpha),
                    np.full(self.band_count, self.prior_beta))
        succ = np.bincount(a, weights=r, minlength=self.band_count)
        tot = np.bincount(a, minlength=self.band_count).astype(float)
        return self.prior_alpha + succ, self.prior_beta + (tot - succ)

    def select_action(self, observation_history: List[Dict],
                      current_time_slot: int) -> int:
        if self.rng is None:
            raise RuntimeError("SlidingWindowThompsonSampling.reset() must be called before use.")
        self._check_truth_leakage(observation_history)
        alpha, beta = self._posterior()
        samples = self.rng.beta(alpha, beta)
        return self._validate_action(int(np.argmax(samples)))

    def update(self, action: int, observation: Dict) -> None:
        self.observation_history.append(observation)
        self._actions.append(int(action))
        self._rewards.append(1.0 if observation["hit"] else 0.0)
        if len(self._actions) > self.window:
            # Trim from the front: these slots are outside the window forever.
            del self._actions[:-self.window]
            del self._rewards[:-self.window]
        self.current_band = action
        self.steps_taken += 1

    def get_diagnostics(self) -> Dict:
        d = super().get_diagnostics()
        alpha, beta = self._posterior()
        mean = alpha / (alpha + beta)
        d.update({
            "window": self.window,
            "posterior_alpha": np.round(alpha, 3).tolist(),
            "posterior_beta": np.round(beta, 3).tolist(),
            "posterior_mean": np.round(mean, 4).tolist(),
            "bands_at_prior": int(np.sum((alpha == self.prior_alpha) & (beta == self.prior_beta))),
        })
        return d


class DiscountedThompsonSampling(BaseScheduler):
    """
    Discounted TS: exponentially-decayed Beta-Bernoulli counts on every arm.

    [ENGINEERING-ASSUMPTION] gamma=0.95 gives an effective memory of about
    1/(1-gamma) = 20 slots, matching the longest scenario-registry emitter
    period. Fixed a priori and reported; swept in the robustness study.
    """

    def __init__(self, band_count: int, gamma: float = 0.95,
                 prior_alpha: float = 1.0, prior_beta: float = 1.0):
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        super().__init__(
            band_count,
            "[BANDIT] Discounted Thompson Sampling (Raj & Kalyani 2017). "
            "All arms decay each slot, so posteriors widen with staleness — the "
            "restless-arm property UCB1 structurally lacks.",
        )
        self.gamma = float(gamma)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.rng: Optional[np.random.Generator] = None
        self.successes = np.zeros(band_count, dtype=float)
        self.failures = np.zeros(band_count, dtype=float)

    def reset(self, seed: int, scenario_config: Dict) -> None:
        self.rng = np.random.default_rng(seed)
        self.successes = np.zeros(self.band_count, dtype=float)
        self.failures = np.zeros(self.band_count, dtype=float)
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0

    def select_action(self, observation_history: List[Dict],
                      current_time_slot: int) -> int:
        if self.rng is None:
            raise RuntimeError("DiscountedThompsonSampling.reset() must be called before use.")
        self._check_truth_leakage(observation_history)
        alpha = self.prior_alpha + self.successes
        beta = self.prior_beta + self.failures
        samples = self.rng.beta(alpha, beta)
        return self._validate_action(int(np.argmax(samples)))

    def update(self, action: int, observation: Dict) -> None:
        self.observation_history.append(observation)
        reward = 1.0 if observation["hit"] else 0.0
        # Decay ALL arms, then credit the played arm.
        self.successes *= self.gamma
        self.failures *= self.gamma
        self.successes[action] += reward
        self.failures[action] += (1.0 - reward)
        self.current_band = action
        self.steps_taken += 1

    def get_diagnostics(self) -> Dict:
        d = super().get_diagnostics()
        alpha = self.prior_alpha + self.successes
        beta = self.prior_beta + self.failures
        n_eff = self.successes + self.failures
        d.update({
            "gamma": self.gamma,
            "effective_memory_slots": 1.0 / max(1e-12, 1.0 - self.gamma),
            "effective_sample_size": np.round(n_eff, 3).tolist(),
            "posterior_mean": np.round(alpha / (alpha + beta), 4).tolist(),
            # Posterior width is the staleness signal; report it as evidence.
            "posterior_std": np.round(
                np.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))), 4
            ).tolist(),
        })
        return d

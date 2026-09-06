"""
PS26055 Rising / restless-index bandit scheduler.

[BANDIT] [LITERATURE-GROUNDED]
  - Whittle (1988) — restless bandits and the Whittle index.
  - Metelli, Trovo, Pirola & Restelli (2022), "Stochastic Rising Bandits".
  - Levine, Crammer & Mannor (2017) — Rotting bandits (the mirror case).
  - Apfeld & Charlish (2021) — Markov/semi-Markov channel occupancy for ES.

-----------------------------------------------------------------------
Why a RISING index is the structurally correct model for ES search
-----------------------------------------------------------------------
In a standard bandit an arm's expected reward is fixed, so information about it
only improves with sampling. In ES band search the opposite holds for the
quantity that matters:

    P(band b holds a detectable emitter NOW | last observed k slots ago)

is INCREASING in k. Every slot you do not look at a band, the chance that
something has appeared, or that a periodic emitter has come back around, grows.
The expected value of dwelling on a band therefore RISES with its staleness.

This is what UCB1 cannot express: its exploration bonus sqrt(2 ln t / n_b)
depends on the visit COUNT, not on the RECENCY. Two bands sampled 50 times each
get identical bonuses whether one was last seen 2 slots ago and the other 400.

This policy makes staleness the explicit driver. For each band it maintains a
hazard estimate — the empirical probability that a dwell scores a hit, as a
function of how stale the band was when that dwell happened — and selects the
band with the highest predicted hit probability at its current staleness. It is
the natural non-parametric relative of the Whittle index for this problem, and
it is the honest competitor to the periodicity-aware methods: if a policy that
knows nothing about periods but tracks staleness properly already matches ESPE,
then scan-period estimation has not earned its complexity (Gate 3).

[ENGINEERING-ASSUMPTION] The hazard curve is estimated by binning staleness
into geometric buckets rather than fitting a parametric form, because the true
recurrence law differs by emitter family (deterministic for periodic, geometric
for Markov, heavy-tailed for semi-Markov) and committing to one shape would
build in an advantage on the families that match it.
"""

from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np

from ...core.scheduler_interface import BaseScheduler, BandPrediction


class RisingBanditScheduler(BaseScheduler):
    """
    Staleness-indexed restless bandit with a non-parametric hazard estimate.

    Reward model: P(hit | band b, staleness age) estimated from the policy's own
    dwell outcomes, pooled across bands for the age-shape and specialised per
    band for the level.
    """

    def __init__(self, band_count: int,
                 age_bin_edges: Optional[List[int]] = None,
                 prior_strength: float = 2.0,
                 optimism: float = 0.5,
                 max_staleness: Optional[int] = None):
        super().__init__(
            band_count,
            "[BANDIT] Rising restless-index bandit (Whittle 1988; Metelli et al. 2022; "
            "Apfeld & Charlish 2021). Selects on P(hit | staleness), the quantity "
            "UCB1's visit-count bonus cannot represent.",
        )
        # [ENGINEERING-ASSUMPTION] Geometric bins: recurrence effects are strong
        # at short ages and flatten out, so uniform bins would waste resolution.
        self.age_bin_edges = list(age_bin_edges) if age_bin_edges else [1, 2, 4, 8, 16, 32, 64, 128]
        self.n_bins = len(self.age_bin_edges) + 1
        self.prior_strength = float(prior_strength)
        self.optimism = float(optimism)
        self.max_staleness = max_staleness

        self.rng: Optional[np.random.Generator] = None
        self._last_seen = np.full(band_count, -1, dtype=np.int64)
        # Per-(band, age-bin) Beta-ish counts.
        self._succ = np.zeros((band_count, self.n_bins), dtype=float)
        self._tot = np.zeros((band_count, self.n_bins), dtype=float)
        # Pooled across bands, for the shared age shape.
        self._pool_succ = np.zeros(self.n_bins, dtype=float)
        self._pool_tot = np.zeros(self.n_bins, dtype=float)

    # -----------------------------------------------------------------
    def reset(self, seed: int, scenario_config: Dict) -> None:
        self.rng = np.random.default_rng(seed)
        self._last_seen = np.full(self.band_count, -1, dtype=np.int64)
        self._succ = np.zeros((self.band_count, self.n_bins), dtype=float)
        self._tot = np.zeros((self.band_count, self.n_bins), dtype=float)
        self._pool_succ = np.zeros(self.n_bins, dtype=float)
        self._pool_tot = np.zeros(self.n_bins, dtype=float)
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0

    def _age_bin(self, age: int) -> int:
        return int(np.searchsorted(self.age_bin_edges, age, side="right"))

    def _staleness(self, current_time_slot: int) -> np.ndarray:
        ages = current_time_slot - self._last_seen
        # Never-visited bands are maximally stale: they get first claim, which
        # is the correct behaviour and also guarantees full initial coverage.
        ages[self._last_seen < 0] = current_time_slot + 1
        return ages

    def _index(self, current_time_slot: int) -> np.ndarray:
        ages = self._staleness(current_time_slot)
        bins = np.fromiter((self._age_bin(int(a)) for a in ages),
                           dtype=np.intp, count=self.band_count)

        # Shared age-shape prior, shrinking each band toward the pooled curve.
        pool_rate = np.divide(
            self._pool_succ + 0.5, self._pool_tot + 1.0,
            out=np.full(self.n_bins, 0.5), where=True)

        rows = np.arange(self.band_count)
        s = self._succ[rows, bins]
        n = self._tot[rows, bins]
        prior = pool_rate[bins] * self.prior_strength
        rate = (s + prior) / (n + self.prior_strength)

        # Optimism scaled by posterior width: an unmeasured (band, age) cell is
        # explored, but the bonus shrinks as evidence accrues — unlike UCB1 this
        # bonus can grow again when a band drifts into a stale, unmeasured bin.
        width = np.sqrt(rate * (1.0 - rate) / (n + self.prior_strength))
        return rate + self.optimism * width

    def select_action(self, observation_history: List[Dict],
                      current_time_slot: int) -> int:
        if self.rng is None:
            raise RuntimeError("RisingBanditScheduler.reset() must be called before use.")
        self._check_truth_leakage(observation_history)

        # Hard staleness cap, when configured, dominates the index.
        if self.max_staleness is not None:
            ages = self._staleness(current_time_slot)
            if (ages >= self.max_staleness).any():
                return self._validate_action(int(np.argmax(ages)))

        idx = self._index(current_time_slot)
        best = np.flatnonzero(idx >= idx.max() - 1e-12)
        # Random tie-break, from the policy's own Generator: deterministic
        # tie-breaking by lowest index would bias the scan toward low bands.
        return self._validate_action(int(best[0] if best.size == 1
                                         else self.rng.choice(best)))

    def update(self, action: int, observation: Dict) -> None:
        self.observation_history.append(observation)
        slot = int(observation["time_slot"])
        hit = 1.0 if observation["hit"] else 0.0

        prev = self._last_seen[action]
        age = (slot + 1) if prev < 0 else (slot - int(prev))
        b = self._age_bin(age)

        self._succ[action, b] += hit
        self._tot[action, b] += 1.0
        self._pool_succ[b] += hit
        self._pool_tot[b] += 1.0

        self._last_seen[action] = slot
        self.current_band = action
        self.steps_taken += 1

    # -----------------------------------------------------------------
    def predict(self, current_time_slot: int) -> Optional[BandPrediction]:
        """
        Forecast from the hazard curve: P(active at next slot) per band.

        [SCIENTIFIC] This is a genuine probabilistic forecast but NOT a timing
        forecast — a hazard model says how likely activity is, not when it
        recurs. `predicted_next_activity_slot` is therefore left None, so this
        policy is scored on PS metric 6 and correctly excluded from metric 7
        rather than contributing a fabricated time estimate.
        """
        if self._pool_tot.sum() < self.band_count:
            return None  # not yet one dwell per band; no basis for a forecast
        ages = self._staleness(current_time_slot) + 1  # staleness at next slot
        bins = np.fromiter((self._age_bin(int(a)) for a in ages),
                           dtype=np.intp, count=self.band_count)
        pool_rate = (self._pool_succ + 0.5) / (self._pool_tot + 1.0)
        rows = np.arange(self.band_count)
        s = self._succ[rows, bins]
        n = self._tot[rows, bins]
        prior = pool_rate[bins] * self.prior_strength
        probs = (s + prior) / (n + self.prior_strength)
        return BandPrediction(
            issued_at_slot=current_time_slot,
            about_time_slot=current_time_slot + 1,
            band_activity_probability=np.clip(probs, 0.0, 1.0).tolist(),
            predicted_next_activity_slot=None,
            method_note=("[LITERATURE-GROUNDED] Non-parametric hazard P(hit | staleness); "
                         "probabilistic only, no timing claim."),
        )

    def get_diagnostics(self) -> Dict:
        d = super().get_diagnostics()
        pool_rate = (self._pool_succ + 0.5) / (self._pool_tot + 1.0)
        d.update({
            "age_bin_edges": self.age_bin_edges,
            "pooled_hazard_by_age_bin": np.round(pool_rate, 4).tolist(),
            "pooled_samples_by_age_bin": self._pool_tot.astype(int).tolist(),
            "optimism": self.optimism,
            "max_staleness": self.max_staleness,
            # The headline evidence: hazard should be increasing in staleness.
            "hazard_is_rising": bool(np.all(np.diff(pool_rate[self._pool_tot > 0]) >= -0.05))
            if (self._pool_tot > 0).sum() > 1 else None,
        })
        return d

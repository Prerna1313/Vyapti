"""
PS26055 Periodicity estimation from sparse, unevenly-sampled ES observations.

Shared by ESPE, Gaussian ESPE and the coverage-constrained scheduler.

=======================================================================
[SCIENTIFIC] Why the obvious estimator is wrong, and what is used instead
=======================================================================
An ES receiver observes a band only when it dwells on it. So the observation
series for a band is sparse and unevenly sampled, and two standard approaches
both fail:

FAILURE 1 — MEDIAN OF INTER-HIT INTERVALS.
The intuitive estimator is `median(diff(hit_slots))`. It is badly biased here.
If the true scan period is 10 slots but the scheduler only dwells on that band
every third visit, the OBSERVED intervals are 30, 20, 40... — multiples of the
true period. The median returns roughly 30, not 10. The estimator's bias grows
with how little the band is watched, i.e. it is worst exactly where a
periodicity-aware policy most needs it.

FAILURE 2 — PHASE FOLDING SCORED ONLY ON HITS.
The standard fix for uneven sampling is phase folding: for each candidate
period p, fold hit times into phase mod p and measure how concentrated they
are (circular concentration R, Rayleigh statistic nR^2). This correctly handles
sparse sampling and correctly suppresses INTEGER MULTIPLES of the true period
(folding a period-10 signal at p=20 splits hits into two antipodal clusters, so
R collapses).

But it fails catastrophically on DIVISORS. Fold a period-10 signal at p=5 and
every hit lands at phase 0, giving R = 1.0 — a perfect score. At p=1 every
possible time series scores R = 1.0. A hits-only criterion therefore selects
p=1, "this band is always active", for every band. Preferring the smallest
high-scoring period, which is the usual harmonic-suppression heuristic, walks
straight into this.

THE FIX — SCORE AGAINST MISSES TOO.
The divisor ambiguity is only an ambiguity if you ignore the negative evidence.
"I dwelled on band 3 at slot 47 and saw nothing" refutes p=1. This module
therefore fits a full generative model (period p, phase offset s, active-window
width w) by BALANCED ACCURACY over every observed slot, hits and misses alike:

    predicted_active(t)  <=>  ((t - s) mod p) < w
    score = 0.5 * ( hits inside window / all hits
                  + misses outside window / all misses )

p=1 forces w in [1,1) — an empty window — or w=p which predicts always-active
and is refuted by any miss. Divisors are refuted because a period-5 model with a
3-slot window predicts activity at slots 5-7, where misses were actually
recorded. Balanced accuracy rather than raw accuracy because activity is rare:
a model predicting "never active" scores high raw accuracy and must not win.

Search is O(p^2) per candidate period, fully vectorised over (offset, width),
and refitting is throttled to when new evidence for a band actually arrives.
=======================================================================
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class PeriodicModel:
    """A fitted (period, phase offset, active-window width) hypothesis."""
    period: int
    offset: int          # phase at which the active window starts
    width: int           # active-window length in slots
    score: float         # balanced accuracy on observed slots, in [0, 1]
    n_hits: int
    n_misses: int
    fitted_at_slot: int

    def is_active_at(self, t: int) -> bool:
        return ((t - self.offset) % self.period) < self.width

    def activity_probability_at(self, t: int) -> float:
        """
        Model-implied P(active at t), softened by fit quality.

        [ENGINEERING-ASSUMPTION] A model with balanced accuracy `score` is
        trusted proportionally: a perfect fit asserts 1/0, a chance-level fit
        (score 0.5) asserts the unconditional duty cycle. Emitting a hard 0/1
        from a marginal fit would make the calibration plot (mandatory figure
        11) look confident and be wrong, and would corrupt PS metric 6.
        """
        duty = self.width / self.period
        confidence = float(np.clip(2.0 * (self.score - 0.5), 0.0, 1.0))
        hard = 1.0 if self.is_active_at(t) else 0.0
        return confidence * hard + (1.0 - confidence) * duty

    def next_active_slot(self, after: int, horizon: Optional[int] = None) -> Optional[int]:
        """First slot strictly after `after` at which the model predicts activity."""
        limit = self.period if horizon is None else min(self.period, max(1, horizon))
        for d in range(1, limit + 1):
            if self.is_active_at(after + d):
                return after + d
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period, "offset": self.offset, "width": self.width,
            "score": round(self.score, 4), "duty_cycle": round(self.width / self.period, 4),
            "n_hits": self.n_hits, "n_misses": self.n_misses,
            "fitted_at_slot": self.fitted_at_slot,
        }


class PeriodicityEstimator:
    """
    Per-band periodic-model fitter over the scheduler's own observations.

    Uses ONLY (slot, band, hit) triples the scheduler legitimately received.
    It never touches truth; that is what makes the resulting forecast a genuine
    prediction rather than a disguised oracle read.
    """

    def __init__(self, band_count: int,
                 period_min: int = 2, period_max: int = 32,
                 min_hits: int = 3, min_observations: int = 8,
                 min_score: float = 0.65,
                 tie_margin: float = 0.02,
                 refit_min_interval: int = 5):
        self.band_count = band_count
        self.period_min = max(2, int(period_min))   # p=1 excluded: degenerate
        self.period_max = int(period_max)
        self.min_hits = int(min_hits)
        self.min_observations = int(min_observations)
        self.min_score = float(min_score)
        self.tie_margin = float(tie_margin)
        self.refit_min_interval = int(refit_min_interval)

        # Per-band observation records (scheduler-visible data only).
        self._slots: List[List[int]] = [[] for _ in range(band_count)]
        self._hits: List[List[bool]] = [[] for _ in range(band_count)]
        self._models: List[Optional[PeriodicModel]] = [None] * band_count
        self._last_fit_slot: List[int] = [-10 ** 9] * band_count
        self._dirty: List[bool] = [False] * band_count
        self.fit_calls = 0

    # -----------------------------------------------------------------
    def reset(self) -> None:
        self._slots = [[] for _ in range(self.band_count)]
        self._hits = [[] for _ in range(self.band_count)]
        self._models = [None] * self.band_count
        self._last_fit_slot = [-10 ** 9] * self.band_count
        self._dirty = [False] * self.band_count
        self.fit_calls = 0

    def observe(self, band: int, slot: int, hit: bool) -> None:
        """Record one dwell outcome. Misses are recorded: they are evidence."""
        self._slots[band].append(int(slot))
        self._hits[band].append(bool(hit))
        # Only a hit changes which periodic hypotheses are plausible enough to
        # be worth the refit cost; a miss is folded in at the next refit.
        if hit:
            self._dirty[band] = True

    def model(self, band: int) -> Optional[PeriodicModel]:
        return self._models[band]

    def maybe_refit(self, band: int, current_slot: int) -> Optional[PeriodicModel]:
        """Refit `band` if new positive evidence arrived and throttle allows."""
        if not self._dirty[band]:
            return self._models[band]
        if current_slot - self._last_fit_slot[band] < self.refit_min_interval:
            return self._models[band]
        self._dirty[band] = False
        self._last_fit_slot[band] = current_slot
        self._models[band] = self._fit(band, current_slot)
        return self._models[band]

    def refit_all(self, current_slot: int) -> None:
        for b in range(self.band_count):
            self.maybe_refit(b, current_slot)

    # -----------------------------------------------------------------
    def _fit(self, band: int, current_slot: int) -> Optional[PeriodicModel]:
        slots = np.asarray(self._slots[band], dtype=np.int64)
        hits = np.asarray(self._hits[band], dtype=bool)
        n_hits = int(hits.sum())
        n_misses = int((~hits).sum())

        if slots.size < self.min_observations or n_hits < self.min_hits:
            return None
        # Without any negative evidence the divisor ambiguity is unresolvable,
        # so decline to fit rather than emit a model that may be a divisor of
        # the truth. Declining is the honest outcome; a wrong period would
        # actively mislead the scheduler.
        if n_misses == 0:
            return None

        self.fit_calls += 1
        H = float(n_hits)
        M = float(n_misses)
        hit_slots = slots[hits]
        miss_slots = slots[~hits]

        best: Optional[PeriodicModel] = None
        p_max = min(self.period_max, max(self.period_min, int(slots.max() - slots.min())))

        for p in range(self.period_min, p_max + 1):
            h = np.bincount(hit_slots % p, minlength=p).astype(float)
            m = np.bincount(miss_slots % p, minlength=p).astype(float)
            # Circular prefix sums via doubling, so a window may wrap phase 0.
            ch = np.concatenate(([0.0], np.cumsum(np.concatenate((h, h)))))
            cm = np.concatenate(([0.0], np.cumsum(np.concatenate((m, m)))))

            s = np.arange(p, dtype=np.int64)[:, None]        # window start
            w = np.arange(1, p, dtype=np.int64)[None, :]     # window width < p
            end = s + w
            hits_in = ch[end] - ch[s]
            miss_in = cm[end] - cm[s]

            tpr = hits_in / H                       # hits explained by the window
            tnr = (M - miss_in) / M                 # misses correctly left outside
            score = 0.5 * (tpr + tnr)

            k = int(np.argmax(score))
            si, wi = np.unravel_index(k, score.shape)
            cand_score = float(score[si, wi])

            # Parsimony tie-break: among near-equal fits prefer the SHORTER
            # period, which is the more constrained hypothesis. Safe here only
            # because divisors have already been penalised by the miss term.
            if best is None or cand_score > best.score + self.tie_margin:
                best = PeriodicModel(
                    period=p, offset=int(s[si, 0]), width=int(w[0, wi]),
                    score=cand_score, n_hits=n_hits, n_misses=n_misses,
                    fitted_at_slot=current_slot,
                )

        if best is None or best.score < self.min_score:
            return None
        return best

    # -----------------------------------------------------------------
    def activity_probability(self, slot: int, fallback: float) -> np.ndarray:
        """
        Per-band model-implied P(active at `slot`).

        Bands with no fitted model receive `fallback` — normally the observed
        empirical hit rate — so an unfitted band is neither asserted active nor
        asserted silent.
        """
        out = np.full(self.band_count, float(fallback), dtype=float)
        for b in range(self.band_count):
            mdl = self._models[b]
            if mdl is not None:
                out[b] = mdl.activity_probability_at(slot)
        return out

    def next_active_slots(self, after: int) -> List[Optional[int]]:
        return [None if (m := self._models[b]) is None else m.next_active_slot(after)
                for b in range(self.band_count)]

    def empirical_hit_rate(self, band: int) -> Optional[float]:
        hs = self._hits[band]
        return (sum(hs) / len(hs)) if hs else None

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "fit_calls": self.fit_calls,
            "bands_with_model": int(sum(m is not None for m in self._models)),
            "models": {b: (m.to_dict() if m else None) for b, m in enumerate(self._models)},
            "observations_per_band": [len(s) for s in self._slots],
        }

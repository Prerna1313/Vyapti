"""
PS26055 PRIMARY CONTRIBUTION — Coverage-Constrained Periodicity-Aware Bandit.

[CONTRIBUTION] Our formulation. Not from any single published source; it
composes four established ingredients under one novel hard constraint.

=======================================================================
The claim, stated precisely enough to be falsifiable
=======================================================================
Every bandit baseline in this study — UCB1, SW-UCB, SW-TS, Discounted TS, ESPE,
Gaussian ESPE, Rising — optimises an EXPECTATION. None of them can promise
anything about the worst case. In every one of them there exists a run in which
some band is never revisited for an arbitrarily long stretch, because the
selection rule is a comparison of scores and a band whose score stays lowest
stays unvisited. For an ES receiver that is operationally unacceptable: a
newly-arrived threat emitter in a neglected band is invisible for as long as the
neglect lasts, and no expectation-based guarantee bounds that.

This scheduler adds a HARD staleness cap A. If any band's staleness reaches A,
that band is dwelled on next, unconditionally, overriding every learned score.

GUARANTEE (immediate, and the reason the design is worth writing up):
  Every band is dwelled on at least once in any window of A + band_count slots.
  Therefore any emitter transmitting continuously in some band for at least
  A + band_count slots is dwelled upon at least once, and is detected with
  probability at least Pd. Worst-case time-to-first-dwell is bounded by
  A + band_count REGARDLESS of what the learned components believe.

  (The + band_count term is the settling cost: in the worst case every band
  hits the cap in the same slot and they are served in sequence.)

No baseline in the comparison set has any such bound. Round-robin has an even
tighter one — exactly band_count — but pays for it with zero adaptivity and
with the synchronisation lockout described in baselines/fixed.py. The
contribution is that the two properties are not in fact a trade-off: the cap
costs only the slots it actually consumes, and inside the cap the policy is free
to be as opportunistic as any bandit.

FALSIFIABLE PREDICTIONS (must be reported whichever way they resolve):
  P1. worst_case_band_staleness_slots <= A + band_count on EVERY episode.
      A single violation falsifies the guarantee and is a bug, not a result.
  P2. On the delayed-arrival family, this scheduler's p90 first-intercept time
      is lower than every unconstrained bandit's. If it is not, the constraint
      is not buying what is claimed.
  P3. On the stationary continuous family it should LOSE slightly to pure
      exploitation, because the cap forces dwells that a greedy policy would
      spend on a known-productive band. That loss is the price of the guarantee
      and must be quantified, not hidden.

=======================================================================
Decision score
=======================================================================
    S_b(t) = w_v * p_hat_b        recency-weighted hit rate      (exploit)
           + w_q * q_hat_b(t)     fitted scan-period forecast    (predict)
           + w_u * u_b(t)         staleness-scaled uncertainty   (explore)
           + w_a * A_b(t)/A       normalised coverage pressure   (cover)
           - w_c * switch_b       retune cost if b != current    (economise)

subject to:  A_b(t) < A  enforced by pre-emption.

[ENGINEERING-ASSUMPTION] Weights default to 1.0 / 1.0 / 0.5 / 0.3 / 0.1 and the
cap to 3 * band_count. All are fixed a priori, reported in provenance, and swept
in the ablation study (which is what Phase 7's ablation chart is for) rather than
tuned per scenario. Terms are normalised into [0,1] before weighting so the
weights are comparable and an ablation that zeroes one is interpretable.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Sequence
import numpy as np

from ...core.scheduler_interface import BaseScheduler, BandPrediction
from .periodicity import PeriodicityEstimator


class CoverageConstrainedScheduler(BaseScheduler):
    """
    Periodicity-aware bandit with a hard worst-case coverage guarantee.

    Parameters
    ----------
    max_staleness : int, optional
        The cap A. Defaults to 3 * band_count. Setting it to band_count reduces
        the policy to round-robin (the cap fires every slot); setting it to None
        removes the guarantee and yields the unconstrained ablation.
    band_priority : sequence of float, optional
        Static per-band importance for Subproblem F (threat prioritisation).
        [ENGINEERING-ASSUMPTION] Exposed as a BAND-level weight because the
        scheduler cannot see emitter identity — per-emitter threat weighting
        requires deinterleaving and is deferred to TSRD Option 3. Uniform by
        default, so it is inert unless deliberately supplied.
    """

    def __init__(self, band_count: int,
                 max_staleness: Optional[int] = None,
                 value_weight: float = 1.0,
                 periodic_weight: float = 1.0,
                 uncertainty_weight: float = 0.5,
                 coverage_weight: float = 0.3,
                 switch_cost_weight: float = 0.1,
                 recency_factor: float = 0.95,
                 period_min: int = 2,
                 period_max: int = 32,
                 band_priority: Optional[Sequence[float]] = None):
        cap = (3 * band_count) if max_staleness is None else int(max_staleness)
        super().__init__(
            band_count,
            "[CONTRIBUTION] Coverage-Constrained Periodicity-Aware Bandit. "
            f"Hard staleness cap A={cap} guarantees every band is dwelled within "
            f"A+N={cap + band_count} slots, bounding worst-case discovery latency — "
            "a guarantee no expectation-optimising bandit provides.",
        )
        self.max_staleness = cap
        self.value_weight = float(value_weight)
        self.periodic_weight = float(periodic_weight)
        self.uncertainty_weight = float(uncertainty_weight)
        self.coverage_weight = float(coverage_weight)
        self.switch_cost_weight = float(switch_cost_weight)
        if not (0.0 < recency_factor <= 1.0):
            raise ValueError(f"recency_factor must be in (0, 1], got {recency_factor}")
        self.recency_factor = float(recency_factor)

        if band_priority is None:
            self.band_priority = np.ones(band_count, dtype=float)
        else:
            bp = np.asarray(band_priority, dtype=float)
            if bp.shape != (band_count,):
                raise ValueError(
                    f"band_priority must have length {band_count}, got {bp.shape}")
            if (bp < 0).any():
                raise ValueError("band_priority must be non-negative.")
            self.band_priority = bp / max(1e-12, bp.max())

        self.rng: Optional[np.random.Generator] = None
        self.estimator = PeriodicityEstimator(
            band_count, period_min=period_min, period_max=period_max)

        self._reward_ema = np.zeros(band_count, dtype=float)
        self._weight_ema = np.zeros(band_count, dtype=float)
        self._last_seen = np.full(band_count, -1, dtype=np.int64)
        self._current_band: Optional[int] = None

        # Audit counters proving P1 and quantifying the cap's cost.
        self.constraint_activations = 0
        self.observed_max_staleness = 0
        self.constraint_violations = 0

    # -----------------------------------------------------------------
    def reset(self, seed: int, scenario_config: Dict) -> None:
        self.rng = np.random.default_rng(seed)
        self.estimator.reset()
        self._reward_ema = np.zeros(self.band_count, dtype=float)
        self._weight_ema = np.zeros(self.band_count, dtype=float)
        self._last_seen = np.full(self.band_count, -1, dtype=np.int64)
        self._current_band = None
        self.constraint_activations = 0
        self.observed_max_staleness = 0
        self.constraint_violations = 0
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0

    # -----------------------------------------------------------------
    def _staleness(self, t: int) -> np.ndarray:
        ages = t - self._last_seen
        ages[self._last_seen < 0] = t + 1   # never seen: maximally stale
        return ages

    def _value_estimate(self) -> np.ndarray:
        out = np.full(self.band_count, 0.5, dtype=float)
        seen = self._weight_ema > 1e-9
        out[seen] = self._reward_ema[seen] / self._weight_ema[seen]
        return out

    def _uncertainty(self) -> np.ndarray:
        """
        Posterior width of the recency-weighted rate, in [0, 0.5] -> scaled to [0,1].

        Uses the DECAYED effective sample size, so uncertainty grows back as
        evidence ages. This is the property that distinguishes the term from
        UCB1's visit-count bonus, which only ever shrinks.
        """
        n_eff = self._weight_ema
        p = self._value_estimate()
        return 2.0 * np.sqrt(p * (1.0 - p) / (n_eff + 1.0))

    def score_breakdown(self, t: int) -> Dict[str, np.ndarray]:
        """All five terms, normalised, for the ablation study and diagnostics."""
        self.estimator.refit_all(t)
        p_hat = self._value_estimate()
        q_hat = self.estimator.activity_probability(t, fallback=float(p_hat.mean()))
        u = np.clip(self._uncertainty(), 0.0, 1.0)
        ages = self._staleness(t)
        pressure = np.clip(ages / max(1, self.max_staleness), 0.0, 1.0)
        switch = np.zeros(self.band_count, dtype=float)
        if self._current_band is not None:
            switch[:] = 1.0
            switch[self._current_band] = 0.0
        return {
            "value": p_hat, "periodic": q_hat, "uncertainty": u,
            "coverage_pressure": pressure, "switch_cost": switch,
            "staleness": ages.astype(float), "priority": self.band_priority,
        }

    def _scores(self, t: int) -> np.ndarray:
        c = self.score_breakdown(t)
        learned = (self.value_weight * c["value"]
                   + self.periodic_weight * c["periodic"]
                   + self.uncertainty_weight * c["uncertainty"])
        # Threat priority scales the LEARNED part only. Scaling the coverage
        # term too would let a high-priority band starve a low-priority one of
        # its guaranteed revisit, which would silently void the guarantee.
        return (self.band_priority * learned
                + self.coverage_weight * c["coverage_pressure"]
                - self.switch_cost_weight * c["switch_cost"])

    # -----------------------------------------------------------------
    def select_action(self, observation_history: List[Dict],
                      current_time_slot: int) -> int:
        if self.rng is None:
            raise RuntimeError("CoverageConstrainedScheduler.reset() must be called before use.")
        self._check_truth_leakage(observation_history)

        ages = self._staleness(current_time_slot)
        self.observed_max_staleness = max(self.observed_max_staleness, int(ages.max()))

        # --- HARD CONSTRAINT: pre-empt everything -----------------------
        breached = np.flatnonzero(ages >= self.max_staleness)
        if breached.size:
            # Serve the most stale first; ties by lowest index for determinism.
            self.constraint_activations += 1
            if int(ages.max()) > self.max_staleness + self.band_count:
                # Should be unreachable; records a violation of P1 rather than
                # silently continuing, so a regression cannot pass unnoticed.
                self.constraint_violations += 1
                self.audit_flags.append(
                    f"COVERAGE_GUARANTEE_VIOLATION: staleness {int(ages.max())} exceeded "
                    f"A+N={self.max_staleness + self.band_count} at slot {current_time_slot}"
                )
            worst = breached[np.argmax(ages[breached])]
            return self._validate_action(int(worst))

        scores = self._scores(current_time_slot)
        best = np.flatnonzero(scores >= scores.max() - 1e-12)
        band = int(best[0] if best.size == 1 else self.rng.choice(best))
        return self._validate_action(band)

    def update(self, action: int, observation: Dict) -> None:
        self.observation_history.append(observation)
        slot = int(observation["time_slot"])
        hit = bool(observation["hit"])

        self.estimator.observe(action, slot, hit)

        self._reward_ema *= self.recency_factor
        self._weight_ema *= self.recency_factor
        self._reward_ema[action] += 1.0 if hit else 0.0
        self._weight_ema[action] += 1.0

        self._last_seen[action] = slot
        self._current_band = action
        self.current_band = action
        self.steps_taken += 1

    # -----------------------------------------------------------------
    def predict(self, current_time_slot: int) -> Optional[BandPrediction]:
        """Forecast for the next slot; drives PS metrics 6 and 7."""
        if self.estimator.diagnostics()["bands_with_model"] == 0:
            return None
        target = current_time_slot + 1
        p_hat = self._value_estimate()
        probs = self.estimator.activity_probability(target, fallback=float(p_hat.mean()))
        return BandPrediction(
            issued_at_slot=current_time_slot,
            about_time_slot=target,
            band_activity_probability=probs.tolist(),
            predicted_next_activity_slot=self.estimator.next_active_slots(current_time_slot),
            method_note=("[CONTRIBUTION] Coverage-constrained scheduler forecast from "
                         "(period, phase, width) models fitted to hits and misses."),
        )

    def get_diagnostics(self) -> Dict:
        d = super().get_diagnostics()
        t = max(self.steps_taken, 1)
        d.update({
            "max_staleness_cap_A": self.max_staleness,
            "guaranteed_dwell_window_slots": self.max_staleness + self.band_count,
            "observed_max_staleness": self.observed_max_staleness,
            "constraint_activations": self.constraint_activations,
            "constraint_activation_rate": self.constraint_activations / t,
            "constraint_violations": self.constraint_violations,
            # P1 check, per-episode. Must be True on every episode.
            "guarantee_held": self.observed_max_staleness <= self.max_staleness + self.band_count,
            "weights": {
                "value": self.value_weight, "periodic": self.periodic_weight,
                "uncertainty": self.uncertainty_weight,
                "coverage": self.coverage_weight, "switch_cost": self.switch_cost_weight,
            },
            "band_priority": self.band_priority.tolist(),
            "value_estimate": np.round(self._value_estimate(), 4).tolist(),
            "periodicity": self.estimator.diagnostics(),
        })
        return d

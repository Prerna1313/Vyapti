"""
PS26055 ESPE family — Emitter Scan-Period Estimation schedulers.

[BANDIT] [LITERATURE-GROUNDED] Teissier et al. (2024) — multi-armed-bandit
search for radar emitters with scan-period estimation, and its frequency-agile
Gaussian variant.

This is the MANDATORY closest-published baseline named in the algorithm audit.
The proposed coverage-constrained scheduler must be compared against it directly,
and Gate 3 ("periodic awareness earns its complexity") is decided by whether
periodicity-aware methods beat the non-stationary bandits of ucb.py/thompson.py.

-----------------------------------------------------------------------
Structure
-----------------------------------------------------------------------
ESPE combines two signals:

  p_hat[b]  recency-weighted empirical hit rate for band b — "has this band
            been productive lately". Exponentially decayed so a band that was
            productive 500 slots ago is not trusted now.

  q_hat[b]  periodic forecast — "does my fitted scan-period model say band b
            will be illuminated at the NEXT slot". Supplied by
            PeriodicityEstimator, which fits (period, phase, width) against
            both hits and misses (see periodicity.py for why misses are
            essential).

Selection is Boltzmann (softmax) over the combined score rather than argmax.
[ENGINEERING-ASSUMPTION] Softmax is used because a hard argmax on a periodic
forecast is self-reinforcing: the policy dwells only where its current model
predicts activity, so it gathers no evidence that could refute that model, and a
wrong period is never corrected. Stochastic selection keeps a floor of
counter-evidence flowing. This is a real failure mode, not a stylistic choice,
and the temperature is what controls it.

-----------------------------------------------------------------------
[SCIENTIFIC] Both ESPE variants DO forecast, so both implement predict()
-----------------------------------------------------------------------
Because they carry an explicit generative model of band activity, they can be
scored on PS metric 6 (percentage of correct predictions) and metric 7 (average
intercept time error). Round-robin, UCB1 and Thompson sampling cannot: they hold
no model of the future and return None from predict(), so those two metrics are
reported unavailable for them. That asymmetry is a genuine finding about which
algorithm classes can satisfy the full PS metric set, and it is the reason the
prediction hook exists on the scheduler contract at all.
"""

from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np

from ...core.scheduler_interface import BaseScheduler, BandPrediction
from .periodicity import PeriodicityEstimator


class ESPEScheduler(BaseScheduler):
    """
    Emitter Scan-Period Estimation scheduler (Teissier et al. 2024).

    [ENGINEERING-ASSUMPTION] Default hyperparameters are fixed a priori and
    reported, never tuned against the evaluation scenarios:
      recency_factor 0.95 -> ~20-slot memory, matching the longest registry period
      temperature    0.30 -> exploratory enough to refute a wrong period model
      value/periodic weights 1.0 / 1.0 -> equal trust, no implicit preference
    """

    def __init__(self, band_count: int,
                 recency_factor: float = 0.95,
                 temperature: float = 0.30,
                 value_weight: float = 1.0,
                 periodic_weight: float = 1.0,
                 period_min: int = 2,
                 period_max: int = 32):
        super().__init__(
            band_count,
            "[BANDIT] ESPE — Emitter Scan-Period Estimation (Teissier et al. 2024). "
            "Recency-weighted value plus fitted scan-period forecast, Boltzmann "
            "selection. Mandatory closest-published baseline.",
        )
        if not (0.0 < recency_factor <= 1.0):
            raise ValueError(f"recency_factor must be in (0, 1], got {recency_factor}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.recency_factor = float(recency_factor)
        self.temperature = float(temperature)
        self.value_weight = float(value_weight)
        self.periodic_weight = float(periodic_weight)

        self.rng: Optional[np.random.Generator] = None
        self.estimator = PeriodicityEstimator(
            band_count, period_min=period_min, period_max=period_max)
        # Recency-weighted hit rate as a decayed ratio, so it is a genuine rate
        # in [0,1] rather than an unnormalised score.
        self._reward_ema = np.zeros(band_count, dtype=float)
        self._weight_ema = np.zeros(band_count, dtype=float)
        self._last_prediction: Optional[BandPrediction] = None

    # -----------------------------------------------------------------
    def reset(self, seed: int, scenario_config: Dict) -> None:
        self.rng = np.random.default_rng(seed)
        self.estimator.reset()
        self._reward_ema = np.zeros(self.band_count, dtype=float)
        self._weight_ema = np.zeros(self.band_count, dtype=float)
        self._last_prediction = None
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0

    # -----------------------------------------------------------------
    def _value_estimate(self) -> np.ndarray:
        """Recency-weighted hit rate; 0.5 (uninformative) where no evidence."""
        out = np.full(self.band_count, 0.5, dtype=float)
        seen = self._weight_ema > 1e-9
        out[seen] = self._reward_ema[seen] / self._weight_ema[seen]
        return out

    def _scores(self, current_time_slot: int) -> np.ndarray:
        self.estimator.refit_all(current_time_slot)
        p_hat = self._value_estimate()
        # Forecast is about the slot we are ABOUT to dwell on.
        q_hat = self.estimator.activity_probability(
            current_time_slot, fallback=float(p_hat.mean()))
        return self.value_weight * p_hat + self.periodic_weight * q_hat

    def select_action(self, observation_history: List[Dict],
                      current_time_slot: int) -> int:
        if self.rng is None:
            raise RuntimeError("ESPEScheduler.reset() must be called before use.")
        self._check_truth_leakage(observation_history)

        scores = self._scores(current_time_slot)
        # Numerically stable softmax.
        z = (scores - scores.max()) / self.temperature
        expz = np.exp(z)
        probs = expz / expz.sum()
        band = int(self.rng.choice(self.band_count, p=probs))
        return self._validate_action(band)

    def update(self, action: int, observation: Dict) -> None:
        self.observation_history.append(observation)
        slot = int(observation["time_slot"])
        hit = bool(observation["hit"])

        # Feed the periodicity model. Misses are recorded too; they are what
        # disambiguates a true period from its divisors.
        self.estimator.observe(action, slot, hit)

        # Decay every band (information ages everywhere), credit the played one.
        self._reward_ema *= self.recency_factor
        self._weight_ema *= self.recency_factor
        self._reward_ema[action] += 1.0 if hit else 0.0
        self._weight_ema[action] += 1.0

        self.current_band = action
        self.steps_taken += 1

    # -----------------------------------------------------------------
    def predict(self, current_time_slot: int) -> Optional[BandPrediction]:
        """
        Forecast band occupancy at the next slot (PS metrics 6 and 7).

        Returns None until at least one band has a fitted periodic model: before
        that this policy has no forecasting basis, and emitting the flat
        empirical rate as if it were a prediction would score PS metric 6
        against a model that does not exist.
        """
        if self.estimator.diagnostics()["bands_with_model"] == 0:
            return None
        target = current_time_slot + 1
        p_hat = self._value_estimate()
        probs = self.estimator.activity_probability(target, fallback=float(p_hat.mean()))
        pred = BandPrediction(
            issued_at_slot=current_time_slot,
            about_time_slot=target,
            band_activity_probability=probs.tolist(),
            predicted_next_activity_slot=self.estimator.next_active_slots(current_time_slot),
            method_note=("[LITERATURE-GROUNDED] ESPE scan-period model; "
                         "(period, phase, width) fitted to observed hits AND misses."),
        )
        self._last_prediction = pred
        return pred

    def get_diagnostics(self) -> Dict:
        d = super().get_diagnostics()
        d.update({
            "recency_factor": self.recency_factor,
            "temperature": self.temperature,
            "value_estimate": np.round(self._value_estimate(), 4).tolist(),
            "periodicity": self.estimator.diagnostics(),
        })
        return d


class GaussianESPEScheduler(ESPEScheduler):
    """
    Gaussian ESPE — shares positive evidence with spectrally adjacent bands.

    [BANDIT] [LITERATURE-GROUNDED] Teissier et al. (2024) frequency-agile variant.

    Rationale: a frequency-agile emitter does not stay in one band, so evidence
    that band b was productive is partial evidence about b-1 and b+1. A Gaussian
    kernel over band index spreads each observation to its neighbours, which
    lets the policy track a hopping emitter instead of chasing it one band
    behind.

    [ENGINEERING-ASSUMPTION] Band index is used as the proximity metric because
    the environment's bands are contiguous, equal-width slices of the spectrum,
    so adjacency in index IS adjacency in frequency. If the band plan were ever
    made non-contiguous this kernel would have to be rebuilt over centre
    frequencies; the assumption is recorded here because it is invisible at the
    call site.

    NOTE the cost: spreading evidence deliberately BLURS the per-band periodic
    signal, so this variant is expected to beat plain ESPE on the agile families
    and to LOSE to it on the strictly-periodic single-band families. Both
    directions must be reported; the frozen protocol forbids presenting only the
    family where a method wins.
    """

    def __init__(self, band_count: int,
                 recency_factor: float = 0.95,
                 temperature: float = 0.30,
                 value_weight: float = 1.0,
                 periodic_weight: float = 1.0,
                 sigma: float = 1.0,
                 period_min: int = 2,
                 period_max: int = 32):
        super().__init__(band_count, recency_factor, temperature,
                         value_weight, periodic_weight, period_min, period_max)
        self.provenance_note = (
            "[BANDIT] Gaussian ESPE (Teissier et al. 2024 frequency-agile variant). "
            f"Gaussian evidence kernel over band index, sigma={sigma}. Expected to "
            "trade periodic sharpness for agile tracking."
        )
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = float(sigma)
        # Precompute the normalised kernel row for each band.
        idx = np.arange(band_count, dtype=float)
        d2 = (idx[:, None] - idx[None, :]) ** 2
        k = np.exp(-d2 / (2.0 * self.sigma ** 2))
        self._kernel = k / k.sum(axis=1, keepdims=True)

    def update(self, action: int, observation: Dict) -> None:
        self.observation_history.append(observation)
        slot = int(observation["time_slot"])
        hit = bool(observation["hit"])

        # The periodicity model is NOT blurred: a fitted period belongs to the
        # band it was observed in. Only the value estimate is shared.
        self.estimator.observe(action, slot, hit)

        w = self._kernel[action]
        self._reward_ema *= self.recency_factor
        self._weight_ema *= self.recency_factor
        self._reward_ema += w * (1.0 if hit else 0.0)
        self._weight_ema += w

        self.current_band = action
        self.steps_taken += 1

    def get_diagnostics(self) -> Dict:
        d = super().get_diagnostics()
        d["sigma"] = self.sigma
        d["kernel_row_0"] = np.round(self._kernel[0], 4).tolist()
        return d

"""
vyapti_simulator.core.observation_context
=======================================

Multi-step observation context utilities for schedulers.

This module provides helpers for schedulers that want to look at
multiple recent observations rather than just the single most recent one.
This is useful for:
- Tracking temporal patterns (periodic emitters)
- Building observation sequences for sequence models
- Computing rolling statistics over recent observations

Per frozen protocol Gate 0: These helpers only access observation_history,
which contains only PERMITTED_OBSERVATION_KEYS fields. No truth is leaked.

Usage
-----
::

    from vyapti_simulator.core.observation_context import ObservationContext

    ctx = ObservationContext(observation_history, window=10)

    # Recent observations (most recent last)
    recent = ctx.get_recent_observations()

    # Hit rate in window
    hit_rate = ctx.hit_rate()

    # Per-band hit counts
    band_counts = ctx.band_hit_counts()

    # Sequence for RNN/LSTM input
    sequence = ctx.get_feature_matrix()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class ObservationSummary:
    """Compact summary of recent observations."""
    window_size: int
    n_observations: int
    hit_rate: float
    n_unique_bands: int
    most_active_band: Optional[int]
    recency_weighted_hit_rate: float


class ObservationContext:
    """
    Multi-step observation context for schedulers.

    Provides a sliding window view over the observation history,
    along with derived statistics useful for temporal reasoning.

    This is a VIEW class — it does not modify or copy the underlying
    observation_history, and does not store state. Each call recomputes
    from the history.

    Parameters
    ----------
    observation_history : List[Dict]
        The scheduler's observation history (from BaseScheduler.observation_history).
    window : int, optional
        Number of recent observations to consider. If None, uses all history.
    """

    def __init__(
        self,
        observation_history: List[Dict],
        window: Optional[int] = None,
    ):
        self._history = observation_history
        self._window = window

    @property
    def window(self) -> Optional[int]:
        """Sliding window size, or None for full history."""
        return self._window

    @window.setter
    def window(self, value: Optional[int]) -> None:
        self._window = value

    @property
    def history(self) -> List[Dict]:
        """Reference to the underlying observation history."""
        return self._history

    @property
    def n_total(self) -> int:
        """Total number of observations in history."""
        return len(self._history)

    @property
    def n_windowed(self) -> int:
        """Number of observations in the current window."""
        if self._window is None:
            return len(self._history)
        return min(len(self._history), self._window)

    def _windowed_history(self) -> List[Dict]:
        """Get the windowed slice of history (most recent last)."""
        if self._window is None:
            return self._history
        return self._history[-self._window:]

    def get_recent_observations(self) -> List[Dict]:
        """
        Get the most recent N observations (most recent last).

        Returns
        -------
        List[Dict]
            Windowed observation list, most recent at the end.
        """
        return list(self._windowed_history())

    def hit_rate(self, min_observations: int = 1) -> float:
        """
        Compute hit rate over the window.

        Parameters
        ----------
        min_observations : int
            Minimum observations required. Returns 0.0 if fewer.

        Returns
        -------
        float
            Fraction of observations with hit=True.
        """
        obs = self._windowed_history()
        if len(obs) < min_observations:
            return 0.0
        hits = sum(1 for o in obs if o.get("hit", False))
        return hits / len(obs)

    def recency_weighted_hit_rate(self, decay: float = 0.1) -> float:
        """
        Compute exponentially-weighted hit rate, more recent = higher weight.

        Parameters
        ----------
        decay : float
            Decay rate per slot. Higher = more weight on recent.

        Returns
        -------
        float
            Recency-weighted hit rate in [0, 1].
        """
        obs = self._windowed_history()
        if not obs:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        n = len(obs)

        for i, o in enumerate(obs):
            # Weight: most recent (i = n-1) gets weight 1.0
            weight = np.exp(decay * (i - (n - 1)))
            total_weight += weight
            if o.get("hit", False):
                weighted_sum += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def band_hit_counts(self) -> Dict[int, Tuple[int, int]]:
        """
        Per-band hit and miss counts over the window.

        Returns
        -------
        Dict[int, Tuple[hit_count, miss_count]]
            For each band observed, the (hits, misses) tuple.
        """
        counts: Dict[int, Tuple[int, int]] = {}
        for o in self._windowed_history():
            band = o.get("selected_band")
            if band is None:
                continue
            band = int(band)
            if band not in counts:
                counts[band] = (0, 0)
            hits, misses = counts[band]
            if o.get("hit", False):
                counts[band] = (hits + 1, misses)
            else:
                counts[band] = (hits, misses + 1)
        return counts

    def band_hit_rates(self) -> Dict[int, float]:
        """
        Per-band hit rates over the window.

        Returns
        -------
        Dict[int, float]
            For each observed band, hit_rate in [0, 1].
        """
        counts = self.band_hit_counts()
        return {
            band: hits / (hits + misses) if (hits + misses) > 0 else 0.0
            for band, (hits, misses) in counts.items()
        }

    def most_active_band(self) -> Optional[int]:
        """
        Find the band with the most hits in the window.

        Returns
        -------
        int or None
            Band index with highest hit count, or None if no hits.
        """
        counts = self.band_hit_counts()
        if not counts:
            return None
        return max(counts.items(), key=lambda x: x[1][0])[0]

    def least_explored_bands(self, band_count: int, n: int = 5) -> List[int]:
        """
        Find bands that have been observed the least.

        Parameters
        ----------
        band_count : int
            Total number of bands.
        n : int
            Number of bands to return.

        Returns
        -------
        List[int]
            Bands sorted by observation count (ascending).
        """
        counts = self.band_hit_counts()
        all_bands = set(range(band_count))
        observed = set(counts.keys())
        unobserved = all_bands - observed

        # Unobserved bands come first
        result = sorted(unobserved)

        # Then observed bands by total visits (ascending)
        observed_with_counts = [
            (band, sum(counts[band]))
            for band in observed
        ]
        result.extend(band for band, _ in sorted(observed_with_counts, key=lambda x: x[1]))

        return result[:n]

    def time_since_last_hit(self, band: int) -> Optional[int]:
        """
        Slots elapsed since the last hit on a specific band.

        Parameters
        ----------
        band : int
            Band to check.

        Returns
        -------
        int or None
            Slots since last hit, or None if never hit.
        """
        obs = self._windowed_history()
        current_slot = obs[-1].get("time_slot", 0) if obs else 0

        last_hit_slot = None
        for o in reversed(obs):
            if o.get("selected_band") == band and o.get("hit", False):
                last_hit_slot = o.get("time_slot")
                break

        if last_hit_slot is None:
            return None
        return current_slot - last_hit_slot

    def staleness(self, band: int, decay: float = 0.05) -> float:
        """
        Compute staleness score for a band: higher = more stale.

        Parameters
        ----------
        band : int
            Band to score.
        decay : float
            Decay rate. Higher = staleness grows faster.

        Returns
        -------
        float
            Staleness score in [0, 1].
        """
        slots = self.time_since_last_hit(band)
        if slots is None:
            return 1.0  # Never observed = maximally stale
        return min(1.0, 1.0 - np.exp(-decay * slots))

    def get_snr_sequence(self, band: Optional[int] = None) -> List[float]:
        """
        Get SNR estimates from recent observations.

        Parameters
        ----------
        band : int, optional
            Filter to specific band. If None, all bands.

        Returns
        -------
        List[float]
            SNR estimates (dB), most recent last.
        """
        result = []
        for o in self._windowed_history():
            if band is not None and o.get("selected_band") != band:
                continue
            snr = o.get("snr_db_estimate")
            if snr is not None and not np.isnan(snr):
                result.append(float(snr))
        return result

    def get_feature_matrix(
        self,
        include_snr: bool = True,
        include_pulse_count: bool = True,
        include_aoa: bool = True,
    ) -> np.ndarray:
        """
        Build a feature matrix for sequence models (RNN/LSTM input).

        Parameters
        ----------
        include_snr : bool
            Include SNR estimate feature.
        include_pulse_count : bool
            Include pulse count feature.
        include_aoa : bool
            Include mean AoA feature.

        Returns
        -------
        ndarray, shape (n_observations, n_features)
            Feature matrix with one row per observation.
        """
        features = []
        for o in self._windowed_history():
            row = []

            # Hit indicator (always included)
            row.append(1.0 if o.get("hit", False) else 0.0)

            # Band as one-hot (sparse representation)
            band = o.get("selected_band", 0)
            row.append(float(band))

            # SNR
            if include_snr:
                snr = o.get("snr_db_estimate")
                row.append(float(snr) if snr is not None and not np.isnan(snr) else 0.0)

            # Pulse count
            if include_pulse_count:
                pc = o.get("pulse_count", 0)
                row.append(float(pc) if pc is not None else 0.0)

            # AoA
            if include_aoa:
                aoa = o.get("mean_aoa_deg")
                row.append(float(aoa) if aoa is not None and not np.isnan(aoa) else 0.0)

            features.append(row)

        return np.array(features, dtype=np.float32)

    def get_summary(self) -> ObservationSummary:
        """
        Get a compact summary of the observation context.

        Returns
        -------
        ObservationSummary
            Summary statistics.
        """
        obs = self._windowed_history()
        n_obs = len(obs)
        hit_rate = self.hit_rate()
        counts = self.band_hit_counts()

        return ObservationSummary(
            window_size=self._window or n_obs,
            n_observations=n_obs,
            hit_rate=hit_rate,
            n_unique_bands=len(counts),
            most_active_band=self.most_active_band(),
            recency_weighted_hit_rate=self.recency_weighted_hit_rate(),
        )


class SequentialObservationBuffer:
    """
    Stateful buffer that maintains rolling observation context.

    Unlike ObservationContext (which is a view over the scheduler's
    observation_history), this class maintains its own state and
    can be used by schedulers that need to track additional context
    without modifying their observation_history.

    Parameters
    ----------
    window : int
        Maximum number of observations to retain.
    band_count : int
        Number of bands in the environment.
    """

    def __init__(self, window: int, band_count: int):
        self.window = window
        self.band_count = band_count
        self._buffer: List[Dict] = []
        self._band_counts: Dict[int, Tuple[int, int]] = {
            b: (0, 0) for b in range(band_count)
        }

    def add(self, observation: Dict) -> None:
        """Add an observation to the buffer."""
        self._buffer.append(observation)
        if len(self._buffer) > self.window:
            # Evict oldest
            oldest = self._buffer.pop(0)
            band = oldest.get("selected_band")
            if band is not None:
                band = int(band)
                hits, misses = self._band_counts.get(band, (0, 0))
                if oldest.get("hit", False):
                    self._band_counts[band] = (max(0, hits - 1), misses)
                else:
                    self._band_counts[band] = (hits, max(0, misses - 1))

        # Update counts
        band = observation.get("selected_band")
        if band is not None:
            band = int(band)
            hits, misses = self._band_counts.get(band, (0, 0))
            if observation.get("hit", False):
                self._band_counts[band] = (hits + 1, misses)
            else:
                self._band_counts[band] = (hits, misses + 1)

    @property
    def buffer(self) -> List[Dict]:
        """Current buffer contents."""
        return list(self._buffer)

    @property
    def context(self) -> ObservationContext:
        """ObservationContext view over the buffer."""
        return ObservationContext(self._buffer)

    def band_hit_rate(self, band: int) -> float:
        """Hit rate for a specific band."""
        hits, misses = self._band_counts.get(band, (0, 0))
        total = hits + misses
        return hits / total if total > 0 else 0.0

    def all_band_hit_rates(self) -> np.ndarray:
        """All band hit rates as an array."""
        rates = np.zeros(self.band_count, dtype=np.float32)
        for band, (hits, misses) in self._band_counts.items():
            total = hits + misses
            rates[band] = hits / total if total > 0 else 0.0
        return rates

    def reset(self) -> None:
        """Clear the buffer and counts."""
        self._buffer.clear()
        self._band_counts = {b: (0, 0) for b in range(self.band_count)}


__all__ = [
    "ObservationContext",
    "ObservationSummary",
    "SequentialObservationBuffer",
]

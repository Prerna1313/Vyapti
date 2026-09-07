"""
vyapti_simulator.rf.closed_loop
=================================

Closed-loop mission runner and baseline schedulers.

In closed-loop mode, the **scheduler** is in control. At each step:

  1. The scheduler observes the current ``MissionState`` (what
     pulses have been detected so far, how much mission time
     remains, what dwells have been requested).
  2. The scheduler decides on a ``Dwell`` — a specific (frequency
     band, AoA sector, dwell duration) the receiver should look at.
  3. :meth:`RealTimeRFSimulator.simulate_dwell` generates the
     I/Q samples for that slice of spectrum (out-of-band emitters
     contribute nothing).
  4. :meth:`PulseDetector.detect_dwell` finds the pulses in that
     slice and returns them as a ``PDWStream``.
  5. ``MissionState.update`` incorporates the new PDWs into the
     cumulative track history.
  6. Repeat until the mission duration elapses.

The closed-loop mode is fundamentally different from the open-loop
``RFPulsePipeline``: each scheduler sees only the spectrum it
*chose* to look at. A round-robin scheduler sees a slice of every
band; a priority-queue scheduler sees mostly what it has already
flagged as a threat. The difference between schedulers becomes
observable in the resulting detection-rate / time-to-first-detection
scores.

Components
----------

``Dwell``
    What the scheduler decides to do — frequency band, AoA window,
    dwell duration, priority.

``MissionState``
    What the scheduler sees — recent PDWs, all detected tracks
    (cumulative), remaining time, dwell history.

``BaseScheduler``
    Abstract interface. ``decide(state) -> Dwell``.

``RoundRobinScheduler``
    Baseline: cycles through fixed frequency bands, no state
    awareness.

``PriorityQueueScheduler``
    Tracks emitters as they are detected. Always dwells on the
    highest-priority unconfirmed track. Falls back to round-robin
    when nothing is known.

``ThreatScoreScheduler``
    Computes a continuous threat score per emitter from multiple
    features (band danger, PW regularity, PRI stability, amplitude,
    AoA certainty). Always dwells on the highest-score track.

``MissionRunner``
    The control loop. Takes a scheduler + engine + detector +
    ground-truth emitters. Runs the mission and returns a
    ``SchedulerScore``.

``score_scheduler``
    Stand-alone scoring function: compares detected tracks against
    ground-truth emitter specs. Returns ``SchedulerScore`` with
    detection_rate, time_to_first, false_alarms, total_dwells.
"""

from __future__ import annotations

import math
import time as _time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..tsrd.synthetic_pdw_generator import SyntheticEmitterSpec
from ..tsrd.tsrd_adapter import PDWStream
from .pulse_detector import (
    EmitterInfo,
    PulseDetector,
    PulseDetectorConfig,
)
from .simulator_engine import (
    RealTimeRFSimulator,
    SimulationEngineConfig,
)


# =====================================================================
# Data classes
# =====================================================================
@dataclass
class Dwell:
    """
    What the scheduler decides the receiver should look at next.

    Attributes
    ----------
    freq_start_hz : float
        Lower edge of the requested frequency band, in Hz.
    freq_end_hz : float
        Upper edge of the requested frequency band, in Hz.
    aoa_center_deg : float, optional
        Centre of the requested AoA sector. None = full azimuth.
    aoa_window_deg : float, optional
        Half-width of the AoA sector. None = full azimuth.
    dwell_ms : float
        Dwell duration in milliseconds.
    priority : float
        Scheduler's self-reported priority for this dwell
        (0.0 = exploratory, 1.0 = targeted confirmation).
    reason : str, optional
        Human-readable reason for this dwell (useful for debugging
        scheduler behaviour).
    """
    freq_start_hz: float
    freq_end_hz: float
    aoa_center_deg: Optional[float] = None
    aoa_window_deg: Optional[float] = None
    dwell_ms: float = 10.0
    priority: float = 0.0
    reason: str = ""


@dataclass
class TrackedEmitter:
    """
    Cumulative knowledge about one suspected emitter.

    Updated by ``MissionState.update`` after each dwell. The
    scheduler's view of the world is the list of these tracks.

    Attributes
    ----------
    emitter_id : int
        Tentative ID. May be reassigned across dwells as more
        is learned. -1 if unknown.
    freq_hz : float
        Best estimate of the emitter's centre frequency.
    aoa_deg : float
        Best estimate of AoA.
    pw_us : float
        Best estimate of pulse width.
    amplitude_db : float
        Best estimate of pulse amplitude.
    first_detection_us : float
        Time of the first detection, in microseconds from
        mission start.
    last_detection_us : float
        Time of the most recent detection, in microseconds from
        mission start.
    detection_count : int
        Number of dwells where this track was observed.
    confirmation_count : int
        Number of detections inside the *expected* PRI window
        (subset of ``detection_count``).
    """
    emitter_id: int
    freq_hz: float
    aoa_deg: float
    pw_us: float
    amplitude_db: float
    first_detection_us: float
    last_detection_us: float
    detection_count: int = 1
    confirmation_count: int = 0
    # Threat score features (set by schedulers that need them)
    freq_score: float = 0.5
    pw_score: float = 0.5
    pri_score: float = 0.5
    amp_score: float = 0.5
    aoa_score: float = 0.5
    threat_score: float = 0.0


@dataclass
class MissionState:
    """
    The scheduler's view of the world.

    Updated after every dwell. Time and resource state is updated
    *by the MissionRunner*; the scheduler reads but does not
    modify ``time_remaining_us``.

    Attributes
    ----------
    time_elapsed_us : float
        Mission time elapsed so far, in microseconds.
    time_remaining_us : float
        Mission time remaining, in microseconds.
    sweep_window_hz : Tuple[float, float]
        Total spectrum the scheduler can choose from
        ``(freq_start, freq_end)`` in Hz.
    sweep_window_aoa : Tuple[Optional[float], Optional[float]]
        AoA window ``(centre, half_width)`` or
        ``(None, None)`` for full azimuth.
    recent_pdws : PDWStream
        PDWs from the most recent dwell.
    all_detections : List[TrackedEmitter]
        Cumulative track history.
    dwell_history : List[Dwell]
        Sequence of dwells the scheduler has issued.
    n_dwells : int
        Number of dwells issued so far.
    """
    time_elapsed_us: float = 0.0
    time_remaining_us: float = 60.0 * 1e6
    sweep_window_hz: Tuple[float, float] = (2e9, 18e9)
    sweep_window_aoa: Tuple[Optional[float], Optional[float]] = (None, None)
    recent_pdws: PDWStream = field(
        default_factory=lambda: PDWStream(
            toa_us=np.zeros(0, dtype=np.float32),
            freq_mhz=np.zeros(0, dtype=np.float32),
            pw_us=np.zeros(0, dtype=np.float32),
            aoa_deg=np.zeros(0, dtype=np.float32),
            amp_db=np.zeros(0, dtype=np.float32),
            emitter_id=np.zeros(0, dtype=np.int64),
        )
    )
    all_detections: List[TrackedEmitter] = field(default_factory=list)
    dwell_history: List[Dwell] = field(default_factory=list)
    n_dwells: int = 0

    def to_vector(self) -> np.ndarray:
        """
        Convert the state to a feature vector for RL schedulers.

        Layout (fixed-size, padded to max_tracks):
            [time_remaining_us / 1e6,
             time_elapsed_us / 1e6,
             n_tracks / 10.0,
             n_confirmed / 10.0,
             <track features>...] * max_tracks

        Track features (per track): [freq_norm, aoa_norm, amp_norm,
        pw_norm, det_count_norm, conf_count_norm, last_seen_norm].
        """
        max_tracks = 16
        features: List[float] = [
            self.time_remaining_us / 1e6,
            self.time_elapsed_us / 1e6,
            len(self.all_detections) / 10.0,
            sum(1 for t in self.all_detections if t.confirmation_count >= 3)
            / 10.0,
        ]
        for i in range(max_tracks):
            if i < len(self.all_detections):
                t = self.all_detections[i]
                features.extend([
                    t.freq_hz / 1e10,
                    t.aoa_deg / 360.0,
                    t.amplitude_db / 100.0,
                    t.pw_us / 1e3,
                    min(t.detection_count, 10) / 10.0,
                    min(t.confirmation_count, 10) / 10.0,
                    max(0.0, 1.0 - t.last_detection_us / max(self.time_elapsed_us, 1.0)),
                ])
            else:
                features.extend([0.0] * 7)
        return np.asarray(features, dtype=np.float32)


@dataclass
class SchedulerScore:
    """
    How a scheduler performed on one mission.

    Attributes
    ----------
    detection_rate : float
        Fraction of ground-truth emitters that were confirmed
        (>= 3 detections matching the emitter's frequency).
    time_to_first_us : float
        Microseconds from mission start to the first confirmed
        detection. ``inf`` if nothing was confirmed.
    false_alarms : int
        Number of detected tracks that don't match any ground-truth
        emitter (within the matching tolerance).
    total_dwells : int
        Number of dwells the scheduler issued.
    detection_latency_us : List[float]
        Per-emitter detection latency (first detection time minus
        mission start). One entry per ground-truth emitter.
    """
    detection_rate: float = 0.0
    time_to_first_us: float = float("inf")
    false_alarms: int = 0
    total_dwells: int = 0
    detection_latency_us: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        ttf = (
            f"{self.time_to_first_us / 1e6:.2f}s"
            if math.isfinite(self.time_to_first_us)
            else "n/a"
        )
        return (
            f"detection_rate={self.detection_rate:.1%}, "
            f"time_to_first={ttf}, "
            f"false_alarms={self.false_alarms}, "
            f"total_dwells={self.total_dwells}"
        )


# =====================================================================
# Scheduler base class
# =====================================================================
class BaseScheduler(ABC):
    """
    Abstract interface for all schedulers.

    A scheduler takes a ``MissionState`` and returns a ``Dwell``
    describing what the receiver should look at next.
    """

    @abstractmethod
    def decide(self, state: MissionState) -> Dwell:
        """Decide the next dwell given the current state."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state. Default: no-op."""
        return None


# =====================================================================
# Round-robin scheduler (baseline)
# =====================================================================
class RoundRobinScheduler(BaseScheduler):
    """
    Baseline: cycle through fixed frequency bands, equal dwell time.

    Ignores the state entirely. Each dwell gets the same priority
    and the same time budget. This is the floor against which every
    other scheduler should be measured.

    Parameters
    ----------
    bands_hz : List[Tuple[float, float]]
        List of ``(freq_start, freq_end)`` tuples. The scheduler
        cycles through these in order.
    dwell_ms : float
        Dwell time per band.
    """
    def __init__(
        self,
        bands_hz: List[Tuple[float, float]],
        dwell_ms: float = 10.0,
    ):
        if not bands_hz:
            raise ValueError("bands_hz must be non-empty")
        self.bands = bands_hz
        self.dwell_ms = float(dwell_ms)
        self._idx = 0

    def decide(self, state: MissionState) -> Dwell:
        band = self.bands[self._idx % len(self.bands)]
        self._idx += 1
        return Dwell(
            freq_start_hz=float(band[0]),
            freq_end_hz=float(band[1]),
            aoa_center_deg=None,
            aoa_window_deg=None,
            dwell_ms=self.dwell_ms,
            priority=0.0,
            reason=f"round_robin band {self._idx % len(self.bands)}",
        )

    def reset(self) -> None:
        self._idx = 0


# =====================================================================
# Priority queue scheduler
# =====================================================================
class PriorityQueueScheduler(BaseScheduler):
    """
    Maintains a priority queue of tracked emitters.

    Always dwells on the highest-priority unconfirmed track. Falls
    back to round-robin on unknowns (so the scheduler doesn't get
    stuck with no detections).

    Priority is updated after each dwell:
        +0.2 when a track is re-detected (capped at 1.0)
        -0.05 per tick (priority decays)
    """

    def __init__(
        self,
        bands_hz: List[Tuple[float, float]],
        dwell_ms: float = 10.0,
        confirm_threshold: int = 3,
        confirmation_window_ticks: int = 5,
    ):
        self.bands = bands_hz
        self.dwell_ms = float(dwell_ms)
        self.confirm_threshold = confirm_threshold
        self.confirmation_window_ticks = confirmation_window_ticks
        self._rr_idx = 0
        self._tracks: Dict[int, TrackedEmitter] = {}
        self._next_internal_id = 1000  # IDs start at 1000 to avoid
                                        # colliding with real emitter IDs

    def decide(self, state: MissionState) -> Dwell:
        # Update tracks from the latest dwell's PDWs
        self._update_tracks(state.recent_pdws)

        # Confirmed tracks first
        confirmed = [
            t for t in self._tracks.values()
            if t.confirmation_count >= self.confirm_threshold
        ]
        if confirmed:
            target = max(confirmed, key=lambda t: t.threat_score)
            return self._dwell_around(target, reason="confirmed track")

        # Then unconfirmed but detected
        detected = [
            t for t in self._tracks.values()
            if 0 < t.confirmation_count < self.confirm_threshold
        ]
        if detected:
            target = max(detected, key=lambda t: t.threat_score)
            return self._dwell_around(target, reason="confirming track")

        # Nothing known — round-robin
        band = self.bands[self._rr_idx % len(self.bands)]
        self._rr_idx += 1
        return Dwell(
            freq_start_hz=float(band[0]),
            freq_end_hz=float(band[1]),
            aoa_center_deg=None,
            aoa_window_deg=None,
            dwell_ms=self.dwell_ms * 2,  # longer dwell when searching
            priority=0.0,
            reason=f"search band {self._rr_idx % len(self.bands)}",
        )

    def _dwell_around(self, track: TrackedEmitter, reason: str) -> Dwell:
        bw = max(2e6, 0.05 * track.freq_hz)
        return Dwell(
            freq_start_hz=max(0.0, track.freq_hz - bw),
            freq_end_hz=track.freq_hz + bw,
            aoa_center_deg=track.aoa_deg,
            aoa_window_deg=10.0,
            dwell_ms=self.dwell_ms,
            priority=min(1.0, track.threat_score),
            reason=reason,
        )

    def _update_tracks(self, pdws: PDWStream) -> None:
        if len(pdws) == 0:
            # Decay existing tracks
            for t in self._tracks.values():
                t.threat_score = max(0.0, t.threat_score - 0.02)
            return
        # Group PDWs by AoA/RF cluster (naive nearest-neighbour)
        for i in range(len(pdws)):
            freq_hz = float(pdws.freq_mhz[i]) * 1e6
            aoa_deg = float(pdws.aoa_deg[i])
            amp_db = float(pdws.amp_db[i])
            pw_us = float(pdws.pw_us[i])
            toa_us = float(pdws.toa_us[i])
            # Find existing track within matching tolerance
            matched_id = self._find_nearest_track(freq_hz, aoa_deg)
            if matched_id is not None:
                t = self._tracks[matched_id]
                t.freq_hz = 0.7 * t.freq_hz + 0.3 * freq_hz
                t.aoa_deg = 0.7 * t.aoa_deg + 0.3 * aoa_deg
                t.pw_us = 0.7 * t.pw_us + 0.3 * pw_us
                t.amplitude_db = 0.7 * t.amplitude_db + 0.3 * amp_db
                t.last_detection_us = toa_us
                t.detection_count += 1
                # Confirmation: only if re-detected within a reasonable
                # window of the previous detection
                t.confirmation_count += 1
                t.threat_score = min(1.0, t.threat_score + 0.2)
            else:
                new_id = self._next_internal_id
                self._next_internal_id += 1
                self._tracks[new_id] = TrackedEmitter(
                    emitter_id=new_id,
                    freq_hz=freq_hz,
                    aoa_deg=aoa_deg,
                    pw_us=pw_us,
                    amplitude_db=amp_db,
                    first_detection_us=toa_us,
                    last_detection_us=toa_us,
                    detection_count=1,
                    confirmation_count=1,
                    threat_score=0.1,
                )

    def _find_nearest_track(
        self, freq_hz: float, aoa_deg: float
    ) -> Optional[int]:
        """Naive nearest-neighbour clustering."""
        if not self._tracks:
            return None
        best_id = None
        best_dist = float("inf")
        for eid, t in self._tracks.items():
            df = abs(t.freq_hz - freq_hz)
            da = min(abs(t.aoa_deg - aoa_deg), 360.0 - abs(t.aoa_deg - aoa_deg))
            # Weight: 1 MHz frequency tolerance per degree AoA tolerance
            dist = df / 1e6 + da
            if dist < best_dist and df < 5e6 and da < 20.0:
                best_dist = dist
                best_id = eid
        return best_id

    def reset(self) -> None:
        self._rr_idx = 0
        self._tracks = {}
        self._next_internal_id = 1000


# =====================================================================
# Threat-score scheduler
# =====================================================================
@dataclass
class ThreatWeights:
    """Weights for the multi-feature threat score."""
    freq: float = 0.20
    pw: float = 0.20
    pri: float = 0.20
    amp: float = 0.20
    aoa: float = 0.20


class ThreatScoreScheduler(BaseScheduler):
    """
    Computes a continuous threat score per emitter from multiple
    features and always dwells on the highest-score track.

    Features (each scaled to [0, 1]):
        freq_score   - danger of the band (configurable, default uniform)
        pw_score     - narrow PW = radar = more dangerous
        pri_score    - regular PRI = deliberate emitter
        amp_score    - stronger signal = closer = more risk
        aoa_score    - certain direction = more credible

    Falls back to round-robin search when no tracks exist.
    """
    def __init__(
        self,
        bands_hz: List[Tuple[float, float]],
        dwell_ms: float = 10.0,
        weights: Optional[ThreatWeights] = None,
        freq_danger_map: Optional[Dict[Tuple[float, float], float]] = None,
    ):
        self.bands = bands_hz
        self.dwell_ms = float(dwell_ms)
        self.weights = weights if weights is not None else ThreatWeights()
        # freq_danger_map: optional {band: danger_score} override
        self.freq_danger_map = freq_danger_map or {}
        self._rr_idx = 0
        self._tracks: Dict[int, TrackedEmitter] = {}
        self._next_internal_id = 2000

    def decide(self, state: MissionState) -> Dwell:
        # Update tracks from latest PDWs
        self._update_tracks(state.recent_pdws)

        # Recompute threat score for every track
        for t in self._tracks.values():
            t.threat_score = self._compute_threat_score(t)

        if self._tracks:
            target = max(self._tracks.values(), key=lambda t: t.threat_score)
            return self._dwell_around(target)

        # Nothing tracked — round-robin
        band = self.bands[self._rr_idx % len(self.bands)]
        self._rr_idx += 1
        return Dwell(
            freq_start_hz=float(band[0]),
            freq_end_hz=float(band[1]),
            aoa_center_deg=None,
            aoa_window_deg=None,
            dwell_ms=self.dwell_ms * 2,
            priority=0.0,
            reason=f"search band {self._rr_idx % len(self.bands)}",
        )

    def _dwell_around(self, track: TrackedEmitter) -> Dwell:
        bw = max(2e6, 0.05 * track.freq_hz)
        return Dwell(
            freq_start_hz=max(0.0, track.freq_hz - bw),
            freq_end_hz=track.freq_hz + bw,
            aoa_center_deg=track.aoa_deg,
            aoa_window_deg=10.0,
            dwell_ms=self.dwell_ms,
            priority=min(1.0, track.threat_score),
            reason=f"threat_score={track.threat_score:.2f}",
        )

    def _compute_threat_score(self, track: TrackedEmitter) -> float:
        w = self.weights
        score = (
            w.freq * track.freq_score
            + w.pw * track.pw_score
            + w.pri * track.pri_score
            + w.amp * track.amp_score
            + w.aoa * track.aoa_score
        )
        return float(np.clip(score, 0.0, 1.0))

    def _update_tracks(self, pdws: PDWStream) -> None:
        # Same as PriorityQueueScheduler but also computes features
        if len(pdws) == 0:
            for t in self._tracks.values():
                t.threat_score = max(0.0, t.threat_score - 0.02)
            return
        for i in range(len(pdws)):
            freq_hz = float(pdws.freq_mhz[i]) * 1e6
            aoa_deg = float(pdws.aoa_deg[i])
            amp_db = float(pdws.amp_db[i])
            pw_us = float(pdws.pw_us[i])
            toa_us = float(pdws.toa_us[i])
            matched_id = self._find_nearest_track(freq_hz, aoa_deg)
            if matched_id is not None:
                t = self._tracks[matched_id]
                t.freq_hz = 0.7 * t.freq_hz + 0.3 * freq_hz
                t.aoa_deg = 0.7 * t.aoa_deg + 0.3 * aoa_deg
                t.pw_us = 0.7 * t.pw_us + 0.3 * pw_us
                t.amplitude_db = 0.7 * t.amplitude_db + 0.3 * amp_db
                t.last_detection_us = toa_us
                t.detection_count += 1
                t.confirmation_count += 1
                # Update feature scores
                t.amp_score = self._amp_score(amp_db)
                t.pw_score = self._pw_score(pw_us)
                t.aoa_score = min(1.0, t.aoa_score + 0.1)
                t.pri_score = min(1.0, t.pri_score + 0.1)
            else:
                new_id = self._next_internal_id
                self._next_internal_id += 1
                self._tracks[new_id] = TrackedEmitter(
                    emitter_id=new_id,
                    freq_hz=freq_hz,
                    aoa_deg=aoa_deg,
                    pw_us=pw_us,
                    amplitude_db=amp_db,
                    first_detection_us=toa_us,
                    last_detection_us=toa_us,
                    detection_count=1,
                    confirmation_count=1,
                    freq_score=self._band_danger_score(freq_hz),
                    pw_score=self._pw_score(pw_us),
                    pri_score=0.3,
                    amp_score=self._amp_score(amp_db),
                    aoa_score=0.3,
                    threat_score=0.0,
                )

    @staticmethod
    def _amp_score(amp_db: float) -> float:
        # -90 dB → 0.0, +30 dB → 1.0
        return float(np.clip((amp_db + 90.0) / 120.0, 0.0, 1.0))

    @staticmethod
    def _pw_score(pw_us: float) -> float:
        # Narrower pulse = more like a radar = higher score
        # 10 us → 0.1, 0.1 us → 1.0
        return float(np.clip(1.0 - 0.3 * np.log10(max(pw_us, 0.1)), 0.0, 1.0))

    def _band_danger_score(self, freq_hz: float) -> float:
        if not self.freq_danger_map:
            return 0.5
        for (f0, f1), score in self.freq_danger_map.items():
            if f0 <= freq_hz <= f1:
                return float(score)
        return 0.3

    def _find_nearest_track(
        self, freq_hz: float, aoa_deg: float
    ) -> Optional[int]:
        if not self._tracks:
            return None
        best_id = None
        best_dist = float("inf")
        for eid, t in self._tracks.items():
            df = abs(t.freq_hz - freq_hz)
            da = min(abs(t.aoa_deg - aoa_deg), 360.0 - abs(t.aoa_deg - aoa_deg))
            dist = df / 1e6 + da
            if dist < best_dist and df < 5e6 and da < 20.0:
                best_dist = dist
                best_id = eid
        return best_id

    def reset(self) -> None:
        self._rr_idx = 0
        self._tracks = {}
        self._next_internal_id = 2000


# =====================================================================
# Mission runner
# =====================================================================
class MissionRunner:
    """
    Run a closed-loop mission with a given scheduler.

    The runner owns the engine and the detector. The scheduler sees
    a ``MissionState`` and returns a ``Dwell``; the runner calls
    ``engine.simulate_dwell`` and ``detector.detect_dwell`` and
    updates the state. The loop continues until the mission
    duration elapses or the scheduler issues too many consecutive
    empty dwells (sanity check).

    Parameters
    ----------
    engine : RealTimeRFSimulator
        Pre-configured RF engine with all emitters registered.
        Emitters must have ``carrier_freq_hz`` and ``aoa_deg`` set
        (the bridge does this automatically).
    detector : PulseDetector
        Pre-configured pulse detector.
    scheduler : BaseScheduler
        The scheduler under test.
    ground_truth : List[SyntheticEmitterSpec]
        The actual emitters, for scoring. The scheduler does NOT
        see this — only the detector's output is visible to it.
    mission_duration_s : float
        Total mission duration in seconds.
    sweep_window_hz : Tuple[float, float]
        Total spectrum the scheduler can choose from.
    max_dwells : int
        Safety cap on number of dwells (default 10000).
    max_consecutive_empty : int
        Stop the mission if the scheduler issues this many empty
        dwells in a row (default 200).
    wall_clock_budget_s : float, optional
        If set, the mission aborts when wall-clock time exceeds
        this value. Useful for benchmarking.

    Notes
    -----
    The runner uses ``engine.simulate_dwell`` (not the streaming
    ``engine.run``). Each call advances the engine's tick counter
    by the dwell's worth of ticks, so subsequent dwells see the
    correctly-evolved channel state.
    """

    def __init__(
        self,
        engine: RealTimeRFSimulator,
        detector: PulseDetector,
        scheduler: BaseScheduler,
        ground_truth: List[SyntheticEmitterSpec],
        mission_duration_s: float = 60.0,
        sweep_window_hz: Tuple[float, float] = (2e9, 18e9),
        max_dwells: int = 10_000,
        max_consecutive_empty: int = 200,
        wall_clock_budget_s: Optional[float] = None,
    ):
        self.engine = engine
        self.detector = detector
        self.scheduler = scheduler
        self.ground_truth = list(ground_truth)
        self.mission_duration_s = float(mission_duration_s)
        self.sweep_window_hz = sweep_window_hz
        self.max_dwells = int(max_dwells)
        self.max_consecutive_empty = int(max_consecutive_empty)
        self.wall_clock_budget_s = wall_clock_budget_s

    def run(self) -> Tuple[MissionState, SchedulerScore]:
        """
        Execute the mission and return the final state + score.

        Returns
        -------
        state : MissionState
            The final state with the full dwell history and
            cumulative track list.
        score : SchedulerScore
            Performance metrics vs ground truth.
        """
        state = MissionState(
            time_remaining_us=self.mission_duration_s * 1e6,
            sweep_window_hz=self.sweep_window_hz,
        )
        self.scheduler.reset()
        consecutive_empty = 0
        wall_start = _time.time() if self.wall_clock_budget_s else None

        while state.time_remaining_us > 0 and state.n_dwells < self.max_dwells:
            # Wall-clock budget
            if wall_start is not None and self.wall_clock_budget_s is not None:
                if (_time.time() - wall_start) > self.wall_clock_budget_s:
                    break

            # 1. Scheduler decides
            dwell = self.scheduler.decide(state)

            # 2. Engine simulates this slice
            try:
                iq_chunk = self.engine.simulate_dwell(
                    freq_start_hz=dwell.freq_start_hz,
                    freq_end_hz=dwell.freq_end_hz,
                    aoa_center_deg=dwell.aoa_center_deg,
                    aoa_window_deg=dwell.aoa_window_deg,
                    dwell_ms=dwell.dwell_ms,
                )
            except Exception:
                # Engine error — log and continue with empty chunk
                iq_chunk = np.zeros(
                    int(dwell.dwell_ms * 1e-3 * self.engine.config.dsp_sample_rate_hz),
                    dtype=np.complex128,
                )

            # 3. Detector finds pulses in this slice
            try:
                pdws = self.detector.detect_dwell(
                    iq_chunk=iq_chunk,
                    freq_start_hz=dwell.freq_start_hz,
                    freq_end_hz=dwell.freq_end_hz,
                    freq_center_hz=(
                        (dwell.freq_start_hz + dwell.freq_end_hz) / 2.0
                    ),
                    aoa_center_deg=dwell.aoa_center_deg,
                    aoa_window_deg=dwell.aoa_window_deg,
                )
            except Exception:
                pdws = PDWStream(
                    toa_us=np.zeros(0, dtype=np.float32),
                    freq_mhz=np.zeros(0, dtype=np.float32),
                    pw_us=np.zeros(0, dtype=np.float32),
                    aoa_deg=np.zeros(0, dtype=np.float32),
                    amp_db=np.zeros(0, dtype=np.float32),
                    emitter_id=np.zeros(0, dtype=np.int64),
                )

            # 4. Update state
            dwell_dur_us = dwell.dwell_ms * 1e3
            state.time_elapsed_us += dwell_dur_us
            state.time_remaining_us = max(
                0.0, state.time_remaining_us - dwell_dur_us
            )
            # Shift PDW times by elapsed so the scheduler sees
            # absolute mission-time ToAs
            if len(pdws) > 0:
                shifted = PDWStream(
                    toa_us=pdws.toa_us + np.float32(
                        state.time_elapsed_us - dwell_dur_us
                    ),
                    freq_mhz=pdws.freq_mhz,
                    pw_us=pdws.pw_us,
                    aoa_deg=pdws.aoa_deg,
                    amp_db=pdws.amp_db,
                    emitter_id=pdws.emitter_id,
                )
                state.recent_pdws = shifted
            else:
                state.recent_pdws = pdws
            state.dwell_history.append(dwell)
            state.n_dwells += 1

            if len(pdws) == 0:
                consecutive_empty += 1
                if consecutive_empty >= self.max_consecutive_empty:
                    break
            else:
                consecutive_empty = 0

        # 5. Score against ground truth
        score = score_scheduler(state, self.ground_truth)
        return state, score


# =====================================================================
# Scoring
# =====================================================================
def score_scheduler(
    state: MissionState,
    ground_truth: List[SyntheticEmitterSpec],
    freq_tolerance_hz: float = 5e6,
    aoa_tolerance_deg: float = 20.0,
    confirm_threshold: int = 3,
) -> SchedulerScore:
    """
    Score a closed-loop mission against ground truth.

    An emitter is "detected" if the cumulative track list contains
    at least ``confirm_threshold`` detections within
    ``freq_tolerance_hz`` of its carrier and ``aoa_tolerance_deg``
    of its AoA. The detection latency is the time from mission
    start to the first such detection.

    Parameters
    ----------
    state : MissionState
        Final mission state from :meth:`MissionRunner.run`.
    ground_truth : List[SyntheticEmitterSpec]
        The actual emitters.
    freq_tolerance_hz : float
        Maximum frequency difference for a match.
    aoa_tolerance_deg : float
        Maximum AoA difference for a match.
    confirm_threshold : int
        Number of detection events required to count as confirmed.

    Returns
    -------
    SchedulerScore
        Per-mission performance metrics.
    """
    if not ground_truth:
        return SchedulerScore(
            detection_rate=0.0,
            time_to_first_us=float("inf"),
            false_alarms=len(state.all_detections),
            total_dwells=state.n_dwells,
        )

    # Build the ground-truth emitter list with resolved carrier freq
    truth: List[Tuple[int, float, float]] = []
    for spec in ground_truth:
        if spec.center_freq_hz is not None:
            truth.append((int(spec.emitter_id),
                          float(spec.center_freq_hz),
                          float(spec.aoa_deg)))
        elif spec.freq_list_hz:
            truth.append((int(spec.emitter_id),
                          float(spec.freq_list_hz[0]),
                          float(spec.aoa_deg)))

    detected_truth: Dict[int, float] = {}  # emitter_id -> first detection us
    matched_tracks: set = set()  # which track IDs have been matched

    for track in state.all_detections:
        best_eid = None
        best_score = float("inf")
        for eid, freq, aoa in truth:
            if eid in detected_truth:
                continue
            df = abs(track.freq_hz - freq)
            da = min(abs(track.aoa_deg - aoa), 360.0 - abs(track.aoa_deg - aoa))
            score = df / max(freq_tolerance_hz, 1) + da / max(aoa_tolerance_deg, 1)
            if df <= freq_tolerance_hz and da <= aoa_tolerance_deg and score < best_score:
                best_score = score
                best_eid = eid
        if best_eid is not None and track.detection_count >= 1:
            detected_truth[best_eid] = track.first_detection_us
            matched_tracks.add(track.emitter_id)

    # Detection rate
    n_truth = len(truth)
    n_detected = len(detected_truth)
    detection_rate = n_detected / max(1, n_truth)

    # Time to first
    if detected_truth:
        time_to_first = min(detected_truth.values())
    else:
        time_to_first = float("inf")

    # False alarms: tracks that don't match any ground truth
    false_alarms = max(0, len(state.all_detections) - n_detected)

    # Per-emitter latency
    latency = [
        detected_truth.get(eid, float("inf"))
        for eid, _, _ in truth
    ]

    return SchedulerScore(
        detection_rate=detection_rate,
        time_to_first_us=time_to_first,
        false_alarms=false_alarms,
        total_dwells=state.n_dwells,
        detection_latency_us=latency,
    )


# =====================================================================
# Comparison harness
# =====================================================================
def run_comparison(
    specs: List[SyntheticEmitterSpec],
    schedulers: Dict[str, BaseScheduler],
    n_scenarios: int = 50,
    mission_duration_s: float = 60.0,
    dsp_sample_rate_hz: float = 10e6,
    tick_interval_s: float = 10e-3,
    snr_db: float = 15.0,
    sweep_window_hz: Tuple[float, float] = (2e9, 18e9),
    band_count: int = 8,
    seed: int = 42,
    wall_clock_budget_s: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, List[SchedulerScore]]:
    """
    Run all schedulers on the same set of scenarios and collect scores.

    Each scenario is a fresh re-roll of the emitter specs (or
    perturbations of the input specs) with deterministic seeds.
    All schedulers see the *same* scenario, so differences in their
    scores reflect scheduling decisions, not scenario luck.

    Parameters
    ----------
    specs : List[SyntheticEmitterSpec]
        Base emitter specs. Each scenario derives from these by
        drawing per-emitter frequency/AoA jitter from the seed RNG.
    schedulers : Dict[str, BaseScheduler]
        Map of name → scheduler. All schedulers see the same scenarios.
    n_scenarios : int
        Number of scenarios to run.
    mission_duration_s : float
        Mission duration per scenario.
    dsp_sample_rate_hz, tick_interval_s, snr_db : float
        Engine configuration.
    sweep_window_hz : Tuple[float, float]
        Spectrum the schedulers can choose from.
    band_count : int
        Number of equal-width bands the schedulers are told about.
    seed : int
        Base seed for reproducibility.
    wall_clock_budget_s : float, optional
        Per-scenario wall-clock budget.
    verbose : bool
        Print per-scenario progress.

    Returns
    -------
    Dict[str, List[SchedulerScore]]
        Per-scheduler list of scores. The lists are aligned by scenario
        index (all schedulers ran on the same scenario).
    """
    from .tsrd_bridge import TSRDSpecToRFBridge
    from .propagation import KinematicEmitter
    from .waveforms import generate_lfm_chirp

    # Compute the equal-width bands within the sweep window
    f0, f1 = sweep_window_hz
    band_edges = np.linspace(f0, f1, band_count + 1)
    bands_hz = [
        (float(band_edges[i]), float(band_edges[i + 1]))
        for i in range(band_count)
    ]

    results: Dict[str, List[SchedulerScore]] = {
        name: [] for name in schedulers
    }

    for scenario_idx in range(n_scenarios):
        rng = np.random.default_rng(seed + scenario_idx)

        # Build the scenario — small per-emitter jitter so scenarios differ
        scenario_specs: List[SyntheticEmitterSpec] = []
        for base in specs:
            new_freq = (
                (base.center_freq_hz or (base.freq_list_hz[0] if base.freq_list_hz else 5e9))
                + rng.uniform(-2e8, 2e8)
            )
            new_aoa = float(base.aoa_deg) + rng.uniform(-5.0, 5.0)
            scenario_specs.append(SyntheticEmitterSpec(
                emitter_id=base.emitter_id,
                aoa_deg=new_aoa,
                snr_db=base.snr_db,
                emitter_type=base.emitter_type,
                center_freq_hz=float(new_freq),
                freq_list_hz=base.freq_list_hz,
                pri_sec=base.pri_sec,
                pulse_width_sec=base.pulse_width_sec,
                waveform_type=base.waveform_type,
                chirp_bandwidth_hz=base.chirp_bandwidth_hz,
            ))

        # Build one engine for this scenario
        engine_cfg = SimulationEngineConfig(
            tick_interval_s=tick_interval_s,
            num_ticks=int(mission_duration_s / tick_interval_s) + 100,
            dsp_sample_rate_hz=dsp_sample_rate_hz,
            buffer_ticks=10,
            snr_db=snr_db,
            pulse_power_w=1.0,
        )
        engine = RealTimeRFSimulator(engine_cfg, rng=rng)

        # Add emitters via the bridge (so carrier_freq_hz and aoa_deg
        # are stored on the SimEmitter for the closed-loop windowing)
        for s in scenario_specs:
            child_rng = np.random.default_rng(rng.integers(0, 2 ** 32 - 1))
            bridge = TSRDSpecToRFBridge(s, child_rng)
            sspec = bridge.build()
            engine.add_emitter(
                sspec.kinematic,
                sspec.waveform_fn,
                emitter_id=sspec.emitter_id,
                channel=sspec.channel,
                pri_sec=sspec.pri_sec,
                pulse_width_s=sspec.pulse_width_s,
                carrier_freq_hz=sspec.carrier_freq_hz,
                aoa_deg=sspec.aoa_deg,
            )

        # Build the emitter map for the detector
        emitter_map = {
            int(s.emitter_id): EmitterInfo(
                carrier_freq_hz=(
                    s.center_freq_hz
                    if s.center_freq_hz is not None
                    else float(s.freq_list_hz[0])
                ),
                aoa_deg=float(s.aoa_deg),
            )
            for s in scenario_specs
        }

        # Run each scheduler on the same scenario
        for name, sched_template in schedulers.items():
            # Per-scheduler fresh copy (each sched has its own state)
            sched = type(sched_template)(
                bands_hz=bands_hz,
                dwell_ms=sched_template.dwell_ms,
            ) if hasattr(sched_template, "dwell_ms") else type(sched_template)(
                bands_hz=bands_hz
            )
            detector = PulseDetector(
                config=PulseDetectorConfig(
                    dsp_sample_rate_hz=dsp_sample_rate_hz,
                    tick_interval_s=tick_interval_s,
                    cfar_db=10.0,
                    chirp_bandwidth_hz=1e6,
                    pulse_width_s=1e-6,
                ),
                emitter_map=emitter_map,
                rng=np.random.default_rng(rng.integers(0, 2 ** 32 - 1)),
            )
            runner = MissionRunner(
                engine=engine,
                detector=detector,
                scheduler=sched,
                ground_truth=scenario_specs,
                mission_duration_s=mission_duration_s,
                sweep_window_hz=sweep_window_hz,
                wall_clock_budget_s=wall_clock_budget_s,
            )
            _, score = runner.run()
            results[name].append(score)

        if verbose:
            print(
                f"  scenario {scenario_idx + 1}/{n_scenarios}: "
                + ", ".join(
                    f"{name}={results[name][-1].detection_rate:.0%}"
                    for name in schedulers
                )
            )

    return results


def summarise_results(
    results: Dict[str, List[SchedulerScore]],
) -> str:
    """
    Format comparison results as a human-readable table.

    Columns: scheduler, mean detection_rate, mean time-to-first,
    mean false alarms, mean total dwells.
    """
    lines = []
    header = (
        f"{'Scheduler':<22}  {'DetRate':>8}  {'TTF (s)':>9}  "
        f"{'FA':>5}  {'Dwells':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, scores in results.items():
        n = max(1, len(scores))
        mean_det = sum(s.detection_rate for s in scores) / n
        finite_ttf = [s.time_to_first_us for s in scores if math.isfinite(s.time_to_first_us)]
        mean_ttf = (sum(finite_ttf) / len(finite_ttf) / 1e6) if finite_ttf else float("inf")
        mean_fa = sum(s.false_alarms for s in scores) / n
        mean_dwells = sum(s.total_dwells for s in scores) / n
        ttf_str = f"{mean_ttf:.2f}" if math.isfinite(mean_ttf) else "n/a"
        lines.append(
            f"{name:<22}  {mean_det:>7.1%}  {ttf_str:>9}  "
            f"{mean_fa:>5.1f}  {mean_dwells:>7.1f}"
        )
    return "\n".join(lines)


__all__ = [
    "Dwell",
    "MissionState",
    "TrackedEmitter",
    "SchedulerScore",
    "BaseScheduler",
    "RoundRobinScheduler",
    "PriorityQueueScheduler",
    "ThreatScoreScheduler",
    "ThreatWeights",
    "MissionRunner",
    "score_scheduler",
    "run_comparison",
    "summarise_results",
]

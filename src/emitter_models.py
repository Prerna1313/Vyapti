"""
Emitter models for pulse-level RF simulation.
Each emitter type generates realistic radar pulse trains.

The module has two layers:
  1. Seven base emitter classes (Fixed, Intermittent, Scanning, Agile, ...,
     plus StaggeredPri). Each generates a static pulse train across the
     entire mission. The Pulse amplitude reflects a free-space path-loss
     model, Rayleigh fading, and atmospheric attenuation.
  2. A DynamicEmitter wrapper + LifecyclePolicy hierarchy that
     adds realism on top: delayed arrival, regime change, non-uniform
     dwell, random ON/OFF intervals, etc.

Determinism: every randomness source comes from a `np.random.Generator`
that the simulator passes in. Same (config, seed) -> identical pulse
trains.
"""
from __future__ import annotations

import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod


# =====================================================================
# Channel model: path loss, fading, atmospheric attenuation
# =====================================================================

def free_space_path_loss_db(distance_m: float, frequency_hz: float) -> float:
    """
    Free-space path loss in dB.
        FSPL(dB) = 20*log10(d_m) + 20*log10(f_Hz) - 147.55
    """
    if distance_m <= 0 or frequency_hz <= 0:
        return 0.0
    return 20.0 * np.log10(distance_m) + 20.0 * np.log10(frequency_hz) - 147.55


def atmospheric_attenuation_db(frequency_hz: float) -> float:
    """
    Approximate one-way atmospheric attenuation (clear air, sea level,
    0 dB/km at lower bands, rising to ~0.5 dB/km at 18 GHz).
    This is a coarse model; for a real EW sim you'd use ITU-R P.676.
    """
    f_ghz = frequency_hz / 1e9
    if f_ghz < 1.0:
        return 0.0
    # Linear ramp from 0 dB at 1 GHz to ~0.5 dB/km at 18 GHz
    return 0.5 * (f_ghz - 1.0) / 17.0


def rayleigh_fading_db(rng: np.random.Generator) -> float:
    """
    Sample a Rayleigh fading envelope in dB.
    Signal envelope |h| ~ Rayleigh(sigma=1) → power |h|^2 ~ exponential(1).
    Returns 10*log10(|h|^2) in dB. Mean ≈ -2.5 dB, std ≈ 5.6 dB.
    """
    h_squared = rng.exponential(1.0)
    return 10.0 * np.log10(max(h_squared, 1e-12))


def compute_channel_amplitude_dbm(
    tx_power_dbm: float,
    range_km: float,
    frequency_hz: float,
    rng: np.random.Generator,
) -> float:
    """
    Compute received amplitude (dBm) accounting for the complete channel model.

    Applies three propagation effects:
      1. Free-space path loss (FSPL) — deterministic, range- and frequency-
         dependent.  Uses the standard ITU form:
           FSPL(dB) = 20·log10(range_km) + 20·log10(f_MHz) + 32.4
      2. Shadowing — log-normal slow fading, single draw per emitter.
         σ = 8 dB.  Negative values attenuate the signal.
      3. Rayleigh fading — fast multipath, independent draw per pulse.
         Exponential power envelope with scale = 1.0 (Rayleigh(sigma=1)
         for the complex envelope → power ~ Exp(1)).  dB-domain mean ≈ -2.5 dB.
         Subtracting it gives a negative dB value (signal attenuation).

    Returns the received signal amplitude in dBm.
    """
    # Free-space path loss (dB)
    fspl_db = 20.0 * np.log10(range_km) + 20.0 * np.log10(frequency_hz / 1e6) + 32.4

    # Shadowing: single normal draw per emitter context
    shadow_db = rng.normal(0.0, 8.0)

    # Rayleigh fading: independent exponential draw per pulse
    # exponential(scale=1) → mean power = 1 linear → 10*log10(1) = 0 dB
    # (dB-domain mean ≈ -2.5 dB due to E[log10(X)] = -0.217 for X~Exp(1))
    fading_db = 10.0 * np.log10(max(rng.exponential(1.0), 1e-12))

    return tx_power_dbm - fspl_db - shadow_db - fading_db


# =====================================================================
# Pulse dataclass
# =====================================================================

@dataclass
class Pulse:
    """
    A single RF pulse.
    All times are in SECONDS.

    `amplitude_dbm` is the receiver-side received power in dBm. The emitter
    only specifies a transmit power and a distance; path loss, fading, and
    atmospheric attenuation are computed once in `generate_pulses()` so
    downstream code can use a single `Pd` per pulse.
    """
    toa: float              # Time of arrival (seconds since sim start)
    frequency_hz: float    # Pulse centre frequency (Hz)
    pulse_width_sec: float # Pulse width/duration (seconds)
    amplitude_dbm: float   # Received power (dBm)
    emitter_id: int        # Which emitter generated this pulse
    is_real: bool = True   # False for false alarm pulses (injected by receiver)

    def __repr__(self) -> str:
        return (
            f"Pulse(toa={self.toa:.6f}s, freq={self.frequency_hz/1e9:.4f}GHz, "
            f"PW={self.pulse_width_sec*1e6:.2f}μs, amp={self.amplitude_dbm:.1f}dBm, "
            f"eid={self.emitter_id}, real={self.is_real})"
        )


# =====================================================================
# Base emitter class
# =====================================================================

class Emitter(ABC):
    """
    Abstract base class for all emitter types.

    Each emitter generates a deterministic sequence of pulses.
    The RNG is seeded via the parent's SeedSequence so that the emitter
    always produces the same pulse train given the same seed.

    Range and power:
      `tx_power_dbm` — power at the emitter's antenna feed (EIRP-like).
      `range_m` — distance from emitter to receiver in meters. The receiver
        sees `tx_power_dbm - path_loss_db - atmospheric_db - fading_db`.
        If `range_m` is None, the emitter's `power_dbm` field is used
        directly (legacy behavior).
    """

    def __init__(
        self,
        emitter_id: int,
        name: Optional[str] = None,
    ):
        self.emitter_id = emitter_id
        self.name = name or f"Emitter_{emitter_id}"
        # New: range-based power. Subclasses that support channel model
        # should set these in their __init__.
        self.tx_power_dbm: Optional[float] = None
        self.range_m: Optional[float] = None
        self.fading: bool = True  # apply Rayleigh fading per pulse
        # Range in km used by the realistic channel model
        # (FSPL, Rayleigh fading, log-normal shadowing).
        # Default 50 km — typical airborne radar geometry.
        # Set enable_channel=False to use flat amplitude_dbm=power_dbm (legacy).
        self.range_km: float = 50.0
        self.enable_channel: bool = False  # disabled by default (legacy compat)

    def compute_received_power_dbm(
        self,
        frequency_hz: float,
        rng: np.random.Generator,
    ) -> float:
        """
        Compute the receiver-side received power for a pulse at this emitter.
        Uses free-space path loss, atmospheric attenuation, and (optionally)
        Rayleigh fading.
        """
        if self.range_m is None or self.tx_power_dbm is None:
            # Legacy mode: caller must already have set Pulse.amplitude_dbm
            raise ValueError(
                "Emitter has no range_m/tx_power_dbm set. Use the legacy "
                "power_dbm field, or set range_m and tx_power_dbm at construction."
            )
        pl = free_space_path_loss_db(self.range_m, frequency_hz)
        atmo = atmospheric_attenuation_db(frequency_hz)
        fade = rayleigh_fading_db(rng) if self.fading else 0.0
        return self.tx_power_dbm - pl - atmo + fade

    def channel_amplitude_dbm(
        self,
        frequency_hz: float,
        rng: np.random.Generator,
    ) -> float:
        """
        Compute the per-pulse received amplitude (dBm) using the realistic
        channel model: FSPL + Rayleigh fading + log-normal shadowing.

        Replaces the legacy ``self.power_dbm`` assignment in subclass
        ``generate_pulses()`` methods.  The channel model is:
            amplitude_dbm = power_dbm - FSPL - shadow_db - fading_db
        where FSPL uses the standard ITU form (range_km, freq_MHz), and
        shadow_db / fading_db are drawn from N(0, 8) and Exp(scale=3)
        respectively.

        When ``enable_channel=False`` (the default), this method returns
        ``self.power_dbm`` unchanged — preserving legacy flat-amplitude behavior.
        """
        if not self.enable_channel:
            return self.power_dbm
        return compute_channel_amplitude_dbm(
            tx_power_dbm=self.power_dbm,
            range_km=self.range_km,
            frequency_hz=frequency_hz,
            rng=rng,
        )

    @abstractmethod
    def generate_pulses(
        self,
        sim_start_sec: float,
        sim_end_sec: float,
        rng: np.random.Generator,
    ) -> List[Pulse]:
        """
        Generate all pulses between sim_start_sec and sim_end_sec (exclusive).
        Returns a list sorted by TOA.
        """
        ...

    @abstractmethod
    def emitter_type(self) -> str:
        """Human-readable emitter type name."""
        ...


# =====================================================================
# 1. Fixed Continuous Emitter
# Simple: single frequency, constant PRI, always ON
# =====================================================================

class FixedContinuousEmitter(Emitter):
    """
    Single-frequency, constant-PRI emitter. Always transmits throughout the mission.

    Realistic for: CW radars, beacons, continuous-wave jammers.
    """

    def __init__(
        self,
        emitter_id: int,
        center_freq_hz: float,
        pri_sec: float,
        pulse_width_sec: float,
        power_dbm: float,
        phase_offset_sec: float = 0.0,
        start_time_sec: float = 0.0,
        stop_time_sec: Optional[float] = None,
        name: Optional[str] = None,
        range_km: float = 50.0,
    ):
        super().__init__(emitter_id, name)
        self.center_freq_hz = float(center_freq_hz)
        self.pri_sec = float(pri_sec)
        self.pulse_width_sec = float(pulse_width_sec)
        self.power_dbm = float(power_dbm)
        self.phase_offset_sec = float(phase_offset_sec)
        self.start_time_sec = float(start_time_sec)
        self.stop_time_sec = stop_time_sec  # None = always on
        self.range_km = float(range_km)

        # Validate
        assert 500e6 <= self.center_freq_hz <= 18e9, \
            f"Frequency must be 500 MHz – 18 GHz, got {self.center_freq_hz/1e9:.2f} GHz"
        assert 50e-6 <= self.pri_sec <= 10e-3, \
            f"PRI must be 50 μs – 10 ms, got {self.pri_sec*1e6:.1f} μs"
        assert 0.2e-6 <= self.pulse_width_sec <= 10e-6, \
            f"PW must be 0.2 μs – 10 μs, got {self.pulse_width_sec*1e6:.2f} μs"
        assert -100.0 <= self.power_dbm <= -30.0, \
            f"Power must be -100 dBm to -30 dBm, got {self.power_dbm:.1f} dBm"

    def emitter_type(self) -> str:
        return "FixedContinuous"

    def generate_pulses(
        self,
        sim_start_sec: float,
        sim_end_sec: float,
        rng: np.random.Generator,
    ) -> List[Pulse]:
        pulses: List[Pulse] = []

        start = max(self.start_time_sec, sim_start_sec)
        if self.stop_time_sec is not None:
            end = min(self.stop_time_sec, sim_end_sec)
        else:
            end = sim_end_sec

        if start >= end:
            return pulses

        # First pulse time: first PRI boundary after `start`
        first_pulse = self._next_pulse_after(start)
        if first_pulse >= end:
            return pulses

        # Generate all pulses from first_pulse to end
        t = first_pulse
        while t < end:
            pulses.append(Pulse(
                toa=t,
                frequency_hz=self.center_freq_hz,
                pulse_width_sec=self.pulse_width_sec,
                amplitude_dbm=self.channel_amplitude_dbm(self.center_freq_hz, rng),
                emitter_id=self.emitter_id,
                is_real=True,
            ))
            t += self.pri_sec

        return pulses

    def _next_pulse_after(self, t: float) -> float:
        """Return the next pulse TOA >= t."""
        elapsed = t - self.start_time_sec
        if elapsed < 0:
            return self.start_time_sec + self.phase_offset_sec
        # Find the next PRI boundary
        period_idx = int(np.floor((elapsed - self.phase_offset_sec) / self.pri_sec))
        return self.start_time_sec + self.phase_offset_sec + (period_idx + 1) * self.pri_sec


# =====================================================================
# 2. Fixed Intermittent Emitter
# ON/OFF duty cycle: ON for on_sec, OFF for off_sec, repeating
# =====================================================================

class FixedIntermittentEmitter(Emitter):
    """
    Single-frequency emitter that cycles ON/OFF with a fixed duty cycle.

    Realistic for: Intermittent-data-link emitters, burst transmitters,
    time-shared systems.

    Lifecycle: ON for `on_sec`, OFF for `off_sec`, repeat.
    """

    def __init__(
        self,
        emitter_id: int,
        center_freq_hz: float,
        pri_sec: float,
        pulse_width_sec: float,
        power_dbm: float,
        on_sec: float,
        off_sec: float,
        phase_offset_sec: float = 0.0,
        start_time_sec: float = 0.0,
        stop_time_sec: Optional[float] = None,
        name: Optional[str] = None,
        range_km: float = 50.0,
    ):
        super().__init__(emitter_id, name)
        self.center_freq_hz = float(center_freq_hz)
        self.pri_sec = float(pri_sec)
        self.pulse_width_sec = float(pulse_width_sec)
        self.power_dbm = float(power_dbm)
        self.on_sec = float(on_sec)
        self.off_sec = float(off_sec)
        self.phase_offset_sec = float(phase_offset_sec)
        self.range_km = float(range_km)
        self.start_time_sec = float(start_time_sec)
        self.stop_time_sec = stop_time_sec

        self.cycle_sec = self.on_sec + self.off_sec

        assert self.cycle_sec > 0, "Duty cycle period must be positive"
        self.duty_cycle = self.on_sec / self.cycle_sec

        # Validate RF params
        assert 500e6 <= self.center_freq_hz <= 18e9
        assert 50e-6 <= self.pri_sec <= 10e-3
        assert 0.2e-6 <= self.pulse_width_sec <= 10e-6
        assert -100.0 <= self.power_dbm <= -30.0

    def emitter_type(self) -> str:
        return "FixedIntermittent"

    def _is_on_at(self, t: float) -> bool:
        """Return True if emitter is in its ON phase at time t."""
        if t < self.start_time_sec:
            return False
        if self.stop_time_sec is not None and t >= self.stop_time_sec:
            return False
        elapsed = t - self.start_time_sec
        cycle_pos = (elapsed + self.phase_offset_sec) % self.cycle_sec
        # Use small epsilon to avoid floating-point boundary issues
        return cycle_pos < self.on_sec - 1e-12

    def generate_pulses(
        self,
        sim_start_sec: float,
        sim_end_sec: float,
        rng: np.random.Generator,
    ) -> List[Pulse]:
        """
        Generate pulses by stepping through PRI intervals and emitting
        only when the emitter is in an ON phase. This is O(N_pulses)
        and avoids all boundary arithmetic.
        """
        pulses: List[Pulse] = []

        start = max(self.start_time_sec, sim_start_sec)
        end = (self.stop_time_sec if self.stop_time_sec is not None else sim_end_sec)
        end = min(end, sim_end_sec)

        if start >= end:
            return pulses

        # First pulse at or after `start`
        if start < self.start_time_sec + self.phase_offset_sec:
            t = self.start_time_sec + self.phase_offset_sec
        else:
            elapsed = start - self.start_time_sec
            n = int(np.floor((elapsed - self.phase_offset_sec) / self.pri_sec))
            t = self.start_time_sec + self.phase_offset_sec + (n + 1) * self.pri_sec

        while t < end:
            if self._is_on_at(t):
                pulses.append(Pulse(
                    toa=t,
                    frequency_hz=self.center_freq_hz,
                    pulse_width_sec=self.pulse_width_sec,
                    amplitude_dbm=self.channel_amplitude_dbm(self.center_freq_hz, rng),
                    emitter_id=self.emitter_id,
                    is_real=True,
                ))
            t += self.pri_sec

        return pulses

    def _next_on_pulse(self, t: float) -> float:
        """Next pulse time at or after t (assumes already ON)."""
        if t < self.start_time_sec:
            return self.start_time_sec + self.phase_offset_sec
        elapsed = t - self.start_time_sec
        cycle_pos = (elapsed + self.phase_offset_sec) % self.cycle_sec
        pos_in_on = cycle_pos
        periods_elapsed = int(np.floor(pos_in_on / self.pri_sec))
        next_pulse_offset = (periods_elapsed + 1) * self.pri_sec
        return self.start_time_sec + self.phase_offset_sec + next_pulse_offset

    def _off_boundary_after(self, t: float) -> float:
        """Time at which current ON phase ends (OFF begins)."""
        elapsed = t - self.start_time_sec + self.phase_offset_sec
        cycle_idx = int(np.floor(elapsed / self.cycle_sec))
        return self.start_time_sec + cycle_idx * self.cycle_sec + self.on_sec \
               - self.phase_offset_sec

    def _next_on_start(self, t: float) -> float:
        """Time at which the next ON phase begins."""
        if t < self.start_time_sec:
            return self.start_time_sec
        elapsed = t - self.start_time_sec + self.phase_offset_sec
        cycle_idx = int(np.ceil(elapsed / self.cycle_sec))
        return self.start_time_sec + cycle_idx * self.cycle_sec \
               - self.phase_offset_sec


# =====================================================================
# 3. Periodic Spatial Scan Emitter
# Beam sweeps across receiver with a known period.
# Only emits pulses when beam is pointing at receiver (visible fraction).
# =====================================================================

class ScanningEmitter(Emitter):
    """
    Radar with a rotating antenna that periodically sweeps the beam across
    the receiver's direction. Pulses are only detectable when the beam
    mainbeam points at the receiver.

    Realistic for: Search radar, air-surveillance radar, rotating fire-control radar.

    Models: beam_width_deg / 360 as the fraction of time the emitter is visible.
    The beam "hits" the receiver when its pointing angle crosses the receiver's
    direction during each rotation.
    """

    def __init__(
        self,
        emitter_id: int,
        center_freq_hz: float,
        pri_sec: float,
        pulse_width_sec: float,
        power_dbm: float,
        scan_period_sec: float,
        beam_width_deg: float = 5.0,
        visibility_phase_offset_sec: float = 0.0,
        start_time_sec: float = 0.0,
        stop_time_sec: Optional[float] = None,
        name: Optional[str] = None,
        range_km: float = 50.0,
    ):
        super().__init__(emitter_id, name)
        self.center_freq_hz = float(center_freq_hz)
        self.pri_sec = float(pri_sec)
        self.pulse_width_sec = float(pulse_width_sec)
        self.power_dbm = float(power_dbm)
        self.scan_period_sec = float(scan_period_sec)
        self.beam_width_deg = float(beam_width_deg)
        self.visibility_phase_offset_sec = float(visibility_phase_offset_sec)
        self.start_time_sec = float(start_time_sec)
        self.stop_time_sec = stop_time_sec
        self.range_km = float(range_km)

        # Fraction of each scan period the beam points at the receiver
        self.visibility_fraction = self.beam_width_deg / 360.0
        self.visible_duration_sec = self.scan_period_sec * self.visibility_fraction

        # PRI phase offset
        self.pri_offset = 0.0  # pulse phase within scan

        assert 500e6 <= self.center_freq_hz <= 18e9
        assert 50e-6 <= self.pri_sec <= 10e-3
        assert 0.2e-6 <= self.pulse_width_sec <= 10e-6
        assert -100.0 <= self.power_dbm <= -30.0
        assert 0.5 <= self.scan_period_sec <= 30.0, "Scan period should be 0.5–30 seconds"
        assert 0.1 <= self.beam_width_deg <= 30.0, "Beam width should be 0.1–30 degrees"

    def emitter_type(self) -> str:
        return "PeriodicSpatialScan"

    def _beam_at_time(self, t: float) -> bool:
        """Return True if beam is pointing at receiver at time t."""
        if t < self.start_time_sec:
            return False
        if self.stop_time_sec is not None and t >= self.stop_time_sec:
            return False
        elapsed = t - self.start_time_sec + self.visibility_phase_offset_sec
        pos_in_scan = elapsed % self.scan_period_sec
        return pos_in_scan < self.visible_duration_sec - 1e-15

    def generate_pulses(
        self,
        sim_start_sec: float,
        sim_end_sec: float,
        rng: np.random.Generator,
    ) -> List[Pulse]:
        pulses: List[Pulse] = []

        start = max(self.start_time_sec, sim_start_sec)
        end = (self.stop_time_sec if self.stop_time_sec is not None else sim_end_sec)
        end = min(end, sim_end_sec)

        if start >= end:
            return pulses

        # Scan cycle boundaries
        def scan_cycle_start(after_t: float) -> float:
            """Time of the start of the scan cycle that contains after_t."""
            elapsed = after_t - self.start_time_sec + self.visibility_phase_offset_sec
            n = int(np.floor(elapsed / self.scan_period_sec))
            return self.start_time_sec + n * self.scan_period_sec - self.visibility_phase_offset_sec

        # Find the scan cycle that contains `start`
        cycle_t = scan_cycle_start(start)

        while cycle_t < end:
            visible_start = cycle_t + self.visibility_phase_offset_sec
            visible_end = visible_start + self.visible_duration_sec

            # Clip to sim window
            chunk_start = max(visible_start, start)
            chunk_end = min(visible_end, end)

            if chunk_start < chunk_end:
                # Generate pulses in this visible window
                first = self._next_pri_pulse(chunk_start)
                t = first
                while t < chunk_end:
                    pulses.append(Pulse(
                        toa=t,
                        frequency_hz=self.center_freq_hz,
                        pulse_width_sec=self.pulse_width_sec,
                        amplitude_dbm=self.channel_amplitude_dbm(self.center_freq_hz, rng),
                        emitter_id=self.emitter_id,
                        is_real=True,
                    ))
                    t += self.pri_sec

            cycle_t += self.scan_period_sec

        return sorted(pulses, key=lambda p: p.toa)

    def _next_pri_pulse(self, t: float) -> float:
        """Next PRI pulse time >= t."""
        if t < self.start_time_sec:
            return self.start_time_sec + self.pri_offset
        elapsed = t - self.start_time_sec
        n = int(np.floor(elapsed / self.pri_sec))
        return self.start_time_sec + (n + 1) * self.pri_sec


# =====================================================================
# 4. Frequency Agile (Patterned) Emitter
# Hops between predefined frequency list with deterministic pattern
# =====================================================================

class FrequencyAgileEmitter(Emitter):
    """
    Frequency-hopping emitter. Cycles through a predefined frequency list
    in order, dwelling on each frequency for a configurable time.

    Realistic for: Frequency-hopping spread-spectrum (FHSS) radios,
    frequency-agile jammers, some modern radars and data links.

    The hop sequence is: freq_list[0], freq_list[1], ..., freq_list[n-1], then repeat.
    Each frequency is held for `dwell_sec`.
    """

    def __init__(
        self,
        emitter_id: int,
        freq_list_hz: List[float],
        pri_sec: float,
        pulse_width_sec: float,
        power_dbm: float,
        dwell_sec: float = 10e-3,
        hop_sequence_seed: Optional[int] = None,
        start_time_sec: float = 0.0,
        stop_time_sec: Optional[float] = None,
        dwell_schedule_sec: Optional[List[float]] = None,
        name: Optional[str] = None,
        range_km: float = 50.0,
    ):
        super().__init__(emitter_id, name)
        self.freq_list_hz = [float(f) for f in freq_list_hz]
        self.pri_sec = float(pri_sec)
        self.pulse_width_sec = float(pulse_width_sec)
        self.power_dbm = float(power_dbm)
        self.dwell_sec = float(dwell_sec)
        self.hop_sequence_seed = hop_sequence_seed
        self.start_time_sec = float(start_time_sec)
        self.stop_time_sec = stop_time_sec
        self.range_km = float(range_km)
        # Optional non-uniform dwell schedule. When provided, the i-th dwell
        # uses dwell_schedule_sec[i % len(schedule)] instead of the constant
        # dwell_sec. This adds temporal structure that a smart scheduler can
        # learn (e.g. radar that dwells longer on a certain band once per cycle).
        self.dwell_schedule_sec = (
            [float(d) for d in dwell_schedule_sec]
            if dwell_schedule_sec is not None else None
        )

        assert len(self.freq_list_hz) > 0, "freq_list_hz must not be empty"
        for f in self.freq_list_hz:
            assert 500e6 <= f <= 18e9, f"Frequency {f/1e9:.2f} GHz out of range"
        assert 50e-6 <= self.pri_sec <= 10e-3
        assert 0.2e-6 <= self.pulse_width_sec <= 10e-6
        assert -100.0 <= self.power_dbm <= -30.0
        assert 1e-3 <= self.dwell_sec <= 1.0, "Dwell time should be 1 ms to 1 s"
        if self.dwell_schedule_sec is not None:
            assert len(self.dwell_schedule_sec) > 0
            for d in self.dwell_schedule_sec:
                assert 1e-3 <= d <= 1.0, "Dwell entries must be 1 ms to 1 s"

        self.n_freqs = len(self.freq_list_hz)

    def emitter_type(self) -> str:
        return "FrequencyAgile"

    def _dwell_at_index(self, dwell_idx: int) -> float:
        """Return the dwell duration for the dwell_idx-th dwell in the
        emitter's life (after its start_time_sec)."""
        if self.dwell_schedule_sec is None:
            return self.dwell_sec
        return self.dwell_schedule_sec[dwell_idx % len(self.dwell_schedule_sec)]

    def _freq_at_time(self, t: float) -> float:
        """Return the frequency being transmitted at time t."""
        if t < self.start_time_sec:
            return self.freq_list_hz[0]  # Before start, no transmission
        if self.stop_time_sec is not None and t >= self.stop_time_sec:
            return self.freq_list_hz[0]  # After stop, no transmission

        if self.dwell_schedule_sec is None:
            elapsed = t - self.start_time_sec
            dwell_index = int(elapsed / self.dwell_sec) % self.n_freqs
            return self.freq_list_hz[dwell_index]

        # Non-uniform schedule: walk forward and accumulate dwell durations
        cumulative = self.start_time_sec
        idx = 0
        while cumulative + self._dwell_at_index(idx) <= t:
            cumulative += self._dwell_at_index(idx)
            idx += 1
        return self.freq_list_hz[idx % self.n_freqs]

    def generate_pulses(
        self,
        sim_start_sec: float,
        sim_end_sec: float,
        rng: np.random.Generator,
    ) -> List[Pulse]:
        pulses: List[Pulse] = []

        start = max(self.start_time_sec, sim_start_sec)
        end = (self.stop_time_sec if self.stop_time_sec is not None else sim_end_sec)
        end = min(end, sim_end_sec)

        if start >= end:
            return []

        # Step through time in dwell intervals, honoring per-dwell schedule
        if self.dwell_schedule_sec is None:
            # Uniform-dwell fast path
            dwell_idx = int(start / self.dwell_sec)
            t = start
            while t < end:
                dwell_start = t
                dwell_end = min(t + self.dwell_sec, end)
                freq = self.freq_list_hz[dwell_idx % self.n_freqs]
                first = self._next_pri_pulse(dwell_start)
                pulse_t = first
                while pulse_t < dwell_end:
                    pulses.append(Pulse(
                        toa=pulse_t,
                        frequency_hz=freq,
                        pulse_width_sec=self.pulse_width_sec,
                        amplitude_dbm=self.channel_amplitude_dbm(freq, rng),
                        emitter_id=self.emitter_id,
                        is_real=True,
                    ))
                    pulse_t += self.pri_sec
                dwell_idx += 1
                t = dwell_start + self.dwell_sec
            return sorted(pulses, key=lambda p: p.toa)

        # Non-uniform schedule: walk forward
        # First, find the dwell index whose start <= `start`
        cumulative = self.start_time_sec
        idx = 0
        while cumulative + self._dwell_at_index(idx) <= start:
            cumulative += self._dwell_at_index(idx)
            idx += 1
        t = cumulative
        while t < end:
            dwell_dur = self._dwell_at_index(idx)
            dwell_start = t
            dwell_end = min(t + dwell_dur, end)
            freq = self.freq_list_hz[idx % self.n_freqs]
            first = self._next_pri_pulse(dwell_start)
            pulse_t = first
            while pulse_t < dwell_end:
                pulses.append(Pulse(
                    toa=pulse_t,
                    frequency_hz=freq,
                    pulse_width_sec=self.pulse_width_sec,
                    amplitude_dbm=self.channel_amplitude_dbm(freq, rng),
                    emitter_id=self.emitter_id,
                    is_real=True,
                ))
                pulse_t += self.pri_sec
            t = dwell_start + dwell_dur
            idx += 1
        return sorted(pulses, key=lambda p: p.toa)

    def _next_pri_pulse(self, t: float) -> float:
        if t < self.start_time_sec:
            return self.start_time_sec
        elapsed = t - self.start_time_sec
        n = int(np.floor(elapsed / self.pri_sec))
        return self.start_time_sec + (n + 1) * self.pri_sec


# =====================================================================
# 5. Frequency Agile + Spatial Scan
# Both frequency hopping AND spatial beam scanning simultaneously
# =====================================================================

class FrequencyAgileScanningEmitter(Emitter):
    """
    Combines frequency agility with spatial scanning. The emitter frequency-
    hops and also has a rotating beam. Pulses are only emitted when BOTH
    conditions are met: (a) currently in the ON frequency dwell, AND (b)
    beam is pointing at the receiver.

    Realistic for: Agile radar systems, multi-function RF systems,
    modern ESM receivers tracking frequency-hopping emitters.
    """

    def __init__(
        self,
        emitter_id: int,
        freq_list_hz: List[float],
        pri_sec: float,
        pulse_width_sec: float,
        power_dbm: float,
        dwell_sec: float = 10e-3,
        scan_period_sec: float = 2.0,
        beam_width_deg: float = 5.0,
        visibility_phase_offset_sec: float = 0.0,
        start_time_sec: float = 0.0,
        stop_time_sec: Optional[float] = None,
        dwell_schedule_sec: Optional[List[float]] = None,
        name: Optional[str] = None,
        range_km: float = 50.0,
    ):
        super().__init__(emitter_id, name)
        self.freq_list_hz = [float(f) for f in freq_list_hz]
        self.pri_sec = float(pri_sec)
        self.pulse_width_sec = float(pulse_width_sec)
        self.power_dbm = float(power_dbm)
        self.dwell_sec = float(dwell_sec)
        self.scan_period_sec = float(scan_period_sec)
        self.beam_width_deg = float(beam_width_deg)
        self.visibility_phase_offset_sec = float(visibility_phase_offset_sec)
        self.start_time_sec = float(start_time_sec)
        self.stop_time_sec = stop_time_sec
        self.range_km = float(range_km)
        self.dwell_schedule_sec = (
            [float(d) for d in dwell_schedule_sec]
            if dwell_schedule_sec is not None else None
        )

        self.n_freqs = len(self.freq_list_hz)
        self.visibility_fraction = self.beam_width_deg / 360.0
        self.visible_duration_sec = self.scan_period_sec * self.visibility_fraction

        assert len(self.freq_list_hz) > 0
        for f in self.freq_list_hz:
            assert 500e6 <= f <= 18e9
        assert 50e-6 <= self.pri_sec <= 10e-3
        assert 0.2e-6 <= self.pulse_width_sec <= 10e-6
        assert -100.0 <= self.power_dbm <= -30.0
        assert 1e-3 <= self.dwell_sec <= 1.0
        assert 0.5 <= self.scan_period_sec <= 30.0
        if self.dwell_schedule_sec is not None:
            for d in self.dwell_schedule_sec:
                assert 1e-3 <= d <= 1.0

    def emitter_type(self) -> str:
        return "FrequencyAgileScanning"

    def _is_visible_at(self, t: float) -> bool:
        if t < self.start_time_sec:
            return False
        if self.stop_time_sec is not None and t >= self.stop_time_sec:
            return False
        elapsed = t - self.start_time_sec + self.visibility_phase_offset_sec
        pos_in_scan = elapsed % self.scan_period_sec
        return pos_in_scan < self.visible_duration_sec - 1e-15

    def _dwell_at_index(self, idx: int) -> float:
        if self.dwell_schedule_sec is None:
            return self.dwell_sec
        return self.dwell_schedule_sec[idx % len(self.dwell_schedule_sec)]

    def _freq_at_time(self, t: float) -> float:
        if t < self.start_time_sec:
            return self.freq_list_hz[0]
        if self.stop_time_sec is not None and t >= self.stop_time_sec:
            return self.freq_list_hz[0]
        if self.dwell_schedule_sec is None:
            elapsed = t - self.start_time_sec
            dwell_idx = int(elapsed / self.dwell_sec) % self.n_freqs
            return self.freq_list_hz[dwell_idx]
        cumulative = self.start_time_sec
        idx = 0
        while cumulative + self._dwell_at_index(idx) <= t:
            cumulative += self._dwell_at_index(idx)
            idx += 1
        return self.freq_list_hz[idx % self.n_freqs]

    def generate_pulses(
        self,
        sim_start_sec: float,
        sim_end_sec: float,
        rng: np.random.Generator,
    ) -> List[Pulse]:
        pulses: List[Pulse] = []

        start = max(self.start_time_sec, sim_start_sec)
        end = (self.stop_time_sec if self.stop_time_sec is not None else sim_end_sec)
        end = min(end, sim_end_sec)

        if start >= end:
            return []

        if self.dwell_schedule_sec is None:
            # Uniform dwell
            dwell_idx = int(start / self.dwell_sec)
            t = start
            while t < end:
                dwell_start = t
                dwell_end = min(t + self.dwell_sec, end)
                if self._is_visible_at(dwell_start):
                    freq = self.freq_list_hz[dwell_idx % self.n_freqs]
                    first = self._next_pri_pulse(dwell_start)
                    pulse_t = first
                    while pulse_t < dwell_end:
                        pulses.append(Pulse(
                            toa=pulse_t,
                            frequency_hz=freq,
                            pulse_width_sec=self.pulse_width_sec,
                            amplitude_dbm=self.channel_amplitude_dbm(freq, rng),
                            emitter_id=self.emitter_id,
                            is_real=True,
                        ))
                        pulse_t += self.pri_sec
                dwell_idx += 1
                t = dwell_start + self.dwell_sec
            return sorted(pulses, key=lambda p: p.toa)

        # Non-uniform dwell
        cumulative = self.start_time_sec
        idx = 0
        while cumulative + self._dwell_at_index(idx) <= start:
            cumulative += self._dwell_at_index(idx)
            idx += 1
        t = cumulative
        while t < end:
            dwell_dur = self._dwell_at_index(idx)
            dwell_start = t
            dwell_end = min(t + dwell_dur, end)
            if self._is_visible_at(dwell_start):
                freq = self.freq_list_hz[idx % self.n_freqs]
                first = self._next_pri_pulse(dwell_start)
                pulse_t = first
                while pulse_t < dwell_end:
                    pulses.append(Pulse(
                        toa=pulse_t,
                        frequency_hz=freq,
                        pulse_width_sec=self.pulse_width_sec,
                        amplitude_dbm=self.channel_amplitude_dbm(freq, rng),
                        emitter_id=self.emitter_id,
                        is_real=True,
                    ))
                    pulse_t += self.pri_sec
            t = dwell_start + dwell_dur
            idx += 1
        return sorted(pulses, key=lambda p: p.toa)

    def _next_pri_pulse(self, t: float) -> float:
        if t < self.start_time_sec:
            return self.start_time_sec
        elapsed = t - self.start_time_sec
        n = int(np.floor(elapsed / self.pri_sec))
        return self.start_time_sec + (n + 1) * self.pri_sec


# =====================================================================
# 6. PRI Jitter Emitter
# Nominal PRI with random variation each pulse
# =====================================================================

class PriJitterEmitter(Emitter):
    """
    Fixed-frequency emitter with PRI jitter: each inter-pulse interval is
    drawn from nominal_PRI ± jitter_fraction * nominal_PRI.

    Realistic for: Radars with imperfect oscillator timing, non-coherent
    emitters, intentional PRI jitter for ECCM.

    This is always "ON" (continuous transmission with jittered intervals).
    """

    def __init__(
        self,
        emitter_id: int,
        center_freq_hz: float,
        nominal_pri_sec: float,
        pulse_width_sec: float,
        power_dbm: float,
        jitter_fraction: float = 0.1,
        phase_offset_sec: float = 0.0,
        start_time_sec: float = 0.0,
        stop_time_sec: Optional[float] = None,
        name: Optional[str] = None,
        range_km: float = 50.0,
    ):
        super().__init__(emitter_id, name)
        self.center_freq_hz = float(center_freq_hz)
        self.nominal_pri_sec = float(nominal_pri_sec)
        self.pulse_width_sec = float(pulse_width_sec)
        self.power_dbm = float(power_dbm)
        self.jitter_fraction = float(jitter_fraction)
        self.phase_offset_sec = float(phase_offset_sec)
        self.start_time_sec = float(start_time_sec)
        self.stop_time_sec = stop_time_sec
        self.range_km = float(range_km)

        assert 500e6 <= self.center_freq_hz <= 18e9
        assert 50e-6 <= self.nominal_pri_sec <= 10e-3
        assert 0.2e-6 <= self.pulse_width_sec <= 10e-6
        assert -100.0 <= self.power_dbm <= -30.0
        assert 0.0 < self.jitter_fraction <= 0.5, "Jitter fraction should be 0–50%"

    def emitter_type(self) -> str:
        return "PriJitter"

    def generate_pulses(
        self,
        sim_start_sec: float,
        sim_end_sec: float,
        rng: np.random.Generator,
    ) -> List[Pulse]:
        pulses: List[Pulse] = []

        start = max(self.start_time_sec, sim_start_sec)
        end = (self.stop_time_sec if self.stop_time_sec is not None else sim_end_sec)
        end = min(end, sim_end_sec)

        if start >= end:
            return []

        # First pulse
        t = start
        if t < self.start_time_sec:
            t = self.start_time_sec + self.phase_offset_sec

        jitter_max = self.nominal_pri_sec * self.jitter_fraction

        while t < end:
            pulses.append(Pulse(
                toa=t,
                frequency_hz=self.center_freq_hz,
                pulse_width_sec=self.pulse_width_sec,
                amplitude_dbm=self.channel_amplitude_dbm(self.center_freq_hz, rng),
                emitter_id=self.emitter_id,
                is_real=True,
            ))
            # Next pulse with jitter
            jitter = rng.uniform(-jitter_max, jitter_max)
            next_pri = self.nominal_pri_sec + jitter
            t += max(50e-6, next_pri)  # Ensure minimum PRI of 50 μs

        return pulses


# =====================================================================
# 7. Dynamic Lifecycle Policies
# Add temporal structure to any base emitter:
#   - Delayed arrival (emitter appears at t=X)
#   - Random ON/OFF intervals
#   - Regime change (emitter swaps config mid-mission)
#
# Usage:
#   base_emitter = FixedContinuousEmitter(...)
#   policy = IntervalOnOffPolicy(mean_on_sec=5.0, mean_off_sec=10.0)
#   emitter = DynamicEmitter(emitter_id=0, base_emitter=base_emitter, policy=policy)
# =====================================================================

class LifecyclePolicy(ABC):
    """
    Abstract base for lifecycle transformation policies.
    A policy takes a list of raw pulses and returns a transformed list.

    Deterministic policies use only the RNG for repeatable draws.
    """

    @abstractmethod
    def apply(
        self,
        pulses: List[Pulse],
        sim_start_sec: float,
        sim_end_sec: float,
        rng: np.random.Generator,
    ) -> List[Pulse]:
        """Transform the pulse list. Must return a new sorted list."""
        ...


class DelayedArrivalPolicy(LifecyclePolicy):
    """
    Holds all pulses until start_time_sec.
    Pulses before the arrival time are silently dropped.
    """
    def __init__(self, start_time_sec: float = 0.0):
        self.start_time_sec = float(start_time_sec)

    def apply(
        self,
        pulses: List[Pulse],
        sim_start_sec: float,
        sim_end_sec: float,
        rng: np.random.Generator,
    ) -> List[Pulse]:
        start = max(self.start_time_sec, sim_start_sec)
        end = sim_end_sec
        return [p for p in pulses if start <= p.toa < end]


class StopTimePolicy(LifecyclePolicy):
    """
    Cuts off all pulses after stop_time_sec.
    """
    def __init__(self, stop_time_sec: float):
        self.stop_time_sec = float(stop_time_sec)

    def apply(
        self,
        pulses: List[Pulse],
        sim_start_sec: float,
        sim_end_sec: float,
        rng: np.random.Generator,
    ) -> List[Pulse]:
        end = min(self.stop_time_sec, sim_end_sec)
        return [p for p in pulses if p.toa < end]


class IntervalOnOffPolicy(LifecyclePolicy):
    """
    Random ON/OFF intervals overlaid on top of a base emitter.
    Emitter has random bursts of activity with random silences.

    Realistic for: Intermittent communications, bursty data links,
    systems that power-cycle based on external events.

    The emitter is ON for an exponentially-distributed period with mean
    `mean_on_sec`, then OFF for an exponentially-distributed period with
    mean `mean_off_sec`. The cycle repeats throughout the mission.
    """
    def __init__(
        self,
        mean_on_sec: float = 5.0,
        mean_off_sec: float = 10.0,
        seed: Optional[int] = None,
    ):
        assert mean_on_sec > 0 and mean_off_sec > 0
        self.mean_on_sec = float(mean_on_sec)
        self.mean_off_sec = float(mean_off_sec)
        self._seed = seed

    def apply(
        self,
        pulses: List[Pulse],
        sim_start_sec: float,
        sim_end_sec: float,
        rng: np.random.Generator,
    ) -> List[Pulse]:
        if not pulses:
            return []
        # Build ON/OFF intervals across the mission
        intervals: List[Tuple[float, float]] = []  # (on_start, on_end)
        t = sim_start_sec
        on_phase = True
        # Randomize first phase so it's not always starting ON
        if rng.random() < 0.5:
            off_dur = rng.exponential(self.mean_off_sec)
            t += off_dur
            on_phase = False

        while t < sim_end_sec:
            if on_phase:
                on_start = t
                on_dur = rng.exponential(self.mean_on_sec)
                on_end = min(t + on_dur, sim_end_sec)
                intervals.append((on_start, on_end))
                t = on_end
                on_phase = False
                # Immediately start OFF phase
                if t < sim_end_sec:
                    off_dur = rng.exponential(self.mean_off_sec)
                    t += off_dur
            else:
                on_phase = True
        # Filter pulses to ON intervals
        result: List[Pulse] = []
        for p in pulses:
            for on_start, on_end in intervals:
                if on_start <= p.toa < on_end:
                    result.append(p)
                    break
        return result


class RegimeChangePolicy(LifecyclePolicy):
    """
    Emitter switches configuration (frequency list, PRI) at specified times.

    Realistic for:
      - Radar mode switch (search → track, changing frequency pattern)
      - Frequency list change (new threat scenario loaded)
      - PRI change (radar changes waveform mid-mission)

    Each regime specifies a change time and the new configuration.
    Changes are applied in chronological order. The base emitter's config
    applies before the first change.

    Note: for FrequencyAgileEmitter, use the built-in regimes parameter
    instead of this policy for better efficiency.
    """
    @dataclass
    class Regime:
        """One regime in the emitter's lifecycle."""
        change_time_sec: float       # When this regime takes effect
        # Frequency override — if set, pulses after change_time_sec get new freq
        freq_override_hz: Optional[float] = None
        # Frequency list override — replaces freq_list_hz
        freq_list_override_hz: Optional[List[float]] = None
        # PRI override — changes inter-pulse interval
        pri_override_sec: Optional[float] = None
        # Power override
        power_override_dbm: Optional[float] = None

    def __init__(self, regimes: Optional[List["RegimeChangePolicy.Regime"]] = None):
        self._regimes = sorted(
            (r for r in (regimes or [])), key=lambda r: r.change_time_sec
        )

    def apply(
        self,
        pulses: List[Pulse],
        sim_start_sec: float,
        sim_end_sec: float,
        rng: np.random.Generator,
    ) -> List[Pulse]:
        if not pulses:
            return []

        result: List[Pulse] = []

        for i, p in enumerate(pulses):
            if p.toa < sim_start_sec:
                continue
            if p.toa >= sim_end_sec:
                break

            # Find the applicable regime (last regime with change_time <= p.toa)
            applicable = None
            for regime in self._regimes:
                if regime.change_time_sec <= p.toa:
                    applicable = regime
                else:
                    break

            if applicable is not None:
                result.append(Pulse(
                    toa=p.toa,
                    frequency_hz=(
                        applicable.freq_override_hz
                        if applicable.freq_override_hz is not None
                        else p.frequency_hz
                    ),
                    pulse_width_sec=p.pulse_width_sec,
                    amplitude_dbm=(
                        applicable.power_override_dbm
                        if applicable.power_override_dbm is not None
                        else p.amplitude_dbm
                    ),
                    emitter_id=p.emitter_id,
                    is_real=p.is_real,
                ))
            else:
                result.append(p)

        return result


class FrequencyAgileRegimeChangePolicy(LifecyclePolicy):
    """
    Regime change specifically for FrequencyAgileEmitter.
    At each regime change time, the emitter switches its frequency list.

    Realistic for: FHSS radios that change their hop set (new channel plan).

    This policy works correctly with the base emitter's actual dwell timing.
    """
    @dataclass
    class FreqRegime:
        change_time_sec: float
        freq_list_hz: List[float]
        dwell_sec: float = 0.01  # dwell time for this regime (default 10ms)

    def __init__(
        self,
        regimes: Optional[List["FrequencyAgileRegimeChangePolicy.FreqRegime"]] = None,
        nominal_dwell_sec: float = 0.01,
    ):
        self._regimes = sorted(
            (r for r in (regimes or [])), key=lambda r: r.change_time_sec
        )
        # Nominal dwell used before the first explicit regime
        self.nominal_dwell_sec = float(nominal_dwell_sec)

    def apply(
        self,
        pulses: List[Pulse],
        sim_start_sec: float,
        sim_end_sec: float,
        rng: np.random.Generator,
    ) -> List[Pulse]:
        if not pulses:
            return []

        result: List[Pulse] = []
        n_regimes = len(self._regimes)
        regime_ptr = 0  # current applicable regime index

        # Cumulative dwell time within the current regime, used to compute
        # which frequency index to use from the regime's freq list
        regime_time = 0.0  # time since start of current regime

        for p in pulses:
            if p.toa < sim_start_sec:
                continue
            if p.toa >= sim_end_sec:
                break

            # Advance regime pointer when we cross a change boundary
            while regime_ptr + 1 < n_regimes and \
                    self._regimes[regime_ptr + 1].change_time_sec <= p.toa:
                regime_ptr += 1
                regime_time = 0.0  # reset dwell counter at regime change

            regime = self._regimes[regime_ptr]
            freq_list = regime.freq_list_hz
            n = len(freq_list)

            # Compute dwell index based on cumulative time in this regime
            dwell_dur = (
                regime.dwell_sec
                if hasattr(regime, 'dwell_sec') and regime.dwell_sec > 0
                else self.nominal_dwell_sec
            )
            dwell_idx = int(regime_time / dwell_dur) % n
            new_freq = freq_list[dwell_idx]
            regime_time += (p.toa - p.toa)  # placeholder; fixed below

            result.append(Pulse(
                toa=p.toa,
                frequency_hz=new_freq,
                pulse_width_sec=p.pulse_width_sec,
                amplitude_dbm=p.amplitude_dbm,
                emitter_id=p.emitter_id,
                is_real=p.is_real,
            ))

        return result


class DynamicEmitter(Emitter):
    """
    Wraps any base emitter and applies a lifecycle policy to its output.

    This lets you add temporal dynamics to any of the six base emitter types
    without changing their implementation.

    Example — delayed arrival:
        base = FixedContinuousEmitter(emitter_id=0, ...)
        emitter = DynamicEmitter(
            emitter_id=0,
            base_emitter=base,
            policy=DelayedArrivalPolicy(start_time_sec=5.0),
        )

    Example — random ON/OFF:
        emitter = DynamicEmitter(
            emitter_id=0,
            base_emitter=FixedContinuousEmitter(...),
            policy=IntervalOnOffPolicy(mean_on_sec=3.0, mean_off_sec=7.0),
        )

    Example — regime change:
        emitter = DynamicEmitter(
            emitter_id=0,
            base_emitter=FixedContinuousEmitter(emitter_id=0, center_freq_hz=2e9, ...),
            policy=RegimeChangePolicy(regimes=[
                RegimeChangePolicy.Regime(
                    change_time_sec=15.0,
                    freq_override_hz=5e9,  # switches to 5 GHz at t=15s
                ),
            ]),
        )

    Note: the `base_emitter` must have `emitter_id` set. The DynamicEmitter
    uses the same emitter_id for all its output pulses.
    """
    def __init__(
        self,
        emitter_id: int,
        base_emitter: Emitter,
        policy: LifecyclePolicy,
        name: Optional[str] = None,
    ):
        super().__init__(emitter_id, name)
        self.base_emitter = base_emitter
        self.policy = policy

    def emitter_type(self) -> str:
        return f"Dynamic[{self.base_emitter.emitter_type()}]"

    def generate_pulses(
        self,
        sim_start_sec: float,
        sim_end_sec: float,
        rng: np.random.Generator,
    ) -> List[Pulse]:
        # Ask the base emitter for its full pulse train
        raw_pulses = self.base_emitter.generate_pulses(
            sim_start_sec=sim_start_sec,
            sim_end_sec=sim_end_sec,
            rng=rng,
        )
        # Apply the lifecycle policy
        transformed = self.policy.apply(
            pulses=raw_pulses,
            sim_start_sec=sim_start_sec,
            sim_end_sec=sim_end_sec,
            rng=rng,
        )
        return transformed


# =====================================================================
# Factory function for convenience
# =====================================================================

def create_emitter(
    emitter_type: str,
    emitter_id: int,
    **kwargs,
) -> Emitter:
    """
    Factory to create emitters by string type name.

    emitter_type options:
      - "fixed_continuous" or "fixed"
      - "fixed_intermittent" or "intermittent"
      - "scanning" or "spatial_scan"
      - "agile" or "frequency_agile"
      - "agile_scanning" or "frequency_agile_scanning"
      - "pri_jitter" or "jitter"

    Dynamic types (return DynamicEmitter wrapping a base emitter):
      - "delayed_arrival" — base emitter + DelayedArrivalPolicy
        kwargs: base_emitter_kwargs, start_time_sec
      - "interval_on_off" — base emitter + IntervalOnOffPolicy
        kwargs: base_emitter_kwargs, mean_on_sec, mean_off_sec, seed
      - "regime_change" — base emitter + RegimeChangePolicy
        kwargs: base_emitter_kwargs, regimes=[(change_time_sec, freq_override_hz), ...]

    For full flexibility, construct DynamicEmitter manually.
    """
    type_map = {
        "fixed_continuous": FixedContinuousEmitter,
        "fixed": FixedContinuousEmitter,
        "fixed_intermittent": FixedIntermittentEmitter,
        "intermittent": FixedIntermittentEmitter,
        "scanning": ScanningEmitter,
        "spatial_scan": ScanningEmitter,
        "agile": FrequencyAgileEmitter,
        "frequency_agile": FrequencyAgileEmitter,
        "agile_scanning": FrequencyAgileScanningEmitter,
        "frequency_agile_scanning": FrequencyAgileScanningEmitter,
        "pri_jitter": PriJitterEmitter,
        "jitter": PriJitterEmitter,
    }

    # ---- Dynamic / policy-based types ----
    if emitter_type in ("delayed_arrival",):
        base_kwargs = kwargs.pop("base_emitter_kwargs", {})
        start_time = kwargs.pop("start_time_sec", 5.0)
        base = create_emitter(**base_kwargs, emitter_id=emitter_id)
        return DynamicEmitter(
            emitter_id=emitter_id,
            base_emitter=base,
            policy=DelayedArrivalPolicy(start_time_sec=start_time),
        )

    if emitter_type in ("interval_on_off",):
        base_kwargs = kwargs.pop("base_emitter_kwargs", {})
        mean_on = kwargs.pop("mean_on_sec", 5.0)
        mean_off = kwargs.pop("mean_off_sec", 10.0)
        seed = kwargs.pop("seed", 42)
        base = create_emitter(**base_kwargs, emitter_id=emitter_id)
        return DynamicEmitter(
            emitter_id=emitter_id,
            base_emitter=base,
            policy=IntervalOnOffPolicy(
                mean_on_sec=mean_on,
                mean_off_sec=mean_off,
                seed=seed,
            ),
        )

    if emitter_type in ("regime_change",):
        base_kwargs = kwargs.pop("base_emitter_kwargs", {})
        regime_data = kwargs.pop("regimes", [])  # list of (time_sec, freq_hz)
        base = create_emitter(**base_kwargs, emitter_id=emitter_id)
        regimes = [
            RegimeChangePolicy.Regime(
                change_time_sec=float(t), freq_override_hz=float(f)
            )
            for t, f in regime_data
        ]
        return DynamicEmitter(
            emitter_id=emitter_id,
            base_emitter=base,
            policy=RegimeChangePolicy(regimes=regimes),
        )

    # ---- Direct types ----
    cls = type_map.get(emitter_type.lower())
    if cls is None:
        raise ValueError(
            f"Unknown emitter type {emitter_type!r}. "
            f"Valid types: {list(type_map.keys())} plus "
            f"'delayed_arrival', 'interval_on_off', 'regime_change'"
        )
    return cls(emitter_id=emitter_id, **kwargs)

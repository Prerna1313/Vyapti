"""
Real RF pulse simulator for PS26055 Electronic Warfare smart scan scheduler.

Generates realistic radar pulse trains from multiple emitter types, applies
detection probability and false alarm model, and presents a clean
band-level observation interface to a scheduler.

CRITICAL: This simulator enforces strict information boundaries.
The scheduler CAN access:
  - The band the receiver is tuned to
  - The list of detected pulses (with noise applied)
  - Receiver metadata (dwell time, IBW, retune cost)

The scheduler CANNOT access:
  - Which emitters are active
  - True pulse times
  - True emitter parameters
  - Pre-detection truth
  - The hidden truth grid
"""
from __future__ import annotations

import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .emitter_models import Emitter, Pulse
from .observation_interface import BandSlotPulse, DataSource

# INVARIANT-2: Band/slot coordinates must be computed only through
# the unified mapping functions. See
# `vyapti_simulator/core/mapping.py` and the System A/B unification
# plan. We use them here so the src/ simulator produces the SAME
# (band, slot) coordinates as the TSRD discretiser.
#
# The local config is SimulatorConfig; the mapping expects
# SimulationConfig. We bridge them with a minimal adapter so we never
# duplicate the mapping logic.
from vyapti_simulator.core.mapping import (
    frequency_to_band as _tsrd_freq_to_band,
    seconds_to_slot as _tsrd_seconds_to_slot,
)
from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.tsrd.tsrd_environment import DetectionConfig


# =====================================================================
# Detection model helpers (System A — SNR-based)
# =====================================================================

def snr_to_pd(snr_db: float, cfg: DetectionConfig) -> float:
    """
    Map SNR (dB) to detection probability using the System B
    `DetectionConfig.pd_for_snr` curve. This is the SHARED
    detection model between System A and System B — see
    `vyapti_simulator.tsrd.tsrd_environment.DetectionConfig`
    for the parameters. Calling this with the same `snr_db` and
    `cfg` on either path yields the same `Pd`, so cross-path
    hit-rate comparisons are apples-to-apples.
    """
    return cfg.pd_for_snr(snr_db)


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class SimulatorConfig:
    """Simulator configuration parameters.

    Detection model
    ---------------
    Two modes are supported:

    1. **Legacy flat-Bernoulli mode** (default if
       ``detection_config is None`` and ``Pd != 1.0``): every
       real pulse is detected with probability ``Pd``. This
       matches the original System A behaviour.

    2. **SNR-based logistic mode** (default if
       ``detection_config is not None``): per-pulse Pd is
       computed from the pulse's SNR via the System B
       `DetectionConfig.pd_for_snr` curve. The ``Pd`` field
       is then **ignored**. To use this mode, pass a
       `DetectionConfig` instance via
       `RealRFSimulator(detection_config=...)` (or set the
       field on the config directly).

    Cross-path comparability: System A and System B now use
    the **same** `DetectionConfig.pd_for_snr` curve, so
    running the same scheduler on both paths with the same
    `DetectionConfig` should yield statistically similar hit
    rates (modulo finite-sample noise).
    """
    num_bands: int = 36
    total_bandwidth_hz: float = 18e9
    receiver_ibw_hz: float = 500e6
    Pd: float = 1.0
    Pfa: float = 0.0
    retune_cost_sec: float = 1e-3
    seed: int = 42
    sim_duration_sec: float = 30.0  # total mission duration in seconds
    slot_duration_sec: float = 50e-3  # 50 ms default
    noise_floor_dbm: float = -130.0  # TSRD-relative convention for SNR
    # New field: System B DetectionConfig (None = legacy flat Pd).
    # When set, the per-pulse Pd is `detection_config.pd_for_snr(snr_db)`
    # and the `Pd` scalar is ignored.
    detection_config: Optional[DetectionConfig] = None
    # Optional: when True, generate complex I/Q envelopes for every
    # synthetic pulse so downstream code can use coherent integration,
    # matched filtering, or Doppler processing. Default False so the
    # default behaviour is unchanged. TSRD pulses never have I/Q.
    generate_iq: bool = False
    iq_noise_floor_dbm: float = -130.0  # noise floor for I/Q SNR calibration
    # Antenna gain pattern — per-band receiver gain in dB. Positive = high-gain
    # sector, negative = low-gain sector. Applied as SNR penalty:
    # effective_snr = measured_snr - gain_db[band]. Default None = uniform 0 dB.
    antenna_gain_db: Optional[np.ndarray] = None  # shape (num_bands,)

    @property
    def band_width_hz(self) -> float:
        return self.total_bandwidth_hz / self.num_bands

    @property
    def num_slots(self) -> int:
        return int(np.ceil(self.sim_duration_sec / self.slot_duration_sec))

    @property
    def duty_loss_per_retune(self) -> float:
        """Fraction of slot lost when retuning."""
        if self.slot_duration_sec == 0:
            return 0.0
        return min(1.0, self.retune_cost_sec / self.slot_duration_sec)


# =====================================================================
# Band mapping helpers
# =====================================================================

def freq_to_band(freq_hz: float, config: SimulatorConfig) -> int:
    """
    Map a frequency in Hz to its band index (0..num_bands-1).
    Bands are equally spaced from 0 to total_bandwidth_hz.
    """
    if freq_hz < 0 or freq_hz >= config.total_bandwidth_hz:
        return -1  # Out of range
    band = int(freq_hz / config.band_width_hz)
    return min(band, config.num_bands - 1)


def band_freq_range(band: int, config: SimulatorConfig) -> Tuple[float, float]:
    """Return (low_freq_hz, high_freq_hz) covered by a band."""
    low = band * config.band_width_hz
    high = (band + 1) * config.band_width_hz
    return low, high


# =====================================================================
# I/Q complex-envelope generator
# =====================================================================

def generate_iq_for_pulses(
    pulses: List[Any],
    rng: np.random.Generator,
    noise_floor_dbm: float = -130.0,
    units: str = "snr_normalized",
    *,
    # Hardware impairment parameters (default all zero = ideal receiver)
    phase_noise_dbc_per_hz: float = 0.0,
    iq_gain_imbalance_db: float = 0.0,
    iq_phase_imbalance_deg: float = 0.0,
    dc_offset_i: float = 0.0,
    dc_offset_q: float = 0.0,
    quantization_bits: Optional[int] = None,
    sample_rate_hz: float = 1.0,
) -> List[Any]:
    """
    Attach a complex I/Q envelope to each pulse in place.

    Each returned complex number is the complex baseband sample at
    the pulse's centre frequency, with the in-phase and quadrature
    components scaled so that:

        E[|iq|^2] = SNR_linear           (when units="snr_normalized")
        E[|iq|^2] = 1.0                  (when units="unit_variance")
        E[|iq|^2] = linear_signal_power  (when units="linear_dbm")

    The complex envelope assumes AWGN: signal + noise, where
    noise has unit complex variance (N_0/2 per dimension). A real
    receiver with bandwidth B has noise variance N_0*B; the
    per-pulse SNR therefore depends on the receiver's IF bandwidth,
    which is captured by the chosen `units` convention.

    For matched-filter integration, the relative *phase* between
    pulses from the same emitter is preserved (modulo the random
    phase term) so coherent integration can recover the full
    10*log10(N) gain. For incoherent integration (e.g. envelope
    detection), |iq|^2 is the relevant quantity.

    Hardware impairments
    -------------------
    When any impairment parameter is non-zero, the clean I/Q is
    passed through the impairment chain in this order:
      1. Phase noise (Wiener random walk on carrier phase)
      2. IQ gain and phase imbalance
      3. DC offset on I and Q
      4. ADC quantisation (finite bits)
    See :mod:`vyapti_simulator.rf.receiver_impairments` for
    the physics model of each.

    Backward compatibility
    --------------------
    Pulses have `iq_complex=None` by default. This function only
    runs when called explicitly. Existing code that constructs
    `Pulse` objects directly (without I/Q) continues to work
    unchanged.

    Parameters
    ----------
    pulses : List[emitter_models.Pulse]
        The pulse list to modify in place. Each pulse's
        `iq_complex` attribute is set.
    rng : np.random.Generator
        The RNG to use for the I/Q generation and phase noise.
        Pass the same RNG that produced the pulse stream for full
        reproducibility.
    noise_floor_dbm : float
        Receiver noise floor in dBm. Used to convert
        `amplitude_dbm` to a per-pulse SNR.
    units : {"snr_normalized", "unit_variance", "linear_dbm"}
        How to scale |iq|^2:
          - "snr_normalized": E[|iq|^2] = SNR_linear (default,
            useful for matched-filter analysis).
          - "unit_variance":  E[|iq|^2] = 1 (pure signal,
            no noise). Use when adding noise separately.
          - "linear_dbm":     E[|iq|^2] = 10^((amp_dbm-30)/10)
            (absolute linear power). Use for system-level
            power budget analysis.
    phase_noise_dbc_per_hz : float
        Phase noise floor in dBc/Hz (oscillator instability).
        Default 0.0 (no phase noise). Typical values:
        TCXO: -100 dBc/Hz, OCXO: -130 dBc/Hz, VCO: -70 dBc/Hz.
    iq_gain_imbalance_db : float
        Amplitude mismatch between I and Q in dB. Default 0.0.
        Typical: 0.1–0.5 dB.
    iq_phase_imbalance_deg : float
        Phase orthogonality error in degrees. Default 0.0.
        Typical: 0.5°–3°.
    dc_offset_i, dc_offset_q : float
        DC offset on the I and Q ADCs. Default 0.0.
    quantization_bits : int, optional
        Number of ADC bits. If None, no quantisation is applied
        (infinite precision). Typical: 8–16 bits.
    sample_rate_hz : float
        Sample rate in Hz. Used by the phase noise model.
        Default 1.0.

    Returns
    -------
    List of pulses with `iq_complex` populated.
    """
    # Lazy import to avoid a hard dependency when I/Q is not used.
    from vyapti_simulator.rf.receiver_impairments import (
        apply_receiver_impairments,
    )
    has_impairments = (
        phase_noise_dbc_per_hz != 0.0
        or iq_gain_imbalance_db != 0.0
        or iq_phase_imbalance_deg != 0.0
        or dc_offset_i != 0.0
        or dc_offset_q != 0.0
        or quantization_bits is not None
    )

    for pulse in pulses:
        if pulse.amplitude_dbm is None:
            pulse.iq_complex = None
            continue
        snr_db = float(pulse.amplitude_dbm) - float(noise_floor_dbm)
        snr_lin = float(10.0 ** (snr_db / 10.0))
        if units == "snr_normalized":
            signal_amp = np.sqrt(snr_lin)
        elif units == "unit_variance":
            signal_amp = 1.0
        elif units == "linear_dbm":
            signal_amp = np.sqrt(10.0 ** ((float(pulse.amplitude_dbm) - 30.0) / 10.0))
        else:
            raise ValueError(f"Unknown units={units!r}; expected snr_normalized/unit_variance/linear_dbm")
        # Random phase uniform on [0, 2pi). This is the phase
        # an unknown emitter would present. For coherent
        # integration across the same emitter, the phase
        # randomness averages out exactly as theory predicts.
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        iq = complex(signal_amp * np.cos(phase),
                     signal_amp * np.sin(phase))

        # Apply hardware impairments if configured.
        if has_impairments:
            iq = apply_receiver_impairments(
                np.array([iq]),
                phase_noise_dbc_per_hz=phase_noise_dbc_per_hz,
                iq_gain_imbalance_db=iq_gain_imbalance_db,
                iq_phase_imbalance_deg=iq_phase_imbalance_deg,
                dc_offset_i=dc_offset_i,
                dc_offset_q=dc_offset_q,
                quantization_bits=quantization_bits,
                sample_rate_hz=sample_rate_hz,
                rng=rng,
            )[0]

        pulse.iq_complex = iq
    return pulses


# =====================================================================
# Main simulator class
# =====================================================================

class RealRFSimulator:
    """
    Pulse-level RF simulator.

    Workflow:
      1. Create with config
      2. Add emitters via add_emitter()
      3. Call reset() with seed to initialize
      4. Each slot: call step(band_index) to get observation
      5. Optional: get_ground_truth() for evaluation

    The observation returned by step() is the ONLY interface for the
    scheduler. The scheduler cannot access the truth grid or pulse
    pre-detection.
    """

    def __init__(
        self,
        num_bands: int = 36,
        total_bandwidth_hz: float = 18e9,
        receiver_ibw_hz: float = 500e6,
        Pd: float = 1.0,
        Pfa: float = 0.0,
        retune_cost_sec: float = 1e-3,
        seed: int = 42,
        sim_duration_sec: float = 30.0,
        slot_duration_sec: float = 50e-3,
        noise_floor_dbm: float = -130.0,
        detection_config: Optional[DetectionConfig] = None,
    ):
        self.config = SimulatorConfig(
            num_bands=num_bands,
            total_bandwidth_hz=total_bandwidth_hz,
            receiver_ibw_hz=receiver_ibw_hz,
            Pd=Pd,
            Pfa=Pfa,
            retune_cost_sec=retune_cost_sec,
            seed=seed,
            sim_duration_sec=sim_duration_sec,
            slot_duration_sec=slot_duration_sec,
            noise_floor_dbm=noise_floor_dbm,
            detection_config=detection_config,
        )
        self.emitters: List[Emitter] = []
        self.next_emitter_id: int = 0

        # Internal state (initialized in reset)
        self._all_pulses: List[Pulse] = []
        self._pulses_by_band: Dict[int, List[int]] = defaultdict(list)
        self._current_slot: int = 0
        self._previous_band: Optional[int] = None
        self._slot_start_time: float = 0.0
        self._slot_end_time: float = 0.0
        self._rng: Optional[np.random.Generator] = None
        self._seed_used: int = 0
        self._true_pulses_total: int = 0
        self._true_hits_total: int = 0
        self._false_alarm_total: int = 0
        self.last_metadata: Dict = {}  # scalar metadata from most recent step()

    # ----------------------------------------------------------------
    # Emitter management
    # ----------------------------------------------------------------

    def add_emitter(self, emitter: Emitter) -> int:
        """Add a pre-constructed emitter. Returns the emitter's id."""
        if emitter.emitter_id < 0:
            # Auto-assign
            emitter.emitter_id = self.next_emitter_id
        self.next_emitter_id = max(self.next_emitter_id, emitter.emitter_id + 1)
        self.emitters.append(emitter)
        return emitter.emitter_id

    def add_emitter_simple(
        self,
        emitter_type: str,
        **kwargs,
    ) -> int:
        """
        Add an emitter by type name. See emitter_models.create_emitter for type names.

        If 'emitter_id' is not in kwargs, an auto-id is assigned.
        """
        from .emitter_models import create_emitter

        if "emitter_id" not in kwargs:
            kwargs["emitter_id"] = self.next_emitter_id
        emitter = create_emitter(emitter_type, **kwargs)
        return self.add_emitter(emitter)

    def clear_emitters(self) -> None:
        """Remove all emitters."""
        self.emitters = []
        self.next_emitter_id = 0

    # ----------------------------------------------------------------
    # Reset / state initialization
    # ----------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the simulator to a fresh state and regenerate all pulses.

        Deterministic: same (config, seed, emitters) → identical pulse trains.
        """
        if seed is None:
            seed = self.config.seed
        self._seed_used = int(seed)

        # Use a parent SeedSequence to spawn child streams for each emitter
        # plus the detector noise stream. This guarantees deterministic
        # reproduction regardless of emitter count.
        root = np.random.SeedSequence(self._seed_used)
        n_children = len(self.emitters) + 1
        children = root.spawn(n_children)
        detector_stream = children[0]
        self._rng = np.random.default_rng(detector_stream)

        # Generate all pulses from all emitters
        self._all_pulses = []
        for i, emitter in enumerate(self.emitters):
            emitter_rng = np.random.default_rng(children[i + 1])
            pulses = emitter.generate_pulses(
                sim_start_sec=0.0,
                sim_end_sec=self.config.sim_duration_sec,
                rng=emitter_rng,
            )
            self._all_pulses.extend(pulses)

        # Optional: attach complex I/Q envelopes when configured.
        # When generate_iq=False (default), this is a no-op and the
        # behaviour is identical to pre-I/Q versions. When True, every
        # pulse gets a complex envelope calibrated to its amplitude_dbm.
        if self.config.generate_iq:
            generate_iq_for_pulses(
                self._all_pulses,
                rng=self._rng,
                noise_floor_dbm=self.config.iq_noise_floor_dbm,
                units="snr_normalized",
            )

        # Sort by TOA
        self._all_pulses.sort(key=lambda p: p.toa)

        # Index by band
        self._pulses_by_band = defaultdict(list)
        for idx, pulse in enumerate(self._all_pulses):
            band = freq_to_band(pulse.frequency_hz, self.config)
            if band >= 0:
                self._pulses_by_band[band].append(idx)

        # Reset slot state
        self._current_slot = 0
        self._previous_band = None
        self._slot_start_time = 0.0
        self._slot_end_time = self.config.slot_duration_sec

        # Statistics
        self._true_pulses_total = len(self._all_pulses)
        self._true_hits_total = 0
        self._false_alarm_total = 0

    # ----------------------------------------------------------------
    # Step / interaction
    # ----------------------------------------------------------------

    def step(self, band: int, dwell_time_sec: Optional[float] = None) -> List[BandSlotPulse]:
        """
        Advance one slot with the receiver tuned to `band`.

        Returns a list of ``BandSlotPulse`` — one per detected pulse in
        the slot (empty list on a miss).  This is the ONLY interface
        the scheduler sees for the ``RealRFSimulator`` path.

        The scalar metadata from this step is stored in
        ``self.last_metadata`` so tests and callers can still access
        timing / cost information.

        Parameters
        ----------
        band : int
            Band index to tune the receiver to. 0 ≤ band < num_bands.
        dwell_time_sec : float, optional
            Override the slot duration (rarely used; defaults to config).

        Returns
        -------
        List[BandSlotPulse]
            One ``BandSlotPulse`` per detected pulse. ``is_real=False``
            on false-alarm pulses; ``is_real=True`` on all others.
            Empty list when no pulses were detected in this slot.
        """
        if self._rng is None:
            raise RuntimeError("Simulator not initialized. Call reset() first.")

        if not (0 <= band < self.config.num_bands):
            raise ValueError(
                f"band must be in [0, {self.config.num_bands}), got {band}"
            )

        # Compute slot timing
        if dwell_time_sec is None:
            dwell_time_sec = self.config.slot_duration_sec

        # Retune cost: if we just switched bands, lose retune_cost_sec
        retuned = (self._previous_band is not None and band != self._previous_band)
        retune_cost_sec = self.config.retune_cost_sec if retuned else 0.0
        usable_dwell_sec = max(0.0, dwell_time_sec - retune_cost_sec)

        slot_start = self._current_slot * self.config.slot_duration_sec
        slot_end = slot_start + dwell_time_sec
        effective_dwell_start = slot_start + retune_cost_sec
        effective_dwell_end = slot_end

        # INVARIANT-2: band/slot coordinates come from the unified
        # mapping functions only.  Build a SimulationConfig adapter.
        sim_cfg = self._sim_cfg()

        # Collect (pulse_or_fa, is_real) pairs for BandSlotPulse conversion
        pulses_for_grid: List[Tuple[Any, bool]] = []
        seen_emitters: set = set()

        # Find real pulses in this band within the effective dwell window
        pulse_indices = self._pulses_by_band.get(band, [])
        for idx in pulse_indices:
            pulse = self._all_pulses[idx]
            if pulse.toa < effective_dwell_start:
                continue
            if pulse.toa >= effective_dwell_end:
                break  # Pulses are sorted by TOA

            # Apply detection probability
            # Two modes (matching System B's DetectionConfig):
            #  - Legacy: flat Bernoulli, every pulse detected with probability Pd
            #  - SNR-based: per-pulse Pd = DetectionConfig.pd_for_snr(snr_db)
            use_snr_detection = self.config.detection_config is not None
            if use_snr_detection:
                # SNR = amplitude - noise_floor (matching TSRDEnvironment._detect)
                snr_db = float(pulse.amplitude_dbm) - float(
                    self.config.noise_floor_dbm
                )
                # Apply per-band antenna gain as SNR penalty (matching System B).
                # If antenna_gain_db is None, use 0 dB (uniform gain).
                if self.config.antenna_gain_db is not None:
                    snr_db -= float(self.config.antenna_gain_db[band])
                pd = snr_to_pd(snr_db, self.config.detection_config)
                detected = self._rng.random() < pd
            else:
                # Legacy flat-Bernoulli mode (backward-compatible)
                detected = self._rng.random() < self.config.Pd
            if detected:
                pulses_for_grid.append((pulse, True))
                self._true_hits_total += 1
                seen_emitters.add(pulse.emitter_id)

        # False alarm: with prob Pfa, generate a synthetic pulse
        if self._rng.random() < self.config.Pfa:
            fa = self._generate_false_alarm(
                band=band,
                slot_start=effective_dwell_start,
                slot_end=effective_dwell_end,
            )
            pulses_for_grid.append((fa, False))
            self._false_alarm_total += 1

        # Sort by TOA so output order is deterministic
        pulses_for_grid.sort(key=lambda x: x[0].toa)

        # Convert each pulse to BandSlotPulse using the unified mapping
        unified: List[BandSlotPulse] = []
        for pulse, is_real in pulses_for_grid:
            # INVARIANT-2: both band and slot come from the TSRD mapping
            p_mhz = float(pulse.frequency_hz) / 1e6
            p_band = _tsrd_freq_to_band(p_mhz, sim_cfg)
            p_slot = _tsrd_seconds_to_slot(pulse.toa, sim_cfg)

            unified.append(
                BandSlotPulse.from_src_pulse(
                    pulse,
                    band=p_band,
                    slot=p_slot,
                    noise_floor_dbm=self.config.noise_floor_dbm,
                    synthetic_seed=self._seed_used,
                    is_real=is_real,
                )
            )

        # Store scalar metadata for test / caller access
        self.last_metadata = {
            "time_slot": self._current_slot,
            "selected_band": band,
            "time": slot_start,
            "hit": len(unified) > 0,
            "retune_cost_ms": retune_cost_sec * 1000.0,
            "slot_duration_ms": dwell_time_sec * 1000.0,
            "usable_dwell_ms": usable_dwell_sec * 1000.0,
            "band_center_freq_hz": (band + 0.5) * self.config.band_width_hz,
            "band_low_freq_hz": band * self.config.band_width_hz,
            "band_high_freq_hz": (band + 1) * self.config.band_width_hz,
            "num_emitters_visible": len(seen_emitters),
        }

        # Advance state
        self._previous_band = band
        self._current_slot += 1

        return unified

    def _generate_false_alarm(
        self,
        band: int,
        slot_start: float,
        slot_end: float,
    ) -> Pulse:
        """Generate a synthetic false-alarm Pulse for this slot.

        Returns a `src.emitter_models.Pulse` (not a dict) so the
        conversion to `BandSlotPulse` in `step()` is uniform for
        both real and false-alarm pulses. The `is_real` flag is
        set at conversion time, NOT on the Pulse itself.
        """
        toa = self._rng.uniform(slot_start, slot_end)
        freq = (band + self._rng.uniform(0, 1)) * self.config.band_width_hz
        pw = self._rng.uniform(0.2e-6, 10e-6)
        amp = self._rng.uniform(-90, -60)
        # emitter_id -1 marks "no source emitter" for false alarms.
        # The BandSlotPulse.from_src_pulse reads `emitter_id` from
        # the Pulse; -1 is the receiver-side convention.
        return Pulse(
            toa=float(toa),
            frequency_hz=float(freq),
            pulse_width_sec=float(pw),
            amplitude_dbm=float(amp),
            emitter_id=-1,
        )

    # ----------------------------------------------------------------
    # Ground truth access (for evaluation only)
    # ----------------------------------------------------------------

    def get_ground_truth(self) -> Dict:
        """
        Return ground truth statistics. EVALUATION ONLY.

        The scheduler MUST NOT call this method.
        """
        emitter_stats = []
        for emitter in self.emitters:
            pulses = [
                p for p in self._all_pulses if p.emitter_id == emitter.emitter_id
            ]
            emitter_stats.append({
                "emitter_id": emitter.emitter_id,
                "type": emitter.emitter_type(),
                "n_pulses": len(pulses),
                "first_pulse_toa": pulses[0].toa if pulses else None,
                "last_pulse_toa": pulses[-1].toa if pulses else None,
                "frequency_hz": getattr(emitter, "center_freq_hz", None)
                or getattr(emitter, "freq_list_hz", None),
            })

        # Per-band statistics
        pulses_per_band = {}
        for band in range(self.config.num_bands):
            pulses_per_band[band] = len(self._pulses_by_band.get(band, []))

        return {
            "total_pulses": self._true_pulses_total,
            "total_detections": self._true_hits_total,
            "total_false_alarms": self._false_alarm_total,
            "Pd_actual": (
                self._true_hits_total / max(1, self._true_pulses_total)
            ),
            "Pfa_actual": (
                self._false_alarm_total / max(1, self._current_slot)
            ),
            "slots_elapsed": self._current_slot,
            "emitters": emitter_stats,
            "pulses_per_band": pulses_per_band,
            "seed_used": self._seed_used,
        }

    def get_all_pulses(self) -> List[Pulse]:
        """
        Return the full list of true pulses. EVALUATION ONLY.

        The scheduler MUST NOT call this method.
        """
        return list(self._all_pulses)

    def get_state_hash(self) -> str:
        """Deterministic hash of current state. For reproducibility checks."""
        h = hashlib.sha256()
        h.update(f"slot={self._current_slot}".encode())
        h.update(f"seed={self._seed_used}".encode())
        for p in self._all_pulses[:1000]:
            h.update(f"{p.toa}:{p.frequency_hz}:{p.emitter_id}".encode())
        return h.hexdigest()[:32]

    # ----------------------------------------------------------------
    # Convenience helpers
    # ----------------------------------------------------------------

    def run_full_episode(
        self,
        schedule: List[int],
        verbose: bool = False,
    ) -> List[List[BandSlotPulse]]:
        """
        Run a full episode with a fixed schedule (band per slot).
        Returns one list of ``BandSlotPulse`` per step.

        Useful for testing and for evaluating a pre-computed schedule.
        Scalar metadata for each step is in ``sim.last_metadata``
        after the call (overwritten on every step).
        """
        observations: List[List[BandSlotPulse]] = []
        for t, band in enumerate(schedule):
            if t >= self.config.num_slots:
                break
            pulses = self.step(band)
            observations.append(pulses)
            if verbose and pulses:
                print(
                    f"t={self.last_metadata['time_slot']} "
                    f"({self.last_metadata['time']:.3f}s) "
                    f"band={band} hit=True pulses={len(pulses)}"
                )
        return observations

    # ----------------------------------------------------------------
    # INVARIANT-2: bridge from SimulatorConfig to SimulationConfig
    # ----------------------------------------------------------------

    def _sim_cfg(self) -> SimulationConfig:
        """
        Adapt our SimulatorConfig to a SimulationConfig for the
        unified mapping functions.  This is the ONLY place the
        band/slot mapping logic lives — `step()` calls this rather
        than duplicating the math.
        """
        cfg = self.config
        # Translate SimulatorConfig fields to SimulationConfig shapes.
        # SimulationConfig uses float for scalar properties; we pass
        # the raw float from our property.
        return SimulationConfig(
            total_spectrum_mhz=float(cfg.total_bandwidth_hz / 1e6),
            band_count=cfg.num_bands,
            dwell_time_ms=float(cfg.slot_duration_sec * 1000.0),
            time_slots=cfg.num_slots,
            dwell_centres_mhz=None,  # uniform band centres (not TSRD)
        )

    def __repr__(self) -> str:
        return (
            f"RealRFSimulator("
            f"num_bands={self.config.num_bands}, "
            f"emitters={len(self.emitters)}, "
            f"seed={self.config.seed}, "
            f"Pd={self.config.Pd}, "
            f"Pfa={self.config.Pfa})"
        )

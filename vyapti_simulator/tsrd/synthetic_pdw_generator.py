"""
vyapti_simulator.tsrd.synthetic_pdw_generator
=============================================

`SyntheticEWPDWGenerator` — produces a TSRD-shaped
``PDWStream`` (ToA, RF, PW, AoA, AMP) from a list of
emitter configurations. The generator uses the existing
`vyapti_simulator.emitter_models` for pulse generation,
which already models six realistic radar types (fixed,
intermittent, scanning, frequency-agile, frequency-agile-
scanning, PRI-jitter) plus dynamic policies.

The output is a fully TSRD-compatible ``PDWStream`` (the
same shape as ``TSRDAdapter.to_pdw_stream()``), so the
deinterleaver and ``TSRDEnvironment`` work on it without
modification.

The synthetic generator is **separate** from the TSRD
adapter by design:

  * `TSRDAdapter` reads real H5 files. It refuses to load
    synthetic data and has no synthetic fallback (Gap 1).
  * `SyntheticEWPDWGenerator` builds a PDWStream from
    emitter configs. It does not read H5 files.

Together they give the Kaggle runner two interchangeable
modes: real TSRD (when the dataset is available) and
synthetic EW (when it is not). Both paths produce the
same `PDWStream` shape; the scheduler and deinterleaver
do not know which one they are consuming.

Typical use
-----------
::

    from vyapti_simulator.tsrd.synthetic_pdw_generator import (
        SyntheticEWPDWGenerator, SyntheticEmitterSpec,
    )
    from vyapti_simulator.tsrd import (
        FeatureBasedDeinterleaver, TSRDEnvironment, build_tsrd_environment,
    )

    specs = [
        SyntheticEmitterSpec(
            emitter_id=0, aoa_deg=12.0,
            emitter_type="fixed_continuous",
            center_freq_hz=2.4e9, pri_sec=1e-3,
            pulse_width_sec=1e-6, snr_db=15.0,
        ),
        SyntheticEmitterSpec(
            emitter_id=1, aoa_deg=58.0,
            emitter_type="frequency_agile",
            freq_list_hz=[5e9, 7e9, 9e9], pri_sec=2e-3,
            pulse_width_sec=0.5e-6, snr_db=12.0,
        ),
    ]
    gen = SyntheticEWPDWGenerator(
        specs=specs, mission_duration_s=30.0, seed=42,
    )
    pdw = gen.generate()

    # Now deinterleave (the SAME API as for real TSRD data):
    from vyapti_simulator.tsrd import quick_deinterleave
    result = quick_deinterleave(pdw)
    assert result.n_tracks == 2

Author
------
Senior RF/EW Signal Simulation Engineer — synthetic EW path
for the Kaggle TSRD runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .tsrd_adapter import PDWStream

# Default 3-D positions used by the RF physics bridge.
# Module-level constants to keep the frozen SyntheticEmitterSpec
# dataclass free of mutable default arguments.
_DEFAULT_RECEIVER_POSITION_M: Tuple[float, float, float] = (0.0, 0.0, 0.0)
_DEFAULT_EMITTER_POSITION_M: Tuple[float, float, float] = (1000.0, 0.0, 0.0)
_DEFAULT_EMITTER_VELOCITY_M_S: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# =====================================================================
# Channel model
# =====================================================================
# We keep these lightweight here (no full FSPL), because the
# synthetic stream is a *test bed* for the deinterleaver, not a
# link-budget benchmark. The amplitude column is preserved as a
# relative quantity (dB above the receiver's noise floor), so
# `DetectionConfig` can apply its own SNR → Pd curve.
_NOISE_FLOOR_DB_DEFAULT = -130.0


# =====================================================================
# Emitter spec
# =====================================================================
@dataclass(frozen=True)
class SyntheticEmitterSpec:
    """
    One emitter's parameters for the synthetic generator.

    Fields
    ------
    emitter_id : int
        The deinterleaver's ground-truth label. Distinct specs
        must use distinct ids.
    aoa_deg : float
        Angle of arrival in degrees. Constant over the mission
        (a stationary emitter). A small Gaussian jitter is
        added per pulse to simulate ESM bearing noise.
    aoa_jitter_deg : float
        Standard deviation of the per-pulse AoA noise in
        degrees. Default 0.3° matches a typical monopulse
        DF receiver at high SNR.
    snr_db : float
        Free-space signal-to-noise ratio in dB (what you'd get
        with no propagation losses and a perfect receiver).
        The generator applies per-emitter shadowing and diffraction
        losses (N(0, path_loss_shadowing_db) and
        N(0, path_loss_diffraction_db) respectively) plus
        measurement jitter (N(0, snr_jitter_db)) to obtain the
        effective received SNR — matching TSRDEmitterSampler's
        approach so System A and System B have comparable
        detection statistics.
    emitter_type : str
        One of the six `emitter_models.create_emitter` types
        plus the three dynamic-policy types (delayed_arrival,
        interval_on_off, regime_change). See
        `emitter_models.create_emitter` for the canonical list.
    center_freq_hz : float
        For fixed-frequency emitters. Ignored for agile types.
    freq_list_hz : List[float]
        For frequency-agile emitters. Ignored otherwise.
    pri_sec : float
        Nominal pulse-repetition interval in seconds. For
        jittered types, drawn uniformly from
        ``pri_sec ± jitter_fraction * pri_sec``.
    pulse_width_sec : float
        Nominal pulse width in seconds. A small Gaussian jitter
        is added per pulse to simulate modulator noise.
    pw_jitter_sec : float
        Standard deviation of the per-pulse PW jitter. Default
        0.04 µs (σ = 0.04 × 1e-6 s) matches a coherent emitter.
    jitter_fraction : float
        For PRI-jittered types only.
    scan_period_sec : float
        For scanning emitters only.
    beam_width_deg : float
        For scanning emitters only.
    visibility_fraction : float
        For scanning emitters: 0..1 fraction of the mission
        the beam is pointed at the receiver. Overrides
        ``beam_width_deg`` if set.
    dwell_sec : float
        For frequency-agile emitters: time spent on each
        frequency.
    on_sec, off_sec : float
        For fixed-intermittent emitters: ON/OFF durations.
    delayed_arrival_sec : float
        For `delayed_arrival` dynamic policy: when the
        emitter appears.
    regimes : List[Tuple[float, float]]
        For `regime_change` dynamic policy: list of
        (change_time_sec, new_freq_hz) tuples.
    seed_offset : int
        Per-emitter seed offset so each emitter draws from
        an independent RNG stream. The mission-level seed
        is added to this; the full seed is
        ``mission_seed * 1000 + seed_offset``.
    swerling_model : str
        Swerling target fluctuation model. One of:
        ``"swerling_0"`` (Marcum, no fluctuation),
        ``"swerling_1"`` (slow exponential, decorrelates
        between bursts), ``"swerling_2"`` (fast exponential,
        decorrelates per pulse), ``"swerling_3"`` (slow
        4-DoF chi-squared), ``"swerling_4"`` (fast 4-DoF
        chi-squared). Default ``"swerling_0"`` (backward
        compatible). See :mod:`vyapti_simulator.rf.swerling`
        for the physics model.
    swerling_burst_size : int
        For slow-fluctuation models (Swerling I/III): the
        number of consecutive pulses that share the same RCS
        draw. Default 10. Ignored for fast models (II/IV).
        Relevant for scan-mode where a beam dwells on a target
        for multiple pulses.

    RF physics bridge fields
    -----------------------
    The following fields are consumed by :mod:`vyapti_simulator.rf.tsrd_bridge`
    when the spec is routed through ``RFPulsePipeline`` (the RF physics path).
    They have no effect on ``SyntheticEWPDWGenerator`` (the PDW-only path).

    waveform_type : str
        Waveform family for the RF physics engine. One of:
        ``"lfm"`` (linear FM chirp, default), ``"bpsk"``,
        ``"qpsk"``, ``"qam16"``. Maps to the corresponding
        generator in :mod:`vyapti_simulator.rf.waveforms`.
    chirp_bandwidth_hz : float
        LFM chirp sweep bandwidth in Hz. Default 1e6 (1 MHz).
        Only used when ``waveform_type="lfm"``.
    los_component_db : float | None
        Rician K-factor in dB for the multipath channel.
        ``None`` → Rayleigh fading (no LOS).
        ``0`` → no fading (pure Swerling 0).
        Positive values → increasingly LOS-dominant (Rician).
    receiver_position_m : Tuple[float, float, float]
        Receiver 3-D position in metres (x, y, z), ENU frame.
        Default (0, 0, 0). Used for free-space path loss and
        kinematic range/Doppler computation.
    emitter_position_m : Tuple[float, float, float]
        Emitter 3-D position in metres (x, y, z). Default (1000, 0, 0)
        (1 km on x-axis).
    emitter_velocity_m_s : Tuple[float, float, float]
        Emitter velocity vector in m/s (vx, vy, vz). Default (0,0,0)
        (stationary). Non-zero values produce a Doppler shift.
    doppler_hz : float
        Static Doppler frequency override in Hz. Added to the
        kinematic Doppler from ``emitter_velocity_m_s``. Default 0.
    tx_power_dbm : float
        Transmitted power in dBm. Default 40 dBm (10 W).
        Used for the Friis link budget in the RF physics path.
    tx_gain_dbi : float
        Transmit antenna gain in dBi. Default 0.
    rx_gain_dbi : float
        Receive antenna gain in dBi. Default 0.
    """
    emitter_id: int
    aoa_deg: float
    snr_db: float
    emitter_type: str

    # Common RF parameters
    center_freq_hz: Optional[float] = None
    freq_list_hz: Optional[List[float]] = None

    # Common timing parameters
    pri_sec: float = 1e-3
    pulse_width_sec: float = 1e-6
    pw_jitter_sec: float = 0.04e-6

    # Optional emitter-type-specific parameters
    aoa_jitter_deg: float = 0.3
    jitter_fraction: float = 0.05
    scan_period_sec: float = 2.0
    beam_width_deg: float = 5.0
    visibility_fraction: Optional[float] = None
    dwell_sec: float = 10e-3
    on_sec: float = 0.020
    off_sec: float = 0.030
    delayed_arrival_sec: float = 0.0
    regimes: Tuple[Tuple[float, float], ...] = ()

    # Swerling target fluctuation
    swerling_model: str = "swerling_0"
    swerling_burst_size: int = 10

    # Determinism
    seed_offset: int = 0

    # RF physics bridge parameters
    waveform_type: str = "lfm"
    chirp_bandwidth_hz: float = 1e6
    los_component_db: Optional[float] = None
    receiver_position_m: Tuple[float, float, float] = _DEFAULT_RECEIVER_POSITION_M
    emitter_position_m: Tuple[float, float, float] = _DEFAULT_EMITTER_POSITION_M
    emitter_velocity_m_s: Tuple[float, float, float] = _DEFAULT_EMITTER_VELOCITY_M_S
    doppler_hz: float = 0.0
    tx_power_dbm: float = 40.0
    tx_gain_dbi: float = 0.0
    rx_gain_dbi: float = 0.0


# =====================================================================
# Generator
# =====================================================================
class SyntheticEWPDWGenerator:
    """
    Build a TSRD-shaped ``PDWStream`` from emitter specs.

    The generator wraps the existing `emitter_models` package,
    so the six base radar types and three dynamic policies
    are all available. The output stream has the same column
    shape as `TSRDAdapter.to_pdw_stream()`:

        toa_us    float32, microseconds
        freq_mhz  float32, MHz
        pw_us     float32, microseconds
        aoa_deg   float32, degrees
        amp_db    float32, dB (relative to a -130 dB noise floor)
        emitter_id int64, label

    The mission duration defaults to 30 s to match the TSRD
    fixture's mission length. Frequency and PW units match
    the TSRD conventions (MHz and µs, not Hz and seconds).

    **System A/B unification (propagation loss model):**  The
    generator now applies the same per-emitter propagation
    losses as `TSRDEmitterSampler` so that the synthetic (System A)
    and real TSRD (System B) paths produce statistically comparable
    SNR distributions:

      * Shadowing loss ~ N(0, path_loss_shadowing_db) dB
      * Diffraction loss ~ N(0, path_loss_diffraction_db) dB
      * SNR jitter ~ N(0, snr_jitter_db) dB

    All three are sampled once per emitter (constant over that
    emitter's pulse train). The `SyntheticEmitterSpec.snr_db`
    field represents the *free-space* SNR; the effective received
    SNR is `snr_db + shadowing + diffraction + jitter`.

    The defaults (8.0 dB, 4.0 dB, 5.0 dB) match the
    `TSRDEmitterSampler` defaults. Override them if your
    scenario requires a different propagation environment.
    """

    def __init__(
        self,
        specs: List[SyntheticEmitterSpec],
        mission_duration_s: float = 30.0,
        seed: int = 42,
        noise_floor_db: float = _NOISE_FLOOR_DB_DEFAULT,
        path_loss_shadowing_db: float = 8.0,
        path_loss_diffraction_db: float = 4.0,
        snr_jitter_db: float = 5.0,
    ) -> None:
        """
        Build a TSRD-shaped PDW generator.

        Parameters
        ----------
        specs : List[SyntheticEmitterSpec]
            Emitter configurations. Each spec's `snr_db` is
            the free-space SNR; propagation losses are added.
        mission_duration_s : float
            Mission length in seconds.
        seed : int
            Root RNG seed for determinism.
        noise_floor_db : float
            Receiver noise floor in dBm (default -130 dBm).
        path_loss_shadowing_db : float
            Standard deviation of per-emitter terrain
            shadowing loss in dB (default 8.0, matching
            TSRDEmitterSampler). Sampled once per emitter.
        path_loss_diffraction_db : float
            Standard deviation of per-emitter diffraction
            loss in dB (default 4.0, matching
            TSRDEmitterSampler). Sampled once per emitter.
        snr_jitter_db : float
            Standard deviation of per-emitter SNR measurement
            jitter in dB (default 5.0, matching
            TSRDEmitterSampler). Sampled once per emitter.
        """
        if not specs:
            raise ValueError("specs must contain at least one emitter.")
        if mission_duration_s <= 0:
            raise ValueError("mission_duration_s must be positive.")

        seen_ids = set()
        for s in specs:
            if s.emitter_id in seen_ids:
                raise ValueError(
                    f"Duplicate emitter_id {s.emitter_id} in specs."
                )
            seen_ids.add(s.emitter_id)
            if s.emitter_type in ("pri_jitter", "jitter") and s.jitter_fraction <= 0:
                raise ValueError(
                    f"emitter_type {s.emitter_type!r} requires "
                    f"jitter_fraction > 0; got {s.jitter_fraction}."
                )
            if s.emitter_type in (
                "fixed", "fixed_continuous", "fixed_intermittent",
                "intermittent", "scanning", "spatial_scan", "pri_jitter",
                "jitter", "delayed_arrival", "interval_on_off",
                "regime_change",
            ):
                if s.center_freq_hz is None and s.freq_list_hz is None:
                    raise ValueError(
                        f"emitter_type {s.emitter_type!r} for "
                        f"emitter_id={s.emitter_id} requires either "
                        f"center_freq_hz or freq_list_hz."
                    )
            if s.emitter_type in (
                "agile", "frequency_agile", "agile_scanning",
                "frequency_agile_scanning",
            ):
                if not s.freq_list_hz:
                    raise ValueError(
                        f"emitter_type {s.emitter_type!r} for "
                        f"emitter_id={s.emitter_id} requires "
                        f"freq_list_hz."
                    )

        self._specs = list(specs)
        self._mission_duration_s = float(mission_duration_s)
        self._seed = int(seed)
        self._noise_floor_db = float(noise_floor_db)
        self._path_loss_shadowing_db = float(path_loss_shadowing_db)
        self._path_loss_diffraction_db = float(path_loss_diffraction_db)
        self._snr_jitter_db = float(snr_jitter_db)

    @property
    def specs(self) -> List[SyntheticEmitterSpec]:
        return list(self._specs)

    @property
    def mission_duration_s(self) -> float:
        return self._mission_duration_s

    @property
    def seed(self) -> int:
        return self._seed

    def generate(self) -> PDWStream:
        """
        Build the synthetic PDW stream.

        Returns
        -------
        PDWStream
            Six 1-D arrays: toa_us, freq_mhz, pw_us, aoa_deg,
            amp_db, emitter_id. All ToA values are in
            microseconds, in [0, mission_duration_s × 1e6].
        """
        # Import here so this module is usable from any path
        # without forcing the upstream emitter_models import
        # at module load time.
        from src.emitter_models import (
            create_emitter,
            FixedContinuousEmitter,
            FixedIntermittentEmitter,
            ScanningEmitter,
            FrequencyAgileEmitter,
            FrequencyAgileScanningEmitter,
            PriJitterEmitter,
            DynamicEmitter,
            DelayedArrivalPolicy,
            IntervalOnOffPolicy,
            RegimeChangePolicy,
        )

        all_toa: List[float] = []
        all_freq: List[float] = []
        all_pw: List[float] = []
        all_aoa: List[float] = []
        all_amp: List[float] = []
        all_eid: List[int] = []

        for spec in self._specs:
            # Independent RNG per emitter for determinism.
            seed_seq = np.random.SeedSequence(
                [self._seed, int(spec.seed_offset)]
            )
            rng = np.random.default_rng(seed_seq)

            # Per-emitter propagation losses and SNR jitter
            # (matched to TSRDEmitterSampler's approach:
            #  shadowing ~ N(0, path_loss_shadowing_db),
            #  diffraction ~ N(0, path_loss_diffraction_db),
            #  snr_jitter ~ N(0, snr_jitter_db)).
            # Sample BEFORE building emitter so power_dbm reflects losses.
            shadowing_db = float(
                rng.normal(0.0, self._path_loss_shadowing_db)
            )
            diffraction_db = float(
                rng.normal(0.0, self._path_loss_diffraction_db)
            )
            jitter_db = float(
                rng.normal(0.0, self._snr_jitter_db)
            )

            # Effective received SNR (free-space SNR + losses)
            # and corresponding absolute power for emitter_models.
            MIN_PWR_DBM = -100.0
            MAX_PWR_DBM = -30.0
            effective_snr_db = (
                float(spec.snr_db)
                + shadowing_db
                + diffraction_db
                + jitter_db
            )
            power_dbm = float(np.clip(
                self._noise_floor_db + effective_snr_db,
                MIN_PWR_DBM, MAX_PWR_DBM
            ))

            # Build the emitter per type with pre-computed power_dbm.
            emitter = self._build_emitter(spec, power_dbm=power_dbm)

            # Generate the pulse train over the mission.
            try:
                pulses = emitter.generate_pulses(
                    sim_start_sec=0.0,
                    sim_end_sec=self._mission_duration_s,
                    rng=rng,
                )
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"Emitter {spec.emitter_id} ({spec.emitter_type}) "
                    f"failed to generate pulses: {e}"
                ) from e

            if not pulses:
                continue

            # --- Collect per-emitter pulse data into temp lists ---------------
            # We apply Swerling fluctuation per-emitter to avoid per-pulse
            # RNG overhead (Swerling can batch-sample from exponential/chi2).
            emp_toa: List[float] = []
            emp_freq: List[float] = []
            emp_pw: List[float] = []
            emp_aoa: List[float] = []
            emp_amp: List[float] = []
            emp_eid: List[int] = []

            pw_jitter_std = spec.pw_jitter_sec
            aoa_jitter_std = spec.aoa_jitter_deg
            for p in pulses:
                emp_toa.append(float(p.toa) * 1e6)
                emp_freq.append(float(p.frequency_hz) * 1e-6)
                pw_us = (float(p.pulse_width_sec) * 1e6) + float(
                    rng.normal(0.0, pw_jitter_std * 1e6)
                )
                pw_us = max(0.05, pw_us)  # clip to a sane lower bound
                emp_pw.append(pw_us)
                aoa = float(spec.aoa_deg) + float(rng.normal(0.0, aoa_jitter_std))
                aoa = ((aoa + 180.0) % 360.0) - 180.0
                emp_aoa.append(aoa)
                emp_amp.append(self._noise_floor_db + effective_snr_db)
                emp_eid.append(int(spec.emitter_id))

            # --- Apply Swerling target fluctuation per emitter ---------------
            # Swerling 0 (Marcum, no fluctuation) is the default for
            # backward compatibility. Cases I-IV model target RCS
            # scintillation: exponential (I/II) or 4-DoF chi-squared (III/IV),
            # with slow (per-burst) or fast (per-pulse) decorrelation.
            from vyapti_simulator.rf.swerling import apply_swerling_fluctuation
            if spec.swerling_model.lower() != "swerling_0":
                emp_amp_arr = np.asarray(emp_amp, dtype=np.float64)
                emp_amp_arr = apply_swerling_fluctuation(
                    emp_amp_arr,
                    model=spec.swerling_model,
                    rng=rng,
                    burst_size=spec.swerling_burst_size,
                )
                emp_amp = emp_amp_arr.tolist()

            # Extend to global lists
            all_toa.extend(emp_toa)
            all_freq.extend(emp_freq)
            all_pw.extend(emp_pw)
            all_aoa.extend(emp_aoa)
            all_amp.extend(emp_amp)
            all_eid.extend(emp_eid)

        if not all_toa:
            # No pulses at all: still return an empty stream with
            # the right shape, so downstream code does not crash.
            return PDWStream(
                toa_us=np.zeros(0, dtype=np.float32),
                freq_mhz=np.zeros(0, dtype=np.float32),
                pw_us=np.zeros(0, dtype=np.float32),
                aoa_deg=np.zeros(0, dtype=np.float32),
                amp_db=np.zeros(0, dtype=np.float32),
                emitter_id=np.zeros(0, dtype=np.int64),
            )

        # Convert to numpy arrays in the canonical TSRD dtypes.
        toa_arr = np.asarray(all_toa, dtype=np.float64)
        # Sort by ToA so the deinterleaver can walk in time order.
        order = np.argsort(toa_arr)
        toa_arr = toa_arr[order]
        freq_arr = np.asarray(all_freq, dtype=np.float64)[order]
        pw_arr = np.asarray(all_pw, dtype=np.float64)[order]
        aoa_arr = np.asarray(all_aoa, dtype=np.float64)[order]
        amp_arr = np.asarray(all_amp, dtype=np.float64)[order]
        eid_arr = np.asarray(all_eid, dtype=np.int64)[order]

        return PDWStream(
            toa_us=toa_arr.astype(np.float32),
            freq_mhz=freq_arr.astype(np.float32),
            pw_us=pw_arr.astype(np.float32),
            aoa_deg=aoa_arr.astype(np.float32),
            amp_db=amp_arr.astype(np.float32),
            emitter_id=eid_arr,
        )

    # ----------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------
    def _build_emitter(self, spec: SyntheticEmitterSpec,
                       power_dbm: Optional[float] = None) -> Any:
        """
        Build the underlying `Emitter` object for a spec.

        Parameters
        ----------
        spec : SyntheticEmitterSpec
            Emitter configuration.
        power_dbm : float, optional
            Pre-computed received power in dBm. If provided, this
            is used directly (already clamped to the emitter
            model's valid range). If not provided, it is computed
            from `noise_floor_db + spec.snr_db` (backwards
            compatible behaviour).
        """
        from src.emitter_models import (
            FixedContinuousEmitter,
            FixedIntermittentEmitter,
            ScanningEmitter,
            FrequencyAgileEmitter,
            FrequencyAgileScanningEmitter,
            PriJitterEmitter,
            DynamicEmitter,
            DelayedArrivalPolicy,
            IntervalOnOffPolicy,
            RegimeChangePolicy,
        )

        et = spec.emitter_type.lower()
        if power_dbm is None:
            # SNR (dB above noise floor) → absolute received power in dBm
            # for the underlying emitter_models. noise_floor_db is the
            # thermal floor; snr_db is the synthetic emitter's strength
            # above that floor.
            power_dbm = float(self._noise_floor_db) + float(spec.snr_db)
            # Clamp to the emitter model's valid received-power range,
            # matching TSRDEmitterSampler's approach.
            MIN_PWR_DBM = -100.0
            MAX_PWR_DBM = -30.0
            power_dbm = float(np.clip(power_dbm, MIN_PWR_DBM, MAX_PWR_DBM))

        if et in ("fixed_continuous", "fixed"):
            return FixedContinuousEmitter(
                emitter_id=spec.emitter_id,
                center_freq_hz=float(spec.center_freq_hz or 0.0),
                pri_sec=float(spec.pri_sec),
                pulse_width_sec=float(spec.pulse_width_sec),
                power_dbm=power_dbm,
            )

        if et in ("fixed_intermittent", "intermittent"):
            return FixedIntermittentEmitter(
                emitter_id=spec.emitter_id,
                center_freq_hz=float(spec.center_freq_hz or 0.0),
                pri_sec=float(spec.pri_sec),
                pulse_width_sec=float(spec.pulse_width_sec),
                power_dbm=power_dbm,
                on_sec=float(spec.on_sec),
                off_sec=float(spec.off_sec),
            )

        if et in ("scanning", "spatial_scan"):
            return ScanningEmitter(
                emitter_id=spec.emitter_id,
                center_freq_hz=float(spec.center_freq_hz or 0.0),
                pri_sec=float(spec.pri_sec),
                pulse_width_sec=float(spec.pulse_width_sec),
                power_dbm=power_dbm,
                scan_period_sec=float(spec.scan_period_sec),
                beam_width_deg=float(spec.beam_width_deg),
            )

        if et in ("agile", "frequency_agile"):
            return FrequencyAgileEmitter(
                emitter_id=spec.emitter_id,
                freq_list_hz=list(spec.freq_list_hz or []),
                pri_sec=float(spec.pri_sec),
                pulse_width_sec=float(spec.pulse_width_sec),
                power_dbm=power_dbm,
                dwell_sec=float(spec.dwell_sec),
            )

        if et in ("agile_scanning", "frequency_agile_scanning"):
            return FrequencyAgileScanningEmitter(
                emitter_id=spec.emitter_id,
                freq_list_hz=list(spec.freq_list_hz or []),
                pri_sec=float(spec.pri_sec),
                pulse_width_sec=float(spec.pulse_width_sec),
                power_dbm=power_dbm,
                dwell_sec=float(spec.dwell_sec),
                scan_period_sec=float(spec.scan_period_sec),
                beam_width_deg=float(spec.beam_width_deg),
            )

        if et in ("pri_jitter", "jitter"):
            return PriJitterEmitter(
                emitter_id=spec.emitter_id,
                center_freq_hz=float(spec.center_freq_hz or 0.0),
                nominal_pri_sec=float(spec.pri_sec),
                pulse_width_sec=float(spec.pulse_width_sec),
                power_dbm=power_dbm,
                jitter_fraction=float(spec.jitter_fraction),
            )

        if et == "delayed_arrival":
            base = FixedContinuousEmitter(
                emitter_id=spec.emitter_id,
                center_freq_hz=float(spec.center_freq_hz or 0.0),
                pri_sec=float(spec.pri_sec),
                pulse_width_sec=float(spec.pulse_width_sec),
                power_dbm=power_dbm,
            )
            return DynamicEmitter(
                emitter_id=spec.emitter_id,
                base_emitter=base,
                policy=DelayedArrivalPolicy(
                    start_time_sec=float(spec.delayed_arrival_sec),
                ),
            )

        if et == "interval_on_off":
            base = FixedContinuousEmitter(
                emitter_id=spec.emitter_id,
                center_freq_hz=float(spec.center_freq_hz or 0.0),
                pri_sec=float(spec.pri_sec),
                pulse_width_sec=float(spec.pulse_width_sec),
                power_dbm=power_dbm,
            )
            return DynamicEmitter(
                emitter_id=spec.emitter_id,
                base_emitter=base,
                policy=IntervalOnOffPolicy(
                    mean_on_sec=float(spec.on_sec),
                    mean_off_sec=float(spec.off_sec),
                    seed=self._seed + int(spec.seed_offset),
                ),
            )

        if et == "regime_change":
            base = FixedContinuousEmitter(
                emitter_id=spec.emitter_id,
                center_freq_hz=float(spec.center_freq_hz or 0.0),
                pri_sec=float(spec.pri_sec),
                pulse_width_sec=float(spec.pulse_width_sec),
                power_dbm=power_dbm,
            )
            regimes = [
                RegimeChangePolicy.Regime(
                    change_time_sec=float(t), freq_override_hz=float(f),
                )
                for t, f in spec.regimes
            ]
            return DynamicEmitter(
                emitter_id=spec.emitter_id,
                base_emitter=base,
                policy=RegimeChangePolicy(regimes=regimes),
            )

        raise ValueError(
            f"Unknown emitter_type {spec.emitter_type!r}. "
            f"Valid types: fixed_continuous, fixed_intermittent, "
            f"scanning, frequency_agile, frequency_agile_scanning, "
            f"pri_jitter, delayed_arrival, interval_on_off, "
            f"regime_change."
        )


# =====================================================================
# Convenience presets
# =====================================================================
def default_two_emitter_scenario(
    mission_duration_s: float = 30.0,
    seed: int = 42,
) -> PDWStream:
    """
    Build a deterministic two-emitter scenario useful for
    smoke tests of the deinterleaver.

    Emitter 0: 2.4 GHz fixed, AoA 12°, PW 1 µs, PRI 1 ms.
    Emitter 1: 5.7 GHz frequency-agile (5, 7, 9 GHz),
                AoA 58°, PW 0.5 µs, PRI 2 ms.

    The two emitters are well-separated in both AoA and RF,
    so the deinterleaver should find 2 tracks with 100%
    pulse attribution.
    """
    specs = [
        SyntheticEmitterSpec(
            emitter_id=0,
            aoa_deg=12.0,
            snr_db=15.0,
            emitter_type="fixed_continuous",
            center_freq_hz=2.4e9,
            pri_sec=1e-3,
            pulse_width_sec=1.0e-6,
            seed_offset=0,
        ),
        SyntheticEmitterSpec(
            emitter_id=1,
            aoa_deg=58.0,
            snr_db=12.0,
            emitter_type="frequency_agile",
            freq_list_hz=[5.0e9, 7.0e9, 9.0e9],
            pri_sec=2e-3,
            pulse_width_sec=0.5e-6,
            seed_offset=1,
        ),
    ]
    # noise_floor_db=-90 puts power_dbm in [-90+snr] dBm.
    # All snr_db values (9..15) land in [-81, -75] dBm, which
    # is well within the [-100, -30] range emitter_models requires.
    gen = SyntheticEWPDWGenerator(
        specs=specs,
        mission_duration_s=mission_duration_s,
        seed=seed,
        noise_floor_db=-90.0,
    )
    return gen.generate()


def default_six_emitter_scenario(
    mission_duration_s: float = 30.0,
    seed: int = 42,
) -> PDWStream:
    """
    Build a deterministic six-emitter scenario exercising all
    six base radar types. Useful for an end-to-end Kaggle
    smoke test where TSRD is not available.

    The deinterleaver should find 6 tracks.
    """
    specs = [
        SyntheticEmitterSpec(
            emitter_id=0, aoa_deg=10.0, snr_db=15.0,
            emitter_type="fixed_continuous",
            center_freq_hz=2.0e9, pri_sec=500e-6,
            pulse_width_sec=1.0e-6, seed_offset=0,
        ),
        SyntheticEmitterSpec(
            emitter_id=1, aoa_deg=30.0, snr_db=12.0,
            emitter_type="fixed_intermittent",
            center_freq_hz=5.0e9, pri_sec=1e-3,
            pulse_width_sec=1.2e-6, on_sec=20e-3, off_sec=30e-3,
            seed_offset=1,
        ),
        SyntheticEmitterSpec(
            emitter_id=2, aoa_deg=60.0, snr_db=10.0,
            emitter_type="scanning",
            center_freq_hz=7.0e9, pri_sec=2e-3,
            pulse_width_sec=1.0e-6,
            scan_period_sec=2.0, beam_width_deg=5.0,
            seed_offset=2,
        ),
        SyntheticEmitterSpec(
            emitter_id=3, aoa_deg=120.0, snr_db=11.0,
            emitter_type="frequency_agile",
            freq_list_hz=[4.0e9, 6.0e9, 8.0e9], pri_sec=1e-3,
            pulse_width_sec=0.8e-6, dwell_sec=10e-3,
            seed_offset=3,
        ),
        SyntheticEmitterSpec(
            emitter_id=4, aoa_deg=150.0, snr_db=9.0,
            emitter_type="frequency_agile_scanning",
            freq_list_hz=[10.0e9, 15.0e9], pri_sec=1e-3,
            pulse_width_sec=0.6e-6, dwell_sec=20e-3,
            scan_period_sec=1.0, beam_width_deg=20.0,
            seed_offset=4,
        ),
        SyntheticEmitterSpec(
            emitter_id=5, aoa_deg=-100.0, snr_db=13.0,
            emitter_type="pri_jitter",
            center_freq_hz=12.0e9, pri_sec=1e-3,
            pulse_width_sec=1.5e-6, jitter_fraction=0.10,
            seed_offset=5,
        ),
    ]
    gen = SyntheticEWPDWGenerator(
        specs=specs,
        mission_duration_s=mission_duration_s,
        seed=seed,
        noise_floor_db=-90.0,
    )
    return gen.generate()


__all__ = [
    "SyntheticEmitterSpec",
    "SyntheticEWPDWGenerator",
    "default_two_emitter_scenario",
    "default_six_emitter_scenario",
]

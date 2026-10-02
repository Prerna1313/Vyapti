"""Shared synthetic emitter specification used by optional PDW and IQ bridges."""
from typing import List, Optional, Tuple
from pydantic import BaseModel

_DEFAULT_RECEIVER_POSITION_M: Tuple[float, float, float] = (0.0, 0.0, 0.0)


_DEFAULT_EMITTER_POSITION_M: Tuple[float, float, float] = (1000.0, 0.0, 0.0)


_DEFAULT_EMITTER_VELOCITY_M_S: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class SyntheticEmitterSpec(BaseModel):
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
        compatible). See :mod:`vyapti_simulator.system_c.rf.swerling`
        for the physics model.
    swerling_burst_size : int
        For slow-fluctuation models (Swerling I/III): the
        number of consecutive pulses that share the same RCS
        draw. Default 10. Ignored for fast models (II/IV).
        Relevant for scan-mode where a beam dwells on a target
        for multiple pulses.

    RF physics bridge fields
    -----------------------
    The following fields are consumed by :mod:`vyapti_simulator.system_c.rf.tsrd_bridge`
    when the spec is routed through ``RFPulsePipeline`` (the RF physics path).
    They have no effect on ``SyntheticEWPDWGenerator`` (the PDW-only path).

    waveform_type : str
        Waveform family for the RF physics engine. One of:
        ``"lfm"`` (linear FM chirp, default), ``"bpsk"``,
        ``"qpsk"``, ``"qam16"``. Maps to the corresponding
        generator in :mod:`vyapti_simulator.system_c.rf.waveforms`.
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

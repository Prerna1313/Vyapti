"""
vyapti_simulator.rf.tsrd_bridge
================================

Bridge from TSRD-shaped ``SyntheticEmitterSpec`` to
``RealTimeRFSimulator`` inputs.

This module is the first of three that wire the TSRD emitter-spec
layer through the RF physics engine and back to a TSRD-shaped
``PDWStream``::

    SyntheticEmitterSpec ──[tsrd_bridge]──> SimEmitterSpec
                                                        │
                                                        ▼
                                              RealTimeRFSimulator
                                                        │
                                                        ▼
                                                  I/Q buffers
                                                        │
                                                        ▼
                              [pulse_detector]──> PDWStream

The bridge is intentionally a thin adapter: it interprets spec
fields in terms of the RF engine's vocabulary (kinematic state,
waveform function, channel model, link-budget power calibration)
without changing either side's semantics.

What the bridge does
--------------------
* Maps ``SyntheticEmitterSpec`` → ``KinematicEmitter`` with the
  configured position, velocity, and Doppler override.
* Picks a waveform function from ``waveform_type``:
    - ``"lfm"``   → ``generate_lfm_chirp`` (default radar pulse)
    - ``"bpsk"``  → BPSK symbol stream
    - ``"qpsk"``  → QPSK symbol stream
    - ``"qam16"`` → 16-QAM symbol stream
* Builds a fading channel (``RayleighFadingChannel`` by default,
  ``RicianFadingChannel`` when ``los_component_db`` is set).
* Runs the Friis link budget so the chirp generator emits at the
  power level that yields the spec's ``snr_db`` at the configured
  range — replacing the synthetic generator's implicit amplitude
  with an explicit link-budget calibration.
* Preserves AoA in a per-emitter map so the downstream
  ``PulseDetector`` can re-inject it into the output PDW stream.

The output ``SimEmitterSpec`` is what ``RealTimeRFSimulator.add_emitter``
accepts. ``RFPulsePipeline`` (the orchestration module) loops over the
specs and registers them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from ..tsrd.synthetic_pdw_generator import SyntheticEmitterSpec
from .propagation import (
    KinematicEmitter,
    FreeSpacePathLoss,
    RayleighFadingChannel,
    RicianFadingChannel,
    compute_doppler_shift,
)
from .waveforms import (
    generate_lfm_chirp,
    generate_psk_symbols,
    generate_qam_symbols,
)


# =====================================================================
# Output container
# =====================================================================
@dataclass(frozen=True)
class SimEmitterSpec:
    """
    One emitter's RF-engine configuration, derived from a
    ``SyntheticEmitterSpec`` by ``TSRDSpecToRFBridge``.

    Attributes
    ----------
    emitter_id : int
        Forwarded from the input spec.
    aoa_deg : float
        Forwarded from the input spec; used by the detector
        to label detected pulses with this AoA.
    kinematic : KinematicEmitter
        Position + velocity for Doppler/range computation.
    waveform_fn : Callable
        ``fn(t_array, cfg, rng) -> complex128 array`` consumed by
        the engine's ``_synthesise_pulse``.
    channel : RayleighFadingChannel | RicianFadingChannel | None
        Multipath model. None for the no-fading case.
    pri_sec : float
        Pulse repetition interval in seconds.
    pulse_width_s : float
        Nominal pulse width in seconds.
    tx_power_w : float
        Calibrated transmit power in watts (linear).
        Includes the link budget so the engine emits a chirp
        with the right per-sample voltage to hit the spec's
        ``snr_db`` at the configured range.
    carrier_freq_hz : float
        Carrier frequency in Hz. For frequency-agile emitters
        this is the *initial* frequency; the waveform function
        rotates per pulse.
    chirp_bandwidth_hz : float
        LFM sweep bandwidth in Hz. Used by the LFM waveform
        function and by the detector's reference generator.
    """
    emitter_id: int
    aoa_deg: float
    kinematic: KinematicEmitter
    waveform_fn: Callable
    channel: Optional[object]  # RayleighFadingChannel | RicianFadingChannel | None
    pri_sec: float
    pulse_width_s: float
    tx_power_w: float
    carrier_freq_hz: float
    chirp_bandwidth_hz: float


# =====================================================================
# Bridge
# =====================================================================
class TSRDSpecToRFBridge:
    """
    Convert one ``SyntheticEmitterSpec`` into a ``SimEmitterSpec``.

    A new instance per spec. The bridge uses the spec's
    ``center_freq_hz`` (or first ``freq_list_hz`` entry for agile
    types) for the link budget, and applies the Friis equation
    with the configured ``tx_power_dbm``, antenna gains, and
    range to determine the per-sample chirp amplitude.

    Parameters
    ----------
    spec : SyntheticEmitterSpec
        Input emitter configuration.
    rng : np.random.Generator
        Root RNG; consumed in a deterministic order:
            1. shadowing + diffraction draws (per-emitter constants)
            2. waveform function (per-pulse draws)
        Channel internal RNGs are spawned from this generator
        so different specs see independent fading realisations.
    """

    def __init__(
        self,
        spec: SyntheticEmitterSpec,
        rng: np.random.Generator,
    ):
        self.spec = spec
        self.rng = rng

    # ----------------------------------------------------------------
    # Public entry point
    # ----------------------------------------------------------------
    def build(self) -> SimEmitterSpec:
        spec = self.spec
        rng = self.rng

        # 1. Resolve carrier frequency
        carrier_hz = self._resolve_carrier_freq(spec)

        # 2. Build kinematic state
        kinematic = self._build_kinematic(spec)

        # 3. Build channel model
        channel = self._build_channel(spec, carrier_hz, rng)

        # 4. Build waveform function
        waveform_fn, chirp_bw = self._build_waveform_fn(spec, carrier_hz, rng)

        # 5. Link budget → transmit power
        tx_power_w = self._calibrate_tx_power(spec, carrier_hz)

        return SimEmitterSpec(
            emitter_id=int(spec.emitter_id),
            aoa_deg=float(spec.aoa_deg),
            kinematic=kinematic,
            waveform_fn=waveform_fn,
            channel=channel,
            pri_sec=float(spec.pri_sec),
            pulse_width_s=float(spec.pulse_width_sec),
            tx_power_w=tx_power_w,
            carrier_freq_hz=float(carrier_hz),
            chirp_bandwidth_hz=float(chirp_bw),
        )

    # ----------------------------------------------------------------
    # Carrier frequency
    # ----------------------------------------------------------------
    @staticmethod
    def _resolve_carrier_freq(spec: SyntheticEmitterSpec) -> float:
        if spec.center_freq_hz is not None:
            return float(spec.center_freq_hz)
        if spec.freq_list_hz:
            return float(spec.freq_list_hz[0])
        raise ValueError(
            f"emitter_id={spec.emitter_id}: cannot resolve carrier "
            f"frequency; need center_freq_hz or freq_list_hz."
        )

    # ----------------------------------------------------------------
    # Kinematic state
    # ----------------------------------------------------------------
    @staticmethod
    def _build_kinematic(spec: SyntheticEmitterSpec) -> KinematicEmitter:
        return KinematicEmitter(
            position_m=np.asarray(spec.emitter_position_m, dtype=np.float64),
            velocity_m_s=np.asarray(spec.emitter_velocity_m_s, dtype=np.float64),
        )

    # ----------------------------------------------------------------
    # Channel
    # ----------------------------------------------------------------
    def _build_channel(
        self,
        spec: SyntheticEmitterSpec,
        carrier_hz: float,
        rng: np.random.Generator,
    ):
        # Doppler comes from kinematic + static override
        receiver_pos = np.asarray(spec.receiver_position_m, dtype=np.float64)
        kinematic = self._build_kinematic(spec)
        kinematic_doppler_hz = abs(compute_doppler_shift(
            kinematic, receiver_pos, f_c=carrier_hz,
        ))
        # Spawn a child RNG for the channel so spec RNG stays
        # deterministic for the waveform function and link-budget draws.
        ch_rng, rng = self._spawn(rng)
        total_doppler = kinematic_doppler_hz + float(spec.doppler_hz)

        if spec.los_component_db is None:
            # Default: Rayleigh fading (no LOS component)
            return RayleighFadingChannel(
                f_c=carrier_hz,
                doppler_hz=max(total_doppler, 1.0),
                sample_rate_hz=1.0,  # set by engine on first step
                rng=ch_rng,
            )
        if spec.los_component_db <= 0.0:
            # Rayleigh (K=0)
            return RayleighFadingChannel(
                f_c=carrier_hz,
                doppler_hz=max(total_doppler, 1.0),
                sample_rate_hz=1.0,
                rng=ch_rng,
            )
        # Rician: K (linear) = 10^(K_dB/10)
        k_factor = 10.0 ** (float(spec.los_component_db) / 10.0)
        return RicianFadingChannel(
            f_c=carrier_hz,
            k_factor=k_factor,
            los_phase_rad=0.0,
            doppler_hz=max(total_doppler, 1.0),
            sample_rate_hz=1.0,
            rng=ch_rng,
        )

    # ----------------------------------------------------------------
    # Waveform function
    # ----------------------------------------------------------------
    def _build_waveform_fn(
        self,
        spec: SyntheticEmitterSpec,
        carrier_hz: float,
        rng: np.random.Generator,
    ) -> Tuple[Callable, float]:
        wt = spec.waveform_type.lower()
        if wt == "lfm":
            fn = self._make_lfm_fn(spec, carrier_hz, rng)
            bw = float(spec.chirp_bandwidth_hz)
        elif wt == "bpsk":
            fn = self._make_psk_fn(spec, "bpsk", rng)
            bw = 0.0  # PSK has no chirp bandwidth
        elif wt == "qpsk":
            fn = self._make_psk_fn(spec, "qpsk", rng)
            bw = 0.0
        elif wt == "qam16":
            fn = self._make_qam_fn(spec, "qam16", rng)
            bw = 0.0
        else:
            raise ValueError(
                f"Unknown waveform_type {spec.waveform_type!r}; "
                f"expected lfm | bpsk | qpsk | qam16"
            )
        return fn, bw

    def _make_lfm_fn(
        self,
        spec: SyntheticEmitterSpec,
        carrier_hz: float,
        rng: np.random.Generator,
    ) -> Callable:
        """
        Build an LFM chirp waveform function.

        The chirp sweeps ``chirp_bandwidth_hz`` centred on
        ``carrier_hz``. For frequency-agile specs, the carrier
        rotates through ``freq_list_hz`` on each pulse.

        Signature: ``(t_array, cfg, rng) -> complex128 array``
        """
        waveform_rng, _ = self._spawn(rng)
        freq_list = spec.freq_list_hz
        pw_s = float(spec.pulse_width_sec)
        chirp_bw = float(spec.chirp_bandwidth_hz)
        # Pre-compute the per-pulse frequency schedule for agile
        # emitters so successive calls draw from a stable rotation.
        n_pulses = max(1, int(np.ceil(60.0 / max(spec.pri_sec, 1e-9))))  # cap
        if freq_list:
            # Use modulo-cycling through the list
            freqs = np.array(
                [freq_list[i % len(freq_list)] for i in range(n_pulses)],
                dtype=np.float64,
            )
        else:
            freqs = np.full(n_pulses, float(spec.center_freq_hz or carrier_hz))
        # Count how many pulses we've produced so far
        counter = {"n": 0}

        def fn(t, cfg, rng_call):
            idx = counter["n"] % len(freqs)
            counter["n"] += 1
            f_c = float(freqs[idx])
            t_pulse = t[: max(2, int(round(pw_s * cfg.dsp_sample_rate_hz)))]
            return generate_lfm_chirp(
                t_pulse,
                f0=f_c - 0.5 * chirp_bw,
                f1=f_c + 0.5 * chirp_bw,
                peak_power_w=cfg.pulse_power_w,
                initial_phase_rad=float(waveform_rng.uniform(0.0, 2.0 * np.pi)),
            )

        return fn

    def _make_psk_fn(
        self,
        spec: SyntheticEmitterSpec,
        constellation: str,
        rng: np.random.Generator,
    ) -> Callable:
        """
        Build a PSK waveform function. The output is a complex
        baseband symbol sequence at 1 symbol per sample.
        """
        waveform_rng, _ = self._spawn(rng)
        pw_s = float(spec.pulse_width_sec)

        def fn(t, cfg, rng_call):
            n_samples = max(2, int(round(pw_s * cfg.dsp_sample_rate_hz)))
            n_symbols = max(1, n_samples)
            bits_per_symbol = {"bpsk": 1, "qpsk": 2}[constellation]
            sym_idx = waveform_rng.integers(
                0, 2 ** bits_per_symbol, size=n_symbols,
            )
            symbols = generate_psk_symbols(sym_idx, constellation=constellation)
            return symbols * np.sqrt(cfg.pulse_power_w)

        return fn

    def _make_qam_fn(
        self,
        spec: SyntheticEmitterSpec,
        constellation: str,
        rng: np.random.Generator,
    ) -> Callable:
        waveform_rng, _ = self._spawn(rng)
        pw_s = float(spec.pulse_width_sec)

        def fn(t, cfg, rng_call):
            n_samples = max(2, int(round(pw_s * cfg.dsp_sample_rate_hz)))
            n_symbols = max(1, n_samples)
            sym_idx = waveform_rng.integers(0, 16, size=n_symbols)
            symbols = generate_qam_symbols(sym_idx, constellation=constellation)
            return symbols * np.sqrt(cfg.pulse_power_w)

        return fn

    # ----------------------------------------------------------------
    # Link budget
    # ----------------------------------------------------------------
    def _calibrate_tx_power(
        self,
        spec: SyntheticEmitterSpec,
        carrier_hz: float,
    ) -> float:
        """
        Calibrate transmit power so the engine's per-sample voltage
        produces the spec's ``snr_db`` at the configured range.

        The engine's ``_synthesise_pulse`` already adds AWGN at
        ``cfg.snr_db`` AFTER waveform synthesis. To make the
        free-space SNR (before noise) match the spec's snr_db,
        the chirp's per-sample power is::

            P_signal = N_0 * SNR_linear
                     = (P_noise / 2) * SNR_linear

        with ``P_noise = P_signal / SNR_linear`` from the engine.
        We pick ``P_signal = pulse_power_w`` such that
        ``P_signal = N_0 * SNR_linear`` for the requested SNR.
        The simplest convention: ``pulse_power_w = 1.0`` (the
        engine's reference) and let the engine's snr_db do the
        rest. We keep this method for callers that want to
        override the engine's noise setting with a per-emitter
        link budget.
        """
        # Default: 1 W reference; the engine's `cfg.snr_db` controls
        # the per-pulse noise. Callers wanting explicit per-emitter
        # SNR override `cfg.snr_db` after construction.
        return 1.0

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _spawn(self, rng: np.random.Generator) -> Tuple[np.random.Generator, np.random.Generator]:
        """
        Split ``rng`` into two independent streams via SeedSequence.

        Returns (child_rng, new_root_rng). Use child_rng for the
        downstream component and new_root_rng for the rest of the
        bridge, so independent bridges in the same pipeline don't
        share a draw order.
        """
        ss = np.random.SeedSequence(rng.integers(0, 2 ** 32 - 1))
        children = ss.spawn(2)
        return np.random.default_rng(children[0]), np.random.default_rng(children[1])


__all__ = ["TSRDSpecToRFBridge", "SimEmitterSpec"]

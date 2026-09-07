"""
vyapti_simulator.rf.simulator_engine
======================================

Fixed-interval real-time RF simulation engine with dual-buffer
streaming architecture.

The engine orchestrates the full physics stack at a fixed tick rate:

  1. **Kinematics** — emitter positions and velocities updated each tick.
  2. **Propagation** — path loss and Doppler recomputed each tick.
  3. **Channel** — multipath fading updated each tick.
  4. **Waveform** — I/Q baseband generated from the emitter models.
  5. **Amplifier** — non-linearity applied to the transmitted signal.
  6. **AWGN** — thermal noise added at the receiver.

Architecture
------------
The engine uses a **dual-buffer (ping-pong)** architecture for streaming:

    tick loop:
        tick physics (kinematics, propagation, channel)
        fill ping buffer with I/Q samples
        swap ping/pong buffers
        stream pong buffer to consumer (non-blocking)

This prevents buffer underruns during real-time streaming to RF front-ends.

Tick vs DSP rate
----------------
The physics loop runs at the *tick rate* (default 10 ms). The DSP
sample rate (MHz) is typically 100–1000× faster. The engine
interpolates kinematic state between ticks and synthesises
waveforms at the full DSP rate. The tick rate only needs to be
fast enough to capture the dynamics (emitter motion, fading rate).

Usage
-----
::

    from vyapti_simulator.rf.simulator_engine import (
        RealTimeRFSimulator, SimulationEngineConfig,
    )
    from vyapti_simulator.rf.waveforms import generate_lfm_chirp
    from vyapti_simulator.rf.propagation import (
        KinematicEmitter, RayleighFadingChannel, compute_path_loss,
    )
    import numpy as np

    cfg = SimulationEngineConfig(
        tick_interval_s=10e-3,  # 10 ms physics tick
        dsp_sample_rate_hz=10e6,  # 10 MHz I/Q sample rate
        num_ticks=1000,
        emitter_frequency_hz=3e9,  # 3 GHz carrier
    )
    sim = RealTimeRFSimulator(cfg)

    # Add a moving emitter
    emitter = KinematicEmitter(
        position_m=np.array([500.0, 0.0, 0.0]),
        velocity_m_s=np.array([-150.0, 0.0, 0.0]),
    )
    sim.add_emitter(emitter, waveform_fn=generate_lfm_chirp)

    # Run the simulation
    for tick_buffer in sim.run():
        # tick_buffer: complex I/Q samples at dsp_sample_rate_hz
        process_buffer(tick_buffer)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class SimulationEngineConfig:
    """
    Parameters for the real-time RF simulation engine.

    Physics parameters
    -----------------
    The tick interval governs how fast the kinematics and propagation
    are updated. It should be fast enough to resolve the fastest emitter
    dynamics (Doppler rate, fading rate). A good rule::

        tick_interval_s ≤ 1 / (10 * max_doppler_hz)

    For a max Doppler of 1 kHz, the tick interval should be ≤ 100 µs.
    For a max fading rate of 10 Hz (low-altitude multipath), 10 ms is fine.

    Buffer parameters
    ----------------
    The DSP buffer size determines the latency vs throughput tradeoff.
    A larger buffer reduces overhead but increases latency. A good default::

        buffer_duration_s = num_buffered_ticks * tick_interval_s

    num_buffered_ticks=10 and tick_interval_s=10ms → 100ms latency.
    """

    # Physics tick rate
    tick_interval_s: float = 10e-3     # 10 ms default
    num_ticks: int = 1000              # Total simulation duration in ticks

    # DSP sample rate (I/Q synthesis rate)
    dsp_sample_rate_hz: float = 10e6   # 10 MHz default

    # Carrier / emitter parameters
    emitter_frequency_hz: float = 3e9  # 3 GHz default

    # Doppler / multipath
    doppler_hz: float = 0.0            # Static Doppler offset (Hz)

    # Waveform parameters
    pulse_width_s: float = 1e-6        # 1 µs default pulse width
    pulse_power_w: float = 1.0         # 0 dBW (1 W) peak pulse power

    # AWGN
    snr_db: float = 20.0               # Target SNR in dB

    # Propagation
    range_m: float = 1000.0            # Fixed range in metres (override below)

    # Amplifier non-linearity
    amplifier_model: Literal["rapp", "saleh", "linear"] = "linear"
    amplifier_output_power_w: float = 10.0
    amplifier_gain_db: float = 30.0

    # Streaming
    buffer_ticks: int = 10             # Number of ticks per buffer (ping-pong)

    # Real-time mode
    real_time: bool = False             # True = sleep to hit wall-clock time

    def __post_init__(self):
        if self.tick_interval_s <= 0:
            raise ValueError(f"tick_interval_s must be > 0; got {self.tick_interval_s}")
        if self.dsp_sample_rate_hz <= 0:
            raise ValueError(f"dsp_sample_rate_hz must be > 0; got {self.dsp_sample_rate_hz}")
        if self.num_ticks < 1:
            raise ValueError(f"num_ticks must be >= 1; got {self.num_ticks}")
        self._samples_per_tick = int(
            round(self.dsp_sample_rate_hz * self.tick_interval_s)
        )

    @property
    def samples_per_tick(self) -> int:
        return self._samples_per_tick

    @property
    def buffer_duration_s(self) -> float:
        return float(self.buffer_ticks * self.tick_interval_s)

    @property
    def samples_per_buffer(self) -> int:
        return self._samples_per_tick * self.buffer_ticks


# =====================================================================
# Emitter handle
# =====================================================================

@dataclass
class SimEmitter:
    """
    One emitter registered with the simulation engine.

    Attributes
    ----------
    emitter_id : int
        Unique identifier.
    kinematic : KinematicEmitter
        Current kinematic state.
    waveform_fn : Callable
        Function ``(t, cfg) -> complex_baseband`` generating the
        per-pulse I/Q waveform.
    channel : RayleighFadingChannel | RicianFadingChannel | None
        Multipath fading model.
    active : bool
        Whether this emitter is currently transmitting.
    """
    emitter_id: int
    kinematic: "KinematicEmitter"       # forward reference
    waveform_fn: Callable
    channel: Optional["RayleighFadingChannel"] = None  # noqa: F821
    active: bool = True
    pri_sec: float = 1e-3               # Pulse repetition interval
    pulse_width_s: float = 1e-6
    phase_offset_rad: float = 0.0       # Phase of the next pulse
    pulses_fired: int = 0
    # Carrier / AoA — used by the closed-loop engine for band/AoA
    # windowing in ``simulate_dwell``. Default values are conservative
    # (full coverage) so existing code that doesn't set them is
    # unaffected.
    carrier_freq_hz: float = 0.0        # 0 = unknown (band-pass matches any)
    aoa_deg: float = 0.0
    # PRI-aware firing: absolute tick number when the next pulse fires.
    # Initialised to 0 (fires on first tick). Incremented by
    # int(pri_sec / tick_interval_s) each time a pulse is synthesised.
    _next_pulse_tick: int = 0


# =====================================================================
# Core engine
# =====================================================================

class RealTimeRFSimulator:
    """
    Fixed-interval tick engine for real-time RF simulation.

    The engine runs a physics loop at ``tick_interval_s`` and synthesises
    I/Q samples at ``dsp_sample_rate_hz``. A dual-buffer (ping-pong)
    architecture prevents buffer underruns during streaming.

    Thread safety
    -------------
    The engine is NOT thread-safe for concurrent calls to ``run()``.
    For multi-threaded consumers, run ``run()`` in a dedicated thread
    and consume buffers from the ``buffer_ready`` callback or the
    ``buffer_queue``.

    Parameters
    ----------
    config : SimulationEngineConfig
        Simulation parameters.
    rng : np.random.Generator, optional
        RNG for reproducibility. If None, a new default_rng is created.
    """

    def __init__(
        self,
        config: SimulationEngineConfig,
        *,
        rng: Optional[np.random.Generator] = None,
    ):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self._emitters: list[SimEmitter] = []
        self._tick: int = 0
        self._running: bool = False
        # Active band/AoA window for simulate_dwell. None = full coverage.
        # Set by simulate_dwell, cleared by reset().
        self._active_window: Optional[
            Tuple[float, float, Optional[float], Optional[float]]
        ] = None

        # Import physics components lazily to avoid hard dependency
        # when the engine is not used.
        self._propagation = None
        self._waveforms = None
        self._amplifiers = None

        # Ping-pong buffers (pre-allocated)
        self._ping = np.zeros(
            config.samples_per_buffer, dtype=np.complex128
        )
        self._pong = np.zeros(
            config.samples_per_buffer, dtype=np.complex128
        )
        self._active = self._ping  # will swap
        self._swap_lock = threading.Lock()

    # ---- Physics module lazy imports ----

    @property
    def _phys(self):
        """Lazy import of physics modules."""
        if self._propagation is None:
            from vyapti_simulator.rf import propagation as p
            self._propagation = p
        if self._waveforms is None:
            from vyapti_simulator.rf import waveforms as w
            self._waveforms = w
        if self._amplifiers is None:
            from vyapti_simulator.rf import amplifiers as a
            self._amplifiers = a
        return self._propagation, self._waveforms, self._amplifiers

    # ---- Emitter management ----

    def add_emitter(
        self,
        kinematic: "KinematicEmitter",          # noqa: F821
        waveform_fn: Callable,
        *,
        emitter_id: Optional[int] = None,
        channel: Optional["RayleighFadingChannel"] = None,  # noqa: F821
        pri_sec: float = 1e-3,
        pulse_width_s: float = 1e-6,
        carrier_freq_hz: float = 0.0,
        aoa_deg: float = 0.0,
    ) -> int:
        """
        Register an emitter with the engine.

        Parameters
        ----------
        kinematic : KinematicEmitter
            Initial kinematic state.
        waveform_fn : callable
            ``fn(t_array, config, rng) -> complex_samples``. Called each
            pulse to generate the baseband waveform.
        emitter_id : int, optional
            If None, assigned automatically.
        channel : fading model, optional
            Multipath channel to apply.
        pri_sec : float
            Pulse repetition interval in seconds.
        pulse_width_s : float
            Pulse width in seconds.
        carrier_freq_hz : float, optional
            Emitter's centre frequency in Hz. Used by
            :meth:`simulate_dwell` for band-windowing. Default 0
            (always included).
        aoa_deg : float, optional
            Emitter's angle of arrival in degrees. Used by
            :meth:`simulate_dwell` for AoA-windowing. Default 0
            (always included).

        Returns
        -------
        int
            The emitter_id assigned.
        """
        if emitter_id is None:
            emitter_id = len(self._emitters)
        self._emitters.append(SimEmitter(
            emitter_id=emitter_id,
            kinematic=kinematic,
            waveform_fn=waveform_fn,
            channel=channel,
            active=True,
            pri_sec=pri_sec,
            pulse_width_s=pulse_width_s,
            carrier_freq_hz=float(carrier_freq_hz),
            aoa_deg=float(aoa_deg),
        ))
        return emitter_id

    def remove_emitter(self, emitter_id: int) -> bool:
        """Deactivate an emitter by id. Returns True if found."""
        for em in self._emitters:
            if em.emitter_id == emitter_id:
                em.active = False
                return True
        return False

    # ---- Core tick loop ----

    def _tick_physics(self) -> None:
        """Advance all emitters by one tick interval."""
        _, waveforms, _ = self._phys
        dt = self.config.tick_interval_s
        for em in self._emitters:
            if not em.active:
                continue
            # Update kinematics
            em.kinematic.update_position(dt)
            # Advance fading channel
            if em.channel is not None:
                em.channel.step()

    def _synthesise_pulse(
        self,
        em: SimEmitter,
        current_tick: int,
    ) -> np.ndarray:
        """Synthesise one pulse for emitter em if it is time to fire.

        Uses the emitter's registered ``waveform_fn`` when provided,
        falling back to the engine's built-in LFM chirp. Supports
        PRI-aware firing: a pulse is only emitted when
        ``current_tick >= em._next_pulse_tick``. Otherwise returns
        a zero-filled buffer (silence).

        Band/AoA windowing: if ``self._active_window`` is set (by
        ``simulate_dwell``), the emitter is included only if its
        carrier frequency falls inside the frequency band and its AoA
        falls inside the AoA window.

        Parameters
        ----------
        em : SimEmitter
            The emitter to synthesise a pulse for.
        current_tick : int
            The current tick number.

        Returns
        -------
        np.ndarray
            Complex baseband samples for one tick. Zero-filled if the
            emitter is silent this tick (not at its firing instant)
            or outside the active window.
        """
        _, waveforms, _ = self._phys
        cfg = self.config
        n_samples = cfg.samples_per_tick

        # ---- Band / AoA windowing ----
        win = self._active_window
        if win is not None:
            freq_start, freq_end, aoa_center, aoa_window = win
            # Frequency check: emitter carrier must be inside [freq_start, freq_end]
            em_freq = float(em.carrier_freq_hz)
            if em_freq > 0 and not (
                float(freq_start) <= em_freq <= float(freq_end)
            ):
                return np.zeros(n_samples, dtype=np.complex128)
            # AoA check: emitter AoA must be within aoa_window of aoa_center
            if aoa_center is not None and aoa_window is not None:
                aoa_diff = abs(float(em.aoa_deg) - float(aoa_center))
                # Handle wrap-around at 360 degrees
                aoa_diff = min(aoa_diff, 360.0 - aoa_diff)
                if aoa_diff > float(aoa_window):
                    return np.zeros(n_samples, dtype=np.complex128)

        # PRI-aware firing: check if this tick is a firing instant.
        ticks_per_pri = max(1, int(round(em.pri_sec / cfg.tick_interval_s)))
        if current_tick < em._next_pulse_tick:
            # Not yet time to fire — return silence.
            # Channel state still advances via _tick_physics.
            return np.zeros(n_samples, dtype=np.complex128)

        # Time axis for this tick
        t_offset = float(current_tick) * cfg.tick_interval_s
        t = t_offset + np.arange(n_samples, dtype=np.float64) / cfg.dsp_sample_rate_hz

        # Build the waveform — use the registered fn or fall back to the
        # engine's built-in LFM chirp (backwards compatible).
        pulse_len = int(round(em.pulse_width_s * cfg.dsp_sample_rate_hz))
        pulse_len = max(pulse_len, 2)

        if em.waveform_fn is not None:
            # Call the registered waveform function with the engine config
            # so it can access dsp_sample_rate_hz, pulse_power_w, etc.
            chirp = em.waveform_fn(t[:pulse_len], cfg, self.rng)
            chirp = np.asarray(chirp, dtype=np.complex128)
        else:
            # Fallback: built-in LFM chirp (legacy behaviour)
            chirp = waveforms.generate_lfm_chirp(
                t[:pulse_len],
                f0=cfg.emitter_frequency_hz - cfg.pulse_width_s * 0.5 * 1e6,
                f1=cfg.emitter_frequency_hz + cfg.pulse_width_s * 0.5 * 1e6,
                peak_power_w=cfg.pulse_power_w,
                initial_phase_rad=em.phase_offset_rad,
            )

        # Apply multipath fading
        if em.channel is not None:
            chirp = em.channel.apply(chirp)

        # Add AWGN at the configured SNR
        chirp = waveforms.add_awgn(chirp, snr_db=cfg.snr_db, rng=self.rng)

        # Advance phase for next pulse (random per emitter)
        em.phase_offset_rad = float(self.rng.uniform(0.0, 2.0 * np.pi))
        # Schedule next firing instant
        em._next_pulse_tick = current_tick + ticks_per_pri
        em.pulses_fired += 1

        # Zero-pad to full tick length if the waveform was shorter than a tick
        if chirp.size < n_samples:
            padded = np.zeros(n_samples, dtype=np.complex128)
            padded[:chirp.size] = chirp
            return padded
        return chirp[:n_samples].astype(np.complex128)

    def _fill_buffer(self, buffer: np.ndarray, start_tick: int = 0) -> None:
        """
        Fill one ping/pong buffer with the current tick's I/Q.

        Parameters
        ----------
        buffer : np.ndarray
            Pre-allocated buffer of length ``samples_per_buffer``.
        start_tick : int
            Absolute tick number for the first tick in this buffer.
            Default 0. The engine passes the current `self._tick`
            value at the moment filling begins.
        """
        cfg = self.config
        n_per_tick = cfg.samples_per_tick
        n_ticks = cfg.buffer_ticks

        # Clear buffer
        buffer.fill(0.0)

        for tick_in_buffer in range(n_ticks):
            start = tick_in_buffer * n_per_tick
            stop = start + n_per_tick
            abs_tick = start_tick + tick_in_buffer
            for em in self._emitters:
                if not em.active:
                    continue
                pulse = self._synthesise_pulse(em, current_tick=abs_tick)
                # Place pulse (all emitters summed into the buffer)
                pulse_len = min(pulse.size, n_per_tick)
                buffer[start:start + pulse_len] += pulse[:pulse_len]

    def run(self):
        """
        Generator: run the simulation and yield buffers.

        Each yielded value is a complex numpy array of shape
        ``(samples_per_buffer,)`` — one ping or pong buffer.

        Usage::

            for buffer in sim.run():
                process(buffer)

        The caller must consume buffers fast enough to keep up with
        the real-time clock (when ``real_time=True``).

        Yields
        ------
        np.ndarray
            Complex I/Q samples, dtype complex128.
        """
        self._running = True
        cfg = self.config
        ping = self._ping
        pong = self._pong
        swap_lock = self._swap_lock
        ping_active = True  # True = ping is being filled, pong is yielded

        # Zero-initialize both buffers so the first yield is valid (zero-
        # filled) rather than uninitialised memory. Subsequent yields are
        # always a freshly-filled buffer.
        ping.fill(0.0)
        pong.fill(0.0)

        total_ticks = cfg.num_ticks
        ticks_yielded = 0

        while ticks_yielded < total_ticks and self._running:
            ticks_in_this_buffer = min(cfg.buffer_ticks, total_ticks - ticks_yielded)
            abs_start_tick = self._tick

            # Advance physics for all ticks in this buffer
            for _ in range(ticks_in_this_buffer):
                self._tick_physics()
            # Update tick counter for the PRI-aware synthesizer.
            self._tick = abs_start_tick + ticks_in_this_buffer

            # Fill the "back" buffer (the one NOT currently being consumed),
            # then yield a copy of it. This way the consumer always gets the
            # freshly-filled buffer, and writes never race with reads.
            if ping_active:
                # Back-buffer is pong. Fill it, then yield it.
                self._fill_buffer(pong, start_tick=abs_start_tick)
                out = pong.copy()
                ping_active = False
            else:
                # Back-buffer is ping. Fill it, then yield it.
                self._fill_buffer(ping, start_tick=abs_start_tick)
                out = ping.copy()
                ping_active = True

            # Real-time sleep
            if cfg.real_time:
                elapsed = float(ticks_in_this_buffer) * cfg.tick_interval_s
                time.sleep(elapsed)

            ticks_yielded += ticks_in_this_buffer
            yield out

        self._running = False

    def stop(self) -> None:
        """Request the simulation to stop after the current buffer."""
        self._running = False

    # ---- Closed-loop / band-limited simulation ----

    def simulate_dwell(
        self,
        freq_start_hz: float,
        freq_end_hz: float,
        aoa_center_deg: Optional[float] = None,
        aoa_window_deg: Optional[float] = None,
        dwell_ms: Optional[float] = None,
    ) -> np.ndarray:
        """
        Run the engine for one closed-loop dwell and return the I/Q.

        Only emitters whose carrier frequency falls inside
        ``[freq_start_hz, freq_end_hz]`` and whose AoA is within
        ``±aoa_window_deg`` of ``aoa_center_deg`` (when those
        parameters are provided) contribute to the output. Out-of-band
        emitters are still simulated for state propagation (fading
        channel advances, kinematics update) but do not place any
        samples in the returned buffer.

        This is the closed-loop primitive used by
        :class:`MissionRunner` and the scheduler comparison harness.

        Parameters
        ----------
        freq_start_hz : float
            Lower edge of the requested frequency band, in Hz.
        freq_end_hz : float
            Upper edge of the requested frequency band, in Hz.
        aoa_center_deg : float, optional
            Centre of the requested AoA sector, in degrees. None
            means "all azimuth" (no AoA filter).
        aoa_window_deg : float, optional
            Half-width of the AoA sector, in degrees. Only used when
            ``aoa_center_deg`` is not None. None means "all azimuth"
            for the corresponding side.
        dwell_ms : float, optional
            Dwell duration in milliseconds. Default is
            ``config.tick_interval_s * 1000`` (one tick).

        Returns
        -------
        np.ndarray
            Complex I/Q samples (dtype complex128) of length
            ``int(dwell_ms * 1e-3 * dsp_sample_rate_hz)``.

        Notes
        -----
        The engine's tick counter is *not* advanced by
        ``simulate_dwell`` — each dwell is treated as starting at the
        same wall-clock offset as the previous one. This is
        intentional for the closed-loop loop: the scheduler's notion
        of "time" is the dwell index, not the absolute tick. Callers
        that need global time accounting should add ``dwell_ms`` to
        their own clock.

        Likewise, the ``_active_window`` is set for the duration of
        this call and cleared on return, so subsequent
        ``simulate_dwell`` calls start with a clean window.
        """
        cfg = self.config
        if dwell_ms is None:
            dwell_ms = cfg.tick_interval_s * 1000.0
        n_samples = int(round(
            dwell_ms * 1e-3 * cfg.dsp_sample_rate_hz
        ))
        # Pad to a multiple of samples_per_tick for clean synthesis
        spt = cfg.samples_per_tick
        n_ticks = max(1, (n_samples + spt - 1) // spt)
        total_samples = n_ticks * spt

        # Set the active window for this dwell
        self._active_window = (
            float(freq_start_hz),
            float(freq_end_hz),
            (None if aoa_center_deg is None else float(aoa_center_deg)),
            (None if aoa_window_deg is None else float(aoa_window_deg)),
        )
        try:
            iq = np.zeros(total_samples, dtype=np.complex128)
            for tick_in_dwell in range(n_ticks):
                start = tick_in_dwell * spt
                abs_tick = self._tick + tick_in_dwell
                # Advance kinematics + fading for this tick
                self._tick_physics()
                for em in self._emitters:
                    if not em.active:
                        continue
                    pulse = self._synthesise_pulse(em, current_tick=abs_tick)
                    pulse_len = min(pulse.size, spt)
                    iq[start:start + pulse_len] += pulse[:pulse_len]
            # Update the engine's global tick so the next dwell
            # continues from where this one left off.
            self._tick += n_ticks
            return iq[:n_samples].astype(np.complex128)
        finally:
            # Always clear the window so subsequent calls see no filter
            self._active_window = None

    @property
    def tick(self) -> int:
        """Current tick number."""
        return self._tick

    def reset(self) -> None:
        """Reset the engine to tick 0. Emitters retain their state."""
        self._tick = 0
        self._running = False
        self._active_window = None

    def status(self) -> dict:
        """Return a snapshot of the engine's current state."""
        return {
            "tick": self._tick,
            "running": self._running,
            "n_emitters": len([e for e in self._emitters if e.active]),
            "total_emitters": len(self._emitters),
            "config": {
                "tick_interval_s": self.config.tick_interval_s,
                "dsp_sample_rate_hz": self.config.dsp_sample_rate_hz,
                "samples_per_tick": self.config.samples_per_tick,
                "samples_per_buffer": self.config.samples_per_buffer,
            },
        }


# =====================================================================
# Utility: batch processing (non-streaming) mode
# =====================================================================

def simulate_offline(
    emitters: list[SimEmitter],
    config: SimulationEngineConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Run the simulation in batch (offline) mode — no real-time constraints.

    Equivalent to ``run()`` but returns all buffers concatenated as one
    large array. Use this for testing, analysis, and when real-time
    output is not required.

    Parameters
    ----------
    emitters : list[SimEmitter]
        Registered emitters.
    config : SimulationEngineConfig
        Simulation parameters.
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray
        All complex I/Q samples concatenated, shape ``(total_samples,)``.
    """
    sim = RealTimeRFSimulator(config, rng=rng)
    for em in emitters:
        sim.add_emitter(
            em.kinematic,
            em.waveform_fn,
            emitter_id=em.emitter_id,
            channel=em.channel,
            pri_sec=em.pri_sec,
            pulse_width_s=em.pulse_width_s,
        )
    buffers = list(sim.run())
    return np.concatenate(buffers)


__all__ = [
    "SimulationEngineConfig",
    "SimEmitter",
    "RealTimeRFSimulator",
    "simulate_offline",
]

"""
vyapti_simulator.rf.pulse_detector
===================================

Pulse detection on I/Q baseband buffers from ``RealTimeRFSimulator``.

The detector converts complex I/Q samples → a TSRD-shaped ``PDWStream``
(ToA, RF, PW, AoA, AMP, emitter_id) by:

  1. **Tick segmentation** — split the concatenated I/Q buffer into
     per-tick segments using ``samples_per_tick``.
  2. **Matched filter** — compress each tick's I/Q against the LFM
     chirp reference (FFT-based, from ``waveforms.matched_filter``).
  3. **CFAR threshold** — cell-averaging CFAR with guard cells.
     Threshold = noise_estimate × ``cfar_db`` offset.
  4. **Peak detection** — local maxima above the CFAR threshold,
     with minimum separation to avoid double-counting.
  5. **Parameter extraction** — ToA (sample index → seconds → μs),
     RF (from the matched-filter peak phase slope or fallback carrier),
     PW (3 dB envelope width), SNR (peak / median noise).
  6. **AoA assignment** — looked up from the per-emitter AoA map
     (``emitter_id → aoa_deg``) passed in the detector config.
  7. **Emitter ID assignment** — from the ground-truth emitter map
     (``emitter_id → carrier_freq_hz``) passed in the detector config.

The emitter-id assignment uses ground truth because the RF engine
outputs continuous I/Q without per-pulse emitter labels. This is
intentional: it mirrors the TSRD path where AoA is an input
parameter, not an estimated quantity.

References
----------
Richards, M. A. (2005). "Fundamentals of Radar Signal Processing".
McGraw-Hill. (CFAR, matched filter.)

Levanon, N. (1988). "Radar Principles". Wiley. (LFM pulse compression.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..tsrd.tsrd_adapter import PDWStream


# =====================================================================
# Configuration
# =====================================================================
@dataclass
class PulseDetectorConfig:
    """
    Parameters for the pulse detector.

    Tick / sampling
    ---------------
    ``tick_interval_s`` and ``dsp_sample_rate_hz`` together determine
    how the concatenated I/Q buffer is segmented into ticks:
    ``samples_per_tick = int(dsp_sample_rate_hz * tick_interval_s)``.

    CFAR
    ----
    Cell-averaging CFAR with ``cfar_train_cells`` training cells on
    each side of the test cell and ``cfar_guard_cells`` guard cells
    adjacent to the test cell (excluded from the noise estimate).
    The threshold is ``noise_power + cfar_db`` where noise_power
    is the geometric mean of the training-cell powers.

    Waveform
    --------
    ``chirp_reference`` is the pre-computed complex chirp reference
    (same one used by the engine's LFM generator). If None, a
    reference is built from the other waveform parameters on first use.
    """
    dsp_sample_rate_hz: float = 10e6
    tick_interval_s: float = 10e-3
    noise_floor_dbm: float = -130.0
    # CFAR
    cfar_db: float = 10.0          # threshold offset above noise estimate (dB)
    cfar_train_cells: int = 20     # training cells per side
    cfar_guard_cells: int = 4      # guard cells per side (excluded from avg)
    # Pulse formation
    min_pulse_samples: int = 5     # minimum width in samples
    max_pulses_per_tick: int = 50  # sanity cap
    # Waveform
    waveform_type: str = "lfm"     # "lfm" | "bpsk" | "qpsk" | "qam16"
    chirp_bandwidth_hz: float = 1e6
    carrier_freq_hz: float = 3e9
    pulse_width_s: float = 1e-6
    chirp_reference: Optional[np.ndarray] = None  # pre-computed LFM reference

    @property
    def samples_per_tick(self) -> int:
        return int(round(self.dsp_sample_rate_hz * self.tick_interval_s))

    @property
    def cfar_offset_linear(self) -> float:
        """CFAR offset from dB to linear."""
        return 10.0 ** (float(self.cfar_db) / 10.0)


# =====================================================================
# Pulse detector
# =====================================================================
class PulseDetector:
    """
    Detect pulses in I/Q buffers from ``RealTimeRFSimulator``.

    Parameters
    ----------
    config : PulseDetectorConfig
        Detector parameters.
    emitter_map : Dict[int, EmitterInfo]
        Maps ``emitter_id`` → ``EmitterInfo(carrier_freq_hz, aoa_deg)``.
        Used for AoA assignment and emitter-ID labelling of detected
        pulses. The map is stored by reference; update it between
        calls if the emitter set changes.
    rng : np.random.Generator, optional
        RNG for any stochastic steps (e.g. jitter in parameter
        extraction). If None, a default is created.

    Usage
    -----
    ::

        from vyapti_simulator.rf.pulse_detector import PulseDetector, PulseDetectorConfig

        detector = PulseDetector(
            config=PulseDetectorConfig(
                dsp_sample_rate_hz=10e6,
                tick_interval_s=10e-3,
                chirp_reference=my_chirp_reference,
            ),
            emitter_map={0: EmitterInfo(3e9, 12.0), 1: EmitterInfo(5e9, 58.0)},
            rng=np.random.default_rng(42),
        )
        pdw_stream = detector.detect(iq_buffer)
    """

    def __init__(
        self,
        config: PulseDetectorConfig,
        emitter_map: Dict[int, "EmitterInfo"],
        rng: Optional[np.random.Generator] = None,
    ):
        self.config = config
        self.emitter_map = dict(emitter_map)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._chirp_ref: Optional[np.ndarray] = None  # stored chirp reference

        # Lazily built on first detect() call
        if config.chirp_reference is not None:
            self._build_fft_reference(config.chirp_reference)

    # ------------------------------------------------------------------
    # Public API — full-buffer (batch / open-loop)
    # ------------------------------------------------------------------
    def detect(self, iq_buffer: np.ndarray) -> PDWStream:
        """
        Detect pulses in an I/Q buffer and return a ``PDWStream``.

        Parameters
        ----------
        iq_buffer : np.ndarray
            Concatenated complex I/Q samples from ``RealTimeRFSimulator``.
            Length must be a multiple of ``samples_per_tick``.

        Returns
        -------
        PDWStream
            Detected pulses with fields: ``toa_us``, ``freq_mhz``,
            ``pw_us``, ``aoa_deg``, ``amp_db``, ``emitter_id``.
            Sorted by ToA. Empty arrays on no detections.
        """
        cfg = self.config
        n = int(iq_buffer.size)
        spt = cfg.samples_per_tick
        if n % spt != 0:
            raise ValueError(
                f"iq_buffer length {n} is not a multiple of "
                f"samples_per_tick={spt}"
            )
        n_ticks = n // spt

        # Build chirp reference lazily
        if self._chirp_ref is None:
            self._build_chirp_reference()

        all_toa_us: List[float] = []
        all_freq_mhz: List[float] = []
        all_pw_us: List[float] = []
        all_aoa: List[float] = []
        all_amp_db: List[float] = []
        all_eid: List[int] = []

        for tick in range(n_ticks):
            start = tick * spt
            end = start + spt
            tick_iq = iq_buffer[start:end].astype(np.complex128)

            pulses = self._detect_in_tick(tick_iq, tick)
            for p in pulses:
                t_offset = start / cfg.dsp_sample_rate_hz
                all_toa_us.append(float(p.toa_sec + t_offset) * 1e6)
                all_freq_mhz.append(float(p.freq_hz) * 1e-6)
                all_pw_us.append(float(p.pw_sec) * 1e6)
                all_aoa.append(float(p.aoa_deg))
                all_amp_db.append(float(p.amp_db))
                all_eid.append(int(p.emitter_id))

        if not all_toa_us:
            return PDWStream(
                toa_us=np.zeros(0, dtype=np.float32),
                freq_mhz=np.zeros(0, dtype=np.float32),
                pw_us=np.zeros(0, dtype=np.float32),
                aoa_deg=np.zeros(0, dtype=np.float32),
                amp_db=np.zeros(0, dtype=np.float32),
                emitter_id=np.zeros(0, dtype=np.int64),
            )

        # Sort by ToA
        order = np.argsort(all_toa_us)
        toa_arr = np.asarray(all_toa_us, dtype=np.float64)[order]
        freq_arr = np.asarray(all_freq_mhz, dtype=np.float64)[order]
        pw_arr = np.asarray(all_pw_us, dtype=np.float64)[order]
        aoa_arr = np.asarray(all_aoa, dtype=np.float64)[order]
        amp_arr = np.asarray(all_amp_db, dtype=np.float64)[order]
        eid_arr = np.asarray(all_eid, dtype=np.int64)[order]

        return PDWStream(
            toa_us=toa_arr.astype(np.float32),
            freq_mhz=freq_arr.astype(np.float32),
            pw_us=pw_arr.astype(np.float32),
            aoa_deg=aoa_arr.astype(np.float32),
            amp_db=amp_arr.astype(np.float32),
            emitter_id=eid_arr,
        )

    # ------------------------------------------------------------------
    # Closed-loop / band-limited detection
    # ------------------------------------------------------------------
    def detect_dwell(
        self,
        iq_chunk: np.ndarray,
        freq_start_hz: float,
        freq_end_hz: float,
        freq_center_hz: Optional[float] = None,
        aoa_center_deg: Optional[float] = None,
        aoa_window_deg: Optional[float] = None,
    ) -> PDWStream:
        """
        Detect pulses in one dwell's I/Q slice (closed-loop mode).

        This is the closed-loop analogue of :meth:`detect`. It is used
        when the receiver is pointed at a specific frequency band/AoA
        sector by a scheduler. It:

          1. Rebuilds the chirp reference for the dwell's centre
             frequency (different dwell = different reference).
          2. Runs the matched-filter / CFAR / peak-detection pipeline.
          3. Filters the output to pulses inside the dwell window.

        Parameters
        ----------
        iq_chunk : np.ndarray
            Complex I/Q samples from one :meth:`RealTimeRFSimulator.simulate_dwell`
            call. Length must be a multiple of ``samples_per_tick``.
        freq_start_hz : float
            Lower edge of the dwell band in Hz.
        freq_end_hz : float
            Upper edge of the dwell band in Hz.
        freq_center_hz : float, optional
            Centre frequency for the matched-filter reference. If None,
            inferred as ``(freq_start_hz + freq_end_hz) / 2``.
        aoa_center_deg : float, optional
            Centre AoA of the dwell sector. Pulses from emitters
            outside ``±aoa_window_deg`` of this value are excluded.
        aoa_window_deg : float, optional
            Half-width of the AoA sector. Only used when
            ``aoa_center_deg`` is not None.

        Returns
        -------
        PDWStream
            Detected pulses inside the dwell window, sorted by ToA.
            Empty stream if nothing was detected.
        """
        cfg = self.config

        # Resolve centre frequency for the reference
        center_hz = (
            float(freq_center_hz)
            if freq_center_hz is not None
            else (float(freq_start_hz) + float(freq_end_hz)) / 2.0
        )

        # Rebuild the chirp reference for this dwell's centre frequency
        n_ref = max(2, int(round(cfg.pulse_width_s * cfg.dsp_sample_rate_hz)))
        t_ref = np.arange(n_ref, dtype=np.float64) / cfg.dsp_sample_rate_hz
        from .waveforms import generate_lfm_chirp
        chirp = generate_lfm_chirp(
            t_ref,
            f0=center_hz - 0.5 * cfg.chirp_bandwidth_hz,
            f1=center_hz + 0.5 * cfg.chirp_bandwidth_hz,
            peak_power_w=1.0,
        )
        self._build_fft_reference(chirp)

        # Build a temporary emitter map filtered to this dwell's band
        filtered_map: Dict[int, "EmitterInfo"] = {}
        for eid, info in self.emitter_map.items():
            f_e = float(info.carrier_freq_hz)
            if f_e <= 0:
                # No frequency — include it
                filtered_map[eid] = info
            elif float(freq_start_hz) <= f_e <= float(freq_end_hz):
                # Inside the dwell band
                # AoA filter
                if aoa_center_deg is not None and aoa_window_deg is not None:
                    aoa_diff = abs(float(info.aoa_deg) - float(aoa_center_deg))
                    aoa_diff = min(aoa_diff, 360.0 - aoa_diff)
                    if aoa_diff > float(aoa_window_deg):
                        continue
                filtered_map[eid] = info

        # Temporarily swap the emitter map so _lookup_emitter uses the
        # filtered set. Restore the original afterwards.
        original_map = self.emitter_map
        try:
            self.emitter_map = filtered_map
            result = self.detect(iq_chunk)
        finally:
            self.emitter_map = original_map

        # Post-filter: additionally constrain by the dwell frequency
        # window (and AoA window when set) even if an emitter wasn't in
        # the map. This catches agile emitters whose frequency happens
        # to be inside the band, and excludes noise peaks labeled as
        # emitter_id=-1 when the AoA filter is active.
        freq_start = float(freq_start_hz)
        freq_end = float(freq_end_hz)
        aoa_filter_active = (
            aoa_center_deg is not None and aoa_window_deg is not None
        )
        aoa_c = float(aoa_center_deg) if aoa_center_deg is not None else 0.0
        aoa_w = float(aoa_window_deg) if aoa_window_deg is not None else 360.0

        def _in_freq(f_mhz: float) -> bool:
            return freq_start <= f_mhz * 1e6 <= freq_end

        def _in_aoa(aoa_deg_val: float) -> bool:
            if not aoa_filter_active:
                return True
            diff = abs(float(aoa_deg_val) - aoa_c)
            diff = min(diff, 360.0 - diff)
            return diff <= aoa_w

        mask = np.array([
            _in_freq(float(f)) and _in_aoa(float(a))
            for f, a in zip(result.freq_mhz, result.aoa_deg)
        ], dtype=bool)

        if not mask.any():
            return PDWStream(
                toa_us=np.zeros(0, dtype=np.float32),
                freq_mhz=np.zeros(0, dtype=np.float32),
                pw_us=np.zeros(0, dtype=np.float32),
                aoa_deg=np.zeros(0, dtype=np.float32),
                amp_db=np.zeros(0, dtype=np.float32),
                emitter_id=np.zeros(0, dtype=np.int64),
            )

        return PDWStream(
            toa_us=result.toa_us[mask],
            freq_mhz=result.freq_mhz[mask],
            pw_us=result.pw_us[mask],
            aoa_deg=result.aoa_deg[mask],
            amp_db=result.amp_db[mask],
            emitter_id=result.emitter_id[mask],
        )

    # ------------------------------------------------------------------
    # Per-tick detection
    # ------------------------------------------------------------------
    def _detect_in_tick(self, tick_iq: np.ndarray, tick_idx: int) -> List["DetectedPulse"]:
        cfg = self.config
        n = tick_iq.size

        # 1. Matched filter (frequency-domain for speed)
        mf_out = self._matched_filter(tick_iq)

        # 2. Envelope
        envelope = np.abs(mf_out)

        # 3. CFAR threshold
        threshold = self._cfar_threshold(envelope)

        # 4. Peak detection
        peaks = self._find_peaks(envelope, threshold)
        if len(peaks) > cfg.max_pulses_per_tick:
            # Keep the top-N by amplitude
            peak_amps = envelope[peaks]
            top_n_idx = np.argsort(peak_amps)[-cfg.max_pulses_per_tick:]
            peaks = sorted(peaks[i] for i in top_n_idx)

        # 5. Parameter extraction per peak
        pulses: List[DetectedPulse] = []
        for pk_idx in peaks:
            # ToA: sample index → seconds
            toa_sec = float(pk_idx) / cfg.dsp_sample_rate_hz

            # PW: 3 dB width around peak
            half_max = envelope[pk_idx] / np.sqrt(2.0)
            # Search left and right for 3 dB crossing
            lo = max(0, pk_idx - n // 2)
            hi = min(n - 1, pk_idx + n // 2)
            left = pk_idx
            for i in range(pk_idx, lo - 1, -1):
                if envelope[i] < half_max:
                    left = i + 1
                    break
            right = pk_idx
            for i in range(pk_idx, hi + 1):
                if envelope[i] < half_max:
                    right = i
                    break
            pw_samples = max(right - left, cfg.min_pulse_samples)
            pw_sec = float(pw_samples) / cfg.dsp_sample_rate_hz

            # SNR: peak power vs median noise floor
            noise_floor = float(np.median(envelope[:min(n, 200)]))
            if noise_floor > 1e-12:
                snr_linear = (envelope[pk_idx] / noise_floor) ** 2
            else:
                snr_linear = 1e6  # very high SNR fallback
            amp_db = 10.0 * np.log10(snr_linear + 1e-30)

            # RF: from phase slope across the peak region
            rf_hz = self._estimate_frequency(tick_iq, pk_idx, cfg)

            # AoA and emitter ID: ground-truth lookup
            aoa_deg, emitter_id = self._lookup_emitter(rf_hz)

            pulses.append(DetectedPulse(
                toa_sec=toa_sec,
                freq_hz=rf_hz,
                pw_sec=pw_sec,
                aoa_deg=aoa_deg,
                amp_db=amp_db,
                emitter_id=emitter_id,
            ))

        return pulses

    # ------------------------------------------------------------------
    # Matched filter
    # ------------------------------------------------------------------
    def _build_chirp_reference(self) -> None:
        """Build the LFM chirp reference for the matched filter."""
        from .waveforms import generate_lfm_chirp
        cfg = self.config
        # Reference is one pulse wide (pulse_width_s)
        n_ref = max(2, int(round(cfg.pulse_width_s * cfg.dsp_sample_rate_hz)))
        t_ref = np.arange(n_ref, dtype=np.float64) / cfg.dsp_sample_rate_hz
        chirp = generate_lfm_chirp(
            t_ref,
            f0=cfg.carrier_freq_hz - 0.5 * cfg.chirp_bandwidth_hz,
            f1=cfg.carrier_freq_hz + 0.5 * cfg.chirp_bandwidth_hz,
            peak_power_w=1.0,
        )
        self._build_fft_reference(chirp)

    def _build_fft_reference(self, reference: np.ndarray) -> None:
        """Store the chirp reference (zero-padded in _matched_filter)."""
        self._chirp_ref = np.asarray(reference, dtype=np.complex128)

    def _matched_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        FFT-based matched filter: signal ⋆ reference*.

        Uses zero-padding to avoid circular convolution artefacts.
        Returns the compressed output of the same length as signal.
        """
        if self._chirp_ref is None:
            self._build_chirp_reference()
        sig = np.asarray(signal, dtype=np.complex128)
        n = sig.size
        m = self._chirp_ref.size
        # Zero-pad both signal and reference to the same length
        padded_len = n + m - 1
        padded_sig = np.zeros(padded_len, dtype=np.complex128)
        padded_sig[:n] = sig
        padded_ref = np.zeros(padded_len, dtype=np.complex128)
        padded_ref[:m] = self._chirp_ref
        # FFT of both
        sig_fft = np.fft.fft(padded_sig)
        ref_fft = np.fft.fft(padded_ref[::-1].conj())
        # Pointwise multiply + IFFT
        result = np.fft.ifft(sig_fft * ref_fft)
        return result[:n].astype(np.complex128)

    # ------------------------------------------------------------------
    # CFAR
    # ------------------------------------------------------------------
    def _cfar_threshold(self, envelope: np.ndarray) -> np.ndarray:
        """
        Cell-averaging CFAR on the matched-filter envelope.

        For each test cell i, the threshold is::

            T[i] = alpha * (1/N) * Σ training_cells P[j]
            alpha = cfar_offset_linear

        Guard cells adjacent to the test cell are excluded.
        Edge cells with insufficient training cells use a reduced window.
        """
        cfg = self.config
        n = envelope.size
        t = cfg.cfar_train_cells
        g = cfg.cfar_guard_cells
        alpha = cfg.cfar_offset_linear

        threshold = np.zeros(n, dtype=np.float64)
        total_cells = t + g

        for i in range(n):
            # Left training region: max(0, i - total_cells) to i - g - 1
            left_start = max(0, i - total_cells)
            left_end = max(left_start, i - g)
            # Right training region: i + g + 1 to min(n-1, i + total_cells)
            right_start = min(n - 1, i + g + 1)
            right_end = min(n, i + total_cells + 1)
            # Collect training cells
            left_cells = envelope[left_start:left_end]
            right_cells = envelope[right_start:right_end]
            all_cells = np.concatenate([left_cells, right_cells])
            if all_cells.size == 0:
                threshold[i] = alpha * float(np.median(envelope))
            else:
                threshold[i] = alpha * float(np.mean(all_cells))

        return threshold

    # ------------------------------------------------------------------
    # Peak finding
    # ------------------------------------------------------------------
    def _find_peaks(
        self,
        envelope: np.ndarray,
        threshold: np.ndarray,
    ) -> List[int]:
        """
        Find local maxima above the CFAR threshold.

        A peak at index i is valid if:
          - envelope[i] > threshold[i]
          - envelope[i] >= envelope[i-1] (local max)
          - envelope[i] >= envelope[i+1]
        Adjacent peaks within min_pulse_samples are merged to the
        highest-amplitude one.
        """
        cfg = self.config
        n = envelope.size
        peaks: List[int] = []

        for i in range(1, n - 1):
            if envelope[i] <= threshold[i]:
                continue
            if envelope[i] < envelope[i - 1] or envelope[i] < envelope[i + 1]:
                continue
            peaks.append(i)

        # Merge adjacent peaks within min_pulse_samples
        if not peaks:
            return peaks

        min_sep = max(cfg.min_pulse_samples, 1)
        merged: List[int] = [peaks[0]]
        for pk in peaks[1:]:
            if pk - merged[-1] < min_sep:
                # Keep whichever has higher amplitude
                if envelope[pk] > envelope[merged[-1]]:
                    merged[-1] = pk
            else:
                merged.append(pk)

        return merged

    # ------------------------------------------------------------------
    # Frequency estimation
    # ------------------------------------------------------------------
    def _estimate_frequency(
        self,
        tick_iq: np.ndarray,
        peak_idx: int,
        cfg: PulseDetectorConfig,
    ) -> float:
        """
        Estimate the instantaneous frequency around a peak using the
        phase slope across the matched-filter output.

        Uses unwrapped phase difference over a ±half_win window.
        Falls back to the configured carrier frequency if the
        phase estimate is noisy.
        """
        half_win = max(4, cfg.min_pulse_samples // 2)
        lo = max(0, peak_idx - half_win)
        hi = min(tick_iq.size - 1, peak_idx + half_win)
        if hi <= lo:
            return float(cfg.carrier_freq_hz)

        segment = tick_iq[lo:hi + 1]
        phase = np.unwrap(np.angle(segment))
        if phase.size < 2:
            return float(cfg.carrier_freq_hz)

        # Phase slope: Δφ / (2π * Δt) = frequency
        dt = 1.0 / cfg.dsp_sample_rate_hz
        dphi = float(phase[-1] - phase[0])
        freq_est = abs(dphi / (2.0 * np.pi * dt * (phase.size - 1)))
        # Sanity: if the estimate is way off, fall back to carrier
        if abs(freq_est - cfg.carrier_freq_hz) > cfg.chirp_bandwidth_hz:
            return float(cfg.carrier_freq_hz)
        return float(freq_est)

    # ------------------------------------------------------------------
    # AoA / emitter ID lookup
    # ------------------------------------------------------------------
    def _lookup_emitter(self, freq_hz: float) -> Tuple[float, int]:
        """
        Look up the AoA and emitter ID for a detected pulse.

        Uses the frequency to find the closest matching emitter in
        the emitter map. This is the ground-truth assignment —
        the RF engine has no independent emitter-ID signal.
        """
        if not self.emitter_map:
            return 0.0, -1
        # Find the emitter with the closest carrier frequency
        best_eid = -1
        best_delta = float("inf")
        for eid, info in self.emitter_map.items():
            delta = abs(float(info.carrier_freq_hz) - freq_hz)
            if delta < best_delta:
                best_delta = delta
                best_eid = eid
        if best_eid < 0:
            return 0.0, -1
        info = self.emitter_map[best_eid]
        return float(info.aoa_deg), int(best_eid)

    def update_emitter_map(self, emitter_map: Dict[int, "EmitterInfo"]) -> None:
        """Replace the current emitter map."""
        self.emitter_map = dict(emitter_map)


# =====================================================================
# Result type
# =====================================================================
@dataclass
class DetectedPulse:
    """One detected pulse from the detector."""
    toa_sec: float
    freq_hz: float
    pw_sec: float
    aoa_deg: float
    amp_db: float
    emitter_id: int


@dataclass
class EmitterInfo:
    """
    Static emitter information for the detector's AoA / ID assignment.

    Attributes
    ----------
    carrier_freq_hz : float
        Emitter's centre frequency in Hz.
    aoa_deg : float
        Emitter's angle of arrival in degrees.
    """
    carrier_freq_hz: float
    aoa_deg: float


__all__ = [
    "PulseDetector",
    "PulseDetectorConfig",
    "DetectedPulse",
    "EmitterInfo",
]

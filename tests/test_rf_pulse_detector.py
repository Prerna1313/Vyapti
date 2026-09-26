"""
tests.test_rf_pulse_detector
=============================

Tests for :mod:`vyapti_simulator.rf.pulse_detector`.
"""
from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.rf.pulse_detector import (
    PulseDetector,
    PulseDetectorConfig,
    EmitterInfo,
)
from vyapti_simulator.rf.waveforms import generate_lfm_chirp


def _chirp_in_buffer(
    dsp_rate: float,
    tick_samples: int,
    freq_hz: float,
    bw_hz: float,
    pw_s: float,
    rng: np.random.Generator,
    snr_db: float = 20.0,
) -> np.ndarray:
    """Build a single-tick buffer with one chirp pulse at tick start."""
    n_pulse = max(2, int(round(pw_s * dsp_rate)))
    t = np.arange(n_pulse, dtype=np.float64) / dsp_rate
    chirp = generate_lfm_chirp(
        t,
        f0=freq_hz - 0.5 * bw_hz,
        f1=freq_hz + 0.5 * bw_hz,
        peak_power_w=1.0,
    )
    # Add noise at requested SNR
    p_signal = float(np.mean(np.abs(chirp) ** 2))
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    std = np.sqrt(p_noise / 2.0)
    u1 = rng.random(n_pulse); u1 = np.where(u1 == 0, 1e-12, u1)
    u2 = rng.random(n_pulse)
    mag = np.sqrt(-2.0 * np.log(u1))
    noise = std * mag * (np.cos(2 * np.pi * u2) + 1j * np.sin(2 * np.pi * u2))
    pulse = chirp + noise.astype(np.complex128)
    buffer = np.zeros(tick_samples, dtype=np.complex128)
    buffer[:n_pulse] = pulse
    return buffer


class TestPulseDetectorBasics:
    """Smoke tests for the detector's basic plumbing."""

    def test_detect_returns_pdw_stream(self):
        """detect() returns a PDWStream with the right dtypes."""
        cfg = PulseDetectorConfig(
            dsp_sample_rate_hz=1e6, tick_interval_s=1e-3,
            cfar_db=10.0, min_pulse_samples=3,
        )
        det = PulseDetector(
            config=cfg,
            emitter_map={0: EmitterInfo(3e9, 12.0)},
            rng=np.random.default_rng(0),
        )
        # Pure noise
        iq = (np.random.randn(1000) + 1j * np.random.randn(1000)) * 0.01
        pdw = det.detect(iq)
        assert pdw.toa_us.dtype == np.float32
        assert pdw.freq_mhz.dtype == np.float32
        assert pdw.pw_us.dtype == np.float32
        assert pdw.aoa_deg.dtype == np.float32
        assert pdw.amp_db.dtype == np.float32
        assert pdw.emitter_id.dtype == np.int64

    def test_detect_pure_noise_zero_detections(self):
        """Pure noise at low amplitude yields zero detections at high CFAR."""
        cfg = PulseDetectorConfig(
            dsp_sample_rate_hz=1e6, tick_interval_s=1e-3,
            cfar_db=20.0,  # very high threshold
        )
        det = PulseDetector(
            config=cfg,
            emitter_map={0: EmitterInfo(3e9, 12.0)},
            rng=np.random.default_rng(0),
        )
        iq = (np.random.randn(1000) + 1j * np.random.randn(1000)) * 0.001
        pdw = det.detect(iq)
        assert len(pdw) == 0

    def test_detect_finds_pulse(self):
        """A clean chirp is detected with high SNR."""
        cfg = PulseDetectorConfig(
            dsp_sample_rate_hz=1e6, tick_interval_s=1e-3,
            cfar_db=10.0, min_pulse_samples=3,
            carrier_freq_hz=3e9, chirp_bandwidth_hz=1e6, pulse_width_s=1e-6,
        )
        det = PulseDetector(
            config=cfg,
            emitter_map={0: EmitterInfo(3e9, 12.0)},
            rng=np.random.default_rng(0),
        )
        rng = np.random.default_rng(0)
        iq = _chirp_in_buffer(
            dsp_rate=cfg.dsp_sample_rate_hz,
            tick_samples=cfg.samples_per_tick,
            freq_hz=cfg.carrier_freq_hz,
            bw_hz=cfg.chirp_bandwidth_hz,
            pw_s=cfg.pulse_width_s,
            rng=rng,
            snr_db=25.0,  # very high SNR
        )
        pdw = det.detect(iq)
        # Should detect at least one pulse
        assert len(pdw) >= 1

    def test_detect_multiple_pulses(self):
        """Multiple chirp pulses across ticks are all detected."""
        cfg = PulseDetectorConfig(
            dsp_sample_rate_hz=1e6, tick_interval_s=1e-3,
            cfar_db=10.0, min_pulse_samples=3,
            carrier_freq_hz=3e9, chirp_bandwidth_hz=1e6, pulse_width_s=100e-6,
        )
        det = PulseDetector(
            config=cfg,
            emitter_map={0: EmitterInfo(3e9, 12.0)},
            rng=np.random.default_rng(0),
        )
        rng = np.random.default_rng(0)
        n_ticks = 3
        buffers = [
            _chirp_in_buffer(
                dsp_rate=cfg.dsp_sample_rate_hz,
                tick_samples=cfg.samples_per_tick,
                freq_hz=cfg.carrier_freq_hz,
                bw_hz=cfg.chirp_bandwidth_hz,
                pw_s=cfg.pulse_width_s,
                rng=rng,
                snr_db=25.0,
            )
            for _ in range(n_ticks)
        ]
        iq = np.concatenate(buffers)
        pdw = det.detect(iq)
        # One detection per tick at minimum
        assert len(pdw) >= n_ticks

    def test_aoa_assigned_from_map(self):
        """
        Detected pulses carry the emitter's AoA from the map.
        Current AoA is temporary truth-derived behaviour; it will be removed in Gate 3 (measurement-only receiver).
        """
        cfg = PulseDetectorConfig(
            dsp_sample_rate_hz=1e6, tick_interval_s=1e-3,
            cfar_db=10.0, min_pulse_samples=3,
            carrier_freq_hz=3e9, chirp_bandwidth_hz=1e6, pulse_width_s=1e-6,
        )
        det = PulseDetector(
            config=cfg,
            emitter_map={0: EmitterInfo(3e9, 12.0)},
            rng=np.random.default_rng(0),
        )
        rng = np.random.default_rng(0)
        iq = _chirp_in_buffer(
            dsp_rate=cfg.dsp_sample_rate_hz,
            tick_samples=cfg.samples_per_tick,
            freq_hz=cfg.carrier_freq_hz,
            bw_hz=cfg.chirp_bandwidth_hz,
            pw_s=cfg.pulse_width_s,
            rng=rng, snr_db=25.0,
        )
        pdw = det.detect(iq)
        # All detections should have AoA = 12.0
        # Test deterministic replay
        det2 = PulseDetector(
            config=cfg,
            emitter_map={0: EmitterInfo(3e9, 12.0)},
            rng=np.random.default_rng(0),
        )
        rng = np.random.default_rng(0)
        iq2 = _chirp_in_buffer(
            dsp_rate=cfg.dsp_sample_rate_hz,
            tick_samples=cfg.samples_per_tick,
            freq_hz=cfg.carrier_freq_hz,
            bw_hz=cfg.chirp_bandwidth_hz,
            pw_s=cfg.pulse_width_s,
            rng=rng, snr_db=25.0,
        )
        pdw2 = det2.detect(iq2)
        np.testing.assert_array_equal(pdw.aoa_deg, pdw2.aoa_deg)

    def test_emitter_id_assigned_from_map(self):
        """Detected pulses carry the emitter_id from the map."""
        cfg = PulseDetectorConfig(
            dsp_sample_rate_hz=1e6, tick_interval_s=1e-3,
            cfar_db=10.0, min_pulse_samples=3,
            carrier_freq_hz=3e9, chirp_bandwidth_hz=1e6, pulse_width_s=1e-6,
        )
        det = PulseDetector(
            config=cfg,
            emitter_map={7: EmitterInfo(3e9, 12.0)},
            rng=np.random.default_rng(0),
        )
        rng = np.random.default_rng(0)
        iq = _chirp_in_buffer(
            dsp_rate=cfg.dsp_sample_rate_hz,
            tick_samples=cfg.samples_per_tick,
            freq_hz=cfg.carrier_freq_hz,
            bw_hz=cfg.chirp_bandwidth_hz,
            pw_s=cfg.pulse_width_s,
            rng=rng, snr_db=25.0,
        )
        pdw = det.detect(iq)
        # All detections should have emitter_id = 7
        for eid in pdw.emitter_id:
            assert int(eid) == 7

    def test_buffer_length_must_be_multiple_of_tick(self):
        """IQ buffer length must be a multiple of samples_per_tick."""
        cfg = PulseDetectorConfig(
            dsp_sample_rate_hz=1e6, tick_interval_s=1e-3,
        )
        det = PulseDetector(
            config=cfg, emitter_map={}, rng=np.random.default_rng(0),
        )
        # 1500 is not a multiple of 1000
        iq = np.zeros(1500, dtype=np.complex128)
        with pytest.raises(ValueError, match="not a multiple"):
            det.detect(iq)

    def test_sorted_by_toa(self):
        """Output PDWStream is sorted by ToA."""
        cfg = PulseDetectorConfig(
            dsp_sample_rate_hz=1e6, tick_interval_s=1e-3,
            cfar_db=10.0, min_pulse_samples=3,
            carrier_freq_hz=3e9, chirp_bandwidth_hz=1e6, pulse_width_s=1e-6,
        )
        det = PulseDetector(
            config=cfg,
            emitter_map={0: EmitterInfo(3e9, 12.0)},
            rng=np.random.default_rng(0),
        )
        rng = np.random.default_rng(0)
        # Add chirps to ticks 2, 0, 1 (out of order)
        buffers = [np.zeros(cfg.samples_per_tick, dtype=np.complex128) for _ in range(3)]
        for tick in (2, 0, 1):
            buffers[tick] = _chirp_in_buffer(
                dsp_rate=cfg.dsp_sample_rate_hz,
                tick_samples=cfg.samples_per_tick,
                freq_hz=cfg.carrier_freq_hz,
                bw_hz=cfg.chirp_bandwidth_hz,
                pw_s=cfg.pulse_width_s,
                rng=rng, snr_db=25.0,
            )
        iq = np.concatenate(buffers)
        pdw = det.detect(iq)
        # ToA is monotonically non-decreasing
        assert np.all(np.diff(pdw.toa_us.astype(np.float64)) >= 0.0)

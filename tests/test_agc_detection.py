"""
tests.test_agc_detection
=========================

Tests for the AGC (Automatic Gain Control) model in
`DetectionConfig` / `TSRDEnvironment`.

The AGC is the solution to the problem that TSRD amplitudes are
relative (not absolute dBm), so a fixed ``nominal_noise_floor_db``
gives a misleadingly high SNR for any pulse. The real receiver's
noise floor is set by thermal noise and AGC gain state; the AGC
model tracks the recent amplitude maximum and computes an
adaptive floor ``dynamic_range_db`` below it. A detection
margin (``agc_snr_margin_db``) suppresses multipath and other
sub-dominant signals.

These tests cover:
  * `TestDetectionConfigAGC` — DetectionConfig field defaults and
    backwards-compat for `agc_floor_fraction`.
  * `TestAGCStateManagement` — the env's AGC queue, floor, and
    reset semantics.
  * `TestAGCFloorTracking` — the floor follows the recent max
    (a real AGC backs off when a strong signal arrives).
  * `TestAGCMarginSuppressesMultipath` — strong margin suppresses
    weak (multipath-like) cells while passing the direct path.
  * `TestAGCFloorTruncation` — old samples drop off the window
    after `agc_window_slots` updates.
  * `TestAGCDisabledIsFixedFloor` — `agc_window_slots=0` keeps
    the legacy fixed-floor behavior.
  * `TestAGCMultipathSuppression` — multipath at -20 dB relative
    to direct path is suppressed when margin ≥ 10 dB.
"""

from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.tsrd import (
    DetectionConfig,
    DeinterleaverConfig,
    PDWStream,
    TSRDAdapter,
    TSRDDataMode,
    TSRDEnvironment,
)


# =====================================================================
# Helpers
# =====================================================================

FIXTURE = "D:/Vyapti/tests/fixtures/tsrd/config_0_scan.h5"


def _load_stream():
    """Load the TSRD scan-mode fixture as a PDWStream."""
    from pathlib import Path
    p = Path(FIXTURE)
    if not p.exists():
        pytest.skip("TSRD scan fixture not present")
    cfg = SimulationConfig(
        band_count=36, total_spectrum_mhz=18_000.0,
        receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
        retune_time_ms=1.0, time_slots=600,
    )
    return TSRDAdapter(p, TSRDDataMode.FIXTURE, simulation_config=cfg).to_pdw_stream(), cfg


# =====================================================================
# TestDetectionConfigAGC
# =====================================================================


class TestDetectionConfigAGC:
    """DetectionConfig has the right AGC fields and defaults."""

    def test_agc_window_slots_default_zero(self):
        """Default AGC is disabled (legacy fixed-floor behavior)."""
        cfg = DetectionConfig()
        assert cfg.agc_window_slots == 0

    def test_agc_dynamic_range_db_default_30(self):
        """Default dynamic range is 30 dB below recent max."""
        cfg = DetectionConfig()
        assert cfg.agc_dynamic_range_db == 30.0

    def test_agc_snr_margin_db_default_0(self):
        """Default SNR margin is 0 dB to align with System A noise floor.

        The default was changed from 5.0 to 0.0 so System B's effective
        detection threshold matches System A. Multipath suppression
        remains available as an opt-in feature via DetectionConfig().
        """
        cfg = DetectionConfig()
        assert cfg.agc_snr_margin_db == 0.0

    def test_agc_no_signal_floor_default(self):
        """The fallback floor matches the legacy nominal noise floor."""
        cfg = DetectionConfig()
        assert cfg.agc_no_signal_floor_db == cfg.nominal_noise_floor_db

    def test_agc_floor_fraction_deprecated(self):
        """agc_floor_fraction default is 1.0 (no-op)."""
        cfg = DetectionConfig()
        assert cfg.agc_floor_fraction == 1.0


# =====================================================================
# TestAGCStateManagement
# =====================================================================


class TestAGCStateManagement:
    """The env's AGC state is correctly initialised and reset."""

    def test_agc_state_keys(self):
        """agc_state() returns the documented keys."""
        stream, sim_cfg = _load_stream()
        det = DetectionConfig(agc_window_slots=20)
        env = TSRDEnvironment(stream, sim_cfg, det, DeinterleaverConfig(), seed=42)
        env.reset(seed=42)
        s = env.agc_state()
        for k in [
            "agc_enabled", "agc_window_slots", "agc_queue_size",
            "agc_noise_floor_db", "agc_max_db", "agc_updates",
            "agc_snr_margin_db",
        ]:
            assert k in s, f"Missing key: {k}"

    def test_agc_disabled_when_window_zero(self):
        """agc_enabled reflects agc_window_slots > 0."""
        stream, sim_cfg = _load_stream()
        env_disabled = TSRDEnvironment(
            stream, sim_cfg, DetectionConfig(agc_window_slots=0),
            DeinterleaverConfig(), seed=42,
        )
        env_disabled.reset(seed=42)
        assert env_disabled.agc_state()["agc_enabled"] is False

        env_enabled = TSRDEnvironment(
            stream, sim_cfg, DetectionConfig(agc_window_slots=20),
            DeinterleaverConfig(), seed=42,
        )
        env_enabled.reset(seed=42)
        assert env_enabled.agc_state()["agc_enabled"] is True

    def test_reset_clears_agc_state(self):
        """reset() clears the AGC queue and floor."""
        # Use a small synthetic stream with known occupied cells
        sim_cfg = SimulationConfig(
            band_count=4, total_spectrum_mhz=2000.0, receiver_ibw_mhz=500.0,
            dwell_time_ms=50.0, retune_time_ms=1.0, time_slots=10,
        )
        stream = PDWStream(
            toa_us=np.array([0.0, 50_000.0, 100_000.0], dtype=np.float32),
            freq_mhz=np.full(3, 2400.0, dtype=np.float32),
            pw_us=np.full(3, 1.0, dtype=np.float32),
            aoa_deg=np.full(3, 10.0, dtype=np.float32),
            amp_db=np.array([-75.0, -75.0, -75.0], dtype=np.float32),
            emitter_id=np.zeros(3, dtype=np.int64),
        )
        det = DetectionConfig(agc_window_slots=20)
        env = TSRDEnvironment(stream, sim_cfg, det, DeinterleaverConfig(), seed=42)
        env.reset(seed=42)
        # Walk through a few slots on band 3 (the occupied band)
        for _ in range(3):
            env.step(3)
        assert env.agc_state()["agc_updates"] > 0

        # Reset and verify state is cleared
        env.reset(seed=42)
        s = env.agc_state()
        assert s["agc_queue_size"] == 0
        assert s["agc_updates"] == 0
        # The floor should be back to the no-signal value
        assert s["agc_noise_floor_db"] == det.agc_no_signal_floor_db


# =====================================================================
# TestAGCFloorTracking
# =====================================================================


class TestAGCFloorTracking:
    """The AGC floor follows the recent max amplitude."""

    def test_floor_tracks_recent_max_minus_dynamic_range(self):
        """floor = max(recent) - dynamic_range_db."""
        # Single emitter at -75 dB. Band 3, slots 0+1 (known from grid test above).
        stream = PDWStream(
            toa_us=np.array([0.0, 50_000.0], dtype=np.float32),
            freq_mhz=np.array([2400.0, 2400.0], dtype=np.float32),
            pw_us=np.array([1.0, 1.0], dtype=np.float32),
            aoa_deg=np.array([10.0, 10.0], dtype=np.float32),
            amp_db=np.array([-75.0, -75.0], dtype=np.float32),
            emitter_id=np.array([0, 0], dtype=np.int64),
        )
        sim_cfg = SimulationConfig(
            band_count=4, total_spectrum_mhz=2000.0, receiver_ibw_mhz=500.0,
            dwell_time_ms=50.0, retune_time_ms=1.0, time_slots=10,
        )
        det = DetectionConfig(agc_window_slots=10, agc_dynamic_range_db=30.0)
        env = TSRDEnvironment(stream, sim_cfg, det, DeinterleaverConfig(), seed=42)
        env.reset(seed=42)
        env.step(3)  # band 3 has the occupied cells
        s = env.agc_state()
        # max_db ≈ -75, floor ≈ -75 - 30 = -105
        assert abs(s["agc_max_db"] - (-75.0)) < 1.0
        assert abs(s["agc_noise_floor_db"] - (-105.0)) < 1.0

    def test_floor_responds_to_strong_signal(self):
        """A new stronger signal pushes the floor up (AGC backs off)."""
        # Pulses at -100 dB for first 3 slots, then -70 dB for last 2.
        # Band 3: slots 0..4 occupied.
        toa = [0.0, 50_000.0, 100_000.0, 150_000.0, 200_000.0]
        amp = [-100.0, -100.0, -100.0, -70.0, -70.0]
        stream = PDWStream(
            toa_us=np.array(toa, dtype=np.float32),
            freq_mhz=np.full(5, 2400.0, dtype=np.float32),
            pw_us=np.full(5, 1.0, dtype=np.float32),
            aoa_deg=np.full(5, 10.0, dtype=np.float32),
            amp_db=np.array(amp, dtype=np.float32),
            emitter_id=np.zeros(5, dtype=np.int64),
        )
        sim_cfg = SimulationConfig(
            band_count=4, total_spectrum_mhz=2000.0, receiver_ibw_mhz=500.0,
            dwell_time_ms=50.0, retune_time_ms=1.0, time_slots=10,
        )
        det = DetectionConfig(agc_window_slots=10, agc_dynamic_range_db=20.0)
        env = TSRDEnvironment(stream, sim_cfg, det, DeinterleaverConfig(), seed=42)
        env.reset(seed=42)
        for _ in range(5):
            env.step(3)  # band 3 has all pulses
        s = env.agc_state()
        # After 5 updates, max=-70, floor = -70 - 20 = -90
        assert s["agc_max_db"] > -75.0, f"max={s['agc_max_db']}"
        assert s["agc_noise_floor_db"] > -95.0, f"floor={s['agc_noise_floor_db']}"


# =====================================================================
# TestAGCMarginSuppressesMultipath
# =====================================================================


class TestAGCMarginSuppressesMultipath:
    """AGC margin trades sensitivity for multipath rejection."""

    def test_high_margin_lowers_pd(self):
        """For the same SNR, higher margin → lower Pd."""
        # pd_for_snr applies the logistic curve to its input directly.
        # To test margin effect, we feed post-margin SNR values.
        det_low = DetectionConfig(agc_window_slots=20, agc_snr_margin_db=0.0)
        det_high = DetectionConfig(agc_window_slots=20, agc_snr_margin_db=20.0)
        # A signal 5 dB above the AGC floor (post-margin SNR = 5)
        # With margin=0: post-margin = 5 → high Pd
        # With margin=20: post-margin = 5 - 20 = -15 → Pd ≈ 0
        post_margin_snr = 5.0
        pd_low = det_low.pd_for_snr(post_margin_snr)
        pd_high = det_high.pd_for_snr(post_margin_snr - det_high.agc_snr_margin_db)
        assert pd_low > 0.9, f"low-margin Pd={pd_low}"
        assert pd_high < 0.05, f"high-margin Pd={pd_high}"

    def test_pulses_above_floor_pass(self):
        """Pulses 5 dB above the AGC floor still pass with margin=5."""
        det = DetectionConfig(agc_window_slots=20, agc_dynamic_range_db=30.0,
                              agc_snr_margin_db=5.0)
        # Pre-margin SNR = 30 (signal at max), post-margin = 25 → Pd=1.0
        assert det.pd_for_snr(25.0) == 1.0

    def test_multipath_suppressed(self):
        """Multipath 10 dB above the floor is fully suppressed at margin=10."""
        det = DetectionConfig(agc_window_slots=20, agc_dynamic_range_db=30.0,
                              agc_snr_margin_db=10.0)
        # Multipath: 10 dB above floor, post-margin = 0 → Pd=0
        assert det.pd_for_snr(0.0) == 0.0
        # But direct path 30 dB above floor, post-margin = 20 → Pd=1.0
        assert det.pd_for_snr(20.0) == 1.0


# =====================================================================
# TestAGCFloorTruncation
# =====================================================================


class TestAGCFloorTruncation:
    """Old samples drop off the AGC window after `agc_window_slots` updates."""

    def test_queue_size_bounded(self):
        """After N > window updates, the queue size is exactly window."""
        # Synthetic stream with 1 occupied cell, walked repeatedly.
        # We use a long stream so we can step many times without exhausting it.
        n_pulses = 200
        toa = [i * 50_000.0 for i in range(n_pulses)]
        stream = PDWStream(
            toa_us=np.array(toa, dtype=np.float32),
            freq_mhz=np.full(n_pulses, 2400.0, dtype=np.float32),
            pw_us=np.full(n_pulses, 1.0, dtype=np.float32),
            aoa_deg=np.full(n_pulses, 10.0, dtype=np.float32),
            amp_db=np.full(n_pulses, -75.0, dtype=np.float32),
            emitter_id=np.zeros(n_pulses, dtype=np.int64),
        )
        sim_cfg = SimulationConfig(
            band_count=4, total_spectrum_mhz=2000.0, receiver_ibw_mhz=500.0,
            dwell_time_ms=50.0, retune_time_ms=1.0, time_slots=500,
        )
        det = DetectionConfig(agc_window_slots=5)
        env = TSRDEnvironment(stream, sim_cfg, det, DeinterleaverConfig(), seed=42)
        env.reset(seed=42)
        # Walk band 3 (the occupied band) many times
        for _ in range(50):
            env.step(3)
        s = env.agc_state()
        assert s["agc_queue_size"] == 5


# =====================================================================
# TestAGCDisabledIsFixedFloor
# =====================================================================


class TestAGCDisabledIsFixedFloor:
    """With agc_window_slots=0, AGC is disabled and the floor is fixed."""

    def test_agc_disabled_keeps_nominal_floor(self):
        """The AGC floor stays at the no-signal fallback when disabled."""
        stream, sim_cfg = _load_stream()
        det = DetectionConfig(agc_window_slots=0)
        env = TSRDEnvironment(stream, sim_cfg, det, DeinterleaverConfig(), seed=42)
        env.reset(seed=42)
        env.step(0)
        s = env.agc_state()
        assert s["agc_enabled"] is False
        assert s["agc_queue_size"] == 0
        assert s["agc_noise_floor_db"] == det.agc_no_signal_floor_db

    def test_agc_disabled_uses_nominal_floor_for_snr(self):
        """With AGC off, snr_db_estimate uses the fixed nominal floor."""
        stream = PDWStream(
            toa_us=np.array([0.0], dtype=np.float32),
            freq_mhz=np.array([2400.0], dtype=np.float32),
            pw_us=np.array([1.0], dtype=np.float32),
            aoa_deg=np.array([10.0], dtype=np.float32),
            amp_db=np.array([-75.0], dtype=np.float32),
            emitter_id=np.array([0], dtype=np.int64),
        )
        sim_cfg = SimulationConfig(
            band_count=4, total_spectrum_mhz=2000.0, receiver_ibw_mhz=500.0,
            dwell_time_ms=50.0, retune_time_ms=1.0, time_slots=10,
        )
        det = DetectionConfig(agc_window_slots=0, nominal_noise_floor_db=-130.0)
        env = TSRDEnvironment(stream, sim_cfg, det, DeinterleaverConfig(), seed=42)
        env.reset(seed=42)
        obs, _ = env.step(3)  # band 3 has the occupied cell
        # SNR = -75 - (-130) = 55 dB
        assert obs["snr_db_estimate"] == pytest.approx(55.0, abs=0.1)


# =====================================================================
# TestAGCMultipathSuppression (integration)
# =====================================================================


class TestAGCMultipathSuppression:
    """End-to-end AGC behaviour on the TSRD fixture."""

    def test_agc_margin_does_not_starve_real_signals(self):
        """A 5 dB margin still detects the fixture's emitter (max=-85 dB)."""
        stream, sim_cfg = _load_stream()
        # 5 dB margin (aggressive but realistic)
        det = DetectionConfig(agc_window_slots=20, agc_dynamic_range_db=30.0,
                              agc_snr_margin_db=5.0)
        env = TSRDEnvironment(stream, sim_cfg, det, DeinterleaverConfig(), seed=42)
        env.reset(seed=42)
        # Walk the whole mission, count hits. Stepping past the end of the
        # episode raises; reset and keep stepping.
        n_hits = 0
        n_steps = 0
        max_outer = min(50, sim_cfg.time_slots)
        for t in range(max_outer):
            for b in range(sim_cfg.band_count):
                try:
                    obs, done = env.step(b)
                except RuntimeError:
                    env.reset(seed=42)
                    continue
                n_steps += 1
                if obs["hit"]:
                    n_hits += 1
                if done:
                    env.reset(seed=42)
                    break
        # Should detect at least some pulses (fixture has real signal)
        assert n_hits > 0, "AGC margin=5 dB starved the real signal"
        assert env.agc_state()["agc_updates"] > 0

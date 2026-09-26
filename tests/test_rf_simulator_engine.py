"""
tests.test_rf_simulator_engine
===============================

Tests for the real-time RF simulation engine in
``vyapti_simulator.rf.simulator_engine``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.rf.simulator_engine import (
    SimulationEngineConfig,
    SimEmitter,
    RealTimeRFSimulator,
    simulate_offline,
)


# =====================================================================
# Configuration
# =====================================================================

class TestSimulationEngineConfig:
    def test_defaults(self):
        cfg = SimulationEngineConfig()
        assert cfg.tick_interval_s == 10e-3
        assert cfg.dsp_sample_rate_hz == 10e6
        assert cfg.num_ticks == 1000

    def test_samples_per_tick_derived(self):
        cfg = SimulationEngineConfig(tick_interval_s=10e-3, dsp_sample_rate_hz=10e6)
        assert cfg.samples_per_tick == 100_000

    def test_samples_per_buffer(self):
        cfg = SimulationEngineConfig(
            tick_interval_s=10e-3, dsp_sample_rate_hz=10e6, buffer_ticks=5,
        )
        # 5 ticks * 100000 samples/tick = 500000
        assert cfg.samples_per_buffer == 500_000

    def test_buffer_duration_s(self):
        cfg = SimulationEngineConfig(tick_interval_s=10e-3, buffer_ticks=10)
        np.testing.assert_allclose(cfg.buffer_duration_s, 0.1, rtol=1e-9)

    def test_negative_tick_interval_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            SimulationEngineConfig(tick_interval_s=-1.0)

    def test_negative_sample_rate_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            SimulationEngineConfig(dsp_sample_rate_hz=-1.0)

    def test_zero_ticks_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            SimulationEngineConfig(num_ticks=0)


# =====================================================================
# RealTimeRFSimulator — basic smoke
# =====================================================================

class TestRealTimeRFSimulatorBasic:
    def test_initialization(self):
        cfg = SimulationEngineConfig(
            tick_interval_s=10e-3,
            dsp_sample_rate_hz=1e6,
            num_ticks=10,
            buffer_ticks=2,
        )
        rng = np.random.default_rng(0)
        sim = RealTimeRFSimulator(cfg, rng=rng)
        assert sim.config is cfg
        assert sim._tick == 0
        assert not sim._running

    def test_status(self):
        cfg = SimulationEngineConfig(num_ticks=10)
        sim = RealTimeRFSimulator(cfg)
        status = sim.status()
        assert status["tick"] == 0
        assert status["running"] is False
        assert status["n_emitters"] == 0

    def test_reset(self):
        cfg = SimulationEngineConfig(num_ticks=10)
        sim = RealTimeRFSimulator(cfg)
        sim._tick = 99
        sim.reset()
        assert sim.tick == 0
        assert sim._running is False

    def test_stop(self):
        cfg = SimulationEngineConfig(num_ticks=10)
        sim = RealTimeRFSimulator(cfg)
        sim._running = True
        sim.stop()
        assert sim._running is False


# =====================================================================
# Emitter management
# =====================================================================

def _dummy_waveform_fn(t, cfg, rng):
    """Trivial waveform: one sample at peak power, rest zeros."""
    n = len(t)
    out = np.zeros(n, dtype=np.complex128)
    if n > 0:
        out[0] = np.sqrt(cfg.pulse_power_w * 2)
    return out


class TestEmitterManagement:
    def test_add_emitter_returns_id(self):
        from vyapti_simulator.rf.propagation import KinematicEmitter
        cfg = SimulationEngineConfig(num_ticks=10)
        sim = RealTimeRFSimulator(cfg)
        kin = KinematicEmitter(position_m=np.array([500.0, 0.0, 0.0]))
        eid = sim.add_emitter(kin, _dummy_waveform_fn)
        assert eid == 0

    def test_add_multiple_emitters_incremental_ids(self):
        from vyapti_simulator.rf.propagation import KinematicEmitter
        cfg = SimulationEngineConfig(num_ticks=10)
        sim = RealTimeRFSimulator(cfg)
        kin = KinematicEmitter(position_m=np.array([500.0, 0.0, 0.0]))
        eid0 = sim.add_emitter(kin, _dummy_waveform_fn)
        eid1 = sim.add_emitter(kin, _dummy_waveform_fn)
        assert eid1 == eid0 + 1

    def test_remove_emitter_found(self):
        from vyapti_simulator.rf.propagation import KinematicEmitter
        cfg = SimulationEngineConfig(num_ticks=10)
        sim = RealTimeRFSimulator(cfg)
        kin = KinematicEmitter(position_m=np.array([500.0, 0.0, 0.0]))
        eid = sim.add_emitter(kin, _dummy_waveform_fn)
        assert sim.remove_emitter(eid) is True
        assert not any(e.emitter_id == eid and e.active for e in sim._emitters)

    def test_remove_emitter_not_found(self):
        cfg = SimulationEngineConfig(num_ticks=10)
        sim = RealTimeRFSimulator(cfg)
        assert sim.remove_emitter(9999) is False


# =====================================================================
# Run loop
# =====================================================================

class TestRunLoop:
    def test_run_yields_correct_buffer_count(self):
        cfg = SimulationEngineConfig(
            tick_interval_s=10e-3,
            dsp_sample_rate_hz=1e6,
            num_ticks=6,
            buffer_ticks=2,
        )
        sim = RealTimeRFSimulator(cfg)
        buffers = list(sim.run())
        # 6 ticks / 2 ticks per buffer = 3 buffers
        assert len(buffers) == 3
        for buf in buffers:
            assert buf.dtype == np.complex128
            assert buf.shape == (2 * 10_000,)  # 2 ticks * 10000 samples

    def test_run_buffer_dtype_complex128(self):
        cfg = SimulationEngineConfig(
            tick_interval_s=10e-3,
            dsp_sample_rate_hz=1e6,
            num_ticks=2,
            buffer_ticks=1,
        )
        sim = RealTimeRFSimulator(cfg)
        buf = next(sim.run())
        assert buf.dtype == np.complex128

    def test_run_contains_samples(self):
        cfg = SimulationEngineConfig(
            tick_interval_s=10e-3,
            dsp_sample_rate_hz=1e6,
            num_ticks=5,
            buffer_ticks=5,  # one buffer = whole sim
        )
        sim = RealTimeRFSimulator(cfg)
        buf = next(sim.run())
        assert buf.shape == (5 * 10_000,)

    def test_no_emitters_produces_zeros(self):
        cfg = SimulationEngineConfig(
            tick_interval_s=10e-3,
            dsp_sample_rate_hz=1e6,
            num_ticks=2,
            buffer_ticks=2,
        )
        sim = RealTimeRFSimulator(cfg)
        buf = next(sim.run())
        # No emitters: should be all zeros
        np.testing.assert_allclose(buf, 0.0, atol=1e-30)


# =====================================================================
# simulate_offline
# =====================================================================

class TestSimulateOffline:
    def test_simulate_offline_concatenates_all_ticks(self):
        cfg = SimulationEngineConfig(
            tick_interval_s=10e-3,
            dsp_sample_rate_hz=1e6,
            num_ticks=4,
            buffer_ticks=2,
        )
        rng = np.random.default_rng(0)
        result = simulate_offline([], cfg, rng=rng)
        # 4 ticks * 10000 samples = 40000
        assert result.shape == (40_000,)
        assert result.dtype == np.complex128

    def test_simulate_offline_zero_emitters(self):
        cfg = SimulationEngineConfig(
            tick_interval_s=10e-3,
            dsp_sample_rate_hz=1e6,
            num_ticks=3,
        )
        rng = np.random.default_rng(0)
        result = simulate_offline([], cfg, rng=rng)
        np.testing.assert_allclose(result, 0.0, atol=1e-30)


# =====================================================================
# Status snapshot
# =====================================================================

class TestStatus:
    def test_status_shows_emitters(self):
        from vyapti_simulator.rf.propagation import KinematicEmitter
        cfg = SimulationEngineConfig(num_ticks=10)
        sim = RealTimeRFSimulator(cfg)
        kin = KinematicEmitter(position_m=np.array([500.0, 0.0, 0.0]))
        sim.add_emitter(kin, _dummy_waveform_fn)
        status = sim.status()
        assert status["n_emitters"] == 1
        assert status["total_emitters"] == 1

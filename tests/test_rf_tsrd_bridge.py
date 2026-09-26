"""
tests.test_rf_tsrd_bridge
=========================

Tests for :mod:`vyapti_simulator.rf.tsrd_bridge`.
"""
from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.rf.tsrd_bridge import (
    TSRDSpecToRFBridge,
    SimEmitterSpec,
)
from vyapti_simulator.tsrd.synthetic_pdw_generator import (
    SyntheticEmitterSpec,
)


class TestTSRDSpecToRFBridge:
    """Tests for the spec → RF engine bridge."""

    def test_fixed_emitter_maps_correctly(self):
        """Fixed continuous emitter produces valid KinematicEmitter and waveform."""
        spec = SyntheticEmitterSpec(
            emitter_id=0, aoa_deg=12.0, snr_db=15.0,
            emitter_type="fixed_continuous",
            center_freq_hz=3e9, pri_sec=1e-3,
            pulse_width_sec=1e-6,
        )
        rng = np.random.default_rng(42)
        bridge = TSRDSpecToRFBridge(spec, rng)
        sspec = bridge.build()

        assert isinstance(sspec, SimEmitterSpec)
        assert sspec.emitter_id == 0
        assert sspec.aoa_deg == 12.0
        assert sspec.pri_sec == 1e-3
        assert sspec.pulse_width_s == 1e-6
        assert sspec.carrier_freq_hz == 3e9
        # Power is link-budget calibrated from range, carrier frequency,
        # antenna gains and the requested SNR; it is no longer a fixed 1 W.
        assert np.isfinite(sspec.tx_power_w)
        assert sspec.tx_power_w > 0.0
        assert sspec.received_power_w > 0.0
        # Kinematic state is stationary by default
        np.testing.assert_allclose(sspec.kinematic.velocity_m_s, [0, 0, 0])
        assert sspec.kinematic.position_m is not None
        # No channel (Rayleigh by default but no channel returned when los_component_db=None)
        # Actually los_component_db=None → RayleighFadingChannel is returned
        assert sspec.channel is not None

    def test_friis_gains_are_applied_once_and_tx_power_is_preserved(self):
        spec = SyntheticEmitterSpec(
            emitter_id=99,
            aoa_deg=0.0,
            emitter_type="fixed_continuous",
            center_freq_hz=1e9,
            pri_sec=1e-3,
            pulse_width_sec=1e-6,
            snr_db=10.0,
            tx_power_dbm=30.0,
            tx_gain_dbi=3.0,
            rx_gain_dbi=6.0,
            emitter_position_m=(1000.0, 0.0, 0.0),
        )
        sspec = TSRDSpecToRFBridge(spec, np.random.default_rng(0)).build()

        wavelength = 299792458.0 / 1e9
        friis_gain = (wavelength / (4.0 * np.pi * 1000.0)) ** 2
        expected_rx = (
            1.0 * 10.0 ** (3.0 / 10.0) * 10.0 ** (6.0 / 10.0) * friis_gain
        )
        assert sspec.tx_power_w == pytest.approx(1.0)
        assert sspec.received_power_w == pytest.approx(expected_rx)

    def test_rician_channel_when_los_set(self):
        """Rician channel is created when los_component_db is specified."""
        spec = SyntheticEmitterSpec(
            emitter_id=1, aoa_deg=58.0, snr_db=10.0,
            emitter_type="fixed_continuous",
            center_freq_hz=5e9, pri_sec=2e-3,
            pulse_width_sec=0.5e-6,
            los_component_db=6.0,  # 6 dB K-factor
        )
        bridge = TSRDSpecToRFBridge(spec, np.random.default_rng(0))
        sspec = bridge.build()
        # Rician channel was created
        assert sspec.channel is not None
        assert hasattr(sspec.channel, "k_factor")

    def test_rayleigh_channel_when_los_zero(self):
        """Rayleigh channel is used when los_component_db=0."""
        spec = SyntheticEmitterSpec(
            emitter_id=2, aoa_deg=90.0, snr_db=8.0,
            emitter_type="fixed_continuous",
            center_freq_hz=10e9, pri_sec=500e-6,
            pulse_width_sec=0.3e-6,
            los_component_db=0.0,  # K=0 → Rayleigh
        )
        bridge = TSRDSpecToRFBridge(spec, np.random.default_rng(0))
        sspec = bridge.build()
        # Rayleigh channel was created
        assert sspec.channel is not None

    def test_frequency_agile_emitter(self):
        """Agile emitter uses the first freq in the list."""
        spec = SyntheticEmitterSpec(
            emitter_id=3, aoa_deg=120.0, snr_db=11.0,
            emitter_type="frequency_agile",
            freq_list_hz=[4e9, 6e9, 8e9],
            pri_sec=1e-3, pulse_width_sec=0.8e-6,
        )
        bridge = TSRDSpecToRFBridge(spec, np.random.default_rng(0))
        sspec = bridge.build()
        # Carrier is the first in the list
        assert sspec.carrier_freq_hz == 4e9
        assert sspec.emitter_id == 3
        # Waveform function is present
        assert sspec.waveform_fn is not None

    def test_waveform_type_bpsk(self):
        """bpsk waveform_type creates a PSK waveform function."""
        spec = SyntheticEmitterSpec(
            emitter_id=5, aoa_deg=30.0, snr_db=12.0,
            emitter_type="fixed_continuous",
            center_freq_hz=7e9, pri_sec=1e-3,
            pulse_width_sec=1e-6,
            waveform_type="bpsk",
        )
        bridge = TSRDSpecToRFBridge(spec, np.random.default_rng(0))
        sspec = bridge.build()
        # Waveform function was built (call it to verify)
        from vyapti_simulator.rf.simulator_engine import SimulationEngineConfig
        cfg = SimulationEngineConfig()
        # The function truncates to n_pulse samples regardless of input length
        result = sspec.waveform_fn(
            np.arange(100, dtype=np.float64) / cfg.dsp_sample_rate_hz,
            cfg, np.random.default_rng(0),
        )
        # Should be complex and non-zero length (n_pulse = pulse_width * rate)
        assert result.shape[0] > 0
        assert np.iscomplexobj(result)

    def test_waveform_type_qam16(self):
        """qam16 waveform_type creates a QAM waveform function."""
        spec = SyntheticEmitterSpec(
            emitter_id=6, aoa_deg=0.0, snr_db=10.0,
            emitter_type="fixed_continuous",
            center_freq_hz=2e9, pri_sec=1e-3,
            pulse_width_sec=0.5e-6,
            waveform_type="qam16",
        )
        bridge = TSRDSpecToRFBridge(spec, np.random.default_rng(0))
        sspec = bridge.build()
        from vyapti_simulator.rf.simulator_engine import SimulationEngineConfig
        cfg = SimulationEngineConfig()
        result = sspec.waveform_fn(
            np.arange(100, dtype=np.float64) / cfg.dsp_sample_rate_hz,
            cfg, np.random.default_rng(0),
        )
        # Function truncates to n_pulse samples
        assert result.shape[0] > 0
        assert np.iscomplexobj(result)

    def test_kinematic_moving_emitter(self):
        """Moving emitter has non-zero velocity."""
        spec = SyntheticEmitterSpec(
            emitter_id=7, aoa_deg=45.0, snr_db=15.0,
            emitter_type="fixed_continuous",
            center_freq_hz=6e9, pri_sec=1e-3,
            pulse_width_sec=1e-6,
            emitter_velocity_m_s=(100.0, -50.0, 0.0),
        )
        bridge = TSRDSpecToRFBridge(spec, np.random.default_rng(0))
        sspec = bridge.build()
        np.testing.assert_allclose(
            sspec.kinematic.velocity_m_s, [100.0, -50.0, 0.0]
        )

    def test_resolves_carrier_from_center_freq(self):
        """center_freq_hz is used as carrier when specified."""
        spec = SyntheticEmitterSpec(
            emitter_id=8, aoa_deg=10.0, snr_db=14.0,
            emitter_type="fixed_continuous",
            center_freq_hz=9e9, pri_sec=1e-3,
            pulse_width_sec=1e-6,
        )
        bridge = TSRDSpecToRFBridge(spec, np.random.default_rng(0))
        sspec = bridge.build()
        assert sspec.carrier_freq_hz == 9e9

    def test_unknown_waveform_type_raises(self):
        """Unknown waveform_type raises ValueError."""
        spec = SyntheticEmitterSpec(
            emitter_id=9, aoa_deg=15.0, snr_db=13.0,
            emitter_type="fixed_continuous",
            center_freq_hz=3e9, pri_sec=1e-3,
            pulse_width_sec=1e-6,
            waveform_type="unknown_type",
        )
        bridge = TSRDSpecToRFBridge(spec, np.random.default_rng(0))
        with pytest.raises(ValueError, match="waveform_type"):
            bridge.build()

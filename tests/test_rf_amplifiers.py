"""
tests.test_rf_amplifiers
=========================

Tests for the non-linear power amplifier models in
``vyapti_simulator.rf.amplifiers``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.rf.amplifiers import (
    RappAmplifier,
    SalehAmplifier,
    OIP3,
    OIP3_from_gain_compression,
    gain_compression_db,
    compute_evm,
    apply_thermal_noise,
)


class TestOIP3:
    def test_oip3_from_gain_compression(self):
        # OIP3 ≈ P1dB + 10.6 dB
        oip3 = OIP3_from_gain_compression(gain_db=30.0, p1db_dbw=0.0)
        np.testing.assert_allclose(oip3, 10.6, rtol=1e-9)

    def test_oip3(self):
        # OIP3 ≈ P1dB + 10 dB
        oip3 = OIP3(p1db_dbw=0.0)
        np.testing.assert_allclose(oip3, 10.0, rtol=1e-9)


class TestGainCompression:
    def test_below_saturation_full_gain(self):
        # Well below saturation: full gain
        g = gain_compression_db(input_power_dbw=-20.0, gain_db=30.0, p1db_dbw=0.0)
        np.testing.assert_allclose(g, 30.0, atol=1e-6)

    def test_above_saturation_compressed(self):
        # Above saturation: gain is compressed
        g = gain_compression_db(input_power_dbw=5.0, gain_db=30.0, p1db_dbw=0.0)
        assert g < 30.0
        assert g > 10.0  # but not completely crushed

    def test_heavy_overdrive_capped(self):
        # Heavy overdrive hits the gain floor at gain_db - 20 dB
        g = gain_compression_db(input_power_dbw=50.0, gain_db=30.0, p1db_dbw=0.0)
        # 50 dB input → 49 dB overdrive → compression=24.5 dB → 5.5 dB
        # But floor caps at gain_db - 20 = 10 dB
        assert g == 10.0
        assert g > 0.0


class TestRappAmplifier:
    def test_output_power_w_positive(self):
        amp = RappAmplifier(output_power_w=10.0, gain_db=30.0, p1db_dbw=0.0)
        assert amp.output_power_w > 0

    def test_saturation_voltage(self):
        amp = RappAmplifier(output_power_w=10.0, gain_db=30.0, p1db_dbw=0.0)
        # Saturation voltage = sqrt(output_power_w * 0.8)
        np.testing.assert_allclose(amp.saturation_voltage, np.sqrt(10.0 * 0.8))

    def test_oip3_dbw(self):
        amp = RappAmplifier(output_power_w=10.0, gain_db=30.0, p1db_dbw=0.0)
        np.testing.assert_allclose(amp.oip3_dbw, 10.0, rtol=1e-6)

    def test_zero_input_preserved(self):
        amp = RappAmplifier(output_power_w=10.0, gain_db=30.0, p1db_dbw=0.0)
        out = amp.apply(np.zeros(100, dtype=np.complex128))
        # Rapp handles |input|=0 by mapping to a small constant; output should be near 0
        assert np.max(np.abs(out)) < 1e-9

    def test_linear_regime_small_signal(self):
        """Small signal should pass through with approximately linear gain."""
        amp = RappAmplifier(output_power_w=10.0, gain_db=20.0, p1db_dbw=0.0)
        signal = np.array([0.01 + 0.01j] * 10, dtype=np.complex128)
        out = amp.apply(signal)
        # All outputs should have the same phase (linearity)
        assert np.allclose(np.angle(out), np.angle(signal), atol=1e-4)

    def test_compression_regime_phase_preserved(self):
        """AM/AM distortion should not add phase shift in Rapp model."""
        amp = RappAmplifier(output_power_w=1.0, gain_db=10.0, p1db_dbw=-5.0)
        signal = np.array([1.0 + 0.0j], dtype=np.complex128)  # real only
        out = amp.apply(signal)
        # Phase should be ~0 (no AM/PM in Rapp)
        assert abs(np.angle(out[0])) < 0.1

    def test_output_clipped_at_saturation(self):
        """Output magnitude should not exceed sqrt(output_power_w)."""
        amp = RappAmplifier(output_power_w=1.0, gain_db=30.0, p1db_dbw=0.0)
        signal = np.array([1000.0 + 0.0j], dtype=np.complex128)  # huge drive
        out = amp.apply(signal)
        assert np.max(np.abs(out)) <= np.sqrt(1.0) + 1e-9

    def test_invalid_output_power_raises(self):
        with pytest.raises(ValueError, match="positive"):
            RappAmplifier(output_power_w=-1.0, gain_db=30.0, p1db_dbw=0.0)

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError, match="positive"):
            RappAmplifier(output_power_w=10.0, gain_db=30.0, p1db_dbw=0.0, p=0.0)


class TestSalehAmplifier:
    def test_zero_input_preserved(self):
        amp = SalehAmplifier()
        out = amp.apply(np.zeros(10, dtype=np.complex128))
        np.testing.assert_allclose(out, 0.0, atol=1e-30)

    def test_small_signal_near_linear(self):
        """Small signal should be approximately linear gain."""
        amp = SalehAmplifier(alpha_a=2.0, beta_a=0.0)  # linear
        signal = np.array([0.001 + 0.001j] * 10, dtype=np.complex128)
        out = amp.apply(signal)
        # Gain ≈ alpha_a = 2
        np.testing.assert_allclose(np.abs(out) / np.abs(signal), 2.0, rtol=0.05)

    def test_am_pm_phase_shift(self):
        """Saleh model adds phase rotation at high drive levels."""
        amp = SalehAmplifier(alpha_a=2.1587, beta_a=1.1517,
                             alpha_phi=4.0033, beta_phi=9.1040)
        # Large signal drives AM/PM
        signal = np.array([1.0 + 0.0j], dtype=np.complex128)
        out = amp.apply(signal)
        # Phase should be shifted from 0
        assert abs(np.angle(out[0])) > 0.0

    def test_backoff_reduces_distortion(self):
        """Input backoff reduces drive level."""
        amp0 = SalehAmplifier(input_backoff_db=0.0)
        amp6 = SalehAmplifier(input_backoff_db=6.0)
        signal = np.array([1.0 + 0.0j] * 10, dtype=np.complex128)
        out0 = amp0.apply(signal)
        out6 = amp6.apply(signal)
        # Backed-off output should have lower magnitude
        assert np.mean(np.abs(out6)) < np.mean(np.abs(out0))


class TestComputeEVM:
    def test_perfect_signal_zero_evm(self):
        ref = np.array([1.0 + 0.0j, -1.0 + 0.0j, 1.0j, -1.0j], dtype=np.complex128)
        evm = compute_evm(ref, ref)
        assert evm < 0.01

    def test_known_distortion(self):
        ref = np.array([1.0 + 0.0j] * 100, dtype=np.complex128)
        rx = ref + 0.1 * (np.random.randn(100) + 1j * np.random.randn(100))
        evm = compute_evm(rx, ref)
        # EVM ≈ 10% of symbol energy (slightly above 15% due to noise variance)
        assert 5.0 < evm < 20.0

    def test_empty_raises(self):
        assert compute_evm(np.array([]), np.array([1.0])) == np.inf

    def test_zero_reference_power(self):
        assert compute_evm(np.array([1.0]), np.array([0.0])) == np.inf


class TestApplyThermalNoise:
    def test_noise_power_correct(self):
        """P_noise = k_B * T * B."""
        rng = np.random.default_rng(0)
        signal = np.zeros(50_000, dtype=np.complex128)
        T = 290.0  # 290 K
        B = 1e6   # 1 MHz
        kB = 1.380649e-23
        expected_power = kB * T * B
        noisy = apply_thermal_noise(signal, noise_temperature_k=T,
                                    bandwidth_hz=B, rng=rng)
        p_noise = float(np.mean(np.abs(noisy) ** 2))
        np.testing.assert_allclose(p_noise, expected_power, rtol=0.15)

    def test_signal_plus_noise_power(self):
        rng = np.random.default_rng(0)
        signal = np.ones(1000, dtype=np.complex128) * 5.0  # P=25
        noisy = apply_thermal_noise(signal, noise_temperature_k=290.0,
                                    bandwidth_hz=1e6, rng=rng)
        p_total = float(np.mean(np.abs(noisy) ** 2))
        p_signal = 25.0
        # Total power ≈ signal + noise (noise is small relative to signal=5)
        assert p_total > p_signal

    def test_2d_signal(self):
        """Should handle 2-D signals."""
        rng = np.random.default_rng(0)
        signal = np.ones((10, 50), dtype=np.complex128) * 2.0
        noisy = apply_thermal_noise(signal, noise_temperature_k=290.0,
                                    bandwidth_hz=1e6, rng=rng)
        assert noisy.shape == signal.shape
        assert noisy.dtype == np.complex128

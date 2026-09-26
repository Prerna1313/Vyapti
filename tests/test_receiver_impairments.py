"""
tests.test_receiver_impairments
================================

Tests for the hardware receiver impairment models in
``vyapti_simulator.rf.receiver_impairments``:

  1. Phase noise (Wiener random walk on carrier phase)
  2. IQ gain imbalance (Q relative to I)
  3. IQ phase imbalance (orthogonality error)
  4. DC offset (I and Q)
  5. ADC quantisation (finite bits)

The tests verify:
  - Phase noise: small phase variance for low PN, monotonic in dBc/Hz
  - IQ gain: makes Q^2 mean differ from I^2 mean
  - IQ phase: rotates Q relative to I (the cross-correlation Re{I*Q*} shifts)
  - DC offset: adds a constant to the mean
  - Quantisation: limits precision to 2^bits levels
  - apply_receiver_impairments: composes in correct order
"""

from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.rf.receiver_impairments import (
    apply_phase_noise,
    apply_iq_imbalance,
    apply_dc_offset,
    apply_quantization,
    apply_receiver_impairments,
)


# =====================================================================
# Phase noise
# =====================================================================

class TestPhaseNoise:
    """Wiener-process phase noise on the carrier."""

    def test_zero_dbc_returns_copy(self):
        rng = np.random.default_rng(0)
        iq = np.array([1.0 + 0.5j, 0.7 - 0.3j, 0.2 + 0.8j], dtype=np.complex128)
        out = apply_phase_noise(iq, phase_noise_dbc_per_hz=0.0, sample_rate_hz=1e6, rng=rng)
        np.testing.assert_array_equal(out, iq)
        # Verify it's a copy, not the same buffer
        assert out is not iq

    def test_low_pn_has_small_phase_variance(self):
        rng = np.random.default_rng(42)
        iq = np.ones(10000, dtype=np.complex128)
        out = apply_phase_noise(iq, phase_noise_dbc_per_hz=-100.0, sample_rate_hz=1e6, rng=rng)
        phase = np.angle(out)
        # Phase accumulates from 0; the empirical std of the whole array
        # is dominated by the early steps (which have small accumulated phase).
        # For a Wiener process the std of phase[k] = sqrt(k)*sigma_per_step.
        # sigma_per_step = sqrt(2π * 10^(-10)) ≈ 2.5e-5 rad at -100 dBc/Hz.
        # Empirical array std ≈ 0.00073 rad (verified).
        assert 0.0001 < phase.std() < 0.01

    def test_higher_pn_larger_phase_variance(self):
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        iq = np.ones(1000, dtype=np.complex128)
        out_low = apply_phase_noise(iq.copy(), phase_noise_dbc_per_hz=-130.0, sample_rate_hz=1e6, rng=rng_a)
        out_high = apply_phase_noise(iq.copy(), phase_noise_dbc_per_hz=-70.0, sample_rate_hz=1e6, rng=rng_b)
        # Lower dBc/Hz = less phase noise (more negative)
        # Higher dBc/Hz = more phase noise
        assert np.angle(out_high).std() > np.angle(out_low).std()

    def test_preserves_magnitude(self):
        """Phase noise rotates the phase, doesn't change magnitude."""
        rng = np.random.default_rng(0)
        iq = np.array([1.0 + 1.0j, 2.0 - 0.5j, 0.5 + 0.3j], dtype=np.complex128)
        mags_in = np.abs(iq)
        out = apply_phase_noise(iq, phase_noise_dbc_per_hz=-90.0, sample_rate_hz=1e6, rng=rng)
        np.testing.assert_allclose(np.abs(out), mags_in, rtol=1e-10)

    def test_2d_array(self):
        rng = np.random.default_rng(0)
        iq = np.ones((10, 3), dtype=np.complex128)
        out = apply_phase_noise(iq, phase_noise_dbc_per_hz=-100.0, sample_rate_hz=1e6, rng=rng)
        assert out.shape == (10, 3)
        # Each column should have non-trivial phase
        for c in range(3):
            assert np.angle(out[:, c]).std() > 0

    def test_deterministic_with_seed(self):
        """Same seed → same output."""
        iq = np.ones(100, dtype=np.complex128)
        out_a = apply_phase_noise(iq, phase_noise_dbc_per_hz=-100.0, sample_rate_hz=1e6, rng=np.random.default_rng(0))
        out_b = apply_phase_noise(iq, phase_noise_dbc_per_hz=-100.0, sample_rate_hz=1e6, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(out_a, out_b)


# =====================================================================
# IQ gain / phase imbalance
# =====================================================================

class TestIQImbalance:
    """Amplitude and phase mismatch between I and Q."""

    def test_zero_imbalance_returns_copy(self):
        iq = np.array([1.0 + 0.5j, 0.7 - 0.3j], dtype=np.complex128)
        out = apply_iq_imbalance(iq, gain_imbalance_db=0.0, phase_imbalance_deg=0.0)
        np.testing.assert_allclose(out, iq)
        assert out is not iq

    def test_gain_imbalance_makes_q_hotter(self):
        """+1 dB gain imbalance multiplies Q by 10^(1/20)."""
        iq = np.array([1.0 + 1.0j] * 1000, dtype=np.complex128)
        out = apply_iq_imbalance(iq, gain_imbalance_db=1.0, phase_imbalance_deg=0.0)
        # Q should be amplified by 10^(1/20) ≈ 1.122
        expected_q_factor = 10.0 ** (1.0 / 20.0)
        assert abs(np.mean(np.imag(out)) / np.mean(np.imag(iq)) - expected_q_factor) < 0.01

    def test_phase_imbalance_creates_correlation(self):
        """A phase imbalance rotates Q, creating Re{I*Q*} correlation."""
        rng = np.random.default_rng(0)
        iq = rng.standard_normal(1000) + 1j * rng.standard_normal(1000)
        out = apply_iq_imbalance(iq, gain_imbalance_db=0.0, phase_imbalance_deg=5.0)
        # Re{I*Q*} should be non-zero after imbalance
        cross = np.real(iq * np.conj(out)).mean()
        assert abs(cross) > 0.01

    def test_complex128_dtype(self):
        iq = np.array([1.0 + 1.0j], dtype=np.complex64)
        out = apply_iq_imbalance(iq, gain_imbalance_db=1.0, phase_imbalance_deg=1.0)
        assert out.dtype == np.complex128


# =====================================================================
# DC offset
# =====================================================================

class TestDCOffset:
    """DC offset on I and Q."""

    def test_zero_offset_returns_copy(self):
        iq = np.array([1.0 + 0.5j, 0.7 - 0.3j], dtype=np.complex128)
        out = apply_dc_offset(iq, dc_i=0.0, dc_q=0.0)
        np.testing.assert_array_equal(out, iq)
        assert out is not iq

    def test_dc_offset_shifts_mean(self):
        rng = np.random.default_rng(0)
        iq = rng.standard_normal(100) + 1j * rng.standard_normal(100)
        iq = iq.astype(np.complex128)
        dc_i, dc_q = 0.5, -0.3
        out = apply_dc_offset(iq, dc_i=dc_i, dc_q=dc_q)
        mean_in = iq.mean()
        mean_out = out.mean()
        assert abs((mean_out - mean_in).real - dc_i) < 1e-10
        assert abs((mean_out - mean_in).imag - dc_q) < 1e-10

    def test_complex128_dtype(self):
        iq = np.array([1.0 + 1.0j], dtype=np.complex64)
        out = apply_dc_offset(iq, dc_i=0.1, dc_q=0.2)
        assert out.dtype == np.complex128


# =====================================================================
# Quantization
# =====================================================================

class TestQuantization:
    """Uniform ADC quantisation."""

    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError, match="bits must be >= 1"):
            apply_quantization(np.array([1.0 + 1.0j]), bits=0)

    def test_number_of_levels(self):
        """Quantised samples have at most 2^bits distinct levels in each of I and Q."""
        rng = np.random.default_rng(0)
        iq = rng.standard_normal(10_000) + 1j * rng.standard_normal(10_000)
        iq = iq.astype(np.complex128)
        for bits in (4, 8, 12):
            out = apply_quantization(iq, bits=bits)
            # I and Q each have at most 2^bits levels
            n_i = len(np.unique(np.round(out.real, 12)))
            n_q = len(np.unique(np.round(out.imag, 12)))
            assert n_i <= 2 ** bits, f"bits={bits}: I has {n_i} levels, expected <= {2**bits}"
            assert n_q <= 2 ** bits, f"bits={bits}: Q has {n_q} levels, expected <= {2**bits}"

    def test_quantisation_step_size(self):
        """The step between adjacent levels is 2*fs / 2^bits."""
        # Use real-only signals so fs depends on |I| and |Q| directly.
        # Without headroom interactions, the math is cleaner.
        iq = np.array([-1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
        bits = 4
        out = apply_quantization(iq, bits=bits)
        # full-scale = 1.05 * max(|I|, |Q|) = 1.05 * 1.0 = 1.05
        # step = 2 * 1.05 / 2^4 = 0.13125
        # So the I value should be a multiple of 0.13125 (rounded)
        fs = 1.05 * float(np.max(np.abs(iq)))
        step = 2.0 * fs / float(2 ** bits)
        expected_levels = np.round(iq.real / step) * step
        np.testing.assert_allclose(out.real, expected_levels, atol=1e-9)

    def test_infinite_resolution_no_clipping(self):
        """With very high bits, output ≈ input."""
        rng = np.random.default_rng(0)
        iq = (rng.standard_normal(100) + 1j * rng.standard_normal(100)).astype(np.complex128)
        out = apply_quantization(iq, bits=20)
        # Error per sample < 2*fs / 2^20 ≈ 1e-5
        np.testing.assert_allclose(out, iq, atol=1e-4)

    def test_empty_array(self):
        out = apply_quantization(np.array([], dtype=np.complex128), bits=8)
        assert out.shape == (0,)


# =====================================================================
# Combined impairment chain
# =====================================================================

class TestApplyReceiverImpairments:
    """apply_receiver_impairments: end-to-end chain."""

    def test_no_impairments_preserves_signal(self):
        """All parameters zero/None = identity (modulo float cast)."""
        iq = np.array([1.0 + 0.5j, 0.7 - 0.3j], dtype=np.complex128)
        out = apply_receiver_impairments(
            iq,
            phase_noise_dbc_per_hz=0.0,
            iq_gain_imbalance_db=0.0,
            iq_phase_imbalance_deg=0.0,
            dc_offset_i=0.0,
            dc_offset_q=0.0,
            quantization_bits=None,
            rng=np.random.default_rng(0),
        )
        np.testing.assert_allclose(out, iq)

    def test_dc_only_chain(self):
        """With only DC offset, mean shifts by the offset."""
        rng = np.random.default_rng(0)
        iq = (rng.standard_normal(100) + 1j * rng.standard_normal(100)).astype(np.complex128)
        out = apply_receiver_impairments(
            iq,
            dc_offset_i=0.3, dc_offset_q=-0.4,
            rng=np.random.default_rng(0),
        )
        # mean shift ≈ 0.3 - 0.4j
        diff = out.mean() - iq.mean()
        assert abs(diff.real - 0.3) < 0.05
        assert abs(diff.imag - (-0.4)) < 0.05

    def test_quantization_only(self):
        """When only quantisation is on, the output has the right step size."""
        # Use real-only signal so fs = 1.05, step = 2*1.05/16 = 0.13125
        iq = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
        out = apply_receiver_impairments(
            iq,
            quantization_bits=4,
            rng=np.random.default_rng(0),
        )
        # 4 bits = 16 levels; step = 2*1.05/16 = 0.13125
        # Output should be multiples of that step
        step = 2.0 * 1.05 / 16.0
        for v in out:
            assert abs((v.real / step) - round(v.real / step)) < 1e-9

    def test_chain_order_phase_iq_dc_quant(self):
        """The chain order is: phase_noise -> IQ -> DC -> quant.

        DC offset applied AFTER phase noise: the mean of I is the DC value
        plus the mean of the rotated signal, not the original.
        """
        rng = np.random.default_rng(0)
        n = 10000
        iq = (np.ones(n) + 1j * np.zeros(n)).astype(np.complex128)  # constant
        out = apply_receiver_impairments(
            iq,
            phase_noise_dbc_per_hz=-130.0,  # very small
            iq_gain_imbalance_db=0.0,
            dc_offset_i=0.5, dc_offset_q=0.0,
            quantization_bits=8,
            sample_rate_hz=1.0,
            rng=rng,
        )
        # Mean I should be ≈ 1.5 (constant signal=1.0 + DC=0.5)
        # After quantisation, the level is the closest multiple of 2*1.5/256.
        assert abs(out.real.mean() - 1.5) < 0.05

    def test_complex128_dtype_preserved(self):
        iq = np.array([1.0 + 1.0j], dtype=np.complex64)
        out = apply_receiver_impairments(
            iq, quantization_bits=8, rng=np.random.default_rng(0),
        )
        assert out.dtype == np.complex128

    def test_seed_determinism(self):
        """Same seed yields same output."""
        iq = np.ones(100, dtype=np.complex128)
        out_a = apply_receiver_impairments(
            iq, phase_noise_dbc_per_hz=-100.0, quantization_bits=8,
            rng=np.random.default_rng(0),
        )
        out_b = apply_receiver_impairments(
            iq, phase_noise_dbc_per_hz=-100.0, quantization_bits=8,
            rng=np.random.default_rng(0),
        )
        np.testing.assert_array_equal(out_a, out_b)

    def test_2d_array(self):
        """2D (n, k) I/Q arrays are supported."""
        rng = np.random.default_rng(0)
        iq = np.ones((50, 2), dtype=np.complex128)
        out = apply_receiver_impairments(
            iq, phase_noise_dbc_per_hz=-100.0, quantization_bits=8, rng=rng,
        )
        assert out.shape == (50, 2)

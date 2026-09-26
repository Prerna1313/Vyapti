"""
tests.test_rf_waveforms
========================

Tests for the RF waveform synthesizers in
``vyapti_simulator.rf.waveforms``:

  - PSK (BPSK, QPSK, 8-PSK) — phase-shift keying modulators
  - QAM (16-QAM, 64-QAM) — rectangular constellation modulators
  - LFM / Chirp — linear frequency modulation synthesis
  - AWGN — additive white Gaussian noise (Box-Muller)
  - Up/down conversion (complex baseband <-> RF passband)
  - Matched filter (pulse compression)
"""

from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.rf.waveforms import (
    generate_psk_symbols,
    bits_to_psk_symbols,
    generate_qam_symbols,
    generate_lfm_chirp,
    add_awgn,
    estimate_snr,
    complex_baseband_to_rf,
    rf_to_complex_baseband,
    matched_filter,
    coherent_integrate,
)


# =====================================================================
# PSK
# =====================================================================

class TestPSK:
    """Phase-shift keying constellation tests."""

    def test_bpsk_symbols(self):
        syms = generate_psk_symbols(np.array([0, 1]), constellation='bpsk')
        np.testing.assert_array_almost_equal(syms, [1.0 + 0.0j, -1.0 + 0.0j])

    def test_qpsk_symbols(self):
        syms = generate_psk_symbols(np.array([0, 1, 2, 3]), constellation='qpsk')
        # 4 symbols at 45°, 135°, 225°, 315° (all unit magnitude)
        assert np.allclose(np.abs(syms), 1.0)
        expected_angles = [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4]
        actual_angles = np.angle(syms)
        np.testing.assert_allclose(
            np.sort(actual_angles), np.sort(expected_angles), atol=1e-9,
        )

    def test_8psk_eight_points(self):
        syms = generate_psk_symbols(np.arange(8), constellation='8psk')
        assert len(syms) == 8
        assert np.allclose(np.abs(syms), 1.0, atol=1e-9)
        # All distinct
        assert len(set(np.round(syms, 9))) == 8

    def test_invalid_constellation_raises(self):
        with pytest.raises(ValueError, match="bpsk/qpsk/8psk"):
            generate_psk_symbols(np.array([0]), constellation='fsk')

    def test_bits_to_bpsk(self):
        bits = np.array([0, 1, 0, 1])
        syms, _ = bits_to_psk_symbols(bits, constellation='bpsk')
        # 4 BPSK symbols, alternating
        np.testing.assert_array_almost_equal(
            syms, [1.0 + 0.0j, -1.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j]
        )

    def test_bits_to_qpsk(self):
        bits = np.array([0, 0, 0, 1, 1, 1, 1, 0])  # 4 QPSK symbols
        syms, _ = bits_to_psk_symbols(bits, constellation='qpsk')
        assert len(syms) == 4
        assert np.allclose(np.abs(syms), 1.0)

    def test_unit_power_qpsk(self):
        syms = generate_psk_symbols(np.arange(4), constellation='qpsk')
        # All PSK on unit circle
        assert np.allclose(np.abs(syms), 1.0)


# =====================================================================
# QAM
# =====================================================================

class TestQAM:
    """Quadrature amplitude modulation tests."""

    def test_qam16_16_symbols(self):
        syms = generate_qam_symbols(np.arange(16), constellation='qam16')
        assert len(syms) == 16
        assert np.all(np.isfinite(syms))

    def test_qam16_unit_power(self):
        syms = generate_qam_symbols(np.arange(16), constellation='qam16')
        power = np.mean(np.abs(syms) ** 2)
        np.testing.assert_allclose(power, 1.0, atol=1e-9)

    def test_qam64_64_symbols(self):
        syms = generate_qam_symbols(np.arange(64), constellation='qam64')
        assert len(syms) == 64

    def test_qam64_unit_power(self):
        syms = generate_qam_symbols(np.arange(64), constellation='qam64')
        power = np.mean(np.abs(syms) ** 2)
        np.testing.assert_allclose(power, 1.0, atol=1e-9)

    def test_qam16_invalid_index(self):
        with pytest.raises(ValueError, match="out of range"):
            generate_qam_symbols(np.array([16]), constellation='qam16')

    def test_qam_grid_structure(self):
        """QAM16 should lie on a 4x4 grid centred at origin."""
        syms = generate_qam_symbols(np.arange(16), constellation='qam16')
        # Imaginary and real parts should be on a discrete set
        unique_real = np.unique(np.round(syms.real, 6))
        unique_imag = np.unique(np.round(syms.imag, 6))
        assert len(unique_real) == 4
        assert len(unique_imag) == 4
        # Symmetric: sorted_real = -sorted_real[::-1]
        np.testing.assert_allclose(
            np.sort(unique_real), -np.sort(unique_real)[::-1], atol=1e-6
        )


# =====================================================================
# LFM Chirp
# =====================================================================

class TestLFMChirp:
    """Linear frequency modulation waveform tests."""

    def test_shape(self):
        t = np.linspace(0, 10e-6, 100)
        chirp = generate_lfm_chirp(t, f0=1e9, f1=1.1e9)
        assert chirp.shape == t.shape
        assert chirp.dtype == np.complex128

    def test_magnitude_constant(self):
        """Chirp magnitude should be sqrt(2 * peak_power_w)."""
        t = np.linspace(0, 10e-6, 100)
        peak = 5.0
        chirp = generate_lfm_chirp(t, f0=1e9, f1=1.1e9, peak_power_w=peak)
        expected = np.sqrt(2.0 * peak)
        np.testing.assert_allclose(np.abs(chirp), expected, rtol=1e-10)

    def test_non_uniform_grid_raises(self):
        t = np.array([0.0, 1e-6, 3e-6, 4e-6])  # non-uniform
        with pytest.raises(ValueError, match="uniformly spaced"):
            generate_lfm_chirp(t, f0=1e9, f1=1.1e9)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            generate_lfm_chirp(np.array([0.0]), f0=1e9, f1=1.1e9)

    def test_2d_raises(self):
        t = np.zeros((2, 2))
        with pytest.raises(ValueError, match="1-D"):
            generate_lfm_chirp(t, f0=1e9, f1=1.1e9)

    def test_with_awgn(self):
        t = np.linspace(0, 10e-6, 100)
        rng = np.random.default_rng(0)
        chirp_noisy = generate_lfm_chirp(
            t, f0=1e9, f1=1.1e9,
            rng=rng, add_noise=True, snr_db=20.0,
        )
        # Noisy chirp should have more variance than clean
        clean = generate_lfm_chirp(t, f0=1e9, f1=1.1e9)
        assert np.var(np.abs(chirp_noisy)) > np.var(np.abs(clean))

    def test_up_vs_down_chirp(self):
        """f1 > f0 → up-chirp; f1 < f0 → down-chirp."""
        t = np.linspace(0, 1e-6, 1000)
        up = generate_lfm_chirp(t, f0=0.0, f1=1e6)
        down = generate_lfm_chirp(t, f0=1e6, f1=0.0)
        # Up-chirp instantaneous phase: 2π f0 t + π K t²
        # For f0=0, f1=1e6, K=1e12 — phase grows quadratically
        # For down-chirp, f0=1e6, K=-1e12 — also quadratic, different shape
        # Just verify they're not identical
        assert not np.allclose(up, down)

    def test_initial_phase_shifts_signal(self):
        t = np.linspace(0, 1e-6, 100)
        s0 = generate_lfm_chirp(t, f0=0.0, f1=1e6, initial_phase_rad=0.0)
        s_pi = generate_lfm_chirp(t, f0=0.0, f1=1e6, initial_phase_rad=np.pi)
        # Multiply by -1
        np.testing.assert_allclose(s_pi, -s0, rtol=1e-10)


# =====================================================================
# AWGN
# =====================================================================

class TestAWGN:
    """Additive white Gaussian noise tests."""

    def test_mutually_exclusive_modes(self):
        signal = np.ones(10, dtype=np.complex128)
        with pytest.raises(ValueError, match="mutually exclusive"):
            add_awgn(signal, snr_db=10.0, noise_floor_w=1.0)

    def test_requires_one_mode(self):
        signal = np.ones(10, dtype=np.complex128)
        with pytest.raises(ValueError, match="must be given"):
            add_awgn(signal)

    def test_zero_signal_raises(self):
        with pytest.raises(ValueError, match="zero or negative power"):
            add_awgn(np.zeros(10, dtype=np.complex128), snr_db=10.0)

    def test_achieved_snr(self):
        """Empirical SNR matches the requested SNR (within tolerance)."""
        rng = np.random.default_rng(42)
        signal = np.ones(50_000, dtype=np.complex128) * np.sqrt(10.0)
        for snr in (5.0, 10.0, 20.0):
            noisy = add_awgn(signal, snr_db=snr, rng=rng)
            p_signal = float(np.mean(np.abs(signal) ** 2))
            p_noise = float(np.mean(np.abs(noisy - signal) ** 2))
            snr_emp = 10.0 * np.log10(p_signal / p_noise)
            assert abs(snr_emp - snr) < 0.5, f"SNR {snr}: empirical {snr_emp:.2f}"

    def test_noise_floor_mode(self):
        """When noise_floor_w given, total noise power matches."""
        rng = np.random.default_rng(0)
        signal = np.zeros(50_000, dtype=np.complex128)
        noisy = add_awgn(signal, noise_floor_w=1.0, rng=rng)
        p_noise = float(np.mean(np.abs(noisy) ** 2))
        np.testing.assert_allclose(p_noise, 1.0, rtol=0.1)

    def test_box_muller_circular(self):
        """Box-Muller produces circularly-symmetric complex Gaussian."""
        rng = np.random.default_rng(0)
        # Use noise_floor_w mode (no signal, so SNR mode would raise)
        _, noise = add_awgn(np.zeros(10000, dtype=np.complex128),
                            noise_floor_w=2.0, rng=rng, return_noise=True)
        # I and Q have equal variance
        np.testing.assert_allclose(np.var(noise.real), np.var(noise.imag), rtol=0.2)
        # Zero mean
        assert abs(noise.real.mean()) < 0.1
        assert abs(noise.imag.mean()) < 0.1

    def test_return_noise(self):
        rng = np.random.default_rng(0)
        signal = np.ones(10, dtype=np.complex128)
        noisy, noise = add_awgn(signal, snr_db=20.0, rng=rng, return_noise=True)
        np.testing.assert_array_almost_equal(noisy, signal + noise)


class TestSNREstimate:
    def test_estimate_returns_finite_for_clean_signal(self):
        rng = np.random.default_rng(0)
        # 30 dB SNR signal — estimate should be in [25, 35] dB
        signal = np.ones(1000, dtype=np.complex128) * 10.0
        noisy = add_awgn(signal, snr_db=30.0, rng=rng)
        est = estimate_snr(noisy)
        assert 25.0 < est < 35.0

    def test_estimate_low_snr(self):
        rng = np.random.default_rng(0)
        signal = np.ones(1000, dtype=np.complex128) * 1.0
        noisy = add_awgn(signal, snr_db=0.0, rng=rng)
        est = estimate_snr(noisy)
        # 0 dB SNR — estimate should be roughly 0 dB
        assert -5.0 < est < 5.0


# =====================================================================
# Up/down conversion
# =====================================================================

class TestUpDownConversion:
    """Complex baseband <-> RF passband conversion tests."""

    def test_up_down_round_trip_dc(self):
        """DC baseband recovers with the expected 0.5 round-trip conversion gain."""
        f_c = 1e6
        f_s = 5e6
        # Standard upconversion: s_RF(t) = Re{b * exp(jωt)} = |b|/2 in amplitude.
        # After down-mix: mixed_DC = b/2. After LPF (DC gain = 1): recovered = b/2.
        # So round-trip recovers half the input.
        baseband = np.full(500, 2.5 + 1.5j, dtype=np.complex128)
        rf = complex_baseband_to_rf(baseband, f_c=f_c, f_s=f_s)
        assert rf.dtype == np.float64
        recovered = rf_to_complex_baseband(rf, f_c=f_c, f_s=f_s)
        # Recovered ≈ (2.5+1.5j) × 0.5 = (1.25 + 0.75j)
        np.testing.assert_allclose(
            recovered[100:].real, 1.25, rtol=0.05,
        )
        np.testing.assert_allclose(
            recovered[100:].imag, 0.75, rtol=0.05,
        )

    def test_up_down_round_trip_narrowband(self):
        """Narrowband tone within the LPF passband recovers with correct amplitude."""
        f_c = 1e6
        f_s = 5e6
        t = np.arange(500, dtype=np.float64) / f_s
        # Baseband tone at 10 kHz. After up-mix the RF has 10 kHz offset
        # from f_c. After down-mix it returns to 10 kHz baseband. The LPF
        # passes 10 kHz with near-unity gain. Round-trip amplitude = input/2.
        baseband = (np.exp(2j * np.pi * 10e3 * t) * 2.0).astype(np.complex128)
        rf = complex_baseband_to_rf(baseband, f_c=f_c, f_s=f_s)
        recovered = rf_to_complex_baseband(rf, f_c=f_c, f_s=f_s)
        # Recovered amplitude ≈ 1.0 (= 2.0 / 2)
        np.testing.assert_allclose(
            np.abs(recovered[100:]).mean(), 1.0, rtol=0.05,
        )

    def test_nyquist_violation_raises(self):
        with pytest.raises(ValueError, match="Nyquist"):
            complex_baseband_to_rf(
                np.ones(10, dtype=np.complex128), f_c=1e6, f_s=1.5e6,
            )

    def test_empty_input(self):
        rf = complex_baseband_to_rf(np.array([], dtype=np.complex128), f_c=1e6, f_s=5e6)
        assert rf.shape == (0,)
        bb = rf_to_complex_baseband(np.array([]), f_c=1e6, f_s=5e6)
        assert bb.shape == (0,)


# =====================================================================
# Matched filter
# =====================================================================

class TestMatchedFilter:
    """Pulse compression with matched filtering."""

    def test_compresses_lfm(self):
        """Matched filter on an LFM chirp should give a single sharp peak."""
        t = np.linspace(0, 10e-6, 1000)
        chirp = generate_lfm_chirp(t, f0=0.0, f1=1e6)
        compressed = matched_filter(chirp, chirp, mode="sampleless")
        # The peak should be where the chirp matches itself (anywhere;
        # just check it's much larger than the baseline)
        peak = float(np.max(np.abs(compressed)))
        baseline = float(np.median(np.abs(compressed)))
        assert peak > 5 * baseline

    def test_full_mode_peak_position(self):
        """In 'full' correlation mode, peak should be at the centre."""
        t = np.linspace(0, 1e-5, 1000)
        chirp = generate_lfm_chirp(t, f0=0.0, f1=1e6)
        compressed = matched_filter(chirp, chirp, mode="full")
        # Peak near the centre of the output
        peak_idx = int(np.argmax(np.abs(compressed)))
        centre = len(compressed) // 2
        assert abs(peak_idx - centre) < 50

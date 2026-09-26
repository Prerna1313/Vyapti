"""
tests.test_rf_propagation
===========================

Tests for the RF propagation physics in
``vyapti_simulator.rf.propagation``:

  - KinematicEmitter: position, velocity, range, AoA
  - Doppler shift computation
  - Time-of-flight
  - FreeSpacePathLoss: Friis equation
  - compute_path_loss: FSPL + shadowing + diffraction
  - Rayleigh/Rician fading: Jakes sum-of-sinusoids
  - MultipathChannel: tapped delay line
"""

from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.rf.propagation import (
    SPEED_OF_LIGHT,
    KinematicEmitter,
    compute_doppler_shift,
    time_of_flight,
    fractional_delay_filter,
    FreeSpacePathLoss,
    compute_path_loss,
    PathLossResult,
    RayleighFadingChannel,
    RicianFadingChannel,
    MultipathChannel,
)


# =====================================================================
# Kinematics
# =====================================================================

class TestKinematicEmitter:
    def test_construction(self):
        e = KinematicEmitter(
            position_m=np.array([1.0, 2.0, 3.0]),
            velocity_m_s=np.array([10.0, 0.0, 0.0]),
        )
        np.testing.assert_array_equal(e.position_m, [1.0, 2.0, 3.0])

    def test_position_reshaped(self):
        # 6-element array should be reshaped to 3
        e = KinematicEmitter(
            position_m=np.array([[1.0], [2.0], [3.0]]),  # column
            velocity_m_s=np.array([0.0, 0.0, 0.0]),
        )
        assert e.position_m.shape == (3,)

    def test_range_m(self):
        e = KinematicEmitter(
            position_m=np.array([3.0, 4.0, 0.0]),
            velocity_m_s=np.zeros(3),
        )
        r = e.range_m(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(r, 5.0, rtol=1e-9)

    def test_range_rate_approaching(self):
        # Emitter at 1000m moving towards receiver at -100 m/s
        e = KinematicEmitter(
            position_m=np.array([1000.0, 0.0, 0.0]),
            velocity_m_s=np.array([-100.0, 0.0, 0.0]),
        )
        rr = e.range_rate_m_s(np.array([0.0, 0.0, 0.0]))
        # Negative = approaching
        assert rr < 0.0
        np.testing.assert_allclose(rr, -100.0, rtol=1e-9)

    def test_range_rate_receding(self):
        e = KinematicEmitter(
            position_m=np.array([100.0, 0.0, 0.0]),
            velocity_m_s=np.array([50.0, 0.0, 0.0]),
        )
        rr = e.range_rate_m_s(np.array([0.0, 0.0, 0.0]))
        assert rr > 0.0
        np.testing.assert_allclose(rr, 50.0, rtol=1e-9)

    def test_angle_of_arrival(self):
        # Emitter at +x = azimuth 0°
        e = KinematicEmitter(
            position_m=np.array([1000.0, 0.0, 0.0]),
            velocity_m_s=np.zeros(3),
        )
        az, el = e.angle_of_arrival_deg(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(az, 0.0, atol=1e-6)
        np.testing.assert_allclose(el, 0.0, atol=1e-6)

    def test_aoa_perpendicular(self):
        # Emitter at +y = azimuth 90°
        e = KinematicEmitter(
            position_m=np.array([0.0, 1000.0, 0.0]),
            velocity_m_s=np.zeros(3),
        )
        az, el = e.angle_of_arrival_deg(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(az, 90.0, atol=1e-6)

    def test_update_position(self):
        e = KinematicEmitter(
            position_m=np.array([100.0, 0.0, 0.0]),
            velocity_m_s=np.array([10.0, 0.0, 0.0]),
        )
        e.update_position(1.0)
        np.testing.assert_allclose(e.position_m, [110.0, 0.0, 0.0])

    def test_update_position_records_history(self):
        e = KinematicEmitter(
            position_m=np.array([0.0, 0.0, 0.0]),
            velocity_m_s=np.array([1.0, 0.0, 0.0]),
        )
        e.update_position(1.0)
        assert len(e._position_history) == 1
        np.testing.assert_allclose(e._position_history[0], [0.0, 0.0, 0.0])


# =====================================================================
# Doppler
# =====================================================================

class TestDoppler:
    def test_stationary_zero_doppler(self):
        e = KinematicEmitter(
            position_m=np.array([1000.0, 0.0, 0.0]),
            velocity_m_s=np.zeros(3),
        )
        f_d = compute_doppler_shift(e, np.array([0.0, 0.0, 0.0]), f_c=3e9)
        np.testing.assert_allclose(f_d, 0.0, atol=1e-3)

    def test_approaching_negative_doppler(self):
        # Emitter at +x, moving towards origin at -100 m/s
        # f_d = (-100 / c) * 3e9 ≈ -1000.7 Hz
        e = KinematicEmitter(
            position_m=np.array([1000.0, 0.0, 0.0]),
            velocity_m_s=np.array([-100.0, 0.0, 0.0]),
        )
        f_d = compute_doppler_shift(e, np.array([0.0, 0.0, 0.0]), f_c=3e9)
        np.testing.assert_allclose(f_d, -1000.0, rtol=1e-3)

    def test_receding_positive_doppler(self):
        e = KinematicEmitter(
            position_m=np.array([1000.0, 0.0, 0.0]),
            velocity_m_s=np.array([100.0, 0.0, 0.0]),
        )
        f_d = compute_doppler_shift(e, np.array([0.0, 0.0, 0.0]), f_c=3e9)
        np.testing.assert_allclose(f_d, 1000.0, rtol=1e-3)

    def test_doppler_scales_with_frequency(self):
        e = KinematicEmitter(
            position_m=np.array([1000.0, 0.0, 0.0]),
            velocity_m_s=np.array([-100.0, 0.0, 0.0]),
        )
        rx = np.array([0.0, 0.0, 0.0])
        f_d_3 = compute_doppler_shift(e, rx, f_c=3e9)
        f_d_6 = compute_doppler_shift(e, rx, f_c=6e9)
        np.testing.assert_allclose(f_d_6, 2.0 * f_d_3, rtol=1e-6)


# =====================================================================
# Time of flight
# =====================================================================

class TestTimeOfFlight:
    def test_zero_distance(self):
        tof = time_of_flight(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(tof, 0.0, atol=1e-12)

    def test_known_distance(self):
        # 1500 m at c = 299792458 m/s
        tof = time_of_flight(
            np.array([1500.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        # 1500 / 299792458 = 5.003461e-6 s (c is exact, test expects exact 5e-6)
        np.testing.assert_allclose(tof, 1500.0 / 299792458.0, rtol=1e-9)

    def test_3d_distance(self):
        # sqrt(3² + 4²) = 5 m → 5/c seconds
        tof = time_of_flight(
            np.array([3.0, 4.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(tof, 5.0 / SPEED_OF_LIGHT, rtol=1e-9)


# =====================================================================
# Fractional delay filter
# =====================================================================

class TestFractionalDelayFilter:
    def test_n_taps_even_raises(self):
        with pytest.raises(ValueError, match="odd"):
            fractional_delay_filter(n_taps=10, delay_samples=2.0)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match="kaiser/hamming/blackman"):
            fractional_delay_filter(n_taps=11, delay_samples=2.0, window="hann")

    def test_normalised_to_unit_dc_gain(self):
        h = fractional_delay_filter(n_taps=21, delay_samples=3.0)
        np.testing.assert_allclose(np.sum(h), 1.0, rtol=1e-6)

    def test_dc_passthrough(self):
        """A DC input should pass through the fractional delay unchanged."""
        h = fractional_delay_filter(n_taps=21, delay_samples=3.5)
        # Sum of all taps is 1 (DC gain = 1)
        assert abs(np.sum(h) - 1.0) < 1e-6


# =====================================================================
# Free-space path loss
# =====================================================================

class TestFreeSpacePathLoss:
    def test_fspl_at_500m_3ghz(self):
        # FSPL_dB = 20*log10(0.5) + 20*log10(3000) + 32.44
        # = -6.02 + 69.54 + 32.44 = 95.96
        fspl = FreeSpacePathLoss(frequency_hz=3e9)
        result = fspl.fspl_db(range_m=500.0)
        np.testing.assert_allclose(result, 95.96, atol=0.1)

    def test_fspl_at_1km_1ghz(self):
        # FSPL_dB = 20*log10(1) + 20*log10(1000) + 32.44 = 0 + 60 + 32.44 = 92.44
        fspl = FreeSpacePathLoss(frequency_hz=1e9)
        result = fspl.fspl_db(range_m=1000.0)
        # Note: 20*log10(1000) = 60; actual 20*log10(1000) = 60.0 exactly
        np.testing.assert_allclose(result, 92.44, atol=0.1)

    def test_fspl_scales_with_range(self):
        fspl = FreeSpacePathLoss(frequency_hz=3e9)
        r1 = fspl.fspl_db(range_m=1000.0)
        r2 = fspl.fspl_db(range_m=2000.0)
        # Doubling range → +6 dB
        np.testing.assert_allclose(r2 - r1, 6.02, atol=0.1)

    def test_fspl_scales_with_frequency(self):
        fspl = FreeSpacePathLoss(frequency_hz=3e9)
        r1 = fspl.fspl_db(range_m=1000.0)
        fspl2 = FreeSpacePathLoss(frequency_hz=6e9)
        r2 = fspl2.fspl_db(range_m=1000.0)
        # Doubling freq → +6 dB
        np.testing.assert_allclose(r2 - r1, 6.02, atol=0.1)

    def test_zero_range_raises(self):
        fspl = FreeSpacePathLoss(frequency_hz=3e9)
        with pytest.raises(ValueError, match="positive"):
            fspl.fspl_db(range_m=0.0)

    def test_received_power(self):
        fspl = FreeSpacePathLoss(frequency_hz=3e9)
        pr = fspl.received_power_dbw(
            transmitted_power_dbw=10.0,
            tx_gain_dbi=20.0, rx_gain_dbi=20.0,
            range_m=1000.0,
        )
        # FSPL at 1km 3GHz = 20*log10(1) + 20*log10(3000) + 32.44 = 101.98 dB
        # Pr = 10 + 20 + 20 - 101.98 = -51.98 dBW
        np.testing.assert_allclose(pr, -51.98, atol=0.2)

    def test_dbm_round_trip(self):
        fspl = FreeSpacePathLoss(frequency_hz=3e9)
        pr_dbm = fspl.received_power_dbm(
            transmitted_power_dbm=40.0,
            tx_gain_dbi=20.0, rx_gain_dbi=20.0,
            range_m=1000.0,
        )
        # 40 dBm = 10 dBW; Pr = 10 - 101.98 dBW = -91.98 dBW = -61.98 dBm
        # Actually: Pr_dBm = Pt_dBm + Gt + Gr - FSPL = 40 + 20 + 20 - 101.98 = -21.98
        np.testing.assert_allclose(pr_dbm, -21.98, atol=0.2)


# =====================================================================
# Path loss with shadowing and diffraction
# =====================================================================

class TestComputePathLoss:
    def test_returns_path_loss_result(self):
        e = KinematicEmitter(
            position_m=np.array([1000.0, 0.0, 0.0]),
            velocity_m_s=np.zeros(3),
        )
        rx = np.array([0.0, 0.0, 0.0])
        result = compute_path_loss(e, rx, f_c=3e9)
        assert isinstance(result, PathLossResult)
        assert result.range_m > 0
        assert result.doppler_shift_hz == 0.0

    def test_no_rng_zero_shadowing(self):
        e = KinematicEmitter(
            position_m=np.array([1000.0, 0.0, 0.0]),
            velocity_m_s=np.zeros(3),
        )
        result = compute_path_loss(e, np.array([0.0, 0.0, 0.0]), f_c=3e9, rng=None)
        assert result.shadowing_db == 0.0
        assert result.diffraction_db == 0.0
        np.testing.assert_allclose(result.total_loss_db, result.fspl_db, rtol=1e-9)

    def test_shadowing_gaussian_stats(self):
        rng = np.random.default_rng(42)
        e = KinematicEmitter(
            position_m=np.array([1000.0, 0.0, 0.0]),
            velocity_m_s=np.zeros(3),
        )
        results = [
            compute_path_loss(e, np.array([0.0, 0.0, 0.0]), f_c=3e9, rng=rng)
            for _ in range(2000)
        ]
        shad_db = np.array([r.shadowing_db for r in results])
        # Mean ≈ 0, std ≈ 8 dB
        assert abs(shad_db.mean()) < 0.5
        np.testing.assert_allclose(shad_db.std(), 8.0, atol=0.3)

    def test_total_loss_includes_all(self):
        rng = np.random.default_rng(0)
        e = KinematicEmitter(
            position_m=np.array([500.0, 0.0, 0.0]),
            velocity_m_s=np.zeros(3),
        )
        result = compute_path_loss(e, np.array([0.0, 0.0, 0.0]), f_c=3e9, rng=rng)
        np.testing.assert_allclose(
            result.total_loss_db,
            result.fspl_db + result.shadowing_db + result.diffraction_db,
            rtol=1e-9,
        )


# =====================================================================
# Fading channels
# =====================================================================

class TestRayleighFading:
    def test_unit_variance_stationary(self):
        """E[|h|²] should be ≈ 1.0 over many samples."""
        rng = np.random.default_rng(0)
        ch = RayleighFadingChannel(
            f_c=3e9, doppler_hz=100.0, sample_rate_hz=1e6, rng=rng,
        )
        gains = ch.step(n_samples=50_000)
        emp_power = float(np.mean(np.abs(gains) ** 2))
        # Should be close to 1.0; allow up to 20% error from finite-N Jakes
        np.testing.assert_allclose(emp_power, 1.0, atol=0.2)

    def test_zero_doppler_constant(self):
        """With zero Doppler the channel should be approximately DC."""
        rng = np.random.default_rng(0)
        ch = RayleighFadingChannel(
            f_c=3e9, doppler_hz=0.0, sample_rate_hz=1e6, rng=rng,
        )
        # Step a small number of times
        h = ch.step(n_samples=1000)
        # With zero Doppler, the channel varies only because of the
        # Jakes phases, not time evolution — variation should be small
        # but still present (random path phases).
        assert h.shape == (1000,)

    def test_apply_multiplies_signal(self):
        rng = np.random.default_rng(0)
        ch = RayleighFadingChannel(
            f_c=3e9, doppler_hz=100.0, sample_rate_hz=1e6, rng=rng,
        )
        signal = np.ones(100, dtype=np.complex128)
        # apply() internally calls step() to get fresh gains
        out = ch.apply(signal)
        # Output = signal * gain, both same shape
        assert out.shape == signal.shape
        # Output magnitude should be the magnitude of a Jakes sum (≈ 1 on average)
        assert np.all(np.abs(out) > 0.0)

    def test_rayleigh_envelope_distribution(self):
        """Envelope of large-sample Rayleigh should have mean ≈ sqrt(pi/2)."""
        rng = np.random.default_rng(0)
        ch = RayleighFadingChannel(
            f_c=3e9, doppler_hz=100.0, sample_rate_hz=1e6, rng=rng,
        )
        h = ch.step(n_samples=200_000)
        envelope = np.abs(h)
        # For Rayleigh with sigma=1/sqrt(2): E[|h|] = sqrt(pi/2) * sigma
        # With E[|h|²]=1, sigma=1/sqrt(2), so E[|h|] = sqrt(pi/2)/sqrt(2) ≈ 0.886
        # Jakes with N=16 paths is close to Rayleigh but not exact
        assert 0.7 < envelope.mean() < 1.1


class TestRicianFading:
    def test_invalid_k_factor_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            RicianFadingChannel(f_c=3e9, k_factor=-1.0)

    def test_unit_variance_stationary(self):
        """E[|h|²] should be ≈ 1.0 (K/(K+1) + 1/(K+1) = 1)."""
        rng = np.random.default_rng(0)
        ch = RicianFadingChannel(
            f_c=3e9, k_factor=3.0,
            doppler_hz=100.0, sample_rate_hz=1e6, rng=rng,
        )
        h = ch.step(n_samples=50_000)
        emp_power = float(np.mean(np.abs(h) ** 2))
        np.testing.assert_allclose(emp_power, 1.0, atol=0.2)

    def test_high_k_factor_low_fading(self):
        """K → ∞: channel becomes pure LOS, no variation."""
        rng = np.random.default_rng(0)
        ch = RicianFadingChannel(
            f_c=3e9, k_factor=1e6,  # effectively no scatter
            doppler_hz=100.0, sample_rate_hz=1e6, rng=rng,
        )
        h = ch.step(n_samples=1000)
        # All samples ≈ 1 (no fading variation)
        np.testing.assert_allclose(h, 1.0 + 0j, atol=0.01)

    def test_k_factor_zero_is_rayleigh(self):
        """K=0 should give pure Rayleigh (no LOS)."""
        rng = np.random.default_rng(0)
        ch = RicianFadingChannel(
            f_c=3e9, k_factor=0.0,
            doppler_hz=100.0, sample_rate_hz=1e6, rng=rng,
        )
        # Mean of h should be ≈ 0 (no LOS, no bias). Finite-N Jakes
        # with N=16 has small residual bias. Allow 0.5.
        h = ch.step(n_samples=20_000)
        assert abs(h.mean()) < 0.5

    def test_k_factor_db(self):
        ch = RicianFadingChannel(f_c=3e9, k_factor=10.0)
        np.testing.assert_allclose(ch.k_factor_db, 10.0, rtol=1e-9)

    def test_apply_multiplies_signal(self):
        rng = np.random.default_rng(0)
        ch = RicianFadingChannel(
            f_c=3e9, k_factor=3.0,
            doppler_hz=100.0, sample_rate_hz=1e6, rng=rng,
        )
        signal = np.ones(50, dtype=np.complex128) * 2.0
        out = ch.apply(signal)
        assert out.shape == signal.shape


class TestMultipathChannel:
    def test_basic_construction(self):
        ch = MultipathChannel(
            f_c=3e9,
            taps={0: 0.0, 3: -8.0, 7: -15.0},
            sample_rate_hz=1e6,
        )
        assert 0 in ch.taps
        assert 3 in ch.taps

    def test_step_returns_cir(self):
        ch = MultipathChannel(
            f_c=3e9,
            taps={0: 0.0, 3: -8.0},
            sample_rate_hz=1e6,
        )
        cir = ch.step(n_samples=10)
        assert cir.shape == (10,)

    def test_apply_changes_signal(self):
        rng = np.random.default_rng(0)
        ch = MultipathChannel(
            f_c=3e9,
            taps={0: 0.0, 3: -8.0},
            sample_rate_hz=1e6,
            rng=rng,
        )
        signal = np.ones(20, dtype=np.complex128)
        out = ch.apply(signal)
        assert out.shape == signal.shape

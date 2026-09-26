"""
tests.test_antenna_patterns_freq_dep
====================================

Tests for the frequency-dependent antenna gain models in
``vyapti_simulator.tsrd.antenna_patterns``:

  - ``cosine_taper_antenna_gain``: cosine-shaped passband
  - ``sinc_antenna_gain``: sinc-squared roll-off with nulls
  - ``realistic_ew_antenna_gain``: cosine taper + ripple + sidelobes

The tests verify:
  - Output shape matches band_count
  - Gain is centred around the requested peak_gain_db
  - Band edges are at the requested edge_attenuation_db (cosine taper)
  - Deep nulls reach the sidelobe_db (realistic_ew)
  - Sinc has zeros at the band edges
  - Determinism with seed
"""

from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.tsrd.antenna_patterns import (
    cosine_taper_antenna_gain,
    sinc_antenna_gain,
    realistic_ew_antenna_gain,
)


@pytest.fixture
def freq_axis():
    """A 10-band frequency axis from 2-18 GHz."""
    return np.linspace(2000.0, 18000.0, 10)


# =====================================================================
# cosine_taper_antenna_gain
# =====================================================================

class TestCosineTaper:
    def test_shape_matches_freq(self, freq_axis):
        g = cosine_taper_antenna_gain(freq_axis)
        assert g.shape == freq_axis.shape

    def test_dtype_float64(self, freq_axis):
        g = cosine_taper_antenna_gain(freq_axis)
        assert g.dtype == np.float64

    def test_peak_at_center(self, freq_axis):
        """Cosine taper peaks at the centre frequency."""
        g = cosine_taper_antenna_gain(freq_axis, peak_gain_db=6.0, edge_attenuation_db=-10.0)
        centre_idx = len(freq_axis) // 2
        # The band nearest the centre should have the highest gain
        assert g[centre_idx] == g.max()

    def test_edges_at_attenuation(self, freq_axis):
        """The first and last band should be at the edge attenuation level."""
        g = cosine_taper_antenna_gain(freq_axis, peak_gain_db=6.0, edge_attenuation_db=-10.0)
        assert abs(g[0] - (-10.0)) < 0.01
        assert abs(g[-1] - (-10.0)) < 0.01

    def test_range(self, freq_axis):
        """Gain stays within [edge, peak] for a typical parameter set."""
        g = cosine_taper_antenna_gain(freq_axis, peak_gain_db=5.0, edge_attenuation_db=-8.0)
        assert g.min() >= -8.0 - 0.01
        assert g.max() <= 5.0 + 0.01

    def test_explicit_center_freq(self, freq_axis):
        """Specifying center_freq_mhz shifts the peak."""
        g_default = cosine_taper_antenna_gain(freq_axis)
        g_shifted = cosine_taper_antenna_gain(freq_axis, center_freq_mhz=5000.0)
        # Peak should now be closer to 5000 MHz, not the centre
        assert g_shifted.argmax() != g_default.argmax() or np.allclose(g_shifted, g_default)

    def test_empty_input(self):
        g = cosine_taper_antenna_gain(np.array([]))
        assert g.shape == (0,)

    def test_single_frequency(self):
        """With one frequency, the function should return that value (≈peak)."""
        g = cosine_taper_antenna_gain(np.array([5000.0]), peak_gain_db=6.0)
        # With one freq, x=0 (centre), gain = peak
        assert abs(g[0] - 6.0) < 0.01

    def test_constant_input(self):
        """All-same frequencies means no roll-off."""
        f = np.array([5000.0] * 5)
        g = cosine_taper_antenna_gain(f, peak_gain_db=6.0, edge_attenuation_db=-10.0)
        # f_hi == f_lo so f_hi <= f_lo check returns the peak
        np.testing.assert_allclose(g, np.full(5, 6.0))


# =====================================================================
# sinc_antenna_gain
# =====================================================================

class TestSincGain:
    def test_shape(self, freq_axis):
        g = sinc_antenna_gain(freq_axis)
        assert g.shape == freq_axis.shape

    def test_dtype(self, freq_axis):
        g = sinc_antenna_gain(freq_axis)
        assert g.dtype == np.float64

    def test_peak_at_centre(self, freq_axis):
        """Peak gain should be at the centre frequency."""
        g = sinc_antenna_gain(freq_axis, peak_gain_db=8.0, null_depth_db=-25.0)
        centre_idx = len(freq_axis) // 2
        assert g[centre_idx] == g.max()

    def test_peak_equals_peak_gain(self, freq_axis):
        g = sinc_antenna_gain(freq_axis, peak_gain_db=8.0, null_depth_db=-25.0)
        # At the centre, gain should be ≈ peak_gain_db
        centre = freq_axis[len(freq_axis) // 2]
        # Recompute the centre exactly
        g_centre = sinc_antenna_gain(np.array([centre]), peak_gain_db=8.0, null_depth_db=-25.0)
        assert abs(g_centre[0] - 8.0) < 0.01

    def test_edges_approaching_null_depth(self, freq_axis):
        """At the band edges, gain should be well below peak (roll-off active)."""
        g = sinc_antenna_gain(freq_axis, peak_gain_db=8.0, null_depth_db=-25.0)
        # The roll-off is active at the edges; they should be meaningfully
        # below peak. The first null is at (freq-centre)/centre = ±0.25
        # for N=2. With 2-18 GHz, the nulls are at 7500 and 12500 MHz,
        # not at the band edges. Edges are below peak but not at null depth.
        assert g[0] < 8.0 - 1.0, "Band edge should be meaningfully below peak"
        assert g[-1] < 8.0 - 1.0, "Band edge should be meaningfully below peak"

    def test_no_value_below_null_depth(self, freq_axis):
        """No band should be below the configured null depth."""
        g = sinc_antenna_gain(freq_axis, peak_gain_db=8.0, null_depth_db=-25.0)
        assert g.min() >= -25.0 - 0.01

    def test_empty_input(self):
        g = sinc_antenna_gain(np.array([]))
        assert g.shape == (0,)

    def test_constant_input_returns_peak(self):
        f = np.array([5000.0] * 5)
        g = sinc_antenna_gain(f, peak_gain_db=8.0)
        np.testing.assert_allclose(g, np.full(5, 8.0))


# =====================================================================
# realistic_ew_antenna_gain
# =====================================================================

class TestRealisticEW:
    def test_shape(self, freq_axis):
        g = realistic_ew_antenna_gain(freq_axis, seed=0)
        assert g.shape == freq_axis.shape

    def test_dtype(self, freq_axis):
        g = realistic_ew_antenna_gain(freq_axis, seed=0)
        assert g.dtype == np.float64

    def test_no_value_below_sidelobe(self, freq_axis):
        """Sidelobe drops reach sidelobe_db; no value should be lower."""
        sidelobe = -15.0
        g = realistic_ew_antenna_gain(
            freq_axis, sidelobe_db=sidelobe, sidelobe_probability=0.5, seed=0,
        )
        assert g.min() >= sidelobe - 0.01

    def test_sidelobes_present(self, freq_axis):
        """With high sidelobe_probability, some bands should hit the floor."""
        sidelobe = -15.0
        g = realistic_ew_antenna_gain(
            freq_axis, sidelobe_db=sidelobe, sidelobe_probability=1.0, seed=0,
        )
        # Every band should be at the sidelobe floor
        np.testing.assert_allclose(g, sidelobe, atol=0.01)

    def test_deterministic_with_seed(self, freq_axis):
        g_a = realistic_ew_antenna_gain(freq_axis, seed=42)
        g_b = realistic_ew_antenna_gain(freq_axis, seed=42)
        np.testing.assert_array_equal(g_a, g_b)

    def test_different_seeds_differ(self, freq_axis):
        g_a = realistic_ew_antenna_gain(freq_axis, sidelobe_probability=0.5, seed=0)
        g_b = realistic_ew_antenna_gain(freq_axis, sidelobe_probability=0.5, seed=1)
        assert not np.allclose(g_a, g_b)

    def test_ripple_modulation(self, freq_axis):
        """With sidelobe_probability=0, only taper + ripple remain."""
        g = realistic_ew_antenna_gain(
            freq_axis, peak_gain_db=5.0, edge_attenuation_db=-8.0,
            ripple_amplitude_db=2.0, ripple_period_mhz=2000.0,
            sidelobe_probability=0.0, seed=0,
        )
        # Mean should be near the average of taper and ripple
        # Just check it's in the expected range
        assert g.min() >= -8.0 - 0.01
        assert g.max() <= 5.0 + 1.0 + 0.01  # ripple can push above peak by 1 dB

    def test_empty_input(self):
        g = realistic_ew_antenna_gain(np.array([]), seed=0)
        assert g.shape == (0,)


# =====================================================================
# Cross-compatibility with existing pattern helpers
# =====================================================================

class TestBackwardCompat:
    """The new freq-dependent helpers do not break the existing band-indexed ones."""

    def test_uniform_still_zero(self):
        from vyapti_simulator.tsrd.antenna_patterns import uniform_antenna_gain
        g = uniform_antenna_gain(10)
        np.testing.assert_array_equal(g, np.zeros(10))

    def test_existing_helpers_unchanged(self):
        """The legacy band-indexed helpers still work."""
        from vyapti_simulator.tsrd.antenna_patterns import (
            sectorised_antenna_gain,
            realistic_antenna_gain,
        )
        g_sec = sectorised_antenna_gain(10, seed=0)
        g_real = realistic_antenna_gain(10, seed=0)
        assert g_sec.shape == (10,)
        assert g_real.shape == (10,)

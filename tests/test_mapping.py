"""
tests.test_mapping
==================

Unit tests for the frequency <-> band and time <-> slot mapping helpers.
These tests are part of the **Invariant-2 contract**: the grid is
defined once, in `SimulationConfig`, and every other module must use
the helpers in `vyapti_simulator.core.mapping`, not roll its own formula.
"""

import numpy as np
import pytest

from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.core.mapping import (
    band_center_frequency_mhz,
    band_edges_mhz,
    frequency_to_band,
    seconds_to_slot,
    slot_edges_seconds,
    slot_start_seconds,
)


# =====================================================================
# Fixtures: a single, canonical config used by most tests
# =====================================================================
@pytest.fixture
def cfg() -> SimulationConfig:
    """
    The canonical Vyapti grid that matches the TSRD receiver:
        * 500 MHz band width
        * 0 - 18 000 MHz total spectrum  -> 36 bands
        * 50 ms slot  ->  600 slots over 30 s
    """
    return SimulationConfig(
        band_count=36,
        total_spectrum_mhz=18_000.0,
        receiver_ibw_mhz=500.0,
        dwell_time_ms=50.0,
        time_slots=600,
    )


# =====================================================================
# frequency_to_band
# =====================================================================
class TestFrequencyToBand:
    def test_lower_edge_maps_to_band_zero(self, cfg):
        assert frequency_to_band(0.0, cfg) == 0

    def test_first_band_centre(self, cfg):
        # Centre of band 0 is at 250 MHz (midpoint of 0-500 MHz)
        assert frequency_to_band(250.0, cfg) == 0

    def test_second_band_centre(self, cfg):
        # Centre of band 1 is at 750 MHz
        assert frequency_to_band(750.0, cfg) == 1

    def test_last_band_centre(self, cfg):
        # Centre of band 35 is at 17 750 MHz
        assert frequency_to_band(17_750.0, cfg) == 35

    def test_frequency_just_before_upper_edge(self, cfg):
        # 17 999.9 MHz is still in the last band
        assert frequency_to_band(17_999.9, cfg) == 35

    def test_frequency_at_upper_edge_clamps_to_last_band(self, cfg):
        # 18 000 MHz is the upper edge; it is NOT in any new band.
        # The helper must clamp it to the last band.
        assert frequency_to_band(18_000.0, cfg) == 35

    def test_negative_frequency_clamps_to_zero(self, cfg):
        # TSRD has 9 299 pulses below 500 MHz. Even if a future
        # file has a pulse at -10 MHz, the helper must not crash.
        assert frequency_to_band(-10.0, cfg) == 0

    def test_very_high_frequency_clamps_to_last_band(self, cfg):
        assert frequency_to_band(1e9, cfg) == 35

    def test_invalid_total_spectrum_raises(self):
        with pytest.raises(ValueError, match="total_spectrum_mhz"):
            SimulationConfig(band_count=10, total_spectrum_mhz=0.0)

    def test_invalid_band_count_raises(self):
        with pytest.raises(ValueError, match="band_count"):
            SimulationConfig(band_count=0, total_spectrum_mhz=2000.0)

    def test_monotonicity(self, cfg):
        """
        The mapping must be monotonically non-decreasing: if
        f1 < f2 then band(f1) <= band(f2). This is the property
        the future oracle relies on for window queries.
        """
        freqs = [i * 100.0 for i in range(200)]  # 0 -> 19 900 MHz
        bands = [frequency_to_band(f, cfg) for f in freqs]
        for i in range(len(bands) - 1):
            assert bands[i] <= bands[i + 1], (
                f"Non-monotonic at f={freqs[i+1]} MHz: "
                f"band({freqs[i]})={bands[i]} > band({freqs[i+1]})={bands[i+1]}"
            )

    def test_round_trip_with_band_center(self, cfg):
        """
        For every band, frequency_to_band(band_center) == band.
        """
        for b in range(cfg.band_count):
            center = band_center_frequency_mhz(b, cfg)
            assert frequency_to_band(center, cfg) == b, (
                f"Round-trip failed at band {b}: center={center} MHz"
            )


# =====================================================================
# seconds_to_slot
# =====================================================================
class TestSecondsToSlot:
    def test_zero_maps_to_slot_zero(self, cfg):
        assert seconds_to_slot(0.0, cfg) == 0

    def test_one_slot_centre(self, cfg):
        # Centre of slot 0 is at 25 ms; centre of slot 1 is at 75 ms.
        assert seconds_to_slot(0.025, cfg) == 0
        assert seconds_to_slot(0.075, cfg) == 1

    def test_last_slot_centre(self, cfg):
        # Centre of slot 599 is at 29.975 s
        assert seconds_to_slot(29.975, cfg) == 599

    def test_time_just_before_mission_end(self, cfg):
        assert seconds_to_slot(29.999, cfg) == 599

    def test_time_at_mission_end_clamps_to_last_slot(self, cfg):
        # 30.0 s is exactly the end; clamp to the last slot
        assert seconds_to_slot(30.0, cfg) == 599

    def test_negative_time_clamps_to_zero(self, cfg):
        assert seconds_to_slot(-0.1, cfg) == 0

    def test_time_far_in_future_clamps_to_last_slot(self, cfg):
        assert seconds_to_slot(1e6, cfg) == 599

    def test_invalid_slot_duration_raises(self):
        with pytest.raises(ValueError, match="dwell_time_ms"):
            SimulationConfig(
                band_count=10, total_spectrum_mhz=2000.0,
                dwell_time_ms=0.0,
            )

    def test_monotonicity(self, cfg):
        """
        Monotonicity over a dense sweep of 3100 timestamps (0 -> 31 s).
        """
        times = [i * 0.01 for i in range(3100)]
        slots = [seconds_to_slot(t, cfg) for t in times]
        for i in range(len(slots) - 1):
            assert slots[i] <= slots[i + 1], (
                f"Non-monotonic at t={times[i+1]} s: "
                f"slot({times[i]})={slots[i]} > slot({times[i+1]})={slots[i+1]}"
            )

    def test_round_trip_with_slot_start(self, cfg):
        """
        For every slot, seconds_to_slot(slot_start) == slot.
        """
        for s in range(cfg.time_slots):
            start = slot_start_seconds(s, cfg)
            assert seconds_to_slot(start, cfg) == s


# =====================================================================
# Edge case: different grid (the synthetic simulator's default)
# =====================================================================
class TestAlternateGrid:
    """
    The same helpers must work for the synthetic simulator's default
    grid (10 bands, 200 MHz IBW, 10 ms slots, 1000 slots).
    """
    @pytest.fixture
    def small_cfg(self) -> SimulationConfig:
        return SimulationConfig()  # all defaults

    def test_synthetic_grid_lower_edge(self, small_cfg):
        # Default: total_spectrum_mhz=2000, band_count=10
        assert frequency_to_band(0.0, small_cfg) == 0

    def test_synthetic_grid_first_band_centre(self, small_cfg):
        # 2000 MHz / 10 bands = 200 MHz per band; centre of band 0 is at 100 MHz
        assert frequency_to_band(100.0, small_cfg) == 0

    def test_synthetic_grid_last_band(self, small_cfg):
        # 1900 MHz is the centre of band 9
        assert frequency_to_band(1900.0, small_cfg) == 9

    def test_synthetic_grid_slot_round_trip(self, small_cfg):
        # 1000 slots of 10 ms; the start of each must round-trip
        for s in range(small_cfg.time_slots):
            start = slot_start_seconds(s, small_cfg)
            assert seconds_to_slot(start, small_cfg) == s


# =====================================================================
# Edge helpers (band_edges_mhz, slot_edges_seconds)
# =====================================================================
class TestEdgeHelpers:
    def test_band_edges_match_grid(self, cfg):
        lower, upper, width = band_edges_mhz(cfg)
        assert lower == 0.0
        assert upper == 18_000.0
        assert width == 500.0

    def test_slot_edges_match_grid(self, cfg):
        start, end, dur = slot_edges_seconds(cfg)
        assert start == 0.0
        assert end == 30.0
        assert dur == 0.050


# =====================================================================
# TSRD dwell-centre mode (nearest-neighbour assignment)
# =====================================================================
class TestDwellCentreMode:
    """
    When `SimulationConfig.dwell_centres_mhz` is set, frequency_to_band
    uses nearest-neighbour assignment against the recorded tune
    frequencies instead of the uniform-band-centre formula.

    This matters when the H5's dwell centres deviate from the
    (band + 0.5) * band_width formula — e.g. if a non-uniform
    scan pattern was used, or if the two differ at a boundary.
    """

    def test_nearest_centre_is_used(self):
        """A frequency closer to centre[3] than centre[2] goes to band 3."""
        centres = np.linspace(250.0, 17_750.0, 36)
        # Shift band 2 centre to 1350 so band 2 covers [1000, 1700]
        centres[2] = 1350.0
        cfg = SimulationConfig(
            band_count=36, total_spectrum_mhz=18_000.0,
            dwell_centres_mhz=centres,
        )
        # 1400 MHz: closer to centre[2]=1350 (dist=50) than centre[3]=1750 (dist=350)
        assert frequency_to_band(1400.0, cfg) == 2

    def test_shifted_centre_resolves_cross_boundary(self):
        """
        When a dwell centre crosses a uniform-band boundary, the
        discretisation changes. Example: if band 2's centre is moved
        from 1250 (uniform) to 1750 (uniform band 3's centre),
        then a frequency at 1500 MHz which uniformly maps to band 3
        may map to band 2 under the shifted-centre scheme.
        """
        centres = np.linspace(250.0, 17_750.0, 36)
        centres[1] = 1750.0  # Band 1 centre moved to uniform band 3's centre
        cfg = SimulationConfig(
            band_count=36, total_spectrum_mhz=18_000.0,
            dwell_centres_mhz=centres,
        )
        # Uniform: 1500 MHz -> band 3 (centre 1750, distance 250)
        # TSRD: 1500 MHz -> band 1 (centre 1750, distance 250) vs band 2 (centre 2250, dist 750) -> band 1
        uniform_cfg = SimulationConfig(band_count=36, total_spectrum_mhz=18_000.0)
        assert frequency_to_band(1500.0, uniform_cfg) == 3
        assert frequency_to_band(1500.0, cfg) == 1

    def test_band_center_returns_dwell_centre(self):
        """band_center_frequency_mhz returns the configured centre, not the uniform one."""
        centres = np.array([250.0, 1350.0] + list(np.linspace(2250.0, 17_750.0, 34)), dtype=np.float64)
        cfg = SimulationConfig(
            band_count=36, total_spectrum_mhz=18_000.0,
            dwell_centres_mhz=centres,
        )
        # Band 0: uniform centre=250, TSRD centre=250
        assert band_center_frequency_mhz(0, cfg) == 250.0
        # Band 1: uniform centre=750, TSRD centre=1350
        assert band_center_frequency_mhz(1, cfg) == 1350.0
        uniform_cfg = SimulationConfig(band_count=36, total_spectrum_mhz=18_000.0)
        assert band_center_frequency_mhz(1, uniform_cfg) == 750.0

    def test_nan_frequency_clamps_to_band_zero(self):
        """NaN frequencies are a graceful degradation to band 0."""
        centres = np.linspace(250.0, 17_750.0, 36)
        cfg = SimulationConfig(band_count=36, total_spectrum_mhz=18_000.0, dwell_centres_mhz=centres)
        assert frequency_to_band(float("nan"), cfg) == 0
        assert frequency_to_band(float("-inf"), cfg) == 0

    def test_wrong_length_dwell_centres_raises(self):
        """Mismatched length raises ValueError, not silently truncating."""
        centres = np.linspace(250.0, 17_750.0, 10)  # wrong length
        with pytest.raises(ValueError, match="length"):
            SimulationConfig(band_count=36, total_spectrum_mhz=18_000.0, dwell_centres_mhz=centres)

    def test_non_finite_dwell_centres_raises(self):
        """NaN in dwell_centres_mhz raises, not silently corrupts the assignment."""
        centres = np.linspace(250.0, 17_750.0, 36)
        centres[5] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            SimulationConfig(band_count=36, total_spectrum_mhz=18_000.0, dwell_centres_mhz=centres)

    def test_round_trip_nearest_centre(self):
        """band_center_frequency_mhz(round_trip) == band when centres are non-uniform."""
        centres = np.linspace(250.0, 17_750.0, 36)
        cfg = SimulationConfig(band_count=36, total_spectrum_mhz=18_000.0, dwell_centres_mhz=centres)
        for b in range(36):
            centre = band_center_frequency_mhz(b, cfg)
            assert frequency_to_band(centre, cfg) == b, f"Round-trip failed at band {b}"

    def test_uniform_tsrc_fixture_centres_match_uniform(self):
        """
        The TSRD fixture has uniformly-spaced dwell centres at
        [250, 750, 1250, ...] which coincide with the uniform
        band centres. This is a regression test: the two paths
        must agree for the fixture.
        """
        centres = np.array(
            [250.0, 750.0, 1250.0, 1750.0, 2250.0, 2750.0, 3250.0, 3750.0,
             4250.0, 4750.0, 5250.0, 5750.0, 6250.0, 6750.0, 7250.0, 7750.0,
             8250.0, 8750.0, 9250.0, 9750.0, 10250.0, 10750.0, 11250.0, 11750.0,
             12250.0, 12750.0, 13250.0, 13750.0, 14250.0, 14750.0, 15250.0,
             15750.0, 16250.0, 16750.0, 17250.0, 17750.0],
            dtype=np.float64,
        )
        cfg_uniform = SimulationConfig(band_count=36, total_spectrum_mhz=18_000.0)
        cfg_tsrd = SimulationConfig(
            band_count=36, total_spectrum_mhz=18_000.0,
            dwell_centres_mhz=centres,
        )
        # Test across a range of frequencies
        test_freqs = np.linspace(0.0, 18_000.0, 200)
        for f in test_freqs:
            u = frequency_to_band(f, cfg_uniform)
            t = frequency_to_band(f, cfg_tsrd)
            assert u == t, (
                f"Uniform and TSRD schemes disagree at {f:.1f} MHz: "
                f"uniform band {u}, TSRD band {t}"
            )

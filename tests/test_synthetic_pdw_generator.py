"""
tests.test_synthetic_pdw_generator
==================================

Tests for `SyntheticEWPDWGenerator` — the synthetic EW PDW
generator that produces TSRD-shaped `PDWStream` objects from
`SyntheticEmitterSpec` lists.

The generator is the fallback path for Kaggle runs when TSRD is
unavailable. It must:

  * Emit dtypes that match the TSRD schema (float32 for the
    five PDW fields, int64 for emitter_id).
  * Honour the per-emitter pulse count implied by the spec
    (PRI × duration).
  * Honour `aoa_jitter_deg` and `pw_jitter_sec` as noise
    parameters.
  * Round-trip through `quick_deinterleave` for the canonical
    2-emitter and 6-emitter scenarios (i.e. the AoA-first
    deinterleaver finds the right number of tracks).
  * Be deterministic given the same seed.

Author
------
Senior RF/EW Signal Simulation Engineer — Vyapti Stage 2 tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.tsrd import (
    SyntheticEmitterSpec,
    SyntheticEWPDWGenerator,
    default_two_emitter_scenario,
    default_six_emitter_scenario,
    quick_deinterleave,
)


# =====================================================================
# Helpers
# =====================================================================


def _build_simple_spec(
    emitter_id: int = 0,
    aoa_deg: float = 10.0,
    center_freq_hz: float = 2.4e9,
    pri_sec: float = 1.0e-3,
    pulse_width_sec: float = 1.0e-6,
    snr_db: float = 15.0,
    pw_jitter_sec: float = 0.0,
    aoa_jitter_deg: float = 0.0,
    seed_offset: int = 0,
) -> SyntheticEmitterSpec:
    return SyntheticEmitterSpec(
        emitter_id=emitter_id,
        aoa_deg=aoa_deg,
        snr_db=snr_db,
        emitter_type="fixed_continuous",
        center_freq_hz=center_freq_hz,
        pri_sec=pri_sec,
        pulse_width_sec=pulse_width_sec,
        pw_jitter_sec=pw_jitter_sec,
        aoa_jitter_deg=aoa_jitter_deg,
        seed_offset=seed_offset,
    )


# =====================================================================
# TestSchema
# =====================================================================


class TestSchema:
    """The output stream must be TSRD-shaped."""

    def test_dtypes_match_tsrd(self):
        """toa/freq/pw/aoa/amp are float32, emitter_id is int64."""
        specs = [_build_simple_spec()]
        gen = SyntheticEWPDWGenerator(
            specs, mission_duration_s=0.1, seed=42, noise_floor_db=-90.0,
        )
        s = gen.generate()
        assert s.toa_us.dtype == np.float32
        assert s.freq_mhz.dtype == np.float32
        assert s.pw_us.dtype == np.float32
        assert s.aoa_deg.dtype == np.float32
        assert s.amp_db.dtype == np.float32
        assert s.emitter_id.dtype == np.int64

    def test_units_are_us_mhz(self):
        """ToA in microseconds, freq in MHz, PW in microseconds."""
        specs = [_build_simple_spec(pri_sec=500e-6, pulse_width_sec=2e-6)]
        gen = SyntheticEWPDWGenerator(
            specs, mission_duration_s=0.1, seed=42, noise_floor_db=-90.0,
        )
        s = gen.generate()
        # ToA must be > 0; PW around 2 µs (±0.1 µs of noise)
        assert (s.toa_us > 0).all()
        assert 1.5 < s.pw_us.mean() < 2.5
        # freq_mhz: 2.4 GHz = 2400 MHz
        assert 2300 < s.freq_mhz.mean() < 2500


# =====================================================================
# TestPulseCounts
# =====================================================================


class TestPulseCounts:
    """The generator should produce approximately the right number of pulses."""

    def test_single_emitter_pulse_count_close_to_expected(self):
        """PRI=1 ms, duration=1 s → ~1000 pulses (allow ±20% slack)."""
        specs = [_build_simple_spec(pri_sec=1e-3)]
        gen = SyntheticEWPDWGenerator(
            specs, mission_duration_s=1.0, seed=42, noise_floor_db=-90.0,
        )
        s = gen.generate()
        assert 800 < len(s) < 1200

    def test_no_pulses_for_zero_duration(self):
        """Duration 0 is rejected by the constructor (positive enforced)."""
        specs = [_build_simple_spec()]
        with pytest.raises(ValueError):
            SyntheticEWPDWGenerator(
                specs, mission_duration_s=0.0, seed=42, noise_floor_db=-90.0,
            )


# =====================================================================
# TestJitter
# =====================================================================


class TestJitter:
    """Per-pulse noise must be applied with the right scale."""

    def test_pw_jitter_adds_noise(self):
        """Non-zero pw_jitter_sec → PW std > 0."""
        specs = [_build_simple_spec(pw_jitter_sec=0.1e-6)]
        gen = SyntheticEWPDWGenerator(
            specs, mission_duration_s=0.5, seed=42, noise_floor_db=-90.0,
        )
        s = gen.generate()
        # With σ=0.1 µs and N=500, sample std should be ~0.1 µs
        assert s.pw_us.std() > 0.05

    def test_aoa_jitter_adds_noise(self):
        """Non-zero aoa_jitter_deg → AoA std > 0."""
        specs = [_build_simple_spec(aoa_jitter_deg=0.5)]
        gen = SyntheticEWPDWGenerator(
            specs, mission_duration_s=0.5, seed=42, noise_floor_db=-90.0,
        )
        s = gen.generate()
        # σ=0.5° → sample std ~0.5°
        assert s.aoa_deg.std() > 0.2

    def test_aoa_mean_matches_spec(self):
        """AoA mean is the spec's aoa_deg regardless of jitter."""
        specs = [_build_simple_spec(aoa_deg=33.0, aoa_jitter_deg=1.0)]
        gen = SyntheticEWPDWGenerator(
            specs, mission_duration_s=0.5, seed=42, noise_floor_db=-90.0,
        )
        s = gen.generate()
        # With N=500 pulses and σ=1°, mean is within ~0.1° of truth
        assert abs(s.aoa_deg.mean() - 33.0) < 0.5


# =====================================================================
# TestDeterminism
# =====================================================================


class TestDeterminism:
    """The generator must be reproducible given the same seed."""

    def test_same_seed_same_output(self):
        specs = [_build_simple_spec()]
        a = SyntheticEWPDWGenerator(
            specs, mission_duration_s=0.5, seed=42, noise_floor_db=-90.0,
        ).generate()
        b = SyntheticEWPDWGenerator(
            specs, mission_duration_s=0.5, seed=42, noise_floor_db=-90.0,
        ).generate()
        assert np.array_equal(a.toa_us, b.toa_us)
        assert np.array_equal(a.aoa_deg, b.aoa_deg)
        assert np.array_equal(a.pw_us, b.pw_us)

    def test_different_seed_different_output(self):
        # With aoa_jitter > 0, the per-pulse noise is RNG-driven
        specs = [_build_simple_spec(aoa_jitter_deg=1.0, pw_jitter_sec=0.05e-6)]
        a = SyntheticEWPDWGenerator(
            specs, mission_duration_s=0.5, seed=42, noise_floor_db=-90.0,
        ).generate()
        b = SyntheticEWPDWGenerator(
            specs, mission_duration_s=0.5, seed=43, noise_floor_db=-90.0,
        ).generate()
        # Different seeds → different per-pulse noise
        assert not np.array_equal(a.aoa_deg, b.aoa_deg)
        assert not np.array_equal(a.pw_us, b.pw_us)


# =====================================================================
# TestDeinterleaverRoundTrip
# =====================================================================


class TestDeinterleaverRoundTrip:
    """The deinterleaver must find the right number of tracks."""

    def test_two_emitter_scenario_finds_two_tracks(self):
        """default_two_emitter_scenario → 2 tracks, 100% attribution."""
        stream = default_two_emitter_scenario(
            mission_duration_s=1.0, seed=42,
        )
        result = quick_deinterleave(stream)
        assert result.n_tracks == 2
        assert result.pulse_attribution_rate == 1.0
        eids = sorted(t.dominant_emitter_id for t in result.tracks)
        assert eids == [0, 1]

    def test_six_emitter_scenario_finds_six_tracks(self):
        """default_six_emitter_scenario → 6 tracks (one per emitter type)."""
        stream = default_six_emitter_scenario(
            mission_duration_s=2.0, seed=99,
        )
        result = quick_deinterleave(stream)
        assert result.n_tracks == 6
        # 6 unique dominant emitter ids
        eids = sorted(t.dominant_emitter_id for t in result.tracks)
        assert eids == [0, 1, 2, 3, 4, 5]
        # All pulses attributed
        assert result.pulse_attribution_rate == 1.0


# =====================================================================
# TestEmissionValidation
# =====================================================================


class TestEmissionValidation:
    """Spec validation must reject invalid inputs."""

    def test_empty_specs_rejected(self):
        with pytest.raises(ValueError, match="at least one emitter"):
            SyntheticEWPDWGenerator([], mission_duration_s=1.0, seed=42)

    def test_duplicate_emitter_id_rejected(self):
        specs = [_build_simple_spec(0), _build_simple_spec(0)]
        with pytest.raises(ValueError, match="Duplicate emitter_id"):
            SyntheticEWPDWGenerator(
                specs, mission_duration_s=1.0, seed=42,
            )

    def test_pri_jitter_requires_jitter_fraction(self):
        spec = SyntheticEmitterSpec(
            emitter_id=0, aoa_deg=10.0, snr_db=15.0,
            emitter_type="pri_jitter",
            center_freq_hz=2.4e9, pri_sec=1e-3, pulse_width_sec=1e-6,
            jitter_fraction=0.0, seed_offset=0,
        )
        with pytest.raises(ValueError, match="jitter_fraction"):
            SyntheticEWPDWGenerator(
                [spec], mission_duration_s=1.0, seed=42,
            )

    def test_agile_requires_freq_list(self):
        spec = SyntheticEmitterSpec(
            emitter_id=0, aoa_deg=10.0, snr_db=15.0,
            emitter_type="frequency_agile",
            pri_sec=1e-3, pulse_width_sec=1e-6,
            seed_offset=0,
        )
        with pytest.raises(ValueError, match="freq_list_hz"):
            SyntheticEWPDWGenerator(
                [spec], mission_duration_s=1.0, seed=42,
            )

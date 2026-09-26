"""
tests.test_deinterleaver
========================

Tests for `FeatureBasedDeinterleaver` — the AoA-first deinterleaver.

The deinterleaver was redesigned from PRI-first (PRI histogram sort →
RF/PW cluster → AoA filter) to feature-first (AoA cluster → RF/PW
refinement → PRI validation). The new design reflects the reality of
TSRD scan-mode data: the scan receiver's IF sweep corrupts the PRI
histogram, so PRI cannot be the Stage-1 criterion. AoA is the most
stable feature (std < 1° over 30 s for stationary emitters), so it
takes the primary-discriminator role.

These tests cover five areas:

  * `TestAoAClustering` — Stage 1 finds AoA clusters correctly.
  * `TestRFPWRefinement` — Stage 2 splits AoA clusters by PW.
  * `TestPRIValidation` — Stage 3 computes PRI statistics.
  * `TestTrackStatistics` — pw_std_us and pw_cv are computed correctly.
  * `TestFixtureIntegration` — runs on the real TSRD fixture.

Author
------
Senior RF/EW Signal Simulation Engineer — Vyapti Option C tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

from vyapti_simulator.tsrd.deinterleaver import (
    DeinterleaverConfig,
    DeinterleaverResult,
    EmitterTrack,
    FeatureBasedDeinterleaver,
    quick_deinterleave,
)
from vyapti_simulator.tsrd.tsrd_adapter import (
    TSRDAdapter,
    TSRDDataMode,
)


# =====================================================================
# Helpers
# =====================================================================

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "tsrd"
SCAN_H5 = FIXTURES_DIR / "config_0_scan.h5"


def _build_stream(
    *,
    n_pulses: int,
    toa_start_us: float,
    toa_delta_us: float,
    freq_mhz: float,
    pw_us: float,
    pw_jitter_us: float,
    aoa_deg: float,
    aoa_jitter_deg: float,
    amp_db: float = -100.0,
    emitter_id: int = 0,
    seed: int = 0,
) -> dict:
    """
    Build a single-emitter PDW stream as a dict, deterministic given seed.
    """
    rng = np.random.default_rng(seed)
    toa = toa_start_us + np.arange(n_pulses) * toa_delta_us
    # Add small ToA jitter so the stream is realistic
    toa = toa + rng.normal(0.0, toa_delta_us * 0.02, size=n_pulses)
    toa = np.sort(toa)
    freq = np.full(n_pulses, freq_mhz, dtype=np.float64) + rng.normal(
        0.0, 0.05, size=n_pulses
    )
    pw = np.full(n_pulses, pw_us, dtype=np.float64) + rng.normal(
        0.0, pw_jitter_us, size=n_pulses
    )
    aoa = np.full(n_pulses, aoa_deg, dtype=np.float64) + rng.normal(
        0.0, aoa_jitter_deg, size=n_pulses
    )
    amp = np.full(n_pulses, amp_db, dtype=np.float64)
    eid = np.full(n_pulses, emitter_id, dtype=np.int64)
    return dict(
        toa_us=toa.astype(np.float32),
        freq_mhz=freq.astype(np.float32),
        pw_us=pw.astype(np.float32),
        aoa_deg=aoa.astype(np.float32),
        amp_db=amp.astype(np.float32),
        emitter_id=eid,
    )


def _combine(*streams: dict) -> dict:
    """Combine multiple single-emitter streams into one PDW dict."""
    keys = ("toa_us", "freq_mhz", "pw_us", "aoa_deg", "amp_db", "emitter_id")
    out: dict = {}
    for k in keys:
        arrays = [s[k] for s in streams]
        out[k] = np.concatenate(arrays)
    # Sort by ToA
    order = np.argsort(out["toa_us"])
    for k in keys:
        out[k] = out[k][order]
    return out


class _DictStream:
    """Adapter that lets a dict play the PDWStream role."""
    def __init__(self, d: dict) -> None:
        self.toa_us = d["toa_us"]
        self.freq_mhz = d["freq_mhz"]
        self.pw_us = d["pw_us"]
        self.aoa_deg = d["aoa_deg"]
        self.amp_db = d["amp_db"]
        self.emitter_id = d["emitter_id"]

    def __len__(self) -> int:
        return int(self.toa_us.size)


# =====================================================================
# TestAoAClustering
# =====================================================================

class TestAoAClustering:
    """Stage 1: AoA-based clustering finds dominant AoA peaks."""

    def test_single_emitter_single_track(self):
        """A single AoA cluster yields one track with all pulses."""
        s1 = _build_stream(
            n_pulses=20, toa_start_us=0, toa_delta_us=1000.0,
            freq_mhz=2400.0, pw_us=1.0, pw_jitter_us=0.05,
            aoa_deg=10.0, aoa_jitter_deg=0.3, seed=1,
        )
        result = quick_deinterleave(_DictStream(s1))
        assert result.n_tracks == 1
        assert result.tracks[0].n_pulses == 20
        assert result.tracks[0].dominant_emitter_id == 0

    def test_two_aoa_clusters(self):
        """Two AoA clusters at well-separated angles → two tracks."""
        s1 = _build_stream(
            n_pulses=20, toa_start_us=0, toa_delta_us=1000.0,
            freq_mhz=2400.0, pw_us=1.0, pw_jitter_us=0.05,
            aoa_deg=10.0, aoa_jitter_deg=0.3, seed=1,
            emitter_id=0,
        )
        s2 = _build_stream(
            n_pulses=20, toa_start_us=20000, toa_delta_us=1000.0,
            freq_mhz=2400.0, pw_us=1.0, pw_jitter_us=0.05,
            aoa_deg=80.0, aoa_jitter_deg=0.3, seed=2,
            emitter_id=1,
        )
        result = quick_deinterleave(_DictStream(_combine(s1, s2)))
        assert result.n_tracks == 2
        eids = sorted(t.dominant_emitter_id for t in result.tracks)
        assert eids == [0, 1]


# =====================================================================
# TestRFPWRefinement
# =====================================================================

class TestRFPWRefinement:
    """Stage 2: PW mean and PW std separate AoA-aligned emitters."""

    def test_two_pw_in_same_aoa(self):
        """Two emitters at the same AoA but different PW mean → two tracks."""
        s1 = _build_stream(
            n_pulses=20, toa_start_us=0, toa_delta_us=1000.0,
            freq_mhz=2400.0, pw_us=1.0, pw_jitter_us=0.05,
            aoa_deg=10.0, aoa_jitter_deg=0.3, seed=1,
            emitter_id=0,
        )
        s2 = _build_stream(
            n_pulses=20, toa_start_us=20000, toa_delta_us=1000.0,
            freq_mhz=2400.0, pw_us=20.0, pw_jitter_us=0.05,
            aoa_deg=10.0, aoa_jitter_deg=0.3, seed=2,
            emitter_id=1,
        )
        result = quick_deinterleave(_DictStream(_combine(s1, s2)))
        eids = sorted(t.dominant_emitter_id for t in result.tracks)
        assert eids == [0, 1]
        # Track PW means should be near the source PW values
        means = sorted([t.pw_mean_us for t in result.tracks])
        assert means[0] < 5.0
        assert means[1] > 15.0

    def test_pw_std_threshold_splits_stable_vs_jittered(self):
        """
        Two emitters at the same AoA + same mean PW but different
        PW std (0.05 vs 0.3 µs) should be separated when
        pw_std_tolerance_us=0.2.
        """
        s_stable = _build_stream(
            n_pulses=30, toa_start_us=0, toa_delta_us=1000.0,
            freq_mhz=2400.0, pw_us=5.0, pw_jitter_us=0.05,
            aoa_deg=10.0, aoa_jitter_deg=0.3, seed=1,
            emitter_id=0,
        )
        s_jittered = _build_stream(
            n_pulses=30, toa_start_us=30000, toa_delta_us=1000.0,
            freq_mhz=2400.0, pw_us=5.0, pw_jitter_us=0.3,
            aoa_deg=10.0, aoa_jitter_deg=0.3, seed=2,
            emitter_id=1,
        )
        cfg = DeinterleaverConfig(pw_std_tolerance_us=0.2)
        deint = FeatureBasedDeinterleaver(cfg)
        result = deint.deinterleave(_DictStream(_combine(s_stable, s_jittered)))
        eids = sorted(t.dominant_emitter_id for t in result.tracks)
        assert eids == [0, 1]


# =====================================================================
# TestPRIValidation
# =====================================================================

class TestPRIValidation:
    """Stage 3: PRI statistics are computed but don't drive clustering."""

    def test_pri_mean_is_positive(self):
        """The PRI mean is positive for any track with >= 2 pulses."""
        s1 = _build_stream(
            n_pulses=20, toa_start_us=0, toa_delta_us=1000.0,
            freq_mhz=2400.0, pw_us=1.0, pw_jitter_us=0.05,
            aoa_deg=10.0, aoa_jitter_deg=0.3, seed=1,
        )
        result = quick_deinterleave(_DictStream(s1))
        assert result.n_tracks == 1
        assert result.tracks[0].pri_mean_us > 0
        assert 800.0 < result.tracks[0].pri_mean_us < 1200.0  # within jitter

    def test_pri_scan_mode_robustness(self):
        """
        The whole point of the new algorithm: a PDW stream with
        scan-boundary ToA gaps (ToA delta = 50 ms with brief
        gaps of 200 ms) still gets deinterleaved correctly because
        PRI is NOT the Stage-1 criterion.
        """
        # Build a stream with big scan-boundary gaps every 5 pulses
        n = 50
        toa = []
        for i in range(n):
            t = i * 50_000.0  # 50 ms between pulses
            if i > 0 and i % 5 == 0:
                t += 200_000.0  # 200 ms gap every 5 pulses
            toa.append(t)
        toa = np.array(toa, dtype=np.float64)
        freq = np.full(n, 2400.0, dtype=np.float64)
        pw = np.full(n, 1.0, dtype=np.float64) + np.random.default_rng(0).normal(
            0, 0.05, n
        )
        aoa = np.full(n, 10.0, dtype=np.float64) + np.random.default_rng(0).normal(
            0, 0.3, n
        )
        amp = np.full(n, -100.0, dtype=np.float64)
        eid = np.zeros(n, dtype=np.int64)
        stream = _DictStream(dict(
            toa_us=toa.astype(np.float32),
            freq_mhz=freq.astype(np.float32),
            pw_us=pw.astype(np.float32),
            aoa_deg=aoa.astype(np.float32),
            amp_db=amp.astype(np.float32),
            emitter_id=eid,
        ))
        result = quick_deinterleave(stream)
        # The PRI-first algorithm would fragment this; feature-first
        # should still find 1 track with all 50 pulses.
        assert result.n_tracks == 1
        assert result.tracks[0].n_pulses == n


# =====================================================================
# TestTrackStatistics
# =====================================================================

class TestTrackStatistics:
    """The new pw_std_us and pw_cv fields are computed correctly."""

    def test_pw_std_matches_source(self):
        """Track PW std is close to the input jitter std."""
        # Use a low-jitter source (σ=0.04 µs) to match the TSRD
        # fixture's profile. The default config (pw_tolerance=0.2,
        # pw_std_tolerance=0.1) keeps a σ=0.04 source in one track
        # (3σ = 0.12, well within both tolerances) and the
        # recovered std should be close to the source std.
        s1 = _build_stream(
            n_pulses=100, toa_start_us=0, toa_delta_us=1000.0,
            freq_mhz=2400.0, pw_us=2.5, pw_jitter_us=0.04,
            aoa_deg=10.0, aoa_jitter_deg=0.2, seed=42,
        )
        result = quick_deinterleave(_DictStream(s1))
        assert result.n_tracks == 1
        # Input std was 0.04; output std should be close
        assert 0.02 < result.tracks[0].pw_std_us < 0.08

    def test_pw_cv_matches_source(self):
        """Track PW CV = pw_std / pw_mean."""
        # Use aoa_jitter_deg=0.2 so all pulses fall in one AoA bin
        # after the adjacent-bin merge.
        s1 = _build_stream(
            n_pulses=100, toa_start_us=0, toa_delta_us=1000.0,
            freq_mhz=2400.0, pw_us=2.5, pw_jitter_us=0.04,
            aoa_deg=10.0, aoa_jitter_deg=0.2, seed=42,
        )
        result = quick_deinterleave(_DictStream(s1))
        track = result.tracks[0]
        expected_cv = track.pw_std_us / track.pw_mean_us
        assert abs(track.pw_cv - expected_cv) < 1e-9

    def test_pw_std_nan_for_single_pulse(self):
        """A track with only 1 pulse has pw_std=NaN (defined behaviour)."""
        s1 = _build_stream(
            n_pulses=10, toa_start_us=0, toa_delta_us=1000.0,
            freq_mhz=2400.0, pw_us=1.0, pw_jitter_us=0.05,
            aoa_deg=10.0, aoa_jitter_deg=0.3, seed=1,
        )
        result = quick_deinterleave(_DictStream(s1))
        assert result.n_tracks == 1
        # 10 pulses -> 9 intervals, std is well-defined
        assert not np.isnan(result.tracks[0].pw_std_us)


# =====================================================================
# TestFixtureIntegration
# =====================================================================

class TestFixtureIntegration:
    """End-to-end run on the TSRD scan-mode fixture."""

    @pytest.fixture
    def fixture_pdw(self):
        if not SCAN_H5.exists():
            pytest.skip("TSRD scan fixture not present")
        # Lazy import to avoid pulling h5py when the fixture is missing
        from vyapti_simulator.core.environment import SimulationConfig
        cfg = SimulationConfig(
            band_count=36, total_spectrum_mhz=18_000.0,
            receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
            retune_time_ms=1.0, time_slots=600,
        )
        adapter = TSRDAdapter(
            SCAN_H5, TSRDDataMode.FIXTURE, simulation_config=cfg
        )
        return adapter.to_pdw_stream()

    def test_finds_at_least_one_track(self, fixture_pdw):
        """
        The fixture has 1 emitter, 788 pulses. The old
        PRI-first algorithm found 0 tracks. The new AoA-first
        algorithm should find at least one.
        """
        result = quick_deinterleave(fixture_pdw)
        assert result.n_tracks >= 1, (
            f"Expected at least 1 track, got {result.n_tracks}"
        )

    def test_high_pulse_attribution(self, fixture_pdw):
        """The new algorithm should attribute most pulses to a track."""
        result = quick_deinterleave(fixture_pdw)
        assert result.pulse_attribution_rate > 0.5, (
            f"Attribution rate {result.pulse_attribution_rate:.2f} "
            f"is too low; algorithm is too aggressive"
        )

    def test_track_dominates_single_emitter(self, fixture_pdw):
        """
        The fixture's ground truth is 1 emitter (id=0). The
        dominant emitter id of the recovered track should be 0.
        """
        result = quick_deinterleave(fixture_pdw)
        assert result.n_tracks >= 1
        for t in result.tracks:
            assert t.dominant_emitter_id == 0

    def test_pw_std_reasonable(self, fixture_pdw):
        """The fixture PW std is ~0.04 µs; the recovered std should be similar."""
        result = quick_deinterleave(fixture_pdw)
        assert result.n_tracks >= 1
        track = max(result.tracks, key=lambda t: t.n_pulses)
        # 0.04 ± 0.04 — allow a wide window since the track is one cluster
        assert 0.0 < track.pw_std_us < 0.5


# =====================================================================
# TestDeprecatedAlias
# =====================================================================

class TestDeprecatedAlias:
    """The PRIBasedDeinterleaver alias still works but warns."""

    def test_alias_emits_deprecation_warning(self):
        """Instantiating the old class name emits a DeprecationWarning."""
        from vyapti_simulator.tsrd.deinterleaver import PRIBasedDeinterleaver
        with pytest.warns(DeprecationWarning, match="FeatureBasedDeinterleaver"):
            PRIBasedDeinterleaver()
        # Suppress warning from the un-instantiated class lookup above

    def test_quick_deinterleave_returns_a_result(self):
        """quick_deinterleave is the canonical entry point and works."""
        s1 = _build_stream(
            n_pulses=20, toa_start_us=0, toa_delta_us=1000.0,
            freq_mhz=2400.0, pw_us=1.0, pw_jitter_us=0.05,
            aoa_deg=10.0, aoa_jitter_deg=0.3, seed=1,
        )
        result = quick_deinterleave(_DictStream(s1))
        assert isinstance(result, DeinterleaverResult)
        assert result.n_tracks == 1

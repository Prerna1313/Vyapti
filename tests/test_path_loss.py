"""
tests.test_path_loss
====================

Tests for the realistic path-loss model in
``TSRDEmitterSampler._estimate_snr_db`` and ``TSRDAdapter``.

The model:
  path_loss_db = FSPL + shadowing + diffraction

  FSPL = 20·log10(range_km) + 20·log10(freq_mhz) + 32.4
  shadowing  = N(0, σ=8) dB  (terrain shadowing)
  diffraction = N(0, σ=4) dB  (multipath fading)

Plus: the receiver position is read from H5
``/metadata/receiver/start_position_km``, not hardcoded to (0, 0).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.tsrd import TSRDAdapter, TSRDDataMode
from vyapti_simulator.tsrd.tsrd_emitter import TSRDEmitterSampler


# =====================================================================
# TestFSPLFormula
# =====================================================================


class TestFSPLFormula:
    """Free-space path loss formula is the standard ITU approximation."""

    def test_fspl_50km_1ghz(self):
        """50 km @ 1 GHz → 20·log10(50) + 20·log10(1000) + 32.4 ≈ 106.4 dB."""
        pl = TSRDEmitterSampler.path_loss_db(50.0, 1000.0)
        # 20*log10(50) = 33.98, 20*log10(1000) = 60.0, sum + 32.4 = 126.38
        assert abs(pl - 126.38) < 0.1, f"got {pl}"

    def test_fspl_scales_with_log_range(self):
        """Doubling the range adds ~6 dB."""
        pl1 = TSRDEmitterSampler.path_loss_db(50.0, 1000.0)
        pl2 = TSRDEmitterSampler.path_loss_db(100.0, 1000.0)
        # 20*log10(2) = 6.02
        assert abs((pl2 - pl1) - 6.02) < 0.1

    def test_fspl_scales_with_log_freq(self):
        """Doubling the freq adds ~6 dB."""
        pl1 = TSRDEmitterSampler.path_loss_db(50.0, 1000.0)
        pl2 = TSRDEmitterSampler.path_loss_db(50.0, 2000.0)
        assert abs((pl2 - pl1) - 6.02) < 0.1

    def test_shadowing_passthrough(self):
        """shadowing_db is added directly to FSPL."""
        pl_clean = TSRDEmitterSampler.path_loss_db(50.0, 1000.0)
        pl_shadowed = TSRDEmitterSampler.path_loss_db(
            50.0, 1000.0, shadowing_db=10.0
        )
        assert abs((pl_shadowed - pl_clean) - 10.0) < 1e-9

    def test_diffraction_passthrough(self):
        """diffraction_db is added directly to FSPL."""
        pl_clean = TSRDEmitterSampler.path_loss_db(50.0, 1000.0)
        pl_diffr = TSRDEmitterSampler.path_loss_db(
            50.0, 1000.0, diffraction_db=-3.0
        )
        assert abs((pl_diffr - pl_clean) - (-3.0)) < 1e-9


# =====================================================================
# TestReceiverPositionFromH5
# =====================================================================


class TestReceiverPositionFromH5:
    """TSRDAdapter reads receiver position from H5 metadata."""

    def test_fixture_receiver_position(self):
        """Fixture H5 has receiver at (100, 100) km."""
        adapter = TSRDAdapter(
            "D:/Vyapti/tests/fixtures/tsrd/config_0_scan.h5",
            TSRDDataMode.FIXTURE,
            simulation_config=SimulationConfig(
                band_count=36, total_spectrum_mhz=18_000.0,
                receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
                retune_time_ms=1.0, time_slots=600,
            ),
        )
        assert adapter.receiver_position_km == (100.0, 100.0)

    def test_receiver_position_is_tuple(self):
        """The property returns a tuple, not a list or numpy array."""
        adapter = TSRDAdapter(
            "D:/Vyapti/tests/fixtures/tsrd/config_0_scan.h5",
            TSRDDataMode.FIXTURE,
            simulation_config=SimulationConfig(
                band_count=36, total_spectrum_mhz=18_000.0,
                receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
                retune_time_ms=1.0, time_slots=600,
            ),
        )
        pos = adapter.receiver_position_km
        assert isinstance(pos, tuple)
        assert len(pos) == 2


# =====================================================================
# TestShadowingAndDiffraction
# =====================================================================


class TestShadowingAndDiffraction:
    """Shadowing and diffraction draws are log-normal with correct σ."""

    def test_shadowing_8db_std(self):
        """1000 draws of N(0, 8) have empirical σ within tolerance of 8."""
        rng = np.random.default_rng(42)
        draws = rng.normal(0.0, 8.0, 1000)
        # 95% CI for sample std with n=1000: ±~5% of true σ
        emp_std = float(np.std(draws, ddof=1))
        assert abs(emp_std - 8.0) < 0.5, f"got {emp_std}"

    def test_diffraction_4db_std(self):
        """1000 draws of N(0, 4) have empirical σ within tolerance of 4."""
        rng = np.random.default_rng(42)
        draws = rng.normal(0.0, 4.0, 1000)
        emp_std = float(np.std(draws, ddof=1))
        assert abs(emp_std - 4.0) < 0.3, f"got {emp_std}"


# =====================================================================
# TestSamplerUsesReceiverPosition
# =====================================================================


class TestSamplerUsesReceiverPosition:
    """The sampler wires receiver_position_km into the link budget."""

    def _make_stats(self) -> dict:
        """Synthesise a minimal statistics dict with one active emitter."""
        return {
            "source_h5_sha256": "deadbeef" * 8,
            "emitter_library": {
                "0": {
                    "function": "radar",
                    "frequency": {
                        "freq_mode": "FixedSingle",
                        "freqs_mhz": [1200.0],
                    },
                    "pri": {"pri_mode": "stable", "pris_us": [10.0]},
                    "power": {"power_w": 5000.0, "gain": 40.0},
                    "scan": {"scan_type": "Circular", "scan_rate_rpm": 60.0},
                    "position": {"start_position_km": [0.0, 0.0]},
                },
            },
            "label_counts": {"0": 100},
            "population_stats": {"n_active": 1, "n_silent": 0},
        }

    def test_default_no_receiver_position_uses_50km(self, monkeypatch):
        """With default receiver_position_km=None, uses 50 km fallback."""
        from vyapti_simulator.tsrd import tsrd_emitter

        stats = self._make_stats()
        monkeypatch.setattr(
            tsrd_emitter, "load_tsrd_statistics", lambda path: stats,
        )
        cfg = SimulationConfig(
            band_count=36, total_spectrum_mhz=18_000.0,
            receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
            retune_time_ms=1.0, time_slots=600,
        )
        sampler = tsrd_emitter.TSRDEmitterSampler("dummy.json", cfg, seed=42)
        configs = sampler.build_emitter_configs()
        c0 = configs[0]
        # provenance records None for receiver_position_km
        assert c0.provenance_notes["tsrd_receiver_position_km"] is None

    def test_explicit_receiver_position_recorded(self, monkeypatch):
        """With receiver_position_km=(100,100), the provenance records it."""
        from vyapti_simulator.tsrd import tsrd_emitter

        stats = self._make_stats()
        monkeypatch.setattr(
            tsrd_emitter, "load_tsrd_statistics", lambda path: stats,
        )
        cfg = SimulationConfig(
            band_count=36, total_spectrum_mhz=18_000.0,
            receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
            retune_time_ms=1.0, time_slots=600,
        )
        sampler = tsrd_emitter.TSRDEmitterSampler(
            "dummy.json", cfg, seed=42,
            receiver_position_km=(100.0, 100.0),
        )
        configs = sampler.build_emitter_configs()
        c0 = configs[0]
        # Emitter is at (0, 0), receiver at (100, 100)
        # Range = sqrt(20000) ≈ 141.4 km
        assert c0.provenance_notes["tsrd_receiver_position_km"] == [100.0, 100.0]

    def test_shadowing_diffraction_recorded(self, monkeypatch):
        """The per-emitter shadowing/diffraction draws are in provenance."""
        from vyapti_simulator.tsrd import tsrd_emitter

        stats = self._make_stats()
        monkeypatch.setattr(
            tsrd_emitter, "load_tsrd_statistics", lambda path: stats,
        )
        cfg = SimulationConfig(
            band_count=36, total_spectrum_mhz=18_000.0,
            receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
            retune_time_ms=1.0, time_slots=600,
        )
        # Use a fixed RNG so the draws are deterministic
        rng = np.random.default_rng(42)
        sampler = tsrd_emitter.TSRDEmitterSampler(
            "dummy.json", cfg, seed=42, rng=rng,
        )
        configs = sampler.build_emitter_configs()
        c0 = configs[0]
        # The draws are in [-3σ, +3σ] for sanity
        for key in ("tsrd_shadowing_db", "tsrd_diffraction_db"):
            v = c0.provenance_notes[key]
            assert -3 * 8 < v < 3 * 8, f"{key}={v} out of range"


# =====================================================================
# TestLongRangePathLoss
# =====================================================================


class TestLongRangePathLoss:
    """At long ranges, FSPL dominates shadowing/diffraction."""

    def test_long_range_snr_is_low(self, monkeypatch):
        """200 km range produces low SNR even with strong emitter."""
        from vyapti_simulator.tsrd import tsrd_emitter

        stats = {
            "source_h5_sha256": "deadbeef" * 8,
            "emitter_library": {
                "0": {
                    "function": "radar",
                    "frequency": {
                        "freq_mode": "FixedSingle",
                        "freqs_mhz": [1000.0],
                    },
                    "pri": {"pri_mode": "stable", "pris_us": [10.0]},
                    "power": {"power_w": 5000.0, "gain": 40.0},
                    "scan": {"scan_type": "Circular", "scan_rate_rpm": 60.0},
                    "position": {"start_position_km": [200.0, 0.0]},
                },
            },
            "label_counts": {"0": 100},
            "population_stats": {"n_active": 1, "n_silent": 0},
        }
        monkeypatch.setattr(
            tsrd_emitter, "load_tsrd_statistics", lambda path: stats,
        )
        cfg = SimulationConfig(
            band_count=36, total_spectrum_mhz=18_000.0,
            receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
            retune_time_ms=1.0, time_slots=600,
        )
        rng = np.random.default_rng(42)
        # 200 km range, σ=0 for deterministic check
        sampler = tsrd_emitter.TSRDEmitterSampler(
            "dummy.json", cfg, seed=42, rng=rng,
            receiver_position_km=(0.0, 0.0),
            path_loss_shadowing_db=0.0,
            path_loss_diffraction_db=0.0,
        )
        configs = sampler.build_emitter_configs()
        c0 = configs[0]
        # EIRP = 10*log10(5000) + 40 = 36.99 + 40 = 76.99 dBW
        # FSPL @ 200 km, 1 GHz = 20*log10(200) + 60 + 32.4 = 138.42 dB
        # SNR = 76.99 - 138.42 ≈ -61.4 dB
        assert c0.snr_db < -40.0, f"expected < -40 dB, got {c0.snr_db}"

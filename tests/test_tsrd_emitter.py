"""
tests.test_tsrd_emitter
========================

Unit tests for `TSRDEmitterSampler`.

The sampler is the locked plan's Phase-4 integration point. It must:
  1. Load the JSON through `load_tsrd_statistics` so the hash check fires.
  2. Produce non-empty, well-formed `EmitterConfig` lists.
  3. Be deterministic for a given seed (Gate 0 property).
  4. Drop straight into `VyaptiEnv.reset()` and produce a
     truth grid of the right shape.

To avoid touching the real TSRD artefact and its real manifest hash,
the fixture is a 4-emitter synthetic stats JSON written to a temp dir,
along with a manifest whose SHA-256 matches. The real JSON is
covered by V6/V8 of the plan.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from vyapti_simulator.core.environment import (
    EmitterBehaviorType,
    EmitterConfig,
    VyaptiEnv,
    SimulationConfig,
)
from vyapti_simulator.tsrd.tsrd_emitter import (
    FREQ_MODE_TO_BEHAVIOR,
    TSRDEmitterSampler,
)


# =====================================================================
# Fixtures: a tiny self-contained TSRD stats JSON + matching manifest
# =====================================================================
def _sha256_of_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _make_synthetic_stats_bytes() -> bytes:
    """
    A 4-emitter synthetic stats dict, JSON-serialised with sorted keys
    and indent=2 (matching the extractor's output). Three of the four
    emitters are "active" (label_count > 0); one is silent.
    """
    payload = {
        "dataset": "synthetic-fixture",
        "extractor_version": "test",
        "extraction_date_utc": "2026-01-01T00:00:00+00:00",
        "slot_ms": 50.0,
        "source_file": "fixture.h5",
        "source_h5_sha256": "0" * 64,
        "file_info": {
            "data_shape": [10, 5],
            "feature_names": ["f0"],
            "collection_time_s": 1.0,
            "num_pulses_reported": 10,
            "scan_mode": "fixture",
            "bandwidth_mhz": 500.0,
            "sensitivity_dbm": -90.0,
            "gain_db": 30.0,
            "freq_range_mhz": [0.0, 18000.0],
            "dwell_centres_mhz": [0.0],
            "dwell_times_s": [0.05],
        },
        "population_stats": {
            "n_active": 3,
            "n_silent": 1,
            "freq_mode_counts": {
                "FixedSingle": 2,
                "HoppingSawtooth": 1,
                "RandomRange": 0,
            },
            "pri_mode_counts": {"Fixed": 3},
            "scan_type_counts": {"Circular": 3},
            "period_slots": {"min": 10, "median": 30, "max": 120},
            "freq_range_mhz": {"min": 1000.0, "max": 15000.0},
            "snr_db": {"min": 10.0, "median": 20.0, "max": 30.0},
        },
        "emitter_library": {
            "0": {
                "function": "Radar-A",
                "frequency": {
                    "freq_mode": "FixedSingle",
                    "freqs_mhz": [3000.0],
                },
                "pri": {"pri_mode": "Fixed", "pris_us": [1000.0]},
                "pulse_width": {"pw_mode": "Fixed", "pws_us": [10.0]},
                "power": {"power_w": 100000.0, "gain": 40.0},
                "scan": {"scan_type": "Circular", "scan_rate_rpm": 12.0},
                "position": {"start_position_km": [0.0, 0.0]},
            },
            "1": {
                "function": "Radar-B",
                "frequency": {
                    "freq_mode": "FixedSingle",
                    "freqs_mhz": [9000.0, 9200.0],
                },
                "pri": {"pri_mode": "Fixed", "pris_us": [2000.0]},
                "pulse_width": {"pw_mode": "Fixed", "pws_us": [5.0]},
                "power": {"power_w": 50000.0, "gain": 35.0},
                "scan": {"scan_type": "Circular", "scan_rate_rpm": 6.0},
                "position": {"start_position_km": [0.0, 0.0]},
            },
            "2": {
                "function": "Radar-C",
                "frequency": {
                    "freq_mode": "HoppingSawtooth",
                    "freqs_mhz": [12000.0],
                },
                "pri": {"pri_mode": "Fixed", "pris_us": [1500.0]},
                "pulse_width": {"pw_mode": "Fixed", "pws_us": [8.0]},
                "power": {"power_w": 80000.0, "gain": 38.0},
                "scan": {"scan_type": "Circular", "scan_rate_rpm": 0.0},
                "position": {"start_position_km": [0.0, 0.0]},
            },
            # emitter 3: silent (no pulses), so it must NOT be returned.
            "3": {
                "function": "Silent-X",
                "frequency": {
                    "freq_mode": "FixedSingle",
                    "freqs_mhz": [100.0],
                },
                "pri": {"pri_mode": "Fixed", "pris_us": [500.0]},
                "pulse_width": {"pw_mode": "Fixed", "pws_us": [2.0]},
                "power": {"power_w": 1.0, "gain": 1.0},
                "scan": {"scan_type": "Circular", "scan_rate_rpm": 0.0},
                "position": {"start_position_km": [0.0, 0.0]},
            },
        },
        "label_counts": {"0": 100, "1": 80, "2": 50, "3": 0},
    }
    return json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")


def _write_fixture(tmp_path: Path) -> Path:
    """
    Write the synthetic stats JSON + a matching manifest under
    `tmp_path`. The loader's repo-root walk starts at the JSON's
    directory and walks upward looking for `data_provenance/manifest.json`.
    Putting the manifest in a `data_provenance` sibling of the JSON
    makes the walk hit it on the first step.
    """
    data_dir = tmp_path / "vyapti_simulator" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    stats_path = data_dir / "tsrd_statistics.json"
    body = _make_synthetic_stats_bytes()
    stats_path.write_bytes(body)

    prov_dir = tmp_path / "data_provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "1.0.0",
        "datasets": {
            "tsrd_statistics": {
                "sha256": _sha256_of_bytes(body),
                "path": "vyapti_simulator/data/tsrd_statistics.json",
                "provenance": "test fixture",
            }
        },
    }
    (prov_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    return stats_path


def _tsrd_grid_config() -> SimulationConfig:
    """
    The locked TSRD grid from the plan: 36 bands, 18 000 MHz, 50 ms
    slots, 600 slots (30 s mission).
    """
    return SimulationConfig(
        band_count=36,
        total_spectrum_mhz=18000.0,
        receiver_ibw_mhz=500.0,
        dwell_time_ms=50.0,
        retune_time_ms=1.0,
        time_slots=600,
        max_emitters=35,
        detection_probability=1.0,
        false_alarm_probability=0.0,
    )


# =====================================================================
# Tests
# =====================================================================
class TestTSRDEmitterSampler:

    def test_runs_without_raising(self, tmp_path: Path):
        """
        Smoke test: the sampler accepts a path + config + seed, hashes
        the JSON against the manifest, and returns a list of
        EmitterConfig without raising.
        """
        stats_path = _write_fixture(tmp_path)
        cfg = _tsrd_grid_config()
        sampler = TSRDEmitterSampler(stats_path, cfg, seed=7)
        configs = sampler.build_emitter_configs()
        assert isinstance(configs, list)
        assert len(configs) >= 1
        # Silent emitter (id=3) must NOT appear.
        ids = {c.emitter_id for c in configs}
        assert 3 not in ids
        assert ids.issubset({0, 1, 2})

    def test_determinism_same_seed(self, tmp_path: Path):
        """
        Two samplers built from the same fixture with the same seed
        must produce identical config lists (Gate 0).
        """
        stats_path = _write_fixture(tmp_path)
        cfg = _tsrd_grid_config()
        s1 = TSRDEmitterSampler(stats_path, cfg, seed=42)
        s2 = TSRDEmitterSampler(stats_path, cfg, seed=42)
        c1 = s1.build_emitter_configs()
        c2 = s2.build_emitter_configs()
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a.emitter_id == b.emitter_id
            assert a.behavior == b.behavior
            assert a.active_bands == b.active_bands
            assert a.period_slots == b.period_slots
            assert a.visibility_fraction == b.visibility_fraction
            assert a.arrival_slot == b.arrival_slot
            # SNR is jittered from a normal draw, so it must also
            # match (same seed, same draws, same jitter).
            assert a.snr_db == b.snr_db
            # Provenance is also a function of the seed, so it must match.
            assert a.provenance_notes == b.provenance_notes

    def test_configs_are_non_empty_and_well_formed(self, tmp_path: Path):
        """
        Every returned config has at least one active_band, a known
        behavior, a non-negative snr_db, a finite visibility, and an
        arrival_slot inside the mission window.
        """
        stats_path = _write_fixture(tmp_path)
        cfg = _tsrd_grid_config()
        sampler = TSRDEmitterSampler(stats_path, cfg, seed=123)
        configs = sampler.build_emitter_configs()
        for ec in configs:
            assert isinstance(ec, EmitterConfig)
            assert ec.behavior in EmitterBehaviorType
            assert len(ec.active_bands) >= 1
            assert all(0 <= b < cfg.band_count for b in ec.active_bands)
            assert ec.arrival_slot >= 0
            assert ec.arrival_slot < cfg.time_slots
            assert 0.0 < ec.visibility_fraction < 1.0
            assert math.isfinite(ec.snr_db)
            # Provenance must reference the TSRD source.
            assert "tsrd_source_tx_id" in ec.provenance_notes
            assert "tsrd_source_h5_sha256" in ec.provenance_notes
            assert (
                ec.provenance_notes["tsrd_provenance_label"]
                == "[TSRD-DERIVED]"
            )

    def test_truth_grid_shape_after_reset(self, tmp_path: Path):
        """
        Plug the sampled configs into a real `VyaptiEnv` and
        confirm the hidden truth grid has the locked TSRD shape
        `(n_emitters, 36, 600)`.
        """
        stats_path = _write_fixture(tmp_path)
        cfg = _tsrd_grid_config()
        sampler = TSRDEmitterSampler(stats_path, cfg, seed=99)
        configs = sampler.build_emitter_configs()

        env = VyaptiEnv(cfg, configs)
        env.reset(seed=99, emitter_family_config=configs)

        truth = env.hidden_truth
        n_emit = len(configs)
        assert truth.grid.shape == (n_emit, cfg.band_count, cfg.time_slots)
        assert truth.grid.dtype == bool
        # At least one emitter must illuminate at least one band/slot
        # — if not, the sampler handed us configs the truth generator
        # couldn't render, which would be a sampling bug.
        assert truth.grid.any()


# =====================================================================
# Direct mapping-table test (cheap, no fixtures)
# =====================================================================
class TestFreqModeToBehavior:
    def test_table_is_total_over_known_modes(self):
        # Every freq_mode produced by the extractor is mapped to a
        # known enum member; no silent fallthrough.
        known = {
            "FixedSingle",
            "FixedMultiSimultaneous",
            "HoppingLinear",
            "HoppingSawtooth",
            "RandomFixed",
            "RandomRange",
        }
        assert set(FREQ_MODE_TO_BEHAVIOR.keys()) == known
        for v in FREQ_MODE_TO_BEHAVIOR.values():
            assert v in EmitterBehaviorType

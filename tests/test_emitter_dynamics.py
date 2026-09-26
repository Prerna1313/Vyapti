"""
tests.test_emitter_dynamics
===========================

Tests for emitter lifecycle dynamics in the TSRD environment:
delayed arrival (emitters power up mid-mission) and the
per-emitter arrival_slot wiring from H5 metadata or PDW data.

These tests cover:
  * `TestAdapterStartTime` — `TSRDAdapter` reads `start_time_s`
    from H5 ``power_config`` and exposes it via property.
  * `TestInferArrivalSlotsFromPDW` — `TSRDAdapter.infer_arrival_slots_from_pdw()`
    correctly finds the first pulse of each emitter.
  * `TestAdapterInferArrivalSlotsFromH5` — H5 ``start_time_s`` → slot.
  * `TestEnvironmentArrivalSlots` — `TSRDEnvironment.emitter_arrival_slots`
    returns the inferred map.
  * `TestSamplerStartTime` — `TSRDEmitterSampler` reads
    ``power.start_time_s`` from the statistics JSON.
"""

from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.tsrd import (
    DetectionConfig,
    DeinterleaverConfig,
    PDWStream,
    TSRDAdapter,
    TSRDDataMode,
    TSRDEnvironment,
)


# =====================================================================
# TestAdapterStartTime
# =====================================================================


class TestAdapterStartTime:
    """TSRDAdapter reads start_time_s from H5 power_config."""

    def test_fixture_start_times_are_zero(self):
        """Fixture has no start_time_s in power_config → all 0.0."""
        adapter = TSRDAdapter(
            "D:/Vyapti/tests/fixtures/tsrd/config_0_scan.h5",
            TSRDDataMode.FIXTURE,
            simulation_config=SimulationConfig(
                band_count=36, total_spectrum_mhz=18_000.0,
                receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
                retune_time_ms=1.0, time_slots=600,
            ),
        )
        # Must call to_emitter_configs() to populate the start times
        # (they are read during H5 traversal in to_emitter_configs).
        adapter.to_emitter_configs(
            visibility_fraction=0.3,
            arrival_slot=0,
            snr_db=20.0,
        )
        assert adapter.transmitter_start_times_s == {
            "transmitters_0": 0.0,
            "transmitters_1": 0.0,
        }


# =====================================================================
# TestInferArrivalSlotsFromPDW
# =====================================================================


class TestInferArrivalSlotsFromPDW:
    """infer_arrival_slots_from_pdw() finds the first pulse per emitter."""

    def _make_stream(self, pulses: list[dict]) -> PDWStream:
        """Build a PDWStream from a list of {toa_us, emitter_id} dicts."""
        toa = np.array([p["toa_us"] for p in pulses], dtype=np.float32)
        eid = np.array([p["emitter_id"] for p in pulses], dtype=np.int64)
        return PDWStream(
            toa_us=toa,
            freq_mhz=np.full(len(pulses), 2400.0, dtype=np.float32),
            pw_us=np.full(len(pulses), 1.0, dtype=np.float32),
            aoa_deg=np.full(len(pulses), 10.0, dtype=np.float32),
            amp_db=np.full(len(pulses), -75.0, dtype=np.float32),
            emitter_id=eid,
        )

    def _make_adapter(self, stream: PDWStream) -> TSRDAdapter:
        cfg = SimulationConfig(
            band_count=4, total_spectrum_mhz=2000.0,
            receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
            retune_time_ms=1.0, time_slots=600,
        )
        # TSRDAdapter normally reads from H5; for this test we use
        # a private _from_pdw_stream classmethod we add for testing.
        adapter = TSRDAdapter.__new__(TSRDAdapter)
        # Manually set the required fields for infer_arrival_slots_from_pdw
        adapter._simulation_config = cfg
        adapter._data = np.column_stack([
            stream.toa_us.astype(np.float64),
            stream.freq_mhz.astype(np.float64),
            stream.pw_us.astype(np.float64),
            stream.aoa_deg.astype(np.float64),
            stream.amp_db.astype(np.float64),
        ])
        adapter._labels = stream.emitter_id.astype(np.int64)
        adapter._n_pulses = len(stream)
        return adapter

    def test_single_emitter_arrives_at_slot_0(self):
        """Emitter with first pulse at 0s → arrival_slot 0."""
        stream = self._make_stream([
            {"toa_us": 0.0, "emitter_id": 0},
            {"toa_us": 100.0, "emitter_id": 0},
        ])
        adapter = self._make_adapter(stream)
        result = adapter.infer_arrival_slots_from_pdw()
        assert result == {0: 0}

    def test_two_emitters_different_arrival_slots(self):
        """Emitter 0 arrives at slot 0; emitter 1 arrives at slot 200."""
        # slot_s = 0.05s, so emitter 0 at 0s → slot 0
        # emitter 1 at 10s → 10 / 0.05 = 200
        stream = self._make_stream([
            {"toa_us": 0.0, "emitter_id": 0},
            {"toa_us": 10_000_000.0, "emitter_id": 1},  # 10 s
        ])
        adapter = self._make_adapter(stream)
        result = adapter.infer_arrival_slots_from_pdw()
        assert result[0] == 0
        assert result[1] == 200

    def test_empty_stream_returns_empty_dict(self):
        """No pulses → empty dict."""
        adapter = self._make_adapter(self._make_stream([]))
        result = adapter.infer_arrival_slots_from_pdw()
        assert result == {}


# =====================================================================
# TestAdapterInferArrivalSlotsFromH5
# =====================================================================


class TestAdapterInferArrivalSlotsFromH5:
    """infer_arrival_slots_from_h5() converts start_time_s to slots."""

    def test_fixture_zero_start_times(self):
        """Fixture has no start_time_s → all arrival slots 0."""
        adapter = TSRDAdapter(
            "D:/Vyapti/tests/fixtures/tsrd/config_0_scan.h5",
            TSRDDataMode.FIXTURE,
            simulation_config=SimulationConfig(
                band_count=36, total_spectrum_mhz=18_000.0,
                receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
                retune_time_ms=1.0, time_slots=600,
            ),
        )
        # Populate start times (requires to_emitter_configs call)
        adapter.to_emitter_configs(
            visibility_fraction=0.3, arrival_slot=0, snr_db=20.0,
        )
        result = adapter.infer_arrival_slots_from_h5()
        # Both start at 0.0 s → slot 0
        assert result == {0: 0, 1: 0}


# =====================================================================
# TestEnvironmentArrivalSlots
# =====================================================================


class TestEnvironmentArrivalSlots:
    """TSRDEnvironment.emitter_arrival_slots returns the inferred map."""

    def _make_stream(self, pulses: list[dict]) -> PDWStream:
        toa = np.array([p["toa_us"] for p in pulses], dtype=np.float32)
        eid = np.array([p["emitter_id"] for p in pulses], dtype=np.int64)
        return PDWStream(
            toa_us=toa,
            freq_mhz=np.full(len(pulses), 2400.0, dtype=np.float32),
            pw_us=np.full(len(pulses), 1.0, dtype=np.float32),
            aoa_deg=np.full(len(pulses), 10.0, dtype=np.float32),
            amp_db=np.full(len(pulses), -75.0, dtype=np.float32),
            emitter_id=eid,
        )

    def test_delayed_arrival_from_first_pulse(self):
        """Emitters arriving mid-mission have correct arrival_slot in property."""
        # slot_s = 0.05s
        # emitter 0 first pulse at 0s → slot 0
        # emitter 1 first pulse at 5s → 5 / 0.05 = 100
        stream = self._make_stream([
            {"toa_us": 0.0, "emitter_id": 0},
            {"toa_us": 5_000_000.0, "emitter_id": 1},  # 5s = slot 100
            {"toa_us": 5_000_100.0, "emitter_id": 1},  # second pulse, 100 µs later
        ])
        sim_cfg = SimulationConfig(
            band_count=4, total_spectrum_mhz=2000.0,
            receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
            retune_time_ms=1.0, time_slots=600,
        )
        env = TSRDEnvironment(
            stream, sim_cfg, DetectionConfig(),
            DeinterleaverConfig(), seed=42,
        )
        env.reset(seed=42)
        arrivals = env.emitter_arrival_slots()
        assert arrivals[0] == 0
        assert arrivals[1] == 100

    def test_property_cached_between_steps(self):
        """The property caches and returns the same dict on repeated calls."""
        stream = self._make_stream([
            {"toa_us": 0.0, "emitter_id": 0},
        ])
        sim_cfg = SimulationConfig(
            band_count=4, total_spectrum_mhz=2000.0,
            receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
            retune_time_ms=1.0, time_slots=600,
        )
        env = TSRDEnvironment(
            stream, sim_cfg, DetectionConfig(),
            DeinterleaverConfig(), seed=42,
        )
        env.reset(seed=42)
        r1 = env.emitter_arrival_slots()
        env.step(3)  # step without changing the PDW
        r2 = env.emitter_arrival_slots()
        assert r1 is r2  # same dict object
        assert r1 == {0: 0}

    def test_arrival_slots_cleared_on_reset(self):
        """reset() clears the cache so dynamics are re-inferred."""
        stream = self._make_stream([
            {"toa_us": 0.0, "emitter_id": 0},
        ])
        sim_cfg = SimulationConfig(
            band_count=4, total_spectrum_mhz=2000.0,
            receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
            retune_time_ms=1.0, time_slots=600,
        )
        env = TSRDEnvironment(
            stream, sim_cfg, DetectionConfig(),
            DeinterleaverConfig(), seed=42,
        )
        env.reset(seed=42)
        _ = env.emitter_arrival_slots()
        assert env._emitter_arrival_slots is not None
        env.reset(seed=99)
        assert env._emitter_arrival_slots is None


# =====================================================================
# TestSamplerStartTime
# =====================================================================


class TestSamplerStartTime:
    """TSRDEmitterSampler reads start_time_s from statistics JSON."""

    def test_sampler_arrival_slot_prefers_h5_start_time(self, monkeypatch):
        """When the entry's power.start_time_s > 0, arrival_slot uses it.

        We mock load_tsrd_statistics to return a synthetic dict with
        start_time_s = 5.0 so the test doesn't depend on fixture files.
        """
        from vyapti_simulator.tsrd import tsrd_emitter

        fake_stats = {
            "source_h5_sha256": "deadbeef" * 8,
            "emitter_library": {
                "0": {
                    "function": "radar",
                    "frequency": {
                        "freq_mode": "FixedSingle",
                        "freqs_mhz": [1200.0],
                    },
                    "pri": {"pri_mode": "stable", "pris_us": [10.0]},
                    "power": {"power_w": 1000.0, "gain": 30.0,
                              "start_time_s": 5.0},
                    "scan": {"scan_type": "Circular", "scan_rate_rpm": 60.0},
                    "position": {"start_position_km": [0.0, 0.0]},
                },
            },
            "label_counts": {"0": 100},
            "population_stats": {"n_active": 1, "n_silent": 0},
        }
        monkeypatch.setattr(
            tsrd_emitter, "load_tsrd_statistics", lambda path: fake_stats,
        )
        cfg = SimulationConfig(
            band_count=36, total_spectrum_mhz=18_000.0,
            receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
            retune_time_ms=1.0, time_slots=600,
        )
        sampler = tsrd_emitter.TSRDEmitterSampler(
            "dummy_path.json", cfg, seed=42,
        )
        configs = sampler.build_emitter_configs()
        assert len(configs) == 1
        c0 = configs[0]
        # 5s / 0.05s per slot = 100
        assert c0.arrival_slot == 100, f"got {c0.arrival_slot}"
        assert c0.provenance_notes.get("tsrd_start_time_s") == 5.0

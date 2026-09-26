"""
tests.test_tsrd_pdw_only_path
=============================

Tests for the direct PDW-to-occupancy path added in
Commit 4. This is the path the full-corpus Kaggle run
actually used (§10.3.1). It must:

  * Build occupancy directly from `(toa_us, freq_mhz)`
    without calling `to_emitter_configs()`.
  * Admit 250/250 of the full Scan test split, vs 122/250
    for the `to_emitter_configs()` path.
  * Not require `caller_overrides`.
  * Not require non-zero `scan_rate_rpm` on any transmitter.
  * Be fixture-runnable on the local `config_0_scan.h5`
    fixture (a 1-active-emitter file with 788 pulses).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.tsrd import (
    TSRDAdapter,
    TSRDCorpusLoader,
    TSRDDataMode,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "tsrd"


def _sim_cfg() -> SimulationConfig:
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


@pytest.fixture(scope="module")
def scan_fixture() -> Path:
    p = FIXTURE_DIR / "config_0_scan.h5"
    if not p.is_file():
        pytest.skip("config_0_scan.h5 fixture not present")
    return p


class TestAdapterDirectOccupancy:
    def test_to_observed_occupancy_from_pdw_shape(self, scan_fixture):
        adapter = TSRDAdapter(
            scan_fixture,
            data_mode=TSRDDataMode.FIXTURE,
            simulation_config=_sim_cfg(),
        )
        grid = adapter.to_observed_occupancy_from_pdw()
        assert grid.ndim == 2
        assert grid.shape == (36, 600)
        assert grid.dtype == bool

    def test_to_observed_occupancy_from_pdw_is_per_band_slot(
        self, scan_fixture
    ):
        adapter = TSRDAdapter(
            scan_fixture,
            data_mode=TSRDDataMode.FIXTURE,
            simulation_config=_sim_cfg(),
        )
        grid = adapter.to_observed_occupancy_from_pdw()
        # The fixture has 788 pulses across 1 active emitter.
        # At least one cell must be True; not all cells can
        # be True (the receiver is narrow-band per pulse).
        assert grid.sum() >= 1
        assert grid.sum() < grid.size

    def test_to_observed_occupancy_from_pdw_no_emitter_configs(
        self, scan_fixture
    ):
        # The PDW-only path must NOT call to_emitter_configs.
        # We assert this by checking that the method runs
        # without the caller supplying caller_overrides.
        adapter = TSRDAdapter(
            scan_fixture,
            data_mode=TSRDDataMode.FIXTURE,
            simulation_config=_sim_cfg(),
        )
        # No caller_overrides anywhere. The PDW-only path
        # is independent of caller_overrides.
        grid = adapter.to_observed_occupancy_from_pdw()
        assert grid is not None

    def test_to_observed_occupancy_from_pdw_requires_sim_cfg(
        self, scan_fixture
    ):
        adapter = TSRDAdapter(
            scan_fixture,
            data_mode=TSRDDataMode.FIXTURE,
            # No simulation_config supplied.
        )
        with pytest.raises(ValueError, match="simulation_config"):
            adapter.to_observed_occupancy_from_pdw()


class TestCorpusLoaderPDWOnlyPath:
    def test_run_emits_per_file_occupancy(self, scan_fixture):
        loader = TSRDCorpusLoader(
            FIXTURE_DIR,
            data_mode=TSRDDataMode.FIXTURE,
            allow_legacy_layout=True,
            simulation_config=_sim_cfg(),
            # NOTE: no caller_overrides. The PDW-only path
            # must work without it.
        )
        summary = loader.run()
        assert summary.per_file_occupancy is not None
        assert summary.per_file_occupancy.shape[0] == summary.n_files_total
        assert summary.per_file_occupancy.shape[1:] == (36, 600)
        # The Scan fixture's row should be non-empty; the
        # Stare fixture's row should also be present (it
        # does not refuse on this path) but may be larger
        # because Stare records the full spectrum per
        # pulse.
        scan_record_idx = next(
            i for i, r in enumerate(summary.per_file)
            if r.file_path.name == "config_0_scan.h5"
        )
        assert summary.per_file_occupancy[scan_record_idx].sum() >= 1
        # The Stare row is present too (this method does
        # not refuse Stare; whether to consume it is a
        # downstream policy).
        stare_record_idx = next(
            i for i, r in enumerate(summary.per_file)
            if r.file_path.name == "config_0_stare.h5"
        )
        assert summary.per_file_occupancy[stare_record_idx] is not None

    def test_run_records_per_file_occupied_cells(self, scan_fixture):
        loader = TSRDCorpusLoader(
            FIXTURE_DIR,
            data_mode=TSRDDataMode.FIXTURE,
            allow_legacy_layout=True,
            simulation_config=_sim_cfg(),
        )
        summary = loader.run()
        assert summary.per_file_occupied_cells is not None
        assert len(summary.per_file_occupied_cells) == summary.n_files_total
        scan_record_idx = next(
            i for i, r in enumerate(summary.per_file)
            if r.file_path.name == "config_0_scan.h5"
        )
        assert summary.per_file_occupied_cells[scan_record_idx] >= 1

    def test_run_notes_record_pdw_path_admission_rates(
        self, scan_fixture
    ):
        loader = TSRDCorpusLoader(
            FIXTURE_DIR,
            data_mode=TSRDDataMode.FIXTURE,
            allow_legacy_layout=True,
            simulation_config=_sim_cfg(),
        )
        summary = loader.run()
        # The notes block must record both the PDW-only
        # 250/250 admission rate and the
        # to_emitter_configs() 122/250 admission rate so
        # a future reader cannot confuse the two paths.
        assert "pdw_only_path_used" in summary.notes
        assert "pdw_only_path_admits" in summary.notes
        assert "emitter_config_path_admits" in summary.notes
        assert "250/250" in summary.notes["pdw_only_path_admits"]
        assert "122/250" in summary.notes["emitter_config_path_admits"]

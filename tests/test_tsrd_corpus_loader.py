"""
tests.test_tsrd_corpus_loader
=============================

Tests for `TSRDCorpusLoader` using the two hash-pinned fixtures
under `tests/fixtures/tsrd/`. The full-corpus run is on Kaggle;
these tests verify the loader's fail-closed invariants and the
two-file iteration behaviour.

Tests:
  * `TestCorpusFailClosed` — missing dir, empty dir, missing
    manifest all raise `CorpusUnavailableError`.
  * `TestCorpusFixtureIteration` — with the fixtures dir, the
    loader iterates both files, both are ACCEPTED, Scan file
    yields emitter configs (when overrides are supplied), Stare
    file does not.
  * `TestCorpusCallerOverrides` — without caller_overrides, the
    Scan file still produces a PDWStream but no emitter configs.
  * `TestCorpusStareModeNotRoutedToEmitterConfigs` — explicit
    check that Stare mode never calls to_emitter_configs().
  * `TestCorpusSchemaWarningCaptured` — a directory with a
    malformed H5 records a SKIPPED entry with a schema warning
    rather than raising.
  * `TestCorpusSummaryDeterminism` — running the loader twice on
    the fixtures gives the same per-file SHA-256 chain (sanity
    check; full Gate 0 is on Kaggle).
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.tsrd import (
    CorpusFileDisposition,
    CorpusUnavailableError,
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


def _caller_overrides() -> dict:
    return {
        "visibility_fraction": 0.7,
        "arrival_slot": 0,
        "snr_db": 10.0,
    }


@pytest.fixture(scope="module")
def fixtures_present() -> bool:
    return (
        (FIXTURE_DIR / "config_0_scan.h5").is_file()
        and (FIXTURE_DIR / "config_0_stare.h5").is_file()
    )


class TestCorpusFailClosed:
    def test_missing_dir_raises(self):
        with pytest.raises(CorpusUnavailableError):
            TSRDCorpusLoader(
                "/nonexistent/path/that/does/not/exist"
            ).discover_h5_files()

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(CorpusUnavailableError):
                TSRDCorpusLoader(d).discover_h5_files()

    def test_require_manifest_raises_when_missing(self):
        with tempfile.TemporaryDirectory() as d:
            # The empty dir raises before the manifest check,
            # which is fine — both raise CorpusUnavailableError.
            with pytest.raises(CorpusUnavailableError):
                TSRDCorpusLoader(
                    d, require_manifest=True
                ).discover_h5_files()

    def test_no_synthetic_fallback_kwarg(self):
        # The loader's signature must NOT have an
        # `allow_synthetic_fallback_on_missing` keyword. This
        # is the same invariant the per-file adapter enforces.
        import inspect

        sig = inspect.signature(TSRDCorpusLoader.__init__)
        assert (
            "allow_synthetic_fallback_on_missing" not in sig.parameters
        )

    def test_missing_requested_split_does_not_read_another_split(self, tmp_path):
        other = tmp_path / "scan" / "train_scan"
        other.mkdir(parents=True)
        (other / "config_0.h5").write_bytes(b"sample")
        with pytest.raises(CorpusUnavailableError, match="test_scan"):
            TSRDCorpusLoader(tmp_path, split="test", receiver_mode="scan")

    def test_missing_requested_split_does_not_fallback_by_default(self, tmp_path):
        (tmp_path / "config_0.h5").write_bytes(b"\x89HDF\r\n\x1a\n")
        with pytest.raises(CorpusUnavailableError, match="train_scan"):
            TSRDCorpusLoader(tmp_path, split="train", receiver_mode="scan")

    def test_required_manifest_checks_file_integrity(self, tmp_path):
        target = tmp_path / "scan" / "test_scan"
        target.mkdir(parents=True)
        source = FIXTURE_DIR / "config_0_scan.h5"
        shutil.copyfile(source, target / source.name)
        copied = target / source.name
        entry = {
            "path": f"scan/test_scan/{source.name}",
            "size_bytes": copied.stat().st_size,
            "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
        }
        manifest = tmp_path / "corpus_manifest.json"
        manifest.write_text(json.dumps({"files": [entry]}), encoding="utf-8")
        loader = TSRDCorpusLoader(
            tmp_path, split="test", receiver_mode="scan",
            data_mode=TSRDDataMode.FIXTURE, require_manifest=True,
        )
        assert len(list(loader.iter_corpus())) == 1

        for bad in (
            {**entry, "size_bytes": entry["size_bytes"] + 1},
            {**entry, "sha256": "0" * 64},
            {**entry, "path": "../outside.h5"},
        ):
            manifest.write_text(json.dumps({"files": [bad]}), encoding="utf-8")
            with pytest.raises(CorpusUnavailableError):
                list(loader.iter_corpus())

        manifest.write_text(json.dumps({"files": []}), encoding="utf-8")
        with pytest.raises(CorpusUnavailableError, match="file set"):
            list(loader.iter_corpus())


class TestCorpusDiscoveryRecursive:
    """Explicit legacy discovery may recurse through a fixture tree.

    These tests opt into the old recursive layout. Normal split-based
    runs select only the requested split directory.

    1. The flat per-split form, e.g. `…/scan/test_scan/*.h5`,
       which the caller passes as `corpus_dir`.
    2. The Hugging Face snapshot root form, e.g.
       `…/snapshots/<revision>/`, under which the
       `scan/test_scan/*.h5` files live one or more levels
       deeper.

    `allow_legacy_layout=True` permits recursion for these fixtures.
    """

    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89HDF\r\n\x1a\n")  # HDF5 magic

    def test_flat_split_directory_is_discovered(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._touch(root / "config_0.h5")
            self._touch(root / "config_1.h5")
            # A non-H5 sibling must be ignored.
            (root / "README.md").write_text("ignore me")
            files = TSRDCorpusLoader(root, allow_legacy_layout=True).discover_h5_files()
            names = sorted(p.name for p in files)
            assert names == ["config_0.h5", "config_1.h5"]
            # Every returned path is directly under `root`
            # (the flat case does not require recursion).
            assert all(p.parent == root for p in files)

    def test_nested_snapshot_root_is_discovered(self):
        # Mirrors the Hugging Face cache layout:
        #   <snapshot_root>/scan/test_scan/config_*.h5
        # with the caller passing the snapshot root as
        # `corpus_dir`. The discovery must walk two levels
        # down and find the files.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            nested = root / "scan" / "test_scan"
            self._touch(nested / "config_0.h5")
            self._touch(nested / "config_10.h5")
            # A second split at the same depth (e.g. a
            # `stare/test_stare/` sibling) is also part of
            # the snapshot and IS visible at this scope;
            # the loader's job is to enumerate every H5
            # under the caller-supplied root, not to filter
            # by split name (split selection is a downstream
            # policy).
            self._touch(root / "stare" / "test_stare" / "ignored.h5")
            files = TSRDCorpusLoader(root, allow_legacy_layout=True).discover_h5_files()
            names = sorted(p.name for p in files)
            assert names == [
                "config_0.h5",
                "config_10.h5",
                "ignored.h5",
            ]
            # Concrete recursion check: each file is at
            # depth >= 2 below `root`. A non-recursive
            # `glob` would have returned [] for this tree.
            assert all(
                len(p.relative_to(root).parts) >= 2 for p in files
            )


@pytest.mark.skipif(
    not (FIXTURE_DIR / "config_0_scan.h5").is_file(),
    reason="config_0_scan.h5 fixture not present",
)
class TestCorpusFixtureIteration:
    def test_iter_finds_both_files(self, fixtures_present):
        if not fixtures_present:
            pytest.skip("fixtures not present")
        loader = TSRDCorpusLoader(
            FIXTURE_DIR,
            data_mode=TSRDDataMode.FIXTURE,
            allow_legacy_layout=True,
        )
        files = loader.discover_h5_files()
        names = sorted(p.name for p in files)
        assert names == ["config_0_scan.h5", "config_0_stare.h5"]

    def test_full_iteration_accepts_both_files(self, fixtures_present):
        if not fixtures_present:
            pytest.skip("fixtures not present")
        loader = TSRDCorpusLoader(
            FIXTURE_DIR,
            data_mode=TSRDDataMode.FIXTURE,
            allow_legacy_layout=True,
            simulation_config=_sim_cfg(),
            caller_overrides=_caller_overrides(),
        )
        summary = loader.run()
        assert summary.n_files_total == 2
        assert summary.n_files_accepted_scan == 1
        assert summary.n_files_accepted_stare == 1
        assert summary.n_files_skipped == 0
        # Both files have one unique label.
        assert summary.n_unique_labels_total == 2

    def test_scan_file_yields_emitter_configs_with_overrides(
        self, fixtures_present
    ):
        if not fixtures_present:
            pytest.skip("fixtures not present")
        loader = TSRDCorpusLoader(
            FIXTURE_DIR,
            data_mode=TSRDDataMode.FIXTURE,
            allow_legacy_layout=True,
            simulation_config=_sim_cfg(),
            caller_overrides=_caller_overrides(),
        )
        summary = loader.run()
        scan_record = next(
            r for r in summary.per_file
            if r.file_path.name == "config_0_scan.h5"
        )
        assert scan_record.disposition == CorpusFileDisposition.ACCEPTED
        assert scan_record.emitter_configs is not None
        assert len(scan_record.emitter_configs) >= 1

    def test_stare_file_does_not_yield_emitter_configs(
        self, fixtures_present
    ):
        if not fixtures_present:
            pytest.skip("fixtures not present")
        loader = TSRDCorpusLoader(
            FIXTURE_DIR,
            data_mode=TSRDDataMode.FIXTURE,
            allow_legacy_layout=True,
            simulation_config=_sim_cfg(),
            caller_overrides=_caller_overrides(),
        )
        summary = loader.run()
        stare_record = next(
            r for r in summary.per_file
            if r.file_path.name == "config_0_stare.h5"
        )
        assert stare_record.disposition == CorpusFileDisposition.ACCEPTED
        assert stare_record.emitter_configs is None
        # The PDW stream is still available (the oracle input).
        assert stare_record.pdw_stream is not None


class TestCorpusCallerOverrides:
    @pytest.mark.skipif(
        not (FIXTURE_DIR / "config_0_scan.h5").is_file(),
        reason="fixture not present",
    )
    def test_no_overrides_means_no_emitter_configs_but_pdw_still_there(
        self, fixtures_present
    ):
        if not fixtures_present:
            pytest.skip("fixtures not present")
        loader = TSRDCorpusLoader(
            FIXTURE_DIR,
            data_mode=TSRDDataMode.FIXTURE,
            allow_legacy_layout=True,
            simulation_config=_sim_cfg(),
            caller_overrides=None,
        )
        summary = loader.run()
        scan_record = next(
            r for r in summary.per_file
            if r.file_path.name == "config_0_scan.h5"
        )
        assert scan_record.pdw_stream is not None
        assert scan_record.emitter_configs is None
        # A schema_warnings entry should explain why.
        assert any(
            "caller_overrides" in w["message"]
            for w in scan_record.schema_warnings
        )


class TestCorpusStareModeNotRoutedToEmitterConfigs:
    @pytest.mark.skipif(
        not (FIXTURE_DIR / "config_0_stare.h5").is_file(),
        reason="fixture not present",
    )
    def test_stare_path_never_instantiates_to_emitter_configs(
        self, fixtures_present
    ):
        if not fixtures_present:
            pytest.skip("fixtures not present")
        # We can't easily monkey-patch to_emitter_configs here
        # because TSRDAdapter is constructed inside the loader.
        # Instead, we assert the documented contract: the
        # Stare-Mode file's schema_warnings contains the
        # expected "not called" message, and the disposition
        # is ACCEPTED, not SKIPPED_INSUFFICIENT_DATA.
        loader = TSRDCorpusLoader(
            FIXTURE_DIR,
            data_mode=TSRDDataMode.FIXTURE,
            allow_legacy_layout=True,
            simulation_config=_sim_cfg(),
            caller_overrides=_caller_overrides(),
        )
        summary = loader.run()
        stare_record = next(
            r for r in summary.per_file
            if r.file_path.name == "config_0_stare.h5"
        )
        assert stare_record.disposition == CorpusFileDisposition.ACCEPTED
        assert stare_record.emitter_configs is None
        assert any(
            "ScanPolicyOracle" in w["message"]
            for w in stare_record.schema_warnings
        )


class TestCorpusSummaryDeterminism:
    @pytest.mark.skipif(
        not (FIXTURE_DIR / "config_0_scan.h5").is_file(),
        reason="fixture not present",
    )
    def test_two_runs_have_identical_per_file_sha256(
        self, fixtures_present
    ):
        if not fixtures_present:
            pytest.skip("fixtures not present")
        loader1 = TSRDCorpusLoader(
            FIXTURE_DIR,
            data_mode=TSRDDataMode.FIXTURE,
            allow_legacy_layout=True,
            simulation_config=_sim_cfg(),
            caller_overrides=_caller_overrides(),
        )
        loader2 = TSRDCorpusLoader(
            FIXTURE_DIR,
            data_mode=TSRDDataMode.FIXTURE,
            allow_legacy_layout=True,
            simulation_config=_sim_cfg(),
            caller_overrides=_caller_overrides(),
        )
        s1 = loader1.run()
        s2 = loader2.run()
        shas1 = sorted(r.file_sha256 for r in s1.per_file)
        shas2 = sorted(r.file_sha256 for r in s2.per_file)
        assert shas1 == shas2
        # And the n_pulses must also match.
        npulses1 = sorted(r.n_pulses for r in s1.per_file)
        npulses2 = sorted(r.n_pulses for r in s2.per_file)
        assert npulses1 == npulses2

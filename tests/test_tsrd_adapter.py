"""
tests.test_tsrd_adapter
=======================

Tests for `TSRDAdapter`. Twenty-two tests in five classes:

  * `TestFixtureHash` (2)         -- chain of custody.
  * `TestDataModes` (3)           -- real_tsrd / fixture /
                                     synthetic builder.
  * `TestReceiverMode` (2)        -- Scan / Stare discovery;
                                     Stare rejects grid.
  * `TestPDWStream` (3)           -- 6 canonical keys; dtypes;
                                     values preserved.
  * `TestObservedOccupancy` (3)   -- shape; only observed
                                     cells True; docstring
                                     guard.
  * `TestEmitterConfigAtomicity` (3) -- Q5 no-partial-list
                                     rule; caller overrides
                                     marked as assumptions.
  * `TestSchema` (6)              -- your required schema
                                     validations.

Tests that need a real H5 fixture skip cleanly when the
fixture is not present. The hard-fail test
(`test_real_tsrd_mode_raises_when_file_missing`) and the
synthetic-builder test do not need a fixture.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pytest

from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.tsrd import (
    EXPECTED_H5_FEATURE_NAMES,
    PDW_STREAM_FIELDS,
    FREQ_MODE_TO_BEHAVIOR,
    TSRDAdapter,
    TSRDDataMode,
    TSRDReceiverMode,
    DataIntegrityError,
    DataUnavailableError,
    InsufficientDataError,
    StareModeOracleError,
    UnknownTSRDFieldError,
)
from tests._builders.synthetic_pdw import (
    PDWStream,
    SyntheticPDWBuilder,
)


# =====================================================================
# Paths
# =====================================================================
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "tsrd"
FIXTURE_MANIFEST = FIXTURES_DIR / "manifest.json"
SCAN_H5 = FIXTURES_DIR / "config_0_scan.h5"
STARE_H5 = FIXTURES_DIR / "config_0_stare.h5"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _tsrd_grid_config() -> SimulationConfig:
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


def _write_minimal_h5(
    path: Path,
    *,
    receiver_mode: str,
    n_pulses: int,
    n_emitters: int,
    freq_modes: list,
    freqs_mhz_per_emitter: list,
    scan_rates_rpm: list,
) -> None:
    """
    Write a minimal TSRD-shaped H5 to `path` so the adapter
    can be exercised without the real fixtures. The schema
    matches the canonical TSRD: 5 feature columns, integer
    labels, /metadata/transmitters/transmitters_X groups.
    """
    rng = np.random.default_rng(0)
    labels = rng.integers(0, n_emitters, size=n_pulses).astype(np.int64)
    data = np.zeros((n_pulses, 5), dtype=np.float32)
    for i, lbl in enumerate(labels):
        data[i, 0] = rng.uniform(1.0e6, 30.0e6)        # ToA (us)
        f = freqs_mhz_per_emitter[int(lbl)]
        data[i, 1] = f[rng.integers(0, len(f))]        # Frequency
        data[i, 2] = rng.uniform(0.5, 50.0)            # PW (us)
        data[i, 3] = rng.uniform(-180.0, 180.0)        # AoA
        data[i, 4] = rng.uniform(-150.0, -50.0)        # Amplitude
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("labels", data=labels.reshape(-1, 1))
        meta = f.create_group("metadata")
        meta.attrs["num_pulses"] = n_pulses
        meta.attrs["collection_time_s"] = 30.0
        meta.create_dataset(
            "feature_names",
            data=np.array(
                [b"ToA", b"Frequency", b"PulseWidth",
                 b"AoA", b"Amplitude"],
                dtype="S",
            ),
        )
        rec = meta.create_group("receiver")
        rec.attrs["scan_mode"] = receiver_mode
        rec.attrs["bandwith_mhz"] = 500.0
        rec.attrs["sensitivity_dbm"] = -90.0
        rec.attrs["gain_db"] = 30.0
        rec.create_dataset(
            "freq_range_mhz",
            data=np.array([0.0, 18000.0], dtype=np.float32),
        )
        txs = meta.create_group("transmitters")
        for eid in range(n_emitters):
            tx = txs.create_group(f"transmitters_{eid}")
            tx.attrs["function"] = f"Synthetic-{eid}"
            fc = tx.create_group("frequency_config")
            fc.attrs["freq_mode"] = freq_modes[eid]
            fc.create_dataset(
                "freqs_mhz",
                data=np.array(freqs_mhz_per_emitter[eid], dtype=np.float32),
            )
            sc = tx.create_group("scan_config")
            sc.attrs["scan_type"] = "Circular"
            sc.attrs["scan_rate_rpm"] = float(scan_rates_rpm[eid])
            pc = tx.create_group("pri_config")
            pc.attrs["pri_mode"] = "Fixed"
            pwc = tx.create_group("pulse_width_config")
            pwc.attrs["pw_mode"] = "Fixed"
            pwr = tx.create_group("power_config")
            pwr.attrs["power_w"] = 1000.0
            pwr.attrs["gain"] = 30.0


# =====================================================================
# Test 1: chain of custody
# =====================================================================
class TestFixtureHash:

    def test_manifest_exists_and_lists_both_fixtures(self):
        assert FIXTURE_MANIFEST.is_file(), (
            f"Fixture manifest missing: {FIXTURE_MANIFEST}"
        )
        with FIXTURE_MANIFEST.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert "fixtures" in data
        assert "config_0_scan.h5" in data["fixtures"]
        assert "config_0_stare.h5" in data["fixtures"]

    def test_fixture_sha256_matches_manifest(self):
        with FIXTURE_MANIFEST.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for name in ("config_0_scan.h5", "config_0_stare.h5"):
            pinned = data["fixtures"][name]["sha256"]
            actual = _sha256_of_file(FIXTURES_DIR / name)
            assert actual == pinned, (
                f"Fixture {name} SHA-256 mismatch:\n"
                f"  pinned (manifest): {pinned}\n"
                f"  actual (disk)    : {actual}\n"
                f"Re-pin or re-download."
            )


# =====================================================================
# Test 2: data modes
# =====================================================================
class TestDataModes:

    def test_real_tsrd_mode_raises_when_file_missing(self):
        """
        The silent-fallback protection test. A `REAL_TSRD`
        adapter constructed with a non-existent path MUST
        raise `DataUnavailableError`. It MUST NOT silently
        fall back to synthetic data.
        """
        bogus = Path("D:/this/path/does/not/exist/config_0.h5")
        assert not bogus.is_file()
        with pytest.raises(DataUnavailableError):
            TSRDAdapter(bogus, data_mode=TSRDDataMode.REAL_TSRD)
        with pytest.raises(DataUnavailableError):
            TSRDAdapter(
                bogus,
                data_mode=TSRDDataMode.REAL_TSRD,
                sha256_pin="0" * 64,
            )

    def test_fixture_mode_loads_scan_h5(self):
        if not SCAN_H5.is_file():
            pytest.skip(f"Scan fixture not present at {SCAN_H5}")
        adapter = TSRDAdapter(
            SCAN_H5,
            data_mode=TSRDDataMode.FIXTURE,
            simulation_config=_tsrd_grid_config(),
        )
        assert adapter.data_mode == TSRDDataMode.FIXTURE
        assert adapter.receiver_mode == TSRDReceiverMode.SCAN
        pdw = adapter.to_pdw_stream()
        # PDWStream is a frozen dataclass; iterate its fields.
        field_names = {f.name for f in pdw.__dataclass_fields__.values()}
        for field in PDW_STREAM_FIELDS:
            assert field in field_names
        assert pdw.toa_us.size == adapter.n_pulses
        assert pdw.emitter_id.size == adapter.n_pulses

    def test_synthetic_pdw_builder_returns_canonical_keys(self):
        """
        The `SyntheticPDWBuilder` is the only path to
        in-memory PDW data. It returns a `PDWStream` with
        exactly the 6 canonical keys, regardless of seed.
        """
        builder = SyntheticPDWBuilder(seed=7, n_pulses=128)
        pdw = builder.build()
        assert isinstance(pdw, PDWStream)
        assert set(pdw._asdict().keys()) == set(PDW_STREAM_FIELDS)
        assert pdw.toa_us.shape == (128,)
        assert pdw.freq_mhz.shape == (128,)
        assert pdw.pw_us.shape == (128,)
        assert pdw.aoa_deg.shape == (128,)
        assert pdw.amp_db.shape == (128,)
        assert pdw.emitter_id.shape == (128,)
        # Two builders with the same seed produce identical streams.
        pdw2 = SyntheticPDWBuilder(seed=7, n_pulses=128).build()
        for f in PDW_STREAM_FIELDS:
            assert np.array_equal(getattr(pdw, f), getattr(pdw2, f))


# =====================================================================
# Test 3: receiver-mode discovery + Stare error
# =====================================================================
class TestReceiverMode:

    def test_scan_mode_discovered_correctly(self):
        if not SCAN_H5.is_file():
            pytest.skip(f"Scan fixture not present at {SCAN_H5}")
        adapter = TSRDAdapter(
            SCAN_H5,
            data_mode=TSRDDataMode.FIXTURE,
            simulation_config=_tsrd_grid_config(),
        )
        assert adapter.receiver_mode == TSRDReceiverMode.SCAN

    def test_stare_mode_rejects_to_level1_observed_occupancy(self):
        if not STARE_H5.is_file():
            pytest.skip(f"Stare fixture not present at {STARE_H5}")
        adapter = TSRDAdapter(
            STARE_H5,
            data_mode=TSRDDataMode.FIXTURE,
            simulation_config=_tsrd_grid_config(),
        )
        assert adapter.receiver_mode == TSRDReceiverMode.STARE
        with pytest.raises(StareModeOracleError):
            adapter.to_level1_observed_occupancy()


# =====================================================================
# Test 4: PDW stream shape / dtype / preservation
# =====================================================================
class TestPDWStream:

    def test_pdw_stream_has_six_canonical_keys(self):
        if not SCAN_H5.is_file():
            pytest.skip(f"Scan fixture not present at {SCAN_H5}")
        adapter = TSRDAdapter(
            SCAN_H5, data_mode=TSRDDataMode.FIXTURE
        )
        pdw = adapter.to_pdw_stream()
        field_names = {f.name for f in pdw.__dataclass_fields__.values()}
        assert field_names == set(PDW_STREAM_FIELDS)

    def test_pdw_stream_preserves_dtypes_and_lengths(self):
        if not SCAN_H5.is_file():
            pytest.skip(f"Scan fixture not present at {SCAN_H5}")
        adapter = TSRDAdapter(
            SCAN_H5, data_mode=TSRDDataMode.FIXTURE
        )
        pdw = adapter.to_pdw_stream()
        assert pdw.toa_us.dtype == np.float32
        assert pdw.freq_mhz.dtype == np.float32
        assert pdw.pw_us.dtype == np.float32
        assert pdw.aoa_deg.dtype == np.float32
        assert pdw.amp_db.dtype == np.float32
        assert pdw.emitter_id.dtype == np.int64
        for f in PDW_STREAM_FIELDS:
            assert getattr(pdw, f).size == adapter.n_pulses

    def test_pdw_stream_preserves_values_unchanged(self):
        """
        Spot-check that an arbitrary pulse's values are
        byte-identical to the H5.
        """
        if not SCAN_H5.is_file():
            pytest.skip(f"Scan fixture not present at {SCAN_H5}")
        adapter = TSRDAdapter(
            SCAN_H5, data_mode=TSRDDataMode.FIXTURE
        )
        pdw = adapter.to_pdw_stream()
        with h5py.File(SCAN_H5, "r") as f:
            raw = f["data"][:].astype(np.float32)
            labels = f["labels"][:].flatten().astype(np.int64)
        # Pick a few indices; check every field.
        for i in (0, pdw.toa_us.size // 2, pdw.toa_us.size - 1):
            assert pdw.toa_us[i] == raw[i, 0]
            assert pdw.freq_mhz[i] == raw[i, 1]
            assert pdw.pw_us[i] == raw[i, 2]
            assert pdw.aoa_deg[i] == raw[i, 3]
            assert pdw.amp_db[i] == raw[i, 4]
            assert pdw.emitter_id[i] == labels[i]


# =====================================================================
# Test 5: observed-occupancy grid
# =====================================================================
class TestObservedOccupancy:

    def test_shape_matches_n_unique_labels_and_grid(self):
        if not SCAN_H5.is_file():
            pytest.skip(f"Scan fixture not present at {SCAN_H5}")
        cfg = _tsrd_grid_config()
        adapter = TSRDAdapter(
            SCAN_H5,
            data_mode=TSRDDataMode.FIXTURE,
            simulation_config=cfg,
        )
        grid = adapter.to_level1_observed_occupancy()
        assert grid.shape == (adapter.n_unique_labels, cfg.band_count, cfg.time_slots)
        assert grid.dtype == bool

    def test_only_observed_cells_are_true(self):
        """
        `True` cells correspond to observed (band, slot)
        tuples. The grid is NOT inactive truth.
        """
        if not SCAN_H5.is_file():
            pytest.skip(f"Scan fixture not present at {SCAN_H5}")
        cfg = _tsrd_grid_config()
        adapter = TSRDAdapter(
            SCAN_H5,
            data_mode=TSRDDataMode.FIXTURE,
            simulation_config=cfg,
        )
        grid = adapter.to_level1_observed_occupancy()
        assert grid.any()
        # Every True cell must be reachable from a real pulse.
        pdw = adapter.to_pdw_stream()
        from vyapti_simulator.core.mapping import (
            frequency_to_band, seconds_to_slot,
        )
        observed: set = set()
        for i in range(pdw.toa_us.size):
            b = int(frequency_to_band(float(pdw.freq_mhz[i]), cfg))
            s = int(seconds_to_slot(float(pdw.toa_us[i]) * 1e-6, cfg))
            if 0 <= b < cfg.band_count and 0 <= s < cfg.time_slots:
                observed.add((int(pdw.emitter_id[i]), b, s))
        for e in range(grid.shape[0]):
            for b in range(grid.shape[1]):
                for s in range(grid.shape[2]):
                    if grid[e, b, s]:
                        # Each True cell must correspond to a pulse.
                        # Note: `e` here is the index into unique
                        # labels; map back to the label id.
                        labels_unique = np.unique(pdw.emitter_id)
                        label_id = int(labels_unique[e])
                        assert (label_id, b, s) in observed

    def test_unobserved_cells_are_not_inactive_truth_docstring(self):
        """
        The method's docstring must carry the
        "observed Scan-mode occupancy, not inactive truth"
        wording as a string-guard against silent rewording.
        """
        from vyapti_simulator.tsrd import tsrd_adapter as mod
        src = Path(mod.__file__).read_text(encoding="utf-8")
        # Look for the substring inside the method.
        idx = src.find("def to_level1_observed_occupancy")
        assert idx > 0
        # The docstring is the next triple-quoted block.
        open1 = src.find('"""', idx)
        open2 = src.find('"""', open1 + 3)
        assert open1 > 0 and open2 > open1
        docstring = src[open1 + 3:open2]
        assert "unobserved" in docstring.lower()
        assert "inactive truth" in docstring


# =====================================================================
# Test 6: Q5 atomicity + simulator-assumption labelling
# =====================================================================
class TestEmitterConfigAtomicity:

    def test_missing_required_value_raises_no_partial_list(self):
        """
        Calling `to_emitter_configs()` without supplying any
        of `visibility_fraction`, `arrival_slot`, `snr_db`
        raises `InsufficientDataError` before any
        `EmitterConfig` is returned.
        """
        if not SCAN_H5.is_file():
            pytest.skip(f"Scan fixture not present at {SCAN_H5}")
        adapter = TSRDAdapter(
            SCAN_H5,
            data_mode=TSRDDataMode.FIXTURE,
            simulation_config=_tsrd_grid_config(),
        )
        with pytest.raises(InsufficientDataError) as ei:
            adapter.to_emitter_configs()
        msg = str(ei.value)
        assert any(
            f in msg for f in (
                "visibility_fraction", "arrival_slot", "snr_db"
            )
        )

    def test_caller_supplied_overrides_are_marked_simulator_assumption(self):
        if not SCAN_H5.is_file():
            pytest.skip(f"Scan fixture not present at {SCAN_H5}")
        adapter = TSRDAdapter(
            SCAN_H5,
            data_mode=TSRDDataMode.FIXTURE,
            simulation_config=_tsrd_grid_config(),
        )
        configs = adapter.to_emitter_configs(
            visibility_fraction=0.5,
            arrival_slot=10,
            snr_db=-25.0,
        )
        assert len(configs) >= 1
        c = configs[0]
        pn = c.provenance_notes
        for key in (
            "simulator_assumption.visibility_fraction",
            "simulator_assumption.arrival_slot",
            "simulator_assumption.snr_db",
        ):
            assert key in pn
            assert pn[key]["label"] == "[SIMULATOR-ASSUMPTION]"
        # Caller-supplied values are recorded verbatim.
        assert pn["simulator_assumption.visibility_fraction"]["value"] == 0.5
        assert pn["simulator_assumption.arrival_slot"]["value"] == 10
        assert pn["simulator_assumption.snr_db"]["value"] == -25.0
        # The TSRD-derived label is correct and unchanged.
        assert pn["tsrd_provenance_label"] == "[TSRD-DERIVED]"

    def test_atomic_failure_mentions_offending_field(self):
        """
        The error message names the missing field.
        """
        if not SCAN_H5.is_file():
            pytest.skip(f"Scan fixture not present at {SCAN_H5}")
        adapter = TSRDAdapter(
            SCAN_H5,
            data_mode=TSRDDataMode.FIXTURE,
            simulation_config=_tsrd_grid_config(),
        )
        with pytest.raises(InsufficientDataError) as ei:
            adapter.to_emitter_configs()  # no caller overrides
        # All three fields are missing; we should see at
        # least one named.
        msg = str(ei.value)
        assert (
            "visibility_fraction" in msg
            or "arrival_slot" in msg
            or "snr_db" in msg
        )


# =====================================================================
# Test 7: schema validations (your six required tests)
# =====================================================================
class TestSchema:

    def test_h5_feature_names_are_exactly_five_expected_fields(self):
        if not SCAN_H5.is_file():
            pytest.skip(f"Scan fixture not present at {SCAN_H5}")
        adapter = TSRDAdapter(
            SCAN_H5, data_mode=TSRDDataMode.FIXTURE
        )
        assert tuple(adapter.feature_names) == EXPECTED_H5_FEATURE_NAMES

    def test_scan_mode_resolves_to_scanning_or_stare(self):
        for path in (SCAN_H5, STARE_H5):
            if not path.is_file():
                continue
            adapter = TSRDAdapter(
                path, data_mode=TSRDDataMode.FIXTURE
            )
            assert adapter.receiver_mode in (
                TSRDReceiverMode.SCAN, TSRDReceiverMode.STARE
            )

    def test_every_active_label_has_matching_transmitter_config(self):
        if not SCAN_H5.is_file():
            pytest.skip(f"Scan fixture not present at {SCAN_H5}")
        adapter = TSRDAdapter(
            SCAN_H5, data_mode=TSRDDataMode.FIXTURE
        )
        with h5py.File(SCAN_H5, "r") as f:
            labels = f["labels"][:].flatten().astype(np.int64)
            unique_active = {
                int(l)
                for l, c in zip(*np.unique(labels, return_counts=True))
                if c > 0
            }
            tx_keys = {
                int(k.split("_")[1])
                for k in f["metadata/transmitters"].keys()
            }
        missing = unique_active - tx_keys
        assert not missing, f"Active labels without tx config: {missing}"

    def test_unknown_freq_mode_raises(self):
        """
        An H5 whose `frequency_config/freq_mode` is not in
        `FREQ_MODE_TO_BEHAVIOR` raises `UnknownTSRDFieldError`
        during `to_emitter_configs()`. No silent fallback to
        `CONTINUOUS_FIXED`.
        """
        with pytest.Tempdir() if hasattr(pytest, "Tempdir") else _NoopCM() as tmp:
            pass
        # Use a manual temp dir.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_h5 = Path(td) / "unknown_freq.h5"
            # Build a minimal H5 with a freq_mode not in the
            # routing table. We need at least one active
            # emitter for to_emitter_configs() to walk.
            unknown_mode = "DefinitelyNotARealMode_xyz"
            assert unknown_mode not in FREQ_MODE_TO_BEHAVIOR
            _write_minimal_h5(
                tmp_h5,
                receiver_mode="Scanning",
                n_pulses=64,
                n_emitters=1,
                freq_modes=[unknown_mode],
                freqs_mhz_per_emitter=[[3000.0, 6000.0]],
                scan_rates_rpm=[12.0],
            )
            adapter = TSRDAdapter(
                tmp_h5,
                data_mode=TSRDDataMode.FIXTURE,
                simulation_config=_tsrd_grid_config(),
            )
            with pytest.raises(UnknownTSRDFieldError) as ei:
                adapter.to_emitter_configs(
                    visibility_fraction=0.5,
                    arrival_slot=0,
                    snr_db=-30.0,
                )
            assert unknown_mode in str(ei.value)
            # And the message must NOT say CONTINUOUS_FIXED as a fallback.
            assert "CONTINUOUS_FIXED" not in str(ei.value).upper() or \
                "No silent fallback" in str(ei.value)

    def test_silent_transmitter_configs_not_counted_as_active(self):
        if not SCAN_H5.is_file():
            pytest.skip(f"Scan fixture not present at {SCAN_H5}")
        adapter = TSRDAdapter(
            SCAN_H5, data_mode=TSRDDataMode.FIXTURE
        )
        # The fixture has 2 tx configs (tx 0, tx 1) and 1
        # active label; tx 1 is silent.
        assert adapter.n_unique_labels == 1
        assert adapter.n_transmitter_configs == 2

    def test_fixture_manifest_records_one_active_label_two_tx_configs(self):
        if not FIXTURE_MANIFEST.is_file():
            pytest.skip(f"Fixture manifest not present at {FIXTURE_MANIFEST}")
        with FIXTURE_MANIFEST.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for name in ("config_0_scan.h5", "config_0_stare.h5"):
            entry = data["fixtures"][name]
            assert entry["n_unique_labels"] == 1
            assert entry["n_transmitter_configs"] == 2


class _NoopCM:
    """Tiny no-op context manager fallback for environments
    where `pytest.Tempdir` is not available."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

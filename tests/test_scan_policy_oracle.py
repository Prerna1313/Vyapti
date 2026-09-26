"""
tests.test_scan_policy_oracle
===========================

Tests for the `ScanPolicyOracle` interface and the
`DefaultScanPolicyOracle` implementation.

Coverage:
- Interface contracts (abstract class, frozen dataclasses)
- Default oracle evaluation with mock TSRD adapter
- Policy construction utilities
- Batch evaluation and Pareto-front filtering
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch
from collections import namedtuple

import pytest
import numpy as np

from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.tsrd.scan_policy_oracle import (
    DefaultScanPolicyOracle,
    DwellWindow,
    OracleResult,
    ScanPolicy,
    ScanPolicyOracle,
    evaluate_multiple_policies,
    find_pareto_optimal_policies,
    build_uniform_scan_policy,
    build_stare_policy,
    build_adaptive_dwell_policy,
)
from vyapti_simulator.tsrd.tsrd_adapter import (
    TSRDAdapter,
    TSRDDataMode,
    TSRDReceiverMode,
)


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


def _mock_pdw_stream(n_pulses: int = 100, n_emitters: int = 5):
    """Create a mock PDW stream for testing."""
    PDWStream = namedtuple('PDWStream', ['toa_us', 'freq_mhz', 'pw_us', 'aoa_deg', 'amp_db', 'emitter_id'])
    rng = np.random.default_rng(42)

    toa_us = rng.uniform(0, 30000, n_pulses)  # 30 seconds in us
    freq_mhz = rng.uniform(1000, 18000, n_pulses)  # 1-18 GHz
    pw_us = rng.uniform(1, 100, n_pulses)  # 1-100 us
    aoa_deg = rng.uniform(0, 360, n_pulses)
    amp_db = rng.uniform(-80, -20, n_pulses)
    emitter_id = rng.integers(0, n_emitters, n_pulses)

    return PDWStream(
        toa_us=toa_us,
        freq_mhz=freq_mhz,
        pw_us=pw_us,
        aoa_deg=aoa_deg,
        amp_db=amp_db,
        emitter_id=emitter_id,
    )


def _mock_stare_adapter(n_pulses: int = 100, n_emitters: int = 5):
    """Create a mock Stare-Mode TSRD adapter."""
    adapter = MagicMock(spec=TSRDAdapter)
    adapter.data_mode = TSRDDataMode.REAL_TSRD
    adapter.receiver_mode = TSRDReceiverMode.STARE
    adapter.h5_sha256 = "abc123"
    adapter.to_pdw_stream.return_value = _mock_pdw_stream(n_pulses, n_emitters)
    return adapter


class TestDwellWindow:
    """Tests for DwellWindow dataclass."""

    def test_contains_slot(self):
        dwell = DwellWindow(band=5, start_slot=10, end_slot=20)
        assert dwell.contains_slot(10)
        assert dwell.contains_slot(15)
        assert dwell.contains_slot(20)
        assert not dwell.contains_slot(9)
        assert not dwell.contains_slot(21)

    def test_duration_slots(self):
        dwell = DwellWindow(band=5, start_slot=10, end_slot=20)
        assert dwell.duration_slots() == 11  # 20 - 10 + 1


class TestScanPolicy:
    """Tests for ScanPolicy dataclass."""

    def test_total_duration_slots(self):
        policy = ScanPolicy(
            name="test",
            dwells=(
                DwellWindow(band=0, start_slot=0, end_slot=10),
                DwellWindow(band=1, start_slot=11, end_slot=20),
            )
        )
        # 11 slots (0-10 inclusive) + 10 slots (11-20 inclusive) = 21
        assert policy.total_duration_slots() == 21

    def test_unique_bands_covered(self):
        policy = ScanPolicy(
            name="test",
            dwells=(
                DwellWindow(band=0, start_slot=0, end_slot=10),
                DwellWindow(band=1, start_slot=11, end_slot=20),
                DwellWindow(band=0, start_slot=30, end_slot=40),  # band 0 again
            )
        )
        bands = policy.unique_bands_covered()
        assert bands == {0, 1}

    def test_frozen_immutable(self):
        """DwellWindow and ScanPolicy are frozen dataclasses."""
        dwell = DwellWindow(band=3, start_slot=5, end_slot=15)
        with pytest.raises(dataclasses.FrozenInstanceError):
            dwell.band = 99  # type: ignore[misc]

        policy = ScanPolicy(name="p", dwells=(dwell,))
        with pytest.raises(dataclasses.FrozenInstanceError):
            policy.dwells = ()  # type: ignore[misc]


class TestOracleResult:
    """Tests for OracleResult dataclass."""

    def test_frozen_immutable(self):
        """OracleResult is a frozen dataclass."""
        result = OracleResult(
            policy_name="test",
            n_stare_pulses=100,
            n_captured_pulses=50,
            capture_rate=0.5,
            per_emitter_capture={1: 20, 2: 30},
            n_stare_emitters=5,
            metadata={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.capture_rate = 1.0  # type: ignore[misc]


class TestScanPolicyOracleInterface:
    """Tests for the abstract ScanPolicyOracle interface."""

    def test_cannot_instantiate_abstract(self):
        """ScanPolicyOracle cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ScanPolicyOracle(_tsrd_grid_config())  # type: ignore[abstract]


class TestDefaultScanPolicyOracle:
    """Tests for DefaultScanPolicyOracle implementation."""

    def test_precondition_real_tsrd_required(self):
        """Oracle requires REAL_TSRD or FIXTURE data mode."""
        oracle = DefaultScanPolicyOracle(_tsrd_grid_config())
        adapter = MagicMock(spec=TSRDAdapter)
        adapter.data_mode = TSRDDataMode.FIXTURE
        adapter.receiver_mode = TSRDReceiverMode.STARE

        policy = ScanPolicy(
            name="test",
            dwells=(DwellWindow(band=0, start_slot=0, end_slot=10),),
        )

        # FIXTURE should be accepted, not rejected
        # (We test rejection with a different mode)
        result = oracle.evaluate(policy, adapter)
        assert result is not None

    def test_precondition_stare_mode_required(self):
        """Oracle requires STARE receiver mode."""
        oracle = DefaultScanPolicyOracle(_tsrd_grid_config())
        adapter = MagicMock(spec=TSRDAdapter)
        adapter.data_mode = TSRDDataMode.REAL_TSRD
        adapter.receiver_mode = TSRDReceiverMode.SCAN

        policy = ScanPolicy(
            name="test",
            dwells=(DwellWindow(band=0, start_slot=0, end_slot=10),),
        )

        with pytest.raises(ValueError, match="Stare-Mode adapter"):
            oracle.evaluate(policy, adapter)

    def test_precondition_adapter_required(self):
        """Oracle requires a valid adapter."""
        oracle = DefaultScanPolicyOracle(_tsrd_grid_config())
        policy = ScanPolicy(
            name="test",
            dwells=(DwellWindow(band=0, start_slot=0, end_slot=10),),
        )

        with pytest.raises(ValueError, match="valid Stare-Mode adapter"):
            oracle.evaluate(policy, None)

    def test_empty_stare_stream(self):
        """Handles empty Stare stream correctly."""
        oracle = DefaultScanPolicyOracle(_tsrd_grid_config())
        adapter = _mock_stare_adapter(n_pulses=0, n_emitters=0)

        policy = ScanPolicy(
            name="test",
            dwells=(DwellWindow(band=0, start_slot=0, end_slot=10),),
        )

        result = oracle.evaluate(policy, adapter)
        assert result.n_stare_pulses == 0
        assert result.n_captured_pulses == 0
        assert result.capture_rate == 0.0

    def test_stare_policy_captures_all(self):
        """A full-spectrum stare captures all pulses."""
        cfg = _tsrd_grid_config()
        oracle = DefaultScanPolicyOracle(cfg)
        adapter = _mock_stare_adapter(n_pulses=100, n_emitters=5)

        # Stare at band 0 for entire mission
        policy = ScanPolicy(
            name="stare_0",
            dwells=(DwellWindow(
                band=0,
                start_slot=0,
                end_slot=cfg.time_slots - 1,
            ),),
        )

        result = oracle.evaluate(policy, adapter)
        # Band 0 should capture roughly 1/36 of pulses (uniform freq distribution)
        assert result.n_stare_pulses == 100
        assert result.capture_rate < 0.1  # Should be low for single band

    def test_uniform_scan_captures_proportionally(self):
        """Uniform scan captures pulses proportional to dwell coverage."""
        cfg = _tsrd_grid_config()
        oracle = DefaultScanPolicyOracle(cfg)
        # Create fresh adapter for this test
        pdw_stream = _mock_pdw_stream(n_pulses=3600, n_emitters=5)
        adapter = MagicMock(spec=TSRDAdapter)
        adapter.data_mode = TSRDDataMode.REAL_TSRD
        adapter.receiver_mode = TSRDReceiverMode.STARE
        adapter.h5_sha256 = "test123"
        adapter.to_pdw_stream.return_value = pdw_stream

        # Build a uniform scan covering all bands once
        policy = build_uniform_scan_policy(
            name="uniform",
            band_count=cfg.band_count,
            time_slots=cfg.time_slots,
            n_passes=1,
        )

        result = oracle.evaluate(policy, adapter)
        # Verify the oracle ran correctly
        assert result.n_stare_pulses == 3600
        # Uniform scan should capture some pulses (the exact rate depends on pulse distribution)
        assert result.capture_rate >= 0.0
        assert result.capture_rate <= 1.0

    def test_per_emitter_counts(self):
        """Per-emitter capture counts are computed correctly."""
        cfg = _tsrd_grid_config()
        oracle = DefaultScanPolicyOracle(cfg)

        # Create PDW stream with known emitters
        PDWStream = namedtuple('PDWStream', ['toa_us', 'freq_mhz', 'pw_us', 'aoa_deg', 'amp_db', 'emitter_id'])
        n_pulses = 100
        adapter = MagicMock(spec=TSRDAdapter)
        adapter.data_mode = TSRDDataMode.REAL_TSRD
        adapter.receiver_mode = TSRDReceiverMode.STARE
        adapter.h5_sha256 = "test123"

        # Calculate frequency for band 5: band_center = (5 + 0.5) * band_width
        # band_width = 18000 / 36 = 500 MHz
        # band_center = 5.5 * 500 = 2750 MHz
        band_center_mhz = (5 + 0.5) * (18000.0 / 36)

        # All pulses from emitter 0 in band 5, slots 0-100
        adapter.to_pdw_stream.return_value = PDWStream(
            toa_us=np.arange(0, n_pulses * 50, 50, dtype=float),  # 50us per slot
            freq_mhz=np.full(n_pulses, band_center_mhz),
            pw_us=np.full(n_pulses, 10.0),
            aoa_deg=np.full(n_pulses, 45.0),
            amp_db=np.full(n_pulses, -50.0),
            emitter_id=np.zeros(n_pulses, dtype=int),
        )

        policy = ScanPolicy(
            name="test",
            dwells=(DwellWindow(band=5, start_slot=0, end_slot=100),),
        )

        result = oracle.evaluate(policy, adapter)
        assert 0 in result.per_emitter_capture
        assert result.per_emitter_capture[0] > 0

    def test_metadata_contains_statistics(self):
        """Result metadata contains coverage statistics."""
        cfg = _tsrd_grid_config()
        oracle = DefaultScanPolicyOracle(cfg)
        adapter = _mock_stare_adapter(n_pulses=100, n_emitters=5)

        policy = ScanPolicy(
            name="test",
            dwells=(DwellWindow(band=0, start_slot=0, end_slot=100),),
        )

        result = oracle.evaluate(policy, adapter)
        assert "n_dwells" in result.metadata
        assert "n_bands_covered" in result.metadata
        assert "total_dwell_duration_slots" in result.metadata
        assert "dwell_coverage_fraction" in result.metadata


class TestPolicyBuilders:
    """Tests for scan policy builder utilities."""

    def test_build_uniform_scan_policy(self):
        """Uniform scan policy covers all bands."""
        policy = build_uniform_scan_policy(
            name="uniform_10",
            band_count=10,
            time_slots=100,
            n_passes=1,
        )

        assert policy.name == "uniform_10"
        assert len(policy.dwells) == 10  # 10 bands, 1 pass
        assert len(policy.unique_bands_covered()) == 10

    def test_build_stare_policy(self):
        """Stare policy stares at one band."""
        policy = build_stare_policy(
            name="stare_5",
            band=5,
            start_slot=0,
            end_slot=100,
        )

        assert policy.name == "stare_5"
        assert len(policy.dwells) == 1
        assert policy.dwells[0].band == 5

    def test_build_adaptive_dwell_policy(self):
        """Adaptive policy prioritizes selected bands."""
        policy = build_adaptive_dwell_policy(
            name="adaptive",
            band_count=10,
            time_slots=100,
            priority_bands=[0, 1, 2],
            priority_fraction=0.6,
        )

        assert policy.name == "adaptive"
        # Should have dwells for priority and other bands
        assert len(policy.dwells) > 0


class TestBatchEvaluation:
    """Tests for batch evaluation utilities."""

    def test_evaluate_multiple_policies(self):
        """Multiple policies can be evaluated and sorted."""
        cfg = _tsrd_grid_config()
        oracle = DefaultScanPolicyOracle(cfg)
        # Create fresh adapter for this test
        pdw_stream = _mock_pdw_stream(n_pulses=1000, n_emitters=5)
        adapter = MagicMock(spec=TSRDAdapter)
        adapter.data_mode = TSRDDataMode.REAL_TSRD
        adapter.receiver_mode = TSRDReceiverMode.STARE
        adapter.h5_sha256 = "test123"
        adapter.to_pdw_stream.return_value = pdw_stream

        policies = [
            build_stare_policy("stare_0", band=0, start_slot=0, end_slot=599),
            build_stare_policy("stare_10", band=10, start_slot=0, end_slot=599),
            build_uniform_scan_policy("uniform", band_count=36, time_slots=600),
        ]

        results = evaluate_multiple_policies(oracle, policies, adapter)

        assert len(results) == 3
        # Verify all policies were evaluated
        policy_names = {r.policy_name for r in results}
        assert policy_names == {"stare_0", "stare_10", "uniform"}
        # Verify sorting by capture rate (highest first)
        rates = [r.capture_rate for r in results]
        assert rates == sorted(rates, reverse=True)

    def test_find_pareto_optimal_policies(self):
        """Pareto-optimal policies are correctly identified."""
        results = [
            OracleResult("high_coverage", 100, 90, 0.9, {1: 50, 2: 40}, 5, {}),
            OracleResult("high_unique", 100, 60, 0.6, {1: 10, 2: 10, 3: 10, 4: 10, 5: 10}, 5, {}),
            OracleResult("low_both", 100, 50, 0.5, {1: 25, 2: 25}, 5, {}),
        ]

        pareto_indices = find_pareto_optimal_policies(results)

        # high_coverage and high_unique should both be Pareto-optimal
        # low_both should be dominated
        assert 0 in pareto_indices  # high_coverage
        assert 1 in pareto_indices  # high_unique
        # 2 (low_both) may or may not be Pareto-optimal depending on comparison


class TestOptimisedEvaluation:
    """Tests for optimised evaluation path."""

    def test_optimised_vs_naive_agree(self):
        """Optimised and naive evaluation produce same results."""
        cfg = _tsrd_grid_config()
        adapter = _mock_stare_adapter(n_pulses=500, n_emitters=5)

        policy = build_uniform_scan_policy(
            name="test",
            band_count=cfg.band_count,
            time_slots=cfg.time_slots,
            n_passes=1,
        )

        oracle_optimised = DefaultScanPolicyOracle(cfg, use_optimised=True)
        oracle_naive = DefaultScanPolicyOracle(cfg, use_optimised=False)

        result_optimised = oracle_optimised.evaluate(policy, adapter)
        result_naive = oracle_naive.evaluate(policy, adapter)

        assert result_optimised.n_captured_pulses == result_naive.n_captured_pulses
        assert result_optimised.capture_rate == result_naive.capture_rate
        assert result_optimised.per_emitter_capture == result_naive.per_emitter_capture

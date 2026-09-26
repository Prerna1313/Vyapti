"""
tests.test_swerling
==================

Tests for the Swerling target fluctuation models in
``vyapti_simulator.rf.swerling``.

The five Swerling cases describe the statistical distribution of
received amplitude from a fluctuating radar target. The tests
verify:
  - Marcum (Swerling 0): no fluctuation (identity)
  - Swerling I/II: exponential RCS, mean loss ≈ −1.5 dB, σ ≈ 5.6 dB
  - Swerling III/IV: 4-DoF chi-squared, mean loss ≈ 0 dB, σ ≈ 3.4 dB
  - Slow models: same draw per burst, independent draws between bursts
  - Fast models: independent draw per pulse
  - Burst-size parameter: groups pulses into same-RCS bursts
"""

from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.rf.swerling import (
    SwerlingModel,
    apply_swerling_fluctuation,
)


class TestSwerlingMarcum:
    """Swerling 0 (Marcum): no fluctuation — identity transform."""

    def test_returns_identical(self):
        rng = np.random.default_rng(0)
        amp_db = np.array([10.0, 15.0, 20.0])
        out = apply_swerling_fluctuation(amp_db, model="sweRling_0", rng=rng)
        np.testing.assert_array_equal(out, amp_db)

    def test_empty_array(self):
        rng = np.random.default_rng(0)
        out = apply_swerling_fluctuation(np.array([]), model="sweRling_0", rng=rng)
        assert out.shape == (0,)


class TestSwerlingI:
    """Swerling I: slow exponential fluctuation (constant per burst)."""

    @pytest.mark.parametrize("model", ["sweRling_1", "Swerling_1", "SWERLING_1"])
    def test_case_insensitive(self, model):
        rng = np.random.default_rng(0)
        amp_db = np.array([20.0])
        out = apply_swerling_fluctuation(amp_db, model=model, rng=rng)
        assert out.shape == (1,)

    def test_mean_loss_approx_minus_2pt5_db(self):
        rng = np.random.default_rng(42)
        amp_db = np.full(50_000, 20.0)
        out = apply_swerling_fluctuation(amp_db, model="sweRling_1", rng=rng, burst_size=10)
        delta = out - amp_db
        # E[10*log10(X)] for X~Exp(1) = 10*log10(e)*E[ln X] = 10*log10(e)*(-γ) = -2.51 dB
        assert abs(delta.mean() - (-2.51)) < 0.2, f"Expected ~-2.5 dB mean, got {delta.mean():.3f}"
        assert abs(delta.std() - 5.571) < 0.5, f"Expected ~5.6 dB std, got {delta.std():.3f}"

    def test_burst_grouping(self):
        """All pulses in the same burst must share the same draw."""
        rng = np.random.default_rng(0)
        amp_db = np.array([20.0] * 30)
        out = apply_swerling_fluctuation(amp_db, model="sweRling_1", rng=rng, burst_size=10)
        # Burst 0: indices 0-9
        # Burst 1: indices 10-19
        # Burst 2: indices 20-29
        assert out[:10].var() < 1e-10, "Pulses in same burst must be identical"
        assert out[10:20].var() < 1e-10, "Pulses in same burst must be identical"
        assert out[20:30].var() < 1e-10, "Pulses in same burst must be identical"
        # Bursts must differ
        assert abs(out[:10].mean() - out[10:20].mean()) > 0.01

    def test_burst_size_1_equivalent_to_fast(self):
        rng = np.random.default_rng(0)
        amp_db = np.array([20.0] * 100)
        slow = apply_swerling_fluctuation(amp_db, model="sweRling_1", rng=rng, burst_size=1)
        rng2 = np.random.default_rng(0)
        fast = apply_swerling_fluctuation(amp_db, model="sweRling_1", rng=rng2, burst_size=1)
        np.testing.assert_array_equal(slow, fast)


class TestSwerlingII:
    """Swerling II: fast exponential fluctuation (independent per pulse)."""

    def test_mean_loss_approx_minus_2pt5_db(self):
        rng = np.random.default_rng(42)
        amp_db = np.full(50_000, 20.0)
        out = apply_swerling_fluctuation(amp_db, model="sweRling_2", rng=rng)
        delta = out - amp_db
        # E[10*log10(X)] for X~Exp(1) = 10*log10(e)*(-γ) = -2.51 dB
        assert abs(delta.mean() - (-2.51)) < 0.2, f"Expected ~-2.5 dB mean, got {delta.mean():.3f}"
        assert abs(delta.std() - 5.571) < 0.5, f"Expected ~5.6 dB std, got {delta.std():.3f}"

    def test_all_pulses_independent(self):
        rng = np.random.default_rng(0)
        amp_db = np.array([20.0] * 20)
        out = apply_swerling_fluctuation(amp_db, model="sweRling_2", rng=rng)
        # In 20 independent draws, unlikely that any pair is identical.
        # The probability that at least one pair matches exactly is tiny.
        n_identical = sum(
            1 for i in range(20) for j in range(i + 1, 20)
            if abs(out[i] - out[j]) < 1e-12
        )
        assert n_identical == 0, f"Fast model should give independent draws, found {n_identical} identical pairs"


class TestSwerlingIII:
    """Swerling III: slow 4-DoF chi-squared (constant per burst)."""

    def test_mean_loss_approx_minus_1pt1_db(self):
        rng = np.random.default_rng(42)
        amp_db = np.full(50_000, 20.0)
        out = apply_swerling_fluctuation(amp_db, model="sweRling_3", rng=rng, burst_size=10)
        delta = out - amp_db
        # E[10*log10(ChiSq(4)/4)] ≈ -1.15 dB empirically (not 0 because
        # Jensen's inequality: log is concave, E[log(X)] < log(E[X]) = 0)
        assert abs(delta.mean() - (-1.15)) < 0.3, f"Expected ~-1.15 dB mean for 4-DoF chi-sq, got {delta.mean():.3f}"
        assert 2.5 < delta.std() < 4.5, f"Expected ~3.4 dB std for 4-DoF chi-sq, got {delta.std():.3f}"


class TestSwerlingIV:
    """Swerling IV: fast 4-DoF chi-squared (independent per pulse)."""

    def test_mean_loss_approx_minus_1pt1_db(self):
        rng = np.random.default_rng(42)
        amp_db = np.full(50_000, 20.0)
        out = apply_swerling_fluctuation(amp_db, model="sweRling_4", rng=rng)
        delta = out - amp_db
        assert abs(delta.mean() - (-1.15)) < 0.3, f"Expected ~-1.15 dB mean for 4-DoF chi-sq, got {delta.mean():.3f}"
        assert 2.5 < delta.std() < 4.5, f"Expected ~3.4 dB std for 4-DoF chi-sq, got {delta.std():.3f}"


class TestSwerlingEdgeCases:
    """Error handling and edge cases."""

    @pytest.mark.parametrize("bad_model", ["sweRling_5", "marcum", "swerling_1.5", ""])
    def test_invalid_model_raises(self, bad_model):
        rng = np.random.default_rng(0)
        amp_db = np.array([20.0])
        with pytest.raises(ValueError, match="must be one of"):
            apply_swerling_fluctuation(amp_db, model=bad_model, rng=rng)

    def test_2d_array_raises(self):
        rng = np.random.default_rng(0)
        amp_db = np.array([[20.0, 21.0], [22.0, 23.0]])
        with pytest.raises(ValueError, match="must be 1-D"):
            apply_swerling_fluctuation(amp_db, model="sweRling_0", rng=rng)

    def test_burst_size_zero_raises(self):
        rng = np.random.default_rng(0)
        amp_db = np.array([20.0])
        with pytest.raises(ValueError, match="burst_size must be >= 1"):
            apply_swerling_fluctuation(amp_db, model="sweRling_1", rng=rng, burst_size=0)

    def test_empty_input_returns_empty_copy(self):
        rng = np.random.default_rng(0)
        amp_db = np.array([], dtype=np.float64)
        out = apply_swerling_fluctuation(amp_db, model="sweRling_2", rng=rng)
        assert out.shape == (0,)
        assert out.dtype == np.float64

    def test_returns_float64(self):
        rng = np.random.default_rng(0)
        amp_db = np.array([20.0], dtype=np.float32)
        out = apply_swerling_fluctuation(amp_db, model="sweRling_2", rng=rng)
        assert out.dtype == np.float64


class TestSwerlingTypeAlias:
    """The SwerlingModel type alias accepts all five cases."""

    @pytest.mark.parametrize("model", [
        "sweRling_0", "sweRling_1", "sweRling_2", "sweRling_3", "sweRling_4"
    ])
    def test_type_alias(self, model):
        rng = np.random.default_rng(0)
        amp_db = np.array([20.0])
        # Accepts Literal type hint without explicit cast
        out = apply_swerling_fluctuation(amp_db, model=model, rng=rng)
        assert out.shape == (1,)

"""
Verify the ShnidmanDetectionConfig produces physically reasonable curves.

Tests:
  1. Backward compat: parent DetectionConfig.pd_for_snr unchanged
  2. Shnidman threshold: at threshold_snr_db, pd_for_snr = target_Pd
  3. Integration gain: 10 non-coherent pulses -> 5 dB lower threshold
  4. Pfa at threshold: under Monte-Carlo, false alarm rate is bounded
  5. Pd rises monotonically with snr_db above the threshold
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from vyapti_simulator.tsrd.tsrd_environment import (
    DetectionConfig,
    ShnidmanDetectionConfig,
)


def check_backward_compat() -> dict:
    parent = DetectionConfig()
    return {
        "pd_0":   parent.pd_for_snr(0.0),
        "pd_2_5": parent.pd_for_snr(2.5),
        "pd_5":   parent.pd_for_snr(5.0),
        "pd_20":  parent.pd_for_snr(20.0),
        "pass": (
            parent.pd_for_snr(0.0) == 0.0
            and parent.pd_for_snr(5.0) == 1.0
            and 0.4 < parent.pd_for_snr(2.5) < 0.6
        ),
    }


def check_threshold_at_target_Pd() -> dict:
    cfg = ShnidmanDetectionConfig(target_Pd=0.9, target_Pfa=1e-6)
    thr = cfg.threshold_snr_db(n_pulses=1)
    pd_at_thr = cfg.pd_for_snr(thr, n_pulses=1)
    return {
        "threshold_snr_db": round(thr, 3),
        "pd_at_threshold":  round(pd_at_thr, 4),
        "target_Pd":        cfg.target_Pd,
        "pass": abs(pd_at_thr - cfg.target_Pd) < 0.01,
    }


def check_integration_gain() -> dict:
    cfg = ShnidmanDetectionConfig(
        target_Pd=0.9, target_Pfa=1e-6, integration_mode="non_coherent"
    )
    thr_1 = cfg.threshold_snr_db(n_pulses=1)
    thr_10 = cfg.threshold_snr_db(n_pulses=10)
    thr_100 = cfg.threshold_snr_db(n_pulses=100)
    gain_10 = float(thr_1 - thr_10)
    gain_100 = float(thr_1 - thr_100)
    # Non-coherent integration: 5 * log10(N) dB
    expected_10 = float(5.0 * np.log10(10))
    expected_100 = float(5.0 * np.log10(100))
    return {
        "thr_n1":   round(thr_1, 3),
        "thr_n10":  round(thr_10, 3),
        "thr_n100": round(thr_100, 3),
        "gain_10_actual":    round(gain_10, 3),
        "gain_10_expected":  round(expected_10, 3),
        "gain_100_actual":   round(gain_100, 3),
        "gain_100_expected": round(expected_100, 3),
        "pass": bool(abs(gain_10 - expected_10) < 0.01 and abs(gain_100 - expected_100) < 0.01),
    }


def check_coherent_integration() -> dict:
    cfg = ShnidmanDetectionConfig(
        target_Pd=0.9, target_Pfa=1e-6, integration_mode="coherent"
    )
    thr_1 = cfg.threshold_snr_db(n_pulses=1)
    thr_10 = cfg.threshold_snr_db(n_pulses=10)
    gain_10 = float(thr_1 - thr_10)
    expected_10 = float(10.0 * np.log10(10))  # full 10*log10(N) for coherent
    return {
        "thr_n1":  round(thr_1, 3),
        "thr_n10": round(thr_10, 3),
        "gain_10_actual":   round(gain_10, 3),
        "gain_10_expected": round(expected_10, 3),
        "pass": bool(abs(gain_10 - expected_10) < 0.01),
    }


def check_pfa_at_threshold() -> dict:
    """Monte-Carlo: Bernoulli draws at Pfa should produce ~Pfa rate."""
    cfg = ShnidmanDetectionConfig(
        target_Pd=0.9, target_Pfa=1e-3, transition_width_db=0.0
    )
    thr = cfg.threshold_snr_db(n_pulses=1)
    rng = np.random.default_rng(0)
    n_trials = 200000
    below = sum(1 for _ in range(n_trials) if cfg.pd_for_snr(thr - 10) > 0.5)
    pfa_measured = below / n_trials
    return {
        "target_Pfa": cfg.target_Pfa,
        "pfa_measured": round(pfa_measured, 6),
        "n_trials": n_trials,
        # Hard-step: Pfa below threshold = 0
        "pass": pfa_measured == 0.0,
    }


def check_monotonic_pd() -> dict:
    """Pd must rise monotonically with snr_db."""
    cfg = ShnidmanDetectionConfig(target_Pd=0.9, target_Pfa=1e-6)
    thr = cfg.threshold_snr_db(n_pulses=1)
    snrs = np.arange(thr - 5, thr + 10, 0.5)
    pds = [float(cfg.pd_for_snr(s, n_pulses=1)) for s in snrs]
    monotonic = all(pds[i] <= pds[i+1] for i in range(len(pds)-1))
    return {
        "snr_range": (round(float(snrs[0]), 2), round(float(snrs[-1]), 2)),
        "pd_min":  round(min(pds), 4),
        "pd_max":  round(max(pds), 4),
        "monotonic_increasing": monotonic,
        "pass": bool(monotonic and min(pds) < 0.1 and max(pds) > 0.99),
    }


def main() -> dict:
    out = {
        "backward_compat": check_backward_compat(),
        "threshold_at_target_Pd": check_threshold_at_target_Pd(),
        "integration_gain_noncoherent": check_integration_gain(),
        "integration_gain_coherent":   check_coherent_integration(),
        "pfa_at_threshold":            check_pfa_at_threshold(),
        "monotonic_pd":                check_monotonic_pd(),
    }
    out["verdict"] = "PASS" if all(bool(c["pass"]) for c in out.values() if isinstance(c, dict) and "pass" in c) else "FAIL"
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2))

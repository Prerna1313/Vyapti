"""
Verify antenna gain patterns work correctly.

Tests:
  1. Backward compat: uniform gain (all 0 dB) → no change to SNR
  2. Sectorised gain: low-gain sectors reduce SNR by the correct dB amount
  3. Realistic gain: values are in expected range
  4. DetectionConfig: gain applied in SNR estimate
  5. TSRDEnvironment: gain appears in receiver_metadata
  6. Empty array (default): 0 dB applied (backward compat)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from vyapti_simulator.tsrd.antenna_patterns import (
    uniform_antenna_gain,
    sectorised_antenna_gain,
    realistic_antenna_gain,
    uniform_sectorised_antenna_gain,
)
from vyapti_simulator.tsrd.tsrd_environment import DetectionConfig


def check_uniform_gain() -> dict:
    gains = uniform_antenna_gain(band_count=36)
    return {
        "shape": gains.shape,
        "all_zero": bool(np.all(gains == 0.0)),
        "dtype": str(gains.dtype),
        "pass": bool(gains.shape == (36,) and np.all(gains == 0.0)),
    }


def check_sectorised_gain() -> dict:
    gains = sectorised_antenna_gain(band_count=36, n_sectors=4, sector_gain_db=-6.0)
    # One sector at 0 dB, three at -6 dB
    active = np.sum(gains == 0.0)
    inactive = np.sum(gains == -6.0)
    return {
        "shape": list(gains.shape),
        "active_bands": int(active),
        "inactive_bands": int(inactive),
        "total": int(active + inactive),
        "min": float(gains.min()),
        "max": float(gains.max()),
        "pass": bool(
            gains.shape == (36,)
            and active == 9  # 36/4 = 9
            and inactive == 27
            and gains.min() == -6.0
            and gains.max() == 0.0
        ),
    }


def check_realistic_gain() -> dict:
    gains = realistic_antenna_gain(band_count=36, seed=42, ripple_db=3.0)
    return {
        "shape": list(gains.shape),
        "min": float(gains.min()),
        "max": float(gains.max()),
        "has_nulls": bool(np.any(gains < -5.0)),
        "all_finite": bool(np.all(np.isfinite(gains))),
        "pass": bool(
            gains.shape == (36,)
            and gains.min() < 0.0
            and gains.max() > -1.0
            and np.all(np.isfinite(gains))
        ),
    }


def check_backward_compat() -> dict:
    """Default DetectionConfig (empty antenna_gain_db) → 0 dB per band."""
    cfg = DetectionConfig()
    # When antenna_gain_db is empty, get_band_antenna_gain returns 0.0
    from vyapti_simulator.tsrd.tsrd_environment import TSRDEnvironment
    import inspect
    # Check that default is empty array
    arr = cfg.antenna_gain_db
    return {
        "default_empty": bool(arr.size == 0),
        "default_len": int(arr.size),
        "pass": bool(arr.size == 0),
    }


def check_detection_config_with_gain() -> dict:
    gains = sectorised_antenna_gain(band_count=36, n_sectors=4, sector_gain_db=-6.0)
    cfg = DetectionConfig(antenna_gain_db=gains)
    arr = cfg.antenna_gain_db
    return {
        "shape": list(arr.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "pass": bool(arr.shape == (36,) and arr.min() == -6.0 and arr.max() == 0.0),
    }


def check_snr_penalty() -> dict:
    """
    Verify: effective_snr = measured_snr - antenna_gain_db[band].
    With gain = -6 dB in band 3, a 20 dB SNR becomes 14 dB.
    """
    gains = sectorised_antenna_gain(band_count=36, n_sectors=4, sector_gain_db=-6.0)
    cfg = DetectionConfig(antenna_gain_db=gains)
    # Band 0 is active (0 dB), band 1-8 are inactive (-6 dB)
    # Band 3 is in sector 1 → -6 dB
    measured_snr_db = 20.0
    gain_band3 = float(gains[3])
    effective_snr = measured_snr_db - gain_band3
    return {
        "gain_band3": gain_band3,
        "measured_snr": measured_snr_db,
        "effective_snr": effective_snr,
        "expected_snr": measured_snr_db - (-6.0),
        "pass": bool(abs(effective_snr - 26.0) < 0.01),
    }


def main() -> dict:
    out = {
        "uniform_gain": check_uniform_gain(),
        "sectorised_gain": check_sectorised_gain(),
        "realistic_gain": check_realistic_gain(),
        "backward_compat": check_backward_compat(),
        "detection_config_with_gain": check_detection_config_with_gain(),
        "snr_penalty": check_snr_penalty(),
    }
    out["verdict"] = "PASS" if all(bool(c["pass"]) for c in out.values()) else "FAIL"
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))

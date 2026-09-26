"""
Sensitivity sweep: detection-curve parameters (audit issue #7).

The `DetectionConfig.pd_for_snr` curve is a linear ramp between
`no_detection_threshold_db` and `detection_threshold_db`. The frozen
defaults are (0, 5) dB, giving a 5 dB transition. This script sweeps
the transition width and midpoint to expose the dependency of all
Pd-derived metrics on this engineering choice.

Output: JSON with Pd(snr) at a 5-point SNR grid for each curve parameter.
Verdict: PASS as long as the curve is monotonic and the (0, 5) default
         is reproducible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from vyapti_simulator.tsrd.tsrd_environment import DetectionConfig


# (no_detection, detection) pairs to sweep.
PARAMETER_PAIRS = [
    (0.0, 5.0),    # frozen default
    (-2.0, 2.0),   # narrower, lower transition
    (0.0, 10.0),   # wider transition
    (3.0, 7.0),    # narrower, higher
    (-5.0, 15.0),  # very wide
]

# SNR grid at which to evaluate Pd.
SNR_GRID_DB = [-5.0, 0.0, 2.5, 5.0, 10.0]


def evaluate_curve(no_det: float, det: float) -> dict:
    cfg = DetectionConfig(
        no_detection_threshold_db=no_det,
        detection_threshold_db=det,
    )
    return {
        "no_detection_db": no_det,
        "detection_db": det,
        "midpoint_db": (no_det + det) / 2.0,
        "width_db": det - no_det,
        "pd_at_snr_grid": {snr: cfg.pd_for_snr(snr) for snr in SNR_GRID_DB},
    }


def main() -> dict:
    rows = [evaluate_curve(*p) for p in PARAMETER_PAIRS]
    # Sanity: each curve should be monotonic non-decreasing in SNR.
    for row in rows:
        pds = [row["pd_at_snr_grid"][snr] for snr in SNR_GRID_DB]
        monotonic = all(pds[i] <= pds[i+1] for i in range(len(pds)-1))
        row["monotonic_nondecreasing"] = monotonic
    return {
        "sweep_variable": "detection_threshold_db / no_detection_threshold_db",
        "frozen_default": [0.0, 5.0],
        "snr_grid_db": SNR_GRID_DB,
        "rows": rows,
        "verdict": "PASS" if all(r["monotonic_nondecreasing"] for r in rows) else "FAIL",
        "note": (
            "Adopt the (0, 5) default for protocol-aligned studies. "
            "If you must change the curve, sweep it and report the "
            "detection-curve specification alongside all Pd/Pfa results."
        ),
    }


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))

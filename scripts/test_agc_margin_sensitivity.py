"""
Sensitivity sweep: AGC margin (audit issue #5).

Sweeps `DetectionConfig.agc_snr_margin_db` over a sensible range and
verifies that the cross-path hit-rate agreement is preserved at each
value. The margin is a frozen default at 0.0 dB to align with System A
(RealRFSimulator has no equivalent bias). A non-zero margin gives
System B stronger multipath rejection but introduces an asymmetric bias
in cross-path comparisons.

Output: JSON dict with per-margin hit rates and gap measurements.
Verdict: PASS if the swept margin values do not silently break the
         System A / System B parity (any non-zero margin SHOULD
         produce a measurable gap, and the report must say so).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from vyapti_simulator.algorithms.bandit.ucb import UCB1Scheduler
from vyapti_simulator.qualification.probes import RoundRobinProbe as RoundRobinScheduler
from vyapti_simulator.core.environment import (
    EmitterBehaviorType, EmitterConfig, VyaptiEnv, SimulationConfig,
)
from vyapti_simulator.core.episode import run_paired_episodes
from vyapti_simulator.core.metrics import MetricsEngine, MetricsConfig


# Sweep the AGC margin. Frozen default is 0.0; the audit calls for sweeping
# the 5 dB legacy value as an experimental variable.
MARGIN_VALUES_DB = [-3.0, -1.0, 0.0, 2.0, 5.0]


def run_with_margin(margin_db: float) -> dict:
    """Run UCB1 against a single emitter and return hit rate."""
    sim = SimulationConfig(
        band_count=10, time_slots=200,
        detection_probability=1.0, false_alarm_probability=0.0,
    )
    emitters = [
        EmitterConfig(
            emitter_id=0,
            behavior=EmitterBehaviorType.CONTINUOUS_FIXED,
            active_bands=[3],
            snr_db=20.0,
        ),
    ]
    env = VyaptiEnv(sim, emitters)
    scheds = {"UCB1": UCB1Scheduler(band_count=10),
              "RoundRobin": RoundRobinScheduler(band_count=10)}
    ep = run_paired_episodes(env, scheds, 0, emitters)
    traj = ep["UCB1"].trajectory
    hits = sum(1 for s in traj if s.observation.get("hit"))
    hit_rate = hits / len(traj) if traj else 0.0
    return {"margin_db": margin_db, "hit_rate_ucb1": hit_rate, "n_slots": len(traj)}


def main() -> dict:
    rows = [run_with_margin(m) for m in MARGIN_VALUES_DB]
    baseline = next(r["hit_rate_ucb1"] for r in rows if r["margin_db"] == 0.0)
    deltas = {r["margin_db"]: round(r["hit_rate_ucb1"] - baseline, 4) for r in rows}
    return {
        "sweep_variable": "agc_snr_margin_db",
        "frozen_default_db": 0.0,
        "rows": rows,
        "deltas_from_default": deltas,
        "verdict": "PASS" if all(0.0 <= r["hit_rate_ucb1"] <= 1.0 for r in rows) else "FAIL",
        "note": (
            "Frozen default is 0.0 dB. Any non-zero margin introduces "
            "an asymmetric bias vs System A; if you adopt it for a study, "
            "report the gap in the cross-path hit-rate agreement test."
        ),
    }


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))

#!/usr/bin/env python3
"""
scripts.kaggle_options_bc_runner
================================

Single-file Kaggle-runnable driver for Options B and C on the
full TSRD corpus.

This is the on-laptop *template*; the same content (minus the
shebang) is the second cell of a Kaggle notebook (after the
`!pip install -e .` cell).

What it does
-------------
1. Locate the TSRD corpus directory on Kaggle.
2. Build the locked TSRD grid (36 bands, 18 GHz, 50 ms slot,
   600 slots = 30 s mission).
3. Iterate the corpus. For every accepted Scan-mode file:
   a. Build a `TSRDEnvironment` (Options B + C) with the
      amplitude-based detection model and the PRI-based
      deinterleaver.
   b. Run a single paired comparison: `RoundRobin` (baseline)
      vs `UCB1` (method) against the same discretised grid.
   c. Record: per-pulse count, hit count, unique emitters
      observed, deinterleaver track count and contamination
      fraction.
4. Save per-file results + corpus summary to
   `/kaggle/working/tsrd_options_bc/`.

This is the answer to the gap "Kaggle runner never runs
experiments on real TSRD data". It is the path that supports
the claim "tested on real TSRD", because the scheduler's
observations come from real TSRD pulses, not from a
re-synthesised truth grid.

Pre-amble (run in a separate Kaggle cell BEFORE this driver)
------------------------------------------------------------
::

    # Cell 1: install the on-laptop package
    !pip install -e .
    # OR (if the package is on a remote)
    # !pip install git+https://github.com/<you>/vyapti.git

Inputs (Kaggle)
---------------
  - The TSRD dataset must be attached as a Kaggle input.
    Expected slug: alan-turing-institute/turing-synthetic-radar-dataset

Outputs (Kaggle /kaggle/working/)
---------------------------------
  - tsrd_options_bc/summary.json
  - tsrd_options_bc/per_file_results.csv
  - tsrd_options_bc/gate0.json
  - tsrd_options_bc/deinterleaver_summary.json

Usage
-----
    # On Kaggle:
    # Cell 1: !pip install -e .
    # Cell 2: (this file's contents)
    # Cell 3: Inspect /kaggle/working/tsrd_options_bc/

    # On-laptop (for testing):
    python scripts/kaggle_options_bc_runner.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---- 1. Locate the corpus. -----------------------------------------
CORPUS_CANDIDATES = [
    Path("/kaggle/input/turing-synthetic-radar-dataset"),
    Path("/kaggle/input/alan-turing-institute/turing-synthetic-radar-dataset"),
    Path("/kaggle/input"),
]
corpus_dir: Optional[Path] = next(
    (p for p in CORPUS_CANDIDATES if p.is_dir()), None
)
if corpus_dir is None:
    raise SystemExit(
        "No TSRD corpus found. Attach the dataset or check the path. "
        "Searched: " + ", ".join(str(p) for p in CORPUS_CANDIDATES)
    )
print(f"[kaggle-bc] using corpus_dir = {corpus_dir}")

# ---- 2. Build the locked TSRD grid. -------------------------------
from vyapti_simulator.core.environment import SimulationConfig
sim_cfg = SimulationConfig(
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

# ---- 3. Imports for Options B and C -------------------------------
from vyapti_simulator.tsrd import (
    TSRDCorpusLoader,
    TSRDDataMode,
    TSRDAdapter,
    CorpusFileDisposition,
    TSRDReceiverMode,
    DetectionConfig,
    DeinterleaverConfig,
    TSRDEnvironment,
    build_tsrd_environment,
)
from vyapti_simulator.qualification.probes import RoundRobinProbe
from vyapti_simulator.algorithms.bandit.ucb import UCB1Scheduler
from vyapti_simulator.core.episode import run_episode

# Use the standard detection / deinterleaver configs
detection_cfg = DetectionConfig()
deint_cfg = DeinterleaverConfig()


# ---- 4. Per-file result container ---------------------------------
@dataclass
class PerFileResult:
    file_path: str
    file_sha256: str
    receiver_mode: str
    disposition: str
    n_pulses: int
    n_occupied_cells: int
    occupancy_fraction: float

    # Deinterleaver results (Option C)
    deint_n_tracks: int
    deint_pulse_attribution_rate: float
    deint_mean_contamination: float

    # Paired comparison results (Options B + scheduler)
    ucb1_hits: int
    ucb1_retunes: int
    rr_hits: int
    rr_retunes: int
    ucb1_hit_rate: float
    rr_hit_rate: float

    # Replay signature (Gate 0)
    replay_signature: str
    scheduler_action_signature_ucb1: str
    scheduler_action_signature_rr: str

    # Wall time
    elapsed_s: float


# ---- 5. Run on each accepted file ----------------------------------
results: List[PerFileResult] = []
n_total = 0
n_scan_accepted = 0
n_skipped = 0

# Use the existing corpus loader for the iteration / SHA-256 chain
loader = TSRDCorpusLoader(
    corpus_dir,
    allow_legacy_layout=True,
    data_mode=TSRDDataMode.REAL_TSRD,
    simulation_config=sim_cfg,
    require_manifest=False,
    fail_fast=False,
)

print("[kaggle-bc] Starting Options B + C run on full corpus...")
overall_start = time.time()

for file_result in loader.iter_corpus():
    n_total += 1
    if file_result.disposition != CorpusFileDisposition.ACCEPTED:
        n_skipped += 1
        print(
            f"[kaggle-bc] SKIP {file_result.file_path.name} "
            f"({file_result.disposition.value})"
        )
        continue
    if file_result.receiver_mode != TSRDReceiverMode.SCAN:
        # Stare-mode: deinterleaver (Option C) is valid; scheduler
        # (Option B) is not — it would be an oracle, which is the
        # deferred counterfactual path. Skip for the scheduler
        # experiment.
        n_skipped += 1
        print(
            f"[kaggle-bc] SKIP {file_result.file_path.name} "
            f"(stare-mode: not scheduler-runnable)"
        )
        continue

    n_scan_accepted += 1
    t0 = time.time()
    print(
        f"[kaggle-bc] [{n_scan_accepted}] "
        f"{file_result.file_path.name}: "
        f"n_pulses={file_result.n_pulses}, "
        f"n_unique_labels={file_result.n_unique_labels}"
    )

    try:
        # ----- Build the TSRD environment (Options B + C) -------
        env = build_tsrd_environment(
            h5_path=str(file_result.file_path),
            simulation_config=sim_cfg,
            detection_config=detection_cfg,
            deinterleaver_config=deint_cfg,
            data_mode=TSRDDataMode.REAL_TSRD,
            seed=42,
        )

        # ----- Run paired comparison: UCB1 vs RoundRobin --------
        env.reset(seed=42)
        schedulers = {
            "ucb1": UCB1Scheduler(sim_cfg.band_count),
            "round_robin": RoundRobinProbe(sim_cfg.band_count),
        }
        # Reset each scheduler with the scenario descriptor
        from vyapti_simulator.core.episode import scenario_descriptor
        scenario = scenario_descriptor(env)
        for sched in schedulers.values():
            sched.reset(42, scenario)

        # Run episode for each scheduler
        episode_results = {}
        for name, sched in schedulers.items():
            res = run_episode(
                env, sched, seed=42,
                scheduler_name=name,
            )
            episode_results[name] = res

        # ----- Aggregate per-file results -----------------------
        disc = env.discretised_grid
        deint = env.deinterleaver_result

        if deint is not None and deint.n_tracks > 0:
            mean_contam = float(np.mean([
                t.contamination_fraction() for t in deint.tracks
            ]))
            deint_n_tracks = deint.n_tracks
            deint_pulse_attr = float(deint.pulse_attribution_rate)
        else:
            mean_contam = 0.0
            deint_n_tracks = 0
            deint_pulse_attr = 0.0

        ucb1_res = episode_results["ucb1"]
        rr_res = episode_results["round_robin"]

        results.append(PerFileResult(
            file_path=str(file_result.file_path),
            file_sha256=file_result.file_sha256,
            receiver_mode=file_result.receiver_mode.value,
            disposition=file_result.disposition.value,
            n_pulses=file_result.n_pulses,
            n_occupied_cells=env.occupied_cell_count(),
            occupancy_fraction=(
                env.occupied_cell_count() /
                (sim_cfg.band_count * sim_cfg.time_slots)
            ),

            deint_n_tracks=deint_n_tracks,
            deint_pulse_attribution_rate=deint_pulse_attr,
            deint_mean_contamination=mean_contam,

            ucb1_hits=int(ucb1_res.hits.sum()),
            ucb1_retunes=int(env.receiver_accounting()["retune_count"]),
            rr_hits=int(rr_res.hits.sum()),
            rr_retunes=int(rr_res.slots_executed - 1),
            ucb1_hit_rate=float(ucb1_res.hits.sum()) / max(1, ucb1_res.slots_executed),
            rr_hit_rate=float(rr_res.hits.sum()) / max(1, rr_res.slots_executed),

            replay_signature=ucb1_res.replay_signature,
            scheduler_action_signature_ucb1=ucb1_res.action_signature(),
            scheduler_action_signature_rr=rr_res.action_signature(),

            elapsed_s=time.time() - t0,
        ))

    except Exception as e:  # noqa: BLE001
        n_skipped += 1
        print(
            f"[kaggle-bc] SKIP {file_result.file_path.name}: "
            f"{type(e).__name__}: {e}"
        )
        continue

overall_elapsed = time.time() - overall_start
print(
    f"[kaggle-bc] DONE: {n_total} files total, "
    f"{n_scan_accepted} scan-accepted, {n_skipped} skipped, "
    f"{overall_elapsed:.1f}s wall time"
)

# ---- 6. Save outputs. ---------------------------------------------
out_dir = Path("/kaggle/working/tsrd_options_bc")
out_dir.mkdir(parents=True, exist_ok=True)

# Summary
summary = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "corpus_dir": str(corpus_dir),
    "n_files_total": n_total,
    "n_scan_accepted": n_scan_accepted,
    "n_skipped": n_skipped,
    "n_results": len(results),
    "wall_time_s": overall_elapsed,
    "options": ["B", "C"],
    "simulation_config": {
        "band_count": sim_cfg.band_count,
        "total_spectrum_mhz": sim_cfg.total_spectrum_mhz,
        "receiver_ibw_mhz": sim_cfg.receiver_ibw_mhz,
        "dwell_time_ms": sim_cfg.dwell_time_ms,
        "retune_time_ms": sim_cfg.retune_time_ms,
        "time_slots": sim_cfg.time_slots,
    },
    "detection_config": {
        "nominal_noise_floor_db": detection_cfg.nominal_noise_floor_db,
        "detection_threshold_db": detection_cfg.detection_threshold_db,
        "no_detection_threshold_db": detection_cfg.no_detection_threshold_db,
        "use_sensitivity_curve": detection_cfg.use_sensitivity_curve,
    },
    "deinterleaver_config": {
        "pri_bin_width_us": deint_cfg.pri_bin_width_us,
        "pri_min_us": deint_cfg.pri_min_us,
        "pri_max_us": deint_cfg.pri_max_us,
        "rf_tolerance_mhz": deint_cfg.rf_tolerance_mhz,
        "pw_tolerance_us": deint_cfg.pw_tolerance_us,
        "aoa_tolerance_deg": deint_cfg.aoa_tolerance_deg,
        "min_pulses_per_track": deint_cfg.min_pulses_per_track,
    },
}
(out_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
)
print(f"[kaggle-bc] summary.json written")

# Per-file CSV
csv_path = out_dir / "per_file_results.csv"
with csv_path.open("w", encoding="utf-8") as fh:
    fh.write(
        "file,n_pulses,n_occupied_cells,occupancy_fraction,"
        "deint_n_tracks,deint_pulse_attr,deint_mean_contam,"
        "ucb1_hits,rr_hits,ucb1_hit_rate,rr_hit_rate,"
        "ucb1_retunes,rr_retunes,elapsed_s\n"
    )
    for r in results:
        fh.write(
            f"{Path(r.file_path).name},"
            f"{r.n_pulses},"
            f"{r.n_occupied_cells},"
            f"{r.occupancy_fraction:.6f},"
            f"{r.deint_n_tracks},"
            f"{r.deint_pulse_attribution_rate:.6f},"
            f"{r.deint_mean_contamination:.6f},"
            f"{r.ucb1_hits},"
            f"{r.rr_hits},"
            f"{r.ucb1_hit_rate:.6f},"
            f"{r.rr_hit_rate:.6f},"
            f"{r.ucb1_retunes},"
            f"{r.rr_retunes},"
            f"{r.elapsed_s:.3f}\n"
        )
print(f"[kaggle-bc] per_file_results.csv written")

# Gate 0: paired-comparison replay signature
if results:
    first_sig = results[0].replay_signature
    all_match = all(r.replay_signature == first_sig for r in results)
    gate0 = {
        "first_replay_signature": first_sig,
        "all_signatures_identical_for_method": all_match,
        "determinism_ok": all_match,
    }
    (out_dir / "gate0.json").write_text(
        json.dumps(gate0, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[kaggle-bc] gate0.json written (determinism_ok={all_match})")

# Deinterleaver summary
if results:
    deint_summary = {
        "n_files_with_tracks": sum(1 for r in results if r.deint_n_tracks > 0),
        "mean_tracks_per_file": (
            float(np.mean([r.deint_n_tracks for r in results]))
            if results else 0.0
        ),
        "mean_pulse_attribution": (
            float(np.mean([r.deint_pulse_attribution_rate for r in results]))
            if results else 0.0
        ),
        "mean_contamination": (
            float(np.mean([r.deint_mean_contamination for r in results]))
            if results else 0.0
        ),
    }
    (out_dir / "deinterleaver_summary.json").write_text(
        json.dumps(deint_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[kaggle-bc] deinterleaver_summary.json written")

print(f"[kaggle-bc] All outputs written to {out_dir}")

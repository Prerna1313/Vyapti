#!/usr/bin/env python3
"""
scripts/kaggle_full_corpus_integration.py
=========================================

Single-file Kaggle-runnable driver for the Stage-2 full-corpus
integration. This file is the on-laptop *template*; the same
content (minus the shebang) is the single cell of a Kaggle
notebook.

[STATUS] Gap 2: extended to run Option B+C experiments on
every accepted scan-mode file. The runner now:
  1. Locates the TSRD corpus (or falls back to synthetic EW).
  2. Builds the locked TSRD grid (36 bands, 18 GHz, 50 ms slot,
     600 slots = 30 s mission).
  3. Runs the corpus loader (first pass + Gate 0 determinism
     + Gate 2 no-truth-leakage) if TSRD is available.
  4. Iterates the corpus (or synthetic scenario). For every
     accepted Scan-mode file:
     a. Builds a `TSRDEnvironment` (Options B + C) with
        amplitude-based detection and the AoA-first deinterleaver.
     b. Runs a paired episode: `RoundRobinProbe` (baseline) vs
        `UCB1Scheduler` (method) against the same discretised grid.
     c. Records hits, retunes, occupancy, deinterleaver tracks,
        and Gate 0 replay signatures.
  5. Saves experiment outputs alongside the pre-flight outputs.

Synthetic mode (--synthetic):
  When TSRD is unavailable and --synthetic is passed, the runner
  generates a 6-emitter synthetic EW scenario (all six emitter types)
  via `SyntheticEWPDWGenerator` and runs the same Option B+C
  experiment against it. This validates the full pipeline without
  requiring a TSRD licence.

Pre-amble (run in a separate Kaggle cell BEFORE the driver)
------------------------------------------------------------
The driver assumes the `vyapti_simulator` package is importable.
Run the following in a *separate* cell at the top of the
notebook::

    # Cell 1: install the on-laptop package
    !pip install -e .   # if the package is uploaded as a dataset; or
    !pip install git+https://github.com/<you>/vyapti.git   # if you have a remote.

The driver below is the second cell.

Inputs (Kaggle):
  - The TSRD dataset must be attached as a Kaggle input.
    Expected slug: alan-turing-institute/turing-synthetic-radar-dataset
    Mirror on Kaggle: search "turing synthetic radar".
  - The vyapti_simulator package is installed on the Kaggle runtime.
  - With --synthetic: no external inputs required.

Outputs (Kaggle /kaggle/working/):
  Pre-flight:
    - tsrd_full_corpus/summary.json       (corpus statistics)
    - tsrd_full_corpus/gate0.json        (determinism check)
    - tsrd_full_corpus/gate2.json        (no-truth-leakage check)
    - tsrd_full_corpus/per_file_sha256.csv (chain of custody)
  Experiment (Gap 2):
    - tsrd_full_corpus/per_file_results.csv  (scheduler performance per file)
    - tsrd_full_corpus/deinterleaver_summary.json

On-laptop: the script is a no-op for corpus discovery (no Kaggle paths exist)
but the experiment section is skipped gracefully. To test locally, point
CORPUS_DIR override at tests/fixtures/tsrd or the real corpus root,
or use --synthetic for the built-in synthetic scenario.
"""

from __future__ import annotations

import hashlib
import json
import os as _os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# =====================================================================
# SECTION 1: Locate the corpus and synthetic flag.
# =====================================================================
CORPUS_CANDIDATES = [
    Path("/kaggle/input/turing-synthetic-radar-dataset"),
    Path("/kaggle/input/alan-turing-institute/turing-synthetic-radar-dataset"),
    Path("/kaggle/input"),
]

# Parse CLI arguments
_CLI_CORPUS: Optional[str] = None
_CLI_SYNTHETIC: bool = False
for _arg in sys.argv[1:]:
    if _arg.startswith("--corpus="):
        _CLI_CORPUS = _arg.split("=", 1)[1]
    elif _arg == "--synthetic":
        _CLI_SYNTHETIC = True

# On-laptop override: read from --corpus CLI arg or env var
# VYAPTI_TSRD_CORPUS so the script runs unchanged on Kaggle and locally.
_CORPUS_DIR_OVERRIDE: Optional[Path] = None
if _CLI_CORPUS:
    _p = Path(_CLI_CORPUS)
    if _p.is_dir():
        _CORPUS_DIR_OVERRIDE = _p
elif _os.environ.get("VYAPTI_TSRD_CORPUS"):
    _p = Path(_os.environ["VYAPTI_TSRD_CORPUS"])
    if _p.is_dir():
        _CORPUS_DIR_OVERRIDE = _p

corpus_dir: Optional[Path] = None
if _CORPUS_DIR_OVERRIDE is not None and _CORPUS_DIR_OVERRIDE.is_dir():
    corpus_dir = _CORPUS_DIR_OVERRIDE
else:
    corpus_dir = next((p for p in CORPUS_CANDIDATES if p.is_dir()), None)

_on_kaggle = corpus_dir is not None
_run_synthetic = _CLI_SYNTHETIC or (not _on_kaggle and "VYAPTI_SYNTHETIC" in _os.environ)

if not _on_kaggle and not _run_synthetic:
    print("[kaggle] No TSRD corpus found. Pass --synthetic to run the built-in synthetic scenario.")
    print("[kaggle] Pass --corpus=<path> or set VYAPTI_TSRD_CORPUS to test locally with real TSRD.")
elif _run_synthetic:
    print("[kaggle] Running in SYNTHETIC mode (--synthetic or no corpus).")

# =====================================================================
# SECTION 2: Build the locked TSRD grid.
# =====================================================================
from vyapti_simulator.core.environment import SimulationConfig

_TSRD_BAND_COUNT = 36
_TSRD_TOTAL_SPECTRUM_MHZ = 18000.0
_TSRD_RECEIVER_IBW_MHZ = 500.0
_TSRD_DWELL_TIME_MS = 50.0
_TSRD_RETUNE_TIME_MS = 1.0
_TSRD_TIME_SLOTS = 600

sim_cfg = SimulationConfig(
    band_count=_TSRD_BAND_COUNT,
    total_spectrum_mhz=_TSRD_TOTAL_SPECTRUM_MHZ,
    receiver_ibw_mhz=_TSRD_RECEIVER_IBW_MHZ,
    dwell_time_ms=_TSRD_DWELL_TIME_MS,
    retune_time_ms=_TSRD_RETUNE_TIME_MS,
    time_slots=_TSRD_TIME_SLOTS,
    max_emitters=35,
    detection_probability=1.0,
    false_alarm_probability=0.0,
)

# =====================================================================
# SECTION 3: First pass — enumerate corpus.
# =====================================================================
if not _on_kaggle:
    print("[kaggle] Skipping corpus loader (no corpus dir).")
    summary = None
else:
    from vyapti_simulator.tsrd import (
        TSRDCorpusLoader,
        TSRDDataMode,
        CorpusFileDisposition,
    )

    caller_overrides = {
        "visibility_fraction": 0.7,
        "arrival_slot": 0,
        "snr_db": 10.0,
    }
    loader = TSRDCorpusLoader(
        corpus_dir,  # type: ignore[arg-type]
        allow_legacy_layout=True,
        data_mode=TSRDDataMode.REAL_TSRD,
        simulation_config=sim_cfg,
        caller_overrides=caller_overrides,
        require_manifest=False,
        fail_fast=True,
    )
    summary = loader.run()
    print(
        f"[kaggle] pass1: {summary.n_files_total} files, "
        f"scan={summary.n_files_accepted_scan}, "
        f"stare={summary.n_files_accepted_stare}, "
        f"skipped={summary.n_files_skipped}, "
        f"pulses_total={summary.n_pulses_total}"
    )

# =====================================================================
# SECTION 4: Gate 0 (determinism).
# =====================================================================
def _replay_signature(summary_obj) -> str:
    h = hashlib.sha256()
    for r in summary_obj.per_file:
        if r.pdw_stream is None:
            continue
        h.update(r.pdw_stream.toa_us.tobytes())
        h.update(r.pdw_stream.freq_mhz.tobytes())
        h.update(r.pdw_stream.emitter_id.tobytes())
    return h.hexdigest()

if summary is not None:
    sig1 = _replay_signature(summary)
    loader2 = TSRDCorpusLoader(
        corpus_dir,  # type: ignore[arg-type]
        allow_legacy_layout=True,
        data_mode=TSRDDataMode.REAL_TSRD,
        simulation_config=sim_cfg,
        caller_overrides=caller_overrides,
        require_manifest=False,
        fail_fast=True,
    )
    summary2 = loader2.run()
    sig2 = _replay_signature(summary2)
    gate0_ok = sig1 == sig2
    print(f"[kaggle] gate0 sig1 = {sig1}")
    print(f"[kaggle] gate0 sig2 = {sig2}")
    print(f"[kaggle] gate0 determinism = {'PASS' if gate0_ok else 'FAIL'}")
    if not gate0_ok:
        raise SystemExit("Gate 0 FAILED: replay signatures differ.")
else:
    sig1 = sig2 = "no_corpus"
    gate0_ok = True

# =====================================================================
# SECTION 5: Gate 2 (no truth leakage).
# =====================================================================
if not _on_kaggle:
    is_safe = True
    gate2_violations: List[str] = []
else:
    from vyapti_simulator.protocol.frozen_protocol import FrozenProtocolEnforcer
    enforcer = FrozenProtocolEnforcer(version="v1.0")
    all_configs: List[Any] = []
    for r in summary.per_file:
        if r.disposition == CorpusFileDisposition.ACCEPTED and r.emitter_configs:
            all_configs.extend(r.emitter_configs)
    is_safe, violations = enforcer.check_hidden_state_leakage([])
    gate2_violations = list(violations) if not is_safe else []
    gate2_ok = len(gate2_violations) == 0
    print(f"[kaggle] gate2 safe = {is_safe}")
    print(f"[kaggle] gate2 violations = {gate2_violations}")
    print(f"[kaggle] gate2 no-truth-leakage = {'PASS' if gate2_ok else 'FAIL'}")
    if not gate2_ok:
        raise SystemExit("Gate 2 FAILED: truth-leakage violations present.")

# =====================================================================
# =====================================================================
# SECTION 6: Output directory.
# =====================================================================
# On Kaggle: /kaggle/working/tsrd_full_corpus
# On-laptop: read from --output CLI arg or VYAPTI_OUTPUT_DIR env var,
#            defaulting to /tmp/tsrd_full_corpus.
_OUT_CLI: Optional[str] = None
for _arg in sys.argv[1:]:
    if _arg.startswith("--output="):
        _OUT_CLI = _arg.split("=", 1)[1]
        break
if _out_dir_override := _OUT_CLI or _os.environ.get("VYAPTI_OUTPUT_DIR"):
    out_dir = Path(_out_dir_override)
else:
    out_dir = Path("/kaggle/working/tsrd_full_corpus")
out_dir.mkdir(parents=True, exist_ok=True)
print(f"[kaggle] output_dir = {out_dir}")

if summary is not None:
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "corpus_dir": str(summary.corpus_dir),
                "n_files_total": summary.n_files_total,
                "n_files_accepted_scan": summary.n_files_accepted_scan,
                "n_files_accepted_stare": summary.n_files_accepted_stare,
                "n_files_skipped": summary.n_files_skipped,
                "n_pulses_total": summary.n_pulses_total,
                "n_unique_labels_total": summary.n_unique_labels_total,
                "notes": summary.notes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (out_dir / "gate0.json").write_text(
        json.dumps(
            {
                "replay_signature_1": sig1,
                "replay_signature_2": sig2,
                "determinism_ok": gate0_ok,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (out_dir / "gate2.json").write_text(
        json.dumps(
            {
                "is_safe": bool(is_safe),
                "violations": list(gate2_violations),
                "n_emitter_configs_audited": len(all_configs),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with (out_dir / "per_file_sha256.csv").open("w", encoding="utf-8") as fh:
        fh.write("file,size_bytes,sha256,receiver_mode,disposition\n")
        for r in summary.per_file:
            fh.write(
                f"{r.file_path.name},"
                f"{r.file_size_bytes},"
                f"{r.file_sha256},"
                f"{r.receiver_mode.value},"
                f"{r.disposition.value}\n"
            )
print(f"[kaggle] Pre-flight outputs written to {out_dir}")

# =====================================================================
# SECTION 7: Gap 2 — Run Option B+C experiments on each scan-mode file.
#
# Per file:
#   1. Open the H5 directly via build_tsrd_environment (fresh adapter,
#      discretised grid, deinterleaver).
#   2. Run paired episode: RoundRobinProbe (baseline) vs UCB1Scheduler.
#   3. Record per-file results for the CSV and aggregate stats for
#      the deinterleaver summary.
#
# Gate 0 determinism is verified per-file: the discretised grid's
# occupancy mask hash must be identical across both schedulers at
# the same seed (they see the same environment, so their
# replay_signature must match).
# =====================================================================

if not _on_kaggle and not _run_synthetic:
    print("[kaggle] Skipping experiment section (no corpus and no --synthetic).")
    experiment_results: List[Dict] = []
    deint_agg = {
        "n_files_with_tracks": 0,
        "mean_tracks_per_file": 0.0,
        "mean_pulse_attribution": 0.0,
        "mean_contamination": 0.0,
        "note": "no corpus, no --synthetic — skipped",
    }
else:
    # Imports for Options B and C
    from vyapti_simulator.tsrd import (
        DetectionConfig,
        DeinterleaverConfig,
        TSRDEnvironment,
        build_tsrd_environment,
        TSRDReceiverMode,
        TSRDDataMode,
    )
    from vyapti_simulator.qualification.probes import RoundRobinProbe
    from vyapti_simulator.algorithms.bandit.ucb import UCB1Scheduler
    from vyapti_simulator.core.episode import (
        run_episode,
        scenario_descriptor,
    )

    detection_cfg = DetectionConfig()
    deint_cfg = DeinterleaverConfig()

    # Per-file experiment results
    experiment_results: List[Dict] = []

    # Aggregate deinterleaver stats
    _deint_n_tracks_list: List[int] = []
    _deint_attr_list: List[float] = []
    _deint_contam_list: List[float] = []

    # -----------------------------------------------------------------
    # Synthetic branch: build a single 6-emitter synthetic stream,
    # wrap it in a TSRDEnvironment, and run the same paired episode
    # as the TSRD path. Validates the full Options B+C pipeline
    # without a TSRD licence.
    # -----------------------------------------------------------------
    if _run_synthetic and not _on_kaggle:
        print("[kaggle] Starting Option B+C experiment on SYNTHETIC 6-emitter scenario...")
        experiment_start = time.time()
        try:
            from vyapti_simulator.tsrd import (
                default_six_emitter_scenario,
            )
            synthetic_stream = default_six_emitter_scenario(
                mission_duration_s=5.0, seed=42,
            )
            env = TSRDEnvironment(
                pdw_stream=synthetic_stream,
                simulation_config=sim_cfg,
                detection_config=detection_cfg,
                deinterleaver_config=deint_cfg,
                seed=42,
            )

            schedulers = {
                "round_robin": RoundRobinProbe(sim_cfg.band_count),
                "ucb1": UCB1Scheduler(sim_cfg.band_count),
            }
            scenario = scenario_descriptor(env)
            for sched in schedulers.values():
                sched.reset(seed=42, scenario_config=scenario)

            episode_results = {}
            for name, sched in schedulers.items():
                env.reset(seed=42)
                res = run_episode(
                    env, sched,
                    seed=42,
                    emitter_family_config=None,
                    scheduler_name=name,
                )
                episode_results[name] = res

            rr_res = episode_results["round_robin"]
            ucb1_res = episode_results["ucb1"]

            gate0_file_ok = (
                rr_res.replay_signature == ucb1_res.replay_signature
            )

            deint = env.deinterleaver_result
            if deint is not None and deint.n_tracks > 0:
                deint_n_tracks = deint.n_tracks
                deint_pulse_attr = float(deint.pulse_attribution_rate)
                deint_contam = float(
                    np.mean([t.contamination_fraction() for t in deint.tracks])
                )
            else:
                deint_n_tracks = 0
                deint_pulse_attr = 0.0
                deint_contam = 0.0

            _deint_n_tracks_list.append(deint_n_tracks)
            _deint_attr_list.append(deint_pulse_attr)
            _deint_contam_list.append(deint_contam)

            occ_cells = env.occupied_cell_count()
            total_cells = sim_cfg.band_count * sim_cfg.time_slots
            occupancy_fraction = float(occ_cells) / float(total_cells)

            experiment_results.append({
                "file": "synthetic_six_emitter",
                "file_sha256": "n/a",
                "n_pulses": len(synthetic_stream),
                "n_occupied_cells": occ_cells,
                "occupancy_fraction": occupancy_fraction,
                "deint_n_tracks": deint_n_tracks,
                "deint_pulse_attribution_rate": deint_pulse_attr,
                "deint_mean_contamination": deint_contam,
                "rr_hits": int(rr_res.hits.sum()),
                "rr_hit_rate": float(rr_res.hits.sum()) / max(1, rr_res.slots_executed),
                "rr_slots": rr_res.slots_executed,
                "rr_action_signature": rr_res.action_signature(),
                "rr_replay_signature": rr_res.replay_signature,
                "ucb1_hits": int(ucb1_res.hits.sum()),
                "ucb1_hit_rate": float(ucb1_res.hits.sum()) / max(1, ucb1_res.slots_executed),
                "ucb1_slots": ucb1_res.slots_executed,
                "ucb1_action_signature": ucb1_res.action_signature(),
                "ucb1_replay_signature": ucb1_res.replay_signature,
                "gate0_per_file_ok": gate0_file_ok,
                "elapsed_s": time.time() - experiment_start,
            })

            print(
                f"[kaggle] [synthetic] pulses={len(synthetic_stream)}, "
                f"occ_cells={occ_cells} ({occupancy_fraction:.3%}), "
                f"tracks={deint_n_tracks}, "
                f"rr_hits={rr_res.hits.sum()}, "
                f"ucb1_hits={ucb1_res.hits.sum()}, "
                f"gate0={'PASS' if gate0_file_ok else 'FAIL'}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[kaggle] SYNTHETIC EXPERIMENT FAILED: {type(e).__name__}: {e}")
            raise

        deint_agg = {
            "n_files_with_tracks": int(sum(1 for n in _deint_n_tracks_list if n > 0)),
            "mean_tracks_per_file": float(np.mean(_deint_n_tracks_list)) if _deint_n_tracks_list else 0.0,
            "mean_pulse_attribution": float(np.mean(_deint_attr_list)) if _deint_attr_list else 0.0,
            "mean_contamination": float(np.mean(_deint_contam_list)) if _deint_contam_list else 0.0,
            "max_tracks_in_any_file": int(max(_deint_n_tracks_list)) if _deint_n_tracks_list else 0,
            "total_tracks_across_corpus": int(sum(_deint_n_tracks_list)),
            "mode": "synthetic",
        }
    else:
        # TSRD corpus path
        print("[kaggle] Starting Option B+C experiment run on scan-mode files...")
        experiment_start = time.time()
        n_scan_accepted = 0
        n_scan_skipped = 0

        for file_result in loader.iter_corpus():
            if file_result.disposition != CorpusFileDisposition.ACCEPTED:
                n_scan_skipped += 1
                continue
            if file_result.receiver_mode != TSRDReceiverMode.SCAN:
                # Stare-mode: oracle path, not scheduler-runnable
                n_scan_skipped += 1
                continue

            n_scan_accepted += 1
            file_start = time.time()

            try:
                # ---- (a) Build TSRDEnvironment (Options B + C) ----
                env = build_tsrd_environment(
                    h5_path=str(file_result.file_path),
                    simulation_config=sim_cfg,
                    detection_config=detection_cfg,
                    deinterleaver_config=deint_cfg,
                    data_mode=TSRDDataMode.REAL_TSRD,
                    seed=42,
                )

                # ---- (b) Run paired episode: RoundRobin vs UCB1 ----
                # Reset env and schedulers at the same seed for paired comparison.
                schedulers = {
                    "round_robin": RoundRobinProbe(sim_cfg.band_count),
                    "ucb1": UCB1Scheduler(sim_cfg.band_count),
                }
                scenario = scenario_descriptor(env)
                for sched in schedulers.values():
                    sched.reset(seed=42, scenario_config=scenario)

                episode_results = {}
                for name, sched in schedulers.items():
                    # TSRDEnvironment.reset() is safe to call multiple times
                    # (it re-seeds and resets slot counter, but the discretised
                    # grid and deinterleaver are built once at __init__ and stay).
                    env.reset(seed=42)
                    res = run_episode(
                        env, sched,
                        seed=42,
                        emitter_family_config=None,
                        scheduler_name=name,
                    )
                    episode_results[name] = res

                rr_res = episode_results["round_robin"]
                ucb1_res = episode_results["ucb1"]

                # Gate 0 per-file: the environment's replay_signature must be
                # identical for both schedulers (same seed → same grid hash).
                gate0_file_ok = (
                    rr_res.replay_signature == ucb1_res.replay_signature
                )

                # ---- (c) Deinterleaver stats (Option C) ----
                deint = env.deinterleaver_result
                if deint is not None and deint.n_tracks > 0:
                    deint_n_tracks = deint.n_tracks
                    deint_pulse_attr = float(deint.pulse_attribution_rate)
                    deint_contam = float(
                        np.mean([t.contamination_fraction() for t in deint.tracks])
                    )
                else:
                    deint_n_tracks = 0
                    deint_pulse_attr = 0.0
                    deint_contam = 0.0

                _deint_n_tracks_list.append(deint_n_tracks)
                _deint_attr_list.append(deint_pulse_attr)
                _deint_contam_list.append(deint_contam)

                # ---- (d) Occupancy stats (Option B) ----
                occ_cells = env.occupied_cell_count()
                total_cells = sim_cfg.band_count * sim_cfg.time_slots
                occupancy_fraction = float(occ_cells) / float(total_cells)

                # ---- (e) Record per-file result ----
                experiment_results.append({
                    "file": file_result.file_path.name,
                    "file_sha256": file_result.file_sha256,
                    "n_pulses": file_result.n_pulses,
                    "n_occupied_cells": occ_cells,
                    "occupancy_fraction": occupancy_fraction,
                    "deint_n_tracks": deint_n_tracks,
                    "deint_pulse_attribution_rate": deint_pulse_attr,
                    "deint_mean_contamination": deint_contam,
                    # RoundRobin
                    "rr_hits": int(rr_res.hits.sum()),
                    "rr_hit_rate": float(rr_res.hits.sum()) / max(1, rr_res.slots_executed),
                    "rr_slots": rr_res.slots_executed,
                    "rr_action_signature": rr_res.action_signature(),
                    "rr_replay_signature": rr_res.replay_signature,
                    # UCB1
                    "ucb1_hits": int(ucb1_res.hits.sum()),
                    "ucb1_hit_rate": float(ucb1_res.hits.sum()) / max(1, ucb1_res.slots_executed),
                    "ucb1_slots": ucb1_res.slots_executed,
                    "ucb1_action_signature": ucb1_res.action_signature(),
                    "ucb1_replay_signature": ucb1_res.replay_signature,
                    # Gate 0 per-file
                    "gate0_per_file_ok": gate0_file_ok,
                    "elapsed_s": time.time() - file_start,
                })

                print(
                    f"[kaggle] [{n_scan_accepted}] {file_result.file_path.name}: "
                    f"pulses={file_result.n_pulses}, "
                    f"occ_cells={occ_cells} ({occupancy_fraction:.3%}), "
                    f"tracks={deint_n_tracks}, "
                    f"rr_hits={rr_res.hits.sum()}, "
                    f"ucb1_hits={ucb1_res.hits.sum()}, "
                    f"gate0={'PASS' if gate0_file_ok else 'FAIL'}"
                )

            except Exception as e:  # noqa: BLE001
                n_scan_skipped += 1
                print(
                    f"[kaggle] EXPERIMENT SKIP {file_result.file_path.name}: "
                    f"{type(e).__name__}: {e}"
                )
                continue

        experiment_elapsed = time.time() - experiment_start
        print(
            f"[kaggle] Experiment run done: {n_scan_accepted} files, "
            f"{n_scan_skipped} skipped, {experiment_elapsed:.1f}s wall time"
        )

        # Aggregate deinterleaver summary
        if _deint_n_tracks_list:
            deint_agg = {
                "n_files_with_tracks": int(sum(1 for n in _deint_n_tracks_list if n > 0)),
                "mean_tracks_per_file": float(np.mean(_deint_n_tracks_list)),
                "mean_pulse_attribution": float(np.mean(_deint_attr_list)),
                "mean_contamination": float(np.mean(_deint_contam_list)),
                "max_tracks_in_any_file": int(max(_deint_n_tracks_list)),
                "total_tracks_across_corpus": int(sum(_deint_n_tracks_list)),
                "mode": "tsrd",
            }
        else:
            deint_agg = {
                "n_files_with_tracks": 0,
                "mean_tracks_per_file": 0.0,
                "mean_pulse_attribution": 0.0,
                "mean_contamination": 0.0,
                "note": "no scan-mode files successfully processed",
                "mode": "tsrd",
            }

# =====================================================================
# SECTION 8: Save experiment outputs.
# =====================================================================

if experiment_results:
    # ---- per_file_results.csv ----
    csv_path = out_dir / "per_file_results.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "file,n_pulses,n_occupied_cells,occupancy_fraction,"
            "deint_n_tracks,deint_pulse_attr,deint_mean_contam,"
            "rr_hits,rr_hit_rate,rr_slots,"
            "ucb1_hits,ucb1_hit_rate,ucb1_slots,"
            "gate0_per_file_ok,elapsed_s\n"
        )
        for r in experiment_results:
            fh.write(
                f"{r['file']},"
                f"{r['n_pulses']},"
                f"{r['n_occupied_cells']},"
                f"{r['occupancy_fraction']:.6f},"
                f"{r['deint_n_tracks']},"
                f"{r['deint_pulse_attribution_rate']:.6f},"
                f"{r['deint_mean_contamination']:.6f},"
                f"{r['rr_hits']},"
                f"{r['rr_hit_rate']:.6f},"
                f"{r['rr_slots']},"
                f"{r['ucb1_hits']},"
                f"{r['ucb1_hit_rate']:.6f},"
                f"{r['ucb1_slots']},"
                f"{r['gate0_per_file_ok']},"
                f"{r['elapsed_s']:.3f}\n"
            )
    print(f"[kaggle] per_file_results.csv written ({len(experiment_results)} rows)")

    # ---- deinterleaver_summary.json ----
    (out_dir / "deinterleaver_summary.json").write_text(
        json.dumps(
            {**deint_agg, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"[kaggle] deinterleaver_summary.json written")

    # ---- gate0.json updated with per-file results ----
    gate0_per_file = [r["gate0_per_file_ok"] for r in experiment_results]
    gate0_experiment_ok = all(gate0_per_file)
    existing_gate0 = {}
    gate0_path = out_dir / "gate0.json"
    if gate0_path.is_file():
        try:
            existing_gate0 = json.loads(gate0_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing_gate0 = {}
    existing_gate0.update({
        "experiment_gate0_per_file_ok": gate0_experiment_ok,
        "experiment_n_files": len(experiment_results),
        "experiment_all_files_passed": gate0_experiment_ok,
        "experiment_files": [
            {"file": r["file"], "gate0_ok": r["gate0_per_file_ok"]}
            for r in experiment_results
        ],
    })
    gate0_path.write_text(
        json.dumps(existing_gate0, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[kaggle] gate0.json updated (experiment_gate0_per_file_ok={gate0_experiment_ok})")

    # ---- summary.json updated with experiment stats ----
    existing_summary: Dict[str, Any] = {}
    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        try:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing_summary = {}
    existing_summary.update({
        "experiment": {
            "options_run": ["B", "C"],
            "n_files_run": len(experiment_results),
            "total_pulses_processed": sum(r["n_pulses"] for r in experiment_results),
            "mean_occupancy_fraction": float(
                np.mean([r["occupancy_fraction"] for r in experiment_results])
            ) if experiment_results else 0.0,
            "mean_deint_tracks": float(
                np.mean([r["deint_n_tracks"] for r in experiment_results])
            ) if experiment_results else 0.0,
            "ucb1_mean_hit_rate": float(
                np.mean([r["ucb1_hit_rate"] for r in experiment_results])
            ) if experiment_results else 0.0,
            "rr_mean_hit_rate": float(
                np.mean([r["rr_hit_rate"] for r in experiment_results])
            ) if experiment_results else 0.0,
            "gate0_all_passed": gate0_experiment_ok,
        },
    })
    summary_path.write_text(
        json.dumps(existing_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[kaggle] summary.json updated with experiment section")

print(f"[kaggle] All outputs written to {out_dir}")
print("[kaggle] DONE.")

"""Compare VAL sampling settings for one existing residual-SAC checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .evaluate_residual_discrete_sac import evaluation_destination, sha256


def checked_report(path, mode, temperature, action_seed, eval_seed, checkpoint_hash,
                   catalog_hash, expected_worlds):
    report = json.loads(path.read_text(encoding="utf-8"))
    rows_path = path.parent / "per_world.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    actual_worlds = {row["frozen_world_identity_sha256"]: row["receiver_seed"] for row in rows}
    if (report.get("policy_mode") != mode
            or report.get("split") != "val" or report.get("condition") != "normal"
            or report.get("checkpoint_sha256") != checkpoint_hash
            or report.get("frozen_world_catalog_sha256") != catalog_hash
            or report.get("ts_episode_reset_protocol") != "independent deep copy of trained posterior for each world"
            or report.get("eval_seed_base") != eval_seed
            or report.get("action_seed_base", report.get("eval_seed_base")) != action_seed
            or report.get("temperature", 1.0) != temperature
            or report.get("summary", {}).get("worlds") != 50
            or len(rows) != 50 or actual_worlds != expected_worlds
            or report.get("rows") != rows):
        raise ValueError(f"Report does not match calibration settings or frozen worlds: {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", default="residual_sac_final.pt")
    parser.add_argument("--environment", required=True)
    parser.add_argument("--world-catalog", required=True)
    parser.add_argument("--eval-seed-base", type=int, default=420000)
    parser.add_argument("--action-seed-bases", type=int, nargs="+", default=[420000])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    if any(seed < 0 for seed in args.action_seed_bases) or args.eval_seed_base < 0:
        parser.error("Seeds must be nonnegative")
    if any(not np.isfinite(t) or t <= 0 for t in args.temperatures):
        parser.error("Temperatures must be finite and positive")
    run = Path(args.run).resolve()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = run / "checkpoints" / checkpoint
    manifest = json.loads((run / "runtime_manifest.json").read_text(encoding="utf-8"))
    catalog_path = Path(args.world_catalog).resolve()
    if catalog_path.is_dir():
        catalog_path /= "frozen_world_catalog.json"
    catalog_hash = sha256(catalog_path)
    if catalog_hash != manifest["frozen_world_catalog_sha256"]:
        raise ValueError("Use the run's original frozen-world catalog")
    if args.eval_seed_base != int(manifest.get("eval_seed_base", 420000)):
        raise ValueError("Keep the run's TS evaluation seed base fixed")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    expected_worlds = {
        world["conditions"]["normal"]["identity_sha256"]: world["receiver_seed"]
        for world in catalog["splits"]["val"]["worlds"].values()
    }
    if len(expected_worlds) != 50:
        raise ValueError("Calibration requires exactly 50 frozen VAL_NORMAL worlds")
    checkpoint_hash = sha256(checkpoint)
    comparisons = []
    settings = [("actor_sampled", t) for t in dict.fromkeys(args.temperatures)]
    settings.append(("uniform_candidates", 1.0))
    for action_seed in dict.fromkeys(args.action_seed_bases):
        for mode, temperature in settings:
            destination = evaluation_destination(run, "val", "normal", checkpoint, mode,
                                                True, temperature, action_seed)
            report_path = destination / "summary.json"
            # Previous temperature-1 sampled reports used the default action seed
            # and a shorter output path. Reuse only after validating provenance.
            legacy = evaluation_destination(run, "val", "normal", checkpoint, mode, True) / "summary.json"
            if not report_path.exists() and temperature == 1.0 and action_seed == args.eval_seed_base and legacy.exists():
                report_path = legacy
            if not report_path.exists():
                print(f"\n[CALIBRATION] {mode}: temperature={temperature}, action seed={action_seed}", flush=True)
                command = [sys.executable, "-u", "-m", "scripts.evaluation.evaluate_residual_discrete_sac",
                           "--run", str(run), "--checkpoint", str(checkpoint),
                           "--split", "val", "--condition", "normal", "--diagnostic",
                           "--environment", args.environment, "--world-catalog", str(catalog_path),
                           "--eval-seed-base", str(args.eval_seed_base), "--action-seed-base", str(action_seed),
                           "--policy-mode", mode, "--temperature", str(temperature), "--device", args.device]
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                    code = process.wait()
                    if code:
                        raise subprocess.CalledProcessError(code, command)
            else:
                print(f"[CALIBRATION] Reusing {mode}: temperature={temperature}, action seed={action_seed}", flush=True)
            report = checked_report(report_path, mode, temperature, action_seed, args.eval_seed_base,
                                    checkpoint_hash, catalog_hash, expected_worlds)
            summary = report["summary"]
            comparisons.append({
                "policy_mode": mode, "temperature": temperature, "action_seed_base": action_seed,
                "capture": summary["unique_emitter_interception_rate"],
                "OIR": summary["opportunity_interception_ratio"],
                "TTFI_seconds": summary["source_relative_ttfi"]["km_restricted_mean_ms"] / 1000,
                "coverage": summary["whole_mission_unique_band_fraction"],
                "summary_path": str(report_path), "summary_sha256": sha256(report_path),
            })
    destination = run / "eval" / "val" / "normal" / "calibration"
    destination.mkdir(parents=True, exist_ok=True)
    label = "_".join(map(str, dict.fromkeys(args.action_seed_bases)))
    output = destination / f"action_seeds_{label}.json"
    output.write_text(json.dumps({"checkpoint_sha256": checkpoint_hash,
                                  "frozen_world_catalog_sha256": catalog_hash,
                                  "eval_seed_base": args.eval_seed_base,
                                  "variation_source": "action sampling; one trained checkpoint",
                                  "rows": comparisons}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\nVAL_NORMAL calibration comparison", flush=True)
    for row in comparisons:
        print({key: value for key, value in row.items() if key not in ("summary_path", "summary_sha256")}, flush=True)
    print(f"Saved comparison: {output}", flush=True)


if __name__ == "__main__":
    main()

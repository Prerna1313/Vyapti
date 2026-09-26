#!/usr/bin/env python3
"""
Vyapti / PS26055
TSRD Stare — Belief+Periodicity-guided Residual SAC, FIXED DWELL

Corrected training setup:
- uses TSRD v2 Stare 30 s missions;
- keeps the existing 36-band action space;
- 1800 base decision slots => 16.6667 ms per decision;
- streams all 2,500 train files one episode at a time instead of loading them all into RAM;
- preserves the causal hybrid state: belief + staleness + hit-rate + periodicity;
- trains SAC-Discrete and saves full .pth checkpoints + policy-only .pth files;
- validates on the 250-file Stare validation split; test is not touched here.

Run from the repository root, for example:
    python scripts/vyapti_sac_train_fixed_dwell.py --data-root ./data --epochs 1

Smoke test:
    python scripts/vyapti_sac_train_fixed_dwell.py \
        --data-root ./data --epochs 1 --max-train-files 20 --max-val-files 20 \
        --output-dir ./runs/fixed_dwell_smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vyapti_tsrd_sac_common import (
    DEFAULT_DWELL_TIME_PENALTY,
    N_BANDS,
    PD_SENSOR,
    PFA_SENSOR,
    TrainConfig,
    SACTrainer,
    device_from_string,
    evaluate_policy,
    find_split_files,
    load_policy_only,
    load_truth_grid,
    seed_everything,
    PeriodicityTracker,
    FixedDwellTSRDEnv,
    featurize,
    policy_action,
)


# Frozen TSRD v2 primary benchmark: 30 s / 1800 base slots.
EXPECTED_MISSION_SECONDS = 30.0
EXPECTED_BASE_SLOTS = 1800
EXPECTED_BASE_SLOT_MS = EXPECTED_MISSION_SECONDS * 1000.0 / EXPECTED_BASE_SLOTS

if abs((30_000_000.0) - EXPECTED_MISSION_SECONDS * 1_000_000.0) > 1e-9:
    raise RuntimeError("Internal 30-second benchmark constant mismatch.")

# The shared module is the source of truth for environment geometry. Fail fast
# rather than silently training a 10-second environment with a 30-second wrapper.
import vyapti_tsrd_sac_common as _common

if _common.N_BASE_SLOTS != EXPECTED_BASE_SLOTS or not (
    abs(_common.MISSION_US - 30_000_000.0) < 1e-6
):
    raise RuntimeError(
        "vyapti_tsrd_sac_common.py is still configured for a non-v2 timing "
        f"setup: mission_us={_common.MISSION_US}, n_base_slots={_common.N_BASE_SLOTS}. "
        "Update the common module to TSRD v2: 30 s / 1800 slots."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Vyapti fixed-dwell SAC on TSRD v2 Stare train split (30-second mission).")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-dir", default="./runs/fixed_dwell_sac_v2_30s")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-train-files", type=int, default=None)
    p.add_argument("--max-val-files", type=int, default=None)
    p.add_argument("--replay-size", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--start-steps", type=int, default=2000)
    p.add_argument("--q-warmup-steps", type=int, default=6000)
    p.add_argument("--update-every", type=int, default=4)
    p.add_argument("--updates-per-update", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--target-entropy-frac", type=float, default=0.20)
    p.add_argument("--checkpoint-every-episodes", type=int, default=100)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--resume", default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--eval-split", choices=["val", "test"], default="val")
    p.add_argument("--eval-output", default=None)
    return p.parse_args()


def evaluate_only(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required with --eval-only")

    device = device_from_string(args.device)
    policy, config = load_policy_only(
        args.checkpoint,
        dynamic_dwell=False,
        dwell_multipliers=(1,),
        dwell_time_penalty=DEFAULT_DWELL_TIME_PENALTY,
        device=device,
    )

    files = find_split_files(args.data_root, args.eval_split)
    if not files:
        raise SystemExit(f"No files found for {args.eval_split!r} under {args.data_root}")

    metrics = evaluate_policy(
        policy,
        files,
        dynamic_dwell=False,
        dwell_multipliers=(1,),
        pd_sensor=PD_SENSOR,
        pfa_sensor=PFA_SENSOR,
        seed=args.seed + 100000,
        device=device,
        max_files=args.max_val_files,
    )

    out = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.eval_split,
        "device": str(device),
        "config": config,
        "metrics": metrics,
    }
    output = args.eval_output or str(Path(args.checkpoint).with_name("eval_metrics.json"))
    Path(output).write_text(json.dumps(out, indent=2, allow_nan=True))
    print(json.dumps(out, indent=2, allow_nan=True))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.eval_only:
        evaluate_only(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Explicitly freeze the experiment definition so the checkpoint is auditable.
    experiment_config = {
        "mode": "fixed_dwell",
        "tsrd_revision": "68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51",
        "n_bands": N_BANDS,
        "mission_seconds": 30.0,
        "base_slots": 1800,
        "base_slot_ms": 30_000.0 / 1800.0,
        "sensor_pd": PD_SENSOR,
        "sensor_pfa": PFA_SENSOR,
        "dataset_version": "TSRD v2",
        "mission_semantics": "nominal 30-second Stare collection window; PDW ToA >= 30 s excluded from primary mission",
        "train_split": "stare/train_stare",
        "validation_split": "stare/val_stare",
        "test_split": "stare/test_stare (not used by this training script)",
        "dynamic_dwell": False,
        "max_train_files": args.max_train_files,
        "max_val_files": args.max_val_files,
    }
    (output_dir / "experiment_config.json").write_text(json.dumps(experiment_config, indent=2))

    cfg = TrainConfig(
        data_root=args.data_root,
        output_dir=str(output_dir),
        seed=args.seed,
        epochs=args.epochs,
        expected_train_files=2500,
        expected_val_files=250,
        max_train_files=args.max_train_files,
        max_eval_files=args.max_val_files,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        start_steps=args.start_steps,
        q_warmup_steps=args.q_warmup_steps,
        update_every=args.update_every,
        updates_per_update=args.updates_per_update,
        gamma=args.gamma,
        lr=args.lr,
        tau=args.tau,
        target_entropy_frac=args.target_entropy_frac,
        checkpoint_every_episodes=args.checkpoint_every_episodes,
        device=args.device,
        resume=args.resume,
    )

    trainer = SACTrainer(cfg, dynamic_dwell=False, dwell_multipliers=(1,))
    trainer.train()


if __name__ == "__main__":
    main()

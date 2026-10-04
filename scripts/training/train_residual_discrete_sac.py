"""Train the imported conservative residual Discrete-SAC learner on TRAIN-250."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from vyapti_simulator.system_b.tsrd import residual_sac_train250_adapter as adapter
from vyapti_simulator.system_b.tsrd.policies import residual_discrete_sac as learner


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default="training_setup/environments/train250_composed.json")
    parser.add_argument("--world-catalog", default="runs/round_robin/frozen_world_catalog.json",
                        help="Shared frozen_world_catalog.json, or its containing run directory")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--eval-seed-base", type=int, default=420000,
                        help="Fixed base for policy-side evaluation sampling across algorithms")
    parser.add_argument("--max-actions", type=int, default=400_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--run", required=True, help="New output folder, normally runs/residual_discrete_sac")
    args = parser.parse_args()

    output = Path(args.run).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new run directory; refusing to overwrite {output}")
    env_path = Path(args.environment).resolve()
    world_catalog = Path(args.world_catalog).resolve()
    if args.max_actions < 1 or args.seed < 0 or args.eval_seed_base < 0:
        raise ValueError("max-actions must be positive and seeds nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    learner.DEVICE = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    os.environ["VYAPTI_RESIDUAL_ENVIRONMENT"] = str(env_path)
    os.environ["VYAPTI_RESIDUAL_WORLD_CATALOG"] = str(world_catalog)
    factory = adapter.Train250WorldFactory()

    output.mkdir(parents=True)
    manifest = {
        "algorithm": "residual_discrete_sac",
        "uploaded_source": "vyapti_residual_discrete_sac_train250.py",
        "learner_source_sha256": sha256(Path(learner.__file__).resolve()),
        "seed": args.seed,
        "eval_seed_base": args.eval_seed_base,
        "max_actions": args.max_actions,
        "device": str(learner.DEVICE),
        "environment_spec": str(env_path),
        "environment_spec_sha256": sha256(env_path),
        "world_catalog_path": str(factory.catalog_path),
        "frozen_world_catalog_sha256": factory.catalog_sha256,
        "dataset_source_hashes": {
            split: factory.catalog["splits"][split]["source_pool_sha256"]
            for split in ("val", "test")
        },
        "policy_logic_modified": True,
        "policy_logic_change_note": "Contextual-TS posterior learns observed HIT/MISS rate; SAC reward target, actor, critics, replay and updates are unchanged",
        "ts_feedback_schema": learner.TS_FEEDBACK_SCHEMA,
        "ts_feedback_scale": learner.TS_FEEDBACK_SCALE,
        "receiver_seed_protocol": "VAL/TEST receiver RNG is pinned per world in the shared frozen catalog",
    }
    (output / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"[TRAIN] Residual Discrete SAC: max_actions={args.max_actions:,}; "
        f"seed={args.seed}; device={learner.DEVICE}; "
        f"VAL/TEST catalog={factory.catalog_sha256}",
        flush=True,
    )
    learner.train(
        factory=factory,
        output_dir=output,
        seed=args.seed,
        transition=learner.make_calibrated_transition(),
        prior_active=np.full(learner.N_BANDS, learner.CAL_PRIOR_ACTIVE, dtype=np.float64),
        max_actions=args.max_actions,
        eval_seed_base=args.eval_seed_base,
    )


if __name__ == "__main__":
    main()

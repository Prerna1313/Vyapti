"""Evaluate one residual-SAC checkpoint on shared frozen VAL/TEST worlds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from vyapti_simulator.system_b.tsrd import residual_sac_train250_adapter as adapter
from vyapti_simulator.system_b.tsrd.experiment import _summary
from vyapti_simulator.system_b.tsrd.policies import residual_discrete_sac as learner


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path or filename under RUN/checkpoints")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--condition", choices=("normal", "beam_periodic", "beam_stochastic"), default="normal")
    parser.add_argument("--final", action="store_true", help="Required to evaluate TEST")
    parser.add_argument("--environment", default="training_setup/environments/train250_composed.json")
    parser.add_argument("--world-catalog", default="runs/round_robin/frozen_world_catalog.json")
    parser.add_argument("--eval-seed-base", type=int, default=420000,
                        help="Fixed policy-side RNG base, independent of the training seed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    run = Path(args.run).resolve()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (run / "checkpoints" / checkpoint).resolve()
    if args.split == "test" and not args.final:
        parser.error("TEST evaluation requires --final")
    if args.eval_seed_base < 0:
        parser.error("--eval-seed-base must be nonnegative")
    manifest = json.loads((run / "runtime_manifest.json").read_text(encoding="utf-8"))
    recorded_eval_seed = int(manifest.get("eval_seed_base", 420000))
    if args.eval_seed_base != recorded_eval_seed:
        raise ValueError(
            f"Evaluation seed base {args.eval_seed_base} differs from this run's "
            f"frozen setting {recorded_eval_seed}; use the recorded base for comparable results"
        )
    if sha256(Path(learner.__file__).resolve()) != manifest["learner_source_sha256"]:
        raise ValueError("Residual-SAC learner source changed since training")
    if sha256(checkpoint) not in {
        sha256(p) for p in (run / "checkpoints").glob("*.pt")
    }:
        raise ValueError("Checkpoint is not present in this run's checkpoints folder")
    if args.split == "test":
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        if selection.get("checkpoint_sha256") != sha256(checkpoint):
            raise ValueError("TEST may only use the checkpoint frozen from VAL")
        val_report = (
            run / "eval" / "val" / "normal" / "checkpoints"
            / Path(selection["checkpoint_file"]).stem / "summary.json"
        )
        if not val_report.is_file() or sha256(val_report) != selection.get("validation_summary_sha256"):
            raise ValueError("The selected VAL report is missing or changed after checkpoint selection")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    learner.DEVICE = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    os.environ["VYAPTI_RESIDUAL_ENVIRONMENT"] = str(Path(args.environment).resolve())
    os.environ["VYAPTI_RESIDUAL_WORLD_CATALOG"] = str(Path(args.world_catalog).resolve())
    os.environ["VYAPTI_RESIDUAL_CONDITION"] = args.condition
    factory = adapter.Train250WorldFactory()

    if args.split == "test":
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        if selection.get("frozen_world_catalog_sha256") != factory.catalog_sha256:
            raise ValueError("TEST world catalog differs from the catalog recorded at selection")

    recipes = factory.load_val_recipes() if args.split == "val" else factory.load_test_recipes()
    result = learner.evaluate_checkpoint(
        factory, checkpoint, args.split, recipes, eval_seed_base=args.eval_seed_base
    )
    scorecards = [row for row in result["rows"] if "cell_level" in row]
    if len(scorecards) != len(recipes):
        raise ValueError("Some evaluation worlds did not produce a benchmark scorecard")
    result["summary"] = _summary(scorecards)
    result["condition"] = args.condition
    result["checkpoint_sha256"] = sha256(checkpoint)
    result["frozen_world_catalog_sha256"] = factory.catalog_sha256
    result["action_mode"] = "actor_argmax_with_seeded_contextual_ts_proposals"
    result["receiver_rng_source"] = "receiver_seed pinned by shared frozen-world catalog"
    result["protocol_run"] = str(factory.protocol_run)
    result["per_world_identity_check"] = "every world rebuilt and hash-verified against frozen catalog"

    destination = run / "eval" / args.split / args.condition / "checkpoints" / checkpoint.stem
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    (destination / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (destination / "per_world.jsonl").open("w", encoding="utf-8") as stream:
        for row in result["rows"]:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps({
        "split": args.split,
        "condition": args.condition,
        "checkpoint": checkpoint.name,
        "worlds": result["summary"]["worlds"],
        "unique_emitter_interception_rate": result["summary"]["unique_emitter_interception_rate"],
        "OIR": result["summary"]["opportunity_interception_ratio"],
        "Pd": result["summary"]["conditional_pd"],
        "Pfa": result["summary"]["true_pfa"],
        "saved_to": str(destination),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

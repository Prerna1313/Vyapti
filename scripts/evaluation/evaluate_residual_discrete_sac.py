"""Evaluate one residual-SAC checkpoint on shared frozen VAL/TEST worlds."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
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


class DiagnosticPolicy:
    """Evaluation-only choices; checkpoint networks and learner stay unchanged."""

    def __init__(self, agent, mode: str, seed: int, temperature: float = 1.0):
        if mode not in ("actor_argmax", "ts_only", "actor_sampled", "uniform_candidates"):
            raise ValueError(mode)
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        self.agent = agent
        self.mode = mode
        self.temperature = float(temperature)
        # Separate stream from Thompson proposal sampling and receiver noise.
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))

    @torch.no_grad()
    def select_eval_action(self, observation, candidates):
        if self.mode == "actor_argmax":
            return self.agent.select_eval_action(observation, candidates)
        if self.mode == "ts_only":
            return 0, {"q_advantage": 0.0, "policy_prob": 1.0, "used_prior": 1.0}
        x = torch.as_tensor(observation, dtype=torch.float32, device=learner.DEVICE).unsqueeze(0)
        c = torch.as_tensor(candidates, dtype=torch.float32, device=learner.DEVICE).unsqueeze(0)
        if self.mode == "uniform_candidates":
            p = np.full(len(candidates), 1.0 / len(candidates), dtype=np.float64)
        else:
            probs, log_probs = self.agent.policy_distribution(x, c, prior_bias=0.0)
            if self.temperature != 1.0:
                if log_probs is None:
                    log_probs = probs.double().log()
                probs = torch.softmax(log_probs.double() / self.temperature, dim=-1)
            p = probs[0].cpu().numpy().astype(np.float64)
        p /= p.sum()
        choice = int(self.rng.choice(len(p), p=p))
        q = torch.minimum(self.agent.q1.all_values(x, c), self.agent.q2.all_values(x, c))[0]
        return choice, {
            "q_advantage": float((q[choice] - q[0]).item()),
            "policy_prob": float(p[choice]),
            "used_prior": float(choice == 0),
        }


def evaluate_ablation(factory, checkpoint, recipes, mode, eval_seed_base, split="val",
                      temperature=1.0, action_seed_base=None):
    action_seed_base = eval_seed_base if action_seed_base is None else action_seed_base
    agent, transition, prior, dwell, trained_ts = learner.load_agent(checkpoint)
    # load_state_dict uses np.asarray and can alias its input arrays. Preserve
    # the trained posterior and copy it again for each world's online updates.
    trained_posterior = deepcopy(trained_ts.state_dict())
    rows, diagnostics = [], []
    for i, recipe in enumerate(recipes):
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(recipes):
            print(f"[EVAL {mode}] {split.upper()} world {i + 1}/{len(recipes)} "
                  f"{recipe.get('catalog_world_id', i)}", flush=True)
        env = (factory.make_val_world(recipe) if split == "val"
               else factory.make_test_world(recipe))
        receiver_seed = int(recipe["receiver_seed"])
        policy_seed = int(eval_seed_base + i)
        env.reset(seed=receiver_seed)
        ts = learner.ContextualThompsonSampler(learner.N_BANDS, seed=policy_seed)
        ts.load_state_dict(deepcopy(trained_posterior))
        action_seed = int(action_seed_base + i)
        policy = DiagnosticPolicy(agent, mode, action_seed, temperature)
        trajectory, diag = learner.rollout_policy_on_world(
            env, policy, transition, prior, dwell, ts,
        )
        metric = dict(factory.score_episode(env, trajectory))
        metric.update(world_id=int(recipe.get("world_id", i)),
                      receiver_seed=receiver_seed, policy_eval_seed=policy_seed)
        if mode in ("actor_sampled", "uniform_candidates"):
            metric["actor_action_seed"] = action_seed
        rows.append(metric)
        diagnostics.append(diag)
    return {"split": split, "checkpoint": str(checkpoint),
            "eval_seed_base": eval_seed_base, "n_worlds": len(rows),
            "action_seed_base": action_seed_base, "temperature": float(temperature),
            "rows": rows, "diagnostics": diagnostics}


def evaluation_destination(run, split, condition, checkpoint, mode,
                           diagnostic=False, temperature=1.0, action_seed_base=None):
    destination = Path(run) / "eval" / split / condition
    if diagnostic or mode != "actor_argmax":
        destination = destination / "ablations" / mode
        if temperature != 1.0 or action_seed_base is not None:
            destination = destination / f"temperature_{float(temperature)}"
            if action_seed_base is not None:
                destination = destination / f"action_seed_{action_seed_base}"
    return destination / "checkpoints" / Path(checkpoint).stem


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
    parser.add_argument("--policy-mode", choices=("actor_argmax", "ts_only", "actor_sampled", "uniform_candidates"),
                        default="actor_argmax", help="VAL diagnostics: TS proposal only or sampled actor")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Save VAL reports separately, including a fresh actor-argmax reference")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="VAL sampled-actor logit temperature; independent of training alpha")
    parser.add_argument("--action-seed-base", type=int,
                        help="VAL action sampling seed base; TS and receiver seeds stay fixed")
    args = parser.parse_args()

    run = Path(args.run).resolve()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (run / "checkpoints" / checkpoint).resolve()
    if args.split == "test" and not args.final:
        parser.error("TEST evaluation requires --final")
    if args.split == "test" and (args.policy_mode != "actor_argmax" or args.diagnostic):
        parser.error("Diagnostic policy modes are available only on VAL")
    if args.eval_seed_base < 0:
        parser.error("--eval-seed-base must be nonnegative")
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    if args.temperature != 1.0 and args.policy_mode != "actor_sampled":
        parser.error("Temperature tuning requires --policy-mode actor_sampled")
    if args.action_seed_base is not None and (
            args.action_seed_base < 0 or args.policy_mode not in ("actor_sampled", "uniform_candidates")):
        parser.error("--action-seed-base must be nonnegative and used with a sampling mode")
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
    if factory.catalog_sha256 != manifest["frozen_world_catalog_sha256"]:
        raise ValueError("Evaluation catalog differs from the run's frozen catalog")

    if args.split == "test":
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        if selection.get("frozen_world_catalog_sha256") != factory.catalog_sha256:
            raise ValueError("TEST world catalog differs from the catalog recorded at selection")

    recipes = factory.load_val_recipes() if args.split == "val" else factory.load_test_recipes()
    diagnostic = args.diagnostic or args.policy_mode != "actor_argmax"
    destination = evaluation_destination(run, args.split, args.condition, checkpoint,
                                        args.policy_mode, diagnostic, args.temperature,
                                        args.action_seed_base)
    if destination.exists():
        raise FileExistsError(destination)
    result = evaluate_ablation(factory, checkpoint, recipes, args.policy_mode,
                              args.eval_seed_base, split=args.split,
                              temperature=args.temperature, action_seed_base=args.action_seed_base)
    scorecards = [row for row in result["rows"] if "cell_level" in row]
    if len(scorecards) != len(recipes):
        raise ValueError("Some evaluation worlds did not produce a benchmark scorecard")
    result["summary"] = _summary(scorecards)
    result["condition"] = args.condition
    result["checkpoint_sha256"] = sha256(checkpoint)
    result["frozen_world_catalog_sha256"] = factory.catalog_sha256
    result["policy_mode"] = args.policy_mode
    result["action_mode"] = {
        "actor_argmax": "actor_argmax_with_seeded_contextual_ts_proposals",
        "ts_only": "seeded_contextual_ts_proposal_only",
        "actor_sampled": "seeded_actor_sampling_with_seeded_contextual_ts_proposals",
        "uniform_candidates": "seeded_uniform_sampling_over_causal_candidates",
    }[args.policy_mode]
    result["diagnostic_only"] = diagnostic
    result["ts_episode_reset_protocol"] = "independent deep copy of trained posterior for each world"
    if args.policy_mode in ("actor_sampled", "uniform_candidates"):
        result["actor_rng_protocol"] = "per-world numpy SeedSequence([action_seed_base + world_index, 1]); independent of TS and receiver"
    result["receiver_rng_source"] = "receiver_seed pinned by shared frozen-world catalog"
    result["protocol_run"] = str(factory.protocol_run)
    result["per_world_identity_check"] = "every world rebuilt and hash-verified against frozen catalog"

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
        "policy_mode": args.policy_mode,
        "temperature": args.temperature,
        "action_seed_base": result["action_seed_base"],
        "worlds": result["summary"]["worlds"],
        "unique_emitter_interception_rate": result["summary"]["unique_emitter_interception_rate"],
        "OIR": result["summary"]["opportunity_interception_ratio"],
        "Pd": result["summary"]["conditional_pd"],
        "Pfa": result["summary"]["true_pfa"],
        "saved_to": str(destination),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

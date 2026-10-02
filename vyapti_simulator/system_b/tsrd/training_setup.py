"""Resolve environment, held-out worlds, algorithm, and run budget separately."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def _read(path: Path):
    content = path.read_bytes()
    return json.loads(content), {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def resolve_setup(environment_path, algorithm_path, *, seed: int, episodes: int,
                  checkpoint_every: int, save_initial: bool = False) -> dict:
    environment_path = Path(environment_path).resolve()
    algorithm_path = Path(algorithm_path).resolve()
    environment, env_source = _read(environment_path)
    algorithm, algo_source = _read(algorithm_path)
    if environment.get("schema") != "vyapti_environment_spec_v1":
        raise ValueError("Unknown environment spec")
    if algorithm.get("schema") != "vyapti_algorithm_spec_v1":
        raise ValueError("Unknown algorithm spec")
    evaluation_path = (environment_path.parent / environment["evaluation_spec"]).resolve()
    evaluation, eval_source = _read(evaluation_path)
    if evaluation.get("schema") != "vyapti_evaluation_spec_v1":
        raise ValueError("Unknown evaluation spec")
    if any(isinstance(x, bool) or int(x) != x for x in (seed, episodes, checkpoint_every)):
        raise ValueError("Seed, episodes, and checkpoint interval must be integers")
    if seed < 0 or episodes < 1 or checkpoint_every < 1:
        raise ValueError("Set a nonnegative seed and positive episode/checkpoint budgets")
    extension = algorithm.get("checkpoint_extension")
    if not isinstance(extension, str) or re.fullmatch(r"\.[A-Za-z0-9]+", extension) is None:
        raise ValueError("Algorithm must declare a simple checkpoint extension")
    result = {key: value for key, value in environment.items()
              if key not in ("schema", "evaluation_spec")}
    result["evaluation"] = {key: value for key, value in evaluation.items() if key != "schema"}
    protocol_spec = evaluation.get("analysis_protocol_spec")
    if protocol_spec:
        protocol, protocol_source = _read((evaluation_path.parent / protocol_spec).resolve())
        if protocol.get("schema") != "mode_b_evaluation_protocol":
            raise ValueError("Unknown Mode-B analysis protocol")
        result["analysis_protocol"] = protocol
    mode = result.pop("evaluation_mode", result["evaluation"].get("world_mode", "source_replay"))
    result["evaluation"]["world_mode"] = mode
    for split in ("val", "test"):
        if mode == "source_replay":
            result["evaluation"][f"{split}_composed_worlds"] = []
        else:
            for recipe in result["evaluation"].get(f"{split}_composed_worlds", []):
                recipe.setdefault("source_config_ids", list(result["evaluation"][f"{split}_config_ids"]))
    result["algorithm"] = {key: value for key, value in algorithm.items() if key != "schema"}
    result.update(seed=int(seed), train_episodes=int(episodes), checkpoint_file="final" + extension,
                  checkpointing={"interval_episodes": int(checkpoint_every), "save_initial": bool(save_initial)},
                  setup_sources={"environment": env_source, "evaluation": eval_source, "algorithm": algo_source},
                  checkpoint_selection_required=True)
    if protocol_spec:
        result["setup_sources"]["analysis_protocol"] = protocol_source
    for key in ("data_root", "cache_root"):
        result[key] = str((environment_path.parent / result[key]).resolve())
    return result


def resolve_plan(path) -> dict:
    plan_path = Path(path).resolve()
    plan, source = _read(plan_path)
    if plan.get("schema") != "vyapti_run_plan_v1":
        raise ValueError("Unknown run plan")
    result = resolve_setup(plan_path.parent / plan["environment"], plan_path.parent / plan["algorithm"],
                           seed=plan["seed"], episodes=plan["episodes"],
                           checkpoint_every=plan["checkpoint_every"], save_initial=plan.get("save_initial", False))
    result["setup_sources"]["plan"] = source
    return result

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
                  checkpoint_every: int, save_initial: bool = False,
                  execution_mode: str = "training") -> dict:
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
    if (seed < 0 or checkpoint_every < 1 or execution_mode not in {"training", "online_baseline"}
            or (execution_mode == "training" and episodes < 1)
            or (execution_mode == "online_baseline" and episodes != 0)):
        raise ValueError("Set a nonnegative seed and a positive training budget, or zero episodes for an online baseline")
    extension = algorithm.get("checkpoint_extension", ".json")
    if execution_mode == "training" and (
            not isinstance(extension, str) or re.fullmatch(r"\.[A-Za-z0-9]+", extension) is None):
        raise ValueError("Training algorithms need a simple checkpoint extension")
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
    result["evaluation"]["require_frozen_world_catalog"] = (
        bool(evaluation.get("require_frozen_world_catalog", False)) and mode == "emitter_recombined")
    for split in ("val", "test"):
        if mode == "source_replay":
            result["evaluation"][f"{split}_composed_worlds"] = []
        else:
            for recipe in result["evaluation"].get(f"{split}_composed_worlds", []):
                recipe.setdefault("source_config_ids", list(result["evaluation"][f"{split}_config_ids"]))
    belief_model = None
    belief_model_source = None
    belief_model_spec = algorithm.get("belief_model_spec")
    if belief_model_spec is not None:
        if not isinstance(belief_model_spec, str) or not belief_model_spec:
            raise ValueError("belief_model_spec must be a relative calibration JSON path")
        belief_path = (algorithm_path.parent / belief_model_spec).resolve()
        belief_model, belief_model_source = _read(belief_path)
        if belief_model.get("schema") != "vyapti_band_activity_hmm_calibration_v1":
            raise ValueError("Unknown band-activity HMM calibration schema")
        probability_fields = (
            "prior_active_probability",
            "inactive_to_active_probability",
            "active_to_active_probability",
        )
        for field in probability_fields:
            value = float(belief_model[field])
            if not 0.0 < value < 1.0:
                raise ValueError(f"HMM calibration {field} must be in (0, 1)")
    result["algorithm"] = {
        key: value for key, value in algorithm.items()
        if key not in ("schema", "belief_model_spec")
    }
    if belief_model is not None:
        settings = result["algorithm"].setdefault("settings", {})
        for field in ("prior_active_probability", "inactive_to_active_probability",
                      "active_to_active_probability"):
            settings[field] = float(belief_model[field])
        result["belief_model_calibration"] = belief_model
    if algorithm.get("name") == "ppo_lstm" and execution_mode == "training":
        result["algorithm"].setdefault("settings", {})["training_episodes_target"] = int(episodes)
    if algorithm.get("name") in {"discrete_sac", "ppo_lstm", "belief_ucb"}:
        policy_settings = algorithm.get("settings", {})
        receiver = result["receiver"]
        paired_values = (
            ("detection_probability", receiver.get("detection_probability")),
            ("false_alarm_probability", receiver.get("false_alarm_probability")),
        )
        for setting_name, receiver_value in paired_values:
            if receiver_value is None or float(policy_settings.get(setting_name, float("nan"))) != float(receiver_value):
                raise ValueError(f"{algorithm['name']} {setting_name} must match the frozen receiver operating point")
        if float(policy_settings.get("base_slot_seconds", float("nan"))) != float(receiver["base_slot_duration_ms"]) / 1000:
            raise ValueError(f"{algorithm['name']} base_slot_seconds must match the receiver's base slot")
    result.update(seed=int(seed), train_episodes=int(episodes), execution_mode=execution_mode,
                  setup_sources={"environment": env_source, "evaluation": eval_source, "algorithm": algo_source},
                  val_selection_required=True)
    if belief_model_source is not None:
        result["setup_sources"]["belief_model"] = belief_model_source
    if execution_mode == "training":
        result.update(checkpoint_file="final" + extension,
                      checkpointing={"interval_episodes": int(checkpoint_every), "save_initial": bool(save_initial)})
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

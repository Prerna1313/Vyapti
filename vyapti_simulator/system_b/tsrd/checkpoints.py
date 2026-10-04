"""Checkpoint audit records and a single frozen VAL-based selection."""
import hashlib
import json
from pathlib import Path

from .algorithm_interface import algorithm_fingerprint


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    if not isinstance(name, str) or not name or name in (".", "..", "index.json") or any(c in name for c in "/\\:"):
        raise ValueError("Checkpoint must be a filename inside checkpoints/")
    return name


def save_checkpoint(run, algorithm, name, *, episodes, steps, kind):
    directory = Path(run) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / safe_name(name)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_name(target.stem + ".tmp" + target.suffix)
    algorithm.save(temporary)
    temporary.replace(target)
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"schema": "vyapti_checkpoint_index_v1", "checkpoints": []}
    record = {"file": name, "completed_episodes": episodes, "receiver_steps": steps,
              "kind": kind, "sha256": file_hash(target)}
    index["checkpoints"].append(record)
    pending = index_path.with_suffix(".tmp")
    pending.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    pending.replace(index_path)
    return record


def resolve_checkpoint(run, config, status, name=None):
    name = safe_name(name or config.get("checkpoint_file", "final.json"))
    index_path = Path(run) / "checkpoints/index.json"
    if index_path.exists():
        if status.get("checkpoint_index_sha256") != file_hash(index_path):
            raise ValueError("Checkpoint index changed after training")
        records = json.loads(index_path.read_text())["checkpoints"]
        matches = [r for r in records if r["file"] == name]
        if len(matches) != 1:
            raise ValueError("Checkpoint is not in this run's audit index")
        expected_hash = matches[0]["sha256"]
    else:
        if name != config.get("checkpoint_file", "final.json"):
            raise ValueError("Legacy run contains only its final audited checkpoint")
        expected_hash = status["checkpoint_sha256"]
    path = Path(run) / "checkpoints" / name
    if file_hash(path) != expected_hash:
        raise ValueError("Checkpoint changed after training")
    return path, expected_hash


def evaluation_directory(run, config, name, split, condition, *, action_mode="greedy", action_seed=None):
    root = Path(run) / "eval" / split
    if action_mode != "greedy":
        if action_seed is None:
            raise ValueError("Sampled evaluation output requires an action seed")
        root = root / "action_modes" / action_mode / f"seed_{int(action_seed)}"
    if name is not None and name != config.get("checkpoint_file", "final.json"):
        root = root / "checkpoints" / Path(name).stem
    return root / condition


def _ppo_lstm_selection_key(report):
    """Lexicographic VAL preference from primary discovery through reward."""
    summary = report.get("summary", {})
    ttfi = summary.get("source_relative_ttfi", {})

    def higher(value):
        return float(value) if value is not None else float("-inf")

    def lower(value):
        return -float(value) if value is not None else float("-inf")

    return (
        higher(summary.get("unique_emitter_interception_rate")),
        lower(ttfi.get("km_restricted_mean_ms")),
        lower(ttfi.get("censored_fraction")),
        higher(summary.get("mean_rolling_coverage")),
        higher(summary.get("fraction_worlds_reaching_90pct_band_coverage")),
        higher(summary.get("opportunity_interception_ratio")),
        higher(summary.get("policy_reward_per_decision")),
    )


def freeze_selection(run_dir, name=None, *, baseline=False,
                     reason="Selected using frozen VAL_NORMAL results"):
    run = Path(run_dir).resolve()
    config_path = run / "config.json"
    config = json.loads(config_path.read_text())
    status = json.loads((run / "status.json").read_text())
    if status.get("state") != "complete" or status.get("config_sha256") != file_hash(config_path):
        raise ValueError("Selection requires a complete run with its original config")
    target = run / "selection.json"
    if target.exists():
        raise FileExistsError(target)
    is_baseline = config.get("execution_mode") == "online_baseline"
    if is_baseline:
        if not baseline or name is not None:
            raise ValueError("An online baseline selection uses --baseline and has no checkpoint")
        digest = algorithm_fingerprint(config)
        if status.get("algorithm_sha256") != digest:
            raise ValueError("Online baseline algorithm changed after run initialization")
        artifact_dir = evaluation_directory(run, config, None, "val", "normal")
        report_key = "policy_sha256"
    else:
        if baseline or name is None:
            raise ValueError("A trained policy selection requires a checkpoint filename")
        checkpoint, digest = resolve_checkpoint(run, config, status, name)
        artifact_dir = evaluation_directory(run, config, checkpoint.name, "val", "normal")
        report_key = "checkpoint_sha256"
    summary_path = artifact_dir / "summary.json"
    report = json.loads(summary_path.read_text())
    if (report["split"] != "val" or report["condition"] != "normal"
            or report.get(report_key) != digest):
        raise ValueError("Selection needs the policy's completed VAL_NORMAL report")
    is_ppo_lstm_receiver_only = config.get("algorithm", {}).get("module", "").endswith(
        ".ppo_lstm_receiver_only")
    if is_ppo_lstm_receiver_only:
        summary = report["summary"]
        candidate_reports = []
        for candidate_path in sorted((run / "eval" / "val").rglob("summary.json")):
            candidate = json.loads(candidate_path.read_text())
            if (candidate.get("split") == "val" and candidate.get("condition") == "normal"
                    and candidate.get("action_mode", "greedy") == "greedy"
                    and candidate.get("checkpoint_file")):
                candidate_reports.append(candidate)
        if candidate_reports:
            best_report = max(candidate_reports, key=_ppo_lstm_selection_key)
            if _ppo_lstm_selection_key(report) != _ppo_lstm_selection_key(best_report):
                raise ValueError(
                    "PPO-LSTM receiver-only selection must follow the configured VAL metric priority; "
                    f"best evaluated checkpoint is {best_report['checkpoint_file']}"
                )
        ttfi = summary.get("source_relative_ttfi", {})
        metrics = {
            "unique_emitter_interception_rate": summary.get("unique_emitter_interception_rate"),
            "restricted_mean_ttfi_ms": ttfi.get("km_restricted_mean_ms"),
            "censored_fraction": ttfi.get("censored_fraction"),
            "mean_rolling_coverage": summary.get("mean_rolling_coverage"),
            "worlds_reaching_90pct_band_coverage": summary.get("worlds_reaching_90pct_band_coverage"),
            "fraction_worlds_reaching_90pct_band_coverage": summary.get(
                "fraction_worlds_reaching_90pct_band_coverage"),
            "interception_by_deadline": summary.get("interception_by_deadline"),
            "revisit_interval_distribution_s": summary.get("revisit_interval_distribution_s"),
            "world_count": summary.get("worlds"),
            "opportunity_interception_ratio": summary.get("opportunity_interception_ratio"),
            "policy_reward_per_decision": summary.get("policy_reward_per_decision"),
            "conditional_pd": summary.get("conditional_pd"),
            "true_pfa": summary.get("true_pfa"),
        }
        if metrics["unique_emitter_interception_rate"] is None:
            raise ValueError("VAL unique-emitter interception metric is unavailable")
        selection_metric = "VAL_NORMAL.unique_emitter_interception_rate"
        selection_value = metrics["unique_emitter_interception_rate"]
    else:
        metric = report["summary"]["opportunity_interception_ratio"]
        if metric is None:
            raise ValueError("VAL selection metric is unavailable")
        selection_metric = "VAL_NORMAL.opportunity_interception_ratio"
        selection_value = metric
        metrics = None
    selection = {"artifact_type": "online_baseline" if is_baseline else "checkpoint",
                 "config_sha256": status["config_sha256"], "reason": reason,
                 "metric": selection_metric, "value": selection_value,
                 "validation_summary_sha256": file_hash(summary_path)}
    if is_ppo_lstm_receiver_only:
        selection["validation_metrics"] = metrics
        selection["evaluated_checkpoint_priority_order"] = [
            candidate["checkpoint_file"] for candidate in sorted(
                candidate_reports, key=_ppo_lstm_selection_key, reverse=True)
        ]
        selection["selection_priority"] = [
            "unique_emitter_interception_rate:max",
            "restricted_mean_ttfi_ms:min",
            "censored_fraction:min",
            "mean_rolling_coverage:max",
            "worlds_reaching_90pct_band_coverage:max",
            "opportunity_interception_ratio:max",
            "policy_reward_per_decision:max",
        ]
    if config.get("evaluation", {}).get("require_frozen_world_catalog"):
        catalog_path = run / "frozen_world_catalog.json"
        if not catalog_path.is_file():
            raise ValueError("Freeze VAL/TEST world recipes and hashes before selecting a run")
        from .frozen_world_catalog import load_world_catalog
        load_world_catalog(run, status["config_sha256"])
        selection["frozen_world_catalog_sha256"] = file_hash(catalog_path)
    if is_baseline:
        selection.update(algorithm_name=config.get("algorithm", {}).get("name"), policy_sha256=digest)
    else:
        selection.update(checkpoint_file=checkpoint.name, checkpoint_sha256=digest)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(selection, indent=2) + "\n")
    return selection

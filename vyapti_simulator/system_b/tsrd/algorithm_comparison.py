"""Automated pilot/final seed experiments with paired worlds and sealed TEST."""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .training_setup import resolve_setup
from .experiment import train, evaluate, _json, _hash
from .checkpoints import evaluation_directory, freeze_selection
from .frequency_agility import fit_train_reference
from .evaluation_statistics import paired_world_bootstrap, summarize_training_seeds


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _identity(row):
    return {key: row.get(key) for key in ("config_id", "world_seed", "receiver_seed", "condition",
        "illumination_seed", "source_sha256", "sources", "replay_signature", "agility_reference_sha256")}


def paired_report(entries, protocol, *, split):
    """Average seeds within each world, then resample world-level differences."""
    report = {"split": split, "world_ci_unit": "world after averaging training seeds",
              "training_seed_variation_is_separate": True, "conditions": {}}
    algorithms = sorted({entry["algorithm"] for entry in entries})
    seed_sets = [{e["seed"] for e in entries if e["algorithm"] == name} for name in algorithms]
    if (not entries or any(seeds != seed_sets[0] for seeds in seed_sets) or
            len({(e["algorithm"], e["seed"]) for e in entries}) != len(entries)):
        raise ValueError("Paired algorithms need the same unique training-seed set")
    frozen_environment = None
    for condition in protocol["conditions"]:
        groups, identity = {}, {}
        for algorithm in algorithms:
            runs = sorted((e for e in entries if e["algorithm"] == algorithm), key=lambda e: e["seed"])
            seed_values, world_values, regimes, oracle_values = [], {}, {}, {}
            ttfi_values, prediction_values = [], []
            for entry in runs:
                run = Path(entry["run"])
                config = _read(run / "config.json")
                if config["seed"] != entry["seed"] or config["algorithm"]["name"] != algorithm:
                    raise ValueError("Run identity differs from the comparison entry")
                environment = {key: config.get(key) for key in ("contract_version", "receiver", "action", "reward",
                    "world", "evaluation", "analysis_protocol", "agility_reference_sha256")}
                if frozen_environment is not None and environment != frozen_environment:
                    raise ValueError("Paired comparison requires identical receiver and world contracts")
                frozen_environment = environment
                directory = evaluation_directory(run, config, entry["checkpoint"], split, condition)
                summary = _read(directory / "summary.json")
                if _hash(directory / "per_world.jsonl") != summary.get("per_world_sha256"):
                    raise ValueError("Per-world report changed after evaluation")
                if summary["checkpoint_sha256"] != entry["checkpoint_sha256"]:
                    raise ValueError("Comparison report checkpoint differs from frozen choice")
                seed_values.append(summary["summary"]["opportunity_interception_ratio"])
                ttfi_values.append(summary["analysis"]["ttfi_distributions"]["policy"]["rmst_ms"])
                prediction_values.append(summary["analysis"]["band_prediction_metrics"])
                rows = [_read_line(line) for line in (directory / "per_world.jsonl").read_text().splitlines()]
                if len({r["config_id"] for r in rows}) != len(rows):
                    raise ValueError("Duplicate world IDs in evaluation")
                current = {r["config_id"]: _identity(r) for r in rows}
                if identity and identity != current:
                    raise ValueError("Paired comparison requires identical worlds, source hashes, and seeds")
                identity = current
                for row in rows:
                    key = row["config_id"]
                    regimes[key] = row["agility"]["regime"]
                    world_values.setdefault(key, []).append(row["scorecard"]["cell_level"]["occupied_cell_detection_ratio"])
                    oracle_values.setdefault(key, []).append(row["oracle_comparison"]["privileged_oir"])
            valid_seeds = [(e["seed"], value) for e, value in zip(runs, seed_values) if value is not None]
            seed_summary = summarize_training_seeds([v for _, v in valid_seeds], seeds=[s for s, _ in valid_seeds]) if valid_seeds else None
            mean = lambda values: float(np.mean(values)) if all(v is not None for v in values) else None
            groups[algorithm] = {"seed_variability": seed_summary,
                "undefined_seed_count": len(runs) - len(valid_seeds),
                "world_mean_oir": {key: mean(values) for key, values in world_values.items()},
                "oracle_world_mean_oir": {key: mean(values) for key, values in oracle_values.items()},
                "agility_regimes": regimes,
                "per_seed_reports": [{"seed": e["seed"], "run": e["run"], "checkpoint": e["checkpoint"]} for e in runs]}
            valid_ttfi = [(e["seed"], value) for e, value in zip(runs, ttfi_values) if value is not None]
            groups[algorithm]["ttfi_rmst_ms_seed_variability"] = summarize_training_seeds(
                [value for _, value in valid_ttfi], seeds=[seed for seed, _ in valid_ttfi]) if valid_ttfi else None
            groups[algorithm]["band_prediction_by_seed"] = [{"seed": e["seed"], **value}
                for e, value in zip(runs, prediction_values)]
        comparisons = []
        settings = protocol["uncertainty"]
        def interval(a, b, name_a, name_b):
            keys = sorted(key for key in a if a[key] is not None and b[key] is not None)
            result = paired_world_bootstrap({key: a[key] for key in keys}, {key: b[key] for key in keys},
                name_a=name_a, name_b=name_b, samples=settings["resamples"],
                seed=settings.get("bootstrap_seed", 20261002), confidence=settings["confidence"]) if keys else None
            return {"paired_ci": result, "undefined_world_count": len(a) - len(keys)}
        for algorithm, group in groups.items():
            group["oracle_gap"] = interval(group["oracle_world_mean_oir"], group["world_mean_oir"], "privileged", algorithm)
        for a, b in combinations(algorithms, 2):
            first, second = groups[a]["world_mean_oir"], groups[b]["world_mean_oir"]
            strata = {}
            for regime in ("low", "medium", "high"):
                keys = [key for key in first if groups[a]["agility_regimes"][key] == regime]
                strata[regime] = interval({key: first[key] for key in keys}, {key: second[key] for key in keys}, a, b)
            comparisons.append({"algorithm_a": a, "algorithm_b": b, **interval(first, second, a, b), "agility_strata": strata})
        report["conditions"][condition] = {"algorithms": groups, "paired_comparisons": comparisons}
    return report


def _read_line(line):
    return json.loads(line)


def run_comparison(environment, algorithm_paths, output, *, stage, episodes, checkpoint_every):
    """Train 3 pilot or 5 final seeds; select solely on fixed VAL_NORMAL."""
    if stage not in {"pilot", "final"}:
        raise ValueError("stage must be pilot or final")
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(root)
    algorithms = list(algorithm_paths)
    if not algorithms:
        raise ValueError("Provide at least one algorithm spec")
    base = resolve_setup(environment, algorithms[0], seed=0, episodes=episodes, checkpoint_every=checkpoint_every)
    protocol = base.get("analysis_protocol")
    if protocol is None:
        raise ValueError("Comparison needs an environment with a frozen analysis protocol")
    seeds = protocol["training_seeds"]["development" if stage == "pilot" else "final_comparison"]
    if len(seeds) != (3 if stage == "pilot" else 5) or len(set(seeds)) != len(seeds):
        raise ValueError("Pilot requires 3 unique seeds; final requires 5")
    specs = [resolve_setup(environment, path, seed=seeds[0], episodes=episodes, checkpoint_every=checkpoint_every) for path in algorithms]
    names = [spec["algorithm"]["name"] for spec in specs]
    if len(set(names)) != len(names) or any(not name.isidentifier() for name in names):
        raise ValueError("Algorithm names must be unique simple identifiers")
    root.mkdir(parents=True)
    manifest = {"stage": stage, "status": "training_and_validation", "training_seeds": seeds,
        "episodes_per_seed": episodes, "checkpoint_every": checkpoint_every, "analysis_protocol": protocol,
        "setup_sources": [spec["setup_sources"] for spec in specs], "entries": [], "test_revealed": False}
    _json(root / "manifest.json", manifest)
    try:
        reference = fit_train_reference(base["cache_root"], expected_configs=base.get("expected_train_configs", 250),
            channel_width_mhz=protocol["frequency_agility"].get("channel_width_mhz", 500.0))
        _json(root / "agility_reference.json", reference)
        for path, algorithm in zip(algorithms, names):
            for seed in seeds:
                config = resolve_setup(environment, path, seed=seed, episodes=episodes, checkpoint_every=checkpoint_every)
                expected = specs[names.index(algorithm)]["setup_sources"]
                if config["setup_sources"] != expected:
                    raise ValueError("Input specifications changed during comparison")
                run = root / "runs" / algorithm / f"seed_{seed}"
                train(config, run, agility_reference=reference)
                checkpoints = _read(run / "checkpoints/index.json")["checkpoints"]
                candidates = []
                for checkpoint in checkpoints:
                    if checkpoint["kind"] == "initial":
                        continue
                    result = evaluate(run, "val", checkpoint=checkpoint["file"], condition="normal")
                    value = result["summary"]["opportunity_interception_ratio"]
                    if value is not None:
                        candidates.append((value, checkpoint["file"]))
                if not candidates:
                    raise ValueError("VAL has no defined OIR for checkpoint selection")
                # Deterministic tie-break; the selection record explains it.
                _, chosen = sorted(candidates, key=lambda row: (-row[0], row[1]))[0]
                selection = freeze_selection(run, chosen, reason="Maximum frozen VAL_NORMAL pooled OIR; filename tie-break")
                for condition in protocol["conditions"]:
                    if condition != "normal":
                        evaluate(run, "val", checkpoint=chosen, condition=condition)
                entry = {"algorithm": algorithm, "seed": seed, "run": str(run), "checkpoint": chosen,
                    "checkpoint_sha256": selection["checkpoint_sha256"], "selection_sha256": _hash(run / "selection.json"),
                    "config_sha256": _hash(run / "config.json")}
                entry["validation_reports"] = {condition: _hash(
                    evaluation_directory(run, config, chosen, "val", condition) / "summary.json")
                    for condition in protocol["conditions"]}
                manifest["entries"].append(entry)
                _json(root / "manifest.json", manifest)
        paired = paired_report(manifest["entries"], protocol, split="val")
        _json(root / "val_comparison.json", paired)
        means = {name: paired["conditions"]["normal"]["algorithms"][name]["seed_variability"] for name in names}
        if any(value is None or value["seed_count"] != len(seeds) for value in means.values()):
            raise ValueError("Algorithm selection needs defined VAL seed means")
        selected = sorted(names, key=lambda name: (-means[name]["mean"], name))[0]
        frozen = {"selected_algorithm": selected, "selection_metric": "mean_seed_VAL_NORMAL_pooled_OIR",
            "tie_break": "algorithm_name", "stage": stage, "training_seeds": seeds,
            "val_comparison_sha256": _hash(root / "val_comparison.json"), "entries": manifest["entries"],
            "analysis_protocol": protocol}
        _json(root / "comparison_selection.json", frozen)
        manifest.update(status="validation_frozen", selection_sha256=_hash(root / "comparison_selection.json"))
        _json(root / "manifest.json", manifest)
        return paired
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _json(root / "manifest.json", manifest)
        raise


def finalize_test(output):
    """Reveal TEST once for the algorithm/config selected by a final 5-seed run."""
    root = Path(output).resolve()
    manifest = _read(root / "manifest.json")
    if manifest["stage"] != "final" or manifest["status"] != "validation_frozen" or manifest["test_revealed"]:
        raise ValueError("TEST requires a frozen final comparison that has not revealed TEST")
    if _hash(root / "comparison_selection.json") != manifest["selection_sha256"]:
        raise ValueError("Frozen comparison selection changed")
    frozen = _read(root / "comparison_selection.json")
    if _hash(root / "val_comparison.json") != frozen["val_comparison_sha256"]:
        raise ValueError("Frozen VAL comparison changed")
    entries = [entry for entry in frozen["entries"] if entry["algorithm"] == frozen["selected_algorithm"]]
    for entry in entries:
        run = Path(entry["run"])
        if (_hash(run / "selection.json") != entry["selection_sha256"] or
                _hash(run / "config.json") != entry["config_sha256"] or
                _hash(run / "checkpoints" / entry["checkpoint"]) != entry["checkpoint_sha256"]):
            raise ValueError("Selected run changed before TEST")
        config = _read(run / "config.json")
        for condition, digest in entry["validation_reports"].items():
            directory = evaluation_directory(run, config, entry["checkpoint"], "val", condition)
            if (_hash(directory / "summary.json") != digest or
                    _hash(directory / "per_world.jsonl") != _read(directory / "summary.json")["per_world_sha256"]):
                raise ValueError("Selected validation evidence changed before TEST")
    manifest.update(status="test_started", test_revealed=True)
    _json(root / "manifest.json", manifest)
    try:
        for entry in entries:
            evaluate(entry["run"], "test", final=True, checkpoint=entry["checkpoint"], condition="all")
        report = paired_report(entries, frozen["analysis_protocol"], split="test")
        _json(root / "test_comparison.json", report)
        manifest["status"] = "complete"
        _json(root / "manifest.json", manifest)
        return report
    except Exception as exc:
        manifest.update(status="test_failed", error=f"{type(exc).__name__}: {exc}")
        _json(root / "manifest.json", manifest)
        raise

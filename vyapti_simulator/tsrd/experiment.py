"""Frozen TRAIN-250 recorded-PDW training and held-out replay protocol."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from vyapti_simulator.core.metrics import TrajectoryStep
from vyapti_simulator.core.receiver_observation import ReceiverObservation

from .replay_scorecard import score_recorded_replay, _km_quantile, _km_restricted_mean
from .evaluation_illumination import (
    IlluminationSettings, apply_evaluation_illumination, annotate_illumination_score,
)
from .train250_cache import (
    build_stare_evaluation_world, build_train_pool_from_cache,
    compose_heldout_world, write_runtime_manifest,
)


CONTRACT = "train250_recorded_pdw_v1"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=_serialize_receiver) + "\n", encoding="utf-8")


def _append(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False, default=_serialize_receiver) + "\n")


def _serialize_receiver(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def load_contract(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("contract_version") != CONTRACT:
        raise ValueError("Unknown experiment contract")
    receiver = config["receiver"]
    if receiver.get("detector_model", "recorded_stare_bernoulli") != "recorded_stare_bernoulli":
        raise ValueError("The recorded STARE runner uses Bernoulli detection; RF CFAR requires I/Q")
    if (receiver["profile"] != "binary_v1" or config["action"]["dwell_slots"] != [1]
            or config["reward"] != "detector_positive: +1 for an observed hit, 0 otherwise; no hidden truth to policy"):
        raise ValueError("Receiver, action, or reward differs from the frozen v1 contract")
    if int(config["train_episodes"]) < 1:
        raise ValueError("train_episodes must be positive")
    return config


def _receiver_options(config: dict) -> dict:
    r = config["receiver"]
    return {
        "receiver_profile": r["profile"],
        "passband_halfwidth_mhz": float(r["passband_halfwidth_mhz"]),
        "detection_probability": float(r["detection_probability"]),
        "false_alarm_probability": float(r["false_alarm_probability"]),
        "retune_time_ms": float(r["retune_time_ms"]),
    }


def _policy_module(config: dict):
    module = importlib.import_module(config["policy_module"])
    if not callable(getattr(module, "create", None)):
        raise ValueError("Policy module must export create(bands, seed, checkpoint=None)")
    return module


def _run_episode(world, policy, *, training: bool, episode: int, log_path: Path):
    history = []
    trajectory = []
    reward_total = 0.0
    while not world.done:
        slot = world.current_slot
        # The policy receives only past public receiver observations.
        band = int(policy.select_action(tuple(history), slot))
        if not 0 <= band < world.n_bands:
            raise ValueError(f"Policy selected invalid band {band}")
        observation, _ = world.step(band)
        public = ReceiverObservation.from_mapping(observation)
        policy.observe(public)
        reward = float(bool(public["hit"]))
        reward_total += reward
        history.append(public)
        trajectory.append(TrajectoryStep(action=band, time_slot=slot, observation=public))
        _append(log_path, {"episode": episode, "slot": slot, "action_band": band,
                           "observation": public, "detector_positive_reward": reward})
    if training:
        policy.finish_training_episode()
    return trajectory, reward_total


def train(config_path: str | Path, run_dir: str | Path) -> Path:
    """Create one immutable run directory and a model-neutral policy checkpoint."""
    config = load_contract(config_path)
    run = Path(run_dir).resolve()
    if run.exists():
        raise FileExistsError(run)
    data = Path(config["data_root"]).resolve()
    cache_path = Path(config["cache_root"]).resolve()
    pool, cache = build_train_pool_from_cache(
        data, cache_path, expected_configs=int(config.get("expected_train_configs", 250)),
        **_receiver_options(config))
    run.mkdir(parents=True)
    # Persist absolute dataset locations so a later evaluation is independent of cwd.
    config = dict(config, data_root=str(data), cache_root=str(cache_path))
    _json(run / "config.json", config)
    r = config["receiver"]
    write_runtime_manifest(cache, run / "runtime_manifest.json",
                           receiver_profile=r["profile"],
                           detection_probability=r["detection_probability"],
                           false_alarm_probability=r["false_alarm_probability"],
                           retune_time_ms=r["retune_time_ms"], training_seed=config["seed"])
    rng = np.random.default_rng(int(config["seed"]))
    policy = _policy_module(config).create(pool.config.band_count, int(config["seed"]))
    status = {"state": "running", "completed_episodes": 0,
              "config_sha256": _hash(run / "config.json")}
    _json(run / "status.json", status)
    try:
        distribution = cache["emitter_count_distribution"]
        for episode in range(int(config["train_episodes"])):
            count = int(rng.choice(distribution))
            seed = int(rng.integers(1, np.iinfo(np.int32).max))
            world, sources = pool.sample_world(seed, count)
            world.reset(seed=seed)
            policy.reset_episode()
            _, reward = _run_episode(world, policy, training=True, episode=episode,
                                     log_path=run / "train" / "steps.jsonl")
            _append(run / "train" / "episodes.jsonl", {
                "episode": episode, "world_seed": seed, "receiver_seed": seed,
                "emitter_count": count, "sources": sources,
                "reward_total": reward, "receiver_accounting": world.receiver_accounting(),
                "replay_signature": world.replay_signature(),
            })
            status["completed_episodes"] = episode + 1
            _json(run / "status.json", status)
        checkpoint = run / "checkpoints" / "final.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        policy.save(checkpoint)
        status.update(state="complete", checkpoint_sha256=_hash(checkpoint))
        _json(run / "status.json", status)
        return checkpoint
    except Exception as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        _json(run / "status.json", status)
        raise


def _evaluation_ids(data: Path, split: str, spec, expected_count: int | None = None) -> list[str]:
    directory = data / "stare" / f"{split}_stare"
    available = {p.stem for p in directory.glob("config_*.h5")}
    ids = sorted(available, key=lambda value: int(value.split("_")[1])) if spec == "all" else list(spec)
    if not ids or len(ids) != len(set(ids)) or any(value not in available for value in ids):
        raise ValueError(f"Invalid or empty fixed {split} world list")
    if expected_count is not None and len(ids) != expected_count:
        raise ValueError(f"Expected {expected_count} fixed {split} configs, found {len(ids)}")
    return ids


def _seed(base: int, split: str, config_id: str) -> int:
    value = hashlib.sha256(f"{base}:{split}:{config_id}".encode()).digest()
    return int.from_bytes(value[:4], "little")


def _summary(scores: list[dict]) -> dict:
    true = sum(row["event_level"]["true_detections"] for row in scores)
    false = sum(row["event_level"]["false_alarms"] for row in scores)
    eligible = sum(row["cell_level"]["eligible_selected_cells"] for row in scores)
    empty = sum(row["cell_level"]["empty_selected_cells"] for row in scores)
    occupied = sum(row["cell_level"]["occupied_cell_count"] for row in scores)
    emitters = sum(row["emitter_interception"]["eligible_emitters"] for row in scores)
    intercepted = sum(row["emitter_interception"]["intercepted_emitters"] for row in scores)
    stress_rows = [row["illumination_stress"] for row in scores if "illumination_stress" in row]
    source_eligible = sum(row["source_eligible_emitters"] for row in stress_rows)
    source_intercepted = sum(row["source_intercepted_emitters"] for row in stress_rows)
    source_ttfi = [record for row in stress_rows for record in row["source_relative_ttfi"].values()]
    durations = [record["ttfi_ms"] for record in source_ttfi]
    events = [not record["ttfi_censored"] for record in source_ttfi]
    mean_ttfi, restriction_ms = _km_restricted_mean(durations, events)
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    return {"worlds": len(scores), "true_detections": true, "false_alarms": false,
            "conditional_pd": ratio(true, eligible), "true_pfa": ratio(false, empty),
            "opportunity_interception_ratio": ratio(true, occupied),
            "unique_emitter_interception_rate": ratio(intercepted, emitters),
            "source_population_interception_rate": ratio(source_intercepted, source_eligible),
            "source_relative_ttfi": {
                "km_restricted_mean_ms": mean_ttfi, "restriction_ms": restriction_ms,
                "km_median_ms": _km_quantile(durations, events, 0.5),
                "censored_fraction": ratio(source_eligible - source_intercepted, source_eligible),
            }}


def plot_run(run_dir: str | Path) -> list[Path]:
    """Regenerate figures from saved logs without changing any metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run_dir)
    output = run / "plots"
    output.mkdir(exist_ok=True)
    written = []
    episode_log = run / "train" / "episodes.jsonl"
    if episode_log.is_file():
        rows = [json.loads(line) for line in episode_log.read_text(encoding="utf-8").splitlines()]
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot([r["episode"] + 1 for r in rows], [r["reward_total"] for r in rows])
        axis.set(xlabel="Training episode", ylabel="Detector-positive hits",
                 title="TRAIN-250 learning curve")
        fig.tight_layout()
        path = output / "train_reward.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    for split in ("val", "test"):
        for path in sorted((run / "eval" / split).glob("*/summary.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            summary = report["summary"]
            names = ("opportunity_interception_ratio", "conditional_pd", "true_pfa",
                     "unique_emitter_interception_rate")
            fig, axis = plt.subplots(figsize=(8, 4))
            values = [summary[name] if summary[name] is not None else np.nan for name in names]
            axis.bar(range(len(names)), values)
            for index, value in enumerate(values):
                if np.isnan(value):
                    axis.text(index, 0.02, "Unavailable", ha="center", fontsize=8)
            axis.set_xticks(range(len(names)), [name.replace("_", "\n") for name in names])
            axis.set(ylabel="Rate", ylim=(0, 1), title=f"{report['label']} recorded-PDW metrics")
            fig.tight_layout()
            suffix = "" if report["condition"] == "normal" else f"_{report['condition']}"
            figure = output / f"{split}{suffix}_metrics.png"
            fig.savefig(figure, dpi=150)
            plt.close(fig)
            written.append(figure)
    return written


def evaluate(run_dir: str | Path, split: str, *, final: bool = False,
             condition: str = "normal") -> dict:
    """Evaluate one frozen checkpoint on the contract's fixed held-out worlds."""
    if split not in {"val", "test"} or (split == "test" and not final):
        raise ValueError("TEST requires explicit final=True; split must be val or test")
    run = Path(run_dir).resolve()
    config = load_contract(run / "config.json")
    conditions = config["evaluation"].get("illumination_conditions", {"normal": {}})
    if condition == "all":
        return {"split": split, "conditions": {
            name: evaluate(run, split, final=final, condition=name) for name in conditions
        }}
    if condition not in conditions:
        raise ValueError(f"Illumination condition is not in the frozen contract: {condition}")
    settings = IlluminationSettings(mode=condition, **conditions[condition])
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if _hash(run / "config.json") != status.get("config_sha256"):
        raise ValueError("Experiment config changed after training")
    if status.get("state") != "complete":
        raise ValueError("Training must complete before evaluation")
    checkpoint = run / "checkpoints" / "final.json"
    frozen_hash = _hash(checkpoint)
    if frozen_hash != status["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after training")
    target = run / "eval" / split / condition
    if target.exists():
        raise FileExistsError(target)
    data = Path(config["data_root"]).resolve()
    ids = _evaluation_ids(
        data, split, config["evaluation"][f"{split}_config_ids"],
        config["evaluation"].get(f"expected_{split}_configs"),
    )
    composed = config["evaluation"].get(f"{split}_composed_worlds", [])
    target.mkdir(parents=True)
    _json(target / "worlds.json", {"split": split, "config_ids": ids,
                                   "detector_model": "recorded_stare_bernoulli",
                                   "condition": condition,
                                   "illumination_settings": conditions[condition],
                                   "illumination_seed_base": config["evaluation"].get("illumination_seed", 20261004),
                                   "composed_worlds": composed,
                                   "receiver_seed_base": config["evaluation"]["receiver_seed"],
                                   "checkpoint_sha256": frozen_hash})
    scores = []
    policy_module = _policy_module(config)
    for index, config_id in enumerate(ids):
        seed = _seed(int(config["evaluation"]["receiver_seed"]), split, config_id)
        world = build_stare_evaluation_world(data, split, config_id,
                                             receiver_seed=seed, **_receiver_options(config))
        illumination_seed = _seed(int(config["evaluation"].get("illumination_seed", 20261004)),
                                  split, config_id)
        world, illumination = apply_evaluation_illumination(
            world, split=split, settings=settings, illumination_seed=illumination_seed,
            receiver_seed=seed,
        )
        _json(target / "world_logs" / f"{config_id}.json", illumination)
        policy = policy_module.create(world.n_bands, seed, checkpoint)
        trajectory, _ = _run_episode(world, policy, training=False, episode=index,
                                     log_path=target / "steps.jsonl")
        score = score_recorded_replay(world, trajectory)
        annotate_illumination_score(score, illumination, world.config.slot_duration_s())
        scores.append(score)
        source = data / "stare" / f"{split}_stare" / f"{config_id}.h5"
        _append(target / "per_world.jsonl", {
            "config_id": config_id, "receiver_seed": seed,
            "condition": condition, "illumination_seed": illumination_seed,
            "illumination_audit_file": f"world_logs/{config_id}.json",
            "source_recorded_pulses_in_mission": illumination["source_recorded_pulses_in_mission"],
            "visible_recorded_pulses_in_mission": illumination["visible_recorded_pulses_in_mission"],
            "source_sha256": _hash(source), "replay_signature": world.replay_signature(),
            "receiver_accounting": world.receiver_accounting(), "scorecard": score,
        })
    for offset, recipe in enumerate(composed):
        sources = list(recipe["source_config_ids"])
        if any(source not in ids for source in sources):
            raise ValueError("Composed world must use fixed held-out source list")
        seed = _seed(int(config["evaluation"]["receiver_seed"]), split, recipe["id"])
        world, provenance = compose_heldout_world(
            data, split, sources, world_seed=int(recipe["world_seed"]),
            emitter_count=int(recipe["emitter_count"]), receiver_seed=seed,
            **_receiver_options(config),
        )
        illumination_seed = _seed(int(config["evaluation"].get("illumination_seed", 20261004)),
                                  split, recipe["id"])
        world, illumination = apply_evaluation_illumination(
            world, split=split, settings=settings, illumination_seed=illumination_seed,
            receiver_seed=seed,
        )
        _json(target / "world_logs" / f"{recipe['id']}.json", illumination)
        policy = policy_module.create(world.n_bands, seed, checkpoint)
        trajectory, _ = _run_episode(world, policy, training=False,
                                     episode=len(ids) + offset, log_path=target / "steps.jsonl")
        score = score_recorded_replay(world, trajectory)
        annotate_illumination_score(score, illumination, world.config.slot_duration_s())
        scores.append(score)
        _append(target / "per_world.jsonl", {
            "config_id": recipe["id"], "kind": "composed_heldout",
            "condition": condition, "illumination_seed": illumination_seed,
            "illumination_audit_file": f"world_logs/{recipe['id']}.json",
            "world_seed": int(recipe["world_seed"]), "receiver_seed": seed,
            "sources": provenance,
            "source_sha256": {source: _hash(data / "stare" / f"{split}_stare" / f"{source}.h5")
                              for source in sources},
            "replay_signature": world.replay_signature(),
            "receiver_accounting": world.receiver_accounting(), "scorecard": score,
        })
    if _hash(checkpoint) != frozen_hash:
        raise RuntimeError("Frozen checkpoint changed during evaluation")
    report = {"split": split, "condition": condition,
              "detector_model": "recorded_stare_bernoulli",
              "label": f"{split.upper()}_{condition.upper()}",
              "opportunity_basis": "recorded pulses visible under the configured illumination gate",
              "checkpoint_sha256": frozen_hash,
              "contract_version": CONTRACT, "summary": _summary(scores)}
    _json(target / "summary.json", report)
    return report

"""Frozen TRAIN-250 recorded-PDW training and held-out replay protocol."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
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
from .world_composer import WorldComposer, GENERATOR_VERSION, DEFAULT_TIME_OFFSET_US
from .algorithm_interface import (
    PublicState, PublicTransition, create_algorithm, algorithm_provenance,
    algorithm_fingerprint,
)
from .checkpoints import save_checkpoint, resolve_checkpoint, evaluation_directory, safe_name
from .frequency_agility import fit_train_reference, world_features, classify_rate, validate_train_reference
from .privileged_scheduler import evaluate_privileged, compare_privileged
from .prediction_evaluation import collect_prediction, score_predictions
from .evaluation_reporting import analysis_summary
from .frozen_world_catalog import load_world_catalog, verify_catalog_rebuild


CONTRACTS = {"train250_recorded_pdw_v1", "train250_recorded_pdw_v2"}


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


def load_contract(path: str | Path | dict) -> dict:
    config = dict(path) if isinstance(path, dict) else json.loads(Path(path).read_text(encoding="utf-8"))
    contract = config.get("contract_version")
    if contract not in CONTRACTS:
        raise ValueError("Unknown experiment contract")
    receiver = config["receiver"]
    if receiver.get("detector_model", "recorded_stare_bernoulli") != "recorded_stare_bernoulli":
        raise ValueError("The recorded STARE runner uses Bernoulli detection; RF CFAR requires I/Q")
    if receiver["profile"] != "binary_v1":
        raise ValueError("This runner currently requires the binary_v1 receiver")
    if contract == "train250_recorded_pdw_v1":
        if (config["action"].get("dwell_slots") != [1]
                or config["reward"] != "detector_positive: +1 for an observed hit, 0 otherwise; no hidden truth to policy"):
            raise ValueError("Receiver, action, or reward differs from the frozen v1 contract")
    else:
        profile = config["action"].get("dwell_slots_by_band")
        expected_profile = [2 if index in {0, 1, 6, 7, 17, 18, 19} else 1 for index in range(36)]
        if (not isinstance(profile, list) or len(profile) != int(receiver.get("band_count", 36))
                or any(type(value) is not int or value not in (1, 2) for value in profile)
                or profile != expected_profile
                or receiver.get("retune_time_ms") != 0.3
                or receiver.get("same_band_retune_time_ms") != 0.0
                or receiver.get("band_change_retune_time_ms") != 0.3
                or receiver.get("mission_duration_s") != 30.0
                or receiver.get("base_slot_duration_ms") != 50.0
                or receiver.get("receiver_bandwidth_mhz") != 1000.0
                or receiver.get("band_spacing_mhz") != 500.0
                or receiver.get("spectrum_mhz") != [0.0, 18000.0]
                or not isinstance(config.get("reward"), dict)
                or receiver.get("passband_halfwidth_mhz") != 500.0
                or receiver.get("detection_probability") != 0.9
                or receiver.get("false_alarm_probability") != 0.05
                or config["reward"].get("name") != "truth_based_intercept_utility_v2"
                or config["reward"].get("lambda_fa") != 1.0
                or config["reward"].get("mission_duration_s") != 30.0):
            raise ValueError("Receiver, native dwell profile, retune, or reward differs from the frozen v2 contract")
    execution_mode = config.get("execution_mode", "training")
    if execution_mode not in {"training", "online_baseline"}:
        raise ValueError("execution_mode must be training or online_baseline")
    episodes = int(config["train_episodes"])
    if (execution_mode == "training" and episodes < 1) or (execution_mode == "online_baseline" and episodes != 0):
        raise ValueError("Training needs positive episodes; an online baseline needs exactly zero pretraining episodes")
    if execution_mode == "training":
        safe_name(config.get("checkpoint_file", "final.json"))
    elif config.get("checkpoint_file") is not None or config.get("checkpointing") is not None:
        raise ValueError("online_baseline runs do not use checkpoints")
    if not config.get("algorithm") and not config.get("policy_module"):
        raise ValueError("Select an algorithm implementation explicitly")
    schedule = config.get("checkpointing", {})
    interval = schedule.get("interval_episodes")
    if interval is not None and (type(interval) is not int or interval < 1):
        raise ValueError("Checkpoint interval must be a positive episode count")
    world = config.get("world", {"mode": "emitter_recombined", "spatial_model": "tsrd_native"})
    if world["mode"] != "emitter_recombined" or world["spatial_model"] != "tsrd_native":
        raise ValueError("Training requires emitter_recombined with tsrd_native; illumination is evaluation-only")
    WorldComposer(time_offset_us=world.get("time_offset_us", DEFAULT_TIME_OFFSET_US))
    evaluation_mode = config["evaluation"].get("world_mode", "source_replay")
    if evaluation_mode not in ("source_replay", "emitter_recombined"):
        raise ValueError("Unknown evaluation world mode")
    if evaluation_mode == "emitter_recombined":
        for split in ("val", "test"):
            recipes = config["evaluation"].get(f"{split}_composed_worlds", [])
            if not recipes or len({r["id"] for r in recipes}) != len(recipes):
                raise ValueError("Compositional evaluation requires unique frozen world recipes")
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


def _strongest_emitter_in_selected_window(
    world, band: int, slot: int, retune_cost_s: float,
) -> int | None:
    """Attribute one binary-v1 cell hit to its strongest in-window emitter."""
    data, labels = world.recorded_pulses
    if data is None or labels is None or len(data) == 0:
        return None
    slot_s = world.config.slot_duration_s()
    start_us = round((slot * slot_s + retune_cost_s) * 1e6, 9)
    end_us = round((slot + 1) * slot_s * 1e6, 9)
    lo = int(np.searchsorted(data[:, 0], start_us, side="left"))
    hi = int(np.searchsorted(data[:, 0], end_us, side="left"))
    window = data[lo:hi]
    present = np.abs(window[:, 1] - world.receiver_geometry[0][band]) <= world.receiver_geometry[1]
    candidates = np.flatnonzero(present)
    if not len(candidates):
        return None
    strongest = int(candidates[np.argmax(window[candidates, 4])])
    return int(labels[lo + strongest])


def _eligible_world_emitter_first_slots(world) -> dict[int, int]:
    """Map each eligible emitter to its first in-scope pulse opportunity slot."""
    grid = np.asarray(world.hidden_truth.grid, dtype=bool)
    if grid.ndim != 3:
        raise ValueError("Hidden truth grid must have emitter, band, and slot axes")
    per_emitter_slots = grid.any(axis=1)
    first_slots = {}
    for config, occupied_slots in zip(world.hidden_truth.emitter_configs, per_emitter_slots):
        slots = np.flatnonzero(occupied_slots)
        if slots.size:
            first_slots[int(config.emitter_id)] = int(slots[0])
    return first_slots


def _emitters_with_prior_opportunity(first_slots: dict[int, int], slot: int) -> set[int]:
    """Return emitters whose first eligible opportunity predates this action."""
    return {emitter_id for emitter_id, first_slot in first_slots.items() if first_slot < slot}


def _run_episode(world, policy, *, training: bool, episode: int, log_path: Path,
                 prediction_rows: list | None = None, config: dict | None = None):
    trajectory = []
    reward_total = 0.0
    policy.reset_episode(training=training)
    state = PublicState(world.current_slot, None)
    contract = config.get("contract_version", "train250_recorded_pdw_v1") if config else "train250_recorded_pdw_v1"
    first_opportunity_slots = (_eligible_world_emitter_first_slots(world)
                               if contract == "train250_recorded_pdw_v2" else {})
    eligible_ids = set(first_opportunity_slots)
    intercepted_ids: set[int] = set()
    reward_totals = {"new_intercept_utility": 0.0, "elapsed_cost": 0.0,
                     "false_alarm_cost": 0.0, "new_first_intercepts": 0,
                     "false_alarms": 0, "empty_opportunities": 0,
                     "detector_evaluation_opportunities": 0,
                     "eligible_emitters": len(eligible_ids),
                     "band_decisions": 0, "receiver_steps": 0}
    reward_spec = config.get("reward", {}) if config else {}
    reward_mission_s = float(reward_spec.get("mission_duration_s", 30.0))
    lambda_fa = float(reward_spec.get("lambda_fa", 1.0))
    dwell_profile = config.get("action", {}).get("dwell_slots_by_band") if config else None
    if dwell_profile is not None and len(dwell_profile) != world.n_bands:
        raise ValueError("Frozen TRAIN dwell profile does not match the receiver bands")
    while not world.done:
        slot = world.current_slot
        band = int(policy.select_action(state, training=training))
        if not 0 <= band < world.n_bands:
            raise ValueError(f"Policy selected invalid band {band}")
        requested_slots = int(dwell_profile[band]) if dwell_profile else 1
        actual_slots = min(requested_slots, world.n_slots - slot)
        reward_totals["band_decisions"] += 1
        reward_totals["receiver_steps"] += actual_slots
        observations = []
        observed_ids: set[int] = set()
        false_alarms = 0
        empty_opportunities = 0
        for _ in range(actual_slots):
            observation, _ = world.step(band)
            observations.append(observation)
            if contract == "train250_recorded_pdw_v2":
                emitter_id = _strongest_emitter_in_selected_window(
                    world, band, int(observation["time_slot"]),
                    float(observation["retune_cost_s"]))
                if emitter_id is None:
                    empty_opportunities += 1
                    if observation["hit"]:
                        false_alarms += 1
                elif observation["hit"] and emitter_id in eligible_ids:
                    # binary_v1 returns one cell-level hit; credit at most
                    # its strongest in-band emitter for this receiver look.
                    observed_ids.add(emitter_id)
        final_observation = dict(observations[-1])
        final_observation["hit"] = any(bool(item["hit"]) for item in observations)
        final_observation["retune_cost_s"] = float(sum(item["retune_cost_s"] for item in observations))
        final_observation["dwell_elapsed_s"] = float(sum(item["dwell_elapsed_s"] for item in observations))
        final_observation["receiver_metadata"] = dict(final_observation["receiver_metadata"])
        final_observation["receiver_metadata"]["dwell_time_ms"] = actual_slots * 50.0
        public = ReceiverObservation.from_mapping(final_observation)
        if contract == "train250_recorded_pdw_v2":
            new_ids = observed_ids - intercepted_ids
            # Emitters accrue elapsed cost only after their first in-scope
            # opportunity has occurred before this action begins.
            previously_eligible = _emitters_with_prior_opportunity(first_opportunity_slots, slot)
            unresolved_before = len(previously_eligible - intercepted_ids)
            unresolved_fraction = (unresolved_before / len(eligible_ids)) if eligible_ids else 0.0
            retune = float(observations[0]["retune_cost_s"])
            dwell_seconds = actual_slots * world.config.slot_duration_s()
            intercept_utility = len(new_ids) / len(eligible_ids) if eligible_ids else 0.0
            elapsed_seconds = dwell_seconds + retune
            elapsed_cost = unresolved_fraction * elapsed_seconds / reward_mission_s
            false_alarm_rate = false_alarms / empty_opportunities if empty_opportunities else 0.0
            false_alarm_cost = lambda_fa * false_alarm_rate * elapsed_seconds / reward_mission_s
            reward = intercept_utility - elapsed_cost - false_alarm_cost
            intercepted_ids.update(new_ids)
            reward_totals["new_intercept_utility"] += intercept_utility
            reward_totals["elapsed_cost"] += elapsed_cost
            reward_totals["false_alarm_cost"] += false_alarm_cost
            reward_totals["new_first_intercepts"] += len(new_ids)
            reward_totals["false_alarms"] += false_alarms
            reward_totals["empty_opportunities"] += empty_opportunities
            reward_totals["detector_evaluation_opportunities"] += actual_slots
            reward_record = {"new_first_intercepts": len(new_ids), "eligible_emitters": len(eligible_ids),
                             "emitters_with_prior_opportunity": len(previously_eligible),
                             "unresolved_emitters_before_action": unresolved_before,
                             "new_intercept_utility": intercept_utility,
                             "unresolved_fraction_before_action": unresolved_fraction,
                             "dwell_seconds": dwell_seconds, "retune_dead_time_s": retune,
                             "mission_duration_s": reward_mission_s, "lambda_fa": lambda_fa,
                             "elapsed_cost": elapsed_cost, "false_alarms": false_alarms,
                             "empty_opportunities": empty_opportunities,
                             "detector_evaluation_opportunities": actual_slots,
                             "false_alarm_rate": false_alarm_rate,
                             "false_alarm_cost": false_alarm_cost, "reward": reward}
        else:
            reward = float(bool(public["hit"]))
            reward_record = {"detector_positive_reward": reward}
        next_state = PublicState(world.current_slot, public)
        transition = PublicTransition(state, band, reward, next_state, bool(world.done))
        policy.observe(transition, training=training)
        if not training and prediction_rows is not None:
            prediction = collect_prediction(policy, next_state, bands=world.n_bands)
            if prediction is not None:
                prediction_rows.append(prediction)
        reward_total += reward
        state = next_state
        for offset, observation in enumerate(observations):
            trajectory.append(TrajectoryStep(action=band, time_slot=slot + offset, observation=observation))
            item = {"episode": episode, "slot": slot + offset, "action_band": band,
                    "observation": observation}
            if offset == 0:
                item["decision_reward"] = reward_record
                item["dwell_slots"] = actual_slots
            _append(log_path, item)
    diagnostics = policy.end_episode(training=training)
    if training and diagnostics:
        _append(log_path.parent / "algorithm_updates.jsonl", {"episode": episode, "diagnostics": diagnostics})
    if diagnostics:
        # Evaluation diagnostics belong to the world record as well: online
        # bandits reset between worlds, and their decision counts are useful
        # audit/plot data rather than persistent checkpoint state.
        reward_totals["policy_diagnostics"] = diagnostics
    return trajectory, reward_total, reward_totals


def train(config_path: str | Path | dict, run_dir: str | Path, *, agility_reference=None) -> Path:
    """Create one immutable run directory; online baselines use zero pretraining episodes."""
    config = load_contract(config_path)
    run = Path(run_dir).resolve()
    if run.exists():
        raise FileExistsError(run)
    training_started = time.perf_counter()
    data = Path(config["data_root"]).resolve()
    cache_path = Path(config["cache_root"]).resolve()
    pool, cache = build_train_pool_from_cache(
        data, cache_path, expected_configs=int(config.get("expected_train_configs", 250)),
        time_offset_us=config.get("world", {}).get("time_offset_us", DEFAULT_TIME_OFFSET_US),
        **_receiver_options(config))
    if config.get("analysis_protocol"):
        if agility_reference is None:
            agility_reference = fit_train_reference(cache_path,
                expected_configs=int(config.get("expected_train_configs", 250)),
                channel_width_mhz=config["analysis_protocol"]["frequency_agility"].get("channel_width_mhz", 500.0))
        validate_train_reference(agility_reference, cache)
        if agility_reference["channel_width_mhz"] != config["analysis_protocol"]["frequency_agility"].get("channel_width_mhz", 500.0):
            raise ValueError("Agility channel width differs from the frozen protocol")
    execution_mode = config.get("execution_mode", "training")
    algorithm = config.get("algorithm", {})
    if algorithm.get("name") == "ppo_lstm" and execution_mode == "training":
        config = dict(config)
        config["algorithm"] = dict(algorithm)
        config["algorithm"]["settings"] = dict(algorithm.get("settings", {}))
        target = config["algorithm"]["settings"].get("training_episodes_target")
        if target is not None and int(target) != int(config["train_episodes"]):
            raise ValueError("PPO-LSTM learning-rate schedule must match train_episodes")
        config["algorithm"]["settings"]["training_episodes_target"] = int(config["train_episodes"])
    policy = create_algorithm(config, bands=pool.config.band_count, seed=int(config["seed"]))
    if execution_mode == "training" and not callable(getattr(policy, "save", None)):
        raise ValueError("Training algorithms must implement save(path)")
    run.mkdir(parents=True)
    if execution_mode == "online_baseline":
        for relative in ("eval/val", "eval/test", "plots"):
            (run / relative).mkdir(parents=True, exist_ok=True)
    if agility_reference is not None:
        _json(run / "agility_reference.json", agility_reference)
        config = dict(config, agility_reference_sha256=_hash(run / "agility_reference.json"))
    # Persist absolute dataset locations so a later evaluation is independent of cwd.
    config = dict(config, data_root=str(data), cache_root=str(cache_path))
    _json(run / "config.json", config)
    r = config["receiver"]
    write_runtime_manifest(cache, run / "runtime_manifest.json",
                           receiver_profile=r["profile"],
                           detection_probability=r["detection_probability"],
                           false_alarm_probability=r["false_alarm_probability"],
                           retune_time_ms=r["retune_time_ms"], training_seed=config["seed"],
                           time_offset_us=pool.composer.time_offset_us,
                           algorithm_details=algorithm_provenance(config),
                           training_budget={"episodes": int(config["train_episodes"]),
                                            "receiver_steps_per_episode": pool.config.time_slots,
                                            "band_decisions_per_episode_range": ([pool.config.time_slots // 2,
                                                pool.config.time_slots] if config["contract_version"] == "train250_recorded_pdw_v2"
                                                else [pool.config.time_slots, pool.config.time_slots])},
                           checkpointing=config.get("checkpointing") if execution_mode == "training" else None,
                           analysis_details={"protocol": config.get("analysis_protocol"),
                               "agility_reference_sha256": config.get("agility_reference_sha256"),
                               "reward_contract": config.get("reward"),
                               "dwell_slots_by_band": config["action"].get("dwell_slots_by_band"),
                               "dwell_profile_source": config["action"].get("dwell_profile_source"),
                               "dataset_paper": agility_reference.get("dataset_paper") if agility_reference else None})
    rng = np.random.default_rng(int(config["seed"]))
    status = {"state": "running", "execution_mode": config.get("execution_mode", "training"),
              "completed_episodes": 0,
              "completed_receiver_steps": 0,
              "config_sha256": _hash(run / "config.json")}
    _json(run / "status.json", status)
    try:
        distribution = cache["emitter_count_distribution"]
        schedule = config.get("checkpointing", {}) if execution_mode == "training" else {}
        interval = schedule.get("interval_episodes")
        extension = Path(config.get("checkpoint_file", "final.json")).suffix if execution_mode == "training" else None
        if schedule.get("save_initial", False):
            save_checkpoint(run, policy, "initial" + extension, episodes=0, steps=0, kind="initial")
        for episode in range(int(config["train_episodes"])):
            if episode == 0:
                print(f"[TRAIN] starting {int(config['train_episodes'])} episodes; sampling first world",
                      flush=True)
            count = int(rng.choice(distribution))
            seed = int(rng.integers(1, np.iinfo(np.int32).max))
            world, sources = pool.sample_world(seed, count)
            if episode == 0:
                print(f"[TRAIN] first world ready (emitters={count}); starting episode 1",
                      flush=True)
            world.reset(seed=seed)
            trajectory, reward, reward_components = _run_episode(
                world, policy, training=True, episode=episode,
                log_path=run / "train" / "steps.jsonl", config=config)
            elapsed_s = time.perf_counter() - training_started
            _append(run / "train" / "episodes.jsonl", {
                "episode": episode, "world_seed": seed, "receiver_seed": seed,
                "emitter_count": count, "sources": sources,
                "reward_total": reward, "reward_components": reward_components,
                "receiver_accounting": world.receiver_accounting(),
                "replay_signature": world.replay_signature(),
                "training_elapsed_wall_clock_s": elapsed_s,
            })
            status["completed_episodes"] = episode + 1
            status["completed_receiver_steps"] += len(trajectory)
            status["wall_clock_training_s"] = elapsed_s
            status["environment_steps"] = status["completed_receiver_steps"]
            status["steps_per_second"] = (
                status["completed_receiver_steps"] / elapsed_s if elapsed_s > 0 else None
            )
            if interval and (episode + 1) % interval == 0 and episode + 1 < int(config["train_episodes"]):
                save_checkpoint(run, policy, f"episode_{episode + 1:06d}" + extension,
                                episodes=episode + 1, steps=status["completed_receiver_steps"], kind="scheduled")
            _json(run / "status.json", status)
            if episode == 0 or (episode + 1) % 10 == 0 or episode + 1 == int(config["train_episodes"]):
                print(f"[TRAIN] completed {episode + 1}/{int(config['train_episodes'])} episodes "
                      f"receiver_steps={status['completed_receiver_steps']} "
                      f"reward={reward:.6f} elapsed={elapsed_s / 60.0:.1f} min", flush=True)
        if execution_mode == "training":
            record = save_checkpoint(run, policy, config.get("checkpoint_file", "final.json"),
                                     episodes=status["completed_episodes"], steps=status["completed_receiver_steps"], kind="final")
            status.update(state="complete", checkpoint_sha256=record["sha256"],
                          checkpoint_index_sha256=_hash(run / "checkpoints/index.json"))
            result_path = run / "checkpoints" / record["file"]
        else:
            status.update(state="complete", algorithm_sha256=algorithm_fingerprint(config))
            result_path = run
        elapsed_s = time.perf_counter() - training_started
        elapsed_key = ("baseline_setup_wall_clock_s" if status["execution_mode"] == "online_baseline"
                       else "wall_clock_training_s")
        status[elapsed_key] = elapsed_s
        status.update(
            environment_steps=status["completed_receiver_steps"],
            steps_per_second=(status["completed_receiver_steps"] / elapsed_s if elapsed_s > 0 else None),
        )
        _json(run / "status.json", status)
        if config.get("evaluation", {}).get("require_frozen_world_catalog"):
            from .frozen_world_catalog import freeze_world_catalog
            freeze_world_catalog(run)
        return result_path
    except Exception as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        elapsed_s = time.perf_counter() - training_started
        elapsed_key = ("baseline_setup_wall_clock_s" if status["execution_mode"] == "online_baseline"
                       else "wall_clock_training_s")
        status[elapsed_key] = elapsed_s
        status.update(
            environment_steps=status["completed_receiver_steps"],
            steps_per_second=(status["completed_receiver_steps"] / elapsed_s if elapsed_s > 0 else None),
        )
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
    emitter_counts = [row["emitter_interception"] for row in scores]
    opportunity_counts = [row["illumination"] for row in scores]
    revisits = [row["revisit"] for row in scores]
    coverage_times = [row["time_to_90pct_band_coverage_s"] for row in revisits
                      if row["time_to_90pct_band_coverage_s"] is not None]
    return {"worlds": len(scores), "true_detections": true, "false_alarms": false,
            "conditional_pd": ratio(true, eligible), "true_pfa": ratio(false, empty),
            "opportunity_interception_ratio": ratio(true, occupied),
            "unique_emitter_interception_rate": ratio(intercepted, emitters),
            "eligibility": {
                "physical_emitters": sum(row["physical_emitters"] for row in emitter_counts),
                "active_emitters": sum(row["active_emitters"] for row in emitter_counts),
                "eligible_intercept_emitters": emitters,
                "intercepted_emitters": intercepted,
            },
            "missed_emitter_slot_opportunity_count": sum(
                row["missed_emitter_slot_opportunity_count"] for row in opportunity_counts
            ),
            "time_to_90pct_band_coverage_s": (
                float(np.mean(coverage_times)) if coverage_times else None
            ),
            "worlds_reaching_90pct_band_coverage": len(coverage_times),
            "mean_world_max_blind_interval_s": (
                float(np.mean([row["max_blind_interval_s"] for row in revisits])) if revisits else None
            ),
            "mean_action_entropy": (
                float(np.mean([row["action_entropy"] for row in revisits])) if revisits else None
            ),
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
        axis.set(xlabel="Training episode", ylabel="Cumulative training reward",
                 title="TRAIN-250 learning curve")
        fig.tight_layout()
        path = output / "train_reward.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
        component_names = ("new_intercept_utility", "elapsed_cost", "false_alarm_cost")
        if rows and all(isinstance(row.get("reward_components"), dict) for row in rows):
            matrix = np.asarray([[row["reward_components"].get(name, 0.0) for name in component_names]
                                 for row in rows], dtype=float)
            window = min(25, len(matrix))
            if window:
                smooth = np.vstack([np.convolve(matrix[:, i], np.ones(window) / window, mode="valid")
                                    for i in range(matrix.shape[1])]).T
                fig, axis = plt.subplots(figsize=(9, 5))
                labels = ("First-intercept utility", "Elapsed-time cost", "False-alarm cost")
                for i, label in enumerate(labels):
                    axis.plot(np.arange(window, window + len(smooth)), smooth[:, i], label=label)
                axis.set(xlabel=f"Training episode ({window}-episode rolling mean)",
                         ylabel="Reward component", title="Training reward components")
                axis.legend()
                fig.tight_layout()
                path = output / "train_reward_components.png"
                fig.savefig(path, dpi=150)
                plt.close(fig)
                written.append(path)
    update_log = run / "train" / "algorithm_updates.jsonl"
    if update_log.is_file():
        updates = [json.loads(line) for line in update_log.read_text(encoding="utf-8").splitlines()]
        sac_keys = ("mean_critic1_loss", "mean_critic2_loss", "mean_actor_loss", "mean_temperature_loss")
        if updates and any(key in updates[-1].get("diagnostics", {}) for key in sac_keys):
            fig, axis = plt.subplots(figsize=(9, 5))
            for key in sac_keys:
                points = [(row["episode"] + 1, row["diagnostics"][key]) for row in updates
                          if key in row.get("diagnostics", {})]
                if points:
                    axis.plot([p[0] for p in points], [p[1] for p in points], label=key.replace("_", " "))
            axis.set(xlabel="Training episode", ylabel="Loss", title="Discrete SAC update losses")
            axis.legend()
            fig.tight_layout()
            path = output / "train_discrete_sac_losses.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)
        mcts_points = [
            (row["episode"] + 1, row["diagnostics"]["reward_model_parameter_norm"])
            for row in updates
            if "reward_model_parameter_norm" in row.get("diagnostics", {})
        ]
        if mcts_points:
            fig, axis = plt.subplots(figsize=(8, 4))
            axis.plot([point[0] for point in mcts_points], [point[1] for point in mcts_points])
            axis.set(xlabel="Training episode", ylabel="Reward-model parameter norm",
                     title="Belief-MCTS Bayesian reward-model progress")
            fig.tight_layout()
            path = output / "train_belief_mcts_reward_model.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)
        ppo_keys = ("mean_policy_loss", "mean_value_loss", "mean_entropy", "mean_kl", "mean_clip_fraction")
        if updates and any(key in updates[-1].get("diagnostics", {}) for key in ppo_keys):
            fig, axis = plt.subplots(figsize=(9, 5))
            for key in ppo_keys:
                points = [(row["episode"] + 1, row["diagnostics"][key]) for row in updates
                          if key in row.get("diagnostics", {})]
                if points:
                    axis.plot([p[0] for p in points], [p[1] for p in points],
                              label=key.removeprefix("mean_").replace("_", " "))
            axis.set(xlabel="Training episode", ylabel="PPO statistic",
                     title="PPO-LSTM policy and value update metrics")
            axis.legend()
            fig.tight_layout()
            path = output / "train_ppo_lstm_metrics.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)
    for split in ("val", "test"):
        for path in sorted((run / "eval" / split).rglob("summary.json")):
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
            if "checkpoints" in path.relative_to(run / "eval" / split).parts:
                suffix = "_" + Path(report["checkpoint_file"]).stem + suffix
            figure = output / f"{split}{suffix}_metrics.png"
            fig.savefig(figure, dpi=150)
            plt.close(fig)
            written.append(figure)
            run_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            algorithm_name = run_config.get("algorithm", {}).get("name")
            if algorithm_name in {"ucb1", "round_robin", "belief_ucb", "belief_mcts", "contextual_thompson"}:
                world_log = path.parent / "per_world.jsonl"
                world_rows = [json.loads(line) for line in world_log.read_text(encoding="utf-8").splitlines()]
                diagnostic_rows = [row for row in world_rows if row.get("policy_diagnostics")]
                if diagnostic_rows:
                    counts = np.asarray([row["policy_diagnostics"]["band_selection_counts"]
                                         for row in diagnostic_rows], dtype=float)
                    fig, axis = plt.subplots(figsize=(11, 4.5))
                    centers = counts.mean(axis=0)
                    errors = counts.std(axis=0, ddof=1) if len(counts) > 1 else np.zeros(counts.shape[1])
                    axis.bar(np.arange(counts.shape[1]), centers, yerr=errors, capsize=2)
                    axis.set(xlabel="Band index", ylabel="Selections per world",
                             title=f"{report['label']} {algorithm_name.replace('_', ' ')} band selections (mean ± SD)")
                    fig.tight_layout()
                    figure = output / f"{split}{suffix}_{algorithm_name}_band_counts.png"
                    fig.savefig(figure, dpi=150)
                    plt.close(fig)
                    written.append(figure)

                    fig, axis = plt.subplots(figsize=(12, 7))
                    image = axis.imshow(counts, aspect="auto", interpolation="nearest", cmap="viridis")
                    axis.set(xlabel="Band index", ylabel="World index",
                             title=f"{report['label']} {algorithm_name.replace('_', ' ')} per-world band selections")
                    fig.colorbar(image, ax=axis, label="Decision count")
                    fig.tight_layout()
                    figure = output / f"{split}{suffix}_{algorithm_name}_band_counts_by_world.png"
                    fig.savefig(figure, dpi=150)
                    plt.close(fig)
                    written.append(figure)

                    steps_log = path.parent / "steps.jsonl"
                    decisions_by_world = {}
                    for item in map(json.loads, steps_log.read_text(encoding="utf-8").splitlines()):
                        reward = item.get("decision_reward")
                        if reward is not None:
                            decisions_by_world.setdefault(item["episode"], []).append(reward)
                    curves = []
                    window = 25
                    for rows_for_world in decisions_by_world.values():
                        matrix = np.asarray([[r["reward"], r["new_intercept_utility"],
                                              -r["elapsed_cost"], -r["false_alarm_cost"]]
                                             for r in rows_for_world], dtype=float)
                        if len(matrix) >= window:
                            smoothed = np.vstack([
                                np.convolve(matrix[:, col], np.ones(window) / window, mode="valid")
                                for col in range(matrix.shape[1])
                            ]).T
                            curves.append(smoothed)
                    if curves:
                        max_length = max(map(len, curves))
                        aggregate = np.full((len(curves), max_length, 4), np.nan)
                        for index, curve in enumerate(curves):
                            aggregate[index, :len(curve)] = curve
                        mean = np.nanmean(aggregate, axis=0)
                        fig, axis = plt.subplots(figsize=(9, 5))
                        labels = ("Total v2 reward", "First-intercept utility",
                                  "Negative elapsed cost", "Negative false-alarm cost")
                        for column, label in enumerate(labels):
                            axis.plot(np.arange(window, window + max_length), mean[:, column], label=label)
                        axis.set(xlabel="Band decision within world (25-decision rolling mean)",
                                 ylabel="Reward contribution",
                                 title=f"{report['label']} {algorithm_name.replace('_', ' ')} online reward components")
                        axis.legend()
                        fig.tight_layout()
                        figure = output / f"{split}{suffix}_{algorithm_name}_reward_components.png"
                        fig.savefig(figure, dpi=150)
                        plt.close(fig)
                        written.append(figure)
            if run_config.get("algorithm", {}).get("name") in {"discrete_sac", "ppo_lstm"}:
                algorithm_name = run_config["algorithm"]["name"]
                world_log = path.parent / "per_world.jsonl"
                world_rows = [json.loads(line) for line in world_log.read_text(encoding="utf-8").splitlines()]
                diagnostic_rows = [row for row in world_rows if row.get("policy_diagnostics")]
                if diagnostic_rows:
                    counts = np.asarray([row["policy_diagnostics"]["band_selection_counts"]
                                         for row in diagnostic_rows], dtype=float)
                    fig, axis = plt.subplots(figsize=(11, 4.5))
                    axis.bar(np.arange(counts.shape[1]), counts.mean(axis=0))
                    axis.set(xlabel="Band index", ylabel="Selections per world",
                             title=f"{report['label']} {algorithm_name.replace('_', ' ')} band selections")
                    fig.tight_layout()
                    figure = output / f"{split}{suffix}_{algorithm_name}_band_counts.png"
                    fig.savefig(figure, dpi=150)
                    plt.close(fig)
                    written.append(figure)
            analysis = report.get("analysis")
            if analysis:
                fig, axis = plt.subplots(figsize=(7, 4))
                for name, distribution in analysis["ttfi_distributions"].items():
                    cdf = distribution["km_cdf"]
                    if cdf:
                        axis.step([0.0] + [row["time_ms"] / 1000 for row in cdf], [0.0] + [row["cdf"] for row in cdf],
                                  where="post", label=name)
                axis.set(xlabel="Time since first recorded opportunity (s)", ylabel="Kaplan-Meier interception CDF",
                         ylim=(0, 1), title=report["label"] + " TTFI (censored emitters retained)")
                if axis.lines:
                    axis.legend()
                fig.tight_layout()
                figure = output / f"{split}{suffix}_ttfi_cdf.png"
                fig.savefig(figure, dpi=150)
                plt.close(fig)
                written.append(figure)
                points = analysis["agility_and_density_points"]
                fig, axes = plt.subplots(2, 3, figsize=(13, 8))
                features = ("mean_emitter_channel_transition_rate_hz", "mean_complete_channel_dwell_s", "mean_frequency_span_mhz")
                labels = ("Mean observed channel-change rate (Hz)", "Mean complete channel dwell (s)", "Mean frequency span (MHz)")
                for axis_row, metric, ylabel, scale in (
                        (axes[0], "oir", "Recorded OIR", 1.0),
                        (axes[1], "policy_ttfi_rmst_ms", "Restricted mean TTFI (s)", 0.001)):
                    for axis, feature, label in zip(axis_row, features, labels):
                        for regime in ("low", "medium", "high"):
                            selected = [row for row in points if row["agility_regime"] == regime
                                        and row[feature] is not None and row[metric] is not None]
                            if selected:
                                axis.scatter([row[feature] for row in selected],
                                             [row[metric] * scale for row in selected], label=regime)
                        axis.set(xlabel=label, ylabel=ylabel)
                        if metric == "oir":
                            axis.set_ylim(0, 1)
                        if axis.collections:
                            axis.legend()
                fig.suptitle(report["label"] + " native agility")
                fig.tight_layout()
                figure = output / f"{split}{suffix}_agility.png"
                fig.savefig(figure, dpi=150)
                plt.close(fig)
                written.append(figure)
    for split in ("val", "test"):
        matrix_path = run / "eval" / split / "spatial_agility_matrix.json"
        if not matrix_path.is_file():
            continue
        matrix_record = json.loads(matrix_path.read_text(encoding="utf-8"))
        conditions = list(matrix_record["spatial_condition_by_train_derived_frequency_agility"])
        regimes = ("low", "medium", "high")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for axis, metric, label in (
            (axes[0], "opportunity_interception_ratio", "Recorded OIR"),
            (axes[1], "unique_emitter_interception_rate", "Unique emitter interception rate"),
        ):
            values = np.full((len(regimes), len(conditions)), np.nan)
            for ci, condition_name in enumerate(conditions):
                for ri, regime in enumerate(regimes):
                    summary = matrix_record["spatial_condition_by_train_derived_frequency_agility"][condition_name].get(regime)
                    if summary is not None:
                        value = summary.get(metric)
                        if value is not None:
                            values[ri, ci] = value
            width = 0.24
            x = np.arange(len(conditions))
            for ri, regime in enumerate(regimes):
                axis.bar(x + (ri - 1) * width, values[ri], width, label=regime)
            axis.set_xticks(x, conditions, rotation=15)
            axis.set(ylabel=label, ylim=(0, 1), title=label)
            axis.legend(title="TRAIN agility")
        fig.suptitle(f"{split.upper()} spatial condition × native frequency agility")
        fig.tight_layout()
        figure = output / f"{split}_spatial_agility_matrix.png"
        fig.savefig(figure, dpi=150)
        plt.close(fig)
        written.append(figure)
    # A separate model-selection curve is useful when each saved checkpoint
    # has been replayed on the same VAL worlds.
    for split in ("val", "test"):
        checkpoint_reports = []
        final_report = run / "eval" / split / "normal" / "summary.json"
        if final_report.is_file():
            report = json.loads(final_report.read_text(encoding="utf-8"))
            if report.get("checkpoint_file"):
                checkpoint_reports.append(report)
        for summary_path in (run / "eval" / split / "checkpoints").rglob("summary.json"):
            report = json.loads(summary_path.read_text(encoding="utf-8"))
            if report.get("condition") == "normal" and report.get("checkpoint_file"):
                checkpoint_reports.append(report)
        if len(checkpoint_reports) > 1:
            checkpoint_reports.sort(key=lambda report: report["checkpoint_file"])
            fig, axis = plt.subplots(figsize=(8, 4.5))
            for metric, label in (("opportunity_interception_ratio", "OIR"),
                                  ("unique_emitter_interception_rate", "Emitter interception rate")):
                axis.plot([r["checkpoint_file"] for r in checkpoint_reports],
                          [r["summary"].get(metric) for r in checkpoint_reports], marker="o", label=label)
            axis.set(xlabel="Checkpoint", ylabel="VAL_NORMAL metric", ylim=(0, 1),
                     title="Checkpoint-by-checkpoint VAL selection metrics")
            axis.tick_params(axis="x", rotation=20)
            axis.legend()
            fig.tight_layout()
            figure = output / f"{split}_checkpoint_metrics.png"
            fig.savefig(figure, dpi=150)
            plt.close(fig)
            written.append(figure)
    return written


def evaluate(run_dir: str | Path, split: str, *, final: bool = False,
             condition: str = "normal", checkpoint: str | None = None) -> dict:
    """Evaluate a frozen checkpoint or fixed online baseline on held-out worlds."""
    if split not in {"val", "test"} or (split == "test" and not final):
        raise ValueError("TEST requires explicit final=True; split must be val or test")
    run = Path(run_dir).resolve()
    config = load_contract(run / "config.json")
    baseline = config.get("execution_mode") == "online_baseline"
    if baseline and checkpoint is not None:
        raise ValueError("The online baseline has no checkpoint")
    conditions = config["evaluation"].get("illumination_conditions", {"normal": {}})
    if condition == "all":
        condition_reports = {
            name: evaluate(run, split, final=final, condition=name, checkpoint=checkpoint) for name in conditions
        }
        matrix = {
            spatial_condition: {
                regime: report.get("analysis", {}).get("agility_strata", {}).get(regime)
                for regime in ("low", "medium", "high", "unavailable")
            }
            for spatial_condition, report in condition_reports.items()
        }
        combined = {"split": split, "primary_metric": "opportunity_interception_ratio",
                    "spatial_condition_by_train_derived_frequency_agility": matrix,
                    "conditions": condition_reports}
        if config["evaluation"].get("require_frozen_world_catalog"):
            _, catalog_sha256 = load_world_catalog(run, _hash(run / "config.json"))
            combined["frozen_world_catalog_sha256"] = catalog_sha256
            _json(run / "eval" / split / "spatial_agility_matrix.json", {
                key: value for key, value in combined.items() if key != "conditions"})
        return combined
    if condition not in conditions:
        raise ValueError(f"Illumination condition is not in the frozen contract: {condition}")
    settings = IlluminationSettings(mode=condition, **conditions[condition])
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if _hash(run / "config.json") != status.get("config_sha256"):
        raise ValueError("Experiment config changed after training")
    if status.get("state") != "complete":
        raise ValueError("Training must complete before evaluation")
    runtime_path = run / "runtime_manifest.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("algorithm") is not None and runtime["algorithm"] != algorithm_provenance(config):
        raise ValueError("Algorithm adapter source changed since training")
    policy_hash = algorithm_fingerprint(config) if baseline else None
    if baseline and status.get("algorithm_sha256") != policy_hash:
        raise ValueError("Online baseline algorithm changed after run initialization")
    selected_hash = None
    selection_required = config.get("val_selection_required", config.get("checkpoint_selection_required", False))
    if split == "test" and selection_required:
        selection_path = run / "selection.json"
        if not selection_path.exists():
            raise ValueError("Freeze a VAL selection before final TEST evaluation")
        selection = json.loads(selection_path.read_text())
        if selection["config_sha256"] != status["config_sha256"]:
            raise ValueError("Selection belongs to a different config")
        if baseline:
            if selection.get("artifact_type") != "online_baseline" or selection.get("policy_sha256") != policy_hash:
                raise ValueError("TEST can only evaluate the online algorithm frozen from VAL")
            selected_hash = selection["policy_sha256"]
            val_name = None
        else:
            if selection.get("artifact_type") != "checkpoint":
                raise ValueError("TEST requires a checkpoint selected from VAL")
            if checkpoint is not None and checkpoint != selection["checkpoint_file"]:
                raise ValueError("TEST can only evaluate the checkpoint frozen from VAL")
            checkpoint = selection["checkpoint_file"]
            selected_hash = selection["checkpoint_sha256"]
            val_name = checkpoint
        validation_path = evaluation_directory(run, config, val_name, "val", "normal") / "summary.json"
        if _hash(validation_path) != selection["validation_summary_sha256"]:
            raise ValueError("Selected VAL report changed after selection")
        if config["evaluation"].get("require_frozen_world_catalog"):
            _, current_catalog_sha256 = load_world_catalog(run, status["config_sha256"])
            if selection.get("frozen_world_catalog_sha256") != current_catalog_sha256:
                raise ValueError("Frozen world catalog changed after VAL selection")
    if baseline:
        frozen_hash = policy_hash
        if selected_hash is not None and frozen_hash != selected_hash:
            raise ValueError("Selected online algorithm changed")
        checkpoint_path = None
        checkpoint_name = None
    else:
        checkpoint_path, frozen_hash = resolve_checkpoint(run, config, status, checkpoint)
        checkpoint_name = checkpoint_path.name
        if selected_hash is not None and frozen_hash != selected_hash:
            raise ValueError("Selected checkpoint changed")
    target = evaluation_directory(run, config, checkpoint_name, split, condition)
    if target.exists():
        # A completed report is immutable by default. An interrupted evaluation
        # has no summary.json commit marker, so discard its partial artifacts
        # and replay the same frozen worlds deterministically.
        if (target / "summary.json").exists():
            raise FileExistsError(target)
        shutil.rmtree(target)
    data = Path(config["data_root"]).resolve()
    frozen_catalog = None
    frozen_catalog_sha256 = None
    if config["evaluation"].get("require_frozen_world_catalog"):
        if config["evaluation"].get("world_mode") != "emitter_recombined":
            raise ValueError("Frozen world catalog requires emitter_recombined evaluation")
        frozen_catalog, frozen_catalog_sha256 = load_world_catalog(run, status["config_sha256"])
    ids = _evaluation_ids(
        data, split, config["evaluation"][f"{split}_config_ids"],
        config["evaluation"].get(f"expected_{split}_configs"),
    )
    composed = config["evaluation"].get(f"{split}_composed_worlds", [])
    composed_only = config["evaluation"].get("world_mode") == "emitter_recombined"
    protocol = config.get("analysis_protocol")
    reference = None
    if protocol:
        reference_path = run / "agility_reference.json"
        if _hash(reference_path) != config.get("agility_reference_sha256"):
            raise ValueError("Frozen TRAIN agility reference changed after training")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if reference["fit_split"] != "train":
            raise ValueError("Agility thresholds must be fitted on TRAIN only")
    target.mkdir(parents=True)
    worlds_record = {"split": split, "config_ids": ids,
                                   "contract_version": config["contract_version"],
                                   "detector_model": "recorded_stare_bernoulli",
                                   "training_reward": config["reward"],
                                   "retune_time_ms": config["receiver"]["retune_time_ms"],
                                   "dwell_slots_by_band": config["action"].get("dwell_slots_by_band"),
                                   "condition": condition,
                                   "illumination_settings": conditions[condition],
                                   "illumination_seed_base": config["evaluation"].get("illumination_seed", 20261004),
                                   "composed_worlds": composed,
                                   "generator_version": GENERATOR_VERSION,
                                   "world_mode": config["evaluation"].get("world_mode", "source_replay"),
                                   "world_settings": config.get("world", {}),
                                   "receiver_seed_base": config["evaluation"]["receiver_seed"]}
    worlds_record["policy_sha256" if baseline else "checkpoint_sha256"] = frozen_hash
    if frozen_catalog_sha256 is not None:
        worlds_record["frozen_world_catalog_sha256"] = frozen_catalog_sha256
    if baseline:
        worlds_record["policy_type"] = "online_baseline"
    else:
        worlds_record["checkpoint_file"] = checkpoint_name
    _json(target / "worlds.json", worlds_record)
    scores = []
    analysis_rows = []
    evaluation_world_count = len(composed) if composed_only else len(ids)
    print(f"[EVAL] {split.upper()}_{condition.upper()}: "
          f"replaying {evaluation_world_count} frozen worlds", flush=True)
    completed_worlds = 0

    def report_world_progress(row):
        nonlocal completed_worlds
        completed_worlds += 1
        oir = row["scorecard"]["cell_level"].get("occupied_cell_detection_ratio")
        oir_text = "n/a" if oir is None else f"{oir:.4f}"
        print(f"[EVAL] {split.upper()}_{condition.upper()} "
              f"world {completed_worlds}/{evaluation_world_count}: {row['config_id']} "
              f"OIR={oir_text}", flush=True)

    def enhance(row, world, native_agility, predictions, trajectory, illumination, seed):
        if not protocol:
            return row
        native_agility["regime"] = classify_rate(native_agility["mean_emitter_channel_transition_rate_hz"], reference)
        native_agility["basis"] = "complete unmodified source trajectories before illumination stress"
        row["agility"] = native_agility
        row["agility_reference_sha256"] = config["agility_reference_sha256"]
        row["prediction_metrics"] = score_predictions(predictions, world, trajectory)
        _json(target / "prediction_logs" / (row["config_id"] + ".json"), {"predictions": predictions})
        oracle = evaluate_privileged(world, receiver_seed=seed, illumination=illumination)
        _json(target / "oracle_logs" / (row["config_id"] + ".json"), oracle)
        row["oracle_comparison"] = compare_privileged(row["scorecard"], oracle)
        analysis_rows.append(row)
        return row

    for index, config_id in enumerate([] if composed_only else ids):
        seed = _seed(int(config["evaluation"]["receiver_seed"]), split, config_id)
        world = build_stare_evaluation_world(data, split, config_id,
                                             receiver_seed=seed, **_receiver_options(config))
        native_agility = world_features(*world.recorded_pulses, channel_width_mhz=reference["channel_width_mhz"]) if protocol else None
        illumination_seed = _seed(int(config["evaluation"].get("illumination_seed", 20261004)),
                                  split, config_id)
        world, illumination = apply_evaluation_illumination(
            world, split=split, settings=settings, illumination_seed=illumination_seed,
            receiver_seed=seed,
        )
        frozen_identity = None
        if frozen_catalog is not None:
            source_hashes = {config_id: _hash(data / "stare" / f"{split}_stare" / f"{config_id}.h5")}
            frozen_identity = verify_catalog_rebuild(
                frozen_catalog, split=split, world_id=config_id, condition=condition,
                recipe={"id": config_id, "source_config_ids": [config_id]},
                source_file_sha256=source_hashes, receiver_seed=seed,
                illumination_seed=illumination_seed,
                visibility_mask_sha256=illumination["visibility_mask_sha256"],
                replay_signature=world.replay_signature())
        _json(target / "world_logs" / f"{config_id}.json", illumination)
        policy = create_algorithm(config, bands=world.n_bands, seed=seed, checkpoint=checkpoint_path,
                                  restore_rng=False)
        predictions = []
        trajectory, _, reward_components = _run_episode(
            world, policy, training=False, episode=index, log_path=target / "steps.jsonl",
            prediction_rows=predictions if protocol else None, config=config)
        score = score_recorded_replay(world, trajectory)
        annotate_illumination_score(score, illumination, world.config.slot_duration_s())
        scores.append(score)
        source = data / "stare" / f"{split}_stare" / f"{config_id}.h5"
        row = {
            "config_id": config_id, "receiver_seed": seed,
            "condition": condition, "illumination_seed": illumination_seed,
            "illumination_audit_file": f"world_logs/{config_id}.json",
            "source_recorded_pulses_in_mission": illumination["source_recorded_pulses_in_mission"],
            "visible_recorded_pulses_in_mission": illumination["visible_recorded_pulses_in_mission"],
            "source_sha256": _hash(source), "replay_signature": world.replay_signature(),
            "receiver_accounting": world.receiver_accounting(), "scorecard": score,
            "policy_reward_components": reward_components,
            "policy_diagnostics": reward_components.get("policy_diagnostics"),
        }
        if frozen_identity is not None:
            row["frozen_world_identity_sha256"] = frozen_identity
        logged_row = enhance(row, world, native_agility, predictions, trajectory, illumination, seed)
        _append(target / "per_world.jsonl", logged_row)
        report_world_progress(logged_row)
    for offset, recipe in enumerate(composed):
        sources = list(recipe["source_config_ids"])
        if any(source not in ids for source in sources):
            raise ValueError("Composed world must use fixed held-out source list")
        seed = _seed(int(config["evaluation"]["receiver_seed"]), split, recipe["id"])
        world, provenance = compose_heldout_world(
            data, split, sources, world_seed=int(recipe["world_seed"]),
            emitter_count=int(recipe["emitter_count"]), receiver_seed=seed,
            time_offset_us=config.get("world", {}).get("time_offset_us", DEFAULT_TIME_OFFSET_US),
            **_receiver_options(config),
        )
        native_agility = world_features(*world.recorded_pulses, channel_width_mhz=reference["channel_width_mhz"]) if protocol else None
        illumination_seed = _seed(int(config["evaluation"].get("illumination_seed", 20261004)),
                                  split, recipe["id"])
        world, illumination = apply_evaluation_illumination(
            world, split=split, settings=settings, illumination_seed=illumination_seed,
            receiver_seed=seed,
        )
        source_hashes = {source: _hash(data / "stare" / f"{split}_stare" / f"{source}.h5")
                         for source in sources}
        frozen_identity = None
        if frozen_catalog is not None:
            frozen_identity = verify_catalog_rebuild(
                frozen_catalog, split=split, world_id=recipe["id"], condition=condition,
                recipe=recipe, source_file_sha256=source_hashes,
                receiver_seed=seed, illumination_seed=illumination_seed,
                visibility_mask_sha256=illumination["visibility_mask_sha256"],
                replay_signature=world.replay_signature())
        _json(target / "world_logs" / f"{recipe['id']}.json", illumination)
        policy = create_algorithm(config, bands=world.n_bands, seed=seed, checkpoint=checkpoint_path,
                                  restore_rng=False)
        predictions = []
        trajectory, _, reward_components = _run_episode(
            world, policy, training=False, episode=len(ids) + offset,
            log_path=target / "steps.jsonl", prediction_rows=predictions if protocol else None,
            config=config)
        score = score_recorded_replay(world, trajectory)
        annotate_illumination_score(score, illumination, world.config.slot_duration_s())
        scores.append(score)
        row = {
            "config_id": recipe["id"], "kind": "composed_heldout",
            "condition": condition, "illumination_seed": illumination_seed,
            "illumination_audit_file": f"world_logs/{recipe['id']}.json",
            "world_seed": int(recipe["world_seed"]), "receiver_seed": seed,
            "sources": provenance,
            "source_sha256": source_hashes,
            "replay_signature": world.replay_signature(),
            "receiver_accounting": world.receiver_accounting(), "scorecard": score,
            "policy_reward_components": reward_components,
            "policy_diagnostics": reward_components.get("policy_diagnostics"),
        }
        if frozen_identity is not None:
            row["frozen_world_identity_sha256"] = frozen_identity
        logged_row = enhance(row, world, native_agility, predictions, trajectory, illumination, seed)
        _append(target / "per_world.jsonl", logged_row)
        report_world_progress(logged_row)
    if baseline:
        if algorithm_fingerprint(config) != frozen_hash:
            raise RuntimeError("Frozen online algorithm changed during evaluation")
    elif _hash(checkpoint_path) != frozen_hash:
        raise RuntimeError("Frozen checkpoint changed during evaluation")
    report = {"split": split, "condition": condition,
              "detector_model": "recorded_stare_bernoulli",
              "training_reward": config["reward"],
              "retune_time_ms": config["receiver"]["retune_time_ms"],
              "dwell_slots_by_band": config["action"].get("dwell_slots_by_band"),
              "label": f"{split.upper()}_{condition.upper()}",
              "opportunity_basis": "recorded pulses visible under the configured illumination gate",
              "contract_version": config["contract_version"], "summary": _summary(scores)}
    if baseline:
        report.update(policy_type="online_baseline", algorithm_name=config["algorithm"]["name"],
                      policy_sha256=frozen_hash)
    else:
        report.update(checkpoint_sha256=frozen_hash, checkpoint_file=checkpoint_name)
    report["per_world_sha256"] = _hash(target / "per_world.jsonl")
    report["worlds_sha256"] = _hash(target / "worlds.json")
    if protocol:
        report["analysis"] = analysis_summary(analysis_rows, protocol, _summary)
        report["analysis_protocol"] = protocol
        report["agility_reference_sha256"] = config["agility_reference_sha256"]
    _json(target / "summary.json", report)
    return report

"""Algorithm-independent public state, transitions, and loading.

Algorithms own their optimizer and replay/rollout buffers. The runner supplies
public transitions, allowing both on-policy and off-policy learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import importlib
import json
import math
import hashlib
from pathlib import Path
from typing import Protocol

from vyapti_simulator.core.receiver_observation import ReceiverObservation


# Exact source transitions that add evaluation-time action sampling only.
# They preserve training action selection, model initialization, and checkpoints.
_EVALUATION_ONLY_SOURCE_COMPATIBILITY = {
    "discrete_sac": {
        ("9a0ae779b9b66231fdfecbef9daf5d5069a163235852b90cd8bc21bbdc74779e",
         "84bade76c4f701ff2f69a61a0a31e3429b549724ff93c0cb0f6cdc6e3d778e75")
    },
}


@dataclass(frozen=True, slots=True)
class PublicState:
    time_slot: int
    previous_observation: ReceiverObservation | None

    def __post_init__(self):
        if type(self.time_slot) is not int or self.time_slot < 0:
            raise ValueError("Public state needs a nonnegative slot")
        if self.previous_observation is not None and type(self.previous_observation) is not ReceiverObservation:
            raise TypeError("Algorithm state may only carry a closed ReceiverObservation")


@dataclass(frozen=True, slots=True)
class PublicTransition:
    state: PublicState
    action: int
    reward: float
    next_state: PublicState
    terminated: bool
    truncated: bool = False

    def __post_init__(self):
        if type(self.state) is not PublicState or type(self.next_state) is not PublicState:
            raise TypeError("Transition states must be closed PublicState records")
        if (type(self.action) is not int or self.action < 0 or not math.isfinite(self.reward)
                or type(self.terminated) is not bool or type(self.truncated) is not bool):
            raise ValueError("Invalid public transition")


class Algorithm(Protocol):
    def reset_episode(self, *, training: bool) -> None: ...
    def select_action(self, state: PublicState, *, training: bool) -> int: ...
    def observe(self, transition: PublicTransition, *, training: bool) -> None: ...
    def end_episode(self, *, training: bool) -> dict | None: ...


class LegacyEpisodeAdapter:
    """Keep previously saved UCB/episode-based runs replayable."""
    def __init__(self, policy, *, reward_bounds=None):
        self.policy = policy
        self.reward_bounds = reward_bounds
        self.history = []

    def reset_episode(self, *, training):
        self.history = []
        self.policy.reset_episode()

    def select_action(self, state, *, training):
        return self.policy.select_action(tuple(self.history), state.time_slot)

    def observe(self, transition, *, training):
        observation = transition.next_state.previous_observation
        reward_observer = getattr(self.policy, "observe_reward", None)
        if self.reward_bounds is not None and callable(reward_observer):
            reward_observer(transition.action, transition.reward, self.reward_bounds)
        else:
            self.policy.observe(observation)
        self.history.append(observation)

    def end_episode(self, *, training):
        if training:
            return self.policy.finish_training_episode()
        return None

    def save(self, path):
        self.policy.save(path)

    def predict_band_activity(self, state):
        hook = getattr(self.policy, "predict_band_activity", None)
        return hook(state) if callable(hook) else None

    def predict_next_intercept_slot(self, state):
        hook = getattr(self.policy, "predict_next_intercept_slot", None)
        return hook(state) if callable(hook) else None


def algorithm_spec(config: dict) -> dict:
    if "algorithm" in config:
        return config["algorithm"]
    return {"name": config.get("policy_module", "").rsplit(".", 1)[-1],
            "module": config["policy_module"], "api": "legacy_episode",
            "settings": {}}


def create_algorithm(config: dict, *, bands: int, seed: int, checkpoint: Path | None = None,
                     restore_rng: bool = True, evaluation_action_mode: str = "greedy",
                     action_seed: int | None = None):
    spec = algorithm_spec(config)
    module = importlib.import_module(spec["module"])
    factory = getattr(module, "create", None)
    if not callable(factory):
        raise ValueError("Algorithm module must export create")
    settings = deepcopy(spec.get("settings", {}))
    # Belief-UCB evaluates the receiver-only value of an action. Supply the
    # resolved environment timing here so its per-second scores cannot drift
    # from the dwell schedule used by the runner.
    if spec.get("name") == "belief_ucb":
        action = config.get("action", {})
        receiver = config.get("receiver", {})
        dwell_profile = action.get("dwell_slots_by_band")
        if dwell_profile is not None:
            settings["dwell_slots_by_band"] = deepcopy(dwell_profile)
        if "base_slot_duration_ms" in receiver:
            settings["base_slot_seconds"] = float(receiver["base_slot_duration_ms"]) / 1000.0
        if "mission_duration_s" in receiver:
            settings["mission_duration_s"] = float(receiver["mission_duration_s"])
        retune_ms = receiver.get("band_change_retune_time_ms", receiver.get("retune_time_ms"))
        if retune_ms is not None:
            settings["band_change_retune_time_s"] = float(retune_ms) / 1000.0
    # Belief-MCTS uses native band dwell lengths inside hypothetical branches.
    # Always take them from the resolved environment contract, not a copied
    # vector in an algorithm file.
    if spec.get("name") == "belief_mcts":
        dwell_profile = config.get("action", {}).get("dwell_slots_by_band")
        if dwell_profile is None:
            raise ValueError("belief_mcts requires the environment's native dwell profile")
        settings["native_dwell_slots"] = deepcopy(dwell_profile)
    if spec.get("name") == "ppo_lstm":
        dwell_profile = config.get("action", {}).get("dwell_slots_by_band")
        if dwell_profile is not None:
            settings["native_dwell_slots"] = deepcopy(dwell_profile)
    if spec.get("name") == "contextual_thompson":
        dwell_profile = config.get("action", {}).get("dwell_slots_by_band")
        if dwell_profile is None:
            raise ValueError("contextual_thompson requires the environment's native dwell profile")
        settings["native_dwell_slots"] = deepcopy(dwell_profile)
    if spec.get("name") == "recurrent_distributional_dqn":
        dwell_profile = config.get("action", {}).get("dwell_slots_by_band")
        if dwell_profile is None:
            raise ValueError("recurrent_distributional_dqn requires the environment's native dwell profile")
        settings["native_dwell_slots"] = deepcopy(dwell_profile)
    if spec.get("api") == "legacy_episode":
        reward_bounds = settings.pop("reward_bounds", None)
        if settings:
            raise ValueError("Legacy episode factories only accept the runner's reward_bounds setting")
        if reward_bounds is not None and (
                not isinstance(reward_bounds, list) or len(reward_bounds) != 2
                or not all(math.isfinite(float(value)) for value in reward_bounds)
                or float(reward_bounds[0]) >= float(reward_bounds[1])):
            raise ValueError("reward_bounds must be a finite [minimum, maximum] pair")
        is_v2 = config.get("contract_version") == "train250_recorded_pdw_v2"
        if not is_v2:
            reward_bounds = None
        elif reward_bounds is None:
            raise ValueError("A legacy policy used with v2 must declare reward_bounds")
        legacy_policy = factory(bands, seed, checkpoint)
        if reward_bounds is not None and not callable(getattr(legacy_policy, "observe_reward", None)):
            raise ValueError("A legacy policy used with v2 must implement observe_reward")
        algorithm = LegacyEpisodeAdapter(legacy_policy, reward_bounds=reward_bounds)
    elif spec.get("api") == "public_transitions":
        if spec.get("name") == "contextual_thompson":
            algorithm = factory(bands=bands, seed=seed, settings=settings,
                                checkpoint=checkpoint, restore_rng=restore_rng)
        else:
            algorithm = factory(bands=bands, seed=seed, settings=settings, checkpoint=checkpoint)
    else:
        raise ValueError("Algorithm api must be public_transitions or legacy_episode")
    for method in ("reset_episode", "select_action", "observe", "end_episode"):
        if not callable(getattr(algorithm, method, None)):
            raise ValueError(f"Algorithm is missing {method}")
    if evaluation_action_mode != "greedy":
        if spec.get("name") not in {"ppo_lstm", "discrete_sac"}:
            raise ValueError("Sampled evaluation is currently supported only for ppo_lstm and discrete_sac")
        setter = getattr(algorithm, "set_evaluation_action_mode", None)
        if not callable(setter):
            raise ValueError(f"{spec.get('name')} does not support sampled evaluation")
        setter(evaluation_action_mode, action_seed)
    return algorithm


def algorithm_provenance(config: dict) -> dict:
    spec = deepcopy(algorithm_spec(config))
    module = importlib.import_module(spec["module"])
    source = getattr(module, "__file__", None)
    spec["adapter_source"] = None
    if source is not None and Path(source).is_file():
        path = Path(source).resolve()
        spec["adapter_source"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    calibration = config.get("belief_model_calibration")
    if calibration is not None:
        calibration_source = config.get("setup_sources", {}).get("belief_model", {})
        spec["belief_model_calibration"] = {
            "calibration_id": calibration["calibration_id"],
            "source_sha256": calibration_source.get("sha256"),
            "source_pool_fingerprint": calibration["source"]["source_pool_fingerprint"],
            "prior_active_probability": calibration["prior_active_probability"],
            "inactive_to_active_probability": calibration["inactive_to_active_probability"],
            "active_to_active_probability": calibration["active_to_active_probability"],
        }
    return spec


def algorithm_provenance_matches(config: dict, recorded: dict) -> bool:
    """Match run provenance independent of machine paths and audited eval-only edits."""
    current = algorithm_provenance(config)
    recorded_normalized = deepcopy(recorded)
    current_normalized = deepcopy(current)
    recorded_source = recorded_normalized.pop("adapter_source", None)
    current_source = current_normalized.pop("adapter_source", None)
    if recorded_normalized != current_normalized:
        return False
    if not isinstance(recorded_source, dict) or not isinstance(current_source, dict):
        return recorded_source == current_source
    old_digest = recorded_source.get("sha256")
    new_digest = current_source.get("sha256")
    name = algorithm_spec(config).get("name")
    return old_digest == new_digest or (old_digest, new_digest) in _EVALUATION_ONLY_SOURCE_COMPATIBILITY.get(name, set())


def algorithm_fingerprint(config: dict) -> str:
    """Content identity for policy code/settings when no weights are saved."""
    provenance = algorithm_provenance(config)
    encoded = json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

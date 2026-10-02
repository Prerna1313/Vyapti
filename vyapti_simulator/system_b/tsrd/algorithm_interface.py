"""Algorithm-independent public state, transitions, and loading.

Algorithms own their optimizer and replay/rollout buffers. The runner supplies
public transitions, allowing both on-policy and off-policy learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import importlib
import math
import hashlib
from pathlib import Path
from typing import Protocol

from vyapti_simulator.core.receiver_observation import ReceiverObservation


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
    def save(self, path: Path) -> None: ...


class LegacyEpisodeAdapter:
    """Keep previously saved UCB/episode-based runs replayable."""
    def __init__(self, policy):
        self.policy = policy
        self.history = []

    def reset_episode(self, *, training):
        self.history = []
        self.policy.reset_episode()

    def select_action(self, state, *, training):
        return self.policy.select_action(tuple(self.history), state.time_slot)

    def observe(self, transition, *, training):
        observation = transition.next_state.previous_observation
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


def create_algorithm(config: dict, *, bands: int, seed: int, checkpoint: Path | None = None):
    spec = algorithm_spec(config)
    module = importlib.import_module(spec["module"])
    factory = getattr(module, "create", None)
    if not callable(factory):
        raise ValueError("Algorithm module must export create")
    settings = deepcopy(spec.get("settings", {}))
    if spec.get("api") == "legacy_episode":
        if settings:
            raise ValueError("Legacy episode factories do not accept algorithm settings")
        algorithm = LegacyEpisodeAdapter(factory(bands, seed, checkpoint))
    elif spec.get("api") == "public_transitions":
        algorithm = factory(bands=bands, seed=seed, settings=settings, checkpoint=checkpoint)
    else:
        raise ValueError("Algorithm api must be public_transitions or legacy_episode")
    for method in ("reset_episode", "select_action", "observe", "end_episode", "save"):
        if not callable(getattr(algorithm, method, None)):
            raise ValueError(f"Algorithm is missing {method}")
    return algorithm


def algorithm_provenance(config: dict) -> dict:
    spec = deepcopy(algorithm_spec(config))
    module = importlib.import_module(spec["module"])
    source = getattr(module, "__file__", None)
    spec["adapter_source"] = None
    if source is not None and Path(source).is_file():
        path = Path(source).resolve()
        spec["adapter_source"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return spec

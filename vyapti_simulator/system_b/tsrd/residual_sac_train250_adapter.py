"""WorldFactory bridge for the imported residual Discrete-SAC experiment.

The learner receives only per-slot detector outcomes and scalar rewards.  This
adapter delegates source loading, composition, receiver simulation, illumination
and benchmark scoring to the existing Mode-B implementation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation_illumination import (
    IlluminationSettings,
    annotate_illumination_score,
    apply_evaluation_illumination,
)
from .experiment import (
    _eligible_world_emitter_first_slots,
    _emitters_with_prior_opportunity,
    _receiver_options,
    _strongest_emitter_in_selected_window,
)
from .frozen_world_catalog import _valid_catalog_hash, load_world_catalog, verify_catalog_rebuild
from .replay_scorecard import score_recorded_replay
from .train250_cache import build_train_pool_from_cache, compose_heldout_world


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


class ResidualWorld:
    """Single-world wrapper exposing the uploaded learner's expected API."""

    def __init__(self, world, dwell_slots: list[int], *, fixed_seed: int | None = None,
                 illumination_audit: dict[str, Any] | None = None,
                 frozen_identity: str | None = None):
        self.world = world
        self.native_dwell_slots = np.asarray(dwell_slots, dtype=np.int32)
        self._fixed_seed = fixed_seed
        self.illumination_audit = illumination_audit
        self.frozen_identity = frozen_identity
        self._first_slots: dict[int, int] = {}
        self._eligible: set[int] = set()
        self._intercepted: set[int] = set()

    def dwell_slots_for_band(self, band: int) -> int:
        return int(self.native_dwell_slots[int(band)])

    def reset(self, seed: int | None = None):
        actual_seed = self._fixed_seed if self._fixed_seed is not None else seed
        if actual_seed is None:
            raise ValueError("Residual world reset requires a receiver seed")
        self.world.reset(seed=int(actual_seed))
        self._first_slots = _eligible_world_emitter_first_slots(self.world)
        self._eligible = set(self._first_slots)
        self._intercepted = set()
        return None

    @property
    def done(self) -> bool:
        return bool(self.world.done)

    def step_dwell_training(self, band: int, dwell_slots: int, reward_mode: str):
        if reward_mode != "truth_based_intercept_utility_v2":
            raise ValueError(f"Unsupported shared reward mode: {reward_mode}")
        if self.done:
            raise RuntimeError("Episode is finished; reset before stepping")
        band = int(band)
        dwell_slots = int(dwell_slots)
        if dwell_slots not in (1, 2) or dwell_slots != self.dwell_slots_for_band(band):
            raise ValueError("Learner action must use the receiver's native dwell")

        start_slot = int(self.world.current_slot)
        actual_slots = min(dwell_slots, self.world.n_slots - start_slot)
        unresolved_before = len(
            _emitters_with_prior_opportunity(self._first_slots, start_slot) - self._intercepted
        )
        looks: list[dict[str, Any]] = []
        newly_intercepted: set[int] = set()
        false_alarms = 0
        empty_opportunities = 0

        for _ in range(actual_slots):
            observation, _done = self.world.step(band)
            look = dict(observation)
            looks.append(look)
            emitter_id = _strongest_emitter_in_selected_window(
                self.world,
                band,
                int(look["time_slot"]),
                float(look["retune_cost_s"]),
            )
            if emitter_id is None:
                empty_opportunities += 1
                if bool(look["hit"]):
                    false_alarms += 1
            elif bool(look["hit"]) and emitter_id in self._eligible:
                if emitter_id not in self._intercepted:
                    newly_intercepted.add(emitter_id)

        eligible_count = len(self._eligible)
        dwell_seconds = actual_slots * self.world.config.slot_duration_s()
        retune_seconds = float(looks[0]["retune_cost_s"])
        elapsed_seconds = dwell_seconds + retune_seconds
        new_intercept_utility = len(newly_intercepted) / eligible_count if eligible_count else 0.0
        unresolved_fraction = unresolved_before / eligible_count if eligible_count else 0.0
        false_alarm_rate = false_alarms / empty_opportunities if empty_opportunities else 0.0
        elapsed_cost = unresolved_fraction * elapsed_seconds / 30.0
        false_alarm_cost = false_alarm_rate * elapsed_seconds / 30.0
        reward = new_intercept_utility - elapsed_cost - false_alarm_cost
        self._intercepted.update(newly_intercepted)
        return looks, float(reward), self.done


class Train250WorldFactory:
    """Compose fresh TRAIN worlds and rebuild shared frozen VAL/TEST recipes."""

    def __init__(self):
        env_file = Path(os.environ.get(
            "VYAPTI_RESIDUAL_ENVIRONMENT",
            "training_setup/environments/train250_composed.json",
        )).resolve()
        self.environment = json.loads(env_file.read_text(encoding="utf-8"))
        self.data_root = _resolve_path(env_file.parent, self.environment["data_root"])
        self.cache_root = _resolve_path(env_file.parent, self.environment["cache_root"])
        self.receiver_options = _receiver_options(self.environment)
        self.dwell_slots = list(self.environment["action"]["dwell_slots_by_band"])
        self.condition = os.environ.get("VYAPTI_RESIDUAL_CONDITION", "normal")
        catalog_source = Path(os.environ.get(
            "VYAPTI_RESIDUAL_WORLD_CATALOG",
            "runs/round_robin/frozen_world_catalog.json",
        )).resolve()
        if catalog_source.is_dir():
            self.protocol_run = catalog_source
            catalog_path = catalog_source / "frozen_world_catalog.json"
            rr_config_path = catalog_source / "config.json"
            self.catalog, self.catalog_sha256 = load_world_catalog(
                catalog_source, _sha256(rr_config_path)
            )
        else:
            self.protocol_run = catalog_source.parent
            catalog_path = catalog_source
            self.catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            if not _valid_catalog_hash(self.catalog):
                raise ValueError(f"Invalid frozen-world catalog: {catalog_path}")
            self.catalog_sha256 = _sha256(catalog_path)
        self.catalog_path = catalog_path
        if self.catalog.get("receiver_settings") != self.environment.get("receiver"):
            raise ValueError("Frozen baseline catalog receiver settings differ from TRAIN-250 environment")
        if self.catalog.get("world_settings") != self.environment.get("world"):
            raise ValueError("Frozen baseline catalog world settings differ from TRAIN-250 environment")
        if self.condition not in self.catalog.get("conditions", {}):
            raise ValueError(f"Condition {self.condition!r} is absent from the frozen world catalog")
        self.condition_settings = self.catalog["conditions"][self.condition]
        self._train_pool = None

    def _pool(self):
        if self._train_pool is None:
            self._train_pool, _ = build_train_pool_from_cache(
                self.data_root,
                self.cache_root,
                expected_configs=int(self.environment.get("expected_train_configs", 250)),
                **self.receiver_options,
            )
        return self._train_pool

    def make_train_world(self, seed: int):
        pool = self._pool()
        rng = np.random.default_rng(int(seed))
        distribution = pool.cache["emitter_count_distribution"]
        emitter_count = int(rng.choice(distribution))
        world, _sources = pool.sample_world(int(seed), emitter_count)
        return ResidualWorld(world, self.dwell_slots)

    def _recipes(self, split: str) -> list[dict[str, Any]]:
        split_record = self.catalog["splits"][split]
        result = []
        for i, (world_id, entry) in enumerate(split_record["worlds"].items()):
            cond = entry["conditions"][self.condition]
            result.append({
                **entry["recipe"],
                "world_id": i,
                "catalog_world_id": world_id,
                "receiver_seed": entry["receiver_seed"],
                "source_file_sha256": entry["source_file_sha256"],
                "illumination_seed": cond["illumination_seed"],
                "illumination_settings": cond["illumination_settings"],
            })
        return result

    def load_val_recipes(self):
        return self._recipes("val")

    def load_test_recipes(self):
        return self._recipes("test")

    def _heldout_world(self, recipe: dict[str, Any], split: str):
        world_id = recipe["catalog_world_id"]
        source_ids = list(recipe["source_config_ids"])
        world, _ = compose_heldout_world(
            self.data_root,
            split,
            source_ids,
            world_seed=int(recipe["world_seed"]),
            emitter_count=int(recipe["emitter_count"]),
            receiver_seed=int(recipe["receiver_seed"]),
            time_offset_us=int(self.environment["world"]["time_offset_us"]),
            **self.receiver_options,
        )
        world, audit = apply_evaluation_illumination(
            world,
            split=split,
            settings=IlluminationSettings(
                mode=self.condition,
                **recipe["illumination_settings"],
            ),
            illumination_seed=int(recipe["illumination_seed"]),
            receiver_seed=int(recipe["receiver_seed"]),
        )
        hashes = {
            source_id: _sha256(self.data_root / "stare" / f"{split}_stare" / f"{source_id}.h5")
            for source_id in source_ids
        }
        frozen_identity = verify_catalog_rebuild(
            self.catalog,
            split=split,
            world_id=world_id,
            condition=self.condition,
            recipe={key: recipe[key] for key in ("id", "emitter_count", "world_seed", "source_config_ids")},
            source_file_sha256=hashes,
            receiver_seed=int(recipe["receiver_seed"]),
            illumination_seed=int(recipe["illumination_seed"]),
            visibility_mask_sha256=audit["visibility_mask_sha256"],
            replay_signature=world.replay_signature(),
        )
        return ResidualWorld(
            world,
            self.dwell_slots,
            fixed_seed=int(recipe["receiver_seed"]),
            illumination_audit=audit,
            frozen_identity=frozen_identity,
        )

    def make_val_world(self, recipe: dict[str, Any]):
        return self._heldout_world(recipe, "val")

    def make_test_world(self, recipe: dict[str, Any]):
        return self._heldout_world(recipe, "test")

    def score_episode(self, env: ResidualWorld, trajectory: list[Any]) -> dict[str, Any]:
        score = score_recorded_replay(
            env.world,
            trajectory,
            coverage_window_slots=sum(self.dwell_slots),
        )
        if env.illumination_audit is not None:
            annotate_illumination_score(
                score,
                env.illumination_audit,
                env.world.config.slot_duration_s(),
            )
        if env.frozen_identity is not None:
            score["frozen_world_identity_sha256"] = env.frozen_identity
        return score

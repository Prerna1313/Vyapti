"""Frozen, hash-verified recipes for held-out TSRD worlds.

TRAIN worlds remain ephemeral. This catalog contains only VAL/TEST recipes,
source-file hashes, deterministic seeds, and world signatures, never snapshots.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .evaluation_illumination import IlluminationSettings, apply_evaluation_illumination
from .train250_cache import compose_heldout_world
from .world_composer import GENERATOR_VERSION, DEFAULT_TIME_OFFSET_US

CATALOG_NAME = "frozen_world_catalog.json"
CATALOG_SCHEMA = "vyapti_frozen_world_catalog_v1"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _derived_seed(base: int, split: str, world_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}:{split}:{world_id}".encode()).digest()[:4], "little")


def _receiver_options(config: dict) -> dict:
    receiver = config["receiver"]
    return {
        "receiver_profile": receiver["profile"],
        "passband_halfwidth_mhz": float(receiver["passband_halfwidth_mhz"]),
        "detection_probability": float(receiver["detection_probability"]),
        "false_alarm_probability": float(receiver["false_alarm_probability"]),
        "retune_time_ms": float(receiver["retune_time_ms"]),
    }


def freeze_world_catalog(run_dir: str | Path) -> dict:
    """Rebuild and fingerprint every frozen held-out recipe without running a policy."""
    run = Path(run_dir).resolve()
    config_path = run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("evaluation", {}).get("world_mode") != "emitter_recombined":
        raise ValueError("Frozen world catalogs require emitter_recombined held-out worlds")
    target = run / CATALOG_NAME
    if target.exists():
        raise FileExistsError(target)

    data_root = Path(config["data_root"]).resolve()
    receiver_seed_base = int(config["evaluation"]["receiver_seed"])
    illumination_seed_base = int(config["evaluation"].get("illumination_seed", 20261004))
    conditions = config["evaluation"].get("illumination_conditions", {"normal": {}})
    world_settings = config.get("world", {})
    receiver_options = _receiver_options(config)
    time_offset_us = int(world_settings.get("time_offset_us", DEFAULT_TIME_OFFSET_US))
    splits = {}

    for split in ("val", "test"):
        config_ids = list(config["evaluation"][f"{split}_config_ids"])
        expected_count = config["evaluation"].get(f"expected_{split}_configs")
        if expected_count is not None and len(config_ids) != int(expected_count):
            raise ValueError(f"Frozen {split.upper()} config count does not match its contract")
        source_hashes = {}
        for config_id in config_ids:
            source = data_root / "stare" / f"{split}_stare" / f"{config_id}.h5"
            if not source.is_file():
                raise FileNotFoundError(source)
            source_hashes[config_id] = _file_hash(source)
        pool_hash = _object_hash(source_hashes)
        worlds = {}
        recipes = config["evaluation"].get(f"{split}_composed_worlds", [])
        if len(recipes) != int(config["evaluation"].get(f"expected_{split}_worlds", len(recipes))):
            raise ValueError(f"Frozen {split.upper()} world count does not match its contract")
        for recipe in recipes:
            world_id = recipe["id"]
            source_ids = list(recipe.get("source_config_ids", config_ids))
            if (not source_ids or len(set(source_ids)) != len(source_ids)
                    or any(source_id not in source_hashes for source_id in source_ids)):
                raise ValueError(f"Invalid source pool for frozen world {world_id}")
            receiver_seed = _derived_seed(receiver_seed_base, split, world_id)
            illumination_seed = _derived_seed(illumination_seed_base, split, world_id)
            source_world, _provenance = compose_heldout_world(
                data_root, split, source_ids,
                world_seed=int(recipe["world_seed"]),
                emitter_count=int(recipe["emitter_count"]),
                receiver_seed=receiver_seed, time_offset_us=time_offset_us,
                **receiver_options,
            )
            condition_records = {}
            for condition, raw_settings in conditions.items():
                settings = IlluminationSettings(mode=condition, **raw_settings)
                world, audit = apply_evaluation_illumination(
                    source_world, split=split, settings=settings,
                    illumination_seed=illumination_seed, receiver_seed=receiver_seed,
                )
                record = {
                    "illumination_seed": illumination_seed,
                    "illumination_settings": raw_settings,
                    "visibility_mask_sha256": audit["visibility_mask_sha256"],
                    "replay_signature": world.replay_signature(),
                }
                record["identity_sha256"] = _object_hash({
                    "split": split, "world_id": world_id, "recipe": recipe,
                    "source_config_ids": source_ids, "source_pool_sha256": pool_hash,
                    "receiver_seed": receiver_seed, "condition": condition, **record,
                })
                condition_records[condition] = record
            worlds[world_id] = {
                "recipe": recipe,
                "source_config_ids": source_ids,
                "world_seed": int(recipe["world_seed"]),
                "emitter_count": int(recipe["emitter_count"]),
                "receiver_seed": receiver_seed,
                "source_pool_sha256": pool_hash,
                "source_file_sha256": {key: source_hashes[key] for key in source_ids},
                "conditions": condition_records,
            }
        splits[split] = {
            "source_config_ids": config_ids,
            "source_pool_sha256": pool_hash,
            "source_file_sha256": source_hashes,
            "worlds": worlds,
        }

    catalog = {
        "schema": CATALOG_SCHEMA,
        "policy_evaluation_performed": False,
        "world_storage": "recipe_and_hash_only; worlds are deterministically rebuilt",
        "train_world_storage": "generate_per_episode_then_discard",
        "config_sha256": _file_hash(config_path),
        "generator_version": GENERATOR_VERSION,
        "world_settings": world_settings,
        "receiver_settings": config["receiver"],
        "conditions": conditions,
        "splits": splits,
    }
    catalog["catalog_sha256"] = _object_hash(catalog)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(catalog, indent=2, allow_nan=False) + "\n")
    return catalog


def load_world_catalog(run_dir: str | Path, config_sha256: str) -> tuple[dict, str]:
    path = Path(run_dir) / CATALOG_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Freeze held-out worlds before evaluation: {path}")
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog.get("schema") != CATALOG_SCHEMA or catalog.get("config_sha256") != config_sha256:
        raise ValueError("Frozen world catalog belongs to a different config or schema")
    recorded_hash = catalog.get("catalog_sha256")
    payload = dict(catalog)
    payload.pop("catalog_sha256", None)
    if recorded_hash != _object_hash(payload):
        raise ValueError("Frozen world catalog was modified after creation")
    return catalog, _file_hash(path)


def verify_catalog_rebuild(
    catalog: dict, *, split: str, world_id: str, condition: str,
    recipe: dict, source_file_sha256: dict[str, str], receiver_seed: int,
    illumination_seed: int, visibility_mask_sha256: str, replay_signature: str,
) -> str:
    """Reject a rebuilt world unless recipe, source pool, seeds, and signature match."""
    try:
        split_record = catalog["splits"][split]
        entry = split_record["worlds"][world_id]
        expected_condition = entry["conditions"][condition]
    except KeyError as exc:
        raise ValueError("World is absent from the frozen held-out catalog") from exc
    if source_file_sha256 != entry["source_file_sha256"]:
        raise ValueError("Held-out source file hashes differ from the frozen catalog")
    if (recipe != entry["recipe"] or receiver_seed != entry["receiver_seed"]
            or illumination_seed != expected_condition["illumination_seed"]
            or visibility_mask_sha256 != expected_condition["visibility_mask_sha256"]
            or replay_signature != expected_condition["replay_signature"]):
        raise ValueError("Deterministic held-out world rebuild differs from the frozen catalog")
    return expected_condition["identity_sha256"]

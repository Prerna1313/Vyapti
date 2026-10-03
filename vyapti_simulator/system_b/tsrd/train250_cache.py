"""TRAIN-250 cached emitter histories for the recorded-STARE receiver.

The cache is an input format. Every scheduler receives a TSRDStareEnvironment;
no scheduler needs access to NPZ files or source emitter identities.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .tsrd_adapter import stare_pulse_occupancy
from .tsrd_environment import TSRDStareEnvironment
from .world_composer import EmitterRealization, WorldComposer, GENERATOR_VERSION, DEFAULT_TIME_OFFSET_US, source_emitter_metadata


CACHE_SCHEMA = "vyapti_tsrd_emitter_pool_v1"
INDEX_SCHEMA = "vyapti_emitter_index_v1"
DEFAULT_CENTRES_MHZ = 250.0 + 500.0 * np.arange(36, dtype=np.float64)
PDW_FIELDS = (
    "pdw_toa_us_", "pdw_frequency_mhz_", "pdw_pulse_width_",
    "pdw_aoa_deg_", "pdw_amplitude_dbm_",
)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Cache path must be relative to its root: {relative}")
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Cache path escapes its root: {relative}")
    return resolved


def load_train250_cache(cache_root: str | Path, *, expected_configs: int = 250) -> dict[str, Any]:
    """Validate a frozen cache index without decompressing all PDW arrays."""
    root = Path(cache_root).resolve()
    manifest_path = root / "manifest.json"
    index_path = root / "emitter_index.json"
    manifest = _read_json(manifest_path)
    selection = _read_json(root / "selected_config_ids.json")
    index = _read_json(index_path)
    summaries = _read_json(root / "config_summaries.json")
    selected = selection.get("selected_config_ids")
    entries = index.get("entries")
    if manifest.get("cache_schema") != CACHE_SCHEMA or index.get("schema") != INDEX_SCHEMA:
        raise ValueError("Unsupported TRAIN-250 cache/index schema")
    if manifest.get("pool_name") != "train_250":
        raise ValueError("Expected TRAIN-250 cache")
    if (not isinstance(selected, list) or len(selected) != expected_configs
            or len(set(selected)) != expected_configs
            or manifest.get("selected_train_configs") != expected_configs
            or manifest.get("selected_config_ids") != selected):
        raise ValueError("TRAIN-250 selection disagrees with cache manifest")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Cache emitter index is empty")
    if manifest.get("selected_emitter_contributions") != len(entries):
        raise ValueError("Cache emitter count disagrees with manifest")
    if {row["config_id"] for row in summaries.get("configs", [])} != set(selected):
        raise ValueError("Cache summaries disagree with selected configs")
    if len({row["uid"] for row in entries}) != len(entries):
        raise ValueError("Cache emitter UIDs are not unique")
    if {row["config_id"] for row in entries} != set(selected):
        raise ValueError("Cache emitter index disagrees with selected configs")
    for row in entries:
        config_id = row["config_id"]
        if row["stare_file"] != f"stare/train_stare/{config_id}.h5":
            raise ValueError(f"Stale or cross-split source path for {row['uid']}")
        if row["npz_file"] != f"configs/{config_id}.npz":
            raise ValueError(f"Wrong NPZ path for {row['uid']}")
        if int(row["recorded_pulse_count"]) <= 0:
            raise ValueError(f"Empty indexed emitter: {row['uid']}")
    files = {p.stem for p in (root / "configs").glob("*.npz")}
    if files != set(selected):
        raise ValueError("Cache NPZ files disagree with selected configs")
    distribution = np.asarray(
        list(Counter(row["config_id"] for row in entries).values()), dtype=np.int64,
    )
    return {
        "cache_root": root,
        "manifest": manifest,
        "selected_config_ids": selected,
        "emitter_index": entries,
        "config_summaries": summaries["configs"],
        "emitter_count_distribution": distribution,
        "manifest_sha256": _sha256(manifest_path),
        "emitter_index_sha256": _sha256(index_path),
        "source_pool_fingerprint": manifest["source_pool_fingerprint"],
    }


class CachedTSRDTrain250Pool:
    """Sample complete emitter histories, then compose a fresh recorded-PDW world."""

    def __init__(
        self, data_root: str | Path, cache: dict[str, Any], *,
        band_centres_mhz=None, passband_halfwidth_mhz: float = 500.0,
        receiver_profile: str | None = None,
        detection_probability: float | None = None,
        false_alarm_probability: float | None = None,
        retune_time_ms: float | None = None,
        amplitude_midpoint_db: float = -90.0,
        amplitude_scale_db: float = 5.0,
        max_observed_pdws: int = 32,
        time_offset_us: int = DEFAULT_TIME_OFFSET_US,
    ):
        self.data_root = Path(data_root).resolve()
        self.cache = cache
        self.cache_root = Path(cache["cache_root"]).resolve()
        self.contributions = list(cache["emitter_index"])
        self.composer = WorldComposer(time_offset_us=time_offset_us)
        contract = cache["manifest"]["receiver_contract"]
        self.receiver_profile = receiver_profile or contract["receiver_profile"]
        self.detection_probability = float(
            contract["detection_probability"] if detection_probability is None
            else detection_probability
        )
        self.false_alarm_probability = float(
            contract["false_alarm_probability"] if false_alarm_probability is None
            else false_alarm_probability
        )
        self.retune_time_ms = float(
            contract["retune_time_ms"] if retune_time_ms is None else retune_time_ms
        )
        self.amplitude_midpoint_db = amplitude_midpoint_db
        self.amplitude_scale_db = amplitude_scale_db
        self.max_observed_pdws = max_observed_pdws
        centres = DEFAULT_CENTRES_MHZ if band_centres_mhz is None else band_centres_mhz
        self.centres_mhz = np.asarray(centres, dtype=np.float64)
        if (self.centres_mhz.ndim != 1 or len(self.centres_mhz) < 2
                or not np.all(np.isfinite(self.centres_mhz))
                or not np.all(np.diff(self.centres_mhz) > 0)):
            raise ValueError("Receiver centres must be finite and strictly increasing")
        for config_id in cache["selected_config_ids"]:
            source = _inside(self.data_root, f"stare/train_stare/{config_id}.h5")
            if not source.is_file():
                raise FileNotFoundError(source)
        anchor_id = cache["selected_config_ids"][0]
        anchor_file = _inside(self.data_root, f"stare/train_stare/{anchor_id}.h5")
        anchor = TSRDStareEnvironment.from_stare_mode(
            str(anchor_file), band_centres_mhz=self.centres_mhz,
            passband_halfwidth_mhz=passband_halfwidth_mhz,
            receiver_profile=self.receiver_profile,
            detection_probability=self.detection_probability,
            false_alarm_probability=self.false_alarm_probability,
            retune_time_ms=self.retune_time_ms,
            amplitude_midpoint_db=amplitude_midpoint_db,
            amplitude_scale_db=amplitude_scale_db,
            max_observed_pdws=max_observed_pdws,
        )
        self.config = anchor.config
        self.centres_mhz, self.halfwidth_mhz = anchor.receiver_geometry

    def sample_world(self, seed: int, emitter_count: int):
        chosen = self.composer.select(self.contributions, lambda row: row["config_id"],
                                      seed=seed, count=emitter_count)
        by_file = defaultdict(list)
        for world_id, row in enumerate(chosen):
            by_file[row["npz_file"]].append((world_id, row))
        realizations = {}
        for relative, requested in by_file.items():
            with np.load(_inside(self.cache_root, relative), allow_pickle=False) as z:
                for world_id, row in requested:
                    suffix = str(int(row["source_label"]))
                    arrays = []
                    for prefix in PDW_FIELDS:
                        key = prefix + suffix
                        if key not in z:
                            raise ValueError(f"Missing {key} in {relative}")
                        arrays.append(np.asarray(z[key], dtype=np.float64))
                    count = int(row["recorded_pulse_count"])
                    if any(values.ndim != 1 or len(values) != count for values in arrays):
                        raise ValueError(f"PDW length mismatch for {row['uid']}")
                    if not np.all(np.isfinite(arrays)):
                        raise ValueError(f"Nonfinite PDW field for {row['uid']}")
                    if np.any(np.diff(arrays[0]) < 0):
                        raise ValueError(f"Unsorted ToA for {row['uid']}")
                    if int(z["total_pulses_" + suffix]) != count:
                        raise ValueError(f"Pulse count mismatch for {row['uid']}")
                    realizations[world_id] = EmitterRealization(
                        f"train:{row['uid']}", row["config_id"], int(row["source_label"]),
                        row["stare_file"], np.column_stack(arrays), **source_emitter_metadata(
                            _inside(self.data_root, row["stare_file"]), int(row["source_label"])))
        data, labels, sources = self.composer.compose(
            [realizations[i] for i in range(len(chosen))], seed=seed)
        occupancy = stare_pulse_occupancy(
            data, self.centres_mhz, self.halfwidth_mhz, self.config.time_slots,
        )
        world = TSRDStareEnvironment(
            self.config.band_count, self.config.time_slots, occupancy, None,
            self.config, stare_data=data, stare_labels=labels,
            band_centres_mhz=self.centres_mhz,
            passband_halfwidth_mhz=self.halfwidth_mhz,
            receiver_profile=self.receiver_profile,
            amplitude_midpoint_db=self.amplitude_midpoint_db,
            amplitude_scale_db=self.amplitude_scale_db,
            max_observed_pdws=self.max_observed_pdws,
        )
        return world, sorted(sources, key=lambda row: row["world_emitter_id"])


def build_train_pool_from_cache(
    data_root: str | Path, cache_root: str | Path, *,
    expected_configs: int = 250, **receiver_options,
) -> tuple[CachedTSRDTrain250Pool, dict[str, Any]]:
    cache = load_train250_cache(cache_root, expected_configs=expected_configs)
    pool = CachedTSRDTrain250Pool(data_root, cache, **receiver_options)
    return pool, cache


def sample_fresh_training_world(pool: CachedTSRDTrain250Pool, rng: np.random.Generator):
    """Match the source per-config emitter-count distribution; mix emitters anew."""
    distribution = pool.cache["emitter_count_distribution"]
    emitter_count = int(rng.choice(distribution))
    world_seed = int(rng.integers(1, np.iinfo(np.int32).max))
    world, sources = pool.sample_world(world_seed, emitter_count)
    world.reset(seed=world_seed)
    return world, sources, world_seed, emitter_count


def build_stare_evaluation_world(
    data_root: str | Path, split: str, config_id: str, *,
    receiver_seed: int, band_centres_mhz=None,
    passband_halfwidth_mhz: float = 500.0,
    receiver_profile: str = "binary_v1",
    detection_probability: float = 0.9,
    false_alarm_probability: float = 0.05,
    retune_time_ms: float = 0.3,
    amplitude_midpoint_db: float = -90.0,
    amplitude_scale_db: float = 5.0,
    max_observed_pdws: int = 32,
) -> TSRDStareEnvironment:
    """Load a held-out STARE recording directly; never consult the train cache."""
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    if re.fullmatch(r"config_\d+", config_id) is None:
        raise ValueError("config_id must be config_<integer>")
    root = Path(data_root).resolve()
    source = _inside(root, f"stare/{split}_stare/{config_id}.h5")
    if not source.is_file():
        raise FileNotFoundError(source)
    centres = DEFAULT_CENTRES_MHZ if band_centres_mhz is None else band_centres_mhz
    world = TSRDStareEnvironment.from_stare_mode(
        str(source), band_centres_mhz=centres,
        passband_halfwidth_mhz=passband_halfwidth_mhz,
        receiver_profile=receiver_profile,
        detection_probability=detection_probability,
        false_alarm_probability=false_alarm_probability,
        retune_time_ms=retune_time_ms,
        amplitude_midpoint_db=amplitude_midpoint_db,
        amplitude_scale_db=amplitude_scale_db,
        max_observed_pdws=max_observed_pdws,
    )
    world.reset(seed=int(receiver_seed))
    return world


def compose_heldout_world(
    data_root: str | Path, split: str, config_ids: list[str], *,
    world_seed: int, emitter_count: int, receiver_seed: int,
    band_centres_mhz=None, passband_halfwidth_mhz: float = 500.0,
    receiver_profile: str = "binary_v1", detection_probability: float = 0.9,
    false_alarm_probability: float = 0.05, retune_time_ms: float = 0.3,
    time_offset_us: int = DEFAULT_TIME_OFFSET_US,
) -> tuple[TSRDStareEnvironment, list[dict]]:
    """Compose a deterministic VAL/TEST world solely from its named split.

    Complete per-emitter histories receive one additive ToA offset. No train cache,
    future receiver observations, or emitter labels reach the scheduler.
    """
    import h5py

    if split not in ("val", "test") or not config_ids or len(set(config_ids)) != len(config_ids):
        raise ValueError("Need unique VAL or TEST source config IDs")
    root = Path(data_root).resolve()
    centres = DEFAULT_CENTRES_MHZ if band_centres_mhz is None else np.asarray(band_centres_mhz)
    anchor = build_stare_evaluation_world(
        root, split, config_ids[0], receiver_seed=receiver_seed,
        band_centres_mhz=centres, passband_halfwidth_mhz=passband_halfwidth_mhz,
        receiver_profile=receiver_profile, detection_probability=detection_probability,
        false_alarm_probability=false_alarm_probability, retune_time_ms=retune_time_ms,
    )
    candidates = []
    for config_id in config_ids:
        if re.fullmatch(r"config_\d+", config_id) is None:
            raise ValueError("Invalid held-out config ID")
        path = _inside(root, f"stare/{split}_stare/{config_id}.h5")
        with h5py.File(path) as source:
            labels, counts = np.unique(source["labels"][:], return_counts=True)
            collection_s = float(source["metadata/receiver"].attrs["collection_time_s"])
            halfwidth = float(source["metadata/receiver"].attrs["bandwith_mhz"])
        if (not np.isclose(collection_s, anchor.n_slots * anchor.config.slot_duration_s())
                or not np.isclose(halfwidth, anchor.receiver_geometry[1])):
            raise ValueError(f"Incompatible receiver geometry in {config_id}")
        candidates.extend((config_id, int(label), int(count))
                          for label, count in zip(labels, counts) if count > 0)
    composer = WorldComposer(time_offset_us=time_offset_us)
    chosen = composer.select(candidates, lambda row: row[0], seed=world_seed, count=emitter_count)
    groups = defaultdict(list)
    for new_id, candidate in enumerate(chosen):
        groups[candidate[0]].append((new_id, candidate))
    realizations = {}
    for config_id, requested in groups.items():
        path = _inside(root, f"stare/{split}_stare/{config_id}.h5")
        with h5py.File(path) as source:
            data = np.asarray(source["data"][:])
            source_labels = np.asarray(source["labels"][:]).reshape(-1)
        for new_id, (_, label, count) in requested:
            selected = data[source_labels == label]
            if len(selected) != count:
                raise ValueError("Held-out emitter count changed during composition")
            realizations[new_id] = EmitterRealization(
                f"{split}:{config_id}:emitter:{label}", config_id, label,
                f"stare/{split}_stare/{config_id}.h5", selected,
                **source_emitter_metadata(path, label))
    data, world_labels, provenance = composer.compose(
        [realizations[i] for i in range(len(chosen))], seed=world_seed)
    occupancy = stare_pulse_occupancy(data, centres, anchor.receiver_geometry[1], anchor.n_slots)
    world = TSRDStareEnvironment(
        anchor.n_bands, anchor.n_slots, occupancy, None, anchor.config,
        stare_data=data, stare_labels=world_labels, band_centres_mhz=centres,
        passband_halfwidth_mhz=anchor.receiver_geometry[1], receiver_profile=receiver_profile,
    )
    world.reset(seed=receiver_seed)
    return world, sorted(provenance, key=lambda row: row["world_emitter_id"])


def write_runtime_manifest(
    cache: dict[str, Any], output_path: str | Path, *,
    receiver_profile: str, detection_probability: float,
    false_alarm_probability: float, retune_time_ms: float,
    training_seed: int | None = None,
    time_offset_us: int = DEFAULT_TIME_OFFSET_US,
    algorithm_details: dict | None = None,
    training_budget: dict | None = None,
    checkpointing: dict | None = None,
    analysis_details: dict | None = None,
) -> Path:
    """Write run provenance to an output directory, never into the input cache."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    payload = {
        "generator_version": GENERATOR_VERSION,
        "evidence_scope": "tsrd_emitter_realization_compositional_environment",
        "source_config_rule": "max_one_emitter_per_source_config_per_world",
        "world_mode": "emitter_recombined",
        "spatial_model": "tsrd_native",
        "sampling_recipe": "uniform distinct source configs, then uniform emitter within each source",
        "time_offset_us": int(time_offset_us),
        "time_offset_distribution": "independent discrete uniform [-time_offset_us,+time_offset_us]",
        "recorded_cache_role": "complete source traces; distinct from runtime recombined worlds",
        "pool_version": cache["manifest"]["cache_schema"],
        "TSRD_revision": cache["manifest"].get("TSRD_revision", "unknown"),
        "beam_state": "unobserved_in_source",
        "illumination_state": "unobserved_in_source",
        "physical_emission_denominator_available": False,
        "pool_name": cache["manifest"]["pool_name"],
        "cache_schema": cache["manifest"]["cache_schema"],
        "source_pool_fingerprint": cache["source_pool_fingerprint"],
        "manifest_sha256": cache["manifest_sha256"],
        "emitter_index_sha256": cache["emitter_index_sha256"],
        "selected_train_configs": len(cache["selected_config_ids"]),
        "emitter_contributions": len(cache["emitter_index"]),
        "receiver_profile": receiver_profile,
        "detector_model": ("recorded_stare_bernoulli" if receiver_profile == "binary_v1"
                           else "recorded_pdw_amplitude_logistic"),
        "false_alarm_model": "bernoulli_per_truth_empty_detector_evaluation_opportunity",
        "cfar_active": False,
        "detection_probability": float(detection_probability),
        "false_alarm_probability": float(false_alarm_probability),
        "retune_time_ms": float(retune_time_ms),
        "training_seed": training_seed,
        "algorithm": algorithm_details,
        "training_budget": training_budget,
        "evaluation_analysis": analysis_details,
    }
    if checkpointing is not None:
        payload["checkpointing"] = checkpointing
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target

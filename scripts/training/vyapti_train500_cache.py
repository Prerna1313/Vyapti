from __future__ import annotations

"""Loader/adapter for the user's already-created TSRD TRAIN-500 cache.

Important:
- DO NOT rerun HDF5 preprocessing from 2,500 files.
- The existing cache is authoritative for the 500-config selection.
- The current TSRDTrainWorldPool API exposes sample_world(), but no cache-backed
  constructor. To preserve its exact world-generation semantics, this adapter
  creates a 500-file symlink corpus and instantiates TSRDTrainWorldPool on ONLY
  those selected 500 configs.
- The saved emitter-contribution cache is used for selection/provenance checks;
  the simulator remains responsible for exact world construction.
"""

import hashlib
import json
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

N_BANDS = 36
BAND_CENTRES_MHZ = 250.0 + 500.0 * np.arange(N_BANDS, dtype=np.float64)
RECEIVER_PROFILE = "binary_v1"
AMPLITUDE_MIDPOINT_DB = -90.0
AMPLITUDE_SCALE_DB = 5.0
MAX_OBSERVED_PDWS = 32
RECEIVER_DETECTION_PROBABILITY = 0.90
RECEIVER_FALSE_ALARM_PROBABILITY = 0.05
RECEIVER_RETUNE_TIME_MS = 1.0

CACHE_SCHEMA = "vyapti_tsrd_emitter_pool_v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def config_sort_key(config_id: str) -> tuple[int, str]:
    try:
        return int(str(config_id).split("_")[-1]), str(config_id)
    except ValueError:
        return 10**9, str(config_id)


def fingerprint_cache(manifest: dict[str, Any], selected_config_ids: list[str], emitter_index: dict[str, Any]) -> str:
    payload = {
        "cache_schema": manifest.get("cache_schema"),
        "pool_name": manifest.get("pool_name"),
        "selection_method": manifest.get("selection_method"),
        "selection_seed": manifest.get("selection_seed"),
        "selected_config_ids": selected_config_ids,
        "emitter_uids": [row["uid"] for row in emitter_index["entries"]],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_train500_cache(cache_root: Path) -> dict[str, Any]:
    cache_root = Path(cache_root)
    manifest = load_json(cache_root / "manifest.json")
    if manifest.get("cache_schema") != CACHE_SCHEMA:
        raise RuntimeError(f"Unexpected cache_schema={manifest.get('cache_schema')!r}")
    if int(manifest.get("source_train_configs", -1)) != 2500:
        raise RuntimeError("Cache does not describe the full 2,500-config TRAIN corpus")
    if int(manifest.get("selected_train_configs", -1)) != 500:
        raise RuntimeError("Cache does not describe exactly 500 selected TRAIN configs")

    selected_payload = load_json(cache_root / "selected_config_ids.json")
    selected = sorted(
        [str(x) for x in selected_payload["selected_config_ids"]],
        key=config_sort_key,
    )
    if len(selected) != 500 or len(set(selected)) != 500:
        raise RuntimeError("selected_config_ids.json does not contain exactly 500 unique configs")

    emitter_index = load_json(cache_root / "emitter_index.json")
    entries = list(emitter_index.get("entries", []))
    if len(entries) == 0:
        raise RuntimeError("emitter_index.json contains no emitter contributions")
    uids = [str(row["uid"]) for row in entries]
    if len(uids) != len(set(uids)):
        raise RuntimeError("Duplicate emitter contribution UIDs in cached emitter index")
    entry_configs = {str(row["config_id"]) for row in entries}
    if entry_configs != set(selected):
        missing = sorted(set(selected) - entry_configs, key=config_sort_key)
        extra = sorted(entry_configs - set(selected), key=config_sort_key)
        raise RuntimeError(f"Emitter index/config selection mismatch; missing={missing[:5]}, extra={extra[:5]}")

    config_summaries = load_json(cache_root / "config_summaries.json")
    summaries = {str(x["config_id"]): x for x in config_summaries.get("configs", [])}
    if set(summaries) != set(selected):
        raise RuntimeError("config_summaries.json does not exactly cover the selected 500 configs")

    fingerprint = fingerprint_cache(manifest, selected, emitter_index)
    runtime_pool_root = cache_root / "selected_pool_corpus"
    return {
        "cache_root": cache_root,
        "manifest": manifest,
        "selected_config_ids": selected,
        "emitter_index": entries,
        "config_summaries": summaries,
        "fingerprint": fingerprint,
        "runtime_pool_root": runtime_pool_root,
        "emitter_count_distribution": np.asarray(
            list(Counter(str(row["config_id"]) for row in entries).values()),
            dtype=np.int64,
        ),
    }



@dataclass(frozen=True)
class CachedEmitterContribution:
    """One cached emitter contribution from the frozen TRAIN-500 pool."""

    stare_file: Path
    scan_file: Path | None
    npz_file: Path
    source_label: int
    recorded_pulse_count: int


class CachedTSRDTrainWorldPool:
    """
    Cache-backed replacement for TSRDTrainWorldPool.sample_world().

    The frozen TRAIN-500 preprocessing already stores each emitter's original
    PDW history in the per-config NPZ cache. This runtime pool reuses those
    histories instead of reading HDF5 ``data[:]`` for every fresh world.

    Public contract intentionally matches the existing code path:

        sample_world(seed, emitter_count) -> (env, sources)

    The world composition remains sampled without replacement from the same
    cached emitter-contribution pool. The selected emitter PDWs are rebuilt
    into the same five-column TSRD ``stare_data`` representation before the
    normal occupancy/environment construction.

    A single anchor HDF5 is opened once at initialization to recover the
    installed TSRD environment configuration/receiver geometry. It is not
    touched during subsequent world generation.
    """

    def __init__(
        self,
        corpus_root: str | Path,
        cache_root: str | Path,
        cache: dict[str, Any],
        *,
        band_centres_mhz=None,
        receiver_profile: str = "binary_v1",
        amplitude_midpoint_db: float = -90.0,
        amplitude_scale_db: float = 5.0,
        max_observed_pdws: int = 32,
        detection_probability: float = 0.90,
        false_alarm_probability: float = 0.05,
        retune_time_ms: float = 1.0,
        npz_lru_size: int = 4,
    ):
        self.corpus_root = Path(corpus_root).resolve()
        self.cache_root = Path(cache_root).resolve()
        self.explicit_centres_mhz = (
            None
            if band_centres_mhz is None
            else np.asarray(band_centres_mhz, dtype=np.float64)
        )
        self.receiver_profile_options = {
            "receiver_profile": receiver_profile,
            "amplitude_midpoint_db": amplitude_midpoint_db,
            "amplitude_scale_db": amplitude_scale_db,
            "max_observed_pdws": max_observed_pdws,
        }
        self.detection_probability = float(detection_probability)
        self.false_alarm_probability = float(false_alarm_probability)
        self.retune_time_ms = float(retune_time_ms)
        self._npz_lru_size = max(1, int(npz_lru_size))
        self._npz_lru: OrderedDict[str, Any] = OrderedDict()

        selected = list(cache["selected_config_ids"])
        if len(selected) != 500 or len(set(selected)) != 500:
            raise RuntimeError(
                f"Cached runtime pool requires exactly 500 selected configs; got {len(selected)}"
            )

        anchor_file = (
            self.corpus_root
            / "stare"
            / "train_stare"
            / f"{selected[0]}.h5"
        )
        if not anchor_file.exists():
            raise FileNotFoundError(f"Anchor HDF5 missing: {anchor_file}")

        from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment

        print(
            "[CachedTSRDTrainWorldPool] "
            f"one-time anchor read: {anchor_file.name}"
        )
        anchor = TSRDStareEnvironment.from_stare_mode(
            stare_file=str(anchor_file),
            scan_file=None,
            band_centres_mhz=self.explicit_centres_mhz,
            **self.receiver_profile_options,
            detection_probability=self.detection_probability,
            false_alarm_probability=self.false_alarm_probability,
            retune_time_ms=self.retune_time_ms,
        )

        self.config = anchor.config
        self.centres_mhz, self.halfwidth_mhz = anchor.receiver_geometry
        del anchor

        configs_root = self.cache_root / "configs"
        if not configs_root.is_dir():
            raise FileNotFoundError(f"Missing cache configs directory: {configs_root}")

        contributions: list[CachedEmitterContribution] = []
        for config_id in selected:
            npz_file = configs_root / f"{config_id}.npz"
            if not npz_file.exists():
                raise FileNotFoundError(f"Cached config NPZ missing: {npz_file}")

            with np.load(npz_file, allow_pickle=False) as z:
                labels = np.asarray(z["emitter_labels"], dtype=np.int64).reshape(-1)
                for label in np.unique(labels):
                    label = int(label)
                    toa_key = f"pdw_toa_us_{label}"
                    pulse_key = f"total_pulses_{label}"
                    if toa_key not in z.files:
                        raise RuntimeError(f"Missing {toa_key} in {npz_file}")
                    pulse_count = (
                        int(np.asarray(z[pulse_key]).item())
                        if pulse_key in z.files
                        else int(np.asarray(z[toa_key]).shape[0])
                    )
                    if pulse_count <= 0:
                        continue
                    contributions.append(
                        CachedEmitterContribution(
                            stare_file=self.corpus_root / "stare" / "train_stare" / f"{config_id}.h5",
                            scan_file=None,
                            npz_file=npz_file,
                            source_label=label,
                            recorded_pulse_count=pulse_count,
                        )
                    )

        if not contributions:
            raise RuntimeError("Cached TRAIN-500 pool contains no emitter contributions")

        self.contributions = contributions
        print(
            "[CachedTSRDTrainWorldPool] "
            f"configs={len(selected):,} emitters={len(self.contributions):,}"
        )

    def _get_npz(self, path: Path):
        key = str(path)
        z = self._npz_lru.pop(key, None)
        if z is not None:
            self._npz_lru[key] = z
            return z
        z = np.load(path, allow_pickle=False)
        self._npz_lru[key] = z
        while len(self._npz_lru) > self._npz_lru_size:
            _old_key, old_z = self._npz_lru.popitem(last=False)
            try:
                old_z.close()
            except Exception:
                pass
        return z

    def sample_world(self, seed: int, emitter_count: int):
        if not 1 <= int(emitter_count) <= len(self.contributions):
            raise ValueError(
                f"emitter_count must be in [1, {len(self.contributions)}], got {emitter_count}"
            )

        from vyapti_simulator.tsrd.tsrd_adapter import stare_pulse_occupancy
        from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment

        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(
            len(self.contributions),
            size=int(emitter_count),
            replace=False,
        )

        by_file: dict[Path, list[tuple[int, CachedEmitterContribution]]] = defaultdict(list)
        for world_id, index in enumerate(chosen):
            contribution = self.contributions[int(index)]
            by_file[contribution.npz_file].append((world_id, contribution))

        data_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        sources: list[dict[str, Any]] = []

        for npz_file, requested in by_file.items():
            z = self._get_npz(npz_file)
            for world_id, contribution in requested:
                label = int(contribution.source_label)
                suffix = str(label)
                keys = {
                    "toa": f"pdw_toa_us_{suffix}",
                    "frequency": f"pdw_frequency_mhz_{suffix}",
                    "pulse_width": f"pdw_pulse_width_{suffix}",
                    "aoa": f"pdw_aoa_deg_{suffix}",
                    "amplitude": f"pdw_amplitude_dbm_{suffix}",
                }
                missing = [k for k in keys.values() if k not in z.files]
                if missing:
                    raise RuntimeError(
                        f"Missing cached PDW arrays in {npz_file.name}, emitter {label}: {missing}"
                    )

                toa = np.asarray(z[keys["toa"]], dtype=np.float64)
                frequency = np.asarray(z[keys["frequency"]], dtype=np.float64)
                pulse_width = np.asarray(z[keys["pulse_width"]], dtype=np.float64)
                aoa = np.asarray(z[keys["aoa"]], dtype=np.float64)
                amplitude = np.asarray(z[keys["amplitude"]], dtype=np.float64)

                n = len(toa)
                if not (
                    len(frequency) == len(pulse_width) == len(aoa) == len(amplitude) == n
                ):
                    raise RuntimeError(
                        f"Cached PDW field length mismatch in {npz_file.name}, emitter {label}"
                    )
                if n == 0:
                    continue

                emitter_data = np.column_stack(
                    (toa, frequency, pulse_width, aoa, amplitude)
                )
                data_parts.append(emitter_data)
                label_parts.append(np.full(n, int(world_id), dtype=np.int64))
                sources.append(
                    {
                        "world_emitter_id": int(world_id),
                        "source_file": contribution.stare_file.relative_to(self.corpus_root).as_posix(),
                        "source_label": label,
                        "recorded_pulse_count": int(n),
                    }
                )

        if not data_parts:
            raise RuntimeError("Cached sample_world produced no PDWs")

        data = np.concatenate(data_parts, axis=0)
        labels = np.concatenate(label_parts, axis=0)

        occupancy = stare_pulse_occupancy(
            data,
            self.centres_mhz,
            self.halfwidth_mhz,
            self.config.time_slots,
        )

        env = TSRDStareEnvironment(
            self.config.band_count,
            self.config.time_slots,
            occupancy,
            None,
            self.config,
            stare_data=data,
            stare_labels=labels,
            band_centres_mhz=self.centres_mhz,
            passband_halfwidth_mhz=self.halfwidth_mhz,
            **self.receiver_profile_options,
            detection_probability=self.detection_probability,
            false_alarm_probability=self.false_alarm_probability,
            retune_time_ms=self.retune_time_ms,
        )

        return env, sorted(sources, key=lambda x: x["world_emitter_id"])


def build_train_pool_from_cache(corpus_root: Path, cache_root: Path):
    cache = load_train500_cache(Path(cache_root))
    pool = CachedTSRDTrainWorldPool(
        corpus_root=Path(corpus_root),
        cache_root=Path(cache_root),
        cache=cache,
        band_centres_mhz=BAND_CENTRES_MHZ,
        receiver_profile=RECEIVER_PROFILE,
        amplitude_midpoint_db=AMPLITUDE_MIDPOINT_DB,
        amplitude_scale_db=AMPLITUDE_SCALE_DB,
        max_observed_pdws=MAX_OBSERVED_PDWS,
        detection_probability=RECEIVER_DETECTION_PROBABILITY,
        false_alarm_probability=RECEIVER_FALSE_ALARM_PROBABILITY,
        retune_time_ms=RECEIVER_RETUNE_TIME_MS,
        npz_lru_size=4,
    )

    actual = {Path(str(c.stare_file)).stem for c in pool.contributions}
    expected = set(cache["selected_config_ids"])
    if actual != expected:
        missing = sorted(expected - actual, key=config_sort_key)
        extra = sorted(actual - expected, key=config_sort_key)
        raise RuntimeError(
            f"Cached runtime pool selected-config mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )

    cache_counts = Counter(str(x["config_id"]) for x in cache["emitter_index"])
    pool_counts = Counter(Path(str(c.stare_file)).stem for c in pool.contributions)
    if cache_counts != pool_counts:
        raise RuntimeError(
            "Cached emitter contribution counts do not match runtime cached pool"
        )

    cache["pool"] = pool
    return pool, cache


def emitter_count_distribution_from_cache(cache: dict[str, Any]) -> np.ndarray:
    counts = Counter(str(row["config_id"]) for row in cache["emitter_index"])
    values = np.asarray(list(counts.values()), dtype=np.int64)
    if values.size != 500:
        raise RuntimeError(f"Expected 500 per-config emitter counts, got {values.size}")
    return values


def sample_fresh_training_world(pool: Any, rng: np.random.Generator, distribution: np.ndarray | None = None):
    if distribution is None:
        counts = Counter(Path(str(c.stare_file)).stem for c in pool.contributions)
        distribution = np.asarray(list(counts.values()), dtype=np.int64)
    distribution = np.asarray(distribution, dtype=np.int64)
    if distribution.size != 500 or np.any(distribution <= 0):
        raise RuntimeError(f"Expected 500 positive per-config emitter counts, got shape={distribution.shape}")
    emitter_count = int(rng.choice(distribution))
    emitter_count = max(1, min(emitter_count, len(pool.contributions)))
    world_seed = int(rng.integers(1, np.iinfo(np.int32).max))
    env, sources = pool.sample_world(world_seed, emitter_count)
    if len(sources) != emitter_count:
        raise RuntimeError(f"sample_world returned {len(sources)} sources, expected {emitter_count}")
    return env, sources, world_seed, emitter_count


def write_runtime_manifest(cache: dict[str, Any]) -> None:
    save_json(
        Path(cache["cache_root"]) / "runtime_pool_manifest.json",
        {
            "schema": "vyapti_runtime_selected_pool_v1",
            "source_pool_fingerprint": cache["fingerprint"],
            "selected_configs": 500,
            "selected_config_ids": cache["selected_config_ids"],
            "selected_emitter_contributions": len(cache["emitter_index"]),
            "runtime_pool_root": str(cache["runtime_pool_root"]),
            "note": "500 selected HDF5 files are represented by symlinks; no source bytes are duplicated.",
        },
    )

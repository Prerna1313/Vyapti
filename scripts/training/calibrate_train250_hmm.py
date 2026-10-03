"""Fit a shared receiver-only band-occupancy HMM from TRAIN-250 PDWs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from vyapti_simulator.system_b.tsrd.train250_cache import load_train250_cache


N_BANDS = 36
N_SLOTS = 600
SLOT_US = 50_000.0
BAND_CENTRES_MHZ = 250.0 + 500.0 * np.arange(N_BANDS, dtype=np.float64)
PASSBAND_HALF_WIDTH_MHZ = 500.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _occupancy_grid(path: Path) -> np.ndarray:
    """Map TRAIN PDW time/frequency observations to 36 x 600 binary cells."""
    with h5py.File(path, "r") as source:
        if "metadata/feature_names" not in source or "metadata/receiver" not in source:
            raise ValueError(f"TRAIN source lacks required TSRD metadata: {path}")
        fields = [value.decode("utf-8") if isinstance(value, bytes) else str(value)
                  for value in source["metadata/feature_names"][:]]
        if "ToA" not in fields or "Frequency" not in fields:
            raise ValueError(f"TRAIN source lacks ToA/Frequency PDW fields: {path}")
        mode = source["metadata/receiver"].attrs.get("scan_mode")
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")
        if mode != "Stare":
            raise ValueError(f"Expected TRAIN STARE data, found {mode!r}: {path}")
        data = source["data"][:]

    toa = np.asarray(data[:, fields.index("ToA")], dtype=np.float64)
    frequency = np.asarray(data[:, fields.index("Frequency")], dtype=np.float64)
    valid = np.isfinite(toa) & np.isfinite(frequency) & (toa >= 0.0)
    slots = np.floor(np.where(valid, toa, 0.0) / SLOT_US).astype(np.int64)
    valid &= slots < N_SLOTS
    occupancy = np.zeros((N_BANDS, N_SLOTS), dtype=np.bool_)
    for band, centre in enumerate(BAND_CENTRES_MHZ):
        in_band = valid & (np.abs(frequency - centre) <= PASSBAND_HALF_WIDTH_MHZ)
        occupancy[band, slots[in_band]] = True
    return occupancy


def fit_transition_parameters(grids: list[np.ndarray]) -> dict:
    """Pool adjacent-slot counts and use Jeffreys Beta(1/2,1/2) smoothing."""
    if not grids:
        raise ValueError("At least one TRAIN occupancy grid is required")
    counts = {"00": 0, "01": 0, "10": 0, "11": 0}
    occupied = 0
    for grid in grids:
        grid = np.asarray(grid, dtype=np.bool_)
        if grid.shape != (N_BANDS, N_SLOTS):
            raise ValueError(f"Expected occupancy shape {(N_BANDS, N_SLOTS)}, got {grid.shape}")
        before, after = grid[:, :-1], grid[:, 1:]
        counts["00"] += int(np.count_nonzero(~before & ~after))
        counts["01"] += int(np.count_nonzero(~before & after))
        counts["10"] += int(np.count_nonzero(before & ~after))
        counts["11"] += int(np.count_nonzero(before & after))
        occupied += int(np.count_nonzero(grid))
    cells = len(grids) * N_BANDS * N_SLOTS
    return {
        "prior_active_probability": (occupied + 0.5) / (cells + 1.0),
        "inactive_to_active_probability": (counts["01"] + 0.5) / (counts["00"] + counts["01"] + 1.0),
        "active_to_active_probability": (counts["11"] + 0.5) / (counts["10"] + counts["11"] + 1.0),
        "counts": {**counts, "occupied_cells": occupied, "total_cells": cells},
    }


def calibrate(data_root: Path, cache_root: Path,
              progress: Callable[[str], None] | None = None) -> dict:
    cache = load_train250_cache(cache_root, expected_configs=250)
    ids = cache["selected_config_ids"]
    grids = []
    for index, config_id in enumerate(ids, start=1):
        path = data_root / "stare" / "train_stare" / f"{config_id}.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        grids.append(_occupancy_grid(path))
        if progress and (index % 25 == 0 or index == len(ids)):
            progress(f"processed {index}/{len(ids)} TRAIN configs")

    selected_path = cache_root / "selected_config_ids.json"
    manifest_path = cache_root / "manifest.json"
    index_path = cache_root / "emitter_index.json"
    fitted = fit_transition_parameters(grids)
    return {
        "schema": "vyapti_band_activity_hmm_calibration_v1",
        "calibration_id": "train250_observed_band_occupancy_50ms_v1",
        "source": {
            "dataset_split": "TRAIN only",
            "selected_config_count": len(ids),
            "selected_config_ids": ids,
            "source_pool_fingerprint": cache["source_pool_fingerprint"],
            "cache_manifest_sha256": _sha256(manifest_path),
            "emitter_index_sha256": _sha256(index_path),
            "selected_config_ids_sha256": _sha256(selected_path),
        },
        "observation_definition": (
            "A band-slot is active if any recorded TRAIN PDW has ToA in that "
            "50 ms slot and Frequency within 500 MHz of the 36 receiver band centres."
        ),
        "estimator": (
            "Pooled adjacent-slot empirical transitions across bands and TRAIN configs; "
            "Jeffreys Beta(1/2,1/2) smoothing on each conditional probability and prior."
        ),
        "labels_or_eval_data_used": False,
        "bands": N_BANDS,
        "slots_per_config": N_SLOTS,
        "slot_duration_ms": 50.0,
        **{key: value for key, value in fitted.items() if key != "counts"},
        "transition_counts": fitted["counts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--cache-root", type=Path, default=Path("Data/cache/tsrd/train_250"))
    parser.add_argument("--output", type=Path,
                        default=Path("training_setup/belief_models/train250_observed_band_hmm.json"))
    args = parser.parse_args()
    result = calibrate(args.data_root, args.cache_root,
                       progress=lambda message: print(f"[HMM-CAL] {message}", flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "calibration_id", "prior_active_probability",
        "inactive_to_active_probability", "active_to_active_probability", "transition_counts",
    )}, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

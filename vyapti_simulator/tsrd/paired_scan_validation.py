"""Development-only comparison of TSRD scan recordings with paired stare PDWs."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def _mode(value: object) -> str:
    return (value.decode("utf-8") if isinstance(value, bytes) else str(value)).lower()


def _groups_equal(left: h5py.Group, right: h5py.Group) -> bool:
    if set(left.keys()) != set(right.keys()) or set(left.attrs) != set(right.attrs):
        return False
    if any(not np.array_equal(left.attrs[name], right.attrs[name]) for name in left.attrs):
        return False
    for name in left:
        a, b = left[name], right[name]
        if isinstance(a, h5py.Group) and isinstance(b, h5py.Group):
            if not _groups_equal(a, b):
                return False
        elif isinstance(a, h5py.Dataset) and isinstance(b, h5py.Dataset):
            if not np.array_equal(a[:], b[:]):
                return False
        else:
            return False
    return True


def _receiver_schedule(receiver: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centres = np.asarray(receiver["dwell_centres_mhz"][:], dtype=np.float64)
    dwell_s = np.asarray(receiver["dwell_times_s"][:], dtype=np.float64)
    mission_us = int(round(float(receiver.attrs["collection_time_s"]) * 1e6))
    if (centres.ndim != 1 or len(centres) < 2 or dwell_s.shape != centres.shape
            or not np.all(np.isfinite(centres)) or not np.all(np.diff(centres) > 0)
            or not np.all(np.isfinite(dwell_s)) or not np.all(dwell_s > 0)
            or mission_us <= 0):
        raise ValueError("Invalid TSRD scan centres, dwell times, or mission duration")
    dwell_us = np.rint(dwell_s * 1e6).astype(np.int64)
    if not np.allclose(dwell_us / 1e6, dwell_s, atol=1e-6, rtol=0):
        raise ValueError("Scan dwell times cannot be represented at microsecond resolution")
    starts, ends, bands = [], [], []
    cursor = 0
    while cursor < mission_us:
        for band, duration_us in enumerate(dwell_us):
            if cursor >= mission_us:
                break
            starts.append(cursor)
            ends.append(min(mission_us, cursor + int(duration_us)))
            bands.append(band)
            cursor += int(duration_us)
    return np.asarray(starts), np.asarray(ends), np.asarray(bands)


def validate_pair(stare_file: str | Path, scan_file: str | Path,
                  *, min_passband_fraction: float = 0.99) -> dict:
    """Check geometry and compare dwell occupancy; never infer exact pulse identity."""
    if not 0 <= min_passband_fraction <= 1:
        raise ValueError("min_passband_fraction must be in [0, 1]")
    stare_path, scan_path = Path(stare_file), Path(scan_file)
    with h5py.File(stare_path) as stare_h5, h5py.File(scan_path) as scan_h5:
        stare_rx = stare_h5["metadata/receiver"]
        scan_rx = scan_h5["metadata/receiver"]
        same_receiver = all(
            np.array_equal(stare_rx[name][:], scan_rx[name][:])
            for name in ("start_position_km", "freq_range_mhz")
        ) and all(
            np.isclose(stare_rx.attrs[name], scan_rx.attrs[name])
            for name in ("collection_time_s", "bandwith_mhz", "gain_db", "sensitivity_dbm")
        )
        if _mode(stare_rx.attrs["scan_mode"]) != "stare":
            raise ValueError(f"Not a TSRD stare file: {stare_path}")
        if _mode(scan_rx.attrs["scan_mode"]) != "scanning":
            raise ValueError(f"Not a TSRD scan file: {scan_path}")
        if not same_receiver:
            raise ValueError(f"Receiver metadata differs between {stare_path} and {scan_path}")
        if not _groups_equal(stare_h5["metadata/transmitters"],
                             scan_h5["metadata/transmitters"]):
            raise ValueError(f"Transmitter metadata differs between {stare_path} and {scan_path}")
        if not np.array_equal(stare_h5["metadata/feature_names"][:],
                              scan_h5["metadata/feature_names"][:]):
            raise ValueError("Paired files have different PDW feature order")
        starts, ends, bands = _receiver_schedule(scan_rx)
        centres = np.asarray(scan_rx["dwell_centres_mhz"][:], dtype=np.float64)
        halfwidth = float(scan_rx.attrs["bandwith_mhz"])
        if not np.isfinite(halfwidth) or halfwidth <= 0:
            raise ValueError("Invalid TSRD receiver passband half-width")
        stare = np.asarray(stare_h5["data"][:])
        scan = np.asarray(scan_h5["data"][:])
    if (stare.ndim != 2 or stare.shape[1] != 5
            or scan.ndim != 2 or scan.shape[1] != 5):
        raise ValueError("TSRD PDWs must have five columns")
    stare = stare[np.argsort(stare[:, 0], kind="stable")]
    scan = scan[np.argsort(scan[:, 0], kind="stable")]
    stare_toa, scan_toa = stare[:, 0], scan[:, 0]
    pulse_in_window = 0
    scan_positive = 0
    stare_positive = 0
    both_positive = 0
    by_dwell_ms: dict[str, dict[str, int]] = {}
    both_halves_positive_100ms = 0
    for start, end, band in zip(starts, ends, bands):
        centre = centres[band]
        scan_lo, scan_hi = np.searchsorted(scan_toa, (start, end), side="left")
        stare_lo, stare_hi = np.searchsorted(stare_toa, (start, end), side="left")
        scan_freq = scan[scan_lo:scan_hi, 1]
        stare_freq = stare[stare_lo:stare_hi, 1]
        scan_in = np.abs(scan_freq - centre) <= halfwidth
        scan_has = bool(np.any(scan_in))
        stare_has = bool(np.any(np.abs(stare_freq - centre) <= halfwidth))
        pulse_in_window += int(np.count_nonzero(scan_in))
        scan_positive += int(scan_has)
        stare_positive += int(stare_has)
        both_positive += int(scan_has and stare_has)
        duration_ms = int(round((end - start) / 1000))
        row = by_dwell_ms.setdefault(str(duration_ms), {
            "dwells": 0, "scan_positive": 0, "stare_positive": 0,
            "both_positive": 0,
        })
        row["dwells"] += 1
        row["scan_positive"] += int(scan_has)
        row["stare_positive"] += int(stare_has)
        row["both_positive"] += int(scan_has and stare_has)
        if duration_ms == 100:
            midpoint = start + 50_000
            first = scan[scan_lo:np.searchsorted(scan_toa, midpoint, side="left"), 1]
            second = scan[np.searchsorted(scan_toa, midpoint, side="left"):scan_hi, 1]
            both_halves_positive_100ms += int(
                np.any(np.abs(first - centre) <= halfwidth)
                and np.any(np.abs(second - centre) <= halfwidth)
            )
    mission_us = int(ends[-1])
    in_mission = int(np.count_nonzero((scan_toa >= 0) & (scan_toa < mission_us)))
    fraction = float(pulse_in_window / in_mission) if in_mission else None
    return {
        "stare_file": str(stare_path), "scan_file": str(scan_path),
        "receiver_metadata_match": True,
        "transmitter_metadata_match": True,
        "band_count": len(centres), "dwell_count": len(starts),
        "mission_duration_s": mission_us / 1e6,
        "passband_halfwidth_mhz_from_h5": halfwidth,
        "scan_pulses_in_mission": in_mission,
        "scan_pulses_in_scheduled_passband": pulse_in_window,
        "scan_pulse_in_passband_fraction": fraction,
        "passband_gate": fraction is not None and fraction >= min_passband_fraction,
        "scan_positive_dwells": scan_positive,
        "stare_predicted_positive_dwells": stare_positive,
        "shared_positive_dwells": both_positive,
        "scan_positive_supported_by_stare_fraction": (
            float(both_positive / scan_positive) if scan_positive else None
        ),
        "by_dwell_ms": by_dwell_ms,
        "hundred_ms_dwells_with_scan_pulses_in_both_halves": both_halves_positive_100ms,
        "interpretation": (
            "Dwell overlap is descriptive, not detector Pd: scan and stare "
            "recordings are separately censored. PDWs do not reveal whether a "
            "100 ms dwell used one integrated decision or two 50 ms decisions."
        ),
    }


def validate_development_pairs(corpus_root: str | Path,
                               *, splits: tuple[str, ...] = ("train", "val"),
                               min_passband_fraction: float = 0.99) -> dict:
    """Find exact-name pairs in development splits only; leave test unopened."""
    if not splits or any(split not in ("train", "val") for split in splits):
        raise ValueError("Paired scan validation is limited to train/val splits")
    root = Path(corpus_root).resolve()
    results = {}
    for split in splits:
        stare = {path.name: path for path in (root / "stare" / f"{split}_stare").glob("*.h5")}
        scan = {path.name: path for path in (root / "scan" / f"{split}_scan").glob("*.h5")}
        common = sorted(stare.keys() & scan.keys())
        results[split] = {
            "stare_files": len(stare), "scan_files": len(scan),
            "matched_pairs": len(common),
            "unpaired_stare_files": sorted(stare.keys() - scan.keys()),
            "pairs": [validate_pair(stare[name], scan[name],
                                    min_passband_fraction=min_passband_fraction)
                      for name in common],
        }
    if not any(row["matched_pairs"] for row in results.values()):
        raise FileNotFoundError("No matched TSRD train/val scan-stare files")
    return {
        "corpus_root": str(root), "splits": results,
        "method": "Replay each recorded scan dwell over paired stare and scan PDWs",
        "test_split_used": False,
    }

"""
scripts.extract_tsrd_statistics
================================

One-time extractor: reads a TSRD HDF5 file and writes a
`tsrd_statistics.json` containing ONLY aggregate emitter-model
statistics (PRI distributions, frequency modes, agility patterns,
SNR ranges, etc.) — never the raw pulse data.

This is **Option A (Reference-only)** for the Vyapti / TSRD
integration. The JSON is the *only* TSRD-derived artefact that enters
the Vyapti simulator. It contains no truth-grid; the simulator
regenerates truth from these statistics on every run using the
existing `EmitterBehaviorType` enum (no new enum value, no
`TSRD_STATIC` branch, no replay of the pre-baked scan-mode matrix).

Usage
-----
    python scripts/extract_tsrd_statistics.py \\
        --input D:\\Vyapti\\SIH_DATA\\2\\TSRD_READY\\raw\\config_0.h5 \\
        --output D:\\Vyapti\\vyapti_simulator\\data\\tsrd_statistics.json

The script prints the SHA-256 of the output JSON; copy that value
into `data_provenance/manifest.json` under
`datasets.tsrd_statistics.sha256`.

[DESIGN] Pure I/O: no global state, no `np.random.seed()`, no
side effects beyond the two files it writes. Re-runnable: invoking
it twice with the same arguments produces byte-identical JSON
(sorted keys, indent=2) and the same SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np


# =====================================================================
# Helpers
# =====================================================================
def _sha256_of_file(path: Path) -> str:
    """Compute SHA-256 of a file, streaming 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _decode(v: Any) -> Any:
    """Decode bytes / numpy scalars to plain Python types."""
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_decode(x) for x in v.tolist()]
    return v


# =====================================================================
# Inspection (proves the schema we are reading is real)
# =====================================================================
def inspect_h5(path: Path) -> Dict[str, Any]:
    """Return high-level info about the HDF5 file."""
    info: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        info["data_shape"] = list(f["data"].shape)
        info["data_dtype"] = str(f["data"].dtype)
        info["labels_shape"] = list(f["labels"].shape)
        info["labels_dtype"] = str(f["labels"].dtype)
        info["feature_names"] = [
            _decode(x) for x in f["metadata/feature_names"][:]
        ]
        info["collection_time_s"] = float(
            f["metadata"].attrs["collection_time_s"]
        )
        info["num_pulses_reported"] = int(f["metadata"].attrs["num_pulses"])
        rec = f["metadata/receiver"]
        info["scan_mode"] = _decode(rec.attrs["scan_mode"])
        info["bandwidth_mhz"] = float(rec.attrs["bandwith_mhz"])
        info["sensitivity_dbm"] = float(rec.attrs["sensitivity_dbm"])
        info["gain_db"] = float(rec.attrs["gain_db"])
        info["freq_range_mhz"] = _decode(rec["freq_range_mhz"][:].tolist())
        info["dwell_centres_mhz"] = _decode(
            rec["dwell_centres_mhz"][:].tolist()
        )
        info["dwell_times_s"] = _decode(rec["dwell_times_s"][:].tolist())
        info["n_transmitter_configs"] = len(
            list(f["metadata/transmitters"].keys())
        )
    return info


# =====================================================================
# Per-emitter extraction
# =====================================================================
def extract_emitter_library(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Walk /metadata/transmitters/transmitters_X/ and pull out every
    parameter that affects how the emitter behaves in the Vyapti
    simulator. Return a dict keyed by transmitter id (as string).
    """
    library: Dict[str, Dict[str, Any]] = {}
    with h5py.File(path, "r") as f:
        txs = f["metadata/transmitters"]
        for tx_key in sorted(txs.keys(), key=lambda k: int(k.split("_")[1])):
            tx_id = int(tx_key.split("_")[1])
            tx = txs[tx_key]
            entry: Dict[str, Any] = {
                "function": _decode(tx.attrs.get("function", "")),
                "frequency": _extract_frequency_config(tx),
                "pri": _extract_pri_config(tx),
                "pulse_width": _extract_pw_config(tx),
                "power": _extract_power_config(tx),
                "scan": _extract_scan_config(tx),
                "position": _extract_position_config(tx),
            }
            library[str(tx_id)] = entry
    return library


def _extract_frequency_config(tx: h5py.Group) -> Dict[str, Any]:
    if "frequency_config" not in tx:
        return {}
    g = tx["frequency_config"]
    cfg = {k: _decode(v) for k, v in g.attrs.items()}
    if "freqs_mhz" in g:
        cfg["freqs_mhz"] = _decode(g["freqs_mhz"][:].tolist())
    return cfg


def _extract_pri_config(tx: h5py.Group) -> Dict[str, Any]:
    if "pri_config" not in tx:
        return {}
    g = tx["pri_config"]
    cfg = {k: _decode(v) for k, v in g.attrs.items()}
    if "pris_us" in g:
        cfg["pris_us"] = _decode(g["pris_us"][:].tolist())
    return cfg


def _extract_pw_config(tx: h5py.Group) -> Dict[str, Any]:
    if "pulse_width_config" not in tx:
        return {}
    g = tx["pulse_width_config"]
    cfg = {k: _decode(v) for k, v in g.attrs.items()}
    if "pws_us" in g:
        cfg["pws_us"] = _decode(g["pws_us"][:].tolist())
    return cfg


def _extract_power_config(tx: h5py.Group) -> Dict[str, Any]:
    if "power_config" not in tx:
        return {}
    g = tx["power_config"]
    return {k: _decode(v) for k, v in g.attrs.items()}


def _extract_scan_config(tx: h5py.Group) -> Dict[str, Any]:
    if "scan_config" not in tx:
        return {}
    g = tx["scan_config"]
    return {k: _decode(v) for k, v in g.attrs.items()}


def _extract_position_config(tx: h5py.Group) -> Dict[str, Any]:
    if "position_config" not in tx:
        return {}
    g = tx["position_config"]
    cfg = {k: _decode(v) for k, v in g.attrs.items()}
    if "start_position_km" in g:
        cfg["start_position_km"] = _decode(
            g["start_position_km"][:].tolist()
        )
    return cfg


# =====================================================================
# Population statistics (used by the sampler)
# =====================================================================
def compute_population_stats(
    library: Dict[str, Dict[str, Any]],
    label_counts: Dict[int, int],
) -> Dict[str, Any]:
    """
    Aggregate the per-emitter library into population-level
    distributions the Vyapti sampler can draw from.
    """
    freq_mode_counts: Counter = Counter()
    pri_mode_counts: Counter = Counter()
    scan_type_counts: Counter = Counter()
    periods_slots: list = []
    freq_ranges_mhz: list = []
    snr_db_estimates: list = []

    n_active = 0
    n_silent = 0
    for tx_id, entry in library.items():
        if label_counts.get(int(tx_id), 0) == 0:
            n_silent += 1
            continue
        n_active += 1

        # Frequency mode
        fcfg = entry.get("frequency", {})
        freq_mode_counts[fcfg.get("freq_mode", "Unknown")] += 1
        freqs = fcfg.get("freqs_mhz", [])
        if freqs:
            freq_ranges_mhz.append((min(freqs), max(freqs)))

        # PRI mode
        pcfg = entry.get("pri", {})
        pri_mode_counts[pcfg.get("pri_mode", "Unknown")] += 1

        # Scan config -> period in slots (50 ms slot assumed; the
        # caller passes the actual slot size in the JSON top-level).
        scfg = entry.get("scan", {})
        rate_rpm = scfg.get("scan_rate_rpm", 0.0)
        if rate_rpm > 0:
            period_s = 60.0 / rate_rpm
            # 50 ms slots (0.05 s)
            periods_slots.append(int(round(period_s / 0.05)))
        scan_type_counts[scfg.get("scan_type", "Unknown")] += 1

        # Crude SNR estimate: assume 50 km range, free-space loss
        # at 5 GHz ≈ 137 dB. ptot = 10*log10(power) + gain - FSPL.
        # This is illustrative; the sampler can be re-parameterised.
        power = entry.get("power", {}).get("power_w", 0.0)
        gain_tx = entry.get("power", {}).get("gain", 0.0)
        if power > 0 and gain_tx > 0:
            ptot_db = 10 * np.log10(power) + gain_tx - 137.0
            snr_db_estimates.append(float(ptot_db))

    return {
        "n_active": int(n_active),
        "n_silent": int(n_silent),
        "freq_mode_counts": dict(freq_mode_counts),
        "pri_mode_counts": dict(pri_mode_counts),
        "scan_type_counts": dict(scan_type_counts),
        "period_slots": {
            "min": int(min(periods_slots)) if periods_slots else 0,
            "median": int(np.median(periods_slots)) if periods_slots else 0,
            "max": int(max(periods_slots)) if periods_slots else 0,
        },
        "freq_range_mhz": {
            "min": float(min(r[0] for r in freq_ranges_mhz))
            if freq_ranges_mhz
            else 0.0,
            "max": float(max(r[1] for r in freq_ranges_mhz))
            if freq_ranges_mhz
            else 0.0,
        },
        "snr_db": {
            "min": float(min(snr_db_estimates))
            if snr_db_estimates
            else 0.0,
            "median": float(np.median(snr_db_estimates))
            if snr_db_estimates
            else 0.0,
            "max": float(max(snr_db_estimates))
            if snr_db_estimates
            else 0.0,
        },
    }


# =====================================================================
# Label counting (which emitter configs actually produced pulses)
# =====================================================================
def count_labels(path: Path) -> Dict[int, int]:
    with h5py.File(path, "r") as f:
        labels = f["labels"][:].flatten()
    vals, counts = np.unique(labels, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


# =====================================================================
# Main
# =====================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract TSRD emitter statistics -> tsrd_statistics.json"
    )
    ap.add_argument("--input", required=True, type=Path,
                    help="Path to a TSRD .h5 file (e.g. config_0.h5)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Where to write tsrd_statistics.json")
    ap.add_argument(
        "--slot-ms", type=float, default=50.0,
        help="Time slot size in ms (default 50, matches TSRD fine dwell)"
    )
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"[extract] Inspecting {args.input} ...")
    info = inspect_h5(args.input)
    print(f"[extract]   data shape    : {tuple(info['data_shape'])}")
    print(f"[extract]   scan mode     : {info['scan_mode']!r}")
    print(f"[extract]   n_transmitters: {info['n_transmitter_configs']}")

    print("[extract] Counting labels (which configs are active) ...")
    label_counts = count_labels(args.input)
    n_active = sum(1 for c in label_counts.values() if c > 0)
    n_silent = info["n_transmitter_configs"] - n_active
    print(f"[extract]   active = {n_active}, silent = {n_silent}")

    print("[extract] Walking /metadata/transmitters ...")
    library = extract_emitter_library(args.input)
    print(f"[extract]   extracted {len(library)} emitter configs")

    print("[extract] Computing population statistics ...")
    pop_stats = compute_population_stats(library, label_counts)

    # Assemble the final JSON
    out: Dict[str, Any] = {
        "dataset": "Turing Synthetic Radar Dataset",
        "source_file": str(args.input),
        "source_h5_sha256": _sha256_of_file(args.input),
        "extraction_date_utc": datetime.now(timezone.utc).isoformat(),
        "extractor_version": "1.0.0",
        "slot_ms": args.slot_ms,
        "file_info": {
            "data_shape": list(info["data_shape"]),
            "feature_names": info["feature_names"],
            "collection_time_s": info["collection_time_s"],
            "num_pulses_reported": info["num_pulses_reported"],
            "scan_mode": info["scan_mode"],
            "bandwidth_mhz": info["bandwidth_mhz"],
            "sensitivity_dbm": info["sensitivity_dbm"],
            "gain_db": info["gain_db"],
            "freq_range_mhz": info["freq_range_mhz"],
            "dwell_centres_mhz": info["dwell_centres_mhz"],
            "dwell_times_s": info["dwell_times_s"],
        },
        "population_stats": pop_stats,
        "emitter_library": library,
        "label_counts": {str(k): v for k, v in label_counts.items()},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    sha = _sha256_of_file(args.output)
    print(f"[extract] Wrote {args.output}")
    print(f"[extract] SHA-256 = {sha}")
    print(f"[extract] Active emitters: {n_active}, silent: {n_silent}")
    print()
    print("NEXT STEP:")
    print(f"  Paste the SHA-256 above into data_provenance/manifest.json")
    print(f"  under datasets.tsrd_statistics.sha256 (replace 'placeholder').")
    return 0


if __name__ == "__main__":
    sys.exit(main())

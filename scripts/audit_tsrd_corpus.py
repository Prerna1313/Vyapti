"""Audit TSRD HDF5 integrity across all splits without scoring schedulers.

Read-only data quality checks include schema, numeric ranges, timestamps,
labels, receiver metadata, scan schedule passband, and scan/stare pairing.
Test files are inspected only for data quality; no policy is evaluated here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from vyapti_simulator.system_b.tsrd.paired_scan_validation import (
    _groups_equal,
    _receiver_schedule,
)


EXPECTED_FEATURES = ("ToA", "Frequency", "PulseWidth", "AoA", "Amplitude")
EXPECTED_COUNTS = {"train": 2500, "val": 250, "test": 250}
FIELD_NAMES = ("toa_us", "freq_mhz", "pw_us", "aoa_deg", "amp_db")


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _problem(category: str, detail: str) -> dict[str, str]:
    return {"category": category, "detail": detail}


def audit_h5(path: Path, expected_mode: str, chunk_size: int) -> dict:
    """Read one file in chunks and return every observed anomaly."""
    row: dict = {"path": str(path), "anomalies": []}
    anomalies = row["anomalies"]
    try:
        with h5py.File(path, "r") as f:
            for key in ("data", "labels", "metadata/feature_names",
                        "metadata/receiver", "metadata/transmitters"):
                if key not in f:
                    anomalies.append(_problem("schema", f"Missing {key}"))
            if anomalies:
                return row
            data, labels = f["data"], f["labels"]
            if data.ndim != 2 or data.shape[1] != 5 or labels.ndim not in (1, 2):
                anomalies.append(_problem("schema", "Invalid data or label dimensions"))
                return row
            if labels.shape[0] != data.shape[0] or (labels.ndim == 2 and labels.shape[1] != 1):
                anomalies.append(_problem("schema", "Data and label lengths differ"))
                return row
            row.update(n_pdws=int(data.shape[0]), data_dtype=str(data.dtype),
                       label_dtype=str(labels.dtype))
            if data.dtype != np.dtype("float32") or not np.issubdtype(labels.dtype, np.integer):
                anomalies.append(_problem("schema", "Unexpected PDW or label dtype"))
            features = tuple(_decode(x) for x in f["metadata/feature_names"][:])
            row["feature_names"] = list(features)
            if features != EXPECTED_FEATURES:
                anomalies.append(_problem("schema", f"Unexpected feature order: {features}"))
            rx = f["metadata/receiver"]
            mode = _decode(rx.attrs.get("scan_mode", ""))
            row["receiver_mode"] = mode
            if mode.lower() != ("scanning" if expected_mode == "scan" else "stare"):
                anomalies.append(_problem("receiver", f"Unexpected receiver mode: {mode}"))
            collection_s = float(rx.attrs.get("collection_time_s", np.nan))
            halfwidth_mhz = float(rx.attrs.get("bandwith_mhz", np.nan))
            row["collection_time_s"] = collection_s
            row["passband_halfwidth_mhz_attr"] = halfwidth_mhz
            if not np.isfinite(collection_s) or collection_s <= 0:
                anomalies.append(_problem("receiver", "Invalid collection time"))
            if not np.isfinite(halfwidth_mhz) or halfwidth_mhz <= 0:
                anomalies.append(_problem("receiver", "Invalid passband half-width"))
            centres = np.asarray(rx["dwell_centres_mhz"][:], dtype=float) if "dwell_centres_mhz" in rx else np.array([])
            row["dwell_centres_count"] = len(centres)
            if expected_mode == "scan" and (len(centres) != 36 or not np.all(np.isfinite(centres))
                                                 or not np.all(np.diff(centres) > 0)):
                anomalies.append(_problem("receiver", "Invalid scan tune centres"))
            if expected_mode == "stare" and len(centres):
                anomalies.append(_problem("receiver", "Stare file unexpectedly has tune centres"))
            schedule = None
            if expected_mode == "scan":
                try:
                    starts, ends, bands = _receiver_schedule(rx)
                    schedule = (starts, ends, bands)
                    row["scheduled_dwells"] = len(starts)
                    if not np.isclose(ends[-1] / 1e6, collection_s, atol=1e-6):
                        anomalies.append(_problem("receiver", "Schedule duration differs from collection time"))
                except Exception as exc:
                    anomalies.append(_problem("receiver", f"Invalid scan schedule: {type(exc).__name__}: {exc}"))
            tx_ids = set()
            for name in f["metadata/transmitters"]:
                try:
                    tx_ids.add(int(name.rsplit("_", 1)[1]))
                except (IndexError, ValueError):
                    anomalies.append(_problem("schema", f"Unexpected transmitter key: {name}"))
            row["transmitter_configs"] = len(tx_ids)

            minima = np.full(5, np.inf)
            maxima = np.full(5, -np.inf)
            nonfinite = np.zeros(5, dtype=np.int64)
            range_counts = Counter()
            observed_labels: set[int] = set()
            previous_toa = -np.inf
            passband_in = 0
            passband_eligible = 0
            for start in range(0, len(data), chunk_size):
                end = min(start + chunk_size, len(data))
                values = np.asarray(data[start:end], dtype=np.float64)
                label_chunk = np.asarray(labels[start:end]).reshape(-1)
                observed_labels.update(int(x) for x in np.unique(label_chunk))
                finite = np.isfinite(values)
                nonfinite += np.count_nonzero(~finite, axis=0)
                for col in range(5):
                    valid = values[finite[:, col], col]
                    if len(valid):
                        minima[col] = min(minima[col], float(valid.min()))
                        maxima[col] = max(maxima[col], float(valid.max()))
                toa, freq, pw, aoa = values[:, 0], values[:, 1], values[:, 2], values[:, 3]
                if len(toa):
                    range_counts["decreasing_toa"] += int(toa[0] < previous_toa)
                    range_counts["decreasing_toa"] += int(np.count_nonzero(np.diff(toa) < 0))
                    previous_toa = float(toa[-1])
                range_counts["toa_before_zero"] += int(np.count_nonzero(toa < 0))
                range_counts["toa_at_or_after_nominal_duration"] += int(np.count_nonzero(toa >= collection_s * 1e6))
                range_counts["nonpositive_frequency"] += int(np.count_nonzero(freq <= 0))
                range_counts["nonpositive_pulse_width"] += int(np.count_nonzero(pw <= 0))
                range_counts["aoa_outside_180_deg"] += int(np.count_nonzero(np.abs(aoa) > 180))
                if schedule is not None and np.isfinite(halfwidth_mhz):
                    starts, ends, bands = schedule
                    index = np.searchsorted(starts, toa, side="right") - 1
                    in_timeline = (index >= 0) & (index < len(starts))
                    safe = np.clip(index, 0, len(starts) - 1)
                    in_timeline &= toa < ends[safe]
                    passband_eligible += int(np.count_nonzero(in_timeline))
                    passband_in += int(np.count_nonzero(
                        in_timeline & (np.abs(freq - centres[bands[safe]]) <= halfwidth_mhz)
                    ))
            row["observed_label_count"] = len(observed_labels)
            missing_labels = sorted(observed_labels - tx_ids)
            row["labels_without_transmitter_config"] = missing_labels
            if missing_labels:
                anomalies.append(_problem("labels", f"Labels lack transmitter config: {missing_labels}"))
            row["feature_min"] = {key: (float(value) if np.isfinite(value) else None)
                                  for key, value in zip(FIELD_NAMES, minima)}
            row["feature_max"] = {key: (float(value) if np.isfinite(value) else None)
                                  for key, value in zip(FIELD_NAMES, maxima)}
            row["nonfinite_by_feature"] = dict(zip(FIELD_NAMES, map(int, nonfinite)))
            row["range_counts"] = dict(range_counts)
            if np.any(nonfinite):
                anomalies.append(_problem("values", "Nonfinite PDW values"))
            for category, count in range_counts.items():
                # TSRD v2 Table II explicitly includes ToAs > 30 s while its
                # receiver collection duration is 30 s. This relation is
                # reported, but does not establish a corrupt PDW or an
                # absolute ToA clock origin.
                if count and category != "toa_at_or_after_nominal_duration":
                    anomalies.append(_problem("values", f"{category}: {count}"))
            if schedule is not None:
                fraction = passband_in / passband_eligible if passband_eligible else None
                row["scan_pulse_in_scheduled_passband_fraction"] = fraction
                if fraction is not None and fraction < 0.99:
                    anomalies.append(_problem("receiver", f"Passband fraction below 0.99: {fraction}"))
    except Exception as exc:
        anomalies.append(_problem("read_error", f"{type(exc).__name__}: {exc}"))
    return row


def audit_pair(stare_path: Path, scan_path: Path) -> list[dict[str, str]]:
    """Compare paired receiver and transmitter metadata, without pulse scoring."""
    issues = []
    try:
        with h5py.File(stare_path, "r") as stare, h5py.File(scan_path, "r") as scan:
            a, b = stare["metadata/receiver"], scan["metadata/receiver"]
            for key in ("collection_time_s", "bandwith_mhz", "gain_db", "sensitivity_dbm"):
                if key not in a.attrs or key not in b.attrs or not np.array_equal(a.attrs[key], b.attrs[key]):
                    issues.append(_problem("pair_receiver", f"Receiver attribute differs: {key}"))
            for key in ("start_position_km", "freq_range_mhz"):
                if key not in a or key not in b or not np.array_equal(a[key][:], b[key][:]):
                    issues.append(_problem("pair_receiver", f"Receiver dataset differs: {key}"))
            if not np.array_equal(stare["metadata/feature_names"][:], scan["metadata/feature_names"][:]):
                issues.append(_problem("pair_schema", "Feature order differs"))
            if not _groups_equal(stare["metadata/transmitters"], scan["metadata/transmitters"]):
                issues.append(_problem("pair_transmitters", "Transmitter metadata differs"))
    except Exception as exc:
        issues.append(_problem("pair_read_error", f"{type(exc).__name__}: {exc}"))
    return issues


def audit_corpus(root: Path, chunk_size: int = 250_000) -> dict:
    """Audit every expected file; preserve all anomalies by split and file."""
    root = root.resolve()
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    rows = []
    pair_rows = []
    summaries = {}
    for split, expected in EXPECTED_COUNTS.items():
        split_rows = []
        split_pairs = []
        for mode in ("scan", "stare"):
            directory = root / mode / f"{split}_{mode}"
            found = {p.name: p for p in directory.glob("*.h5")}
            if len(found) != expected:
                split_rows.append({
                    "path": str(directory), "n_pdws": 0,
                    "anomalies": [_problem("inventory", f"Expected {expected} HDF5 files; found {len(found)}")],
                })
            for i in range(expected):
                name = f"config_{i}.h5"
                path = found.get(name)
                if path is None:
                    split_rows.append({
                        "path": str(directory / name), "n_pdws": 0,
                        "anomalies": [_problem("inventory", "Expected HDF5 file missing")],
                    })
                    continue
                row = audit_h5(path, mode, chunk_size)
                row.update(split=split, mode=mode)
                split_rows.append(row)
                if (i + 1) % 250 == 0:
                    print(f"audited {split}/{mode}: {i + 1}/{expected}", flush=True)
            unexpected = sorted(set(found) - {f"config_{i}.h5" for i in range(expected)})
            for name in unexpected:
                split_rows.append({
                    "path": str(found[name]), "n_pdws": 0,
                    "anomalies": [_problem("inventory", "Unexpected HDF5 file name")],
                })
            print(f"audited {split}/{mode}: {len(found)} files", flush=True)
        for i in range(expected):
            name = f"config_{i}.h5"
            scan = root / "scan" / f"{split}_scan" / name
            stare = root / "stare" / f"{split}_stare" / name
            issues = audit_pair(stare, scan) if scan.is_file() and stare.is_file() else [
                _problem("pair_inventory", "Missing scan or stare companion")
            ]
            split_pairs.append({"config": name, "anomalies": issues})
        category_counts = Counter(
            issue["category"] for row in split_rows for issue in row["anomalies"]
        )
        pair_category_counts = Counter(
            issue["category"] for row in split_pairs for issue in row["anomalies"]
        )
        summaries[split] = {
            "expected_pairs": expected,
            "scan_files_audited": sum(r.get("mode") == "scan" for r in split_rows),
            "stare_files_audited": sum(r.get("mode") == "stare" for r in split_rows),
            "scan_pdws": sum(r.get("n_pdws", 0) for r in split_rows if r.get("mode") == "scan"),
            "stare_pdws": sum(r.get("n_pdws", 0) for r in split_rows if r.get("mode") == "stare"),
            "empty_recordings": sum(r.get("n_pdws") == 0 for r in split_rows if r.get("mode")),
            "files_with_anomalies": sum(bool(r["anomalies"]) for r in split_rows),
            "pairs_with_anomalies": sum(bool(r["anomalies"]) for r in split_pairs),
            "file_anomaly_categories": dict(category_counts),
            "pair_anomaly_categories": dict(pair_category_counts),
        }
        rows.extend(split_rows)
        pair_rows.extend({"split": split, **r} for r in split_pairs)
    return {
        "version": "1.1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Full corpus data-quality audit only; test used for no scheduler decisions",
        "declared_feature_units": {
            "toa_us": "microseconds", "freq_mhz": "MHz", "pw_us": "microseconds",
            "aoa_deg": "degrees", "amp_db": "dB (not assumed dBm)",
        },
        "unit_limit": "Numeric plausibility does not independently prove upstream units",
        "chunk_size": chunk_size,
        "summary": summaries,
        "files": rows,
        "pairs": pair_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()
    root, output = args.corpus_root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error("Output must be outside the source corpus")
    report = audit_corpus(root, args.chunk_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    for split, summary in report["summary"].items():
        print(split, summary, flush=True)
    print(output)


if __name__ == "__main__":
    main()

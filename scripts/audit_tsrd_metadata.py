"""Audit TSRD H5 schema and candidate PDW-label/transmitter metadata mismatches.

Frequency checks are diagnostic only. A flagged label is never remapped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


EXPECTED_FEATURES = ("ToA", "Frequency", "PulseWidth", "AoA", "Amplitude")


def audit_file(path: Path, *, chunk_size: int = 250_000,
               margin_mhz: float = 50.0, suspect_fraction: float = 0.95,
               min_pulses: int = 20) -> dict:
    if chunk_size <= 0 or margin_mhz < 0 or not 0 < suspect_fraction <= 1:
        raise ValueError("Invalid audit thresholds")
    issues: list[str] = []
    with h5py.File(path, "r") as f:
        features = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in f["metadata/feature_names"][:]
        )
        if features != EXPECTED_FEATURES:
            issues.append(f"Unexpected feature order: {features}")
        data, labels = f["data"], f["labels"]
        if data.ndim != 2 or data.shape[1] != 5 or len(data) != len(labels):
            raise ValueError(f"Invalid PDW or label shape in {path}")
        receiver = f["metadata/receiver"]
        transmitters = f["metadata/transmitters"]
        tx_ids = [int(name.rsplit("_", 1)[1]) for name in transmitters]
        width = max(tx_ids, default=-1) + 1
        low = np.full(width, np.nan)
        high = np.full(width, np.nan)
        for tx_id in tx_ids:
            values = np.asarray(
                transmitters[f"transmitters_{tx_id}/frequency_config/freqs_mhz"][:],
                dtype=float,
            )
            if len(values) and np.all(np.isfinite(values)):
                low[tx_id], high[tx_id] = float(values.min()), float(values.max())
        counts = np.zeros(width, dtype=np.int64)
        outside = np.zeros(width, dtype=np.int64)
        missing_labels: set[int] = set()
        for start in range(0, len(data), chunk_size):
            end = min(start + chunk_size, len(data))
            frequency = np.asarray(data[start:end, 1], dtype=float)
            label = np.asarray(labels[start:end]).reshape(-1).astype(np.int64)
            valid = (label >= 0) & (label < width)
            missing_labels.update(int(x) for x in np.unique(label[~valid]))
            label = label[valid]
            frequency = frequency[valid]
            if not len(label):
                continue
            missing = ~np.isfinite(low[label]) | ~np.isfinite(high[label])
            missing_labels.update(int(x) for x in np.unique(label[missing]))
            label = label[~missing]
            frequency = frequency[~missing]
            counts += np.bincount(label, minlength=width)
            out = (~np.isfinite(frequency)
                   | (frequency < low[label] - margin_mhz)
                   | (frequency > high[label] + margin_mhz))
            outside += np.bincount(label[out], minlength=width)
        suspects = [
            {"label": int(i), "pulses": int(counts[i]),
             "outside_fraction": float(outside[i] / counts[i]),
             "nominal_frequency_range_mhz": [float(low[i]), float(high[i])]}
            for i in np.flatnonzero(counts >= min_pulses)
            if outside[i] / counts[i] >= suspect_fraction
        ]
        return {
            "file": str(path), "pdws": len(data),
            "transmitter_configs": len(transmitters),
            "observed_labels_checked": int(np.count_nonzero(counts)),
            "receiver_mode": str(receiver.attrs.get("scan_mode", "")),
            "collection_time_s": float(receiver.attrs["collection_time_s"]),
            "bandwith_mhz_attr": float(receiver.attrs["bandwith_mhz"]),
            "dwell_centres_count": len(receiver["dwell_centres_mhz"]),
            "missing_nominal_frequency_labels": sorted(missing_labels),
            "suspect_label_joins": suspects,
            "schema_issues": issues,
        }


def audit_corpus(corpus_root: str | Path, *, max_files: int | None = None,
                 chunk_size: int = 250_000) -> dict:
    root = Path(corpus_root).resolve()
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be positive")
    splits = {}
    for split in ("train", "val", "test"):
        files = sorted((root / "stare" / f"{split}_stare").glob("*.h5"))
        if not files:
            raise FileNotFoundError(f"No TSRD stare files for {split} under {root}")
        rows = [audit_file(path, chunk_size=chunk_size)
                for path in files[:max_files]]
        splits[split] = {
            "files_audited": len(rows),
            "pdws_audited": sum(row["pdws"] for row in rows),
            "files_with_suspect_joins": sum(bool(row["suspect_label_joins"]) for row in rows),
            "files_with_schema_issues": sum(bool(row["schema_issues"]) for row in rows),
            "files": rows,
        }
    return {
        "corpus_root": str(root),
        "note": "Frequency-envelope flags are candidates for review, not corrected labels.",
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()
    report = audit_corpus(
        args.corpus_root, max_files=args.max_files, chunk_size=args.chunk_size
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for split, result in report["splits"].items():
        print(split, result["files_audited"], "files,",
              result["files_with_suspect_joins"], "with suspect joins")
    print(output)


if __name__ == "__main__":
    main()

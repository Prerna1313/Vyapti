"""Reinterpret a completed TSRD audit using the v2 paper's ToA range.

The original per-file counts are preserved. This only changes the status of
ToA at or after the nominal 30 s receiver duration from "bad data" to a
documented temporal relation. No HDF5 file is edited or re-read.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def reclassify(report: dict) -> dict:
    if report.get("version") != "1.0.0":
        raise ValueError("Expected original full-corpus audit version 1.0.0")
    for row in report["files"]:
        counts = row.get("range_counts", {})
        minimum_toa = row.get("feature_min", {}).get("toa_us")
        if minimum_toa is not None and minimum_toa < 0:
            raise ValueError(f"Cannot reclassify negative ToA in {row.get('path')}")
        after = counts.pop("toa_outside_collection", 0)
        counts["toa_at_or_after_nominal_duration"] = after
        counts["toa_before_zero"] = 0
        row["anomalies"] = [
            issue for issue in row["anomalies"]
            if not issue["detail"].startswith("toa_outside_collection:")
        ]
    for split in ("train", "val", "test"):
        rows = [row for row in report["files"] if row.get("split") == split]
        summary = report["summary"][split]
        summary["files_with_anomalies"] = sum(bool(row["anomalies"]) for row in rows)
        summary["file_anomaly_categories"] = dict(Counter(
            issue["category"] for row in rows for issue in row["anomalies"]
        ))
        summary["files_with_toa_at_or_after_nominal_duration"] = sum(
            row["range_counts"]["toa_at_or_after_nominal_duration"] > 0 for row in rows
        )
        summary["pdws_at_or_after_nominal_duration"] = sum(
            row["range_counts"]["toa_at_or_after_nominal_duration"] for row in rows
        )
    report["version"] = "1.1.0"
    report["timing_interpretation"] = {
        "source": "https://arxiv.org/html/2602.03856v2",
        "evidence": "Table II reports stare ToA maxima 43.49M/43.17M/43.33M microseconds while Section II-A says collection time was 30 s",
        "conclusion": "ToA beyond nominal collection duration is a published dataset characteristic, not by itself a file integrity failure",
        "unresolved": "The paper does not specify an absolute ToA clock origin or exact receiver interval boundaries",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Preserve the original report; choose a new output path")
    report = reclassify(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for split, summary in report["summary"].items():
        print(split, "integrity anomalies", summary["files_with_anomalies"],
              "after nominal duration", summary["pdws_at_or_after_nominal_duration"])
    print(args.output)


if __name__ == "__main__":
    main()

"""Audit label-to-transmitter frequency compatibility in every scan file.

This is a data-quality diagnostic only. A suspect join does not prove a bad
file-local label or license automatic relabelling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_tsrd_metadata import audit_file


EXPECTED_COUNTS = {"train": 2500, "val": 250, "test": 250}


def audit_scan_labels(root: Path) -> dict:
    root = root.resolve()
    splits = {}
    for split, expected in EXPECTED_COUNTS.items():
        directory = root / "scan" / f"{split}_scan"
        files = sorted(directory.glob("*.h5"))
        flagged = []
        errors = []
        total_pdws = 0
        for path in files:
            try:
                row = audit_file(path)
                total_pdws += row["pdws"]
                if (row["suspect_label_joins"] or row["missing_nominal_frequency_labels"]
                        or row["schema_issues"]):
                    flagged.append(row)
            except Exception as exc:
                errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
        splits[split] = {
            "expected_files": expected,
            "files_found": len(files),
            "pdws_audited": total_pdws,
            "files_with_suspect_joins": sum(bool(row["suspect_label_joins"]) for row in flagged),
            "files_with_missing_nominal_frequency_labels": sum(
                bool(row["missing_nominal_frequency_labels"]) for row in flagged
            ),
            "files_with_schema_issues": sum(bool(row["schema_issues"]) for row in flagged),
            "flagged_files": flagged,
            "read_errors": errors,
        }
        print(f"{split}: {len(files)} scan files, {len(flagged)} flagged, {len(errors)} read errors", flush=True)
    return {
        "version": "1.0.0",
        "scope": "Full scan-mode frequency-envelope diagnostic; test is data QA only",
        "thresholds": {"margin_mhz": 50.0, "suspect_fraction": 0.95, "min_pulses": 20},
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.corpus_root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error("Output must be outside the source corpus")
    report = audit_scan_labels(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

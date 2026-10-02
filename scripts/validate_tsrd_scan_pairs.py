"""Validate matching TSRD train/validation scan-stare pairs without test leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vyapti_simulator.system_b.tsrd.paired_scan_validation import validate_development_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-passband-fraction", type=float, default=0.99)
    args = parser.parse_args()
    report = validate_development_pairs(
        args.corpus_root, min_passband_fraction=args.min_passband_fraction
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for split, row in report["splits"].items():
        print(f"{split}: {row['matched_pairs']} paired files")
        for pair in row["pairs"]:
            print(Path(pair["scan_file"]).name,
                  f"passband={pair['scan_pulse_in_passband_fraction']:.4f}",
                  f"shared_dwells={pair['shared_positive_dwells']}")
    print(output)


if __name__ == "__main__":
    main()

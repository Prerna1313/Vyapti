"""Inventory explicit PDW unit annotations in every TSRD HDF5 file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py


def audit_units(root: Path) -> dict:
    root = root.resolve()
    splits = {}
    for split in ("train", "val", "test"):
        modes = {}
        for mode in ("scan", "stare"):
            paths = sorted((root / mode / f"{split}_{mode}").glob("*.h5"))
            attrs = Counter()
            explicit_unit_files = 0
            errors = []
            for path in paths:
                try:
                    with h5py.File(path, "r") as f:
                        found = set()
                        for name, item in (("root", f), ("data", f["data"]),
                                           ("feature_names", f["metadata/feature_names"])):
                            for key in item.attrs:
                                attrs[f"{name}/{key}"] += 1
                                if "unit" in key.lower():
                                    found.add(f"{name}/{key}")
                        if found:
                            explicit_unit_files += 1
                except Exception as exc:
                    errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            modes[mode] = {
                "files_found": len(paths), "files_with_explicit_pdw_unit_attrs": explicit_unit_files,
                "root_data_feature_attr_counts": dict(attrs), "read_errors": errors,
            }
            print(f"unit metadata {split}/{mode}: {len(paths)} files", flush=True)
        splits[split] = modes
    return {
        "scope": "Explicit unit attributes on HDF5 root, data, and feature_names; no scheduler use",
        "limit": "Receiver attribute names and upstream documentation may carry units even if PDW columns have no unit attributes",
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.corpus_root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error("Output must be outside source corpus")
    report = audit_units(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

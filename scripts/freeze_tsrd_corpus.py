"""Hash a local TSRD corpus without inferring an upstream source revision.

The output is a content inventory for the local snapshot. A Hugging Face
revision is recorded only when supplied from independent download evidence.
HDF5 files are listed under ``files`` to match TSRDCorpusLoader's manifest
schema; CSV and TXT sidecars are listed under ``auxiliary_files``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_COUNTS = {"train": 2500, "val": 250, "test": 250}
CONFIG_NAME = re.compile(r"config_(\d+)\.h5\Z")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_corpus(root: Path, source_revision: str | None = None) -> dict:
    """Return file hashes and layout diagnostics for every local source file."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if source_revision is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", source_revision):
        raise ValueError("source_revision must be a full 40-character Git SHA")

    files = []
    auxiliary_files = []
    errors = []
    split_counts: dict[str, dict[str, int]] = {}
    for split, expected in EXPECTED_COUNTS.items():
        names: dict[str, set[str]] = {}
        for mode in ("scan", "stare"):
            directory = root / mode / f"{split}_{mode}"
            paths = sorted(directory.glob("*.h5")) if directory.is_dir() else []
            names[mode] = {p.name for p in paths}
            split_counts.setdefault(split, {})[mode] = len(paths)
            ids = set()
            for path in paths:
                match = CONFIG_NAME.fullmatch(path.name)
                if match is None:
                    errors.append(f"Unexpected HDF5 name: {path.relative_to(root).as_posix()}")
                else:
                    ids.add(int(match.group(1)))
                before = path.stat()
                files.append({
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": before.st_size,
                    "sha256": _hash_file(path),
                })
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    errors.append(f"File changed while hashing: {path.relative_to(root).as_posix()}")
            if len(paths) != expected or ids != set(range(expected)):
                errors.append(
                    f"{split}/{mode}: expected config_0..config_{expected - 1}; "
                    f"found {len(paths)} files and {len(ids)} matching IDs"
                )
            print(f"hashed {split}/{mode}: {len(paths)} HDF5 files", flush=True)
        if names["scan"] != names["stare"]:
            errors.append(f"{split}: scan/stare HDF5 filenames differ")

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".h5" or path.name == "corpus_manifest.json":
            continue
        before = path.stat()
        auxiliary_files.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": before.st_size,
            "sha256": _hash_file(path),
        })
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            errors.append(f"File changed while hashing: {path.relative_to(root).as_posix()}")

    files.sort(key=lambda row: row["path"])
    auxiliary_files.sort(key=lambda row: row["path"])
    canonical = json.dumps(
        {"files": files, "auxiliary_files": auxiliary_files},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "version": "1.0.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "chain_of_custody": {
            "dataset_slug": "alan-turing-institute/turing-synthetic-radar-dataset",
            "dataset_revision": source_revision.lower() if source_revision else None,
            "revision_status": (
                "supplied_by_operator_unverified_against_local_files"
                if source_revision else "unknown_local_download_revision"
            ),
            "local_root_at_freeze": str(root),
        },
        "split_counts": split_counts,
        "h5_file_count": len(files),
        "auxiliary_file_count": len(auxiliary_files),
        "total_source_bytes": sum(row["size_bytes"] for row in files + auxiliary_files),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "layout_anomalies": errors,
        "files": files,
        "auxiliary_files": auxiliary_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    root = args.corpus_root.resolve()
    output = args.output.resolve()
    if output == root or root in output.parents:
        parser.error("Output must be outside the source corpus")
    manifest = freeze_corpus(root, args.source_revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(
        f"manifest={output} h5={manifest['h5_file_count']} "
        f"auxiliary={manifest['auxiliary_file_count']} "
        f"layout_anomalies={len(manifest['layout_anomalies'])} "
        f"inventory_sha256={manifest['inventory_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

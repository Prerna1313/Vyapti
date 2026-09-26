"""
tests._v7_helper
=================

One-shot V7 helper: builds an isolated manifest tree in a temp dir
whose SHA-256 does NOT match the real tsrd_statistics.json, then
invokes the loader against that tree and confirms DataIntegrityError
fires. The real `data_provenance/manifest.json` is never touched.

Run from the repo root:

    python tests/_v7_helper.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure the repo root is on sys.path so the helper works whether it
# is invoked as `python tests/_v7_helper.py` or `python -m tests._v7_helper`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REAL_STATS = REPO_ROOT / "vyapti_simulator" / "data" / "tsrd_statistics.json"

# Use the public API the same way the CLI does.
from vyapti_simulator.core.data_loader import (
    DataIntegrityError,
    load_tsrd_statistics,
)


def main() -> int:
    if not REAL_STATS.is_file():
        print(f"ERROR: real stats file missing: {REAL_STATS}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="v7_") as tmp:
        scratch = Path(tmp)
        # 1. Copy the REAL JSON bytes to the scratch tree.
        scratch_stats = scratch / "tsrd_statistics.json"
        shutil.copy(REAL_STATS, scratch_stats)
        # 2. Write a manifest in the scratch tree with a WRONG hash.
        scratch_prov = scratch / "data_provenance"
        scratch_prov.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": "1.0.0",
            "datasets": {
                "tsrd_statistics": {
                    "sha256": "f" * 64,  # wrong on purpose
                    "path": "tsrd_statistics.json",
                    "provenance": "v7 scratch fixture",
                }
            },
        }
        (scratch_prov / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        # 3. The loader resolves relative paths against the repo root
        #    (parent of vyapti_simulator). It then walks up looking for
        #    data_provenance/manifest.json. To force it to find the
        #    SCRATCH manifest, we point it at the absolute path; the
        #    walk-from-parent routine in the loader will keep walking
        #    up. So instead we monkey-patch the manifest finder.
        from vyapti_simulator.core import data_loader as dl
        original_find = dl._find_repo_root

        def _scratch_root(start: Path):
            # The scratch stats file lives at scratch/tsrd_statistics.json.
            # We just need the walk to land on `scratch` first.
            cur = start.resolve()
            for _ in range(8):
                if cur == scratch:
                    return scratch
                if cur.parent == cur:
                    break
                cur = cur.parent
            return original_find(start)

        dl._find_repo_root = _scratch_root
        try:
            load_tsrd_statistics(scratch_stats)
        except DataIntegrityError as exc:
            print("V7 PASS: DataIntegrityError raised as expected.")
            print(f"  message: {str(exc).splitlines()[0]}")
            return 0
        else:
            print("V7 FAIL: loader returned without raising.")
            return 1
        finally:
            dl._find_repo_root = original_find


if __name__ == "__main__":
    sys.exit(main())

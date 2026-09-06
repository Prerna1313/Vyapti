"""
vyapti_simulator.core.data_loader
==================================

File-property hash-checked loader for derived data artefacts.

[INVARIANT-2] The grid is defined in `SimulationConfig`. The data that
populates the simulator is governed by `data_provenance/manifest.json`,
which records the SHA-256 of every artefact that the simulator is
allowed to load. This module is the SINGLE entry point for loading
those artefacts; any other module that wants to read, e.g.,
`tsrd_statistics.json` MUST go through `load_tsrd_statistics()` so
that the integrity check fires on every load.

[DESIGN-DISCIPLINE] The hash check is a property of **loading the
file itself**, not tied to any CLI flag or run-mode. There is no
`verify=False` / `bypass=True` / `skip_check=True` keyword. If you
can call this function, the check runs. This is the file-property
contract you locked at the start of the TSRD Option-A work.

Errors
------
* `DataIntegrityError` is raised when:
    - the manifest is missing or unreadable,
    - the requested path is not listed in the manifest,
    - the file's SHA-256 does not match the recorded hash,
    - the file does not exist on disk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union


class DataIntegrityError(Exception):
    """
    Raised when a derived data artefact fails its SHA-256 integrity check.

    This is a hard failure. The frozen protocol's Gate-0 (determinism)
    depends on every run seeing the same artefact bytes; if a hash
    mismatch is ignored, the run is no longer reproducible.
    """


# =====================================================================
# Internal helpers
# =====================================================================
def _sha256_of_file(path: Path) -> str:
    """Compute SHA-256 of a file, streaming 1 MiB chunks at a time."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _find_repo_root(start: Path) -> Path:
    """
    Walk upward from `start` until we find a directory containing
    `data_provenance/manifest.json`. Fall back to `start` if not found.
    """
    cur = start.resolve()
    for _ in range(8):  # bounded walk; protects against runaway
        if (cur / "data_provenance" / "manifest.json").is_file():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start


def _load_manifest(repo_root: Path) -> Dict[str, Any]:
    """
    Load and return `data_provenance/manifest.json`. Raises
    `DataIntegrityError` if the manifest itself is missing.
    """
    manifest_path = repo_root / "data_provenance" / "manifest.json"
    if not manifest_path.is_file():
        raise DataIntegrityError(
            f"Manifest not found at {manifest_path}. "
            f"Create data_provenance/manifest.json with the schema "
            f"{{'version': '1.0.0', 'datasets': {{...}}}} before loading "
            f"any TSRD-derived artefact."
        )
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataIntegrityError(
            f"Manifest at {manifest_path} is unreadable or invalid JSON: {exc}"
        ) from exc


# =====================================================================
# Public API
# =====================================================================
def load_tsrd_statistics(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load the TSRD aggregate statistics JSON, verifying SHA-256 against
    `data_provenance/manifest.json` BEFORE returning any data.

    The check is unconditional: there is no flag to disable it. If you
    can call this function, the check runs.

    Parameters
    ----------
    path : str | Path
        Path to `tsrd_statistics.json`. May be relative to the repo root
        (the directory containing `vyapti_simulator/`) or absolute.

    Returns
    -------
    dict
        The parsed JSON content.

    Raises
    ------
    DataIntegrityError
        If the file is missing, the manifest is missing/invalid, the
        file is unlisted in the manifest, or the SHA-256 does not match.
    """
    path = Path(path)
    if not path.is_absolute():
        # Resolve relative to repo root (parent of the package)
        repo_root = _find_repo_root(Path(__file__).resolve().parent.parent.parent)
        path = (repo_root / path).resolve()

    if not path.is_file():
        raise DataIntegrityError(
            f"TSRD statistics file not found: {path}"
        )

    actual_sha = _sha256_of_file(path)

    # Walk up to find the manifest (works whether path was given
    # absolute or relative).
    repo_root = _find_repo_root(path.parent)
    manifest = _load_manifest(repo_root)

    datasets = manifest.get("datasets", {})
    entry = datasets.get("tsrd_statistics")
    if entry is None:
        raise DataIntegrityError(
            f"Manifest does not list a 'tsrd_statistics' entry. "
            f"Add one to data_provenance/manifest.json with the SHA-256 "
            f"of {path.name}. (Actual SHA-256 of the file on disk: "
            f"{actual_sha})"
        )

    expected_sha = entry.get("sha256", "").lower()
    if not expected_sha or expected_sha == "placeholder":
        raise DataIntegrityError(
            f"Manifest's 'tsrd_statistics.sha256' is empty or a placeholder. "
            f"Fill it with the real SHA-256 of {path.name}. "
            f"(Actual SHA-256: {actual_sha})"
        )

    if actual_sha != expected_sha:
        raise DataIntegrityError(
            f"SHA-256 mismatch for {path.name}:\n"
            f"  expected (from manifest): {expected_sha}\n"
            f"  actual   (file on disk)  : {actual_sha}\n"
            f"Either the file was modified after the manifest was sealed, "
            f"or the manifest is stale. Re-run scripts/extract_tsrd_statistics.py "
            f"and update the manifest's sha256 field."
        )

    # All checks passed. Read and return.
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataIntegrityError(
            f"tsrd_statistics.json at {path} is unreadable or invalid JSON: {exc}"
        ) from exc

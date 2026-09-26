"""
tests.test_data_integrity
==========================

CI guard: the SHA-256 hash check on `tsrd_statistics.json` is a
property of LOADING the file, not a CLI flag. These tests pin that
contract.

A wrong hash MUST abort the load with `DataIntegrityError`.
A missing manifest entry MUST abort the load.
A correct hash MUST return the parsed JSON.
"""

import hashlib
import json
import textwrap
from pathlib import Path

import pytest

from vyapti_simulator.core.data_loader import (
    DataIntegrityError,
    load_tsrd_statistics,
    _sha256_of_file,
)


def _write_json(path: Path, payload: dict) -> str:
    """Write `payload` to `path` and return the SHA-256 of the bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2)
    return _sha256_of_file(path)


def _write_manifest(manifest_path: Path, sha256: str | None) -> None:
    """
    Write a minimal manifest. If `sha256` is None, the entry is missing.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "1.0.0",
        "datasets": {},
    }
    if sha256 is not None:
        payload["datasets"]["tsrd_statistics"] = {
            "sha256": sha256,
            "path": "stats.json",
            "provenance": "synthetic test fixture",
        }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2)


class TestLoadTSRDStatistics:
    """
    The hash check is a file property: it fires every time the function
    is called. These three tests cover the three observable outcomes.
    """

    def test_correct_hash_returns_payload(self, tmp_path: Path, monkeypatch):
        # Build a tiny self-contained repo layout under tmp_path so the
        # loader's repo-root walk finds the manifest we write.
        repo = tmp_path
        stats = repo / "stats.json"
        manifest = repo / "data_provenance" / "manifest.json"
        sha = _write_json(stats, {"hello": "world", "n": 7})
        _write_manifest(manifest, sha)

        # The loader's repo-root walk looks for data_provenance/manifest.json
        # starting from the file's parent and walking upward. We give it
        # an absolute path so it walks from tmp_path upward. The walk
        # must find (repo)/data_provenance/manifest.json.
        loaded = load_tsrd_statistics(stats)
        assert loaded == {"hello": "world", "n": 7}

    def test_wrong_hash_raises(self, tmp_path: Path):
        repo = tmp_path
        stats = repo / "stats.json"
        manifest = repo / "data_provenance" / "manifest.json"
        _write_json(stats, {"hello": "world", "n": 7})
        _write_manifest(manifest, sha256="0" * 64)  # intentionally wrong

        with pytest.raises(DataIntegrityError, match="SHA-256 mismatch"):
            load_tsrd_statistics(stats)

    def test_unlisted_path_raises(self, tmp_path: Path):
        repo = tmp_path
        stats = repo / "stats.json"
        manifest = repo / "data_provenance" / "manifest.json"
        _write_json(stats, {"hello": "world", "n": 7})
        # Manifest has the file's datasets map EMPTY (no entry at all)
        _write_manifest(manifest, sha256=None)

        with pytest.raises(DataIntegrityError, match="does not list"):
            load_tsrd_statistics(stats)

    def test_missing_file_raises(self, tmp_path: Path):
        repo = tmp_path
        manifest = repo / "data_provenance" / "manifest.json"
        _write_manifest(manifest, sha256="0" * 64)

        with pytest.raises(DataIntegrityError, match="not found"):
            load_tsrd_statistics(repo / "does_not_exist.json")

    def test_placeholder_sha_raises(self, tmp_path: Path):
        """
        A manifest entry with a literal "placeholder" sha256 must also
        be rejected, so a half-filled manifest cannot silently bypass
        the check.
        """
        repo = tmp_path
        stats = repo / "stats.json"
        manifest = repo / "data_provenance" / "manifest.json"
        _write_json(stats, {"x": 1})
        _write_manifest(manifest, sha256="placeholder")

        with pytest.raises(DataIntegrityError, match="placeholder"):
            load_tsrd_statistics(stats)

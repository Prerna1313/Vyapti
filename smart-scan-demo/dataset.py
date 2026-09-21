# =============================================================================
# PS26055 / Vyapti — lazy TSRD dataset loading
# =============================================================================
#
# This is the code that used to live at module scope in the original
# main.py (its "6. CONFIG + DATA LOADING" section). It downloaded the
# TSRD dataset from HuggingFace and scanned the whole corpus the moment
# main.py was imported — which is the behavior that made main.py unsafe
# to `import` from a FastAPI process.
#
# Here it is a plain function, `load_tsrd_dataset()`, that does nothing
# until you call it. It caches its result in-process, so the first caller
# pays the download/scan cost and everyone after that (CLI script or
# FastAPI adapter) gets the cached list back instantly.
#
# Importing this module — `import dataset` or `from dataset import
# load_tsrd_dataset` — triggers NO network access and NO disk scanning.
# Only calling load_tsrd_dataset() does.
#
# --------------------------------------------------------------------------
# Multi-config support (folder 3 integration):
# load_tsrd_dataset() now scans BOTH:
#   SIH_DATA/2/TSRD_READY/raw/  (config_0.h5 — original)
#   SIH_DATA/3/                 (config_1, 106, 115, 124, 160, 214, 216, 223)
# Episodes from all configs are pooled into VALID_EPISODES and iterated
# round-robin during simulation runs, giving access to all 9 TSRD scenarios.
# --------------------------------------------------------------------------

from pathlib import Path
from typing import List, Optional, Dict, Any
import sys
_vyapti_root = Path(__file__).resolve().parent.parent
if str(_vyapti_root) not in sys.path:
    sys.path.insert(0, str(_vyapti_root))

from vyapti_simulator.tsrd import TSRDCorpusLoader, CorpusFileResult

TSRD_REPO_ID = "alan-turing-institute/turing-synthetic-radar-dataset"
TSRD_REVISION = "68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51"
DEFAULT_CACHE_DIR = "/root/.cache/huggingface/hub"

# Process-wide cache. None = not loaded yet. This mirrors the original
# main.py's VALID_EPISODES global, just populated lazily instead of at
# import time.
_VALID_EPISODES: Optional[List[CorpusFileResult]] = None
_SCAN_DIRS: Optional[List[Path]] = None
_CONFIG_STATS: Optional[List[Dict[str, Any]]] = None


class TSRDLoadError(RuntimeError):
    """Raised when the TSRD dataset cannot be downloaded or scanned.

    Kept as a distinct exception type so the FastAPI layer can catch it
    specifically and return a clean "dataset unavailable" API response
    instead of a bare 500, e.g. when HuggingFace is unreachable or the
    revision pin is stale.
    """


def _scan_dir(scan_dir: Path, label: str) -> List[CorpusFileResult]:
    """Scan a single directory and return valid CorpusFileResult episodes."""
    if not scan_dir.is_dir():
        print(f"[TSRD] WARNING: {label} dir not found: {scan_dir}")
        return []
    loader = TSRDCorpusLoader(corpus_dir=str(scan_dir), require_manifest=False)
    results: List[CorpusFileResult] = []
    for res in loader.iter_corpus():
        if res.pdw_stream is None:
            continue
        results.append(res)
    return results


def load_tsrd_dataset(
    force_reload: bool = False,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> List[CorpusFileResult]:
    """
    Scan ALL available TSRD H5 files (config_0 from SIH_DATA/2 AND the 8 new
    configs from SIH_DATA/3), returning a merged VALID_EPISODES list.

    Episodes are ordered so that config_0 comes first (preserving original
    behaviour), followed by the folder-3 configs in alphabetical order.

    Safe to call multiple times — result is cached in-process.

    Raises TSRDLoadError if NO valid episodes are found at all.
    """
    global _VALID_EPISODES, _SCAN_DIRS, _CONFIG_STATS

    if _VALID_EPISODES is not None and not force_reload:
        return _VALID_EPISODES

    try:
        base = Path(__file__).parent

        # --- Original single config directory ---
        dir0 = base / "SIH_DATA" / "2" / "TSRD_READY" / "raw"
        # --- New folder-3 configs ---
        dir3 = base / "SIH_DATA" / "3"

        all_episodes: List[CorpusFileResult] = []
        stats: List[Dict[str, Any]] = []
        scan_dirs: List[Path] = []

        print(f"[TSRD] Scanning config_0 from: {dir0}")
        ep0 = _scan_dir(dir0, "config_0")
        if ep0:
            all_episodes.extend(ep0)
            stats.append({"dir": str(dir0), "label": "config_0", "count": len(ep0)})
            scan_dirs.append(dir0)
            print(f"[TSRD]   config_0: {len(ep0)} valid episode(s)")
        else:
            print("[TSRD]   config_0: no valid episodes found (skipping)")

        print(f"[TSRD] Scanning folder-3 configs from: {dir3}")
        ep3 = _scan_dir(dir3, "folder3")
        if ep3:
            # Group by filename for stats
            by_file: Dict[str, int] = {}
            for ep in ep3:
                fname = Path(ep.path).stem if hasattr(ep, "path") else "unknown"
                by_file[fname] = by_file.get(fname, 0) + 1
            all_episodes.extend(ep3)
            stats.append({"dir": str(dir3), "label": "folder3", "count": len(ep3), "by_file": by_file})
            scan_dirs.append(dir3)
            print(f"[TSRD]   folder3: {len(ep3)} valid episode(s) across {len(by_file)} configs")
        else:
            print("[TSRD]   folder3: no valid episodes found (skipping)")

        if not all_episodes:
            raise TSRDLoadError("No valid TSRD episodes found in any scan directory.")

        print(f"[TSRD] Total valid episodes available: {len(all_episodes)}")

        _VALID_EPISODES = all_episodes
        _SCAN_DIRS = scan_dirs
        _CONFIG_STATS = stats
        return _VALID_EPISODES

    except TSRDLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - intentionally broad, re-raised as our own type
        raise TSRDLoadError(
            f"Could not load the TSRD dataset ({TSRD_REPO_ID} @ {TSRD_REVISION}): {exc}"
        ) from exc


def is_dataset_loaded() -> bool:
    """True once load_tsrd_dataset() has succeeded at least once in this process."""
    return _VALID_EPISODES is not None


def dataset_size() -> Optional[int]:
    """Number of valid episodes found, or None if not loaded yet."""
    return len(_VALID_EPISODES) if _VALID_EPISODES is not None else None


def dataset_scan_dir() -> Optional[str]:
    """Return the primary scan directory (config_0 original), or None."""
    if _SCAN_DIRS:
        return str(_SCAN_DIRS[0])
    return None


def dataset_config_stats() -> Optional[List[Dict[str, Any]]]:
    """Return per-directory scan stats, or None if not loaded yet."""
    return _CONFIG_STATS

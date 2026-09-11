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

from pathlib import Path
from typing import List, Optional
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
_SCAN_DIR: Optional[Path] = None


class TSRDLoadError(RuntimeError):
    """Raised when the TSRD dataset cannot be downloaded or scanned.

    Kept as a distinct exception type so the FastAPI layer can catch it
    specifically and return a clean "dataset unavailable" API response
    instead of a bare 500, e.g. when HuggingFace is unreachable or the
    revision pin is stale.
    """


def load_tsrd_dataset(
    force_reload: bool = False,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> List[CorpusFileResult]:
    """
    Download (if needed) and scan the TSRD corpus, returning the same
    VALID_EPISODES list the original main.py built at import time.

    Safe to call multiple times — after the first successful call the
    result is cached in-process and returned immediately, so this is
    cheap to call from every /api/simulation/start request.

    Raises TSRDLoadError if the download or corpus scan fails, instead
    of letting an arbitrary huggingface_hub/vyapti_simulator exception
    propagate — callers (the FastAPI routes) should catch this and
    report a clear "TSRD unavailable" status rather than crashing.
    """
    global _VALID_EPISODES, _SCAN_DIR

    if _VALID_EPISODES is not None and not force_reload:
        return _VALID_EPISODES

    try:
        scan_dir = Path(__file__).parent / "SIH_DATA" / "2" / "TSRD_READY" / "raw"
        print(f"[TSRD] Using local TSRD data from: {scan_dir}")

        loader = TSRDCorpusLoader(corpus_dir=str(scan_dir), require_manifest=False)
        valid_episodes: List[CorpusFileResult] = []
        for res in loader.iter_corpus():
            if res.pdw_stream is None:
                continue
            valid_episodes.append(res)
        print(f"[TSRD] Found {len(valid_episodes)} valid episodes.")

        _VALID_EPISODES = valid_episodes
        _SCAN_DIR = scan_dir
        return _VALID_EPISODES

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
    return str(_SCAN_DIR) if _SCAN_DIR is not None else None

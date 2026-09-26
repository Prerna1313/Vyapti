#!/usr/bin/env python3
"""
scripts.aggregate_tsrd_statistics
=================================

Gap 3 — Run `extract_tsrd_statistics.py` on 5 representative TSRD
configs and merge the per-file statistics into one aggregate JSON.

The 5-config sample is enough to characterise the corpus range
without downloading the full 250-file test split. The aggregated
`tsrd_statistics_aggregate.json` is the file the simulator carries
on-laptop; the full corpus only needs to be accessed on Kaggle.

Usage (Kaggle)
--------------
    python scripts/aggregate_tsrd_statistics.py \\
        --corpus /kaggle/input/turing-synthetic-radar-dataset \\
        --configs 0 50 100 150 200 \\
        --output vyapti_simulator/data/tsrd_statistics_aggregate.json

Usage (on-laptop, fixtures)
---------------------------
    python scripts/aggregate_tsrd_statistics.py \\
        --corpus tests/fixtures/tsrd \\
        --configs 0 \\
        --output /tmp/tsrd_stats_agg.json

The script:
  1. Calls `extract_tsrd_statistics.py` once per config (in-process,
     no subprocess overhead), capturing the full per-file JSON dict.
  2. Merges the population_stats across all 5:
       - period_slots, freq_range_mhz, snr_db  -> global min/median/max
       - freq_mode_counts, pri_mode_counts, scan_type_counts -> summed
  3. Concatenates the per-emitter libraries (all 5*2 = 10 entries
     in the test corpus, with tx_ids from 0..N-1 per file).
  4. Computes a global "n_active" and "n_silent" sum.
  5. Writes the merged JSON with the same schema as the single-file
     output, so `TSRDEmitterSampler` consumes it unchanged.

[DESIGN] Re-uses `extract_tsrd_statistics.extract_emitter_library` and
`compute_population_stats` rather than shelling out. This keeps the
aggregation fast (one H5 open per file, no repeated JSON parsing)
and avoids needing the script to be on $PATH on the Kaggle runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Re-use the existing extractor helpers
from extract_tsrd_statistics import (  # type: ignore
    extract_emitter_library,
    compute_population_stats,
    count_labels,
    _sha256_of_file,
    inspect_h5,
)


# =====================================================================
# Merging
# =====================================================================
def merge_population_stats(
    pop_stats_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merge a list of per-file population_stats into one aggregate.

    Strategy:
      - period_slots, freq_range_mhz, snr_db -> global min/median/max
        across all input files (min-of-mins, max-of-maxes, median of medians).
      - freq_mode_counts, pri_mode_counts, scan_type_counts -> summed
        (each file is a separate population, the union is the aggregate).
    """
    if not pop_stats_list:
        return {}

    def collect_min_max_median(field: str) -> Dict[str, int]:
        vals_min, vals_med, vals_max = [], [], []
        for ps in pop_stats_list:
            sub = ps.get(field, {})
            if not sub:
                continue
            mn, md, mx = sub.get("min"), sub.get("median"), sub.get("max")
            if isinstance(mn, (int, float)):
                vals_min.append(mn)
            if isinstance(md, (int, float)):
                vals_med.append(md)
            if isinstance(mx, (int, float)):
                vals_max.append(mx)
        if not vals_min and not vals_max:
            return {"min": 0, "median": 0, "max": 0}
        return {
            "min": int(min(vals_min)) if vals_min else 0,
            "median": int(np.median(vals_med)) if vals_med else 0,
            "max": int(max(vals_max)) if vals_max else 0,
        }

    def collect_freq_range() -> Dict[str, float]:
        mins, maxs = [], []
        for ps in pop_stats_list:
            fr = ps.get("freq_range_mhz", {})
            if fr:
                mins.append(fr.get("min", 0.0))
                maxs.append(fr.get("max", 0.0))
        if not mins:
            return {"min": 0.0, "max": 0.0}
        return {"min": float(min(mins)), "max": float(max(maxs))}

    def collect_snr_db() -> Dict[str, float]:
        mins, meds, maxs = [], [], []
        for ps in pop_stats_list:
            s = ps.get("snr_db", {})
            if s:
                mins.append(s.get("min", 0.0))
                meds.append(s.get("median", 0.0))
                maxs.append(s.get("max", 0.0))
        if not mins:
            return {"min": 0.0, "median": 0.0, "max": 0.0}
        return {
            "min": float(min(mins)),
            "median": float(np.median(meds)) if meds else 0.0,
            "max": float(max(maxs)),
        }

    def collect_counts(field: str) -> Dict[str, int]:
        c: Counter = Counter()
        for ps in pop_stats_list:
            sub = ps.get(field, {})
            for k, v in sub.items():
                c[k] += int(v)
        return dict(c)

    return {
        "n_active_total": int(sum(ps.get("n_active", 0) for ps in pop_stats_list)),
        "n_silent_total": int(sum(ps.get("n_silent", 0) for ps in pop_stats_list)),
        "period_slots": collect_min_max_median("period_slots"),
        "freq_range_mhz": collect_freq_range(),
        "snr_db": collect_snr_db(),
        "freq_mode_counts": collect_counts("freq_mode_counts"),
        "pri_mode_counts": collect_counts("pri_mode_counts"),
        "scan_type_counts": collect_counts("scan_type_counts"),
    }


def merge_emitter_libraries(
    libraries: List[Dict[str, Dict[str, Any]]],
    source_files: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Concatenate per-emitter library entries across files.

    Each input library is keyed by string tx_id (0..N-1 within that
    file). Across files, TSRD uses per-file-scoped labels so the same
    tx_id from different files refers to different physical emitters.
    We disambiguate by prefixing the key with the file index::

        "0_from_file_0": <entry from config_0, tx 0>
        "0_from_file_1": <entry from config_50, tx 0>
        ...

    This preserves the existing `TSRDEmitterSampler` semantics while
    keeping each entry's full metadata (freq_mode, scan_type, position,
    power, etc.) traceable back to the source H5.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for f_idx, (lib, src) in enumerate(zip(libraries, source_files)):
        src_path = Path(src)
        # Compute SHA-256 if the file exists; otherwise store empty string
        try:
            src_sha = _sha256_of_file(src_path) if src_path.is_file() else ""
        except OSError:
            src_sha = ""
        for tx_key, entry in lib.items():
            new_key = f"{tx_key}__from_{src_path.stem}"
            new_entry = dict(entry)
            new_entry["__source_h5__"] = str(src)
            new_entry["__source_h5_sha256__"] = src_sha
            merged[new_key] = new_entry
    return merged


def merge_label_counts(
    label_counts_list: List[Dict[int, int]],
    source_files: List[str],
) -> Dict[str, int]:
    """Concatenate label counts, prefixed by source file."""
    merged: Dict[str, int] = {}
    for f_idx, (counts, src) in enumerate(zip(label_counts_list, source_files)):
        stem = Path(src).stem
        for label, count in counts.items():
            merged[f"{label}__{stem}"] = int(count)
    return merged


# =====================================================================
# Main
# =====================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate TSRD statistics from 5 representative configs."
    )
    ap.add_argument("--corpus", required=True, type=Path,
                    help="Directory containing the TSRD .h5 files")
    ap.add_argument("--configs", type=int, nargs="+", default=[0, 50, 100, 150, 200],
                    help="Config numbers to sample (default: 0 50 100 150 200)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Where to write tsrd_statistics_aggregate.json")
    ap.add_argument("--slot-ms", type=float, default=50.0,
                    help="Slot size in ms (default 50, matches TSRD fine dwell)")
    args = ap.parse_args()

    if not args.corpus.is_dir():
        print(f"ERROR: corpus dir not found: {args.corpus}", file=sys.stderr)
        return 1

    # Resolve each config -> H5 file
    h5_paths: List[Path] = []
    for n in args.configs:
        # TSRD layout on Kaggle: scan/test_scan/config_<n>.h5
        # TSRD layout in fixture: config_0_scan.h5
        candidates = [
            args.corpus / f"config_{n}.h5",
            args.corpus / f"config_{n}_scan.h5",
            args.corpus / "scan" / "test_scan" / f"config_{n}.h5",
            args.corpus / "scan" / f"config_{n}.h5",
        ]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            print(f"ERROR: config_{n}.h5 not found under {args.corpus}", file=sys.stderr)
            return 1
        h5_paths.append(found)

    print(f"[aggregate] Sampling {len(h5_paths)} configs:")
    for p in h5_paths:
        print(f"  {p}")

    # Extract per-file
    libraries: List[Dict[str, Dict[str, Any]]] = []
    pop_stats_list: List[Dict[str, Any]] = []
    label_counts_list: List[Dict[int, int]] = []
    file_infos: List[Dict[str, Any]] = []
    source_h5_sha256s: List[str] = []

    for p in h5_paths:
        print(f"[aggregate]   extracting {p.name} ...")
        file_infos.append(inspect_h5(p))
        label_counts = count_labels(p)
        library = extract_emitter_library(p)
        pop_stats = compute_population_stats(library, label_counts)
        libraries.append(library)
        pop_stats_list.append(pop_stats)
        label_counts_list.append(label_counts)
        source_h5_sha256s.append(_sha256_of_file(p))
        n_active = sum(1 for c in label_counts.values() if c > 0)
        print(f"      active={n_active}, library_size={len(library)}")

    # Merge
    print("[aggregate] Merging ...")
    merged_pop = merge_population_stats(pop_stats_list)
    merged_lib = merge_emitter_libraries(
        libraries, [str(p) for p in h5_paths]
    )
    merged_labels = merge_label_counts(
        label_counts_list, [str(p) for p in h5_paths]
    )

    # Assemble final JSON with the same schema as the single-file output
    out: Dict[str, Any] = {
        "dataset": "Turing Synthetic Radar Dataset",
        "source_files": [str(p) for p in h5_paths],
        "source_h5_sha256s": source_h5_sha256s,
        "extraction_date_utc": datetime.now(timezone.utc).isoformat(),
        "extractor_version": "1.1.0",
        "aggregator_version": "1.0.0",
        "slot_ms": args.slot_ms,
        "n_configs_aggregated": len(h5_paths),
        "file_infos": file_infos,
        "population_stats": merged_pop,
        "emitter_library": merged_lib,
        "label_counts": {str(k): v for k, v in merged_labels.items()},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    sha = _sha256_of_file(args.output)
    print(f"[aggregate] Wrote {args.output}")
    print(f"[aggregate] SHA-256 = {sha}")
    print(
        f"[aggregate] Aggregate n_active={merged_pop.get('n_active_total', '?')}, "
        f"emitter_library size={len(merged_lib)}"
    )
    print()
    print("NEXT STEP:")
    print(f"  Paste the SHA-256 above into data_provenance/manifest.json")
    print(f"  under datasets.tsrd_statistics_aggregate.sha256.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

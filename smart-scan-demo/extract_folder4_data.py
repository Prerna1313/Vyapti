# =============================================================================
# extract_folder4_data.py — Pre-process TSRD HDF5 STARE files from Vyapti/4/
# =============================================================================
#
# Folder 4 contains STARE-mode data: the receiver dwells at a fixed frequency
# (dwell_centres_mhz / dwell_times_s are empty arrays), unlike folder 3 which
# has SCAN-mode data with a sweeping receiver.
#
# Run once (from the smart-scan-demo/ directory) to generate:
#   lib/tsrd_folder4_configs.json       — per-config metadata for frontend
#   lib/tsrd_all_emitters.json          — merged emitter list (scan + stare)
#   SIH_DATA/4/{cid}_obs_matrix.npy    — 36-band occupancy matrix per config
#   SIH_DATA/4/{cid}_band_activity.npy — 36-band activity fractions per config
#   SIH_DATA/4/{cid}_pulse_counts.npy  — 36-band pulse count matrix per config
#
# Usage:
#   cd smart-scan-demo
#   python extract_folder4_data.py

import os
import json
import numpy as np
from pathlib import Path

try:
    import h5py
except ImportError:
    raise SystemExit("h5py is required: pip install h5py")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent          # smart-scan-demo/
FOLDER4      = SCRIPT_DIR.parent / "4"                  # Vyapti/4/
LIB_DIR      = SCRIPT_DIR / "lib"
SIH_DATA_4   = SCRIPT_DIR / "SIH_DATA" / "4"
SIH_DATA_4.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 36-band receiver definitions for STARE data
# Stare data covers ~1.0–11.4 GHz, so we use FREQ_MIN=500 MHz to ensure
# those frequencies land in bands 1..21 (500 MHz IBW per band).
# ---------------------------------------------------------------------------
N_BANDS   = 36
FREQ_MIN  = 500.0     # MHz  (covers 500–18500 MHz across 36 bands)
BAND_IBW  = 500.0     # MHz per band

def freq_to_band(freq_mhz: float) -> int:
    """Map a frequency in MHz → band index 0..35 (STARE mapping: 500 MHz base)."""
    b = int((freq_mhz - FREQ_MIN) / BAND_IBW)
    return max(0, min(N_BANDS - 1, b))

def extract_stare_config(h5_path: Path) -> dict:
    """
    Extract all useful info from one TSRD STARE HDF5 file.
    Returns a dict with metadata + numpy arrays attached under '_arrays'.

    Key difference from scan: dwell_centres_mhz and dwell_times_s will be
    empty (stare receiver), so we mark dataMode='stare' and skip them.
    """
    raw_config_id = h5_path.stem          # e.g. "config_1"
    config_id     = f"stare_{raw_config_id}"  # e.g. "stare_config_1"
    print(f"  Processing {h5_path.name} ...", end=" ", flush=True)

    with h5py.File(h5_path, "r") as f:
        # ---- PDW stream ----
        data   = f["data"][:]       # (N, 5): ToA, Freq_MHz, PW_us, AoA_deg, Amp_dBm
        labels = f["labels"][:].flatten().astype(int)

        feat_names = [
            x.decode() if isinstance(x, bytes) else x
            for x in f["metadata"]["feature_names"][:]
        ]

        # column indices
        col = {n: i for i, n in enumerate(feat_names)}
        toa_col   = col.get("ToA",        0)
        freq_col  = col.get("Frequency",  1)
        pw_col    = col.get("PulseWidth", 2)
        aoa_col   = col.get("AoA",        3)
        amp_col   = col.get("Amplitude",  4)

        freqs_mhz  = data[:, freq_col].astype(float)
        pws_us     = data[:, pw_col].astype(float)
        aoas_deg   = data[:, aoa_col].astype(float)
        amps_dbm   = data[:, amp_col].astype(float)
        toas_s     = data[:, toa_col].astype(float)

        # ---- Receiver info (stare: dwell arrays will be empty) ----
        rec             = f["metadata"]["receiver"]
        dwell_centres   = rec["dwell_centres_mhz"][:].tolist()   # [] for stare
        dwell_times_s   = rec["dwell_times_s"][:].tolist()        # [] for stare
        freq_range_mhz  = rec["freq_range_mhz"][:].tolist()
        rx_pos_km       = rec["start_position_km"][:].tolist()

        # ---- Transmitter metadata ----
        tx_group  = f["metadata"]["transmitters"]
        tx_keys   = sorted(tx_group.keys(), key=lambda k: int(k.split("_")[1]))
        unique_labels = sorted(np.unique(labels).tolist())

        emitters = []
        for tx_key in tx_keys:
            tx_id = int(tx_key.split("_")[1])
            tx    = tx_group[tx_key]

            def _arr(subgroup, key):
                try:
                    return tx[subgroup][key][:].tolist()
                except Exception:
                    return []

            freqs   = _arr("frequency_config", "freqs_mhz")
            pris    = _arr("pri_config",        "pris_us")
            pws     = _arr("pulse_width_config","pws_us")
            pos     = _arr("position_config",   "start_position_km")

            # Mean frequency for band mapping (stare FREQ_MIN=500 MHz)
            mean_freq = float(np.mean(freqs)) if freqs else 5000.0
            band_idx  = freq_to_band(mean_freq)

            # Pulses from this transmitter
            mask      = labels == tx_id
            tx_pulses = int(mask.sum())

            # Agility: multiple freqs = frequency-agile
            agility   = round(min(1.0, (len(freqs) - 1) / 5.0), 2) if freqs else 0.0

            emitters.append({
                "id":           f"{config_id}_tx{tx_id}",
                "configId":     config_id,
                "transmitterId": tx_id,
                "type":         "RADAR",
                "dataMode":     "stare",
                "band":         band_idx,
                "freq":         round(mean_freq / 1000.0, 3),      # GHz
                "freq_mhz":     round(mean_freq, 2),
                "freqs_mhz":    [round(x, 2) for x in freqs],
                "pri":          f"{round(float(np.mean(pris)), 1)} µs" if pris else "N/A",
                "pris_us":      [round(x, 2) for x in pris],
                "pw":           f"{round(float(np.mean(pws)), 2)} µs" if pws else "N/A",
                "pws_us":       [round(x, 2) for x in pws],
                "pos_km":       [round(x, 2) for x in pos] if pos else [0.0, 0.0],
                "pulses":       tx_pulses,
                "active":       tx_pulses > 0,
                "agility":      agility,
                "detected":     False,
                "power":        "N/A",
            })

        # ---- 36-band occupancy / activity matrix ----
        # Discretise ToA into 290 time-slots (matching existing obs_matrix shape).
        # For stare data the slot density is much higher (millions of pulses vs
        # hundreds of thousands for scan), so the obs_matrix tends to saturate.
        N_SLOTS   = 290
        toa_min   = toas_s.min() if len(toas_s) else 0.0
        toa_max   = toas_s.max() if len(toas_s) else 1.0
        toa_range = max(toa_max - toa_min, 1e-9)

        obs_matrix   = np.zeros((N_SLOTS, N_BANDS), dtype=np.int8)
        pulse_counts = np.zeros((N_SLOTS, N_BANDS), dtype=np.int32)

        slot_idx = np.clip(
            ((toas_s - toa_min) / toa_range * (N_SLOTS - 1)).astype(int),
            0, N_SLOTS - 1
        )
        band_idx_arr = np.vectorize(freq_to_band)(freqs_mhz)

        for s, b in zip(slot_idx, band_idx_arr):
            obs_matrix[s, b]   = 1
            pulse_counts[s, b] += 1

        band_activity = obs_matrix.mean(axis=0)   # fraction of slots occupied per band

    print(f"OK  ({len(emitters)} tx, {data.shape[0]:,} pulses)")

    return {
        "configId":        config_id,
        "rawConfigId":     raw_config_id,
        "filename":        h5_path.name,
        "dataMode":        "stare",
        "pulseCount":      int(data.shape[0]),
        "txCount":         len(emitters),
        "uniqueEmitters":  len(unique_labels),
        "freqMinMhz":      round(float(freqs_mhz.min()), 1),
        "freqMaxMhz":      round(float(freqs_mhz.max()), 1),
        "freqRangeMhz":    [round(x, 1) for x in freq_range_mhz],
        "rxPositionKm":    [round(x, 2) for x in rx_pos_km],
        "dwellCentresMhz": dwell_centres,    # [] — stare mode (no scan)
        "dwellTimesS":     dwell_times_s,    # [] — stare mode (no scan)
        "bandActivity":    [round(float(x), 4) for x in band_activity],
        "_emitters":       emitters,
        "_obs_matrix":     obs_matrix,
        "_pulse_counts":   pulse_counts,
    }


def main():
    # ---- find all H5 files in folder 4 ----
    if not FOLDER4.is_dir():
        raise SystemExit(f"Folder 4 not found at {FOLDER4}")

    h5_files = sorted(FOLDER4.glob("*.h5"), key=lambda p: p.name)
    if not h5_files:
        raise SystemExit(f"No .h5 files found in {FOLDER4}")

    print(f"\nFound {len(h5_files)} STARE HDF5 files in {FOLDER4}:")
    for f in h5_files:
        print(f"  {f.name}  ({f.stat().st_size // 1024:,} KB)")

    print("\nExtracting stare data...")
    stare_configs = []
    stare_emitters = []

    for h5_path in h5_files:
        info = extract_stare_config(h5_path)

        # Save obs matrix and pulse count matrix as .npy
        cid = info["configId"]
        np.save(SIH_DATA_4 / f"{cid}_obs_matrix.npy",    info["_obs_matrix"])
        np.save(SIH_DATA_4 / f"{cid}_pulse_counts.npy",  info["_pulse_counts"])
        np.save(SIH_DATA_4 / f"{cid}_band_activity.npy",
                np.array(info["bandActivity"], dtype=np.float32))

        # Accumulate emitters
        stare_emitters.extend(info["_emitters"])

        # Strip internal numpy arrays before JSON serialisation
        config_meta = {k: v for k, v in info.items() if not k.startswith("_")}
        stare_configs.append(config_meta)

    # ---- Write folder 4 JSON ----
    folder4_json = LIB_DIR / "tsrd_folder4_configs.json"
    with folder4_json.open("w") as f:
        json.dump(stare_configs, f, indent=2)
    print(f"\n[OK] Wrote {folder4_json}  ({len(stare_configs)} stare configs)")

    # ---- Merge into tsrd_all_emitters.json ----
    # Load existing (scan) emitters, strip any old stare entries, then append new stare
    all_emitters_path = LIB_DIR / "tsrd_all_emitters.json"
    scan_emitters = []
    if all_emitters_path.is_file():
        try:
            with all_emitters_path.open() as f:
                existing = json.load(f)
            # Keep only non-stare emitters (from folder 3 + config_0)
            scan_emitters = [e for e in existing if e.get("dataMode") != "stare"]
            print(f"[OK] Loaded {len(scan_emitters)} existing scan emitters from tsrd_all_emitters.json")
        except Exception as e:
            print(f"[WARN] Could not read existing all_emitters: {e}")

    merged_emitters = scan_emitters + stare_emitters
    with all_emitters_path.open("w") as f:
        json.dump(merged_emitters, f, indent=2)
    print(f"[OK] Updated {all_emitters_path}  ({len(merged_emitters)} total emitters: "
          f"{len(scan_emitters)} scan + {len(stare_emitters)} stare)")

    # ---- Summary ----
    total_pulses  = sum(c["pulseCount"] for c in stare_configs)
    total_tx      = sum(c["txCount"]    for c in stare_configs)
    print(f"\n=== Summary ===")
    print(f"Stare configs processed : {len(stare_configs)}")
    print(f"Total stare pulses      : {total_pulses:,}")
    print(f"Total stare emitters    : {total_tx}")
    print(f"NPY files in            : {SIH_DATA_4}")
    print(f"\nAll done!")


if __name__ == "__main__":
    main()

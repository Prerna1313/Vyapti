# =============================================================================
# extract_folder3_data.py — Pre-process TSRD HDF5 files from Vyapti/3/
# =============================================================================
#
# Run once (from the smart-scan-demo/ directory) to generate:
#   lib/tsrd_folder3_configs.json   — per-config metadata for frontend
#   lib/tsrd_all_emitters.json      — merged emitter list from all 9 configs
#   SIH_DATA/3/{cid}_obs_matrix.npy — 36-band occupancy matrix per config
#   SIH_DATA/3/{cid}_band_activity.npy — 36-band activity fractions per config
#
# Usage:
#   cd smart-scan-demo
#   python extract_folder3_data.py

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
FOLDER3      = SCRIPT_DIR.parent / "3"                  # Vyapti/3/
LIB_DIR      = SCRIPT_DIR / "lib"
SIH_DATA_3   = SCRIPT_DIR / "SIH_DATA" / "3"
SIH_DATA_3.mkdir(parents=True, exist_ok=True)

# 36-band receiver definitions (500 MHz IBW, 2–20 GHz)
N_BANDS   = 36
FREQ_MIN  = 2000.0    # MHz
BAND_IBW  = 500.0     # MHz per band

def freq_to_band(freq_mhz: float) -> int:
    """Map a frequency in MHz → band index 0..35."""
    b = int((freq_mhz - FREQ_MIN) / BAND_IBW)
    return max(0, min(N_BANDS - 1, b))

def extract_config(h5_path: Path) -> dict:
    """
    Extract all useful info from one TSRD HDF5 file.
    Returns a dict with metadata + numpy arrays attached under '_arrays'.
    """
    config_id = h5_path.stem   # e.g. "config_1"
    print(f"  Processing {h5_path.name} ...", end=" ", flush=True)

    with h5py.File(h5_path, "r") as f:
        # ---- PDW stream ----
        data   = f["data"][:]      # (N, 5): ToA, Freq_MHz, PW_us, AoA_deg, Amp_dBm
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

        # ---- Receiver info ----
        rec             = f["metadata"]["receiver"]
        dwell_centres   = rec["dwell_centres_mhz"][:].tolist()
        dwell_times_s   = rec["dwell_times_s"][:].tolist()
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

            # Mean frequency for band mapping
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
        # Build a band-vs-time-slot occupancy matrix.
        # We discretise ToA into 290 time-slots (matching existing obs_matrix shape)
        N_SLOTS  = 290
        toa_min  = toas_s.min() if len(toas_s) else 0.0
        toa_max  = toas_s.max() if len(toas_s) else 1.0
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

    print(f"OK  ({len(emitters)} tx, {data.shape[0]} pulses)")

    return {
        "configId":        config_id,
        "filename":        h5_path.name,
        "pulseCount":      int(data.shape[0]),
        "txCount":         len(emitters),
        "uniqueEmitters":  len(unique_labels),
        "freqMinMhz":      round(float(freqs_mhz.min()), 1),
        "freqMaxMhz":      round(float(freqs_mhz.max()), 1),
        "freqRangeMhz":    [round(x, 1) for x in freq_range_mhz],
        "rxPositionKm":    [round(x, 2) for x in rx_pos_km],
        "dwellCentresMhz": [round(x, 1) for x in dwell_centres],
        "dwellTimesS":     [round(x, 4) for x in dwell_times_s],
        "bandActivity":    [round(float(x), 4) for x in band_activity],
        "_emitters":       emitters,
        "_obs_matrix":     obs_matrix,
        "_pulse_counts":   pulse_counts,
    }


def main():
    # ---- find all H5 files in folder 3 ----
    if not FOLDER3.is_dir():
        raise SystemExit(f"Folder 3 not found at {FOLDER3}")

    h5_files = sorted(FOLDER3.glob("*.h5"), key=lambda p: p.name)
    if not h5_files:
        raise SystemExit(f"No .h5 files found in {FOLDER3}")

    print(f"\nFound {len(h5_files)} HDF5 files in {FOLDER3}:")
    for f in h5_files:
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")

    print("\nExtracting data...")
    configs   = []
    all_emitters = []

    for h5_path in h5_files:
        info = extract_config(h5_path)

        # Save obs matrix and pulse count matrix as .npy
        cid = info["configId"]
        np.save(SIH_DATA_3 / f"{cid}_obs_matrix.npy",   info["_obs_matrix"])
        np.save(SIH_DATA_3 / f"{cid}_pulse_counts.npy", info["_pulse_counts"])
        np.save(SIH_DATA_3 / f"{cid}_band_activity.npy",
                np.array(info["bandActivity"], dtype=np.float32))

        # Accumulate emitters
        all_emitters.extend(info["_emitters"])

        # Strip internal numpy arrays before JSON serialisation
        config_meta = {k: v for k, v in info.items() if not k.startswith("_")}
        configs.append(config_meta)

    # ---- Also include config_0 basic entry (from existing SIH_DATA/2) ----
    existing_obs = SCRIPT_DIR / "SIH_DATA" / "2" / "TSRD_READY" / "observation_matrix.npy"
    if existing_obs.is_file():
        m = np.load(existing_obs)
        ba = m.mean(axis=0).tolist()
        # Pad or trim to 36 bands
        if len(ba) < 36:
            ba = ba + [0.0] * (36 - len(ba))
        else:
            ba = ba[:36]

        # Also load existing 72-emitter list from JSON if available
        existing_json = LIB_DIR / "tsrd_emitters_72.json"
        existing_emitters = []
        if existing_json.is_file():
            with existing_json.open() as fj:
                ex_raw = json.load(fj)
                for e in ex_raw:
                    ec = dict(e)
                    ec["configId"] = "config_0"
                    if "id" not in ec:
                        ec["id"] = f"config_0_tx{ec.get('index', 0)}"
                    existing_emitters.append(ec)

        config0_meta = {
            "configId":        "config_0",
            "filename":        "config_0.h5",
            "pulseCount":      0,              # original file in separate location
            "txCount":         len(existing_emitters),
            "uniqueEmitters":  len(existing_emitters),
            "freqMinMhz":      500.0,
            "freqMaxMhz":      18000.0,
            "freqRangeMhz":    [500.0, 18000.0],
            "rxPositionKm":    [100.0, 100.0],
            "dwellCentresMhz": [250.0 + i * 500.0 for i in range(36)],
            "dwellTimesS":     [0.05] * 36,
            "bandActivity":    [round(float(x), 4) for x in ba],
        }
        # Prepend config_0 so it's first
        configs.insert(0, config0_meta)
        all_emitters = existing_emitters + all_emitters

    # ---- Write JSON outputs ----
    configs_json = LIB_DIR / "tsrd_folder3_configs.json"
    with configs_json.open("w") as f:
        json.dump(configs, f, indent=2)
    print(f"\n[OK] Wrote {configs_json}  ({len(configs)} configs)")

    emitters_json = LIB_DIR / "tsrd_all_emitters.json"
    with emitters_json.open("w") as f:
        json.dump(all_emitters, f, indent=2)
    print(f"[OK] Wrote {emitters_json}  ({len(all_emitters)} total emitters)")

    # ---- Summary ----
    total_pulses   = sum(c["pulseCount"] for c in configs)
    total_tx       = sum(c["txCount"]    for c in configs)
    print(f"\n=== Summary ===")
    print(f"Configs processed : {len(configs)}")
    print(f"Total pulses      : {total_pulses:,}")
    print(f"Total emitters    : {total_tx}")
    print(f"NPY files in      : {SIH_DATA_3}")
    print(f"\nAll done!")


if __name__ == "__main__":
    main()

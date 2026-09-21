# Vyapti Hybrid Meta‑UCB — Local Setup Instructions

This document tells you exactly which files to copy to your local system and how to run the hybrid scheduler.

## Files you must copy from Kaggle to local

From your Kaggle `/kaggle/working` directory, copy these files to your local project folder:

1. `vyapti_hybrid_meta_ucb_combiner.py`
2. `vyapti_hybrid_scheduler.py`
3. `vyapti_meta_ucb_combiner_frozen.py` (the frozen baseline combiner)
4. `base_expert_regret_analysis_V3.json` (V3 empirical parameters)

Example (on your local machine, assuming you have SSH access to Kaggle):

```bash
scp user@kaggle:/kaggle/working/vyapti_hybrid_meta_ucb_combiner.py ./vyapti_local/
scp user@kaggle:/kaggle/working/vyapti_hybrid_scheduler.py ./vyapti_local/
scp user@kaggle:/kaggle/working/vyapti_meta_ucb_combiner_frozen.py ./vyapti_local/
scp user@kaggle:/kaggle/working/base_expert_regret_analysis_V3.json ./vyapti_local/
```

Or download them manually from the Kaggle output panel.

## Directory structure (recommended)

On your local machine, use something like:

```text
vyapti_local/
├── vyapti_meta_ucb_combiner_frozen.py
├── vyapti_hybrid_meta_ucb_combiner.py
├── vyapti_hybrid_scheduler.py
├── base_expert_regret_analysis_V3.json
└── run_hybrid_local.py
```

## Minimal local evaluation script

Create `run_hybrid_local.py` with this content:

```python
#!/usr/bin/env python3
"""
Minimal local test of HybridMetaScheduler.
"""

import json
from pathlib import Path

import numpy as np

from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.core.metrics import MetricsConfig
from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment
from vyapti_simulator.tsrd.tsrd_metrics import TSRDMetricsEngine
from vyapti_simulator.core.episode import run_paired_episodes

from vyapti_hybrid_scheduler import HybridMetaScheduler, BAND_COUNT, HORIZON

PARAMS_FILE = Path("base_expert_regret_analysis_V3.json")

def main():
    with PARAMS_FILE.open("r", encoding="utf-8") as f:
        document = json.load(f)
    expert_params = document.get("experts", document)

    sim_config = SimulationConfig(
        receiver_ibw_mhz=500.0,
        total_spectrum_mhz=18000.0,
        band_count=BAND_COUNT,
        dwell_time_ms=50.0,
        retune_time_ms=1.0,
        max_emitters=35,
        time_slots=HORIZON,
        detection_probability=0.9,
        false_alarm_probability=0.01,
    )
    metrics_config = MetricsConfig()

    # Replace these with your actual local TSRD stare/scan test files.
    stare_file = "path/to/your/test_stare/config_0.h5"
    scan_file = "path/to/your/test_scan/config_0.h5"

    env = TSRDStareEnvironment.from_stare_mode(
        stare_file=str(stare_file),
        scan_file=str(scan_file),
        sim_config=sim_config,
    )

    scheduler = HybridMetaScheduler(
        band_count=BAND_COUNT,
        expert_params=expert_params,
        horizon=HORIZON,
    )

    run = run_paired_episodes(
        env=env,
        schedulers={"Hybrid-Meta": scheduler},
        seed=42,
    )

    trajectory = run["Hybrid-Meta"].trajectory

    engine = TSRDMetricsEngine(
        config=metrics_config,
        stare_mode_file=str(stare_file),
        scan_mode_file=str(scan_file),
    )

    metrics = engine.record_result(
        episode_id=0,
        seed=42,
        scheduler_name="Hybrid-Meta",
        scenario_config={
            "band_count": BAND_COUNT,
            "time_slots": HORIZON,
            "algorithm": "Hybrid Meta-UCB Scheduler",
        },
        trajectory=trajectory,
        truth_grid=env.hidden_truth,
    )

    print("Episode 0 metrics (Hybrid Meta-UCB):")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
```

Adjust `stare_file` and `scan_file` to point to your local TSRD test files.

## Required JSON file

The only JSON file you must copy is:

- `base_expert_regret_analysis_V3.json`

This contains the empirical `C(alpha)` values for UCB1, Whittle‑Inspired, RLessUCB, and ARP. The hybrid combiner uses the ARP entry as a conservative proxy for the new `Hybrid-Context` expert.

No other JSON files are required to run the hybrid scheduler locally. Your existing TSRD stare/scan HDF5 files are the only other data dependency.

## Notes

- Keep `vyapti_meta_ucb_combiner_frozen.py` unchanged; it is the frozen baseline.
- `vyapti_hybrid_meta_ucb_combiner.py` and `vyapti_hybrid_scheduler.py` are your new hybrid implementation.
- This local setup is intended for debugging and smaller experiments. For the full 250‑episode benchmark, you will still need the TSRD test split and an evaluation loop similar to your Kaggle Cell 4.
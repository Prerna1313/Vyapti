# Vyapti Simulator - TSRD Metrics Usage Guide

This guide explains exactly how to import, configure, and call the new Comprehensive TSRD Metrics Engine in your Kaggle notebooks or local experiments.

## Overview of New Features
We have added a completely new metrics engine (TSRDMetricsEngine) that calculates 34 new metrics including:
* **True Pd / Pfa** (Conditional on the scheduler's actual visited dwells)
* **Waste Rate & Dwell Efficiency**
* **Threat Assessment** (High/Medium/Low based on PRI, Frequency, PW)
* **Spatial Analysis** (AoA clustering, Range statistics)
* **Spectral Congestion** (Occupancy %)

---

## Method 1: The "Easy Way" (Recommended)
If you are using the standard ExperimentRunner, everything is automated. You do not need to manually instantiate the TSRD environment or metrics engine.

### Imports Needed:
`python
from vyapti_simulator.algorithms.base import BaseScheduler
from vyapti_simulator.core.metrics import MetricsConfig
from vyapti_simulator.experiments.experiment_runner import ExperimentConfig, ExperimentRunner
`

### Usage Code:
`python
# 1. Create your configs (Defaults are now True for all comprehensive metrics)
metrics_config = MetricsConfig()
exp_config = ExperimentConfig(band_count=36, time_slots=600)

# 2. Instantiate the Runner
runner = ExperimentRunner(config=exp_config)

# 3. Define your schedulers
schedulers = {
    "MyCustomScheduler": lambda: MyScheduler(36)
}

# 4. Run the comparison!
# Passing use_tsr=True automatically loads TSRDStareEnvironment and TSRDMetricsEngine
report = runner.run_paired_comparison(
    schedulers=schedulers,
    density=15,
    use_tsr=True, 
    tsrd_scenario="train_stare/config_0"
)

# 5. The output dictionary will now contain your comprehensive metrics!
print(report["MyCustomScheduler"])
`

---

## Method 2: The "Manual Way" (Low-Level Control)
If you are writing a custom training loop and want to manually pair a scheduler to the environment and extract metrics directly.

### Imports Needed:
`python
from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.core.metrics import MetricsConfig
from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment
from vyapti_simulator.tsrd.tsrd_metrics import TSRDMetricsEngine
from vyapti_simulator.core.episode import run_paired_episodes
`

### Usage Code:
`python
STARE_PATH = "/kaggle/input/turing-synthetic-radar-dataset/train_stare/config_0.h5"
SCAN_PATH = "/kaggle/input/turing-synthetic-radar-dataset/train_scan/config_0.h5"

# 1. Initialize configurations
sim_config = SimulationConfig(band_count=36, time_slots=600)
metrics_config = MetricsConfig()

# 2. Initialize the TSRD Environment
env = TSRDStareEnvironment.from_stare_mode(
    stare_file=STARE_PATH,
    scan_file=SCAN_PATH,
    sim_config=sim_config
)

# 3. Initialize the Metrics Engine
metrics_engine = TSRDMetricsEngine(
    config=metrics_config,
    stare_mode_file=STARE_PATH,
    scan_mode_file=SCAN_PATH
)

# 4. Run your episode
schedulers = {"MyScheduler": MyScheduler(band_count=36)}
episode_results = run_paired_episodes(env, schedulers, seed=42)

# 5. Extract the metrics by passing your scheduler's trajectory to the engine
final_metrics = metrics_engine.record_result(
    episode_id=0,
    seed=42,
    scheduler_name="MyScheduler",
    scenario_config={},
    trajectory=episode_results["MyScheduler"].trajectory,
    truth_grid=env.hidden_truth
)

# 6. Print the results!
import json
print(json.dumps(final_metrics["threat_assessment"], indent=2))
print(json.dumps(final_metrics["spatial_analysis"], indent=2))
print(json.dumps(final_metrics["comprehensive_detection"], indent=2))
`

---

## Method 3: Direct Data Extraction
If you simply want to extract Numpy arrays or Metadata dictionaries from the .h5 files without running the simulator.

### Imports Needed:
`python
from vyapti_simulator.tsrd.tsrd_adapter import load_stare_mode_as_occupancy_grid, extract_emitter_metadata
`

### Usage Code:
`python
# Returns a 2D boolean grid (36 bands x 600 slots) of physical truth
occupancy_grid = load_stare_mode_as_occupancy_grid("path/to/stare.h5")

# Returns a dictionary of PRIs, Frequencies, Types, Pulse Widths
metadata = extract_emitter_metadata("path/to/stare.h5")
`

---

## Metric Output Dictionary Keys
When you output the metrics, look for these specific keys in the returned JSON/Dict:
* comprehensive_detection (true_pd, true_pfa, f1_score)
* comprehensive_coverage (occupancy_rate, time_utilization)
* comprehensive_scheduler (observed_hit_rate, exploration_rate)
* comprehensive_efficiency (dwell_efficiency, waste_rate)
* comprehensive_latency
* per_band_metrics
* 	emporal_metrics (learning curves)
* 	hreat_assessment (threat distribution, classification)
* spatial_analysis (AoA clusters, range stats)
* spectral_environment (congestion)

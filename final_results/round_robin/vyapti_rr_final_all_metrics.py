#!/usr/bin/env python3
"""
Vyapti - Round Robin Final (Save ALL 34+ Metrics)
"""

import numpy as np
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from datetime import datetime

from vyapti_simulator.core.scheduler_interface import BaseScheduler
from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.core.metrics import MetricsConfig
from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment
from vyapti_simulator.core.episode import run_paired_episodes
from vyapti_simulator.tsrd.tsrd_metrics import TSRDMetricsEngine

# =============================================================================
# ROUND ROBIN SCHEDULER
# =============================================================================
class RoundRobinScheduler(BaseScheduler):
    def __init__(self, band_count: int):
        super().__init__(band_count, provenance_note="Round Robin Baseline")
        self.t = 0
        self.band_count = band_count
    
    def reset(self, seed: int, scenario_config=None):
        self.t = 0
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        band = self.t % self.band_count
        self.t += 1
        return band
    
    def update(self, action: int, observation: dict):
        pass

# =============================================================================
# SETUP
# =============================================================================
print("="*80)
print("VYAPTI - ROUND ROBIN (250 EPISODES, ALL METRICS)")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

snapshot_path = snapshot_download(
    repo_id="alan-turing-institute/turing-synthetic-radar-dataset",
    repo_type="dataset",
    revision="68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51",
    cache_dir="/root/.cache/huggingface/hub",
)

TSRD_ROOT = Path(snapshot_path)
TEST_DIR = TSRD_ROOT / "stare" / "test_stare"
test_files = sorted(TEST_DIR.glob("config_*.h5"))

print(f"TSRD test files: {len(test_files)}")

BAND_COUNT = 36
TIME_SLOTS = 600
sim_config = SimulationConfig(band_count=BAND_COUNT, time_slots=TIME_SLOTS, dwell_time_ms=50.0)
metrics_config = MetricsConfig()

all_episodes = []
all_pd = []
all_pfa = []

# =============================================================================
# RUN ALL EPISODES
# =============================================================================
print("\n" + "="*80)
print("RUNNING EPISODES")
print("="*80)

for ep_idx, test_file in enumerate(test_files):
    config_name = test_file.stem
    scan_file = str(TSRD_ROOT / "scan" / "test_scan" / f"{config_name}.h5")
    stare_file = str(test_file)
    
    if not Path(scan_file).exists():
        continue
    
    # Initialize
    env = TSRDStareEnvironment.from_stare_mode(stare_file=stare_file, scan_file=scan_file, sim_config=sim_config)
    metrics_engine = TSRDMetricsEngine(config=metrics_config, stare_mode_file=stare_file, scan_mode_file=scan_file)
    
    # Run
    schedulers = {"RR": RoundRobinScheduler(band_count=BAND_COUNT)}
    results = run_paired_episodes(env=env, schedulers=schedulers, seed=ep_idx)
    trajectory = results["RR"].trajectory
    
    # Compute ALL 34+ metrics
    metrics = metrics_engine.record_result(
        episode_id=ep_idx, seed=ep_idx, scheduler_name="RR",
        scenario_config={}, trajectory=trajectory, truth_grid=env.hidden_truth
    )
    
    # Store episode with ALL metrics
    all_episodes.append({
        'episode': ep_idx,
        'config': config_name,
        'seed': ep_idx,
        'all_metrics': metrics  # Complete 34+ metrics dict
    })
    
    # Track Pd, Pfa for aggregate
    all_pd.append(metrics['comprehensive_detection']['true_pd'])
    all_pfa.append(metrics['comprehensive_detection']['true_pfa'])
    
    # Print progress every 10
    if (ep_idx + 1) % 10 == 0 or ep_idx == len(test_files) - 1:
        pd = metrics['comprehensive_detection']['true_pd']
        pfa = metrics['comprehensive_detection']['true_pfa']
        hits = metrics['comprehensive_detection']['true_hits']
        print(f"Episode {ep_idx+1:3d} ({config_name}): Pd={pd:.6f}, Pfa={pfa:.6f}, Hits={hits}")

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================
print("\n" + "="*80)
print("AGGREGATE RESULTS")
print("="*80)

n = len(all_pd)
pd_mean = np.mean(all_pd)
pd_std = np.std(all_pd)
pd_95ci = 1.96 * pd_std / np.sqrt(n)

pfa_mean = np.mean(all_pfa)
pfa_std = np.std(all_pfa)

print(f"\nEpisodes: {n}")
print(f"\n📊 DETECTION:")
print(f"  Pd:  {pd_mean:.6f} ± {pd_std:.6f}")
print(f"  Pfa: {pfa_mean:.6f} ± {pfa_std:.6f}")
print(f"\n📊 95% CI:")
print(f"  Pd: [{pd_mean - pd_95ci:.6f}, {pd_mean + pd_95ci:.6f}]")

# =============================================================================
# SAVE ALL METRICS
# =============================================================================
output_dir = Path("vyapti_results")
output_dir.mkdir(exist_ok=True)

# 1. Save ALL 34+ metrics for all 250 episodes
all_file = output_dir / "rr_all_episodes_all_metrics.json"
with open(all_file, 'w') as f:
    json.dump(all_episodes, f, indent=2, default=str)
print(f"\n✓ Saved ALL metrics: {all_file}")
print(f"  File size: {all_file.stat().st_size / 1024 / 1024:.1f} MB")

# 2. Save aggregate stats
aggregate = {
    'scheduler': 'RoundRobin',
    'n_episodes': n,
    'pd_mean': float(pd_mean),
    'pd_std': float(pd_std),
    'pd_95ci_lower': float(pd_mean - pd_95ci),
    'pd_95ci_upper': float(pd_mean + pd_95ci),
    'pfa_mean': float(pfa_mean),
    'pfa_std': float(pfa_std),
    'timestamp': datetime.now().isoformat()
}

agg_file = output_dir / "rr_aggregate.json"
with open(agg_file, 'w') as f:
    json.dump(aggregate, f, indent=2)
print(f"✓ Saved aggregate: {agg_file}")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ COMPLETE - ALL 34+ METRICS SAVED")
print("="*80)
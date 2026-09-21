#!/usr/bin/env python3
"""
Vyapti - PLAIN THOMPSON SAMPLING (NOT sliding window, NOT discounted)
FULL RUN: ALL 250 TSRD TEST EPISODES

Pure Thompson Sampling:
- Beta-Bernoulli conjugate model
- Remembers ALL data (no forgetting)
- Optimal for STATIONARY bandits

Expected Pd: ~0.085 - 0.11 (based on similar algorithms)
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
# PLAIN THOMPSON SAMPLING
# =============================================================================
class ThompsonScheduler(BaseScheduler):
    """
    Plain Thompson Sampling (Beta-Bernoulli).
    
    NO sliding window, NO discounting - remembers ALL data.
    """
    def __init__(self, band_count: int):
        super().__init__(band_count, provenance_note="Thompson Sampling")
        self.band_count = band_count
        
        # Beta posterior parameters
        # alpha = successes + 1, beta = failures + 1
        self.alpha = np.ones(band_count)
        self.beta_param = np.ones(band_count)  # Renamed to avoid conflict with Beta distribution
        
        self.t = 0
        self.rng = np.random.default_rng()
    
    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(self.band_count)
        self.beta_param = np.ones(self.band_count)
        self.t = 0
    
    def get_scores(self):
        """Return posterior mean for all bands."""
        return self.alpha / (self.alpha + self.beta_param)
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        self.t = current_time_slot
        
        if self.t < self.band_count:
            return self.t % self.band_count
        
        # Thompson sampling: sample from Beta posterior
        samples = self.rng.beta(self.alpha, self.beta_param)
        
        return int(np.argmax(samples))
    
    def update(self, action: int, observation: dict):
        hit = observation.get('hit', False)
        
        # Update Beta posterior
        self.alpha[action] += hit
        self.beta_param[action] += (1 - hit)
        
        self.t += 1


# =============================================================================
# RUN - ALL 250 TSRD EPISODES
# =============================================================================
print("="*80)
print("VYAPTI - PLAIN THOMPSON SAMPLING")
print("FULL RUN: ALL 250 TSRD TEST EPISODES")
print("Pure Thompson: Beta-Bernoulli, NO sliding window, NO discounting")
print("Expected Pd: ~0.085 - 0.11")
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
sim_config = SimulationConfig(band_count=BAND_COUNT, time_slots=600, dwell_time_ms=50.0)
metrics_config = MetricsConfig()

all_results = {}

name = "Thompson_Plain"
print(f"\n{'='*80}\nRUNNING {name}\n{'='*80}")

all_pd, all_pfa = [], []
all_episodes_data = []

# ALL 250 TSRD EPISODES
for ep_idx, test_file in enumerate(test_files):
    config_name = test_file.stem
    scan_file = str(TSRD_ROOT / "scan" / "test_scan" / f"{config_name}.h5")
    stare_file = str(test_file)
    
    if not Path(scan_file).exists():
        print(f"⚠️ Missing scan file: {scan_file}")
        continue
    
    if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
        print(f"Episode {ep_idx+1}: {config_name}")
    
    env = TSRDStareEnvironment.from_stare_mode(
        stare_file=stare_file, scan_file=scan_file, sim_config=sim_config
    )
    
    metrics_engine = TSRDMetricsEngine(
        config=metrics_config,
        stare_mode_file=stare_file,
        scan_mode_file=scan_file
    )
    
    schedulers = {name: ThompsonScheduler(band_count=BAND_COUNT)}
    
    episode_results = run_paired_episodes(env=env, schedulers=schedulers, seed=ep_idx)
    
    trajectory = episode_results[name].trajectory
    
    # Saves ALL 34+ TSRD metrics
    metrics = metrics_engine.record_result(
        episode_id=ep_idx, seed=ep_idx, scheduler_name=name,
        scenario_config={}, trajectory=trajectory, truth_grid=env.hidden_truth
    )
    
    all_episodes_data.append({
        'episode': ep_idx,
        'config': config_name,
        'seed': ep_idx,
        'all_metrics': metrics  # ← ALL 34+ METRICS
    })
    
    all_pd.append(metrics['comprehensive_detection']['true_pd'])
    all_pfa.append(metrics['comprehensive_detection']['true_pfa'])
    
    if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
        pd = metrics['comprehensive_detection']['true_pd']
        hits = metrics['comprehensive_detection']['true_hits']
        print(f"  Pd={pd:.6f}, Hits={hits}")

n = len(all_pd)
if n > 0:
    pd_mean = np.mean(all_pd)
    pd_std = np.std(all_pd)
    pfa_mean = np.mean(all_pfa)
    pfa_std = np.std(all_pfa)
    
    print(f"\n📊 {name}: Pd={pd_mean:.3f} ± {pd_std:.3f}")
    
    all_results[name] = {
        'aggregate': {
            'pd_mean': float(pd_mean),
            'pd_std': float(pd_std),
            'pfa_mean': float(pfa_mean),
            'pfa_std': float(pfa_std),
            'n_episodes': n,
            'algorithm': 'Plain Thompson Sampling',
            'description': 'Beta-Bernoulli Thompson Sampling (no sliding window, no discounting)',
            'expected_pd': '~0.085 - 0.11',
            'metrics_saved': 'ALL 34+ TSRD metrics',
        },
        'episodes': all_episodes_data
    }

# =============================================================================
# SAVE
# =============================================================================
output_dir = Path("vyapti_results")
output_dir.mkdir(exist_ok=True)

all_file = output_dir / "thompson_plain_TSRD_250ep_ALL_METRICS.json"
with open(all_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n✓ Saved ALL metrics: {all_file}")
print(f"  File size: {all_file.stat().st_size / 1024 / 1024:.1f} MB")

agg_file = output_dir / "thompson_plain_TSRD_250ep_aggregate.json"
with open(agg_file, 'w') as f:
    json.dump(all_results[name]['aggregate'], f, indent=2)
print(f"✓ Saved aggregate: {agg_file}")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ COMPLETE - PLAIN THOMPSON SAMPLING (ALL 250 TSRD EPISODES, ALL 34+ METRICS)")
print("="*80)
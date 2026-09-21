#!/usr/bin/env python3
"""
Vyapti - Clarkson-Optimized Periodic Scheduler (250 EPISODES, ALL METRICS)
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
# CLARKSON-OPTIMIZED PERIODIC SCHEDULER
# =============================================================================
class ClarksonScheduler(BaseScheduler):
    """
    Clarkson's algorithm for periodic emitter interception.
    
    Key ideas:
    1. Track hits per band to estimate occupancy probability
    2. Prioritize bands with higher hit rates
    3. Revisit bands at intervals matching emitter PRI
    4. Balance exploration vs exploitation
    """
    def __init__(self, band_count: int, dwell_time_ms: float = 50.0):
        super().__init__(band_count, provenance_note="Clarkson-Optimized Periodic")
        self.band_count = band_count
        self.dwell_time_ms = dwell_time_ms
        
        # Per-band statistics
        self.visits = np.zeros(band_count, dtype=int)
        self.hits = np.zeros(band_count, dtype=int)
        
        # Estimated occupancy probability per band
        self.occupancy_prob = np.ones(band_count) / band_count
        
        # Last visit time per band (for periodic revisits)
        self.last_visit = np.zeros(band_count, dtype=int)
        
        # Estimated PRI per band (in time slots)
        self.estimated_pri = np.ones(band_count) * 10  # Default 10 slots
        
        self.t = 0
        self.epsilon = 0.1  # Exploration rate (10% random)
    
    def reset(self, seed: int, scenario_config=None):
        self.visits = np.zeros(self.band_count, dtype=int)
        self.hits = np.zeros(self.band_count, dtype=int)
        self.occupancy_prob = np.ones(self.band_count) / self.band_count
        self.last_visit = np.zeros(self.band_count, dtype=int)
        self.estimated_pri = np.ones(self.band_count) * 10
        self.t = 0
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            band = np.random.randint(0, self.band_count)
        else:
            # Score each band: occupancy_prob * recency_bonus
            scores = self.occupancy_prob.copy()
            
            # Recency bonus: prefer bands not visited recently
            time_since_visit = self.t - self.last_visit
            recency_bonus = np.clip(time_since_visit / self.estimated_pri, 0, 2)
            scores *= (1 + recency_bonus)
            
            # Add small noise to break ties
            scores += np.random.uniform(0, 0.01, self.band_count)
            
            # Select band with highest score
            band = np.argmax(scores)
        
        return band
    
    def update(self, action: int, observation: dict):
        self.visits[action] += 1
        
        # Update hits if detection occurred
        if observation.get('hit', False):
            self.hits[action] += 1
        
        # Update occupancy probability (Laplace smoothing)
        self.occupancy_prob = (self.hits + 1) / (self.visits + 2)
        
        # Update last visit time
        self.last_visit[action] = self.t
        self.t += 1

# =============================================================================
# SETUP
# =============================================================================
print("="*80)
print("VYAPTI - CLARKSON SCHEDULER (250 EPISODES, ALL METRICS)")
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
    schedulers = {"Clarkson": ClarksonScheduler(band_count=BAND_COUNT)}
    results = run_paired_episodes(env=env, schedulers=schedulers, seed=ep_idx)
    trajectory = results["Clarkson"].trajectory
    
    # Compute ALL 34+ metrics
    metrics = metrics_engine.record_result(
        episode_id=ep_idx, seed=ep_idx, scheduler_name="Clarkson",
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
all_file = output_dir / "clarkson_all_episodes_all_metrics.json"
with open(all_file, 'w') as f:
    json.dump(all_episodes, f, indent=2, default=str)
print(f"\n✓ Saved ALL metrics: {all_file}")
print(f"  File size: {all_file.stat().st_size / 1024 / 1024:.1f} MB")

# 2. Save aggregate stats
aggregate = {
    'scheduler': 'Clarkson',
    'n_episodes': n,
    'pd_mean': float(pd_mean),
    'pd_std': float(pd_std),
    'pd_95ci_lower': float(pd_mean - pd_95ci),
    'pd_95ci_upper': float(pd_mean + pd_95ci),
    'pfa_mean': float(pfa_mean),
    'pfa_std': float(pfa_std),
    'timestamp': datetime.now().isoformat()
}

agg_file = output_dir / "clarkson_aggregate.json"
with open(agg_file, 'w') as f:
    json.dump(aggregate, f, indent=2)
print(f"✓ Saved aggregate: {agg_file}")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ COMPLETE - ALL 34+ METRICS SAVED")
print("="*80)
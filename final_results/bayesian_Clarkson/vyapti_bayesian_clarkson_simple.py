#!/usr/bin/env python3
"""
Vyapti - Bayesian + Clarkson (SIMPLE - No PRI learning)
Just occupancy estimation + smart revisiting, no complex PRI
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
# BAYESIAN + CLARKSON SIMPLE
# =============================================================================
class BayesianClarksonScheduler(BaseScheduler):
    """
    Simple Bayesian bandit with Clarkson-style revisiting:
    1. Thompson sampling for occupancy estimation
    2. Recency bonus (time since last visit)
    3. NO PRI learning (too noisy in sparse TSRD)
    4. Moderate exploration
    """
    def __init__(self, band_count: int, dwell_time_ms: float = 50.0):
        super().__init__(band_count, provenance_note="Bayesian Clarkson - Simple occupancy + recency")
        self.band_count = band_count
        self.dwell_time_ms = dwell_time_ms
        
        # Weak informative prior (assume ~1-2% base occupancy)
        self.alpha = np.ones(band_count) * 1.5
        self.beta_param = np.ones(band_count) * 50.0
        
        # Visit tracking (for recency bonus)
        self.last_visit = np.full(band_count, -1, dtype=int)
        self.visit_count = np.zeros(band_count, dtype=int)
        self.hit_count = np.zeros(band_count, dtype=int)
        
        # Moderate exploration
        self.epsilon = 0.15  # 15% exploration throughout
        
        self.t = 0
    
    def reset(self, seed: int, scenario_config=None):
        np.random.seed(seed)
        self.alpha = np.ones(self.band_count) * 1.5
        self.beta_param = np.ones(self.band_count) * 50.0
        self.last_visit = np.full(self.band_count, -1, dtype=int)
        self.visit_count = np.zeros(self.band_count, dtype=int)
        self.hit_count = np.zeros(self.band_count, dtype=int)
        self.epsilon = 0.15
        self.t = 0
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        # Early episode: force exploration
        if self.t < self.band_count:
            return self.t % self.band_count
        
        # Thompson sampling with recency bonus
        if np.random.random() < self.epsilon:
            # Random exploration
            band = np.random.randint(0, self.band_count)
        else:
            # Thompson sample from posterior
            samples = np.random.beta(
                self.alpha + self.hit_count,
                self.beta_param + self.visit_count - self.hit_count
            )
            
            # Recency bonus: higher for bands not visited recently
            time_since_visit = self.t - self.last_visit
            time_since_visit = np.maximum(time_since_visit, 1)
            
            # Simple recency: linear bonus up to 20 steps
            recency_bonus = np.clip(time_since_visit / 20.0, 0, 1.5)
            
            # Combined score
            scores = samples * (1.0 + recency_bonus)
            
            band = np.argmax(scores)
        
        return band
    
    def update(self, action: int, observation: dict):
        hit = observation.get('hit', False)
        
        # Update counts
        self.visit_count[action] += 1
        if hit:
            self.hit_count[action] += 1
        
        # Update last visit
        self.last_visit[action] = self.t
        self.t += 1

# =============================================================================
# SETUP
# =============================================================================
print("="*80)
print("VYAPTI - BAYESIAN CLARKSON (SIMPLE)")
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
test_files = list(TEST_DIR.glob("config_*.h5"))

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
    schedulers = {"BayesianClarkson": BayesianClarksonScheduler(band_count=BAND_COUNT)}
    results = run_paired_episodes(env=env, schedulers=schedulers, seed=ep_idx)
    trajectory = results["BayesianClarkson"].trajectory
    
    # Compute metrics
    metrics = metrics_engine.record_result(
        episode_id=ep_idx, seed=ep_idx, scheduler_name="BayesianClarkson",
        scenario_config={}, trajectory=trajectory, truth_grid=env.hidden_truth
    )
    
    all_episodes.append({
        'episode': ep_idx,
        'config': config_name,
        'seed': ep_idx,
        'all_metrics': metrics
    })
    
    all_pd.append(metrics['comprehensive_detection']['true_pd'])
    all_pfa.append(metrics['comprehensive_detection']['true_pfa'])
    
    # Print progress
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

# Compare
print(f"\n📊 COMPARISON:")
print(f"  RR Pd:              0.085 ± 0.026")
print(f"  Clarkson V1 Pd:     0.069 ± 0.025")
print(f"  BayesianClarkson Pd: {pd_mean:.3f} ± {pd_std:.3f}")

if pd_mean > 0.085:
    improvement = (pd_mean - 0.085) / 0.085 * 100
    print(f"  ✓ IMPROVEMENT: +{improvement:.1f}% over RR")
else:
    degradation = (0.085 - pd_mean) / 0.085 * 100
    print(f"  ✗ Still {degradation:.1f}% worse than RR")

# =============================================================================
# SAVE
# =============================================================================
output_dir = Path("vyapti_results")
output_dir.mkdir(exist_ok=True)

all_file = output_dir / "bayesian_clarkson_all_episodes_all_metrics.json"
with open(all_file, 'w') as f:
    json.dump(all_episodes, f, indent=2, default=str)
print(f"\n✓ Saved ALL metrics: {all_file}")

aggregate = {
    'scheduler': 'BayesianClarkson',
    'n_episodes': n,
    'pd_mean': float(pd_mean),
    'pd_std': float(pd_std),
    'pfa_mean': float(pfa_mean),
    'pfa_std': float(pfa_std),
    'timestamp': datetime.now().isoformat(),
}

agg_file = output_dir / "bayesian_clarkson_aggregate.json"
with open(agg_file, 'w') as f:
    json.dump(aggregate, f, indent=2)
print(f"✓ Saved aggregate: {agg_file}")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ COMPLETE - BAYESIAN CLARKSON")
print("="*80)
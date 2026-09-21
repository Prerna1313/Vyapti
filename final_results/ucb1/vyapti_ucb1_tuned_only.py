#!/usr/bin/env python3
"""
Vyapti - UCB1-Tuned ONLY (Fast test)
Variance-aware UCB1 - should beat classic UCB1
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
# UCB1-TUNED
# =============================================================================
class UCB1TunedScheduler(BaseScheduler):
    """
    UCB1-Tuned: variance-aware confidence bound
    Generally beats classic UCB1 in practice
    """
    def __init__(self, band_count: int):
        super().__init__(band_count, provenance_note="UCB1-Tuned")
        self.band_count = band_count
        self.hit_count = np.zeros(band_count, dtype=float)
        self.visit_count = np.zeros(band_count, dtype=float)
        self.t = 0
    
    def reset(self, seed: int, scenario_config=None):
        np.random.seed(seed)
        self.hit_count = np.zeros(self.band_count, dtype=float)
        self.visit_count = np.zeros(self.band_count, dtype=float)
        self.t = 0
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        # Force visit all bands once first
        unvisited = np.where(self.visit_count == 0)[0]
        if len(unvisited) > 0:
            return unvisited[0]
        
        n = self.visit_count
        mean = self.hit_count / n
        
        # Bernoulli variance: p(1-p)
        var = mean * (1.0 - mean)
        
        # Variance-aware bonus
        log_term = np.log(self.t + 1) / n
        variance_bonus = np.sqrt(log_term * np.minimum(0.25, var + np.sqrt(2 * log_term)))
        scores = mean + variance_bonus
        
        return int(np.argmax(scores))
    
    def update(self, action: int, observation: dict):
        hit = observation.get('hit', False)
        self.visit_count[action] += 1
        if hit:
            self.hit_count[action] += 1
        self.t += 1

# =============================================================================
# RUN
# =============================================================================
print("="*80)
print("VYAPTI - UCB1-TUNED")
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

all_pd = []
all_pfa = []
all_episodes = []

print("\n" + "="*80)
print("RUNNING EPISODES")
print("="*80)

for ep_idx, test_file in enumerate(test_files):
    config_name = test_file.stem
    scan_file = str(TSRD_ROOT / "scan" / "test_scan" / f"{config_name}.h5")
    stare_file = str(test_file)
    
    if not Path(scan_file).exists():
        continue
    
    env = TSRDStareEnvironment.from_stare_mode(
        stare_file=stare_file, scan_file=scan_file, sim_config=sim_config
    )
    metrics_engine = TSRDMetricsEngine(
        config=metrics_config, stare_mode_file=stare_file, scan_mode_file=scan_file
    )
    
    schedulers = {"UCB1_Tuned": UCB1TunedScheduler(band_count=BAND_COUNT)}
    results = run_paired_episodes(env=env, schedulers=schedulers, seed=ep_idx)
    trajectory = results["UCB1_Tuned"].trajectory
    
    metrics = metrics_engine.record_result(
        episode_id=ep_idx, seed=ep_idx, scheduler_name="UCB1_Tuned",
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
    
    if (ep_idx + 1) % 10 == 0 or ep_idx == len(test_files) - 1:
        pd = metrics['comprehensive_detection']['true_pd']
        pfa = metrics['comprehensive_detection']['true_pfa']
        hits = metrics['comprehensive_detection']['true_hits']
        print(f"Episode {ep_idx+1:3d} ({config_name}): Pd={pd:.6f}, Pfa={pfa:.6f}, Hits={hits}")

# =============================================================================
# AGGREGATE
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
print(f"  RR Pd:         0.085 ± 0.026")
print(f"  Clarkson V1 Pd: 0.069 ± 0.025")
print(f"  UCB1-Tuned Pd:  {pd_mean:.3f} ± {pd_std:.3f}")

if pd_mean > 0.085:
    improvement = (pd_mean - 0.085) / 0.085 * 100
    print(f"  ✓ IMPROVEMENT: +{improvement:.1f}% over RR")
else:
    degradation = (0.085 - pd_mean) / 0.085 * 100
    print(f"  ✗ {degradation:.1f}% worse than RR")

# =============================================================================
# SAVE
# =============================================================================
output_dir = Path("vyapti_results")
output_dir.mkdir(exist_ok=True)

all_file = output_dir / "ucb1_tuned_all_episodes_all_metrics.json"
with open(all_file, 'w') as f:
    json.dump(all_episodes, f, indent=2, default=str)
print(f"\n✓ Saved ALL metrics: {all_file}")

aggregate = {
    'scheduler': 'UCB1-Tuned',
    'n_episodes': n,
    'pd_mean': float(pd_mean),
    'pd_std': float(pd_std),
    'pfa_mean': float(pfa_mean),
    'pfa_std': float(pfa_std),
    'timestamp': datetime.now().isoformat(),
}

agg_file = output_dir / "ucb1_tuned_aggregate.json"
with open(agg_file, 'w') as f:
    json.dump(aggregate, f, indent=2)
print(f"✓ Saved aggregate: {agg_file}")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ COMPLETE - UCB1-TUNED")
print("="*80)
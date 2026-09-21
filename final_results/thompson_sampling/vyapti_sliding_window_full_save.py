#!/usr/bin/env python3
"""
Vyapti - Sliding-Window / Discounted TS (FULL METRICS SAVE)
"""

import numpy as np
import json
from pathlib import Path
from collections import deque
from huggingface_hub import snapshot_download
from datetime import datetime

from vyapti_simulator.core.scheduler_interface import BaseScheduler
from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.core.metrics import MetricsConfig
from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment
from vyapti_simulator.core.episode import run_paired_episodes
from vyapti_simulator.tsrd.tsrd_metrics import TSRDMetricsEngine

# =============================================================================
# SLIDING-WINDOW TS (window_size=12)
# =============================================================================
class SlidingWindowTS_Scheduler(BaseScheduler):
    def __init__(self, band_count: int, window_size: int = 12):
        super().__init__(band_count, provenance_note=f"Sliding-Window TS (W={window_size})")
        self.band_count = band_count
        self.window_size = window_size
        self.history = [deque(maxlen=window_size) for _ in range(band_count)]
        self.hit_count = np.zeros(band_count, dtype=float)
        self.visit_count = np.zeros(band_count, dtype=float)
        self.t = 0
    
    def reset(self, seed: int, scenario_config=None):
        np.random.seed(seed)
        self.history = [deque(maxlen=self.window_size) for _ in range(self.band_count)]
        self.hit_count = np.zeros(self.band_count, dtype=float)
        self.visit_count = np.zeros(self.band_count, dtype=float)
        self.t = 0
    
    def _update_counts(self):
        for band in range(self.band_count):
            hits = sum(1 for (hit, _) in self.history[band] if hit)
            visits = len(self.history[band])
            self.hit_count[band] = hits
            self.visit_count[band] = visits
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        self._update_counts()
        
        if self.t < self.band_count:
            return self.t % self.band_count
        
        alpha = 1.0 + self.hit_count
        beta = 1.0 + self.visit_count - self.hit_count
        samples = np.random.beta(alpha, beta)
        
        return int(np.argmax(samples))
    
    def update(self, action: int, observation: dict):
        hit = observation.get('hit', False)
        self.history[action].append((hit, self.t))
        self.t += 1

# =============================================================================
# DISCOUNTED TS (gamma=0.99)
# =============================================================================
class DiscountedTS_Scheduler(BaseScheduler):
    def __init__(self, band_count: int, gamma: float = 0.99):
        super().__init__(band_count, provenance_note=f"Discounted TS (γ={gamma})")
        self.band_count = band_count
        self.gamma = gamma
        self.hit_count = np.zeros(band_count, dtype=float)
        self.effective_visits = np.zeros(band_count, dtype=float)
        self.t = 0
    
    def reset(self, seed: int, scenario_config=None):
        np.random.seed(seed)
        self.hit_count = np.zeros(self.band_count, dtype=float)
        self.effective_visits = np.zeros(self.band_count, dtype=float)
        self.t = 0
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        if self.t < self.band_count:
            return self.t % self.band_count
        
        alpha = 1.0 + self.hit_count
        beta = 1.0 + np.maximum(self.effective_visits - self.hit_count, 0.01)
        samples = np.random.beta(alpha, beta)
        
        return int(np.argmax(samples))
    
    def update(self, action: int, observation: dict):
        hit = observation.get('hit', False)
        
        self.hit_count *= self.gamma
        self.effective_visits *= self.gamma
        
        self.effective_visits[action] += 1.0
        if hit:
            self.hit_count[action] += 1.0
        
        self.t += 1

# =============================================================================
# RUN
# =============================================================================
print("="*80)
print("VYAPTI - SLIDING-WINDOW / DISCOUNTED TS (FULL SAVE)")
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
sim_config = SimulationConfig(band_count=BAND_COUNT, time_slots=600, dwell_time_ms=50.0)
metrics_config = MetricsConfig()

# =============================================================================
# RUN BOTH - SAVE ALL METRICS
# =============================================================================
all_results = {}  # Will store ALL metrics for all episodes

for name, Scheduler in [
    ("SlidingWindow_TS_W12", SlidingWindowTS_Scheduler),
    ("Discounted_TS_G99", DiscountedTS_Scheduler)
]:
    print(f"\n{'='*80}")
    print(f"RUNNING {name}")
    print(f"{'='*80}")
    
    all_pd, all_pfa = [], []
    all_episodes_data = []  # Store ALL metrics per episode
    
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
        
        schedulers = {name: Scheduler(band_count=BAND_COUNT)}
        episode_results = run_paired_episodes(env=env, schedulers=schedulers, seed=ep_idx)
        trajectory = episode_results[name].trajectory
        
        # Get ALL 34+ metrics
        metrics = metrics_engine.record_result(
            episode_id=ep_idx, seed=ep_idx, scheduler_name=name,
            scenario_config={}, trajectory=trajectory, truth_grid=env.hidden_truth
        )
        
        # Store ALL metrics for this episode
        all_episodes_data.append({
            'episode': ep_idx,
            'config': config_name,
            'seed': ep_idx,
            'all_metrics': metrics  # ← ALL 34+ metrics!
        })
        
        all_pd.append(metrics['comprehensive_detection']['true_pd'])
        all_pfa.append(metrics['comprehensive_detection']['true_pfa'])
        
        if (ep_idx + 1) % 10 == 0 or ep_idx == len(test_files) - 1:
            pd = metrics['comprehensive_detection']['true_pd']
            pfa = metrics['comprehensive_detection']['true_pfa']
            hits = metrics['comprehensive_detection']['true_hits']
            print(f"Episode {ep_idx+1:3d} ({config_name}): Pd={pd:.6f}, Pfa={pfa:.6f}, Hits={hits}")
    
    # Aggregate
    n = len(all_pd)
    pd_mean = np.mean(all_pd)
    pd_std = np.std(all_pd)
    pfa_mean = np.mean(all_pfa)
    pfa_std = np.std(all_pfa)
    
    print(f"\n📊 {name}:")
    print(f"  Pd:  {pd_mean:.6f} ± {pd_std:.6f}")
    print(f"  Pfa: {pfa_mean:.6f} ± {pfa_std:.6f}")
    
    # Store for this scheduler
    all_results[name] = {
        'aggregate': {
            'pd_mean': float(pd_mean),
            'pd_std': float(pd_std),
            'pfa_mean': float(pfa_mean),
            'pfa_std': float(pfa_std),
            'n_episodes': n
        },
        'episodes': all_episodes_data  # ← ALL episodes with ALL metrics!
    }

# =============================================================================
# COMPARISON
# =============================================================================
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\n{'Scheduler':<25} {'Pd':<18}")
print(f"{'-'*25} {'-'*18}")
print(f"{'Round Robin':<25} 0.085 ± 0.026")
print(f"{'UCB1':<25} 0.150 ± 0.051  ← BEST")
print(f"{'SlidingWindow_TS_W12':<25} {all_results['SlidingWindow_TS_W12']['aggregate']['pd_mean']:.3f} ± {all_results['SlidingWindow_TS_W12']['aggregate']['pd_std']:.3f}")
print(f"{'Discounted_TS_G99':<25} {all_results['Discounted_TS_G99']['aggregate']['pd_mean']:.3f} ± {all_results['Discounted_TS_G99']['aggregate']['pd_std']:.3f}")

# =============================================================================
# SAVE ALL METRICS
# =============================================================================
output_dir = Path("vyapti_results")
output_dir.mkdir(exist_ok=True)

# Save ALL metrics for ALL episodes
all_file = output_dir / "sliding_window_discounted_ALL_METRICS.json"
with open(all_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n✓ Saved ALL metrics: {all_file}")
print(f"  File size: {all_file.stat().st_size / 1024 / 1024:.1f} MB")

# Save just aggregate (for quick reference)
agg_file = output_dir / "sliding_window_discounted_aggregate.json"
aggregate_only = {
    name: data['aggregate'] for name, data in all_results.items()
}
with open(agg_file, 'w') as f:
    json.dump(aggregate_only, f, indent=2)
print(f"✓ Saved aggregate: {agg_file}")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ COMPLETE - FULL SAVE")
print("="*80)
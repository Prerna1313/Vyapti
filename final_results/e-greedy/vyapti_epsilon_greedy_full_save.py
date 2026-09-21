#!/usr/bin/env python3
"""
Vyapti - Epsilon-Greedy (FULL METRICS SAVE)
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
# EPSILON-GREEDY
# =============================================================================
class EpsilonGreedyScheduler(BaseScheduler):
    def __init__(self, band_count: int, epsilon: float = 0.1):
        super().__init__(band_count, provenance_note=f"Eps-Greedy (ε={epsilon})")
        self.band_count = band_count
        self.epsilon = epsilon
        self.hit_count = np.zeros(band_count, dtype=float)
        self.visit_count = np.zeros(band_count, dtype=float)
        self.t = 0
        self.rng = np.random.RandomState()
    
    def reset(self, seed: int, scenario_config=None):
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        self.hit_count = np.zeros(self.band_count, dtype=float)
        self.visit_count = np.zeros(self.band_count, dtype=float)
        self.t = 0
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        unvisited = np.where(self.visit_count == 0)[0]
        if len(unvisited) > 0:
            return unvisited[0]
        
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.band_count)
        else:
            hit_rates = self.hit_count / np.maximum(self.visit_count, 1)
            return int(np.argmax(hit_rates))
    
    def update(self, action: int, observation: dict):
        hit = observation.get('hit', False)
        self.visit_count[action] += 1
        if hit:
            self.hit_count[action] += 1
        self.t += 1

# =============================================================================
# EPSILON-GREEDY DECAY
# =============================================================================
class EpsilonGreedyDecayScheduler(BaseScheduler):
    def __init__(self, band_count: int, eps_start: float = 0.5, decay: float = 0.99, eps_floor: float = 0.01):
        super().__init__(band_count, provenance_note=f"Eps-Decay ({eps_start}→{eps_floor})")
        self.band_count = band_count
        self.eps_start = eps_start
        self.decay = decay
        self.eps_floor = eps_floor
        self.epsilon = eps_start
        self.hit_count = np.zeros(band_count, dtype=float)
        self.visit_count = np.zeros(band_count, dtype=float)
        self.t = 0
        self.rng = np.random.RandomState()
    
    def reset(self, seed: int, scenario_config=None):
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        self.hit_count = np.zeros(self.band_count, dtype=float)
        self.visit_count = np.zeros(self.band_count, dtype=float)
        self.epsilon = self.eps_start
        self.t = 0
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        unvisited = np.where(self.visit_count == 0)[0]
        if len(unvisited) > 0:
            return unvisited[0]
        
        self.epsilon = max(self.eps_floor, self.eps_start * (self.decay ** self.t))
        
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.band_count)
        else:
            hit_rates = self.hit_count / np.maximum(self.visit_count, 1)
            return int(np.argmax(hit_rates))
    
    def update(self, action: int, observation: dict):
        hit = observation.get('hit', False)
        self.visit_count[action] += 1
        if hit:
            self.hit_count[action] += 1
        self.t += 1

# =============================================================================
# EPSILON-GREEDY SLOW DECAY
# =============================================================================
class EpsilonGreedySlowDecayScheduler(BaseScheduler):
    def __init__(self, band_count: int, eps_start: float = 0.3, decay: float = 0.995, eps_floor: float = 0.05):
        super().__init__(band_count, provenance_note=f"Eps-SlowDecay ({eps_start}→{eps_floor})")
        self.band_count = band_count
        self.eps_start = eps_start
        self.decay = decay
        self.eps_floor = eps_floor
        self.epsilon = eps_start
        self.hit_count = np.zeros(band_count, dtype=float)
        self.visit_count = np.zeros(band_count, dtype=float)
        self.t = 0
        self.rng = np.random.RandomState()
    
    def reset(self, seed: int, scenario_config=None):
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        self.hit_count = np.zeros(self.band_count, dtype=float)
        self.visit_count = np.zeros(self.band_count, dtype=float)
        self.epsilon = self.eps_start
        self.t = 0
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        unvisited = np.where(self.visit_count == 0)[0]
        if len(unvisited) > 0:
            return unvisited[0]
        
        self.epsilon = max(self.eps_floor, self.eps_start * (self.decay ** self.t))
        
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.band_count)
        else:
            hit_rates = self.hit_count / np.maximum(self.visit_count, 1)
            return int(np.argmax(hit_rates))
    
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
print("VYAPTI - EPSILON-GREEDY (FULL SAVE)")
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
# RUN ALL - SAVE ALL METRICS
# =============================================================================
all_results = {}

# Fixed epsilon values
for eps in [0.05, 0.10, 0.15, 0.20]:
    name = f"EpsGreedy_{eps}"
    print(f"\n{'='*80}\nRUNNING {name}\n{'='*80}")
    
    all_pd, all_pfa = [], []
    all_episodes_data = []
    
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
        
        schedulers = {name: EpsilonGreedyScheduler(band_count=BAND_COUNT, epsilon=eps)}
        episode_results = run_paired_episodes(env=env, schedulers=schedulers, seed=ep_idx)
        trajectory = episode_results[name].trajectory
        
        metrics = metrics_engine.record_result(
            episode_id=ep_idx, seed=ep_idx, scheduler_name=name,
            scenario_config={}, trajectory=trajectory, truth_grid=env.hidden_truth
        )
        
        all_episodes_data.append({
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
            print(f"Episode {ep_idx+1:3d} ({config_name}): Pd={pd:.6f}, Hits={hits}")
    
    n = len(all_pd)
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
            'epsilon': eps
        },
        'episodes': all_episodes_data
    }

# Decay variants
for name, Scheduler in [
    ("EpsGreedy_Decay_Fast", EpsilonGreedyDecayScheduler),
    ("EpsGreedy_Decay_Slow", EpsilonGreedySlowDecayScheduler)
]:
    print(f"\n{'='*80}\nRUNNING {name}\n{'='*80}")
    
    all_pd, all_pfa = [], []
    all_episodes_data = []
    
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
        
        metrics = metrics_engine.record_result(
            episode_id=ep_idx, seed=ep_idx, scheduler_name=name,
            scenario_config={}, trajectory=trajectory, truth_grid=env.hidden_truth
        )
        
        all_episodes_data.append({
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
            print(f"Episode {ep_idx+1:3d} ({config_name}): Pd={pd:.6f}, Hits={hits}")
    
    n = len(all_pd)
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
            'n_episodes': n
        },
        'episodes': all_episodes_data
    }

# =============================================================================
# SAVE
# =============================================================================
output_dir = Path("vyapti_results")
output_dir.mkdir(exist_ok=True)

all_file = output_dir / "epsilon_greedy_ALL_METRICS.json"
with open(all_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n✓ Saved ALL metrics: {all_file}")
print(f"  File size: {all_file.stat().st_size / 1024 / 1024:.1f} MB")

agg_file = output_dir / "epsilon_greedy_aggregate.json"
aggregate_only = {name: data['aggregate'] for name, data in all_results.items()}
with open(agg_file, 'w') as f:
    json.dump(aggregate_only, f, indent=2)
print(f"✓ Saved aggregate: {agg_file}")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ COMPLETE")
print("="*80)
#!/usr/bin/env python3
"""
Vyapti - Simplified Predictive Sampling
(Inspired by Liu, Van Roy & Xu 2023, but SIMPLIFIED)

Core idea from paper: Sample future trajectory, compute future value of information.

Our simplification: Instead of full trajectory sampling, we:
1. Estimate non-stationarity rate from recent data
2. Adjust exploration based on how quickly information becomes obsolete
3. Sample from posterior, but weight by information freshness

This is NOT a full implementation - just captures the key insight.
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
# SIMPLIFIED PREDICTIVE SAMPLING (defensible approximation)
# =============================================================================
class SimplifiedPredictiveSamplingScheduler(BaseScheduler):
    """
    Simplified Predictive Sampling (Liu et al. 2023 inspired).
    
    Key insight: Information value decays in non-stationary environments.
    
    Algorithm:
    1. Estimate non-stationarity rate (how fast does band change?)
    2. Sample from posterior (like Thompson)
    3. Discount sample by information age (older info = less valuable)
    4. Choose band with highest discounted value
    
    This is SIMPLIFIED - doesn't sample full future trajectories.
    """
    def __init__(self, band_count: int, window_size: int = 20, 
                 decay_rate: float = 0.05):
        super().__init__(band_count, 
                        provenance_note=f"Simplified Predictive Sampling (δ={decay_rate})")
        self.band_count = band_count
        self.window_size = window_size
        self.decay_rate = decay_rate  # How fast information becomes obsolete
        
        # Sliding window for recent observations
        self.history = [deque(maxlen=window_size) for _ in range(band_count)]
        
        # Track when we last got "fresh" information
        self.last_info_time = np.full(band_count, -1, dtype=int)
        
        # Estimate non-stationarity (variance in recent window)
        self.nonstationarity = np.ones(band_count) * 0.5  # Prior
        
        self.rng = np.random.default_rng()
        self.t = 0
    
    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.history = [deque(maxlen=self.window_size) for _ in range(self.band_count)]
        self.last_info_time = np.full(self.band_count, -1, dtype=int)
        self.nonstationarity = np.ones(self.band_count) * 0.5
        self.t = 0
    
    def _compute_posterior(self, band):
        """Compute Beta posterior from sliding window"""
        if len(self.history[band]) == 0:
            return 1.0, 1.0  # Prior
        
        hits = sum(1 for (hit, _) in self.history[band] if hit)
        visits = len(self.history[band])
        
        # Beta posterior
        alpha = 1.0 + hits
        beta = 1.0 + visits - hits
        
        return alpha, beta
    
    def _estimate_nonstationarity(self, band):
        """Estimate how fast this band's reward is changing"""
        if len(self.history[band]) < 10:
            return 0.5  # Prior
        
        # Compute variance in hit rate over time
        hits = np.array([1 if hit else 0 for (hit, _) in self.history[band]])
        
        if len(hits) < 2:
            return 0.5
        
        # Simple measure: variance of recent hit rates
        # High variance = rapidly changing = high non-stationarity
        variance = np.var(hits)
        
        return variance  # In [0, 0.25] for Bernoulli
    
    def _compute_information_decay(self, band):
        """
        Compute how much the information has decayed since last observation.
        
        From Predictive Sampling paper: information value decays in non-stationary env.
        """
        time_since_info = self.t - self.last_info_time[band]
        time_since_info = max(time_since_info, 1)
        
        # Exponential decay
        # Fast-changing bands (high nonstationarity) decay faster
        decay = np.exp(-self.decay_rate * self.nonstationarity[band] * time_since_info)
        
        return decay  # In [0, 1]
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        # Force visit all bands once first
        if self.t < self.band_count:
            return self.t % self.band_count
        
        scores = []
        
        for band in range(self.band_count):
            # 1. Sample from posterior (Thompson-like)
            alpha, beta_param = self._compute_posterior(band)
            sample = self.rng.beta(alpha, beta_param)
            
            # 2. Discount by information decay (Predictive Sampling insight)
            info_decay = self._compute_information_decay(band)
            
            # Discounted score
            # Old/stale information gets down-weighted
            discounted_sample = sample * info_decay
            
            # 3. Bonus for high non-stationarity (explore rapidly-changing bands)
            # This captures the "value of information" idea
            ns_bonus = 0.1 * self.nonstationarity[band] * (1 - info_decay)
            
            # Combined score
            score = discounted_sample + ns_bonus
            scores.append(score)
        
        scores = np.array(scores)
        
        # Tiny noise for tie-breaking
        scores += self.rng.uniform(0, 1e-10, self.band_count)
        
        return int(np.argmax(scores))
    
    def update(self, action: int, observation: dict):
        hit = observation.get('hit', False)
        
        # Add to history
        self.history[action].append((hit, self.t))
        
        # Update last info time (we just got fresh information!)
        self.last_info_time[action] = self.t
        
        # Update non-stationarity estimate
        self.nonstationarity[action] = self._estimate_nonstationarity(action)
        
        self.t += 1

# =============================================================================
# RUN
# =============================================================================
print("="*80)
print("VYAPTI - SIMPLIFIED PREDICTIVE SAMPLING")
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

# Test different decay rates
for decay_rate in [0.02, 0.05, 0.1]:
    name = f"SimplifiedPredictive_δ{decay_rate}"
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
        
        schedulers = {name: SimplifiedPredictiveSamplingScheduler(
            band_count=BAND_COUNT,
            window_size=20,
            decay_rate=decay_rate
        )}
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
            'decay_rate': decay_rate,
            'window_size': 20
        },
        'episodes': all_episodes_data
    }

# =============================================================================
# SAVE
# =============================================================================
output_dir = Path("vyapti_results")
output_dir.mkdir(exist_ok=True)

all_file = output_dir / "simplified_predictive_ALL_METRICS.json"
with open(all_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n✓ Saved ALL metrics: {all_file}")
print(f"  File size: {all_file.stat().st_size / 1024 / 1024:.1f} MB")

agg_file = output_dir / "simplified_predictive_aggregate.json"
aggregate_only = {name: data['aggregate'] for name, data in all_results.items()}
with open(agg_file, 'w') as f:
    json.dump(aggregate_only, f, indent=2)
print(f"✓ Saved aggregate: {agg_file}")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ COMPLETE")
print("="*80)
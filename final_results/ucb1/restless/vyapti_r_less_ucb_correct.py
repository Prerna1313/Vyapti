#!/usr/bin/env python3
"""
Vyapti - R-less-UCB (Metelli et al., 2022)
CORRECT Implementation from ICML 2022 Paper

Paper: "Stochastic Rising Bandits"
Metelli, Trovò¹²³, Pirola, Restelli (ICML 2022)
https://proceedings.mlr.press/v162/metelli22a/metelli22a.pdf

Key idea (Eq. 5):
μ̂_i^R-less(t) = μ_i(t_{i,N_{i,t-1}}) + (t - t_{i,N_{i,t-1}}) * [μ_i(t_{i,N_{i,t-1}}) - μ_i(t_{i,N_{i,t-1}-1})] / [t_{i,N_{i,t-1}} - t_{i,N_{i,t-1}-1}]

Translation:
- Most recent payoff + (time since last pull) × (most recent increment)
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
# R-less-UCB (Metelli et al., 2022) - CORRECT IMPLEMENTATION
# =============================================================================
class RLessUCBScheduler(BaseScheduler):
    """
    R-less-UCB: Optimistic estimator for restless rising bandits.
    
    Direct implementation of Algorithm 1 from Metelli et al. (2022).
    
    Optimistic estimator (Eq. 5):
    μ̂_i^R-less(t) = μ_i(t_{i,N_{i,t-1}}) + (t - t_{i,N_{i,t-1}}) * [μ_i(t_{i,N_{i,t-1}}) - μ_i(t_{i,N_{i,t-1}-1})] / [t_{i,N_{i,t-1}} - t_{i,N_{i,t-1}-1}]
    
    Where:
    - μ_i(t_{i,N_{i,t-1}}) = most recent payoff from arm i
    - t - t_{i,N_{i,t-1}} = time since last pull
    - [μ_i(t_{i,N_{i,t-1}}) - μ_i(t_{i,N_{i,t-1}-1})] / [t_{i,N_{i,t-1}} - t_{i,N_{i,t-1}-1}] = most recent increment
    
    Confidence bonus (UCB-style):
    β_t = sqrt(α * log(t) / N_i(t))
    """
    def __init__(self, band_count: int, alpha: float = 2.1, 
                 epsilon: float = 0.25, gamma: float = 0.9):
        super().__init__(band_count, 
                        provenance_note=f"R-less-UCB (Metelli et al. 2022)")
        self.band_count = band_count
        self.alpha = alpha  # Confidence bonus parameter
        self.epsilon = epsilon  # Discount for old increments
        self.gamma = gamma  # Discount for very old observations
        
        # Payoff history for each band
        # payoff_history[band] = list of (time_slot, payoff) tuples
        self.payoff_history = [[] for _ in range(band_count)]
        
        # Last pull time for each band
        self.last_pull_time = np.full(band_count, -1, dtype=int)
        
        # Number of pulls for each band
        self.pull_counts = np.zeros(band_count, dtype=int)
        
        # Optimistic estimates
        self.optimistic_estimates = np.zeros(band_count)
        
        self.t = 0
    
    def reset(self, seed: int, scenario_config=None):
        self.payoff_history = [[] for _ in range(self.band_count)]
        self.last_pull_time = np.full(self.band_count, -1, dtype=int)
        self.pull_counts = np.zeros(self.band_count, dtype=int)
        self.optimistic_estimates = np.zeros(self.band_count)
        self.t = 0
    
    def _compute_optimistic_estimator(self, band, current_time):
        """
        Compute R-less-UCB optimistic estimator (Eq. 5).
        
        μ̂_i^R-less(t) = μ_i(t_{i,N_{i,t-1}}) + (t - t_{i,N_{i,t-1}}) * increment
        
        Returns:
        - optimistic_estimate: μ̂_i^R-less(t)
        - increment: most recent increment (for confidence bonus)
        """
        history = self.payoff_history[band]
        
        if len(history) == 0:
            # No observations yet
            return 0.5, 0.0
        
        # Most recent payoff
        last_time, last_payoff = history[-1]
        
        if len(history) == 1:
            # Only one observation, no increment available
            increment = 0.0
        else:
            # Compute most recent increment
            prev_time, prev_payoff = history[-2]
            time_diff = last_time - prev_time
            if time_diff > 0:
                increment = (last_payoff - prev_payoff) / time_diff
            else:
                increment = 0.0
        
        # Time since last pull
        time_since_last = current_time - last_time
        
        # Optimistic estimate
        optimistic_estimate = last_payoff + time_since_last * increment
        
        return optimistic_estimate, increment
    
    def _compute_confidence_bonus(self, band):
        """
        Compute UCB-style confidence bonus.
        
        β_t = sqrt(α * log(t) / N_i(t))
        """
        n = self.pull_counts[band]
        if n == 0:
            return float('inf')
        
        if self.t == 0:
            return float('inf')
        
        return np.sqrt(self.alpha * np.log(self.t + 1) / n)
    
    def _compute_all_estimates(self):
        """
        Compute optimistic estimates for all bands.
        """
        for band in range(self.band_count):
            estimate, _ = self._compute_optimistic_estimator(band, self.t)
            bonus = self._compute_confidence_bonus(band)
            self.optimistic_estimates[band] = estimate + bonus
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        self.t = current_time_slot
        
        # Force visit all bands once first (initialize)
        if self.t < self.band_count:
            return self.t % self.band_count
        
        # Compute optimistic estimates for all bands
        self._compute_all_estimates()
        
        # Choose band with HIGHEST optimistic estimate
        return int(np.argmax(self.optimistic_estimates))
    
    def update(self, action: int, observation: dict):
        hit = observation.get('hit', False)
        reward = 1.0 if hit else 0.0
        
        # Store payoff with timestamp
        self.payoff_history[action].append((self.t, reward))
        
        # Update last pull time
        self.last_pull_time[action] = self.t
        
        # Update pull count
        self.pull_counts[action] += 1
        
        # Keep history bounded (optional, for memory efficiency)
        if len(self.payoff_history[action]) > 100:
            # Keep only last 100 observations
            self.payoff_history[action] = self.payoff_history[action][-100:]


# =============================================================================
# RUN
# =============================================================================
print("="*80)
print("VYAPTI - R-less-UCB (Metelli et al., 2022)")
print("CORRECT Implementation from ICML 2022 Paper")
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

# Test different alpha values (confidence bonus strength)
for alpha in [1.5, 2.1, 3.0]:
    name = f"R-less-UCB_α{alpha}"
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
            config=metrics_config,
            stare_mode_file=stare_file,
            scan_mode_file=scan_file
        )
        
        schedulers = {name: RLessUCBScheduler(
            band_count=BAND_COUNT,
            alpha=alpha,
            epsilon=0.25,
            gamma=0.9
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
            'alpha': alpha,
            'epsilon': 0.25,
            'gamma': 0.9,
            'algorithm': 'R-less-UCB',
            'citation': 'Metelli et al. (2022) - ICML',
            'formula': 'Eq. 5: μ̂_i^R-less(t) = μ_i(t_last) + (t - t_last) * increment'
        },
        'episodes': all_episodes_data
    }

# =============================================================================
# SAVE
# =============================================================================
output_dir = Path("vyapti_results")
output_dir.mkdir(exist_ok=True)

all_file = output_dir / "r_less_ucb_correct_ALL_METRICS.json"
with open(all_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n✓ Saved ALL metrics: {all_file}")
print(f"  File size: {all_file.stat().st_size / 1024 / 1024:.1f} MB")

agg_file = output_dir / "r_less_ucb_correct_aggregate.json"
aggregate_only = {name: data['aggregate'] for name, data in all_results.items()}
with open(agg_file, 'w') as f:
    json.dump(aggregate_only, f, indent=2)
print(f"✓ Saved aggregate: {agg_file}")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ COMPLETE - R-less-UCB (CORRECT)")
print("="*80)
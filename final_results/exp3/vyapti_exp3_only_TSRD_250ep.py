#!/usr/bin/env python3
"""
Vyapti - EXP3 META-LEARNER ONLY
FULL RUN: ALL 250 TSRD TEST EPISODES

Pure Exp3 (Exponential-weight algorithm for Exploration and Exploitation).
No ensemble - just Exp3 selecting among base schedulers.

Environment: TSRD Stare Mode (PARTIALLY OBSERVABLE)
- 36 bands
- Saves ALL 34+ TSRD metrics
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
# BASE SCHEDULER (Simple Belief-Based)
# =============================================================================
class SimpleBeliefScheduler(BaseScheduler):
    """
    Simple belief-based scheduler for Exp3 to select among.
    """
    def __init__(self, band_count: int, strategy='hit_rate', alpha=0.3):
        super().__init__(band_count, 
                        provenance_note=f"Simple Belief ({strategy})")
        self.band_count = band_count
        self.strategy = strategy
        self.alpha = alpha
        
        # Hit history
        self.hit_history = {b: [] for b in range(band_count)}
        
        # EMA of hit rate
        self.ema_hit_rate = np.ones(band_count) * 0.5
        
        # Visit counts
        self.visit_counts = np.zeros(band_count, dtype=int)
        
        self.t = 0
        self.rng = np.random.default_rng()
    
    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.hit_history = {b: [] for b in range(self.band_count)}
        self.ema_hit_rate = np.ones(self.band_count) * 0.5
        self.visit_counts = np.zeros(self.band_count, dtype=int)
        self.t = 0
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        self.t = current_time_slot
        
        # Force visit all bands once first
        if self.t < self.band_count:
            return self.t % self.band_count
        
        # Compute scores based on strategy
        if self.strategy == 'hit_rate':
            scores = self.ema_hit_rate.copy()
        elif self.strategy == 'ucb':
            # UCB1-style exploration bonus
            scores = self.ema_hit_rate + np.sqrt(2 * np.log(self.t + 1) / (self.visit_counts + 1))
        else:  # thompson
            # Thompson sampling
            samples = self.rng.beta(
                self.ema_hit_rate * 10 + 1,
                (1 - self.ema_hit_rate) * 10 + 1
            )
            scores = samples
        
        # Add noise for tie-breaking
        scores = scores + self.rng.uniform(0, 1e-10, self.band_count)
        
        return int(np.argmax(scores))
    
    def update(self, action: int, observation: dict):
        hit = observation.get('hit', False)
        hit_binary = 1.0 if hit else 0.0
        
        # Add to history
        self.hit_history[action].append(hit_binary)
        
        # Keep recent history
        if len(self.hit_history[action]) > 40:
            self.hit_history[action] = self.hit_history[action][-40:]
        
        # Update EMA
        self.ema_hit_rate[action] = (
            self.alpha * hit_binary + 
            (1 - self.alpha) * self.ema_hit_rate[action]
        )
        
        # Update visit counts
        self.visit_counts[action] += 1
        
        self.t += 1


# =============================================================================
# EXP3 META-LEARNER
# =============================================================================
class Exp3MetaLearner:
    """
    Exp3 (Exponential-weight algorithm for Exploration and Exploitation).
    
    Selects among K base schedulers (arms) based on their performance.
    """
    def __init__(self, n_arms, gamma_exp3=0.1):
        self.n_arms = n_arms
        self.gamma = gamma_exp3  # Exploration parameter (0.1 = 10% exploration)
        self.weights = np.ones(n_arms)  # Initialize uniform weights
    
    def get_selection_probs(self):
        """
        Compute probability distribution over arms.
        
        Mix of uniform exploration and weighted exploitation.
        """
        # Exploit: weight-proportional selection
        exploit_probs = self.weights / self.weights.sum()
        
        # Explore: uniform
        explore_probs = np.ones(self.n_arms) / self.n_arms
        
        # Mix: (1 - γ) * exploit + γ * explore
        probs = (1 - self.gamma) * exploit_probs + self.gamma * explore_probs
        
        return probs
    
    def select_arm(self):
        """
        Select arm according to Exp3 distribution.
        """
        probs = self.get_selection_probs()
        return np.random.choice(self.n_arms, p=probs)
    
    def update(self, arm_idx, reward):
        """
        Update weights based on observed reward.
        
        Uses importance weighting to correct for biased sampling.
        """
        # Get current probabilities
        probs = self.get_selection_probs()
        
        # Importance-weighted reward estimate
        # (unbiased estimate of what reward would be if we always picked this arm)
        estimated_reward = reward / probs[arm_idx]
        
        # Update weight multiplicatively
        # Higher reward → higher weight → more likely to be selected
        self.weights[arm_idx] *= np.exp(self.gamma * estimated_reward / self.n_arms)
        
        # Clip weights to prevent numerical overflow
        self.weights = np.clip(self.weights, 1e-10, 1e10)


# =============================================================================
# EXP3 SCHEDULER (Selects Among Base Schedulers)
# =============================================================================
class Exp3Scheduler(BaseScheduler):
    """
    Exp3 meta-learner that selects among multiple base schedulers.
    
    Architecture:
    - Multiple base schedulers (different strategies)
    - Exp3 selects which scheduler to use at each timestep
    - All schedulers learn from every observation (parallel learning)
    """
    def __init__(self, band_count: int, n_arms_to_activate: int = 1,
                 base_configs=None, gamma_exp3=0.1):
        super().__init__(band_count, 
                        provenance_note=f"Exp3 (γ={gamma_exp3})")
        
        self.band_count = band_count
        self.n_arms_to_activate = n_arms_to_activate
        self.gamma_exp3 = gamma_exp3
        
        # Base scheduler configurations
        if base_configs is None:
            self.base_configs = [
                {'strategy': 'hit_rate', 'alpha': 0.1},
                {'strategy': 'hit_rate', 'alpha': 0.3},
                {'strategy': 'hit_rate', 'alpha': 0.5},
                {'strategy': 'ucb', 'alpha': 0.3},
                {'strategy': 'thompson', 'alpha': 0.3},
            ]
        else:
            self.base_configs = base_configs
        
        # Create base schedulers
        self.base_schedulers = []
        for i, config in enumerate(self.base_configs):
            scheduler = SimpleBeliefScheduler(
                band_count=band_count,
                strategy=config['strategy'],
                alpha=config['alpha']
            )
            self.base_schedulers.append(scheduler)
        
        # Exp3 meta-learner
        self.exp3 = Exp3MetaLearner(n_arms=len(self.base_schedulers), gamma_exp3=gamma_exp3)
        
        # Track last selected scheduler
        self.last_scheduler_idx = None
        
        self.rng = np.random.default_rng()
    
    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        for scheduler in self.base_schedulers:
            scheduler.reset(seed=seed)
        self.exp3 = Exp3MetaLearner(n_arms=len(self.base_schedulers), gamma_exp3=self.gamma_exp3)
        self.last_scheduler_idx = None
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        # Exp3 selects which base scheduler to use
        scheduler_idx = self.exp3.select_arm()
        self.last_scheduler_idx = scheduler_idx
        
        # Use selected scheduler's action
        scheduler = self.base_schedulers[scheduler_idx]
        return scheduler.select_action(observation_history, current_time_slot)
    
    def update(self, action: int, observation: dict):
        hit = observation.get('hit', False)
        reward = 1.0 if hit else 0.0
        
        # Update ALL base schedulers (parallel learning)
        for scheduler in self.base_schedulers:
            scheduler.update(action, observation)
        
        # Update Exp3 meta-learner (reward for selected scheduler)
        if self.last_scheduler_idx is not None:
            self.exp3.update(self.last_scheduler_idx, reward)
        
        self.last_scheduler_idx = None


# =============================================================================
# RUN - ALL 250 TSRD EPISODES
# =============================================================================
print("="*80)
print("VYAPTI - EXP3 META-LEARNER SCHEDULER")
print("FULL RUN: ALL 250 TSRD TEST EPISODES")
print("Exp3 selects among different base schedulers")
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

# Different Exp3 configs
exp3_configs = [
    {
        'name': 'Exp3_5base_γ0.1',
        'gamma_exp3': 0.1,
        'base_configs': [
            {'strategy': 'hit_rate', 'alpha': 0.1},
            {'strategy': 'hit_rate', 'alpha': 0.3},
            {'strategy': 'hit_rate', 'alpha': 0.5},
            {'strategy': 'ucb', 'alpha': 0.3},
            {'strategy': 'thompson', 'alpha': 0.3},
        ]
    },
    {
        'name': 'Exp3_3base_γ0.1',
        'gamma_exp3': 0.1,
        'base_configs': [
            {'strategy': 'hit_rate', 'alpha': 0.3},
            {'strategy': 'ucb', 'alpha': 0.3},
            {'strategy': 'thompson', 'alpha': 0.3},
        ]
    },
    {
        'name': 'Exp3_5base_γ0.3',
        'gamma_exp3': 0.3,
        'base_configs': [
            {'strategy': 'hit_rate', 'alpha': 0.1},
            {'strategy': 'hit_rate', 'alpha': 0.3},
            {'strategy': 'hit_rate', 'alpha': 0.5},
            {'strategy': 'ucb', 'alpha': 0.3},
            {'strategy': 'thompson', 'alpha': 0.3},
        ]
    },
]

for config in exp3_configs:
    name = config['name']
    gamma_exp3 = config['gamma_exp3']
    base_configs = config['base_configs']
    
    print(f"\n{'='*80}\nRUNNING {name} (γ_exp3={gamma_exp3})\n{'='*80}")
    
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
        
        schedulers = {name: Exp3Scheduler(
            band_count=BAND_COUNT,
            n_arms_to_activate=1,
            base_configs=base_configs,
            gamma_exp3=gamma_exp3
        )}
        
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
                'gamma_exp3': gamma_exp3,
                'n_base_schedulers': len(base_configs),
                'base_configs': base_configs,
                'algorithm': 'Exp3 Meta-Learner',
                'description': 'Exp3 selects among base schedulers (hit_rate, UCB, Thompson)',
                'metrics_saved': 'ALL 34+ TSRD metrics',
            },
            'episodes': all_episodes_data
        }

# =============================================================================
# SAVE
# =============================================================================
output_dir = Path("vyapti_results")
output_dir.mkdir(exist_ok=True)

all_file = output_dir / "exp3_only_TSRD_250ep_ALL_METRICS.json"
with open(all_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n✓ Saved ALL metrics: {all_file}")
print(f"  File size: {all_file.stat().st_size / 1024 / 1024:.1f} MB")

agg_file = output_dir / "exp3_only_TSRD_250ep_aggregate.json"
aggregate_only = {name: data['aggregate'] for name, data in all_results.items()}
with open(agg_file, 'w') as f:
    json.dump(aggregate_only, f, indent=2)
print(f"✓ Saved aggregate: {agg_file}")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ COMPLETE - EXP3 ONLY (ALL 250 TSRD EPISODES, ALL 34+ METRICS)")
print("="*80)
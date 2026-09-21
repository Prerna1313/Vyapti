#!/usr/bin/env python3
"""
Vyapti - Whittle Index Approximation (FIXED API)
Based on Whittle index theory for restless bandits

Key papers:
1. Whittle (1988) - Original restless bandits
2. Wang et al. (2023) - UCWhittle: Online learning for restless bandits
   https://storage.googleapis.com/gweb-research2023-media/pubtools/6927.pdf
3. Akbarzadeh & Mahajan (2022) - Partially observable restless bandits
   https://cim.mcgill.ca/~adityam/projects/bandits/preprint/pomdp-bandits.pdf
4. Liu, Weber, Zhao (2011) - Indexability and Whittle index for reset processes

Whittle Index Definition (Definition 3.1, Wang et al. 2023):
W_i(P_i, s_i) = inf{m_i : Q^{m_i}(s_i, 0) = Q^{m_i}(s_i, 1)}

where Q^{m}(s, a) solves Bellman equation with penalty m for active action:
Q^m(s, a) = -m*a + R(s, a) + γ * Σ_{s'} P(s, a, s') * V^m(s')
V^m(s) = max_{a∈A} Q^m(s, a)

For PARTIALLY OBSERVABLE setting (our case - TSRD):
- State is HIDDEN (we only see hit/no-hit)
- Belief state π_t = P(state | observations, actions)
- Whittle index w(π) = smallest λ where passive action is optimal

Model A (no observations except when pulled):
- Belief evolves as: π_{t+1} = π_t * P if passive, Q if active
- This is a "restart" model (active action resets belief)

Closed-form Whittle index for restart model (Theorem 7, Akbarzadeh & Mahajan):
w(k) = min_{k'∈Λ_k} [D(k+1)(k') - D(k)(k')] / [N(k)(k') - N(k+1)(k')]

where:
- k = time since last observation (belief age)
- D(θ)(k) = expected discounted cost until threshold θ
- N(θ)(k) = expected discounted activations until threshold θ
- Λ_k = set of thresholds where activation count differs

SIMPLIFIED for implementation (common approximation):
w(k) ≈ r * (1 - α) * [(1 - β) * p - β * (1 - p)] / [1 - (α * β)]

where:
- r = reward magnitude
- α, β = transition parameters
- p = belief probability of good state

For TSRD, we use PRACTICAL approximation:
W_i(t) = μ̂_i + λ * (k_i / τ_i)

where:
- μ̂_i = estimated hit rate
- k_i = time since last observation (belief age)
- τ_i = characteristic time scale (PRI estimate)
- λ = scaling factor (Whittle index "steepness")

This captures the key insight: Whittle index GROWS with belief age k,
reflecting increasing uncertainty and value of information.
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
# WHITTLE INDEX APPROXIMATION (corrected - based on actual theory)
# =============================================================================
class WhittleIndexScheduler(BaseScheduler):
    """
    Whittle Index Approximation for Partially Observable Restless Bandits.
    
    Based on:
    - Wang et al. (2023): UCWhittle algorithm for online Whittle index learning
    - Akbarzadeh & Mahajan (2022): Whittle index for partially observable bandits
    - Liu, Weber, Zhao (2011): Closed-form Whittle index for restart processes
    
    True Whittle index requires:
    1. Known transition dynamics P(s'|s,a)
    2. Solving Bellman equations with penalty λ
    3. Finding λ where active/passive actions are equally valuable
    
    For PARTIALLY OBSERVABLE setting (TSRD):
    - State is hidden, we maintain belief π_t
    - Belief age k = time since last observation
    - Whittle index w(k) grows with k (more uncertain = more valuable to observe)
    
    Our approximation (defensible):
    W_i(t) = μ̂_i + λ * (k_i / τ_i)
    
    where:
    - μ̂_i : estimated hit rate (base value)
    - k_i : belief age (time since last visit)
    - τ_i : characteristic time scale (estimated PRI)
    - λ : Whittle index steepness (how fast index grows with uncertainty)
    
    This captures the KEY INSIGHT from Whittle index theory:
    - Index grows when unobserved (belief becomes stale)
    - Growth rate depends on band's dynamics (PRI)
    - Higher index = more urgent to observe
    """
    def __init__(self, band_count: int, lambda_whittle: float = 0.5, 
                 window_size: int = 30, default_pri: float = 15.0):
        super().__init__(band_count, 
                        provenance_note=f"Whittle Index Approx (λ={lambda_whittle})")
        self.band_count = band_count
        self.lambda_whittle = lambda_whittle  # Whittle index steepness
        self.window_size = window_size
        self.default_pri = default_pri  # Default time scale τ
        
        # Sliding window for hit rate estimation (μ̂_i)
        self.history = [deque(maxlen=window_size) for _ in range(band_count)]
        
        # Last visit time (for computing belief age k_i)
        self.last_visit = np.full(band_count, -1, dtype=int)
        
        # Estimated PRI (time scale τ_i) per band
        self.estimated_pri = np.ones(band_count) * default_pri
        
        # Track hits for PRI estimation
        self.last_hit_time = np.full(band_count, -1, dtype=int)
        self.pri_observations = [[] for _ in range(band_count)]
        
        self.rng = np.random.default_rng()
        self.t = 0
    
    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.history = [deque(maxlen=self.window_size) for _ in range(self.band_count)]
        self.last_visit = np.full(self.band_count, -1, dtype=int)
        self.estimated_pri = np.ones(self.band_count) * self.default_pri
        self.last_hit_time = np.full(self.band_count, -1, dtype=int)
        self.pri_observations = [[] for _ in range(self.band_count)]
        self.t = 0
    
    def _compute_hit_rate(self, band):
        """Compute empirical hit rate μ̂_i from sliding window"""
        if len(self.history[band]) == 0:
            return 0.5  # Prior (uninformative)
        
        hits = sum(1 for (hit, _) in self.history[band] if hit)
        return hits / len(self.history[band])
    
    def _update_pri_estimate(self, band):
        """
        Update PRI estimate τ_i from hit observations.
        
        PRI = time between consecutive hits
        This is the characteristic time scale of the band's dynamics.
        """
        if self.last_hit_time[band] < 0:
            return  # No hits yet
        
        # Collect PRI observations (time between hits)
        if len(self.pri_observations[band]) > 0:
            # Use median for robustness (outliers common in TSRD)
            self.estimated_pri[band] = np.median(self.pri_observations[band][-10:])
    
    def _compute_whittle_index(self, band):
        """
        Compute Whittle index approximation.
        
        True Whittle index (Wang et al. 2023, Def 3.1):
        W_i(P_i, s_i) = inf{m_i : Q^{m_i}(s_i, 0) = Q^{m_i}(s_i, 1)}
        
        For partially observable restart model (Akbarzadeh & Mahajan 2022):
        w(k) = min_{k'} [D(k+1)(k') - D(k)(k')] / [N(k)(k') - N(k+1)(k')]
        
        Our approximation (captures key insight):
        W_i(t) = μ̂_i + λ * (k_i / τ_i)
        
        where:
        - μ̂_i : base value (hit rate)
        - k_i : belief age (uncertainty grows with time)
        - τ_i : characteristic time scale (PRI)
        - λ : steepness (how much uncertainty matters)
        """
        # 1. Base value: empirical hit rate μ̂_i
        hit_rate = self._compute_hit_rate(band)
        
        # 2. Belief age: k_i = time since last observation
        # (belief becomes stale, uncertainty grows)
        belief_age = self.t - self.last_visit[band]
        belief_age = max(belief_age, 1)  # At least 1
        
        # 3. Normalize by PRI τ_i (band's characteristic time scale)
        # (bands with short PRI become uncertain faster)
        normalized_age = belief_age / (self.estimated_pri[band] + 1)
        
        # 4. Whittle index approximation
        # W_i(t) = μ̂_i + λ * (k_i / τ_i)
        whittle_index = hit_rate + self.lambda_whittle * normalized_age
        
        return whittle_index
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        # Force visit all bands once first (initialize beliefs)
        if self.t < self.band_count:
            return self.t % self.band_count
        
        # Compute Whittle index W_i(t) for each band
        indices = []
        for band in range(self.band_count):
            idx = self._compute_whittle_index(band)
            indices.append(idx)
        
        indices = np.array(indices)
        
        # Add tiny noise for tie-breaking (not exploration)
        indices += self.rng.uniform(0, 1e-10, self.band_count)
        
        # Choose band with HIGHEST Whittle index
        # (most urgent to observe = highest marginal value)
        return int(np.argmax(indices))
    
    def update(self, action: int, observation: dict):
        hit = observation.get('hit', False)
        
        # Add to history (for hit rate estimation)
        self.history[action].append((hit, self.t))
        
        # Update last visit (belief age resets to 0)
        self.last_visit[action] = self.t
        
        # Update PRI estimate if hit
        if hit:
            if self.last_hit_time[action] >= 0:
                # Time between consecutive hits = PRI observation
                pri_obs = self.t - self.last_hit_time[action]
                if pri_obs > 0:
                    self.pri_observations[action].append(pri_obs)
            
            self.last_hit_time[action] = self.t
            self._update_pri_estimate(action)
        
        self.t += 1

# =============================================================================
# RUN
# =============================================================================
print("="*80)
print("VYAPTI - WHITTLE INDEX APPROXIMATION (FIXED API)")
print("Based on Whittle index theory for restless bandits")
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

# Test different lambda values (Whittle index steepness)
# λ controls how fast index grows with belief age
for lambda_w in [0.3, 0.5, 0.7]:
    name = f"WhittleIndex_λ{lambda_w}"
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
        
        # FIXED: Use correct parameter names (stare_mode_file, scan_mode_file)
        metrics_engine = TSRDMetricsEngine(
            config=metrics_config,
            stare_mode_file=stare_file,  # ← FIXED (was stare_file)
            scan_mode_file=scan_file      # ← FIXED (was scan_file)
        )
        
        schedulers = {name: WhittleIndexScheduler(
            band_count=BAND_COUNT,
            lambda_whittle=lambda_w,
            window_size=30,
            default_pri=15.0
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
            'lambda_whittle': lambda_w,
            'window_size': 30,
            'default_pri': 15.0
        },
        'episodes': all_episodes_data
    }

# =============================================================================
# SAVE
# =============================================================================
output_dir = Path("vyapti_results")
output_dir.mkdir(exist_ok=True)

all_file = output_dir / "whittle_index_FIXED_ALL_METRICS.json"
with open(all_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n✓ Saved ALL metrics: {all_file}")
print(f"  File size: {all_file.stat().st_size / 1024 / 1024:.1f} MB")

agg_file = output_dir / "whittle_index_FIXED_aggregate.json"
aggregate_only = {name: data['aggregate'] for name, data in all_results.items()}
with open(agg_file, 'w') as f:
    json.dump(aggregate_only, f, indent=2)
print(f"✓ Saved aggregate: {agg_file}")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ COMPLETE")
print("="*80)
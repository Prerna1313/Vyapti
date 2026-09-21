#!/usr/bin/env python3
"""
Vyapti - Belief-State Whittle Scheduler (UCWhittle-Inspired)
FULL RUN: ALL 250 TSRD TEST EPISODES

ADAPTATION of UCWhittle ideas to Vyapti's partially observable TSRD setting.
NOT the faithful Wang et al. (2023) algorithm - adapted for hidden emitter states.

Environment: TSRD Stare Mode (PARTIALLY OBSERVABLE)
- 36 bands
- Hidden emitter states (NOT observable)
- Observations: hit, snrdb, false_alarm (from env.step)
- Belief state maintained via Bayesian update
- K=1 band activated per timestep
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
# BELIEF-STATE WHITTLE SCHEDULER (UCWhittle-Inspired for TSRD)
# =============================================================================
class BeliefStateWhittleScheduler(BaseScheduler):
    """
    Vyapti's adaptation of UCWhittle ideas to partially observable TSRD setting.
    
    Key differences from faithful UCWhittle:
    - Belief state (not fully observable)
    - Only learns active action transitions (passive not observable in TSRD)
    - Heuristic optimistic transitions (not exact L1 confidence region)
    - Per-timestep recomputation (not episode-level)
    
    Should be called "Vyapti Belief-Whittle" NOT "UCWhittle (Wang et al., 2023)"
    """
    def __init__(self, band_count: int, n_arms_to_activate: int = None,
                 gamma: float = 0.99, delta: float = 0.05,
                 n_states: int = 2, cache_steps: int = 5):
        super().__init__(band_count, 
                        provenance_note=f"Vyapti Belief-Whittle (UCWhittle-inspired)")
        self.band_count = band_count
        self.n_arms_to_activate = n_arms_to_activate or 1
        self.gamma = gamma
        self.delta = delta
        self.n_states = n_states
        self.cache_steps = cache_steps
        
        # Transition counts N(s,a,s') - only for active action in TSRD
        self.transition_counts = np.zeros((band_count, n_states, 2, n_states))
        
        # Reward counts and sums
        self.reward_counts = np.zeros((band_count, n_states, 2), dtype=int)
        self.reward_sums = np.zeros((band_count, n_states, 2))
        
        # Belief state for each band
        self.belief = np.ones(band_count) * 0.5
        
        # Total time step
        self.t = 1
        
        # Whittle indices (cached)
        self.whittle_indices = np.zeros(band_count)
        self.last_whittle_update = 0
        
        # Penalty λ
        self.penalty_lambda = 0.0
        
        self.rng = np.random.default_rng()
    
    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.transition_counts = np.zeros((self.band_count, self.n_states, 2, self.n_states))
        self.reward_counts = np.zeros((self.band_count, self.n_states, 2), dtype=int)
        self.reward_sums = np.zeros((self.band_count, self.n_states, 2))
        self.belief = np.ones(self.band_count) * 0.5
        self.t = 1
        self.whittle_indices = np.zeros(self.band_count)
        self.last_whittle_update = 0
        self.penalty_lambda = 0.0
    
    def _discretize_belief(self, belief):
        return 1 if belief >= 0.5 else 0
    
    def _update_belief_bayesian(self, band, hit):
        Pd = 0.9
        Pfa = 0.1
        
        prior = self.belief[band]
        
        if hit:
            likelihood_h1 = Pd
            likelihood_h0 = Pfa
        else:
            likelihood_h1 = 1.0 - Pd
            likelihood_h0 = 1.0 - Pfa
        
        numerator = likelihood_h1 * prior
        denominator = likelihood_h1 * prior + likelihood_h0 * (1.0 - prior)
        
        if denominator > 1e-10:
            posterior = numerator / denominator
        else:
            posterior = 0.5
        
        self.belief[band] = np.clip(posterior, 0.01, 0.99)
    
    def _compute_empirical_transition(self, band):
        counts = self.transition_counts[band]
        P = np.zeros_like(counts)
        
        for s in range(self.n_states):
            for a in range(2):
                total = counts[s, a, :].sum()
                if total > 0:
                    P[s, a, :] = counts[s, a, :] / total
                else:
                    P[s, a, :] = 1.0 / self.n_states
        
        return P
    
    def _compute_empirical_reward(self, band):
        R = np.zeros((self.n_states, 2))
        
        for s in range(self.n_states):
            for a in range(2):
                if self.reward_counts[band, s, a] > 0:
                    R[s, a] = self.reward_sums[band, s, a] / self.reward_counts[band, s, a]
                else:
                    R[s, a] = 0.5
        
        return R
    
    def _compute_confidence_bound(self, band, s, a):
        N_sa = self.transition_counts[band, s, a, :].sum()
        
        if N_sa == 0:
            return float('inf')
        
        size_S = self.n_states
        size_A = 2
        N = self.t
        
        log_term = np.log(2 * size_S * size_A * (N ** 4) / self.delta)
        d = np.sqrt(2 * size_S * log_term / max(1, N_sa))
        
        return d
    
    def _compute_optimistic_transition(self, band, P_empirical):
        P_opt = P_empirical.copy()
        
        for s in range(self.n_states):
            for a in range(2):
                d = self._compute_confidence_bound(band, s, a)
                if d == float('inf'):
                    continue
                
                for s_prime in range(self.n_states):
                    P_opt[s, a, s_prime] = min(P_empirical[s, a, s_prime] + d, 1.0)
                
                total = P_opt[s, a, :].sum()
                if total > 0:
                    P_opt[s, a, :] /= total
        
        return P_opt
    
    def _solve_bellman(self, band, P, R, lambda_penalty):
        V = np.zeros(self.n_states)
        
        for _ in range(20):
            V_new = np.zeros(self.n_states)
            
            for s in range(self.n_states):
                Q_active = R[s, 1] - lambda_penalty + self.gamma * np.dot(P[s, 1, :], V)
                Q_passive = R[s, 0] + self.gamma * np.dot(P[s, 0, :], V)
                
                V_new[s] = max(Q_active, Q_passive)
            
            if np.max(np.abs(V_new - V)) < 1e-6:
                break
            
            V = V_new
        
        return V
    
    def _compute_whittle_index(self, band, P, R):
        lambda_low = -10.0
        lambda_high = 10.0
        
        for _ in range(10):
            lambda_mid = (lambda_low + lambda_high) / 2
            
            V = self._solve_bellman(band, P, R, lambda_mid)
            
            s = self._discretize_belief(self.belief[band])
            
            Q_active = R[s, 1] - lambda_mid + self.gamma * np.dot(P[s, 1, :], V)
            Q_passive = R[s, 0] + self.gamma * np.dot(P[s, 0, :], V)
            
            if Q_active > Q_passive:
                lambda_low = lambda_mid
            else:
                lambda_high = lambda_mid
        
        return (lambda_low + lambda_high) / 2
    
    def _compute_all_whittle_indices(self):
        for band in range(self.band_count):
            P_emp = self._compute_empirical_transition(band)
            R_emp = self._compute_empirical_reward(band)
            
            P_opt = self._compute_optimistic_transition(band, P_emp)
            
            whittle = self._compute_whittle_index(band, P_opt, R_emp)
            
            self.whittle_indices[band] = whittle
    
    def _update_penalty(self):
        sorted_indices = np.sort(self.whittle_indices)[::-1]
        
        if len(sorted_indices) >= self.n_arms_to_activate:
            self.penalty_lambda = sorted_indices[self.n_arms_to_activate - 1]
        else:
            self.penalty_lambda = 0.0
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        self.t = current_time_slot
        
        if self.t < self.band_count:
            return self.t % self.band_count
        
        # Recompute Whittle indices every cache_steps timesteps
        if (self.t - self.last_whittle_update) >= self.cache_steps:
            self._compute_all_whittle_indices()
            self._update_penalty()
            self.last_whittle_update = self.t
        
        indices = self.whittle_indices + self.rng.uniform(0, 1e-10, self.band_count)
        
        return int(np.argmax(indices))
    
    def update(self, action: int, observation: dict):
        hit = observation.get('hit', False)
        reward = 1.0 if hit else 0.0
        
        s_before = self._discretize_belief(self.belief[action])
        
        self._update_belief_bayesian(action, hit)
        
        s_after = self._discretize_belief(self.belief[action])
        
        a = 1
        self.transition_counts[action, s_before, a, s_after] += 1
        
        self.reward_counts[action, s_before, a] += 1
        self.reward_sums[action, s_before, a] += reward
        
        self.t += 1


# =============================================================================
# RUN - ALL 250 TSRD EPISODES
# =============================================================================
print("="*80)
print("VYAPTI - BELIEF-STATE WHITTLE (UCWhittle-Inspired)")
print("FULL RUN: ALL 250 TSRD TEST EPISODES")
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

# Test different delta values
for delta in [0.01, 0.05, 0.1]:
    name = f"BeliefWhittle_δ{delta}"
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
        
        schedulers = {name: BeliefStateWhittleScheduler(
            band_count=BAND_COUNT,
            n_arms_to_activate=1,
            gamma=0.99,
            delta=delta,
            n_states=2,
            cache_steps=5
        )}
        
        episode_results = run_paired_episodes(env=env, schedulers=schedulers, seed=ep_idx)
        
        trajectory = episode_results[name].trajectory
        
        # This saves ALL 34+ TSRD metrics
        metrics = metrics_engine.record_result(
            episode_id=ep_idx, seed=ep_idx, scheduler_name=name,
            scenario_config={}, trajectory=trajectory, truth_grid=env.hidden_truth
        )
        
        all_episodes_data.append({
            'episode': ep_idx,
            'config': config_name,
            'seed': ep_idx,
            'all_metrics': metrics  # ← ALL 34+ METRICS SAVED HERE
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
                'delta': delta,
                'gamma': 0.99,
                'n_states': 2,
                'n_arms_to_activate': 1,
                'cache_steps': 5,
                'algorithm': 'Vyapti Belief-Whittle (UCWhittle-inspired)',
                'note': 'Adaptation for partially observable TSRD - NOT faithful UCWhittle',
                'metrics_saved': 'ALL 34+ TSRD metrics (Pd, Pfa, hits, latencies, etc.)',
            },
            'episodes': all_episodes_data
        }

# =============================================================================
# SAVE
# =============================================================================
output_dir = Path("vyapti_results")
output_dir.mkdir(exist_ok=True)

all_file = output_dir / "belief_whittle_TSRD_250ep_ALL_METRICS.json"
with open(all_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n✓ Saved ALL metrics: {all_file}")
print(f"  File size: {all_file.stat().st_size / 1024 / 1024:.1f} MB")

agg_file = output_dir / "belief_whittle_TSRD_250ep_aggregate.json"
aggregate_only = {name: data['aggregate'] for name, data in all_results.items()}
with open(agg_file, 'w') as f:
    json.dump(aggregate_only, f, indent=2)
print(f"✓ Saved aggregate: {agg_file}")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ COMPLETE - BELIEF-WHITTLE (ALL 250 TSRD EPISODES, ALL 34+ METRICS)")
print("="*80)
#!/usr/bin/env python3
"""
Vyapti - ARP Scheduler (RESEARCH-GRADE VERSION)
Activity-Recency-Periodicity Heuristic

Research-grade temporal baseline with:
- Clean EMA activity estimate (binary observations)
- Robust periodicity estimation (median + MAD)
- Statistical confidence (dispersion + sample size)
- Uncertainty term (exploration bonus)
- Explicit, normalized weights
- Leakage-safe (no access to ground truth)

Position: Below HMM/BOCPD/Whittle, above simple UCB/Thompson.

Formulation:
S_i(t) = w_A * A_i(t) + w_R * C_i(t) * R_i(t) + w_U * U_i(t)

where:
- A_i(t) = exponentially weighted detection activity
- C_i(t) = periodicity confidence (dispersion + sample size)
- R_i(t) = min(Δ_i(t) / τ_i(t), 1)  (temporal revisit priority)
- U_i(t) = 1 / √(n_i + 1)  (uncertainty/exploration bonus)
- w_A, w_R, w_U = normalized weights (w_A + w_R + w_U = 1)

Reference: Simple temporal baseline for Vyapti project.
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
# ARP SCHEDULER (RESEARCH-GRADE)
# =============================================================================
class ARPScheduler(BaseScheduler):
    """
    Activity-Recency-Periodicity Scheduler (Research-Grade Version).
    
    Clean, reproducible temporal heuristic with:
    - Binary activity update (no arbitrary 0.8/0.2)
    - Robust periodicity (median + MAD)
    - Statistical confidence (dispersion + sample size)
    - Uncertainty term (exploration)
    - Explicit, normalized weights
    
    NOT a probabilistic model (no Bayesian update, no HMM).
    Position: below HMM/BOCPD/Whittle, above UCB/Thompson.
    """
    def __init__(self, band_count: int, 
                 activity_weight: float = 0.6,
                 recency_weight: float = 0.25,
                 uncertainty_weight: float = 0.15,
                 activity_eta: float = 0.3,
                 default_interval: float = 20.0,
                 min_detections_for_period: int = 3):
        super().__init__(band_count, 
                        provenance_note=f"ARP (wA={activity_weight}, wR={recency_weight}, wU={uncertainty_weight})")
        
        # Validate and normalize weights
        total_weight = activity_weight + recency_weight + uncertainty_weight
        self.w_A = activity_weight / total_weight
        self.w_R = recency_weight / total_weight
        self.w_U = uncertainty_weight / total_weight
        
        self.activity_eta = activity_eta
        self.default_interval = default_interval
        self.min_detections_for_period = min_detections_for_period
        
        # Activity estimate A_i(t)
        self.activity = np.ones(band_count) * 0.5
        
        # Detection times for periodicity estimation
        self.detection_times = [deque(maxlen=20) for _ in range(band_count)]
        
        # Inter-detection intervals
        self.intervals = [deque(maxlen=20) for _ in range(band_count)]
        
        # Periodicity estimate τ_i(t) (median)
        self.tau = np.ones(band_count) * default_interval
        
        # Dispersion MAD_i (median absolute deviation)
        self.mad = np.ones(band_count) * default_interval
        
        # Periodicity confidence C_i(t)
        self.confidence = np.zeros(band_count)
        
        # Time since last detection Δ_i(t)
        self.time_since_detection = np.zeros(band_count)
        
        # Number of observations n_i(t)
        self.n_observations = np.zeros(band_count, dtype=int)
        
        # Local RNG
        self.rng = np.random.default_rng()
        self.t = 0
    
    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.activity = np.ones(self.band_count) * 0.5
        self.detection_times = [deque(maxlen=20) for _ in range(self.band_count)]
        self.intervals = [deque(maxlen=20) for _ in range(self.band_count)]
        self.tau = np.ones(self.band_count) * self.default_interval
        self.mad = np.ones(self.band_count) * self.default_interval
        self.confidence = np.zeros(self.band_count)
        self.time_since_detection = np.zeros(self.band_count)
        self.n_observations = np.zeros(self.band_count, dtype=int)
        self.t = 0
    
    def _update_activity(self, band, hit):
        """
        Update activity estimate A_i(t) using clean EMA with binary observations.
        
        A_i(t) = (1 - η) * A_i(t-1) + η * y_t
        
        where y_t = 1 if hit, 0 if miss.
        
        This is exponentially weighted recent detection frequency.
        """
        y_t = 1.0 if hit else 0.0
        
        self.activity[band] = (
            (1 - self.activity_eta) * self.activity[band] + 
            self.activity_eta * y_t
        )
    
    def _update_periodicity(self, band, hit):
        """
        Update periodicity estimates with robust statistics.
        
        Compute:
        - τ_i = median(intervals)  (central tendency)
        - MAD_i = median(|intervals - τ_i|)  (dispersion)
        - C_i = confidence based on dispersion + sample size
        """
        if hit:
            # Record detection time
            self.detection_times[band].append(self.t)
            
            # Compute new interval if we have ≥2 detections
            if len(self.detection_times[band]) >= 2:
                times = list(self.detection_times[band])
                new_interval = times[-1] - times[-2]
                self.intervals[band].append(new_interval)
            
            # Update statistics if we have enough intervals
            if len(self.intervals[band]) >= self.min_detections_for_period:
                intervals_arr = np.array(list(self.intervals[band]))
                
                # Median (robust central tendency)
                self.tau[band] = np.median(intervals_arr)
                
                # MAD (robust dispersion)
                median_abs_dev = np.median(np.abs(intervals_arr - self.tau[band]))
                self.mad[band] = median_abs_dev
                
                # Confidence based on dispersion
                # C_disp = 1 / (1 + MAD / (τ + ε))
                epsilon = 1e-6
                C_disp = 1.0 / (1.0 + self.mad[band] / (self.tau[band] + epsilon))
                
                # Confidence based on sample size
                # C_n = 1 - exp(-n / k)  where k = 5
                k = 5.0
                n = len(self.intervals[band])
                C_n = 1.0 - np.exp(-n / k)
                
                # Combined confidence
                self.confidence[band] = C_disp * C_n
            
            # Reset time since detection
            self.time_since_detection[band] = 0
        else:
            # Increment time since last detection
            self.time_since_detection[band] += 1
        
        # Update observation count
        self.n_observations[band] += 1
    
    def _compute_recency_score(self, band):
        """
        Compute temporal revisit priority R_i(t).
        
        R_i(t) = min(Δ_i(t) / (τ_i(t) + ε), 1)
        
        Interpretation:
        - Longer time since detection → higher revisit priority
        - Relative to observed inter-detection interval
        - Capped at 1.0 (don't over-prioritize)
        
        For bands with no periodicity estimate, use default τ_0.
        """
        delta_t = self.time_since_detection[band]
        
        if self.confidence[band] > 0.1:
            # Use estimated τ_i
            tau = self.tau[band]
        else:
            # Use default τ_0 (no periodicity estimate yet)
            tau = self.default_interval
        
        epsilon = 1e-6
        R_i = min(delta_t / (tau + epsilon), 1.0)
        
        return R_i
    
    def _compute_uncertainty(self, band):
        """
        Compute uncertainty/exploration bonus U_i(t).
        
        U_i(t) = 1 / √(n_i(t) + 1)
        
        Bands with few observations get higher uncertainty bonus.
        This encourages exploration of under-observed bands.
        """
        n_i = self.n_observations[band]
        return 1.0 / np.sqrt(n_i + 1)
    
    def _compute_score(self, band):
        """
        Compute total score S_i(t).
        
        S_i(t) = w_A * A_i(t) + w_R * C_i(t) * R_i(t) + w_U * U_i(t)
        
        Components:
        - Activity A_i(t): recent detection frequency
        - Recency R_i(t): temporal revisit priority
        - Confidence C_i(t): reliability of periodicity estimate
        - Uncertainty U_i(t): exploration bonus
        
        Weights w_A, w_R, w_U are normalized (sum to 1).
        """
        A_i = self.activity[band]
        R_i = self._compute_recency_score(band)
        C_i = self.confidence[band]
        U_i = self._compute_uncertainty(band)
        
        S_i = self.w_A * A_i + self.w_R * C_i * R_i + self.w_U * U_i
        
        return S_i
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        # Force visit all bands once first
        if self.t < self.band_count:
            return self.t % self.band_count
        
        # Compute score for each band
        scores = [self._compute_score(band) for band in range(self.band_count)]
        scores = np.array(scores)
        
        # Add tiny noise for tie-breaking (using local RNG)
        scores += self.rng.uniform(0, 1e-10, self.band_count)
        
        return int(np.argmax(scores))
    
    def update(self, action: int, observation: dict):
        hit = observation.get('hit', False)
        
        # Update activity estimate
        self._update_activity(action, hit)
        
        # Update periodicity estimates
        self._update_periodicity(action, hit)
        
        self.t += 1

# =============================================================================
# RUN
# =============================================================================
print("="*80)
print("VYAPTI - ARP SCHEDULER (RESEARCH-GRADE)")
print("Activity-Recency-Periodicity Heuristic")
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

# Test different weight configurations
weight_configs = [
    (0.6, 0.25, 0.15),  # Balanced
    (0.7, 0.2, 0.1),    # Activity-focused
    (0.5, 0.35, 0.15),  # Recency-focused
    (0.5, 0.25, 0.25),  # Uncertainty-focused
]

for w_A, w_R, w_U in weight_configs:
    name = f"ARP_wA{w_A}wR{w_R}wU{w_U}"
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
        
        schedulers = {name: ARPScheduler(
            band_count=BAND_COUNT,
            activity_weight=w_A,
            recency_weight=w_R,
            uncertainty_weight=w_U,
            activity_eta=0.3,
            default_interval=20.0,
            min_detections_for_period=3
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
            'w_A': w_A,
            'w_R': w_R,
            'w_U': w_U,
            'activity_eta': 0.3,
            'default_interval': 20.0,
            'min_detections_for_period': 3,
            'algorithm': 'ARP',
            'model_type': 'research_grade_heuristic',
            'probabilistic': False,
            'citation': 'Activity-Recency-Periodicity Heuristic'
        },
        'episodes': all_episodes_data
    }

# =============================================================================
# SAVE
# =============================================================================
output_dir = Path("vyapti_results")
output_dir.mkdir(exist_ok=True)

all_file = output_dir / "ARP_research_grade_ALL_METRICS.json"
with open(all_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n✓ Saved ALL metrics: {all_file}")
print(f"  File size: {all_file.stat().st_size / 1024 / 1024:.1f} MB")

agg_file = output_dir / "ARP_research_grade_aggregate.json"
aggregate_only = {name: data['aggregate'] for name, data in all_results.items()}
with open(agg_file, 'w') as f:
    json.dump(aggregate_only, f, indent=2)
print(f"✓ Saved aggregate: {agg_file}")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ COMPLETE - ARP RESEARCH-GRADE")
print("="*80)
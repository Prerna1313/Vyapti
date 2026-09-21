#!/usr/bin/env python3
"""
VYAPTI — META-UCB BANDIT COMBINER (CUTKOSKY ALGORITHM 1, EMPIRICAL ADAPTATION)

Implements the Cutkosky Bandit Combiner [1] as a meta-level scheduler
that selects ONE base expert per round and updates only that expert.

This is an EMPIRICAL ADAPTATION for the Vyapti environment.
The theoretical Cutkosky guarantee requires:
  (i) known putative regret bounds C_i t^alpha_i with high-probability properties,
  (ii) R_i satisfying technical conditions in Theorem 1 (including cross-expert conditions),
  (iii) at least one well-specified base algorithm.

These conditions are NOT established for our custom experts in Vyapti/TSRD.

Therefore this implementation:
  - Uses empirical_C(alpha=0.5) from V3 analysis as heuristic C_i
  - Uses R_i = scale * empirical_C * T^alpha (satisfies only R_i >= empirical_C * T^alpha numerically)
  - Explicitly disclaims the theoretical guarantee

Reference:
[1] Cutkosky, Das, Purohit (2020). "Upper Confidence Bounds for
    Combining Stochastic Bandits". https://arxiv.org/abs/2012.13115
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from vyapti_simulator.core.scheduler_interface import BaseScheduler
from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment


# =============================================================================
# CONFIGURATION
# =============================================================================

BAND_COUNT = 36
HORIZON = 600
PD = 0.90
PFA = 0.01

# Empirical parameters from V3 analysis
EMPIRICAL_PARAMS_FILE = Path("/kaggle/working/base_expert_regret_analysis_V3.json")

# CORRECTION 6: Explicit alpha selection policy
ALPHA_USED = 0.5
ALPHA_SELECTION_POLICY = "predeclared_fixed_alpha"

# Delta for confidence bounds (paper uses delta in log(T^3 N / delta))
DELTA = 0.05

# R_i heuristic: scale factor for R_i = scale * empirical_C * T^alpha
# This satisfies R_i >= empirical_C * T^alpha numerically, but empirical_C is NOT
# a theoretically established Cutkosky C_i, and the full Theorem 1 cross-expert
# conditions are NOT verified.
R_EMPIRICAL_SCALE = 1.0
R_METHOD = "first_condition_scale_only"

OUTPUT_FILE = Path("/kaggle/working/meta_ucb_combiner_results.json")


# =============================================================================
# BASE EXPERT CLASSES (same as V3 analysis)
# =============================================================================

class UCB1Scheduler(BaseScheduler):
    def __init__(self, band_count: int):
        super().__init__(band_count, provenance_note="UCB1")
        self.band_count = band_count
        self.counts = np.zeros(band_count, dtype=np.int64)
        self.rewards = np.zeros(band_count, dtype=float)
        self.local_t = 0
        self.rng = np.random.default_rng()

    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.counts.fill(0)
        self.rewards.fill(0.0)
        self.local_t = 0

    def select_action(self, observation_history, current_time_slot: int) -> int:
        if self.local_t < self.band_count:
            return self.local_t % self.band_count
        means = self.rewards / np.maximum(self.counts, 1)
        scores = np.full(self.band_count, -np.inf)
        visited = self.counts > 0
        scores[visited] = means[visited] + np.sqrt(
            2.0 * np.log(self.local_t + 1.0) / self.counts[visited]
        )
        scores += self.rng.uniform(0.0, 1e-12, self.band_count)
        return int(np.argmax(scores))

    def update(self, action: int, observation: dict):
        self.counts[action] += 1
        self.rewards[action] += float(bool(observation.get("hit", False)))
        self.local_t += 1


class WhittleInspiredScheduler(BaseScheduler):
    def __init__(self, band_count: int, lambda_param: float = 0.3):
        super().__init__(band_count, provenance_note="Whittle-Inspired")
        self.band_count = band_count
        self.lambda_param = lambda_param
        self.belief = np.full(band_count, 0.5, dtype=float)
        self.local_t = 0
        self.rng = np.random.default_rng()

    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.belief.fill(0.5)
        self.local_t = 0

    def select_action(self, observation_history, current_time_slot: int) -> int:
        if self.local_t < self.band_count:
            return self.local_t % self.band_count
        scores = self.belief * (1.0 - self.lambda_param)
        scores += self.rng.uniform(0.0, 1e-12, self.band_count)
        return int(np.argmax(scores))

    def update(self, action: int, observation: dict):
        hit = bool(observation.get("hit", False))
        prior = self.belief[action]
        pd_assumed, pfa_assumed = 0.9, 0.01
        if hit:
            lh1, lh0 = pd_assumed, pfa_assumed
        else:
            lh1, lh0 = 1.0 - pd_assumed, 1.0 - pfa_assumed
        numerator = lh1 * prior
        denominator = numerator + lh0 * (1.0 - prior)
        self.belief[action] = np.clip(
            numerator / denominator if denominator > 1e-12 else 0.5,
            0.01,
            0.99,
        )
        self.local_t += 1


class RLessUCBScheduler(BaseScheduler):
    def __init__(self, band_count: int, alpha: float = 1.5):
        super().__init__(band_count, provenance_note="RLessUCB")
        self.band_count = band_count
        self.alpha = alpha
        self.counts = np.zeros(band_count, dtype=np.int64)
        self.rewards = np.zeros(band_count, dtype=float)
        self.last_visit = np.full(band_count, -np.inf, dtype=float)
        self.local_t = 0
        self.rng = np.random.default_rng()

    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.counts.fill(0)
        self.rewards.fill(0.0)
        self.last_visit.fill(-np.inf)
        self.local_t = 0

    def select_action(self, observation_history, current_time_slot: int) -> int:
        if self.local_t < self.band_count:
            return self.local_t % self.band_count
        unvisited = np.flatnonzero(self.counts == 0)
        if len(unvisited):
            return int(unvisited[0])
        means = self.rewards / self.counts
        elapsed = np.maximum(0.0, current_time_slot - self.last_visit)
        scores = means + np.sqrt(self.alpha * np.log(elapsed + 1.0) / self.counts)
        scores += self.rng.uniform(0.0, 1e-12, self.band_count)
        return int(np.argmax(scores))

    def update(self, action: int, observation: dict):
        self.counts[action] += 1
        self.rewards[action] += float(bool(observation.get("hit", False)))
        self.last_visit[action] = float(observation.get("global_t", self.local_t))
        self.local_t += 1


class ARPScheduler(BaseScheduler):
    def __init__(self, band_count: int, wA: float = 0.6, wR: float = 0.25, wU: float = 0.15):
        super().__init__(band_count, provenance_note="ARP heuristic")
        self.band_count = band_count
        self.wA, self.wR, self.wU = wA, wR, wU
        self.hit_history = {b: [] for b in range(band_count)}
        self.ema_hit_rate = np.full(band_count, 0.5, dtype=float)
        self.recency = np.zeros(band_count, dtype=float)
        self.uncertainty = np.full(band_count, 0.5, dtype=float)
        self.local_t = 0
        self.rng = np.random.default_rng()

    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.hit_history = {b: [] for b in range(self.band_count)}
        self.ema_hit_rate.fill(0.5)
        self.recency.fill(0.0)
        self.uncertainty.fill(0.5)
        self.local_t = 0

    def select_action(self, observation_history, current_time_slot: int) -> int:
        if self.local_t < self.band_count:
            return self.local_t % self.band_count
        scores = self.wA * self.ema_hit_rate + self.wR * self.recency + self.wU * self.uncertainty
        scores += self.rng.uniform(0.0, 1e-12, self.band_count)
        return int(np.argmax(scores))

    def update(self, action: int, observation: dict):
        hit = float(bool(observation.get("hit", False)))
        self.hit_history[action].append(hit)
        if len(self.hit_history[action]) > 40:
            self.hit_history[action] = self.hit_history[action][-40:]
        self.ema_hit_rate[action] = 0.3 * hit + 0.7 * self.ema_hit_rate[action]
        self.recency[action] = 1.0 if hit else self.recency[action] * 0.98
        for band in range(self.band_count):
            history = self.hit_history[band][-20:]
            if len(history) > 5:
                self.uncertainty[band] = float(np.std(history))
        self.local_t += 1


EXPERT_CLASSES = {
    "UCB1": UCB1Scheduler,
    "Whittle-Inspired": WhittleInspiredScheduler,
    "RLessUCB": RLessUCBScheduler,
    "ARP": ARPScheduler,
}


# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

def validate_empirical_parameters(
    expert_params: dict,
    expert_names: list,
    alpha: float,
):
    """
    Validate that every expert has exactly one empirical_C(alpha) entry.
    """
    for name in expert_names:
        if name not in expert_params:
            raise KeyError(f"Missing expert parameters for {name}")
        
        candidates = expert_params[name].get("envelope_candidates", [])
        
        matches = [
            c for c in candidates
            if np.isclose(float(c["alpha"]), alpha)
        ]
        
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one empirical_C(alpha={alpha}) "
                f"for {name}; found {len(matches)}"
            )
        
        C = float(matches[0]["empirical_C"])
        
        if not np.isfinite(C) or C <= 0:
            raise ValueError(f"Invalid empirical_C for {name}: {C}")


# =============================================================================
# META-UCB BANDIT COMBINER (CUTKOSKY ALGORITHM 1, EMPIRICAL ADAPTATION)
# =============================================================================

class MetaUCBBanditCombiner(BaseScheduler):
    """
    Cutkosky Bandit Combiner (Algorithm 1): treats each base expert as a meta-level arm.
    
    At each round t:
    1. Compute UCB-like score for each active expert
    2. Select expert i_t with highest score
    3. Get action from expert i_t
    4. Observe reward
    5. Update only expert i_t
    6. Update meta-level statistics for expert i_t
    7. Check elimination condition for expert i_t
    
    This is an EMPIRICAL ADAPTATION. The theoretical guarantee requires
    known putative regret bounds C_i t^alpha_i that are not established
    for our custom experts in Vyapti.
    """
    
    def __init__(
        self,
        band_count: int,
        expert_params: dict,
        horizon: int = 600,
        alpha_used: float = 0.5,
        delta: float = 0.05,
        r_empirical_scale: float = 1.0,
    ):
        super().__init__(band_count, provenance_note="Meta-UCB Combiner (Empirical)")
        self.band_count = band_count
        self.horizon = horizon
        self.alpha_used = alpha_used
        self.delta = delta
        self.r_empirical_scale = r_empirical_scale
        
        # Initialize base experts
        self.expert_names = list(EXPERT_CLASSES.keys())
        self.experts = {}
        for name in self.expert_names:
            self.experts[name] = EXPERT_CLASSES[name](band_count)
        
        # Meta-level statistics (per expert)
        # CORRECTION 1: mu_hat_i(0) = 0, not 0.5
        self.meta_counts = {name: 0 for name in self.expert_names}
        self.meta_rewards = {name: 0.0 for name in self.expert_names}
        self.deviation_sums = {name: 0.0 for name in self.expert_names}
        self.prior_means = {name: 0.0 for name in self.expert_names}
        self.eliminated = {name: False for name in self.expert_names}
        self.elimination_history = []
        
        # CORRECTION: Track selected expert state
        self._selected_expert = None
        
        # Load and validate empirical parameters
        self.empirical_params = expert_params
        validate_empirical_parameters(
            self.empirical_params,
            self.expert_names,
            self.alpha_used,
        )
        
        # CORRECTION 3: exact log term from paper
        self.log_term = np.log(
            (self.horizon ** 3 * len(self.expert_names)) / self.delta
        )
        
        # Precompute R_i for each expert (heuristic, satisfies R_i >= empirical_C * T^alpha numerically)
        # Note: empirical_C is NOT a theoretically established Cutkosky C_i
        self.R = {}
        for name in self.expert_names:
            C_i = self._get_empirical_C(name)
            self.R[name] = self.r_empirical_scale * C_i * (self.horizon ** self.alpha_used)
        
        # Meta-level RNG
        self.rng = np.random.default_rng()
        self.local_t = 0
    
    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        for name in self.expert_names:
            self.experts[name].reset(seed)
            self.meta_counts[name] = 0
            self.meta_rewards[name] = 0.0
            self.deviation_sums[name] = 0.0
            self.prior_means[name] = 0.0
            self.eliminated[name] = False
        self.elimination_history = []
        self._selected_expert = None
        self.local_t = 0
    
    def _get_empirical_C(self, expert_name: str) -> float:
        """
        Extract empirical_C(alpha_used) from V3 analysis output.
        """
        expert_data = self.empirical_params.get(expert_name, {})
        envelope_candidates = expert_data.get("envelope_candidates", [])
        
        for candidate in envelope_candidates:
            if np.isclose(candidate["alpha"], self.alpha_used):
                C = candidate.get("empirical_C", 1.0)
                if not np.isfinite(C) or C <= 0:
                    raise ValueError(
                        f"Invalid empirical_C for {expert_name}: {C}"
                    )
                return float(C)
        
        raise KeyError(
            f"No empirical_C(alpha={self.alpha_used}) found for expert {expert_name}"
        )
    
    def _compute_meta_score(self, expert_name: str) -> float:
        """
        Compute Meta-UCB score for an expert.
        
        Score = mean_reward + confidence_bonus - R_i / T
        
        where confidence_bonus uses:
          C_i * n^(alpha_i - 1) + sqrt(8 log(T^3 N / delta) / n)
        """
        n = self.meta_counts[expert_name]
        
        if n == 0:
            return float('inf')  # Explore unplayed experts
        
        mean_reward = self.meta_rewards[expert_name] / n
        
        # Get empirical C_i for this expert
        C_i = self._get_empirical_C(expert_name)
        alpha_i = self.alpha_used
        
        # CORRECTION 3: exact confidence bonus from paper
        confidence = (
            C_i * (n ** (alpha_i - 1.0))
            + np.sqrt(8.0 * self.log_term / n)
        )
        confidence = min(1.0, confidence)
        
        # R_i penalty (heuristic, satisfies R_i >= empirical_C * T^alpha numerically only)
        R_i = self.R[expert_name]
        
        score = mean_reward + confidence - R_i / self.horizon
        
        return score
    
    def select_action(self, observation_history, current_time_slot: int) -> int:
        # Get active (non-eliminated) experts
        active_experts = [
            name for name in self.expert_names
            if not self.eliminated[name]
        ]
        
        # CORRECTION 5: fail loudly if all eliminated (no UCB1 fallback)
        if len(active_experts) == 0:
            raise RuntimeError(
                "Meta-UCB eliminated all experts; "
                "this indicates the empirical parameterization "
                "is invalid or overly aggressive."
            )
        
        if len(active_experts) == 1:
            selected_expert = active_experts[0]
        else:
            # Select expert with highest meta-score
            scores = {
                name: self._compute_meta_score(name)
                for name in active_experts
            }
            selected_expert = max(scores, key=scores.get)
        
        # Get action from selected expert
        self._selected_expert = selected_expert
        action = self.experts[selected_expert].select_action(
            observation_history, current_time_slot
        )
        
        return action
    
    def update(self, action: int, observation: dict):
        # CORRECTION: Validate that select_action was called first
        if self._selected_expert is None:
            raise RuntimeError("update() called before select_action().")
        
        selected_expert = self._selected_expert
        
        # Get reward
        reward = float(bool(observation.get("hit", False)))
        
        # CORRECTION: compute prior_mean BEFORE updating counts/rewards
        n_before = self.meta_counts[selected_expert]
        if n_before == 0:
            prior_mean = 0.0  # mu_hat_i(0) = 0
        else:
            prior_mean = self.meta_rewards[selected_expert] / n_before
        
        # Update only the selected expert
        self.experts[selected_expert].update(action, observation)
        
        # Update deviation sum for elimination check
        # CORRECTION: uses prior_mean (before update) - reward
        self.deviation_sums[selected_expert] += prior_mean - reward
        
        # Update meta-level statistics
        self.meta_counts[selected_expert] += 1
        self.meta_rewards[selected_expert] += reward
        
        # Update prior_mean for next round
        n = self.meta_counts[selected_expert]
        self.prior_means[selected_expert] = self.meta_rewards[selected_expert] / n
        
        # Check elimination condition
        # CORRECTION 3: exact threshold from paper
        C_i = self._get_empirical_C(selected_expert)
        threshold = (
            C_i * (n ** self.alpha_used)
            + 3.0 * np.sqrt(self.log_term * n)
        )
        
        # CORRECTION 4: proper elimination logging
        if (
            self.deviation_sums[selected_expert] >= threshold
            and not self.eliminated[selected_expert]
        ):
            # Prevent eliminating the last active expert
            active_count = sum(not e for e in self.eliminated.values())
            if active_count <= 1:
                raise RuntimeError(
                    f"Attempted to eliminate final active expert ({selected_expert}); "
                    f"deviation={self.deviation_sums[selected_expert]:.4f}, "
                    f"threshold={threshold:.4f}"
                )
            
            self.eliminated[selected_expert] = True
            
            self.elimination_history.append({
                "expert_name": selected_expert,
                "global_round": int(self.local_t),
                "meta_count": int(n),
                "deviation_sum": float(self.deviation_sums[selected_expert]),
                "threshold": float(threshold),
            })
        
        self.local_t += 1


# =============================================================================
# SIMULATOR
# =============================================================================

def make_sim_config():
    return SimulationConfig(
        band_count=BAND_COUNT,
        time_slots=HORIZON,
        dwell_time_ms=50.0,
        receiver_ibw_mhz=500.0,
        total_spectrum_mhz=18000.0,
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_meta_ucb(
    empirical_params_file: Path,
    test_stare_dir: Path,
    test_scan_dir: Path,
    n_episodes: int = None,
    alpha_used: float = 0.5,
    delta: float = 0.05,
    r_empirical_scale: float = 1.0,
):
    """
    Evaluate Meta-UCB combiner on test episodes.
    """
    
    # Load empirical parameters
    with open(empirical_params_file, "r", encoding="utf-8") as f:
        empirical_params = json.load(f)
    
    expert_params = empirical_params.get("experts", {})
    
    # Validate parameters before proceeding
    validate_empirical_parameters(
        expert_params,
        list(EXPERT_CLASSES.keys()),
        alpha_used,
    )
    
    # Initialize Meta-UCB combiner
    combiner = MetaUCBBanditCombiner(
        band_count=BAND_COUNT,
        expert_params=expert_params,
        horizon=HORIZON,
        alpha_used=alpha_used,
        delta=delta,
        r_empirical_scale=r_empirical_scale,
    )
    
    # Get test files
    test_stare_files = sorted(test_stare_dir.glob("config_*.h5"))
    if n_episodes is not None:
        test_stare_files = test_stare_files[:n_episodes]
    
    paired = [
        f for f in test_stare_files
        if (test_scan_dir / f.name).exists()
    ]
    
    if not paired:
        raise RuntimeError("No paired test scenarios found.")
    
    print(f"Evaluating on {len(paired)} test episodes...")
    
    # Run episodes
    all_rewards = []
    all_selection_counts = []
    all_elimination_histories = []
    
    for idx, stare_file in enumerate(paired):
        combiner.reset(seed=idx)
        
        env = TSRDStareEnvironment.from_stare_mode(
            stare_file=str(stare_file),
            scan_file=str(test_scan_dir / stare_file.name),
            sim_config=make_sim_config(),
        )
        env.reset(seed=idx)
        
        episode_rewards = []
        selection_counts = {name: 0 for name in combiner.expert_names}
        
        for t in range(HORIZON):
            action = combiner.select_action([], t)
            step_result = env.step(action)
            obs = step_result[0] if isinstance(step_result, tuple) else step_result
            obs = dict(obs)
            obs["band"] = int(action)
            obs["global_t"] = t
            
            reward = float(bool(obs.get("hit", False)))
            episode_rewards.append(reward)
            
            selected_expert = combiner._selected_expert
            selection_counts[selected_expert] += 1
            
            combiner.update(action, obs)
        
        all_rewards.append(episode_rewards)
        all_selection_counts.append(selection_counts)
        all_elimination_histories.append(combiner.elimination_history.copy())
        
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(paired)} episodes")
    
    # Aggregate results
    all_rewards = np.asarray(all_rewards)
    mean_reward_rate = float(np.mean(all_rewards))
    mean_episode_reward = float(np.mean(np.sum(all_rewards, axis=1)))
    
    # Selection statistics
    mean_selection_counts = {
        name: float(np.mean([sc[name] for sc in all_selection_counts]))
        for name in combiner.expert_names
    }
    
    # Elimination statistics
    elimination_stats = {}
    for name in combiner.expert_names:
        eliminated_count = sum(
            1 for eh in all_elimination_histories
            if any(e["expert_name"] == name for e in eh)
        )
        elimination_stats[name] = {
            "eliminated_episodes": eliminated_count,
            "elimination_rate": eliminated_count / len(paired),
        }
    
    # Aggregate elimination history
    all_eliminations = []
    for eh in all_elimination_histories:
        all_eliminations.extend(eh)
    
    # Extract meta-parameters for reproducibility
    meta_parameters = {
        name: {
            "empirical_C": combiner._get_empirical_C(name),
            "alpha": float(alpha_used),
            "R": combiner.R[name],
        }
        for name in combiner.expert_names
    }
    
    return {
        "n_episodes": len(paired),
        "mean_reward_rate": mean_reward_rate,
        "mean_episode_reward": mean_episode_reward,
        "std_episode_reward": float(np.std([np.sum(r) for r in all_rewards])),
        "selection_counts": mean_selection_counts,
        "elimination_stats": elimination_stats,
        "total_eliminations": len(all_eliminations),
        "elimination_details": all_eliminations[:100],  # First 100 for inspection
        "per_episode_rewards": [float(np.sum(r)) for r in all_rewards],
        "meta_parameters": meta_parameters,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("VYAPTI — META-UCB BANDIT COMBINER (CUTKOSKY ALGORITHM 1)")
    print("=" * 90)
    print()
    print("EMPIRICAL ADAPTATION — Theoretical guarantee NOT claimed.")
    print()
    print(f"Configuration:")
    print(f"  alpha_used = {ALPHA_USED}")
    print(f"  alpha_selection_policy = {ALPHA_SELECTION_POLICY}")
    print(f"  delta = {DELTA}")
    print(f"  r_empirical_scale = {R_EMPIRICAL_SCALE}")
    print(f"  r_method = {R_METHOD}")
    print(f"  log(T^3 N / delta) = {np.log((HORIZON**3 * 4) / DELTA):.4f}")
    print()
    
    # Check empirical params file
    if not EMPIRICAL_PARAMS_FILE.exists():
        raise RuntimeError(
            f"Empirical params file not found: {EMPIRICAL_PARAMS_FILE}\n"
            "Run vyapti_base_expert_regret_analysis_V3_frozen.py first."
        )
    
    # Download TSRD test set
    print("📥 Downloading TSRD test set...")
    snapshot_path = snapshot_download(
        repo_id="alan-turing-institute/turing-synthetic-radar-dataset",
        repo_type="dataset",
        revision="68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51",
        cache_dir="/root/.cache/huggingface/hub",
    )
    
    tsrd_root = Path(snapshot_path)
    test_stare_dir = tsrd_root / "stare" / "test_stare"
    test_scan_dir = tsrd_root / "scan" / "test_scan"
    
    print(f"✓ Test stare: {test_stare_dir}")
    print(f"✓ Test scan:  {test_scan_dir}")
    print()
    
    # Evaluate Meta-UCB
    print("Evaluating Meta-UCB combiner on 250 test episodes...")
    print()
    
    results = evaluate_meta_ucb(
        empirical_params_file=EMPIRICAL_PARAMS_FILE,
        test_stare_dir=test_stare_dir,
        test_scan_dir=test_scan_dir,
        n_episodes=250,
        alpha_used=ALPHA_USED,
        delta=DELTA,
        r_empirical_scale=R_EMPIRICAL_SCALE,
    )
    
    # Print results
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)
    print()
    print(f"Episodes:           {results['n_episodes']}")
    print(f"Mean reward rate:   {results['mean_reward_rate']:.6f}")
    print(f"Mean episode reward: {results['mean_episode_reward']:.2f} ± {results['std_episode_reward']:.2f}")
    print()
    print("Expert selection counts (mean per episode):")
    for name, count in results["selection_counts"].items():
        print(f"  {name:20s}: {count:6.2f}")
    print()
    print("Elimination statistics:")
    for name, stats in results["elimination_stats"].items():
        print(
            f"  {name:20s}: "
            f"eliminated in {stats['eliminated_episodes']:3d}/{results['n_episodes']} episodes "
            f"({stats['elimination_rate']*100:.1f}%)"
        )
    print()
    print(f"Total eliminations across all episodes: {results['total_eliminations']}")
    print()
    print("Meta-parameters (embedded for reproducibility):")
    for name, params in results["meta_parameters"].items():
        print(
            f"  {name:20s}: "
            f"C={params['empirical_C']:.6f}, "
            f"alpha={params['alpha']:.1f}, "
            f"R={params['R']:.4f}"
        )
    
    # Save results
    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "Meta-UCB Bandit Combiner (Cutkosky Algorithm 1, Empirical Adaptation)",
        "status": "Empirical adaptation — theoretical guarantee NOT claimed",
        "configuration": {
            "alpha_used": ALPHA_USED,
            "alpha_selection_policy": ALPHA_SELECTION_POLICY,
            "delta": DELTA,
            "r_empirical_scale": R_EMPIRICAL_SCALE,
            "r_method": R_METHOD,
            "log_term": float(np.log((HORIZON**3 * 4) / DELTA)),
        },
        "empirical_params_file": str(EMPIRICAL_PARAMS_FILE),
        "test_split": "stare/test_stare + scan/test_scan",
        "n_episodes": results["n_episodes"],
        "horizon": HORIZON,
        "band_count": BAND_COUNT,
        "results": results,
        "scientific_note": (
            "This implements the Cutkosky combiner architecture as an empirical "
            "adaptation. The theoretical regret guarantee requires known putative "
            "regret bounds C_i t^alpha_i with high-probability properties that are "
            "not established for our custom experts in the Vyapti/TSRD environment. "
            f"R_i is set as R_i = {R_EMPIRICAL_SCALE} * empirical_C * T^alpha, which "
            "numerically satisfies R_i >= empirical_C * T^alpha, but empirical_C is "
            "NOT a theoretically established Cutkosky C_i, and the full Theorem-1 "
            "cross-expert conditions are NOT verified. alpha_used=0.5 was predeclared "
            "(not optimized on test data). C_i values are empirical oracle-relative "
            "envelopes from V3 analysis, not theoretical regret bounds."
        ),
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print()
    print(f"💾 Saved: {OUTPUT_FILE}")
    print("=" * 90)


if __name__ == "__main__":
    main()
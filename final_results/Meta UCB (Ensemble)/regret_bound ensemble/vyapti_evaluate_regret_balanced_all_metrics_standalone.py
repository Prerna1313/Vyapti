#!/usr/bin/env python3
"""
VYAPTI — REGRET-BOUND BALANCED ENSEMBLE — FULL EVALUATION (ALL 34+ METRICS)
============================================================================

Evaluates Regret-Balanced Ensemble on 250 TSRD test episodes.
Saves ALL 34+ TSRD metrics per episode + aggregated statistics.

Includes inline RegretBalancedEnsemble class (no external import needed).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.core.metrics import MetricsConfig
from vyapti_simulator.core.episode import run_paired_episodes
from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment
from vyapti_simulator.tsrd.tsrd_metrics import TSRDMetricsEngine
from vyapti_simulator.core.scheduler_interface import BaseScheduler


# =============================================================================
# CONFIGURATION
# =============================================================================

BAND_COUNT = 36
HORIZON = 600
DELTA = 0.05
EMPIRICAL_PARAMS_FILE = Path("/kaggle/working/base_expert_regret_analysis_V3.json")
OUTPUT_FILE = Path("/kaggle/working/vyapti_regret_balanced_all_34_metrics.json")
N_TEST_EPISODES = 250


# =============================================================================
# INLINE BASE EXPERTS (copied from vyapti_regret_balanced_ensemble_fixed.py)
# =============================================================================

class UCB1Scheduler(BaseScheduler):
    def __init__(self, band_count: int = BAND_COUNT):
        super().__init__(band_count, provenance_note="UCB1")
        self.band_count = band_count
        self.counts = np.zeros(band_count, dtype=np.int64)
        self.rewards = np.zeros(band_count, dtype=np.float64)
        self.local_t = 0
        self.rng = np.random.default_rng()

    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.counts.fill(0)
        self.rewards.fill(0.0)
        self.local_t = 0

    def select_action(self, observation_history, current_time_slot: int) -> int:
        if self.local_t < self.band_count:
            return self.local_t
        mean_rewards = self.rewards / np.maximum(self.counts, 1)
        ucb = mean_rewards + np.sqrt(2.0 * np.log(self.local_t + 1) / np.maximum(self.counts, 1))
        ucb += self.rng.uniform(0.0, 1e-12, self.band_count)
        return int(np.argmax(ucb))

    def update(self, action: int, observation: dict):
        hit = float(observation.get("hit", False))
        self.counts[action] += 1
        self.rewards[action] += hit
        self.local_t += 1


class WhittleInspiredScheduler(BaseScheduler):
    def __init__(self, band_count: int = BAND_COUNT, lambda_param: float = 0.3):
        super().__init__(band_count, provenance_note=f"Whittle-Inspired lambda={lambda_param}")
        self.band_count = band_count
        self.lambda_param = lambda_param
        self.belief = np.ones(band_count, dtype=np.float64) * 0.5
        self.local_t = 0
        self.rng = np.random.default_rng()

    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.belief.fill(0.5)
        self.local_t = 0

    def select_action(self, observation_history, current_time_slot: int) -> int:
        if self.local_t < self.band_count:
            return self.local_t
        scores = self.belief * (1.0 - self.lambda_param)
        scores += self.rng.uniform(0.0, 1e-12, self.band_count)
        return int(np.argmax(scores))

    def update(self, action: int, observation: dict):
        hit = bool(observation.get("hit", False))
        Pd, Pfa = 0.9, 0.1
        prior = self.belief[action]
        if hit:
            likelihood_h1, likelihood_h0 = Pd, Pfa
        else:
            likelihood_h1, likelihood_h0 = 1.0 - Pd, 1.0 - Pfa
        numerator = likelihood_h1 * prior
        denominator = likelihood_h1 * prior + likelihood_h0 * (1.0 - prior)
        posterior = numerator / denominator if denominator > 1e-12 else 0.5
        self.belief[action] = np.clip(posterior, 0.01, 0.99)
        self.local_t += 1


class RLessUCBScheduler(BaseScheduler):
    def __init__(self, band_count: int = BAND_COUNT, alpha: float = 1.5):
        super().__init__(band_count, provenance_note=f"R-less-UCB alpha={alpha}")
        self.band_count = band_count
        self.alpha = alpha
        self.counts = np.zeros(band_count, dtype=np.int64)
        self.rewards = np.zeros(band_count, dtype=np.float64)
        self.last_visit = np.full(band_count, -np.inf, dtype=np.float64)
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
            return self.local_t
        mean_rewards = np.zeros(self.band_count)
        visited = self.counts > 0
        mean_rewards[visited] = self.rewards[visited] / self.counts[visited]
        time_since_visit = np.maximum(0.0, current_time_slot - self.last_visit)
        scores = np.full(self.band_count, -np.inf)
        scores[visited] = mean_rewards[visited] + np.sqrt(self.alpha * np.log(time_since_visit[visited] + 1) / self.counts[visited])
        unvisited = self.counts == 0
        if np.any(unvisited):
            return int(np.flatnonzero(unvisited)[0])
        scores += self.rng.uniform(0.0, 1e-12, self.band_count)
        return int(np.argmax(scores))

    def update(self, action: int, observation: dict):
        hit = float(observation.get("hit", False))
        global_t = int(observation.get("global_t", self.local_t))
        self.counts[action] += 1
        self.rewards[action] += hit
        self.last_visit[action] = global_t
        self.local_t += 1


class ARPScheduler(BaseScheduler):
    def __init__(self, band_count: int = BAND_COUNT, wA: float = 0.6, wR: float = 0.25, wU: float = 0.15):
        super().__init__(band_count, provenance_note=f"ARP wA={wA}, wR={wR}, wU={wU}")
        self.band_count = band_count
        self.wA, self.wR, self.wU = wA, wR, wU
        self.hit_history = [[] for _ in range(band_count)]
        self.hit_times = [[] for _ in range(band_count)]
        self.ema_hit_rate = np.ones(band_count) * 0.5
        self.recency = np.zeros(band_count)
        self.uncertainty = np.ones(band_count) * 0.5
        self.local_t = 0
        self.rng = np.random.default_rng()

    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.hit_history = [[] for _ in range(self.band_count)]
        self.hit_times = [[] for _ in range(self.band_count)]
        self.ema_hit_rate.fill(0.5)
        self.recency.fill(0.0)
        self.uncertainty.fill(0.5)
        self.local_t = 0

    def select_action(self, observation_history, current_time_slot: int) -> int:
        if self.local_t < self.band_count:
            return self.local_t
        scores = self.wA * self.ema_hit_rate + self.wR * self.recency + self.wU * self.uncertainty
        scores += self.rng.uniform(0.0, 1e-12, self.band_count)
        return int(np.argmax(scores))

    def update(self, action: int, observation: dict):
        hit = bool(observation.get("hit", False))
        global_t = int(observation.get("global_t", self.local_t))
        self.local_t += 1
        hit_binary = 1.0 if hit else 0.0
        self.hit_history[action].append(hit_binary)
        if hit:
            self.hit_times[action].append(global_t)
        if len(self.hit_history[action]) > 40:
            self.hit_history[action] = self.hit_history[action][-40:]
        if len(self.hit_times[action]) > 40:
            self.hit_times[action] = self.hit_times[action][-40:]
        self.ema_hit_rate[action] = 0.3 * hit_binary + 0.7 * self.ema_hit_rate[action]
        if self.hit_times[action]:
            dt = global_t - self.hit_times[action][-1]
            self.recency[action] = 1.0 / (dt + 1.0)
        else:
            self.recency[action] = 0.0
        if len(self.hit_history[action]) > 5:
            self.uncertainty[action] = np.std(self.hit_history[action][-20:])


# =============================================================================
# REGRET-BOUND BALANCED ENSEMBLE (INLINE)
# =============================================================================

class RegretBalancedEnsemble(BaseScheduler):
    def __init__(self, band_count: int = BAND_COUNT, horizon: int = HORIZON,
                 delta: float = DELTA, empirical_regret_bounds: dict = None):
        super().__init__(band_count, provenance_note="Regret-Balanced Ensemble")
        self.band_count = band_count
        self.horizon = horizon
        self.delta = delta
        self.experts = {
            "UCB1": UCB1Scheduler(band_count),
            "Whittle-Inspired": WhittleInspiredScheduler(band_count),
            "RLessUCB": RLessUCBScheduler(band_count),
            "ARP": ARPScheduler(band_count),
        }
        self.expert_names = list(self.experts.keys())
        self.n_experts = len(self.expert_names)
        self.regret_bounds = empirical_regret_bounds or {
            "UCB1": 50.0, "Whittle-Inspired": 80.0, "RLessUCB": 100.0, "ARP": 120.0,
        }
        self.expert_counts = np.zeros(self.n_experts, dtype=np.int64)
        self.expert_cumulative_rewards = np.zeros(self.n_experts, dtype=np.float64)
        self.expert_cumulative_regret = np.zeros(self.n_experts, dtype=np.float64)
        self.active = np.ones(self.n_experts, dtype=bool)
        log_term = np.log(float(self.horizon) * self.n_experts / self.delta)
        self.C = np.sqrt(2.0 * log_term) * np.array([self.regret_bounds[name] for name in self.expert_names])
        self.global_t = 0
        self.last_expert_idx = None
        self.rng = np.random.default_rng()

    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        for idx, expert in enumerate(self.experts.values()):
            expert.reset(seed=seed + 1000 * idx)
        self.expert_counts.fill(0)
        self.expert_cumulative_rewards.fill(0.0)
        self.expert_cumulative_regret.fill(0.0)
        self.active.fill(True)
        self.global_t = 0
        self.last_expert_idx = None

    def compute_ucb(self, expert_idx: int) -> float:
        n = int(self.expert_counts[expert_idx])
        if n == 0:
            return np.inf
        mean_reward = self.expert_cumulative_rewards[expert_idx] / n
        confidence = self.C[expert_idx] / np.sqrt(n)
        return mean_reward + confidence

    def select_expert(self) -> int:
        unvisited = np.flatnonzero(self.active & (self.expert_counts == 0))
        if len(unvisited) > 0:
            return int(self.rng.choice(unvisited))
        ucb_values = np.full(self.n_experts, -np.inf)
        for i in range(self.n_experts):
            if self.active[i]:
                ucb_values[i] = self.compute_ucb(i)
        max_value = np.max(ucb_values)
        candidates = np.flatnonzero(np.isclose(ucb_values, max_value, rtol=1e-12))
        if len(candidates) == 1:
            return int(candidates[0])
        return int(self.rng.choice(candidates))

    def check_elimination(self, expert_idx: int):
        n = int(self.expert_counts[expert_idx])
        if n == 0:
            return
        empirical_regret = self.expert_cumulative_regret[expert_idx]
        regret_bound = self.regret_bounds[self.expert_names[expert_idx]] * np.sqrt(self.global_t / self.horizon)
        if empirical_regret > regret_bound:
            active_count = int(np.sum(self.active))
            if active_count > 1:
                self.active[expert_idx] = False

    def select_action(self, observation_history, current_time_slot: int) -> int:
        self.global_t = int(current_time_slot)
        expert_idx = self.select_expert()
        self.last_expert_idx = expert_idx
        expert_name = self.expert_names[expert_idx]
        expert = self.experts[expert_name]
        return int(expert.select_action(observation_history, current_time_slot))

    def update(self, action: int, observation: dict):
        if self.last_expert_idx is None:
            raise RuntimeError("update called without select_action")
        expert_idx = self.last_expert_idx
        expert_name = self.expert_names[expert_idx]
        expert = self.experts[expert_name]
        expert.update(action, observation)
        reward = float(observation.get("hit", False))
        n = int(self.expert_counts[expert_idx]) + 1
        self.expert_counts[expert_idx] = n
        self.expert_cumulative_rewards[expert_idx] += reward
        best_mean_reward = 0.0
        for i in range(self.n_experts):
            if self.expert_counts[i] > 0:
                mean_i = self.expert_cumulative_rewards[i] / self.expert_counts[i]
                best_mean_reward = max(best_mean_reward, mean_i)
        instant_regret = best_mean_reward - reward
        self.expert_cumulative_regret[expert_idx] += instant_regret
        self.check_elimination(expert_idx)
        self.last_expert_idx = None


# =============================================================================
# JSON SANITIZATION
# =============================================================================

def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    return obj


# =============================================================================
# LOAD EMPIRICAL REGRET BOUNDS
# =============================================================================

def load_empirical_regret_bounds(params_file: Path):
    with open(params_file, "r", encoding="utf-8") as f:
        params = json.load(f)
    experts = params.get("experts", {})
    regret_bounds = {}
    for expert_name, expert_params in experts.items():
        envelope_candidates = expert_params.get("envelope_candidates", [])
        alpha_05_candidates = [c for c in envelope_candidates if abs(c.get("alpha", 0) - 0.5) < 0.01]
        if alpha_05_candidates:
            empirical_C = alpha_05_candidates[0].get("empirical_C", 100.0)
            regret_bound = empirical_C * (HORIZON ** 0.5)
        else:
            regret_bound = 100.0
        regret_bounds[expert_name] = float(regret_bound)
    return regret_bounds


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_regret_balanced(test_stare_dir: Path, test_scan_dir: Path,
                             empirical_regret_bounds: dict, n_episodes: int = N_TEST_EPISODES):
    print(f"\n{'='*80}")
    print("Evaluating: Regret-Balanced Ensemble (ALL 34+ METRICS)")
    print(f"{'='*80}")
    sim_config = SimulationConfig(
        band_count=BAND_COUNT, time_slots=HORIZON, dwell_time_ms=50.0,
        receiver_ibw_mhz=500.0, total_spectrum_mhz=18000.0,
        detection_probability=0.9, false_alarm_probability=0.01,
    )
    metrics_config = MetricsConfig()
    stare_files = sorted(test_stare_dir.glob("config_*.h5"))
    paired = [f for f in stare_files if (test_scan_dir / f.name).exists()]
    if len(paired) < n_episodes:
        raise RuntimeError(f"Expected at least {n_episodes} paired test episodes, found {len(paired)}.")
    paired = paired[:n_episodes]
    print(f"Paired test episodes available: {len(paired)}")
    print(f"Using first {n_episodes} paired episodes.")
    print(f"First episode: {paired[0].name}")
    print(f"Last episode:  {paired[n_episodes - 1].name}")
    all_episode_metrics = []
    failures = []
    for ep_idx, stare_file in enumerate(paired):
        scan_file = test_scan_dir / stare_file.name
        try:
            env = TSRDStareEnvironment.from_stare_mode(
                stare_file=str(stare_file), scan_file=str(scan_file), sim_config=sim_config,
            )
            metrics_engine = TSRDMetricsEngine(
                config=metrics_config, stare_mode_file=str(stare_file), scan_mode_file=str(scan_file),
            )
            scheduler = RegretBalancedEnsemble(
                band_count=BAND_COUNT, horizon=HORIZON, delta=DELTA,
                empirical_regret_bounds=empirical_regret_bounds,
            )
            paired_result = run_paired_episodes(
                env=env, schedulers={"Regret-Balanced": scheduler}, seed=ep_idx,
            )
            trajectory = paired_result["Regret-Balanced"].trajectory
            metrics = metrics_engine.record_result(
                episode_id=ep_idx, seed=ep_idx, scheduler_name="Regret-Balanced",
                scenario_config={"band_count": BAND_COUNT, "time_slots": HORIZON},
                trajectory=trajectory, truth_grid=env.hidden_truth,
            )
            if ep_idx == 0:
                print("\nTSRD metric structure (first episode):")
                print(json.dumps(json_safe(metrics), indent=2, default=str)[:3000] + "...")
            all_episode_metrics.append(metrics)
            if (ep_idx + 1) % 50 == 0:
                print(f"  {ep_idx + 1}/{len(paired)} episodes completed")
        except Exception as exc:
            failures.append({"episode": ep_idx, "config": stare_file.stem, "error": repr(exc)})
            print(f"  Episode {ep_idx} FAILED: {exc}")
    if failures:
        raise RuntimeError(f"Regret-Balanced failed on {len(failures)} episodes:\n{failures}")
    if len(all_episode_metrics) != n_episodes:
        raise RuntimeError(f"Regret-Balanced: expected {n_episodes} successful episodes, got {len(all_episode_metrics)}.")
    return all_episode_metrics


# =============================================================================
# AGGREGATE ALL METRICS
# =============================================================================

def aggregate_all_metrics(all_episode_metrics: list) -> dict:
    aggregated = {}
    pd_values, pfa_values, intercept_rates, intercept_times, hit_rates = [], [], [], [], []
    for m in all_episode_metrics:
        detection = m.get("comprehensive_detection", {})
        pd = detection.get("true_pd", np.nan)
        pfa = detection.get("true_pfa", np.nan)
        pd_values.append(float(pd))
        pfa_values.append(float(pfa))
        discovery = m.get("discovery_metrics", {})
        emitters_considered = discovery.get("emitters_considered", 1)
        emitters_intercepted = discovery.get("emitters_intercepted", 0)
        intercept_rate = emitters_intercepted / max(emitters_considered, 1)
        intercept_time_raw = discovery.get("mean_first_intercept_time_slots", None)
        intercept_time = float(intercept_time_raw) if intercept_time_raw is not None else np.nan
        intercept_rates.append(intercept_rate)
        intercept_times.append(intercept_time)
        monitoring = m.get("monitoring_metrics", {})
        hit_rate_raw = monitoring.get("average_coverage_fraction", None)
        hit_rate = float(hit_rate_raw) if hit_rate_raw is not None else np.nan
        hit_rates.append(hit_rate)
    aggregated["comprehensive_detection"] = {
        "true_pd": {"mean": float(np.mean(pd_values)), "std": float(np.std(pd_values))},
        "true_pfa": {"mean": float(np.mean(pfa_values)), "std": float(np.std(pfa_values))},
    }
    aggregated["discovery_metrics"] = {
        "interception_probability": {"mean": float(np.mean(intercept_rates)), "std": float(np.std(intercept_rates))},
        "mean_first_intercept_time_slots": {
            "mean": float(np.mean([x for x in intercept_times if np.isfinite(x)])),
            "std": float(np.std([x for x in intercept_times if np.isfinite(x)]))
        } if any(np.isfinite(x) for x in intercept_times) else None,
    }
    aggregated["monitoring_metrics"] = {
        "average_coverage_fraction": {"mean": float(np.mean(hit_rates)), "std": float(np.std(hit_rates))},
    }
    return aggregated


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("VYAPTI — REGRET-BOUND BALANCED ENSEMBLE — ALL 34+ METRICS")
    print("=" * 90)
    print()
    print("Extracting ALL available TSRD metrics, not just Pd/Pfa.")
    print()
    if not EMPIRICAL_PARAMS_FILE.exists():
        raise RuntimeError(f"Empirical params file not found: {EMPIRICAL_PARAMS_FILE}")
    print("📊 Loading empirical regret bounds from V3 analysis...")
    empirical_regret_bounds = load_empirical_regret_bounds(EMPIRICAL_PARAMS_FILE)
    for name, bound in empirical_regret_bounds.items():
        print(f"   {name:20s}: R = {bound:.2f}")
    print("\n📥 Downloading TSRD test set...")
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
    print(f"✓ Test scan: {test_scan_dir}")
    print()
    all_episode_metrics = evaluate_regret_balanced(
        test_stare_dir, test_scan_dir, empirical_regret_bounds, n_episodes=N_TEST_EPISODES,
    )
    aggregated = aggregate_all_metrics(all_episode_metrics)
    test_stare_files = sorted(test_stare_dir.glob("config_*.h5"))
    paired_test_files = [f for f in test_stare_files if (test_scan_dir / f.name).exists()][:N_TEST_EPISODES]
    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "Regret-Balanced Ensemble - ALL 34+ TSRD METRICS",
        "test_split": "stare/test_stare + scan/test_scan",
        "n_episodes": N_TEST_EPISODES,
        "horizon": HORIZON,
        "band_count": BAND_COUNT,
        "test_episode_files": [f.name for f in paired_test_files],
        "empirical_regret_bounds": empirical_regret_bounds,
        "aggregated_metrics": aggregated,
        "all_episode_metrics": all_episode_metrics,
        "scientific_note": "ALL 34+ TSRD metrics saved for each episode.",
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(json_safe(output), f, indent=2)
    print()
    print("=" * 90)
    print("REGRET-BOUND BALANCED ENSEMBLE RESULTS (ALL 34+ METRICS)")
    print("=" * 90)
    print()
    print("Aggregated metrics:")
    for category, metrics in aggregated.items():
        print(f"\n{category}:")
        for metric, stats in metrics.items():
            if stats is not None:
                print(f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f}")
            else:
                print(f"  {metric}: N/A")
    print()
    print(f"💾 Saved ALL metrics: {OUTPUT_FILE}")
    print("=" * 90)


if __name__ == "__main__":
    main()
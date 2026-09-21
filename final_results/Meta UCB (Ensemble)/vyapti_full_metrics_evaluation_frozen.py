#!/usr/bin/env python3
"""
VYAPTI — FULL METRICS EVALUATION WITH META-UCB COMBINER (FROZEN)

Evaluates the Meta-UCB Bandit Combiner (and other schedulers) using the
complete TSRD metrics pipeline (Pd, Pfa, intercept rate, etc.) on the
held-out test set.

This ensures fair comparison across all schedulers using the same
evaluation harness.

Protocol:
  TRAIN (stare/train_stare + scan/train_scan)
    ↓
  V3 empirical C(alpha) analysis
    ↓
  Predeclare: alpha=0.5, R_scale=1.0
    ↓
  NO VALIDATION — PARAMETERS FROZEN AFTER TRAIN/V3 ANALYSIS
    ↓
  FREEZE configuration
    ↓
  250 TEST EPISODES (stare/test_stare + scan/test_scan)
    ↓
  Full TSRD metrics via TSRDMetricsEngine.record_result()

Note: Full TSRD metrics are recorded per episode; selected headline metrics
are aggregated for the comparison table.
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

# Import Meta-UCB combiner and base experts
from vyapti_meta_ucb_combiner_frozen import (
    MetaUCBBanditCombiner,
    UCB1Scheduler,
    WhittleInspiredScheduler,
    RLessUCBScheduler,
    ARPScheduler,
    BAND_COUNT,
    HORIZON,
    ALPHA_USED,
    R_EMPIRICAL_SCALE,
    DELTA,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

EMPIRICAL_PARAMS_FILE = Path("/kaggle/working/base_expert_regret_analysis_V3.json")
OUTPUT_FILE = Path("/kaggle/working/vyapti_full_metrics_results.json")
N_TEST_EPISODES = 250


# =============================================================================
# JSON SANITIZATION
# =============================================================================

def json_safe(obj):
    """
    Sanitize non-finite floats for JSON serialization.
    
    Replaces NaN/Inf with None to produce valid JSON.
    """
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
# SCHEDULER FACTORY
# =============================================================================

def create_scheduler(scheduler_name: str, empirical_params: dict = None):
    """
    Create a scheduler instance by name.
    """
    if scheduler_name == "Meta-UCB":
        if empirical_params is None:
            raise ValueError("empirical_params required for Meta-UCB")
        return MetaUCBBanditCombiner(
            band_count=BAND_COUNT,
            expert_params=empirical_params,
            horizon=HORIZON,
            alpha_used=ALPHA_USED,
            delta=DELTA,
            r_empirical_scale=R_EMPIRICAL_SCALE,
        )
    elif scheduler_name == "UCB1":
        return UCB1Scheduler(band_count=BAND_COUNT)
    elif scheduler_name == "Whittle-Inspired":
        return WhittleInspiredScheduler(band_count=BAND_COUNT)
    elif scheduler_name == "RLessUCB":
        return RLessUCBScheduler(band_count=BAND_COUNT)
    elif scheduler_name == "ARP":
        return ARPScheduler(band_count=BAND_COUNT)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


# =============================================================================
# METRICS AGGREGATION
# =============================================================================

def aggregate_tsr_metrics(
    scheduler_name: str,
    metrics_list: list,
) -> dict:
    """
    Aggregate per-episode TSRD metrics into summary statistics.
    
    Extracts Pd, Pfa from TSRDMetricsEngine.record_result() output structure.
    Other metrics are aggregated if present.
    """
    
    pd_values = []
    pfa_values = []
    intercept_rates = []
    intercept_times = []
    hit_rates = []
    
    for m in metrics_list:
        # Extract from comprehensive_detection (verified structure)
        detection = m.get("comprehensive_detection", {})
        
        pd = detection.get("true_pd", np.nan)
        pfa = detection.get("true_pfa", np.nan)
        
        # Extract intercept metrics if available (structure to be verified)
        intercept = m.get("intercept_analysis", {})
        intercept_rate = intercept.get("intercept_rate", np.nan)
        intercept_time = intercept.get("mean_intercept_time", np.nan)
        
        # Hit rate (structure to be verified)
        hit_rate = m.get("hit_rate", np.nan)
        
        pd_values.append(float(pd))
        pfa_values.append(float(pfa))
        
        if np.isfinite(intercept_rate):
            intercept_rates.append(intercept_rate)
        if np.isfinite(intercept_time):
            intercept_times.append(intercept_time)
        if np.isfinite(hit_rate):
            hit_rates.append(hit_rate)
    
    pd_values = np.asarray(pd_values, dtype=np.float64)
    pfa_values = np.asarray(pfa_values, dtype=np.float64)
    
    # Check for non-finite values
    if np.any(~np.isfinite(pd_values)):
        raise RuntimeError(f"{scheduler_name}: non-finite Pd detected.")
    
    if np.any(~np.isfinite(pfa_values)):
        raise RuntimeError(f"{scheduler_name}: non-finite Pfa detected.")
    
    summary = {
        "scheduler_name": scheduler_name,
        "n_episodes": len(metrics_list),
        
        "Pd": float(np.mean(pd_values)),
        "Pd_std": float(np.std(pd_values)),
        
        "Pfa": float(np.mean(pfa_values)),
        "Pfa_std": float(np.std(pfa_values)),
    }
    
    if intercept_rates:
        intercept_rates = np.asarray(intercept_rates, dtype=np.float64)
        summary["intercept_rate"] = float(np.mean(intercept_rates))
        summary["intercept_rate_std"] = float(np.std(intercept_rates))
    
    if intercept_times:
        intercept_times = np.asarray(intercept_times, dtype=np.float64)
        summary["intercept_time_mean"] = float(np.mean(intercept_times))
        summary["intercept_time_std"] = float(np.std(intercept_times))
    
    if hit_rates:
        hit_rates = np.asarray(hit_rates, dtype=np.float64)
        summary["hit_rate"] = float(np.mean(hit_rates))
        summary["hit_rate_std"] = float(np.std(hit_rates))
    
    return summary


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_scheduler(
    scheduler_name: str,
    test_stare_dir: Path,
    test_scan_dir: Path,
    empirical_params: dict = None,
    n_episodes: int = N_TEST_EPISODES,
):
    """
    Evaluate a scheduler on test episodes using the full TSRD metrics pipeline.
    
    For each episode:
      1. Create fresh environment
      2. Create a fresh scheduler instance
      3. Run paired episode
      4. Record result with TSRDMetricsEngine
      5. Collect metrics
    """
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {scheduler_name}")
    print(f"{'='*80}")
    
    sim_config = SimulationConfig(
        band_count=BAND_COUNT,
        time_slots=HORIZON,
        dwell_time_ms=50.0,
        receiver_ibw_mhz=500.0,
        total_spectrum_mhz=18000.0,
    )
    
    metrics_config = MetricsConfig()
    
    # Get test files
    stare_files = sorted(test_stare_dir.glob("config_*.h5"))
    
    paired = [
        f for f in stare_files
        if (test_scan_dir / f.name).exists()
    ]
    
    if len(paired) < n_episodes:
        raise RuntimeError(
            f"Expected at least {n_episodes} paired test episodes, "
            f"found {len(paired)}."
        )
    
    paired = paired[:n_episodes]
    
    print(f"Paired test episodes available: {len(paired)}")
    print(f"Using first {n_episodes} paired episodes.")
    print(f"First episode: {paired[0].name}")
    print(f"Last episode:  {paired[n_episodes - 1].name}")
    
    per_episode_metrics = []
    failures = []
    
    for ep_idx, stare_file in enumerate(paired):
        scan_file = test_scan_dir / stare_file.name
        
        try:
            # Create environment
            env = TSRDStareEnvironment.from_stare_mode(
                stare_file=str(stare_file),
                scan_file=str(scan_file),
                sim_config=sim_config,
            )
            
            # Create metrics engine
            metrics_engine = TSRDMetricsEngine(
                config=metrics_config,
                stare_mode_file=str(stare_file),
                scan_mode_file=str(scan_file),
            )
            
            # Create a fresh scheduler instance
            scheduler = create_scheduler(
                scheduler_name,
                empirical_params,
            )
            
            # Run paired episode
            paired_result = run_paired_episodes(
                env=env,
                schedulers={scheduler_name: scheduler},
                seed=ep_idx,
            )
            
            # Extract trajectory
            trajectory = paired_result[scheduler_name].trajectory
            
            # Record result with metrics engine
            metrics = metrics_engine.record_result(
                episode_id=ep_idx,
                seed=ep_idx,
                scheduler_name=scheduler_name,
                scenario_config={
                    "band_count": BAND_COUNT,
                    "time_slots": HORIZON,
                },
                trajectory=trajectory,
                truth_grid=env.hidden_truth,
            )
            
            # Verify metric structure on first episode
            if ep_idx == 0:
                print("\nTSRD metric structure (first episode):")
                print(json.dumps(json_safe(metrics), indent=2, default=str)[:2000] + "...")
            
            per_episode_metrics.append(metrics)
            
            if (ep_idx + 1) % 50 == 0:
                print(f"  {ep_idx + 1}/{len(paired)} episodes completed")
        
        except Exception as exc:
            failures.append({
                "episode": ep_idx,
                "config": stare_file.stem,
                "error": repr(exc),
            })
            print(f"  Episode {ep_idx} FAILED: {exc}")
    
    # Check for failures
    if failures:
        raise RuntimeError(
            f"{scheduler_name} failed on {len(failures)} episodes:\n{failures}"
        )
    
    if len(per_episode_metrics) != n_episodes:
        raise RuntimeError(
            f"{scheduler_name}: expected {n_episodes} successful episodes, "
            f"got {len(per_episode_metrics)}."
        )
    
    # Aggregate metrics
    summary = aggregate_tsr_metrics(scheduler_name, per_episode_metrics)
    
    print(f"  Pd:              {summary['Pd']:.6f} ± {summary['Pd_std']:.6f}")
    print(f"  Pfa:             {summary['Pfa']:.6f} ± {summary['Pfa_std']:.6f}")
    if "intercept_rate" in summary:
        print(f"  Intercept rate:  {summary['intercept_rate']:.6f} ± {summary['intercept_rate_std']:.6f}")
    if "intercept_time_mean" in summary:
        print(f"  Intercept time:  {summary['intercept_time_mean']:.2f} ± {summary['intercept_time_std']:.2f}")
    if "hit_rate" in summary:
        print(f"  Hit rate:        {summary['hit_rate']:.6f} ± {summary['hit_rate_std']:.6f}")
    
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("VYAPTI — FULL METRICS EVALUATION (FROZEN)")
    print("=" * 90)
    print()
    print("Evaluating all schedulers with complete TSRD metrics pipeline.")
    print("Full TSRD metrics are recorded per episode; selected headline")
    print("metrics are aggregated for the comparison table.")
    print()
    
    # Check empirical params file
    if not EMPIRICAL_PARAMS_FILE.exists():
        raise RuntimeError(
            f"Empirical params file not found: {EMPIRICAL_PARAMS_FILE}\n"
            "Run vyapti_base_expert_regret_analysis_V3_frozen.py first."
        )
    
    # Load empirical parameters
    with open(EMPIRICAL_PARAMS_FILE, "r", encoding="utf-8") as f:
        empirical_params_full = json.load(f)
    
    empirical_params = empirical_params_full.get("experts", {})
    
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
    
    # Schedulers to evaluate
    schedulers = [
        "UCB1",
        "Whittle-Inspired",
        "RLessUCB",
        "ARP",
        "Meta-UCB",
    ]
    
    results = []
    
    for scheduler_name in schedulers:
        result = evaluate_scheduler(
            scheduler_name,
            test_stare_dir,
            test_scan_dir,
            empirical_params if scheduler_name == "Meta-UCB" else None,
            n_episodes=N_TEST_EPISODES,
        )
        results.append(result)
    
    # Create comparison table
    print("\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    print()
    print(f"{'Scheduler':20s} {'Pd':12s} {'Pfa':12s} {'Intercept Rate':16s} {'Hit Rate':12s}")
    print("-" * 75)
    
    for result in results:
        pd_str = f"{result['Pd']:.6f} ± {result['Pd_std']:.4f}"
        pfa_str = f"{result['Pfa']:.6f} ± {result['Pfa_std']:.4f}"
        int_rate_str = f"{result.get('intercept_rate', np.nan):.6f}"
        hit_str = f"{result.get('hit_rate', np.nan):.6f}"
        
        print(
            f"{result['scheduler_name']:20s} "
            f"{pd_str:12s} "
            f"{pfa_str:12s} "
            f"{int_rate_str:16s} "
            f"{hit_str:12s}"
        )
    
    # CORRECTION: Use exact same paired selection logic as evaluation
    test_stare_files = sorted(test_stare_dir.glob("config_*.h5"))
    
    paired_test_files = [
        f for f in test_stare_files
        if (test_scan_dir / f.name).exists()
    ]
    
    if len(paired_test_files) < N_TEST_EPISODES:
        raise RuntimeError(
            f"Expected at least {N_TEST_EPISODES} paired test episodes, "
            f"found {len(paired_test_files)}."
        )
    
    paired_test_files = paired_test_files[:N_TEST_EPISODES]
    
    # Save results (with JSON sanitization)
    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "Full TSRD metrics on held-out test set",
        "test_split": "stare/test_stare + scan/test_scan",
        "n_episodes": N_TEST_EPISODES,
        "horizon": HORIZON,
        "band_count": BAND_COUNT,
        "test_episode_files": [f.name for f in paired_test_files],
        "meta_ucb_configuration": {
            "alpha_used": ALPHA_USED,
            "alpha_selection_policy": "predeclared_fixed_alpha",
            "delta": DELTA,
            "r_empirical_scale": R_EMPIRICAL_SCALE,
            "r_method": "first_condition_scale_only",
            "empirical_params_file": str(EMPIRICAL_PARAMS_FILE),
        },
        "schedulers": results,
        "comparison_table": [
            {
                "scheduler": r["scheduler_name"],
                "Pd": r["Pd"],
                "Pd_std": r["Pd_std"],
                "Pfa": r["Pfa"],
                "Pfa_std": r["Pfa_std"],
                "intercept_rate": r.get("intercept_rate", np.nan),
                "intercept_time_mean": r.get("intercept_time_mean", np.nan),
                "hit_rate": r.get("hit_rate", np.nan),
            }
            for r in results
        ],
        "scientific_note": (
            "All schedulers evaluated on the same 250-episode held-out test set "
            "using the identical TSRD metrics pipeline (TSRDMetricsEngine.record_result). "
            "Full TSRD metrics are recorded per episode; selected headline metrics are "
            "aggregated for the comparison table. Meta-UCB is an empirical adaptation "
            "of Cutkosky Algorithm 1; theoretical guarantee not claimed. Each episode "
            "uses a fresh scheduler instance. No validation set was used; parameters "
            "were frozen after V3 training analysis."
        ),
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(json_safe(output), f, indent=2)
    
    print()
    print(f"💾 Saved: {OUTPUT_FILE}")
    print("=" * 90)


if __name__ == "__main__":
    main()
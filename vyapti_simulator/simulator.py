#!/usr/bin/env python3
"""
PS26055 simulator CLI — executable entry point.

Supersedes the prior hardcoded loop. The canonical episode runner is now
`core.episode.run_episode`, which the conformance suite, the experiments package,
and this CLI all call. There is no other loop.

Usage
-----
Single paired comparison:
    python -m vyapti_simulator.simulator \\
        --method vyapti_simulator.algorithms.bandit.ucb:UCB1Scheduler \\
        --baseline vyapti_simulator.qualification.probes:RoundRobinProbe \\
        --bands 10 --slots 1000 --seed 42

Run conformance first:
    python -m vyapti_simulator.qualification.conformance --scheduler mod:Class

Full experiment protocol:
    See vyapti_simulator.experiments.experiment_runner
"""

from __future__ import annotations
import argparse
import importlib
import json
import sys
import time
from typing import Any, Dict

from .core.environment import (
    PS26055Environment, SimulationConfig, EmitterConfig, EmitterBehaviorType,
)
from .core.episode import run_paired_episodes
from .core.metrics import MetricsEngine, MetricsConfig
from .qualification import default_conformance_scenario


# =====================================================================
# Locked TSRD grid (Phase 5 of the plan)
# 500 MHz / 0-18 000 MHz / 36 bands; 50 ms slot; 600 slots = 30 s
# =====================================================================
TSRD_BAND_COUNT = 36
TSRD_TOTAL_SPECTRUM_MHZ = 18000.0
TSRD_RECEIVER_IBW_MHZ = 500.0
TSRD_DWELL_TIME_MS = 50.0
TSRD_TIME_SLOTS = 600


def _build_tsrd_config() -> SimulationConfig:
    """Return a SimulationConfig matching the locked TSRD grid."""
    return SimulationConfig(
        band_count=TSRD_BAND_COUNT,
        total_spectrum_mhz=TSRD_TOTAL_SPECTRUM_MHZ,
        receiver_ibw_mhz=TSRD_RECEIVER_IBW_MHZ,
        dwell_time_ms=TSRD_DWELL_TIME_MS,
        time_slots=TSRD_TIME_SLOTS,
        # The plan preserves the existing retune + detector defaults.
        retune_time_ms=1.0,
        detection_probability=1.0,
        false_alarm_probability=0.0,
    )


def _load_class(spec: str):
    """Load a scheduler class from 'module.path:ClassName'."""
    if ":" not in spec:
        raise ValueError(
            f"Scheduler spec must be 'module:Class', got {spec!r}. "
            f"Example: vyapti_simulator.algorithms.bandit.ucb:UCB1Scheduler"
        )
    module_path, class_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_path)
    if not hasattr(module, class_name):
        raise AttributeError(f"{module_path} has no {class_name!r}")
    return getattr(module, class_name)


def run_comparison(
    method_spec: str,
    baseline_spec: str,
    band_count: int = 10,
    time_slots: int = 1000,
    seed: int = 42,
    method_kwargs: Dict[str, Any] = None,
    baseline_kwargs: Dict[str, Any] = None,
    tsrd_statistics_path: str = None,
) -> Dict[str, Any]:
    """
    Run one paired comparison: two schedulers at one seed against identical truth.

    Returns a report dict suitable for JSON serialization, containing:
      - episode results (actions, hits, diagnostics) for both schedulers
      - metrics computed from truth post-episode
      - replay signatures proving the pairing held

    If `tsrd_statistics_path` is provided, the locked TSRD grid is used
    (36 bands, 18 000 MHz, 50 ms slots, 600 slots = 30 s mission) and
    the emitter family is drawn from the TSRD aggregate statistics via
    `TSRDEmitterSampler`. The `band_count` and `time_slots` arguments
    are then ignored — the locked grid wins, because silently mixing
    grids would break the mapping's invariant.
    """
    MethodClass = _load_class(method_spec)
    BaselineClass = _load_class(baseline_spec)

    method_kwargs = method_kwargs or {}
    baseline_kwargs = baseline_kwargs or {}

    # ------------------------------------------------------------------
    # Scenario construction: TSRD-driven (Option A) OR synthetic default
    # ------------------------------------------------------------------
    tsrd_active = tsrd_statistics_path is not None
    if tsrd_active:
        # Lazy import: keep the TSRD import off the default CLI's
        # critical path so users without h5py etc. still run.
        from .tsrd.tsrd_emitter import TSRDEmitterSampler

        config = _build_tsrd_config()
        sampler = TSRDEmitterSampler(
            tsrd_statistics_path=tsrd_statistics_path,
            simulation_config=config,
            seed=seed,
        )
        emitters = sampler.build_emitter_configs()
        scenario_block = {
            "band_count": config.band_count,
            "time_slots": config.time_slots,
            "seed": seed,
            "emitter_count": len(emitters),
            "emitter_families": sorted({e.behavior.value for e in emitters}),
            "replay_signature": None,  # filled in after the run
            "tsrd_statistics_path": str(tsrd_statistics_path),
            "tsrd_source_h5_sha256": sampler.source_h5_sha256,
            "tsrd_n_active": sampler.n_active,
            "tsrd_n_silent": sampler.n_silent,
        }
    else:
        config = SimulationConfig(
            band_count=band_count,
            time_slots=time_slots,
            receiver_ibw_mhz=200.0,
            total_spectrum_mhz=float(band_count * 200),
            dwell_time_ms=10.0,
            retune_time_ms=1.0,
            detection_probability=1.0,
            false_alarm_probability=0.0,
        )
        emitters = default_conformance_scenario(band_count, time_slots)
        scenario_block = {
            "band_count": band_count,
            "time_slots": time_slots,
            "seed": seed,
            "emitter_count": len(emitters),
            "emitter_families": sorted({e.behavior.value for e in emitters}),
            "replay_signature": None,
            "tsrd_statistics_path": None,
        }

    env = PS26055Environment(config, emitters)

    schedulers = {
        "method": MethodClass(config.band_count, **method_kwargs),
        "baseline": BaselineClass(config.band_count, **baseline_kwargs),
    }

    results = run_paired_episodes(env, schedulers, seed, emitters)

    # Verify pairing held
    sigs = {name: res.replay_signature for name, res in results.items()}
    if len(set(sigs.values())) != 1:
        raise RuntimeError(
            f"Paired comparison broken: truth diverged between schedulers.\n"
            + "\n".join(f"  {name}: {sig}" for name, sig in sigs.items())
        )

    # Compute metrics
    metrics_engine = MetricsEngine(MetricsConfig())
    truth = env.hidden_truth
    metrics = {}
    for name, res in results.items():
        m = metrics_engine.record_result(
            episode_id=0,
            seed=seed,
            scheduler_name=name,
            scenario_config={},
            trajectory=res.trajectory,
            truth_grid=truth,
            receiver_accounting=res.receiver_accounting,
        )
        metrics[name] = m

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "simulator_version": "vyapti_simulator.core.episode v1",
        "frozen_protocol_version": "v1.0",
        "method": {
            "spec": method_spec,
            "name": results["method"].scheduler_name,
            "provenance": schedulers["method"].provenance_note,
            "slots_executed": results["method"].slots_executed,
            "predictions_emitted": results["method"].predictions_emitted,
            "audit_flags": results["method"].audit_flags,
            "diagnostics": results["method"].diagnostics,
            "metrics": metrics["method"],
        },
        "baseline": {
            "spec": baseline_spec,
            "name": results["baseline"].scheduler_name,
            "provenance": schedulers["baseline"].provenance_note,
            "slots_executed": results["baseline"].slots_executed,
            "predictions_emitted": results["baseline"].predictions_emitted,
            "audit_flags": results["baseline"].audit_flags,
            "diagnostics": results["baseline"].diagnostics,
            "metrics": metrics["baseline"],
        },
        "scenario": dict(
            scenario_block,
            replay_signature=results["method"].replay_signature,
        ),
        "pairing_verified": True,
        "note": (
            "Single-seed demonstration. Full experiments require sweeps over the "
            "paired seed set with Wilcoxon signed-rank tests and confidence intervals. "
            "See vyapti_simulator.experiments.experiment_runner."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m vyapti_simulator.simulator",
        description="PS26055 simulator CLI. Runs one paired comparison.",
    )
    ap.add_argument("--method", required=True, metavar="module:Class",
                    help="Scheduler under test, e.g. ...algorithms.bandit.ucb:UCB1Scheduler")
    ap.add_argument("--baseline", required=True, metavar="module:Class",
                    help="Reference scheduler, e.g. ...qualification.probes:RoundRobinProbe")
    ap.add_argument("--method-kwargs", default="{}", metavar="JSON",
                    help='Extra kwargs for method constructor, e.g. \'{"c": 1.5}\'')
    ap.add_argument("--baseline-kwargs", default="{}", metavar="JSON",
                    help="Extra kwargs for baseline constructor")
    ap.add_argument("--bands", type=int, default=10)
    ap.add_argument("--slots", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--tsrd-statistics", metavar="PATH", default=None,
        help=(
            "Path to a TSRD aggregate-statistics JSON produced by "
            "scripts/extract_tsrd_statistics.py. When provided, the "
            "simulator uses the locked TSRD grid (36 bands, 18 000 MHz, "
            "50 ms slots, 600 slots) and samples the emitter family "
            "from the JSON. The SHA-256 in data_provenance/manifest.json "
            "is verified on every load."
        ),
    )
    ap.add_argument("--json", metavar="PATH",
                    help="Write full report as JSON to this path")
    ap.add_argument("--compact", action="store_true",
                    help="Print compact summary instead of full JSON")
    args = ap.parse_args(argv)

    try:
        method_kwargs = json.loads(args.method_kwargs)
        baseline_kwargs = json.loads(args.baseline_kwargs)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in --method-kwargs or --baseline-kwargs: {exc}", file=sys.stderr)
        return 2

    try:
        report = run_comparison(
            args.method, args.baseline,
            band_count=args.bands, time_slots=args.slots, seed=args.seed,
            method_kwargs=method_kwargs, baseline_kwargs=baseline_kwargs,
            tsrd_statistics_path=args.tsrd_statistics,
        )
    except Exception as exc:
        print(f"Comparison failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"Report written to {args.json}")

    if args.compact:
        m_met = report["method"]["metrics"]
        b_met = report["baseline"]["metrics"]
        print("=" * 72)
        print(f"PAIRED COMPARISON — seed {report['scenario']['seed']}")
        print("=" * 72)
        print(f"  Method:   {report['method']['name']}")
        print(f"  Baseline: {report['baseline']['name']}")
        print(f"  Scenario: {report['scenario']['band_count']} bands, "
              f"{report['scenario']['time_slots']} slots, "
              f"{report['scenario']['emitter_count']} emitters")
        print()
        print("  Detection:")
        print(f"    Method   Pd={m_met['detection_metrics']['probability_of_detection']:.3f}  "
              f"Pfa={m_met['detection_metrics']['probability_of_false_alarm']:.3f}")
        print(f"    Baseline Pd={b_met['detection_metrics']['probability_of_detection']:.3f}  "
              f"Pfa={b_met['detection_metrics']['probability_of_false_alarm']:.3f}")
        print()
        print("  Discovery:")
        m_disc = m_met['discovery_metrics']
        b_disc = b_met['discovery_metrics']
        print(f"    Method   intercept_prob={m_disc['interception_probability']:.3f}  "
              f"mean_latency={m_disc.get('mean_intercept_latency_from_arrival_slots', 'N/A')}")
        print(f"    Baseline intercept_prob={b_disc['interception_probability']:.3f}  "
              f"mean_latency={b_disc.get('mean_intercept_latency_from_arrival_slots', 'N/A')}")
        print()
        print("  Coverage:")
        m_cov = m_met['monitoring_metrics']['coverage_detail']
        b_cov = b_met['monitoring_metrics']['coverage_detail']
        print(f"    Method   rolling_mean={m_cov['mean_rolling_coverage']:.3f}  "
              f"worst_staleness={m_cov['worst_case_band_staleness_slots']}  "
              f"never_visited={m_cov['bands_never_visited']}")
        print(f"    Baseline rolling_mean={b_cov['mean_rolling_coverage']:.3f}  "
              f"worst_staleness={b_cov['worst_case_band_staleness_slots']}  "
              f"never_visited={b_cov['bands_never_visited']}")
        print()
        print("  Monitoring:")
        m_mon = m_met['monitoring_metrics']
        b_mon = b_met['monitoring_metrics']
        print(f"    Method   post_discovery_capture={m_mon['post_discovery_capture_efficiency']}  "
              f"intercept_rate={m_mon['average_intercept_rate_per_slot']:.3f}")
        print(f"    Baseline post_discovery_capture={b_mon['post_discovery_capture_efficiency']}  "
              f"intercept_rate={b_mon['average_intercept_rate_per_slot']:.3f}")
        print()
        print("  Prediction (PS metrics 6 & 7):")
        for label, met in (("Method", m_met), ("Baseline", b_met)):
            pm = met['prediction_metrics']
            if pm['percentage_correct_predictions'] is None:
                print(f"    {label:<8} UNAVAILABLE — {pm['unavailable_reason'][:60]}...")
            else:
                eta = pm['average_intercept_time_error_slots']
                print(f"    {label:<8} correct={pm['percentage_correct_predictions']:.1f}%  "
                      f"brier={pm['brier_score']:.4f}  "
                      f"eta_error={eta if eta is None else f'{eta:.2f}'} slots")
        print("=" * 72)
    elif not args.json:
        print(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())

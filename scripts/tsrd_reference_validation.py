"""Run paired, fixed-receiver controls on the TSRD validation split only.

This is an integration reference, not a trained scheduler benchmark. It uses
recorded stare PDWs and scan-file geometry under the binary_v1 replay model.
The test split is intentionally unavailable from this script.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vyapti_simulator.algorithms.bandit.ucb import UCB1Scheduler
from vyapti_simulator.core.episode import run_episode
from vyapti_simulator.qualification.probes import RoundRobinProbe
from vyapti_simulator.tsrd.benchmark_protocol import _seed_for_file, _summarize
from vyapti_simulator.tsrd.corpus_loader import iter_tsr_replay_pairs
from vyapti_simulator.tsrd.replay_scorecard import score_recorded_replay
from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment


def run_validation(
    corpus_root: Path,
    *,
    seed: int = 42,
    detection_probability: float = 0.9,
    false_alarm_probability: float = 0.05,
    retune_time_ms: float = 1.0,
    limit: int | None = None,
) -> dict:
    """Score two controls with the same per-file receiver noise on validation."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    pairs = list(iter_tsr_replay_pairs(corpus_root, "val"))
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError("No TSRD validation pairs found")

    policies = {
        "round_robin_probe": RoundRobinProbe,
        "ucb1_reference": UCB1Scheduler,
    }
    scorecards = {name: [] for name in policies}
    per_file = []
    started = time.perf_counter()
    for index, (stare_file, scan_file) in enumerate(pairs, 1):
        env = TSRDStareEnvironment.from_stare_mode(
            str(stare_file), str(scan_file),
            detection_probability=detection_probability,
            false_alarm_probability=false_alarm_probability,
            retune_time_ms=retune_time_ms,
            receiver_profile="binary_v1",
        )
        receiver_seed = _seed_for_file(stare_file.stem, seed)
        file_result = {"config": stare_file.stem, "policies": {}}
        for name, policy_type in policies.items():
            policy = policy_type(env.n_bands)
            episode = run_episode(
                env, policy, seed=receiver_seed, scheduler_name=name
            )
            score = score_recorded_replay(env, episode.trajectory)
            scorecards[name].append(score)
            file_result["policies"][name] = {
                "true_detections": score["event_level"]["true_detections"],
                "false_alarms": score["event_level"]["false_alarms"],
                "illumination_interception_ratio": score["illumination"][
                    "illumination_interception_ratio"
                ],
                "unique_emitter_interception_rate": score[
                    "emitter_interception"
                ]["unique_emitter_interception_rate"],
                "conditional_pd": score["cell_level"]["conditional_pd"],
                "true_pfa": score["cell_level"]["true_pfa"],
            }
        per_file.append(file_result)
        if index % 25 == 0 or index == len(pairs):
            print(f"validation {index}/{len(pairs)}", flush=True)

    return {
        "scope": "Validation-only recorded-PDW reference; no test files opened",
        "corpus_root": str(corpus_root.resolve()),
        "receiver_profile": "binary_v1",
        "dwell_profile": "fixed_50_ms",
        "metric_contract_version": "tsrd_recorded_pulse_emitter_v3",
        "seed": seed,
        "receiver_options": {
            "detection_probability": detection_probability,
            "false_alarm_probability": false_alarm_probability,
            "retune_time_ms": retune_time_ms,
        },
        "files_evaluated": len(pairs),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "summaries": {
            name: _summarize(rows) for name, rows in scorecards.items()
        },
        "per_file": per_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.resolve() == args.corpus_root.resolve() or (
        args.corpus_root.resolve() in args.output.resolve().parents
    ):
        parser.error("Output must be outside the TSRD corpus")
    report = run_validation(args.corpus_root, seed=args.seed, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

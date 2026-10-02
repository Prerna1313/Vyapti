"""Screen TSRD reward definitions on causal baselines before model training.

This detects obvious ranking failures; finite baseline tests cannot prove a
reward safe for every environment or learned policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vyapti_simulator.core.episode import run_episode
from vyapti_simulator.core.scheduler_interface import BaseScheduler
from vyapti_simulator.qualification.probes import (
    RoundRobinProbe, StaticBandProbe, UniformRandomProbe,
)
from vyapti_simulator.system_b.tsrd.corpus_loader import iter_tsr_replay_pairs
from vyapti_simulator.system_b.tsrd.replay_scorecard import score_recorded_replay
from vyapti_simulator.system_b.tsrd.tsrd_environment import TSRDStareEnvironment


class ScreenProbe(BaseScheduler):
    """Additional causal pathologies and a simple positive-rate greedy policy."""

    def __init__(self, band_count: int, mode: str):
        super().__init__(band_count, f"[REWARD-SCREEN] {mode}")
        self.mode = mode

    def reset(self, seed: int, scenario_config: dict) -> None:
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0
        self.visits = np.zeros(self.band_count, dtype=int)
        self.positives = np.zeros(self.band_count, dtype=int)

    def select_action(self, observation_history: list[dict], current_time_slot: int) -> int:
        self._check_truth_leakage(observation_history)
        t = current_time_slot
        if self.mode == "two_band_ping_pong":
            band = t % min(2, self.band_count)
        elif self.mode == "periodic_two_slot_sweep":
            band = (t // 2) % self.band_count
        elif self.mode == "greedy_busy_band":
            if t < self.band_count or t % 20 == 0:
                band = (t if t < self.band_count else t // 20) % self.band_count
            else:
                mean_positive = (self.positives + 0.5) / (self.visits + 1.0)
                band = int(np.argmax(mean_positive))
        else:
            raise ValueError(f"Unknown screen probe: {self.mode}")
        return self._validate_action(band)

    def update(self, action: int, observation: dict) -> None:
        self._check_truth_leakage([observation])
        self.visits[action] += 1
        self.positives[action] += int(bool(observation["hit"]))
        self.observation_history.append(observation)
        self.steps_taken += 1


def _reward_values(score: dict, band_count: int, *,
                   coverage_weight: float | None,
                   discovery_bonus: float | None) -> dict[str, float]:
    decisions = score["decision_level"]["total_dwells"]
    positives = score["reward"]["detector_positive_total"]
    true_hits = score["event_level"]["true_detections"]
    false_alarms = score["event_level"]["false_alarms"]
    values = {
        "detector_positive": positives / decisions,
        "oracle_true_detection": true_hits / decisions,
        "oracle_false_alarm_aware": (true_hits - false_alarms) / decisions,
    }
    if coverage_weight is not None:
        counts = np.asarray(score["revisit"]["visit_counts"], dtype=float)
        shares = counts / counts.sum()
        concentration = float(np.sum(shares ** 2) - 1.0 / band_count)
        values["coverage_balanced"] = positives / decisions - coverage_weight * concentration
    if discovery_bonus is not None:
        discoveries = score["emitter_interception"]["intercepted_emitters"]
        values["oracle_discovery_focus"] = (
            discovery_bonus * discoveries - false_alarms
        ) / decisions
    return values


def screen_rewards(
    corpus_root: str | Path, *, split: str = "val", max_files: int = 10,
    detection_probability: float = 0.9, false_alarm_probability: float = 0.05,
    retune_time_ms: float = 0.0, coverage_weight: float | None = None,
    discovery_bonus: float | None = None, seed: int = 0,
    band_centres_mhz=None,
    receiver_profile: str = "binary_v1",
    amplitude_midpoint_db: float = -90.0,
    amplitude_scale_db: float = 5.0,
    max_observed_pdws: int = 32,
) -> dict:
    if split not in ("train", "val"):
        raise ValueError("Reward screening may use only train or validation files")
    if max_files <= 0:
        raise ValueError("max_files must be positive")
    if coverage_weight is not None and coverage_weight < 0:
        raise ValueError("coverage_weight must be nonnegative")
    if discovery_bonus is not None and discovery_bonus < 0:
        raise ValueError("discovery_bonus must be nonnegative")
    pairs = list(iter_tsr_replay_pairs(
        corpus_root, split, allow_stare_only=band_centres_mhz is not None,
    ))[:max_files]
    if not pairs:
        raise ValueError("No TSRD files selected for reward screening")
    names = (
        "single_band_camper", "two_band_ping_pong", "greedy_busy_band",
        "round_robin", "uniform_random", "periodic_two_slot_sweep",
    )
    results: dict[str, list[dict]] = {name: [] for name in names}
    for file_index, (stare_file, scan_file) in enumerate(pairs):
        env = TSRDStareEnvironment.from_stare_mode(
            str(stare_file), str(scan_file) if scan_file is not None else None,
            band_centres_mhz=band_centres_mhz,
            receiver_profile=receiver_profile,
            amplitude_midpoint_db=amplitude_midpoint_db,
            amplitude_scale_db=amplitude_scale_db,
            max_observed_pdws=max_observed_pdws,
            detection_probability=detection_probability,
            false_alarm_probability=false_alarm_probability,
            retune_time_ms=retune_time_ms,
        )
        schedulers = {
            "single_band_camper": StaticBandProbe(env.n_bands),
            "two_band_ping_pong": ScreenProbe(env.n_bands, "two_band_ping_pong"),
            "greedy_busy_band": ScreenProbe(env.n_bands, "greedy_busy_band"),
            "round_robin": RoundRobinProbe(env.n_bands),
            "uniform_random": UniformRandomProbe(env.n_bands),
            "periodic_two_slot_sweep": ScreenProbe(env.n_bands, "periodic_two_slot_sweep"),
        }
        for name, scheduler in schedulers.items():
            episode = run_episode(env, scheduler, seed=seed + file_index)
            score = score_recorded_replay(env, episode.trajectory)
            results[name].append({
                "config": stare_file.stem,
                "detected_emitter_slots": score["illumination"]["detected_emitter_slot_opportunity_count"],
                "emitter_slots": score["illumination"]["emitter_slot_opportunity_count"],
                "coverage": len(set(episode.actions.tolist())) / env.n_bands,
                "rewards": _reward_values(
                    score, env.n_bands,
                    coverage_weight=coverage_weight,
                    discovery_bonus=discovery_bonus,
                ),
            })

    summary = {}
    for name, rows in results.items():
        reward_names = rows[0]["rewards"]
        detected = sum(r["detected_emitter_slots"] for r in rows)
        offered = sum(r["emitter_slots"] for r in rows)
        summary[name] = {
            "pooled_illumination": float(detected / offered) if offered else None,
            "mean_unique_band_coverage": float(np.mean([r["coverage"] for r in rows])),
            "mean_rewards_per_decision": {
                reward: float(np.mean([r["rewards"][reward] for r in rows]))
                for reward in reward_names
            },
        }
    flags = {}
    reference = summary["round_robin"]
    for reward in summary["round_robin"]["mean_rewards_per_decision"]:
        flags[reward] = []
        for pathology in ("single_band_camper", "two_band_ping_pong", "greedy_busy_band"):
            row = summary[pathology]
            if (reference["pooled_illumination"] is not None
                    and row["pooled_illumination"] is not None
                    and row["pooled_illumination"] < reference["pooled_illumination"]
                    and row["mean_unique_band_coverage"] < reference["mean_unique_band_coverage"]
                    and row["mean_rewards_per_decision"][reward]
                    >= reference["mean_rewards_per_decision"][reward]):
                flags[reward].append(pathology)
    return {
        "split": split,
        "files_screened": len(pairs),
        "metric_contract": (
            "tsrd_recorded_pulse_emitter_v5" if receiver_profile == "pdw_v2"
            else "tsrd_recorded_pulse_emitter_v4"
        ),
        "receiver_options": {
            "detection_probability": detection_probability,
            "false_alarm_probability": false_alarm_probability,
            "retune_time_ms": retune_time_ms,
        },
        "receiver_profile_options": {
            "receiver_profile": receiver_profile,
            "amplitude_midpoint_db": amplitude_midpoint_db,
            "amplitude_scale_db": amplitude_scale_db,
            "max_observed_pdws": max_observed_pdws,
        },
        "reward_notes": {
            "oracle_true_detection": "Simulator-only current-step truth feedback",
            "oracle_false_alarm_aware": "Simulator-only current-step truth feedback",
            "oracle_discovery_focus": "Optimistic emitter attribution; not causally observable",
        },
        "policy_summary": summary,
        "pathology_flags": flags,
        "interpretation": "A clear flag is not a mathematical safety guarantee",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pd", type=float, default=0.9)
    parser.add_argument("--pfa", type=float, default=0.05)
    parser.add_argument("--retune-ms", type=float, default=0.0)
    parser.add_argument("--coverage-weight", type=float)
    parser.add_argument("--discovery-bonus", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--band-centres-mhz", type=float, nargs="+",
        help="Explicit receiver tune centres when paired scan files are absent",
    )
    parser.add_argument("--receiver-profile", choices=("binary_v1", "pdw_v2"),
                        default="binary_v1")
    parser.add_argument("--amplitude-midpoint-db", type=float, default=-90.0)
    parser.add_argument("--amplitude-scale-db", type=float, default=5.0)
    parser.add_argument("--max-observed-pdws", type=int, default=32)
    args = parser.parse_args()
    report = screen_rewards(
        args.corpus_root, split=args.split, max_files=args.max_files,
        detection_probability=args.pd, false_alarm_probability=args.pfa,
        retune_time_ms=args.retune_ms, coverage_weight=args.coverage_weight,
        discovery_bonus=args.discovery_bonus, seed=args.seed,
        band_centres_mhz=args.band_centres_mhz,
        receiver_profile=args.receiver_profile,
        amplitude_midpoint_db=args.amplitude_midpoint_db,
        amplitude_scale_db=args.amplitude_scale_db,
        max_observed_pdws=args.max_observed_pdws,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()

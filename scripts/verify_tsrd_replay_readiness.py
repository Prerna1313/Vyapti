"""Check repeatability and score accounting on one development TSRD pair.

This is a validation instrument, not a scheduler performance comparison.
Test files are explicitly disallowed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import tomllib
from pathlib import Path

import numpy as np

from vyapti_simulator.core.episode import PERMITTED_SCENARIO_KEYS, run_episode
from vyapti_simulator.core.scheduler_interface import PERMITTED_OBSERVATION_KEYS
from vyapti_simulator.qualification.probes import RoundRobinProbe
from vyapti_simulator.tsrd.benchmark_protocol import _assert_scorecard_consistent, _seed_for_file
from vyapti_simulator.tsrd.replay_scorecard import score_recorded_replay
from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment

PROFILING_KEYS = frozenset({
    "select_action_ms", "predict_ms", "wall_clock_ms", "memory_delta_bytes",
})


class _RecordingProbe(RoundRobinProbe):
    def __init__(self, band_count: int):
        super().__init__(band_count)
        self.policy_history = []

    def select_action(self, observation_history, current_time_slot):
        if observation_history:
            self.policy_history.append(dict(observation_history[-1]))
        return super().select_action(observation_history, current_time_slot)


def _stable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_stable)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_counts_from_trajectory(env, trajectory, score) -> dict:
    """Recount selected passband events directly from PDWs and observations."""
    data, labels = env.recorded_pulses
    centres, halfwidth = env.receiver_geometry
    toa = data[:, 0].astype(float, copy=False)
    frequency = data[:, 1].astype(float, copy=False)
    in_mission_count = int(np.count_nonzero(
        np.isfinite(toa) & (toa >= 0) & (toa < env.n_slots * 50_000.0)
    ))
    occupied = positives = true_hits = false_alarms = covered = 0
    for step in trajectory:
        slot = int(step.time_slot)
        start_us = (slot * 0.05 + float(step.observation["retune_cost_s"])) * 1e6
        end_us = (slot + 1) * 0.05 * 1e6
        left = int(np.searchsorted(toa, start_us, side="left"))
        right = int(np.searchsorted(toa, end_us, side="left"))
        in_band = np.abs(frequency[left:right] - centres[int(step.action)]) <= halfwidth
        in_band_count = int(np.count_nonzero(in_band))
        covered += in_band_count
        is_occupied = in_band_count > 0
        is_positive = bool(step.observation["hit"])
        occupied += is_occupied
        positives += is_positive
        true_hits += is_occupied and is_positive
        false_alarms += not is_occupied and is_positive
    empty = len(trajectory) - occupied
    cell = score["cell_level"]
    event = score["event_level"]
    pulse = score["pulse_level"]
    assert (occupied, empty) == (cell["eligible_selected_cells"], cell["empty_selected_cells"])
    assert (true_hits, false_alarms) == (event["true_detections"], event["false_alarms"])
    assert positives == true_hits + false_alarms
    assert covered == pulse["covered_recorded_pulses"]
    assert pulse["recorded_pulses"] == len(labels)
    assert pulse["recorded_pulses_in_mission"] == in_mission_count
    assert pulse["recorded_pulses_outside_mission"] == len(labels) - in_mission_count
    assert cell["conditional_pd"] == (true_hits / occupied if occupied else None)
    assert cell["true_pfa"] == (false_alarms / empty if empty else None)
    assert pulse["recorded_pulse_coverage_ratio"] == (
        covered / in_mission_count if in_mission_count else None
    )
    assert pulse["physical_pulse_interception_ratio"] is None
    assert pulse["offered_pulse_capture_ratio"] is None
    return {
        "selected_occupied_slots": occupied, "selected_empty_slots": empty,
        "true_detections": true_hits, "false_alarms": false_alarms,
        "covered_recorded_pulses": covered,
    }


def verify(root: Path, config: int = 0, base_seed: int = 42) -> dict:
    root = root.resolve()
    stare = root / "stare" / "val_stare" / f"config_{config}.h5"
    scan = root / "scan" / "val_scan" / f"config_{config}.h5"
    if not stare.is_file() or not scan.is_file():
        raise FileNotFoundError("Validation scan/stare pair is missing")
    env = TSRDStareEnvironment.from_stare_mode(
        str(stare), str(scan), receiver_profile="binary_v1",
        detection_probability=0.9, false_alarm_probability=0.05,
        retune_time_ms=1.0,
    )
    seed = _seed_for_file(stare.stem, base_seed)
    runs = []
    noise_fields = []
    for _ in range(2):
        policy = _RecordingProbe(env.n_bands)
        episode = run_episode(env, policy, seed=seed)
        observations = [
            {key: value for key, value in step.observation.items() if key not in PROFILING_KEYS}
            for step in episode.trajectory
        ]
        assert len(observations) == env.n_slots == 600
        assert all(set(obs) <= PERMITTED_OBSERVATION_KEYS for obs in observations)
        assert all("labels" not in obs and "hidden_truth" not in obs for obs in observations)
        assert policy.policy_history == observations[:-1], "Policy input differs from receiver measurements"
        assert all(PROFILING_KEYS.isdisjoint(obs) for obs in policy.policy_history)
        score = score_recorded_replay(env, episode.trajectory)
        _assert_scorecard_consistent(score, episode.trajectory)
        independent_counts = _check_counts_from_trajectory(env, episode.trajectory, score)
        runs.append({
            "actions_sha256": _digest(episode.actions.tolist()),
            "hits_sha256": _digest(episode.hits.tolist()),
            "observations_sha256": _digest(observations),
            "scorecard_sha256": _digest(score),
            "replay_signature": episode.replay_signature,
            "recorded_pulses": score["pulse_level"]["recorded_pulses"],
            "recorded_pulses_outside_mission": score["pulse_level"]["recorded_pulses_outside_mission"],
            "selected_occupied_slots": score["cell_level"]["eligible_selected_cells"],
            "selected_empty_slots": score["cell_level"]["empty_selected_cells"],
            "true_detections": score["event_level"]["true_detections"],
            "false_alarms": score["event_level"]["false_alarms"],
            "observed_positive_slots": sum(bool(obs["hit"]) for obs in observations),
            "independently_recounted": independent_counts,
        })
        noise_fields.append(env._noise_field.copy())
    changed = [key for key in runs[0] if runs[0][key] != runs[1][key]]
    assert not changed, f"Same seed changed replay fields: {changed}"
    assert np.array_equal(noise_fields[0], noise_fields[1]), "Same seed changed noise"
    env.reset(seed=seed + 1)
    assert not np.array_equal(noise_fields[0], env._noise_field), "Different seed did not change noise"
    assert runs[0]["observed_positive_slots"] == (
        runs[0]["true_detections"] + runs[0]["false_alarms"]
    )
    versions = {}
    for package in ("vyapti-simulator", "numpy", "h5py", "scipy", "pytest", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project_version = tomllib.loads(project_file.read_text(encoding="utf-8"))["project"]["version"]
    return {
        "version": "1.0.0",
        "scope": "Single validation pair repeatability and score accounting; no test access",
        "pair": f"val/config_{config}.h5",
        "profile": {
            "receiver_profile": "binary_v1", "dwell_profile": "fixed_50_ms",
            "detection_probability": 0.9, "false_alarm_probability": 0.05,
            "retune_time_ms": 1.0,
        },
        "seed_scheme": "little_endian_first_32_bits_sha256('{base_seed}:{config_stem}')",
        "base_seed": base_seed,
        "derived_receiver_seed": seed,
        "scenario_key_whitelist": sorted(PERMITTED_SCENARIO_KEYS),
        "observation_key_whitelist": sorted(PERMITTED_OBSERVATION_KEYS),
        "policy_effective_keys": sorted(PERMITTED_OBSERVATION_KEYS - PROFILING_KEYS),
        "same_seed_identical": True,
        "different_seed_changes_noise": True,
        "accounting_checked": True,
        "run": runs[0],
        "software": {"python": platform.python_version(),
                     "source_project_version": project_version, **versions},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()
    root, output = args.corpus_root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error("Output must be outside the source corpus")
    report = verify(root, args.config, args.base_seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

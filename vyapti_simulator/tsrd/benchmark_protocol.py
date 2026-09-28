"""Split-safe TSRD world training and frozen-checkpoint evaluation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from vyapti_simulator.core.episode import run_episode
from vyapti_simulator.core.metrics import MetricsConfig, TrajectoryStep, compute_exploration_metrics

from .corpus_loader import iter_tsr_replay_pairs
from .replay_scorecard import _km_quantile, _km_restricted_mean, score_recorded_replay
from .tsrd_environment import TSRDStareEnvironment
from .world_pool import TSRDTrainWorldPool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_for_file(stem: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{stem}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _input_inventory(corpus_root: Path, allow_stare_only: bool) -> dict:
    """Record a cheap file inventory, not a substitute for content hashes."""
    splits = {}
    for split in ("train", "val", "test"):
        files = []
        for stare, scan in iter_tsr_replay_pairs(
            corpus_root, split, allow_stare_only=allow_stare_only
        ):
            for path in (stare, scan):
                if path is None:
                    continue
                stat = path.stat()
                files.append({
                    "path": path.relative_to(corpus_root).as_posix(),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                })
        if not files:
            raise ValueError(f"No TSRD {split} replay files")
        splits[split] = sorted(files, key=lambda row: row["path"])
    payload = json.dumps(splits, sort_keys=True, separators=(",", ":"))
    return {
        "basis": "relative_path_size_mtime_not_content_hash",
        "inventory_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "splits": splits,
    }


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _assert_scorecard_consistent(score: dict, trajectory: list) -> None:
    """Reject reports whose recorded event counts disagree with observations."""
    positives = sum(bool(step.observation["hit"]) for step in trajectory)
    true_hits = score["event_level"]["true_detections"]
    false_alarms = score["event_level"]["false_alarms"]
    cell = score["cell_level"]
    pulse = score["pulse_level"]
    if (
        positives != true_hits + false_alarms
        or true_hits != cell["detected_occupied_cell_count"]
        or true_hits > cell["eligible_selected_cells"]
        or false_alarms > cell["empty_selected_cells"]
        or pulse["recorded_pulses_in_mission"] + pulse["recorded_pulses_outside_mission"]
           != pulse["recorded_pulses"]
        or pulse["covered_recorded_pulses"] > pulse["recorded_pulses_in_mission"]
        or (pulse["captured_recorded_pulses"] is not None
            and pulse["captured_recorded_pulses"] > pulse["covered_recorded_pulses"])
        or score["reward"]["detector_positive_total"] != positives
        or score["reward"]["false_alarm_aware_total"] != true_hits - false_alarms
    ):
        raise ValueError("TSRD scorecard is inconsistent with its recorded trajectory")


def _summarize(rows: list[dict]) -> dict:
    illumination_n = sum(r["illumination"]["detected_emitter_slot_opportunity_count"] for r in rows)
    illumination_d = sum(r["illumination"]["emitter_slot_opportunity_count"] for r in rows)
    discovered = sum(r["emitter_interception"]["intercepted_emitters"] for r in rows)
    eligible_emitters = sum(r["emitter_interception"]["eligible_emitters"] for r in rows)
    cells = sum(r["cell_level"]["occupied_cell_count"] for r in rows)
    true_hits = sum(r["event_level"]["true_detections"] for r in rows)
    false_alarms = sum(r["event_level"]["false_alarms"] for r in rows)
    observed = sum(r["cell_level"]["eligible_selected_cells"] for r in rows)
    empty = sum(r["cell_level"]["empty_selected_cells"] for r in rows)
    duration_s = sum(r["mission_duration_s"] for r in rows)
    repeated = sum(r["event_level"]["repeated_emitter_slot_detections"] for r in rows)
    ttfi_records = [emitter for row in rows for emitter in row["per_emitter"].values()]
    durations = [emitter["ttfi_ms"] for emitter in ttfi_records]
    events = [not emitter["ttfi_censored"] for emitter in ttfi_records]
    observed_durations = [duration for duration, event in zip(durations, events) if event]
    restricted_mean_ms, restriction_ms = _km_restricted_mean(durations, events)
    revisit_intervals = [
        value for row in rows for value in row["revisit"]["intervals_s"]
    ]
    total_dwells = sum(r["decision_level"]["total_dwells"] for r in rows)
    true_hit_dwells = sum(r["decision_level"]["true_hit_dwells"] for r in rows)
    empty_dwells = sum(r["decision_level"]["empty_dwells"] for r in rows)
    switches = sum(r["revisit"]["switch_count"] for r in rows)
    switch_opportunities = sum(max(0, r["decision_level"]["total_dwells"] - 1) for r in rows)
    eligible_emitter_slots = sum(
        r["emitter_interception"]["eligible_observed_emitter_slot_count"] for r in rows
    )
    covered_pulses = sum(r["pulse_level"]["covered_recorded_pulses"] for r in rows)
    recorded_pulses = sum(r["pulse_level"]["recorded_pulses_in_mission"] for r in rows)
    outside_mission_pulses = sum(r["pulse_level"]["recorded_pulses_outside_mission"] for r in rows)
    captured_counts = [r["pulse_level"]["captured_recorded_pulses"] for r in rows]
    def ratio(n, d):
        return float(n / d) if n is not None and d else None
    return {
        "files_evaluated": len(rows),
        "pooled_illumination_interception_ratio": ratio(illumination_n, illumination_d),
        "pooled_unique_emitter_interception_rate": ratio(discovered, eligible_emitters),
        "pooled_occupied_cell_detection_ratio": ratio(true_hits, cells),
        "opportunity_interception_ratio": ratio(true_hits, cells),
        "average_intercept_rate_hz": ratio(true_hits, duration_s),
        "average_seconds_per_intercept": ratio(duration_s, true_hits),
        "pooled_per_emitter_opportunity_pd_upper_bound": ratio(illumination_n, eligible_emitter_slots),
        "unique_emitter_discoveries_per_s": ratio(discovered, duration_s),
        "unique_emitter_discoveries_per_sec": ratio(discovered, duration_s),
        "repeated_true_detections_per_s": ratio(repeated, duration_s),
        "repeated_true_detection_events_per_sec": ratio(repeated, duration_s),
        "false_alarms_per_s": ratio(false_alarms, duration_s),
        "ttfi_mean_observed_ms": float(sum(observed_durations) / len(observed_durations)) if observed_durations else None,
        "ttfi_median_observed_ms": float(np.median(observed_durations)) if observed_durations else None,
        "ttfi_p90_observed_ms": float(np.percentile(observed_durations, 90)) if observed_durations else None,
        "ttfi_km_median_ms": _km_quantile(durations, events, 0.5),
        "ttfi_km_p90_ms": _km_quantile(durations, events, 0.9),
        "censored_mean_ttfi_s": ratio(restricted_mean_ms, 1000) if restricted_mean_ms is not None else None,
        "censored_median_ttfi_s": ratio(_km_quantile(durations, events, 0.5), 1000),
        "censored_p95_ttfi_s": ratio(_km_quantile(durations, events, 0.95), 1000),
        "censored_mean_restriction_s": ratio(restriction_ms, 1000) if restriction_ms is not None else None,
        "ttfi_censoring_fraction": ratio(len(events) - discovered, len(events)),
        "conditional_pd": ratio(true_hits, observed),
        "true_pfa": ratio(false_alarms, empty),
        "total_dwells": total_dwells,
        "true_detection_yield_per_dwell": ratio(true_hit_dwells, total_dwells),
        "empty_scan_fraction": ratio(empty_dwells, total_dwells),
        "mean_revisit_s": float(np.mean(revisit_intervals)) if revisit_intervals else None,
        "p95_revisit_s": float(np.percentile(revisit_intervals, 95)) if revisit_intervals else None,
        "max_revisit_s": float(np.max(revisit_intervals)) if revisit_intervals else None,
        "switch_rate": ratio(switches, switch_opportunities),
        "physical_pulse_interception_ratio": None,
        "offered_pulse_capture_ratio": None,
        "recorded_pulse_coverage_ratio": ratio(covered_pulses, recorded_pulses),
        "recorded_pulses_in_mission": recorded_pulses,
        "recorded_pulses_outside_mission": outside_mission_pulses,
        "recorded_pulse_detected_ratio": (
            ratio(sum(captured_counts), recorded_pulses)
            if all(count is not None for count in captured_counts) else None
        ),
        "emitter_slot_opportunities": illumination_d,
        "detected_emitter_slot_opportunities": illumination_n,
        "eligible_emitters": eligible_emitters,
        "intercepted_emitters": discovered,
        "true_detections": true_hits,
        "false_alarms": false_alarms,
        "reward_detector_positive_total": true_hits + false_alarms,
        "reward_false_alarm_aware_total": true_hits - false_alarms,
    }


class TSRDBenchmarkProtocol:
    """Trainer/checkpoint adapters are algorithm-specific; split handling is not."""

    def __init__(
        self, corpus_root: str | Path, output_dir: str | Path, *,
        detection_probability: float = 1.0,
        false_alarm_probability: float = 0.0,
        retune_time_ms: float = 0.0,
        seed: int = 0,
        dwell_profile: str = "fixed_50_ms",
        band_centres_mhz=None,
        passband_halfwidth_mhz: float = 500.0,
        receiver_profile: str = "binary_v1",
        amplitude_midpoint_db: float = -90.0,
        amplitude_scale_db: float = 5.0,
        max_observed_pdws: int = 32,
    ):
        self.corpus_root = Path(corpus_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir == self.corpus_root or self.corpus_root in self.output_dir.parents:
            raise ValueError("Checkpoint/output directory must be outside the TSRD corpus")
        self.receiver_options = {
            "detection_probability": detection_probability,
            "false_alarm_probability": false_alarm_probability,
            "retune_time_ms": retune_time_ms,
            "passband_halfwidth_mhz": passband_halfwidth_mhz,
        }
        self.band_centres_mhz = (
            None if band_centres_mhz is None else np.asarray(band_centres_mhz, dtype=np.float64)
        )
        self.receiver_profile_options = {
            "receiver_profile": receiver_profile,
            "amplitude_midpoint_db": amplitude_midpoint_db,
            "amplitude_scale_db": amplitude_scale_db,
            "max_observed_pdws": max_observed_pdws,
        }
        self.seed = int(seed)
        if dwell_profile not in ("fixed_50_ms", "mixed_50_100_ms"):
            raise ValueError("dwell_profile must be fixed_50_ms or mixed_50_100_ms")
        self.dwell_profile = dwell_profile
        self._train_pool = None

    def train_worlds(self, count: int, emitter_count: int) -> Iterable[tuple[TSRDStareEnvironment, list[dict]]]:
        """Yield reproducible train-only worlds; provenance is for audit, not policy input."""
        if count <= 0:
            raise ValueError("count must be positive")
        if self._train_pool is None:
            self._train_pool = TSRDTrainWorldPool(
                self.corpus_root, band_centres_mhz=self.band_centres_mhz,
                **self.receiver_profile_options, **self.receiver_options,
            )
        pool = self._train_pool
        for i in range(count):
            env, provenance = pool.sample_world(self.seed + i, emitter_count)
            env.reset(seed=self.seed + i)
            yield env, provenance

    def _evaluate_split(
        self, split: str, checkpoint: Path,
        load_checkpoint: Callable[[Path, int], object],
    ) -> dict:
        pairs = list(iter_tsr_replay_pairs(
            self.corpus_root, split,
            allow_stare_only=self.band_centres_mhz is not None,
        ))
        if not pairs:
            raise ValueError(f"No TSRD {split} replay pairs")
        frozen_hash = _sha256(checkpoint)
        rows = []
        for file_index, (stare_file, scan_file) in enumerate(pairs):
            env = TSRDStareEnvironment.from_stare_mode(
                str(stare_file), str(scan_file) if scan_file is not None else None,
                band_centres_mhz=self.band_centres_mhz,
                **self.receiver_options, **self.receiver_profile_options
            )
            policy = load_checkpoint(checkpoint, env.n_bands)
            receiver_seed = _seed_for_file(stare_file.stem, self.seed)
            if self.dwell_profile == "mixed_50_100_ms":
                from .mixed_dwell import run_mixed_dwell_episode
                result = run_mixed_dwell_episode(env, policy, seed=receiver_seed)
                trajectory = result.base_trajectory
            else:
                result = run_episode(
                    env, policy, seed=receiver_seed,
                    scheduler_name=checkpoint.stem,
                )
                trajectory = result.trajectory
            from .tsrd_metrics import TSRDMetricsEngine
            metrics = TSRDMetricsEngine(
                MetricsConfig(
                    use_tsr_stare_mode=True,
                    compute_comprehensive_metrics=True,
                    compute_per_band_metrics=False,
                    compute_temporal_metrics=False,
                ),
                stare_mode_file=str(stare_file),
                scan_mode_file=str(scan_file) if scan_file is not None else None,
                receiver_env=env,
            ).record_result(
                episode_id=file_index,
                seed=receiver_seed,
                scheduler_name=checkpoint.stem,
                scenario_config={"slot_duration_s": env.config.slot_duration_s()},
                trajectory=trajectory,
                truth_grid=env.hidden_truth,
                receiver_accounting=result.receiver_accounting,
            )
            score = metrics["tsrd_replay_scorecard"]
            if self.dwell_profile == "mixed_50_100_ms":
                score = score_recorded_replay(
                    env, trajectory, decision_visits=result.decisions
                )
                decision_steps = [
                    TrajectoryStep(action=band, time_slot=start,
                                   observation=result.decision_observations[i])
                    for i, (start, _, band) in enumerate(result.decisions)
                ]
                exploration = compute_exploration_metrics(decision_steps, env.n_bands)
                metrics["tier_b_scheduler"].update({
                    **score["decision_level"],
                    "action_entropy": exploration["entropy_of_actions"],
                    "normalized_action_entropy": exploration["normalized_action_entropy"],
                    "unique_band_coverage": exploration["coverage_rate"],
                    "exploration_fraction": exploration["exploration_fraction"],
                })
            _assert_scorecard_consistent(score, trajectory)
            rows.append({
                "config": stare_file.stem,
                **score,
                **{key: metrics[key] for key in (
                    "tier_a_ps26055", "tier_b_scheduler", "tier_c_prediction",
                    "tier_d_environment", "tier_e_implementation",
                )},
            })
        if _sha256(checkpoint) != frozen_hash:
            raise RuntimeError("Frozen checkpoint changed during evaluation")
        return {
            "split": split,
            "checkpoint_sha256": frozen_hash,
            "summary": _summarize(rows),
            "per_config": rows,
        }

    def run(
        self,
        candidate_ids: list[str],
        train_candidate: Callable[[str, Iterable[tuple[TSRDStareEnvironment, list[dict]]], Path, str], None],
        load_checkpoint: Callable[[Path, int], object],
        *, worlds_per_candidate: int, emitter_count: int,
        reward_mode: str = "detector_positive",
        evaluate_test: bool = False,
    ) -> dict:
        """Train and select on development data; test only on explicit request."""
        if reward_mode not in ("detector_positive", "false_alarm_aware"):
            raise ValueError("Unknown reward mode")
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Provide unique candidate IDs")
        if worlds_per_candidate <= 0 or emitter_count <= 0:
            raise ValueError("Training world and emitter counts must be positive")
        if any(not c.isidentifier() for c in candidate_ids):
            raise ValueError("Candidate IDs must be simple identifiers")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "tsrd_benchmark_report.json"
        if report_path.exists():
            raise FileExistsError(f"Use a fresh output directory: {report_path} exists")
        manifest_path = self.output_dir / "tsrd_run_manifest.json"
        if manifest_path.exists():
            raise FileExistsError(f"Use a fresh output directory: {manifest_path} exists")
        manifest = {
            "status": "started",
            "corpus_root": str(self.corpus_root),
            "metric_contract_version": (
                "tsrd_recorded_pulse_emitter_v4"
                if self.receiver_profile_options["receiver_profile"] == "pdw_v2"
                else "tsrd_recorded_pulse_emitter_v3"
            ),
            "input_inventory": _input_inventory(
                self.corpus_root, self.band_centres_mhz is not None
            ),
            "seed_scheme": {
                "train_world": "root_seed_plus_world_index",
                "evaluation_file": "first_32_bits_sha256(root_seed:config_stem)",
                "receiver_noise": "replay_signature_and_receiver_seed",
                "optimizer_seed": "trainer_callback_owned",
            },
            "seed": self.seed,
            "candidate_ids": candidate_ids,
            "worlds_per_candidate": worlds_per_candidate,
            "emitters_per_world": emitter_count,
            "reward_mode": reward_mode,
            "evaluate_test": bool(evaluate_test),
            "receiver_options": self.receiver_options,
            "receiver_profile_options": self.receiver_profile_options,
            "dwell_profile": self.dwell_profile,
            "band_centres_mhz": (
                self.band_centres_mhz.tolist() if self.band_centres_mhz is not None else None
            ),
        }
        _write_json(manifest_path, manifest)
        def mark_failed(phase: str, exc: Exception, candidate_id: str | None = None) -> None:
            manifest["status"] = "failed"
            manifest["failure"] = {
                "phase": phase,
                "candidate_id": candidate_id,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            _write_json(manifest_path, manifest)

        validation = {}
        checkpoints = {}
        for candidate_id in candidate_ids:
            checkpoint = self.output_dir / f"{candidate_id}.checkpoint"
            if checkpoint.exists():
                raise FileExistsError(f"Use a fresh output directory: {checkpoint} exists")
            consumed = 0
            def counted_worlds():
                nonlocal consumed
                for world in self.train_worlds(worlds_per_candidate, emitter_count):
                    consumed += 1
                    yield world
            try:
                train_candidate(
                    candidate_id, counted_worlds(),
                    checkpoint, reward_mode,
                )
                if consumed != worlds_per_candidate:
                    raise ValueError(
                        f"Trainer {candidate_id} consumed {consumed} of "
                        f"{worlds_per_candidate} declared training worlds"
                    )
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Trainer did not save {checkpoint}")
                checkpoints[candidate_id] = checkpoint
                validation[candidate_id] = self._evaluate_split(
                    "val", checkpoint, load_checkpoint
                )
            except Exception as exc:
                mark_failed("training_or_validation", exc, candidate_id)
                raise

        def selection_score(candidate_id: str) -> float:
            score = validation[candidate_id]["summary"]["pooled_illumination_interception_ratio"]
            return float("-inf") if score is None else score

        try:
            selected = max(candidate_ids, key=selection_score)
            if selection_score(selected) == float("-inf"):
                raise ValueError("Validation has no recorded emitter-slot opportunities")
            test = (
                self._evaluate_split("test", checkpoints[selected], load_checkpoint)
                if evaluate_test else None
            )
        except Exception as exc:
            mark_failed("selection_or_test", exc)
            raise
        report = {
            "metric_contract_version": (
                "tsrd_recorded_pulse_emitter_v4"
                if self.receiver_profile_options["receiver_profile"] == "pdw_v2"
                else "tsrd_recorded_pulse_emitter_v3"
            ),
            "selection_metric": "pooled_illumination_interception_ratio",
            "selected_candidate": selected,
            "reward_mode": reward_mode,
            "receiver_options": self.receiver_options,
            "receiver_profile_options": self.receiver_profile_options,
            "receiver_geometry": {
                "source": ("explicit_centres" if self.band_centres_mhz is not None
                           else "paired_scan_metadata"),
                "band_centres_mhz": (self.band_centres_mhz.tolist()
                                     if self.band_centres_mhz is not None else None),
            },
            "dwell_profile": self.dwell_profile,
            "corpus_root": str(self.corpus_root),
            "seed": self.seed,
            "train_pool_contributions": len(self._train_pool.contributions),
            "train_worlds_per_candidate": worlds_per_candidate,
            "train_emitters_per_world": emitter_count,
            "validation": validation,
            "held_out_test": test,
            "evaluation_stage": "final" if evaluate_test else "development",
        }
        _write_json(report_path, report)
        manifest["status"] = "complete" if evaluate_test else "development_complete"
        manifest["selected_candidate"] = selected
        manifest["checkpoint_sha256"] = {
            candidate_id: validation[candidate_id]["checkpoint_sha256"]
            for candidate_id in candidate_ids
        }
        _write_json(manifest_path, manifest)
        return report

    def finalize_test(self, load_checkpoint: Callable[[Path, int], object]) -> dict:
        """Evaluate the selected frozen checkpoint once after development."""
        manifest_path = self.output_dir / "tsrd_run_manifest.json"
        report_path = self.output_dir / "tsrd_benchmark_report.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "development_complete" or report.get("evaluation_stage") != "development":
            raise ValueError("Expected a completed development run without test results")
        expected_settings = {
            "corpus_root": str(self.corpus_root),
            "seed": self.seed,
            "receiver_options": self.receiver_options,
            "receiver_profile_options": self.receiver_profile_options,
            "dwell_profile": self.dwell_profile,
            "band_centres_mhz": (
                self.band_centres_mhz.tolist() if self.band_centres_mhz is not None else None
            ),
        }
        if any(manifest.get(key) != value for key, value in expected_settings.items()):
            raise ValueError("Replay settings differ from the development run")
        if manifest["input_inventory"]["inventory_sha256"] != _input_inventory(
            self.corpus_root, self.band_centres_mhz is not None
        )["inventory_sha256"]:
            raise ValueError("Source inventory changed since development")
        selected = report["selected_candidate"]
        checkpoint = self.output_dir / f"{selected}.checkpoint"
        if _sha256(checkpoint) != manifest["checkpoint_sha256"][selected]:
            raise ValueError("Selected checkpoint changed since development")
        test = self._evaluate_split("test", checkpoint, load_checkpoint)
        report["held_out_test"] = test
        report["evaluation_stage"] = "final"
        _write_json(report_path, report)
        manifest["status"] = "complete"
        manifest["evaluate_test"] = True
        _write_json(manifest_path, manifest)
        return report

"""TSRD corpus protocol and label-aware replay metric regression tests."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.core.episode import run_episode
from vyapti_simulator.core.mapping import frequency_to_bands
from vyapti_simulator.core.metrics import MetricsConfig, MetricsEngine
from vyapti_simulator.core.scheduler_interface import BandPrediction
from vyapti_simulator.qualification.probes import RoundRobinProbe
from vyapti_simulator.tsrd.benchmark_protocol import (
    TSRDBenchmarkProtocol, _assert_scorecard_consistent, _summarize,
)
from vyapti_simulator.tsrd.corpus_loader import CorpusUnavailableError, iter_tsr_replay_pairs
from vyapti_simulator.tsrd.mixed_dwell import DwellAction, run_mixed_dwell_episode
from vyapti_simulator.tsrd.paired_scan_validation import (
    validate_development_pairs, validate_pair,
)
from vyapti_simulator.tsrd.pdw_discretiser import discretise_pdw_to_grid
from vyapti_simulator.tsrd.replay_scorecard import score_recorded_replay
from vyapti_simulator.tsrd.tsrd_adapter import PDWStream, stare_pulse_occupancy
from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment
from vyapti_simulator.tsrd.tsrd_metrics import TSRDMetricsEngine
from vyapti_simulator.tsrd.world_pool import TSRDTrainWorldPool
from scripts.audit_tsrd_metadata import audit_corpus
from scripts.pre_train_reward_screen import screen_rewards


def _small_world():
    centres = np.array([100.0, 600.0])
    data = np.array([
        [1_000.0, 600.0, 1.0, 0.0, -70.0],
        [5_000.0, 100.0, 1.0, 0.0, -70.0],
        [55_000.0, 100.0, 1.0, 0.0, -70.0],
        [110_000.0, 600.0, 1.0, 0.0, -70.0],
    ])
    labels = np.array([30, 10, 10, 20])
    cfg = SimulationConfig(
        band_count=2, time_slots=3, dwell_time_ms=50.0,
        receiver_ibw_mhz=500.0, total_spectrum_mhz=1000.0,
        detection_probability=1.0, false_alarm_probability=0.0,
    )
    occupancy = stare_pulse_occupancy(data, centres, 250.0, 3)
    return TSRDStareEnvironment(
        2, 3, occupancy, None, cfg, stare_data=data, stare_labels=labels,
        band_centres_mhz=centres, passband_halfwidth_mhz=250.0,
    )


def test_replay_scorecard_has_censored_emitter_and_distinct_denominators():
    env = _small_world()
    env.reset(seed=1)
    trajectory = []
    for band in (0, 0, 1):
        obs, _ = env.step(band)
        trajectory.append(SimpleNamespace(action=band, time_slot=obs["time_slot"], observation=obs))
    score = score_recorded_replay(env, trajectory)
    assert score["emitter_interception"]["unique_emitter_interception_rate"] == pytest.approx(2 / 3)
    assert score["illumination"]["emitter_slot_opportunity_count"] == 4
    assert score["illumination"]["detected_emitter_slot_opportunity_count"] == 3
    assert score["illumination"]["illumination_interception_ratio"] == pytest.approx(0.75)
    assert score["cell_level"]["occupied_cell_detection_ratio"] == pytest.approx(0.75)
    assert score["cell_level"]["conditional_pd"] == 1.0
    assert score["ttfi"]["censoring_fraction"] == pytest.approx(1 / 3)
    assert score["per_emitter"]["30"]["ttfi_censored"] is True
    assert score["per_emitter"]["30"]["emitter_first_intercept_ms"] is None
    assert score["event_level"]["repeated_emitter_slot_detections"] == 1
    assert score["reward"]["detector_positive_total"] == 3
    assert score["cell_level"]["conditional_pd"] == 1.0
    assert score["cell_level"]["occupied_cell_detection_ratio"] == 0.75
    assert score["pulse_level"]["physical_pulse_interception_ratio"] is None
    assert score["pulse_level"]["offered_pulse_capture_ratio"] is None
    assert score["pulse_level"]["recorded_pulse_coverage_ratio"] == 0.75
    assert score["revisit"]["mean_revisit_s"] == pytest.approx(0.05)
    assert score["revisit"]["switch_rate"] == 0.5
    assert score["ttfi"]["censored_mean_ttfi_s"] is not None
    assert score["ttfi"]["censored_mean_ttfi_s"] == pytest.approx(0.05)
    assert score["ttfi"]["censored_median_ttfi_s"] == 0.0
    assert score["ttfi"]["censored_p95_ttfi_s"] is None
    assert score["per_emitter"]["10"]["emitter_first_emission_ms"] == 5.0
    assert score["per_emitter"]["10"]["emitter_first_opportunity_slot"] == 0
    assert score["per_emitter"]["10"]["emitter_first_intercept_slot"] == 0
    assert score["cell_level"]["true_pfa"] is None
    assert score["revisit"]["p95_revisit_s"] == pytest.approx(0.05)
    assert score["event_level"]["repeated_true_detection_events_per_sec"] == pytest.approx(1 / 0.15)
    assert score["emitter_interception"]["per_emitter_opportunity_pd"]["30"] is None


def test_tier_b_uses_real_exploration_keys():
    trajectory = [
        SimpleNamespace(action=0, time_slot=0, observation={"hit": True}, prediction=None),
        SimpleNamespace(action=1, time_slot=1, observation={"hit": False}, prediction=None),
    ]
    grid = np.array([[True, False], [False, False]])
    metrics = TSRDMetricsEngine(MetricsConfig()).compute_comprehensive_metrics(
        trajectory, grid
    )
    assert metrics["tier_a_ps26055"]["conditional_pd"] == 1.0
    assert metrics["tier_b_scheduler"]["action_entropy"] == pytest.approx(1.0)
    assert metrics["tier_b_scheduler"]["unique_band_coverage"] == 1.0


def test_prediction_p95_is_a_real_percentile():
    truth = SimpleNamespace(
        band_count=2, time_slots=3,
        grid=np.array([[[False, True, False], [False, False, True]]]),
    )
    forecast = BandPrediction(
        issued_at_slot=0, about_time_slot=1,
        band_activity_probability=[1.0, 0.0],
        predicted_next_activity_slot=[2, 2],
    )
    trajectory = [SimpleNamespace(prediction=forecast)]
    metrics = MetricsEngine(MetricsConfig())._prediction_metrics(trajectory, truth)
    assert metrics["median_intercept_time_error_slots"] == 0.5
    assert metrics["p95_intercept_time_error_slots"] == pytest.approx(0.95)


def test_receiver_noise_is_keyed_by_world_band_and_slot():
    env = _small_world()
    env.config.detection_probability = 0.5
    env.config.false_alarm_probability = 0.5
    signature = env.replay_signature()
    env.reset(seed=17)
    field = env._noise_field.copy()
    first = [env.step(band)[0]["hit"] for band in (0, 0, 1)]
    env.reset(seed=17)
    second = [env.step(band)[0]["hit"] for band in (1, 0, 1)]
    assert first[-1] == second[-1]
    assert np.array_equal(field, env._noise_field)
    env.reset(seed=18)
    assert not np.array_equal(field, env._noise_field)
    assert env.replay_signature() == signature


def test_receiver_empirical_pd_and_pfa_match_configured_operating_point():
    centres = np.array([100.0])
    slots = 600
    cfg = SimulationConfig(
        band_count=1, time_slots=slots, dwell_time_ms=50.0,
        receiver_ibw_mhz=500.0, total_spectrum_mhz=500.0,
        detection_probability=0.8, false_alarm_probability=0.1,
    )
    empty = TSRDStareEnvironment(
        1, slots, np.zeros((1, slots), dtype=bool), None, cfg,
        stare_data=np.empty((0, 5)), stare_labels=np.empty(0, dtype=int),
        band_centres_mhz=centres, passband_halfwidth_mhz=250.0,
    )
    empty.reset(seed=31)
    pfa = sum(empty.step(0)[0]["hit"] for _ in range(slots)) / slots
    assert pfa == pytest.approx(0.1, abs=0.05)

    data = np.column_stack((
        np.arange(slots) * 50_000.0 + 10_000.0,
        np.full(slots, 100.0), np.ones(slots), np.zeros(slots),
        np.full(slots, -70.0),
    ))
    occupied = TSRDStareEnvironment(
        1, slots, np.ones((1, slots), dtype=bool), None, cfg,
        stare_data=data, stare_labels=np.ones(slots, dtype=int),
        band_centres_mhz=centres, passband_halfwidth_mhz=250.0,
    )
    occupied.reset(seed=31)
    pd = sum(occupied.step(0)[0]["hit"] for _ in range(slots)) / slots
    assert pd == pytest.approx(0.8, abs=0.05)


class _TwoSlotProbe(RoundRobinProbe):
    def select_action(self, observation_history, current_time_slot):
        return DwellAction(super().select_action(observation_history, current_time_slot), 2)


def test_mixed_dwell_consumes_two_independent_looks_and_clips_at_end():
    env = _small_world()
    episode = run_mixed_dwell_episode(env, _TwoSlotProbe(2), seed=5)
    assert episode.base_slots_consumed == 3
    assert episode.decision_count == 2
    assert episode.decisions == [(0, 2, 0), (2, 1, 1)]
    assert episode.decision_observations[0]["slots_consumed"] == 2
    assert len(episode.decision_observations[0]["detector_positives"]) == 2
    assert episode.decision_observations[1]["slots_consumed"] == 1
    assert env._noise_field[0, 0] != env._noise_field[0, 1]
    score = score_recorded_replay(
        env, episode.base_trajectory, decision_visits=episode.decisions
    )
    assert score["decision_level"]["total_dwells"] == 2
    assert score["decision_level"]["base_slots_consumed"] == 3
    assert score["decision_level"]["true_detection_yield_per_dwell"] == 1.0
    assert set(episode.decision_observations[0]).isdisjoint(
        {"emitter_id", "hidden_truth", "true_occupancy"}
    )


def test_mixed_dwell_30_second_mission_has_300_decisions():
    fixture = Path(__file__).parent / "fixtures" / "tsrd"
    env = TSRDStareEnvironment.from_stare_mode(
        str(fixture / "config_0_stare.h5"), str(fixture / "config_0_scan.h5")
    )
    episode = run_mixed_dwell_episode(env, _TwoSlotProbe(36), seed=8)
    assert episode.base_slots_consumed == 600
    assert episode.decision_count == 300
    assert env.done
    assert env.current_slot * env.config.slot_duration_s() == pytest.approx(30.0)
    score = score_recorded_replay(env, episode.base_trajectory, episode.decisions)
    assert score["mission_duration_s"] == pytest.approx(30.0)
    assert score["decision_level"]["total_dwells"] == 300


def test_overlap_mapping_is_opt_in():
    cfg = SimulationConfig(
        band_count=2, time_slots=1, dwell_time_ms=50.0,
        total_spectrum_mhz=1000.0, receiver_ibw_mhz=1000.0,
        dwell_centres_mhz=np.array([250.0, 750.0]),
    )
    pdw = PDWStream(
        toa_us=np.array([1000.0]), freq_mhz=np.array([490.0]),
        pw_us=np.array([1.0]), aoa_deg=np.array([0.0]),
        amp_db=np.array([-70.0]), emitter_id=np.array([7]),
    )
    assert frequency_to_bands(490.0, cfg) == (0,)
    assert frequency_to_bands(490.0, cfg, overlap=True) == (0, 1)
    disjoint = discretise_pdw_to_grid(pdw, cfg)
    overlapping = discretise_pdw_to_grid(pdw, cfg, overlap=True)
    assert disjoint[0, 0].pulse_count == 1
    assert disjoint[1, 0].pulse_count == 0
    assert overlapping[0, 0].pulse_count == 1
    assert overlapping[1, 0].pulse_count == 1


def test_replay_signature_covers_emitter_attribution():
    env = _small_world()
    data, labels = env.recorded_pulses
    centres, halfwidth = env.receiver_geometry
    relabelled = TSRDStareEnvironment(
        env.n_bands, env.n_slots, env.occupancy_grid, None, env.config,
        stare_data=data, stare_labels=np.array([30, 10, 10, 10]),
        band_centres_mhz=centres, passband_halfwidth_mhz=halfwidth,
    )
    assert env.replay_signature() != relabelled.replay_signature()


def test_training_reward_modes_are_explicit():
    env = _small_world()
    env.config.false_alarm_probability = 1.0
    env.reset(seed=2)
    env.step(0)
    _, reward, _ = env.step_training(1, reward_mode="detector_positive")
    assert reward == 1.0  # Slot 1, band 1 is empty: positive detector reward.
    env.reset(seed=2)
    env.step(0)
    _, reward, _ = env.step_training(1, reward_mode="false_alarm_aware")
    assert reward == -1.0
    with pytest.raises(ValueError, match="reward mode"):
        env.step_training(0, reward_mode="unknown")


def test_scorecard_counts_retune_lost_pulse_as_false_alarm():
    centres = np.array([100.0, 600.0])
    data = np.array([[51_000.0, 100.0, 1.0, 0.0, -70.0]])
    cfg = SimulationConfig(
        band_count=2, time_slots=2, dwell_time_ms=50.0,
        receiver_ibw_mhz=500.0, total_spectrum_mhz=1000.0,
        retune_time_ms=2.0, detection_probability=1.0,
        false_alarm_probability=1.0,
    )
    env = TSRDStareEnvironment(
        2, 2, stare_pulse_occupancy(data, centres, 250.0, 2), None, cfg,
        stare_data=data, stare_labels=np.array([7]),
        band_centres_mhz=centres, passband_halfwidth_mhz=250.0,
    )
    env.reset(seed=1)
    trajectory = []
    for band in (1, 0):
        obs, _ = env.step(band)
        trajectory.append(SimpleNamespace(action=band, time_slot=obs["time_slot"], observation=obs))
    score = score_recorded_replay(env, trajectory)
    assert score["cell_level"]["occupied_cell_count"] == 1
    assert score["cell_level"]["eligible_selected_cells"] == 0
    assert score["event_level"]["true_detections"] == 0
    assert score["event_level"]["false_alarms"] == 2
    assert score["per_emitter"]["7"]["ttfi_censored"] is True


def _fixture_corpus(root: Path):
    fixture = Path(__file__).parent / "fixtures" / "tsrd"
    for split in ("train", "val", "test"):
        stare = root / "stare" / f"{split}_stare"
        scan = root / "scan" / f"{split}_scan"
        stare.mkdir(parents=True)
        scan.mkdir(parents=True)
        shutil.copyfile(fixture / "config_0_stare.h5", stare / "config_0.h5")
        shutil.copyfile(fixture / "config_0_scan.h5", scan / "config_0.h5")


def _stare_only_corpus(root: Path):
    fixture = Path(__file__).parent / "fixtures" / "tsrd" / "config_0_stare.h5"
    for split in ("train", "val", "test"):
        stare = root / "stare" / f"{split}_stare"
        stare.mkdir(parents=True)
        shutil.copyfile(fixture, stare / "config_0.h5")


def test_stare_only_requires_explicit_geometry_and_audits_metadata(tmp_path):
    corpus = tmp_path / "corpus"
    _stare_only_corpus(corpus)
    with pytest.raises(CorpusUnavailableError):
        list(iter_tsr_replay_pairs(corpus, "train"))
    assert list(iter_tsr_replay_pairs(corpus, "train", allow_stare_only=True))[0][1] is None
    centres = np.arange(36, dtype=float) * 500.0 + 250.0
    with pytest.raises(ValueError, match="tune centres"):
        TSRDStareEnvironment.from_stare_mode(
            str(corpus / "stare/train_stare/config_0.h5")
        )
    pool = TSRDTrainWorldPool(corpus, band_centres_mhz=centres)
    env, _ = pool.sample_world(seed=3, emitter_count=1)
    assert np.array_equal(env.receiver_geometry[0], centres)
    with pytest.raises(ValueError, match="increasing"):
        TSRDStareEnvironment.from_stare_mode(
            str(corpus / "stare/train_stare/config_0.h5"),
            band_centres_mhz=[250.0, 250.0],
        )
    audit = audit_corpus(corpus)
    assert all(row["files_audited"] == 1 for row in audit["splits"].values())
    assert audit["splits"]["train"]["files"][0]["schema_issues"] == []

    protocol = TSRDBenchmarkProtocol(
        corpus, tmp_path / "results", band_centres_mhz=centres, seed=3
    )

    def train(candidate_id, worlds, checkpoint, reward_mode):
        next(iter(worlds))
        checkpoint.write_text("ready", encoding="ascii")

    def load(checkpoint, band_count):
        return RoundRobinProbe(band_count)

    report = protocol.run(
        ["round_robin"], train, load, worlds_per_candidate=1, emitter_count=1,
        evaluate_test=True,
    )
    assert report["receiver_geometry"]["source"] == "explicit_centres"
    assert report["held_out_test"]["summary"]["files_evaluated"] == 1


def test_paired_scan_validation_replays_development_schedule_only(tmp_path):
    corpus = tmp_path / "corpus"
    _fixture_corpus(corpus)
    report = validate_development_pairs(corpus)
    assert report["test_split_used"] is False
    assert set(report["splits"]) == {"train", "val"}
    pair = report["splits"]["train"]["pairs"][0]
    assert pair["receiver_metadata_match"]
    assert pair["transmitter_metadata_match"]
    assert pair["passband_gate"]
    assert pair["dwell_count"] == 502
    assert {"50", "100"} <= set(pair["by_dwell_ms"])
    assert sum(x["dwells"] for x in pair["by_dwell_ms"].values()) == 502
    with pytest.raises(ValueError, match="train/val"):
        validate_development_pairs(corpus, splits=("test",))


def test_paired_scan_validation_rejects_wrong_companion(tmp_path):
    corpus = tmp_path / "corpus"
    _fixture_corpus(corpus)
    stare = corpus / "stare/train_stare/config_0.h5"
    scan = corpus / "scan/train_scan/config_0.h5"
    import h5py
    with h5py.File(scan, "r+") as f:
        f["metadata/receiver"].attrs["sensitivity_dbm"] += 1.0
    with pytest.raises(ValueError, match="Receiver metadata differs"):
        validate_pair(stare, scan)


def _measured_world():
    centres = np.array([100.0, 600.0])
    data = np.array([
        [1_000.0, 100.0, 2.0, 30.0, -40.0],
        [51_000.0, 100.0, 2.0, 30.0, -130.0],
    ])
    config = SimulationConfig(
        band_count=2, time_slots=2, dwell_time_ms=50.0,
        receiver_ibw_mhz=500.0, total_spectrum_mhz=1000.0,
        detection_probability=1.0, false_alarm_probability=1.0,
        retune_time_ms=0.0,
    )
    return TSRDStareEnvironment(
        2, 2, stare_pulse_occupancy(data, centres, 250.0, 2), None, config,
        stare_data=data, stare_labels=np.array([17, 17]),
        band_centres_mhz=centres, passband_halfwidth_mhz=250.0,
        receiver_profile="pdw_v2", amplitude_midpoint_db=-80.0,
        amplitude_scale_db=2.0, max_observed_pdws=8,
    )


def test_pdw_receiver_exposes_only_causal_measured_pulses_and_noise():
    env = _measured_world()
    env.reset(seed=19)
    first, _ = env.step(0)
    second, _ = env.step(1)
    assert first["hit"] is True
    assert first["receiver_measurement"]["pulse_count"] == 1
    assert first["receiver_measurement"]["pdws"][0]["frequency_mhz"] == 100.0
    assert second["hit"] is True  # Empty selected band, configured Pfa = 1.
    assert second["receiver_measurement"]["pulse_count"] == 1
    assert 350.0 <= second["receiver_measurement"]["pdws"][0]["frequency_mhz"] <= 850.0
    assert "emitter_id" not in str(first["receiver_measurement"])
    assert "emitter_id" not in str(second["receiver_measurement"])
    probe = RoundRobinProbe(2)
    probe._check_truth_leakage([first])
    probe._check_truth_leakage([second])
    contaminated = dict(first)
    contaminated["receiver_measurement"] = {
        "pulse_count": 1,
        "pdws": [{**first["receiver_measurement"]["pdws"][0], "emitter_id": 17}],
    }
    with pytest.raises(ValueError, match="measured PDW"):
        probe._check_truth_leakage([contaminated])


def test_episode_policy_receives_measurements_not_truth():
    class ObservationProbe(RoundRobinProbe):
        def reset(self, seed, scenario_config):
            self.seen_scenario = dict(scenario_config)
            self.seen_observations = []
            super().reset(seed, scenario_config)

        def select_action(self, observation_history, current_time_slot):
            if observation_history:
                self.seen_observations.append(dict(observation_history[-1]))
            return super().select_action(observation_history, current_time_slot)

    env = _measured_world()
    policy = ObservationProbe(2)
    run_episode(env, policy, seed=19)
    assert set(policy.seen_scenario).isdisjoint({"emitter_id", "hidden_truth", "labels"})
    assert policy.seen_observations
    for obs in policy.seen_observations:
        assert set(obs).isdisjoint({"emitter_id", "hidden_truth", "labels"})
        measurement = obs.get("receiver_measurement")
        if measurement is not None:
            assert all(set(pdw).isdisjoint({"emitter_id", "hidden_truth", "labels"})
                       for pdw in measurement["pdws"])


def test_pdw_receiver_miss_and_common_potential_noise():
    env = _measured_world()
    env.config.false_alarm_probability = 0.0
    env.reset(seed=23)
    env.step(1)
    observed, _ = env.step(0)
    assert observed["hit"] is False  # Very weak recorded pulse.
    assert observed["receiver_measurement"] is None
    env.reset(seed=23)
    env.step(0)
    repeated, _ = env.step(0)
    assert repeated == observed
    assert env.replay_signature() != _small_world().replay_signature()


def test_pdw_receiver_mixed_dwell_groups_only_completed_looks():
    env = _measured_world()
    episode = run_mixed_dwell_episode(env, _TwoSlotProbe(2), seed=29)
    measurement = episode.decision_observations[0]["receiver_measurement"]
    assert len(measurement) == 2
    assert measurement[0]["pulse_count"] == 1
    assert measurement[1] is None
    RoundRobinProbe(2)._check_truth_leakage(episode.decision_observations)


def test_pdw_receiver_profile_flows_through_train_and_evaluation(tmp_path):
    corpus = tmp_path / "corpus"
    _fixture_corpus(corpus)
    protocol = TSRDBenchmarkProtocol(
        corpus, tmp_path / "results", receiver_profile="pdw_v2",
        detection_probability=0.9, false_alarm_probability=0.05,
        amplitude_midpoint_db=-90.0, amplitude_scale_db=5.0,
    )

    def train(candidate_id, worlds, checkpoint, reward_mode):
        world, _ = next(iter(worlds))
        assert world.receiver_profile == "pdw_v2"
        assert world.config.detection_probability == 0.9
        checkpoint.write_text("ready", encoding="ascii")

    def load(checkpoint, band_count):
        return RoundRobinProbe(band_count)

    report = protocol.run(
        ["round_robin"], train, load, worlds_per_candidate=1, emitter_count=1,
        evaluate_test=True,
    )
    assert report["receiver_profile_options"]["receiver_profile"] == "pdw_v2"
    assert report["metric_contract_version"] == "tsrd_recorded_pulse_emitter_v4"
    assert report["held_out_test"]["summary"]["files_evaluated"] == 1


def test_pdw_receiver_scores_only_captured_emitter_when_band_is_shared():
    centres = np.array([100.0, 600.0])
    data = np.array([
        [1_000.0, 100.0, 2.0, 10.0, -40.0],
        [1_100.0, 100.0, 2.0, 20.0, -130.0],
    ])
    config = SimulationConfig(
        band_count=2, time_slots=2, dwell_time_ms=50.0,
        receiver_ibw_mhz=500.0, total_spectrum_mhz=1000.0,
        detection_probability=1.0, false_alarm_probability=0.0,
        retune_time_ms=0.0,
    )
    env = TSRDStareEnvironment(
        2, 2, stare_pulse_occupancy(data, centres, 250.0, 2), None, config,
        stare_data=data, stare_labels=np.array([1, 2]),
        band_centres_mhz=centres, passband_halfwidth_mhz=250.0,
        receiver_profile="pdw_v2", amplitude_midpoint_db=-80.0,
        amplitude_scale_db=2.0,
    )
    env.reset(seed=1)
    trajectory = []
    for band in (0, 0):
        observation, _ = env.step(band)
        trajectory.append(SimpleNamespace(
            action=band, time_slot=observation["time_slot"],
            observation=observation,
        ))
    score = score_recorded_replay(env, trajectory)
    assert score["per_emitter"]["1"]["emitter_intercepted"] is True
    assert score["per_emitter"]["2"]["emitter_intercepted"] is False
    assert score["illumination"]["illumination_interception_ratio"] == 0.5
    assert score["pulse_level"]["recorded_pulse_detected_ratio"] == 0.5
    assert score["emitter_interception"]["attribution_limit"] == "captured_recorded_pdw_labels"
    assert score["pulse_level"]["physical_pulse_interception_ratio"] is None


def test_train_pool_and_frozen_split_protocol(tmp_path):
    corpus = tmp_path / "corpus"
    _fixture_corpus(corpus)
    pool = TSRDTrainWorldPool(corpus)
    env, sources = pool.sample_world(seed=11, emitter_count=1)
    assert env.hidden_truth.grid.shape[0] == 1
    assert sources[0]["source_file"].startswith("stare/train_stare/")
    assert env.replay_signature() == pool.sample_world(seed=11, emitter_count=1)[0].replay_signature()

    protocol = TSRDBenchmarkProtocol(corpus, tmp_path / "results", seed=11)
    calls = []
    original_evaluate = protocol._evaluate_split

    def tracked_evaluate(split, checkpoint, loader):
        calls.append((split, checkpoint.name))
        return original_evaluate(split, checkpoint, loader)

    protocol._evaluate_split = tracked_evaluate

    def train(candidate_id, worlds, checkpoint, reward_mode):
        assert reward_mode == "detector_positive"
        world, provenance = next(iter(worlds))
        assert world.n_slots == 600
        assert all(source["source_file"].startswith("stare/train_stare/") for source in provenance)
        checkpoint.write_text("0" if candidate_id == "first" else "1", encoding="ascii")

    def load(checkpoint, band_count):
        return RoundRobinProbe(band_count, start_band=int(checkpoint.read_text()))

    report = protocol.run(
        ["first", "second"], train, load, worlds_per_candidate=1, emitter_count=1
    )
    assert report["evaluation_stage"] == "development"
    assert report["held_out_test"] is None
    assert [split for split, _ in calls] == ["val", "val"]
    report = protocol.finalize_test(load)
    assert [split for split, _ in calls] == ["val", "val", "test"]
    with pytest.raises(ValueError, match="development run without test results"):
        protocol.finalize_test(load)
    assert calls[-1][1] == f"{report['selected_candidate']}.checkpoint"
    assert report["held_out_test"]["summary"]["files_evaluated"] == 1
    assert report["validation"]["first"]["summary"]["files_evaluated"] == 1
    row = report["validation"]["first"]["per_config"][0]
    assert {
        "conditional_pd", "true_pfa", "opportunity_interception_ratio",
        "emitter_interception_ratio", "average_intercept_rate_hz",
        "average_seconds_per_intercept",
    } <= row["tier_a_ps26055"].keys()
    assert {
        "true_detection_yield_per_dwell", "empty_scan_fraction", "total_dwells",
        "action_entropy", "normalized_action_entropy", "unique_band_coverage",
        "exploration_fraction",
    } <= row["tier_b_scheduler"].keys()
    assert {
        "prediction_accuracy", "valid_prediction_count", "prediction_count",
        "average_intercept_time_error", "median_intercept_time_error",
        "p95_intercept_time_error",
    } <= row["tier_c_prediction"].keys()
    assert {
        "active_band_time_opportunities", "unique_occupied_bands",
        "environment_band_occupancy", "environment_time_occupancy",
    } <= row["tier_d_environment"].keys()
    assert {"mean_latency_s", "max_latency_s", "p95_latency_s"} <= row["tier_e_implementation"].keys()
    assert row["tier_a_ps26055"]["conditional_pd"] == row["cell_level"]["conditional_pd"]
    assert row["tier_a_ps26055"]["opportunity_interception_ratio"] == row["cell_level"]["occupied_cell_detection_ratio"]
    assert (tmp_path / "results" / "tsrd_benchmark_report.json").is_file()
    manifest = json.loads((tmp_path / "results" / "tsrd_run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["input_inventory"]["basis"] == "relative_path_size_mtime_not_content_hash"
    assert len(manifest["input_inventory"]["splits"]["train"]) == 2
    assert manifest["checkpoint_sha256"]["first"] == report["validation"]["first"]["checkpoint_sha256"]


def test_scorecard_consistency_rejects_corrupted_counts():
    env = _small_world()
    env.reset(seed=1)
    trajectory = []
    for band in (0, 0, 1):
        obs, _ = env.step(band)
        trajectory.append(SimpleNamespace(action=band, time_slot=obs["time_slot"], observation=obs))
    score = score_recorded_replay(env, trajectory)
    _assert_scorecard_consistent(score, trajectory)
    score["event_level"]["false_alarms"] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        _assert_scorecard_consistent(score, trajectory)


def test_empty_recording_keeps_undefined_replay_metrics_null():
    cfg = SimulationConfig(
        band_count=2, time_slots=3, dwell_time_ms=50.0,
        receiver_ibw_mhz=500.0, total_spectrum_mhz=1000.0,
        detection_probability=1.0, false_alarm_probability=0.0,
    )
    env = TSRDStareEnvironment(
        2, 3, np.zeros((2, 3), dtype=bool), None, cfg,
        stare_data=np.empty((0, 5)), stare_labels=np.empty(0, dtype=int),
        band_centres_mhz=np.array([100.0, 600.0]),
        passband_halfwidth_mhz=250.0,
    )
    episode = run_episode(env, RoundRobinProbe(2), seed=7)
    score = score_recorded_replay(env, episode.trajectory)
    _assert_scorecard_consistent(score, episode.trajectory)
    assert score["pulse_level"]["recorded_pulses"] == 0
    assert score["pulse_level"]["recorded_pulse_coverage_ratio"] is None
    assert score["illumination"]["illumination_interception_ratio"] is None
    assert score["emitter_interception"]["unique_emitter_interception_rate"] is None
    assert score["ttfi"]["censoring_fraction"] is None
    assert score["cell_level"]["conditional_pd"] is None
    assert score["cell_level"]["true_pfa"] == 0.0
    assert _summarize([score])["pooled_illumination_interception_ratio"] is None


def test_pooled_replay_ratio_uses_counts_not_mean_of_world_ratios():
    first = _small_world()
    first_episode = run_episode(first, RoundRobinProbe(2), seed=1)
    first_score = score_recorded_replay(first, first_episode.trajectory)
    one_pulse = np.array([[1_000.0, 100.0, 1.0, 0.0, -70.0]])
    cfg = SimulationConfig(
        band_count=2, time_slots=3, dwell_time_ms=50.0,
        receiver_ibw_mhz=500.0, total_spectrum_mhz=1000.0,
        detection_probability=1.0, false_alarm_probability=0.0,
    )
    second = TSRDStareEnvironment(
        2, 3, stare_pulse_occupancy(one_pulse, np.array([100.0, 600.0]), 250.0, 3),
        None, cfg, stare_data=one_pulse, stare_labels=np.array([5]),
        band_centres_mhz=np.array([100.0, 600.0]), passband_halfwidth_mhz=250.0,
    )
    second_episode = run_episode(second, RoundRobinProbe(2), seed=1)
    second_score = score_recorded_replay(second, second_episode.trajectory)
    rows = [first_score, second_score]
    summary = _summarize(rows)
    numerator = sum(r["illumination"]["detected_emitter_slot_opportunity_count"] for r in rows)
    denominator = sum(r["illumination"]["emitter_slot_opportunity_count"] for r in rows)
    per_world_mean = np.mean([r["illumination"]["illumination_interception_ratio"] for r in rows])
    assert summary["pooled_illumination_interception_ratio"] == pytest.approx(numerator / denominator)
    assert summary["pooled_illumination_interception_ratio"] != pytest.approx(per_world_mean)


def test_scheduler_history_excludes_nondeterministic_profiling_fields():
    class RecordingProbe(RoundRobinProbe):
        def __init__(self, bands):
            super().__init__(bands)
            self.seen = []

        def select_action(self, observation_history, current_time_slot):
            if observation_history:
                self.seen.append(dict(observation_history[-1]))
            return super().select_action(observation_history, current_time_slot)

    env = _small_world()
    first, second = RecordingProbe(2), RecordingProbe(2)
    episode_a = run_episode(env, first, seed=19)
    episode_b = run_episode(env, second, seed=19)
    profiling = {"select_action_ms", "predict_ms", "wall_clock_ms", "memory_delta_bytes"}
    assert first.seen == second.seen
    assert all(profiling.isdisjoint(obs) for obs in first.seen)
    assert profiling <= set(episode_a.trajectory[0].observation)
    assert np.array_equal(episode_a.actions, episode_b.actions)
    assert np.array_equal(episode_a.hits, episode_b.hits)


def test_out_of_mission_pdw_is_reported_but_not_coverage_denominator():
    centres = np.array([100.0, 600.0])
    data = np.array([
        [1_000.0, 100.0, 1.0, 0.0, -70.0],
        [160_000.0, 100.0, 1.0, 0.0, -70.0],
    ])
    cfg = SimulationConfig(
        band_count=2, time_slots=3, dwell_time_ms=50.0,
        receiver_ibw_mhz=500.0, total_spectrum_mhz=1000.0,
        detection_probability=1.0, false_alarm_probability=0.0,
    )
    env = TSRDStareEnvironment(
        2, 3, stare_pulse_occupancy(data, centres, 250.0, 3), None, cfg,
        stare_data=data, stare_labels=np.array([5, 5]),
        band_centres_mhz=centres, passband_halfwidth_mhz=250.0,
    )
    episode = run_episode(env, RoundRobinProbe(2), seed=3)
    score = score_recorded_replay(env, episode.trajectory)
    _assert_scorecard_consistent(score, episode.trajectory)
    assert score["pulse_level"]["recorded_pulses"] == 2
    assert score["pulse_level"]["recorded_pulses_in_mission"] == 1
    assert score["pulse_level"]["recorded_pulses_outside_mission"] == 1
    assert score["pulse_level"]["recorded_pulse_coverage_ratio"] == 1.0
    pooled = _summarize([score])
    assert pooled["recorded_pulses_in_mission"] == 1
    assert pooled["recorded_pulses_outside_mission"] == 1
    assert pooled["recorded_pulse_coverage_ratio"] == 1.0


def test_failed_training_leaves_incomplete_manifest_and_no_report(tmp_path):
    corpus = tmp_path / "corpus"
    _fixture_corpus(corpus)
    output = tmp_path / "failed_results"
    protocol = TSRDBenchmarkProtocol(corpus, output)

    def fail_train(candidate_id, worlds, checkpoint, reward_mode):
        raise RuntimeError("deliberate training failure")

    with pytest.raises(RuntimeError, match="deliberate"):
        protocol.run(
            ["failed"], fail_train, lambda checkpoint, bands: RoundRobinProbe(bands),
            worlds_per_candidate=1, emitter_count=1,
        )
    manifest = json.loads((output / "tsrd_run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure"]["phase"] == "training_or_validation"
    assert manifest["failure"]["type"] == "RuntimeError"
    assert not (output / "tsrd_benchmark_report.json").exists()


def test_reward_screen_uses_development_splits_only(tmp_path):
    corpus = tmp_path / "corpus"
    _fixture_corpus(corpus)
    with pytest.raises(ValueError, match="only train or validation"):
        screen_rewards(corpus, split="test", max_files=1)
    report = screen_rewards(
        corpus, split="val", max_files=1, seed=3,
        coverage_weight=0.2, discovery_bonus=2.0,
    )
    assert report["files_screened"] == 1
    assert set(report["policy_summary"]) == {
        "single_band_camper", "two_band_ping_pong", "greedy_busy_band",
        "round_robin", "uniform_random", "periodic_two_slot_sweep",
    }
    assert "detector_positive" in report["pathology_flags"]
    assert "coverage_balanced" in report["pathology_flags"]
    assert "oracle_discovery_focus" in report["pathology_flags"]


def test_checkpoint_protocol_mixed_dwell_profile(tmp_path):
    corpus = tmp_path / "corpus"
    _fixture_corpus(corpus)
    protocol = TSRDBenchmarkProtocol(
        corpus, tmp_path / "mixed_results", dwell_profile="mixed_50_100_ms"
    )

    def train(candidate_id, worlds, checkpoint, reward_mode):
        next(iter(worlds))
        checkpoint.write_text("ready", encoding="ascii")

    def load(checkpoint, band_count):
        assert checkpoint.read_text(encoding="ascii") == "ready"
        return _TwoSlotProbe(band_count)

    report = protocol.run(
        ["two_slot"], train, load, worlds_per_candidate=1, emitter_count=1,
        evaluate_test=True,
    )
    assert report["dwell_profile"] == "mixed_50_100_ms"
    assert report["held_out_test"]["per_config"][0]["decision_level"]["total_dwells"] == 300

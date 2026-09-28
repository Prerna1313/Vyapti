"""Regression checks for the TSRD receiver's observable contract."""

import numpy as np
import pytest
from types import SimpleNamespace

from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.core.metrics import MetricsConfig
from vyapti_simulator.core.scheduler_interface import PERMITTED_OBSERVATION_KEYS
from vyapti_simulator.tsrd.tsrd_adapter import PDWStream
from vyapti_simulator.tsrd.tsrd_environment import (
    DetectionConfig,
    TSRDEnvironment,
    TSRDStareEnvironment,
)
from vyapti_simulator.tsrd.tsrd_metrics import TSRDMetricsEngine


HIDDEN_FIELDS = {
    "pulse_count", "energy_db", "max_amplitude_db", "mean_pulse_width_us",
    "mean_aoa_deg", "snr_db_estimate", "coherent_integration_gain_db",
    "n_pulses_dominant_emitter",
}


def _config():
    return SimulationConfig(
        receiver_ibw_mhz=500.0,
        total_spectrum_mhz=1000.0,
        band_count=2,
        time_slots=2,
        dwell_time_ms=10.0,
        retune_time_ms=2.0,
        detection_probability=1.0,
        false_alarm_probability=0.0,
    )


def _stream(toa_us):
    return PDWStream(
        toa_us=np.array([toa_us], dtype=np.float32),
        freq_mhz=np.array([100.0], dtype=np.float32),
        pw_us=np.array([1.0], dtype=np.float32),
        aoa_deg=np.array([10.0], dtype=np.float32),
        amp_db=np.array([-75.0], dtype=np.float32),
        emitter_id=np.array([7], dtype=np.int64),
    )


def test_oracle_aided_integration_is_not_the_default():
    assert not DetectionConfig().coherent_integration_enabled


@pytest.mark.parametrize("toa_us,expected_hit", [(11_000.0, False), (13_000.0, True)])
def test_main_tsrd_receiver_respects_retune_window(toa_us, expected_hit):
    env = TSRDEnvironment(
        _stream(toa_us),
        _config(),
        DetectionConfig(false_alarm_probability=0.0),
        seed=7,
    )
    env.reset(seed=7)
    env.step(1)
    observation, _ = env.step(0)
    assert observation["hit"] is expected_hit
    assert observation["dwell_elapsed_s"] == pytest.approx(0.008)
    assert env.receiver_accounting()["retune_count"] == 1.0
    assert HIDDEN_FIELDS.isdisjoint(observation)
    assert set(observation) <= PERMITTED_OBSERVATION_KEYS


@pytest.mark.parametrize("toa_us,expected_hit", [(11_000.0, False), (13_000.0, True)])
def test_stare_receiver_respects_retune_window(toa_us, expected_hit):
    truth = np.zeros((2, 2), dtype=bool)
    truth[0, 1] = True
    stare_data = np.array([[toa_us, 100.0, 1.0, 10.0, -75.0]])
    env = TSRDStareEnvironment(
        2, 2, truth, np.zeros_like(truth), _config(), stare_data=stare_data
    )
    env.reset(seed=7)
    env.step(1)
    observation, _ = env.step(0)
    assert observation["hit"] is expected_hit
    assert observation["dwell_elapsed_s"] == pytest.approx(0.008)
    assert env.receiver_accounting()["retune_count"] == 1.0
    assert HIDDEN_FIELDS.isdisjoint(observation)
    assert set(observation) <= PERMITTED_OBSERVATION_KEYS


def test_stare_scorecard_labels_recorded_pulse_basis_and_uses_slot_duration():
    trajectory = [
        SimpleNamespace(action=0, time_slot=0, observation={"hit": True}, prediction=None),
        SimpleNamespace(action=0, time_slot=1, observation={"hit": True}, prediction=None),
    ]
    grid = np.array([[True, False], [False, True]])
    metrics = TSRDMetricsEngine(MetricsConfig()).compute_comprehensive_metrics(
        trajectory, grid, scenario_config={"dwell_time_ms": 10.0, "retune_time_ms": 2.0}
    )
    tier_a = metrics["tier_a_ps26055"]
    assert tier_a["metric_contract_version"] == "tsrd_recorded_pulse_v4_legacy_full_slot"
    assert tier_a["truth_basis"] == "recorded_stare_pulse_band_slot"
    assert tier_a["opportunity_denominator"] == 2
    assert tier_a["opportunity_interception_ratio"] == pytest.approx(0.5)
    assert tier_a["average_intercept_rate_hz"] == pytest.approx(50.0)
    assert metrics["per_band_metrics"]["0"]["true_hits"] == 1
    assert metrics["per_band_metrics"]["0"]["conditional_pd"] == pytest.approx(1.0)


def test_stare_replay_uses_scan_metadata_only_for_geometry():
    from pathlib import Path
    from vyapti_simulator.tsrd.tsrd_adapter import load_stare_mode_as_occupancy_grid

    fixture = Path(__file__).parent / "fixtures" / "tsrd"
    stare = fixture / "config_0_stare.h5"
    scan = fixture / "config_0_scan.h5"
    with pytest.raises(ValueError, match="tune centres"):
        TSRDStareEnvironment.from_stare_mode(str(stare))

    env = TSRDStareEnvironment.from_stare_mode(str(stare), str(scan))
    grid = load_stare_mode_as_occupancy_grid(str(stare), scan_file=str(scan))
    assert env.config.band_count == 36
    assert env.config.time_slots == 600
    assert env.config.dwell_time_ms == 50.0
    assert np.array_equal(grid, env.occupancy_grid)
    assert np.array_equal(env.hidden_truth.grid.any(axis=0), grid)
    assert not grid[0].any()
    assert grid[2].any()
    assert grid[3].any()
    assert env.replay_signature() != "dummy_signature"


def test_retune_lost_pulse_is_not_a_true_hit_in_replay_metrics():
    cfg = _config()
    cfg.false_alarm_probability = 1.0
    truth = np.zeros((2, 2), dtype=bool)
    truth[0, 1] = True
    env = TSRDStareEnvironment(
        2, 2, truth, None, cfg,
        stare_data=np.array([[11_000.0, 100.0, 1.0, 0.0, -75.0]]),
    )
    env.reset(seed=5)
    first, _ = env.step(1)
    second, _ = env.step(0)
    assert second["hit"] is True
    trajectory = [
        SimpleNamespace(action=1, time_slot=0, observation=first, prediction=None),
        SimpleNamespace(action=0, time_slot=1, observation=second, prediction=None),
    ]
    metrics = TSRDMetricsEngine(MetricsConfig()).compute_comprehensive_metrics(
        trajectory, truth, receiver_env=env
    )["tier_a_ps26055"]
    assert metrics["opportunity_interception_ratio"] == 0.0
    assert metrics["conditional_pd"] is None
    assert metrics["true_pfa"] == 1.0


def test_training_reward_distinguishes_recorded_capture_from_false_alarm():
    cfg = _config()
    cfg.false_alarm_probability = 1.0
    truth = np.zeros((2, 2), dtype=bool)
    truth[0, 1] = True
    env = TSRDStareEnvironment(
        2, 2, truth, None, cfg,
        stare_data=np.array([[11_000.0, 100.0, 1.0, 0.0, -75.0]]),
    )
    env.reset(seed=5)
    env.step(1)
    observation, reward, done = env.step_training(0)
    assert observation["hit"] is True
    assert reward == -1.0
    assert done
    assert HIDDEN_FIELDS.isdisjoint(observation)

    env = TSRDStareEnvironment(
        2, 2, truth, None, cfg,
        stare_data=np.array([[13_000.0, 100.0, 1.0, 0.0, -75.0]]),
    )
    env.reset(seed=5)
    env.step(1)
    _, reward, done = env.step_training(0)
    assert reward == 1.0
    assert done


def test_scan_fixture_supports_metadata_bandwidth_as_halfwidth():
    from pathlib import Path
    import h5py

    scan = Path(__file__).parent / "fixtures" / "tsrd" / "config_0_scan.h5"
    with h5py.File(scan, "r") as f:
        data = f["data"][:]
        centres = f["metadata/receiver/dwell_centres_mhz"][:]
        dwell_s = f["metadata/receiver/dwell_times_s"][:]
        halfwidth = float(f["metadata/receiver"].attrs["bandwith_mhz"])
    phase_s = (data[:, 0] * 1e-6) % float(dwell_s.sum())
    active_band = np.searchsorted(np.cumsum(dwell_s), phase_s, side="right")
    active_band = np.clip(active_band, 0, len(centres) - 1)
    offset_mhz = np.abs(data[:, 1] - centres[active_band])
    assert np.mean(offset_mhz <= halfwidth) > 0.98
    assert np.mean(offset_mhz <= halfwidth / 2) < 0.8


def test_experiment_runner_uses_tsr_split_and_oir_primary_metric(tmp_path):
    import shutil
    from pathlib import Path
    from vyapti_simulator.experiments.experiment_runner import (
        ExperimentConfig, ExperimentRunner,
    )
    from vyapti_simulator.qualification.probes import RoundRobinProbe
    from vyapti_simulator.tsrd.corpus_loader import iter_tsr_replay_pairs

    source = Path(__file__).parent / "fixtures" / "tsrd"
    stare_dir = tmp_path / "stare" / "train_stare"
    scan_dir = tmp_path / "scan" / "train_scan"
    stare_dir.mkdir(parents=True)
    scan_dir.mkdir(parents=True)
    shutil.copyfile(source / "config_0_stare.h5", stare_dir / "config_0.h5")
    shutil.copyfile(source / "config_0_scan.h5", scan_dir / "config_0.h5")
    assert list(iter_tsr_replay_pairs(tmp_path, "train")) == [
        (stare_dir / "config_0.h5", scan_dir / "config_0.h5")
    ]
    with pytest.raises(ValueError, match="split"):
        list(iter_tsr_replay_pairs(tmp_path, "unknown"))
    config = ExperimentConfig(
        train_seeds=[1, 2], eval_seeds=[3], test_seeds=[4],
        bootstrap_samples=16, tsrd_dataset_root=str(tmp_path),
        tsrd_split="train",
    )
    report = ExperimentRunner(config).run_paired_comparison(
        schedulers={
            "sweep_0": lambda: RoundRobinProbe(36),
            "sweep_1": lambda: RoundRobinProbe(36, start_band=1),
        },
        density=1, use_tsr=True, tsrd_scenario="config_0",
    )
    assert report["band_count"] == 36
    assert report["time_slots"] == 600
    assert report["primary_metric"] == "tier_a_ps26055.opportunity_interception_ratio"
    assert report["tsrd_metric_contract_version"] == "tsrd_recorded_pulse_v4"
    assert report["rf_physics"]["source"] == "tsrd_recorded_stare_pdw_replay"
    assert report["per_scheduler_results"]["sweep_0"]["mean_opportunity_interception_ratio"] >= 0

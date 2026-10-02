"""Held-out gates retain recorded frequency agility and never enter TRAIN."""

import numpy as np
import pytest

from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.tsrd.evaluation_illumination import (
    IlluminationSettings, _intervals, apply_evaluation_illumination, annotate_illumination_score,
)
from vyapti_simulator.core.metrics import TrajectoryStep
from vyapti_simulator.tsrd.replay_scorecard import score_recorded_replay
from vyapti_simulator.tsrd.tsrd_adapter import stare_pulse_occupancy
from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment


def _world():
    centres = np.array([250.0, 750.0, 1250.0])
    cfg = SimulationConfig(band_count=3, time_slots=100, dwell_time_ms=50.0,
                           detection_probability=1.0, false_alarm_probability=0.0,
                           retune_time_ms=1.0)
    toa = np.arange(0, 5_000_000, 1000, dtype=float)
    # One fixed-frequency and one frequency-agile emitter coexist.
    fixed = np.column_stack((toa, np.full(len(toa), 250.0), np.ones(len(toa)),
                             np.zeros(len(toa)), np.full(len(toa), -50.0)))
    agile = fixed.copy()
    agile[:, 1] = centres[np.arange(len(toa)) % 3]
    data = np.concatenate((fixed, agile))
    labels = np.repeat([3, 8], len(toa))
    occupancy = stare_pulse_occupancy(data, centres, 100.0, cfg.time_slots)
    return TSRDStareEnvironment(3, cfg.time_slots, occupancy, None, cfg,
                                stare_data=data, stare_labels=labels,
                                band_centres_mhz=centres, passband_halfwidth_mhz=100.0)


@pytest.mark.parametrize("mode", ["normal", "beam_periodic", "beam_stochastic"])
def test_each_condition_is_split_guarded_and_preserves_source(tmp_path, mode):
    source = _world()
    raw, raw_labels = source.recorded_pulses
    settings = IlluminationSettings(mode=mode)
    with pytest.raises(ValueError, match="only for VAL or TEST"):
        apply_evaluation_illumination(source, split="train", settings=settings,
                                     illumination_seed=44, receiver_seed=5)
    first, audit = apply_evaluation_illumination(source, split="val", settings=settings,
                                                illumination_seed=44, receiver_seed=5)
    second, again = apply_evaluation_illumination(_world(), split="val", settings=settings,
                                                 illumination_seed=44, receiver_seed=5)
    assert audit == again
    assert first.replay_signature() == second.replay_signature()
    np.testing.assert_array_equal(source.recorded_pulses[0], raw)
    np.testing.assert_array_equal(source.recorded_pulses[1], raw_labels)
    visible, labels = first.recorded_pulses
    for label in np.unique(labels):
        selected = visible[labels == label]
        original = raw[raw_labels == label]
        indices = np.searchsorted(original[:, 0], selected[:, 0])
        np.testing.assert_array_equal(selected, original[indices])
    if mode == "normal":
        np.testing.assert_array_equal(visible, raw)
    else:
        assert 0 < len(visible) < len(raw)
        # Spatial gating coexists with an unchanged agile RF sequence.
        assert len(np.unique(visible[labels == 8, 1])) == 3
        different, changed = apply_evaluation_illumination(
            source, split="test", settings=settings, illumination_seed=45, receiver_seed=5)
        assert changed["visibility_mask_sha256"] != audit["visibility_mask_sha256"]
        assert different.replay_signature() != first.replay_signature()
    assert [first.step(t % 3)[0] for t in range(20)] == [second.step(t % 3)[0] for t in range(20)]


def test_ctmc_has_continuous_exponential_residence_and_complete_partition():
    settings = IlluminationSettings(mode="beam_stochastic")
    intervals, parameters = _intervals(settings, 100.0, np.random.default_rng(5))
    assert intervals[0]["start_s"] == 0
    assert intervals[-1]["end_s"] == 100
    assert parameters["off_to_on_rate_hz"] == 1 / 1.8
    assert parameters["on_to_off_rate_hz"] == 5
    for previous, current in zip(intervals, intervals[1:]):
        assert previous["end_s"] == current["start_s"]
        assert previous["illuminated"] != current["illuminated"]
    assert any(not np.isclose(interval["end_s"] / 0.05, round(interval["end_s"] / 0.05))
               for interval in intervals[:-1])


def test_fully_occluded_emitters_remain_in_source_population_as_censored():
    world, audit = apply_evaluation_illumination(
        _world(), split="val",
        settings=IlluminationSettings(mode="beam_stochastic", mean_on_s=0.001, mean_off_s=1e9),
        illumination_seed=44, receiver_seed=5,
    )
    assert len(world.recorded_pulses[0]) == 0
    trajectory = []
    while not world.done:
        slot = world.current_slot
        obs, _ = world.step(slot % 3)
        trajectory.append(TrajectoryStep(slot % 3, obs, slot))
    score = score_recorded_replay(world, trajectory)
    annotate_illumination_score(score, audit, world.config.slot_duration_s())
    assert score["cell_level"]["conditional_pd"] is None
    assert score["illumination_stress"]["source_eligible_emitters"] == 2
    assert score["illumination_stress"]["source_population_interception_rate"] == 0
    assert all(record["ttfi_censored"] for record in
               score["illumination_stress"]["source_relative_ttfi"].values())

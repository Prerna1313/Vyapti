"""Receiver measurements must never carry source identity or hidden state."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError

import pytest

from tests.test_train250_cache import _small_pool
from vyapti_simulator.core.receiver_observation import ReceiverObservation
from vyapti_simulator.core.environment import SimulationConfig, VyaptiEnv, EmitterConfig, EmitterBehaviorType
from vyapti_simulator.system_b.tsrd.train250_cache import build_train_pool_from_cache


FORBIDDEN = {"truth", "emitter_id", "source_config_id", "source_label", "pool_id",
             "future_frequency", "hidden_truth", "beam_state", "illumination_state",
             "type_id", "operating_mode", "world_seed", "SNRtruth", "snr_truth",
             "true_occupancy", "future_pulse", "time_offset_us", "local_label"}


def _observation():
    return {"time_slot": 0, "selected_band": 0, "hit": True,
            "retune_cost_s": 0.0, "dwell_elapsed_s": 0.05,
            "receiver_metadata": {"dwell_time_ms": 50.0, "retune_time_ms": 1.0}}


def _assert_no_forbidden(value):
    if isinstance(value, Mapping):
        assert not set(value).intersection(FORBIDDEN)
        for nested in value.values():
            _assert_no_forbidden(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_no_forbidden(nested)


@pytest.mark.parametrize("field", sorted(FORBIDDEN))
def test_unknown_fields_rejected_at_observation_and_metadata_boundary(field):
    raw = _observation()
    with pytest.raises(ValueError, match="Forbidden"):
        ReceiverObservation.from_mapping(dict(raw, **{field: 17}))
    raw["receiver_metadata"][field] = 17
    with pytest.raises(ValueError, match="nested"):
        ReceiverObservation.from_mapping(raw)


def test_nested_pdw_identity_rejected_and_measurements_are_immutable():
    raw = _observation()
    pdw = {"toa_offset_us": 1.0, "frequency_mhz": 250.0, "pulse_width_us": 1.0,
           "aoa_deg": 0.0, "amplitude_db": -80.0}
    raw["receiver_measurement"] = {"pulse_count": 1, "pdws": [dict(pdw, emitter_id=7)]}
    with pytest.raises(ValueError, match="nested"):
        ReceiverObservation.from_mapping(raw)
    raw["receiver_measurement"]["pdws"] = [pdw]
    obs = ReceiverObservation.from_mapping(raw)
    with pytest.raises((FrozenInstanceError, TypeError, AttributeError)):
        obs.emitter_id = 7
    with pytest.raises(FrozenInstanceError):
        obs.hit = False
    with pytest.raises(FrozenInstanceError):
        obs.receiver_measurement.pdws[0].frequency_mhz = 1000.0
    with pytest.raises(TypeError):
        obs.receiver_metadata["source_config_id"] = "config_7"
    # A serialized copy can be changed without changing the live observation.
    copy = obs.to_dict()
    copy["receiver_metadata"]["source_config_id"] = "config_7"
    assert "source_config_id" not in obs.receiver_metadata
    _assert_no_forbidden(obs)


@pytest.mark.parametrize("profile", ["binary_v1", "pdw_v2"])
def test_every_cached_episode_observation_is_typed_and_label_free(tmp_path, profile):
    data, cache = _small_pool(tmp_path)
    pool, _ = build_train_pool_from_cache(data, cache, expected_configs=2,
                                         receiver_profile=profile)
    world, sources = pool.sample_world(seed=5, emitter_count=2)
    assert sources[0]["source_label"] is not None
    world.reset(seed=9)
    while not world.done:
        observation, leaked = world.step(world.current_slot % world.n_bands)
        assert type(observation) is ReceiverObservation
        assert not leaked
        _assert_no_forbidden(observation)


def test_shared_synthetic_receiver_has_the_same_closed_observation_type():
    world = VyaptiEnv(SimulationConfig(band_count=3, time_slots=20), [
        EmitterConfig(emitter_id=7, behavior=EmitterBehaviorType.CONTINUOUS_FIXED, active_bands=[0])
    ])
    world.reset(seed=9)
    while not world.done:
        observation, leaked = world.step(world.current_time_slot % 3)
        assert type(observation) is ReceiverObservation
        assert not leaked
        _assert_no_forbidden(observation)

"""Generic configuration validation and metadata completeness checks."""

from dataclasses import fields

import pytest

from vyapti_simulator.config.master_config import MasterSimulationConfig
from vyapti_simulator.core.environment import SimulationConfig


@pytest.mark.parametrize(
    "field,value",
    [
        ("band_count", 0),
        ("time_slots", 0),
        ("total_spectrum_mhz", 0.0),
        ("dwell_time_ms", 0.0),
        ("detection_probability", 1.2),
        ("false_alarm_probability", -0.1),
    ],
)
def test_invalid_config_rejected_at_construction(field, value):
    with pytest.raises(ValueError, match=field):
        SimulationConfig(**{field: value})


def test_master_config_provenance_covers_every_parameter():
    config = MasterSimulationConfig()
    parameters = {f.name for f in fields(config)} - {"provenance_map"}
    assert parameters <= config.provenance_map.keys()

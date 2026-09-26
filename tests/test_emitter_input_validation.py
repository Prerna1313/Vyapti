"""Constructor validation must work with and without Python assertions."""

import pytest

from src.emitter_models import (
    FixedContinuousEmitter,
    FixedIntermittentEmitter,
    FrequencyAgileEmitter,
    FrequencyAgileScanningEmitter,
    IntervalOnOffPolicy,
    PriJitterEmitter,
    ScanningEmitter,
)


BASE = {
    "emitter_id": 1,
    "pri_sec": 1e-3,
    "pulse_width_sec": 1e-6,
    "power_dbm": -50.0,
}


@pytest.mark.parametrize(
    ("constructor", "params", "message"),
    [
        (FixedContinuousEmitter, {**BASE, "center_freq_hz": 100e6}, "Frequency"),
        (FixedIntermittentEmitter, {**BASE, "center_freq_hz": 1e9, "on_sec": 0, "off_sec": 0}, "Duty cycle"),
        (ScanningEmitter, {**BASE, "center_freq_hz": 1e9, "scan_period_sec": 0.1}, "Scan period"),
        (FrequencyAgileEmitter, {**BASE, "freq_list_hz": []}, "freq_list_hz"),
        (FrequencyAgileScanningEmitter, {**BASE, "freq_list_hz": []}, "freq_list_hz"),
        (PriJitterEmitter, {**BASE, "center_freq_hz": 1e9, "nominal_pri_sec": 1e-3,
                            "jitter_fraction": 0}, "Jitter fraction"),
        (IntervalOnOffPolicy, {"mean_on_sec": 0}, "mean_on_sec"),
    ],
)
def test_invalid_constructor_input_raises_value_error(constructor, params, message):
    if constructor is PriJitterEmitter:
        params = {key: value for key, value in params.items() if key != "pri_sec"}
    with pytest.raises(ValueError, match=message):
        constructor(**params)

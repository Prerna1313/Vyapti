from __future__ import annotations

from vyapti_simulator.system_c.rf.calibration import run_receiver_calibration
from vyapti_simulator.system_c.rf.pulse_detector import PulseDetectorConfig


def test_receiver_calibration_is_seeded_and_reports_tick_level_intervals():
    config = PulseDetectorConfig(
        dsp_sample_rate_hz=1e6,
        tick_interval_s=1e-3,
        cfar_db=8.0,
        cfar_train_cells=4,
        cfar_guard_cells=4,
        pulse_width_s=8e-6,
        chirp_bandwidth_hz=100e3,
    )
    kwargs = dict(
        trials=3,
        seed=17,
        snr_grid_db=(0.0, 10.0),
        cfar_offset_grid_db=(6.0, 12.0),
        config=config,
    )
    first = run_receiver_calibration(**kwargs)
    second = run_receiver_calibration(**kwargs)

    assert first == second
    assert first["trials_per_point"] == 3
    assert len(first["pd_vs_snr"]) == 2
    assert len(first["false_alarm_vs_cfar_offset"]) == 2
    for row in first["pd_vs_snr"]:
        assert row["trials"] == 3
        assert len(row["pd_ci95_wilson"]) == 2
    for row in first["false_alarm_vs_cfar_offset"]:
        assert row["trials"] == 3
        assert len(row["pfa_per_tick_ci95_wilson"]) == 2
        assert "false_alarm_probability_per_tick" in row

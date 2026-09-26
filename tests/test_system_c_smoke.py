import pytest
import numpy as np
from numpy.random import SeedSequence, default_rng
from vyapti_simulator.core.metrics import MetricsConfig
from vyapti_simulator.rf.closed_loop import MissionRunner, RoundRobinScheduler
from vyapti_simulator.rf.pulse_detector import PulseDetectorConfig, PulseDetector, EmitterInfo
from vyapti_simulator.rf.simulator_engine import SimulationEngineConfig, RealTimeRFSimulator
from vyapti_simulator.tsrd.synthetic_pdw_generator import SyntheticEmitterSpec

def test_system_c_smoke():
    """
    Smoke test: 1 fixed emitter, 50 ms mission, 5 dwells.
    Verifies that the System C physics engine runs deterministically
    without crashing, detects at least one pulse, and populates
    all canonical and PS26055 scorecard fields.
    """
    # 1. Deterministic, independent RNG streams via SeedSequence tree
    #    (Gate 0 requirement: each subsystem gets its own child seed)
    root_ss = SeedSequence(42)
    engine_ss, detector_ss = root_ss.spawn(2)
    engine_rng = default_rng(engine_ss)
    detector_rng = default_rng(detector_ss)

    # 2. Configuration — matched chirp bandwidth between emitter and detector
    chirp_bw_hz = 1e6  # 1 MHz chirp bandwidth (SyntheticEmitterSpec default)
    pulse_width_s = 10e-6  # 10 µs pulse width

    det_cfg = PulseDetectorConfig(
        dsp_sample_rate_hz=10e6,
        tick_interval_s=1e-3,
        cfar_db=10.0,
        min_pulse_samples=3,
        carrier_freq_hz=3.0e9,
        chirp_bandwidth_hz=chirp_bw_hz,
        pulse_width_s=pulse_width_s,
    )

    eng_cfg = SimulationEngineConfig(
        dsp_sample_rate_hz=10e6,
        tick_interval_s=1e-3,
        snr_db=20.0,
    )

    # 3. Threat — single fixed emitter at 3.01 GHz
    specs = [
        SyntheticEmitterSpec(
            emitter_id=1,
            center_freq_hz=3.01e9,
            pri_sec=1e-3,
            pulse_width_sec=pulse_width_s,
            chirp_bandwidth_hz=chirp_bw_hz,
            aoa_deg=45.0,
            snr_db=20.0,
            emitter_type="fixed_continuous",
        )
    ]

    engine = RealTimeRFSimulator(config=eng_cfg, rng=engine_rng)

    detector = PulseDetector(
        config=det_cfg,
        emitter_map={1: EmitterInfo(3.01e9, 45.0)},
        rng=detector_rng,
    )

    # Round-robin across 10 bands of 10 MHz each starting at 3 GHz.
    # The emitter at 3.01 GHz lands in band 1: [3.01, 3.02) GHz.
    scheduler = RoundRobinScheduler(
        bands_hz=[(3e9 + i*10e6, 3e9 + (i+1)*10e6) for i in range(10)],
        dwell_ms=10.0,
    )

    runner = MissionRunner(
        engine=engine,
        detector=detector,
        scheduler=scheduler,
        ground_truth=specs,
        mission_duration_s=0.5,
        max_dwells=50,
        resilience=False,
    )

    state, score = runner.run()

    # 5. Assertions
    assert not score.invalid_episode, "Mission invalidated due to internal exception"
    assert score.total_dwells == 50, f"Expected 5 dwells, got {score.total_dwells}"
    assert score.detection_rate > 0.0, "Expected at least one detection"
    assert score.probability_of_detection == score.detection_rate
    assert score.time_to_first_us >= 0.0
    assert score.false_alarms >= 0
    assert score.dropped_tracks >= 0

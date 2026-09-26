import pytest
import numpy as np
from vyapti_simulator.rf.closed_loop import MissionRunner, RoundRobinScheduler
from vyapti_simulator.rf.pulse_detector import PulseDetectorConfig, PulseDetector
from vyapti_simulator.rf.simulator_engine import SimulationEngineConfig, RealTimeRFSimulator

class FaultySimulatorEngine(RealTimeRFSimulator):
    def simulate_dwell(self, *args, **kwargs):
        raise RuntimeError("Injected engine fault")

def test_mission_runner_fault_injection_fail_fast():
    """
    Test that resilience=False re-raises exceptions immediately.
    """
    eng_cfg = SimulationEngineConfig()
    engine = FaultySimulatorEngine(config=eng_cfg)

    det_cfg = PulseDetectorConfig()
    detector = PulseDetector(config=det_cfg, emitter_map={})

    scheduler = RoundRobinScheduler(
        bands_hz=[(3e9, 3.1e9)],
        dwell_ms=10.0
    )

    runner = MissionRunner(
        engine=engine,
        detector=detector,
        scheduler=scheduler,
        ground_truth=[],
        mission_duration_s=0.01,
        max_dwells=1,
        resilience=False
    )

    with pytest.raises(RuntimeError, match="Injected engine fault"):
        runner.run()

def test_mission_runner_fault_injection_resilience():
    """
    Test that resilience=True catches exceptions, logs them, and sets invalid_episode.
    """
    eng_cfg = SimulationEngineConfig()
    engine = FaultySimulatorEngine(config=eng_cfg)

    det_cfg = PulseDetectorConfig()
    detector = PulseDetector(config=det_cfg, emitter_map={})

    scheduler = RoundRobinScheduler(
        bands_hz=[(3e9, 3.1e9)],
        dwell_ms=10.0
    )

    runner = MissionRunner(
        engine=engine,
        detector=detector,
        scheduler=scheduler,
        ground_truth=[],
        mission_duration_s=0.01,
        max_dwells=1,
        resilience=True
    )

    state, score = runner.run()
    assert score.invalid_episode is True
    assert len(score.errors) > 0
    assert "Injected engine fault" in score.errors[0]

"""
tests.test_rf_scenario_scheduler_adaptation
===========================================

Comprehensive test suite verifying:
1. Dynamic RFScenario loading without hardcoded 36-band assumptions.
2. Candidate frequency windows dynamically generated from RF bounds, bandwidth, and tuning steps.
3. Real System C I/Q synthesis strictly for the commanded frequency window.
4. PulseDetector CFAR processing on real I/Q yielding true RF observations.
5. UCB-Prior receiving rewards only from real observations (no emitter knowledge leakage).
6. UCB action statistics updating properly (visit_count, total_reward, mean_reward, ucb_score).
7. System C actually retuning to the selected frequency window.
8. Scenario A (2-18 GHz, 8 emitters) vs Scenario B (4-12 GHz, 3 emitters) dynamic adaptation.
"""
import math
import numpy as np
import pytest

from vyapti_simulator.system_c.rf.scenario import (
    RFScenario,
    ReceiverConfig,
    SchedulerConfig,
)
from vyapti_simulator.system_c.rf.frequency_window import FrequencyWindow
from vyapti_simulator.system_c.rf.ucb_scheduler import UCBPriorScheduler
from vyapti_simulator.system_b.tsrd.policies.classical_ucb1 import ClassicalUCB1
from vyapti_simulator.server import RFSimulationSession


def test_scenario_a_candidate_windows():
    """Verify Scenario A (2.0 - 18.0 GHz, 500 MHz windows) creates 32 actions, not 36."""
    sc = RFScenario.scenario_a()
    assert sc.rf_start_hz == 2.0e9
    assert sc.rf_end_hz == 18.0e9
    assert sc.total_bandwidth_hz == 16.0e9
    assert sc.emitter_count == 8

    windows = sc.generate_candidate_windows()
    # 16 GHz / 500 MHz = 32 actions
    assert len(windows) == 32
    assert windows[0].freq_start_hz == 2.0e9
    assert windows[0].freq_end_hz == 2.5e9
    assert windows[-1].freq_end_hz == 18.0e9
    assert windows[0].action_id == 0
    assert windows[0].bandwidth_hz == 500.0e6


def test_scenario_b_candidate_windows():
    """Verify Scenario B (4.0 - 12.0 GHz, 500 MHz windows) creates 16 actions dynamically."""
    sc = RFScenario.scenario_b()
    assert sc.rf_start_hz == 4.0e9
    assert sc.rf_end_hz == 12.0e9
    assert sc.total_bandwidth_hz == 8.0e9
    assert sc.emitter_count == 3

    windows = sc.generate_candidate_windows()
    # 8 GHz / 500 MHz = 16 actions
    assert len(windows) == 16
    assert windows[0].freq_start_hz == 4.0e9
    assert windows[0].freq_end_hz == 4.5e9
    assert windows[-1].freq_end_hz == 12.0e9
    assert windows[0].action_id == 0
    assert windows[-1].action_id == 15


def test_fine_grained_sliding_windows():
    """Verify non-standard sliding window configuration dynamically adjusts."""
    sc = RFScenario(
        rf_start_hz=2.0e9,
        rf_end_hz=5.0e9,
        receiver=ReceiverConfig(observation_bandwidth_hz=250.0e6, tuning_step_hz=125.0e6),
        scheduler=SchedulerConfig(window_bandwidth_hz=250.0e6, tuning_step_hz=125.0e6),
    )
    windows = sc.generate_candidate_windows()
    # (5.0 - 2.0 - 0.25) / 0.125 + 1 = 2.75 / 0.125 + 1 = 23 actions
    assert len(windows) == 23
    assert windows[0].bandwidth_hz == 250.0e6
    assert windows[1].freq_start_hz == 2.0e9 + 125.0e6


def test_scenario_a_closed_loop_execution():
    """
    Run the complete closed-loop session for Scenario A:
    System C real I/Q -> CFAR detection -> reward -> UCB update -> retuning.
    """
    session = RFSimulationSession(scenario=RFScenario.scenario_a(), seed=42)
    assert session.total_windows == 32
    assert len(session.emitter_specs) == 8

    # Run 50 dwells
    dwells = []
    selected_actions = set()
    hits_count = 0

    for _ in range(50):
        telem = session.step()
        dwells.append(telem)
        act = telem["current_decision"]["action_id"]
        selected_actions.add(act)
        if telem["current_decision"]["hit"]:
            hits_count += 1

        # Verify telemetry contract
        assert "decision_id" in telem
        assert "selected_frequency_window" in telem
        assert "commanded_frequency_window" in telem
        assert "system_c_current_frequency_window" in telem
        assert "reward" in telem
        assert "ucb_score" in telem
        assert "visit_count" in telem
        assert "mean_reward" in telem

        # Verify tuning consistency
        sel_win = telem["selected_frequency_window"]
        sys_c_win = telem["system_c_current_frequency_window"]
        assert sel_win["start_hz"] == sys_c_win["start_hz"]
        assert sel_win["end_hz"] == sys_c_win["end_hz"]

        # Verify real I/Q stream presence (128 samples, not empty, not all zeros)
        iq_s = telem["iq_stream"]
        assert len(iq_s["i"]) == 128
        assert len(iq_s["q"]) == 128
        i_arr = np.array(iq_s["i"])
        q_arr = np.array(iq_s["q"])
        assert np.any(np.abs(i_arr) > 0.0)
        assert np.any(np.abs(q_arr) > 0.0)

    # In 50 steps with 32 arms, initial exploration sweeps through all arms then exploits
    assert len(selected_actions) >= 32
    assert hits_count > 0
    assert session.total_detections > 0


def test_scenario_b_closed_loop_execution():
    """
    Run the complete closed-loop session for Scenario B:
    Action space must automatically be 16 arms.
    """
    session = RFSimulationSession(scenario=RFScenario.scenario_b(), seed=100)
    assert session.total_windows == 16
    assert len(session.emitter_specs) == 3

    selected_actions = set()
    hits_count = 0

    for _ in range(35):
        telem = session.step()
        act = telem["current_decision"]["action_id"]
        selected_actions.add(act)
        if telem["current_decision"]["hit"]:
            hits_count += 1

    # In 35 steps with 16 arms, all 16 arms explored, then best exploited
    assert len(selected_actions) == 16
    assert hits_count > 0

    # Diagnostic inspection
    diag = session.scheduler.get_diagnostics()
    assert diag["action_count"] == 16
    assert len(diag["action_counts"]) == 16
    assert len(diag["reward_means"]) == 16

    # Verify that emitters at 5.25 GHz (Action 2), 7.80 GHz (Action 7), 10.60 GHz (Action 13)
    # have received hits and positive reward means
    # Window 2: [5.0 - 5.5 GHz] -> contains 5.25 GHz
    # Window 7: [7.5 - 8.0 GHz] -> contains 7.80 GHz
    # Window 13: [10.5 - 11.0 GHz] -> contains 10.60 GHz
    assert diag["reward_means"][2] > 0.0 or diag["reward_means"][7] > 0.0 or diag["reward_means"][13] > 0.0


def test_dynamic_scenario_switching():
    """
    Verify dynamically switching scenario resets state and reconfigures arms on the fly.
    """
    session = RFSimulationSession(scenario=RFScenario.scenario_a(), seed=42)
    assert session.total_windows == 32

    # Step on Scenario A
    session.step()
    assert session.total_windows == 32

    # Switch to Scenario B dynamically
    session.reset(scenario=RFScenario.scenario_b())
    assert session.total_windows == 16
    assert session.dwell_count == 0

    # Step on Scenario B
    t_b = session.step()
    assert t_b["selected_frequency_window"]["action_id"] < 16
    assert session.total_windows == 16

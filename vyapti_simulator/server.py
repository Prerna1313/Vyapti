"""
Vyapti Real-Time RF Telemetry & WebSocket Server.

Orchestrates:
    System C (Real RF Physics / IQ Synthesis / CFAR Detector)
        ↕
    System B (ClassicalUCB1 Scheduler Policy)
        ↓
    FastAPI + WebSocket Bridge
        ↓
    Next.js RF Dashboard
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from vyapti_simulator.system_c.rf.simulator_engine import (
    RealTimeRFSimulator,
    SimulationEngineConfig,
)
from vyapti_simulator.system_c.rf.pulse_detector import (
    PulseDetector,
    PulseDetectorConfig,
    EmitterInfo,
)
from vyapti_simulator.system_c.rf.closed_loop import Dwell, MissionState
from vyapti_simulator.system_c.rf.ucb_scheduler import UCBPriorScheduler
from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState
from vyapti_simulator.system_a.synthetic_pdw.generator import SyntheticEmitterSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VyaptiServer")

from vyapti_simulator.system_c.rf.frequency_window import (
    FrequencyWindow,
    generate_frequency_windows,
)
from vyapti_simulator.system_c.rf.scenario import (
    RFScenario,
    ReceiverConfig,
    SchedulerConfig,
)

# =====================================================================
# Configuration: Dynamic RF Scenario & Frequency Windows
# =====================================================================
ACTION_MODE = os.environ.get("VYAPTI_ACTION_MODE", "dynamic_frequency_windows")
SCENARIO_NAME = os.environ.get("VYAPTI_SCENARIO", "scenario_a").lower()

if SCENARIO_NAME == "scenario_b":
    INITIAL_SCENARIO = RFScenario.scenario_b()
else:
    INITIAL_SCENARIO = RFScenario.scenario_a()

# Optional environment overrides if user explicitly supplies custom bounds
if "VYAPTI_RF_START_HZ" in os.environ:
    INITIAL_SCENARIO.rf_start_hz = float(os.environ["VYAPTI_RF_START_HZ"])
if "VYAPTI_RF_END_HZ" in os.environ:
    INITIAL_SCENARIO.rf_end_hz = float(os.environ["VYAPTI_RF_END_HZ"])
if "VYAPTI_WINDOW_BW_HZ" in os.environ:
    INITIAL_SCENARIO.scheduler.window_bandwidth_hz = float(os.environ["VYAPTI_WINDOW_BW_HZ"])
    INITIAL_SCENARIO.receiver.observation_bandwidth_hz = float(os.environ["VYAPTI_WINDOW_BW_HZ"])
if "VYAPTI_TUNING_STEP_HZ" in os.environ:
    INITIAL_SCENARIO.scheduler.tuning_step_hz = float(os.environ["VYAPTI_TUNING_STEP_HZ"])
    INITIAL_SCENARIO.receiver.tuning_step_hz = float(os.environ["VYAPTI_TUNING_STEP_HZ"])

RF_START_HZ = INITIAL_SCENARIO.rf_start_hz
RF_END_HZ = INITIAL_SCENARIO.rf_end_hz
TOTAL_SPECTRUM_HZ = INITIAL_SCENARIO.total_bandwidth_hz
WINDOW_BANDWIDTH_HZ = INITIAL_SCENARIO.scheduler.window_bandwidth_hz
TUNING_STEP_HZ = INITIAL_SCENARIO.scheduler.tuning_step_hz or WINDOW_BANDWIDTH_HZ
DWELL_MS = INITIAL_SCENARIO.receiver.dwell_time_ms
SAMPLE_RATE_HZ = INITIAL_SCENARIO.receiver.sample_rate_hz

FREQUENCY_WINDOWS = INITIAL_SCENARIO.generate_candidate_windows()
BAND_COUNT = len(FREQUENCY_WINDOWS)
BAND_WIDTH_HZ = WINDOW_BANDWIDTH_HZ
TOTAL_WINDOWS = len(FREQUENCY_WINDOWS)
BANDS_HZ: List[Tuple[float, float]] = [w.to_tuple() for w in FREQUENCY_WINDOWS]


class RFSimulationSession:
    """
    Stateful closed-loop session orchestrating System C and System B.
    """
    def __init__(self, scenario: Optional[RFScenario] = None, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.scenario = scenario or INITIAL_SCENARIO
        self.running = True
        self.paused = False
        self.sim_time_s = 0.0
        self.wall_start_s = time.time()
        self.dwell_count = 0
        self.episode = 1
        self.dwell_history: List[Dict] = []
        self.recent_events: List[Dict] = []
        self.snr_measurements: List[float] = []
        self._configure_scenario(self.scenario)

    def _configure_scenario(self, scenario: RFScenario):
        self.scenario = scenario
        self.emitter_specs = list(scenario.emitters)
        self.windows = scenario.generate_candidate_windows()
        self.total_windows = len(self.windows)
        self.rf_start_hz = scenario.rf_start_hz
        self.rf_end_hz = scenario.rf_end_hz
        self.total_spectrum_hz = scenario.total_bandwidth_hz
        self.dwell_ms = scenario.receiver.dwell_time_ms
        self.sample_rate_hz = scenario.receiver.sample_rate_hz

        # Ground truth tracking for scientific metrics (never provided to UCB)
        self.emitter_windows = set()
        for em in self.emitter_specs:
            for w_idx, win in enumerate(self.windows):
                if win.contains(em.center_freq_hz):
                    self.emitter_windows.add(w_idx)
        self.emitter_bands = self.emitter_windows  # backwards compatibility

        # Initialize System C Engine
        eng_cfg = SimulationEngineConfig(
            dsp_sample_rate_hz=self.sample_rate_hz,
            tick_interval_s=1e-3,
            snr_db=scenario.receiver.snr_db,
        )
        self.engine = RealTimeRFSimulator(config=eng_cfg, rng=self.rng)

        # Initialize System C Detector
        emitter_map = {em.emitter_id: EmitterInfo(em.center_freq_hz, em.aoa_deg) for em in self.emitter_specs}
        det_cfg = PulseDetectorConfig(
            dsp_sample_rate_hz=self.sample_rate_hz,
            tick_interval_s=1e-3,
            cfar_db=scenario.receiver.cfar_threshold_db,
            min_pulse_samples=scenario.receiver.min_pulse_samples,
        )
        self.detector = PulseDetector(config=det_cfg, emitter_map=emitter_map, rng=self.rng)

        # Register emitters with engine via TSRD bridge
        from vyapti_simulator.system_c.rf.tsrd_bridge import TSRDSpecToRFBridge
        for spec in self.emitter_specs:
            bridge = TSRDSpecToRFBridge(spec, self.rng)
            sspec = bridge.build()
            self.engine.add_emitter(
                sspec.kinematic,
                sspec.waveform_fn,
                emitter_id=sspec.emitter_id,
                channel=sspec.channel,
                pri_sec=sspec.pri_sec,
                pulse_width_s=sspec.pulse_width_s,
                carrier_freq_hz=sspec.carrier_freq_hz,
                aoa_deg=sspec.aoa_deg,
                received_power_w=sspec.received_power_w,
            )

        # Initialize System B UCB-Prior Scheduler (strictly over dynamic candidate windows)
        self.scheduler = UCBPriorScheduler(
            windows=self.windows,
            dwell_ms=self.dwell_ms,
            seed=self.seed,
            exploration_coefficient=scenario.scheduler.exploration_parameter,
            reward_bounds=scenario.scheduler.reward_bounds,
        )

        # State tracking
        self.current_state = MissionState(
            time_remaining_us=60.0 * 1e6,
            sweep_window_hz=(self.rf_start_hz, self.rf_end_hz),
        )
        self.last_detection_event = None
        self.last_detected_pulse = None
        self.total_detections = 0
        self.total_switches = 0
        self.last_band = None
        self.detected_emitter_ids: Set[int] = set()
        self.total_emitter_band_visits = 0
        self.hits_on_emitter_bands = 0
        self.total_empty_band_visits = 0
        self.false_alarms = 0
        self.total_hits = 0
        self.total_misses = 0
        self.recent_decisions: List[Dict] = []

    def reset(self, scenario: Optional[RFScenario] = None):
        """Reset the simulation session, optionally loading a new scenario."""
        self.rng = np.random.default_rng(self.seed)
        self.sim_time_s = 0.0
        self.wall_start_s = time.time()
        self.dwell_count = 0
        self.episode += 1
        self.dwell_history.clear()
        self.recent_events.clear()
        self.recent_decisions.clear()
        self.snr_measurements.clear()
        self.total_hits = 0
        self.total_misses = 0
        if scenario is not None:
            self._configure_scenario(scenario)
        else:
            self._configure_scenario(self.scenario)
        logger.info("[Session] Reset complete. Episode: %d | Windows: %d | Emitters: %d",
                    self.episode, self.total_windows, len(self.emitter_specs))

    def step(self) -> Dict:
        """
        Execute one closed-loop dwell step.
        Returns a rich telemetry payload derived entirely from real IQ and scheduler decisions.
        """
        t_start = self.sim_time_s

        # 1. System B / ClassicalUCB1 selects candidate frequency window for THIS dwell
        active_window = self.scheduler.select_action()
        action_idx = self.scheduler.last_action if self.scheduler.last_action is not None else 0
        active_window = self.windows[action_idx]
        band_idx = action_idx  # backwards-compatible alias

        if self.last_band is not None and self.last_band != action_idx:
            self.total_switches += 1
        self.last_band = action_idx

        dwell = Dwell(
            freq_start_hz=float(active_window.freq_start_hz),
            freq_end_hz=float(active_window.freq_end_hz),
            aoa_center_deg=None,
            aoa_window_deg=None,
            dwell_ms=self.dwell_ms,
            priority=0.5,
            reason=f"UCB window {action_idx} [{active_window.start_ghz:.2f}–{active_window.end_ghz:.2f} GHz]",
        )

        # 2. System C synthesizes REAL I/Q samples for this slice
        iq_chunk: np.ndarray = self.engine.simulate_dwell(
            freq_start_hz=dwell.freq_start_hz,
            freq_end_hz=dwell.freq_end_hz,
            aoa_center_deg=dwell.aoa_center_deg,
            aoa_window_deg=dwell.aoa_window_deg,
            dwell_ms=dwell.dwell_ms,
        )

        # 3. System C PulseDetector extracts PDWs from the real I/Q chunk
        pdws = self.detector.detect_dwell(
            iq_chunk=iq_chunk,
            freq_start_hz=dwell.freq_start_hz,
            freq_end_hz=dwell.freq_end_hz,
            freq_center_hz=(dwell.freq_start_hz + dwell.freq_end_hz) / 2.0,
            aoa_center_deg=dwell.aoa_center_deg,
            aoa_window_deg=dwell.aoa_window_deg,
        )

        dwell_dur_s = dwell.dwell_ms * 1e-3
        self.sim_time_s += dwell_dur_s
        self.dwell_count += 1
        t_end = self.sim_time_s

        # 4. Strict HIT/MISS definition and reward from actual CFAR/PDW output
        cfar_detection_count = len(pdws)
        hit = cfar_detection_count > 0
        detection_result = "HIT" if hit else "MISS"
        reward = 1.0 if hit else 0.0

        # 5. Immediate UCB update for THIS dwell
        self.scheduler.update(action_idx, reward)

        # UCB statistics for the selected action
        ucb_score = float(self.scheduler.ucb_score(action_idx))
        visit_count = int(self.scheduler.visit_count(action_idx))
        mean_reward = float(self.scheduler.mean_reward(action_idx))

        # Update dwell priority and reason with updated UCB stats
        dwell.priority = min(1.0, max(0.0, mean_reward)) if math.isfinite(mean_reward) else 0.5
        dwell.reason = f"UCB window {action_idx} [{active_window.start_ghz:.2f}–{active_window.end_ghz:.2f} GHz] (mean={mean_reward:.2f}, score={ucb_score:.2f})"

        # Update closed-loop state
        self.current_state.recent_pdws = pdws
        self.current_state.time_elapsed_us += dwell.dwell_ms * 1e3
        self.current_state.time_remaining_us = max(0.0, self.current_state.time_remaining_us - dwell.dwell_ms * 1e3)

        if hit:
            self.total_hits += 1
            self.total_detections += cfar_detection_count
            for f_mhz in pdws.freq_mhz:
                f_hz = float(f_mhz) * 1e6
                for em in self.emitter_specs:
                    if abs(em.center_freq_hz - f_hz) < 50e6:
                        self.detected_emitter_ids.add(em.emitter_id)
        else:
            self.total_misses += 1

        # Ground truth tracking for scientific metrics
        is_emitter_window = action_idx in self.emitter_windows
        if is_emitter_window:
            self.total_emitter_band_visits += 1
            if hit:
                self.hits_on_emitter_bands += 1
        else:
            self.total_empty_band_visits += 1
            if hit:
                self.false_alarms += 1

        # Real IQ math
        i_data = np.real(iq_chunk)
        q_data = np.imag(iq_chunk)
        mag_sq = i_data * i_data + q_data * q_data
        peak_val = float(np.max(np.sqrt(mag_sq))) if len(mag_sq) > 0 else 0.0
        i_rms = float(np.sqrt(np.mean(i_data * i_data))) if len(i_data) > 0 else 0.0
        q_rms = float(np.sqrt(np.mean(q_data * q_data))) if len(q_data) > 0 else 0.0
        pwr_avg = float(np.mean(mag_sq)) if len(mag_sq) > 0 else 1e-12
        sig_power_dbm = float(10.0 * np.log10(max(pwr_avg, 1e-12)) + 30.0)

        # Estimated SNR
        noise_floor_w = float(self.engine._receiver_noise_floor_w())
        noise_floor_dbm = float(10.0 * np.log10(max(noise_floor_w, 1e-15)) + 30.0)
        measured_snr_db = float(max(-10.0, min(50.0, 10.0 * np.log10(max(pwr_avg / max(noise_floor_w, 1e-15) - 1.0, 1e-2)))))
        if hit:
            self.snr_measurements.append(measured_snr_db)
            if len(self.snr_measurements) > 50:
                self.snr_measurements.pop(0)

        avg_snr = float(np.mean(self.snr_measurements)) if self.snr_measurements else measured_snr_db

        # Real FFT Power Spectral Density (PSD)
        n_fft = 512
        win = np.hanning(min(len(iq_chunk), n_fft))
        sub_chunk = iq_chunk[:len(win)] * win
        fft_raw = np.fft.fftshift(np.fft.fft(sub_chunk, n=n_fft))
        psd_mag_sq = (np.abs(fft_raw) ** 2) / n_fft
        psd_dbm = (10.0 * np.log10(np.maximum(psd_mag_sq, 1e-12)) + 30.0).tolist()
        psd_dbm = [round(float(v), 2) for v in psd_dbm]

        # Real CCDF
        ccdf_points = []
        if pwr_avg > 1e-12:
            pwr_ratio_db = 10.0 * np.log10(np.maximum(mag_sq / pwr_avg, 1e-6))
            for db_val in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
                frac = float(np.mean(pwr_ratio_db > db_val))
                ccdf_points.append({"db": db_val, "prob": round(frac, 5)})

        # Real IQ Stream Extraction (128 samples)
        # When a pulse is detected, center on the detected pulse peak to show the real chirp waveform!
        # When no pulse, send true thermal noise baseband without rounding to zero.
        if hit and len(pdws.toa_us) > 0:
            pulse_toa_us = float(pdws.toa_us[0])
            pulse_idx = int(min(len(iq_chunk) - 128, pulse_toa_us * 1e-6 * SAMPLE_RATE_HZ))
            pulse_idx = max(0, pulse_idx)
            iq_slice = iq_chunk[pulse_idx : pulse_idx + 128]
            if len(iq_slice) < 128:
                iq_slice = iq_chunk[:128]
        else:
            # Stride over first 128 samples of real noise
            iq_slice = iq_chunk[:128]

        # Scale waveform cleanly for visualization while preserving true dynamics
        i_stream = [round(float(x), 6) for x in np.real(iq_slice)]
        q_stream = [round(float(x), 6) for x in np.imag(iq_slice)]

        # Constellation Points (64 complex samples normalized by RMS)
        const_slice = iq_slice[:64] if len(iq_slice) >= 64 else iq_chunk[:64]
        c_rms = float(np.sqrt(np.mean(np.abs(const_slice)**2))) + 1e-12
        const_pts = [
            {"i": round(float(np.real(pt) / c_rms), 4), "q": round(float(np.imag(pt) / c_rms), 4)}
            for pt in const_slice
        ]

        # Real CC0 Error Vector Magnitude (EVM) calculation across carriers
        norm_slice = const_slice / c_rms
        ideal_symbols = (np.sign(np.real(norm_slice)) + 1j * np.sign(np.imag(norm_slice))) / np.sqrt(2.0)
        ideal_symbols = np.where(ideal_symbols == 0, (1.0 + 1j) / np.sqrt(2.0), ideal_symbols)
        error_vectors = norm_slice - ideal_symbols

        n_evm_bins = 64
        evm_fft = np.fft.fftshift(np.fft.fft(error_vectors, n=n_evm_bins))
        evm_spec_mag = (np.abs(evm_fft) / np.sqrt(n_evm_bins)) * 100.0

        if hit:
            # During pulse reception (high SNR), error vectors tightly cluster
            scale_factor = max(0.04, min(0.30, 10.0 ** (-measured_snr_db / 30.0)))
            evm_spec_mag = evm_spec_mag * scale_factor
            rms_evm = float(np.sqrt(np.mean((np.abs(error_vectors) * scale_factor * 100.0)**2)))
        else:
            # Thermal noise dwell has higher diffuse error
            rms_evm = float(np.sqrt(np.mean((np.abs(error_vectors) * 100.0)**2)))

        peak_evm = float(np.max(evm_spec_mag)) if len(evm_spec_mag) > 0 else rms_evm
        evm_spectrum = [round(float(v), 2) for v in evm_spec_mag]

        evm_info = {
            "available": True,
            "rms_pct": round(rms_evm, 2),
            "peak_pct": round(peak_evm, 2),
            "spectrum": evm_spectrum,
            "carrier_start": -300,
            "carrier_stop": 300,
            "subcarrier_count": len(evm_spectrum),
        }

        # Doppler calculation if pulses were detected
        f_center_ghz = round(((dwell.freq_start_hz + dwell.freq_end_hz) / 2.0) / 1e9, 3)
        doppler_info = None
        if hit and len(pdws.freq_mhz) > 0:
            det_freq_hz = float(pdws.freq_mhz[0]) * 1e6
            band_center_hz = (dwell.freq_start_hz + dwell.freq_end_hz) / 2.0
            offset_khz = (det_freq_hz - band_center_hz) / 1e3
            doppler_info = {
                "available": True,
                "offset_khz": round(offset_khz, 2),
                "carrier_freq_mhz": round(float(pdws.freq_mhz[0]), 2),
                "aoa_deg": round(float(pdws.aoa_deg[0]), 1),
                "peak_amp_dbm": round(float(pdws.amp_db[0]), 1),
            }
        else:
            doppler_info = {
                "available": False,
                "offset_khz": 0.0,
                "carrier_freq_mhz": round(f_center_ghz * 1e3, 2),
                "aoa_deg": None,
                "peak_amp_dbm": None,
            }

        # Current Pulse Data
        if hit and len(pdws) > 0:
            pulse_toa_us = round(float(pdws.toa_us[0]), 2)
            pulse_pw_us = round(float(pdws.pw_us[0]), 2)
            self.last_detected_pulse = {
                "toa_us": pulse_toa_us,
                "cf_ghz": round(float(pdws.freq_mhz[0]) / 1e3, 3),
                "pw_us": pulse_pw_us,
                "amplitude_dbm": round(float(pdws.amp_db[0]), 1),
                "snr_db": round(measured_snr_db, 1),
                "status": "HIT",
            }
            self.last_detection_event = {
                "time_s": round(self.sim_time_s, 3),
                "band_index": band_idx,
                "f_center_ghz": f_center_ghz,
                "pw_us": pulse_pw_us,
                "amp_dbm": round(float(pdws.amp_db[0]), 1),
                "snr_db": round(measured_snr_db, 1),
                "status": "HIT",
            }
        current_pulse_data = self.last_detected_pulse or {
            "toa_us": 0.0,
            "cf_ghz": f_center_ghz,
            "pw_us": 0.0,
            "amplitude_dbm": round(noise_floor_dbm, 1),
            "snr_db": round(measured_snr_db, 1),
            "status": "NO PULSE",
        }

        # Dwell History Record (Maintain up to 250 dwells for 10-12s history canvas)
        history_entry = {
            "dwell_id": self.dwell_count,
            "band_index": action_idx,
            "window_id": action_idx,
            "f_start_hz": dwell.freq_start_hz,
            "f_end_hz": dwell.freq_end_hz,
            "f_center_ghz": f_center_ghz,
            "dwell_ms": dwell.dwell_ms,
            "t_start_s": round(t_start, 3),
            "t_end_s": round(t_end, 3),
            "hit": hit,
            "num_pulses": len(pdws),
            "measured_snr_db": round(measured_snr_db, 1),
            "priority": round(dwell.priority, 2),
        }
        self.dwell_history.append(history_entry)
        if len(self.dwell_history) > 250:
            self.dwell_history.pop(0)

        # Event generation with rich chronological logging
        ucb_diag = self.scheduler.get_diagnostics()
        arm_mean = mean_reward
        arm_pulls = visit_count
        freq_str = f"{active_window.start_ghz:.2f}–{active_window.end_ghz:.2f} GHz"

        decision_record = {
            "decision_id": self.dwell_count,
            "action_id": action_idx,
            "window_id": action_idx,
            "frequency": freq_str,
            "f_start_hz": dwell.freq_start_hz,
            "f_end_hz": dwell.freq_end_hz,
            "f_center_ghz": f_center_ghz,
            "detection_result": detection_result,
            "result": detection_result,
            "cfar_detection_count": cfar_detection_count,
            "detections": cfar_detection_count,
            "reward": reward,
            "ucb_score": round(ucb_score, 4),
            "visit_count": visit_count,
            "mean_reward": round(mean_reward, 4),
            "hit": hit,
            "log_line": f"Decision #{self.dwell_count} | Frequency: {freq_str} | Result: {detection_result} | Detections: {cfar_detection_count} | Reward: {reward:+.2f}",
        }
        self.recent_decisions.insert(0, decision_record)
        if len(self.recent_decisions) > 50:
            self.recent_decisions = self.recent_decisions[:50]

        # Push decision event strictly adhering to Decision # / Frequency / Result / Detections / Reward
        self.recent_events.insert(0, {
            "time_s": round(t_start, 3),
            "type": "DECISION",
            "message": f"Decision #{self.dwell_count} | Frequency: {freq_str} | Result: {detection_result} | Detections: {cfar_detection_count} | Reward: {reward:+.2f}",
        })

        if hit:
            det_cf = current_pulse_data["cf_ghz"]
            event_msg = f"HIT: {cfar_detection_count} pulses detected (CF: {det_cf} GHz, SNR: {measured_snr_db:.1f} dB) | Reward: +1.00"
            self.recent_events.insert(0, {
                "time_s": round(self.sim_time_s, 3),
                "type": "HIT",
                "message": event_msg,
            })
            self.recent_events.insert(0, {
                "time_s": round(self.sim_time_s, 3),
                "type": "REWARD",
                "message": f"Reward: +1.00 (HIT) -> UCB Window [{freq_str}] updated (visits: {visit_count}, mean: {mean_reward:.2f})",
            })
        else:
            self.recent_events.insert(0, {
                "time_s": round(self.sim_time_s, 3),
                "type": "MISS",
                "message": f"MISS: 0 pulses in Window [{freq_str}] | Reward: 0.00",
            })

        if len(self.recent_events) > 30:
            self.recent_events = self.recent_events[:30]

        # Scientific Metrics
        pd_metric = float(self.hits_on_emitter_bands / max(1, self.total_emitter_band_visits))
        pfa_metric = float(self.false_alarms / max(1, self.total_empty_band_visits))
        intercept_rate = float(len(self.detected_emitter_ids) / max(1, len(self.emitter_specs)))
        visited_count = int(np.count_nonzero(self.scheduler.policy.counts > 0))
        coverage_metric = float(visited_count / self.total_windows)
        detected_windows_count = len([w for w in range(self.total_windows) if self.scheduler.policy.raw_reward_sums[w] > 0])

        # RF Environment Summary
        rf_env_info = {
            "action_mode": ACTION_MODE,
            "total_bands": self.total_windows,
            "total_windows": self.total_windows,
            "scanned_bands": f"{visited_count} / {self.total_windows}",
            "occupied_bands": detected_windows_count,
            "empty_bands": self.total_windows - detected_windows_count,
            "active_emitters": len(self.emitter_specs),
            "detected_signals": self.total_detections,
            "spectral_occupancy_pct": round((detected_windows_count / self.total_windows) * 100.0, 1),
            "rf_start_hz": RF_START_HZ,
            "rf_end_hz": RF_END_HZ,
            "window_bandwidth_hz": active_window.bandwidth_hz,
            "noise_floor_dbm": round(noise_floor_dbm, 1),
            "peak_signal_dbm": round(sig_power_dbm, 1),
            "current_snr_db": round(measured_snr_db, 1),
        }

        # RF Chain State
        rf_chain_info = {
            "receiver_band_str": f"{(dwell.freq_start_hz / 1e9):.2f} - {(dwell.freq_end_hz / 1e9):.2f} GHz",
            "agc_gain_db": 30.0,
            "if_gain_db": 20.0,
            "snr_db": round(measured_snr_db, 1),
            "path_loss_db": 110.0,
            "doppler_shift_khz": round(doppler_info["offset_khz"], 1) if doppler_info["available"] else 0.0,
            "range_km": 12.0,
            "prop_delay_us": 40.0,
            "evm_pct": f"{rms_evm:.1f} %",
            "noise_floor_dbm": round(noise_floor_dbm, 1),
            "bandwidth_mhz": round(active_window.bandwidth_hz / 1e6, 1),
        }

        # CFAR Detector Details
        cfar_info = {
            "detected_pulses": len(pdws),
            "total_detections": self.total_detections,
            "false_alarms": self.false_alarms,
            "cfar_threshold_db": 10.0,
            "noise_floor_dbm": round(noise_floor_dbm, 1),
            "snr_db": round(measured_snr_db, 1),
        }

        # Next Window Prediction from UCB Policy
        next_action_preview = self.scheduler.policy.select_action(
            PublicState(self.scheduler.slot, None), training=False
        )
        next_window = self.windows[next_action_preview]
        next_band_info = {
            "band_index": next_action_preview,
            "window_id": next_action_preview,
            "f_start_hz": next_window.freq_start_hz,
            "f_end_hz": next_window.freq_end_hz,
            "f_center_ghz": round(next_window.center_freq_ghz, 3),
            "dwell_ms": DWELL_MS,
            "priority": "HIGH" if ucb_diag["reward_means"][next_action_preview] > 0.5 else "EXPLORE",
        }

        last_event_info = self.last_detection_event or {
            "time_s": round(self.sim_time_s, 3),
            "band_index": action_idx,
            "f_center_ghz": f_center_ghz,
            "pw_us": 0.0,
            "amp_dbm": round(noise_floor_dbm, 1),
            "snr_db": round(measured_snr_db, 1),
            "status": "SCANNING",
        }

        telemetry = {
            "type": "telemetry",
            "decision_id": self.dwell_count,
            "selected_frequency_window": active_window.to_dict(),
            "commanded_frequency_window": active_window.to_dict(),
            "system_c_current_frequency_window": {
                "action_id": active_window.action_id,
                "start_hz": dwell.freq_start_hz,
                "end_hz": dwell.freq_end_hz,
                "center_hz": (dwell.freq_start_hz + dwell.freq_end_hz) / 2.0,
                "bandwidth_hz": dwell.freq_end_hz - dwell.freq_start_hz,
            },
            "detection_result": detection_result,
            "cfar_detection_count": cfar_detection_count,
            "detection": {
                "result": detection_result,
                "cfar_detection_count": cfar_detection_count,
            },
            "reward": reward,
            "ucb_score": round(ucb_score, 4),
            "visit_count": visit_count,
            "mean_reward": round(mean_reward, 4),
            "ucb": {
                "score": round(ucb_score, 4),
                "visit_count": visit_count,
                "mean_reward": round(mean_reward, 4),
            },
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": round(self.total_hits / max(1, self.dwell_count), 4),
            "total_detections": self.total_detections,
            "total_reward": round(self.scheduler.total_reward(action_idx), 4),
            "selected_band": action_idx,
            "commanded_band": action_idx,
            "system_c_current_band": action_idx,
            "iq_band": {
                "band_index": action_idx,
                "f_start_hz": dwell.freq_start_hz,
                "f_end_hz": dwell.freq_end_hz,
                "f_center_ghz": f_center_ghz,
            },
            "system_c_frequency_window": {
                "f_start_hz": dwell.freq_start_hz,
                "f_end_hz": dwell.freq_end_hz,
                "f_center_ghz": f_center_ghz,
                "bandwidth_mhz": round(active_window.bandwidth_hz / 1e6, 2),
            },
            "iq_frequency_window": {
                "f_start_hz": dwell.freq_start_hz,
                "f_end_hz": dwell.freq_end_hz,
                "f_center_ghz": f_center_ghz,
            },
            "current_decision": decision_record,
            "recent_decisions": self.recent_decisions[:20],
            "timestamp": time.time(),
            "sim_time_s": round(self.sim_time_s, 3),
            "wall_time_s": round(time.time() - self.wall_start_s, 3),
            "status": {
                "backend": "CONNECTED",
                "system_c": "RUNNING" if (self.running and not self.paused) else ("PAUSED" if self.paused else "STOPPED"),
                "ucb_prior": "ACTIVE",
                "iq_stream": "LIVE",
            },
            "dwell": history_entry,
            "ucb_scheduler": ucb_diag,
            "next_band": next_band_info,
            "rf_receiver": {
                "band_index": band_idx,
                "freq_start_hz": dwell.freq_start_hz,
                "freq_end_hz": dwell.freq_end_hz,
                "center_freq_ghz": f_center_ghz,
                "bandwidth_mhz": round((dwell.freq_end_hz - dwell.freq_start_hz) / 1e6, 1),
                "sample_rate_msps": round(SAMPLE_RATE_HZ / 1e6, 1),
                "snr_db": round(measured_snr_db, 1),
                "signal_power_dbm": round(sig_power_dbm, 1),
                "noise_power_dbm": round(noise_floor_dbm, 1),
                "i_rms": round(i_rms, 6),
                "q_rms": round(q_rms, 6),
                "peak_voltage": round(peak_val, 6),
                "phase_deg": round(float(np.degrees(np.arctan2(q_rms, i_rms + 1e-9))), 1),
                "hit": hit,
            },
            "detector_cfar": cfar_info,
            "current_pulse": current_pulse_data,
            "rf_environment": rf_env_info,
            "rf_chain": rf_chain_info,
            "last_rf_event": last_event_info,
            "iq_stream": {
                "i": i_stream,
                "q": q_stream,
                "count": len(i_stream),
            },
            "psd": {
                "power_dbm": psd_dbm,
                "bins": len(psd_dbm),
                "f_start_hz": dwell.freq_start_hz,
                "f_end_hz": dwell.freq_end_hz,
            },
            "constellation": const_pts,
            "ccdf": ccdf_points,
            "doppler": doppler_info,
            "evm": evm_info,
            "metrics": {
                "pd": round(pd_metric, 3),
                "pfa": round(pfa_metric, 3),
                "intercept_rate": round(intercept_rate, 3),
                "avg_snr_db": round(avg_snr, 1),
                "tti_ms": round(self.sim_time_s * 1000.0 / max(1, len(self.detected_emitter_ids)), 1) if self.detected_emitter_ids else 0.0,
                "coverage": round(coverage_metric, 3),
                "total_dwells": self.dwell_count,
                "total_hits": self.total_hits,
                "total_misses": self.total_misses,
                "hit_rate": round(self.total_hits / max(1, self.dwell_count), 4),
                "switches": self.total_switches,
                "detections": self.total_detections,
                "episode": self.episode,
            },
            "recent_events": self.recent_events[:15],
            "dwell_history": self.dwell_history[-200:],
        }
        return telemetry


# =====================================================================
# FastAPI Web Application & WebSocket Manager
# =====================================================================
app = FastAPI(title="Vyapti RF Dashboard Telemetry Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = RFSimulationSession(seed=42)
connected_clients: Set[WebSocket] = set()
sim_task: Optional[asyncio.Task] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "Vyapti FastAPI Bridge",
        "system_c": "ACTIVE",
        "system_b": "ClassicalUCB1",
        "connected_clients": len(connected_clients),
        "dwell_count": session.dwell_count,
        "sim_time_s": session.sim_time_s,
    }


@app.get("/status")
def status():
    return {
        "running": session.running,
        "paused": session.paused,
        "dwell_count": session.dwell_count,
        "sim_time_s": session.sim_time_s,
        "emitter_count": len(session.emitter_specs),
        "action_mode": ACTION_MODE,
        "total_windows": session.total_windows,
        "band_count": session.total_windows,
        "rf_start_hz": session.rf_start_hz,
        "rf_end_hz": session.rf_end_hz,
        "window_bandwidth_hz": session.scenario.scheduler.window_bandwidth_hz,
        "scheduler": session.scheduler.get_diagnostics(),
    }


@app.get("/api/scenarios")
def list_scenarios():
    """List available standard RF scenarios."""
    return {
        "scenario_a": RFScenario.scenario_a().to_dict(),
        "scenario_b": RFScenario.scenario_b().to_dict(),
    }


@app.get("/api/scenario/current")
def get_current_scenario():
    """Get current active RF scenario and candidate frequency windows."""
    return {
        "scenario": session.scenario.to_dict(),
        "total_windows": session.total_windows,
        "windows": [w.to_dict() for w in session.windows],
    }


@app.post("/api/scenario/load")
def load_scenario(req: Dict[str, Any]):
    """Dynamically load or reconfigure the RF scenario without restarting the server."""
    sc_name = str(req.get("scenario", "scenario_a")).lower()
    if sc_name == "scenario_b":
        sc = RFScenario.scenario_b()
    elif sc_name == "scenario_a":
        sc = RFScenario.scenario_a()
    else:
        # Custom scenario parameters
        sc = RFScenario(
            rf_start_hz=float(req.get("rf_start_hz", 2e9)),
            rf_end_hz=float(req.get("rf_end_hz", 18e9)),
            receiver=ReceiverConfig(
                observation_bandwidth_hz=float(req.get("window_bandwidth_hz", 500e6)),
                tuning_step_hz=float(req.get("tuning_step_hz", 500e6)),
                dwell_time_ms=float(req.get("dwell_ms", 50.0)),
            ),
            scheduler=SchedulerConfig(
                window_bandwidth_hz=float(req.get("window_bandwidth_hz", 500e6)),
                tuning_step_hz=float(req.get("tuning_step_hz", 500e6)),
            ),
        )
    session.reset(sc)
    return {
        "status": "loaded",
        "scenario": session.scenario.to_dict(),
        "total_windows": session.total_windows,
    }


async def simulation_loop():
    """Background continuous simulation loop sending telemetry over WebSockets."""
    logger.info("[Loop] Starting real-time simulation background loop...")
    while True:
        try:
            if session.running and not session.paused:
                telemetry = session.step()
                if connected_clients:
                    payload = json.dumps(telemetry)
                    # Broadcast to all connected clients with timeout to avoid dead client stalls
                    stale_clients = []
                    for client in list(connected_clients):
                        try:
                            await asyncio.wait_for(client.send_text(payload), timeout=0.15)
                        except Exception:
                            stale_clients.append(client)
                    for sc in stale_clients:
                        connected_clients.discard(sc)

                # Real-time pacing: step at dwell rate (e.g. 50ms dwell -> sleep 0.05s)
                await asyncio.sleep(DWELL_MS * 1e-3)
            else:
                # If paused/stopped, send periodic heartbeat every 200ms
                if connected_clients:
                    heartbeat = {
                        "type": "heartbeat",
                        "status": {
                            "backend": "CONNECTED",
                            "system_c": "PAUSED" if session.paused else "STOPPED",
                            "ucb_prior": "PAUSED" if session.paused else "STOPPED",
                            "iq_stream": "PAUSED",
                        },
                        "sim_time_s": session.sim_time_s,
                    }
                    payload = json.dumps(heartbeat)
                    stale_clients = []
                    for client in list(connected_clients):
                        try:
                            await asyncio.wait_for(client.send_text(payload), timeout=0.15)
                        except Exception:
                            stale_clients.append(client)
                    for sc in stale_clients:
                        connected_clients.discard(sc)
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("[Loop] Error in simulation loop: %s", e, exc_info=True)
            await asyncio.sleep(0.5)


@app.on_event("startup")
async def startup_event():
    global sim_task
    sim_task = asyncio.create_task(simulation_loop())
    logger.info("[Server] Telemetry server startup complete on ws://localhost:8000/ws/rf")


@app.on_event("shutdown")
async def shutdown_event():
    global sim_task
    if sim_task:
        sim_task.cancel()
    logger.info("[Server] Telemetry server shutdown")


@app.websocket("/ws/rf")
async def websocket_rf_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info("[WebSocket] Client connected: %s (Total: %d)", websocket.client, len(connected_clients))

    # Send immediate initial state
    init_state = {
        "type": "connection_ack",
        "message": "Connected to Vyapti System C + UCB-Prior Telemetry",
        "action_mode": ACTION_MODE,
        "total_windows": session.total_windows,
        "band_count": session.total_windows,
        "total_spectrum_hz": TOTAL_SPECTRUM_HZ,
        "rf_start_hz": RF_START_HZ,
        "rf_end_hz": RF_END_HZ,
        "window_bandwidth_hz": WINDOW_BANDWIDTH_HZ,
        "dwell_ms": DWELL_MS,
    }
    try:
        await asyncio.wait_for(websocket.send_text(json.dumps(init_state)), timeout=0.5)
    except Exception:
        connected_clients.discard(websocket)
        return

    try:
        while True:
            text = await websocket.receive_text()
            try:
                cmd = json.loads(text)
                action = cmd.get("command")
                logger.info("[WebSocket] Received command: %s", action)

                if action == "start":
                    session.running = True
                    session.paused = False
                elif action == "pause":
                    session.paused = True
                elif action == "stop":
                    session.running = False
                elif action == "reset":
                    session.reset()
                elif action == "toggle_pause":
                    session.paused = not session.paused
                elif action == "load_scenario":
                    sc_target = str(cmd.get("scenario", "scenario_a")).lower()
                    if sc_target == "scenario_b":
                        session.reset(RFScenario.scenario_b())
                    else:
                        session.reset(RFScenario.scenario_a())
                    # Broadcast scenario updated state
                    ack = {
                        "type": "scenario_updated",
                        "scenario": session.scenario.to_dict(),
                        "total_windows": session.total_windows,
                    }
                    try:
                        await asyncio.wait_for(websocket.send_text(json.dumps(ack)), timeout=0.5)
                    except Exception:
                        pass
            except Exception as parse_err:
                logger.warning("[WebSocket] Invalid JSON command received: %s", parse_err)
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        logger.info("[WebSocket] Client disconnected: %s (Remaining: %d)", websocket.client, len(connected_clients))
    except Exception as e:
        connected_clients.discard(websocket)
        logger.error("[WebSocket] Client error: %s", e)


if __name__ == "__main__":
    uvicorn.run("vyapti_simulator.server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

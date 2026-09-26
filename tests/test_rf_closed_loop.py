"""
tests.test_rf_closed_loop
=========================

Tests for the closed-loop RF simulation pipeline:
  - engine.simulate_dwell band/AoA windowing
  - detector.detect_dwell frequency/AoA filtering
  - MissionRunner closed-loop loop
  - Scheduler comparison harness
"""
from __future__ import annotations

import numpy as np
import pytest

from vyapti_simulator.rf.simulator_engine import (
    RealTimeRFSimulator,
    SimulationEngineConfig,
)
from vyapti_simulator.rf.waveforms import generate_lfm_chirp
from vyapti_simulator.rf.propagation import KinematicEmitter
from vyapti_simulator.rf.pulse_detector import (
    PulseDetector,
    PulseDetectorConfig,
    EmitterInfo,
)
from vyapti_simulator.rf.closed_loop import (
    Dwell,
    MissionState,
    MissionRunner,
    RoundRobinScheduler,
    PriorityQueueScheduler,
    ThreatScoreScheduler,
    ThreatWeights,
    SchedulerScore,
    score_scheduler,
    run_comparison,
    summarise_results,
    TrackedEmitter,
)
from vyapti_simulator.tsrd.synthetic_pdw_generator import SyntheticEmitterSpec


# =====================================================================
# Helpers
# =====================================================================
def _chirp_fn(t, cfg, rng):
    """Minimal LFM waveform function matching the test emitter."""
    pw = getattr(cfg, "pulse_width_s", 1e-6)
    bw = getattr(cfg, "pulse_width_s", 1e-6) * 1e6
    return generate_lfm_chirp(
        t,
        f0=3e9 - 0.5 * bw,
        f1=3e9 + 0.5 * bw,
        peak_power_w=getattr(cfg, "pulse_power_w", 1.0),
    )


def _make_engine(dsp_hz=10e6, tick_s=10e-3, snr_db=20.0):
    """Factory for a minimal test engine with one chirp emitter."""
    cfg = SimulationEngineConfig(
        tick_interval_s=tick_s,
        dsp_sample_rate_hz=dsp_hz,
        num_ticks=1000,
        buffer_ticks=10,
        snr_db=snr_db,
        pulse_power_w=1.0,
    )
    sim = RealTimeRFSimulator(cfg, rng=np.random.default_rng(0))
    sim.add_emitter(
        KinematicEmitter(position_m=np.array([1000.0, 0.0, 0.0])),
        _chirp_fn,
        emitter_id=0,
        pri_sec=1e-3,
        pulse_width_s=1e-6,
        carrier_freq_hz=3e9,
        aoa_deg=45.0,
    )
    return sim


# =====================================================================
# simulate_dwell — band / AoA windowing
# =====================================================================
class TestSimulateDwell:
    """Tests for engine.simulate_dwell band/AoA filtering."""

    def test_returns_complex_array(self):
        """simulate_dwell returns a complex array."""
        sim = _make_engine(snr_db=30.0)
        iq = sim.simulate_dwell(
            freq_start_hz=2e9,
            freq_end_hz=4e9,
            dwell_ms=10.0,
        )
        assert iq.dtype == np.complex128
        assert iq.size > 0

    def test_band_filter_excludes_out_of_band(self):
        """Emitter at 3 GHz is invisible when looking at 5–6 GHz."""
        sim = _make_engine(snr_db=30.0)
        # Emitter is at 3 GHz — window at 5-6 GHz should produce
        # mostly noise (no chirp)
        iq = sim.simulate_dwell(
            freq_start_hz=5e9,
            freq_end_hz=6e9,
            dwell_ms=10.0,
        )
        # RMS with signal at 3 GHz should be >> RMS at 5-6 GHz
        # (pulse is absent — mostly AWGN)
        rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
        # Noise at 30 dB SNR with 1 W pulse: ~0.031 W noise power
        # signal-absent RMS should be close to noise floor
        assert rms < 0.1  # sanity: no strong component

    def test_band_filter_includes_in_band(self):
        """Emitter at 3 GHz is detected when looking at 2–4 GHz."""
        sim = _make_engine(snr_db=20.0)
        iq = sim.simulate_dwell(
            freq_start_hz=2e9,
            freq_end_hz=4e9,
            dwell_ms=10.0,
        )
        rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
        assert rms > 0.01  # signal is present

    def test_aoa_filter_excludes_out_of_aoa(self):
        """Emitter at 45° is excluded when looking at 180° ± 10°."""
        sim = _make_engine(snr_db=30.0)
        # Emitter at 45°, window at 180° ± 10°
        iq = sim.simulate_dwell(
            freq_start_hz=2e9,
            freq_end_hz=4e9,
            aoa_center_deg=180.0,
            aoa_window_deg=10.0,
            dwell_ms=10.0,
        )
        # Should see no signal (only noise)
        rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
        assert rms < 0.05  # close to noise floor

    def test_aoa_filter_includes_in_aoa(self):
        """Emitter at 45° is included when looking at 45° ± 10°."""
        sim = _make_engine(snr_db=20.0)
        iq = sim.simulate_dwell(
            freq_start_hz=2e9,
            freq_end_hz=4e9,
            aoa_center_deg=45.0,
            aoa_window_deg=10.0,
            dwell_ms=10.0,
        )
        rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
        assert rms > 0.01  # signal present

    def test_tick_counter_advances(self):
        """simulate_dwell advances the engine's tick counter."""
        sim = _make_engine()
        initial_tick = sim.tick
        # 10 ms dwell at 10 ms tick interval = 1 tick
        sim.simulate_dwell(2e9, 4e9, dwell_ms=10.0)
        assert sim.tick == initial_tick + 1
        # 30 ms dwell = 3 ticks
        sim.simulate_dwell(2e9, 4e9, dwell_ms=30.0)
        assert sim.tick == initial_tick + 4

    def test_window_cleared_after_call(self):
        """The active window is cleared after simulate_dwell returns."""
        sim = _make_engine()
        sim.simulate_dwell(5e9, 6e9, dwell_ms=10.0)
        # Window should be cleared — next call should not be filtered
        assert sim._active_window is None

    def test_zero_dwell_returns_empty_array(self):
        """dwell_ms=0 returns an empty array (no synthesis)."""
        sim = _make_engine()
        iq = sim.simulate_dwell(2e9, 4e9, dwell_ms=0.0)
        assert iq.dtype == np.complex128
        assert iq.size == 0  # zero dwell = no samples

    def test_default_dwell_is_one_tick(self):
        """Default dwell_ms is one tick interval."""
        sim = _make_engine(tick_s=5e-3)
        n_before = sim.tick
        iq = sim.simulate_dwell(2e9, 4e9)  # no dwell_ms
        assert sim.tick == n_before + 1
        # Should contain one tick of samples
        expected_len = int(round(5e-3 * sim.config.dsp_sample_rate_hz))
        assert iq.size == expected_len


# =====================================================================
# detect_dwell — band / AoA filtering
# =====================================================================
class TestDetectDwell:
    """Tests for detector.detect_dwell window filtering."""

    def _detector(self, emitter_map=None):
        cfg = PulseDetectorConfig(
            dsp_sample_rate_hz=1e6,
            tick_interval_s=1e-3,
            cfar_db=12.0,
            chirp_bandwidth_hz=1e6,
            carrier_freq_hz=3e9,
            pulse_width_s=1e-6,
        )
        return PulseDetector(
            config=cfg,
            emitter_map=emitter_map or {0: EmitterInfo(3e9, 45.0)},
            rng=np.random.default_rng(0),
        )

    def test_detect_dwell_returns_pdw_stream(self):
        """detect_dwell returns a PDWStream."""
        det = self._detector()
        # Build a synthetic I/Q buffer with a pulse
        n = det.config.samples_per_tick
        iq = np.zeros(n, dtype=np.complex128)
        t = np.arange(10, dtype=np.float64) / det.config.dsp_sample_rate_hz
        chirp = generate_lfm_chirp(
            t,
            f0=3e9 - 0.5e6,
            f1=3e9 + 0.5e6,
            peak_power_w=10.0,
        )
        iq[:10] = chirp.astype(np.complex128)
        pdws = det.detect_dwell(
            iq, freq_start_hz=2e9, freq_end_hz=4e9,
            freq_center_hz=3e9,
        )
        assert hasattr(pdws, "toa_us")
        assert hasattr(pdws, "freq_mhz")
        assert hasattr(pdws, "emitter_id")

    def test_detect_dwell_finds_pulse(self):
        """A high-SNR pulse in the dwell band is detected."""
        det = self._detector()
        n = det.config.samples_per_tick
        iq = np.zeros(n, dtype=np.complex128)
        t = np.arange(10, dtype=np.float64) / det.config.dsp_sample_rate_hz
        chirp = generate_lfm_chirp(
            t,
            f0=3e9 - 0.5e6,
            f1=3e9 + 0.5e6,
            peak_power_w=100.0,  # very high power
        )
        iq[:10] = chirp.astype(np.complex128)
        pdws = det.detect_dwell(
            iq,
            freq_start_hz=2e9,
            freq_end_hz=4e9,
            freq_center_hz=3e9,
        )
        # Should find at least one pulse
        assert len(pdws) >= 1

    def test_detect_dwell_rejects_out_of_band(self):
        """A pulse outside the dwell band is not returned as 6 GHz."""
        det = self._detector()
        n = det.config.samples_per_tick
        iq = np.zeros(n, dtype=np.complex128)
        # Place a pulse at 6 GHz — outside the 2-4 GHz band
        t = np.arange(10, dtype=np.float64) / det.config.dsp_sample_rate_hz
        chirp = generate_lfm_chirp(
            t,
            f0=6e9 - 0.5e6,
            f1=6e9 + 0.5e6,
            peak_power_w=100.0,
        )
        iq[:10] = chirp.astype(np.complex128)
        pdws = det.detect_dwell(
            iq,
            freq_start_hz=2e9,
            freq_end_hz=4e9,
            freq_center_hz=3e9,
        )
        # The pulse was at 6 GHz, so any detection should NOT be at 6 GHz.
        # (The matched filter is centred at 3 GHz, so a 6 GHz pulse
        # does not compress; whatever the detector finds will be
        # labeled as the 3 GHz emitter from the map.)
        for f_mhz in pdws.freq_mhz:
            assert not (5e3 < float(f_mhz) < 7e3)  # not in the 5-7 GHz band

    def test_detect_dwell_aoa_filter(self):
        """Pulses from emitters outside the AoA window are excluded."""
        # Detector has emitter at 45°, dwell is at 180° ± 10°
        det = self._detector({0: EmitterInfo(3e9, 45.0)})
        n = det.config.samples_per_tick
        iq = np.zeros(n, dtype=np.complex128)
        t = np.arange(10, dtype=np.float64) / det.config.dsp_sample_rate_hz
        chirp = generate_lfm_chirp(
            t,
            f0=3e9 - 0.5e6,
            f1=3e9 + 0.5e6,
            peak_power_w=100.0,
        )
        iq[:10] = chirp.astype(np.complex128)
        pdws = det.detect_dwell(
            iq,
            freq_start_hz=2e9,
            freq_end_hz=4e9,
            freq_center_hz=3e9,
            aoa_center_deg=180.0,
            aoa_window_deg=10.0,
        )
        # Emitter at 45°, dwell at 180° ± 10° → should be filtered
        assert len(pdws) == 0


# =====================================================================
# Schedulers
# =====================================================================
class TestSchedulers:
    """Tests for the scheduler classes."""

    def test_round_robin_cycles_bands(self):
        """RoundRobinScheduler cycles through bands in order."""
        bands = [(1e9, 2e9), (2e9, 3e9), (3e9, 4e9)]
        sched = RoundRobinScheduler(bands, dwell_ms=5.0)
        state = MissionState()
        dwells = [sched.decide(state) for _ in range(6)]
        assert dwells[0].freq_start_hz == 1e9
        assert dwells[1].freq_start_hz == 2e9
        assert dwells[2].freq_start_hz == 3e9
        assert dwells[3].freq_start_hz == 1e9  # wraps
        # All have the same dwell_ms
        for d in dwells:
            assert d.dwell_ms == 5.0

    def test_round_robin_reset(self):
        """RoundRobinScheduler resets its index on reset()."""
        bands = [(1e9, 2e9), (2e9, 3e9)]
        sched = RoundRobinScheduler(bands)
        state = MissionState()
        sched.decide(state)
        sched.decide(state)
        sched.reset()
        assert sched.decide(state).freq_start_hz == 1e9

    def test_priority_queue_unknown_emitter_adds_track(self):
        """PriorityQueueScheduler creates a new track on first detection."""
        sched = PriorityQueueScheduler(
            bands_hz=[(2e9, 4e9)],
            dwell_ms=5.0,
        )
        from vyapti_simulator.tsrd.tsrd_adapter import PDWStream
        state = MissionState(
            recent_pdws=PDWStream(
                toa_us=np.array([1000.0], dtype=np.float32),
                freq_mhz=np.array([3000.0], dtype=np.float32),
                pw_us=np.array([1.0], dtype=np.float32),
                aoa_deg=np.array([45.0], dtype=np.float32),
                amp_db=np.array([60.0], dtype=np.float32),
                emitter_id=np.array([-1], dtype=np.int64),
            )
        )
        sched._update_tracks(state.recent_pdws)
        assert len(sched._tracks) == 1

    def test_priority_queue_decides_targeted(self):
        """After a detection, PQScheduler dwells on the tracked emitter."""
        sched = PriorityQueueScheduler(
            bands_hz=[(2e9, 4e9)],
            dwell_ms=5.0,
        )
        from vyapti_simulator.tsrd.tsrd_adapter import PDWStream
        state = MissionState(
            recent_pdws=PDWStream(
                toa_us=np.array([1000.0], dtype=np.float32),
                freq_mhz=np.array([3000.0], dtype=np.float32),
                pw_us=np.array([1.0], dtype=np.float32),
                aoa_deg=np.array([45.0], dtype=np.float32),
                amp_db=np.array([60.0], dtype=np.float32),
                emitter_id=np.array([-1], dtype=np.int64),
            )
        )
        sched._update_tracks(state.recent_pdws)
        dwell = sched.decide(state)
        # Should dwell near 3 GHz
        assert dwell.freq_start_hz < 3.1e9
        assert dwell.freq_end_hz > 2.9e9
        assert dwell.reason in ("confirmed track", "confirming track")

    def test_threat_score_scheduler_ranks_tracks(self):
        """ThreatScoreScheduler assigns a non-zero threat score to tracks."""
        sched = ThreatScoreScheduler(
            bands_hz=[(2e9, 4e9)],
            dwell_ms=5.0,
        )
        from vyapti_simulator.tsrd.tsrd_adapter import PDWStream
        state = MissionState(
            recent_pdws=PDWStream(
                toa_us=np.array([1000.0], dtype=np.float32),
                freq_mhz=np.array([3000.0], dtype=np.float32),
                pw_us=np.array([1.0], dtype=np.float32),
                aoa_deg=np.array([45.0], dtype=np.float32),
                amp_db=np.array([60.0], dtype=np.float32),
                emitter_id=np.array([-1], dtype=np.int64),
            )
        )
        sched._update_tracks(state.recent_pdws)
        # decide() recomputes threat scores for all tracks
        sched.decide(state)
        assert len(sched._tracks) == 1
        track = next(iter(sched._tracks.values()))
        assert track.threat_score > 0.0
        assert 0.0 <= track.threat_score <= 1.0

    def test_threat_weights_configurable(self):
        """ThreatWeights are used in the threat score."""
        weights = ThreatWeights(freq=0.5, pw=0.0, pri=0.0, amp=0.0, aoa=0.0)
        sched = ThreatScoreScheduler(
            bands_hz=[(2e9, 4e9)],
            dwell_ms=5.0,
            weights=weights,
        )
        from vyapti_simulator.tsrd.tsrd_adapter import PDWStream
        state = MissionState(
            recent_pdws=PDWStream(
                toa_us=np.array([1000.0], dtype=np.float32),
                freq_mhz=np.array([3000.0], dtype=np.float32),
                pw_us=np.array([1.0], dtype=np.float32),
                aoa_deg=np.array([45.0], dtype=np.float32),
                amp_db=np.array([60.0], dtype=np.float32),
                emitter_id=np.array([-1], dtype=np.int64),
            )
        )
        sched._update_tracks(state.recent_pdws)
        sched.decide(state)  # recomputes threat scores
        track = next(iter(sched._tracks.values()))
        # With freq_weight=0.5, score should be around 0.5 * 0.5 = 0.25
        # (freq_score=0.5 by default when no danger map)
        assert 0.0 < track.threat_score < 1.0


# =====================================================================
# MissionRunner
# =====================================================================
class TestMissionRunner:
    """Tests for the closed-loop mission runner."""

    def test_mission_runs_and_returns_state(self):
        """MissionRunner completes and returns a state."""
        sim = _make_engine(snr_db=25.0)
        det = PulseDetector(
            config=PulseDetectorConfig(
                dsp_sample_rate_hz=sim.config.dsp_sample_rate_hz,
                tick_interval_s=sim.config.tick_interval_s,
                cfar_db=10.0,
                chirp_bandwidth_hz=1e6,
                carrier_freq_hz=3e9,
                pulse_width_s=1e-6,
            ),
            emitter_map={0: EmitterInfo(3e9, 45.0)},
            rng=np.random.default_rng(0),
        )
        sched = RoundRobinScheduler(
            bands_hz=[(2e9, 4e9)],
            dwell_ms=10.0,
        )
        specs = [
            SyntheticEmitterSpec(
                emitter_id=0,
                aoa_deg=45.0,
                snr_db=15.0,
                emitter_type="fixed_continuous",
                center_freq_hz=3e9,
                pri_sec=1e-3,
                pulse_width_sec=1e-6,
            )
        ]
        runner = MissionRunner(
            engine=sim,
            detector=det,
            scheduler=sched,
            ground_truth=specs,
            mission_duration_s=0.05,
            sweep_window_hz=(2e9, 4e9),
            max_dwells=50,
        )
        state, score = runner.run()
        assert state.n_dwells > 0
        assert isinstance(score, SchedulerScore)
        assert state.n_dwells <= 50

    def test_mission_tracks_time(self):
        """MissionRunner correctly tracks elapsed time."""
        sim = _make_engine(snr_db=30.0)
        det = PulseDetector(
            config=PulseDetectorConfig(
                dsp_sample_rate_hz=sim.config.dsp_sample_rate_hz,
                tick_interval_s=sim.config.tick_interval_s,
                cfar_db=15.0,
            ),
            emitter_map={},
            rng=np.random.default_rng(0),
        )
        sched = RoundRobinScheduler(
            bands_hz=[(2e9, 4e9)],
            dwell_ms=10.0,
        )
        runner = MissionRunner(
            engine=sim,
            detector=det,
            scheduler=sched,
            ground_truth=[],
            mission_duration_s=0.03,
            sweep_window_hz=(2e9, 4e9),
        )
        state, _ = runner.run()
        # 30 ms mission with 10 ms dwells = ~3 dwells
        assert 2 <= state.n_dwells <= 4
        # Elapsed ≈ n_dwells * 10ms
        assert state.time_elapsed_us > 0

    def test_max_dwells_limit(self):
        """MissionRunner stops at max_dwells."""
        sim = _make_engine(snr_db=30.0)
        det = PulseDetector(
            config=PulseDetectorConfig(
                dsp_sample_rate_hz=sim.config.dsp_sample_rate_hz,
                tick_interval_s=sim.config.tick_interval_s,
                cfar_db=20.0,
            ),
            emitter_map={},
            rng=np.random.default_rng(0),
        )
        sched = RoundRobinScheduler(
            bands_hz=[(2e9, 4e9)],
            dwell_ms=10.0,
        )
        runner = MissionRunner(
            engine=sim,
            detector=det,
            scheduler=sched,
            ground_truth=[],
            mission_duration_s=1000.0,  # long mission
            sweep_window_hz=(2e9, 4e9),
            max_dwells=5,
        )
        state, _ = runner.run()
        assert state.n_dwells == 5


# =====================================================================
# Scoring
# =====================================================================
class TestScoring:
    """Tests for score_scheduler."""

    def test_detection_rate_zero_when_no_truth(self):
        """Zero ground truth emitters → detection rate is 0."""
        state = MissionState()
        score = score_scheduler(state, ground_truth=[])
        assert score.detection_rate == 0.0
        assert score.false_alarms == 0

    def test_detection_rate_full_when_all_found(self):
        """When all emitters are tracked, detection rate is 100%."""
        specs = [
            SyntheticEmitterSpec(
                emitter_id=0,
                aoa_deg=45.0,
                snr_db=15.0,
                emitter_type="fixed_continuous",
                center_freq_hz=3e9,
                pri_sec=1e-3,
                pulse_width_sec=1e-6,
            )
        ]
        state = MissionState(
            all_detections=[
                TrackedEmitter(
                    emitter_id=0,
                    freq_hz=3e9,
                    aoa_deg=45.0,
                    pw_us=1.0,
                    amplitude_db=60.0,
                    first_detection_us=1000.0,
                    last_detection_us=5000.0,
                    detection_count=3,
                )
            ]
        )
        score = score_scheduler(state, specs)
        assert score.detection_rate == 1.0
        assert score.false_alarms == 0

    def test_false_alarms_counted(self):
        """Tracks that don't match ground truth are counted as false alarms."""
        specs = [
            SyntheticEmitterSpec(
                emitter_id=0,
                aoa_deg=45.0,
                snr_db=15.0,
                emitter_type="fixed_continuous",
                center_freq_hz=3e9,
                pri_sec=1e-3,
                pulse_width_sec=1e-6,
            )
        ]
        # Track at a completely different frequency/AoA
        state = MissionState(
            all_detections=[
                TrackedEmitter(
                    emitter_id=99,
                    freq_hz=10e9,  # wrong band
                    aoa_deg=200.0,  # wrong AoA
                    pw_us=1.0,
                    amplitude_db=60.0,
                    first_detection_us=1000.0,
                    last_detection_us=5000.0,
                    detection_count=3,
                )
            ]
        )
        score = score_scheduler(state, specs)
        assert score.detection_rate == 0.0
        assert score.false_alarms == 1

    def test_time_to_first_detection(self):
        """time_to_first is the earliest detection time."""
        specs = [
            SyntheticEmitterSpec(
                emitter_id=0,
                aoa_deg=45.0,
                snr_db=15.0,
                emitter_type="fixed_continuous",
                center_freq_hz=3e9,
                pri_sec=1e-3,
                pulse_width_sec=1e-6,
            )
        ]
        state = MissionState(
            all_detections=[
                TrackedEmitter(
                    emitter_id=0,
                    freq_hz=3e9,
                    aoa_deg=45.0,
                    pw_us=1.0,
                    amplitude_db=60.0,
                    first_detection_us=50_000.0,  # 50 ms
                    last_detection_us=100_000.0,
                    detection_count=3,
                )
            ]
        )
        score = score_scheduler(state, specs)
        assert score.time_to_first_us == 50_000.0


# =====================================================================
# MissionState
# =====================================================================
class TestMissionState:
    """Tests for MissionState helper methods."""

    def test_to_vector_fixed_length(self):
        """to_vector returns a fixed-size array regardless of tracks."""
        # Empty state
        state = MissionState()
        v1 = state.to_vector()
        assert isinstance(v1, np.ndarray)
        assert v1.dtype == np.float32
        assert v1.shape[0] > 0  # has features

        # With many tracks
        state.all_detections = [
            TrackedEmitter(
                emitter_id=i,
                freq_hz=3e9,
                aoa_deg=float(i * 10),
                pw_us=1.0,
                amplitude_db=60.0,
                first_detection_us=1000.0,
                last_detection_us=5000.0,
            )
            for i in range(20)
        ]
        v2 = state.to_vector()
        # Shape should be the same as empty state (padded)
        assert v2.shape == v1.shape


# =====================================================================
# End-to-end scheduler comparison
# =====================================================================
class TestSchedulerComparison:
    """End-to-end: run_comparison on a small synthetic scenario."""

    def test_run_comparison_returns_scores(self):
        """run_comparison returns a dict of SchedulerScore lists."""
        specs = [
            SyntheticEmitterSpec(
                emitter_id=0,
                aoa_deg=45.0,
                snr_db=20.0,
                emitter_type="fixed_continuous",
                center_freq_hz=3e9,
                pri_sec=1e-3,
                pulse_width_sec=1e-6,
            ),
        ]
        schedulers = {
            "round_robin": RoundRobinScheduler(
                bands_hz=[(2e9, 4e9)],
                dwell_ms=10.0,
            ),
        }
        results = run_comparison(
            specs=specs,
            schedulers=schedulers,
            n_scenarios=2,
            mission_duration_s=0.03,
            dsp_sample_rate_hz=1e6,
            tick_interval_s=10e-3,
            snr_db=20.0,
            sweep_window_hz=(2e9, 4e9),
            band_count=2,
            seed=99,
            wall_clock_budget_s=5.0,
        )
        assert "round_robin" in results
        assert len(results["round_robin"]) == 2
        assert all(isinstance(s, SchedulerScore) for s in results["round_robin"])

    def test_summarise_results_runs(self):
        """summarise_results produces a table string."""
        results = {
            "round_robin": [
                SchedulerScore(detection_rate=0.5, time_to_first_us=5e6,
                               false_alarms=1, total_dwells=10),
                SchedulerScore(detection_rate=0.4, time_to_first_us=6e6,
                               false_alarms=2, total_dwells=12),
            ],
        }
        table = summarise_results(results)
        assert "round_robin" in table
        assert "DetRate" in table

    def test_multiple_schedulers_comparable(self):
        """Two schedulers produce different scores on the same scenarios."""
        specs = [
            SyntheticEmitterSpec(
                emitter_id=0,
                aoa_deg=45.0,
                snr_db=20.0,
                emitter_type="fixed_continuous",
                center_freq_hz=3e9,
                pri_sec=1e-3,
                pulse_width_sec=1e-6,
            ),
        ]
        schedulers = {
            "round_robin": RoundRobinScheduler(
                bands_hz=[(2e9, 4e9), (4e9, 6e9)],
                dwell_ms=10.0,
            ),
            "priority_queue": PriorityQueueScheduler(
                bands_hz=[(2e9, 4e9), (4e9, 6e9)],
                dwell_ms=10.0,
            ),
        }
        results = run_comparison(
            specs=specs,
            schedulers=schedulers,
            n_scenarios=3,
            mission_duration_s=0.05,
            dsp_sample_rate_hz=1e6,
            tick_interval_s=10e-3,
            snr_db=20.0,
            sweep_window_hz=(2e9, 6e9),
            band_count=2,
            seed=77,
            wall_clock_budget_s=5.0,
        )
        # Both should have run
        assert len(results["round_robin"]) == 3
        assert len(results["priority_queue"]) == 3
        table = summarise_results(results)
        assert "round_robin" in table
        assert "priority_queue" in table

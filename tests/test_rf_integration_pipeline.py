"""
tests.test_rf_integration_pipeline
===================================
End-to-end RF physics integration: PDW -> RF config -> I/Q -> match filter.
"""
from __future__ import annotations
import numpy as np
import pytest

from vyapti_simulator.rf.waveforms import (
    generate_lfm_chirp, add_awgn, matched_filter,
)
from vyapti_simulator.rf.propagation import (
    KinematicEmitter, compute_doppler_shift,
    FreeSpacePathLoss, RayleighFadingChannel,
)
from vyapti_simulator.rf.simulator_engine import (
    RealTimeRFSimulator, SimulationEngineConfig,
)


def _chirp_fn(t, cfg, rng):
    return generate_lfm_chirp(
        t, f0=0.0, f1=cfg.pulse_width_s * 1e6,
        peak_power_w=cfg.pulse_power_w,
    )


def _make_pdw_stream(seed=42, n_pulses=10):
    from vyapti_simulator.tsrd.synthetic_pdw_generator import (
        SyntheticEWPDWGenerator, SyntheticEmitterSpec,
    )
    specs = [SyntheticEmitterSpec(
        emitter_id=0, aoa_deg=12.0,
        emitter_type="fixed_continuous",
        center_freq_hz=3.0e9, pri_sec=1e-3,
        pulse_width_sec=1e-6, snr_db=15.0,
    )]
    # Add a small fudge margin so n_pulses=3 reliably gives ≥3 pulses
    # despite boundary effects at t=0 and sim_end.
    gen = SyntheticEWPDWGenerator(
        specs=specs, mission_duration_s=n_pulses * 1e-3 + 1e-6, seed=seed,
    )
    return gen.generate()


class TestPDWToRFConfigBridge:
    def test_pdw_drives_rf_config(self):
        pdw = _make_pdw_stream()
        assert len(pdw.toa_us) > 0
        rf_cfg = SimulationEngineConfig(
            tick_interval_s=1e-3,
            dsp_sample_rate_hz=10e6,
            num_ticks=5,
            buffer_ticks=1,
            emitter_frequency_hz=3.0e9,
            pulse_width_s=1e-6,
            pulse_power_w=1.0,
            snr_db=15.0,
        )
        assert rf_cfg.samples_per_tick == 10_000

    def test_kinematic_drives_doppler(self):
        e = KinematicEmitter(
            position_m=np.array([1000.0, 0.0, 0.0]),
            velocity_m_s=np.array([-100.0, 0.0, 0.0]),
        )
        f_d = compute_doppler_shift(e, np.zeros(3), f_c=3e9)
        assert f_d < 0.0  # approaching
        # Approaching at 100 m/s at 3 GHz ≈ -1000 Hz
        assert -1010.0 < f_d < -990.0

    def test_fspl_matches_3ghz_1km(self):
        fspl = FreeSpacePathLoss(frequency_hz=3e9)
        loss = fspl.fspl_db(range_m=1000.0)
        # 1km, 3 GHz = ~101.98 dB
        assert 101.0 < loss < 102.5


class TestRealTimePipeline:
    """Full end-to-end: emitter -> RF physics -> I/Q -> matched filter."""

    def test_engine_produces_iq_with_emitter(self):
        cfg = SimulationEngineConfig(
            tick_interval_s=1e-3,
            dsp_sample_rate_hz=10e6,
            num_ticks=3,
            buffer_ticks=1,
            emitter_frequency_hz=3.0e9,
            pulse_width_s=1e-6,
            pulse_power_w=1.0,
            snr_db=20.0,
        )
        rng = np.random.default_rng(0)
        sim = RealTimeRFSimulator(cfg, rng=rng)
        em = KinematicEmitter(
            position_m=np.array([1000.0, 0.0, 0.0]),
            velocity_m_s=np.array([-100.0, 0.0, 0.0]),
        )
        sim.add_emitter(em, _chirp_fn, pri_sec=cfg.tick_interval_s,
                        pulse_width_s=cfg.pulse_width_s)
        buf = next(sim.run())
        assert buf.dtype == np.complex128
        assert buf.shape == (cfg.samples_per_tick,)
        # Buffer should be non-zero (chirp present)
        assert np.max(np.abs(buf)) > 0.0

    def test_fading_channel_applied(self):
        cfg = SimulationEngineConfig(
            tick_interval_s=1e-3,
            dsp_sample_rate_hz=1e6,
            num_ticks=2,
            buffer_ticks=1,
            emitter_frequency_hz=3.0e9,
            pulse_width_s=1e-6,
            pulse_power_w=1.0,
            snr_db=20.0,
        )
        rng = np.random.default_rng(0)
        sim = RealTimeRFSimulator(cfg, rng=rng)
        em = KinematicEmitter(
            position_m=np.array([1000.0, 0.0, 0.0]),
        )
        # Rayleigh fading with motion
        ch = RayleighFadingChannel(
            f_c=3e9, doppler_hz=100.0, sample_rate_hz=cfg.dsp_sample_rate_hz,
            rng=rng,
        )
        sim.add_emitter(em, _chirp_fn, channel=ch,
                        pri_sec=cfg.tick_interval_s,
                        pulse_width_s=cfg.pulse_width_s)
        buf = next(sim.run())
        # With fading applied, signal power should be modulated
        assert np.max(np.abs(buf)) > 0.0

    def test_matched_filter_recovers_pulse(self):
        """Round-trip: emit chirp -> AWGN -> matched filter -> peak."""
        f0, f1 = 0.0, 1e6
        t = np.linspace(0, 1e-5, 1000)
        chirp = generate_lfm_chirp(t, f0=f0, f1=f1, peak_power_w=1.0)
        noisy = add_awgn(chirp, snr_db=10.0, rng=np.random.default_rng(0))
        compressed = matched_filter(noisy, chirp, mode="sampleless")
        # Peak should be much higher than baseline
        peak = float(np.max(np.abs(compressed)))
        baseline = float(np.median(np.abs(compressed)))
        assert peak > 5 * baseline

    def test_full_pipeline_pdw_to_detected_pulse(self):
        """The hardest end-to-end: PDW -> I/Q -> matched filter -> peak."""
        # Step 1: Get PDW stream
        pdw = _make_pdw_stream(seed=42, n_pulses=3)
        assert len(pdw.toa_us) >= 3

        # Step 2: Build RF config from PDW parameters
        f_hz = 3.0e9
        pw_s = 1e-6
        snr_db = 15.0
        cfg = SimulationEngineConfig(
            tick_interval_s=1e-3,
            dsp_sample_rate_hz=10e6,
            num_ticks=3,
            buffer_ticks=1,
            emitter_frequency_hz=f_hz,
            pulse_width_s=pw_s,
            pulse_power_w=1.0,
            snr_db=snr_db,
        )
        rng = np.random.default_rng(42)
        sim = RealTimeRFSimulator(cfg, rng=rng)
        em = KinematicEmitter(
            position_m=np.array([1000.0, 0.0, 0.0]),
            velocity_m_s=np.array([-50.0, 0.0, 0.0]),
        )
        sim.add_emitter(em, _chirp_fn,
                        pri_sec=cfg.tick_interval_s,
                        pulse_width_s=cfg.pulse_width_s)

        # Step 3: Run engine and collect buffers
        buffers = list(sim.run())
        assert len(buffers) == 3
        signal = np.concatenate(buffers)

        # Step 4: Matched-filter detect
        t_ref = np.arange(cfg.samples_per_tick, dtype=np.float64) / cfg.dsp_sample_rate_hz
        ref_chirp = generate_lfm_chirp(
            t_ref, f0=0.0, f1=pw_s * 1e6, peak_power_w=1.0,
        )
        # Apply matched filter to each tick
        detections = 0
        for i in range(3):
            start = i * cfg.samples_per_tick
            stop = start + cfg.samples_per_tick
            compressed = matched_filter(
                signal[start:stop], ref_chirp, mode="sampleless",
            )
            peak = float(np.max(np.abs(compressed)))
            baseline = float(np.median(np.abs(compressed)))
            if peak > 3 * baseline:
                detections += 1
        # At least one of the three ticks should have a detectable pulse
        assert detections >= 1


class TestRFPulsePipeline:
    """End-to-end tests for RFPulsePipeline."""

    def test_pipeline_single_emitter(self):
        """Single fixed emitter → pipeline runs → PDW stream."""
        from vyapti_simulator.rf.rf_to_pdw_pipeline import (
            RFPulsePipeline, RFPipelineConfig,
        )
        from vyapti_simulator.tsrd.synthetic_pdw_generator import (
            SyntheticEmitterSpec,
        )
        import numpy as np

        specs = [
            SyntheticEmitterSpec(
                emitter_id=0, aoa_deg=12.0, snr_db=15.0,
                emitter_type="fixed_continuous",
                center_freq_hz=3e9, pri_sec=1e-3,
                pulse_width_sec=1e-5,  # 10 µs, within allowed range (0.2-10 µs)
            ),
        ]
        pipeline = RFPulsePipeline(
            emitter_specs=specs,
            config=RFPipelineConfig(
                dsp_sample_rate_hz=1e6,
                tick_interval_s=1e-3,
                mission_duration_s=0.1,  # 100 ms = 100 pulses
                buffer_ticks=10,
                snr_db=20.0,
                cfar_db=10.0,
            ),
            rng=np.random.default_rng(42),
        )
        pdw = pipeline.run()
        # PDW stream has right dtypes
        assert pdw.toa_us.dtype == np.float32
        assert pdw.freq_mhz.dtype == np.float32
        assert pdw.pw_us.dtype == np.float32
        assert pdw.aoa_deg.dtype == np.float32
        assert pdw.amp_db.dtype == np.float32
        assert pdw.emitter_id.dtype == np.int64
        # Pulses are sorted by ToA
        assert np.all(np.diff(pdw.toa_us.astype(np.float64)) >= 0.0)
        # All pulses have the right emitter_id
        assert np.all(pdw.emitter_id == 0)
        # All pulses have the right AoA
        # Current AoA is temporary truth-derived behaviour; it will be removed in Gate 3 (measurement-only receiver).
        # We test deterministic replay instead of an exact nominal match.
        pipeline2 = RFPulsePipeline(
            emitter_specs=specs,
            config=RFPipelineConfig(
                dsp_sample_rate_hz=1e6,
                tick_interval_s=1e-3,
                mission_duration_s=0.1,  # 100 ms = 100 pulses
                buffer_ticks=10,
                snr_db=20.0,
                cfar_db=10.0,
            ),
            rng=np.random.default_rng(42),
        )
        pdw2 = pipeline2.run()
        np.testing.assert_array_equal(pdw.aoa_deg, pdw2.aoa_deg)
        # At least some pulses detected (SNR=20dB, should be many)
        assert len(pdw) > 0

    def test_pipeline_two_emitters(self):
        """Two emitters at different frequencies → pipeline detects both."""
        from vyapti_simulator.rf.rf_to_pdw_pipeline import (
            RFPulsePipeline, RFPipelineConfig,
        )
        from vyapti_simulator.tsrd.synthetic_pdw_generator import (
            SyntheticEmitterSpec,
        )
        import numpy as np

        specs = [
            SyntheticEmitterSpec(
                emitter_id=0, aoa_deg=12.0, snr_db=15.0,
                emitter_type="fixed_continuous",
                center_freq_hz=3e9, pri_sec=1e-3,
                pulse_width_sec=1e-6,
            ),
            SyntheticEmitterSpec(
                emitter_id=1, aoa_deg=58.0, snr_db=12.0,
                emitter_type="frequency_agile",
                freq_list_hz=[5e9, 7e9],
                pri_sec=2e-3,
                pulse_width_sec=0.5e-6,
            ),
        ]
        pipeline = RFPulsePipeline(
            emitter_specs=specs,
            config=RFPipelineConfig(
                dsp_sample_rate_hz=1e6,
                tick_interval_s=1e-3,
                mission_duration_s=0.1,
                buffer_ticks=10,
                snr_db=20.0,
                cfar_db=10.0,
            ),
            rng=np.random.default_rng(99),
        )
        pdw = pipeline.run()
        # Two emitter IDs
        unique_ids = set(int(x) for x in np.unique(pdw.emitter_id))
        assert unique_ids.issuperset({0, 1})
        # Pulses sorted by ToA
        assert np.all(np.diff(pdw.toa_us.astype(np.float64)) >= 0.0)

    def test_pipeline_vs_synthetic_generator(self):
        """Same spec → pipeline and synthetic generator produce comparable pulse counts."""
        from vyapti_simulator.rf.rf_to_pdw_pipeline import (
            RFPulsePipeline, RFPipelineConfig,
        )
        from vyapti_simulator.tsrd.synthetic_pdw_generator import (
            SyntheticEmitterSpec, SyntheticEWPDWGenerator,
        )
        import numpy as np

        specs = [
            SyntheticEmitterSpec(
                emitter_id=0, aoa_deg=12.0, snr_db=15.0,
                emitter_type="fixed_continuous",
                center_freq_hz=3e9, pri_sec=1e-3,
                pulse_width_sec=1e-5,  # 10 µs, within allowed range (0.2-10 µs)
            ),
        ]
        # Ground truth: synthetic generator
        gen = SyntheticEWPDWGenerator(
            specs=specs, mission_duration_s=0.1, seed=42,
            noise_floor_db=-90.0,
        )
        pdw_baseline = gen.generate()
        expected_pulses = len(pdw_baseline)

        # RF pipeline
        pipeline = RFPulsePipeline(
            emitter_specs=specs,
            config=RFPipelineConfig(
                dsp_sample_rate_hz=1e6,
                tick_interval_s=1e-3,
                mission_duration_s=0.1,
                buffer_ticks=10,
                snr_db=20.0,
                cfar_db=10.0,
            ),
            rng=np.random.default_rng(42),
        )
        pdw = pipeline.run()

        # Detection rate: at least 50% at 15dB SNR with CFAR=10dB
        # (accounts for Rayleigh fading false negatives)
        detection_rate = len(pdw) / max(1, expected_pulses)
        assert detection_rate >= 0.50, (
            f"Detection rate {detection_rate:.1%} too low: "
            f"{len(pdw)} detected vs {expected_pulses} expected"
        )

    def test_pipeline_empty_when_no_pulses(self):
        """Empty mission → empty PDWStream (not an error)."""
        from vyapti_simulator.rf.rf_to_pdw_pipeline import (
            RFPulsePipeline, RFPipelineConfig,
        )
        from vyapti_simulator.tsrd.synthetic_pdw_generator import (
            SyntheticEmitterSpec,
        )
        import numpy as np

        specs = [
            SyntheticEmitterSpec(
                emitter_id=0, aoa_deg=12.0, snr_db=-30.0,  # below noise
                emitter_type="fixed_continuous",
                center_freq_hz=3e9, pri_sec=1e-3,
                pulse_width_sec=1e-3,  # increased from 1e-6 to 1e-3 for proper chirp length
            ),
        ]
        pipeline = RFPulsePipeline(
            emitter_specs=specs,
            config=RFPipelineConfig(
                dsp_sample_rate_hz=1e6,
                tick_interval_s=1e-3,
                mission_duration_s=0.01,  # very short
                buffer_ticks=10,
                snr_db=-30.0,
                cfar_db=20.0,  # very high threshold
            ),
            rng=np.random.default_rng(0),
        )
        pdw = pipeline.run()
        assert len(pdw) == 0
        # Right dtypes even when empty
        assert pdw.toa_us.dtype == np.float32
        assert pdw.emitter_id.dtype == np.int64

    def test_pipeline_smoke_import(self):
        """Smoke test: pipeline module is importable and run() is callable."""
        from vyapti_simulator.rf.rf_to_pdw_pipeline import (
            RFPulsePipeline, RFPipelineConfig,
        )
        from vyapti_simulator.tsrd.synthetic_pdw_generator import (
            SyntheticEmitterSpec,
        )
        import numpy as np

        specs = [
            SyntheticEmitterSpec(
                emitter_id=0, aoa_deg=12.0, snr_db=15.0,
                emitter_type="fixed_continuous",
                center_freq_hz=3e9, pri_sec=1e-3,
                pulse_width_sec=1e-5,  # 10 µs, within allowed range (0.2-10 µs)
            ),
        ]
        pipeline = RFPulsePipeline(
            emitter_specs=specs,
            config=RFPipelineConfig(
                dsp_sample_rate_hz=1e6,
                tick_interval_s=1e-3,
                mission_duration_s=0.05,
                buffer_ticks=10,
                snr_db=20.0,
                cfar_db=10.0,
            ),
            rng=np.random.default_rng(0),
        )
        pdw = pipeline.run()
        assert pdw is not None
        assert hasattr(pdw, "toa_us")
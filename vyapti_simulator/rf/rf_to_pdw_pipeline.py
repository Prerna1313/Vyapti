"""
vyapti_simulator.rf.rf_to_pdw_pipeline
=======================================

Orchestration: SyntheticEmitterSpec → RF Physics → I/Q → PDWStream.

This is the third of the three modules that wire the TSRD emitter
layer through the RF physics engine and back to a TSRD-shaped
``PDWStream``.

Pipeline
--------

    SyntheticEmitterSpec (list)
            │
            │  TSRDSpecToRFBridge.build() per spec
            ▼
    List[SimEmitterSpec]
            │
            │  RealTimeRFSimulator.add_emitter(...) per spec
            ▼
    RealTimeRFSimulator
            │
            │  sim.run() → list of I/Q buffers
            ▼
    np.concatenate(buffers)  # I/Q stream
            │
            │  PulseDetector.detect(...)
            ▼
    PDWStream  # TSRD-shaped, drops into the deinterleaver /
              # TSRDEnvironment without any further conversion

Usage
-----

::

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
            freq_list_hz=[5e9, 7e9, 9e9], pri_sec=2e-3,
            pulse_width_sec=0.5e-6,
        ),
    ]

    pipeline = RFPulsePipeline(
        emitter_specs=specs,
        config=RFPipelineConfig(
            dsp_sample_rate_hz=10e6,
            tick_interval_s=10e-3,
            mission_duration_s=30.0,
        ),
        rng=np.random.default_rng(42),
    )
    pdw_stream = pipeline.run()

    # pdw_stream is a PDWStream identical in shape to
    # SyntheticEWPDWGenerator output. Feed it to the deinterleaver:
    from vyapti_simulator.tsrd import quick_deinterleave
    result = quick_deinterleave(pdw_stream)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..tsrd.synthetic_pdw_generator import SyntheticEmitterSpec
from ..tsrd.tsrd_adapter import PDWStream
from .simulator_engine import (
    RealTimeRFSimulator,
    SimulationEngineConfig,
)
from .tsrd_bridge import TSRDSpecToRFBridge, SimEmitterSpec
from .pulse_detector import (
    PulseDetector,
    PulseDetectorConfig,
    EmitterInfo,
)


# =====================================================================
# Pipeline configuration
# =====================================================================
@dataclass
class RFPipelineConfig:
    """
    End-to-end pipeline configuration.

    DSP / physics
    -------------
    ``dsp_sample_rate_hz`` × ``tick_interval_s`` = samples per tick.
    The default 10 MHz × 10 ms = 100,000 samples per tick (40 MB
    of complex128). Lower sample rates speed up simulation but
    reduce the matched-filter resolution.

    Detection
    ---------
    ``cfar_db`` sets the cell-averaging CFAR threshold. Higher
    values reduce false alarms but require stronger pulses to be
    detected. The default 10 dB is appropriate for SNR ≥ 12 dB.

    Real-time
    ---------
    ``real_time=True`` makes ``sim.run()`` sleep to hit wall-clock
    time. Off by default — useful for batch experiments and CI.
    """
    dsp_sample_rate_hz: float = 10e6
    tick_interval_s: float = 10e-3
    mission_duration_s: float = 30.0
    buffer_ticks: int = 10
    # Per-emitter global SNR; can be overridden per spec by the
    # spec's snr_db field. If both are given, spec wins.
    snr_db: float = 20.0
    pulse_power_w: float = 1.0
    # Waveform
    chirp_bandwidth_hz: float = 1e6
    waveform_type: str = "lfm"
    # Detection
    cfar_db: float = 10.0
    cfar_train_cells: int = 20
    cfar_guard_cells: int = 4
    min_pulse_samples: int = 5
    noise_floor_dbm: float = -130.0
    # Streaming
    real_time: bool = False

    @property
    def num_ticks(self) -> int:
        return int(np.ceil(self.mission_duration_s / self.tick_interval_s))


# =====================================================================
# Pipeline
# =====================================================================
class RFPulsePipeline:
    """
    End-to-end pipeline: TSRD emitter specs → RF physics → PDW stream.

    The pipeline owns the RF engine, the bridges, and the detector.
    ``run()`` is idempotent: calling it multiple times yields the
    same ``PDWStream`` (provided the same RNG is used).

    Parameters
    ----------
    emitter_specs : List[SyntheticEmitterSpec]
        Input emitter configurations. Each must have either
        ``center_freq_hz`` or ``freq_list_hz`` set so a carrier
        can be resolved.
    config : RFPipelineConfig
        Pipeline-level configuration.
    rng : np.random.Generator
        Root RNG. Per-spec child streams are spawned via
        ``np.random.SeedSequence`` so each spec sees an
        independent deterministic stream.
    """

    def __init__(
        self,
        emitter_specs: List[SyntheticEmitterSpec],
        config: RFPipelineConfig,
        rng: np.random.Generator,
    ):
        self.emitter_specs = list(emitter_specs)
        self.config = config
        self.rng = rng

        # Validate specs (mirror SyntheticEWPDWGenerator's checks)
        seen_ids = set()
        for s in self.emitter_specs:
            if s.emitter_id in seen_ids:
                raise ValueError(
                    f"Duplicate emitter_id {s.emitter_id} in specs"
                )
            seen_ids.add(s.emitter_id)
            if s.center_freq_hz is None and not s.freq_list_hz:
                raise ValueError(
                    f"emitter_id={s.emitter_id}: need center_freq_hz "
                    f"or freq_list_hz"
                )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self) -> PDWStream:
        """
        Build bridges, run the engine, detect pulses, return PDWStream.

        For multiple emitters at different carriers, the detector is
        run once per emitter (with the appropriate carrier) and the
        per-emitter detections are merged. This ensures each emitter's
        matched-filter reference matches its actual chirp centre
        frequency, so all emitters are detected.

        Returns
        -------
        PDWStream
            TSRD-shaped PDW stream with fields:
            ``toa_us``, ``freq_mhz``, ``pw_us``, ``aoa_deg``,
            ``amp_db``, ``emitter_id``. Sorted by ToA.
            Empty stream (zero-length arrays) when no pulses
            were detected.
        """
        cfg = self.config

        # 1. Build per-spec RF configurations via the bridge
        sim_specs, emitter_map, _ = self._build_sim_specs()

        if not sim_specs:
            # No emitters — return empty PDWStream
            return PDWStream(
                toa_us=np.zeros(0, dtype=np.float32),
                freq_mhz=np.zeros(0, dtype=np.float32),
                pw_us=np.zeros(0, dtype=np.float32),
                aoa_deg=np.zeros(0, dtype=np.float32),
                amp_db=np.zeros(0, dtype=np.float32),
                emitter_id=np.zeros(0, dtype=np.int64),
            )

        # 2. Build the engine
        engine_cfg = SimulationEngineConfig(
            tick_interval_s=cfg.tick_interval_s,
            num_ticks=cfg.num_ticks,
            dsp_sample_rate_hz=cfg.dsp_sample_rate_hz,
            buffer_ticks=cfg.buffer_ticks,
            snr_db=cfg.snr_db,
            pulse_power_w=cfg.pulse_power_w,
            real_time=cfg.real_time,
        )
        sim = RealTimeRFSimulator(engine_cfg, rng=self.rng)
        for sspec in sim_specs:
            sim.add_emitter(
                sspec.kinematic,
                sspec.waveform_fn,
                emitter_id=sspec.emitter_id,
                channel=sspec.channel,
                pri_sec=sspec.pri_sec,
                pulse_width_s=sspec.pulse_width_s,
            )

        # 3. Run the engine and concatenate I/Q
        buffers = list(sim.run())
        if not buffers:
            return PDWStream(
                toa_us=np.zeros(0, dtype=np.float32),
                freq_mhz=np.zeros(0, dtype=np.float32),
                pw_us=np.zeros(0, dtype=np.float32),
                aoa_deg=np.zeros(0, dtype=np.float32),
                amp_db=np.zeros(0, dtype=np.float32),
                emitter_id=np.zeros(0, dtype=np.int64),
            )
        iq_buffer = np.concatenate(buffers)

        # 4. Detect pulses per-emitter (one matched-filter reference
        # per carrier) and merge the results.
        all_pdw = self._detect_per_emitter(iq_buffer, sim_specs, emitter_map)
        return all_pdw

    # ------------------------------------------------------------------
    # Per-emitter detection
    # ------------------------------------------------------------------
    def _detect_per_emitter(
        self,
        iq_buffer: np.ndarray,
        sim_specs: List[SimEmitterSpec],
        emitter_map: Dict[int, EmitterInfo],
    ) -> PDWStream:
        """
        Run the detector once per emitter, then merge per-emitter
        detections into a single sorted PDWStream.

        This is required when emitters are at different carrier
        frequencies — the detector's matched filter is carrier-specific
        and would miss emitters whose frequency is far from the
        configured reference.
        """
        cfg = self.config
        per_emitter_pdws: List[PDWStream] = []
        for sspec, input_spec in zip(sim_specs, self.emitter_specs):
            detector_cfg = self._build_detector_config(
                sspec, input_spec
            )
            # Per-emitter emitter_map: only this emitter
            single_emitter_map = {
                sspec.emitter_id: emitter_map[sspec.emitter_id]
            }
            detector = PulseDetector(
                config=detector_cfg,
                emitter_map=single_emitter_map,
                rng=self.rng,
            )
            per_emitter_pdws.append(detector.detect(iq_buffer))

        if not per_emitter_pdws:
            return PDWStream(
                toa_us=np.zeros(0, dtype=np.float32),
                freq_mhz=np.zeros(0, dtype=np.float32),
                pw_us=np.zeros(0, dtype=np.float32),
                aoa_deg=np.zeros(0, dtype=np.float32),
                amp_db=np.zeros(0, dtype=np.float32),
                emitter_id=np.zeros(0, dtype=np.int64),
            )

        # Concatenate
        if len(per_emitter_pdws) == 1:
            return per_emitter_pdws[0]

        toa = np.concatenate([p.toa_us for p in per_emitter_pdws])
        freq = np.concatenate([p.freq_mhz for p in per_emitter_pdws])
        pw = np.concatenate([p.pw_us for p in per_emitter_pdws])
        aoa = np.concatenate([p.aoa_deg for p in per_emitter_pdws])
        amp = np.concatenate([p.amp_db for p in per_emitter_pdws])
        eid = np.concatenate([p.emitter_id for p in per_emitter_pdws])

        # Sort by ToA
        order = np.argsort(toa.astype(np.float64))
        return PDWStream(
            toa_us=toa[order].astype(np.float32),
            freq_mhz=freq[order].astype(np.float32),
            pw_us=pw[order].astype(np.float32),
            aoa_deg=aoa[order].astype(np.float32),
            amp_db=amp[order].astype(np.float32),
            emitter_id=eid[order],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_sim_specs(
        self,
    ) -> tuple[List[SimEmitterSpec], Dict[int, EmitterInfo], List[SyntheticEmitterSpec]]:
        """
        Build one ``SimEmitterSpec`` per input spec.

        Returns
        -------
        sim_specs : List[SimEmitterSpec]
            RF engine configurations, one per input spec.
        emitter_map : Dict[int, EmitterInfo]
            Per-emitter AoA and carrier frequency for the detector.
        specs : List[SyntheticEmitterSpec]
            The original specs (echoed for callers).
        """
        # Spawn one RNG per spec for deterministic per-emitter
        # independence. We do this from a SeedSequence so the
        # pipeline-level RNG state advances deterministically.
        ss_root = np.random.SeedSequence(self.rng.integers(0, 2 ** 32 - 1))
        children = ss_root.spawn(len(self.emitter_specs))

        sim_specs: List[SimEmitterSpec] = []
        emitter_map: Dict[int, EmitterInfo] = {}
        for spec, child in zip(self.emitter_specs, children):
            child_rng = np.random.default_rng(child)
            bridge = TSRDSpecToRFBridge(spec, child_rng)
            sspec = bridge.build()
            sim_specs.append(sspec)
            carrier_hz = self._resolve_carrier(spec)
            emitter_map[sspec.emitter_id] = EmitterInfo(
                carrier_freq_hz=carrier_hz,
                aoa_deg=float(sspec.aoa_deg),
            )
        return sim_specs, emitter_map, self.emitter_specs

    def _build_detector_config(
        self,
        sspec: SimEmitterSpec,
        input_spec: SyntheticEmitterSpec,
    ) -> PulseDetectorConfig:
        cfg = self.config
        # Carrier frequency from the bridge output (handles agile freq_list)
        carrier_hz = float(sspec.carrier_freq_hz)
        return PulseDetectorConfig(
            dsp_sample_rate_hz=cfg.dsp_sample_rate_hz,
            tick_interval_s=cfg.tick_interval_s,
            noise_floor_dbm=cfg.noise_floor_dbm,
            cfar_db=cfg.cfar_db,
            cfar_train_cells=cfg.cfar_train_cells,
            cfar_guard_cells=cfg.cfar_guard_cells,
            min_pulse_samples=cfg.min_pulse_samples,
            waveform_type=cfg.waveform_type,
            chirp_bandwidth_hz=float(sspec.chirp_bandwidth_hz),
            carrier_freq_hz=carrier_hz,
            pulse_width_s=float(sspec.pulse_width_s),
        )

    @staticmethod
    def _resolve_carrier(spec: SyntheticEmitterSpec) -> float:
        if spec.center_freq_hz is not None:
            return float(spec.center_freq_hz)
        if spec.freq_list_hz:
            return float(spec.freq_list_hz[0])
        raise ValueError(
            f"emitter_id={spec.emitter_id}: cannot resolve carrier"
        )


__all__ = ["RFPulsePipeline", "RFPipelineConfig"]

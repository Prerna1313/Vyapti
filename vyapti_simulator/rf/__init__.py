"""
vyapti_simulator.rf
====================

RF physics simulation modules for the Vyapti EW simulator.

Submodules
-----------
``waveforms``
    LFM chirp, PSK, QAM generators; AWGN; matched filter;
    baseband ↔ RF conversion.
``propagation``
    Kinematic emitters; free-space path loss; Rayleigh, Rician,
    and multipath fading channels.
``amplifiers``
    Rapp and Saleh amplifier non-linearity models.
``simulator_engine``
    Fixed-interval real-time RF simulation engine with ping-pong
    dual-buffer streaming.
``receiver_impairments``
    Phase noise, I/Q imbalance, DC offset, ADC quantisation.
``swerling``
    Swerling target fluctuation models (RCS scintillation).
``tsrd_bridge``
    Bridge from ``SyntheticEmitterSpec`` to ``SimEmitterSpec`` for
    the RF physics engine.
``pulse_detector``
    CFAR detection and parameter extraction on I/Q buffers →
    TSRD-shaped ``PDWStream``.
``rf_to_pdw_pipeline``
    End-to-end orchestration: specs → RF engine → I/Q → PDW stream.

Public API
-----------
The primary user-facing entry point is
:class:`rf_to_pdw_pipeline.RFPulsePipeline`. Individual components
are importable directly from their submodules.
"""

from .simulator_engine import (
    RealTimeRFSimulator,
    SimulationEngineConfig,
    SimEmitter,
    simulate_offline,
)
from .waveforms import (
    generate_lfm_chirp,
    generate_psk_symbols,
    generate_qam_symbols,
    add_awgn,
    estimate_snr,
    matched_filter,
    coherent_integrate,
    complex_baseband_to_rf,
    rf_to_complex_baseband,
)
from .propagation import (
    KinematicEmitter,
    FreeSpacePathLoss,
    compute_path_loss,
    compute_doppler_shift,
    RayleighFadingChannel,
    RicianFadingChannel,
    MultipathChannel,
)
from .amplifiers import (
    RappAmplifier,
    SalehAmplifier,
)
from .receiver_impairments import (
    apply_receiver_impairments,
)
from .swerling import (
    apply_swerling_fluctuation,
)
from .tsrd_bridge import (
    TSRDSpecToRFBridge,
    SimEmitterSpec,
)
from .pulse_detector import (
    PulseDetector,
    PulseDetectorConfig,
    DetectedPulse,
    EmitterInfo,
)
from .rf_to_pdw_pipeline import (
    RFPulsePipeline,
    RFPipelineConfig,
)
from .closed_loop import (
    Dwell,
    MissionState,
    TrackedEmitter,
    SchedulerScore,
    BaseScheduler,
    RoundRobinScheduler,
    PriorityQueueScheduler,
    ThreatScoreScheduler,
    ThreatWeights,
    MissionRunner,
    score_scheduler,
    run_comparison,
    summarise_results,
)


__all__ = [
    # Engine
    "RealTimeRFSimulator",
    "SimulationEngineConfig",
    "SimEmitter",
    "simulate_offline",
    # Waveforms
    "generate_lfm_chirp",
    "generate_psk_symbols",
    "generate_qam_symbols",
    "add_awgn",
    "estimate_snr",
    "matched_filter",
    "coherent_integrate",
    "complex_baseband_to_rf",
    "rf_to_complex_baseband",
    # Propagation
    "KinematicEmitter",
    "FreeSpacePathLoss",
    "compute_path_loss",
    "compute_doppler_shift",
    "RayleighFadingChannel",
    "RicianFadingChannel",
    "MultipathChannel",
    # Amplifiers
    "RappAmplifier",
    "SalehAmplifier",
    # Receiver impairments
    "apply_receiver_impairments",
    # Swerling
    "apply_swerling_fluctuation",
    # Bridge
    "TSRDSpecToRFBridge",
    "SimEmitterSpec",
    # Detector
    "PulseDetector",
    "PulseDetectorConfig",
    "DetectedPulse",
    "EmitterInfo",
    # Pipeline
    "RFPulsePipeline",
    "RFPipelineConfig",
    # Closed-loop
    "Dwell",
    "MissionState",
    "TrackedEmitter",
    "SchedulerScore",
    "BaseScheduler",
    "RoundRobinScheduler",
    "PriorityQueueScheduler",
    "ThreatScoreScheduler",
    "ThreatWeights",
    "MissionRunner",
    "score_scheduler",
    "run_comparison",
    "summarise_results",
]

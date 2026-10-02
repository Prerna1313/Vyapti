"""System C: synthetic RF and IQ physics. Public exports load only when requested."""
from importlib import import_module as _import_module

_EXPORTS = {
    'RealTimeRFSimulator': ('.simulator_engine', 'RealTimeRFSimulator'),
    'SimulationEngineConfig': ('.simulator_engine', 'SimulationEngineConfig'),
    'SimEmitter': ('.simulator_engine', 'SimEmitter'),
    'simulate_offline': ('.simulator_engine', 'simulate_offline'),
    'generate_lfm_chirp': ('.waveforms', 'generate_lfm_chirp'),
    'generate_psk_symbols': ('.waveforms', 'generate_psk_symbols'),
    'generate_qam_symbols': ('.waveforms', 'generate_qam_symbols'),
    'add_awgn': ('.waveforms', 'add_awgn'),
    'estimate_snr': ('.waveforms', 'estimate_snr'),
    'matched_filter': ('.waveforms', 'matched_filter'),
    'coherent_integrate': ('.waveforms', 'coherent_integrate'),
    'complex_baseband_to_rf': ('.waveforms', 'complex_baseband_to_rf'),
    'rf_to_complex_baseband': ('.waveforms', 'rf_to_complex_baseband'),
    'KinematicEmitter': ('.propagation', 'KinematicEmitter'),
    'FreeSpacePathLoss': ('.propagation', 'FreeSpacePathLoss'),
    'compute_path_loss': ('.propagation', 'compute_path_loss'),
    'compute_doppler_shift': ('.propagation', 'compute_doppler_shift'),
    'RayleighFadingChannel': ('.propagation', 'RayleighFadingChannel'),
    'RicianFadingChannel': ('.propagation', 'RicianFadingChannel'),
    'MultipathChannel': ('.propagation', 'MultipathChannel'),
    'RappAmplifier': ('.amplifiers', 'RappAmplifier'),
    'SalehAmplifier': ('.amplifiers', 'SalehAmplifier'),
    'apply_receiver_impairments': ('.receiver_impairments', 'apply_receiver_impairments'),
    'apply_swerling_fluctuation': ('.swerling', 'apply_swerling_fluctuation'),
    'TSRDSpecToRFBridge': ('.tsrd_bridge', 'TSRDSpecToRFBridge'),
    'SimEmitterSpec': ('.tsrd_bridge', 'SimEmitterSpec'),
    'PulseDetector': ('.pulse_detector', 'PulseDetector'),
    'PulseDetectorConfig': ('.pulse_detector', 'PulseDetectorConfig'),
    'DetectedPulse': ('.pulse_detector', 'DetectedPulse'),
    'EmitterInfo': ('.pulse_detector', 'EmitterInfo'),
    'RFPulsePipeline': ('.rf_to_pdw_pipeline', 'RFPulsePipeline'),
    'RFPipelineConfig': ('.rf_to_pdw_pipeline', 'RFPipelineConfig'),
    'Dwell': ('.closed_loop', 'Dwell'),
    'MissionState': ('.closed_loop', 'MissionState'),
    'TrackedEmitter': ('.closed_loop', 'TrackedEmitter'),
    'SchedulerScore': ('.closed_loop', 'SchedulerScore'),
    'BaseScheduler': ('.closed_loop', 'BaseScheduler'),
    'RoundRobinScheduler': ('.closed_loop', 'RoundRobinScheduler'),
    'PriorityQueueScheduler': ('.closed_loop', 'PriorityQueueScheduler'),
    'ThreatScoreScheduler': ('.closed_loop', 'ThreatScoreScheduler'),
    'ThreatWeights': ('.closed_loop', 'ThreatWeights'),
    'MissionRunner': ('.closed_loop', 'MissionRunner'),
    'score_scheduler': ('.closed_loop', 'score_scheduler'),
    'run_comparison': ('.closed_loop', 'run_comparison'),
    'summarise_results': ('.closed_loop', 'summarise_results'),
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(_import_module(module, __name__), attribute)
    globals()[name] = value
    return value

def __dir__():
    return sorted(set(globals()) | set(__all__))

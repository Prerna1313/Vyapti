"""System B: recorded TSRD scan and stare data. Public exports load only when requested."""
from importlib import import_module as _import_module

_EXPORTS = {
    'TSRDAdapter': ('.tsrd_adapter', 'TSRDAdapter'),
    'TSRDDataMode': ('.tsrd_adapter', 'TSRDDataMode'),
    'TSRDReceiverMode': ('.tsrd_adapter', 'TSRDReceiverMode'),
    'PDWStream': ('.tsrd_adapter', 'PDWStream'),
    'PDW_STREAM_FIELDS': ('.tsrd_adapter', 'PDW_STREAM_FIELDS'),
    'EXPECTED_H5_FEATURE_NAMES': ('.tsrd_adapter', 'EXPECTED_H5_FEATURE_NAMES'),
    'DataUnavailableError': ('.tsrd_adapter', 'DataUnavailableError'),
    'DataIntegrityError': ('.tsrd_adapter', 'DataIntegrityError'),
    'StareModeOracleError': ('.tsrd_adapter', 'StareModeOracleError'),
    'UnknownTSRDFieldError': ('.tsrd_adapter', 'UnknownTSRDFieldError'),
    'InsufficientDataError': ('.tsrd_adapter', 'InsufficientDataError'),
    'UnsupportedTSRDSchemaError': ('.tsrd_adapter', 'UnsupportedTSRDSchemaError'),
    'ScanPolicyOracle': ('.scan_policy_oracle', 'ScanPolicyOracle'),
    'DefaultScanPolicyOracle': ('.scan_policy_oracle', 'DefaultScanPolicyOracle'),
    'ScanPolicy': ('.scan_policy_oracle', 'ScanPolicy'),
    'DwellWindow': ('.scan_policy_oracle', 'DwellWindow'),
    'OracleResult': ('.scan_policy_oracle', 'OracleResult'),
    'evaluate_multiple_policies': ('.scan_policy_oracle', 'evaluate_multiple_policies'),
    'find_pareto_optimal_policies': ('.scan_policy_oracle', 'find_pareto_optimal_policies'),
    'build_uniform_scan_policy': ('.scan_policy_oracle', 'build_uniform_scan_policy'),
    'build_stare_policy': ('.scan_policy_oracle', 'build_stare_policy'),
    'build_adaptive_dwell_policy': ('.scan_policy_oracle', 'build_adaptive_dwell_policy'),
    'TSRDEmitterSampler': ('.tsrd_emitter', 'TSRDEmitterSampler'),
    'FREQ_MODE_TO_BEHAVIOR': ('.emitter_modes', 'FREQ_MODE_TO_BEHAVIOR'),
    'CorpusFileDisposition': ('.corpus_loader', 'CorpusFileDisposition'),
    'CorpusFileResult': ('.corpus_loader', 'CorpusFileResult'),
    'CorpusSummary': ('.corpus_loader', 'CorpusSummary'),
    'CorpusUnavailableError': ('.corpus_loader', 'CorpusUnavailableError'),
    'TSRDCorpusLoader': ('.corpus_loader', 'TSRDCorpusLoader'),
    'BandSlotCell': ('.pdw_discretiser', 'BandSlotCell'),
    'DiscretisedGrid': ('.pdw_discretiser', 'DiscretisedGrid'),
    'discretise_pdw_to_grid': ('.pdw_discretiser', 'discretise_pdw_to_grid'),
    'discretize_pdw_to_bands': ('.pdw_discretiser', 'discretize_pdw_to_bands'),
    'iter_pulses_in_grid': ('.pdw_discretiser', 'iter_pulses_in_grid'),
    'aggregate_cell_pulses': ('.pdw_discretiser', 'aggregate_cell_pulses'),
    'DeinterleaverConfig': ('.deinterleaver', 'DeinterleaverConfig'),
    'EmitterTrack': ('.deinterleaver', 'EmitterTrack'),
    'DeinterleaverResult': ('.deinterleaver', 'DeinterleaverResult'),
    'FeatureBasedDeinterleaver': ('.deinterleaver', 'FeatureBasedDeinterleaver'),
    'PRIBasedDeinterleaver': ('.deinterleaver', 'PRIBasedDeinterleaver'),
    'quick_deinterleave': ('.deinterleaver', 'quick_deinterleave'),
    'DetectionConfig': ('.tsrd_environment', 'DetectionConfig'),
    'ShnidmanDetectionConfig': ('.tsrd_environment', 'ShnidmanDetectionConfig'),
    'TSRDEnvironment': ('.tsrd_environment', 'TSRDEnvironment'),
    'build_tsrd_environment': ('.tsrd_environment', 'build_tsrd_environment'),
    'write_scenario_to_h5': ('.h5_writer', 'write_scenario_to_h5'),
    'H5ScenarioConfig': ('.h5_writer', 'H5ScenarioConfig'),
    'uniform_antenna_gain': ('.antenna_patterns', 'uniform_antenna_gain'),
    'sectorised_antenna_gain': ('.antenna_patterns', 'sectorised_antenna_gain'),
    'realistic_antenna_gain': ('.antenna_patterns', 'realistic_antenna_gain'),
    'uniform_sectorised_antenna_gain': ('.antenna_patterns', 'uniform_sectorised_antenna_gain'),
    'bridge_local_emitters_to_grid': ('.local_emitter_bridge', 'bridge_local_emitters_to_grid'),
    'local_emitters_to_pdw_stream': ('.local_emitter_bridge', 'local_emitters_to_pdw_stream'),
    'BridgeResult': ('.local_emitter_bridge', 'BridgeResult'),
    'DEFAULT_NOISE_FLOOR_DBM': ('.local_emitter_bridge', 'DEFAULT_NOISE_FLOOR_DBM'),
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

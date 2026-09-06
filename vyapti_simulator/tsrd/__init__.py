"""PS26055 TSRD package.

Public surfaces:

  * `TSRDAdapter` -- read a single TSRD H5 file (real or
    fixture) and expose its 5-field PDW stream and (Scan
    Mode only) its observed-occupancy grid. Fail-closed on
    missing files; no synthetic fallback.
  * `PDWStream` -- the canonical 6-field PDW stream
    (toa_us, freq_mhz, pw_us, aoa_deg, amp_db, emitter_id).
  * `ScanPolicyOracle` / `DefaultScanPolicyOracle` -- abstract
    interface for the deferred Stare-Mode counterfactual
    oracle, plus a default stub whose `evaluate()` raises
    `NotImplementedError`.
  * `TSRDEmitterSampler` -- the existing Option-A sampler
    that turns the aggregate `tsrd_statistics.json` into
    `EmitterConfig` objects. Preserved unchanged.
  * `TSRDCorpusLoader` -- iterate a directory of full-corpus
    TSRD H5 files and produce per-file `PDWStream` /
    `EmitterConfig` records. Stage-2 deliverable; no
    fixture or synthetic fallback.
  * `FREQ_MODE_TO_BEHAVIOR` -- the routing table from TSRD's
    six frequency modes onto six of the thirteen existing
    `EmitterBehaviorType` enum members. Preserved unchanged.

NEW in Option B (PDW-to-band/slot discretisation):

  * `BandSlotCell` -- the aggregated pulse data for one
    (band, slot) cell on the simulation grid.
  * `DiscretisedGrid` -- the complete discretised grid
    covering all (band, slot) cells of one PDW stream.
  * `discretise_pdw_to_grid` -- the Option B discretiser.
  * `iter_pulses_in_grid` -- memory-efficient cell iterator.

NEW in Option C (Deinterleaver):

  * `DeinterleaverConfig` -- hyperparameters for the
    PRI-based deinterleaver.
  * `EmitterTrack` -- one deinterleaved emitter track.
  * `DeinterleaverResult` -- the output of one pass.
  * `FeatureBasedDeinterleaver` -- AoA-first three-stage
    deinterleaver: AoA cluster → RF/PW refinement →
    PRI validation. `PRIBasedDeinterleaver` is a deprecated
    alias pointing here.
  * `quick_deinterleave` -- one-shot functional entry.

NEW: TSRD environment (Options B + C wired for the scheduler):

  * `DetectionConfig` -- amplitude → SNR → Pd mapping.
  * `TSRDEnvironment` -- the scheduler-facing environment
    that uses the discretised grid and the deinterleaver.
    Drop-in replacement for `PS26055Environment` on real
    TSRD data.
  * `build_tsrd_environment` -- factory helper for the
    Kaggle runner.

Errors:

  * `DataUnavailableError` -- a real TSRD file was requested
    but is missing. Hard failure; no fallback.
  * `DataIntegrityError` -- H5 SHA-256 pin mismatch.
  * `StareModeOracleError` -- code tried to discretise Stare
    Mode into a level-1 grid. Stare Mode is the oracle.
  * `UnknownTSRDFieldError` -- a TSRD field value is not in
    the routing table (e.g. an unknown `freq_mode`).
  * `InsufficientDataError` -- an `EmitterConfig` field
    cannot be derived from the H5 and was not supplied by
    the caller.
  * `UnsupportedTSRDSchemaError` -- the H5's schema differs
    from what the adapter expects.
"""

from .tsrd_adapter import (
    TSRDAdapter,
    TSRDDataMode,
    TSRDReceiverMode,
    PDWStream,
    PDW_STREAM_FIELDS,
    EXPECTED_H5_FEATURE_NAMES,
    DataUnavailableError,
    DataIntegrityError,
    StareModeOracleError,
    UnknownTSRDFieldError,
    InsufficientDataError,
    UnsupportedTSRDSchemaError,
)
from .scan_policy_oracle import (
    ScanPolicyOracle,
    DefaultScanPolicyOracle,
    ScanPolicy,
    DwellWindow,
    OracleResult,
)
from .tsrd_emitter import TSRDEmitterSampler, FREQ_MODE_TO_BEHAVIOR
from .corpus_loader import (
    CorpusFileDisposition,
    CorpusFileResult,
    CorpusSummary,
    CorpusUnavailableError,
    TSRDCorpusLoader,
)

# Option B — PDW discretisation
#
# NOTE: ``BandSlotPulse`` is no longer re-exported from this module.
# It is the unified dataclass from ``src.observation_interface`` and
# is imported from there directly:
#     from src.observation_interface import BandSlotPulse
from .pdw_discretiser import (
    BandSlotCell,
    DiscretisedGrid,
    discretise_pdw_to_grid,
    discretize_pdw_to_bands,
    iter_pulses_in_grid,
    aggregate_cell_pulses,
)

# Option C — Deinterleaver
from .deinterleaver import (
    DeinterleaverConfig,
    EmitterTrack,
    DeinterleaverResult,
    FeatureBasedDeinterleaver,
    PRIBasedDeinterleaver,  # deprecated alias
    quick_deinterleave,
)

# TSRD environment (B + C wired)
from .tsrd_environment import (
    DetectionConfig,
    TSRDEnvironment,
    build_tsrd_environment,
)

# Synthetic EW PDW generator (for Kaggle paths without TSRD)
from .synthetic_pdw_generator import (
    SyntheticEmitterSpec,
    SyntheticEWPDWGenerator,
    default_two_emitter_scenario,
    default_six_emitter_scenario,
)

# Bridge: src/emitter_models → TSRD band/slot grid (System A/B unification)
from .local_emitter_bridge import (
    bridge_local_emitters_to_grid,
    local_emitters_to_pdw_stream,
    BridgeResult,
    DEFAULT_NOISE_FLOOR_DBM,
)

__all__ = [
    # Adapter
    "TSRDAdapter",
    "TSRDDataMode",
    "TSRDReceiverMode",
    "PDWStream",
    "PDW_STREAM_FIELDS",
    "EXPECTED_H5_FEATURE_NAMES",
    "DataUnavailableError",
    "DataIntegrityError",
    "StareModeOracleError",
    "UnknownTSRDFieldError",
    "InsufficientDataError",
    "UnsupportedTSRDSchemaError",
    # Scan policy oracle
    "ScanPolicyOracle",
    "DefaultScanPolicyOracle",
    "ScanPolicy",
    "DwellWindow",
    "OracleResult",
    # Option-A sampler (preserved)
    "TSRDEmitterSampler",
    "FREQ_MODE_TO_BEHAVIOR",
    # Full-corpus loader (Stage 2)
    "TSRDCorpusLoader",
    "CorpusFileDisposition",
    "CorpusFileResult",
    "CorpusSummary",
    "CorpusUnavailableError",
    # Option B — PDW discretisation
    # BandSlotPulse is from src.observation_interface, not re-exported here
    "BandSlotCell",
    "DiscretisedGrid",
    "discretise_pdw_to_grid",
    "discretize_pdw_to_bands",
    "iter_pulses_in_grid",
    "aggregate_cell_pulses",
    # Option C — Deinterleaver
    "DeinterleaverConfig",
    "EmitterTrack",
    "DeinterleaverResult",
    "FeatureBasedDeinterleaver",
    "PRIBasedDeinterleaver",  # deprecated alias
    "quick_deinterleave",
    # TSRD environment (B + C wired)
    "DetectionConfig",
    "TSRDEnvironment",
    "build_tsrd_environment",
    # Synthetic EW PDW generator
    "SyntheticEmitterSpec",
    "SyntheticEWPDWGenerator",
    "default_two_emitter_scenario",
    "default_six_emitter_scenario",
    # System A/B unification bridge
    "bridge_local_emitters_to_grid",
    "local_emitters_to_pdw_stream",
    "BridgeResult",
    "DEFAULT_NOISE_FLOOR_DBM",
]

"""Compatibility import for new training scripts; implementation lives in the package."""

from vyapti_simulator.system_b.tsrd.train250_cache import (
    CachedTSRDTrain250Pool,
    build_stare_evaluation_world,
    build_train_pool_from_cache,
    load_train250_cache,
    sample_fresh_training_world,
    write_runtime_manifest,
)

__all__ = [
    "CachedTSRDTrain250Pool",
    "build_stare_evaluation_world",
    "build_train_pool_from_cache",
    "load_train250_cache",
    "sample_fresh_training_world",
    "write_runtime_manifest",
]

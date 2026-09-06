"""PS26055 Core — environment, receiver, scheduler contract, metrics, episode runner."""

from .environment import (
    ProvenanceTag,
    ProvenanceLabel,
    EmitterBehaviorType,
    HELD_OUT_BEHAVIORS,
    EmitterConfig,
    HiddenTruthGrid,
    SimulationConfig,
    PS26055Environment,
)
from .mapping import (
    frequency_to_band,
    seconds_to_slot,
    band_edges_mhz,
    slot_edges_seconds,
    band_center_frequency_mhz,
    slot_start_seconds,
)
from .receiver import ReceiverPhysicsConfig, ReceiverModel
from .scheduler_interface import (
    SchedulerInterface,
    BaseScheduler,
    BandPrediction,
    PERMITTED_OBSERVATION_KEYS,
    FORBIDDEN_OBSERVATION_KEYS,
)
from .metrics import MetricsConfig, MetricsEngine, TrajectoryStep
from .episode import (
    EpisodeResult,
    run_episode,
    run_paired_episodes,
    scenario_descriptor,
    PERMITTED_SCENARIO_KEYS,
)

__all__ = [
    # provenance
    "ProvenanceTag",
    "ProvenanceLabel",
    # environment
    "EmitterBehaviorType",
    "HELD_OUT_BEHAVIORS",
    "EmitterConfig",
    "HiddenTruthGrid",
    "SimulationConfig",
    "PS26055Environment",
    # band/time mapping (Invariant 2)
    "frequency_to_band",
    "seconds_to_slot",
    "band_edges_mhz",
    "slot_edges_seconds",
    "band_center_frequency_mhz",
    "slot_start_seconds",
    # receiver
    "ReceiverPhysicsConfig",
    "ReceiverModel",
    # scheduler contract
    "SchedulerInterface",
    "BaseScheduler",
    "BandPrediction",
    "PERMITTED_OBSERVATION_KEYS",
    "FORBIDDEN_OBSERVATION_KEYS",
    # metrics
    "MetricsConfig",
    "MetricsEngine",
    "TrajectoryStep",
    # episode execution
    "EpisodeResult",
    "run_episode",
    "run_paired_episodes",
    "scenario_descriptor",
    "PERMITTED_SCENARIO_KEYS",
]

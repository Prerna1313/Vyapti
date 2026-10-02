"""TSRD metadata mapping, independent of synthetic emitter implementations."""
from vyapti_simulator.core.environment import EmitterBehaviorType

FREQ_MODE_TO_BEHAVIOR = {
    "FixedSingle": EmitterBehaviorType.CONTINUOUS_FIXED,
    "FixedMultiSimultaneous": EmitterBehaviorType.CONTINUOUS_FIXED,
    "HoppingLinear": EmitterBehaviorType.PATTERNED,
    "HoppingSawtooth": EmitterBehaviorType.PATTERNED,
    "RandomFixed": EmitterBehaviorType.PSEUDO_RANDOM_AGILE,
    "RandomRange": EmitterBehaviorType.RANDOM_HOPPER,
}

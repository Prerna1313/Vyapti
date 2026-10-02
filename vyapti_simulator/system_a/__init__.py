"""System A: synthetic pulse-description-word generation."""

from .synthetic_pdw import (
    SyntheticEWPDWGenerator,
    SyntheticEmitterSpec,
    default_two_emitter_scenario,
    default_six_emitter_scenario,
)

__all__ = [
    "SyntheticEWPDWGenerator",
    "SyntheticEmitterSpec",
    "default_two_emitter_scenario",
    "default_six_emitter_scenario",
]

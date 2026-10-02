"""Synthetic PDW generation for controlled development and examples."""

from .generator import (
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

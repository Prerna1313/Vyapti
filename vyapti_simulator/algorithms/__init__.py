"""
PS26055 Algorithms — scheduler policies.

Import convention: import concrete schedulers from their submodules, e.g.
    from vyapti_simulator.algorithms.bandit.ucb import UCB1Scheduler
    from vyapti_simulator.algorithms.bandit.thompson import SlidingWindowThompsonSampling
    from vyapti_simulator.qualification.probes import RoundRobinProbe   # baseline

This top-level package intentionally re-exports lazily via __getattr__ so that
importing `vyapti_simulator.algorithms` never forces every submodule (and its
numpy/scipy cost) to load, and so a partially-implemented submodule cannot break
the whole package import during development.

NOTE 2026-09-05: The `baselines/` subpackage was deleted. The Round-Robin
and Random-Markov baselines are now in `vyapti_simulator.qualification.probes`
(RoundRobinProbe, UniformRandomProbe). Clarkson's optimized policy is owned
by the algorithm team — do not re-introduce `baselines/clarkson.py` here.
"""

from importlib import import_module
from typing import Any

# Public name -> (submodule, attribute)
_REGISTRY = {
    # bandits
    "UCB1Scheduler": ("vyapti_simulator.algorithms.bandit.ucb", "UCB1Scheduler"),
    "SlidingWindowThompsonSampling": ("vyapti_simulator.algorithms.bandit.thompson", "SlidingWindowThompsonSampling"),
    "DiscountedThompsonSampling": ("vyapti_simulator.algorithms.bandit.thompson", "DiscountedThompsonSampling"),
    "ESPEScheduler": ("vyapti_simulator.algorithms.bandit.espe", "ESPEScheduler"),
    "GaussianESPEScheduler": ("vyapti_simulator.algorithms.bandit.espe", "GaussianESPEScheduler"),
    "RisingBanditScheduler": ("vyapti_simulator.algorithms.bandit.rising", "RisingBanditScheduler"),
    # primary contribution
    "CoverageConstrainedScheduler": ("vyapti_simulator.algorithms.bandit.coverage_constrained", "CoverageConstrainedScheduler"),
}

__all__ = list(_REGISTRY.keys())


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute loading
    if name in _REGISTRY:
        module_path, attr = _REGISTRY[name]
        module = import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def available_schedulers() -> dict:
    """Registry of public scheduler name -> (module, attribute)."""
    return dict(_REGISTRY)

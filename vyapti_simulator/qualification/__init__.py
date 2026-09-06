"""
PS26055 Gate 0 qualification — instrument calibration and admissibility.

This package qualifies the SIMULATOR and certifies SCHEDULERS. It contains no
research algorithms; see `algorithms/README.md` for the ownership boundary.

  probes.py       Known inputs with known correct readings. Includes the
                  truth-reading oracles that establish performance ceilings,
                  without which a raw metric is uninterpretable.
  conformance.py  The admissibility gate any scheduler must pass before its
                  results may be reported under the frozen protocol.

    python -m vyapti_simulator.qualification.conformance --self-test
    python -m vyapti_simulator.qualification.conformance --scheduler mod:Class
"""

from .probes import (
    RoundRobinProbe,
    UniformRandomProbe,
    StaticBandProbe,
    OmniscientCaptureOracle,
    GreedyDiscoveryOracle,
    BLIND_PROBES,
    ORACLE_PROBES,
)
from .conformance import (
    CheckResult,
    ConformanceReport,
    run_conformance,
    self_test,
    default_conformance_scenario,
)

__all__ = [
    "RoundRobinProbe", "UniformRandomProbe", "StaticBandProbe",
    "OmniscientCaptureOracle", "GreedyDiscoveryOracle",
    "BLIND_PROBES", "ORACLE_PROBES",
    "CheckResult", "ConformanceReport", "run_conformance", "self_test",
    "default_conformance_scenario",
]

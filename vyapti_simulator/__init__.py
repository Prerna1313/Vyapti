"""
PS26055 Simulation Platform — Package Root

Algorithm-agnostic Electronic Support (ES) receiver scheduling simulator built
to the Common Simulation and Evaluation Protocol v1.0 (FROZEN).

Import convention: this package uses RELATIVE imports internally. Run it as
    python -m vyapti_simulator.main
    python -m vyapti_simulator.simulator
from the directory that CONTAINS vyapti_simulator/ (i.e. D:\\Vyapti).

Provenance labels used throughout:
  [PS-DEFINED]              Directly from SIH 26055 problem statement
  [LITERATURE-GROUNDED]     Clarkson 2003/2011/2019, Glaude 2015, Teissier 2024, ...
  [TSRD-DERIVED]            Turing Synthetic Radar Dataset (arXiv:2602.03856)
  [ENGINEERING-ASSUMPTION]  Justified engineering choice, explicitly stated
  [EXPERIMENTAL-VARIABLE]   Sweep parameter for experiments
"""

__version__ = "1.0.0"
__protocol_version__ = "v1.0_FROZEN"
__architecture_version__ = "simplified_hybrid_v1_post_audit"

# Provenance tag vocabulary is the one thing every module needs; re-export it
# from the root so callers never have to reach into core.environment for it.
from .core.environment import ProvenanceTag, ProvenanceLabel  # noqa: E402

__all__ = [
    "__version__",
    "__protocol_version__",
    "__architecture_version__",
    "ProvenanceTag",
    "ProvenanceLabel",
]

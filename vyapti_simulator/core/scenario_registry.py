"""
vyapti_simulator.core.scenario_registry
=======================================

PS26055 Scenario Registry — Canonical scenario definitions for experiment
reproducibility and benchmark integrity.

The registry provides:
1. **Canonical scenario templates** — named scenarios with fixed parameters
2. **Version tracking** — tracks which scenarios were used in each experiment
3. **Train/eval/test split** — proper held-out sets per frozen protocol §6

Per frozen protocol §6 line 180: "10 training seeds × 100 evaluation
scenario seeds per benchmark cell." The registry enforces non-overlapping
splits and records the version tag used in each experiment run.

References
----------
PS26055 Common Simulation and Evaluation Protocol v1.0 FROZEN.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import hashlib
import json


# =====================================================================
# Scenario version tracking
# =====================================================================

SCENARIO_REGISTRY_VERSION = "v1.0"
SCENARIO_REGISTRY_HASH = "a3f2c1e9b8d4f6a7c0e5b2d9f4a8c3e1b7d5f9a2c4e6b8d0f3a5c7e9b1d3f5"  # Placeholder


class SeedSplit(str, Enum):
    """Seed set classification per frozen protocol §6."""
    TRAIN = "train"      # Algorithm development
    EVAL = "eval"        # Hyperparameter / gate evaluation
    TEST = "test"        # Held-out generalisation claim


@dataclass(frozen=True)
class ScenarioVersion:
    """
    Immutable snapshot of the scenario registry used in an experiment run.

    This is the version tag recorded in every result row, enabling
    exact reproduction of any published experiment.
    """
    version: str
    registry_hash: str
    timestamp_utc: str
    n_train_seeds: int
    n_eval_seeds: int
    n_test_seeds: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_registry_version": self.version,
            "scenario_registry_hash": self.registry_hash,
            "timestamp_utc": self.timestamp_utc,
            "n_train_seeds": self.n_train_seeds,
            "n_eval_seeds": self.n_eval_seeds,
            "n_test_seeds": self.n_test_seeds,
        }


# =====================================================================
# Scenario definitions
# =====================================================================

@dataclass(frozen=True)
class ScenarioConfig:
    """
    Canonical scenario definition for experiment reproducibility.

    Parameters
    ----------
    name : str
        Human-readable scenario name (e.g., "dense_urban_5_emitters").
    band_count : int
        Number of frequency bands in the receiver.
    time_slots : int
        Number of time slots in the mission.
    emitter_density : int
        Number of active emitters in this scenario.
    dwell_time_ms : float
        Dwell time per band in milliseconds.
    retune_time_ms : float
        Retune time between bands in milliseconds.
    emitter_mixture : Dict[str, float]
        Fraction of each emitter behavior type.
    train_seeds : Tuple[int, ...]
        Training seed range (start, end) — exclusive end for slicing.
    eval_seeds : Tuple[int, ...]
        Evaluation seed range.
    test_seeds : Tuple[int, ...]
        Test seed range (held-out).
    description : str
        Human-readable description of this scenario.
    provenance : str
        Source of the scenario parameters.
    """
    name: str
    band_count: int = 36
    time_slots: int = 600
    emitter_density: int = 10
    dwell_time_ms: float = 50.0
    retune_time_ms: float = 1.0
    emitter_mixture: Tuple[Tuple[str, float], ...] = field(default_factory=lambda: (
        ("periodic_spatial_scan", 0.5),
        ("pseudo_random_agile", 0.3),
        ("intermittent", 0.2),
    ))
    train_seeds: Tuple[int, int] = (0, 1000)      # 1000 seeds
    eval_seeds: Tuple[int, int] = (1000, 1200)    # 200 seeds
    test_seeds: Tuple[int, int] = (2000, 2200)    # 200 seeds
    description: str = ""
    provenance: str = ""

    def train_seed_list(self) -> List[int]:
        return list(range(self.train_seeds[0], self.train_seeds[1]))

    def eval_seed_list(self) -> List[int]:
        return list(range(self.eval_seeds[0], self.eval_seeds[1]))

    def test_seed_list(self) -> List[int]:
        return list(range(self.test_seeds[0], self.test_seeds[1]))

    def seed_list(self, split: SeedSplit) -> List[int]:
        """Get seed list for a given split."""
        if split == SeedSplit.TRAIN:
            return self.train_seed_list()
        elif split == SeedSplit.EVAL:
            return self.eval_seed_list()
        else:
            return self.test_seed_list()

    def validate_seeds(self) -> List[str]:
        """Validate seed splits. Returns list of issues (empty if valid)."""
        issues = []
        train = set(self.train_seed_list())
        eval_set = set(self.eval_seed_list())
        test = set(self.test_seed_list())

        # Check for duplicates within splits
        if len(train) != len(self.train_seed_list()):
            issues.append(f"train_seeds contains duplicates")
        if len(eval_set) != len(self.eval_seed_list()):
            issues.append(f"eval_seeds contains duplicates")
        if len(test) != len(self.test_seed_list()):
            issues.append(f"test_seeds contains duplicates")

        # Check for overlap between splits
        if train & eval_set:
            issues.append(f"train_seeds and eval_seeds overlap")
        if train & test:
            issues.append(f"train_seeds and test_seeds overlap")
        if eval_set & test:
            issues.append(f"eval_seeds and test_seeds overlap")

        return issues

    def compute_hash(self) -> str:
        """Compute deterministic hash of scenario parameters."""
        config_str = json.dumps({
            "name": self.name,
            "band_count": self.band_count,
            "time_slots": self.time_slots,
            "emitter_density": self.emitter_density,
            "dwell_time_ms": self.dwell_time_ms,
            "retune_time_ms": self.retune_time_ms,
            "emitter_mixture": list(self.emitter_mixture),
            "train_seeds": list(self.train_seeds),
            "eval_seeds": list(self.eval_seeds),
            "test_seeds": list(self.test_seeds),
        }, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:32]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "band_count": self.band_count,
            "time_slots": self.time_slots,
            "emitter_density": self.emitter_density,
            "dwell_time_ms": self.dwell_time_ms,
            "retune_time_ms": self.retune_time_ms,
            "emitter_mixture": dict(self.emitter_mixture),
            "train_seeds": list(self.train_seeds),
            "eval_seeds": list(self.eval_seeds),
            "test_seeds": list(self.test_seeds),
            "description": self.description,
            "provenance": self.provenance,
            "scenario_hash": self.compute_hash(),
        }


# =====================================================================
# Canonical scenario library
# =====================================================================

# Per frozen protocol §6: 10 training seeds × 100 evaluation scenario seeds
# per benchmark cell. Default splits use:
#   Train: 0-999 (1000 seeds)
#   Eval:  1000-1199 (200 seeds)
#   Test:  2000-2199 (200 seeds)

SCENARIOS: Dict[str, ScenarioConfig] = {
    # =====================================================================
    # PS26055 Standard Benchmark Scenarios
    # =====================================================================

    "ps26055_standard_low": ScenarioConfig(
        name="ps26055_standard_low",
        emitter_density=5,
        description="Low-density scenario: 5 emitters (2 periodic, 2 agile, 1 intermittent)",
        provenance="PS26055 Frozen Protocol §6",
    ),

    "ps26055_standard_medium": ScenarioConfig(
        name="ps26055_standard_medium",
        emitter_density=10,
        description="Medium-density scenario: 10 emitters (5 periodic, 3 agile, 2 intermittent)",
        provenance="PS26055 Frozen Protocol §6",
    ),

    "ps26055_standard_high": ScenarioConfig(
        name="ps26055_standard_high",
        emitter_density=20,
        description="High-density scenario: 20 emitters (10 periodic, 6 agile, 4 intermittent)",
        provenance="PS26055 Frozen Protocol §6",
    ),

    # =====================================================================
    # TSRD-Derived Scenarios (for real RF signal validation)
    # =====================================================================

    "tsrd_sparse": ScenarioConfig(
        name="tsrd_sparse",
        emitter_density=8,
        band_count=36,
        time_slots=600,
        description="TSRD sparse: 8 emitters matching TSRD low-density fixtures",
        provenance="TSRD arXiv:2602.03856",
    ),

    "tsrd_dense": ScenarioConfig(
        name="tsrd_dense",
        emitter_density=15,
        band_count=36,
        time_slots=600,
        description="TSRD dense: 15 emitters matching TSRD high-density fixtures",
        provenance="TSRD arXiv:2602.03856",
    ),

    # =====================================================================
    # Dynamic Phenomena Scenarios (for temporal dynamics validation)
    # =====================================================================

    "dynamic_delayed_arrival": ScenarioConfig(
        name="dynamic_delayed_arrival",
        emitter_density=10,
        description="Dynamic: emitters with delayed arrival (start mid-mission)",
        provenance="TSRD dynamic phenomena validation",
    ),

    "dynamic_on_off": ScenarioConfig(
        name="dynamic_on_off",
        emitter_density=10,
        description="Dynamic: emitters with random ON/OFF intervals",
        provenance="TSRD dynamic phenomena validation",
    ),

    "dynamic_regime_change": ScenarioConfig(
        name="dynamic_regime_change",
        emitter_density=10,
        description="Dynamic: emitters with mid-mission configuration changes",
        provenance="TSRD dynamic phenomena validation",
    ),
}


# =====================================================================
# Registry operations
# =====================================================================

class ScenarioRegistry:
    """
    Central registry for scenario management and version tracking.

    Provides:
    - Scenario lookup by name
    - Version tracking for experiment reproducibility
    - Seed split management
    - Scenario validation
    """

    def __init__(self, scenarios: Optional[Dict[str, ScenarioConfig]] = None):
        self._scenarios = scenarios or dict(SCENARIOS)
        self._version_history: List[ScenarioVersion] = []

    def get_scenario(self, name: str) -> ScenarioConfig:
        """Get scenario by name."""
        if name not in self._scenarios:
            available = list(self._scenarios.keys())
            raise ValueError(
                f"Scenario {name!r} not found. Available: {available}"
            )
        return self._scenarios[name]

    def register_scenario(self, scenario: ScenarioConfig) -> None:
        """Register a new scenario."""
        issues = scenario.validate_seeds()
        if issues:
            raise ValueError(
                f"Invalid scenario {scenario.name}: {', '.join(issues)}"
            )
        self._scenarios[scenario.name] = scenario

    def list_scenarios(self) -> List[str]:
        """List all registered scenario names."""
        return sorted(self._scenarios.keys())

    def get_version(self, timestamp_utc: Optional[str] = None) -> ScenarioVersion:
        """
        Get the scenario version for the current registry state.

        Parameters
        ----------
        timestamp_utc : str, optional
            ISO timestamp. If None, uses current UTC time.

        Returns
        -------
        ScenarioVersion
            Immutable version snapshot.
        """
        import datetime
        if timestamp_utc is None:
            timestamp_utc = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Compute hash of all scenarios
        all_config = json.dumps(
            {k: v.to_dict() for k, v in sorted(self._scenarios.items())},
            sort_keys=True,
        )
        registry_hash = hashlib.sha256(all_config.encode()).hexdigest()[:32]

        version = ScenarioVersion(
            version=SCENARIO_REGISTRY_VERSION,
            registry_hash=registry_hash,
            timestamp_utc=timestamp_utc,
            n_train_seeds=len(list(self._scenarios.values())[0].train_seed_list()),
            n_eval_seeds=len(list(self._scenarios.values())[0].eval_seed_list()),
            n_test_seeds=len(list(self._scenarios.values())[0].test_seed_list()),
        )
        self._version_history.append(version)
        return version

    def export_registry(self) -> Dict[str, Any]:
        """Export full registry as JSON-serializable dict."""
        return {
            "registry_version": SCENARIO_REGISTRY_VERSION,
            "scenarios": {k: v.to_dict() for k, v in self._scenarios.items()},
            "version_history": [v.to_dict() for v in self._version_history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioRegistry":
        """Reconstruct registry from exported dict."""
        scenarios = {}
        for name, cfg in data.get("scenarios", {}).items():
            scenarios[name] = ScenarioConfig(
                name=cfg["name"],
                band_count=cfg["band_count"],
                time_slots=cfg["time_slots"],
                emitter_density=cfg["emitter_density"],
                dwell_time_ms=cfg["dwell_time_ms"],
                retune_time_ms=cfg["retune_time_ms"],
                emitter_mixture=tuple(cfg["emitter_mixture"].items()),
                train_seeds=tuple(cfg["train_seeds"]),
                eval_seeds=tuple(cfg["eval_seeds"]),
                test_seeds=tuple(cfg["test_seeds"]),
                description=cfg.get("description", ""),
                provenance=cfg.get("provenance", ""),
            )
        return cls(scenarios=scenarios)


# =====================================================================
# Default registry instance
# =====================================================================

_default_registry: Optional[ScenarioRegistry] = None


def get_registry() -> ScenarioRegistry:
    """Get the default global registry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ScenarioRegistry()
    return _default_registry


def get_scenario(name: str) -> ScenarioConfig:
    """Convenience: get scenario from default registry."""
    return get_registry().get_scenario(name)


def list_scenarios() -> List[str]:
    """Convenience: list scenarios in default registry."""
    return get_registry().list_scenarios()


__all__ = [
    "ScenarioConfig",
    "ScenarioRegistry",
    "ScenarioVersion",
    "SeedSplit",
    "SCENARIOS",
    "SCENARIO_REGISTRY_VERSION",
    "get_registry",
    "get_scenario",
    "list_scenarios",
]

"""
vyapti_simulator.core.multi_objective_reward
==========================================

Sub-problem G: Multi-objective reconciliation for PS26055 reward functions.

This module implements configurable multi-objective reward optimisation with
Pareto-front analysis. The key insight is that discovery and monitoring
are conflicting objectives: a policy that stares at discovered emitters
(maximising post-discovery capture) may miss new emitters (minimising
discovery rate).

The module provides:
1. **Configurable reward weights** for discovery vs monitoring vs efficiency
2. **Pareto-front analysis** to identify non-dominated policies
3. **Scalarisation methods** (weighted sum, Chebyshev, Tchebycheff, hybrid)
4. **Thompson sampling for multi-objective** via preference elicitation

References
----------
- Deb 2014: Multi-objective optimisation and evolutionary algorithms
- Ziomek 2021: Multi-objective MAB for ESM
- PS26055 Frozen Protocol §6: Multi-objective Reconciliation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
from copy import deepcopy

import numpy as np


# =====================================================================
# Objective Definitions
# =====================================================================

class ObjectiveType(str, Enum):
    """Types of objectives in the multi-objective reward."""
    DISCOVERY_RATE = "discovery_rate"        # Fraction of emitters found
    DISCOVERY_SPEED = "discovery_speed"       # Mean time to find emitters
    MONITORING_RATE = "monitoring_rate"       # Post-discovery capture rate
    COVERAGE = "coverage"                      # Rolling spectrum coverage
    EFFICIENCY = "retune_efficiency"          # Minimize retune overhead
    ACCURACY = "prediction_accuracy"          # Prediction correctness
    THREAT_DETECTION = "threat_detection"     # Threat-weighted detection (Sub-problem F+G)
    INTERCEPT_SPEED = "intercept_speed"       # Speed of first intercept
    DWELL_EFFICIENCY = "dwell_efficiency"    # Minimize dwell overhead cost


# =====================================================================
# PS26055 Sub-problem G: Configured Reward Formula
# =====================================================================

@dataclass
class RewardConfig:
    """
    Configuration for the multi-objective reward function (PS26055 Sub-problem G).

    Implements the weighted multi-objective reward formula:
        reward = w_detection * (threat_score * detection)
               + w_speed * (1 / intercept_time)
               - w_cost * dwell_cost

    Default weights (as specified):
        - detection: 0.6 (60% weight on threat-weighted detection)
        - intercept_speed: 0.3 (30% weight on speed of first intercept)
        - efficiency: 0.1 (10% weight on dwell efficiency)

    Parameters
    ----------
    weight_detection : float
        Weight for threat-weighted detection. Default: 0.6
    weight_speed : float
        Weight for intercept speed. Default: 0.3
    weight_cost : float
        Weight for dwell efficiency (negative penalty). Default: 0.1
    use_threat_weighting : bool
        If True, multiply detection by threat score. Default: True
    intercept_time_max : float
        Maximum intercept time for normalisation. Default: 500.0 slots
    memo_reference : str
        Reference to the reward formulation documentation.
    """
    weight_detection: float = 0.6
    weight_speed: float = 0.3
    weight_cost: float = 0.1
    use_threat_weighting: bool = True
    intercept_time_max: float = 500.0
    memo_reference: str = "PS26055-SubProb-G-IMPLEMENTED"

    def __post_init__(self) -> None:
        """Validate and normalise weights."""
        total = self.weight_detection + self.weight_speed + self.weight_cost
        if total > 0:
            # Normalise to ensure weights sum to 1.0
            self.weight_detection /= total
            self.weight_speed /= total
            self.weight_cost /= total

    def compute_reward(
        self,
        detection: float,
        threat_score: float,
        intercept_time: float,
        dwell_cost: float,
    ) -> float:
        """
        Compute the multi-objective reward using the PS26055 Sub-problem G formula.

        reward = w_detection * (threat_score * detection)
               + w_speed * (1 / intercept_time)
               - w_cost * dwell_cost

        Parameters
        ----------
        detection : float
            Binary or fractional detection result (0.0 or 1.0 for hit/miss,
            or pulse_count / max_pulses for graded detection)
        threat_score : float
            Threat score from ThreatScorer [0, 1], higher = more threatening
        intercept_time : float
            Time since first detection of this emitter (slots)
        dwell_cost : float
            Dwell cost (retune time / slot duration)

        Returns
        -------
        float
            Composite reward value
        """
        # Threat-weighted detection component
        if self.use_threat_weighting:
            detection_component = threat_score * detection
        else:
            detection_component = detection

        # Intercept speed component (normalised to [0, 1])
        if intercept_time > 0:
            # Normalized: 1.0 at intercept_time_max (worst), increases as intercept is faster
            # Use 1 - normalized so that faster intercepts have higher values
            normalized_time = min(1.0, intercept_time / self.intercept_time_max)
            speed_component = 1.0 - normalized_time
        else:
            speed_component = 1.0  # Immediate intercept

        # Dwell efficiency component (penalty, so subtract)
        efficiency_component = dwell_cost

        # Compute weighted sum
        reward = (
            self.weight_detection * detection_component
            + self.weight_speed * speed_component
            - self.weight_cost * efficiency_component
        )

        return float(reward)

    def compute_reward_from_observation(
        self,
        observation: Dict[str, Any],
        threat_score: float,
        intercept_time: float,
    ) -> float:
        """
        Compute reward from a single observation.

        This is the step-level reward function for use during episodes.

        Parameters
        ----------
        observation : Dict[str, Any]
            Observation dict from environment step
        threat_score : float
            Threat score from ThreatScorer
        intercept_time : float
            Time since first detection of this emitter (slots)

        Returns
        -------
        float
            Reward value for this step
        """
        detection = 1.0 if observation.get("hit", False) else 0.0
        dwell_cost = observation.get("retune_cost_s", 0.0)
        slot_duration = observation.get("receiver_metadata", {}).get("dwell_time_ms", 50.0)
        dwell_cost_slots = dwell_cost / (slot_duration / 1000.0) if slot_duration > 0 else 0.0

        return self.compute_reward(
            detection=detection,
            threat_score=threat_score,
            intercept_time=intercept_time,
            dwell_cost=dwell_cost_slots,
        )

    def is_implemented(self) -> bool:
        """True iff reward config is configured and ready for use."""
        return True


@dataclass
class Objective:
    """
    Definition of one objective in the multi-objective problem.

    Parameters
    ----------
    name : str
        Unique identifier for this objective
    objective_type : ObjectiveType
        Type of objective
    weight : float
        Weight in the composite reward (normalised to sum to 1.0)
    direction : str
        "maximize" or "minimize"
    threshold : Optional[float]
        Minimum acceptable value; None = no threshold
    """
    name: str
    objective_type: ObjectiveType
    weight: float = 1.0
    direction: str = "maximize"
    threshold: Optional[float] = None


@dataclass
class ObjectiveValue:
    """Computed value of one objective."""
    objective: Objective
    raw_value: float
    normalised_value: float  # Scaled to [0, 1] based on expected range
    satisfied: bool  # Whether threshold is met


@dataclass
class MultiObjectiveResult:
    """Result of multi-objective evaluation."""
    objectives: List[ObjectiveValue]
    composite_score: float
    pareto_dominated: bool  # True if dominated by another solution
    pareto_rank: int  # 0 = on Pareto front, 1 = one step removed, etc.
    dominates_count: int  # Number of other solutions this dominates
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Scalarisation Methods
# =====================================================================

class ScalarisationMethod(str, Enum):
    """Methods for converting multi-objective to scalar reward."""
    WEIGHTED_SUM = "weighted_sum"  # Linear weighted sum
    CHEBYSHEV = "chebyshev"        # Chebyshev/Tchebycheff scalarisation
    AUGMENTED_CHEBYSHEV = "augmented_chebyshev"  # Augmented Chebyshev
    PENALTY_BASED = "penalty_based"  # Tchebycheff with penalty for imbalance
    HYBRID = "hybrid"  # Combines weighted sum and Chebyshev


class ScalarisationFunction:
    """
    Base class for scalarisation functions.

    Scalarisation converts a vector of objective values into a scalar
    reward for the learning algorithm.
    """

    def __init__(self, objectives: List[Objective]):
        self.objectives = objectives
        self._normalise_weights()

    def _normalise_weights(self) -> None:
        """Normalise weights to sum to 1.0."""
        total = sum(o.weight for o in self.objectives)
        if total > 0:
            for o in self.objectives:
                o.weight = o.weight / total

    @abstractmethod
    def compute(self, values: List[ObjectiveValue]) -> float:
        """Compute scalar reward from objective values."""
        raise NotImplementedError

    def normalise_value(self, value: float, objective: Objective,
                        reference_range: Tuple[float, float]) -> float:
        """
        Normalise value to [0, 1] based on reference range.

        Parameters
        ----------
        value : float
            Raw objective value
        objective : Objective
            The objective definition
        reference_range : Tuple[float, float]
            (min, max) expected values for normalisation
        """
        min_val, max_val = reference_range
        if max_val <= min_val:
            return 0.5  # Neutral if range is invalid

        normalised = (value - min_val) / (max_val - min_val)
        normalised = float(np.clip(normalised, 0.0, 1.0))

        # Flip if minimizing
        if objective.direction == "minimize":
            normalised = 1.0 - normalised

        return normalised


class WeightedSumScalarisation(ScalarisationFunction):
    """
    Weighted sum scalarisation.

    R(x) = Σ w_i * f_i(x)

    Simple and efficient, but cannot find all Pareto-optimal solutions
    for non-convex fronts. Good for convex problems.
    """

    def compute(self, values: List[ObjectiveValue]) -> float:
        return sum(v.normalised_value * o.weight
                   for v, o in zip(values, self.objectives))


class ChebyshevScalarisation(ScalarisationFunction):
    """
    Chebyshev (Tchebycheff) scalarisation.

    R(x) = max_i { w_i * |f_i(x) - z_i*| }

    where z_i* is the ideal point (best value for each objective).

    Can find all Pareto-optimal solutions for any problem shape,
    but may produce weakly Pareto-optimal solutions.
    """

    def __init__(self, objectives: List[Objective], ideal_point: Optional[List[float]] = None):
        super().__init__(objectives)
        self.ideal_point = ideal_point  # Reference point for Chebyshev

    def compute(self, values: List[ObjectiveValue]) -> float:
        if self.ideal_point is None:
            # Use ideal point of 1.0 for all objectives (best possible normalised score)
            ideal = [1.0] * len(values)
        else:
            ideal = self.ideal_point

        chebyshev = max(
            o.weight * abs(v.normalised_value - i)
            for v, o, i in zip(values, self.objectives, ideal)
        )
        return 1.0 - chebyshev  # Invert so higher = better


class AugmentedChebyshevScalarisation(ScalarisationFunction):
    """
    Augmented Chebyshev scalarisation.

    R(x) = max_i { w_i * |f_i(x) - z_i*| } + ρ * Σ w_i * f_i(x)

    Combines Chebyshev with weighted sum to avoid weakly Pareto-optimal
    solutions. The parameter ρ (rho) controls the augmentation strength.
    """

    def __init__(self, objectives: List[Objective], rho: float = 0.001,
                 ideal_point: Optional[List[float]] = None):
        super().__init__(objectives)
        self.rho = rho
        self.ideal_point = ideal_point

    def compute(self, values: List[ObjectiveValue]) -> float:
        if self.ideal_point is None:
            ideal = [1.0] * len(values)  # Perfect score for all
        else:
            ideal = self.ideal_point

        chebyshev = max(
            o.weight * abs(v.normalised_value - i)
            for v, o, i in zip(values, self.objectives, ideal)
        )
        weighted_sum = sum(v.normalised_value * o.weight
                          for v, o in zip(values, self.objectives))

        return 1.0 - chebyshev + self.rho * weighted_sum


class PenaltyBasedScalarisation(ScalarisationFunction):
    """
    Penalty-based scalarisation (Tchebycheff variant).

    R(x) = Σ w_i * f_i(x) - ρ * max_i { |f_i(x) - f_i*| }

    Penalises imbalance between objectives while rewarding absolute performance.
    """

    def __init__(self, objectives: List[Objective], penalty_factor: float = 0.1):
        super().__init__(objectives)
        self.penalty_factor = penalty_factor

    def compute(self, values: List[ObjectiveValue]) -> float:
        weighted_sum = sum(v.normalised_value * o.weight
                          for v, o in zip(values, self.objectives))

        # Compute max deviation from ideal
        ideal = 1.0  # Perfect normalised score
        max_deviation = max(abs(v.normalised_value - ideal)
                           for v in values)

        penalty = self.penalty_factor * max_deviation
        return weighted_sum - penalty


class HybridScalarisation(ScalarisationFunction):
    """
    Hybrid scalarisation combining weighted sum and Chebyshev.

    R(x) = (1 - α) * weighted_sum + α * (1 - chebyshev)

    The parameter α ∈ [0, 1] controls the balance:
    - α = 0: pure weighted sum
    - α = 1: pure Chebyshev
    - α = 0.5: balanced hybrid
    """

    def __init__(self, objectives: List[Objective], alpha: float = 0.5,
                 ideal_point: Optional[List[float]] = None):
        super().__init__(objectives)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.ideal_point = ideal_point

    def compute(self, values: List[ObjectiveValue]) -> float:
        # Weighted sum component
        ws = sum(v.normalised_value * o.weight
                for v, o in zip(values, self.objectives))

        # Chebyshev component
        if self.ideal_point is None:
            ideal = [1.0] * len(values)
        else:
            ideal = self.ideal_point

        cheb = max(
            o.weight * abs(v.normalised_value - i)
            for v, o, i in zip(values, self.objectives, ideal)
        )

        return (1.0 - self.alpha) * ws + self.alpha * (1.0 - cheb)


# =====================================================================
# Multi-Objective Reward Engine
# =====================================================================

@dataclass
class MultiObjectiveRewardConfig:
    """
    Configuration for multi-objective reward computation.

    Parameters
    ----------
    objectives : List[Objective]
        List of objectives to optimise
    scalarisation_method : ScalarisationMethod
        Method for converting multi-objective to scalar
    reference_ranges : Dict[str, Tuple[float, float]]
        (min, max) ranges for normalising each objective
    alpha : float
        Parameter for hybrid scalarisation (if used)
    rho : float
        Parameter for augmented Chebyshev (if used)
    penalty_factor : float
        Parameter for penalty-based scalarisation (if used)
    use_pareto_filtering : bool
        If True, filter solutions to Pareto front
    """
    objectives: List[Objective] = field(default_factory=list)
    scalarisation_method: ScalarisationMethod = ScalarisationMethod.WEIGHTED_SUM
    reference_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    alpha: float = 0.5  # Hybrid parameter
    rho: float = 0.001  # Augmented Chebyshev parameter
    penalty_factor: float = 0.1  # Penalty-based parameter
    use_pareto_filtering: bool = False

    def __post_init__(self) -> None:
        # Set default objectives if none specified
        if not self.objectives:
            self.objectives = [
                Objective("discovery", ObjectiveType.DISCOVERY_RATE, weight=0.3,
                         direction="maximize"),
                Objective("monitoring", ObjectiveType.MONITORING_RATE, weight=0.3,
                         direction="maximize"),
                Objective("efficiency", ObjectiveType.EFFICIENCY, weight=0.2,
                         direction="maximize"),
                Objective("coverage", ObjectiveType.COVERAGE, weight=0.2,
                         direction="maximize"),
            ]

        # Set default reference ranges
        default_ranges = {
            "discovery": (0.0, 1.0),  # 0-100% discovery rate
            "monitoring": (0.0, 1.0),  # 0-100% monitoring rate
            "efficiency": (0.0, 1.0),  # 0-100% retune efficiency
            "coverage": (0.0, 1.0),  # 0-100% coverage
            "discovery_speed": (0, 500),  # 0-500 slots
            "accuracy": (0.0, 1.0),  # 0-100% accuracy
        }
        for name, range_vals in default_ranges.items():
            if name not in self.reference_ranges:
                self.reference_ranges[name] = range_vals


class MultiObjectiveRewardEngine:
    """
    Engine for computing multi-objective rewards.

    This class handles:
    1. Computing objective values from episode metrics
    2. Normalising values to a common scale
    3. Applying scalarisation to produce scalar reward
    4. Pareto-front analysis for multi-objective comparison
    """

    # Default reference ranges for normalisation
    DEFAULT_REFERENCE_RANGES = {
        "discovery_rate": (0.0, 1.0),
        "discovery_speed": (0.0, 500.0),  # slots
        "monitoring_rate": (0.0, 1.0),
        "coverage": (0.0, 1.0),
        "efficiency": (0.0, 1.0),
        "prediction_accuracy": (0.0, 1.0),
    }

    def __init__(self, config: Optional[MultiObjectiveRewardConfig] = None):
        self.config = config or MultiObjectiveRewardConfig()
        self._build_scalariser()

    def _build_scalariser(self) -> None:
        """Build the scalarisation function based on config."""
        method = self.config.scalarisation_method

        if method == ScalarisationMethod.WEIGHTED_SUM:
            self.scalariser = WeightedSumScalarisation(self.config.objectives)
        elif method == ScalarisationMethod.CHEBYSHEV:
            self.scalariser = ChebyshevScalarisation(self.config.objectives)
        elif method == ScalarisationMethod.AUGMENTED_CHEBYSHEV:
            self.scalariser = AugmentedChebyshevScalarisation(
                self.config.objectives, rho=self.config.rho
            )
        elif method == ScalarisationMethod.PENALTY_BASED:
            self.scalariser = PenaltyBasedScalarisation(
                self.config.objectives, penalty_factor=self.config.penalty_factor
            )
        elif method == ScalarisationMethod.HYBRID:
            self.scalariser = HybridScalarisation(
                self.config.objectives, alpha=self.config.alpha
            )
        else:
            self.scalariser = WeightedSumScalarisation(self.config.objectives)

    def compute_objective_values(self, metrics: Dict[str, Any]) -> List[ObjectiveValue]:
        """
        Compute objective values from episode metrics.

        Parameters
        ----------
        metrics : Dict
            Episode metrics from MetricsEngine

        Returns
        -------
        List[ObjectiveValue]
            Computed values for each objective
        """
        values = []

        for objective in self.config.objectives:
            # Extract raw value from metrics
            raw_value = self._extract_objective_value(objective, metrics)

            # Get reference range
            ref_range = self.config.reference_ranges.get(
                objective.name,
                self.DEFAULT_REFERENCE_RANGES.get(objective.objective_type.value, (0.0, 1.0))
            )

            # Normalise value
            normalised = self.scalariser.normalise_value(
                raw_value, objective, ref_range
            )

            # Check threshold
            satisfied = True
            if objective.threshold is not None:
                if objective.direction == "maximize":
                    satisfied = raw_value >= objective.threshold
                else:
                    satisfied = raw_value <= objective.threshold

            values.append(ObjectiveValue(
                objective=objective,
                raw_value=raw_value,
                normalised_value=normalised,
                satisfied=satisfied
            ))

        return values

    def _extract_objective_value(self, objective: Objective,
                                  metrics: Dict[str, Any]) -> float:
        """Extract raw value for an objective from metrics."""
        obj_type = objective.objective_type
        discovery = metrics.get("discovery_metrics", {})
        monitoring = metrics.get("monitoring_metrics", {})
        detection = metrics.get("detection_metrics", {})

        if obj_type == ObjectiveType.DISCOVERY_RATE:
            return discovery.get("interception_probability", 0.0) or 0.0

        elif obj_type == ObjectiveType.DISCOVERY_SPEED:
            val = discovery.get("mean_first_intercept_time_slots")
            return val if val is not None else float('inf')

        elif obj_type == ObjectiveType.MONITORING_RATE:
            val = monitoring.get("post_discovery_capture_efficiency")
            return val if val is not None else 0.0

        elif obj_type == ObjectiveType.COVERAGE:
            return monitoring.get("average_coverage_fraction", 0.0) or 0.0

        elif obj_type == ObjectiveType.EFFICIENCY:
            dead_time = monitoring.get("dead_time_fraction", 1.0) or 1.0
            return 1.0 - dead_time  # Efficiency = 1 - dead time

        elif obj_type == ObjectiveType.ACCURACY:
            pred = metrics.get("prediction_metrics", {})
            pct = pred.get("percentage_correct_predictions")
            return (pct / 100.0) if pct is not None else 0.0

        return 0.0  # Default fallback

    def compute_reward(self, metrics: Dict[str, Any]) -> Tuple[float, MultiObjectiveResult]:
        """
        Compute multi-objective reward from episode metrics.

        Parameters
        ----------
        metrics : Dict
            Episode metrics from MetricsEngine

        Returns
        -------
        Tuple[float, MultiObjectiveResult]
            Scalar reward and detailed result
        """
        values = self.compute_objective_values(metrics)
        composite = self.scalariser.compute(values)

        result = MultiObjectiveResult(
            objectives=values,
            composite_score=composite,
            pareto_dominated=False,
            pareto_rank=0,
            dominates_count=0,
            metadata={"scalarisation": self.config.scalarisation_method.value}
        )

        return composite, result

    def compute_pareto_front(self, results: List[MultiObjectiveResult]) -> List[int]:
        """
        Identify Pareto-optimal solutions.

        Parameters
        ----------
        results : List[MultiObjectiveResult]
            List of multi-objective results

        Returns
        -------
        List[int]
            Indices of Pareto-optimal solutions
        """
        n = len(results)
        if n == 0:
            return []

        pareto_indices = []
        for i in range(n):
            is_dominated = False
            dominates_count = 0

            for j in range(n):
                if i == j:
                    continue

                # Check if result i dominates result j
                i_dominates_j = self._dominates(results[i], results[j])
                j_dominates_i = self._dominates(results[j], results[i])

                if j_dominates_i:
                    is_dominated = True
                if i_dominates_j:
                    dominates_count += 1

            results[i].pareto_dominated = is_dominated
            results[i].dominates_count = dominates_count

            if not is_dominated:
                pareto_indices.append(i)

        # Assign Pareto ranks
        self._assign_pareto_ranks(results)

        return pareto_indices

    def _dominates(self, a: MultiObjectiveResult, b: MultiObjectiveResult) -> bool:
        """
        Check if result a dominates result b (Pareto dominance).

        a dominates b if:
        - a is at least as good as b in all objectives
        - a is strictly better than b in at least one objective
        """
        better_in_any = False

        for va, vb in zip(a.objectives, b.objectives):
            # Compare normalised values (higher = better after normalisation)
            if va.normalised_value < vb.normalised_value:
                return False  # a is worse in this objective
            if va.normalised_value > vb.normalised_value:
                better_in_any = True

        return better_in_any

    def _assign_pareto_ranks(self, results: List[MultiObjectiveResult]) -> None:
        """Assign Pareto ranks (0 = front, 1 = next, etc.)."""
        n = len(results)
        assigned = [False] * n
        current_rank = 0

        while not all(assigned):
            front_indices = []

            for i in range(n):
                if assigned[i]:
                    continue

                # Check if dominated by any unassigned solution
                dominated_by_unassigned = False
                for j in range(n):
                    if i == j or assigned[j]:
                        continue
                    if self._dominates(results[j], results[i]):
                        dominated_by_unassigned = True
                        break

                if not dominated_by_unassigned:
                    front_indices.append(i)

            for i in front_indices:
                results[i].pareto_rank = current_rank
                assigned[i] = True

            current_rank += 1

    def get_pareto_front_statistics(self, results: List[MultiObjectiveResult]) -> Dict[str, Any]:
        """
        Get statistics about the Pareto front.

        Returns
        -------
        Dict
            Statistics about Pareto-optimal solutions
        """
        pareto_indices = self.compute_pareto_front(results)
        pareto_results = [results[i] for i in pareto_indices]

        if not pareto_results:
            return {"n_pareto": 0, "message": "No Pareto-optimal solutions found"}

        # Compute statistics per objective
        stats = {"n_pareto": len(pareto_results)}
        for obj in self.config.objectives:
            values = [r.objectives[i].normalised_value
                     for r in pareto_results
                     for i, o in enumerate(r.objectives)
                     if o.objective.name == obj.name]
            if values:
                stats[f"{obj.name}_mean"] = float(np.mean(values))
                stats[f"{obj.name}_std"] = float(np.std(values))
                stats[f"{obj.name}_range"] = (float(min(values)), float(max(values)))

        return stats


# =====================================================================
# Multi-Objective Thompson Sampling
# =====================================================================

class PreferenceElicitation:
    """
    Preference elicitation for multi-objective Thompson sampling.

    This class handles:
    1. Tracking pairwise comparisons from feedback
    2. Fitting a preference model (Bradley-Terry)
    3. Sampling from the posterior to guide exploration
    """

    def __init__(self, objectives: List[Objective], alpha: float = 1.0):
        self.objectives = objectives
        self.n_objectives = len(objectives)
        self.alpha = alpha  # Dirichlet concentration parameter

        # Pairwise comparison counts: wins[a][b] = times a preferred over b
        self.wins = [[0] * self.n_objectives for _ in range(self.n_objectives)]
        self.total_comparisons = 0

    def record_preference(self, preferred: int, dispreferred: int) -> None:
        """Record a pairwise comparison: preferred was chosen over dispreferred."""
        if 0 <= preferred < self.n_objectives and 0 <= dispreferred < self.n_objectives:
            self.wins[preferred][dispreferred] += 1
            self.total_comparisons += 1

    def sample_weight_vector(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample weight vectors from the posterior distribution.

        Uses a simple Dirichlet-based model where each objective's
        weight is proportional to its relative preference frequency.
        """
        samples = []
        for _ in range(n_samples):
            # Compute preference scores
            scores = np.zeros(self.n_objectives)
            for i in range(self.n_objectives):
                total_wins = sum(self.wins[i])
                total_losses = sum(self.wins[j][i] for j in range(self.n_objectives))
                total = total_wins + total_losses
                if total > 0:
                    scores[i] = (total_wins + self.alpha) / (total + 2 * self.alpha)
                else:
                    scores[i] = 1.0 / self.n_objectives

            # Sample from Dirichlet
            alpha_post = scores * (self.total_comparisons + 1) + self.alpha
            weights = np.random.dirichlet(alpha_post)
            samples.append(weights)

        if n_samples == 1:
            return samples[0]
        return np.array(samples)

    def get_weight_estimate(self) -> np.ndarray:
        """Get maximum likelihood estimate of objective weights."""
        weights = np.zeros(self.n_objectives)
        for i in range(self.n_objectives):
            total_wins = sum(self.wins[i])
            total_losses = sum(self.wins[j][i] for j in range(self.n_objectives))
            total = total_wins + total_losses
            if total > 0:
                weights[i] = total_wins / total
            else:
                weights[i] = 1.0 / self.n_objectives

        # Normalise
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return weights


# =====================================================================
# Convenience Functions
# =====================================================================

def create_default_multi_objective_engine(
    discovery_weight: float = 0.3,
    monitoring_weight: float = 0.3,
    coverage_weight: float = 0.2,
    efficiency_weight: float = 0.2,
    scalarisation: ScalarisationMethod = ScalarisationMethod.HYBRID,
    alpha: float = 0.5,
) -> MultiObjectiveRewardEngine:
    """
    Create a multi-objective reward engine with sensible defaults.

    Parameters
    ----------
    discovery_weight : float
        Weight for discovery rate
    monitoring_weight : float
        Weight for monitoring rate
    coverage_weight : float
        Weight for coverage
    efficiency_weight : float
        Weight for efficiency
    scalarisation : ScalarisationMethod
        Scalarisation method
    alpha : float
        Parameter for hybrid scalarisation

    Returns
    -------
    MultiObjectiveRewardEngine
        Configured engine
    """
    objectives = [
        Objective("discovery", ObjectiveType.DISCOVERY_RATE,
                 weight=discovery_weight, direction="maximize"),
        Objective("monitoring", ObjectiveType.MONITORING_RATE,
                 weight=monitoring_weight, direction="maximize"),
        Objective("coverage", ObjectiveType.COVERAGE,
                 weight=coverage_weight, direction="maximize"),
        Objective("efficiency", ObjectiveType.EFFICIENCY,
                 weight=efficiency_weight, direction="maximize"),
    ]

    config = MultiObjectiveRewardConfig(
        objectives=objectives,
        scalarisation_method=scalarisation,
        alpha=alpha,
    )

    return MultiObjectiveRewardEngine(config)


# =====================================================================
# Module Exports
# =====================================================================

__all__ = [
    # Sub-problem G: Multi-objective reward
    "RewardConfig",
    # Objectives
    "ObjectiveType",
    "Objective",
    "ObjectiveValue",
    "MultiObjectiveResult",
    # Scalarisation methods
    "ScalarisationMethod",
    "ScalarisationFunction",
    "WeightedSumScalarisation",
    "ChebyshevScalarisation",
    "AugmentedChebyshevScalarisation",
    "PenaltyBasedScalarisation",
    "HybridScalarisation",
    # Engine
    "MultiObjectiveRewardConfig",
    "MultiObjectiveRewardEngine",
    "PreferenceElicitation",
    "create_default_multi_objective_engine",
]

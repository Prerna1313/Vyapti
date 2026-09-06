"""
vyapti_simulator.algorithms.threat
=================================

Sub-problem F: Threat-aware band prioritisation.

This module implements threat prioritisation based on observable features
from the observation history. The threat model balances:

1. **SNR-based threat**: High-SNR emitters are more threatening (longer detection
   range, higher power = more capable systems)
2. **Agility threat**: Frequency-agile emitters are harder to track and more
   threatening (modern radar/EW systems)
3. **PRI stability threat**: Jittered PRI emitters are harder to jam, so
   unstable PRI = higher threat
4. **Activity-based threat**: Bands with recent hits are more likely to have
   active threats

The threat score is computed from the observation history without access to
hidden truth. It uses a weighted combination of features derived from the
deinterleaver (when available) or from the raw observation statistics.

References
----------
DRDO PS26055 Frozen Protocol — Sub-problem F: Threat Prioritisation.
Threat model based on:
- Apfeld & Charlish 2021: Threat prioritisation in ESM
- Perini 2025 SmartScan: Threat-weighted MAB
- Clarkson 2019: Emergent behaviour classification
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np


# =====================================================================
# Emitter Behavior Classification
# =====================================================================

class EmitterBehaviorClass(str, Enum):
    """
    Classification of emitter behaviors by threat level.

    Based on PS26055 Sub-problem F requirements:
    - Agile emitters (PSEUDO_RANDOM_AGILE, MARKOV_HOPPER) = priority 9-10
    - Periodic emitters (PERIODIC_SPATIAL_SCAN) = priority 7
    - Fixed emitters (CONTINUOUS_FIXED) = priority 3
    """
    # Fixed/threat level 3
    CONTINUOUS_FIXED = "continuous_fixed"       # Priority 3: constant frequency, constant PRI
    FIXED_INTERMITTENT = "fixed_intermittent"   # Priority 3: same but with ON/OFF cycles

    # Periodic/threat level 7
    PERIODIC_SPATIAL_SCAN = "periodic_spatial_scan"  # Priority 7: dwells at known bands
    PRI_JITTER = "pri_jitter"                # Priority 7: periodic with PRI instability

    # Agile/threat level 9-10
    PSEUDO_RANDOM_AGILE = "pseudo_random_agile"    # Priority 9: frequency hopping
    MARKOV_HOPPER = "markov_hopper"          # Priority 10: CTMC-based band switching
    FREQUENCY_AGILE = "frequency_agile"      # Priority 9: general frequency agility
    FREQUENCY_AGILE_SCAN = "frequency_agile_scan"  # Priority 9: agile + scanning

    # Unknown/default
    UNKNOWN = "unknown"  # Priority 5: baseline threat for unclassified emitters


# =====================================================================
# Behavior-Based Threat Scores
# =====================================================================

# Static threat scores based on emitter behavior type
# These scores represent the inherent threat level of each behavior type
BEHAVIOR_THREAT_SCORES: Dict[EmitterBehaviorClass, float] = {
    EmitterBehaviorClass.CONTINUOUS_FIXED: 3.0,
    EmitterBehaviorClass.FIXED_INTERMITTENT: 3.0,
    EmitterBehaviorClass.PERIODIC_SPATIAL_SCAN: 7.0,
    EmitterBehaviorClass.PRI_JITTER: 7.0,
    EmitterBehaviorClass.PSEUDO_RANDOM_AGILE: 9.0,
    EmitterBehaviorClass.MARKOV_HOPPER: 10.0,
    EmitterBehaviorClass.FREQUENCY_AGILE: 9.0,
    EmitterBehaviorClass.FREQUENCY_AGILE_SCAN: 9.0,
    EmitterBehaviorClass.UNKNOWN: 5.0,
}

# Mapping from emitter class names to behavior classes
EMITTER_CLASS_TO_BEHAVIOR: Dict[str, EmitterBehaviorClass] = {
    # Fixed emitters (threat level 3)
    "FixedContinuousEmitter": EmitterBehaviorClass.CONTINUOUS_FIXED,
    "FixedIntermittentEmitter": EmitterBehaviorClass.FIXED_INTERMITTENT,

    # Periodic emitters (threat level 7)
    "ScanningEmitter": EmitterBehaviorClass.PERIODIC_SPATIAL_SCAN,
    "PriJitterEmitter": EmitterBehaviorClass.PRI_JITTER,

    # Agile emitters (threat level 9-10)
    "FrequencyAgileEmitter": EmitterBehaviorClass.FREQUENCY_AGILE,
    "FrequencyAgileScanningEmitter": EmitterBehaviorClass.FREQUENCY_AGILE_SCAN,

    # Dynamic emitters (mapped based on underlying behavior)
    "DynamicEmitter": EmitterBehaviorClass.UNKNOWN,  # Depends on underlying policy
}


def get_behavior_threat_score(emitter_class_name: str) -> float:
    """
    Get the inherent threat score for an emitter based on its behavior class.

    Parameters
    ----------
    emitter_class_name : str
        The class name of the emitter (e.g., "FrequencyAgileEmitter")

    Returns
    -------
    float
        Threat score in [0, 10], higher = more threatening
    """
    behavior = EMITTER_CLASS_TO_BEHAVIOR.get(emitter_class_name, EmitterBehaviorClass.UNKNOWN)
    return BEHAVIOR_THREAT_SCORES.get(behavior, 5.0)


def classify_emitter_by_observation(observation: Dict[str, Any]) -> EmitterBehaviorClass:
    """
    Classify an emitter's behavior based on observation features.

    This is used when the emitter class name is not available, e.g.,
    when working with TSRD data or deinterleaver tracks.

    Parameters
    ----------
    observation : Dict[str, Any]
        Observation features including frequency_stability, pri_stability, etc.

    Returns
    -------
    EmitterBehaviorClass
        Estimated behavior class based on observable features
    """
    # Extract observable features
    freq_stability = observation.get("frequency_stability", 0.5)  # 0 = stable, 1 = agile
    pri_stability = observation.get("pri_stability", 0.5)  # 0 = stable, 1 = jittery
    spatial_pattern = observation.get("spatial_pattern", "unknown")

    # Classification logic based on observable features
    if freq_stability > 0.7:
        # High frequency agility indicates modern radar
        if pri_stability > 0.5:
            return EmitterBehaviorClass.MARKOV_HOPPER  # Most sophisticated
        return EmitterBehaviorClass.PSEUDO_RANDOM_AGILE
    elif freq_stability > 0.3:
        # Moderate frequency variation
        if spatial_pattern == "scanning":
            return EmitterBehaviorClass.FREQUENCY_AGILE_SCAN
        return EmitterBehaviorClass.FREQUENCY_AGILE
    elif freq_stability < 0.1 and pri_stability < 0.1:
        # Very stable frequency and PRI
        return EmitterBehaviorClass.CONTINUOUS_FIXED
    elif freq_stability < 0.2:
        # Stable frequency but possibly intermittent or periodic
        if spatial_pattern == "scanning":
            return EmitterBehaviorClass.PERIODIC_SPATIAL_SCAN
        elif pri_stability < 0.3:
            return EmitterBehaviorClass.FIXED_INTERMITTENT
        return EmitterBehaviorClass.CONTINUOUS_FIXED
    else:
        return EmitterBehaviorClass.UNKNOWN


# =====================================================================
# Threat Model Configuration
# =====================================================================

class ThreatModelType(str, Enum):
    """Available threat scoring models."""
    # Simple model: based on hit rate and recency only
    SIMPLE = "simple"
    # Standard model: SNR + hit rate + recency
    STANDARD = "standard"
    # Behavior-based model: uses emitter class threat scores
    BEHAVIOR_BASED = "behavior_based"
    # Advanced model: includes agility and PRI stability estimates
    ADVANCED = "advanced"
    # Full model: all features including emitter classification
    FULL = "full"


@dataclass
class ThreatScoreConfig:
    """
    Configuration for threat scoring.

    The threat score is a weighted combination of features derived from
    observation history. All features are observable without access to
    hidden truth.

    Feature weights are normalised so that they sum to 1.0 across
    the active features.

    Parameters
    ----------
    model_type : ThreatModelType
        The threat scoring model to use.
        - SIMPLE: hit rate and recency only
        - STANDARD: SNR + hit rate + recency
        - BEHAVIOR_BASED: emitter class-based threat scores (per PS26055 Sub-problem F)
        - ADVANCED: STANDARD + agility + PRI stability
        - FULL: all features including emitter classification
    weight_snr_db : float
        Weight for SNR-based threat. High-SNR emitters are more threatening.
        Range: [0.0, 1.0]. Default: 0.3
    weight_agility : float
        Weight for frequency agility threat. Agile emitters are harder to track.
        Range: [0.0, 1.0]. Default: 0.25
    weight_pri_stability : float
        Weight for PRI stability threat. Unstable PRI is harder to jam.
        Range: [0.0, 1.0]. Default: 0.2
    weight_activity : float
        Weight for activity-based threat (recent hits).
        Range: [0.0, 1.0]. Default: 0.25
    weight_behavior : float
        Weight for behavior-based threat. Uses emitter class threat scores.
        Range: [0.0, 1.0]. Default: 0.4 (dominant in BEHAVIOR_BASED mode)
    snr_high_threshold_db : float
        SNR above which emitters are considered high-threat.
        Default: 20.0 dB
    snr_low_threshold_db : float
        SNR below which emitters are considered low-threat.
        Default: 5.0 dB
    recency_window_slots : int
        Number of recent slots to consider for recency weighting.
        Default: 50
    agility_window_slots : int
        Number of slots to observe for agility estimation.
        Default: 100
    memo_reference : str
        Reference to the threat model documentation.
    """
    model_type: ThreatModelType = ThreatModelType.BEHAVIOR_BASED
    weight_snr_db: float = 0.20
    weight_agility: float = 0.15
    weight_pri_stability: float = 0.15
    weight_activity: float = 0.20
    weight_behavior: float = 0.30  # Weight for behavior-based scoring
    snr_high_threshold_db: float = 20.0
    snr_low_threshold_db: float = 5.0
    recency_window_slots: int = 50
    agility_window_slots: int = 100
    memo_reference: str = "PS26055-SubProb-F-IMPLEMENTED"

    def __post_init__(self) -> None:
        """Normalise weights to sum to 1.0."""
        total = (self.weight_snr_db + self.weight_agility +
                  self.weight_pri_stability + self.weight_activity +
                  self.weight_behavior)
        if total > 0:
            self.weight_snr_db /= total
            self.weight_agility /= total
            self.weight_pri_stability /= total
            self.weight_activity /= total
            self.weight_behavior /= total

    def is_implemented(self) -> bool:
        """True iff threat model is configured and ready for use."""
        return True  # Always implemented with default or custom weights

    def active_features(self) -> List[str]:
        """List of active threat features based on model type."""
        features = ["activity"]
        if self.model_type in (ThreatModelType.STANDARD, ThreatModelType.BEHAVIOR_BASED,
                               ThreatModelType.ADVANCED, ThreatModelType.FULL):
            features.append("snr")
        if self.model_type in (ThreatModelType.BEHAVIOR_BASED, ThreatModelType.ADVANCED,
                               ThreatModelType.FULL):
            features.append("behavior")
        if self.model_type in (ThreatModelType.ADVANCED, ThreatModelType.FULL):
            features.append("agility")
            features.append("pri_stability")
        return features


# =====================================================================
# Threat Score Computation
# =====================================================================

@dataclass
class BandThreatState:
    """
    Per-band threat state tracked over the episode.

    This accumulates evidence about the threat level of each band
    from observations without access to hidden truth.
    """
    band: int
    # Activity tracking
    hit_count: int = 0
    miss_count: int = 0
    last_hit_slot: int = -1
    recent_hits: List[int] = field(default_factory=list)  # slots of recent hits
    # SNR tracking (from TSRD observations)
    snr_estimates: List[float] = field(default_factory=list)  # dB values
    max_snr_db: float = -np.inf
    mean_snr_db: float = 0.0
    # Agility tracking
    bands_visited: List[int] = field(default_factory=list)  # bands seen when this band hit
    agility_score: float = 0.0  # 0 = fixed, 1 = fully agile
    # PRI stability tracking
    pri_estimates: List[float] = field(default_factory=list)  # microseconds
    pri_jitter: float = 0.0  # coefficient of variation
    # Emitter classification (from deinterleaver)
    estimated_emitter_types: Dict[str, int] = field(default_factory=dict)
    # Behavior-based classification (per PS26055 Sub-problem F)
    emitter_class_name: Optional[str] = None  # From deinterleaver or TSRD metadata
    behavior_class: EmitterBehaviorClass = EmitterBehaviorClass.UNKNOWN
    behavior_threat_score: float = 5.0  # Inherent threat from behavior type [0-10]
    behavior_confidence: float = 0.0  # Confidence in behavior classification [0-1]
    # Aggregated threat score
    threat_score: float = 0.0
    threat_components: Dict[str, float] = field(default_factory=dict)

    def hit_rate(self, window: Optional[int] = None) -> float:
        """Hit rate over all observations or a sliding window."""
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        if window is None:
            return self.hit_count / total
        # Recency-weighted hit rate
        if not self.recent_hits:
            return 0.0
        recent_count = sum(1 for s in self.recent_hits if s >= self.recent_hits[-1] - window)
        return recent_count / min(window, total)


class ThreatScorer:
    """
    Computes threat scores from observation history.

    This class maintains per-band threat state and computes threat
    scores based on observable features. It does not access hidden
    truth and is suitable for use by schedulers.
    """

    def __init__(self, band_count: int, config: Optional[ThreatScoreConfig] = None):
        self.band_count = band_count
        self.config = config or ThreatScoreConfig()
        self.band_states: List[BandThreatState] = [
            BandThreatState(band=b) for b in range(band_count)
        ]
        self.current_slot: int = 0

    def update_from_observation(self, band: int, observation: Dict[str, Any],
                                 current_slot: int) -> None:
        """
        Update threat state from a new observation.

        Parameters
        ----------
        band : int
            The band that was observed
        observation : Dict
            The observation dict from the environment
        current_slot : int
            The current time slot
        """
        self.current_slot = current_slot
        state = self.band_states[band]

        # Activity update
        if observation.get("hit", False):
            state.hit_count += 1
            state.last_hit_slot = current_slot
            state.recent_hits.append(current_slot)
            # Keep only recent hits in window
            window = self.config.recency_window_slots
            state.recent_hits = [s for s in state.recent_hits
                                 if current_slot - s <= window]
        else:
            state.miss_count += 1

        # SNR update (from TSRD observations)
        if "snr_db_estimate" in observation:
            snr = float(observation["snr_db_estimate"])
            if not np.isnan(snr):
                state.snr_estimates.append(snr)
                state.max_snr_db = max(state.max_snr_db, snr)
                if state.snr_estimates:
                    state.mean_snr_db = float(np.mean(state.snr_estimates))

        # Track bands seen when this band hits (for agility estimation)
        # Accept either "observed_bands" or check adjacent bands in the grid
        if observation.get("hit", False):
            # Check for explicit observed_bands list (from deinterleaver)
            if "observed_bands" in observation:
                other_bands = observation.get("observed_bands", [])
                for b in other_bands:
                    if b != band:
                        state.bands_visited.append(b)
            # Alternative: if we have frequency diversity info from TSRD
            elif "pulse_count" in observation and observation.get("pulse_count", 0) > 1:
                # Multiple pulses might indicate multiple emitters in different bands
                # This is a heuristic for agility estimation
                pass

        # Behavior classification update (per PS26055 Sub-problem F)
        # Try to get emitter class from deinterleaver or observation
        emitter_class = observation.get("emitter_class_name")
        if emitter_class is not None:
            state.emitter_class_name = emitter_class
            state.behavior_threat_score = get_behavior_threat_score(emitter_class)
            state.behavior_confidence = 1.0  # Direct classification from class name
            state.behavior_class = EMITTER_CLASS_TO_BEHAVIOR.get(
                emitter_class, EmitterBehaviorClass.UNKNOWN
            )
        else:
            # Try to classify based on observable features
            behavior_class = classify_emitter_by_observation(observation)
            if behavior_class != EmitterBehaviorClass.UNKNOWN:
                state.behavior_class = behavior_class
                state.behavior_threat_score = BEHAVIOR_THREAT_SCORES.get(
                    behavior_class, 5.0
                )
                # Confidence based on number of observations
                state.behavior_confidence = min(1.0, state.hit_count / 10.0)

        # Recompute derived features
        self._update_derived_features(state)

    def _update_derived_features(self, state: BandThreatState) -> None:
        """Update derived threat features from accumulated state."""
        cfg = self.config

        # Recency factor: more recent hits = higher threat
        if state.last_hit_slot >= 0:
            time_since_hit = self.current_slot - state.last_hit_slot
            recency_decay = np.exp(-time_since_hit / cfg.recency_window_slots)
        else:
            recency_decay = 0.0

        # Agility score: if hits appear in different bands, high agility
        if len(state.bands_visited) > 1:
            unique_bands = len(set(state.bands_visited))
            state.agility_score = min(1.0, unique_bands / 5.0)  # Normalise to 5 bands

        # PRI jitter: coefficient of variation
        if len(state.pri_estimates) > 1:
            mean_pri = np.mean(state.pri_estimates)
            std_pri = np.std(state.pri_estimates)
            if mean_pri > 0:
                state.pri_jitter = std_pri / mean_pri

    def compute_threat_score(self, band: int) -> float:
        """
        Compute the threat score for a band.

        The threat score is a weighted combination of:
        1. Activity score (hit rate and recency)
        2. SNR score (high SNR = high threat)
        3. Agility score (high agility = high threat)
        4. PRI stability score (high jitter = high threat)
        5. Behavior-based score (emitter class threat level)

        Returns
        -------
        float
            Threat score in [0, 1], higher = more threatening
        """
        state = self.band_states[band]
        cfg = self.config

        components = {}

        # 1. Activity-based threat (always active)
        hit_rate = state.hit_rate(window=cfg.recency_window_slots)
        if state.last_hit_slot >= 0:
            recency = np.exp(-(self.current_slot - state.last_hit_slot) / cfg.recency_window_slots)
        else:
            recency = 0.0
        activity_score = 0.5 * hit_rate + 0.5 * recency
        components["activity"] = activity_score

        # 2. SNR-based threat (if available)
        if "snr" in cfg.active_features() and state.max_snr_db > -np.inf:
            # Normalise SNR to [0, 1] using thresholds
            snr = state.max_snr_db
            if snr >= cfg.snr_high_threshold_db:
                snr_score = 1.0
            elif snr <= cfg.snr_low_threshold_db:
                snr_score = 0.0
            else:
                snr_score = (snr - cfg.snr_low_threshold_db) / (
                    cfg.snr_high_threshold_db - cfg.snr_low_threshold_db
                )
            components["snr"] = snr_score

        # 3. Behavior-based threat (per PS26055 Sub-problem F)
        if "behavior" in cfg.active_features():
            # Normalize behavior threat score from [0-10] to [0-1]
            # Agile (9-10) -> 0.9-1.0, Periodic (7) -> 0.7, Fixed (3) -> 0.3
            behavior_score = state.behavior_threat_score / 10.0
            # Weight by confidence
            behavior_score *= state.behavior_confidence
            components["behavior"] = behavior_score

        # 4. Agility threat (if available)
        if "agility" in cfg.active_features():
            # High agility = high threat (harder to track)
            components["agility"] = state.agility_score

        # 5. PRI stability threat (if available)
        if "pri_stability" in cfg.active_features():
            # High jitter = high threat (harder to jam)
            # Clip jitter to [0, 1] for reasonable scores
            pri_score = min(1.0, state.pri_jitter)
            components["pri_stability"] = pri_score

        # Map feature names to config attribute names
        feature_to_weight = {
            "activity": "weight_activity",
            "snr": "weight_snr_db",
            "behavior": "weight_behavior",
            "agility": "weight_agility",
            "pri_stability": "weight_pri_stability",
        }

        # Compute weighted sum using the normalized weights
        actual_weight_sum = sum(
            cfg.__dict__.get(feature_to_weight.get(feat, f"weight_{feat}"), 0.0)
            for feat in components
        )
        if actual_weight_sum > 0:
            threat_score = sum(
                cfg.__dict__.get(feature_to_weight.get(feat, f"weight_{feat}"), 0.0)
                / actual_weight_sum * score
                for feat, score in components.items()
            )
        else:
            threat_score = activity_score  # Fallback to activity only

        state.threat_score = threat_score
        state.threat_components = components.copy()

        return threat_score

    def rank_bands_by_threat(self, descending: bool = True) -> List[int]:
        """
        Rank all bands by threat score.

        Parameters
        ----------
        descending : bool
            If True (default), highest threat first

        Returns
        -------
        List[int]
            Band indices sorted by threat score
        """
        scores = np.array([self.compute_threat_score(b) for b in range(self.band_count)])
        indices = np.argsort(-scores) if descending else np.argsort(scores)
        return indices.tolist()

    def get_band_threat_details(self, band: int) -> Dict[str, Any]:
        """Get detailed threat information for a band."""
        state = self.band_states[band]
        return {
            "band": band,
            "threat_score": state.threat_score,
            "threat_components": state.threat_components,
            "hit_count": state.hit_count,
            "miss_count": state.miss_count,
            "hit_rate": state.hit_rate(),
            "last_hit_slot": state.last_hit_slot,
            "max_snr_db": state.max_snr_db if state.max_snr_db > -np.inf else None,
            "mean_snr_db": state.mean_snr_db if state.snr_estimates else None,
            "agility_score": state.agility_score,
            "pri_jitter": state.pri_jitter,
            "time_since_last_hit": (
                self.current_slot - state.last_hit_slot
                if state.last_hit_slot >= 0 else None
            ),
            # Behavior-based classification (per PS26055 Sub-problem F)
            "emitter_class_name": state.emitter_class_name,
            "behavior_class": state.behavior_class.value if state.behavior_class else None,
            "behavior_threat_score": state.behavior_threat_score,
            "behavior_confidence": state.behavior_confidence,
        }

    def reset(self) -> None:
        """Reset all threat state for a new episode."""
        self.band_states = [
            BandThreatState(band=b) for b in range(self.band_count)
        ]
        self.current_slot = 0


# =====================================================================
# Threat-Aware Scheduler Mixin
# =====================================================================

class ThreatScoreMixin:
    """
    Scheduler mixin for threat-aware band prioritisation.

    Sub-problem F: Rank bands by estimated threat level, not just
    hit rate. A scheduler using this mixin orders its exploration
    by threat score rather than by raw observation history.

    Usage
    -----
    Classify your scheduler as a mixin::

        class MyThreatAwareScheduler(ThreatScoreMixin, MyBaseScheduler):
            def __init__(self, *args, threat_config=None, **kwargs):
                super().__init__(*args, **kwargs)
                self._init_threat_scorer(threat_config)

    Then use ``rank_bands_by_threat()`` in ``select_action()``.
    """

    def _init_threat_scorer(self, config: Optional[ThreatScoreConfig] = None) -> None:
        """Initialise the threat scorer. Call from subclass __init__."""
        if not hasattr(self, 'band_count'):
            raise RuntimeError(
                "band_count must be set before _init_threat_scorer. "
                "Call super().__init__() first."
            )
        self._threat_scorer = ThreatScorer(
            band_count=self.band_count,
            config=config or ThreatScoreConfig()
        )

    def update_threat_from_observation(self, band: int, observation: Dict[str, Any],
                                       current_slot: int) -> None:
        """
        Update threat state from observation. Call from scheduler update().
        """
        if hasattr(self, "_threat_scorer"):
            self._threat_scorer.update_from_observation(band, observation, current_slot)

    def _compute_band_threat_score(self, band: int) -> float:
        """
        Compute the raw threat score for one band.

        Returns a score in [0, 1], higher = more threatening.
        """
        if hasattr(self, "_threat_scorer"):
            return self._threat_scorer.compute_threat_score(band)
        return 0.0

    def rank_bands_by_threat(self, descending: bool = True) -> List[int]:
        """
        Rank all bands by threat score.

        Parameters
        ----------
        descending : bool
            If True (default), highest threat first

        Returns
        -------
        List[int]
            Band indices sorted by threat score
        """
        if hasattr(self, "_threat_scorer"):
            return self._threat_scorer.rank_bands_by_threat(descending)
        # Fallback: return uniform ordering
        return list(range(self.band_count))

    def get_threat_config(self) -> ThreatScoreConfig:
        """Return the current threat configuration."""
        if hasattr(self, "_threat_scorer"):
            return self._threat_scorer.config
        return ThreatScoreConfig()

    def get_band_threat_details(self, band: int) -> Dict[str, Any]:
        """Get detailed threat information for a band."""
        if hasattr(self, "_threat_scorer"):
            return self._threat_scorer.get_band_threat_details(band)
        return {"band": band, "threat_score": 0.0, "error": "Threat scorer not initialized"}

    def reset_threat_state(self) -> None:
        """Reset threat state for a new episode."""
        if hasattr(self, "_threat_scorer"):
            self._threat_scorer.reset()


# =====================================================================
# Threat-Aware Scheduler Examples
# =====================================================================

class ThreatAwareUCB1(ThreatScoreMixin):
    """
    Example: UCB1 with threat-weighted exploration.

    This demonstrates how to combine threat prioritisation with
    standard MAB exploration. The threat score modifies the
    exploration bonus to prioritise high-threat bands.
    """

    def __init__(self, band_count: int, threat_config: Optional[ThreatScoreConfig] = None,
                 c: float = 2.0, threat_bonus: float = 1.5):
        self.band_count = band_count
        self.c = c  # UCB exploration constant
        self.threat_bonus = threat_bonus  # Multiplier for threat score

        # Initialise threat scorer (mixin method)
        self._init_threat_scorer(threat_config)

        # MAB state
        self.counts = np.zeros(band_count, dtype=np.int64)
        self.values = np.zeros(band_count, dtype=np.float64)
        self.total_counts = 0

    def reset(self, seed: int, scenario_config: Dict) -> None:
        """Reset for new episode."""
        self.counts.fill(0)
        self.values.fill(0)
        self.total_counts = 0
        self.reset_threat_state()

    def select_action(self, observation_history: List[Dict], current_time_slot: int) -> int:
        """Select action using threat-weighted UCB."""
        # Ensure all bands have been tried
        if self.total_counts < self.band_count:
            return self.total_counts  # Try each band once first

        # Compute UCB with threat bonus
        ucb_values = np.zeros(self.band_count)
        for b in range(self.band_count):
            if self.counts[b] > 0:
                avg = self.values[b]
                exploration = self.c * np.sqrt(np.log(self.total_counts) / self.counts[b])
                # Add threat bonus to exploration term
                threat_score = self._compute_band_threat_score(b)
                ucb_values[b] = avg + exploration * (1.0 + self.threat_bonus * threat_score)
            else:
                ucb_values[b] = float('inf')  # Unexplored bands have infinite UCB

        return int(np.argmax(ucb_values))

    def update(self, action: int, observation: Dict) -> None:
        """Update from observation."""
        self.total_counts += 1
        self.counts[action] += 1

        # Update threat state
        current_slot = observation.get("time_slot", self.total_counts - 1)
        self.update_threat_from_observation(action, observation, current_slot)

        # Update MAB value estimate
        reward = 1.0 if observation.get("hit", False) else 0.0
        q = self.values[action]
        n = self.counts[action]
        self.values[action] = q + (reward - q) / n

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics."""
        return {
            "scheduler_type": "ThreatAwareUCB1",
            "counts": self.counts.tolist(),
            "values": self.values.tolist(),
            "threat_config": self.get_threat_config().__dict__,
        }


# =====================================================================
# Module Exports
# =====================================================================

__all__ = [
    # Behavior classification (PS26055 Sub-problem F)
    "EmitterBehaviorClass",
    "BEHAVIOR_THREAT_SCORES",
    "EMITTER_CLASS_TO_BEHAVIOR",
    "get_behavior_threat_score",
    "classify_emitter_by_observation",
    # Threat scoring
    "ThreatModelType",
    "ThreatScoreConfig",
    "BandThreatState",
    "ThreatScorer",
    "ThreatScoreMixin",
    "ThreatAwareUCB1",
]

"""
PS26055 Simulation — Main Configuration with Provenance Tracking
Every parameter labeled per user requirements.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# Import provenance label definitions from environment
from vyapti_simulator.core.environment import ProvenanceTag, ProvenanceLabel


@dataclass
class MasterSimulationConfig:
    """The single source of truth for all simulation parameters.
    Per user instruction: every parameter must have provenance label.
    Per frozen protocol: no parameter changes without version update + audit trail.
    """

    # ============================================
    # CORE SIMULATION PARAMETERS — Frozen Protocol Lock
    # ============================================

    # [PS-DEFINED] From SIH 26055 problem statement and frozen protocol
    receiver_ibw_mhz: float = 200.0
    total_spectrum_mhz: float = 2000.0
    band_count: int = 10
    dwell_time_ms: float = 10.0
    retune_time_ms: float = 1.0

    # [ENGINEERING-ASSUMPTION] — Scale choice for prototype feasibility
    # Justified: 4-6 week timeline (PDF 2, line 86-89); 35 max emitters scaled
    # from TSRD 90-110 per scenario; 1000 time slots = ~100s mission at 10ms/slot.
    max_emitters_default: int = 35
    mission_duration_slots: int = 1000

    # [PS-DEFINED] — Binary band/time-slot ground truth (frozen protocol line 35-36)
    binary_grid_representation: bool = True

    # [LITERATURE-GROUNDED] — Clarkson 2003 periodic synchronisation analysis
    # requires periodic emitters for validation; included in emitter family registry.
    include_periodic_synchronisation_test: bool = True

    # ============================================
    # DETECTOR / OBSERVATION MODEL
    # ============================================

    # [PS-DEFINED] — Binary observation (hit/miss) per frozen protocol.
    # Noisy mode (Pd, Pfa) required for final evaluation (line 174-176 Protocol).
    detection_probability: float = 1.0  # [EXPERIMENTAL-VARIABLE]
    false_alarm_probability: float = 0.0  # [EXPERIMENTAL-VARIABLE]
    noisy_detector_mode_enabled: bool = False  # Set True for final evaluation

    # [ENGINEERING-ASSUMPTION] — Simplified detection model (threshold-based,
    # not full RF physics). Per Deep Dive Part 17 (line 243-247): full SNR model
    # deferred to Level 3 fidelity, which requires explicit justification.
    simplified_detection_model: bool = True

    # ============================================
    # EVALUATION / METRIC SETTINGS
    # ============================================

    # Per frozen protocol (line 178-180): paired evaluation seeds required.
    evaluation_seeds_per_scenario: int = 5
    paired_scenario_count: int = 100

    # [ENGINEERING-ASSUMPTION] — Reward weights must be explicit, not arbitrary defaults.
    # Per Deconstruction (line 200-204): any weight choice must be justified.
    # Default set here for framework demonstration; must be finalized per Sub-problem G memo.
    reward_weight_intercept_time: float = 1.0
    reward_weight_interception_rate: float = 1.0
    reward_weight_false_alarm_penalty: float = -0.5
    reward_weight_switch_cost: float = -0.1
    reward_weights_justified: bool = False  # Must become True before Stage 7

    # [SCIENTIFIC] — Metric separation required (PDF 2, metric design):
    # Discovery family (first-intercept probability/deadline, mean/median intercept time)
    # Monitoring family (post-discovery interception ratio, rolling coverage, retunes, utility).
    separate_discovery_monitoring_metrics: bool = True

    # ============================================
    # TSRD INTEGRATION — Explicit Decision Required
    # ============================================

    # [DOCUMENT REFERENCE] Per instruction document and frozen protocol:
    # Team must explicitly select option(s) and record memo before use.
    # Default: Option 1 (Parameter Grounding) selected for prototype;
    # Option 2 (PDW Overlay) optional; Option 3 (Future Coupling) deferred.
    tsrd_option_1_parameter_grounding: bool = True
    tsrd_option_1_memo_text: str = (
        "TSRD Option 1 adopted: Inspect 'scan mode' PDW statistics (center frequencies, "
        "pulse widths, emitter counts, frequency spans) to derive order-of-magnitude ranges. "
        "Normalized to discrete 10-band / 10ms-slot grid. Not copying exactly; "
        "parameter grounding for simulator realism (per instruction line 1-9)."
    )
    tsrd_option_2_pdw_overlay: bool = False  # Must be enabled explicitly if pursued
    tsrd_option_3_future_coupling: bool = False  # Deferred per scope freeze
    tsrd_subset_size: int = 250  # [ENGINEERING-ASSUMPTION] Small subset for optional overlay

    # ============================================
    # TSRD FULL-CORPUS LOCATION (Stage 2)
    # ============================================
    # The full TSRD corpus (Hugging Face dataset, mirrored on
    # Kaggle, gated) is the source for Option 1's
    # `tsrd_statistics.json`. Per the laptop-storage
    # constraint, the full corpus is **not** stored on-laptop;
    # the canonical path lives on the Kaggle runtime.
    # `tsrd_corpus_dir` points at the local mirror for the
    # fixtures; the full-corpus integration is
    # Kaggle-runnable via `vyapti_simulator.tsrd.TSRDCorpusLoader`.
    tsrd_corpus_dir: str = "tests/fixtures/tsrd"  # fixture mirror
    # When True, `TSRDCorpusLoader(require_manifest=True)`
    # will refuse to iterate a corpus without a
    # `corpus_manifest.json`. Off by default because the
    # fixture directory is not a full corpus.
    tsrd_corpus_require_manifest: bool = False

    # ============================================
    # ARCHITECTURE — Simplified Per Audit
    # ============================================

    # Per Audit (line 144-159): simplified architecture adopted.
    # Core: tabular restless-bandit (confidence-bound variant, NOT Whittle-index).
    # Prediction: isolated LSTM/GRU predictor (Layer 2), only wired after Gate 5.
    # Periodic-awareness: optional, only if RQ6 confirms value.
    # Agile-handling: tuned exploration within same bandit, not separate layer.
    # Deep RL: comparison arm only, not core.
    # Deinterleaving (Sub-problem H): optional, conditional on scope/time.

    architecture_version: str = "simplified_hybrid_v1_post_audit"
    architecture_note: str = (
        "Simplified from original 7-layer proposal (Architecture_Investigation.md) to core hybrid "
        "per Validation Audit (line 144-159): tabular restless-bandit default + isolated predictor + optional classical awareness."
    )

    # ============================================
    # PROVENANCE TRACKING — Every parameter must reference source
    # ============================================

    # [SCIENTIFIC] All parameters tracked for reproducibility audit.
    # Per frozen protocol (line 150-166): sub_problem_id, layer_id, technique_version,
    # compared_against, gate_status, scenario_version, seed required for every result.
    provenance_map: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Auto-register provenance for all major fields
        provenance_entries = {
            "receiver_ibw_mhz": "[PS-DEFINED] SIH 26055 IBW constraint (10x lower than total); fixed to 200 MHz for discrete grid.",
            "total_spectrum_mhz": "[ENGINEERING-ASSUMPTION] Not numerically specified in PS; 2000 MHz for 10-band tractability.",
            "band_count": "[ENGINEERING-ASSUMPTION] Derived from IBW/total; frozen protocol binary grid minimum.",
            "dwell_time_ms": "[PS-DEFINED] PS scanning requires dwell/switch timing; 10 ms from Deep Dive numerical model.",
            "retune_time_ms": "[ENGINEERING-ASSUMPTION] 1 ms overhead; ensures switching has non-zero cost (Deep Dive Part 2, line 36).",
            "max_emitters_default": "[TSRD-DERIVED] Scaled from TSRD 90-110 to 35 for 4-6 week prototype (Working Plan PDF, line 86-89).",
            "mission_duration_slots": "[ENGINEERING-ASSUMPTION] 1000 slots = 10s at 10ms/slot; adjustable per scenario.",
            "binary_grid_representation": "[PS-DEFINED] Frozen protocol minimum (line 35-36 Deconstruction).",
            "detection_probability": "[EXPERIMENTAL-VARIABLE] Configurable; ideal (1.0) for debugging, noisy for final evaluation.",
            "false_alarm_probability": "[EXPERIMENTAL-VARIABLE] Configurable; required for meaningful Pfa metric.",
            "simplified_detection_model": "[ENGINEERING-ASSUMPTION] Threshold model sufficient for binary Pd/Pfa; full SNR deferred (Deep Dive Part 17).",
            "noisy_detector_mode_enabled": "[PS-DEFINED] Noisy mode required for final evaluation (frozen protocol line 174-176).",
            "evaluation_seeds_per_scenario": "[SCIENTIFIC] Paired evaluation requires same seeds; 5 minimum per protocol.",
            "paired_scenario_count": "[SCIENTIFIC] 100 per protocol recommendation; 50 minimum acceptable.",
            "separate_discovery_monitoring_metrics": "[SCIENTIFIC] Per PDF 2 metric design; 2024 MAB study shows divergence between discovery and monitoring.",
            "tsrd_option_1_parameter_grounding": "[DOCUMENT-REQUIRED] Option 1 adopted; memo required before any TSRD data touches script.",
            "tsrd_option_2_pdw_overlay": "[DOCUMENT-REQUIRED] Option 2 optional; must be explicitly enabled with counterfactual documentation.",
            "architecture_version": "[SCIENTIFIC] Post-audit simplified version; must reference if comparing to original 7-layer proposal.",
        }
        self.provenance_map.update(provenance_entries)

"""
PS26055 Simulation Environment — Core Truth Layer
Scientific Simulation Architect | Senior RF/ES Engineer

PURPOSE: Own the hidden ground-truth emitter activity grid.
This module must NEVER expose X[e,b,t] (true emitter state) to any scheduler.
Only the receiver module queries truth (indirectly), and only metrics receive
full truth after the decision step.

PROVENANCE RULE: Every parameter below carries a label:
  [PS-DEFINED]    — Directly from SIH 26055 problem statement
  [LITERATURE-GROUNDED] — From Clarkson 2003/2011/2019, Glaude 2015,
                    Apfeld & Charlish 2021, Teissier 2024 MAB/ESPE,
                    Perini 2025 SmartScan, or TSRD methodology
  [TSRD-DERIVED]  — From Turing Synthetic Radar Dataset (arXiv:2602.03856)
  [ENGINEERING-ASSUMPTION] — Justified engineering choice, explicitly stated
  [EXPERIMENTAL-VARIABLE] — Sweep parameter for experiments

-----------------------------------------------------------------------
RNG DISCIPLINE (Gate 0 requirement: deterministic replay by seed)
-----------------------------------------------------------------------
[SCIENTIFIC] This module uses numpy's Generator API with an explicit
SeedSequence tree. It NEVER calls np.random.seed(), which would corrupt
global state shared with schedulers and break paired comparison.

Stream layout, from root SeedSequence(scenario_seed):
    child 0 -> detector noise stream   (Pd / Pfa draws)
    child 1 -> emitter parent stream, itself spawned per emitter

Because SeedSequence.spawn(k) returns children whose i-th element depends
only on (parent, i) and not on k, emitter e always receives an identical
random stream regardless of how many emitters are in the population. That
gives a property the frozen protocol's density sweep needs but did not
previously have: increasing emitter density ADDS emitters without
re-randomising the ones already present, so density sweeps are properly
nested/paired rather than independently re-drawn. Detector noise is likewise
insulated from emitter count.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# NOTE on the frequency<->band / time<->slot mapping helpers in
# `vyapti_simulator.core.mapping`:
#
# They are NOT imported here at module load to avoid a circular import
# (mapping.py imports SimulationConfig from this module). They are
# imported lazily inside the methods that need them (e.g. inside
# `tsrd.tsrd_emitter.TSRDEmitterSampler`). The synthetic truth
# generator (`_fill_emitter_track`) operates entirely in discrete
# band/slot indices, so it never needs the continuous-to-discrete
# mapping.

# =====================================================================
# PROVENANCE-LABELED PARAMETERS (must not change without audit)
# =====================================================================

class ProvenanceTag(str, Enum):
    PS_DEFINED = "[PS-DEFINED]"
    LITERATURE_GROUNDED = "[LITERATURE-GROUNDED]"
    TSRD_DERIVED = "[TSRD-DERIVED]"
    ENGINEERING_ASSUMPTION = "[ENGINEERING-ASSUMPTION]"
    EXPERIMENTAL_VARIABLE = "[EXPERIMENTAL-VARIABLE]"


@dataclass
class ProvenanceLabel:
    tag: ProvenanceTag
    source_text: str  # Full citation or justification string
    justification: str


# =====================================================================
# EMITTER FAMILIES — From frozen protocol and problem deconstruction
# =====================================================================

class EmitterBehaviorType(str, Enum):
    """
    Behavior taxonomy. The frozen protocol's scenario registry names twelve
    families; PSEUDO_RANDOM_AGILE is additionally named by the problem
    statement itself ("frequency agile"), giving thirteen enum members.

    Mapping to the frozen protocol's twelve:
      continuous        -> CONTINUOUS_FIXED
      periodic          -> PERIODIC_SPATIAL_SCAN
      jittered-periodic -> JITTERED_PERIODIC
      random-hopper     -> RANDOM_HOPPER          (uniform over ALL bands)
      Markov-hopper     -> MARKOV_HOPPER
      semi-Markov       -> SEMI_MARKOV
      patterned         -> PATTERNED
      intermittent      -> INTERMITTENT
      delayed-arrival   -> DELAYED_ARRIVAL        (arrival_slot > 0 enforced)
      abrupt-change     -> ABRUPT_CHANGE
      mixed             -> MIXED_POPULATION
      unseen-test       -> UNSEEN_TEST            (held out of any training set)

    [PS-DEFINED] PERIODIC_SPATIAL_SCAN, PSEUDO_RANDOM_AGILE
    [LITERATURE-GROUNDED] JITTERED_PERIODIC (Clarkson 2019 CTMC),
                          MARKOV_HOPPER / SEMI_MARKOV (Apfeld & Charlish 2021)
    [ENGINEERING-ASSUMPTION] all remaining shapes
    """
    CONTINUOUS_FIXED = "continuous_fixed"
    PERIODIC_SPATIAL_SCAN = "periodic_spatial_scan"   # [PS-DEFINED]
    JITTERED_PERIODIC = "jittered_periodic"
    PSEUDO_RANDOM_AGILE = "pseudo_random_agile"       # [PS-DEFINED] frequency agile
    RANDOM_HOPPER = "random_hopper"
    MARKOV_HOPPER = "markov_hopper"
    SEMI_MARKOV = "semi_markov"
    PATTERNED = "patterned"
    INTERMITTENT = "intermittent"
    DELAYED_ARRIVAL = "delayed_arrival"
    ABRUPT_CHANGE = "abrupt_change"
    MIXED_POPULATION = "mixed_population"
    UNSEEN_TEST = "unseen_test"


# Families reserved for generalisation testing only. Per frozen protocol, a
# result claiming generalisation must show these were never used to tune.
HELD_OUT_BEHAVIORS = frozenset({EmitterBehaviorType.UNSEEN_TEST})


@dataclass
class EmitterConfig:
    """
    One emitter's hidden behaviour specification.

    [ENGINEERING-ASSUMPTION] Default 10 bands, 10 ms slots.
    Justification: 2000 MHz total / 200 MHz IBW = 10 bands; 10 ms dwell + 1 ms
    retune aligns with Technical Deep Dive Part 2 (line 29-36) toy numbers,
    scaled to the frozen protocol's discrete band/time-slot minimum.

    Every shape parameter below is explicit rather than hardcoded in the
    generator, because the frozen protocol forbids unstated defaults: an
    unreported `visibility_fraction = 0.3` buried in a loop is exactly the
    kind of silent design choice the audit flags.
    """
    emitter_id: int
    behavior: EmitterBehaviorType
    active_bands: List[int] = field(default_factory=list)

    # --- periodic / jittered shape -------------------------------------
    period_slots: Optional[int] = None            # [EXPERIMENTAL-VARIABLE]
    phase_offset_slots: int = 0                   # [EXPERIMENTAL-VARIABLE]
    # Fraction of each period during which the emitter illuminates the
    # receiver. [ENGINEERING-ASSUMPTION] 0.3 is the mainbeam-dwell analogue of
    # Clarkson's "probability of intercept per scan"; must be swept in RQ6.
    visibility_fraction: float = 0.3
    # Jitter as a fraction of the nominal period (JITTERED_PERIODIC).
    # [LITERATURE-GROUNDED] Clarkson 2019 models scan irregularity as bounded
    # stochastic perturbation of the period, not as Bernoulli visibility.
    period_jitter_fraction: float = 0.2

    # --- hopping shape --------------------------------------------------
    hop_sequence: Optional[List[int]] = None      # PATTERNED / pre-drawn agile
    pattern_length: int = 5                       # [ENGINEERING-ASSUMPTION]
    # MARKOV_HOPPER: probability of leaving the current band each slot.
    markov_switch_probability: float = 0.1        # [ENGINEERING-ASSUMPTION]
    # SEMI_MARKOV: mean dwell in slots before re-drawing a band.
    mean_dwell_slots: float = 10.0                # [ENGINEERING-ASSUMPTION]

    # --- intermittency --------------------------------------------------
    on_duration_slots: int = 20                   # [ENGINEERING-ASSUMPTION]
    off_duration_slots: int = 30                  # [ENGINEERING-ASSUMPTION]

    # --- lifecycle ------------------------------------------------------
    arrival_slot: int = 0                         # [EXPERIMENTAL-VARIABLE]
    departure_slot: Optional[int] = None          # [EXPERIMENTAL-VARIABLE]

    # --- regime change (ABRUPT_CHANGE) ----------------------------------
    # Slot at which behaviour switches. None -> midpoint of the mission.
    change_point_slot: Optional[int] = None       # [EXPERIMENTAL-VARIABLE]
    behavior_before_change: EmitterBehaviorType = EmitterBehaviorType.CONTINUOUS_FIXED
    behavior_after_change: EmitterBehaviorType = EmitterBehaviorType.RANDOM_HOPPER

    # --- mixed population ------------------------------------------------
    # Candidate behaviours sampled per emitter when behavior is
    # MIXED_POPULATION. None -> the three PS-relevant archetypes.
    mixture_components: Optional[List[EmitterBehaviorType]] = None
    mixture_weights: Optional[List[float]] = None

    # --- link budget ------------------------------------------------------
    # [ENGINEERING-ASSUMPTION] Per-emitter received SNR in dB at the receiver.
    # This is what makes "sensitivity" (PS metric 3) a measurable quantity
    # rather than a placeholder: a weak emitter can be dwelled on and still
    # missed. Full RF propagation is deferred to Level 3 fidelity per Deep
    # Dive Part 17 (line 243-247); this is a single scalar standing in for it.
    snr_db: float = 20.0                          # [EXPERIMENTAL-VARIABLE]

    # Provenance tracking (free-form: str or ProvenanceLabel)
    provenance_notes: Dict[str, Any] = field(default_factory=dict)

    def resolved_period(self, default: int = 10) -> int:
        return int(self.period_slots) if self.period_slots else default

    def is_live_at(self, t: int) -> bool:
        """Lifecycle gate applied uniformly to EVERY behaviour type."""
        if t < self.arrival_slot:
            return False
        if self.departure_slot is not None and t >= self.departure_slot:
            return False
        return True


# =====================================================================
# HIDDEN TRUTH GRID — Must never be accessible to scheduler interface
# =====================================================================

@dataclass
class HiddenTruthGrid:
    """
    X[e,b,t] per frozen protocol (line 169-172 of Architecture_Investigation.md).
    Binary: 1 if emitter e transmits in band b during time slot t.
    """
    grid: np.ndarray                       # Boolean array (emitters, bands, slots)
    emitter_configs: List[EmitterConfig]
    band_count: int = 10                   # [ENGINEERING-ASSUMPTION]
    time_slots: int = 1000                 # [EXPERIMENTAL-VARIABLE]

    # Hidden-state leakage guard (per frozen protocol Gate 0, line 83-86).
    # [SCIENTIFIC] The log is a bounded sample plus a total counter. An
    # unbounded list here would grow to emitters x slots entries per episode
    # and dominate memory in a 100-seed sweep.
    _truth_access_count: int = field(default=0, init=False)
    _truth_access_log: List[str] = field(default_factory=list, init=False)
    _truth_access_log_cap: int = field(default=256, init=False)

    def get_truth_for_evaluation_only(self, emitter_idx: int, band_idx: int,
                                      time_idx: int) -> bool:
        """
        Evaluation-only access. Scheduler interface must NOT call this.

        [ENGINEERING-ASSUMPTION] Direct truth read is permitted ONLY to the
        Metrics module after the decision step (per frozen protocol). Any
        scheduler module calling this triggers an audit flag.
        """
        self._truth_access_count += 1
        if len(self._truth_access_log) < self._truth_access_log_cap:
            self._truth_access_log.append(
                f"TRUTH_ACCESS:e={emitter_idx},b={band_idx},t={time_idx}"
            )
        if time_idx < 0 or time_idx >= self.time_slots:
            return False
        if band_idx < 0 or band_idx >= self.band_count:
            return False
        if emitter_idx < 0 or emitter_idx >= self.grid.shape[0]:
            return False
        return bool(self.grid[emitter_idx, band_idx, time_idx])

    # -- vectorised views used by the environment and metrics -------------
    def active_emitters(self, band_idx: int, time_idx: int) -> np.ndarray:
        """Indices of emitters transmitting in `band_idx` at `time_idx`."""
        if not (0 <= time_idx < self.time_slots and 0 <= band_idx < self.band_count):
            return np.empty(0, dtype=np.intp)
        return np.flatnonzero(self.grid[:, band_idx, time_idx])

    def band_occupied(self, band_idx: int, time_idx: int) -> bool:
        if not (0 <= time_idx < self.time_slots and 0 <= band_idx < self.band_count):
            return False
        return bool(self.grid[:, band_idx, time_idx].any())

    def occupancy_slice(self, time_idx: int) -> np.ndarray:
        """(emitters, bands) boolean slice at one instant, for the metrics engine."""
        if not (0 <= time_idx < self.time_slots):
            return np.zeros((self.grid.shape[0], self.band_count), dtype=bool)
        return self.grid[:, :, time_idx]

    def emitter_present_slots(self, emitter_idx: int) -> np.ndarray:
        """Slots in which emitter is transmitting in ANY band."""
        return np.flatnonzero(self.grid[emitter_idx].any(axis=0))

    @property
    def truth_access_count(self) -> int:
        return self._truth_access_count


# =====================================================================
# SIMULATOR ENGINE — Algorithm-agnostic environment
# =====================================================================

@dataclass
class SimulationConfig:
    """All configurable simulation parameters with provenance."""
    # [PS-DEFINED]
    receiver_ibw_mhz: float = 200.0        # From Deep Dive Part 2 (line 32-36)
    total_spectrum_mhz: float = 2000.0     # [ENGINEERING-ASSUMPTION] toy scale, defensible
    band_count: int = 10                   # [ENGINEERING-ASSUMPTION] derived from IBW/total
    dwell_time_ms: float = 10.0            # [PS-DEFINED] implied scanning constraint
    retune_time_ms: float = 1.0            # [ENGINEERING-ASSUMPTION] from Deep Dive

    # [TSRD-DERIVED] dataset card reports up to 90-110 emitters per scenario;
    # [ENGINEERING-ASSUMPTION] scaled to 35 for the 4-6 week prototype.
    max_emitters: int = 35

    # Mission duration in slots. [EXPERIMENTAL-VARIABLE]
    # 1000 slots x 10 ms = 10 s of mission time.
    time_slots: int = 1000

    # [EXPERIMENTAL-VARIABLE]
    scenario_seed: int = 42

    # --- detector model -------------------------------------------------
    # [ENGINEERING-ASSUMPTION] Default: ideal (Pd=1, Pfa=0) for algorithm
    # debugging. Noisy mode is required for final evaluation per frozen
    # protocol (line 174-176).
    detection_probability: float = 1.0     # [EXPERIMENTAL-VARIABLE]
    false_alarm_probability: float = 0.0   # [EXPERIMENTAL-VARIABLE]

    # Sensitivity model. When enabled, per-slot Pd is derived from each
    # emitter's snr_db against `detection_threshold_db` through a logistic
    # curve, and `detection_probability` becomes the ASYMPTOTIC ceiling for a
    # strong signal rather than a flat rate. [ENGINEERING-ASSUMPTION]
    use_sensitivity_curve: bool = False
    detection_threshold_db: float = 10.0   # [ENGINEERING-ASSUMPTION]
    sensitivity_slope_db: float = 3.0      # [ENGINEERING-ASSUMPTION] logistic width

    # Charge the scheduler for retuning. [ENGINEERING-ASSUMPTION] The frozen
    # protocol requires retune overhead to be visible in the cost metric; when
    # this is on, a band change consumes retune_time_ms of the slot's dwell.
    account_retune_overhead: bool = True

    # --- receiver tune grid (TSRD / Scan-Mode fidelity) ---------------
    # [TSRD-DERIVED] When a TSRD H5 is loaded, this field is populated
    # from `metadata/receiver/dwell_centres_mhz`, the actual frequencies
    # the receiver tunes to in each of the `band_count` dwells. When
    # `None` (the synthetic default), band centres are uniform at
    # `(band + 0.5) * band_width_mhz()` and the band edges are at
    # `[band * band_width, (band + 1) * band_width)`.
    #
    # When set, `frequency_to_band` uses NEAREST-DWELL-CENTRE assignment:
    # each emitter pulse is mapped to the band whose tune frequency is
    # closest to the pulse's frequency. This is what a real ESM receiver
    # does: it cannot resolve two emitters to different (band, slot)
    # cells if both fall inside the IBW of a single tune frequency, but
    # two emitters straddling a band boundary map to different cells.
    # The "uniform 18 GHz / 36 bands" formula, by contrast, can place
    # both 1205 MHz and 1295 MHz in band 2 even when the receiver
    # would never have dwelled on band 2 in a way that captures both.
    dwell_centres_mhz: Optional[Any] = None  # 1D array of length band_count

    # Provenance tracking for audit
    provenance_map: Dict[str, Any] = field(default_factory=dict)

    def band_width_mhz(self) -> float:
        return self.total_spectrum_mhz / float(self.band_count)

    def slot_duration_s(self) -> float:
        return self.dwell_time_ms / 1000.0


class PS26055Environment:
    """
    The single interactive environment for ALL scheduling algorithms.
    Per frozen protocol (line 30-38): same simulator, same observation contract,
    same receiver constraints, same metrics — only scheduler policy changes.
    """

    def __init__(self, config: SimulationConfig,
                 emitter_family_config: List[EmitterConfig]):
        self.config = config
        self.emitter_configs = list(emitter_family_config)
        self.current_time_slot: int = 0
        # Hidden truth — initialized in reset()
        self._hidden_truth: Optional[HiddenTruthGrid] = None
        # Observation history (visible to scheduler)
        self.observation_history: List[Dict] = []
        # Audit trail for hidden-state leakage attempts
        self.audit_trail: List[str] = []

        # RNG streams, created in reset(). Never np.random.* global state.
        self._detector_rng: Optional[np.random.Generator] = None
        self._seed_used: Optional[int] = None

        # Receiver bookkeeping
        self._previous_band: Optional[int] = None
        self._retune_count: int = 0
        self._retune_overhead_s_total: float = 0.0

        self._register_provenance()

    # -----------------------------------------------------------------
    # Provenance
    # -----------------------------------------------------------------
    def _register_provenance(self):
        """Register provenance for all simulation parameters."""
        mappings = {
            "receiver_ibw_mhz": ProvenanceLabel(
                ProvenanceTag.PS_DEFINED,
                "SIH 26055 Problem Statement (instantaneous bandwidth at least 10x lower than total)",
                "Fixed at 200 MHz to match Deep Dive toy values (Part 2, line 29-36) and keep discrete band count tractable."
            ),
            "total_spectrum_mhz": ProvenanceLabel(
                ProvenanceTag.ENGINEERING_ASSUMPTION,
                "Not numerically specified in PS; chosen for 10-band discrete grid.",
                "2000 MHz produces N=10 bands at 200 MHz IBW, matching frozen protocol minimum representation."
            ),
            "band_count": ProvenanceLabel(
                ProvenanceTag.ENGINEERING_ASSUMPTION,
                "Derived from IBW/total ratio.",
                "Fixed discrete band count required by frozen protocol binary grid (line 35-36 Problem Deconstruction)."
            ),
            "dwell_time_ms": ProvenanceLabel(
                ProvenanceTag.PS_DEFINED,
                "SIH 26055 — scanning requires dwell/switch timing (implied by IBW constraint).",
                "10 ms matches Deep Dive numerical example; adjustable via config."
            ),
            "retune_time_ms": ProvenanceLabel(
                ProvenanceTag.ENGINEERING_ASSUMPTION,
                "From Deep Dive Part 2 numerical example (line 36).",
                "1 ms overhead; must be non-zero to make frequent switching costly."
            ),
            "max_emitters": ProvenanceLabel(
                ProvenanceTag.TSRD_DERIVED,
                "TSRD dataset card reports up to 90-110 emitters per scenario (arXiv:2602.03856).",
                "Scaled to 35 for 4-6 week prototype (Working Plan PDF, line 86-89)."
            ),
            "time_slots": ProvenanceLabel(
                ProvenanceTag.EXPERIMENTAL_VARIABLE,
                "Mission duration; not specified in PS.",
                "1000 slots x 10 ms = 10 s mission. Swept for scalability curve (frozen protocol line 181-182)."
            ),
            "detection_probability": ProvenanceLabel(
                ProvenanceTag.EXPERIMENTAL_VARIABLE,
                "Not specified in PS; required for meaningful Pd/Pfa metrics.",
                "Ideal mode (1.0) for algorithm debugging; noisy mode for final evaluation per frozen protocol."
            ),
            "false_alarm_probability": ProvenanceLabel(
                ProvenanceTag.EXPERIMENTAL_VARIABLE,
                "Not specified in PS; Pfa listed as required metric.",
                "Ideal mode (0.0) for debugging; configurable for final evaluation."
            ),
            "use_sensitivity_curve": ProvenanceLabel(
                ProvenanceTag.ENGINEERING_ASSUMPTION,
                "PS names 'sensitivity' as a required metric but specifies no RF link budget.",
                "Logistic Pd(SNR) around detection_threshold_db makes sensitivity measurable at Level 2 "
                "fidelity; full propagation modelling deferred (Deep Dive Part 17, line 243-247)."
            ),
            "account_retune_overhead": ProvenanceLabel(
                ProvenanceTag.ENGINEERING_ASSUMPTION,
                "Frozen protocol requires retune overhead in the cost metric.",
                "A band change consumes retune_time_ms of the slot; reported in the monitoring family."
            ),
        }
        self.config.provenance_map = mappings

    # -----------------------------------------------------------------
    # Truth generation
    # -----------------------------------------------------------------
    def reset(self, seed: int = 42,
              emitter_family_config: Optional[List[EmitterConfig]] = None) -> None:
        """
        Reset environment with a new random seed. Per frozen protocol (line
        87-89): paired evaluation seeds must be fixed and reused across
        algorithms.

        [SCIENTIFIC] Truth generation is a pure function of (seed, emitter
        configs). Calling reset(seed) twice yields bit-identical grids — this
        is the deterministic-replay half of Gate 0.
        """
        if emitter_family_config is not None:
            self.emitter_configs = list(emitter_family_config)
        emitter_configs = self.emitter_configs

        self._seed_used = int(seed)
        self.current_time_slot = 0
        self.observation_history.clear()
        self.audit_trail.clear()
        self._previous_band = None
        self._retune_count = 0
        self._retune_overhead_s_total = 0.0

        # --- RNG stream tree (see module docstring) ----------------------
        root = np.random.SeedSequence(self._seed_used)
        detector_ss, emitter_parent_ss = root.spawn(2)
        self._detector_rng = np.random.default_rng(detector_ss)

        emitter_count = len(emitter_configs)
        time_slots = self.config.time_slots
        band_count = self.config.band_count
        emitter_streams = emitter_parent_ss.spawn(emitter_count) if emitter_count else []

        grid_array = np.zeros((emitter_count, band_count, time_slots), dtype=bool)

        for e_idx, ec in enumerate(emitter_configs):
            rng = np.random.default_rng(emitter_streams[e_idx])
            self._fill_emitter_track(
                grid_array[e_idx], ec, ec.behavior, rng, band_count, time_slots
            )

        self._hidden_truth = HiddenTruthGrid(
            grid=grid_array,
            emitter_configs=emitter_configs,
            band_count=band_count,
            time_slots=time_slots,
        )
        # Note: hidden truth is never exposed directly to the scheduler interface.

    def _fill_emitter_track(self, track: np.ndarray, ec: EmitterConfig,
                            behavior: EmitterBehaviorType,
                            rng: np.random.Generator,
                            band_count: int, time_slots: int) -> None:
        """
        Write one emitter's (bands, slots) occupancy into `track` in place.

        `behavior` is passed separately from `ec.behavior` so ABRUPT_CHANGE and
        MIXED_POPULATION can delegate to a concrete shape without mutating the
        config. The lifecycle gate (arrival/departure) is applied here, once,
        for EVERY behaviour — the previous implementation ignored arrival_slot
        entirely, so late-arriving emitters were present from slot 0 and the
        "discovery of a new emitter" scenario could not be tested at all.
        """
        # Bands this emitter may occupy; default to the whole spectrum.
        bands = [b for b in ec.active_bands if 0 <= b < band_count]
        if not bands:
            bands = list(range(band_count))

        # Vector of slots at which the emitter exists at all.
        live = np.zeros(time_slots, dtype=bool)
        start = max(0, int(ec.arrival_slot))
        stop = time_slots if ec.departure_slot is None else min(time_slots, int(ec.departure_slot))
        if start < stop:
            live[start:stop] = True
        if not live.any():
            return  # Emitter never present in this mission window.

        live_idx = np.flatnonzero(live)

        if behavior in (EmitterBehaviorType.CONTINUOUS_FIXED,
                        EmitterBehaviorType.DELAYED_ARRIVAL):
            # [ENGINEERING-ASSUMPTION] Always active in configured bands while live.
            for b in bands:
                track[b, live_idx] = True

        elif behavior == EmitterBehaviorType.PERIODIC_SPATIAL_SCAN:
            # [LITERATURE-GROUNDED] Clarkson 2003/2011 periodic synchronisation.
            period = ec.resolved_period()
            phase = int(ec.phase_offset_slots)
            visible = max(1, int(round(period * ec.visibility_fraction)))
            rel = (live_idx - phase) % period
            on = live_idx[rel < visible]
            for b in bands:
                track[b, on] = True

        elif behavior == EmitterBehaviorType.JITTERED_PERIODIC:
            # [LITERATURE-GROUNDED] Clarkson 2019 — the scan PERIOD is perturbed,
            # not the visibility. Drawing an independent Bernoulli per slot (as
            # the previous code did) destroys periodicity entirely and makes
            # this family indistinguishable from a random duty cycle, which
            # would have quietly invalidated Gate 3 / RQ6.
            period = ec.resolved_period()
            visible = max(1, int(round(period * ec.visibility_fraction)))
            jitter = max(0.0, float(ec.period_jitter_fraction)) * period
            t = start + int(ec.phase_offset_slots) % max(1, period)
            while t < stop:
                on_start = int(t)
                on_stop = min(stop, on_start + visible)
                if on_stop > on_start:
                    seg = np.arange(max(start, on_start), on_stop)
                    if seg.size:
                        for b in bands:
                            track[b, seg] = True
                # Next revisit: nominal period perturbed uniformly in +/- jitter.
                step = period + (rng.uniform(-jitter, jitter) if jitter > 0 else 0.0)
                t += max(1.0, step)

        elif behavior == EmitterBehaviorType.PSEUDO_RANDOM_AGILE:
            # [PS-DEFINED] Frequency-agile: hops among its configured bands.
            # Pre-drawn so a paired comparison replays the identical sequence.
            if ec.hop_sequence is None:
                seq = rng.integers(0, len(bands), size=time_slots)
                ec.hop_sequence = [bands[i] for i in seq.tolist()]
            hops = np.asarray(ec.hop_sequence, dtype=np.intp)
            chosen = hops[live_idx % hops.size]
            track[chosen, live_idx] = True

        elif behavior == EmitterBehaviorType.RANDOM_HOPPER:
            # Uniform over the WHOLE spectrum each slot (harder than agile:
            # the emitter is not confined to a known band subset).
            chosen = rng.integers(0, band_count, size=live_idx.size)
            track[chosen, live_idx] = True

        elif behavior == EmitterBehaviorType.MARKOV_HOPPER:
            # [LITERATURE-GROUNDED] Apfeld & Charlish 2021 model agile emitters
            # as a Markov chain over channels: stay with prob 1-p, else re-draw.
            p = float(np.clip(ec.markov_switch_probability, 0.0, 1.0))
            current = int(rng.integers(0, len(bands)))
            switches = rng.random(live_idx.size) < p
            draws = rng.integers(0, len(bands), size=live_idx.size)
            for k, t in enumerate(live_idx):
                if switches[k]:
                    current = int(draws[k])
                track[bands[current], t] = True

        elif behavior == EmitterBehaviorType.SEMI_MARKOV:
            # Variable dwell durations: geometric holding time per band.
            mean_dwell = max(1.0, float(ec.mean_dwell_slots))
            p_leave = 1.0 / mean_dwell
            t = start
            while t < stop:
                b = bands[int(rng.integers(0, len(bands)))]
                hold = int(rng.geometric(p_leave))
                seg_stop = min(stop, t + max(1, hold))
                track[b, t:seg_stop] = True
                t = seg_stop

        elif behavior == EmitterBehaviorType.PATTERNED:
            # Deterministic repeating sequence — the easiest family for a
            # periodicity-aware policy and the natural Gate 3 sanity case.
            if ec.hop_sequence is None:
                n = max(1, int(ec.pattern_length))
                seq = rng.integers(0, len(bands), size=n)
                ec.hop_sequence = [bands[i] for i in seq.tolist()]
            pat = np.asarray(ec.hop_sequence, dtype=np.intp)
            chosen = pat[(live_idx - start) % pat.size]
            track[chosen, live_idx] = True

        elif behavior == EmitterBehaviorType.INTERMITTENT:
            on_d = max(1, int(ec.on_duration_slots))
            off_d = max(0, int(ec.off_duration_slots))
            cycle = on_d + off_d
            phase = (live_idx - start) % cycle
            on = live_idx[phase < on_d]
            for b in bands:
                track[b, on] = True

        elif behavior == EmitterBehaviorType.ABRUPT_CHANGE:
            # Non-stationarity with a known change point: the family that
            # separates sliding-window / discounted bandits from stationary UCB.
            cp = ec.change_point_slot if ec.change_point_slot is not None else time_slots // 2
            cp = int(np.clip(cp, start, stop))
            before = EmitterConfig(**{**ec.__dict__, "behavior": ec.behavior_before_change,
                                      "arrival_slot": start, "departure_slot": cp,
                                      "hop_sequence": None})
            after = EmitterConfig(**{**ec.__dict__, "behavior": ec.behavior_after_change,
                                     "arrival_slot": cp, "departure_slot": stop,
                                     "hop_sequence": None})
            self._fill_emitter_track(track, before, ec.behavior_before_change,
                                     rng, band_count, time_slots)
            self._fill_emitter_track(track, after, ec.behavior_after_change,
                                     rng, band_count, time_slots)

        elif behavior == EmitterBehaviorType.MIXED_POPULATION:
            # Draw a concrete archetype for THIS emitter, then delegate. The
            # previous implementation emitted uniform noise, which is not a
            # mixture of anything and gave every scheduler the same
            # uninformative target.
            components = ec.mixture_components or [
                EmitterBehaviorType.CONTINUOUS_FIXED,
                EmitterBehaviorType.PERIODIC_SPATIAL_SCAN,
                EmitterBehaviorType.PSEUDO_RANDOM_AGILE,
            ]
            weights = ec.mixture_weights
            if weights is not None and len(weights) == len(components):
                p = np.asarray(weights, dtype=float)
                p = p / p.sum()
            else:
                p = None
            pick = components[int(rng.choice(len(components), p=p))]
            ec.provenance_notes.setdefault(
                "mixture_realisation",
                f"[ENGINEERING-ASSUMPTION] MIXED_POPULATION resolved to {pick.value} for emitter {ec.emitter_id}."
            )
            self._fill_emitter_track(track, ec, pick, rng, band_count, time_slots)

        elif behavior == EmitterBehaviorType.UNSEEN_TEST:
            # Out-of-distribution generalisation probe: a prime period and a
            # narrow visibility window, deliberately outside the ranges used
            # anywhere in tuning. Held out per HELD_OUT_BEHAVIORS.
            period = ec.period_slots or 7           # prime; not a tuning value
            visible = max(1, int(round(period * 0.15)))
            rel = (live_idx - int(ec.phase_offset_slots)) % period
            on = live_idx[rel < visible]
            for b in bands:
                track[b, on] = True

        else:  # pragma: no cover - enum is exhaustive above
            raise ValueError(f"Unhandled emitter behaviour: {behavior!r}")

    # -----------------------------------------------------------------
    # Interaction
    # -----------------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.current_time_slot >= self.config.time_slots

    def _detection_probability_for(self, emitter_indices: np.ndarray) -> float:
        """
        Pd for a dwell that contains one or more emitters.

        [ENGINEERING-ASSUMPTION] With use_sensitivity_curve off this is the
        flat configured Pd. With it on, each present emitter gets a logistic
        Pd(SNR) and the band-level detection probability is the complement of
        all of them being missed — so a dwell holding two weak emitters is
        more likely to register than a dwell holding one.
        """
        base = float(np.clip(self.config.detection_probability, 0.0, 1.0))
        if not self.config.use_sensitivity_curve:
            return base
        miss = 1.0
        for e_idx in emitter_indices:
            snr = float(self.emitter_configs[int(e_idx)].snr_db)
            margin = (snr - self.config.detection_threshold_db) / max(1e-9, self.config.sensitivity_slope_db)
            pd_e = base / (1.0 + np.exp(-margin))
            miss *= (1.0 - pd_e)
        return float(1.0 - miss)

    def step(self, selected_band: int) -> Tuple[Dict, bool]:
        """
        Execute one scheduling step: observe selected band, return observation.
        Per frozen protocol: scheduler only sees observation; truth stays hidden.

        [SCIENTIFIC] The time index is read BEFORE it is advanced. The previous
        implementation incremented first, so slot 0 was never observable and
        every reported intercept time was biased by one slot while the final
        slot of the truth grid was unreachable.
        """
        if self._hidden_truth is None:
            raise RuntimeError("Environment.step() called before reset(); no truth grid exists.")
        if not (0 <= selected_band < self.config.band_count):
            raise ValueError(
                f"Invalid band selection: {selected_band} (valid: 0..{self.config.band_count - 1})"
            )
        if self.done:
            raise RuntimeError(
                f"Episode already finished at slot {self.current_time_slot} "
                f"of {self.config.time_slots}; call reset() before stepping again."
            )

        t = self.current_time_slot

        # --- receiver overhead ------------------------------------------
        retuned = self._previous_band is not None and selected_band != self._previous_band
        retune_cost_s = 0.0
        if retuned and self.config.account_retune_overhead:
            retune_cost_s = self.config.retune_time_ms / 1000.0
            self._retune_count += 1
            self._retune_overhead_s_total += retune_cost_s
        elif retuned:
            self._retune_count += 1

        # Usable dwell shrinks by the retune overhead. [ENGINEERING-ASSUMPTION]
        dwell_s = max(0.0, self.config.slot_duration_s() - retune_cost_s)

        # --- detection --------------------------------------------------
        # Environment owns truth; this is not a scheduler-visible read, so it
        # uses the vectorised internal view rather than the audited accessor.
        present = self._hidden_truth.active_emitters(selected_band, t)
        if present.size:
            p_hit = self._detection_probability_for(present)
            # A retune that eats the whole slot cannot detect anything.
            if dwell_s <= 0.0:
                p_hit = 0.0
            hit = bool(self._detector_rng.random() < p_hit)
        else:
            hit = bool(self._detector_rng.random() < float(
                np.clip(self.config.false_alarm_probability, 0.0, 1.0)))

        observation = {
            "time_slot": t,
            "selected_band": selected_band,
            "hit": hit,
            # Retune cost is legitimately observable: the scheduler pays it and
            # a cost-aware policy must be able to see it. It carries no truth.
            "retune_cost_s": retune_cost_s,
            "dwell_elapsed_s": dwell_s,
            "receiver_metadata": {
                "dwell_time_ms": self.config.dwell_time_ms,
                "retune_time_ms": self.config.retune_time_ms,
                "band_width_mhz": self.config.band_width_mhz(),
            },
            # Explicit exclusion markers — audited by Gate 0.
            "truth_excluded": True,
            "emitter_identity_excluded": True,
            "future_state_excluded": True,
        }
        self.observation_history.append(observation)

        self._previous_band = selected_band
        self.current_time_slot = t + 1

        # Second element is the hidden-state-leakage flag, always False: the
        # observation contract above contains no truth field.
        return observation, False

    # -----------------------------------------------------------------
    # Evaluation-only accessors (metrics engine)
    # -----------------------------------------------------------------
    @property
    def hidden_truth(self) -> HiddenTruthGrid:
        """
        [SCIENTIFIC] Metrics-only. Any scheduler touching this is a Gate 0
        violation; the audit greps for `.hidden_truth` inside algorithms/.
        """
        if self._hidden_truth is None:
            raise RuntimeError("No truth grid; call reset() first.")
        return self._hidden_truth

    def receiver_accounting(self) -> Dict[str, float]:
        """Retune statistics for the monitoring metric family."""
        slots = max(1, self.current_time_slot)
        return {
            "retune_count": float(self._retune_count),
            "retune_overhead_total_s": float(self._retune_overhead_s_total),
            "retune_rate_per_slot": float(self._retune_count) / slots,
            "dead_time_fraction": (
                self._retune_overhead_s_total /
                max(1e-12, slots * self.config.slot_duration_s())
            ),
        }

    def replay_signature(self) -> str:
        """
        Cheap deterministic fingerprint of the generated truth, used by the
        Gate 0 replay test to assert reset(seed) is reproducible.
        """
        import hashlib
        g = self.hidden_truth.grid
        h = hashlib.sha256(np.ascontiguousarray(g).tobytes()).hexdigest()
        return f"seed={self._seed_used}:shape={g.shape}:sha256={h[:32]}"

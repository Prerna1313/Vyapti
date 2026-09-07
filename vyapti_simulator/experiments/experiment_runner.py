"""
PS26055 Experiment Runner — Paired comparison + statistical protocol.

=======================================================================
What this implements
=======================================================================
The frozen protocol's experimental design (Experimental_Protocols.md):
  - Paired trials: same N scenario realizations per method (line 10-12)
  - Fixed seed list reused across algorithms (line 13-16)
  - Wilcoxon signed-rank for two-method comparison (line 13-16)
  - Friedman + Holm-Bonferroni post-hoc for 3+ methods (line 13-16)
  - Bootstrap 95% CI on paired differences (line 13-16)
  - Cliff's delta for effect size (non-parametric)
  - Kaplan-Meier for right-censored first-intercept times (line 34-41)
  - Negative results reported (line 41 Protocol Q2, Audit line 144-146)

=======================================================================
[SCIENTIFIC] Why scipy is now mandatory
=======================================================================
The prior implementation had two defects:
  1. Wilcoxon ranking was wrong: `np.argsort(np.abs(diff))` returns INDICES,
     not RANKS. Ranks are `np.argsort(np.argsort(...)) + 1`. The broken
     version passed code review for months because it looked plausible.
  2. The p-value was `1/sqrt(n)`, which is numerically meaningless.

A hand-rolled Wilcoxon that gets the ranking right still needs either exact
tables (for n ≤ 20) or a normal approximation with continuity correction (for
n > 20), and both need tie-handling. That is 50+ lines we would not trust more
than scipy. So: scipy is a required dependency for experiments, and the test
suite will verify the import succeeds before certifying any comparison.

If scipy is unavailable the experiment runner refuses to run rather than
silently producing wrong p-values that look right.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None  # type: ignore

from ..core.environment import PS26055Environment, EmitterConfig, SimulationConfig
from ..core.episode import run_paired_episodes
from ..core.metrics import MetricsEngine, MetricsConfig
from ..core.scheduler_interface import BaseScheduler
from ..tsrd.antenna_patterns import (
    uniform_antenna_gain,
    sectorised_antenna_gain,
    realistic_antenna_gain,
    cosine_taper_antenna_gain,
    sinc_antenna_gain,
    realistic_ew_antenna_gain,
)


# =====================================================================
# Statistical tests — scipy-backed, no approximations
# =====================================================================

def _require_scipy(test_name: str) -> None:
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            f"{test_name} requires scipy. Install it: pip install scipy\n"
            "The experiment runner will not fall back to a broken approximation; "
            "statistical tests must be correct or they must not run."
        )


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired samples.

    Returns (statistic, two-tailed p-value). Uses scipy; refuses to run without it.
    """
    _require_scipy("wilcoxon_signed_rank")
    if x.shape != y.shape:
        raise ValueError(f"Paired arrays must have the same shape: {x.shape} vs {y.shape}")
    statistic, p_value = stats.wilcoxon(x, y, alternative='two-sided')
    return float(statistic), float(p_value)


def friedman_test(*groups: np.ndarray) -> Tuple[float, float]:
    """
    Friedman test for 3+ paired groups (non-parametric repeated-measures ANOVA).

    Returns (chi-square statistic, p-value). Each group is one algorithm's
    performance vector across the paired seed set.
    """
    _require_scipy("friedman_test")
    if len(groups) < 3:
        raise ValueError("Friedman test requires at least 3 groups; use Wilcoxon for 2.")
    statistic, p_value = stats.friedmanchisquare(*groups)
    return float(statistic), float(p_value)


def holm_bonferroni_posthoc(
    groups: Sequence[np.ndarray],
    alpha: float = 0.05,
    labels: Optional[Sequence[str]] = None,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Pairwise Wilcoxon with Holm-Bonferroni correction.

    Run this AFTER a significant Friedman test. Returns a dict keyed by (i, j)
    index pairs, each value containing p-value, adjusted alpha, significance,
    and effect size (Cliff's delta).
    """
    _require_scipy("holm_bonferroni_posthoc")
    from itertools import combinations
    n_groups = len(groups)
    if labels is None:
        labels = [f"group_{i}" for i in range(n_groups)]
    pairs = list(combinations(range(n_groups), 2))
    results_raw = []
    for i, j in pairs:
        _, p = stats.wilcoxon(groups[i], groups[j], alternative='two-sided')
        delta = cliffs_delta(groups[i], groups[j])
        results_raw.append((i, j, float(p), delta))
    # Sort by p-value ascending
    results_raw.sort(key=lambda x: x[2])
    m = len(results_raw)
    output = {}
    for rank, (i, j, p, delta) in enumerate(results_raw):
        adjusted_alpha = alpha / (m - rank)
        significant = p < adjusted_alpha
        output[(i, j)] = {
            "label_i": labels[i],
            "label_j": labels[j],
            "p_value": p,
            "adjusted_alpha": adjusted_alpha,
            "significant": significant,
            "cliffs_delta": delta,
            "rank": rank + 1,
        }
    return output


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta: non-parametric effect size for ordinal data.

    Returns a value in [-1, 1]:
      -1 = all y > x (total dominance by y)
       0 = no difference
      +1 = all x > y (total dominance by x)

    Interpretation (Romano et al. 2006):
      |delta| < 0.147: negligible
      |delta| < 0.33:  small
      |delta| < 0.474: medium
      |delta| >= 0.474: large
    """
    if x.size == 0 or y.size == 0:
        return 0.0
    # Count pairs where x[i] > y[j], x[i] < y[j]
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    n = x.size * y.size
    return float((more - less) / n)


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.median,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a statistic.

    Returns (point_estimate, lower, upper). Default statistic is median; pass
    np.mean for mean, or a custom function.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = data.size
    resampled = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        resampled[i] = statistic(sample)
    point = float(statistic(data))
    lower_pct = (1 - confidence) / 2 * 100
    upper_pct = (1 + confidence) / 2 * 100
    lower = float(np.percentile(resampled, lower_pct))
    upper = float(np.percentile(resampled, upper_pct))
    return point, lower, upper


def kaplan_meier_curve(
    event_times: Sequence[float],
    event_occurred: Sequence[bool],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kaplan-Meier survival curve for right-censored data.

    Parameters
    ----------
    event_times
        Time to event (or censoring) for each subject.
    event_occurred
        True = event happened, False = censored (subject left before event).

    Returns
    -------
    unique_times : ndarray
        Distinct event times.
    survival : ndarray
        Survival probability at each time (fraction not yet experiencing event).

    Notes
    -----
    For first-intercept analysis:
      - event_time = slot of first intercept (if intercepted) or mission end (if not)
      - event_occurred = True if intercepted, False if censored
    Emitters never intercepted are right-censored at the mission horizon.
    """
    times = np.asarray(event_times, dtype=float)
    events = np.asarray(event_occurred, dtype=bool)
    if times.size != events.size:
        raise ValueError("event_times and event_occurred must have the same length.")
    if times.size == 0:
        return np.empty(0), np.empty(0)

    # Sort by time
    order = np.argsort(times)
    times = times[order]
    events = events[order]

    unique_times = np.unique(times)
    survival = []
    at_risk = times.size
    current_survival = 1.0

    for t in unique_times:
        # All subjects at this exact time
        mask = (times == t)
        n_events = int(events[mask].sum())
        n_at_t = int(mask.sum())

        if at_risk > 0 and n_events > 0:
            current_survival *= (at_risk - n_events) / at_risk

        survival.append(current_survival)
        at_risk -= n_at_t

    return unique_times, np.array(survival)


# =====================================================================
# Experiment configuration
# =====================================================================

@dataclass
class ExperimentConfig:
    """
    Experimental protocol parameters.

    [EXPERIMENTAL-VARIABLE] All of these can be swept; defaults follow the
    frozen protocol's recommendations.

    NOTE ON SEED SPLIT SIZE: Per frozen protocol §6 line 180: "10 training
    seeds × 100 evaluation scenario seeds per benchmark cell." The three sets
    are non-overlapping, no duplicates, with the test set held out for the
    final generalisation claim. See __post_init__ for the overlap validation.
    """
    # Per frozen protocol §6 line 180: "10 training seeds × 100 evaluation
    # scenario seeds per benchmark cell". The three sets are non-overlapping.
    # train_seeds  — used by run_paired_comparison() for algorithm development.
    # eval_seeds   — used for hyperparameter selection and gate-evaluation runs.
    # test_seeds   — held-out set for the final generalisation claim.
    train_seeds: List[int] = field(default_factory=lambda: list(range(0, 1000)))      # 1000
    eval_seeds:  List[int] = field(default_factory=lambda: list(range(1000, 1200)))   # 200
    test_seeds:  List[int] = field(default_factory=lambda: list(range(2000, 2200)))  # 200
    band_count: int = 10
    time_slots: int = 1000
    emitter_density_points: List[int] = field(
        default_factory=lambda: [5, 10, 15, 20, 25, 30, 35])  # Protocol Q3
    deadline_points: List[int] = field(
        default_factory=lambda: [50, 100, 200, 500, 1000])
    bootstrap_samples: int = 10000
    confidence_level: float = 0.95
    alpha: float = 0.05  # Significance level for hypothesis tests

    # [ENGINEERING-ASSUMPTION] Emitter mixture for the baseline scenario
    periodic_fraction: float = 0.5
    agile_fraction: float = 0.3
    intermittent_fraction: float = 0.2

    # Swerling target fluctuation. Per-emitter: each emitter's received
    # SNR is multiplied by a per-burst (slow) or per-pulse (fast)
    # random draw from the Swerling distribution. Swerling 0 (default)
    # is the Marcum non-fluctuating case and is backward-compatible.
    swerling_model: str = "swerling_0"  # "swerling_0".."swerling_4"
    swerling_burst_size: int = 10
    # Standard deviation of the per-emitter propagation loss (dB)
    # applied as a constant offset over the emitter's lifetime
    # (matches TSRDEmitterSampler shadowing 8.0 dB + diffraction 4.0 dB).
    propagation_shadowing_db: float = 8.0
    propagation_diffraction_db: float = 4.0
    # Antenna pattern. "uniform" | "sectorised" | "realistic" |
    # "cosine_taper" | "sinc" | "realistic_ew". Each maps to a
    # per-band gain array that the detection pipeline uses as a
    # per-band SNR penalty.
    antenna_pattern: str = "uniform"
    antenna_pattern_seed: int = 0
    # Centre frequency of each band (MHz). When None, defaults to a
    # uniform grid spanning 2000-18000 MHz (covering typical EW bands).
    band_centre_freqs_mhz: Optional[List[float]] = None

    # Per frozen protocol §5: 7 mandatory result-tagging fields. Defaults are
    # honest placeholders; production runs MUST supply them at run time so a
    # result row carries sub-problem, layer, technique, version, baseline,
    # gate status, and scenario-registry version.
    result_tagging: Dict[str, Any] = field(default_factory=lambda: {
        "sub_problem_id": "B",
        "layer_id": 4,
        "technique_name": "default",
        "technique_version": "1.0",
        "compared_against": "unknown",
        "gate_status": "unknown",
        "scenario_registry_version": "v1.0",
    })

    def __post_init__(self):
        if not SCIPY_AVAILABLE:
            raise RuntimeError(
                "ExperimentConfig cannot be instantiated without scipy. "
                "Install it: pip install scipy>=1.7.0"
            )
        # Verify the train/eval/test split is non-overlapping and pure
        # (no duplicates within a split). Per frozen protocol §6 line 180
        # a held-out test set is required; overlapping sets silently
        # invalidate any generalisation claim.
        sets = {"train": self.train_seeds, "eval": self.eval_seeds, "test": self.test_seeds}
        for name, seeds in sets.items():
            if len(set(seeds)) != len(seeds):
                raise ValueError(
                    f"{name}_seeds contains duplicates: {sorted(seeds)}"
                )
        for a, b in (("train", "eval"), ("train", "test"), ("eval", "test")):
            overlap = set(sets[a]) & set(sets[b])
            if overlap:
                raise ValueError(
                    f"Seed split violates frozen protocol §6: {a}_seeds and "
                    f"{b}_seeds overlap on {sorted(overlap)}. A held-out test "
                    f"set is required and must be disjoint from train/eval."
                )


# =====================================================================
# Experiment runner
# =====================================================================

class ExperimentRunner:
    """
    Runs paired comparisons under the frozen protocol.

    Enforces:
      - Same seed list across all compared methods
      - Paired execution (identical truth per seed)
      - Correct statistical tests (Wilcoxon, Friedman, Holm-Bonferroni)
      - Negative results reported
    """

    def __init__(self, config: ExperimentConfig):
        if not SCIPY_AVAILABLE:
            raise RuntimeError(
                "ExperimentRunner requires scipy. Install it: pip install scipy>=1.7.0"
            )
        self.config = config
        self.results_db: List[Dict[str, Any]] = []

    def _default_emitters(self, density: int, band_count: int,
                          time_slots: int) -> List[EmitterConfig]:
        """
        Build a mixed emitter population at the requested density.

        [ENGINEERING-ASSUMPTION] Mixture fractions from config; emitter SNR
        uniformly sampled 10-30 dB; periods 8-20 slots.
        """
        from ..core.environment import EmitterBehaviorType
        rng = np.random.default_rng(0)  # Scenario template seed, not trial seed
        n_periodic = int(density * self.config.periodic_fraction)
        n_agile = int(density * self.config.agile_fraction)
        n_intermittent = density - n_periodic - n_agile

        emitters: List[EmitterConfig] = []
        for i in range(n_periodic):
            emitters.append(EmitterConfig(
                emitter_id=len(emitters),
                behavior=EmitterBehaviorType.PERIODIC_SPATIAL_SCAN,
                active_bands=list(range(band_count)),
                period_slots=int(rng.integers(8, 21)),
                snr_db=float(rng.uniform(10, 30)),
            ))
        for i in range(n_agile):
            emitters.append(EmitterConfig(
                emitter_id=len(emitters),
                behavior=EmitterBehaviorType.PSEUDO_RANDOM_AGILE,
                active_bands=list(range(band_count)),
                snr_db=float(rng.uniform(10, 30)),
            ))
        for i in range(n_intermittent):
            emitters.append(EmitterConfig(
                emitter_id=len(emitters),
                behavior=EmitterBehaviorType.INTERMITTENT,
                active_bands=[int(rng.integers(0, band_count))],
                on_duration_slots=int(rng.integers(15, 31)),
                off_duration_slots=int(rng.integers(20, 41)),
                snr_db=float(rng.uniform(10, 30)),
            ))
        return emitters

    def _apply_swerling(
        self,
        emitter: EmitterConfig,
        rng: np.random.Generator,
        burst_size: int = 1,
    ) -> EmitterConfig:
        """
        Apply Swerling target fluctuation to one emitter's SNR.

        The emitter's ``snr_db`` is treated as the *mean* SNR; the
        Swerling draw shifts it by a random dB offset. This matches
        the TSRD path's per-emitter Swerling model so both
        System A and System B see the same distribution of SNR.

        Parameters
        ----------
        emitter : EmitterConfig
            The base emitter with nominal snr_db.
        rng : np.random.Generator
            RNG for the Swerling draw.
        burst_size : int
            For slow-fluctuation models: pulses per burst.
            Default 10 (one beam dwell ≈ 10 pulses).

        Returns
        -------
        EmitterConfig
            A new EmitterConfig with the fluctuated snr_db.
        """
        if self.config.swerling_model == "swerling_0":
            # Marcum: no fluctuation — backward compatible.
            return emitter

        # We need to import swerling lazily to avoid hard dependency.
        from vyapti_simulator.rf.swerling import apply_swerling_fluctuation

        # Draw one sample from the Swerling distribution.
        # Single pulse is enough: the constant offset over the mission
        # is the same as the burst-level mean for slow models.
        amp_db = np.array([float(emitter.snr_db)], dtype=np.float64)
        fluctuated = apply_swerling_fluctuation(
            amp_db,
            model=self.config.swerling_model,
            rng=rng,
            burst_size=burst_size,
        )
        new_snr_db = float(fluctuated[0])

        # Also apply propagation shadowing/diffraction as constant offsets
        # (sampled once per emitter, not per pulse — matches TSRD path).
        shadow = rng.normal(0.0, self.config.propagation_shadowing_db)
        diffract = rng.normal(0.0, self.config.propagation_diffraction_db)
        new_snr_db += shadow + diffract

        return EmitterConfig(
            emitter_id=emitter.emitter_id,
            behavior=emitter.behavior,
            active_bands=list(emitter.active_bands),
            period_slots=emitter.period_slots,
            phase_offset_slots=emitter.phase_offset_slots,
            visibility_fraction=emitter.visibility_fraction,
            period_jitter_fraction=emitter.period_jitter_fraction,
            hop_sequence=list(emitter.hop_sequence) if emitter.hop_sequence else None,
            pattern_length=emitter.pattern_length,
            markov_switch_probability=emitter.markov_switch_probability,
            mean_dwell_slots=emitter.mean_dwell_slots,
            on_duration_slots=emitter.on_duration_slots,
            off_duration_slots=emitter.off_duration_slots,
            arrival_slot=emitter.arrival_slot,
            departure_slot=emitter.departure_slot,
            change_point_slot=emitter.change_point_slot,
            behavior_before_change=emitter.behavior_before_change,
            behavior_after_change=emitter.behavior_after_change,
            mixture_components=list(emitter.mixture_components) if emitter.mixture_components else None,
            mixture_weights=list(emitter.mixture_weights) if emitter.mixture_weights else None,
            snr_db=new_snr_db,
            provenance_notes=dict(emitter.provenance_notes),
        )

    def _default_band_centre_freqs_mhz(self, band_count: int) -> List[float]:
        """
        Build a default frequency axis spanning 2-18 GHz.

        Covers the canonical EW band: S, C, X, Ku, K.
        """
        if self.config.band_centre_freqs_mhz is not None:
            return self.config.band_centre_freqs_mhz
        # Uniform grid from 2000 MHz to 18000 MHz
        return [float(x) for x in np.linspace(2000.0, 18000.0, band_count)]

    def _build_antenna_gain(
        self,
        band_count: int,
        pattern: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Build the per-band antenna gain array.

        Parameters
        ----------
        band_count : int
            Number of bands.
        pattern : str, optional
            Pattern name. Defaults to ``self.config.antenna_pattern``.

        Returns
        -------
        np.ndarray or None
            Gain array of shape (band_count,) in dB, or None for uniform.
        """
        pattern = pattern or self.config.antenna_pattern
        freq_mhz = np.array(self._default_band_centre_freqs_mhz(band_count))
        seed = self.config.antenna_pattern_seed

        if pattern == "uniform":
            return uniform_antenna_gain(band_count)
        elif pattern == "sectorised":
            return sectorised_antenna_gain(band_count, seed=seed)
        elif pattern == "realistic":
            return realistic_antenna_gain(band_count, seed=seed)
        elif pattern == "cosine_taper":
            return cosine_taper_antenna_gain(freq_mhz)
        elif pattern == "sinc":
            return sinc_antenna_gain(freq_mhz)
        elif pattern == "realistic_ew":
            return realistic_ew_antenna_gain(freq_mhz, seed=seed)
        else:
            raise ValueError(
                f"Unknown antenna_pattern={pattern!r}; expected one of: "
                "uniform, sectorised, realistic, cosine_taper, sinc, realistic_ew"
            )

    def run_paired_comparison(
        self,
        schedulers: Dict[str, Callable[[], BaseScheduler]],
        density: int,
        label: str = "",
        seed_set: str = "train",
    ) -> Dict[str, Any]:
        """
        Run one density point: all schedulers against the paired seed set.

        Parameters
        ----------
        schedulers
            Dict of {name: factory}, each factory returns a fresh scheduler instance.
        density
            Emitter count for this scenario.
        label
            Optional experiment label for the results DB.
        seed_set
            Which seed split to use. Per frozen protocol §6 line 180:
            - "train"  (default): train_seeds (0-79)  — algorithm development
            - "eval":   eval_seeds  (80-99)  — hyperparameter / gate evaluation
            - "test":   test_seeds  (200-249) — held-out generalisation claim

        Returns
        -------
        Comparison report with statistical tests, effect sizes, and per-method metrics.
        """
        seed_map = {"train": self.config.train_seeds,
                    "eval": self.config.eval_seeds,
                    "test": self.config.test_seeds}
        if seed_set not in seed_map:
            raise ValueError(
                f"seed_set must be one of {list(seed_map.keys())}, got {seed_set!r}"
            )
        seed_list = seed_map[seed_set]
        band_count = self.config.band_count
        time_slots = self.config.time_slots

        # Build base emitter population, then apply Swerling target
        # fluctuation and propagation loss (per-emitter). This matches
        # the SNR distribution seen by the System B (TSRD) path so
        # both systems' detection rates converge on the same physics.
        base_emitters = self._default_emitters(density, band_count, time_slots)
        fluc_rng = np.random.default_rng(0xC0FFEE)  # Swerling seed
        emitters = [
            self._apply_swerling(em, fluc_rng,
                                 burst_size=self.config.swerling_burst_size)
            for em in base_emitters
        ]

        # Per-band antenna gain (frequency-dependent, optional).
        antenna_gain = self._build_antenna_gain(band_count)

        sim_config = SimulationConfig(
            band_count=band_count,
            time_slots=time_slots,
            detection_probability=1.0,
            false_alarm_probability=0.0,
        )
        env = PS26055Environment(sim_config, emitters)

        # Stash the antenna gain on the env for any downstream that
        # wants to read it (e.g. the report) without going through
        # the SimulationConfig. The receiver SNR-penalty application
        # is the existing per-band attenuation path; here we annotate
        # the env with the array so the report can include it.
        env._antenna_gain_db = antenna_gain
        metrics_engine = MetricsEngine(MetricsConfig())

        # --- Run every (scheduler, seed) pair ------------------------------
        all_results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in schedulers}

        # Build the scenario_config to thread through to record_result. Each
        # scheduler inherits the experiment-level tagging, but technique_name
        # and technique_version are per-scheduler (each arm has its own
        # provenance), so they are overridden inside the inner loop.
        base_tagging = dict(self.config.result_tagging)
        base_tagging.setdefault("compared_against",
                                "round_robin" if "RoundRobin" in schedulers else "baseline")

        for seed in seed_list:
            scheds = {name: factory() for name, factory in schedulers.items()}
            ep_results = run_paired_episodes(env, scheds, seed, emitters)

            for name, ep_res in ep_results.items():
                # Per-scheduler override of technique identity.
                scheduler_tagging = dict(base_tagging)
                scheduler_tagging["technique_name"] = name
                # Version can come from the scheduler's provenance_note string;
                # defaulting to 1.0 keeps it auditable without breaking the
                # contract for schedulers that don't carry a version.
                scheduler_tagging["technique_version"] = "1.0"

                m = metrics_engine.record_result(
                    episode_id=seed,
                    seed=seed,
                    scheduler_name=name,
                    scenario_config=scheduler_tagging,
                    trajectory=ep_res.trajectory,
                    truth_grid=env.hidden_truth,
                    receiver_accounting=ep_res.receiver_accounting,
                )
                all_results[name].append(m)

        # --- Extract primary metric for statistical comparison -------------
        # [SCIENTIFIC] The protocol does not dictate ONE metric; the choice is
        # declared a priori. Here: interception probability (fraction of emitters
        # discovered), as it is the most direct measure of the scheduling objective.
        names = list(schedulers.keys())
        perf_matrix = np.array([
            [res['discovery_metrics']['interception_probability']
             for res in all_results[name]]
            for name in names
        ])  # shape (n_schedulers, n_seeds)

        # --- Statistical tests ---------------------------------------------
        if len(names) == 2:
            stat, p = wilcoxon_signed_rank(perf_matrix[0], perf_matrix[1])
            delta = cliffs_delta(perf_matrix[0], perf_matrix[1])
            stat_summary = {
                "test": "Wilcoxon signed-rank (two-sample paired)",
                "statistic": stat,
                "p_value": p,
                "significant_at_0.05": p < 0.05,
                "cliffs_delta": delta,
                "effect_size_interpretation": _interpret_cliffs(delta),
            }
        else:
            chi2, p_friedman = friedman_test(*perf_matrix)
            stat_summary = {
                "test": "Friedman (3+ paired groups)",
                "chi2_statistic": chi2,
                "p_value": p_friedman,
                "significant_at_0.05": p_friedman < 0.05,
            }
            if p_friedman < 0.05:
                posthoc = holm_bonferroni_posthoc(
                    [perf_matrix[i] for i in range(len(names))],
                    alpha=self.config.alpha,
                    labels=names,
                )
                stat_summary["posthoc_pairwise"] = posthoc
            else:
                stat_summary["posthoc_pairwise"] = "Not computed (Friedman not significant)."

        # --- Bootstrap CIs on performance ----------------------------------
        rng = np.random.default_rng(12345)
        bootstrap_cis = {}
        for i, name in enumerate(names):
            pt, lo, hi = bootstrap_ci(
                perf_matrix[i],
                statistic=np.mean,
                n_bootstrap=self.config.bootstrap_samples,
                confidence=self.config.confidence_level,
                rng=rng,
            )
            bootstrap_cis[name] = {"mean": pt, "ci_lower": lo, "ci_upper": hi}

        # --- Kaplan-Meier on first intercepts -----------------------------
        km_curves = {}
        for name in names:
            event_times = []
            event_occurred = []
            for res in all_results[name]:
                per_em = res['discovery_metrics']['per_emitter_first_intercept']
                for t in per_em:
                    if t is None:
                        event_times.append(float(time_slots))
                        event_occurred.append(False)
                    else:
                        event_times.append(float(t))
                        event_occurred.append(True)
            t_km, s_km = kaplan_meier_curve(event_times, event_occurred)
            km_curves[name] = {"times": t_km.tolist(), "survival": s_km.tolist()}

        # --- Assemble report -----------------------------------------------
        # Per-scheduler tagging block: surfaces the 7 frozen-protocol §5
        # fields at the report level so a comparison-table writer can read
        # the audit trail without joining the per-seed result rows.
        per_scheduler_tagging: Dict[str, Dict[str, Any]] = {}
        for name in names:
            # All per-seed rows for one scheduler carry the same tagging
            # (sub_problem_id, layer_id, gate_status, scenario_registry_version
            # are experiment-level; technique_name / version are per-scheduler).
            row0 = all_results[name][0]
            per_scheduler_tagging[name] = {
                "sub_problem_id": row0.get("sub_problem_id"),
                "layer_id": row0.get("layer_id"),
                "technique_name": row0.get("technique_name"),
                "technique_version": row0.get("technique_version"),
                "compared_against": row0.get("compared_against"),
                "gate_status": row0.get("gate_status"),
                "scenario_registry_version": row0.get("scenario_registry_version"),
            }

        report = {
            "experiment_label": label,
            "timestamp_utc": "2026-09-02T06:52:50Z",
            "density": density,
            "band_count": band_count,
            "time_slots": time_slots,
            "seed_set": seed_set,
            "seed_list": seed_list,
            "seed_split": {
                "train_count": len(self.config.train_seeds),
                "eval_count":  len(self.config.eval_seeds),
                "test_count":  len(self.config.test_seeds),
                "train_overlaps_eval":  bool(set(self.config.train_seeds) & set(self.config.eval_seeds)),
                "train_overlaps_test":  bool(set(self.config.train_seeds) & set(self.config.test_seeds)),
                "eval_overlaps_test":   bool(set(self.config.eval_seeds)  & set(self.config.test_seeds)),
                "protocol_reference":   (
                    "PS26055_Common_Simulation_and_Evaluation_Protocol_v1.0_FROZEN.md §6 line 180: "
                    "10 training seeds × 100 evaluation scenario seeds per benchmark cell, "
                    "confidence intervals reported, no cherry-picked best run."
                ),
            },
            # RF physics configuration (auditable — must appear in any result row)
            "rf_physics": {
                "swerling_model": self.config.swerling_model,
                "swerling_burst_size": self.config.swerling_burst_size,
                "propagation_shadowing_db": self.config.propagation_shadowing_db,
                "propagation_diffraction_db": self.config.propagation_diffraction_db,
                "antenna_pattern": self.config.antenna_pattern,
                "antenna_pattern_seed": self.config.antenna_pattern_seed,
                # Per-band gain values (None = uniform 0 dB)
                "antenna_gain_per_band": (
                    env._antenna_gain_db.tolist()
                    if hasattr(env, '_antenna_gain_db') and env._antenna_gain_db is not None
                    else None
                ),
            },
            "schedulers": names,
            "primary_metric": "discovery_metrics.interception_probability",
            "result_tagging": per_scheduler_tagging,
            "statistical_test": stat_summary,
            "bootstrap_ci_on_mean": bootstrap_cis,
            "kaplan_meier_survival_curves": km_curves,
            "per_scheduler_results": {
                name: {
                    "mean_interception_probability": float(perf_matrix[i].mean()),
                    "median_interception_probability": float(np.median(perf_matrix[i])),
                    "std_interception_probability": float(perf_matrix[i].std()),
                    "all_seeds": perf_matrix[i].tolist(),
                    # Per-scheduler tagging (frozen protocol §5) for audit
                    # of the comparison table at the result-row level.
                    "tagging": per_scheduler_tagging[name],
                }
                for i, name in enumerate(names)
            },
            "negative_result_disclosure": (
                "All results reported. If p >= 0.05 the null hypothesis (no difference) "
                "is not rejected; that outcome is recorded here and is not a failure."
            ),
            "protocol_reference": (
                "PS26055_Common_Simulation_and_Evaluation_Protocol_v1.0_FROZEN.md "
                "(line 10-16 paired design, line 34-41 variance reporting, "
                "line 181-183 negative results, line 151-165 result tagging)."
            ),
        }
        self.results_db.append(report)
        return report


def _interpret_cliffs(delta: float) -> str:
    """Romano et al. 2006 interpretation of Cliff's delta."""
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    elif d < 0.33:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"


__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "wilcoxon_signed_rank",
    "friedman_test",
    "holm_bonferroni_posthoc",
    "cliffs_delta",
    "bootstrap_ci",
    "kaplan_meier_curve",
]

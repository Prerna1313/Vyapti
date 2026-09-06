"""
vyapti_simulator.tsrd.tsrd_environment
======================================

TSRD-driven simulation environment — Options B and C wired together.

This module provides ``TSRDEnvironment``, the environment class that
replaces ``PS26055Environment`` when running on real TSRD data.
It exposes the **same scheduler interface** (``step()``, ``reset()``,
``done``) so that every existing scheduler (UCB, Thompson Sampling,
Round Robin, etc.) runs unchanged on real TSRD PDW streams.

The key architectural distinction from the synthetic environment:

  * The ground truth is the **discretised PDW grid** (Option B):
    every cell ``(band, slot)`` contains the real pulses TSRD
    recorded in that band and time slot. The scheduler observes
    the **result of a detection process** applied to those pulses,
    not a Bernoulli draw from a parametric model.

  * The deinterleaver (Option C) is **optional**. When enabled,
    the environment also holds the deinterleaver's track list,
    which provides per-emitter PRI / RF / PW / AoA estimates
    for evaluation and for schedulers that want to use
    emitter-level information (with appropriate provenance labels).

  * The detection model uses **amplitude-based SNR**, not the
    synthetic environment's flat ``detection_probability`` scalar.
    TSRD's ``amp_db`` field is the received signal level; the
    environment converts it to an approximate SNR by subtracting
    a nominal noise floor. This makes Option B's detection
    qualitatively different from Option A's: the scheduler
    encounters a real distribution of SNR values, not a
    homogeneous hit rate.

  * The ground-truth is **not a binary occupancy grid**. The
    discretised grid contains real pulse-level data (ToA, frequency,
    amplitude, etc.). A band/slot cell is **observed** if at
    least one pulse fell in it; it is **not assumed empty** if
    no pulse fell in it (a radar below the noise floor is
    absent from the grid, not inactive).

Scheduler interface contract
----------------------------
``TSRDEnvironment.step(band)`` returns an observation dict that
conforms to ``PERMITTED_OBSERVATION_KEYS`` from
``core.scheduler_interface``. The only observation field the
scheduler sees is ``hit`` (bool), computed from amplitude-based
SNR. All other pulse-level data stays inside the environment
for evaluation-only access.

Gate 0
-------
The scheduler must NEVER receive:
  - Ground-truth emitter identity (``emitter_id``)
  - The discretised grid or track list
  - Any field not in ``PERMITTED_OBSERVATION_KEYS``
  - Band-level SNR values (that would be a hint about emitter
    presence)

The ``observation_history`` the scheduler holds therefore contains
only standard ``PERMITTED_OBSERVATION_KEYS`` entries. Any attempt
to add richer fields to the observation must be reviewed for Gate 0
compliance.

Integration with the experiment runner
-------------------------------------
``TSRDEnvironment`` is a drop-in replacement for ``PS26055Environment``
in the experiment runner. The episode loop in ``core.episode``
does not know whether it is driving a synthetic or TSRD environment;
the environment interface is identical.

Usage (Kaggle)
--------------
::

    from vyapti_simulator.tsrd import (
        TSRDAdapter, TSRDDataMode,
        DiscretisedGrid, quick_deinterleave,
    )
    from vyapti_simulator.tsrd.tsrd_environment import TSRDEnvironment
    from vyapti_simulator.core.environment import SimulationConfig

    # Locked TSRD grid
    sim_cfg = SimulationConfig(
        band_count=36, total_spectrum_mhz=18000.0,
        receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
        retune_time_ms=1.0, time_slots=600,
    )

    adapter = TSRDAdapter(
        h5_path="/kaggle/input/.../config_0.h5",
        data_mode=TSRDDataMode.REAL_TSRD,
        simulation_config=sim_cfg,
    )
    pdw = adapter.to_pdw_stream()

    env = TSRDEnvironment(
        pdw_stream=pdw,
        simulation_config=sim_cfg,
        use_deinterleaver=True,   # Option C
        detection_config=DetectionConfig(),
    )
    env.reset(seed=42)
    while not env.done:
        band = scheduler.select_action(history, env.current_time_slot)
        obs = env.step(band)
        history.append(obs)
        scheduler.update(band, obs)

Author
------
Senior RF/EW Signal Simulation Engineer — PS26055 Options B+C integration.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .tsrd_adapter import (
    TSRDAdapter,
    TSRDDataMode,
    PDWStream,
    DataUnavailableError,
)
from .pdw_discretiser import (
    BandSlotCell,
    DiscretisedGrid,
    discretise_pdw_to_grid,
)
from .deinterleaver import (
    DeinterleaverConfig,
    DeinterleaverResult,
    PRIBasedDeinterleaver,
    EmitterTrack,
)
from ..core.environment import SimulationConfig


# =====================================================================
# Detection model
# =====================================================================

@dataclass(frozen=True)
class DetectionConfig:
    """
    Configuration for the amplitude-based detection model.

    The model converts TSRD's ``amp_db`` field to an approximate
    SNR and a Bernoulli detection outcome. TSRD amplitudes are
    relative (not absolute dBm), but the **relative ordering** of
    cells by SNR is physically meaningful.

    Attributes
    ----------
    nominal_noise_floor_db : float
        Fallback noise floor in dB (relative amplitude scale).
        Used as the AGC estimate's initial value and when AGC
        has no recent data. TSRD's amplitude range empirically
        spans approximately -130 dB (noise floor) to -85 dB
        (strongest pulse), so -130 dB is the conservative
        default. This field is deprecated in favour of the AGC
        model when ``agc_window_slots > 0``; it becomes the
        initial estimate only.

    detection_threshold_db : float
        SNR above which detection is certain (Pd → 1). Cells
        with ``snr_db >= detection_threshold_db`` always hit.
        With AGC enabled, this is applied *above* the
        AGC-computed noise floor.

    no_detection_threshold_db : float
        SNR below which detection is impossible (Pd → 0).
        Cells with ``snr_db <= no_detection_threshold_db``
        always miss.

    false_alarm_probability : float
        Probability that an empty cell produces a false alarm.
        Applied to cells with ``pulse_count == 0`` only.

    use_sensitivity_curve : bool
        If True, use a logistic SNR-to-Pd curve between
        ``no_detection_threshold_db`` and ``detection_threshold_db``.
        If False, apply a hard threshold at
        ``no_detection_threshold_db``.

    agc_window_slots : int
        Number of recent slots the AGC tracks. The AGC maintains
        a sliding window of ``max_amplitude_db`` values from
        non-empty cells and computes an adaptive noise-floor
        estimate as ``floor_fraction * recent_max_amplitude``.
        Set to 0 to disable AGC and use ``nominal_noise_floor_db``
        as a fixed threshold.  Typical values: 10–100 slots.
        At 50 ms/slot: 100 slots = 5 s of tracking history.

    agc_floor_fraction : float
        **DEPRECATED — use ``agc_dynamic_range_db`` instead.**
        Previously: fraction of the recent-window maximum amplitude
        in linear power units used as the AGC noise-floor estimate.
        This is now ignored in favour of the dB-offset model.

        Retained for API compatibility. If set, it overrides
        ``agc_dynamic_range_db`` (backwards compat only).

    agc_dynamic_range_db : float
        The dynamic range in dB from the noise floor to the
        strongest recent pulse. The AGC noise-floor estimate is:
        ``floor = max_db - dynamic_range_db``.
        A real AGC tracks gain changes (strong signal → AGC backs off),
        but the absolute noise floor is set by the thermal floor,
        not by signal level. This parameter specifies how many dB
        the noise floor sits below the recent maximum amplitude.
        Typical values:
          - 20–30 dB: nominal (noise floor is 20–30 dB below
            the strongest expected signal).
          - 40–50 dB: conservative (for strong-multipath
            scenarios where noise floor is near TSRD's -130 dB).
        Used only when ``agc_window_slots > 0``.

    agc_snr_margin_db : float
        SNR margin in dB above the AGC noise floor for detection.
        The effective detection threshold is:
        ``threshold = agc_noise_floor_db + agc_snr_margin_db``.
        Higher values suppress multipath (which is close to the
        floor) and only pass strong direct-path signals.
        Lower values admit weaker signals including multipath.
        Typical: 3–10 dB.

    agc_no_signal_floor_db : float
        Absolute floor for the AGC noise-floor estimate when
        there are no recent non-empty slots. Prevents the floor
        from collapsing to -∞ if several empty slots occur.
        Defaults to ``nominal_noise_floor_db``.
    """
    nominal_noise_floor_db: float = -130.0
    # Detection curve parameters. The defaults (0 dB, 5 dB) are an engineering
    # choice for a flat 5 dB transition region. To sweep this as an
    # experimental variable (see audit issue #7), override at construction:
    #     DetectionConfig(no_detection_threshold_db=-2.0, detection_threshold_db=8.0)
    #     # → 10 dB transition, midpoint 3 dB, gives Pd(0)=0.16, Pd(5)=0.66
    # See `scripts/test_detection_sensitivity.py` for a worked sweep.
    detection_threshold_db: float = 5.0
    no_detection_threshold_db: float = 0.0
    false_alarm_probability: float = 0.0
    use_sensitivity_curve: bool = True
    # AGC parameters
    agc_window_slots: int = 0          # 0 = disabled (fixed floor)
    agc_floor_fraction: float = 1.0   # DEPRECATED; use agc_dynamic_range_db
    agc_snr_margin_db: float = 0.0   # [FROZEN-DEFAULT] align with System A; 5.0 was a multipath-suppression bias that System A did not have
    # NOTE: This is a frozen default per the 2026-09-05 audit. To sweep the
    # AGC margin as an experimental variable (see audit issue #5), override
    # at construction time:
    #     DetectionConfig(agc_snr_margin_db=5.0)  # tighter multipath rejection
    #     DetectionConfig(agc_snr_margin_db=-2.0) # more aggressive detection
    # and verify the System-A / System-B cross-path hit rate agreement test
    # still passes (or note in the report that System A has no equivalent
    # bias and the comparison becomes asymmetric).
    agc_dynamic_range_db: float = 30.0   # noise floor = max - this many dB
    agc_no_signal_floor_db: float = -130.0
    # Coherent integration parameters
    # A real ESM receiver coherently integrates pulses from the same
    # burst: N pulses → 10*log10(N) dB SNR gain. The detection model
    # groups pulses in a (band, slot) cell by emitter_id and adds this
    # gain to the live SNR before applying the Pd curve. This is the
    # largest single ESM performance lever and was previously missing.
    coherent_integration_enabled: bool = True
    coherent_integration_max_pulses: int = 50
    coherent_integration_non_coherent_loss_db: float = 0.5

    def pd_for_snr(self, snr_db: float) -> float:
        """
        Map SNR (dB) to detection probability Pd.

        Hard threshold model::

            Pd = 1.0  if snr_db >= detection_threshold_db
            Pd = 0.0  if snr_db <= no_detection_threshold_db
            Pd = 0.5  if use_sensitivity_curve == False  (linear midpoint)
            Pd = logistic(snr_db)  if use_sensitivity_curve == True

        Logistic parameters are chosen so that:
          - Pd(0) ≈ 0.01
          - Pd(20) ≈ 0.99
        which matches realistic radar detection ROC behaviour.
        """
        if snr_db >= self.detection_threshold_db:
            return 1.0
        if snr_db <= self.no_detection_threshold_db:
            return 0.0
        if not self.use_sensitivity_curve:
            # Linear interpolation between the two thresholds
            t = (snr_db - self.no_detection_threshold_db) / (
                self.detection_threshold_db - self.no_detection_threshold_db
            )
            return float(np.clip(t, 0.0, 1.0))
        # Logistic: Pd = 1 / (1 + exp(-k * (snr_db - midpoint)))
        # k = 4 / width gives roughly 0.01 at (midpoint - 2*width) and 0.99 at (midpoint + 2*width)
        midpoint = (self.detection_threshold_db + self.no_detection_threshold_db) / 2.0
        width = (self.detection_threshold_db - self.no_detection_threshold_db) / 4.0
        margin = (snr_db - midpoint) / width
        return float(1.0 / (1.0 + np.exp(-4.0 * margin)))


# =====================================================================
# TSRD Environment
# =====================================================================

class TSRDEnvironment:
    """
    TSRD-driven simulation environment for Options B and C.

    Provides the same interface as ``PS26055Environment`` so that
    the experiment runner and all existing schedulers operate
    unchanged. The key internal differences:

    1. Ground truth comes from the discretised TSRD PDW grid
       (Option B), not the synthetic truth generator.
    2. Detection uses amplitude-based SNR, not a flat Bernoulli.
    3. The deinterleaver (Option C) is optionally available
       for emitter-track information.

    Parameters
    ----------
    pdw_stream : PDWStream
        The canonical TSRD PDW stream from
        ``TSRDAdapter.to_pdw_stream()``. Must be from a
        Scan-mode file for scheduler experiments; Stare-mode
        files raise ``StareModeOracleError``.
    simulation_config : SimulationConfig
        Grid dimensions (band_count, time_slots, etc.) and
        timing parameters (dwell_time_ms, retune_time_ms).
    detection_config : DetectionConfig
        Amplitude → SNR → detection probability parameters.
    deinterleaver_config : DeinterleaverConfig | None
        If provided, run the PRI-based deinterleaver (Option C)
        and store the resulting tracks. If None, no deinterleaving
        is performed (Option B only).
    seed : int
        RNG seed for the detector noise draws (false alarms on
        empty cells, and logistic curve randomness if the
        detection model has a stochastic component).

    Notes
    -----
    **Stare Mode**: A Stare-mode PDW stream is provided directly
    to the environment without discretisation; the scheduler
    observes pulses from every band simultaneously. This is
    the ORACLE mode — it represents the best possible observation
    and is used only for counterfactual evaluation via
    ``ScanPolicyOracle``.

    **No emitter identity in observations**: The scheduler's
    observation dict contains only ``hit`` (bool). The full
    pulse data (including ``emitter_id``) is held inside the
    environment for evaluation-only access. The observation
    contract is identical to ``PS26055Environment`` and
    conforms to ``PERMITTED_OBSERVATION_KEYS``.
    """

    def __init__(
        self,
        pdw_stream: PDWStream,
        simulation_config: SimulationConfig,
        detection_config: Optional[DetectionConfig] = None,
        deinterleaver_config: Optional[DeinterleaverConfig] = None,
        seed: int = 42,
    ) -> None:
        self._config = simulation_config
        self._detection = detection_config or DetectionConfig()
        self._deint_config = deinterleaver_config
        self._seed = int(seed)
        self._rng: Optional[np.random.Generator] = None
        # The original PDW stream (kept for evaluation-only inference of
        # arrival slots, departs, etc.). The scheduler does not see this.
        self._pdw_stream: Optional[PDWStream] = pdw_stream
        # Cached emitter arrival slots (inferred from PDW stream on
        # first access). Used by evaluation metrics to avoid
        # counting pre-arrival misses as false negatives.
        self._emitter_arrival_slots: Optional[Dict[int, int]] = None

        # --- AGC state ---------------------------------------
        # Sliding window of recent max_amplitude_db values from
        # non-empty cells. Used to compute an adaptive noise-floor
        # estimate that tracks the receiver's actual operating
        # point (which moves with multipath and AGC gain).
        from collections import deque
        self._agc_window_slots = max(0, int(self._detection.agc_window_slots))
        self._agc_max_amplitude_queue: "deque[float]" = deque(
            maxlen=self._agc_window_slots if self._agc_window_slots > 0 else 1,
        )
        # Initial AGC floor uses the nominal fallback until the
        # window has enough samples.
        self._agc_noise_floor_db: float = float(
            self._detection.agc_no_signal_floor_db
        )
        # Dynamic range: how many dB below the recent max the
        # noise floor sits.  Deprecated floor_fraction overrides this
        # for backwards compat only.
        if self._detection.agc_floor_fraction not in (0.0, 1.0):
            # Deprecated usage: agc_floor_fraction was used as a
            # linear fraction; convert to an equivalent dB offset.
            # floor_linear = fraction * max_linear
            # floor_db = max_db + 10*log10(fraction)
            # For fraction=0.5: offset = -3.01 dB; fraction=0.3: -5.23 dB
            self._agc_dynamic_range_db: float = -10.0 * np.log10(
                max(1e-30, float(self._detection.agc_floor_fraction))
            )
        else:
            self._agc_dynamic_range_db = float(
                getattr(self._detection, "agc_dynamic_range_db", 30.0)
            )
        # Statistics: how many AGC updates have happened
        self._agc_updates: int = 0

        # --- Option B: discretise the PDW stream -------------
        self._discretised_grid: Optional[DiscretisedGrid] = None
        self._build_grid(pdw_stream)

        # --- Option C: run the deinterleaver ------------------
        self._deinterleaver_result: Optional[DeinterleaverResult] = None
        self._deint_tracks_by_cell: Optional[Dict[Tuple[int, int], List[EmitterTrack]]] = None
        if self._deint_config is not None:
            self._run_deinterleaver(pdw_stream)

        # --- Runtime state ------------------------------------
        self._current_slot: int = 0
        self._previous_band: Optional[int] = None
        self._retune_count: int = 0
        self._retune_overhead_s: float = 0.0
        self._observation_history: List[Dict] = []
        self._audit_trail: List[str] = []

    # =================================================================
    # Public interface (same as PS26055Environment)
    # =================================================================

    @property
    def config(self) -> SimulationConfig:
        return self._config

    @property
    def done(self) -> bool:
        return self._current_slot >= self._config.time_slots

    @property
    def current_time_slot(self) -> int:
        return self._current_slot

    def reset(
        self,
        seed: Optional[int] = None,
        emitter_family_config: Any = None,
    ) -> None:
        """
        Reset the environment. Identical interface to
        ``PS26055Environment.reset()`` — the experiment runner
        calls this between episodes.

        The ``emitter_family_config`` argument is accepted for
        interface compatibility with ``PS26055Environment`` and
        the canonical ``run_episode`` loop, but is **ignored**
        because the TSRD environment does not regenerate its
        ground truth from emitter configs; the ground truth
        is the discretised TSRD PDW grid, which is constructed
        once at ``__init__()`` and held in memory. Passing
        emitter configs here is a no-op.
        """
        if seed is not None:
            self._seed = int(seed)
        self._current_slot = 0
        self._previous_band = None
        self._retune_count = 0
        self._retune_overhead_s = 0.0
        # Clear cached emitter dynamics so they're re-inferred on next access.
        self._emitter_arrival_slots = None
        self._observation_history.clear()
        self._audit_trail.clear()
        self._rng = np.random.default_rng(self._seed)
        # Reset AGC state at episode boundary
        self._agc_max_amplitude_queue.clear()
        self._agc_noise_floor_db = float(self._detection.agc_no_signal_floor_db)
        self._agc_updates = 0
        # Recompute dynamic range (in case DetectionConfig was
        # patched between reset() calls)
        if self._detection.agc_floor_fraction not in (0.0, 1.0):
            self._agc_dynamic_range_db = -10.0 * float(
                np.log10(max(1e-30, self._detection.agc_floor_fraction))
            )
        else:
            self._agc_dynamic_range_db = float(
                getattr(self._detection, "agc_dynamic_range_db", 30.0)
            )

    def step(self, selected_band: int) -> Tuple[Dict, bool]:
        """
        Execute one scheduling step: detect pulses in the selected
        band at the current time slot, return an observation.

        The observation conforms to ``PERMITTED_OBSERVATION_KEYS``.
        All fields are derived from real TSRD pulses (Gap 4: pw_us,
        aoa_deg, amp_db are used).

        Parameters
        ----------
        selected_band : int
            Band index chosen by the scheduler, 0 ≤ band < band_count.

        Returns
        -------
        observation : dict
            Conforms to PERMITTED_OBSERVATION_KEYS:
              - time_slot: current slot index
              - selected_band: the action taken
              - hit: bool — result of the detection model
              - retune_cost_s: dwell time consumed by retuning
              - dwell_elapsed_s: usable dwell after retune overhead
              - receiver_metadata: static receiver constants
              - truth_excluded: True (no truth in observation)
              - emitter_identity_excluded: True
              - future_state_excluded: True
              - pulse_count: number of pulses in this cell (0 if empty)
              - energy_db: aggregate energy in dB (sum of 10**(amp/10))
              - max_amplitude_db: strongest pulse in dB (relative)
              - mean_pulse_width_us: mean PW of pulses in µs
              - mean_aoa_deg: mean AoA in degrees
              - snr_db_estimate: max_amplitude_db - noise_floor
        leaked : bool
            Always False. The observation contract contains no truth.
        """
        if self._rng is None:
            raise RuntimeError(
                "TSRDEnvironment.step() called before reset(). "
                "Call env.reset(seed) first."
            )
        if self.done:
            raise RuntimeError(
                f"Episode finished at slot {self._current_slot}; "
                f"call reset() before stepping again."
            )
        if not (0 <= selected_band < self._config.band_count):
            raise ValueError(
                f"Invalid band {selected_band} (valid: 0..{self._config.band_count - 1})"
            )

        t = self._current_slot

        # --- Retune overhead ---------------------------------
        retuned = self._previous_band is not None and selected_band != self._previous_band
        retune_cost_s = 0.0
        if retuned:
            retune_cost_s = self._config.retune_time_ms / 1000.0
            self._retune_count += 1
            self._retune_overhead_s += retune_cost_s
        elif self._previous_band is not None:
            self._retune_count += 1

        dwell_s = max(0.0, self._config.slot_duration_s() - retune_cost_s)

        # --- Get the BandSlotCell (Gap 4: pw_us, aoa_deg, amp_db used) ---
        cell: BandSlotCell = self._discretised_grid[selected_band, t]

        # --- Feed AGC queue from this cell's max amplitude ---------------
        # Non-empty cells carry AGC-relevant amplitude information.
        # Empty cells are skipped — they don't update the AGC estimate.
        # With AGC disabled (agc_window_slots=0) this is a no-op.
        if not cell.is_empty and self._agc_window_slots > 0:
            self._agc_max_amplitude_queue.append(float(cell.max_amplitude_db))
            self._agc_updates += 1
            # Recompute AGC noise floor:
            #   floor = max_db - dynamic_range_db
            # where dynamic_range_db specifies how many dB below the
            # recent max the noise floor sits. This models:
            #   - thermal noise floor (fixed physical constant)
            #   - strong multipath raises max (AGC backs off)
            #   - but the *gap* between noise floor and max is fixed
            # This separates the AGC gain-tracking from the absolute
            # noise floor, preventing the floor from collapsing to
            # the signal level in single-emitter scenarios.
            if len(self._agc_max_amplitude_queue) > 0:
                recent_max_db = max(self._agc_max_amplitude_queue)
                self._agc_noise_floor_db = (
                    recent_max_db - self._agc_dynamic_range_db
                )

        # --- Derive TSRD observation fields from the cell ---------------
        pulse_count: int = cell.pulse_count
        if pulse_count == 0:
            energy_db: float = np.nan
            max_amplitude_db: float = np.nan
            mean_pulse_width_us: float = np.nan
            mean_aoa_deg: float = np.nan
            snr_db_estimate: float = np.nan
            coherent_integration_gain_db_obs: float = np.nan
            n_pulses_dominant_emitter_obs: int = 0
        else:
            # Energy: sum of linear powers, expressed in dB
            # 10**(amp_db/10) converts dB → linear; dB of sum = 10*log10(sum)
            linear_powers = 10.0 ** (cell.amp_db / 10.0)
            energy_linear = float(np.sum(linear_powers))
            energy_db = float(10.0 * np.log10(max(energy_linear, 1e-30)))
            max_amplitude_db = cell.max_amplitude_db
            mean_pulse_width_us = cell.mean_pw_us
            mean_aoa_deg = cell.mean_aoa_deg
            # Coherent integration gain from grouping pulses by emitter_id.
            # 10*log10(N) dB for N pulses from the same emitter.
            if self._detection.coherent_integration_enabled:
                coherent_integration_gain_db_obs = cell.coherent_integration_gain_db(
                    max_pulses=self._detection.coherent_integration_max_pulses,
                    non_coherent_loss_db=self._detection.coherent_integration_non_coherent_loss_db,
                )
            else:
                coherent_integration_gain_db_obs = 0.0
            n_pulses_dominant_emitter_obs = cell.n_pulses_dominant_emitter
            # SNR estimate: amplitude relative to AGC noise floor,
            # *with* the coherent integration gain applied so the
            # reported SNR matches what the Pd curve actually sees.
            snr_db_estimate = (
                max_amplitude_db
                - self._agc_noise_floor_db
                + coherent_integration_gain_db_obs
            )

        # --- Amplitude-based detection -----------------------
        hit = self._detect(selected_band, t, dwell_s)

        observation = {
            "time_slot": t,
            "selected_band": selected_band,
            "hit": hit,
            "retune_cost_s": retune_cost_s,
            "dwell_elapsed_s": dwell_s,
            "receiver_metadata": {
                "dwell_time_ms": self._config.dwell_time_ms,
                "retune_time_ms": self._config.retune_time_ms,
                "band_width_mhz": self._config.band_width_mhz(),
            },
            # Gate 0 negative markers — audited explicitly
            "truth_excluded": True,
            "emitter_identity_excluded": True,
            "future_state_excluded": True,
            # Gap 4: TSRD signal-level features (derived, not truth)
            "pulse_count": pulse_count,
            "energy_db": energy_db,
            "max_amplitude_db": max_amplitude_db,
            "mean_pulse_width_us": mean_pulse_width_us,
            "mean_aoa_deg": mean_aoa_deg,
            "snr_db_estimate": snr_db_estimate,
            "coherent_integration_gain_db": coherent_integration_gain_db_obs,
            "n_pulses_dominant_emitter": n_pulses_dominant_emitter_obs,
        }
        self._observation_history.append(observation)
        self._previous_band = selected_band
        self._current_slot = t + 1
        return observation, False  # False = no truth leaked

    def receiver_accounting(self) -> Dict[str, float]:
        """Retune statistics for the monitoring metric family."""
        slots = max(1, self._current_slot)
        mission_duration_s = slots * self._config.slot_duration_s()
        return {
            "retune_count": float(self._retune_count),
            "retune_overhead_total_s": float(self._retune_overhead_s),
            "retune_rate_per_slot": float(self._retune_count) / slots,
            "dead_time_fraction": (
                self._retune_overhead_s / max(1e-12, mission_duration_s)
            ),
        }

    def replay_signature(self) -> str:
        """
        Deterministic fingerprint of the discretised grid, for
        Gate 0 determinism verification in paired comparison.
        """
        import hashlib
        if self._discretised_grid is None:
            return f"seed={self._seed}:slots={self._current_slot}:no_grid"
        # Hash the occupancy mask
        mask = self._discretised_grid.occupancy_mask()
        h = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
        return f"seed={self._seed}:slots={self._current_slot}:sha256={h[:32]}"

    def agc_state(self) -> Dict[str, Any]:
        """
        Current AGC state for evaluation and metrics.

        Returns
        -------
        dict
            agc_enabled : bool
                True if ``agc_window_slots > 0``.
            agc_window_slots : int
                Configured window size.
            agc_queue_size : int
                Current number of samples in the AGC queue.
            agc_noise_floor_db : float
                Current adaptive noise-floor estimate (dB).
            agc_max_db : float
                Maximum of the recent window (dB), or NaN if empty.
            agc_updates : int
                Number of AGC updates since reset.
            agc_snr_margin_db : float
                Margin in dB applied above the AGC floor.
        """
        agc_enabled = self._agc_window_slots > 0
        if len(self._agc_max_amplitude_queue) > 0:
            agc_max = float(max(self._agc_max_amplitude_queue))
        else:
            agc_max = float("nan")
        return {
            "agc_enabled": agc_enabled,
            "agc_window_slots": self._agc_window_slots,
            "agc_queue_size": len(self._agc_max_amplitude_queue),
            "agc_noise_floor_db": float(self._agc_noise_floor_db),
            "agc_max_db": agc_max,
            "agc_updates": int(self._agc_updates),
            "agc_snr_margin_db": float(self._detection.agc_snr_margin_db),
        }

    # =================================================================
    # Evaluation-only accessors (for MetricsEngine)
    # =================================================================

    @property
    def discretised_grid(self) -> DiscretisedGrid:
        """
        [EVALUATION-ONLY] The discretised TSRD PDW grid (Option B).
        MetricsEngine may access this. Schedulers must NOT.
        """
        if self._discretised_grid is None:
            raise RuntimeError("No discretised grid; call reset() first.")
        return self._discretised_grid

    @property
    def deinterleaver_result(self) -> Optional[DeinterleaverResult]:
        """
        [EVALUATION-ONLY] Deinterleaver output (Option C), or None
        if deinterleaving was not requested.
        """
        return self._deinterleaver_result

    @property
    def observation_history(self) -> List[Dict]:
        """[EVALUATION-ONLY] Observation history for metrics computation."""
        return self._observation_history

    # =================================================================
    # Internal methods
    # =================================================================

    def _build_grid(self, pdw: PDWStream) -> None:
        """Option B: discretise the PDW stream onto the band/slot grid."""
        provenance = {
            "tsrd_environment_version": "1.0.0",
            "discretiser": "tsrd.pdw_discretiser.discretise_pdw_to_grid",
            "nominal_noise_floor_db": self._detection.nominal_noise_floor_db,
            "provenance_label": "[TSRD-DERIVED]",
        }
        self._discretised_grid = discretise_pdw_to_grid(
            pdw,
            self._config,
            nominal_noise_floor_db=self._detection.nominal_noise_floor_db,
            provenance=provenance,
        )

    def _run_deinterleaver(self, pdw: PDWStream) -> None:
        """Option C: run the PRI-based deinterleaver and index tracks by cell."""
        deinterleaver = PRIBasedDeinterleaver(self._deint_config)
        self._deinterleaver_result = deinterleaver.deinterleave(pdw)

        # Index tracks by (band, slot) for fast per-cell lookup
        self._deint_tracks_by_cell: Dict[Tuple[int, int], List[EmitterTrack]] = (
            defaultdict(list)
        )
        for track in self._deinterleaver_result.tracks:
            from ..core.mapping import frequency_to_band, seconds_to_slot
            for k in range(track.n_pulses):
                toa_s = float(track.toa_us[k]) * 1e-6
                band = int(frequency_to_band(float(track.freq_mhz[k]), self._config))
                slot = int(seconds_to_slot(toa_s, self._config))
                if 0 <= band < self._config.band_count and 0 <= slot < self._config.time_slots:
                    self._deint_tracks_by_cell[(band, slot)].append(track)

    def _detect(self, band: int, slot: int, dwell_s: float) -> bool:
        """
        Run the detection model for cell (band, slot).

        Logic:
          1. If dwell_s == 0 (retune ate the whole slot): miss.
          2. Get the cell's BandSlotCell from the discretised grid.
          3. If cell is empty: false-alarm Bernoulli draw.
          4. If cell has pulses: compute live AGC-calibrated SNR
             (max_amplitude_db − agc_noise_floor_db) and apply the
             SNR-to-Pd curve.

        The AGC-calibrated SNR differs from the cell's stored
        ``snr_db`` (which is computed at grid-build time using
        the fixed ``nominal_noise_floor_db``).  The AGC model
        dynamically re-bases the floor on the receiver's recent
        operating point, which moves with multipath and gain
        changes. This makes the detection model robust to absolute
        amplitude calibration drift and to the TSRD's
        relative-only amplitude scale.
        """
        if dwell_s <= 0.0:
            return False

        cell: BandSlotCell = self._discretised_grid[band, slot]

        if cell.is_empty:
            # Empty cell: false alarm draw
            return bool(
                self._rng.random()
                < float(self._detection.false_alarm_probability)
            )

        # Cell with pulses: AGC-calibrated SNR-based detection.
        # The AGC noise floor is `recent_max_db - dynamic_range_db`
        # (or `agc_no_signal_floor_db` if no recent non-empty
        # cells). The effective SNR passed to the Pd curve
        # subtracts this floor AND the AGC margin, so a pulse must
        # be `margin` dB above the AGC floor to be reliably
        # detected. This is the headroom that suppresses multipath
        # when the AGC has tracked up to a strong direct path.
        live_snr_db = (
            float(cell.max_amplitude_db)
            - float(self._agc_noise_floor_db)
            - float(self._detection.agc_snr_margin_db)
        )
        # Coherent integration gain: N pulses from the same emitter
        # add 10*log10(N) dB to the SNR. Applied *before* the
        # margin so the margin still suppresses weak multipath but
        # the direct-path burst gets the full integration boost.
        if (
            self._detection.coherent_integration_enabled
            and cell.pulse_count > 1
        ):
            live_snr_db += cell.coherent_integration_gain_db(
                max_pulses=self._detection.coherent_integration_max_pulses,
                non_coherent_loss_db=self._detection.coherent_integration_non_coherent_loss_db,
            )
        # If AGC margin exceeds the AGC-calibrated SNR, the pulse
        # is below the AGC-derived threshold (e.g. strong multipath
        # when the AGC has tracked up to a strong direct path).
        # We apply the standard Pd curve, which will return a low
        # Pd for small or negative SNR.
        pd = self._detection.pd_for_snr(live_snr_db)
        return bool(self._rng.random() < pd)

    # =================================================================
    # Helpers for metrics / evaluation (evaluation-only)
    # =================================================================

    def cell_at(self, band: int, slot: int) -> BandSlotCell:
        """
        [EVALUATION-ONLY] Return the BandSlotCell for (band, slot).
        """
        return self._discretised_grid[band, slot]

    def emitter_arrival_slots(self) -> Dict[int, int]:
        """
        [EVALUATION-ONLY] Per-emitter arrival slot, inferred from the
        first pulse of each emitter in the PDW stream.

        This is used by evaluation metrics to avoid counting
        pre-arrival misses as false negatives, and by Option A
        (synthetic truth) to configure per-emitter ``arrival_slot``
        correctly when the H5 lacks ``power_config.start_time_s``.

        The cache is cleared on ``reset()`` so each episode gets a fresh
        inference from the current PDW stream.
        """
        if self._emitter_arrival_slots is None:
            self._emitter_arrival_slots = self._infer_arrival_slots_from_pdw()
        return self._emitter_arrival_slots

    def _infer_arrival_slots_from_pdw(self) -> Dict[int, int]:
        """
        Find the first pulse of each emitter and convert ToA → slot.
        """
        if self._pdw_stream is None:
            return {}
        slot_s = self._config.slot_duration_s()
        if slot_s <= 0:
            return {}
        result: Dict[int, int] = {}
        for label in np.unique(self._pdw_stream.emitter_id):
            mask = self._pdw_stream.emitter_id == label
            if not mask.any():
                continue
            first_toa_us = float(self._pdw_stream.toa_us[mask].min())
            slot = max(0, int(first_toa_us * 1e-6 / slot_s))
            result[int(label)] = slot
        return result

    def tracks_at(self, band: int, slot: int) -> List[EmitterTrack]:
        """
        [EVALUATION-ONLY] Return deinterleaved tracks present in (band, slot).
        Returns an empty list if deinterleaving was not requested
        (Option B only) or if no tracks cover this cell.
        """
        if self._deint_tracks_by_cell is None:
            return []
        return self._deint_tracks_by_cell.get((band, slot), [])

    def unique_emitters_in_cell(self, band: int, slot: int) -> np.ndarray:
        """
        [EVALUATION-ONLY] Unique emitter IDs present in (band, slot).
        """
        cell = self._discretised_grid[band, slot]
        return cell.unique_emitter_ids

    def occupancy_mask(self) -> np.ndarray:
        """
        [EVALUATION-ONLY] Boolean mask of observed cells.
        """
        return self._discretised_grid.occupancy_mask()

    def total_pulse_count(self) -> int:
        """
        [EVALUATION-ONLY] Total pulses in the discretised grid.
        """
        return self._discretised_grid.total_pulse_count

    def occupied_cell_count(self) -> int:
        """
        [EVALUATION-ONLY] Number of cells containing at least one pulse.
        """
        return self._discretised_grid.occupied_cell_count


# =====================================================================
# Factory helpers (for the Kaggle runner)
# =====================================================================

def build_tsrd_environment(
    h5_path: str,
    simulation_config: SimulationConfig,
    detection_config: Optional[DetectionConfig] = None,
    deinterleaver_config: Optional[DeinterleaverConfig] = None,
    data_mode: TSRDDataMode = TSRDDataMode.REAL_TSRD,
    sha256_pin: Optional[str] = None,
    seed: int = 42,
) -> TSRDEnvironment:
    """
    Factory: open a TSRD H5 file and build a TSRDEnvironment.

    This is the one-liner for the Kaggle runner::

        env = build_tsrd_environment(
            h5_path="/kaggle/input/.../config_0.h5",
            simulation_config=sim_cfg,
            deinterleaver_config=DeinterleaverConfig(),
        )

    Parameters
    ----------
    h5_path : str
        Path to the TSRD H5 file.
    simulation_config : SimulationConfig
        Grid parameters. Must match the H5's collection parameters
        (band_count, time_slots).
    detection_config : DetectionConfig | None
        Detection model. Defaults to the standard config.
    deinterleaver_config : DeinterleaverConfig | None
        If provided, enables Option C (deinterleaver).
    data_mode : TSRDDataMode
        REAL_TSRD or FIXTURE.
    sha256_pin : str | None
        If provided, verify the H5 SHA-256 matches before loading.
    seed : int
        RNG seed for the detector model.

    Raises
    ------
    DataUnavailableError
        If the H5 file is missing.
    DataIntegrityError
        If the SHA-256 pin does not match.
    StareModeOracleError
        If the H5 is a Stare-mode file (use the oracle for those).
    """
    adapter = TSRDAdapter(
        h5_path=h5_path,
        data_mode=data_mode,
        sha256_pin=sha256_pin,
        simulation_config=simulation_config,
    )
    # [TSRD-DERIVED] If the H5 carries `dwell_centres_mhz`, apply them
    # to the SimulationConfig so `frequency_to_band` uses the real
    # scan-receiver tune grid instead of the uniform-band-centre
    # fallback. Two emitters straddling a band boundary now map to
    # different bands, which is the assignment a real ESM receiver
    # would produce.
    cfg_with_centres = adapter.apply_dwell_centres_to_config(simulation_config)
    pdw = adapter.to_pdw_stream()
    return TSRDEnvironment(
        pdw_stream=pdw,
        simulation_config=cfg_with_centres,
        detection_config=detection_config,
        deinterleaver_config=deinterleaver_config,
        seed=seed,
    )


__all__ = [
    "DetectionConfig",
    "TSRDEnvironment",
    "build_tsrd_environment",
]

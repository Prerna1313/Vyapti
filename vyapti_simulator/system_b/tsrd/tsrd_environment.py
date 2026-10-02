"""
vyapti_simulator.system_b.tsrd.tsrd_environment
======================================

TSRD-driven simulation environment — Options B and C wired together.

This module provides ``TSRDEnvironment``, the environment class that
replaces ``VyaptiEnv`` when running on real TSRD data.
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

The separate ``TSRDStareEnvironment`` below preserves a binary replay profile
and offers an opt-in ``pdw_v2`` profile with causal, label-free measured PDWs.
That profile has its own guarded observation extension and metric contract.

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
``TSRDEnvironment`` is a drop-in replacement for ``VyaptiEnv``
in the experiment runner. The episode loop in ``core.episode``
does not know whether it is driving a synthetic or TSRD environment;
the environment interface is identical.

Usage (Kaggle)
--------------
::

    from vyapti_simulator.system_b.tsrd import (
        TSRDAdapter, TSRDDataMode,
        DiscretisedGrid, quick_deinterleave,
    )
    from vyapti_simulator.system_b.tsrd.tsrd_environment import TSRDEnvironment
    from vyapti_simulator.core.environment import SimulationConfig

    # Locked TSRD grid
    sim_cfg = SimulationConfig(
        band_count=36, total_spectrum_mhz=18000.0,
        receiver_ibw_mhz=500.0, dwell_time_ms=50.0,
        retune_time_ms=0.3, time_slots=600,
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
Senior RF/EW Signal Simulation Engineer — Vyapti Options B+C integration.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
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
    FeatureBasedDeinterleaver,
    EmitterTrack,
)
from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.core.receiver_observation import ReceiverObservation

from vyapti_simulator.core.detection_config import DetectionConfig, ShnidmanDetectionConfig



# =====================================================================
# Detection model
# =====================================================================



# =====================================================================
# Shnidman / Albersheim detection model
# =====================================================================



# =====================================================================
# TSRD Environment
# =====================================================================

class TSRDEnvironment:
    """
    TSRD-driven simulation environment for Options B and C.

    Provides the same interface as ``VyaptiEnv`` so that
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
    contract is identical to ``VyaptiEnv`` and
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
    # Public interface (same as VyaptiEnv)
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
        ``VyaptiEnv.reset()`` — the experiment runner
        calls this between episodes.

        The ``emitter_family_config`` argument is accepted for
        interface compatibility with ``VyaptiEnv`` and
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
        Pulse data is used internally by the detector, not returned to the
        scheduler as pre-detection cell statistics.

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
        dwell_s = max(0.0, self._config.slot_duration_s() - retune_cost_s)

        # --- Get the BandSlotCell (Gap 4: pw_us, aoa_deg, amp_db used) ---
        cell: BandSlotCell = self._discretised_grid[selected_band, t]
        if not cell.is_empty and retune_cost_s > 0.0:
            start_us = (t * self._config.slot_duration_s() + retune_cost_s) * 1e6
            visible = cell.toa_us >= start_us
            if not np.all(visible):
                amplitudes = cell.amp_db[visible]
                cell = replace(
                    cell,
                    pulse_count=int(np.sum(visible)),
                    emitter_ids=cell.emitter_ids[visible],
                    toa_us=cell.toa_us[visible],
                    freq_mhz=cell.freq_mhz[visible],
                    pw_us=cell.pw_us[visible],
                    aoa_deg=cell.aoa_deg[visible],
                    amp_db=amplitudes,
                    max_amplitude_db=float(np.max(amplitudes)) if amplitudes.size else np.nan,
                    min_amplitude_db=float(np.min(amplitudes)) if amplitudes.size else np.nan,
                    snr_db=(float(np.max(amplitudes)) - self._detection.nominal_noise_floor_db
                            if amplitudes.size else np.nan),
                    is_empty=not bool(amplitudes.size),
                )

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

        # --- Amplitude-based detection -----------------------
        hit = self._detect(selected_band, t, dwell_s, cell)

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
                "noise_figure_db": self._detection.noise_figure_db,
                "antenna_gain_db": (
                    self._detection.antenna_gain_db.tolist()
                    if self._detection.antenna_gain_db.size
                    else [0.0] * self._config.band_count
                ),
            },
            # Gate 0 negative markers — audited explicitly
            "truth_excluded": True,
            "emitter_identity_excluded": True,
            "future_state_excluded": True,
        }
        observation = ReceiverObservation.from_mapping(observation)
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

    def _get_band_antenna_gain(self, band: int) -> float:
        """
        Return the antenna gain in dB for a given band.

        If ``antenna_gain_db`` is empty (default), returns 0.0 dB
        (uniform isotropic reference gain). Otherwise returns the
        pre-stored gain for the requested band.

        Applied as an SNR penalty: ``effective_snr = measured_snr - gain_db``.
        A -6 dB antenna gain in band 3 means every pulse in band 3
        appears 6 dB weaker than it would with an isotropic antenna.
        """
        gains = self._detection.antenna_gain_db
        if gains.size == 0:
            return 0.0
        if not (0 <= band < len(gains)):
            return 0.0
        return float(gains[band])

    def _build_grid(self, pdw: PDWStream) -> None:
        """Option B: discretise the PDW stream onto the band/slot grid."""
        provenance = {
            "tsrd_environment_version": "1.0.0",
            "discretiser": "tsrd.pdw_discretiser.discretise_pdw_to_grid",
            "nominal_noise_floor_db": self._detection.nominal_noise_floor_db,
            "noise_figure_db": self._detection.noise_figure_db,
            "provenance_label": "[TSRD-DERIVED]",
        }
        self._discretised_grid = discretise_pdw_to_grid(
            pdw,
            self._config,
            nominal_noise_floor_db=self._detection.nominal_noise_floor_db,
            provenance=provenance,
        )

    def _run_deinterleaver(self, pdw: PDWStream) -> None:
        """Option C: run the feature-based deinterleaver and index tracks by cell."""
        deinterleaver = FeatureBasedDeinterleaver(self._deint_config)
        self._deinterleaver_result = deinterleaver.deinterleave(pdw)

        # Index tracks by (band, slot) for fast per-cell lookup
        self._deint_tracks_by_cell: Dict[Tuple[int, int], List[EmitterTrack]] = (
            defaultdict(list)
        )
        for track in self._deinterleaver_result.tracks:
            from vyapti_simulator.core.mapping import frequency_to_band, seconds_to_slot
            for k in range(track.n_pulses):
                toa_s = float(track.toa_us[k]) * 1e-6
                band = int(frequency_to_band(float(track.freq_mhz[k]), self._config))
                slot = int(seconds_to_slot(toa_s, self._config))
                if 0 <= band < self._config.band_count and 0 <= slot < self._config.time_slots:
                    self._deint_tracks_by_cell[(band, slot)].append(track)

    def _detect(self, band: int, slot: int, dwell_s: float,
                cell: Optional[BandSlotCell] = None) -> bool:
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

        if cell is None:
            cell = self._discretised_grid[band, slot]

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
            - self._get_band_antenna_gain(band)
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


class TSRDStareEnvironment:
    """
    Causal receiver replay over one TSRD stare recording.

    The hidden grid represents recorded-pulse opportunities, not all emitted
    pulses. Each action is one fixed 50 ms receiver dwell; scan-mode recordings
    are not merged into the world because they are separately censored runs.
    """
    def __init__(
        self, n_bands, n_slots, occupancy_grid, scan_grid, sim_config,
        stare_data=None, stare_labels=None, *, band_centres_mhz=None,
        passband_halfwidth_mhz=None, receiver_profile="binary_v1",
        amplitude_midpoint_db=-90.0, amplitude_scale_db=5.0,
        max_observed_pdws=32,
    ):
        self.n_bands = n_bands
        self.n_slots = n_slots
        self._occupancy_grid = np.asarray(occupancy_grid, dtype=bool)
        if self._occupancy_grid.shape != (n_bands, n_slots):
            raise ValueError("Occupancy grid shape does not match receiver configuration")
        self._scan_grid = scan_grid
        self._config = sim_config
        if receiver_profile not in ("binary_v1", "pdw_v2"):
            raise ValueError("receiver_profile must be binary_v1 or pdw_v2")
        if (not np.isfinite(amplitude_midpoint_db)
                or not np.isfinite(amplitude_scale_db) or amplitude_scale_db <= 0
                or not isinstance(max_observed_pdws, int) or max_observed_pdws < 1):
            raise ValueError("Invalid PDW receiver settings")
        self._receiver_profile = receiver_profile
        self._amplitude_midpoint_db = float(amplitude_midpoint_db)
        self._amplitude_scale_db = float(amplitude_scale_db)
        self._max_observed_pdws = max_observed_pdws
        self._centres_mhz = np.asarray(
            band_centres_mhz if band_centres_mhz is not None
            else (np.arange(n_bands) + 0.5) * sim_config.band_width_mhz(),
            dtype=np.float64,
        )
        self._halfwidth_mhz = float(
            passband_halfwidth_mhz if passband_halfwidth_mhz is not None
            else sim_config.receiver_ibw_mhz / 2.0
        )
        if self._centres_mhz.shape != (n_bands,) or self._halfwidth_mhz <= 0:
            raise ValueError("Invalid TSRD receiver passbands")
        self._current_slot = 0
        self._observation_history = []
        self._deinterleaver_result = None
        self._rng = np.random.default_rng()
        self._stare_data = None if stare_data is None else np.asarray(stare_data)
        self._stare_labels = None if stare_labels is None else np.asarray(stare_labels).reshape(-1)
        if receiver_profile == "pdw_v2" and self._stare_data is None:
            raise ValueError("pdw_v2 requires recorded TSRD PDWs")
        if self._stare_data is not None:
            if self._stare_data.ndim != 2 or self._stare_data.shape[1] < 5:
                raise ValueError("TSRD stare data must have five PDW columns")
            if self._stare_labels is not None and len(self._stare_labels) != len(self._stare_data):
                raise ValueError("TSRD labels and PDWs must have equal length")
            if np.any(np.diff(self._stare_data[:, 0]) < 0):
                order = np.argsort(self._stare_data[:, 0], kind="stable")
                self._stare_data = self._stare_data[order]
                if self._stare_labels is not None:
                    self._stare_labels = self._stare_labels[order]
            self._toa_us = self._stare_data[:, 0].astype(np.float64, copy=False)
        else:
            self._toa_us = np.empty(0, dtype=np.float64)
        self._hidden_truth = self._build_hidden_truth()
        import hashlib
        signature = hashlib.sha256()
        signature.update(np.ascontiguousarray(self._occupancy_grid).tobytes())
        signature.update(np.ascontiguousarray(self._centres_mhz).tobytes())
        signature.update(np.float64(self._halfwidth_mhz).tobytes())
        if self._stare_data is not None:
            signature.update(np.ascontiguousarray(self._stare_data).tobytes())
        if self._stare_labels is not None:
            signature.update(np.ascontiguousarray(self._stare_labels).tobytes())
        self._signature = signature.hexdigest()
        self._noise_field = None
        self.reset()

    @classmethod
    def from_stare_mode(
        cls, stare_file: str, scan_file: str = None, sim_config=None, *,
        band_centres_mhz=None,
        passband_halfwidth_mhz: float | None = None,
        receiver_profile: str = "binary_v1",
        amplitude_midpoint_db: float = -90.0,
        amplitude_scale_db: float = 5.0,
        max_observed_pdws: int = 32,
        detection_probability: float = 1.0,
        false_alarm_probability: float = 0.0,
        retune_time_ms: float = 0.0,
    ):
        import h5py
        from .tsrd_adapter import stare_pulse_occupancy

        with h5py.File(stare_file, "r") as f:
            receiver = f["metadata/receiver"]
            mode = receiver.attrs.get("scan_mode", "")
            if isinstance(mode, bytes):
                mode = mode.decode()
            if str(mode).lower() != "stare":
                raise ValueError("TSRD replay world requires a stare-mode recording")
            centres = np.asarray(receiver["dwell_centres_mhz"][:], dtype=np.float64)
            halfwidth = float(receiver.attrs["bandwith_mhz"])
            collection_s = float(receiver.attrs["collection_time_s"])
            data = f["data"][:]
            raw_labels = np.asarray(f["labels"][:])
        if passband_halfwidth_mhz is not None and (
            not np.isfinite(passband_halfwidth_mhz)
            or passband_halfwidth_mhz <= 0
            or not np.isclose(halfwidth, passband_halfwidth_mhz)
        ):
            raise ValueError("Configured passband halfwidth disagrees with TSRD metadata")
        explicit_centres = (None if band_centres_mhz is None else
                            np.asarray(band_centres_mhz, dtype=np.float64))
        if explicit_centres is not None and (
            explicit_centres.ndim != 1 or len(explicit_centres) < 2
            or not np.all(np.isfinite(explicit_centres))
            or not np.all(np.diff(explicit_centres) > 0)
        ):
            raise ValueError("Explicit TSRD tune centres must be finite and increasing")
        if not len(centres):
            if scan_file is not None:
                with h5py.File(scan_file, "r") as scan:
                    centres = np.asarray(
                        scan["metadata/receiver/dwell_centres_mhz"][:],
                        dtype=np.float64,
                    )
            elif explicit_centres is not None:
                centres = explicit_centres
            elif sim_config is not None and sim_config.dwell_centres_mhz is not None:
                centres = np.asarray(sim_config.dwell_centres_mhz, dtype=np.float64)
            else:
                raise ValueError(
                    "Stare metadata has no tune centres; provide the matching "
                    "scan file or an explicit dwell-centre configuration"
                )
        if (centres.ndim != 1 or len(centres) < 2
                or not np.all(np.isfinite(centres))
                or not np.all(np.diff(centres) > 0)):
            raise ValueError("Invalid TSRD tune centres")
        if explicit_centres is not None and not np.array_equal(centres, explicit_centres):
            raise ValueError("Explicit tune centres disagree with TSRD scan metadata")
        labels = raw_labels[:, 0] if raw_labels.ndim == 2 else raw_labels
        if len(labels) != len(data):
            raise ValueError("TSRD labels and PDWs must have equal length")
        n_bands = len(centres)
        n_slots = int(round(collection_s / 0.05))
        spectrum_mhz = float((centres[-1] - centres[0]) + np.median(np.diff(centres)))
        if sim_config is None:
            sim_config = SimulationConfig(
                receiver_ibw_mhz=2.0 * halfwidth,
                total_spectrum_mhz=spectrum_mhz,
                band_count=n_bands,
                time_slots=n_slots,
                dwell_time_ms=50.0,
                retune_time_ms=retune_time_ms,
                detection_probability=detection_probability,
                false_alarm_probability=false_alarm_probability,
                dwell_centres_mhz=centres,
            )
        elif (sim_config.band_count != n_bands or sim_config.time_slots != n_slots
              or not np.isclose(sim_config.dwell_time_ms, 50.0)):
            raise ValueError("TSRD replay requires the H5 band count, 50 ms slots, and full duration")
        else:
            sim_config = replace(
                sim_config, dwell_centres_mhz=centres,
                receiver_ibw_mhz=2.0 * halfwidth,
                total_spectrum_mhz=spectrum_mhz,
            )
        occupancy = stare_pulse_occupancy(data, centres, halfwidth, n_slots)
        return cls(
            n_bands, n_slots, occupancy, None, sim_config,
            stare_data=data, stare_labels=labels,
            band_centres_mhz=centres, passband_halfwidth_mhz=halfwidth,
            receiver_profile=receiver_profile,
            amplitude_midpoint_db=amplitude_midpoint_db,
            amplitude_scale_db=amplitude_scale_db,
            max_observed_pdws=max_observed_pdws,
        )

    @property
    def hidden_truth(self):
        return self._hidden_truth

    @property
    def occupancy_grid(self):
        return self._occupancy_grid.copy()

    @property
    def recorded_pulses(self):
        """Evaluation-only PDWs and labels; never include these in observations."""
        return self._stare_data, self._stare_labels

    def recorded_capture_indices(self, slot: int):
        """Evaluation-only captured stare-row indices for the opt-in PDW receiver."""
        if self._receiver_profile != "pdw_v2":
            raise ValueError("Pulse capture indices are unavailable in binary_v1")
        if not 0 <= slot < len(self._captured_pulse_indices):
            raise ValueError("Slot has not been observed")
        return self._captured_pulse_indices[slot].copy()

    @property
    def receiver_geometry(self):
        return self._centres_mhz.copy(), self._halfwidth_mhz

    @property
    def receiver_profile(self):
        return self._receiver_profile

    def _build_hidden_truth(self):
        from vyapti_simulator.core.environment import (
            HiddenTruthGrid, EmitterConfig, EmitterBehaviorType,
        )
        if self._stare_data is None or self._stare_labels is None:
            mask_3d = self._occupancy_grid[np.newaxis, :, :]
            configs = [EmitterConfig(
                emitter_id=0, behavior=EmitterBehaviorType.MIXED_POPULATION,
                active_bands=np.flatnonzero(self._occupancy_grid.any(axis=1)).tolist(),
            )]
            return HiddenTruthGrid(grid=mask_3d, emitter_configs=configs,
                                   band_count=self.n_bands, time_slots=self.n_slots)

        labels, emitter_index = np.unique(self._stare_labels, return_inverse=True)
        grid = np.zeros((len(labels), self.n_bands, self.n_slots), dtype=bool)
        dwell_us = self._config.dwell_time_ms * 1000.0
        toa = self._stare_data[:, 0].astype(np.float64, copy=False)
        freq = self._stare_data[:, 1].astype(np.float64, copy=False)
        valid = np.isfinite(toa) & np.isfinite(freq) & (toa >= 0)
        slots = np.floor(np.where(valid, toa, 0.0) / dwell_us).astype(np.int64)
        valid &= slots < self.n_slots
        for band, centre in enumerate(self._centres_mhz):
            covered = valid & (np.abs(freq - centre) <= self._halfwidth_mhz)
            grid[emitter_index[covered], band, slots[covered]] = True
        configs = [
            EmitterConfig(
                emitter_id=int(label),
                behavior=EmitterBehaviorType.MIXED_POPULATION,
                active_bands=np.flatnonzero(grid[i].any(axis=1)).tolist(),
            )
            for i, label in enumerate(labels)
        ]
        return HiddenTruthGrid(grid=grid, emitter_configs=configs,
                               band_count=self.n_bands, time_slots=self.n_slots)

    @property
    def config(self):
        return self._config

    @property
    def current_slot(self) -> int:
        return self._current_slot

    def reset(self, seed=None, emitter_family_config=None):
        self._current_slot = 0
        self._observation_history.clear()
        self._captured_pulse_indices = []
        self._rng = np.random.default_rng(seed)
        receiver_seed = int(self._rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        import hashlib
        noise_key = hashlib.sha256(
            f"{self._signature}:{receiver_seed}".encode("ascii")
        ).digest()
        field_seed = int.from_bytes(noise_key[:8], "little")
        self._noise_field = np.random.default_rng(field_seed).random(
            (self.n_bands, self.n_slots)
        )
        if self._receiver_profile == "pdw_v2":
            self._pulse_noise = np.random.default_rng(field_seed ^ 0x5A17C0DE).random(
                len(self._stare_data)
            )
            self._false_pdw_field = np.random.default_rng(
                field_seed ^ 0xDC0FFEE5
            ).random((self.n_bands, self.n_slots, 5))

    def _detect_recorded_pdws(self, band, slot, retune, dwell_s):
        """Capture current-window PDWs with potential noise fixed by world and seed."""
        empty_indices = np.empty(0, dtype=np.intp)
        if dwell_s <= 0:
            return False, None, empty_indices
        start_us = round((slot * self._config.slot_duration_s() + retune) * 1e6, 9)
        end_us = round((slot + 1) * self._config.slot_duration_s() * 1e6, 9)
        lo = int(np.searchsorted(self._toa_us, start_us, side="left"))
        hi = int(np.searchsorted(self._toa_us, end_us, side="left"))
        window = self._stare_data[lo:hi]
        passband = np.isfinite(window[:, 1]) & (
            np.abs(window[:, 1] - self._centres_mhz[band]) <= self._halfwidth_mhz
        )
        valid = passband & np.all(np.isfinite(window[:, :5]), axis=1)
        candidates = np.flatnonzero(valid) + lo
        if len(candidates):
            amplitude = self._stare_data[candidates, 4].astype(np.float64)
            z = np.clip(
                (amplitude - self._amplitude_midpoint_db) / self._amplitude_scale_db,
                -60.0, 60.0,
            )
            pd = float(self._config.detection_probability) / (1.0 + np.exp(-z))
            captured = candidates[self._pulse_noise[candidates] < pd]
            if len(captured):
                selected = captured[np.linspace(
                    0, len(captured) - 1,
                    min(len(captured), self._max_observed_pdws), dtype=int,
                )]
                pdws = [
                    {
                        "toa_offset_us": float(self._stare_data[i, 0] - start_us),
                        "frequency_mhz": float(self._stare_data[i, 1]),
                        "pulse_width_us": float(self._stare_data[i, 2]),
                        "aoa_deg": float(self._stare_data[i, 3]),
                        "amplitude_db": float(self._stare_data[i, 4]),
                    }
                    for i in selected
                ]
                return True, {"pulse_count": int(len(captured)), "pdws": pdws}, captured
            return False, None, empty_indices
        if np.any(passband):
            return False, None, empty_indices
        if self._noise_field[band, slot] >= float(self._config.false_alarm_probability):
            return False, None, empty_indices
        u = self._false_pdw_field[band, slot]
        return True, {
            "pulse_count": 1,
            "pdws": [{
                "toa_offset_us": float(u[0] * dwell_s * 1e6),
                "frequency_mhz": float(self._centres_mhz[band]
                                       + (2 * u[1] - 1) * self._halfwidth_mhz),
                "pulse_width_us": float(0.1 * 1000 ** u[2]),
                "aoa_deg": float(360 * u[3] - 180),
                "amplitude_db": float(self._amplitude_midpoint_db
                                      + (2 * u[4] - 1) * self._amplitude_scale_db),
            }],
        }, empty_indices

    def eligible_recorded_pulse(self, band: int, slot: int, retune_cost_s: float = 0.0) -> bool:
        """Evaluation-side window query; never pass its result to the scheduler."""
        if not (0 <= band < self.n_bands and 0 <= slot < self.n_slots):
            return False
        if retune_cost_s >= self._config.slot_duration_s():
            return False
        if self._stare_data is None:
            return bool(self._occupancy_grid[band, slot]) and retune_cost_s == 0.0
        start_us = round((slot * self._config.slot_duration_s() + retune_cost_s) * 1e6, 9)
        end_us = round((slot + 1) * self._config.slot_duration_s() * 1e6, 9)
        lo = np.searchsorted(self._toa_us, start_us, side="left")
        hi = np.searchsorted(self._toa_us, end_us, side="left")
        freq = self._stare_data[lo:hi, 1]
        return bool(np.any(np.abs(freq - self._centres_mhz[band]) <= self._halfwidth_mhz))

    def step(self, band: int):
        if self.done:
            raise RuntimeError("Episode finished; call reset() before stepping again.")
        band = int(band)
        if not (0 <= band < self.n_bands):
            raise ValueError(f"Invalid band {band}; expected 0..{self.n_bands - 1}")
        previous = self._observation_history[-1] if self._observation_history else None
        retune = (self._config.retune_time_ms / 1000.0
                  if previous is not None and previous["selected_band"] != band else 0.0)
        dwell_s = max(0.0, self._config.slot_duration_s() - retune)
        if self._stare_data is None and retune:
            raise ValueError("Pulse timestamps are required to model retune loss")
        if self._receiver_profile == "pdw_v2":
            hit, measurement, captured = self._detect_recorded_pdws(
                band, self._current_slot, retune, dwell_s
            )
            self._captured_pulse_indices.append(captured)
        else:
            observed_pulse = self.eligible_recorded_pulse(band, self._current_slot, retune)
            hit_probability = (self._config.detection_probability if observed_pulse
                               else self._config.false_alarm_probability)
            hit = bool(
                dwell_s > 0.0
                and self._noise_field[band, self._current_slot] < float(hit_probability)
            )
        obs = {
            "time_slot": self._current_slot,
            "selected_band": band,
            "hit": hit,
            "retune_cost_s": retune,
            "dwell_elapsed_s": max(0.0, self._config.slot_duration_s() - retune),
            "receiver_metadata": {"dwell_time_ms": self._config.dwell_time_ms,
                                   "retune_time_ms": self._config.retune_time_ms},
            "truth_excluded": True,
            "emitter_identity_excluded": True,
            "future_state_excluded": True,
        }
        if self._receiver_profile == "pdw_v2":
            obs["receiver_measurement"] = measurement
        obs = ReceiverObservation.from_mapping(obs)
        self._observation_history.append(obs)
        self._current_slot += 1
        return obs, False

    def step_training(self, band: int, reward_mode: str = "false_alarm_aware"):
        """Training reward; mode is fixed by the experiment before an episode."""
        if reward_mode not in ("detector_positive", "false_alarm_aware"):
            raise ValueError("Unknown TSRD training reward mode")
        obs, _ = self.step(band)
        if reward_mode == "detector_positive":
            return obs, float(obs["hit"]), self.done
        eligible = self.eligible_recorded_pulse(
            band, obs["time_slot"], obs["retune_cost_s"]
        )
        reward = 1.0 if obs["hit"] and eligible else -1.0 if obs["hit"] else 0.0
        return obs, reward, self.done

    def step_dwell(self, band: int, dwell_slots: int):
        """Execute a non-preemptible 50/100 ms dwell as independent base looks."""
        if dwell_slots not in (1, 2):
            raise ValueError("Mixed TSRD dwell must consume one or two 50 ms slots")
        if self.done:
            raise RuntimeError("Episode finished; call reset()")
        actual_slots = min(dwell_slots, self.n_slots - self._current_slot)
        return [self.step(band)[0] for _ in range(actual_slots)]

    def step_dwell_training(
        self, band: int, dwell_slots: int, reward_mode: str = "detector_positive"
    ):
        """Train on a whole dwell; current-step rewards are summed over its looks."""
        if dwell_slots not in (1, 2):
            raise ValueError("Mixed TSRD dwell must consume one or two 50 ms slots")
        if self.done:
            raise RuntimeError("Episode finished; call reset()")
        actual_slots = min(dwell_slots, self.n_slots - self._current_slot)
        looks = [self.step_training(band, reward_mode) for _ in range(actual_slots)]
        return [look[0] for look in looks], sum(look[1] for look in looks), self.done

    @property
    def done(self) -> bool:
        return self._current_slot >= self.n_slots

    def receiver_accounting(self):
        actions = [obs["selected_band"] for obs in self._observation_history]
        retunes = sum(a != b for a, b in zip(actions, actions[1:]))
        overhead_s = sum(obs["retune_cost_s"] for obs in self._observation_history)
        elapsed_s = self._current_slot * self._config.slot_duration_s()
        return {
            "retune_count": float(retunes),
            "retune_overhead_total_s": float(overhead_s),
            "dead_time_fraction": float(overhead_s / elapsed_s) if elapsed_s else 0.0,
        }

    def replay_signature(self):
        import hashlib
        profile = (
            self._receiver_profile, self._amplitude_midpoint_db,
            self._amplitude_scale_db, self._max_observed_pdws,
            float(self._config.detection_probability),
            float(self._config.false_alarm_probability),
            float(self._config.retune_time_ms),
        )
        return hashlib.sha256(f"{self._signature}:{profile!r}".encode("ascii")).hexdigest()

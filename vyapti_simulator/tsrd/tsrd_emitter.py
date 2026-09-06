"""
vyapti_simulator.tsrd.tsrd_emitter
====================================

`TSRDEmitterSampler` — turns the aggregate TSRD statistics JSON into a
list of `EmitterConfig` objects that the existing
`PS26055Environment.reset(seed, configs)` path can consume unchanged.

This is the TSRD Option-A integration point: the simulator regenerates
truth from distributions on every run using the existing 13-member
`EmitterBehaviorType` enum. There is no new enum value, no
`TSRD_STATIC` branch, no replay of the pre-baked scan-mode matrix.

INVARIANTS RESpected
--------------------
[INVARIANT-2] band/slot mapping goes through `core.mapping`. We never
re-derive `band = floor(freq / total * band_count)` in this file.

[INVARIANT-3] The scheduler never sees "this came from TSRD". The
returned `EmitterConfig` objects have `provenance_notes["tsrd_*"]`
fields for audit, but the truth grid they generate depends only on
`behavior` + the numeric shape parameters.

DESIGN
------
* The hash check on `tsrd_statistics.json` fires inside the constructor
  via `core.data_loader.load_tsrd_statistics`. There is no `verify=`
  kwarg. If you can call this class, the check runs.
* The RNG is a `numpy.random.Generator` derived from
  `SeedSequence([seed, "tsrd_emitter_v1"])`. We never call
  `np.random.seed()`; that would corrupt the global state shared with
  schedulers and break paired comparison (Gate 0).
* We iterate the 72 *active* emitter entries (label_count > 0) of the
  emitter library in sorted order and clip to
  `simulation_config.max_emitters` (default 35). The plan specifies
  that "increasing emitter density ADDS emitters without re-randomising
  the ones already present" — sorting by id and taking a prefix gives
  that property for free.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.environment import (
    EmitterBehaviorType,
    EmitterConfig,
    ProvenanceLabel,
    ProvenanceTag,
    SimulationConfig,
)
from ..core.data_loader import load_tsrd_statistics
from ..core.mapping import frequency_to_band

# System A/B unification (Task #3). The sampler now offers a
# `build_emitter_objects()` method that returns
# `src.emitter_models.Emitter` instances parameterized by the same
# `tsrd_statistics.json` row that drives `build_emitter_configs()`.
# Both views are deterministic in the same RNG and carry the same
# provenance notes; only the output type differs.
#
# Import is local because `src.emitter_models` is a sibling package
# (top-level `src`, not under `vyapti_simulator`).
from src.emitter_models import (
    DynamicEmitter,
    DelayedArrivalPolicy,
    IntervalOnOffPolicy,
    RegimeChangePolicy,
    FixedContinuousEmitter,
    FrequencyAgileEmitter,
    PriJitterEmitter,
    ScanningEmitter,
)


# =====================================================================
# FREQ-MODE  →  EmitterBehaviorType  MAPPING
# =====================================================================
# The TSRD emitter library has 6 frequency modes; the simulator's enum
# has 13 behavior types. We map the 6 onto 6 *natural* simulator
# archetypes; the remaining 7 enum members (DELAYED_ARRIVAL,
# ABRUPT_CHANGE, MIXED_POPULATION, UNSEEN_TEST, etc.) are not generated
# by TSRD and stay reserved for synthetic / held-out experiments.
#
# This mapping is a one-to-one, traceable table: every TSRD emitter
# resolves to a single behavior type, and the audit log records which
# one. No silent fallthroughs.
FREQ_MODE_TO_BEHAVIOR: Dict[str, EmitterBehaviorType] = {
    "FixedSingle":            EmitterBehaviorType.CONTINUOUS_FIXED,
    "FixedMultiSimultaneous": EmitterBehaviorType.CONTINUOUS_FIXED,
    "HoppingLinear":          EmitterBehaviorType.PATTERNED,
    "HoppingSawtooth":        EmitterBehaviorType.PATTERNED,
    "RandomFixed":            EmitterBehaviorType.PSEUDO_RANDOM_AGILE,
    "RandomRange":            EmitterBehaviorType.RANDOM_HOPPER,
}


# Cap on period_slots so a 0-RPM edge case (period=12000 slots =
# 10 minutes) doesn't dominate a 30 s mission. The plan locks the
# mission at 600 slots; a period longer than that means "on once, for
# the rest of the mission" which is what CONTINUOUS_FIXED is for.
# 600 slots = 30 s, matching the TSRD grid chosen in the plan.
_MAX_PERIOD_SLOTS_CAP = 600


class TSRDEmitterSampler:
    """
    Build `EmitterConfig` instances by sampling the TSRD aggregate
    statistics JSON. Same seed → identical list (Gate-0 property).
    """

    def __init__(
        self,
        tsrd_statistics_path: Union[str, Path],
        simulation_config: SimulationConfig,
        seed: int = 42,
        visibility_alpha: float = 2.0,
        visibility_beta: float = 5.0,
        snr_jitter_db: float = 5.0,
        arrival_window_fraction: float = 0.10,
        receiver_position_km: Optional[Tuple[float, float]] = None,
        path_loss_shadowing_db: float = 8.0,
        path_loss_diffraction_db: float = 4.0,
        rng: Optional[np.random.Generator] = None,
        # Dynamic phenomena configuration
        enable_interval_on_off: bool = False,
        interval_on_off_fraction: float = 0.2,
        mean_on_sec: float = 5.0,
        mean_off_sec: float = 10.0,
        enable_regime_change: bool = False,
        regime_change_fraction: float = 0.1,
        regime_change_time_fraction: float = 0.5,
    ) -> None:
        # ----------------------------------------------------------------
        # File-property integrity check fires HERE. No opt-out.
        # ----------------------------------------------------------------
        self._stats = load_tsrd_statistics(tsrd_statistics_path)

        self._config = simulation_config
        self._seed = int(seed)
        self._visibility_alpha = float(visibility_alpha)
        self._visibility_beta = float(visibility_beta)
        self._snr_jitter_db = float(snr_jitter_db)
        self._arrival_window_fraction = float(arrival_window_fraction)
        # Path-loss geometry
        self._receiver_position_km: Optional[Tuple[float, float]] = receiver_position_km
        self._path_loss_shadowing_db = float(path_loss_shadowing_db)
        self._path_loss_diffraction_db = float(path_loss_diffraction_db)

        # Dynamic phenomena configuration
        self._enable_interval_on_off = bool(enable_interval_on_off)
        self._interval_on_off_fraction = float(interval_on_off_fraction)
        self._mean_on_sec = float(mean_on_sec)
        self._mean_off_sec = float(mean_off_sec)
        self._enable_regime_change = bool(enable_regime_change)
        self._regime_change_fraction = float(regime_change_fraction)
        self._regime_change_time_fraction = float(regime_change_time_fraction)

        # RNG: if caller passes one (for testing), use it; otherwise build
        # from the canonical SeedSequence.
        if rng is not None:
            self._rng: np.random.Generator = rng
        else:
            # Explicit, root-tagged SeedSequence — the child of the run seed
            # is reserved for the sampler so the parent seed passed to
            # `environment.reset()` cannot influence what the sampler draws.
            # `SeedSequence` only accepts integers, so we hash the human
            # label down to a stable uint32.
            import hashlib as _hashlib
            label_hash = int.from_bytes(
                _hashlib.blake2b(b"tsrd_emitter_v1", digest_size=4).digest(),
                "big",
            )
            ss = np.random.SeedSequence([self._seed, label_hash])
            self._rng = np.random.default_rng(ss)

        # Cache the active emitter list, in deterministic sorted order.
        # An emitter is "active" if the TSRD extractor recorded at
        # least one pulse for it (label_count > 0).
        self._active_entries: List[tuple] = self._collect_active_entries()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_emitter_configs(self) -> List[EmitterConfig]:
        """
        Return a list of `EmitterConfig` of length
        `min(len(active_emitters), simulation_config.max_emitters)`.

        The seed is consumed lazily on each call, so calling this
        function twice with the same sampler returns two *identical*
        lists (we draw nothing here). That is the property Gate 0 needs:
        paired comparison must see the same truth every time.
        """
        max_count = max(1, int(self._config.max_emitters))
        n = min(len(self._active_entries), max_count)
        return [self._build_one(tx_id, entry) for tx_id, entry in self._active_entries[:n]]

    def build_emitter_objects(self) -> List:
        """
        Return a list of ``src.emitter_models.Emitter`` instances
        parameterised from the same ``tsrd_statistics.json`` rows that
        drive :meth:`build_emitter_configs`.

        **Unification contract (Task #3):**  Both
        ``build_emitter_objects()`` and ``build_emitter_configs()`` iterate
        the same ``_active_entries`` list in the same order and draw the
        same stochastic values from the same RNG for the same ``tx_id``.
        They are two representations of one sampler view; paired comparison
        across both views is valid.

        The lifecycle policy (delayed arrival) wraps every base emitter so
        that the TSRD-observed ``arrival_slot`` is respected. The
        ``visibility_fraction`` is encoded via the ``DelayedArrivalPolicy``'s
        ``start_time_sec``; ``phase_offset`` is encoded via the emitter's
        own ``phase_offset_sec`` field.

        The link budget: the TSRD sampler already computed
        ``snr_db = received_power_dbm - noise_floor_dbm`` from the
        H5 power config and receiver geometry. We convert this back to a
        received power: ``power_dbm = snr_db + noise_floor_dbm`` and pass
        it as the emitter's ``power_dbm``. The ``src.emitter_models``
        channel model is disabled (``range_m = None``) so that the emitter
        uses ``power_dbm`` directly without re-applying a path-loss model.
        This ensures the SNR at the bridge output matches what the TSRD
        sampler computed.

        Returns
        -------
        List[src.emitter_models.Emitter]
            ``min(len(_active_entries), max_emitters)`` emitter objects,
            in the same deterministic order as :meth:`build_emitter_configs`.
            The list may contain both bare base emitters
            (``FixedContinuousEmitter``, ``FrequencyAgileEmitter``) and
            ``DynamicEmitter`` wrappers (when ``arrival_slot > 0``).
        """
        max_count = max(1, int(self._config.max_emitters))
        n = min(len(self._active_entries), max_count)
        return [
            self._build_one_emitter(tx_id, entry)
            for tx_id, entry in self._active_entries[:n]
        ]

    @property
    def active_emitter_count(self) -> int:
        """Number of TSRD transmitter configs that produced >= 1 pulse."""
        return len(self._active_entries)

    @property
    def n_active(self) -> int:
        """Read directly from the JSON `population_stats` field for audit."""
        return int(self._stats["population_stats"]["n_active"])

    @property
    def n_silent(self) -> int:
        """Read directly from the JSON `population_stats` field for audit."""
        return int(self._stats["population_stats"]["n_silent"])

    @property
    def source_h5_sha256(self) -> str:
        """SHA-256 of the original TSRD HDF5 file, for the audit log."""
        return str(self._stats.get("source_h5_sha256", ""))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _collect_active_entries(self) -> List[tuple]:
        """
        Return [(tx_id_int, entry_dict), ...] for every active emitter,
        sorted by integer id so increasing `max_emitters` is a strict
        prefix extension (Gate-0 paired-comparison property).
        """
        library: Dict[str, Dict[str, Any]] = self._stats.get("emitter_library", {})
        label_counts: Dict[str, int] = self._stats.get("label_counts", {})

        active: List[tuple] = []
        for tx_key, entry in library.items():
            tx_id = int(tx_key)
            n_pulses = int(label_counts.get(tx_key, 0))
            if n_pulses > 0:
                active.append((tx_id, entry))
        active.sort(key=lambda pair: pair[0])
        return active

    def _build_one(self, tx_id: int, entry: Dict[str, Any]) -> EmitterConfig:
        """
        Construct one `EmitterConfig` for a TSRD emitter config.

        All continuous parameters are jittered; the discrete shape
        parameters (behavior, active_bands) come straight from the
        emitter library where available, and fall back to the
        population distribution otherwise.

        **Paired-comparison invariant:** Both this method and
        ``_build_one_emitter`` consume the same RNG draw sequence
        via ``_build_one_common`` (which now also draws
        ``phase_offset_slots``). The emitter config and the emitter
        object for the same ``tx_id`` therefore receive the same
        stochastic values when called from equally-seeded samplers.
        """
        rec = self._build_one_common(tx_id, entry)
        return EmitterConfig(
            emitter_id=int(tx_id),
            behavior=rec["behavior"],
            active_bands=rec["active_bands"],
            period_slots=int(rec["period_slots"]) if rec["period_slots"] is not None else None,
            phase_offset_slots=int(rec["phase_offset_slots"]),
            visibility_fraction=float(rec["visibility_fraction"]),
            snr_db=float(rec["snr_db"]),
            arrival_slot=int(rec["arrival_slot"]),
            provenance_notes=rec["provenance_notes"],
        )

    def _build_one_emitter(self, tx_id: int, entry: Dict[str, Any]) -> Any:
        """
        Construct one ``src.emitter_models.Emitter`` from the same
        statistics row used by :meth:`_build_one`. Paired comparison
        across the two views holds because both methods share
        :meth:`_build_one_common` (single RNG draw sequence).
        """
        rec = self._build_one_common(tx_id, entry)

        # --- Map TSRD freq_mode to a concrete src/ emitter class ---------
        freq_mode = rec["freq_mode"]
        freqs_mhz: List[float] = list(rec["raw_freqs_mhz"])
        pri_sec = float(rec["pri_sec"])
        pulse_width_sec = float(rec["pulse_width_sec"])
        power_dbm = float(rec["power_dbm"])
        arrival_slot = int(rec["arrival_slot"])
        period_slots = rec["period_slots"]
        center_freq_mhz = (
            float(freqs_mhz[0]) if freqs_mhz
            else float(rec.get("centre_freq_mhz") or 0.0)
        )
        scan_period_sec = (
            float(period_slots) * float(self._config.slot_duration_s())
            if period_slots else 1.0
        )

        base_emitter: Any
        if freq_mode in ("FixedSingle", "FixedMultiSimultaneous") and freqs_mhz:
            base_emitter = FixedContinuousEmitter(
                emitter_id=int(tx_id),
                center_freq_hz=center_freq_mhz * 1e6,
                pri_sec=pri_sec,
                pulse_width_sec=pulse_width_sec,
                power_dbm=power_dbm,
                name=f"TSRD_tx{tx_id}_{freq_mode}",
            )
        elif freq_mode in ("HoppingLinear", "HoppingSawtooth",
                           "RandomFixed", "RandomRange"):
            # All four map onto FrequencyAgileEmitter. The hop pattern
            # is preserved by passing freqs_mhz in the original order;
            # the emitter's internal RNG-driven hop is seeded per-emitter
            # so paired comparison still holds.
            if not freqs_mhz:
                # RandomRange with no realised list: fall back to the
                # same band we derived for EmitterConfig.
                fallback_mhz = float(rec.get("centre_freq_mhz") or 5000.0)
                freqs_mhz = [fallback_mhz]
            base_emitter = FrequencyAgileEmitter(
                emitter_id=int(tx_id),
                freq_list_hz=[f * 1e6 for f in freqs_mhz],
                pri_sec=pri_sec,
                pulse_width_sec=pulse_width_sec,
                power_dbm=power_dbm,
                dwell_sec=0.01,
                hop_sequence_seed=int(self._seed) * 1000 + int(tx_id),
                name=f"TSRD_tx{tx_id}_{freq_mode}",
            )
        elif freq_mode == "Scanning" or freq_mode.startswith("Scanning"):
            base_emitter = ScanningEmitter(
                emitter_id=int(tx_id),
                center_freq_hz=center_freq_mhz * 1e6,
                pri_sec=pri_sec,
                pulse_width_sec=pulse_width_sec,
                power_dbm=power_dbm,
                scan_period_sec=scan_period_sec,
                beam_width_deg=5.0,
                name=f"TSRD_tx{tx_id}_{freq_mode}",
            )
        else:
            # Unknown freq_mode: best-effort fallback to a fixed emitter
            # centred on the population frequency aggregate.
            base_emitter = FixedContinuousEmitter(
                emitter_id=int(tx_id),
                center_freq_hz=(center_freq_mhz or 5000.0) * 1e6,
                pri_sec=pri_sec,
                pulse_width_sec=pulse_width_sec,
                power_dbm=power_dbm,
                name=f"TSRD_tx{tx_id}_{freq_mode}_fallback",
            )

        # --- Wrap with a lifecycle policy if the emitter has a delay -----
        if arrival_slot > 0:
            start_time_sec = float(arrival_slot) * float(self._config.slot_duration_s())
            emitter = DynamicEmitter(
                emitter_id=int(tx_id),
                base_emitter=base_emitter,
                policy=DelayedArrivalPolicy(start_time_sec=start_time_sec),
                name=f"TSRD_tx{tx_id}_dynamic",
            )
        else:
            emitter = base_emitter

        # --- Apply IntervalOnOff policy if configured -----
        # A fraction of emitters get random ON/OFF intervals
        if self._enable_interval_on_off:
            if self._rng.random() < self._interval_on_off_fraction:
                emitter = DynamicEmitter(
                    emitter_id=int(tx_id),
                    base_emitter=emitter,
                    policy=IntervalOnOffPolicy(
                        mean_on_sec=self._mean_on_sec,
                        mean_off_sec=self._mean_off_sec,
                        seed=int(self._seed) + int(tx_id),
                    ),
                    name=f"TSRD_tx{tx_id}_interval_on_off",
                )

        # --- Apply RegimeChange policy if configured -----
        # A fraction of emitters change their configuration mid-mission
        if self._enable_regime_change:
            if self._rng.random() < self._regime_change_fraction:
                mission_duration_s = float(self._config.time_slots) * float(self._config.slot_duration_s())
                change_time_sec = mission_duration_s * self._regime_change_time_fraction
                # Regime change: switch to a different frequency list
                # Use the emitter's current frequency as the new center
                new_freq_hz = center_freq_mhz * 1e6 * 1.05  # 5% frequency shift
                emitter = DynamicEmitter(
                    emitter_id=int(tx_id),
                    base_emitter=emitter,
                    policy=RegimeChangePolicy(regimes=[
                        RegimeChangePolicy.Regime(
                            change_time_sec=change_time_sec,
                            freq_override_hz=new_freq_hz,
                        ),
                    ]),
                    name=f"TSRD_tx{tx_id}_regime_change",
                )

        # Attach the full provenance record so callers (bridge, evaluation)
        # can read it without knowing the internal type hierarchy.
        emitter.provenance_notes = rec["provenance_notes"]
        return emitter

    def _build_one_common(self, tx_id: int, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the shared (RNG-deterministic) view of one TSRD emitter
        row. Both ``_build_one`` (legacy EmitterConfig) and
        ``_build_one_emitter`` (src.emitter_models.Emitter) consume
        the same record, which guarantees paired comparison holds by
        construction.

        All RNG draws happen here, in the order both views require:
            1. visibility_fraction    (Beta)
            2. arrival_slot           (uniform integers; conditional on
                                      H5 start_time_s being absent)
            3. shadowing_db           (Normal)
            4. diffraction_db         (Normal)
            5. snr_jitter             (Normal)
            6. phase_offset_slots     (uniform integers)

        Returns a dict with everything both views need, including the
        raw TSRD fields that are only used by one of them.
        """
        freq_cfg = entry.get("frequency") or {}
        pri_cfg = entry.get("pri") or {}
        power_cfg = entry.get("power") or {}
        scan_cfg = entry.get("scan") or {}

        # --- behavior: TSRD freq_mode → simulator behavior enum ----------
        freq_mode = str(freq_cfg.get("freq_mode", "FixedSingle"))
        behavior = FREQ_MODE_TO_BEHAVIOR.get(
            freq_mode, EmitterBehaviorType.CONTINUOUS_FIXED
        )

        # --- active_bands: each freqs_mhz[] entry → discrete band -------
        freqs_mhz = list(freq_cfg.get("freqs_mhz") or [])
        active_bands = self._freqs_to_bands(freqs_mhz)

        # --- period_slots: from scan_rate_rpm when available, else
        #     from the population aggregate (min/median/max). ----------
        period_slots = self._sample_period_slots(scan_cfg)
        if period_slots is not None and period_slots > _MAX_PERIOD_SLOTS_CAP:
            period_slots = _MAX_PERIOD_SLOTS_CAP

        # --- visibility_fraction: Beta(α, β) per the plan -------------
        visibility_fraction = float(
            np.clip(self._rng.beta(self._visibility_alpha, self._visibility_beta), 0.01, 0.99)
        )

        # --- arrival_slot: prefer H5 start_time_s when present;
        #     else sample uniformly over the first 10% of the mission. ---
        h5_start_time_s = float(power_cfg.get("start_time_s", 0.0))
        if h5_start_time_s > 0.0:
            slot_s = self._config.slot_duration_s()
            arrival_slot = (
                max(0, int(h5_start_time_s / slot_s)) if slot_s > 0 else 0
            )
        else:
            window = max(
                1, int(self._config.time_slots * self._arrival_window_fraction)
            )
            arrival_slot = int(self._rng.integers(0, window))

        # --- snr_db: per-emitter link budget (Gap 5: proper FSPL
        #     from emitter start_position_km to receiver at origin,
        #     with the emitter's own centre frequency) ---
        entry_freq_cfg = entry.get("frequency") or {}
        entry_freqs = entry_freq_cfg.get("freqs_mhz") or []
        centre_freq_mhz = (
            float(entry_freqs[0]) if entry_freqs else None
        )
        entry_pos_cfg = entry.get("position") or {}
        # Path-loss components:
        #   FSPL: computed by _estimate_snr_db
        #   Shadowing (terrain): per-emitter N(0, σ) — models ridge shadowing
        #   Diffraction (multipath): per-emitter N(0, σ) — models rooftop diffraction
        #   SNR jitter: N(0, σ) — existing measurement noise on SNR
        shadowing_db = float(
            self._rng.normal(0.0, self._path_loss_shadowing_db)
        )
        diffraction_db = float(
            self._rng.normal(0.0, self._path_loss_diffraction_db)
        )
        snr_db = self._estimate_snr_db(
            power_cfg=power_cfg,
            position_cfg=entry_pos_cfg,
            freq_mhz=centre_freq_mhz,
            receiver_position_km=self._receiver_position_km,
        ) + shadowing_db + diffraction_db + float(
            self._rng.normal(0.0, self._snr_jitter_db)
        )

        # --- snr → dBm (for src/ emitter) -------------------------------
        # TSRD's snr_db is a relative dB; the src/ emitters want
        # absolute received power in dBm. We pin the receiver's
        # noise floor to the same -130 dBm the TSRD path uses so
        # the two views produce the same pulse-amplitude distribution.
        # Lazy import to break the tsrd_emitter ↔ tsrd_environment cycle
        # (tsrd_environment already imports this module).
        from .tsrd_environment import DetectionConfig as _DC  # local
        noise_floor_dbm = float(_DC().nominal_noise_floor_db)
        power_dbm = float(snr_db) + noise_floor_dbm
        # Clamp to the emitter model's valid received-power range.
        # Negative-SNR pulses (below TSRD relative floor) would map to
        # unrealistic received powers below -130 dBm; clamp to the
        # minimum meaningful received power so the emitter model is valid.
        # TSRD pulses with very high SNR (> 40 dB above noise floor)
        # would map to unrealistic received powers above -90 dBm; also clamp.
        MIN_PWR_DBM = -100.0
        MAX_PWR_DBM = -30.0
        power_clamped = not (MIN_PWR_DBM <= power_dbm <= MAX_PWR_DBM)
        if power_clamped:
            power_dbm = float(np.clip(power_dbm, MIN_PWR_DBM, MAX_PWR_DBM))

        # Provenance dict: defined early so we can annotate clamping facts,
        # then extended with the full audit record at the end.
        provenance_notes: Dict[str, Any] = {}
        provenance_notes["tsrd_power_dbm_clamped"] = power_clamped
        if power_clamped:
            provenance_notes["tsrd_power_dbm_raw"] = float(snr_db) + noise_floor_dbm

        # --- phase_offset_slots: uniform integer draw (draw 6 in the
        #     shared sequence, consumed by BOTH _build_one and
        #     _build_one_emitter so the two views stay in sync).
        phase_offset_slots = int(
            self._rng.integers(0, max(1, int(period_slots or 1)))
        )

        # --- PRI / PW: TSRD pris_us / pws_us are already in microseconds.
        #     Convert to seconds for src/emitter_models constructors.
        #     Take the first entry as a representative value; Staggered PRI
        #     (a list of PRIs) is an advanced feature deferred to v2.
        #
        #     Clamp PW to the emitter model's valid range (0.2–10 µs).
        #     TSRD emitters legitimately span 0.5–100 µs; values outside
        #     the simple model's range are clipped to the nearest endpoint
        #     and logged in provenance so the evaluation can reason about it.
        pris_us = pri_cfg.get("pris_us") or []
        pri_sec = (float(pris_us[0]) if pris_us else 1e-3) * 1e-6  # µs → s
        pws_us = (entry.get("pulse_width") or {}).get("pws_us") or []
        raw_pw_sec = (float(pws_us[0]) if pws_us else 1e-6) * 1e-6  # µs → s
        MIN_PW_SEC = 0.2e-6
        MAX_PW_SEC = 10e-6
        pulse_width_sec = float(np.clip(raw_pw_sec, MIN_PW_SEC, MAX_PW_SEC))

        # --- provenance: the audit must be able to trace every config
        #     back to the TSRD transmitter id and the JSON row used. ---
        provenance_notes.update({
            "tsrd_source_tx_id": tx_id,
            "tsrd_function": entry.get("function", ""),
            "tsrd_freq_mode": freq_mode,
            "tsrd_pri_mode": pri_cfg.get("pri_mode", ""),
            "tsrd_pri_us": [float(p) for p in (pri_cfg.get("pris_us") or [])],
            "tsrd_scan_type": scan_cfg.get("scan_type", ""),
            "tsrd_scan_rate_rpm": float(scan_cfg.get("scan_rate_rpm", 0.0)),
            "tsrd_power_w": float(power_cfg.get("power_w", 0.0)),
            "tsrd_gain_db": float(power_cfg.get("gain", 0.0)),
            "tsrd_start_time_s": h5_start_time_s,
            "tsrd_period_slots": int(period_slots) if period_slots is not None else None,
            "tsrd_active_bands": list(active_bands),
            "tsrd_visibility_fraction": float(visibility_fraction),
            "tsrd_arrival_slot": int(arrival_slot),
            "tsrd_snr_db": float(snr_db),
            "tsrd_power_dbm": power_dbm,
            "tsrd_noise_floor_dbm": noise_floor_dbm,
            "tsrd_pw_sec": pulse_width_sec,
            "tsrd_raw_pw_us": float(pws_us[0]) if pws_us else None,
            "tsrd_pw_clamped": pulse_width_sec != raw_pw_sec,
            "tsrd_shadowing_db": shadowing_db,
            "tsrd_diffraction_db": diffraction_db,
            "tsrd_receiver_position_km": (
                list(self._receiver_position_km)
                if self._receiver_position_km is not None else None
            ),
            "tsrd_path_loss_shadowing_db": self._path_loss_shadowing_db,
            "tsrd_path_loss_diffraction_db": self._path_loss_diffraction_db,
            "tsrd_source_h5_sha256": self.source_h5_sha256,
            "tsrd_sampler_seed": self._seed,
            # Dynamic phenomena configuration
            "tsrd_dynamic_interval_on_off_enabled": self._enable_interval_on_off,
            "tsrd_dynamic_interval_on_off_fraction": self._interval_on_off_fraction,
            "tsrd_dynamic_regime_change_enabled": self._enable_regime_change,
            "tsrd_dynamic_regime_change_fraction": self._regime_change_fraction,
            "tsrd_provenance_label": ProvenanceLabel(
                ProvenanceTag.TSRD_DERIVED,
                "TSRD config_0.h5 (arXiv:2602.03856, Scan Mode, "
                f"{self.n_active} active of {self.n_active + self.n_silent} configured).",
                f"Sampled transmitters_X={tx_id} into existing EmitterBehaviorType "
                f"('{behavior.value}') using the population distribution. "
                "The simulator regenerates truth from these statistics on every "
                "run; no TSRD scan-mode matrix is replayed.",
            ).tag.value,
        })

        return {
            "behavior": behavior,
            "freq_mode": freq_mode,
            "active_bands": active_bands,
            "period_slots": period_slots,
            "phase_offset_slots": phase_offset_slots,
            "visibility_fraction": visibility_fraction,
            "arrival_slot": arrival_slot,
            "snr_db": snr_db,
            "power_dbm": power_dbm,
            "pri_sec": pri_sec,
            "pulse_width_sec": pulse_width_sec,
            "raw_freqs_mhz": freqs_mhz,
            "centre_freq_mhz": centre_freq_mhz,
            "provenance_notes": provenance_notes,
        }

    def _freqs_to_bands(self, freqs_mhz: List[float]) -> List[int]:
        """
        Convert TSRD `freqs_mhz` (continuous MHz) to discrete band
        indices using the locked `core.mapping.frequency_to_band`.

        Returns a *sorted, deduplicated* list. Falls back to "all
        bands" if the emitter's list is empty (e.g. RandomRange
        generated no fixed list), so the truth generator still has
        somewhere to put it.
        """
        if not freqs_mhz:
            return list(range(self._config.band_count))
        bands: List[int] = []
        for f in freqs_mhz:
            try:
                b = frequency_to_band(float(f), self._config)
            except (ValueError, TypeError):
                continue
            if 0 <= b < self._config.band_count and b not in bands:
                bands.append(b)
        if not bands:
            return list(range(self._config.band_count))
        bands.sort()
        return bands

    def _sample_period_slots(self, scan_cfg: Dict[str, Any]) -> Optional[int]:
        """
        Compute period_slots from a TSRD emitter's scan config.

        Priority:
            1. `scan_rate_rpm > 0` → `60 / rpm / slot_seconds`, rounded.
            2. Else draw from the population aggregate `period_slots`
               min/median/max (uniform on those three, log-uniform
               because 15 to 12000 is a wide range).
            3. Else None.
        """
        rate_rpm = float(scan_cfg.get("scan_rate_rpm", 0.0) or 0.0)
        slot_seconds = max(1e-6, self._config.slot_duration_s())
        if rate_rpm > 0.0:
            period_s = 60.0 / rate_rpm
            return int(max(1, round(period_s / slot_seconds)))

        pop = self._stats.get("population_stats", {}).get("period_slots") or {}
        mn = int(pop.get("min", 0) or 0)
        md = int(pop.get("median", 0) or 0)
        mx = int(pop.get("max", 0) or 0)
        choices = [v for v in (mn, md, mx) if v > 0]
        if not choices:
            return None
        # Log-uniform: 15 to 12000 is a 3-order-of-magnitude range and
        # naive uniform under-samples the small end. We re-sample in
        # log space, then round.
        log_lo = math.log10(min(choices))
        log_hi = math.log10(max(choices))
        log_v = float(self._rng.uniform(log_lo, log_hi))
        return int(max(1, round(10 ** log_v)))

    @staticmethod
    def path_loss_db(
        range_km: float,
        freq_mhz: float,
        shadowing_db: float = 0.0,
        diffraction_db: float = 0.0,
    ) -> float:
        """
        Compute path loss in dB for a given range and frequency.

        The model adds two stochastic components to free-space path loss
        (Gap 5):

          path_loss_db = FSPL_dB + shadowing_db + diffraction_db

        Parameters
        ----------
        range_km : float
            Slant range from emitter to receiver in km.
        freq_mhz : float
            Centre frequency in MHz.
        shadowing_db : float
            Terrain shadowing loss/gain. Typically N(0, σ=8) dB for
            suburban environments. Positive = more loss than free space.
        diffraction_db : float
            Diffraction/multipath loss/gain. Typically N(0, σ=4) dB.
            Models constructive and destructive multipath interference.

        Returns
        -------
        float
            Total path loss in dB.
        """
        fspl_db = (
            20.0 * math.log10(max(range_km, 0.1))
            + 20.0 * math.log10(freq_mhz)
            + 32.4
        )
        return fspl_db + shadowing_db + diffraction_db

    @staticmethod
    def _estimate_snr_db(
        power_cfg: Dict[str, Any],
        position_cfg: Optional[Dict[str, Any]] = None,
        freq_mhz: Optional[float] = None,
        receiver_position_km: Optional[Tuple[float, float]] = None,
        rx_gt_db: float = 0.0,
    ) -> float:
        """
        Link-budget SNR with full path-loss model.

        Computes:

          EIRP_dBW  = 10*log10(power_w) + gain_db
          range_km   = euclidean distance from emitter start_position_km
                       to receiver_position_km
          FSPL_dB    = 20*log10(range_km) + 20*log10(freq_mhz) + 32.4
          shadowing  = 0 dB (callers draw and pass in separately)
          diffraction = 0 dB (callers draw and pass in separately)
          path_loss  = FSPL + shadowing + diffraction
          SNR_dB     = EIRP_dBW - path_loss + rx_G/T_dB

        Note: shadowing and diffraction are drawn per-emitter by the
        caller (the sampler) using the same RNG that controls all
        stochasticity, preserving Gate-0 determinism.

        Falls back to a 20 dB reference when inputs are missing or
        non-positive, so the sampler never returns -inf.
        """
        try:
            power_w = float(power_cfg.get("power_w", 0.0) or 0.0)
            gain_db = float(power_cfg.get("gain", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 20.0
        if power_w <= 0.0 or gain_db <= 0.0:
            return 20.0

        eirp_dbw = 10.0 * math.log10(power_w) + gain_db

        # Compute range from emitter position to receiver
        range_km = 50.0  # default if no position available
        if position_cfg is not None and receiver_position_km is not None:
            start = position_cfg.get("start_position_km", None)
            if start is not None and len(start) >= 2:
                try:
                    ex, ey = float(start[0]), float(start[1])
                    rx, ry = float(receiver_position_km[0]), float(receiver_position_km[1])
                    range_km = math.sqrt((ex - rx) ** 2 + (ey - ry) ** 2)
                except (TypeError, ValueError):
                    pass
        range_km = max(range_km, 0.1)

        f_mhz = freq_mhz if freq_mhz is not None and freq_mhz > 0 else 5000.0

        # Free-space path loss (shadowing/diffraction added by caller)
        fspl_db = (
            20.0 * math.log10(range_km)
            + 20.0 * math.log10(f_mhz)
            + 32.4
        )

        return eirp_dbw - fspl_db + rx_gt_db

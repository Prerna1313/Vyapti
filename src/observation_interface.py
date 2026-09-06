"""
src.observation_interface
==========================

Canonical observation contract shared by BOTH:

  * System A — `src.rf_pulse_simulator` / `src.emitter_models`
    (pulse-level simulation with Rayleigh fading, FSPL, atmospheric
    attenuation, dynamic policies)

  * System B — `vyapti_simulator.tsrd.*`
    (TSRD H5 corpus → discretised band/slot grid, with synthetic
    statistics-grounded path for offline runs)

The two systems historically produced different observation shapes
(`Pulse` from `src.emitter_models`, `PDWStream` from
`tsrd.tsrd_adapter`, `BandSlotPulse` from `tsrd.pdw_discretiser`).
This module unifies them into ONE schema so that the scheduler
sees identical input regardless of the data source.

Design rules
------------

1. **The schema is the contract.** The dataclass fields here are the
   authoritative list. Any new field requires an audit entry.

2. **Both SI and TSRD unit conventions are supported.** Real TSRD
   PDWs use microseconds, MHz, and dB (relative). The `src/`
   emitter models use seconds, Hz, and dBm. The unified contract
   stores everything in SI (seconds, Hz, dBm) so the scheduler
   never has to know which convention the data arrived in. The
   `from_tsrd_pulse` and `from_src_pulse` constructors perform
   the unit conversion at the boundary.

3. **Provenance is mandatory.** Every `BandSlotPulse` carries a
   `data_source` tag ("real_tsrd" or "synthetic_dynamics") and a
   `source_h5_sha256` (or None for synthetic). This is required by
   the no-conflation rule: a result derived from TSRD must be
   distinguishable from a result derived from a synthetic scenario
   even when the observation content is byte-identical.

4. **Truth-leak by construction.** The contract carries only
   derived-from-pulses fields. It does NOT carry hidden truth,
   ground truth emitter identity beyond the H5 `emitter_id`
   (which is the per-pulse label, not a scheduling hint), or
   the discretised truth grid. The scheduler interface layer
   (`vyapti_simulator.core.scheduler_interface`) is responsible
   for stripping any residual truth before exposing the
   observation.

5. **Zero branching on data_source inside the scheduler.** The
   contract is the same shape for both sources. The
   `unified_environment.py` module exposes a `data_source` flag
   to the *environment constructor*, not to the scheduler.
   Tests in `tests/test_unified_observation_contract.py`
   enforce this invariant.

Author
------
Senior RF/EW Signal Simulation Engineer — PS26055 System A/B
unification deliverable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# Data source enumeration
# =====================================================================

class DataSource(str, Enum):
    """
    Which subsystem produced the underlying pulse stream.

    This is the ONLY place the data source is named. The
    scheduler must never branch on it; it is a provenance tag
    that travels with every observation so that an audit can
    distinguish a result derived from real TSRD data from a
    result derived from a synthetic scenario.
    """
    REAL_TSRD = "real_tsrd"
    SYNTHETIC_DYNAMICS = "synthetic_dynamics"


# =====================================================================
# Canonical pulse: BandSlotPulse
# =====================================================================

@dataclass(frozen=True)
class BandSlotPulse:
    """
    The unified, system-agnostic pulse schema.

    Every pulse reaching the scheduler — whether it came from a
    real TSRD H5 file or from `src.emitter_models` dynamics —
    is a `BandSlotPulse` with this exact shape.

    Units
    -----
    All times are in SECONDS since the start of the mission.
    All frequencies are in Hz (not MHz).
    All amplitudes are in dBm (absolute received power), not
    TSRD's relative dB scale. Conversion from TSRD's relative
    `amp_db` is performed in `from_tsrd_pulse`.

    Fields
    ------
    toa_s : float
        Time of arrival in seconds from mission start.
    frequency_hz : float
        Centre frequency in Hz. > 0.
    pulse_width_s : float
        Pulse width (duration) in seconds. > 0.
    amplitude_dbm : float
        Received signal power at the receiver in dBm. May be
        negative (typical for EW receivers) or near 0 for very
        strong emitters.
    aoa_deg : float
        Angle of arrival in degrees, in [-180, 180]. The
        `src/` path uses a single AoA per emitter (set at
        construction); the TSRD path preserves the per-pulse
        AoA from the H5.
    emitter_id : int
        Per-pulse ground-truth label. For real TSRD this is the
        H5 `labels` column. For synthetic dynamics this is the
        `emitter_id` from the `Emitter` constructor. The
        scheduler MUST NOT use this to make decisions (it is a
        truth field and is stripped at the scheduler boundary);
        it is preserved here for evaluation only.
    band : int
        Band index this pulse mapped to, via
        `vyapti_simulator.core.mapping.frequency_to_band`.
        Computed at construction time, cached for O(1) lookup.
    slot : int
        Time-slot index this pulse mapped to, via
        `vyapti_simulator.core.mapping.seconds_to_slot`.
        Computed at construction time, cached for O(1) lookup.
    snr_db : float
        Estimated SNR at the receiver: `amplitude_dbm -
        noise_floor_dbm`. Negative values mean the pulse is
        below the noise floor. The noise floor is supplied at
        construction (default -130 dBm, the TSRD-relative
        convention).
    data_source : DataSource
        Which subsystem produced this pulse. Provenance only.
    source_h5_sha256 : Optional[str]
        SHA-256 of the source H5 if `data_source ==
        DataSource.REAL_TSRD`; None for synthetic dynamics.
        For synthetic dynamics the synthetic seed is recorded
        in `provenance`.
    provenance : Dict[str, Any]
        Free-form provenance (TBD). For real TSRD this includes
        `tsrd_freq_mode`, `tsrd_function`, `tsrd_scan_type`,
        `tsrd_scan_rate_rpm`, `tsrd_pri_mode` from the H5
        `metadata/transmitters` group. For synthetic dynamics
        this includes the emitter type, PRI, PW, range_m, and
        tx_power_dbm from the `Emitter` constructor.
    is_real : bool
        True if the pulse corresponds to a real emitter. False if
        it was injected by the receiver's false-alarm model (Pfa)
        and is not grounded in any emitter. Provenance flag;
        preserved for evaluation only. Real-TSRD pulses are always
        `is_real=True`; src/ pulses are real unless the receiver
        injected a false alarm. Default True.

    Author note
    -----------
    `emitter_id`, `data_source`, `source_h5_sha256`, `is_real`,
    and `provenance` are TRUTH OR PROVENANCE fields. They are
    preserved on the BandSlotPulse for evaluation, but the
    scheduler interface layer MUST strip them before exposing
    the observation to a policy. See
    `vyapti_simulator.tsrd.unified_environment.strip_truth`.
    """

    # --- core pulse fields (SI units) -------------------------------
    toa_s: float
    frequency_hz: float
    pulse_width_s: float
    amplitude_dbm: float
    aoa_deg: float
    emitter_id: int

    # --- discretised grid coordinates (cached) ----------------------
    band: int
    slot: int

    # --- derived ----------------------------------------------------
    snr_db: float
    noise_floor_dbm: float

    # --- provenance -------------------------------------------------
    data_source: DataSource
    source_h5_sha256: Optional[str]
    is_real: bool = True
    provenance: Dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------
    @staticmethod
    def from_src_pulse(
        pulse: Any,  # src.emitter_models.Pulse
        band: int,
        slot: int,
        noise_floor_dbm: float = -130.0,
        synthetic_seed: Optional[int] = None,
        provenance: Optional[Dict[str, Any]] = None,
        is_real: bool = True,
    ) -> "BandSlotPulse":
        """
        Construct a `BandSlotPulse` from an `src.emitter_models.Pulse`.

        The `src/` pulse is already in SI units (seconds, Hz, dBm),
        so this is a direct copy with grid-coordinate annotation
        and SNR computation.

        Parameters
        ----------
        pulse : src.emitter_models.Pulse
            The pulse to convert.
        band, slot : int
            Discretised grid coordinates (computed by the caller
            via `core.mapping.frequency_to_band` and
            `core.mapping.seconds_to_slot`).
        noise_floor_dbm : float
            Receiver noise floor in dBm. Default -130 dBm
            matches the TSRD-relative convention; the `src/`
            path uses this only for SNR reporting, not for
            detection.
        synthetic_seed : int | None
            Seed used to generate this pulse, recorded in
            provenance.
        provenance : dict | None
            Additional per-emitter provenance (e.g. PRI, PW,
            emitter type).
        is_real : bool
            Set to False for receiver-injected false-alarm pulses
            that have no underlying emitter. Default True.
        """
        amp_dbm = float(pulse.amplitude_dbm)
        snr_db = amp_dbm - float(noise_floor_dbm)
        prov = dict(provenance or {})
        if synthetic_seed is not None:
            prov.setdefault("synthetic_seed", int(synthetic_seed))
        prov.setdefault("data_origin", "src.emitter_models")
        if not is_real:
            prov.setdefault("is_false_alarm", True)
        return BandSlotPulse(
            toa_s=float(pulse.toa),
            frequency_hz=float(pulse.frequency_hz),
            pulse_width_s=float(pulse.pulse_width_sec),
            amplitude_dbm=amp_dbm,
            aoa_deg=float(getattr(pulse, "aoa_deg", 0.0)),
            emitter_id=int(pulse.emitter_id),
            band=int(band),
            slot=int(slot),
            snr_db=snr_db,
            noise_floor_dbm=float(noise_floor_dbm),
            data_source=DataSource.SYNTHETIC_DYNAMICS,
            source_h5_sha256=None,
            is_real=is_real,
            provenance=prov,
        )

    @staticmethod
    def from_tsrd_pulse(
        toa_us: float,
        freq_mhz: float,
        pw_us: float,
        aoa_deg: float,
        amp_db: float,
        emitter_id: int,
        band: int,
        slot: int,
        *,
        source_h5_sha256: Optional[str] = None,
        noise_floor_dbm: float = -130.0,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> "BandSlotPulse":
        """
        Construct a `BandSlotPulse` from a TSRD-format PDW row.

        TSRD stores times in microseconds, frequencies in MHz,
        and amplitudes in dB relative to an internal scale. This
        constructor converts to the SI convention used by the
        `src/` path:

            toa_s       = toa_us  * 1e-6
            frequency_hz = freq_mhz * 1e6
            pulse_width_s = pw_us * 1e-6
            amplitude_dbm = noise_floor_dbm + amp_db

        The last identity is necessary because TSRD's
        `amp_db` is relative to an internal noise floor, not
        an absolute dBm value. The conversion uses the
        `noise_floor_dbm` parameter (default -130 dBm, the
        TSRD conventional value).

        Parameters
        ----------
        toa_us : float
            Time of arrival in microseconds.
        freq_mhz : float
            Centre frequency in MHz.
        pw_us : float
            Pulse width in microseconds.
        aoa_deg : float
            Angle of arrival in degrees.
        amp_db : float
            Amplitude in dB RELATIVE to the TSRD noise floor.
        emitter_id : int
            H5 `labels` column value.
        band, slot : int
            Discretised grid coordinates.
        source_h5_sha256 : str | None
            SHA-256 of the source H5 file, recorded for
            provenance.
        noise_floor_dbm : float
            TSRD noise floor in dBm (default -130 dBm). The
            absolute dBm is `noise_floor_dbm + amp_db`.
        provenance : dict | None
            Per-emitter metadata from the H5
            `metadata/transmitters` group.
        """
        toa_s = float(toa_us) * 1e-6
        frequency_hz = float(freq_mhz) * 1e6
        pulse_width_s = float(pw_us) * 1e-6
        amplitude_dbm = float(noise_floor_dbm) + float(amp_db)
        snr_db = float(amp_db)
        prov = dict(provenance or {})
        prov.setdefault("data_origin", "tsrd.h5")
        return BandSlotPulse(
            toa_s=toa_s,
            frequency_hz=frequency_hz,
            pulse_width_s=pulse_width_s,
            amplitude_dbm=amplitude_dbm,
            aoa_deg=float(aoa_deg),
            emitter_id=int(emitter_id),
            band=int(band),
            slot=int(slot),
            snr_db=snr_db,
            noise_floor_dbm=float(noise_floor_dbm),
            data_source=DataSource.REAL_TSRD,
            source_h5_sha256=source_h5_sha256,
            provenance=prov,
        )

    # ----------------------------------------------------------------
    # Serialisation helpers
    # ----------------------------------------------------------------
    def to_observation_dict(self) -> Dict[str, Any]:
        """
        Return a scheduler-safe observation dict.

        This is the single function the scheduler-facing
        environment uses to expose one cell's content. The
        returned dict contains ONLY the keys in
        `PERMITTED_OBSERVATION_KEYS` from
        `vyapti_simulator.core.scheduler_interface`. Truth
        fields (`emitter_id`, `data_source`, `source_h5_sha256`,
        `provenance`) are NOT included; they are held by the
        environment for evaluation only.
        """
        # NOTE: the scheduler interface expects `pulse_count`,
        # `energy_db`, `max_amplitude_db`, etc. from the
        # TSRD Option B observation contract. The environment
        # aggregates a list of `BandSlotPulse` into a cell
        # BEFORE calling this method; this function takes a
        # single pulse's contribution to a cell. The actual
        # aggregation is in `unified_environment.py`.
        raise NotImplementedError(
            "BandSlotPulse.to_observation_dict() is not used; "
            "the environment aggregates pulses into a cell and "
            "builds the observation dict from the cell. See "
            "vyapti_simulator.tsrd.unified_environment."
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise this pulse to a JSON-safe dict.

        Useful for logging and for the `per_file_sha256.csv`
        Gate-0 audit trail. Provenance fields are preserved.
        """
        return {
            "toa_s": self.toa_s,
            "frequency_hz": self.frequency_hz,
            "pulse_width_s": self.pulse_width_s,
            "amplitude_dbm": self.amplitude_dbm,
            "aoa_deg": self.aoa_deg,
            "emitter_id": self.emitter_id,
            "band": self.band,
            "slot": self.slot,
            "snr_db": self.snr_db,
            "noise_floor_dbm": self.noise_floor_dbm,
            "data_source": self.data_source.value,
            "source_h5_sha256": self.source_h5_sha256,
            "is_real": self.is_real,
            "provenance": dict(self.provenance),
        }


# =====================================================================
# Observation history container
# =====================================================================

@dataclass
class ObservationHistory:
    """
    The accumulated set of pulses the scheduler has seen, plus
    the cell-level aggregates the scheduler is allowed to
    observe.

    Design
    ------
    The scheduler interface (from
    `vyapti_simulator.core.scheduler_interface`) forbids
    the scheduler from reading `emitter_id`, `data_source`,
    or any other truth field. `ObservationHistory` therefore
    keeps TWO stores:

      * `pulse_archive` — every `BandSlotPulse` ever observed
        (used for evaluation, for the deinterleaver, and for
        forensic replay). Marked EVAL-ONLY.

      * `cell_observations` — one dict per (band, slot) cell
        the scheduler has dwelled on, containing ONLY the
        keys in `PERMITTED_OBSERVATION_KEYS`. This is what the
        scheduler sees.

    The scheduler accesses `cell_observations` only. It never
    touches `pulse_archive`. The conformance test
    `test_unified_observation_contract.py` enforces this
    invariant by intercepting attribute access on a
    instrumented scheduler.

    Fields
    ------
    cell_observations : List[Dict[str, Any]]
        One dict per (band, slot) cell the scheduler has
        dwelled on. Order is chronological. Each dict is a
        member of the `PERMITTED_OBSERVATION_KEYS` whitelist
        from `vyapti_simulator.core.scheduler_interface`.
    pulse_archive : List[BandSlotPulse]
        [EVAL-ONLY] All pulses observed, in chronological
        order. The scheduler must not access this.
    source_label : str
        Human-readable label of the data source ("real_tsrd"
        or "synthetic_dynamics"). This is NOT a truth field
        and is exposed to the scheduler for diagnostic
        purposes only; the scheduler MUST NOT branch on it.
        Gate 0 compliance is enforced by code review and by
        `test_unified_observation_contract.py`.
    mission_seed : Optional[int]
        The seed that drove the current episode. Set by the
        environment; not used by the scheduler for decisions.
    """

    cell_observations: List[Dict[str, Any]] = field(default_factory=list)
    pulse_archive: List[BandSlotPulse] = field(default_factory=list)
    source_label: str = "synthetic_dynamics"
    mission_seed: Optional[int] = None

    def append_pulse(self, pulse: BandSlotPulse) -> None:
        """Append a pulse to the archive (evaluation only)."""
        self.pulse_archive.append(pulse)

    def append_cell_observation(self, obs: Dict[str, Any]) -> None:
        """
        Append a cell observation. The dict is stored as-is;
        callers (the environment) are responsible for stripping
        truth fields before calling.
        """
        self.cell_observations.append(dict(obs))

    def pulses_in_cell(self, band: int, slot: int) -> List[BandSlotPulse]:
        """
        [EVAL-ONLY] All pulses whose (band, slot) annotation
        matches. Used by the deinterleaver and by metrics.
        """
        return [p for p in self.pulse_archive if p.band == band and p.slot == slot]

    def pulses_in_band(self, band: int) -> List[BandSlotPulse]:
        """[EVAL-ONLY] All pulses whose band annotation matches."""
        return [p for p in self.pulse_archive if p.band == band]

    def unique_emitter_ids(self) -> np.ndarray:
        """[EVAL-ONLY] Distinct emitter ids across the archive."""
        if not self.pulse_archive:
            return np.array([], dtype=np.int64)
        return np.unique(np.array([p.emitter_id for p in self.pulse_archive], dtype=np.int64))

    def source_breakdown(self) -> Dict[str, int]:
        """
        [EVAL-ONLY] Count of pulses from each data source.
        Used to confirm a mixed-source experiment is
        reported correctly.
        """
        out: Dict[str, int] = {}
        for p in self.pulse_archive:
            k = p.data_source.value
            out[k] = out.get(k, 0) + 1
        return out

    def to_observation_dict_list(self) -> List[Dict[str, Any]]:
        """
        Return the scheduler-safe observation list.

        This is the list passed to `select_action(...)`. It
        contains only `PERMITTED_OBSERVATION_KEYS` dicts.
        """
        return list(self.cell_observations)


# =====================================================================
# Source-mix container (for downstream reporting)
# =====================================================================

@dataclass(frozen=True)
class SourceMix:
    """
    A summary of which data sources contributed to a result.

    The mixed-source evaluation track (Kaggle full-corpus
    pass, etc.) needs to report not just an aggregate metric
    but the per-source metric split, so a reader can see
    whether a scheduler's behaviour on real TSRD matches
    its behaviour on synthetic dynamics.

    Fields
    ------
    n_real_tsrd : int
        Number of cells whose pulses were all from real TSRD.
    n_synthetic : int
        Number of cells whose pulses were all from synthetic
        dynamics.
    n_mixed : int
        Number of cells that contained pulses from BOTH
        sources (should be 0 unless the experiment explicitly
        mixes sources cell-by-cell).
    h5_sha256_set : Tuple[str, ...]
        Sorted tuple of H5 SHA-256 hashes that contributed.
        Empty for synthetic-only runs.
    """
    n_real_tsrd: int
    n_synthetic: int
    n_mixed: int
    h5_sha256_set: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_real_tsrd": self.n_real_tsrd,
            "n_synthetic": self.n_synthetic,
            "n_mixed": self.n_mixed,
            "h5_sha256_set": list(self.h5_sha256_set),
        }


__all__ = [
    "DataSource",
    "BandSlotPulse",
    "ObservationHistory",
    "SourceMix",
]

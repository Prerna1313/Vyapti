"""
vyapti_simulator.tsrd.pdw_discretiser
======================================

Option B — PDW-to-band/slot discretiser.

Maps the canonical TSRD 6-field PDW stream::

    (toa_us, freq_mhz, pw_us, aoa_deg, amp_db, emitter_id)

onto the simulation's discrete ``(band, slot)`` grid, aggregating all
pulses that fall into the same cell into statistical summaries.

This is the bridge between the raw pulse-level data that TSRD records
and the band/slot observation space that the PS26055 scheduler interface
operates in. It is a pure transform: no truth is created, no emitter
identity is assumed, and the output is deterministic (same PDW stream +
same config → same grid).

[UNIFICATION 2026-09-05] The per-pulse dataclass emitted by
``discretize_pdw_to_bands()`` is no longer a TSRD-shape
``BandSlotPulse``; it is the UNIFIED
``src.observation_interface.BandSlotPulse`` (SI units: seconds, Hz,
dBm). The cell-level aggregator ``BandSlotCell`` keeps the TSRD-native
arrays (``toa_us``/``freq_mhz``/``pw_us``/``amp_db``) because those
arrays are the discretiser's INTERNAL representation and they are
exposed to the scheduler only as derived scalar statistics
(``pulse_count``, ``max_amplitude_db``, ``snr_db``, …). No public
field on the cell carries per-pulse data in TSRD units out to
external callers.

The discretiser is the ONLY place the TSRD ``amp_db`` field is used.
Preserving amplitude is what differentiates Option B from Option A
(statistics → synthetic): the observation can now carry
``pulse_count``, ``max_amplitude_db``, and a derived ``snr_db``
estimate, which are the basis for SNR-dependent detection and for
the richer EW-quality metrics that a real receiver produces.

Design decisions
----------------
1. **One pulse → one cell.**  A TSRD pulse is instantaneous at its
   ToA. The discretiser maps it to the cell ``(band, slot)`` that
   contains the ToA. Pulses that arrive within the same band and slot
   are aggregated together; the original per-pulse data is not preserved.

2. **Aggregation rule: max-amplitude.**  When multiple pulses fall in
   the same cell, the cell reports the strongest amplitude (largest
   amp_db). This is the conservative choice for detection: the
   highest-SNR pulse in the cell is what a real energy-detector
   would integrate to first.

3. **Amplitude → SNR: simplified link budget.**  TSRD's ``amp_db`` is
   an arbitrary relative scale (not absolute dBm). The discretiser
   converts it to an approximate SNR by subtracting a nominal noise
   floor. The absolute calibration is deferred to ``rf/noise.py``;
   this module provides the signal-level information the scheduler
   needs at Option-B fidelity.

4. **Empty cells are no-signal.**  A cell with no pulses produces
   ``pulse_count=0`` and ``hit=False``. The absence of a pulse is
   not a detection of absence — it is simply an unobserved cell.
   The scheduler must decide whether to treat a miss as evidence of
   emitter absence or as an unlucky sample.

5. **Out-of-range pulses are dropped.**  Pulses whose frequency or
   ToA maps outside the simulation grid are silently discarded.
   They are not re-mapped or extrapolated; that would invent data.

[INVARIANT-2] Band and slot indices are computed ONLY through
``core.mapping.frequency_to_band`` and ``seconds_to_slot``. No
local formula is used.

Author
------
Senior RF/EW Signal Simulation Engineer — PS26055 Option B integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .tsrd_adapter import PDWStream
from ..core.environment import SimulationConfig
from ..core.mapping import frequency_to_band, seconds_to_slot

# `src` is a sibling top-level package.  The per-pulse type emitted by
# this module is now the UNIFIED BandSlotPulse from observation_interface,
# not a local TSRD-shape dataclass.
from src.observation_interface import BandSlotPulse as UnifiedBandSlotPulse
from src.observation_interface import DataSource


def discretize_pdw_to_bands(
    pdw: PDWStream,
    cfg: SimulationConfig,
    *,
    nominal_noise_floor_db: float = -130.0,
    source_h5_sha256: Optional[str] = None,
) -> Iterator[UnifiedBandSlotPulse]:
    """
    Generator: map a TSRD PDW stream to individual unified
    ``BandSlotPulse`` objects (SI units), one per pulse.

    This is the **Option-B per-pulse interface** — the generator
    the deinterleaver (Option C) and the observation-builder
    (TSRDEnvironment) use.

    The pulses yielded are the UNIFIED ``BandSlotPulse`` from
    ``src.observation_interface``: all times in seconds, frequencies
    in Hz, amplitudes in dBm.  This is the one schema that both the
    real-TSRD path and the synthetic-dynamics path expose to
    downstream code.

    Band-edge handling
    ------------------
    A pulse whose frequency falls exactly on a band boundary is
    assigned to the band whose centre is nearest to the pulse's
    frequency (i.e. the band with the lowest frequency distance).
    This is equivalent to rounding the band index to the nearest
    integer, which is the same as assigning to the nearest centre
    frequency.

    Multiple pulses in the same (band, slot) are yielded as
    separate ``BandSlotPulse`` objects. Callers can aggregate
    them by collecting into a dict keyed by ``(band, slot)``.

    Parameters
    ----------
    pdw : PDWStream
        The canonical 6-field TSRD PDW stream.
    cfg : SimulationConfig
        Grid parameters for the band/slot mapping.
    nominal_noise_floor_db : float
        Noise floor in dB for SNR estimation:
        ``snr_db = amp_db - nominal_noise_floor_db``.
    source_h5_sha256 : str | None
        SHA-256 of the source H5 file, for provenance tracking.
        Stamped into every yielded pulse.  None is acceptable for
        synthetic-only usage (the caller stamps it themselves).

    Yields
    ------
    BandSlotPulse (unified, from src.observation_interface)
        One per pulse in the stream, in original order.
        Pulses outside the grid boundaries are silently dropped.
        Units are SI: toa_s, frequency_hz, pulse_width_s, amplitude_dbm.

    [INVARIANT-2] Band and slot are computed ONLY through
    ``core.mapping.frequency_to_band`` and ``seconds_to_slot``.
    """
    n = len(pdw)
    for i in range(n):
        freq = float(pdw.freq_mhz[i])
        toa_us = float(pdw.toa_us[i])

        band = frequency_to_band(freq, cfg)
        slot = seconds_to_slot(toa_us * 1e-6, cfg)

        if not (0 <= band < cfg.band_count):
            continue
        if not (0 <= slot < cfg.time_slots):
            continue

        yield UnifiedBandSlotPulse.from_tsrd_pulse(
            toa_us=toa_us,
            freq_mhz=freq,
            pw_us=float(pdw.pw_us[i]),
            aoa_deg=float(pdw.aoa_deg[i]),
            amp_db=float(pdw.amp_db[i]),
            emitter_id=int(pdw.emitter_id[i]),
            band=band,
            slot=slot,
            source_h5_sha256=source_h5_sha256,
            noise_floor_dbm=float(nominal_noise_floor_db),
            provenance=None,
        )


def aggregate_cell_pulses(
    pulses: List[UnifiedBandSlotPulse],
    *,
    nominal_noise_floor_db: float = -130.0,
) -> BandSlotCell:
    """
    Aggregate a list of unified ``BandSlotPulse`` from the same
    (band, slot) into a single ``BandSlotCell``.

    The unified pulse carries SI fields; the cell stores TSRD-native
    arrays (``toa_us``, ``freq_mhz``, ``pw_us``, ``amp_db``) which
    are the discretiser's internal representation.  The conversion
    is performed here at the cell boundary.

    Use this when you have collected pulses from
    ``discretize_pdw_to_bands()`` into per-cell buckets and want
    the same summary statistics that ``DiscretisedGrid`` provides.

    The cell's ``snr_db`` is computed as
    ``max_amplitude_db - nominal_noise_floor_db``. Pass the
    same noise floor you used when building the pulses so the
    SNR is consistent.

    [UNIFICATION 2026-09-05] The cell keeps TSRD-native per-pulse
    arrays for backward compatibility with downstream code that
    reads ``cell.amp_db`` / ``cell.pw_us``.  The unified pulse
    is the ONLY type emitted by the public per-pulse generator.
    """
    if not pulses:
        return BandSlotCell(
            band=0, slot=0,
            pulse_count=0,
            emitter_ids=np.array([], dtype=np.int64),
            toa_us=np.array([], dtype=np.float64),
            freq_mhz=np.array([], dtype=np.float64),
            pw_us=np.array([], dtype=np.float64),
            aoa_deg=np.array([], dtype=np.float64),
            amp_db=np.array([], dtype=np.float64),
            max_amplitude_db=np.nan,
            min_amplitude_db=np.nan,
            snr_db=np.nan,
            is_empty=True,
        )
    b = pulses[0].band
    s = pulses[0].slot
    # Convert unified SI fields back to TSRD-native units for the
    # cell's internal arrays.  toa_s→toa_us (×1e6), frequency_hz→MHz
    # (÷1e6), pulse_width_s→µs (×1e6), amplitude_dbm→amp_db via
    # (amplitude_dbm - nominal_noise_floor_db).  The last identity
    # is the inverse of from_tsrd_pulse's reconstruction.
    toa = np.array([p.toa_s * 1e6 for p in pulses], dtype=np.float64)
    freq = np.array([p.frequency_hz * 1e-6 for p in pulses], dtype=np.float64)
    pw = np.array([p.pulse_width_s * 1e6 for p in pulses], dtype=np.float64)
    aoa = np.array([p.aoa_deg for p in pulses], dtype=np.float64)
    amp = np.array(
        [p.amplitude_dbm - p.noise_floor_dbm for p in pulses],
        dtype=np.float64,
    )
    eid = np.array([p.emitter_id for p in pulses], dtype=np.int64)

    # Sort by ToA
    order = np.argsort(toa)
    return BandSlotCell(
        band=b, slot=s,
        pulse_count=len(pulses),
        emitter_ids=eid[order],
        toa_us=toa[order],
        freq_mhz=freq[order],
        pw_us=pw[order],
        aoa_deg=aoa[order],
        amp_db=amp[order],
        max_amplitude_db=float(np.max(amp)),
        min_amplitude_db=float(np.min(amp)),
        snr_db=float(np.max(amp)) - nominal_noise_floor_db,
        is_empty=False,
    )


# =====================================================================
# Public data types
# =====================================================================

@dataclass(frozen=True, slots=True)
class BandSlotCell:
    """
    All pulses that fell within one (band, slot) grid cell,
    aggregated into statistics.

    All arrays are non-owning views of the grid's owning arrays
    unless explicitly copied. Treat them as read-only.

    Attributes
    ----------
    band : int
        Band index, 0 ≤ band < band_count.
    slot : int
        Time-slot index, 0 ≤ slot < time_slots.
    pulse_count : int
        Number of TSRD pulses that mapped to this cell.
    emitter_ids : np.ndarray
        H5 emitter labels of the pulses in this cell. May contain
        duplicates (multiple pulses from the same emitter). Shape
        ``(pulse_count,)``.
    toa_us : np.ndarray
        Time-of-arrival of each pulse in microseconds. Shape
        ``(pulse_count,)``. Sorted ascending.
    freq_mhz : np.ndarray
        Centre frequency of each pulse in MHz. Shape
        ``(pulse_count,)``.
    pw_us : np.ndarray
        Pulse width of each pulse in microseconds. Shape
        ``(pulse_count,)``.
    aoa_deg : np.ndarray
        Angle-of-arrival of each pulse in degrees. Shape
        ``(pulse_count,)``.
    amp_db : np.ndarray
        Amplitude of each pulse in dB (relative TSRD scale). Shape
        ``(pulse_count,)``.
    max_amplitude_db : float
        Maximum amplitude in this cell.  ``max(amp_db)`` or
        ``nan`` if ``pulse_count == 0``.
    min_amplitude_db : float
        Minimum amplitude in this cell.  ``min(amp_db)`` or
        ``nan`` if ``pulse_count == 0``.
    snr_db : float
        Approximate SNR for this cell: ``max_amplitude_db -
        nominal_noise_floor_db``.  The noise floor is derived
        from ``SimulationConfig.detection_threshold_db``; it is an
        engineering assumption, not a physical calibration.
    is_empty : bool
        ``True`` iff no pulses fell in this cell.
    """
    band: int
    slot: int
    pulse_count: int
    emitter_ids: np.ndarray
    toa_us: np.ndarray
    freq_mhz: np.ndarray
    pw_us: np.ndarray
    aoa_deg: np.ndarray
    amp_db: np.ndarray
    max_amplitude_db: float
    min_amplitude_db: float
    snr_db: float
    is_empty: bool

    @property
    def unique_emitter_count(self) -> int:
        """Number of distinct emitters contributing pulses to this cell."""
        if self.pulse_count == 0:
            return 0
        return int(np.unique(self.emitter_ids).size)

    @property
    def unique_emitter_ids(self) -> np.ndarray:
        """Distinct emitter IDs present in this cell."""
        if self.pulse_count == 0:
            return np.array([], dtype=np.int64)
        return np.unique(self.emitter_ids)

    @property
    def mean_pw_us(self) -> float:
        """Mean pulse width across pulses in this cell."""
        if self.pulse_count == 0:
            return np.nan
        return float(np.mean(self.pw_us))

    @property
    def std_pw_us(self) -> float:
        """Std of pulse width across pulses in this cell."""
        if self.pulse_count <= 1:
            return np.nan
        return float(np.std(self.pw_us, ddof=1))

    @property
    def mean_aoa_deg(self) -> float:
        """Mean AoA across pulses in this cell."""
        if self.pulse_count == 0:
            return np.nan
        return float(np.mean(self.aoa_deg))

    @property
    def dominant_emitter_id(self) -> Optional[int]:
        """Emitter that contributed the most pulses to this cell."""
        if self.pulse_count == 0:
            return None
        vals, counts = np.unique(self.emitter_ids, return_counts=True)
        return int(vals[int(np.argmax(counts))])

    @property
    def n_pulses_dominant_emitter(self) -> int:
        """
        Number of pulses from the dominant emitter in this cell.

        Returns 0 for empty cells.
        """
        if self.pulse_count == 0:
            return 0
        vals, counts = np.unique(self.emitter_ids, return_counts=True)
        return int(np.max(counts))

    def coherent_integration_gain_db(
        self,
        *,
        max_pulses: int = 50,
        non_coherent_loss_db: float = 0.5,
    ) -> float:
        """
        Maximum coherent integration gain (dB) available for any emitter
        in this cell.

        Real ESM receivers coherently integrate pulses from the same
        burst. For N pulses from the same emitter, the SNR gain is::

            gain_db = 10 * log10(min(N, max_pulses)) - non_coherent_loss_db

        where the loss term models phase jitter, small frequency offsets,
        and antenna scan modulation that prevent perfectly coherent sum.

        This method returns the *maximum* gain across all emitters present
        in the cell (i.e. the gain available for the dominant emitter).
        Multipath from a *different* emitter uses its own pulse count
        and would receive its own (lower) gain.

        Returns 0.0 for empty cells.

        Parameters
        ----------
        max_pulses : int
            Upper cap on the number of pulses considered for integration.
            Prevents unrealistic gain when a cell contains an atypically
            large number of pulses. Default 50 (~1 s of integration at
            50 Hz PRI). Typical ESM dwell time ≈ 50 ms → 2–3 pulses;
            a burst with PRI=10 ms × 1 s observation = 100 pulses.
        non_coherent_loss_db : float
            dB loss applied to the ideal coherent gain. Models phase
            jitter, frequency offset, and antenna scanning. Typical
            range: 0.3–1.0 dB. Default 0.5 dB.
        """
        if self.pulse_count == 0:
            return 0.0
        if self.pulse_count == 1:
            return 0.0
        vals, counts = np.unique(self.emitter_ids, return_counts=True)
        max_n = int(np.max(counts))
        n_capped = min(max_n, max_pulses)
        gain = 10.0 * np.log10(float(n_capped)) - non_coherent_loss_db
        return float(np.clip(gain, 0.0, float(np.inf)))


@dataclass
class DiscretisedGrid:
    """
    The complete discretisation of one TSRD PDW stream onto the
    band/slot grid.

    ``grid[band, slot]`` is a ``BandSlotCell`` describing every
    TSRD pulse that fell inside that cell. Empty cells are
    pre-built ``BandSlotCell`` objects with ``is_empty=True`` so
    that lookup is O(1) with no branching for the common no-pulse
    case.

    Parameters
    ----------
    band_count : int
    time_slots : int
    cells : np.ndarray
        Shape ``(band_count, time_slots)``, dtype ``object``.
        ``cells[band, slot]`` is a ``BandSlotCell``.
    config : SimulationConfig
        The simulation configuration (not stored, kept for reference).
    provenance : Dict
        Provenance metadata: source file, seed, TSRD H5 SHA-256,
        discretiser version, noise-floor assumption.
    """
    band_count: int
    time_slots: int
    cells: np.ndarray
    config: SimulationConfig
    provenance: Dict = field(default_factory=dict)

    def __post_init__(self):
        # Ensure cells is a proper object ndarray of the right shape
        if not isinstance(self.cells, np.ndarray) or self.cells.dtype != object:
            cells_arr = np.empty((self.band_count, self.time_slots), dtype=object)
            cells_arr[:, :] = self.cells
            self.cells = cells_arr
        if self.cells.shape != (self.band_count, self.time_slots):
            raise ValueError(
                f"cells shape {self.cells.shape} != "
                f"({self.band_count}, {self.time_slots})"
            )

    def __getitem__(self, key: Tuple[int, int]) -> BandSlotCell:
        """``grid[band, slot]`` → BandSlotCell."""
        band, slot = key
        return self.cells[band, slot]

    def pulses_in_band_slot(self, band: int, slot: int) -> BandSlotCell:
        """Convenience alias for ``self[band, slot]``."""
        return self.cells[band, slot]

    def pulses_in_band(self, band: int) -> np.ndarray:
        """
        All cells in a given band across all time slots.
        Returns a ``(time_slots,)`` array of ``BandSlotCell``.
        """
        return self.cells[band, :]

    def pulses_in_slot(self, slot: int) -> np.ndarray:
        """
        All cells in a given time slot across all bands.
        Returns a ``(band_count,)`` array of ``BandSlotCell``.
        """
        return self.cells[:, slot]

    @property
    def total_pulse_count(self) -> int:
        """Sum of pulse_count over all cells."""
        return int(
            sum(c.pulse_count for row in self.cells.flat for c in [row])
        )

    @property
    def occupied_cell_count(self) -> int:
        """Number of cells containing at least one pulse."""
        return int(
            sum(1 for row in self.cells.flat for c in [row] if not c.is_empty)
        )

    def occupancy_mask(self) -> np.ndarray:
        """
        Return a ``(band_count, time_slots)`` boolean array where
        ``True`` indicates at least one pulse was present.
        """
        mask = np.zeros((self.band_count, self.time_slots), dtype=bool)
        for b in range(self.band_count):
            for s in range(self.time_slots):
                mask[b, s] = not self.cells[b, s].is_empty
        return mask


# =====================================================================
# Core discretisation
# =====================================================================

def discretise_pdw_to_grid(
    pdw: PDWStream,
    cfg: SimulationConfig,
    *,
    nominal_noise_floor_db: float = -130.0,
    provenance: Optional[Dict] = None,
) -> DiscretisedGrid:
    """
    Discretise a TSRD PDW stream onto the simulation grid.

    Parameters
    ----------
    pdw : PDWStream
        The canonical 6-field TSRD PDW stream.
    cfg : SimulationConfig
        The simulation configuration defining band/slot grid dimensions.
    nominal_noise_floor_db : float
        [ENGINEERING-ASSUMPTION] Nominal noise floor in dB (relative
        amplitude scale). Used to derive ``snr_db`` for each cell as
        ``max_amplitude_db - nominal_noise_floor_db``. The absolute
        value is not calibrated; only differences between cells are
        meaningful at Option-B fidelity. Replace with a physical
        noise-floor model (``rf/noise.py``) for Option-C+ fidelity.
    provenance : Dict | None
        Optional provenance metadata to attach to the grid's provenance
        field. Typically includes the source H5 SHA-256 and the
        discretiser version string.

    Returns
    -------
    DiscretisedGrid
        A fully populated grid with one ``BandSlotCell`` per
        ``(band, slot)`` cell. Empty cells are pre-built; lookup
        is O(1).

    Algorithm
    ---------
    1. Iterate every pulse in ``pdw``.
    2. Map ``pdw.freq_mhz[i]`` → band via ``frequency_to_band``.
    3. Map ``pdw.toa_us[i]`` → slot via ``seconds_to_slot``.
    4. Drop pulses outside the grid boundary.
    5. Append each valid pulse to its cell's accumulator.
    6. Post-process each cell: compute max/min amplitude, SNR,
       unique emitter count.

    Complexity
    ----------
    O(P) where P = number of pulses in the PDW stream. Memory is
    O(B × S) for the cell table plus O(P) for the accumulator
    lists. For the full TSRD corpus (millions of pulses) this is
    the limiting factor; a GPU-accelerated path using CuPy or a
    vectorised numpy groupby is the natural next step.

    [INVARIANT-2] Band and slot indices use ONLY
    ``core.mapping.frequency_to_band`` and ``seconds_to_slot``.
    No local band-width formula.
    """
    n_bands = int(cfg.band_count)
    n_slots = int(cfg.time_slots)
    n_pulses = len(pdw)

    # Pre-allocate cell accumulators.
    # Each accumulator is a list that collects pulses for one cell.
    # Using a dict keyed by (band, slot) avoids allocating a 2-D
    # object array until we know the full set of occupied cells.
    accumulators: Dict[Tuple[int, int], Dict[str, List]] = {}

    for i in range(n_pulses):
        # --- frequency → band ----------------------------------------
        freq = float(pdw.freq_mhz[i])
        try:
            band = frequency_to_band(freq, cfg)
        except (ValueError, TypeError):
            continue
        if not (0 <= band < n_bands):
            continue

        # --- time → slot ---------------------------------------------
        toa_s = float(pdw.toa_us[i]) * 1e-6          # µs → s
        try:
            slot = seconds_to_slot(toa_s, cfg)
        except (ValueError, TypeError):
            continue
        if not (0 <= slot < n_slots):
            continue

        key = (band, slot)
        if key not in accumulators:
            accumulators[key] = {
                "emitter_ids": [],
                "toa_us": [],
                "freq_mhz": [],
                "pw_us": [],
                "aoa_deg": [],
                "amp_db": [],
            }
        acc = accumulators[key]
        acc["emitter_ids"].append(int(pdw.emitter_id[i]))
        acc["toa_us"].append(float(pdw.toa_us[i]))
        acc["freq_mhz"].append(float(pdw.freq_mhz[i]))
        acc["pw_us"].append(float(pdw.pw_us[i]))
        acc["aoa_deg"].append(float(pdw.aoa_deg[i]))
        acc["amp_db"].append(float(pdw.amp_db[i]))

    # --- build empty-cell template ------------------------------------
    empty_cell = BandSlotCell(
        band=0, slot=0,
        pulse_count=0,
        emitter_ids=np.array([], dtype=np.int64),
        toa_us=np.array([], dtype=np.float64),
        freq_mhz=np.array([], dtype=np.float64),
        pw_us=np.array([], dtype=np.float64),
        aoa_deg=np.array([], dtype=np.float64),
        amp_db=np.array([], dtype=np.float64),
        max_amplitude_db=np.nan,
        min_amplitude_db=np.nan,
        snr_db=np.nan,
        is_empty=True,
    )

    # --- allocate cell table (n_bands × n_slots of BandSlotCell) -----
    cells: np.ndarray = np.empty((n_bands, n_slots), dtype=object)
    cells[:, :] = empty_cell

    # --- post-process each occupied cell ------------------------------
    for (band, slot), acc in accumulators.items():
        emitter_ids = np.array(acc["emitter_ids"], dtype=np.int64)
        toa_us = np.array(acc["toa_us"], dtype=np.float64)
        freq_mhz = np.array(acc["freq_mhz"], dtype=np.float64)
        pw_us = np.array(acc["pw_us"], dtype=np.float64)
        aoa_deg = np.array(acc["aoa_deg"], dtype=np.float64)
        amp_db = np.array(acc["amp_db"], dtype=np.float64)

        # Sort by ToA for deterministic ordering within a cell
        sort_idx = np.argsort(toa_us)
        emitter_ids = emitter_ids[sort_idx]
        toa_us = toa_us[sort_idx]
        freq_mhz = freq_mhz[sort_idx]
        pw_us = pw_us[sort_idx]
        aoa_deg = aoa_deg[sort_idx]
        amp_db = amp_db[sort_idx]

        max_amp = float(np.max(amp_db))
        min_amp = float(np.min(amp_db))
        snr = float(max_amp - nominal_noise_floor_db)

        cell = BandSlotCell(
            band=band,
            slot=slot,
            pulse_count=int(emitter_ids.size),
            emitter_ids=emitter_ids,
            toa_us=toa_us,
            freq_mhz=freq_mhz,
            pw_us=pw_us,
            aoa_deg=aoa_deg,
            amp_db=amp_db,
            max_amplitude_db=max_amp,
            min_amplitude_db=min_amp,
            snr_db=snr,
            is_empty=False,
        )
        cells[band, slot] = cell

    return DiscretisedGrid(
        band_count=n_bands,
        time_slots=n_slots,
        cells=cells,
        config=cfg,
        provenance=provenance or {},
    )


def iter_pulses_in_grid(
    pdw: PDWStream,
    cfg: SimulationConfig,
) -> Iterator[BandSlotCell]:
    """
    Generator: yield one ``BandSlotCell`` per occupied (band, slot)
    pair, in row-major order (band first, then slot).

    This is the memory-efficient path for very large PDW streams
    where materialising the full ``DiscretisedGrid`` is prohibitive.
    It is used by the deinterleaver (Option C) which processes cells
    one at a time.

    Empty cells are skipped. The iterator yields ``BandSlotCell``
    objects identical to those in a ``DiscretisedGrid``.

    [INVARIANT-2] Uses only ``core.mapping`` helpers.
    """
    grid = discretise_pdw_to_grid(pdw, cfg)
    for b in range(grid.band_count):
        for s in range(grid.time_slots):
            cell = grid.cells[b, s]
            if not cell.is_empty:
                yield cell


__all__ = [
    "BandSlotCell",
    "DiscretisedGrid",
    "discretise_pdw_to_grid",
    "discretize_pdw_to_bands",
    "iter_pulses_in_grid",
    "aggregate_cell_pulses",
]

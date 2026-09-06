"""
vyapti_simulator.core.mapping
===============================

Single source of truth for converting between physical units
(frequency in MHz, time in seconds) and the simulator's discrete
band / time-slot indices.

Both the synthetic truth generator (`_fill_emitter_track` in
`core.environment`) and the TSRD ingestion layer
(`tsrd.tsrd_emitter.TSRDEmitterSampler`) MUST use these helpers to
guarantee that band / slot indices are identical across the two
paths. The future TSRD oracle (`tsrd_oracle.py`, deferred) will
import from here as well.

[INVARIANT-2] The grid is defined exactly once, in
`SimulationConfig.band_count`, `SimulationConfig.total_spectrum_mhz`,
`SimulationConfig.dwell_time_ms`, and `SimulationConfig.time_slots`.
Any code that uses a different band width, frequency range, or slot
duration is a Gate-0 violation and must be caught in code review.

[SCIENTIFIC] These helpers are pure functions of `SimulationConfig`.
They do NOT touch the hidden truth grid; they only translate between
continuous and discrete representations. The mapping is deterministic
and reversible up to the discretisation grid (see the
`band_center_frequency_mhz` and `slot_start_seconds` round-trip tests).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .environment import SimulationConfig


# =====================================================================
# FREQUENCY → BAND
# =====================================================================
def frequency_to_band(freq_mhz: float, cfg: SimulationConfig) -> int:
    """
    Map a centre frequency (MHz) to a band index in
    ``[0, cfg.band_count - 1]``.

    Two operating modes:

    1. **Uniform band centres** (the default; `cfg.dwell_centres_mhz is None`).
       The grid is divided into `band_count` equal-width bands of
       `band_width_mhz = total_spectrum_mhz / band_count` MHz, with
       band ``b`` covering ``[b * band_width, (b + 1) * band_width)`` and
       centre ``(b + 0.5) * band_width``. Formula::

           band = floor( (freq_mhz / total_spectrum_mhz) * band_count )

    2. **TSRD-dwell-centre mode** (`cfg.dwell_centres_mhz` is set). The
       array provides the actual tune frequency for each of the
       `band_count` receivers dwells. Mapping is **nearest-neighbour**:
       the pulse is assigned to the band whose `dwell_centres_mhz[band]`
       is closest to `freq_mhz`. This is the assignment a real ESM
       receiver would produce: two pulses that fall inside the same
       500 MHz IBW land in the same band regardless of where in the
       IBW they sit, but two pulses straddling a band boundary land
       in different bands.

       TSRD Scan-Mode H5 files store these centres in
       `metadata/receiver/dwell_centres_mhz`. The TSRDAdapter reads
       them and stores them on the SimulationConfig; the synthetic
       path leaves the field `None`.

    Edge handling
    -------------
    * Frequencies below the lowest band centre (uniform: < 0 MHz;
      TSRD: < min(dwell_centres_mhz) - bw/2) clip to band 0.
    * Frequencies above the highest band centre clip to the last band.
    * NaN frequencies clip to band 0 (graceful, scheduler sees a miss).

    Parameters
    ----------
    freq_mhz : float
        Centre frequency in MHz. May be negative, NaN, or larger than
        the upper edge; both are clipped to the nearest valid band.
    cfg : SimulationConfig
        The simulation configuration that defines the grid.

    Returns
    -------
    int
        Band index in ``[0, cfg.band_count - 1]``.

    Raises
    ------
    ValueError
        If ``cfg.total_spectrum_mhz <= 0``, ``cfg.band_count <= 0``,
        or if ``cfg.dwell_centres_mhz`` is set but has the wrong length
        or contains non-finite values.
    """
    if cfg.total_spectrum_mhz <= 0:
        raise ValueError(
            f"SimulationConfig.total_spectrum_mhz must be positive, "
            f"got {cfg.total_spectrum_mhz}"
        )
    if cfg.band_count <= 0:
        raise ValueError(
            f"SimulationConfig.band_count must be positive, "
            f"got {cfg.band_count}"
        )

    # [TSRD-DERIVED] If the config has explicit dwell centres, use
    # nearest-neighbour assignment. This is the assignment a real
    # scan receiver would produce given the same tune grid.
    if cfg.dwell_centres_mhz is not None:
        dc = np.asarray(cfg.dwell_centres_mhz, dtype=np.float64)
        if dc.shape != (cfg.band_count,):
            raise ValueError(
                f"dwell_centres_mhz must be 1D of length band_count="
                f"{cfg.band_count}, got shape {dc.shape}"
            )
        if not np.all(np.isfinite(dc)):
            raise ValueError(
                f"dwell_centres_mhz must contain only finite values, "
                f"got {dc}"
            )
        # NaN frequency -> band 0 (graceful: scheduler sees a miss)
        try:
            f = float(freq_mhz)
            if not np.isfinite(f):
                return 0
        except (TypeError, ValueError):
            return 0
        # Nearest-neighbour index
        diffs = np.abs(dc - f)
        return int(np.argmin(diffs))

    # Uniform fallback
    if not np.isfinite(freq_mhz):
        return 0
    rel = float(freq_mhz) / float(cfg.total_spectrum_mhz)
    band = int(rel * cfg.band_count)
    if band < 0:
        return 0
    if band >= cfg.band_count:
        return cfg.band_count - 1
    return band


def band_edges_mhz(cfg: SimulationConfig) -> tuple:
    """
    Return ``(lower_mhz, upper_mhz, band_width_mhz)`` for the grid.

    When `cfg.dwell_centres_mhz` is set, the lower/upper edges are
    computed as the midpoint between adjacent dwell centres, and the
    band_width is the nominal uniform width for reference. When not set,
    the lower edge is 0.0 and upper edge is `total_spectrum_mhz`.

    Convenience helper for documentation, plotting, and the future oracle.
    """
    if cfg.dwell_centres_mhz is not None:
        dc = np.asarray(cfg.dwell_centres_mhz, dtype=np.float64)
        bw = float(cfg.band_width_mhz())
        return float(dc.min() - bw / 2), float(dc.max() + bw / 2), bw
    return 0.0, float(cfg.total_spectrum_mhz), float(cfg.band_width_mhz())


# =====================================================================
# TIME → SLOT
# =====================================================================
def seconds_to_slot(t_s: float, cfg: SimulationConfig) -> int:
    """
    Map a time offset (seconds) from mission start to an integer slot
    index in ``[0, cfg.time_slots - 1]``.

    Formula
    -------
        slot = floor( t_s / slot_duration_s )

    Edge handling
    -------------
    * Negative times are clipped to slot 0.
    * Times at or beyond the mission duration
      (>= time_slots * slot_duration) are clipped to the last slot.

    Parameters
    ----------
    t_s : float
        Time offset in seconds from mission start.
    cfg : SimulationConfig
        The simulation configuration that defines the grid.

    Returns
    -------
    int
        Slot index in ``[0, cfg.time_slots - 1]``.

    Raises
    ------
    ValueError
        If ``cfg.slot_duration_s()`` returns a non-positive value.
    """
    slot_dur = cfg.slot_duration_s()
    if slot_dur <= 0:
        raise ValueError(
            f"SimulationConfig.slot_duration_s() must be positive, "
            f"got {slot_dur}"
        )

    # Floor with a tiny epsilon to defeat IEEE-754 drift on exact
    # slot boundaries (e.g. 0.05 is not exactly representable in
    # binary, so 5 * 0.05 = 0.25000000000000006 and the naive floor
    # would return 4 instead of 5). 1e-9 is well below any sensible
    # slot duration (1 ns vs 50 ms) but enough to absorb float drift.
    slot = int(np.floor(t_s / slot_dur + 1e-9))

    if slot < 0:
        return 0
    if slot >= cfg.time_slots:
        return cfg.time_slots - 1
    return slot


def slot_edges_seconds(cfg: SimulationConfig) -> tuple:
    """
    Return ``(start_s, end_s, slot_duration_s)`` for the time grid.

    Convenience helper for the oracle and for plotting.
    """
    start = 0.0
    end = float(cfg.time_slots) * float(cfg.slot_duration_s())
    return start, end, float(cfg.slot_duration_s())


# =====================================================================
# REVERSE MAPPINGS (for evaluation, NOT for the scheduler)
# =====================================================================
def band_center_frequency_mhz(band: int, cfg: SimulationConfig) -> float:
    """
    Return the centre frequency (MHz) of the given band.

    If `cfg.dwell_centres_mhz` is set, returns the TSRD-recorded
    tune frequency for that band directly. Otherwise returns the
    uniform-band-centre value ``(band + 0.5) * band_width``.

    [EVAL-ONLY] This mapping is the *inverse* of `frequency_to_band` and
    is provided for the oracle and for plotting. The scheduler must
    NEVER use this to "cheat" by converting band → frequency; the
    scheduler only sees the integer band index.
    """
    if not (0 <= band < cfg.band_count):
        raise ValueError(
            f"band {band} out of range [0, {cfg.band_count - 1}]"
        )
    if cfg.dwell_centres_mhz is not None:
        dc = np.asarray(cfg.dwell_centres_mhz, dtype=np.float64)
        if dc.shape != (cfg.band_count,):
            raise ValueError(
                f"dwell_centres_mhz must be 1D of length band_count="
                f"{cfg.band_count}, got shape {dc.shape}"
            )
        return float(dc[band])
    return (band + 0.5) * cfg.band_width_mhz()


def slot_start_seconds(slot: int, cfg: SimulationConfig) -> float:
    """
    Return the start time (seconds) of the given slot.

    The end of the slot is ``slot_start_seconds(slot) + slot_duration_s()``.
    """
    if not (0 <= slot < cfg.time_slots):
        raise ValueError(
            f"slot {slot} out of range [0, {cfg.time_slots - 1}]"
        )
    return float(slot) * cfg.slot_duration_s()

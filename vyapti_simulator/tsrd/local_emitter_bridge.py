"""
vyapti_simulator.tsrd.local_emitter_bridge
==========================================

PDW bridge from `src.emitter_models` to the TSRD band/slot grid.

This is the second of the five unification deliverables: the bridge
that lets the **synthetic emitter pipeline** (System A) feed the
**TSRD band/slot observation pipeline** (System B) so the scheduler
sees one BandSlotPulse schema regardless of source.

Pipeline
--------

    src.emitter_models.Emitter      # System A, e.g. FrequencyAgileEmitter
            │
            │  .generate_pulses(sim_start_sec, sim_end_sec, rng)
            ▼
    List[src.emitter_models.Pulse] # Already SI units (s, Hz, dBm)
            │
            │  _pulses_to_pdw_stream()   # μs/MHz/dB convention required
            ▼                               by downstream `discretize_pdw_to_bands`
    vyapti_simulator.tsrd.tsrd_adapter.PDWStream
            │
            │  discretize_pdw_to_bands()  # SAME function the TSRD path uses
            ▼
    Iterator[pdw_discretiser.BandSlotPulse]   # TSRD-shape per-pulse objects
            │
            │  .to_unified(...)           # unit conversion + provenance stamp
            ▼
    src.observation_interface.BandSlotPulse   # The ONE schema the scheduler sees
            │
            │  collect by (band, slot)
            ▼
    DiscretisedGrid  ← the EXACT same return type as the TSRD path

Why the same function?
----------------------
Per unification requirement (2): "Route both real TSRD PDW streams
and synthetic emitter output through the SAME
``discretize_pdw_to_bands()`` function." This is the single point
where frequency → band and time → slot are computed. If the synthetic
path used a different mapping, the two paths would produce slightly
different (band, slot) coordinates for the same physical pulse and
scheduler comparisons across sources would be meaningless.

What the bridge does NOT do
---------------------------
- It does NOT touch the truth grid. ``generate_pulses`` returns
  pulses; the bridge never inspects whether a pulse is "supposed
  to be there" according to a hidden truth.
- It does NOT run a detection model. The unified environment
  (Task #4) wraps these pulses with the same AGC / detection
  pipeline that the TSRD path uses.
- It does NOT branch on data_source inside the discretiser. The
  TSRD `discretize_pdw_to_bands` is unaware of synthetic vs. real.

Author
------
Senior RF/EW Signal Simulation Engineer — PS26055 System A/B
unification deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np

from .pdw_discretiser import (
    BandSlotCell,
    DiscretisedGrid,
    discretise_pdw_to_grid,
    discretize_pdw_to_bands,
)
from .tsrd_adapter import PDW_STREAM_FIELDS, PDWStream
from ..core.environment import SimulationConfig

# `src` is a sibling top-level package; absolute import keeps the
# bridge free of `..` jumps that would reach above the
# `vyapti_simulator` package root.
from src.observation_interface import BandSlotPulse as UnifiedBandSlotPulse
from src.observation_interface import DataSource


# =====================================================================
# Noise-floor convention
# =====================================================================
# src/emitter_models.Pulse.amplitude is dBm. TSRD's amp_db is a
# relative scale (not absolute dBm). To keep both paths through the
# SAME `discretize_pdw_to_bands` (which expects amp_db as a
# TSRD-relative dB), the bridge reports amp_db = pulse.amplitude_dbm
# after subtracting `noise_floor_dbm`. That subtraction is the SI
# analog of TSRD's "amplitude above noise floor", and it preserves
# the SNR ordering: a 15 dB SNR pulse from src/ is on the same axis
# as a 15 dB SNR pulse from TSRD.
#
# The unified contract's `amplitude_dbm` field is RECOMPUTED in
# `BandSlotPulse.from_tsrd_pulse` as `noise_floor_dbm + amp_db`,
# so the scheduler sees the original dBm value transparently.
DEFAULT_NOISE_FLOOR_DBM: float = -130.0


# =====================================================================
# Result containers
# =====================================================================
@dataclass(frozen=True)
class BridgeResult:
    """
    Output of :func:`bridge_local_emitters_to_grid`.

    Attributes
    ----------
    unified_pulses : tuple[UnifiedBandSlotPulse, ...]
        Per-pulse unified observations, one per input src/ pulse
        that mapped to a valid (band, slot) cell. Preserves the
        order of the underlying PDW stream.

    grid : DiscretisedGrid
        The discretised (band, slot) grid, with one cell per
        occupied cell. **This is the SAME type the TSRD path
        returns** — the unified environment can hand it to the
        detection pipeline without a separate code path.

    dropped : int
        Number of src/ pulses that fell outside the grid
        (out-of-range frequency or time) and were silently
        discarded. Mirrors the discretiser's "Out-of-range
        pulses are dropped" invariant.

    n_input_pulses : int
        Total src/ pulses received, before the drop filter.
    """
    unified_pulses: tuple
    grid: DiscretisedGrid
    dropped: int
    n_input_pulses: int


# =====================================================================
# Pulse → PDWStream
# =====================================================================
def _pulse_to_pdw_arrays(
    pulses: Sequence,
    noise_floor_dbm: float,
) -> dict:
    """
    Convert a sequence of ``src.emitter_models.Pulse`` objects to the
    six numpy arrays that make up a TSRD-shaped ``PDWStream``.

    Unit conversion happens here:
      - ``toa_s``   → ``toa_us``  (s × 1e6)
      - ``freq_hz`` → ``freq_mhz`` (Hz × 1e-6)
      - ``pw_s``    → ``pw_us``   (s × 1e6)
      - ``aoa_deg`` → ``aoa_deg`` (no change)
      - ``amplitude_dbm`` → ``amp_db = amplitude_dbm - noise_floor_dbm``
                            (TSRD-relative dB above noise floor)
      - ``emitter_id``  → ``emitter_id`` (int; -1 if unknown)

    Parameters
    ----------
    pulses : Sequence[src.emitter_models.Pulse]
    noise_floor_dbm : float
        Receiver noise floor in dBm. Used to convert the dBm
        amplitude to TSRD-relative dB.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``toa_us``, ``freq_mhz``, ``pw_us``, ``aoa_deg``,
        ``amp_db``, ``emitter_id``. Empty arrays if no pulses.
    """
    if not pulses:
        # Match the TSRD dtypes (float32 / int64) so the resulting
        # PDWStream is byte-compatible with the real-TSRD path.
        return {
            "toa_us": np.empty(0, dtype=np.float32),
            "freq_mhz": np.empty(0, dtype=np.float32),
            "pw_us": np.empty(0, dtype=np.float32),
            "aoa_deg": np.empty(0, dtype=np.float32),
            "amp_db": np.empty(0, dtype=np.float32),
            "emitter_id": np.empty(0, dtype=np.int64),
        }

    n = len(pulses)
    toa_us = np.empty(n, dtype=np.float32)
    freq_mhz = np.empty(n, dtype=np.float32)
    pw_us = np.empty(n, dtype=np.float32)
    aoa_deg = np.empty(n, dtype=np.float32)
    amp_db = np.empty(n, dtype=np.float32)
    emitter_id = np.empty(n, dtype=np.int64)

    for i, p in enumerate(pulses):
        # Defensive `getattr` calls: src/ emitter models are a
        # living class hierarchy; older or stripped-down Pulse
        # objects may not have every field. Defaults match the
        # convention "if absent, treat as zero / unknown".
        #
        # NOTE: Pulse.toa is the actual field name (not "toa_s").
        # Frequency Agile and base Emitter pulses carry no AoA; the
        # deinterleaver (Option C) assigns it later from the
        # scan schedule or beam geometry. We default to 0.
        toa_us[i] = float(getattr(p, "toa", getattr(p, "time_s", 0.0))) * 1e6
        freq_mhz[i] = float(getattr(p, "frequency_hz", 0.0)) * 1e-6
        pw_us[i] = float(getattr(p, "pulse_width_sec", 0.0)) * 1e6
        aoa_deg[i] = float(getattr(p, "aoa_deg", 0.0))
        amp_dbm = float(getattr(p, "amplitude_dbm", noise_floor_dbm))
        amp_db[i] = amp_dbm - noise_floor_dbm
        eid = int(getattr(p, "emitter_id", -1))
        emitter_id[i] = eid

    return {
        "toa_us": toa_us,
        "freq_mhz": freq_mhz,
        "pw_us": pw_us,
        "aoa_deg": aoa_deg,
        "amp_db": amp_db,
        "emitter_id": emitter_id,
    }


def _pdw_stream_from_arrays(arrays: dict) -> PDWStream:
    """
    Pack the six arrays into a frozen ``PDWStream`` with field order
    matching ``PDW_STREAM_FIELDS``.
    """
    return PDWStream(
        toa_us=arrays["toa_us"],
        freq_mhz=arrays["freq_mhz"],
        pw_us=arrays["pw_us"],
        aoa_deg=arrays["aoa_deg"],
        amp_db=arrays["amp_db"],
        emitter_id=arrays["emitter_id"],
    )


# =====================================================================
# Unified-pulse re-stamp (override data_source → SYNTHETIC_DYNAMICS)
# =====================================================================
def _restamp_as_synthetic(
    pulse: UnifiedBandSlotPulse,
    *,
    provenance: Union[str, dict],
) -> UnifiedBandSlotPulse:
    """
    Re-stamp a unified pulse as ``DataSource.SYNTHETIC_DYNAMICS`` and
    merge in the bridge's provenance label.

    The discretiser (``discretize_pdw_to_bands``) emits unified pulses
    stamped with ``DataSource.REAL_TSRD`` (the conservative default —
    that factory's caller is real TSRD data). For the synthetic path
    we re-stamp each pulse with the synthetic tag so the scheduler
    sees the correct provenance. The pulse is reconstructed
    field-for-field; only ``data_source`` and ``provenance`` change.

    [UNIFICATION 2026-09-05] The unified ``BandSlotPulse`` is
    ``frozen=True``; we cannot mutate it in place. The reconstruction
    is a constant-time copy of 16 fields.
    """
    if isinstance(provenance, str):
        prov_dict: dict = {"label": provenance}
    else:
        prov_dict = dict(provenance or {})
    prov_dict.setdefault("data_origin", "src.emitter_models")
    return UnifiedBandSlotPulse(
        toa_s=pulse.toa_s,
        frequency_hz=pulse.frequency_hz,
        pulse_width_s=pulse.pulse_width_s,
        amplitude_dbm=pulse.amplitude_dbm,
        aoa_deg=pulse.aoa_deg,
        emitter_id=pulse.emitter_id,
        band=pulse.band,
        slot=pulse.slot,
        snr_db=pulse.snr_db,
        noise_floor_dbm=pulse.noise_floor_dbm,
        data_source=DataSource.SYNTHETIC_DYNAMICS,
        source_h5_sha256=None,
        provenance=prov_dict,
    )


# =====================================================================
# Main bridge entry point
# =====================================================================
def bridge_local_emitters_to_grid(
    emitters: Sequence,
    *,
    sim_cfg: SimulationConfig,
    sim_start_sec: float,
    sim_end_sec: float,
    rng: np.random.Generator,
    noise_floor_dbm: float = DEFAULT_NOISE_FLOOR_DBM,
    nominal_noise_floor_db: float = DEFAULT_NOISE_FLOOR_DBM,
    provenance: str = "src.emitter_models",
    source_h5_sha256: Optional[str] = None,
) -> BridgeResult:
    """
    Run a list of ``src.emitter_models.Emitter`` objects through
    ``generate_pulses``, discretise the result with the SAME function
    the TSRD path uses, and return both the per-pulse unified
    observations and a ``DiscretisedGrid``.

    Parameters
    ----------
    emitters : Sequence[src.emitter_models.Emitter]
        The local synthetic emitters. Each must expose
        ``generate_pulses(sim_start_sec, sim_end_sec, rng) -> List[Pulse]``.
        Both base ``Emitter`` and ``DynamicEmitter`` (a thin wrapper
        around a base emitter + a ``LifecyclePolicy``) qualify.

    sim_cfg : SimulationConfig
        The band/slot grid. **MUST be the same config used by the
        TSRD path** for any cross-source comparison to be valid;
        a different ``dwell_centres_mhz`` here would route a pulse
        to a different (band, slot) than the TSRD run.

    sim_start_sec, sim_end_sec : float
        Mission window passed to each emitter's ``generate_pulses``.

    rng : np.random.Generator
        Deterministic numpy Generator. Spent on emitter pulse
        generation; **NOT** consumed by the discretiser (which is
        pure and deterministic in the band/slot mapping). The
        caller owns the SeedSequence; paired-comparison sweeps
        should pass the same root seed to both System A and
        System B calls.

    noise_floor_dbm : float
        Receiver noise floor in dBm. Used to convert each
        ``Pulse.amplitude_dbm`` to a TSRD-relative ``amp_db``
        BEFORE the discretiser runs. Default -130 dBm matches
        TSRD's empirical floor (see ``tsrd_environment.DetectionConfig``).

    nominal_noise_floor_db : float
        The TSRD-relative noise floor passed to
        ``discretize_pdw_to_bands`` for ``snr_db`` computation.
        For paired comparison, this should be the same value used
        in the TSRD-side ``DetectionConfig``. Default
        ``noise_floor_dbm`` because, when the synthetic pulse
        amplitudes are already dB-above-floor, the TSRD-style
        ``snr_db`` equals the original SNR.

    provenance : str
        Free-text tag stored on every unified pulse's
        ``provenance`` field. Useful for distinguishing
        src/ emitter families in evaluation.

    source_h5_sha256 : Optional[str]
        For the synthetic path this is always None. Kept in the
        signature for symmetry with the TSRD-side builder; the
        scheduler interface strips both before exposing the
        observation.

    Returns
    -------
    BridgeResult
        Container with ``unified_pulses`` (per-pulse unified
        ``BandSlotPulse``), ``grid`` (the same ``DiscretisedGrid``
        type the TSRD path returns), ``dropped`` count, and
        ``n_input_pulses`` for observability.
    """
    # 1. Collect all src/ pulses from all emitters.
    all_pulses: List = []
    for em in emitters:
        all_pulses.extend(
            em.generate_pulses(
                sim_start_sec=sim_start_sec,
                sim_end_sec=sim_end_sec,
                rng=rng,
            )
        )
    n_input = len(all_pulses)

    # 2. Convert to a TSRD-shaped PDWStream (units: μs / MHz / dB).
    arrays = _pulse_to_pdw_arrays(all_pulses, noise_floor_dbm=noise_floor_dbm)
    pdw = _pdw_stream_from_arrays(arrays)

    # 3. Run through the SAME discretiser the TSRD path uses.
    #    discretize_pdw_to_bands is a generator → preserve order.
    #    The generator now yields unified BandSlotPulse stamped with
    #    DataSource.REAL_TSRD; we re-stamp each as SYNTHETIC_DYNAMICS.
    unified_pulses_from_discretiser: List[UnifiedBandSlotPulse] = list(
        discretize_pdw_to_bands(
            pdw,
            sim_cfg,
            nominal_noise_floor_db=nominal_noise_floor_db,
        )
    )
    n_after = len(unified_pulses_from_discretiser)
    dropped = n_input - n_after

    # 4. Re-stamp each pulse as synthetic dynamics.
    unified: List[UnifiedBandSlotPulse] = [
        _restamp_as_synthetic(p, provenance=provenance)
        for p in unified_pulses_from_discretiser
    ]

    # 5. Build the SAME DiscretisedGrid the TSRD path returns, using
    #    the SAME TSRD helper. This is the cell-aggregated view the
    #    unified environment's detection pipeline will consume.
    grid: DiscretisedGrid = discretise_pdw_to_grid(pdw, sim_cfg)

    return BridgeResult(
        unified_pulses=tuple(unified),
        grid=grid,
        dropped=dropped,
        n_input_pulses=n_input,
    )


# =====================================================================
# Convenience: build PDWStream directly (for callers that want to
# deinterleave, not discretise)
# =====================================================================
def local_emitters_to_pdw_stream(
    emitters: Sequence,
    *,
    sim_start_sec: float,
    sim_end_sec: float,
    rng: np.random.Generator,
    noise_floor_dbm: float = DEFAULT_NOISE_FLOOR_DBM,
    swerling_models: Optional[Dict[int, str]] = None,
    swerling_burst_size: int = 10,
) -> PDWStream:
    """
    Just the first half of the bridge: src/ emitter output → PDWStream.

    Useful when the caller wants to run the deinterleaver (which
    takes a ``PDWStream``) without going through the discretiser.
    Does not call ``frequency_to_band`` or ``seconds_to_slot`` —
    it is a pure pulse-list → TSRD-arrays transform.

    Parameters
    ----------
    emitters : Sequence[src.emitter_models.Emitter]
        The synthetic emitter objects whose pulses are converted
        to TSRD arrays.
    sim_start_sec, sim_end_sec : float
        Mission window.
    rng : np.random.Generator
        RNG for pulse generation and (optional) Swerling draws.
    noise_floor_dbm : float
        Receiver noise floor in dBm.
    swerling_models : dict[int, str], optional
        Per-emitter Swerling model. Keys are ``emitter.emitter_id``,
        values are model names: ``"swerling_0"`` (default, no
        fluctuation), ``"swerling_1"``, ``"swerling_2"``,
        ``"swerling_3"``, ``"swerling_4"``. See
        :mod:`vyapti_simulator.rf.swerling`. Default None (no
        fluctuation; equivalent to all Swerling 0).
    swerling_burst_size : int
        Burst size for slow-fluctuation models (Swerling I/III).
        Default 10. Ignored for fast models (II/IV).
    """
    from vyapti_simulator.rf.swerling import apply_swerling_fluctuation
    swerling_models = swerling_models or {}

    # Group pulses by emitter so Swerling fluctuation can be applied
    # per-emitter (slow models need to keep bursts intact).
    per_emitter_pulses: Dict[int, List] = {}
    for em in emitters:
        em_pulses = em.generate_pulses(
            sim_start_sec=sim_start_sec,
            sim_end_sec=sim_end_sec,
            rng=rng,
        )
        eid = int(getattr(em, "emitter_id", -1))
        per_emitter_pulses.setdefault(eid, []).extend(em_pulses)

    # Apply Swerling per emitter, then concatenate.
    all_pulses: List = []
    for eid, em_pulses in per_emitter_pulses.items():
        model = swerling_models.get(eid, "swerling_0")
        if model != "swerling_0" and em_pulses:
            arr = _pulse_to_pdw_arrays(em_pulses, noise_floor_dbm=noise_floor_dbm)
            amp_db_in = arr["amp_db"].astype(np.float64)
            amp_db_out = apply_swerling_fluctuation(
                amp_db_in, model=model, rng=rng,
                burst_size=swerling_burst_size,
            )
            # Re-stamp the dBm-relative amplitude back into the Pulse
            # objects so downstream sees the fluctuated value.
            for p, new_amp_db in zip(em_pulses, amp_db_out):
                p.amplitude_dbm = float(new_amp_db) + noise_floor_dbm
        all_pulses.extend(em_pulses)

    arrays = _pulse_to_pdw_arrays(all_pulses, noise_floor_dbm=noise_floor_dbm)
    return _pdw_stream_from_arrays(arrays)

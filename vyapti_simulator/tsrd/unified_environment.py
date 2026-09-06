"""
vyapti_simulator.tsrd.unified_environment
========================================

Unified simulation environment for the PS26055 scheduler experiment.

This is Task #4 of the System A/B unification: a single entry point that
produces a scheduler-compatible environment from EITHER:

  * **Real TSRD data** (``data_source = "real_tsrd"``) — a pre-loaded
    ``PDWStream`` from ``TSRDAdapter``, discretised with the TSRD path's
    band/slot grid and fed to ``TSRDEnvironment``.

  * **Synthetic emitter dynamics** (``data_source = "synthetic_dynamics"``) —
    a list of ``src.emitter_models.Emitter`` instances (built from
    ``TSRDEmitterSampler.build_emitter_objects()``), converted to a
    ``PDWStream`` by the bridge, and discretised with the SAME function
    the TSRD path uses.

Both paths share the same ``TSRDEnvironment`` internals:
the same ``DetectionConfig``, the same AGC pipeline, the same
discretisation function, and the same observation contract.
The scheduler (UCB, Thompson, Round Robin, etc.) is NEVER told
which source produced its observations.

**No data_source branching inside the scheduler.** The environment
is the only place ``data_source`` is consulted. The observation
returned by ``step()`` conforms to ``PERMITTED_OBSERVATION_KEYS``
identically for both sources.

Usage
-----
Real TSRD::

    adapter = TSRDAdapter(h5_path=".../config_0.h5", ...)
    pdw = adapter.to_pdw_stream()
    env = build_unified_environment(
        data_source="real_tsrd",
        pdw_stream=pdw,
        simulation_config=sim_cfg,
        detection_config=DetectionConfig(),
        seed=42,
    )
    scheduler = UCB1Scheduler()
    run_episode(env, scheduler)

Synthetic dynamics::

    sampler = TSRDEmitterSampler(
        tsrd_statistics_path="vyapti_simulator/data/tsrd_statistics.json",
        simulation_config=sim_cfg,
        seed=42,
    )
    emitters = sampler.build_emitter_objects()
    env = build_unified_environment(
        data_source="synthetic_dynamics",
        emitters=emitters,
        simulation_config=sim_cfg,
        detection_config=DetectionConfig(),
        seed=42,
    )
    scheduler = UCB1Scheduler()
    run_episode(env, scheduler)

Author
------
Senior RF/EW Signal Simulation Engineer — PS26055 System A/B
unification deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Union

import numpy as np

from .tsrd_adapter import PDWStream
from .tsrd_environment import (
    DetectionConfig,
    TSRDEnvironment,
)
from .local_emitter_bridge import local_emitters_to_pdw_stream
from ..core.environment import SimulationConfig

# The enum is mirrored here (not re-exported from src/) to keep the
# vyapti_simulator package self-contained and avoid a cross-package
# import in the hot path. The two enums are identical in value.


class DataSource(str, Enum):
    """
    Which subsystem produced the underlying pulse stream.

    This enum is the canonical source tag; it is mirrored from
    ``src.observation_interface.DataSource`` so that the
    vyapti_simulator package never needs to import from src/
    in its public interfaces. The two enum instances are identical.
    """
    REAL_TSRD = "real_tsrd"
    SYNTHETIC_DYNAMICS = "synthetic_dynamics"


# =====================================================================
# Factory
# =====================================================================

def build_unified_environment(
    data_source: Union[str, DataSource],
    simulation_config: SimulationConfig,
    detection_config: Optional[DetectionConfig] = None,
    deinterleaver_config=None,
    seed: int = 42,
    *,
    # Source-specific arguments (only one must be provided)
    pdw_stream: Optional[PDWStream] = None,
    emitters: Optional[Sequence] = None,
    sim_start_sec: float = 0.0,
    sim_end_sec: Optional[float] = None,
    emitter_rng_seed: Optional[int] = None,
) -> UnifiedEnvironment:
    """
    Build a unified environment from either a real TSRD PDW stream
    or a list of synthetic ``src.emitter_models.Emitter`` instances.

    Both paths return a ``UnifiedEnvironment`` subclassing
    ``TSRDEnvironment``; the scheduler interface is identical.

    Parameters
    ----------
    data_source : str | DataSource
        Which path to use: ``"real_tsrd"`` or ``"synthetic_dynamics"``.
        The string value is accepted for convenience; it is validated
        and converted to the enum.

    simulation_config : SimulationConfig
        Grid dimensions and timing parameters. Must be the SAME config
        used by any TSRD adapter for cross-source comparison to be valid.

    detection_config : DetectionConfig | None
        Amplitude → SNR → detection parameters. Defaults to
        ``DetectionConfig()`` (fixed -130 dBm floor, Pd curve).

    deinterleaver_config : DeinterleaverConfig | None
        If provided, run the PRI-based deinterleaver (Option C)
        and store the resulting tracks. Deferred for v2.

    seed : int
        RNG seed for the detector noise draws (false alarms, logistic
        curve randomness). Also used as the seed for the emitter
        bridge RNG when ``data_source == "synthetic_dynamics"``.

    pdw_stream : PDWStream | None
        Required when ``data_source == "real_tsrd"``. The PDW stream
        must be from a Scan-mode H5 file; Stare-mode raises
        ``StareModeOracleError`` inside ``TSRDEnvironment``.

    emitters : Sequence[src.emitter_models.Emitter] | None
        Required when ``data_source == "synthetic_dynamics"``.
        The list of emitter objects produced by
        ``TSRDEmitterSampler.build_emitter_objects()``.

    sim_start_sec, sim_end_sec : float
        Mission window for the emitter bridge. ``sim_end_sec`` defaults
        to ``time_slots * slot_duration_s`` if not supplied.

    emitter_rng_seed : int | None
        RNG seed for the emitter bridge's pulse generation. Defaults
        to ``seed`` if not supplied (same seed drives both the
        detector noise and the emitter pulse generation).

    Returns
    -------
    UnifiedEnvironment
        A ``TSRDEnvironment`` subclass with the unified environment's
        ``data_source`` property set.

    Raises
    ------
    ValueError
        If ``data_source`` is not a valid enum member, or if the
        wrong source-specific argument is missing.

    Examples
    --------
    See module docstring for full usage examples.
    """
    # --- Validate data_source ---
    if isinstance(data_source, str):
        try:
            data_source = DataSource(data_source)
        except ValueError:
            valid = [e.value for e in DataSource]
            raise ValueError(
                f"data_source must be one of {valid}, got {data_source!r}"
            )

    if data_source not in (DataSource.REAL_TSRD, DataSource.SYNTHETIC_DYNAMICS):
        raise ValueError(
            f"data_source must be DataSource.REAL_TSRD or "
            f"DataSource.SYNTHETIC_DYNAMICS, got {data_source!r}"
        )

    # --- Resolve source-specific arguments ---
    if data_source == DataSource.REAL_TSRD:
        if pdw_stream is None:
            raise ValueError(
                "pdw_stream is required when data_source='real_tsrd'"
            )
        _pdw: PDWStream = pdw_stream
        _data_source_tag = DataSource.REAL_TSRD

    elif data_source == DataSource.SYNTHETIC_DYNAMICS:
        if emitters is None:
            raise ValueError(
                "emitters (list of src.emitter_models.Emitter) is required "
                "when data_source='synthetic_dynamics'"
            )
        # Default mission end: last slot boundary
        if sim_end_sec is None:
            sim_end_sec = float(
                simulation_config.time_slots * simulation_config.slot_duration_s()
            )
        # Emitter bridge RNG: separate from detector RNG so they don't
        # share state. Same root seed if not overridden.
        bridge_seed = emitter_rng_seed if emitter_rng_seed is not None else seed
        bridge_rng = np.random.default_rng(bridge_seed)
        _pdw = local_emitters_to_pdw_stream(
            list(emitters),
            sim_start_sec=sim_start_sec,
            sim_end_sec=sim_end_sec,
            rng=bridge_rng,
        )
        _data_source_tag = DataSource.SYNTHETIC_DYNAMICS

    else:
        # Defensive: should not reach here given the enum check above
        raise ValueError(f"Unknown data_source: {data_source}")

    # --- Build the environment ---
    env = UnifiedEnvironment(
        pdw_stream=_pdw,
        simulation_config=simulation_config,
        detection_config=detection_config,
        deinterleaver_config=deinterleaver_config,
        seed=seed,
        _data_source=_data_source_tag,
    )
    return env


# =====================================================================
# Unified environment (subclass of TSRDEnvironment)
# =====================================================================

class UnifiedEnvironment(TSRDEnvironment):
    """
    A ``TSRDEnvironment`` subclass that additionally carries the
    ``data_source`` provenance tag so that evaluation code can
    distinguish real-TSRD runs from synthetic-dynamics runs without
    inspecting the pulse content.

    The scheduler interface (``step()``, ``reset()``, ``done``,
    ``observation_history``) is identical to ``TSRDEnvironment`` and
    conforms to ``PERMITTED_OBSERVATION_KEYS``.

    **No scheduler-branching rule:** ``data_source`` is read-only
    at runtime. The scheduler never branches on it. The evaluation
    engine uses it to label results in the metrics table.

    Attributes
    ----------
    data_source : DataSource
        The source flag this environment was constructed with.
        Read-only; set by the constructor only.

    provenance : dict
        Free-form provenance record. Includes ``data_source``,
        the emitter bridge RNG seed (for synthetic path), and
        the source H5 SHA-256 (for real-TSRD path).
    """

    _data_source: DataSource

    def __init__(
        self,
        pdw_stream: PDWStream,
        simulation_config: SimulationConfig,
        detection_config: Optional[DetectionConfig] = None,
        deinterleaver_config=None,
        seed: int = 42,
        _data_source: DataSource = DataSource.REAL_TSRD,
    ) -> None:
        # Delegate to TSRDEnvironment.__init__; it builds the grid
        # from the PDWStream and runs the deinterleaver if requested.
        super().__init__(
            pdw_stream=pdw_stream,
            simulation_config=simulation_config,
            detection_config=detection_config,
            deinterleaver_config=deinterleaver_config,
            seed=seed,
        )
        self._data_source = _data_source

    @property
    def data_source(self) -> DataSource:
        """Read-only source provenance tag."""
        return self._data_source

    @property
    def provenance(self) -> dict:
        """
        Provenance record for this environment run.

        Includes ``data_source`` and source-specific fields:
          - real_tsrd: ``source_h5_sha256``
          - synthetic_dynamics: ``bridge_emitter_seed``
        """
        prov: dict = {"data_source": self._data_source.value}
        if self._data_source == DataSource.SYNTHETIC_DYNAMICS:
            prov["bridge_emitter_seed"] = self._seed
        return prov

    def replay_signature(self) -> str:
        """
        Deterministic fingerprint for Gate-0 determinism verification.

        Extends ``TSRDEnvironment.replay_signature`` with the
        ``data_source`` tag so that paired comparison between
        real-TSRD and synthetic-dynamics runs can be verified
        as having different ground truth.
        """
        base = super().replay_signature()
        return f"data_source={self._data_source.value}:{base}"

"""
vyapti_simulator.tsrd.h5_writer
================================

Write a synthetic Vyapti scenario to a TSRD-shaped HDF5 file.

The written H5 is compatible with ``TSRDAdapter`` and therefore with
the entire Option B/C pipeline (deinterleaver, TSRDEnvironment, etc.)
without any code changes. Any TSRD file written by this module can be
loaded with::

    from vyapti_simulator.tsrd import TSRDAdapter
    adapter = TSRDAdapter(h5_path="my_scenario.h5")
    pdw = adapter.to_pdw_stream()

Schema
------
The writer produces the same groups as a real TSRD H5 file:

    /metadata/
        feature_names  (dataset, 5 strings: ToA, Frequency, PulseWidth, AoA, Amplitude)
        receiver/     (group)
            scan_mode  (attribute: "scanning" or "stare")
            start_position_km  (dataset, 2 floats: x, y in km)
            dwell_centres_mhz  (dataset, n_bands floats, Scan Mode only)
        transmitters/ (group)
            {i}/       (group per emitter, i = 0..N-1)
                freq_mode           (attribute)
                function            (attribute)
                start_time_s       (attribute)
                pri_mode           (attribute)
                pri_us             (attribute)
                pulse_width_us     (attribute)
                center_frequency_mhz (attribute)
                scan_rate_rpm      (attribute)
                power_w            (attribute)
                gain               (attribute)
                start_position_km  (dataset, 3 floats: x, y, z)
    /data     (dataset, shape (N, 5), float32)
    /labels   (dataset, shape (N, 1), int64)

Usage
-----
::

    from vyapti_simulator.tsrd.h5_writer import write_scenario_to_h5

    write_scenario_to_h5(
        path="my_scenario.h5",
        toa_s=np.array([0.001, 0.002, ...]),
        freq_hz=np.array([2.4e9, 2.4e9, ...]),
        pulse_width_s=np.array([1e-6, 1e-6, ...]),
        aoa_deg=np.array([45.0, 45.0, ...]),
        amplitude_dbm=np.array([-50.0, -50.0, ...]),
        emitter_ids=np.array([0, 0, ...]),
        emitter_metadata=[
            {"freq_mode": "fixed", "function": "search", ...},
            {"freq_mode": "agile", "function": "tracking", ...},
        ],
        scan_mode="scanning",
        receiver_position_km=(0.0, 0.0),
        dwell_centres_mhz=np.linspace(500, 17500, 36),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np

from .tsrd_adapter import EXPECTED_H5_FEATURE_NAMES

#: Maps from scan_mode string to the TSRD H5 schema string (capitalized).
_TSRD_RECEIVER_MODE: dict[str, str] = {
    "scanning": "Scanning",
    "stare": "Stare",
}


@dataclass
class H5ScenarioConfig:
    """
    Configuration for writing a synthetic Vyapti scenario to H5.

    All emitter-level fields in `emitters` are written verbatim to
    the H5's ``/metadata/transmitters`` group so that downstream
    readers (including ``TSRDAdapter`` and ``TSRDEnvironment``) can
    access the full provenance of each synthetic emitter.
    """
    scan_mode: str = "Scanning"
    """'scanning' or 'stare'. Passed as the H5 receiver scan_mode attribute."""
    receiver_position_km: tuple[float, float] = (0.0, 0.0)
    """(x, y) receiver position in km."""
    dwell_centres_mhz: Optional[np.ndarray] = None
    """Per-dwell tune centres in MHz for Scan Mode. Shape (n_bands,)."""
    emitter_metadata: List[Dict[str, Any]] = None
    """Per-emitter metadata dict. Written verbatim to /metadata/transmitters/{i}/."""

    def __post_init__(self):
        if self.emitter_metadata is None:
            self.emitter_metadata = []


def write_scenario_to_h5(
    path: Union[str, Path],
    toa_s: np.ndarray,
    freq_hz: np.ndarray,
    pulse_width_s: np.ndarray,
    aoa_deg: np.ndarray,
    amplitude_dbm: np.ndarray,
    emitter_ids: np.ndarray,
    *,
    config: Optional[H5ScenarioConfig] = None,
    overwrite: bool = False,
) -> str:
    """
    Write a synthetic Vyapti scenario to a TSRD-shaped HDF5 file.

    Parameters
    ----------
    path : str | Path
        Output file path. Will be created (or overwritten if ``overwrite=True``).
    toa_s : np.ndarray
        Time of arrival in seconds. Shape (N,).
    freq_hz : np.ndarray
        Pulse centre frequencies in Hz. Shape (N,).
    pulse_width_s : np.ndarray
        Pulse widths in seconds. Shape (N,).
    aoa_deg : np.ndarray
        Angles of arrival in degrees. Shape (N,).
    amplitude_dbm : np.ndarray
        Received amplitudes in dBm. Shape (N,).
    emitter_ids : np.ndarray
        Integer emitter labels. Shape (N,).
    config : H5ScenarioConfig, optional
        Scenario-level metadata (scan mode, receiver position, per-emitter info).
        Default: reasonable scanning defaults.
    overwrite : bool
        If True, overwrite an existing file. Default False (raises FileExistsError).

    Returns
    -------
    str
        The resolved file path.

    Notes
    -----
    The amplitude is written **as-is** (dBm), not converted to the TSRD
    relative ``amp_db`` scale. Downstream code that reads via
    ``TSRDAdapter`` applies the noise floor conversion::

        amp_db = amplitude_dbm - nominal_noise_floor_dbm

    So ensure ``nominal_noise_floor_dbm`` used by the reader matches
    the noise floor used when generating ``amplitude_dbm``.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {path}. Pass overwrite=True to replace.")

    if config is None:
        config = H5ScenarioConfig()
    if config.scan_mode.lower() not in _TSRD_RECEIVER_MODE:
        raise ValueError(
            f"scan_mode must be 'scanning' or 'stare' (case-insensitive), "
            f"got {config.scan_mode!r}"
        )

    n_pulses = len(toa_s)
    if not (len(freq_hz) == len(pulse_width_s) == len(aoa_deg)
            == len(amplitude_dbm) == len(emitter_ids) == n_pulses):
        raise ValueError(
            "All pulse arrays must have the same length. "
            f"Got: toa={n_pulses}, freq={len(freq_hz)}, "
            f"pw={len(pulse_width_s)}, aoa={len(aoa_deg)}, "
            f"amp={len(amplitude_dbm)}, ids={len(emitter_ids)}"
        )

    # --- Convert to TSRD units ----------------------------------------
    toa_us = (toa_s * 1e6).astype(np.float32)
    freq_mhz = (freq_hz / 1e6).astype(np.float32)
    pw_us = (pulse_width_s * 1e6).astype(np.float32)

    # --- Build data array (ToA, Freq, PW, AoA, Amplitude) ------------
    # Matches EXPECTED_H5_FEATURE_NAMES order exactly
    data = np.column_stack([
        toa_us.astype(np.float32),
        freq_mhz.astype(np.float32),
        pw_us.astype(np.float32),
        aoa_deg.astype(np.float32),
        amplitude_dbm.astype(np.float32),
    ])

    labels = np.atleast_2d(emitter_ids.astype(np.int64)).T

    # --- Write H5 ---------------------------------------------------
    with h5py.File(path, "w") as f:
        # /metadata/feature_names
        f.create_dataset(
            "metadata/feature_names",
            data=[n.encode("utf-8") for n in EXPECTED_H5_FEATURE_NAMES],
        )

        # /metadata/receiver
        rec_grp = f.create_group("metadata/receiver")
        h5_scan_mode = _TSRD_RECEIVER_MODE.get(config.scan_mode, config.scan_mode)
        rec_grp.attrs["scan_mode"] = h5_scan_mode.encode("utf-8")
        rec_grp.create_dataset(
            "start_position_km",
            data=np.asarray(config.receiver_position_km, dtype=np.float64),
        )
        if config.dwell_centres_mhz is not None and config.scan_mode == "scanning":
            rec_grp.create_dataset(
                "dwell_centres_mhz",
                data=np.asarray(config.dwell_centres_mhz, dtype=np.float32),
            )

        # /metadata/transmitters/{i}/
        transmitters_grp = f.create_group("metadata/transmitters")
        for i, meta in enumerate(config.emitter_metadata):
            emitter_grp = transmitters_grp.create_group(str(i))
            for key, value in meta.items():
                if isinstance(value, str):
                    emitter_grp.attrs[key] = value.encode("utf-8")
                elif isinstance(value, (int, float)):
                    emitter_grp.attrs[key] = value
                elif isinstance(value, (list, np.ndarray)):
                    emitter_grp.create_dataset(key, data=np.asarray(value, dtype=np.float64))
                else:
                    emitter_grp.attrs[key] = str(value).encode("utf-8")

        # /data and /labels
        f.create_dataset("data", data=data, dtype=np.float32)
        f.create_dataset("labels", data=labels, dtype=np.int64)

    return str(path.resolve())

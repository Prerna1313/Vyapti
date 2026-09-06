"""
vyapti_simulator.tsrd.tsrd_adapter
==================================

Adapter that reads a single TSRD H5 file via raw `h5py` and
exposes its 5-field PDW stream to the PS26055 simulator. This
is the file-level bridge between the canonical TSRD corpus
(Kaggle-gated) and the two PS26055 use cases:

  - **Scheduler experiments** (Scan Mode -> discretised band/slot
    observed occupancy).
  - **Deinterleaving experiments** (Stare Mode -> counterfactual
    oracle; the 5-field PDW stream is the only valid input).

Citation
--------

The adapter reads the five fields identified by the H5 metadata:
ToA, Frequency, PulseWidth, AoA, and Amplitude. All five source
values are preserved without transformation. AoA is retained as
supplied by TSRD; this adapter does not impose an angular-range
convention. Amplitude is retained as supplied by TSRD and is not
interpreted as received power, dBm, SNR, or a link-budget
quantity.

Data-admission modes
--------------------

The adapter has two admission modes. They are NOT interchangeable,
and there is NO automatic or opt-in fallback to synthetic data
on the real-data paths.

  * `REAL_TSRD`  -- points at the canonical TSRD corpus (Hugging
                    Face, gated, downloaded onto Kaggle or
                    equivalent). This is the only mode that
                    produces results claimable as "tested on
                    real TSRD".
  * `FIXTURE`    -- points at the small, hash-pinned fixture
                    under `tests/fixtures/tsrd/`. Used for unit
                    tests and for machines without Kaggle access.
                    The fixture manifest enforces the SHA-256
                    chain of custody.

Synthetic data (used by tests only) is constructed by a
SEPARATE class (`tests._builders.synthetic_pdw.SyntheticPDWBuilder`)
that the adapter cannot reach. There is no `synthetic` mode on
the adapter; there is no `allow_synthetic_fallback_on_missing`
kwarg. A missing file raises `DataUnavailableError`. A fixture
SHA-256 mismatch raises `DataIntegrityError`.

Receiver modes
--------------

The receiver mode is read from `/metadata/receiver/scan_mode` in
the H5. Valid values are `'Scanning'` (Scan Mode) and `'Stare'`
(Stare Mode).

  * Scan Mode -> `to_level1_observed_occupancy()` returns a
    per-emitter (band, slot) occupancy grid derived from the
    pulses actually present in the H5. Unobserved cells are
    NOT inactive truth; they are unobserved.
  * Stare Mode -> `to_level1_observed_occupancy()` raises
    `StareModeOracleError`. Stare Mode is the ORACLE input for
    the deferred `ScanPolicyOracle`, not the input for the
    scheduler's truth grid.

[DESIGN-DISCIPLINE] The adapter reads H5 files via raw `h5py`
only (not via the upstream `PulseTrain.load()` wrapper). This
keeps the adapter self-contained and decouples the simulator
from the upstream package's versioning. The fixture manifest
is the chain of custody, not the upstream loader.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from .tsrd_emitter import FREQ_MODE_TO_BEHAVIOR
from ..core.environment import (
    EmitterBehaviorType,
    EmitterConfig,
    SimulationConfig,
)


# =====================================================================
# PDW stream
# =====================================================================
@dataclass(frozen=True)
class PDWStream:
    """
    The canonical 6-field TSRD PDW stream.

    All five PDW fields are preserved without transformation.
    The H5 dtype (float32) is preserved. `emitter_id` is the
    H5's `labels` column cast to int64.
    """
    toa_us: np.ndarray      # float32, microseconds
    freq_mhz: np.ndarray    # float32, MHz
    pw_us: np.ndarray       # float32, microseconds
    aoa_deg: np.ndarray     # float32, degrees
    amp_db: np.ndarray      # float32, dB
    emitter_id: np.ndarray  # int64, label

    def __len__(self) -> int:
        return int(self.toa_us.size)


PDW_STREAM_FIELDS: tuple[str, ...] = (
    "toa_us", "freq_mhz", "pw_us", "aoa_deg", "amp_db", "emitter_id",
)

# The five H5 feature_names the adapter expects, in the order
# they appear as columns in `data`. This is the ONLY place the
# H5's column-order assumption is encoded.
EXPECTED_H5_FEATURE_NAMES: tuple[str, ...] = (
    "ToA", "Frequency", "PulseWidth", "AoA", "Amplitude",
)


# =====================================================================
# Errors
# =====================================================================
class DataUnavailableError(FileNotFoundError):
    """
    Raised when a real TSRD file is requested but is not on
    disk. This is a HARD failure. The user explicitly
    required: "do not silently fall back to synthetic data
    when real TSRD data is requested". Inherits from
    `FileNotFoundError` so existing OSError-style error
    handling still works.
    """


class DataIntegrityError(Exception):
    """
    Raised on H5 SHA-256 pin mismatch. Distinct from
    `core.data_loader.DataIntegrityError`; the adapter's
    integrity check is per-instance (`sha256_pin` kwarg) and
    does not consult the global manifest.
    """


class StareModeOracleError(ValueError):
    """
    Raised when code tries to discretise Stare Mode into a
    band/slot grid. Stare Mode is the ORACLE for counterfactual
    scan-policy evaluation, not the input source for the
    scheduler.
    """


class UnknownTSRDFieldError(ValueError):
    """
    Raised when a TSRD field value is not in the routing
    table (e.g. a `freq_mode` not in `FREQ_MODE_TO_BEHAVIOR`).
    No silent fallback to a default.
    """


class InsufficientDataError(ValueError):
    """
    Raised when an `EmitterConfig` field cannot be derived
    from the H5 AND was not supplied by the caller. The error
    message identifies the offending emitter's `tx_id` and
    the missing field name.
    """


class UnsupportedTSRDSchemaError(ValueError):
    """
    Raised when the H5's schema differs from what the adapter
    expects (e.g. `feature_names` is missing one of the 5
    expected fields, or `scan_mode` attr is not 'Scanning'
    or 'Stare').
    """


# =====================================================================
# Data admission modes
# =====================================================================
class TSRDDataMode(str, Enum):
    """Which TSRD data the adapter is allowed to read.

    These values are NOT interchangeable. The `REAL_TSRD` mode
    will refuse to load `FIXTURE` data and vice versa; the
    real-data paths will NOT silently fall back to synthetic
    data.
    """
    REAL_TSRD = "real_tsrd"
    FIXTURE = "fixture"


# =====================================================================
# Receiver modes (discovered from H5)
# =====================================================================
class TSRDReceiverMode(str, Enum):
    """Receiver mode, discovered from `/metadata/receiver/scan_mode`."""
    SCAN = "scan"     # H5 string: 'Scanning'
    STARE = "stare"   # H5 string: 'Stare'


_RECEIVER_MODE_FROM_H5: Dict[str, TSRDReceiverMode] = {
    "Scanning": TSRDReceiverMode.SCAN,
    "Stare": TSRDReceiverMode.STARE,
}


# =====================================================================
# Adapter
# =====================================================================
class TSRDAdapter:
    """
    Read a single TSRD H5 file via raw `h5py` and expose its
    PDW stream and (Scan Mode only) its observed-occupancy
    grid.

    See the module docstring for the data-admission modes,
    the citation, and the rationale for the two consumer
    paths (scheduler vs oracle).

    Parameters
    ----------
    h5_path : str | Path
        Path to a TSRD H5 file. Required; the adapter has no
        in-memory synthetic path.
    data_mode : TSRDDataMode
        Which admission mode this adapter is operating in.
        Must be one of `TSRDDataMode.REAL_TSRD` or
        `TSRDDataMode.FIXTURE`. These are not interchangeable.
    sha256_pin : str | None
        Optional SHA-256 to verify the file against. If
        provided, the adapter raises `DataIntegrityError` on
        mismatch. If `None`, no per-call pin check is
        performed; the fixture manifest performs its own
        check for the `FIXTURE` mode.
    simulation_config : SimulationConfig | None
        Required for `to_level1_observed_occupancy()` and for
        `to_emitter_configs()` (the latter needs the grid to
        map `freqs_mhz` to bands and `scan_rate_rpm` to slot
        period). Not required for `to_pdw_stream()`.
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        data_mode: TSRDDataMode,
        *,
        sha256_pin: Optional[str] = None,
        simulation_config: Optional[SimulationConfig] = None,
    ) -> None:
        self._data_mode = TSRDDataMode(data_mode)
        self._sha256_pin = sha256_pin.lower() if sha256_pin else None
        self._simulation_config = simulation_config
        self._h5_path: Optional[Path] = None
        self._receiver_mode: Optional[TSRDReceiverMode] = None
        self._feature_names: List[str] = []
        self._n_pulses: int = 0
        self._n_unique_labels: int = 0
        self._n_transmitter_configs: int = 0
        self._labels: Optional[np.ndarray] = None
        self._data: Optional[np.ndarray] = None
        self._h5_sha256: str = ""
        # Per-emitter start time (s) from H5 `power_config.start_time_s`.
        # Defaults to 0.0 (always-on) for emitters that lack the attribute.
        # Stored on the adapter so callers (Option A) can use it for
        # per-emitter arrival_slot wiring.
        self._transmitter_start_times_s: Dict[str, float] = {}

        if not isinstance(h5_path, (str, Path)):
            raise TypeError(
                f"h5_path must be str or Path, got {type(h5_path).__name__}"
            )
        path = Path(h5_path)

        if not path.is_file():
            raise DataUnavailableError(
                f"TSRD H5 file not found: {path}. "
                f"data_mode={self._data_mode.value!r} requires this "
                f"file. Refusing to silently substitute synthetic "
                f"data."
            )

        self._h5_path = path

        if self._sha256_pin is not None:
            actual = self._sha256_of_file(path)
            self._h5_sha256 = actual
            if actual.lower() != self._sha256_pin:
                raise DataIntegrityError(
                    f"TSRD H5 SHA-256 mismatch at {path}:\n"
                    f"  pinned   : {self._sha256_pin}\n"
                    f"  actual   : {actual}\n"
                    f"File was modified after pinning, or pin is wrong."
                )

        # Open the file and cache metadata. We do NOT hold the
        # H5 open long-term; the public API methods reopen as
        # needed so the adapter can outlive the file handle.
        with h5py.File(path, "r") as f:
            self._load_metadata(f)
            self._labels = f["labels"][:].flatten().astype(np.int64)
            self._data = f["data"][:].astype(np.float32)
            self._n_pulses = int(self._labels.size)
            unique_labels = np.unique(self._labels)
            self._n_unique_labels = int(unique_labels.size)
            self._n_transmitter_configs = int(
                len(list(f["metadata/transmitters"].keys()))
            )
        # Compute SHA-256 if not already (for provenance).
        if not self._h5_sha256:
            self._h5_sha256 = self._sha256_of_file(path)

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------
    @property
    def data_mode(self) -> TSRDDataMode:
        return self._data_mode

    @property
    def receiver_mode(self) -> TSRDReceiverMode:
        if self._receiver_mode is None:
            raise RuntimeError(
                "receiver_mode is undefined; the adapter was not "
                "fully initialised."
            )
        return self._receiver_mode

    @property
    def n_pulses(self) -> int:
        return self._n_pulses

    @property
    def n_unique_labels(self) -> int:
        return self._n_unique_labels

    @property
    def n_transmitter_configs(self) -> int:
        return self._n_transmitter_configs

    @property
    def transmitter_start_times_s(self) -> Dict[str, float]:
        """
        Per-emitter start time in seconds from H5 ``power_config.start_time_s``.
        Returns 0.0 for emitters that lack the attribute (e.g. fixtures
        that only have ``power_w`` and ``gain``).

        Only populated after ``to_emitter_configs()`` has been called,
        because that is when the H5 is read and the start times are
        extracted from each ``power_config`` group.
        """
        return dict(self._transmitter_start_times_s)

    @property
    def receiver_position_km(self) -> Optional[Tuple[float, float]]:
        """
        Receiver's (x, y) ground position in km from H5
        ``/metadata/receiver/start_position_km``. Returns None if
        the H5 does not carry the field.
        """
        return self._receiver_position_km

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def h5_sha256(self) -> str:
        return self._h5_sha256

    @property
    def dwell_centres_mhz(self) -> Optional[np.ndarray]:
        """
        The receiver's per-dwell tune frequencies in MHz, of length
        ``band_count``, as read from the H5's
        ``metadata/receiver/dwell_centres_mhz``. ``None`` for Stare
        Mode (the receiver uses a single fixed IF centre) or if the
        H5 does not carry the array.

        [TSRD-DERIVED] For Scan-Mode H5 files this is the ground truth
        of "which exact frequency the receiver is tuned to in each of
        the 36 dwells". The frequency-to-band discretisation in
        ``core.mapping`` should use this when available — the
        uniform ``(band + 0.5) * band_width`` formula is a fallback
        for the synthetic path, not the right answer for TSRD.
        """
        return None if self._dwell_centres_mhz is None else self._dwell_centres_mhz.copy()

    def apply_dwell_centres_to_config(self, cfg: SimulationConfig) -> SimulationConfig:
        """
        Return a copy of `cfg` with `dwell_centres_mhz` populated from
        this H5. The original `cfg` is not modified.

        If the H5 has no dwell centres (Stare Mode or unknown schema),
        the copy is identical to the input. If `cfg.band_count` differs
        from `len(dwell_centres_mhz)`, raises `ValueError` — a mismatch
        means the grid geometry does not match the recorded scan
        parameters and a fallback would silently distort the result.
        """
        if self._dwell_centres_mhz is None:
            return cfg
        if cfg.band_count != int(self._dwell_centres_mhz.size):
            raise ValueError(
                f"SimulationConfig.band_count={cfg.band_count} does not "
                f"match H5 dwell_centres_mhz length="
                f"{self._dwell_centres_mhz.size}. Refusing to silently "
                f"truncate or pad the tune grid."
            )
        import copy
        new_cfg = copy.copy(cfg)
        new_cfg.dwell_centres_mhz = self._dwell_centres_mhz.copy()
        return new_cfg

    @property
    def source_h5_path(self) -> Path:
        if self._h5_path is None:
            raise RuntimeError("source_h5_path is undefined.")
        return self._h5_path

    # ----------------------------------------------------------------
    # Hash helper
    # ----------------------------------------------------------------
    @staticmethod
    def _sha256_of_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    # ----------------------------------------------------------------
    # Metadata loading
    # ----------------------------------------------------------------
    def _load_metadata(self, f: h5py.File) -> None:
        # feature_names
        if "metadata/feature_names" not in f:
            raise UnsupportedTSRDSchemaError(
                "H5 is missing /metadata/feature_names. This is a "
                "TSRD schema break."
            )
        feat = f["metadata/feature_names"][:]
        self._feature_names = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in feat.tolist()
        ]
        if tuple(self._feature_names) != EXPECTED_H5_FEATURE_NAMES:
            raise UnsupportedTSRDSchemaError(
                f"H5 feature_names is {self._feature_names!r}; "
                f"expected exactly {list(EXPECTED_H5_FEATURE_NAMES)!r}. "
                f"This is a TSRD schema break."
            )

        # Receiver mode
        if "metadata/receiver" not in f:
            raise UnsupportedTSRDSchemaError(
                "H5 is missing /metadata/receiver. This is a TSRD "
                "schema break."
            )
        rec = f["metadata/receiver"]
        if "scan_mode" not in rec.attrs:
            raise UnsupportedTSRDSchemaError(
                "H5 /metadata/receiver is missing the 'scan_mode' "
                "attribute. This is a TSRD schema break."
            )
        scan_mode_str = rec.attrs["scan_mode"]
        if isinstance(scan_mode_str, bytes):
            scan_mode_str = scan_mode_str.decode("utf-8")
        if scan_mode_str not in _RECEIVER_MODE_FROM_H5:
            raise UnsupportedTSRDSchemaError(
                f"Unknown receiver scan_mode in H5: "
                f"{scan_mode_str!r}. Expected one of "
                f"{list(_RECEIVER_MODE_FROM_H5)}. This is a TSRD "
                f"schema break."
            )
        self._receiver_mode = _RECEIVER_MODE_FROM_H5[scan_mode_str]

        # Receiver ground position (x, y) in km. Used for path-loss
        # computation in TSRDEmitterSampler. Optional — None if the
        # H5 doesn't carry the field (older fixtures).
        self._receiver_position_km: Optional[Tuple[float, float]] = None
        if "start_position_km" in rec:
            pos = rec["start_position_km"][:]
            if len(pos) >= 2:
                self._receiver_position_km = (float(pos[0]), float(pos[1]))

        # Dwell centres (Scan-Mode only; Stare has a single fixed
        # IF centre). Stored on the adapter so callers can apply it
        # to their SimulationConfig. For Stare Mode we leave the
        # attribute as None — the receiver never sweeps.
        self._dwell_centres_mhz: Optional[np.ndarray] = None
        if "dwell_centres_mhz" in rec and self._receiver_mode == TSRDReceiverMode.SCAN:
            self._dwell_centres_mhz = np.asarray(
                rec["dwell_centres_mhz"][:], dtype=np.float32
            )
        elif (
            self._receiver_mode == TSRDReceiverMode.SCAN
            and "scan_mode" in rec.attrs
        ):
            # Some schema variants may store the centres differently;
            # warn-but-continue rather than raise so an unknown schema
            # doesn't kill an otherwise valid file.
            self._dwell_centres_mhz = None

    # ----------------------------------------------------------------
    # Public API: 5-field PDW stream
    # ----------------------------------------------------------------
    def to_pdw_stream(self) -> PDWStream:
        """
        Return the complete real TSRD PDW stream as a
        `PDWStream` of 6 numpy arrays:

            toa_us, freq_mhz, pw_us, aoa_deg, amp_db, emitter_id

        All five PDW values are preserved without
        transformation. The H5 dtype (float32) is preserved.
        `emitter_id` is the H5's `labels` column cast to
        int64.

        Both Scan and Stare modes return a `PDWStream`. Stare
        Mode consumers should go through `ScanPolicyOracle`
        for counterfactual evaluation; this method is the
        only place a Stare-Mode PDW stream can be obtained.
        """
        if self._data is None or self._labels is None:
            raise RuntimeError(
                "Adapter was not fully initialised; H5 data is "
                "missing."
            )
        # Column order from feature_names. We verified this is
        # exactly EXPECTED_H5_FEATURE_NAMES in `_load_metadata`.
        col_map = {name: i for i, name in enumerate(self._feature_names)}
        return PDWStream(
            toa_us=self._data[:, col_map["ToA"]].copy(),
            freq_mhz=self._data[:, col_map["Frequency"]].copy(),
            pw_us=self._data[:, col_map["PulseWidth"]].copy(),
            aoa_deg=self._data[:, col_map["AoA"]].copy(),
            amp_db=self._data[:, col_map["Amplitude"]].copy(),
            emitter_id=self._labels.copy(),
        )

    # ----------------------------------------------------------------
    # Public API: level-1 observed occupancy (Scan Mode only)
    # ----------------------------------------------------------------
    def to_observed_occupancy_from_pdw(self) -> np.ndarray:
        """
        Return a (n_bands, n_slots) bool array of the
        observed occupancy derived DIRECTLY from this file's
        PDW stream (no `to_emitter_configs()` call, no
        period_slots derivation, no FREQ_MODE routing).

        occupancy[b, s] = True iff at least one pulse in the
        PDW stream has (centre frequency → band b) and
        (time-of-arrival → slot s).

        This is the path the full-corpus Kaggle run
        actually used (Commit 4, §10.3.1). It does NOT
        require `caller_overrides` and it does NOT require
        every transmitter to have a non-zero
        `scan_rate_rpm`. The full corpus has 250/250 Scan
        test files that are admitted by this path; the
        `to_emitter_configs()` path admits only 122/250
        because 128/250 files contain at least one
        transmitter with `scan_rate_rpm = 0.0` and
        `period_slots` cannot be derived without inventing
        a default.

        Stare Mode is allowed on this path: in Stare Mode
        the receiver sweeps are not present, but the
        per-pulse (freq, toa) pairs are still the
        canonical observed occupancy. Whether to consume
        Stare-mode occupancy is a downstream policy
        decision; this method does not refuse either mode.

        Returns
        -------
        np.ndarray
            Shape `(n_bands, n_slots)`, dtype `bool`.
            True for observed cells; False elsewhere.

        Raises
        ------
        ValueError
            If `simulation_config` was not supplied to the
            constructor.
        """
        if self._simulation_config is None:
            raise ValueError(
                "to_observed_occupancy_from_pdw() requires "
                "simulation_config. Pass it to the "
                "TSRDAdapter constructor."
            )
        return self._observed_occupancy_from_pdw(
            pdw=self.to_pdw_stream(),
            cfg=self._simulation_config,
        )

    def _observed_occupancy_from_pdw(
        self,
        pdw: PDWStream,
        cfg: SimulationConfig,
    ) -> np.ndarray:
        from ..core.mapping import frequency_to_band, seconds_to_slot
        n_bands = int(cfg.band_count)
        n_slots = int(cfg.time_slots)
        grid = np.zeros((n_bands, n_slots), dtype=bool)
        for i in range(pdw.emitter_id.size):
            band = int(frequency_to_band(float(pdw.freq_mhz[i]), cfg))
            slot = int(seconds_to_slot(float(pdw.toa_us[i]) * 1e-6, cfg))
            if 0 <= band < n_bands and 0 <= slot < n_slots:
                grid[band, slot] = True
        return grid

    def to_level1_observed_occupancy(self) -> np.ndarray:
        """
        Return a (n_unique_labels, n_bands, n_slots) bool
        array.

        grid[e, b, s] = True iff at least one pulse from
        emitter e in the H5 has (freq_mhz, toa_us) mapping
        to (band b, slot s).

        This is the OBSERVED Scan-mode occupancy. Cells that
        are False are unobserved, NOT inactive truth. Without
        an independent completeness proof, the simulator MUST
        NOT treat unobserved Scan cells as inactive truth.
        Callers that need inactive truth must use the
        synthetic truth generator.

        Raises
        ------
        StareModeOracleError
            If `receiver_mode == STARE`. Stare Mode has no
            discretised grid; use the ScanPolicyOracle.
        ValueError
            If `simulation_config` was not supplied to the
            constructor.
        """
        if self.receiver_mode == TSRDReceiverMode.STARE:
            raise StareModeOracleError(
                "to_level1_observed_occupancy() is not valid for "
                "Stare Mode. Stare Mode is the oracle; use "
                "ScanPolicyOracle for counterfactual scan-policy "
                "evaluation."
            )
        if self._simulation_config is None:
            raise ValueError(
                "to_level1_observed_occupancy() requires "
                "simulation_config. Pass it to the TSRDAdapter "
                "constructor."
            )
        return self._scan_mode_observed_occupancy(
            pdw=self.to_pdw_stream(),
            cfg=self._simulation_config,
        )

    def _scan_mode_observed_occupancy(
        self,
        pdw: PDWStream,
        cfg: SimulationConfig,
    ) -> np.ndarray:
        from ..core.mapping import frequency_to_band, seconds_to_slot
        n_bands = int(cfg.band_count)
        n_slots = int(cfg.time_slots)
        unique_labels = np.unique(pdw.emitter_id)
        n_emit = int(unique_labels.size)
        grid = np.zeros((n_emit, n_bands, n_slots), dtype=bool)
        label_to_idx = {int(lab): i for i, lab in enumerate(unique_labels)}
        for i in range(pdw.emitter_id.size):
            e_idx = label_to_idx[int(pdw.emitter_id[i])]
            band = int(frequency_to_band(float(pdw.freq_mhz[i]), cfg))
            slot = int(seconds_to_slot(float(pdw.toa_us[i]) * 1e-6, cfg))
            if 0 <= band < n_bands and 0 <= slot < n_slots:
                grid[e_idx, band, slot] = True
        return grid

    # ----------------------------------------------------------------
    # Public API: per-emitter configs (no invented defaults)
    # ----------------------------------------------------------------
    def to_emitter_configs(
        self,
        *,
        max_emitters: Optional[int] = None,
        visibility_fraction: Optional[float] = None,
        arrival_slot: Optional[int] = None,
        snr_db: Optional[float] = None,
    ) -> List[EmitterConfig]:
        """
        Build `EmitterConfig` objects from observed
        per-emitter H5 metadata.

        Per-emitter fields derived from the H5 (NEVER
        caller-overridable):

          * `emitter_id`   : from
                             `/metadata/transmitters/transmitters_X`
          * `behavior`     : `FREQ_MODE_TO_BEHAVIOR[freq_mode]`;
                             unknown mode raises
                             `UnknownTSRDFieldError`
          * `active_bands` : from `freqs_mhz[]` via
                             `frequency_to_band(f, cfg)`
          * `period_slots` : from `scan_rate_rpm` via
                             `round((60 / scan_rate_rpm) / slot_s)`;
                             `scan_rate_rpm <= 0` raises
                             `InsufficientDataError`
          * `provenance_notes`: chain-of-custody fields

        Per-emitter fields required from the caller (NOT in
        the H5; no silent defaults):

          * `visibility_fraction`
          * `arrival_slot`
          * `snr_db`

        Each of these is an optional keyword argument. If a
        value is not supplied AND is not derivable from the
        H5, the entire call raises `InsufficientDataError`
        before any `EmitterConfig` is constructed.

        Caller-supplied values are recorded in
        `provenance_notes` under `simulator_assumption.*`
        keys with the `[SIMULATOR-ASSUMPTION]` label.

        Atomicity
        ---------
        All active emitters are validated for required
        fields first; the function then constructs the list
        atomically. A failure on any emitter raises before
        any list element is returned.

        Parameters
        ----------
        max_emitters : int | None
            If set, clip the returned list to at most this
            many configs (after filtering silent emitters).
        visibility_fraction, arrival_slot, snr_db : optional
            Caller-supplied simulator assumptions. See above.
        """
        if self._h5_path is None:
            raise RuntimeError("Adapter was not fully initialised.")
        if self._simulation_config is None:
            raise ValueError(
                "to_emitter_configs() requires simulation_config "
                "to map freqs_mhz to bands and scan_rate_rpm to "
                "slot period. Pass it to the TSRDAdapter "
                "constructor."
            )
        cfg = self._simulation_config

        # 1. Gather active emitters and their H5 metadata.
        with h5py.File(self._h5_path, "r") as f:
            labels = f["labels"][:].flatten().astype(np.int64)
            tx_keys = sorted(
                [k for k in f["metadata/transmitters"].keys()],
                key=lambda k: int(k.split("_")[1]),
            )
            active_set = {
                int(l) for l, c in zip(*np.unique(labels, return_counts=True)) if c > 0
            }
            active_label_count: Dict[int, int] = {
                int(l): int(c)
                for l, c in zip(*np.unique(labels, return_counts=True))
                if c > 0
            }
            per_emitter_meta: Dict[int, Dict[str, Any]] = {}
            for k in tx_keys:
                tx_id = int(k.split("_")[1])
                tx = f["metadata/transmitters"][k]
                entry: Dict[str, Any] = {}
                entry["function"] = self._decode_attr(
                    tx.attrs.get("function", "")
                )
                if "frequency_config" in tx:
                    fc = tx["frequency_config"]
                    entry["freq_mode"] = self._decode_attr(
                        fc.attrs.get("freq_mode", "")
                    )
                    if "freqs_mhz" in fc:
                        entry["freqs_mhz"] = fc["freqs_mhz"][:].astype(
                            np.float64
                        ).tolist()
                if "scan_config" in tx:
                    sc = tx["scan_config"]
                    entry["scan_type"] = self._decode_attr(
                        sc.attrs.get("scan_type", "")
                    )
                    entry["scan_rate_rpm"] = float(
                        sc.attrs.get("scan_rate_rpm", 0.0)
                    )
                if "pri_config" in tx:
                    pc = tx["pri_config"]
                    entry["pri_mode"] = self._decode_attr(
                        pc.attrs.get("pri_mode", "")
                    )
                if "power_config" in tx:
                    pwr = tx["power_config"]
                    entry["power_w"] = float(pwr.attrs.get("power_w", 0.0))
                    entry["gain"] = float(pwr.attrs.get("gain", 0.0))
                    entry["start_time_s"] = float(
                        pwr.attrs.get("start_time_s", 0.0)
                    )
                    # Cache on the adapter for the `transmitter_start_times_s` property.
                    self._transmitter_start_times_s[str(k)] = entry["start_time_s"]
                per_emitter_meta[tx_id] = entry

        # 2. Validate ALL active emitters before any construction.
        #    Required from caller (not derivable from H5):
        if visibility_fraction is None:
            self._raise_insufficient(
                tx_id=None,
                field="visibility_fraction",
                reason=(
                    "TSRD H5 does not carry a per-emitter "
                    "visibility fraction. The caller must supply "
                    "this value to enable use in the synthetic "
                    "truth generator."
                ),
            )
        if arrival_slot is None:
            self._raise_insufficient(
                tx_id=None,
                field="arrival_slot",
                reason=(
                    "TSRD H5 does not carry a per-emitter "
                    "arrival_slot. The caller must supply this "
                    "value."
                ),
            )
        if snr_db is None:
            self._raise_insufficient(
                tx_id=None,
                field="snr_db",
                reason=(
                    "TSRD H5 does not carry a per-emitter "
                    "received-SNR scalar (FSPL depends on an "
                    "assumed range, not in the H5). The caller "
                    "must supply this value."
                ),
            )

        # 3. Build per-emitter (validate + route).
        active_ids_sorted = sorted(active_set)
        prepared: List[Dict[str, Any]] = []
        for tx_id in active_ids_sorted:
            meta = per_emitter_meta.get(tx_id, {})
            freq_mode = str(meta.get("freq_mode", ""))
            if freq_mode not in FREQ_MODE_TO_BEHAVIOR:
                raise UnknownTSRDFieldError(
                    f"TSRD emitter tx_id={tx_id} has freq_mode="
                    f"{freq_mode!r}, which is not in "
                    f"FREQ_MODE_TO_BEHAVIOR. No silent fallback to "
                    f"CONTINUOUS_FIXED. Known modes: "
                    f"{sorted(FREQ_MODE_TO_BEHAVIOR.keys())}."
                )
            behavior = FREQ_MODE_TO_BEHAVIOR[freq_mode]

            freqs = meta.get("freqs_mhz", [])
            if not freqs:
                self._raise_insufficient(
                    tx_id=tx_id,
                    field="active_bands",
                    reason=(
                        f"TSRD emitter tx_id={tx_id} has empty "
                        f"freqs_mhz[]; cannot derive active_bands."
                    ),
                )
            active_bands = self._freqs_to_bands(freqs, cfg)
            if not active_bands:
                self._raise_insufficient(
                    tx_id=tx_id,
                    field="active_bands",
                    reason=(
                        f"TSRD emitter tx_id={tx_id} mapped to "
                        f"no bands via frequency_to_band."
                    ),
                )

            scan_rpm = float(meta.get("scan_rate_rpm", 0.0))
            if scan_rpm <= 0:
                self._raise_insufficient(
                    tx_id=tx_id,
                    field="period_slots",
                    reason=(
                        f"TSRD emitter tx_id={tx_id} has "
                        f"scan_rate_rpm={scan_rpm}; cannot derive "
                        f"period_slots (would require a non-zero "
                        f"scan rate)."
                    ),
                )
            period_s = 60.0 / scan_rpm
            slot_s = cfg.slot_duration_s()
            period_slots = max(1, int(round(period_s / slot_s)))

            provenance = self._build_provenance(
                tx_id=tx_id,
                meta=meta,
                label_count=active_label_count.get(tx_id, 0),
            )
            # Caller-supplied values are recorded as
            # simulator assumptions, NOT as TSRD-derived.
            # When the caller did not supply arrival_slot, use the
            # per-emitter start_time_s from H5 power_config (real TSRD)
            # or fall back to 0. Record the source in provenance.
            if arrival_slot is not None:
                effective_arrival = int(arrival_slot)
                self._attach_simulator_assumption(
                    provenance, "arrival_slot", arrival_slot,
                    "TSRD Scan fixture does not carry per-emitter "
                    "arrival_slot; caller supplied this value.",
                )
            else:
                # Infer from H5 power_config.start_time_s if present.
                start_time_s = float(meta.get("start_time_s", 0.0))
                slot_s = cfg.slot_duration_s()
                effective_arrival = max(0, int(start_time_s / slot_s)) if slot_s > 0 else 0
                if start_time_s > 0:
                    provenance["tsrd_start_time_s"] = start_time_s
                    provenance["tsrd_inferred_arrival_slot"] = effective_arrival
            self._attach_simulator_assumption(
                provenance, "visibility_fraction", visibility_fraction,
                "TSRD Scan fixture does not carry per-emitter "
                "visibility; caller supplied this value.",
            )
            self._attach_simulator_assumption(
                provenance, "snr_db", snr_db,
                "TSRD Scan fixture does not carry a per-emitter "
                "received-SNR scalar (FSPL depends on assumed "
                "range, not in the H5); caller supplied this value.",
            )

            prepared.append({
                "emitter_id": tx_id,
                "behavior": behavior,
                "active_bands": active_bands,
                "period_slots": period_slots,
                "visibility_fraction": float(visibility_fraction),
                "arrival_slot": effective_arrival,
                "snr_db": float(snr_db),
                "provenance_notes": provenance,
            })

        # 4. Build the EmitterConfig list atomically.
        configs: List[EmitterConfig] = [
            EmitterConfig(**p) for p in prepared
        ]
        if max_emitters is not None:
            configs = configs[: int(max_emitters)]
        return configs

    def infer_arrival_slots_from_pdw(self) -> Dict[int, int]:
        """
        Infer per-emitter arrival slot from the PDW stream.

        For each unique emitter_id, finds the earliest ToA and
        converts it to a slot index using ``SimulationConfig.slot_duration_s()``.

        Useful when the H5 lacks ``power_config.start_time_s`` (e.g.
        fixtures) but the PDW stream itself encodes the dynamics.

        Returns
        -------
        dict[int, int]
            ``{emitter_id: arrival_slot}`` for every emitter with
            at least one pulse.
        """
        if self._simulation_config is None:
            raise ValueError(
                "infer_arrival_slots_from_pdw() requires "
                "simulation_config to compute slot indices."
            )
        if self._labels is None or self._data is None:
            raise RuntimeError(
                "PDW stream not loaded; call to_pdw_stream() first "
                "or use a constructed adapter."
            )
        if self._n_pulses == 0:
            return {}
        slot_s = self._simulation_config.slot_duration_s()
        if slot_s <= 0:
            return {}
        toa_us = self._data[:, 0]  # column 0 = ToA
        result: Dict[int, int] = {}
        for label in np.unique(self._labels):
            mask = self._labels == label
            if not mask.any():
                continue
            first_toa_us = float(toa_us[mask].min())
            slot = max(0, int(first_toa_us * 1e-6 / slot_s))
            result[int(label)] = slot
        return result

    def infer_arrival_slots_from_h5(self) -> Dict[int, int]:
        """
        Infer per-emitter arrival slot from H5 ``power_config.start_time_s``.

        Falls back to 0 for emitters that lack the attribute. Only
        populated after ``to_emitter_configs()`` has been called.

        Returns
        -------
        dict[int, int]
            ``{emitter_id: arrival_slot}``.
        """
        if self._simulation_config is None:
            raise ValueError(
                "infer_arrival_slots_from_h5() requires simulation_config."
            )
        if not self._transmitter_start_times_s:
            return {}
        slot_s = self._simulation_config.slot_duration_s()
        result: Dict[int, int] = {}
        for tx_key, start_s in self._transmitter_start_times_s.items():
            tx_id = int(tx_key.split("_")[1])
            slot = max(0, int(start_s / slot_s)) if slot_s > 0 else 0
            result[tx_id] = slot
        return result

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    @staticmethod
    def _decode_attr(v: Any) -> Any:
        if isinstance(v, bytes):
            return v.decode("utf-8")
        if isinstance(v, np.generic):
            return v.item()
        return v

    @staticmethod
    def _raise_insufficient(
        *, tx_id: Optional[int], field: str, reason: str
    ) -> None:
        who = f"tx_id={tx_id}" if tx_id is not None else "adapter"
        raise InsufficientDataError(
            f"{who} cannot construct EmitterConfig: missing "
            f"{field!r}. {reason}"
        )

    def _freqs_to_bands(
        self, freqs: List[float], cfg: SimulationConfig
    ) -> List[int]:
        if not freqs:
            return []
        from ..core.mapping import frequency_to_band
        out: List[int] = []
        for f in freqs:
            b = int(frequency_to_band(float(f), cfg))
            if 0 <= b < cfg.band_count and b not in out:
                out.append(b)
        return sorted(out)

    def _build_provenance(
        self,
        *,
        tx_id: int,
        meta: Dict[str, Any],
        label_count: int,
    ) -> Dict[str, Any]:
        return {
            "tsrd_provenance_label": "[TSRD-DERIVED]",
            "tsrd_source_tx_id": int(tx_id),
            "tsrd_source_h5_path": str(self._h5_path),
            "tsrd_source_h5_sha256": self._h5_sha256,
            "tsrd_freq_mode": str(meta.get("freq_mode", "")),
            "tsrd_function": str(meta.get("function", "")),
            "tsrd_scan_type": str(meta.get("scan_type", "")),
            "tsrd_scan_rate_rpm": float(meta.get("scan_rate_rpm", 0.0)),
            "tsrd_pri_mode": str(meta.get("pri_mode", "")),
            "tsrd_n_freqs": int(len(meta.get("freqs_mhz", []))),
            "tsrd_label_count": int(label_count),
            "tsrd_dataset_slug": (
                "alan-turing-institute/turing-synthetic-radar-dataset"
            ),
            "tsrd_dataset_revision": (
                "68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51"
            ),
            "tsrd_github_repo_slug": (
                "alan-turing-institute/turing-deinterleaving-challenge"
            ),
            "tsrd_github_commit_hash": (
                "51a69e5ec950deeda69d17f26ebf958d7e3dce3e"
            ),
            "tsrd_feature_names": list(self._feature_names),
            "tsrd_n_pulses": int(self._n_pulses),
            "tsrd_n_unique_labels": int(self._n_unique_labels),
            "tsrd_n_transmitter_configs": int(self._n_transmitter_configs),
            "tsrd_receiver_mode": self.receiver_mode.value,
            "tsrd_data_mode": self._data_mode.value,
        }

    @staticmethod
    def _attach_simulator_assumption(
        provenance: Dict[str, Any],
        key: str,
        value: Any,
        rationale: str,
    ) -> None:
        provenance[f"simulator_assumption.{key}"] = {
            "label": "[SIMULATOR-ASSUMPTION]",
            "value": value,
            "rationale": rationale,
        }

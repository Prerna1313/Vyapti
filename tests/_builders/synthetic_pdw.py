"""
tests._builders.synthetic_pdw
==============================

`PDWStream` namedtuple and `SyntheticPDWBuilder` for unit tests.

These are the ONLY way the test tree produces a 5-field PDW
stream without an H5 file. They are intentionally separated
from the production `tsrd` package so that the real-data code
path (`TSRDAdapter`) cannot accidentally substitute synthetic
data when real TSRD data is requested.

The canonical 6-field PDW stream:
    toa_us, freq_mhz, pw_us, aoa_deg, amp_db, emitter_id

is the same shape returned by `TSRDAdapter.to_pdw_stream()`.
Tests can swap between real and synthetic streams by changing
which builder produced the `PDWStream`; the consumer code does
not need to know.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class PDWStream(NamedTuple):
    """The canonical 6-field TSRD PDW stream.

    All five PDW fields are preserved without transformation.
    The H5 dtype (float32) is preserved by the real-data path.
    `emitter_id` is the H5's `labels` column cast to int64.
    """
    toa_us: np.ndarray      # float32, microseconds
    freq_mhz: np.ndarray    # float32, MHz
    pw_us: np.ndarray       # float32, microseconds
    aoa_deg: np.ndarray     # float32, degrees
    amp_db: np.ndarray      # float32, dB
    emitter_id: np.ndarray  # int64, label


# Canonical field order, exposed as a constant so tests can
# assert against it without duplicating the list.
PDW_STREAM_FIELDS: tuple[str, ...] = (
    "toa_us", "freq_mhz", "pw_us", "aoa_deg", "amp_db", "emitter_id",
)


class SyntheticPDWBuilder:
    """
    Construct a synthetic `PDWStream` for unit tests.

    The constructor does NOT take an H5 path. The methods return
    `PDWStream` namedtuples directly. The draw is deterministic
    given a seed.

    NOT used by:
      * production code
      * any code path that touches a TSRD H5 file
    """

    def __init__(self, *, seed: int, n_pulses: int = 256) -> None:
        if n_pulses <= 0:
            raise ValueError(f"n_pulses must be positive, got {n_pulses}")
        self._seed = int(seed)
        self._n_pulses = int(n_pulses)

    def build(self) -> PDWStream:
        """Return a `PDWStream` of shape (n_pulses,) per field.

        Drawn from a deterministic stream
        (`np.random.default_rng(self._seed)`).

        Field units match the H5's documented column units
        (toa_us=us, freq_mhz=MHz, pw_us=us, aoa_deg=deg,
        amp_db=dB). Ranges are wide enough to exercise the
        full 0-18000 MHz band; they are NOT claims about TSRD
        data ranges.
        """
        rng = np.random.default_rng(self._seed)
        n = self._n_pulses
        # ToA: sort within a 30 s mission (in us).
        toa = np.sort(
            rng.uniform(0.0, 30.0, size=n).astype(np.float32) * 1e6
        )
        # Frequency: uniform over 500-18000 MHz.
        freq = rng.uniform(500.0, 18000.0, size=n).astype(np.float32)
        # Pulse width: 0.5-50 us.
        pw = rng.uniform(0.5, 50.0, size=n).astype(np.float32)
        # AoA: full [-180, +180] range; preserved as supplied.
        aoa = rng.uniform(-180.0, 180.0, size=n).astype(np.float32)
        # Amplitude: wide range; not interpreted.
        amp = rng.uniform(-200.0, 200.0, size=n).astype(np.float32)
        # Emitter id: a small set of integer labels.
        eid = rng.integers(0, 5, size=n).astype(np.int64)
        return PDWStream(
            toa_us=toa,
            freq_mhz=freq,
            pw_us=pw,
            aoa_deg=aoa,
            amp_db=amp,
            emitter_id=eid,
        )

    def build_with_labels(
        self, *, labels: np.ndarray
    ) -> PDWStream:
        """Return a `PDWStream` with caller-supplied emitter ids.

        Useful for tests that need to assert specific
        emitter-id values in the returned stream. Other
        fields are drawn from the same deterministic stream
        as `build()`.
        """
        if labels.size != self._n_pulses:
            raise ValueError(
                f"labels.size ({labels.size}) must equal "
                f"n_pulses ({self._n_pulses})"
            )
        s = self.build()
        return s._replace(emitter_id=labels.astype(np.int64))

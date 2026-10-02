"""Shared internal PDW stream, including audit truth; never a policy observation."""
from dataclasses import dataclass
import numpy as np

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

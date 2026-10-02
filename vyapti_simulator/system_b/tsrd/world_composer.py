"""Generator-only composition of complete recorded emitter realizations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

import numpy as np

GENERATOR_VERSION = "emitter_recombined_v2"
DEFAULT_TIME_OFFSET_US = 1_000_000
T = TypeVar("T")


def source_emitter_metadata(source_file, local_label: int) -> dict:
    """Read only explicitly recorded attributes for the matching transmitter.

    TSRD stores transmitter labels in transmitters_<local_label>. Its `function`
    is retained verbatim as a source type identifier, never inferred from PDWs.
    """
    import h5py
    with h5py.File(source_file, "r") as source:
        node = source.get(f"metadata/transmitters/transmitters_{int(local_label)}")
        attrs = {} if node is None else node.attrs
        def value(key, fallback=None):
            raw = attrs.get(key, fallback)
            return None if raw is None else (raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
        return {"type_id": value("type_id", attrs.get("function")),
                "operating_mode": value("operating_mode")}


@dataclass(frozen=True, slots=True)
class EmitterRealization:
    """One source-labelled complete trace; metadata never enters observations.

    Unknown type/mode remain None: PDW labels alone cannot identify them.
    Columns are ToA(us), CF(MHz), PW(us), AoA(degrees), amplitude(dB).
    """
    pool_id: str
    source_config_id: str
    local_label: int
    source_file: str
    pdws: np.ndarray
    type_id: str | None = None
    operating_mode: str | None = None

    def __post_init__(self):
        data = np.asarray(self.pdws, dtype=np.float64)
        if (data.ndim != 2 or data.shape[1] != 5 or not len(data)
                or not np.all(np.isfinite(data)) or np.any(np.diff(data[:, 0]) < 0)):
            raise ValueError("Realization must contain a finite, sorted, complete five-column trace")
        # Immutable bytes backing also prevents callers re-enabling writes.
        object.__setattr__(self, "pdws", np.frombuffer(data.tobytes(), dtype=np.float64).reshape(data.shape))


class WorldComposer:
    """Sample distinct sources, then one realization per source, and re-phase.

    Full traces are retained, including events outside the receiver mission.
    No wrapping, cropping, resampling or synthetic frequency patterns occur.
    """
    def __init__(self, *, time_offset_us: int = DEFAULT_TIME_OFFSET_US):
        if isinstance(time_offset_us, bool) or int(time_offset_us) != time_offset_us or time_offset_us < 0:
            raise ValueError("time_offset_us must be a nonnegative integer")
        self.time_offset_us = int(time_offset_us)

    def select(self, candidates: Sequence[T], source_id: Callable[[T], str], *, seed: int, count: int) -> list[T]:
        groups: dict[str, list[T]] = {}
        for candidate in candidates:
            groups.setdefault(source_id(candidate), []).append(candidate)
        if isinstance(count, bool) or int(count) != count or not 1 <= count <= len(groups):
            raise ValueError("emitter_count exceeds distinct source configs or is invalid")
        rng = np.random.default_rng(int(seed))
        keys = sorted(groups)
        chosen = rng.choice(len(keys), size=int(count), replace=False)
        return [groups[keys[int(i)]][int(rng.integers(len(groups[keys[int(i)]])))] for i in chosen]

    def compose(self, realizations: Sequence[EmitterRealization], *, seed: int):
        if not realizations or len({r.source_config_id for r in realizations}) != len(realizations):
            raise ValueError("A world requires at most one emitter per source_config_id")
        if len({r.pool_id for r in realizations}) != len(realizations):
            raise ValueError("Duplicate pool realization")
        # A separate stream makes offsets independent of loading/group traversal.
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x54535244]))
        arrays, labels, sources = [], [], []
        for world_id, realization in enumerate(realizations):
            offset = int(rng.integers(-self.time_offset_us, self.time_offset_us + 1))
            data = realization.pdws.copy()
            data[:, 0] += offset
            arrays.append(data)
            labels.append(np.full(len(data), world_id, dtype=np.int64))
            sources.append({"world_emitter_id": world_id, "pool_id": realization.pool_id,
                            "source_config_id": realization.source_config_id,
                            "source_label": realization.local_label, "source_file": realization.source_file,
                            "type_id": realization.type_id, "operating_mode": realization.operating_mode,
                            "recorded_pulse_count": len(data), "time_offset_us": offset,
                            "generator_version": GENERATOR_VERSION})
        data, world_labels = np.concatenate(arrays), np.concatenate(labels)
        order = np.argsort(data[:, 0], kind="stable")
        return data[order], world_labels[order], sources

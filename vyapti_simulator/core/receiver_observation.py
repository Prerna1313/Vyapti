"""Closed, immutable receiver measurements shared by TSRD schedulers.

Mapping access preserves existing policy syntax. Conversion to ordinary dicts
is reserved for serialization and offline diagnostics. Nested measurements are
also frozen and slotted, so metadata cannot hide labels or future state.
"""

from __future__ import annotations

from collections.abc import Mapping, Iterator
from dataclasses import dataclass, fields, field
import math
from typing import Any


class _Record(Mapping[str, Any]):
    __slots__ = ()

    def __iter__(self) -> Iterator[str]:
        return (item.name for item in fields(self))

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __getitem__(self, key: str):
        if key not in tuple(self):
            raise KeyError(key)
        return getattr(self, key)

    def to_dict(self) -> dict:
        def plain(value):
            if isinstance(value, _Record):
                return value.to_dict()
            if isinstance(value, tuple):
                return [plain(item) for item in value]
            return value
        return {key: plain(self[key]) for key in self}


@dataclass(frozen=True, slots=True)
class ReceiverMetadata(_Record):
    dwell_time_ms: float
    retune_time_ms: float
    band_width_mhz: float | None = None
    noise_figure_db: float | None = None
    antenna_gain_db: tuple[float, ...] | None = None

    def __iter__(self):
        return (item.name for item in fields(self) if getattr(self, item.name) is not None)

    def __post_init__(self):
        for name in ("dwell_time_ms", "retune_time_ms", "band_width_mhz", "noise_figure_db"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"Invalid receiver constant: {name}")
        if self.antenna_gain_db is not None:
            if (not isinstance(self.antenna_gain_db, tuple)
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                           for v in self.antenna_gain_db)):
                raise ValueError("Antenna gains must be an immutable finite tuple")


@dataclass(frozen=True, slots=True)
class MeasuredPDW(_Record):
    toa_offset_us: float
    frequency_mhz: float
    pulse_width_us: float
    aoa_deg: float
    amplitude_db: float

    def __post_init__(self):
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in self.values()):
            raise ValueError("Measured PDW fields must be finite numbers")


@dataclass(frozen=True, slots=True)
class ReceiverMeasurement(_Record):
    pulse_count: int
    pdws: tuple[MeasuredPDW, ...]

    def __post_init__(self):
        if (not isinstance(self.pulse_count, int) or isinstance(self.pulse_count, bool)
                or not isinstance(self.pdws, tuple)
                or not 1 <= len(self.pdws) <= self.pulse_count
                or any(type(pdw) is not MeasuredPDW for pdw in self.pdws)):
            raise ValueError("Invalid immutable PDW measurement")


@dataclass(frozen=True, slots=True)
class ReceiverObservation(_Record):
    time_slot: int
    selected_band: int
    hit: bool
    retune_cost_s: float
    dwell_elapsed_s: float
    receiver_metadata: ReceiverMetadata
    receiver_measurement: ReceiverMeasurement | None = None
    pdw_profile: bool = field(default=False, repr=False)
    truth_excluded: bool = field(default=True, init=False)
    emitter_identity_excluded: bool = field(default=True, init=False)
    future_state_excluded: bool = field(default=True, init=False)

    def __iter__(self):
        return (item.name for item in fields(self)
                if item.name != "pdw_profile"
                and (item.name != "receiver_measurement" or self.pdw_profile))

    def __post_init__(self):
        if type(self.receiver_metadata) is not ReceiverMetadata:
            raise TypeError("Receiver metadata must use the closed ReceiverMetadata type")
        if self.receiver_measurement is not None and type(self.receiver_measurement) is not ReceiverMeasurement:
            raise TypeError("Receiver measurement must use the closed ReceiverMeasurement type")
        if self.receiver_measurement is not None and not self.pdw_profile:
            raise ValueError("PDW measurements require the PDW receiver profile")
        if (type(self.time_slot) is not int or self.time_slot < 0
                or type(self.selected_band) is not int or self.selected_band < 0
                or type(self.hit) is not bool or type(self.pdw_profile) is not bool
                or not isinstance(self.retune_cost_s, (int, float))
                or not isinstance(self.dwell_elapsed_s, (int, float))
                or not math.isfinite(self.retune_cost_s) or self.retune_cost_s < 0
                or not math.isfinite(self.dwell_elapsed_s) or self.dwell_elapsed_s < 0):
            raise ValueError("Invalid receiver observation values")

    @classmethod
    def from_mapping(cls, observation: Mapping) -> ReceiverObservation:
        if type(observation) is cls:
            return observation
        allowed = {item.name for item in fields(cls)} - {"pdw_profile"}
        extra = set(observation) - allowed
        if extra:
            raise ValueError(f"Forbidden receiver observation fields: {sorted(extra)}")
        for marker in ("truth_excluded", "emitter_identity_excluded", "future_state_excluded"):
            if observation.get(marker, True) is not True:
                raise ValueError(f"Invalid receiver safety marker: {marker}")
        metadata = observation["receiver_metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError("Receiver metadata must be a mapping")
        metadata = dict(metadata)
        if "antenna_gain_db" in metadata:
            metadata["antenna_gain_db"] = tuple(metadata["antenna_gain_db"])
        try:
            typed_metadata = ReceiverMetadata(**metadata)
            measurement = observation.get("receiver_measurement")
            typed_measurement = None
            if measurement is not None:
                if not isinstance(measurement, Mapping) or set(measurement) != {"pulse_count", "pdws"}:
                    raise ValueError("Forbidden receiver measurement fields")
                typed_measurement = ReceiverMeasurement(
                    measurement["pulse_count"], tuple(MeasuredPDW(**dict(pdw)) for pdw in measurement["pdws"]))
            return cls(
                time_slot=observation["time_slot"], selected_band=observation["selected_band"],
                hit=observation["hit"], retune_cost_s=observation["retune_cost_s"],
                dwell_elapsed_s=observation["dwell_elapsed_s"],
                receiver_metadata=typed_metadata, receiver_measurement=typed_measurement,
                pdw_profile="receiver_measurement" in observation,
            )
        except TypeError as exc:
            raise ValueError("Undeclared or invalid nested receiver fields") from exc

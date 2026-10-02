"""Train-only TSRD emitter histories composed into recorded-PDW replay worlds."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .corpus_loader import iter_tsr_replay_pairs
from .tsrd_adapter import stare_pulse_occupancy
from .tsrd_environment import TSRDStareEnvironment
from .world_composer import EmitterRealization, WorldComposer, DEFAULT_TIME_OFFSET_US, source_emitter_metadata


@dataclass(frozen=True)
class EmitterContribution:
    stare_file: Path
    scan_file: Path | None
    source_label: int
    recorded_pulse_count: int


class TSRDTrainWorldPool:
    """Index complete per-emitter recorded histories from the train split only."""

    def __init__(
        self, corpus_root: str | Path, *, band_centres_mhz=None,
        passband_halfwidth_mhz: float | None = None,
        receiver_profile: str = "binary_v1", amplitude_midpoint_db: float = -90.0,
        amplitude_scale_db: float = 5.0, max_observed_pdws: int = 32,
        detection_probability: float = 1.0,
        false_alarm_probability: float = 0.0,
        retune_time_ms: float = 0.0,
        time_offset_us: int = DEFAULT_TIME_OFFSET_US,
    ):
        self.corpus_root = Path(corpus_root).resolve()
        self.composer = WorldComposer(time_offset_us=time_offset_us)
        self.explicit_centres_mhz = (
            None if band_centres_mhz is None else np.asarray(band_centres_mhz, dtype=np.float64)
        )
        self.receiver_profile_options = {
            "receiver_profile": receiver_profile,
            "amplitude_midpoint_db": amplitude_midpoint_db,
            "amplitude_scale_db": amplitude_scale_db,
            "max_observed_pdws": max_observed_pdws,
        }
        pairs = list(iter_tsr_replay_pairs(
            self.corpus_root, "train",
            allow_stare_only=self.explicit_centres_mhz is not None,
        ))
        if not pairs:
            raise ValueError("TSRD train split has no replay pairs")
        anchor = TSRDStareEnvironment.from_stare_mode(
            str(pairs[0][0]), str(pairs[0][1]) if pairs[0][1] is not None else None,
            band_centres_mhz=self.explicit_centres_mhz,
            passband_halfwidth_mhz=passband_halfwidth_mhz,
            **self.receiver_profile_options,
            detection_probability=detection_probability,
            false_alarm_probability=false_alarm_probability,
            retune_time_ms=retune_time_ms,
        )
        self.config = anchor.config
        self.centres_mhz, self.halfwidth_mhz = anchor.receiver_geometry
        self.contributions: list[EmitterContribution] = []
        for stare_file, scan_file in pairs:
            with h5py.File(stare_file, "r") as f:
                labels = np.asarray(f["labels"][:]).reshape(-1)
                unique, counts = np.unique(labels, return_counts=True)
            self.contributions.extend(
                EmitterContribution(stare_file, scan_file, int(label), int(count))
                for label, count in zip(unique, counts) if count > 0
            )
        if not self.contributions:
            raise ValueError("TSRD train split contains no recorded emitter histories")

    def sample_world(self, seed: int, emitter_count: int) -> tuple[TSRDStareEnvironment, list[dict]]:
        chosen = self.composer.select(self.contributions, lambda c: c.stare_file.stem,
                                      seed=seed, count=emitter_count)
        by_file: dict[Path, list[tuple[int, EmitterContribution]]] = defaultdict(list)
        for new_id, contribution in enumerate(chosen):
            by_file[contribution.stare_file].append((new_id, contribution))

        realizations = {}
        for stare_file, requested in by_file.items():
            scan_file = requested[0][1].scan_file
            with h5py.File(stare_file, "r") as f:
                collection_s = float(f["metadata/receiver"].attrs["collection_time_s"])
                halfwidth = float(f["metadata/receiver"].attrs["bandwith_mhz"])
                if scan_file is not None:
                    with h5py.File(scan_file, "r") as scan:
                        centres = np.asarray(scan["metadata/receiver/dwell_centres_mhz"][:])
                else:
                    centres = self.centres_mhz
                if (not np.isclose(collection_s, self.config.time_slots * self.config.slot_duration_s())
                        or not np.isclose(halfwidth, self.halfwidth_mhz)
                        or not np.array_equal(centres, self.centres_mhz)):
                    raise ValueError(f"Incompatible TSRD receiver geometry in {stare_file}")
                data = np.asarray(f["data"][:])
                labels = np.asarray(f["labels"][:]).reshape(-1)
            for new_id, contribution in requested:
                selected = labels == contribution.source_label
                if int(selected.sum()) != contribution.recorded_pulse_count:
                    raise ValueError("Source emitter pulse count changed")
                source_id = contribution.stare_file.stem
                realizations[new_id] = EmitterRealization(
                    f"train:{source_id}:emitter:{contribution.source_label}", source_id,
                    contribution.source_label, contribution.stare_file.relative_to(self.corpus_root).as_posix(), data[selected],
                    **source_emitter_metadata(stare_file, contribution.source_label))

        data, labels, sources = self.composer.compose(
            [realizations[i] for i in range(len(chosen))], seed=seed)
        occupancy = stare_pulse_occupancy(
            data, self.centres_mhz, self.halfwidth_mhz, self.config.time_slots
        )
        env = TSRDStareEnvironment(
            self.config.band_count, self.config.time_slots, occupancy, None,
            self.config, stare_data=data, stare_labels=labels,
            band_centres_mhz=self.centres_mhz,
            passband_halfwidth_mhz=self.halfwidth_mhz,
            **self.receiver_profile_options,
        )
        return env, sorted(sources, key=lambda source: source["world_emitter_id"])

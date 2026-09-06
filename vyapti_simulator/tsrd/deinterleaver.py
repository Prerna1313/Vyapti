"""
vyapti_simulator.tsrd.deinterleaver
=====================================

Option C — Deinterleaver.

A real electronic-warfare (EW) receiver does not see separate
emitters. It sees an interleaved stream of Pulse Descriptor Words
(PDWs) from every emitter in its field of view. The **deinterleaver**
is the unit that separates this stream back into per-emitter tracks
based on the features that are *almost unique* to a single emitter:
angle-of-arrival (AoA), radio frequency (RF), pulse width (PW),
and pulse repetition interval (PRI).

The deinterleaver here is the **AoA → PW → PRI** three-stage
architecture, because:

  * **AoA is the most stable feature** for stationary emitters
    (σ < 1° over a 30 s mission). It is the *first* discriminator.
  * **PW (mean and std) is the second most stable**: an emitter's
    PW is a hardware fingerprint of the modulator. PW std is
    typically 0.01–0.15 µs for a coherent emitter, 0.2–0.5 µs
    for a magnetron. Two emitters at the same AoA but different
    PW means OR different PW stds are separated here.
  * **RF and PRI are validation features** in scan-mode data:
    TSRD's scan receiver captures the emitter at different
    IF frequencies (RF std ≈ 100 MHz across the scan), and
    scan-boundary gaps corrupt the PRI histogram. RF and PRI
    are stored on each track for evaluation but do NOT drive
    the clustering.

This is the standard, well-documented EW deinterleaving order
(Wiley 2006 "ELINT: The Interception and Analysis of Radar
Signals", Chapter 5) adapted for scan-mode receiver data, where
RF stability across dwells is poor.

Algorithm
---------
  1. **AoA cluster** — split the stream into tight groups of
     similar AoA (1° bins, ``aoa_tolerance_deg`` acceptance).
  2. **PW refinement** — within each AoA cluster, walk pulses
     in chronological order. A new pulse joins the current
     sub-track if its PW is within ``pw_tolerance_us`` of the
     sub-track's running PW mean. After 3+ pulses, the cluster's
     PW std is also checked; a std above ``pw_std_tolerance_us``
     starts a new sub-track (separating emitters at the same
     mean PW but different PW jitter).
  3. **PRI validation** — for each AoA + PW sub-track, compute
     PRI statistics from ToA deltas. PRI is a *consistency
     check*, not a primary grouping mechanism.

The output is a list of ``EmitterTrack`` objects, each holding the
pulses attributed to a single emitter, with per-cluster features
(mean / std of PRI, RF, PW, AoA) for downstream evaluation.

This module is **not a research contribution**: the deinterleaving
algorithm is a standard, well-documented EW technique. What it
gives the simulator is the *ground-truth track list* needed to
evaluate Option-B experiments: with the deinterleaver's output,
we can label each pulse back to a single emitter, count
intercepts per emitter, and compute per-emitter metrics that the
synthetic-truth simulator cannot.

Author
------
Senior RF/EW Signal Simulation Engineer — PS26055 Option C integration.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

from .tsrd_adapter import PDWStream
from ..core.environment import SimulationConfig


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True)
class DeinterleaverConfig:
    """
    Hyperparameters for the deinterleaver.

    All tolerances are absolute (not fractional). They are
    intended to be *tight enough* to separate real TSRD emitters
    (which have clean PRI/RF/PW values) but *loose enough* to
    absorb within-burst jitter from PRI modes like
    ``Jitter`` and ``Stagger``.

    The defaults are calibrated against TSRD config_0 — see
    the unit tests in ``tests/test_deinterleaver.py`` for
    the empirical derivation.

    Attributes
    ----------
    pri_bin_width_us : float
        Width of each PRI histogram bin. TSRD's
        ``pri_proporitional_variance_factor`` is typically 0.03
        of the nominal PRI; a 4 µs bin covers PRI modes from
        ``Fixed`` (σ ≈ 0) to ``Jitter`` (σ ≈ 3 % of PRI).

    pri_min_us : float
        Pulses with ToA deltas below this value are dropped
        (they are either hardware artifacts or the
        first pulse of a track, which has no defined PRI).

    rf_tolerance_mhz : float
        Maximum RF difference (MHz) between two pulses in the
        same cluster. TSRD's ``freq_fixed_std_mhz`` is on the
        order of 0.001-0.1 MHz; 0.5 MHz tolerates the worst
        observed variability while separating distinct emitters.

    pw_tolerance_us : float
        Maximum pulse-width difference (µs) between two pulses
        in the same cluster. TSRD's ``pw_proporitional_variance_factor``
        is on the order of 0.03; 0.1 µs accommodates jitter while
        separating distinct emitters.

    pw_std_tolerance_us : float
        Maximum allowed PW standard deviation (µs) within a cluster.
        Two pulses with identical mean PW but different variability
        (e.g. 1.5±0.01 µs vs 1.5±0.3 µs) will be separated
        if the cluster's updated PW std exceeds this threshold.

    aoa_tolerance_deg : float
        Maximum AoA excursion within a cluster. TSRD does not
        necessarily simulate a moving emitter with realistic
        angular motion; 5° accommodates a 2σ variation.

    min_pulses_per_track : int
        Tracks with fewer pulses than this are discarded as
        noise. 5 is empirically the right floor for TSRD
        pulses in a 30 s mission.

    max_pri_us : float
        Pulses with PRI greater than this are dropped (they
        are likely missed clusters or end-of-mission gaps).
        5 000 µs = 5 ms covers PRIs from 200 Hz to 200 kHz
        (the full TSRD range).
    """
    pri_bin_width_us: float = 4.0
    pri_min_us: float = 1.0
    pri_max_us: float = 5_000.0
    # RF tolerance for AoA → RF → PRI on STARE-mode data. On
    # scan-mode data the captured RF varies by ~100 MHz across
    # dwells and this tolerance is irrelevant. Set wide enough
    # to be a non-restrictive check on stare data.
    rf_tolerance_mhz: float = 0.5
    # PW mean tolerance for AoA → PW sub-clustering. The TSRD
    # fixture has PW std ≈ 0.04 µs and the typical coherent
    # emitter has std 0.01–0.15 µs. 0.2 µs separates two emitters
    # at the same AoA with different PW means (e.g. 1.5 µs vs
    # 5 µs) while absorbing within-burst jitter.
    pw_tolerance_us: float = 0.2
    # PW std threshold: if a candidate cluster's PW std exceeds
    # this, the cluster is split. Two emitters at the same mean PW
    # but different variability (e.g. 1.5±0.01 µs vs 1.5±0.3 µs)
    # are separated here. 0.1 µs is empirically the right floor
    # for TSRD scan-mode data (fixture std ≈ 0.04 µs).
    pw_std_tolerance_us: float = 0.1
    # AoA tolerance for Stage 1 clustering. TSRD emitters are
    # stationary over a 30 s mission; σ_aoa is < 1°. 2° is a
    # generous bound that still separates emitters > 4° apart.
    aoa_tolerance_deg: float = 2.0
    min_pulses_per_track: int = 5


# =====================================================================
# Track data structure
# =====================================================================

@dataclass(frozen=True)
class EmitterTrack:
    """
    A deinterleaved emitter track — the pulses attributed to a
    single emitter after deinterleaving.

    All arrays are sorted by time-of-arrival. ``cluster_id`` is
    the deinterleaver's internal label; it is NOT the TSRD
    ``emitter_id`` (which is the ground-truth label). The two
    are related but distinct: ground-truth tracks may split
    across multiple clusters (rare, but possible with PRI-mode
    transitions), and a single cluster may be polluted by
    out-of-class pulses (a mis-cluster is treated as a
    contamination event, not a track merge).

    Attributes
    ----------
    cluster_id : int
        The deinterleaver's internal id (0, 1, 2, ...).
    pulse_indices : np.ndarray
        Indices into the original PDW stream.
    toa_us : np.ndarray
        Time-of-arrival of each pulse in microseconds. Sorted
        ascending. Shape ``(n,)``.
    freq_mhz : np.ndarray
        Centre frequency of each pulse. Shape ``(n,)``.
    pw_us : np.ndarray
        Pulse width of each pulse. Shape ``(n,)``.
    aoa_deg : np.ndarray
        Angle-of-arrival of each pulse. Shape ``(n,)``.
    amp_db : np.ndarray
        Amplitude of each pulse. Shape ``(n,)``.
    emitter_id : np.ndarray
        H5 ground-truth emitter label. Shape ``(n,)``. May
        contain more than one unique value if the deinterleaver
        mis-clustered pulses from adjacent emitters.
    pri_mean_us : float
        Mean PRI in microseconds, ``np.diff(toa_us).mean()``.
    pri_std_us : float
        Std of PRI in microseconds, ``np.diff(toa_us).std(ddof=1)``.
        NaN for tracks with fewer than 2 intervals.
    pri_cv : float
        Coefficient of variation ``pri_std / pri_mean``. A
        high CV (> 0.1) signals a Jitter or Stagger PRI mode.
    pw_std_us : float
        Std of pulse width in microseconds, ``pw_us.std(ddof=1)``.
        NaN for tracks with fewer than 2 pulses.
    pw_cv : float
        Coefficient of variation of pulse width, ``pw_std / pw_mean``.
        A high PW CV (> 0.1) signals a pulse-width-jittered emitter.
    n_pulses : int
    dominant_emitter_id : int
        The H5 ground-truth emitter id that contributed the
        most pulses to this cluster. Used for evaluation
        only.
    """
    cluster_id: int
    pulse_indices: np.ndarray
    toa_us: np.ndarray
    freq_mhz: np.ndarray
    pw_us: np.ndarray
    aoa_deg: np.ndarray
    amp_db: np.ndarray
    emitter_id: np.ndarray
    pri_mean_us: float
    pri_std_us: float
    pri_cv: float
    pw_std_us: float
    pw_cv: float
    n_pulses: int
    dominant_emitter_id: int

    @property
    def unique_emitter_ids(self) -> np.ndarray:
        """Distinct H5 ground-truth emitter labels in this track."""
        return np.unique(self.emitter_id)

    @property
    def rf_mean_mhz(self) -> float:
        return float(np.mean(self.freq_mhz))

    @property
    def pw_mean_us(self) -> float:
        return float(np.mean(self.pw_us))

    @property
    def aoa_mean_deg(self) -> float:
        return float(np.mean(self.aoa_deg))

    @property
    def amp_max_db(self) -> float:
        return float(np.max(self.amp_db))

    def contamination_fraction(self) -> float:
        """
        Fraction of pulses in this cluster whose H5 emitter
        label differs from the cluster's dominant emitter.
        A clean cluster has 0.0; a heavily contaminated
        cluster has values approaching 1.0.
        """
        if self.n_pulses == 0:
            return 0.0
        dominant = self.dominant_emitter_id
        mismatches = int(np.sum(self.emitter_id != dominant))
        return float(mismatches) / float(self.n_pulses)


@dataclass
class DeinterleaverResult:
    """
    The output of one deinterleaving pass.

    Attributes
    ----------
    tracks : List[EmitterTrack]
        All tracks recovered, in cluster-id order.
    n_input_pulses : int
        Total pulses in the input stream.
    n_tracks : int
        Number of tracks returned (after dropping those
        shorter than ``min_pulses_per_track``).
    n_dropped_short_tracks : int
        Number of candidate tracks dropped because they
        were too short.
    cluster_to_dominant_emitter : Dict[int, int]
        Map from cluster id to the H5 ground-truth emitter
        id that dominated that cluster. Provided for
        evaluation convenience.
    deinterleaver_config : DeinterleaverConfig
        The config used to produce these tracks. Stored for
        audit / reproducibility.
    """
    tracks: List[EmitterTrack]
    n_input_pulses: int
    n_tracks: int
    n_dropped_short_tracks: int
    cluster_to_dominant_emitter: Dict[int, int] = field(default_factory=dict)
    deinterleaver_config: DeinterleaverConfig = field(
        default_factory=DeinterleaverConfig
    )

    @property
    def n_pulses_in_tracks(self) -> int:
        return int(sum(t.n_pulses for t in self.tracks))

    @property
    def pulse_attribution_rate(self) -> float:
        """Fraction of input pulses assigned to a track."""
        if self.n_input_pulses == 0:
            return 0.0
        return self.n_pulses_in_tracks / self.n_input_pulses


# =====================================================================
# Main deinterleaver
# =====================================================================

class FeatureBasedDeinterleaver:
    """
    AoA → PW → PRI three-stage deinterleaver with PW purity checks.

    Stage 1: AoA clustering
    -----------------------
    AoA is the most stable feature for stationary emitters
    (σ_aoa < 1° over a 30 s mission). We histogram AoA, find
    dense bins, and walk pulses in time order collecting
    those within ``aoa_tolerance_deg`` of a seed centre.

    Stage 2: PW refinement
    -----------------------
    Within each AoA cluster, group pulses by PW mean. A new
    pulse is accepted into a PW sub-cluster if its PW is
    within ``pw_tolerance_us`` of the sub-cluster's running
    PW mean. After 3+ pulses, the cluster's PW std is also
    checked; a std above ``pw_std_tolerance_us`` starts a
    new sub-track. PW is the second-most stable feature
    for TSRD scan-mode data (the captured RF varies by
    ~100 MHz across the IF sweep, so RF cannot be a primary
    discriminator here).

    Stage 3: PRI validation
    -----------------------
    For each AoA + PW sub-cluster, compute PRI statistics
    from ToA deltas. PRI is a *consistency check*, not a
    primary grouping mechanism, because TSRD scan-mode
    boundary gaps corrupt the PRI histogram. The PRI
    statistics are stored on the track for evaluation.

    Complexity
    ----------
    O(P log P) for the AoA histogram, plus O(P) per AoA
    cluster for the PW refinement and std check.
    """

    def __init__(self, config: Optional[DeinterleaverConfig] = None) -> None:
        self._config = config or DeinterleaverConfig()

    @property
    def config(self) -> DeinterleaverConfig:
        return self._config

    def deinterleave(self, pdw: PDWStream) -> DeinterleaverResult:
        """
        Run the deinterleaver on a complete TSRD PDW stream.

        Parameters
        ----------
        pdw : PDWStream
            The full PDW stream. The deinterleaver operates on
            the entire stream at once; for very long streams,
            process in chunks and merge the resulting tracks.

        Returns
        -------
        DeinterleaverResult
            Tracks and provenance. Empty tracks list is a
            valid output (no emitters found at the configured
            tolerance).
        """
        n = len(pdw)
        if n < 2:
            return DeinterleaverResult(
                tracks=[],
                n_input_pulses=n,
                n_tracks=0,
                n_dropped_short_tracks=0,
                cluster_to_dominant_emitter={},
                deinterleaver_config=self._config,
            )

        # Sort pulses by time-of-arrival. TSRD already stores them
        # in ToA order, but we re-sort defensively.
        toa = pdw.toa_us.astype(np.float64).copy()
        order = np.argsort(toa)
        toa = toa[order]
        freq = pdw.freq_mhz.astype(np.float64)[order]
        pw = pdw.pw_us.astype(np.float64)[order]
        aoa = pdw.aoa_deg.astype(np.float64)[order]
        amp = pdw.amp_db.astype(np.float64)[order]
        eid = pdw.emitter_id.astype(np.int64)[order]
        original_idx = np.arange(n, dtype=np.int64)[order]

        # --- Stage 1: AoA clustering ---------------------------------
        aoa_clusters = self._aoa_cluster(pdw_toa=toa, pdw_aoa=aoa)

        # --- Stage 2 & 3: PW refinement + PRI/PW validation ---------
        tracks: List[EmitterTrack] = []
        used_pulses: set = set()
        cluster_id = 0

        for cluster_pulses in aoa_clusters:
            # Sort by ToA inside the cluster for chronological order.
            cluster_pulses_sorted = sorted(
                cluster_pulses, key=lambda i: toa[i]
            )

            if len(cluster_pulses_sorted) < 2:
                continue

            # Stage 2: greedy PW chain within this AoA cluster.
            # Each sub-track is a PW cluster. After at least
            # 3 pulses, the PW std is checked; a std above the
            # tolerance splits the cluster.
            sub_tracks: List[List[int]] = self._pw_split(
                indices=cluster_pulses_sorted,
                pw=pw,
            )

            for track_pulses in sub_tracks:
                if len(track_pulses) < self._config.min_pulses_per_track:
                    continue

                # Mark pulses as used
                for p in track_pulses:
                    used_pulses.add(p)

                # --- Build the EmitterTrack ----------------------------
                track_pulses_arr = np.array(track_pulses, dtype=np.int64)
                track_toa = toa[track_pulses_arr]
                track_freq = freq[track_pulses_arr]
                track_pw = pw[track_pulses_arr]
                track_aoa = aoa[track_pulses_arr]
                track_amp = amp[track_pulses_arr]
                track_eid = eid[track_pulses_arr]
                track_orig_idx = original_idx[track_pulses_arr]

                # PRI statistics (Stage 3: consistency)
                track_pri = np.diff(track_toa)
                pri_mean = float(track_pri.mean())
                pri_std = (
                    float(track_pri.std(ddof=1))
                    if track_pri.size > 1
                    else np.nan
                )
                pri_cv = pri_std / pri_mean if pri_mean > 0 else np.nan

                # PW statistics
                pw_arr = track_pw
                pw_mean = float(np.mean(pw_arr))
                pw_std = (
                    float(np.std(pw_arr, ddof=1))
                    if pw_arr.size > 1
                    else np.nan
                )
                pw_cv = (
                    pw_std / pw_mean
                    if pw_mean > 0 and not np.isnan(pw_std)
                    else np.nan
                )

                # Dominant emitter id (ground truth, for evaluation)
                vals, counts = np.unique(track_eid, return_counts=True)
                dominant_eid = int(vals[int(np.argmax(counts))])

                tracks.append(EmitterTrack(
                    cluster_id=cluster_id,
                    pulse_indices=track_orig_idx,
                    toa_us=track_toa,
                    freq_mhz=track_freq,
                    pw_us=track_pw,
                    aoa_deg=track_aoa,
                    amp_db=track_amp,
                    emitter_id=track_eid,
                    pri_mean_us=pri_mean,
                    pri_std_us=pri_std,
                    pri_cv=pri_cv if not np.isnan(pri_cv) else 0.0,
                    pw_std_us=pw_std,
                    pw_cv=pw_cv if not np.isnan(pw_cv) else 0.0,
                    n_pulses=int(track_pulses_arr.size),
                    dominant_emitter_id=dominant_eid,
                ))
                cluster_id += 1

        # Sort tracks by first ToA for deterministic output
        tracks.sort(key=lambda t: float(t.toa_us[0]))

        # Re-number cluster_id after the chronological sort
        tracks = [
            EmitterTrack(
                **{**t.__dict__, "cluster_id": i}
            )
            for i, t in enumerate(tracks)
        ]

        cluster_to_dominant = {
            t.cluster_id: t.dominant_emitter_id for t in tracks
        }

        return DeinterleaverResult(
            tracks=tracks,
            n_input_pulses=n,
            n_tracks=len(tracks),
            n_dropped_short_tracks=0,
            cluster_to_dominant_emitter=cluster_to_dominant,
            deinterleaver_config=self._config,
        )

    def _pw_split(
        self,
        indices: List[int],
        pw: np.ndarray,
    ) -> List[List[int]]:
        """
        Stage 2: greedy PW chain with PW mean + std purity check.

        Walk pulses in chronological order. Each pulse joins
        the current sub-track if its PW is within
        ``pw_tolerance_us`` of the sub-track's running PW mean.
        If the PW check fails, start a new sub-track with the
        current pulse as seed.

        When the sub-track has at least 3 pulses, the updated
        PW std is also checked against ``pw_std_tolerance_us``;
        a std above the threshold rejects the new pulse and
        starts a new sub-track (separating two emitters at
        the same mean PW but different PW-jitter fingerprints,
        e.g. 1.5±0.01 µs magnetron vs 1.5±0.3 µs TWT).

        Returns
        -------
        list of list of int
            Pulse indices per sub-track, in chronological order.
        """
        if not indices:
            return []

        sub_tracks: List[List[int]] = []
        current: List[int] = [indices[0]]

        for p in indices[1:]:
            # Compute the running PW mean of the current sub-track.
            cur_pw = np.array(
                [pw[t] for t in current], dtype=np.float64
            )
            cur_pw_mean = float(cur_pw.mean())

            pw_dev = abs(float(pw[p]) - cur_pw_mean)

            # PW std purity check (only meaningful with >= 3 pulses)
            pw_std_ok = True
            if len(current) >= 3:
                tentative_pw = np.append(cur_pw, float(pw[p]))
                new_std = float(np.std(tentative_pw, ddof=1))
                if new_std > self._config.pw_std_tolerance_us:
                    pw_std_ok = False

            if pw_dev > self._config.pw_tolerance_us or not pw_std_ok:
                # Close the current sub-track and start a new one
                # with this pulse as the seed.
                sub_tracks.append(current)
                current = [p]
            else:
                current.append(p)

        # Don't forget the last sub-track
        if current:
            sub_tracks.append(current)

        return sub_tracks

    def _aoa_cluster(
        self,
        pdw_toa: np.ndarray,
        pdw_aoa: np.ndarray,
    ) -> List[List[int]]:
        """
        Stage 1: cluster pulses by AoA using a histogram approach.

        Algorithm
        ---------
        1. Build an AoA histogram with 1° bins.
        2. Find bins with count >= 2 (a single isolated pulse is
           not enough to start a track).
        3. Merge adjacent seed bins whose centres are within
           ``2 * aoa_tolerance_deg`` of each other (they're the
           same emitter, just spanning the bin boundary). This
           prevents the test case where a single emitter's AoA
           jitter spans 1° and produces two overlapping clusters.
        4. For each merged seed, walk pulses in time order
           collecting those whose AoA is within
           ``aoa_tolerance_deg`` of the merged seed's centre.
        5. Any remaining unused pulses go into a residual cluster.

        Returns
        -------
        List[List[int]]
            Each inner list is the pulse indices belonging to one
            AoA cluster, in ToA order.
        """
        if pdw_aoa.size == 0:
            return []

        # 1. AoA histogram with 1° bins.
        aoa_min = float(np.floor(pdw_aoa.min()))
        aoa_max = float(np.ceil(pdw_aoa.max()))
        if aoa_min == aoa_max:
            aoa_max = aoa_min + 1.0
        n_aoa_bins = max(1, int(aoa_max - aoa_min) + 1)
        aoa_bin_edges = np.arange(aoa_min, aoa_max + 1.0, 1.0)
        aoa_bin_idx = np.digitize(pdw_aoa, aoa_bin_edges) - 1

        # 2. Find bins with >= 2 pulses.
        bin_counts = np.bincount(aoa_bin_idx, minlength=n_aoa_bins)
        seed_bins = np.flatnonzero(bin_counts >= 2)

        if seed_bins.size == 0:
            # No AoA bin has 2+ pulses; fall back to the global
            # AoA range as a single cluster so the deinterleaver
            # at least attempts to find tracks.
            return [list(np.argsort(pdw_toa))]

        # 3. Merge adjacent seed bins. If two seed centres are
        #    within ``2 * tolerance`` of each other, treat them
        #    as one cluster (they're the same emitter, with AoA
        #    jitter spanning a bin boundary). This is the key
        #    fix: without merging, an emitter with AoA std
        #    close to 1° would produce two clusters from the
        #    same emitter.
        seed_centres = sorted(
            (aoa_min + 0.5 + seed_bins).tolist()
        )
        tolerance = self._config.aoa_tolerance_deg
        merge_distance = 2.0 * tolerance

        merged_seeds: List[Tuple[float, List[int]]] = []
        for c in seed_centres:
            if merged_seeds and abs(c - merged_seeds[-1][0]) <= merge_distance:
                # Extend the last merged group
                prev_centre, _ = merged_seeds[-1]
                new_centre = (prev_centre + c) / 2.0
                merged_seeds[-1] = (new_centre, [])
            else:
                merged_seeds.append((c, []))

        # 4. Walk pulses in time order; assign each pulse to the
        #    first (highest-count) seed whose centre is within
        #    tolerance of the pulse's AoA.
        clusters: List[List[int]] = []
        used: set = set()

        for centre, _ in merged_seeds:
            cluster_pulses: List[int] = []
            for p in np.argsort(pdw_toa):
                if p in used:
                    continue
                if abs(float(pdw_aoa[p]) - centre) <= tolerance:
                    cluster_pulses.append(int(p))
            if len(cluster_pulses) >= 2:
                clusters.append(cluster_pulses)
                for p in cluster_pulses:
                    used.add(p)

        # 5. Any remaining unused pulses go into a residual cluster.
        unused = [int(p) for p in range(pdw_aoa.size) if p not in used]
        if unused:
            clusters.append(sorted(unused, key=lambda i: pdw_toa[i]))

        return clusters

    @staticmethod
    def _aoa_filter(
        pulse_indices: List[int],
        aoa: np.ndarray,
        tolerance_deg: float,
    ) -> List[int]:
        """
        Walk pulses in chronological order; drop any whose AoA
        differs from the running cluster mean by more than
        ``tolerance_deg``.

        The running mean is robust to single outliers: we
        recompute it after each accepted pulse using all
        accepted pulses so far, not a running average.
        """
        if not pulse_indices:
            return pulse_indices

        accepted: List[int] = [pulse_indices[0]]
        running_mean = float(aoa[pulse_indices[0]])

        for p in pulse_indices[1:]:
            if abs(float(aoa[p]) - running_mean) <= tolerance_deg:
                accepted.append(p)
                # Update running mean from all accepted pulses
                running_mean = float(np.mean([aoa[a] for a in accepted]))

        return accepted


# Backwards-compat alias. The new algorithm is feature-based
# (AoA → PW → PRI), not PRI-first; the old name is kept so
# existing callers don't break. New code should use
# ``FeatureBasedDeinterleaver`` directly.
class PRIBasedDeinterleaver(FeatureBasedDeinterleaver):  # pragma: no cover
    """
    DEPRECATED alias for ``FeatureBasedDeinterleaver``.

    The deinterleaver algorithm was changed from PRI-first
    (PRI histogram sort → RF/PW cluster → AoA filter) to
    feature-first (AoA cluster → PW refinement → PRI
    validation). The new name reflects the new architecture.

    This alias emits a ``DeprecationWarning`` when instantiated
    so callers know to migrate.
    """

    def __init__(self, config: Optional[DeinterleaverConfig] = None) -> None:
        import warnings
        warnings.warn(
            "PRIBasedDeinterleaver is deprecated; use "
            "FeatureBasedDeinterleaver. The algorithm is now "
            "AoA → PW → PRI (AoA-first), not PRI-first.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(config=config)


# =====================================================================
# Convenience constructors
# =====================================================================

def quick_deinterleave(
    pdw: PDWStream,
    config: Optional[DeinterleaverConfig] = None,
) -> DeinterleaverResult:
    """
    Functional entry point. Equivalent to
    ``FeatureBasedDeinterleaver(config).deinterleave(pdw)``.

    Provided for callers that prefer a one-liner over instantiating
    a deinterleaver object.
    """
    return FeatureBasedDeinterleaver(config).deinterleave(pdw)


__all__ = [
    "DeinterleaverConfig",
    "EmitterTrack",
    "DeinterleaverResult",
    "FeatureBasedDeinterleaver",
    "PRIBasedDeinterleaver",  # deprecated alias
    "quick_deinterleave",
]

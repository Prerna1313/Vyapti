from __future__ import annotations

"""
Vyapti VariBAD-style Meta-RSAC components.

This module deliberately separates four concerns:

1. Existing causal scheduler state from causal_harness.py.
2. A structured, population-level prior from the existing PO-RMAB artifacts.
3. VariBAD-style sequential latent inference q(z_t | history_<t) trained with
   a VAE-style transition/signal reconstruction objective and sequential KL.
4. A residual discrete SAC control layer conditioned on the latent regime state.

This is a research hybrid, not a verbatim implementation of a published
algorithm.  VariBAD contributes the causal variational task-inference idea;
Residual Policy Learning contributes correction over a useful controller;
and the discrete-SAC side uses an average-twin-Q target motivated by SDSAC.

V1 CHANGES
----------
- Stronger discovery-oriented reward weighting.
- Bayesian detector-surprise term normalized by the receiver false-alarm model.
- Rebalanced structured prior: less Whittle dominance, more UCB exploration.
- Stronger concentration constraint target.


PRIVILEGED DATA RULE
--------------------
Hidden truth, emitter IDs and meta-task descriptors are NEVER supplied to:
- policy inputs
- reward computation
- Bayesian state updates
- context inference during an episode
- validation/test action selection

Meta-task descriptors are used only by the TRAINER to construct different
training-world distributions from the frozen TRAIN-500 source pool.
"""

import copy
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_harness import (
    N_BANDS,
    N_BASE_SLOTS,
    BASE_SLOT_S,
    PD,
    PFA,
    RETUNE_TIME_MS,
    CausalSchedulerState,
)

EPS = 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# V1 STRUCTURED-PRIOR COEFFICIENTS
# ============================================================

# V0 used Whittle=1.00 and UCB=0.30.  V1 deliberately shifts part of the
# structured prior toward causal exploration because UCB was the stronger
# spectrum-discovery baseline in the matched validation protocol.
V1_WHITTLE_W = 0.75
V1_UCB_W = 0.50
V1_PERIODIC_W = 0.20
V1_STALENESS_W = 0.15
V1_AGILITY_W = 0.20
V1_SPATIAL_W = 0.15
V1_PRIORITY_W = 0.25

# V1 concentration target.  The cost remains normalized against a fair
# 1/N_BANDS allocation.
V1_CONCENTRATION_TARGET = 0.05


# ============================================================
# NUMERICAL HELPERS
# ============================================================


def std_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = float(x.mean())
    sd = float(x.std())
    if sd < 1e-8:
        return np.zeros_like(x)
    return (x - mu) / sd


def entropy_np(p: float | np.ndarray) -> np.ndarray:
    x = np.clip(
        np.asarray(p, dtype=np.float64),
        1e-8,
        1.0 - 1e-8,
    )
    return -(
        x * np.log2(x)
        + (1.0 - x) * np.log2(1.0 - x)
    )


def kl_diag_normal(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussian distributions."""
    return 0.5 * (
        logvar_p
        - logvar_q
        - 1.0
        + torch.exp(logvar_q - logvar_p)
        + (mu_p - mu_q).pow(2)
        / torch.exp(logvar_p)
    ).sum(dim=-1)


# ============================================================
# CAUSAL BELIEF SIMULATION
# ============================================================


def transition_posterior(
    state: CausalSchedulerState,
    band: int,
    y: bool,
    belief: np.ndarray,
) -> tuple[float, np.ndarray]:
    """One observation update followed by one Markov transition prediction."""
    band = int(band)
    p = float(
        np.clip(
            belief[band],
            1e-8,
            1.0 - 1e-8,
        )
    )

    if bool(y):
        like_active = float(PD)
        like_inactive = float(PFA)
    else:
        like_active = float(1.0 - PD)
        like_inactive = float(1.0 - PFA)

    den = (
        p * like_active
        + (1.0 - p) * like_inactive
    )

    post = float(
        np.clip(
            p * like_active / max(den, EPS),
            1e-8,
            1.0 - 1e-8,
        )
    )

    nxt = belief.copy()

    for b in range(N_BANDS):
        pb = post if b == band else float(belief[b])
        p01 = float(
            state.belief.transition[b, 0, 1]
        )
        p11 = float(
            state.belief.transition[b, 1, 1]
        )
        nxt[b] = np.clip(
            (1.0 - pb) * p01 + pb * p11,
            1e-8,
            1.0 - 1e-8,
        )

    return post, nxt


def causal_transition_rollout(
    state: CausalSchedulerState,
    action: int,
    positives: list[bool],
) -> tuple[list[float], list[float]]:
    """Compute reward-side belief transitions without mutating real state."""
    belief = np.asarray(
        state.belief.belief,
        dtype=np.float64,
    ).copy()

    pre: list[float] = []
    post: list[float] = []

    for y in positives:
        pre.append(
            float(
                np.clip(
                    belief[int(action)],
                    1e-8,
                    1.0 - 1e-8,
                )
            )
        )

        p_post, belief = transition_posterior(
            state,
            int(action),
            bool(y),
            belief,
        )

        post.append(float(p_post))

    return pre, post


# ============================================================
# ONLINE FREQUENCY-AGILITY CUE
# ============================================================


class FrequencyAgilityCue:
    """
    Observation-only transition cue.

    A transition is recorded between consecutive bands on which the receiver
    observed at least one positive.  No emitter IDs or hidden occupancy are
    consulted.
    """

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = float(smoothing)
        self.counts = np.full(
            (N_BANDS, N_BANDS),
            self.smoothing,
            dtype=np.float64,
        )
        self.last_positive_band = -1

    def reset(self) -> None:
        self.counts.fill(self.smoothing)
        self.last_positive_band = -1

    def observe(
        self,
        band: int,
        positives: list[bool],
    ) -> None:
        if not any(positives):
            return

        band = int(band)

        if self.last_positive_band >= 0:
            self.counts[
                self.last_positive_band,
                band,
            ] += 1.0

        self.last_positive_band = band

    def next_distribution(self) -> np.ndarray:
        if self.last_positive_band < 0:
            return np.full(
                N_BANDS,
                1.0 / N_BANDS,
                dtype=np.float64,
            )

        row = self.counts[
            self.last_positive_band
        ]

        return row / max(
            float(row.sum()),
            EPS,
        )


# ============================================================
# STRUCTURED PRIOR
# ============================================================


class StructuredPrior:
    """
    Loads the already-fitted population-level PO-RMAB artifacts.

    This prior contains population calibration, not current-world truth.
    """

    def __init__(
        self,
        prior_dir: Path,
        expected_fingerprint: str | None = None,
    ):
        prior_dir = Path(prior_dir)

        transition_file = (
            prior_dir / "pormab_transition.json"
        )
        tables_file = (
            prior_dir / "pormab_index_tables.npy"
        )
        diag_file = (
            prior_dir / "pormab_index_diagnostics.json"
        )

        for path in (
            transition_file,
            tables_file,
            diag_file,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing structured prior artifact: {path}"
                )

        payload = json.loads(
            transition_file.read_text(
                encoding="utf-8"
            )
        )

        if expected_fingerprint is not None:
            prior_fp = payload.get(
                "source_pool_fingerprint"
            )
            if prior_fp not in (
                None,
                expected_fingerprint,
            ):
                raise RuntimeError(
                    "PO-RMAB prior fingerprint mismatch: "
                    f"prior={prior_fp}, cache={expected_fingerprint}"
                )

        self.transition = np.asarray(
            payload["transition"],
            dtype=np.float64,
        )

        self.prior_active = np.asarray(
            payload["prior_active"],
            dtype=np.float64,
        )

        diagnostics = json.loads(
            diag_file.read_text(
                encoding="utf-8"
            )
        )

        self.belief_grid = np.asarray(
            diagnostics["belief_grid"],
            dtype=np.float64,
        )

        self.index_tables = np.load(
            tables_file
        )

        if self.transition.shape != (
            N_BANDS,
            2,
            2,
        ):
            raise RuntimeError(
                f"Bad transition shape: {self.transition.shape}"
            )

        if self.prior_active.shape != (
            N_BANDS,
        ):
            raise RuntimeError(
                f"Bad prior shape: {self.prior_active.shape}"
            )

        if self.index_tables.shape[0] != N_BANDS:
            raise RuntimeError(
                f"Bad Whittle table shape: {self.index_tables.shape}"
            )

    def whittle(
        self,
        belief: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            [
                np.interp(
                    np.clip(
                        float(belief[b]),
                        0.0,
                        1.0,
                    ),
                    self.belief_grid,
                    self.index_tables[b],
                )
                for b in range(N_BANDS)
            ],
            dtype=np.float64,
        )


# ============================================================
# CAUSAL RSAC STATE
# ============================================================


@dataclass
class RSACState:
    causal: CausalSchedulerState
    agility: FrequencyAgilityCue
    dwell_slots: int = 2

    @classmethod
    def create(
        cls,
        transition: np.ndarray,
        prior_active: np.ndarray,
        dwell_slots: int = 2,
    ) -> "RSACState":
        return cls(
            causal=CausalSchedulerState.create(
                transition,
                prior_active=prior_active,
            ),
            agility=FrequencyAgilityCue(),
            dwell_slots=int(dwell_slots),
        )

    def reset(
        self,
        prior_active: np.ndarray,
    ) -> None:
        if int(self.dwell_slots) <= 0:
            raise ValueError("dwell_slots must be positive")
        self.causal.reset(
            prior_active
        )
        self.agility.reset()

    @property
    def previous_action(self) -> int:
        return int(
            self.causal.previous_action
        )

    @property
    def previous_positive(self) -> bool:
        return bool(
            self.causal.previous_positive
        )

    def apply_observation(
        self,
        action: int,
        positives: list[bool],
        dwell_slots: int,
    ) -> None:
        # IMPORTANT:
        # causal_harness.CausalSchedulerState.step() does not commit the
        # Bayesian observation posterior.  It MUST be committed here first.
        self.causal.belief.step_observations(
            action,
            positives,
        )

        self.agility.observe(
            action,
            positives,
        )

        self.causal.step(
            action,
            dwell_slots,
            positives,
        )

    def features(
        self,
        prior: StructuredPrior,
    ) -> dict[str, np.ndarray]:
        f = self.causal.pre_action_features()

        belief = np.asarray(
            f["belief"],
            dtype=np.float64,
        )
        belief_entropy = np.asarray(
            f["belief_entropy"],
            dtype=np.float64,
        )
        staleness = np.asarray(
            f["staleness"],
            dtype=np.float64,
        )
        periodicity = np.asarray(
            f["periodicity"],
            dtype=np.float64,
        )
        periodic_confidence = np.asarray(
            f["periodicity_confidence"],
            dtype=np.float64,
        )
        visits = np.asarray(
            f["visit_count_norm"],
            dtype=np.float64,
        )

        agility = self.agility.next_distribution()

        # Explicit D2 cue.
        periodic_opportunity = (
            periodicity * periodic_confidence
        )

        # D1 currently uses the best causal phase/window proxy available in
        # the shared harness.  A learned spatial specialist can replace this
        # without changing the RSAC interface later.
        spatial_opportunity = periodic_opportunity.copy()

        # Operational sensing priority, not hostile-probability classification.
        priority = (
            0.55 * belief
            + 0.25 * belief_entropy
            + 0.20 * periodic_confidence
        )

        whittle = prior.whittle(
            belief
        )

        total_actions = max(
            int(self.causal.elapsed_base_slots),
            1,
        )

        counts = np.asarray(
            self.causal.visit_counts,
            dtype=np.float64,
        )

        ucb = np.sqrt(
            np.log(
                total_actions + 2.0
            )
            /
            np.maximum(
                counts + 1.0,
                1.0,
            )
        )

        # Population prior + causal exploration + prediction cues.
        base_score = (
            V1_WHITTLE_W * std_np(whittle)
            + V1_UCB_W * std_np(ucb)
            + V1_PERIODIC_W * std_np(periodic_opportunity)
            + V1_STALENESS_W * std_np(staleness)
            + V1_AGILITY_W * std_np(agility)
            + V1_SPATIAL_W * std_np(spatial_opportunity)
            + V1_PRIORITY_W * std_np(priority)
        )

        base_logits = np.clip(
            base_score,
            -5.0,
            5.0,
        ).astype(np.float32)

        prev_onehot = np.zeros(
            N_BANDS,
            dtype=np.float32,
        )

        if 0 <= self.previous_action < N_BANDS:
            prev_onehot[
                self.previous_action
            ] = 1.0

        elapsed_norm = np.float32(
            np.clip(
                self.causal.elapsed_base_slots
                / max(N_BASE_SLOTS, 1),
                0.0,
                1.0,
            )
        )

        switch_available = np.float32(
            self.previous_action >= 0
        )

        # Current benchmark action dwell is 2 base slots by default.
        # The feature is an accounting feature, not physical time simulation.
        retune_fraction = np.float32(
            (
                float(RETUNE_TIME_MS)
                / 1000.0
            )
            / max(
                BASE_SLOT_S * max(self.dwell_slots, 1),
                1e-9,
            )
        )

        observation = np.concatenate(
            [
                belief.astype(np.float32),
                belief_entropy.astype(np.float32),
                staleness.astype(np.float32),
                visits.astype(np.float32),
                periodicity.astype(np.float32),
                periodic_confidence.astype(np.float32),
                np.asarray(
                    f["period_norm"],
                    dtype=np.float32,
                ),
                agility.astype(np.float32),
                periodic_opportunity.astype(np.float32),
                spatial_opportunity.astype(np.float32),
                priority.astype(np.float32),
                ucb.astype(np.float32),
                whittle.astype(np.float32),
                base_logits,
                prev_onehot,
                np.asarray(
                    [
                        elapsed_norm,
                        float(self.previous_positive),
                        switch_available,
                        retune_fraction,
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )

        return {
            "observation": observation.astype(
                np.float32
            ),
            "base_logits": base_logits,
            "belief": belief.astype(
                np.float32
            ),
            "belief_entropy": belief_entropy.astype(
                np.float32
            ),
            "staleness": staleness.astype(
                np.float32
            ),
            "periodicity": periodicity.astype(
                np.float32
            ),
            "periodicity_confidence": periodic_confidence.astype(
                np.float32
            ),
            "periodic_opportunity": periodic_opportunity.astype(
                np.float32
            ),
            "spatial_opportunity": spatial_opportunity.astype(
                np.float32
            ),
            "agility": agility.astype(
                np.float32
            ),
            "priority": priority.astype(
                np.float32
            ),
            "ucb": ucb.astype(
                np.float32
            ),
            "whittle": whittle.astype(
                np.float32
            ),
        }

    def observation(
        self,
        prior: StructuredPrior,
    ) -> tuple[np.ndarray, np.ndarray]:
        f = self.features(
            prior
        )
        return (
            f["observation"],
            f["base_logits"],
        )


# ============================================================
# META-TRAINING TASK DESCRIPTORS
# ============================================================


@dataclass(frozen=True)
class EmitterDescriptor:
    index: int
    periodic_score: float
    agility_score: float
    spatial_score: float
    irregularity_score: float


def _descriptor_from_arrays(
    index: int,
    toa: np.ndarray,
    freq: np.ndarray,
    aoa: np.ndarray,
) -> EmitterDescriptor:
    toa = np.asarray(
        toa,
        dtype=np.float64,
    )
    freq = np.asarray(
        freq,
        dtype=np.float64,
    )
    aoa = np.asarray(
        aoa,
        dtype=np.float64,
    )

    if toa.size < 3:
        return EmitterDescriptor(
            index=index,
            periodic_score=0.0,
            agility_score=0.0,
            spatial_score=0.0,
            irregularity_score=1.0,
        )

    gaps = np.diff(toa)
    gaps = gaps[gaps > 0]

    gap_cv = (
        float(
            np.std(gaps)
            / max(
                float(np.mean(gaps)),
                EPS,
            )
        )
        if gaps.size
        else 1.0
    )

    periodic_score = float(
        np.exp(
            -min(gap_cv, 6.0)
        )
    )

    if freq.size > 1:
        fd = np.abs(
            np.diff(freq)
        )
        agility_score = float(
            np.clip(
                np.mean(
                    fd > 100.0
                )
                + np.ptp(freq)
                / 4000.0,
                0.0,
                1.0,
            )
        )
    else:
        agility_score = 0.0

    if aoa.size > 1:
        ad = np.abs(
            np.diff(aoa)
        )
        spatial_score = float(
            np.clip(
                np.mean(
                    ad > 5.0
                )
                + np.ptp(aoa)
                / 180.0,
                0.0,
                1.0,
            )
        )
    else:
        spatial_score = 0.0

    irregularity_score = float(
        np.clip(
            gap_cv / 2.0,
            0.0,
            1.0,
        )
    )

    return EmitterDescriptor(
        index=index,
        periodic_score=periodic_score,
        agility_score=agility_score,
        spatial_score=spatial_score,
        irregularity_score=irregularity_score,
    )


class MetaTaskPool:
    """Task-specific view over the immutable runtime TRAIN-500 pool."""

    def __init__(
        self,
        parent_pool: Any,
        contributions: list[Any],
        name: str,
    ):
        if not contributions:
            raise ValueError(
                f"Meta-task {name} has no contributions"
            )
        self.parent = parent_pool
        self.contributions = list(
            contributions
        )
        self.name = str(name)

    def sample_world(
        self,
        seed: int,
        emitter_count: int,
    ):
        view = copy.copy(
            self.parent
        )
        view.contributions = (
            self.contributions
        )
        emitter_count = max(
            1,
            min(
                int(emitter_count),
                len(self.contributions),
            ),
        )
        return view.sample_world(
            int(seed),
            emitter_count,
        )


class MetaTaskSampler:
    """
    Constructs trainer-only behavior regimes from the frozen cache.

    Each NPZ file is opened once during initialization, avoiding ~17k small
    opens for the current 17,826 emitter-contribution pool.
    """

    def __init__(
        self,
        pool: Any,
        seed: int = 20260930,
    ):
        self.pool = pool
        self.rng = np.random.default_rng(
            int(seed)
        )

        grouped: dict[
            Path,
            list[tuple[int, int]],
        ] = defaultdict(list)

        for i, contribution in enumerate(
            pool.contributions
        ):
            grouped[
                Path(
                    str(
                        contribution.npz_file
                    )
                )
            ].append(
                (
                    i,
                    int(
                        contribution.source_label
                    ),
                )
            )

        descriptors: list[EmitterDescriptor] = []

        # Preserve one descriptor per contribution index.
        descriptor_by_index: dict[
            int,
            EmitterDescriptor,
        ] = {}

        for npz_path, entries in grouped.items():
            with np.load(
                npz_path,
                allow_pickle=False,
            ) as z:
                for index, label in entries:
                    descriptor_by_index[
                        index
                    ] = _descriptor_from_arrays(
                        index=index,
                        toa=z[
                            f"pdw_toa_us_{label}"
                        ],
                        freq=z[
                            f"pdw_frequency_mhz_{label}"
                        ],
                        aoa=z[
                            f"pdw_aoa_deg_{label}"
                        ],
                    )

        descriptors = [
            descriptor_by_index[i]
            for i in range(
                len(pool.contributions)
            )
        ]

        self.descriptors = descriptors
        self.tasks = self._build_tasks()

        if len(self.tasks) < 3:
            raise RuntimeError(
                "Unable to construct at least three meta-training regimes"
            )

    def _build_tasks(
        self,
    ) -> list[MetaTaskPool]:
        d = self.descriptors

        p = np.asarray(
            [x.periodic_score for x in d],
            dtype=np.float64,
        )
        a = np.asarray(
            [x.agility_score for x in d],
            dtype=np.float64,
        )
        s = np.asarray(
            [x.spatial_score for x in d],
            dtype=np.float64,
        )
        irr = np.asarray(
            [x.irregularity_score for x in d],
            dtype=np.float64,
        )

        q25 = float(
            np.quantile(
                np.concatenate(
                    [p, a, s, irr]
                ),
                0.25,
            )
        )
        _ = q25

        p25, p50, p75 = np.quantile(
            p,
            [0.25, 0.50, 0.75],
        )
        a25, a50, a75 = np.quantile(
            a,
            [0.25, 0.50, 0.75],
        )
        s25, s50, s75 = np.quantile(
            s,
            [0.25, 0.50, 0.75],
        )
        i75 = float(
            np.quantile(
                irr,
                0.75,
            )
        )

        masks = {
            "periodic": (
                (p >= p75)
                & (a <= a50)
                & (s <= s50)
            ),
            "frequency_agile": (
                a >= a75
            ),
            "spatial_scan_like": (
                s >= s75
            ),
            "bursty_irregular": (
                irr >= i75
            ),
            "stationary_like": (
                (a <= a25)
                & (s <= s50)
                & (p >= p50)
            ),
        }

        _ = p25, s25

        tasks: list[MetaTaskPool] = []

        minimum = max(
            64,
            int(
                0.01
                * len(d)
            ),
        )

        for name, mask in masks.items():
            indices = np.flatnonzero(
                mask
            ).tolist()

            if len(indices) >= minimum:
                tasks.append(
                    MetaTaskPool(
                        self.pool,
                        [
                            self.pool.contributions[i]
                            for i in indices
                        ],
                        name,
                    )
                )

        # Mixed task retains all 17,826 contributions.
        tasks.append(
            MetaTaskPool(
                self.pool,
                list(
                    self.pool.contributions
                ),
                "mixed",
            )
        )

        return tasks

    def sample(
        self,
        emitter_count_distribution: np.ndarray,
    ) -> tuple[
        MetaTaskPool,
        int,
        int,
    ]:
        task = self.tasks[
            int(
                self.rng.integers(
                    len(self.tasks)
                )
            )
        ]

        emitter_count = int(
            self.rng.choice(
                emitter_count_distribution
            )
        )

        emitter_count = max(
            1,
            min(
                emitter_count,
                len(task.contributions),
            ),
        )

        world_seed = int(
            self.rng.integers(
                1,
                np.iinfo(
                    np.int32
                ).max,
            )
        )

        return (
            task,
            world_seed,
            emitter_count,
        )


# ============================================================
# VYAPTI R1 REWARD
# ============================================================


@dataclass
class RewardBreakdown:
    reward: float
    concentration_cost: float
    detection_surprise: float
    information_gain: float
    coverage: float
    prediction: float
    retune_cost: float


class VyaptiRewardV1:
    """V1 causal discovery utility + separate concentration cost.

    Reward:
        r = 1.00*D_Bayes + 0.60*IG + 0.10*Coverage
            + 0.03*Prediction - 0.10*Retune

    D_Bayes sums normalized negative log receiver-positive likelihoods over
    causal positive observations only.  It uses Pd/Pfa, not hidden truth.
    """

    # V1 discovery-oriented weights.
    HIT_W = 1.00
    IG_W = 0.60
    COVERAGE_W = 0.10
    PRED_W = 0.03
    RETUNE_W = 0.10

    # Normalize Bayesian positive-detection surprise by -log(PFA) so a very
    # unexpected positive remains O(1) instead of dominating the reward.
    BAYES_SURPRISE_NORMALIZER = max(-math.log(max(float(PFA), EPS)), EPS)

    def compute(
        self,
        state: RSACState,
        prior: StructuredPrior,
        action: int,
        positives: list[bool],
        dwell_slots: int,
    ) -> RewardBreakdown:
        action = int(action)
        dwell_slots = int(dwell_slots)

        pre, posts = causal_transition_rollout(
            state.causal,
            action,
            positives,
        )

        detection_surprise = float(
            sum(
                -math.log(
                    max(
                        float(p) * float(PD)
                        + (1.0 - float(p)) * float(PFA),
                        EPS,
                    )
                )
                / self.BAYES_SURPRISE_NORMALIZER
                for y, p in zip(
                    positives,
                    pre,
                )
                if bool(y)
            )
        )

        information_gain = float(
            sum(
                float(
                    entropy_np(p0)
                    - entropy_np(p1)
                )
                for p0, p1 in zip(
                    pre,
                    posts,
                )
            )
        )

        f = state.features(
            prior
        )

        p = float(
            f["belief"][action]
        )
        s = float(
            f["staleness"][action]
        )

        coverage = float(
            s
            * (
                0.5
                + 0.5 * float(
                    entropy_np(p)
                )
            )
        )

        prediction = float(
            f["periodic_opportunity"][action]
            + 0.25
            * f["agility"][action]
            + 0.25
            * f["spatial_opportunity"][action]
        )

        previous_action = (
            state.previous_action
        )

        switched = float(
            previous_action >= 0
            and previous_action != action
        )

        retune_fraction = (
            float(RETUNE_TIME_MS)
            / 1000.0
        ) / max(
            BASE_SLOT_S
            * dwell_slots,
            1e-9,
        )

        retune_cost = (
            switched
            * retune_fraction
        )

        # Project the current dwell into the concentration constraint.
        elapsed = float(
            state.causal.elapsed_base_slots
        )

        projected_total = (
            elapsed
            + float(dwell_slots)
        )

        projected_action_slots = (
            float(
                state.causal.visit_counts[
                    action
                ]
                + 1
            )
            * float(dwell_slots)
        )

        share = (
            projected_action_slots
            / max(
                projected_total,
                1.0,
            )
        )

        fair = 1.0 / N_BANDS

        concentration_cost = max(
            0.0,
            (
                share
                - fair
            )
            / max(
                1.0 - fair,
                EPS,
            ),
        )

        reward = (
            self.HIT_W
            * detection_surprise
            + self.IG_W
            * information_gain
            + self.COVERAGE_W
            * coverage
            + self.PRED_W
            * prediction
            - self.RETUNE_W
            * retune_cost
        )

        return RewardBreakdown(
            reward=float(reward),
            concentration_cost=float(
                concentration_cost
            ),
            detection_surprise=float(
                detection_surprise
            ),
            information_gain=float(
                information_gain
            ),
            coverage=float(coverage),
            prediction=float(prediction),
            retune_cost=float(retune_cost),
        )


# ============================================================
# VARIBAD-STYLE CONTEXT ENCODER
# ============================================================


class ContextEncoder(nn.Module):
    """
    Sequential approximate posterior.

    q_0(z) is emitted by the learned posterior heads from the zero
    recurrent state, matching online execution exactly.

    q_t(z | h_<t) is emitted before action t.
    Transition t = (obs_t, action_t, context_signal_t) is consumed only after
    action t occurs, producing q_{t+1}.

    This causal alignment prevents reward_t / obs_{t+1} leakage into z_t.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 64,
        hidden: int = 256,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)

        token_dim = (
            self.obs_dim
            + N_BANDS
            + 1
        )

        self.token = nn.Sequential(
            nn.Linear(
                token_dim,
                hidden,
            ),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            batch_first=True,
        )

        self.mu = nn.Linear(
            hidden,
            latent_dim,
        )

        self.logvar = nn.Linear(
            hidden,
            latent_dim,
        )

    def initial_hidden(
        self,
        batch: int,
        device: torch.device = DEVICE,
    ) -> torch.Tensor:
        return torch.zeros(
            1,
            int(batch),
            self.hidden,
            device=device,
        )

    def posterior_from_hidden(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = hidden[0]
        return (
            self.mu(h),
            self.logvar(h).clamp(
                -8.0,
                5.0,
            ),
        )

    @staticmethod
    def sample(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if deterministic:
            return mu

        std = torch.exp(
            0.5 * logvar
        )

        return mu + std * torch.randn_like(
            std
        )

    def step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        context_signal: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        action_oh = F.one_hot(
            action.long(),
            num_classes=N_BANDS,
        ).float()

        token = torch.cat(
            [
                obs,
                action_oh,
                context_signal.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        token = self.token(
            token
        ).unsqueeze(1)

        _out, next_hidden = self.gru(
            token,
            hidden,
        )

        mu, logvar = (
            self.posterior_from_hidden(
                next_hidden
            )
        )

        return (
            mu,
            logvar,
            next_hidden,
        )

    def sequence(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        context_signal: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Parameters
        ----------
        obs: [B,T,D]
        actions: [B,T]
        context_signal: [B,T]

        Returns
        -------
        means, logvars: [B,T+1,Z]

        Index t represents q_t before transition t.
        """
        B, T, _ = obs.shape

        hidden = self.initial_hidden(
            B,
            obs.device,
        )

        # Match online execution exactly: q_0 is the posterior produced by
        # the learned posterior heads from the zero recurrent state.  Do not
        # replace this with an independent N(0, I) initialization because the
        # posterior-head biases are trainable.
        mu0, logvar0 = self.posterior_from_hidden(
            hidden
        )

        means = [
            mu0
        ]
        logvars = [
            logvar0
        ]

        for t in range(T):
            mu, lv, hidden = self.step(
                obs[:, t],
                actions[:, t],
                context_signal[:, t],
                hidden,
            )
            means.append(mu)
            logvars.append(lv)

        return (
            torch.stack(
                means,
                dim=1,
            ),
            torch.stack(
                logvars,
                dim=1,
            ),
        )


# ============================================================
# VARIBAD-STYLE DECODER
# ============================================================


class MetaDecoder(nn.Module):
    """
    Reconstructs the next causal observation and receiver-observable
    context signal from (obs_t, action_t, z_t).

    We intentionally do NOT decode privileged task IDs.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 64,
        hidden: int = 256,
    ):
        super().__init__()

        inp = (
            obs_dim
            + N_BANDS
            + latent_dim
        )

        self.net = nn.Sequential(
            nn.Linear(
                inp,
                hidden,
            ),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.GELU(),
        )

        self.next_obs = nn.Linear(
            hidden,
            obs_dim,
        )

        self.signal = nn.Linear(
            hidden,
            1,
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        action_oh = F.one_hot(
            action.long(),
            num_classes=N_BANDS,
        ).float()

        h = self.net(
            torch.cat(
                [
                    obs,
                    action_oh,
                    z,
                ],
                dim=-1,
            )
        )

        return (
            self.next_obs(h),
            self.signal(h).squeeze(-1),
        )


# ============================================================
# RESIDUAL ACTOR
# ============================================================


class ResidualMetaActor(nn.Module):
    """
    Final logits:

        base_structured_logits + residual(obs, z)

    The residual head starts near zero, so the policy initially remains close
    to the structured controller instead of beginning from arbitrary logits.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 64,
        hidden: int = 256,
        residual_scale: float = 1.0,
    ):
        super().__init__()

        self.residual_scale = float(
            residual_scale
        )

        self.net = nn.Sequential(
            nn.Linear(
                obs_dim + latent_dim,
                hidden,
            ),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                N_BANDS,
            ),
        )

        nn.init.normal_(
            self.net[-1].weight,
            mean=0.0,
            std=1e-3,
        )

        nn.init.zeros_(
            self.net[-1].bias
        )

    def residual_logits(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.residual_scale
            * self.net(
                torch.cat(
                    [obs, z],
                    dim=-1,
                )
            )
        )

    def distribution(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        logits = (
            base_logits
            + self.residual_logits(
                obs,
                z,
            )
        )

        logp = F.log_softmax(
            logits,
            dim=-1,
        )

        probs = logp.exp()

        return (
            logits,
            probs,
            logp,
        )


# ============================================================
# DISCRETE Q NETWORK
# ============================================================


class DiscreteQ(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 64,
        hidden: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                obs_dim + latent_dim,
                hidden,
            ),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                N_BANDS,
            ),
        )

    def forward(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            torch.cat(
                [obs, z],
                dim=-1,
            )
        )


# ============================================================
# EPISODE REPLAY
# ============================================================


@dataclass
class EpisodeData:
    # T+1 because obs[t] is pre-action state and obs[t+1] is post-action state.
    obs: np.ndarray
    base_logits: np.ndarray

    # T transition-aligned arrays.
    actions: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray
    context_signal: np.ndarray
    dones: np.ndarray

    lengths: int
    task_name: str


class EpisodeReplay:

    def __init__(
        self,
        capacity: int = 600,
        seed: int = 20260930,
    ):
        self.capacity = int(capacity)
        self.episodes: list[
            EpisodeData
        ] = []
        self.rng = np.random.default_rng(
            int(seed)
        )

    def add(
        self,
        ep: EpisodeData,
    ) -> None:
        self.episodes.append(
            ep
        )

        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def __len__(self) -> int:
        return len(
            self.episodes
        )

    def ready(
        self,
        minimum: int,
    ) -> bool:
        return len(
            self.episodes
        ) >= int(minimum)

    def sample_window_batch(
        self,
        batch_episodes: int,
        context_len: int,
        train_len: int,
        train_start: int | None = None,
    ) -> tuple[
        torch.Tensor,
        ...
    ]:
        """
        Sample causal prefixes ending in a randomly chosen later training
        window rather than always training on the first 96 transitions.

        The same ``train_start`` is used across the batch so all sequences
        remain rectangular tensors.  When omitted, it is sampled uniformly
        from ``[context_len, min_episode_length - train_len]`` over the
        selected batch candidates.  Every SAC transition therefore retains
        its full history from t=0 through the requested training window.
        """
        if not self.ready(
            batch_episodes
        ):
            raise RuntimeError(
                "Replay is not ready"
            )

        batch_episodes = int(batch_episodes)
        context_len = int(context_len)
        train_len = int(train_len)

        if context_len < 0 or train_len <= 0:
            raise ValueError(
                "context_len must be >= 0 and train_len must be > 0"
            )

        minimum_required = context_len + train_len
        candidates = [
            ep
            for ep in self.episodes
            if int(ep.lengths) >= minimum_required
        ]

        if len(candidates) < batch_episodes:
            raise RuntimeError(
                "Not enough episodes for the requested context/training window"
            )

        indices = self.rng.choice(
            len(candidates),
            size=batch_episodes,
            replace=False,
        )

        selected = [
            candidates[int(idx)]
            for idx in indices
        ]

        common_max_start = min(
            int(ep.lengths) - train_len
            for ep in selected
        )

        if common_max_start < context_len:
            raise RuntimeError(
                "Selected episodes are too short for the requested context/training window"
            )

        if train_start is None:
            train_start = int(
                self.rng.integers(
                    context_len,
                    common_max_start + 1,
                )
            )
        else:
            train_start = int(train_start)
            if not (
                context_len
                <= train_start
                <= common_max_start
            ):
                raise ValueError(
                    f"train_start={train_start} outside ["
                    f"{context_len}, {common_max_start}]"
                )

        total_len = train_start + train_len

        obs_b = []
        base_b = []
        action_b = []
        reward_b = []
        cost_b = []
        signal_b = []
        done_b = []

        for ep in selected:
            # T+1 state observations for T transitions.  The prefix is kept
            # from t=0 so posterior inference sees the full causal history.
            obs_b.append(
                ep.obs[: total_len + 1]
            )
            base_b.append(
                ep.base_logits[: total_len + 1]
            )
            action_b.append(
                ep.actions[:total_len]
            )
            reward_b.append(
                ep.rewards[:total_len]
            )
            cost_b.append(
                ep.costs[:total_len]
            )
            signal_b.append(
                ep.context_signal[:total_len]
            )
            done_b.append(
                ep.dones[:total_len]
            )

        return (
            torch.as_tensor(
                np.stack(obs_b),
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.as_tensor(
                np.stack(base_b),
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.as_tensor(
                np.stack(action_b),
                dtype=torch.long,
                device=DEVICE,
            ),
            torch.as_tensor(
                np.stack(reward_b),
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.as_tensor(
                np.stack(cost_b),
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.as_tensor(
                np.stack(signal_b),
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.as_tensor(
                np.stack(done_b),
                dtype=torch.float32,
                device=DEVICE,
            ),
        )



# ============================================================
# HYBRID MODEL
# ============================================================


class VariBADResidualSDSAC(nn.Module):
    """
    Neural modules only. Optimizers are owned by the trainer.

    The name SDSAC is shorthand for the stable-discrete-SAC-inspired variant:
    average twin-Q target + conservative cost critic. It is NOT claimed to
    reproduce the full published SD-SAC algorithm with every paper-specific
    entropy-penalty/Q-clipping detail.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 64,
        enc_hidden: int = 256,
        actor_hidden: int = 256,
        residual_scale: float = 1.0,
    ):
        super().__init__()

        self.obs_dim = int(
            obs_dim
        )
        self.latent_dim = int(
            latent_dim
        )

        self.encoder = ContextEncoder(
            obs_dim,
            latent_dim,
            enc_hidden,
        )

        self.decoder = MetaDecoder(
            obs_dim,
            latent_dim,
            enc_hidden,
        )

        self.actor = ResidualMetaActor(
            obs_dim,
            latent_dim,
            actor_hidden,
            residual_scale,
        )

        self.q1 = DiscreteQ(
            obs_dim,
            latent_dim,
            actor_hidden,
        )
        self.q2 = DiscreteQ(
            obs_dim,
            latent_dim,
            actor_hidden,
        )

        self.q1_target = copy.deepcopy(
            self.q1
        )
        self.q2_target = copy.deepcopy(
            self.q2
        )

        self.cq1 = DiscreteQ(
            obs_dim,
            latent_dim,
            actor_hidden,
        )
        self.cq2 = DiscreteQ(
            obs_dim,
            latent_dim,
            actor_hidden,
        )

        self.cq1_target = copy.deepcopy(
            self.cq1
        )
        self.cq2_target = copy.deepcopy(
            self.cq2
        )

        # Trainable SAC temperature and concentration multiplier.
        self.log_alpha = nn.Parameter(
            torch.tensor(
                math.log(0.20),
                dtype=torch.float32,
            )
        )
        self.log_lambda = nn.Parameter(
            torch.tensor(
                math.log(0.10),
                dtype=torch.float32,
            )
        )

        self.to(DEVICE)

    def save(
        self,
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save model parameters plus trainer metadata.

        Optimizer state is intentionally not included because the current
        Vyapti training script uses checkpoints for best-model evaluation and
        final inference, not exact optimizer-state resume.
        """
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        torch.save(
            {
                "model": self.state_dict(),
                "metadata": dict(metadata or {}),
            },
            path,
        )

    def load(
        self,
        path: Path,
    ) -> dict[str, Any]:
        """Load model parameters and return stored metadata."""
        path = Path(path)
        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(
                f"Invalid VariBAD-RSAC checkpoint: {path}"
            )

        self.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        metadata = checkpoint.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Checkpoint metadata must be a dict: {path}"
            )
        return metadata

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp(1e-5, 20.0)

    @property
    def lam(self) -> torch.Tensor:
        return self.log_lambda.exp().clamp(1e-6, 50.0)

    def infer_sequences(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        context_signal: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        mu, logvar = self.encoder.sequence(
            obs,
            actions,
            context_signal,
        )

        z = self.encoder.sample(
            mu,
            logvar,
            deterministic=deterministic,
        )

        return (
            z,
            mu,
            logvar,
        )

    def meta_loss(
        self,
        obs_state_seq: torch.Tensor,
        actions: torch.Tensor,
        context_signal: torch.Tensor,
        train_start: int,
        kl_weight: float,
    ) -> dict[str, torch.Tensor]:
        """VariBAD-style ELBO over a causal state sequence.

        obs_state_seq has shape [B,T+1,D].  q_t is inferred from history
        before transition t.  Decoder predicts (obs_{t+1}, signal_t) from
        (obs_t, action_t, z_t).  Sequential KL compares q_{t+1} || q_t.
        """
        current_obs = obs_state_seq[:, :-1]
        target_next_obs = obs_state_seq[:, 1:]

        z, mu, logvar = self.infer_sequences(
            current_obs,
            actions,
            context_signal,
            deterministic=False,
        )

        B, T, _ = current_obs.shape
        if train_start < 0 or train_start >= T:
            raise ValueError(
                "train_start outside sequence"
            )

        pred_next, pred_signal = self.decoder(
            current_obs.reshape(
                -1,
                self.obs_dim,
            ),
            actions.reshape(-1),
            z[:, :T].reshape(
                -1,
                self.latent_dim,
            ),
        )

        pred_next = pred_next.reshape(
            B,
            T,
            self.obs_dim,
        )

        pred_signal = pred_signal.reshape(
            B,
            T,
        )

        state_loss = F.smooth_l1_loss(
            pred_next[:, train_start:],
            target_next_obs[:, train_start:],
            reduction="mean",
        )

        signal_loss = F.mse_loss(
            pred_signal[:, train_start:],
            context_signal[:, train_start:],
            reduction="mean",
        )

        # One KL term per transition t: KL(q_{t+1} || q_t).
        kl_terms = kl_diag_normal(
            mu[:, 1:],
            logvar[:, 1:],
            mu[:, :-1].detach(),
            logvar[:, :-1].detach(),
        )

        kl_loss = kl_terms[:, train_start:].mean()

        total = (
            state_loss
            + 0.25 * signal_loss
            + float(kl_weight) * kl_loss
        )

        return {
            "loss": total,
            "state_loss": state_loss,
            "signal_loss": signal_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def soft_update(
        self,
        source: nn.Module,
        target: nn.Module,
        tau: float,
    ) -> None:
        for target_param, source_param in zip(
            target.parameters(),
            source.parameters(),
        ):
            target_param.data.mul_(
                1.0 - tau
            )
            target_param.data.add_(
                tau * source_param.data
            )

    @torch.no_grad()
    def policy_action(
        self,
        obs: np.ndarray,
        base_logits: np.ndarray,
        z: torch.Tensor,
        deterministic: bool,
    ) -> int:
        obs_t = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        base_t = torch.as_tensor(
            base_logits,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        if z.shape[0] != 1:
            raise ValueError(
                "policy_action expects a single latent state"
            )

        logits, probs, _logp = (
            self.actor.distribution(
                obs_t,
                z,
                base_t,
            )
        )

        if deterministic:
            return int(
                torch.argmax(
                    logits,
                    dim=-1,
                ).item()
            )

        return int(
            torch.distributions.Categorical(
                probs=probs
            ).sample().item()
        )


# ============================================================
# END
# ============================================================

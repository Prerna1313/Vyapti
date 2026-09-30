from __future__ import annotations

"""
Vyapti Residual Discrete SAC | TRAIN-500 online-world protocol

Architecture
------------
Structured population prior
    PO-RMAB / Whittle
    UCB
    periodic / scan-window cue
    frequency-agility cue
    operational priority
            |
            v
      base action logits
            |
            +-------------------+
                                |
Observation ----------------> residual actor
                                |
                                v
                     final policy logits
                                |
                                v
                         36-band action

Critics
-------
Twin reward Q critics
Twin concentration-cost Q critics
Automatic entropy temperature
Adaptive concentration Lagrange multiplier

Training world
--------------
Existing TRAIN-500 cached source pool
        ->
fresh online world
        ->
existing TSRD receiver
        ->
causal observations
        ->
Residual SAC

No truth enters action selection or reward.
"""

import argparse
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from causal_harness import (
    N_BANDS,
    VAL_FILES,
    TEST_FILES,
    VAL_REPLAY_SEED,
    TEST_REPLAY_SEED,
    DEFAULT_SEED,
    build_replay_registry,
    validate_dataset,
    detector_sequence,
    save_json,
    _seed_for_file,
)

from vyapti_train500_cache import (
    BAND_CENTRES_MHZ,
    RECEIVER_PROFILE,
    AMPLITUDE_MIDPOINT_DB,
    AMPLITUDE_SCALE_DB,
    MAX_OBSERVED_PDWS,
    RECEIVER_DETECTION_PROBABILITY,
    RECEIVER_FALSE_ALARM_PROBABILITY,
    RECEIVER_RETUNE_TIME_MS,
    build_train_pool_from_cache,
    load_train500_cache,
    sample_fresh_training_world,
    write_runtime_manifest,
)

from vyapti_rsac_components import (
    StructuredPrior,
    RSACState,
    VyaptiRewardV1,
    VyaptiRSACEpisode,
)


# ============================================================
# DEVICE / NUMERICAL CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DEFAULT RSAC HYPERPARAMETERS
# ============================================================

DEFAULT_DWELL_SLOTS = 2

TOTAL_ACTION_STEPS = 200_000

REPLAY_CAPACITY = 100_000
BATCH_SIZE = 256

LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 3e-4
LEARNING_RATE_ALPHA = 3e-4
LEARNING_RATE_LAMBDA = 1e-4

GAMMA = 0.997

TAU = 0.005

MIN_REPLAY = 2_000

TRAIN_EVERY = 1

TARGET_ENTROPY_FRACTION = 0.70

INITIAL_ALPHA = 0.20
INITIAL_LAMBDA = 0.10

CONCENTRATION_TARGET = 0.10

RESIDUAL_SCALE = 1.00

HIDDEN = 256

VAL_EVERY = 50_000

DEFAULT_SEED = 20260929


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int) -> None:

    random.seed(int(seed))
    np.random.seed(int(seed))

    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            int(seed)
        )


# ============================================================
# REPLAY BUFFER
# ============================================================

@dataclass
class ReplayBatch:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    concentration_cost: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        seed: int,
    ):

        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)

        self.obs = np.zeros(
            (self.capacity, self.obs_dim),
            dtype=np.float32,
        )

        self.next_obs = np.zeros(
            (self.capacity, self.obs_dim),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            self.capacity,
            dtype=np.int64,
        )

        self.rewards = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.costs = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.dones = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.position = 0
        self.size = 0

        self.rng = np.random.default_rng(
            int(seed)
        )

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        concentration_cost: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:

        i = int(self.position)

        self.obs[i] = obs
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.costs[i] = float(
            concentration_cost
        )
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)

        self.position = (
            self.position + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity,
        )

    def can_sample(
        self,
        batch_size: int,
    ) -> bool:

        return (
            self.size
            >= max(
                int(batch_size),
                MIN_REPLAY,
            )
        )

    def sample(
        self,
        batch_size: int,
    ) -> ReplayBatch:

        indices = self.rng.integers(
            0,
            self.size,
            size=int(batch_size),
        )

        return ReplayBatch(
            obs=torch.as_tensor(
                self.obs[indices],
                dtype=torch.float32,
                device=DEVICE,
            ),
            action=torch.as_tensor(
                self.actions[indices],
                dtype=torch.long,
                device=DEVICE,
            ),
            reward=torch.as_tensor(
                self.rewards[indices],
                dtype=torch.float32,
                device=DEVICE,
            ),
            concentration_cost=torch.as_tensor(
                self.costs[indices],
                dtype=torch.float32,
                device=DEVICE,
            ),
            next_obs=torch.as_tensor(
                self.next_obs[indices],
                dtype=torch.float32,
                device=DEVICE,
            ),
            done=torch.as_tensor(
                self.dones[indices],
                dtype=torch.float32,
                device=DEVICE,
            ),
        )


# ============================================================
# NETWORK BUILDING BLOCK
# ============================================================

def mlp(
    input_dim: int,
    hidden: int,
    output_dim: int,
) -> nn.Sequential:

    return nn.Sequential(
        nn.Linear(
            input_dim,
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
            output_dim,
        ),
    )


# ============================================================
# RESIDUAL ACTOR
# ============================================================

class ResidualActor(nn.Module):
    """
    Actor learns only a correction to structured base logits.

    Final policy:

        logits = base_logits + residual_scale * delta

    The last layer is initialized near zero so that the initial
    policy is approximately the structured prior.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden: int = HIDDEN,
        n_actions: int = N_BANDS,
        residual_scale: float = RESIDUAL_SCALE,
    ):

        super().__init__()

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.residual_scale = float(
            residual_scale
        )

        self.encoder = nn.Sequential(
            nn.Linear(
                self.obs_dim,
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

        self.residual_head = nn.Linear(
            hidden,
            self.n_actions,
        )

        # Near-zero residual at initialization.
        nn.init.normal_(
            self.residual_head.weight,
            mean=0.0,
            std=1e-3,
        )

        nn.init.zeros_(
            self.residual_head.bias
        )

    def forward(
        self,
        obs: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:

        z = self.encoder(obs)

        residual = (
            self.residual_head(z)
        )

        return (
            base_logits
            + self.residual_scale
            * residual
        )

    def distribution(
        self,
        obs: torch.Tensor,
        base_logits: torch.Tensor,
    ):

        logits = self.forward(
            obs,
            base_logits,
        )

        log_probs = torch.log_softmax(
            logits,
            dim=-1,
        )

        probs = torch.exp(
            log_probs
        )

        return (
            logits,
            probs,
            log_probs,
        )


# ============================================================
# DISCRETE Q NETWORK
# ============================================================

class DiscreteQNetwork(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        hidden: int = HIDDEN,
        n_actions: int = N_BANDS,
    ):

        super().__init__()

        self.net = mlp(
            obs_dim,
            hidden,
            n_actions,
        )

    def forward(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:

        return self.net(obs)


# ============================================================
# TRAINER
# ============================================================

class ResidualSAC:

    def __init__(
        self,
        obs_dim: int,
        seed: int,
    ):

        self.obs_dim = int(obs_dim)

        # ----------------------------------------------------
        # Actor
        # ----------------------------------------------------
        self.actor = ResidualActor(
            obs_dim=self.obs_dim,
            hidden=HIDDEN,
            n_actions=N_BANDS,
            residual_scale=RESIDUAL_SCALE,
        ).to(DEVICE)

        # ----------------------------------------------------
        # Reward critics
        # ----------------------------------------------------
        self.q1 = DiscreteQNetwork(
            self.obs_dim
        ).to(DEVICE)

        self.q2 = DiscreteQNetwork(
            self.obs_dim
        ).to(DEVICE)

        self.q1_target = DiscreteQNetwork(
            self.obs_dim
        ).to(DEVICE)

        self.q2_target = DiscreteQNetwork(
            self.obs_dim
        ).to(DEVICE)

        self.q1_target.load_state_dict(
            self.q1.state_dict()
        )

        self.q2_target.load_state_dict(
            self.q2.state_dict()
        )

        # ----------------------------------------------------
        # Constraint critic
        # ----------------------------------------------------
        self.cq1 = DiscreteQNetwork(
            self.obs_dim
        ).to(DEVICE)

        self.cq2 = DiscreteQNetwork(
            self.obs_dim
        ).to(DEVICE)

        self.cq1_target = DiscreteQNetwork(
            self.obs_dim
        ).to(DEVICE)

        self.cq2_target = DiscreteQNetwork(
            self.obs_dim
        ).to(DEVICE)

        self.cq1_target.load_state_dict(
            self.cq1.state_dict()
        )

        self.cq2_target.load_state_dict(
            self.cq2.state_dict()
        )

        # ----------------------------------------------------
        # Optimizers
        # ----------------------------------------------------
        self.actor_optimizer = (
            torch.optim.Adam(
                self.actor.parameters(),
                lr=LEARNING_RATE_ACTOR,
            )
        )

        self.critic_optimizer = (
            torch.optim.Adam(
                list(
                    self.q1.parameters()
                )
                + list(
                    self.q2.parameters()
                ),
                lr=LEARNING_RATE_CRITIC,
            )
        )

        self.cost_optimizer = (
            torch.optim.Adam(
                list(
                    self.cq1.parameters()
                )
                + list(
                    self.cq2.parameters()
                ),
                lr=LEARNING_RATE_CRITIC,
            )
        )

        # ----------------------------------------------------
        # Automatic entropy temperature
        # ----------------------------------------------------
        target_entropy = (
            TARGET_ENTROPY_FRACTION
            * math.log(N_BANDS)
        )

        self.target_entropy = float(
            target_entropy
        )

        self.log_alpha = nn.Parameter(
            torch.tensor(
                math.log(
                    INITIAL_ALPHA
                ),
                dtype=torch.float32,
                device=DEVICE,
            )
        )

        self.alpha_optimizer = (
            torch.optim.Adam(
                [self.log_alpha],
                lr=LEARNING_RATE_ALPHA,
            )
        )

        # ----------------------------------------------------
        # Concentration Lagrange multiplier
        # ----------------------------------------------------
        self.log_lambda = nn.Parameter(
            torch.tensor(
                math.log(
                    INITIAL_LAMBDA
                ),
                dtype=torch.float32,
                device=DEVICE,
            )
        )

        self.lambda_optimizer = (
            torch.optim.Adam(
                [self.log_lambda],
                lr=LEARNING_RATE_LAMBDA,
            )
        )

        self.rng = np.random.default_rng(
            int(seed) + 981_771
        )

    @property
    def alpha(self) -> torch.Tensor:

        return torch.exp(
            self.log_alpha
        )

    @property
    def lagrange_lambda(self) -> torch.Tensor:

        return torch.exp(
            self.log_lambda
        )

    def _soft_update(
        self,
        source: nn.Module,
        target: nn.Module,
    ) -> None:

        with torch.no_grad():

            for target_param, source_param in zip(
                target.parameters(),
                source.parameters(),
            ):

                target_param.data.mul_(
                    1.0 - TAU
                )

                target_param.data.add_(
                    TAU * source_param.data
                )

    def update(
        self,
        batch: ReplayBatch,
    ) -> dict[str, float]:

        obs = batch.obs
        action = batch.action
        reward = batch.reward
        cost = batch.concentration_cost
        next_obs = batch.next_obs
        done = batch.done

        # ----------------------------------------------------
        # Base logits are stored in observation.
        #
        # Observation layout from components.py:
        #
        # 12 x N_BANDS arrays +
        # previous-action one-hot +
        # elapsed +
        # previous-positive
        #
        # base_logits is the 13th N_BANDS block.
        # ----------------------------------------------------
        base_start = (
            12 * N_BANDS
        )

        base_end = (
            13 * N_BANDS
        )

        base_logits = obs[
            :,
            base_start:base_end,
        ]

        next_base_logits = next_obs[
            :,
            base_start:base_end,
        ]

        with torch.no_grad():

            (
                _next_logits,
                next_probs,
                next_log_probs,
            ) = self.actor.distribution(
                next_obs,
                next_base_logits,
            )

            q1_next = (
                self.q1_target(next_obs)
            )

            q2_next = (
                self.q2_target(next_obs)
            )

            next_q = torch.minimum(
                q1_next,
                q2_next,
            )

            next_v = torch.sum(
                next_probs
                * (
                    next_q
                    - self.alpha.detach()
                    * next_log_probs
                ),
                dim=1,
            )

            reward_target = (
                reward
                + GAMMA
                * (1.0 - done)
                * next_v
            )

            cq1_next = (
                self.cq1_target(next_obs)
            )

            cq2_next = (
                self.cq2_target(next_obs)
            )

            next_cq = torch.maximum(
                cq1_next,
                cq2_next,
            )

            next_cost_v = torch.sum(
                next_probs
                * next_cq,
                dim=1,
            )

            cost_target = (
                cost
                + GAMMA
                * (1.0 - done)
                * next_cost_v
            )

        # ----------------------------------------------------
        # Reward critics
        # ----------------------------------------------------
        q1_all = self.q1(obs)
        q2_all = self.q2(obs)

        q1_selected = q1_all.gather(
            1,
            action.unsqueeze(1),
        ).squeeze(1)

        q2_selected = q2_all.gather(
            1,
            action.unsqueeze(1),
        ).squeeze(1)

        critic_loss = (
            torch.mean(
                (
                    q1_selected
                    - reward_target
                ) ** 2
            )
            +
            torch.mean(
                (
                    q2_selected
                    - reward_target
                ) ** 2
            )
        )

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(
                self.q1.parameters()
            )
            + list(
                self.q2.parameters()
            ),
            5.0,
        )

        self.critic_optimizer.step()

        # ----------------------------------------------------
        # Concentration critics
        # ----------------------------------------------------
        cq1_all = self.cq1(obs)
        cq2_all = self.cq2(obs)

        cq1_selected = cq1_all.gather(
            1,
            action.unsqueeze(1),
        ).squeeze(1)

        cq2_selected = cq2_all.gather(
            1,
            action.unsqueeze(1),
        ).squeeze(1)

        cost_critic_loss = (
            torch.mean(
                (
                    cq1_selected
                    - cost_target
                ) ** 2
            )
            +
            torch.mean(
                (
                    cq2_selected
                    - cost_target
                ) ** 2
            )
        )

        self.cost_optimizer.zero_grad(
            set_to_none=True
        )

        cost_critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(
                self.cq1.parameters()
            )
            + list(
                self.cq2.parameters()
            ),
            5.0,
        )

        self.cost_optimizer.step()

        # ----------------------------------------------------
        # Actor
        # ----------------------------------------------------
        (
            _logits,
            probs,
            log_probs,
        ) = self.actor.distribution(
            obs,
            base_logits,
        )

        q1_actor = self.q1(obs)
        q2_actor = self.q2(obs)

        q_actor = torch.minimum(
            q1_actor,
            q2_actor,
        )

        cq1_actor = self.cq1(obs)
        cq2_actor = self.cq2(obs)

        cq_actor = torch.maximum(
            cq1_actor,
            cq2_actor,
        )

        lambda_value = (
            self.lagrange_lambda.detach()
        )

        actor_per_action = (
            self.alpha.detach()
            * log_probs
            - q_actor
            + lambda_value
            * cq_actor
        )

        actor_loss = torch.sum(
            probs
            * actor_per_action,
            dim=1,
        ).mean()

        self.actor_optimizer.zero_grad(
            set_to_none=True
        )

        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            5.0,
        )

        self.actor_optimizer.step()

        # ----------------------------------------------------
        # Automatic entropy alpha
        # ----------------------------------------------------
        entropy = -torch.sum(
            probs.detach()
            * log_probs.detach(),
            dim=1,
        )

        alpha_loss = (
            self.log_alpha
            * (
                entropy
                - self.target_entropy
            )
        ).mean()

        self.alpha_optimizer.zero_grad(
            set_to_none=True
        )

        alpha_loss.backward()

        self.alpha_optimizer.step()

        # ----------------------------------------------------
        # Lagrange multiplier
        # ----------------------------------------------------
        mean_cost = cost.mean().detach()

        lambda_loss = -(
            self.lagrange_lambda
            * (
                mean_cost
                - CONCENTRATION_TARGET
            )
        )

        self.lambda_optimizer.zero_grad(
            set_to_none=True
        )

        lambda_loss.backward()

        self.lambda_optimizer.step()

        with torch.no_grad():

            self.log_alpha.clamp_(
                -8.0,
                3.0,
            )

            self.log_lambda.clamp_(
                -8.0,
                4.0,
            )

        # ----------------------------------------------------
        # Target networks
        # ----------------------------------------------------
        self._soft_update(
            self.q1,
            self.q1_target,
        )

        self._soft_update(
            self.q2,
            self.q2_target,
        )

        self._soft_update(
            self.cq1,
            self.cq1_target,
        )

        self._soft_update(
            self.cq2,
            self.cq2_target,
        )

        return {
            "critic_loss": float(
                critic_loss.detach().cpu()
            ),
            "cost_critic_loss": float(
                cost_critic_loss.detach().cpu()
            ),
            "actor_loss": float(
                actor_loss.detach().cpu()
            ),
            "alpha_loss": float(
                alpha_loss.detach().cpu()
            ),
            "lambda_loss": float(
                lambda_loss.detach().cpu()
            ),
            "alpha": float(
                self.alpha.detach().cpu()
            ),
            "lambda": float(
                self.lagrange_lambda.detach().cpu()
            ),
            "entropy": float(
                entropy.mean().detach().cpu()
            ),
            "mean_cost": float(
                mean_cost.cpu()
            ),
        }

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool,
    ) -> int:

        x = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        base_start = (
            12 * N_BANDS
        )

        base_end = (
            13 * N_BANDS
        )

        base_logits = x[
            :,
            base_start:base_end,
        ]

        with torch.no_grad():

            (
                logits,
                probs,
                _log_probs,
            ) = self.actor.distribution(
                x,
                base_logits,
            )

        if deterministic:

            return int(
                torch.argmax(
                    logits,
                    dim=-1,
                ).item()
            )

        distribution = torch.distributions.Categorical(
            probs=probs
        )

        return int(
            distribution.sample().item()
        )

    def save(
        self,
        path: Path,
        step: int,
        metadata: dict[str, Any],
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "step": int(step),
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "cq1": self.cq1.state_dict(),
                "cq2": self.cq2.state_dict(),
                "cq1_target": self.cq1_target.state_dict(),
                "cq2_target": self.cq2_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "log_lambda": self.log_lambda.detach().cpu(),
                "metadata": metadata,
            },
            path,
        )

    def load(
        self,
        path: Path,
    ) -> dict[str, Any]:

        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=False,
        )

        self.actor.load_state_dict(
            checkpoint["actor"]
        )

        self.q1.load_state_dict(
            checkpoint["q1"]
        )

        self.q2.load_state_dict(
            checkpoint["q2"]
        )

        self.q1_target.load_state_dict(
            checkpoint["q1_target"]
        )

        self.q2_target.load_state_dict(
            checkpoint["q2_target"]
        )

        self.cq1.load_state_dict(
            checkpoint["cq1"]
        )

        self.cq2.load_state_dict(
            checkpoint["cq2"]
        )

        self.cq1_target.load_state_dict(
            checkpoint["cq1_target"]
        )

        self.cq2_target.load_state_dict(
            checkpoint["cq2_target"]
        )

        self.log_alpha.data.copy_(
            checkpoint["log_alpha"].to(
                DEVICE
            )
        )

        self.log_lambda.data.copy_(
            checkpoint["log_lambda"].to(
                DEVICE
            )
        )

        return checkpoint.get(
            "metadata",
            {},
        )


# ============================================================
# REPLAY ENV FACTORY
# ============================================================

def build_replay_environment(
    stare_file: Path,
):
    from vyapti_simulator.tsrd.tsrd_environment import (
        TSRDStareEnvironment,
    )

    return TSRDStareEnvironment.from_stare_mode(
        stare_file=str(
            stare_file
        ),
        band_centres_mhz=BAND_CENTRES_MHZ,
        receiver_profile=RECEIVER_PROFILE,
        amplitude_midpoint_db=AMPLITUDE_MIDPOINT_DB,
        amplitude_scale_db=AMPLITUDE_SCALE_DB,
        max_observed_pdws=MAX_OBSERVED_PDWS,
        detection_probability=(
            RECEIVER_DETECTION_PROBABILITY
        ),
        false_alarm_probability=(
            RECEIVER_FALSE_ALARM_PROBABILITY
        ),
        retune_time_ms=(
            RECEIVER_RETUNE_TIME_MS
        ),
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate_model(
    agent: ResidualSAC,
    corpus_root: Path,
    registry: list[dict[str, Any]],
    prior: StructuredPrior,
    prior_active: np.ndarray,
    transition: np.ndarray,
    dwell_slots: int,
    seed: int,
    stochastic: bool = False,
) -> dict[str, Any]:

    from vyapti_simulator.tsrd.benchmark_protocol import (
        score_recorded_replay,
        _assert_scorecard_consistent,
    )

    from vyapti_simulator.core.metrics import (
        TrajectoryStep,
    )

    rows = []

    bar = tqdm(
        registry,
        desc=(
            "STOCHASTIC VAL RSAC"
            if stochastic
            else "DETERMINISTIC VAL RSAC"
        ),
        unit="scene",
    )

    agent.actor.eval()

    for recipe in bar:

        stare_file = Path(
            recipe["source_file"]
        )

        env = build_replay_environment(
            stare_file
        )

        receiver_seed = _seed_for_file(
            stare_file.stem,
            seed,
        )

        env.reset(
            seed=int(receiver_seed)
        )

        state = RSACState.create(
            transition,
            prior_active,
        )

        state.reset(
            prior_active
        )

        wrapper = VyaptiRSACEpisode(
            env=env,
            state=state,
            prior=prior,
            reward_engine=VyaptiRewardV1(),
            dwell_slots=dwell_slots,
        )

        trajectory = []

        action_counts = np.zeros(
            N_BANDS,
            dtype=np.int64,
        )

        switch_count = 0
        previous_action = -1

        while not env.done:

            obs = wrapper.current_observation()

            action = agent.select_action(
                obs,
                deterministic=not stochastic,
            )

            if (
                previous_action >= 0
                and action != previous_action
            ):
                switch_count += 1

            previous_action = action

            action_counts[
                action
            ] += 1

            looks, _env_reward, done = (
                env.step_dwell_training(
                    action,
                    dwell_slots,
                    reward_mode="detector_positive",
                )
            )

            looks = list(looks)

            if (
                len(looks) != dwell_slots
                and not done
            ):
                raise RuntimeError(
                    f"Expected {dwell_slots} looks, "
                    f"got {len(looks)}"
                )

            positives = detector_sequence(
                looks
            )

            for look in looks:

                trajectory.append(
                    TrajectoryStep(
                        action=int(action),
                        time_slot=int(
                            look["time_slot"]
                        ),
                        observation=look,
                    )
                )

            state.apply_observation(
                action,
                positives,
                len(looks),
            )

            if done:
                break

        score = score_recorded_replay(
            env,
            trajectory,
        )

        _assert_scorecard_consistent(
            score,
            trajectory,
        )

        score[
            "unique_bands_visited"
        ] = int(
            np.count_nonzero(
                action_counts
            )
        )

        score[
            "switch_count"
        ] = int(
            switch_count
        )

        score[
            "switch_rate"
        ] = float(
            switch_count
            /
            max(
                len(trajectory) - 1,
                1,
            )
        )

        score[
            "max_action_share"
        ] = float(
            np.max(
                action_counts
            )
            /
            max(
                np.sum(action_counts),
                1,
            )
        )
        score["action_distribution"] = [int(x) for x in action_counts]

        score["world_id"] = int(
            recipe["world_id"]
        )

        score["source_file"] = str(
            stare_file
        )

        rows.append(
            score
        )

    from vyapti_simulator.tsrd.benchmark_protocol import (
        _summarize,
    )

    summary = _summarize(
        rows
    )

    # Additional action-allocation diagnostics.
    if rows:

        unique = [
            r["unique_bands_visited"]
            for r in rows
        ]

        max_share = [
            r["max_action_share"]
            for r in rows
        ]

        summary[
            "mean_unique_bands_visited"
        ] = float(
            np.mean(unique)
        )

        summary[
            "mean_max_action_share"
        ] = float(
            np.mean(max_share)
        )

    return {
        "policy": (
            "ResidualDiscreteSAC"
            + (
                "_stochastic"
                if stochastic
                else "_deterministic"
            )
        ),
        "split": "val",
        "rows": rows,
        "summary": summary,
    }


# ============================================================
# CHECKPOINT RANKING
# ============================================================

def ranking_key(
    summary: dict[str, Any],
) -> tuple[float, float, float]:

    oir = float(
        summary.get(
            "opportunity_interception_ratio",
            float("-inf"),
        )
    )

    ttfi = summary.get(
        "censored_median_ttfi_s",
        None,
    )

    if ttfi is None:
        ttfi_value = float("inf")
    else:
        ttfi_value = float(ttfi)

    eir = float(
        summary.get(
            "pooled_unique_emitter_interception_rate",
            summary.get(
                "unique_emitter_interception_rate",
                float("-inf"),
            ),
        )
    )

    return (
        oir,
        -ttfi_value,
        eir,
    )


# ============================================================
# TRAIN
# ============================================================

def run_training(
    args: argparse.Namespace,
) -> None:

    seed_everything(
        args.seed
    )

    corpus_root = Path(
        args.corpus_root
    )

    cache_root = Path(
        args.cache_root
    )

    prior_dir = Path(
        args.prior_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Dataset + frozen cache
    # --------------------------------------------------------

    counts = validate_dataset(
        corpus_root
    )

    cache = load_train500_cache(
        cache_root
    )

    pool, cache = (
        build_train_pool_from_cache(
            corpus_root,
            cache_root,
        )
    )

    write_runtime_manifest(
        cache
    )

    # --------------------------------------------------------
    # Population-level structured prior
    # --------------------------------------------------------

    prior = StructuredPrior(
        prior_dir,
        expected_fingerprint=cache[
            "fingerprint"
        ],
    )

    transition = prior.transition

    prior_active = prior.prior_active

    # --------------------------------------------------------
    # Determine observation dimension
    # --------------------------------------------------------

    dummy_state = RSACState.create(
        transition,
        prior_active,
    )

    dummy_state.reset(
        prior_active
    )

    dummy_obs, _ = (
        dummy_state.observation(
            prior
        )
    )

    obs_dim = int(
        len(dummy_obs)
    )

    print("=" * 88)
    print(
        "VYAPTI RESIDUAL DISCRETE SAC"
    )
    print("=" * 88)
    print(
        f"TRAIN configs          : "
        f"{counts['train']:,}"
    )
    print(
        "TRAIN source pool      : 500 "
        "cached configs"
    )
    print(
        f"Cache fingerprint      : "
        f"{cache['fingerprint'][:16]}..."
    )
    print(
        f"Observation dimension  : "
        f"{obs_dim}"
    )
    print(
        f"Actions                : "
        f"{N_BANDS} frequency bands"
    )
    print(
        f"Dwell                  : "
        f"{50 * args.dwell_slots} ms"
    )
    print(
        f"Total action steps     : "
        f"{args.max_action_steps:,}"
    )
    print(
        "Reward                 : "
        "VyaptiRewardV1"
    )
    print(
        "Truth in scheduler     : NO"
    )
    print(
        "Retune physical model  : "
        "existing receiver; 1 ms"
    )
    print(
        "Retune in R1           : "
        "explicit causal cost"
    )
    print(
        "Concentration control  : "
        "Lagrange cost critic"
    )
    print(
        "Entropy                : "
        "automatic SAC temperature"
    )

    metadata = {
        "algorithm": "Vyapti-Residual-Discrete-SAC",
        "seed": int(args.seed),
        "corpus_root": str(
            corpus_root
        ),
        "cache_root": str(
            cache_root
        ),
        "prior_dir": str(
            prior_dir
        ),
        "source_pool_fingerprint": cache[
            "fingerprint"
        ],
        "n_bands": N_BANDS,
        "dwell_slots": int(
            args.dwell_slots
        ),
        "dwell_ms": int(
            50 * args.dwell_slots
        ),
        "pd": float(
            RECEIVER_DETECTION_PROBABILITY
        ),
        "pfa": float(
            RECEIVER_FALSE_ALARM_PROBABILITY
        ),
        "retune_ms": float(
            RECEIVER_RETUNE_TIME_MS
        ),
        "obs_dim": obs_dim,
        "residual_scale": RESIDUAL_SCALE,
        "gamma": GAMMA,
        "tau": TAU,
        "target_entropy_fraction": (
            TARGET_ENTROPY_FRACTION
        ),
        "concentration_target": (
            CONCENTRATION_TARGET
        ),
        "reward_version": "R1",
        "hidden_truth_for_policy": False,
    }

    save_json(
        output_dir / "run_manifest.json",
        metadata,
    )

    agent = ResidualSAC(
        obs_dim=obs_dim,
        seed=args.seed,
    )

    replay = ReplayBuffer(
        capacity=REPLAY_CAPACITY,
        obs_dim=obs_dim,
        seed=args.seed,
    )

    reward_engine = VyaptiRewardV1()

    world_rng = np.random.default_rng(
        int(args.seed)
    )

    global_step = 0
    update_count = 0

    running_rewards = deque(
        maxlen=200
    )

    best_key = None

    train_worlds = 0

    t0 = time.perf_counter()

    while global_step < args.max_action_steps:

        # ----------------------------------------------------
        # Fresh world from existing TRAIN-500 pool
        # ----------------------------------------------------

        env, _sources, world_seed, emitter_count = (
            sample_fresh_training_world(
                pool,
                world_rng,
            )
        )

        train_worlds += 1

        state = RSACState.create(
            transition,
            prior_active,
        )

        state.reset(
            prior_active
        )

        wrapper = VyaptiRSACEpisode(
            env=env,
            state=state,
            prior=prior,
            reward_engine=reward_engine,
            dwell_slots=args.dwell_slots,
        )

        obs = wrapper.reset(
            seed=int(world_seed),
            prior_active=prior_active,
        )

        episode_reward = 0.0

        while (
            not env.done
            and global_step
            < args.max_action_steps
        ):

            # ------------------------------------------------
            # SAC policy action
            # ------------------------------------------------

            action = agent.select_action(
                obs,
                deterministic=False,
            )

            next_obs, reward, cost, done, info = (
                wrapper.step(
                    action
                )
            )

            replay.add(
                obs=obs,
                action=action,
                reward=reward,
                concentration_cost=cost,
                next_obs=next_obs,
                done=done,
            )

            episode_reward += float(
                reward
            )

            obs = next_obs

            global_step += 1

            # ------------------------------------------------
            # SAC update
            # ------------------------------------------------

            if (
                replay.can_sample(
                    BATCH_SIZE
                )
                and global_step % TRAIN_EVERY
                == 0
            ):

                batch = replay.sample(
                    BATCH_SIZE
                )

                stats = agent.update(
                    batch
                )

                update_count += 1

            if done:
                break

        running_rewards.append(
            episode_reward
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if global_step % 1000 == 0:

            elapsed = (
                time.perf_counter()
                - t0
            )

            rate = (
                global_step
                /
                max(
                    elapsed,
                    1e-9,
                )
            )

            print(
                f"[RSAC] "
                f"steps={global_step:,} "
                f"worlds={train_worlds:,} "
                f"updates={update_count:,} "
                f"avg_ep_reward="
                f"{np.mean(running_rewards):.4f} "
                f"alpha="
                f"{float(agent.alpha):.5f} "
                f"lambda="
                f"{float(agent.lagrange_lambda):.5f} "
                f"rate={rate:.1f}/s"
            )

        # ----------------------------------------------------
        # Validation checkpoints
        # ----------------------------------------------------

        if (
            global_step > 0
            and global_step
            % args.val_every == 0
        ):

            val_registry = (
                build_replay_registry(
                    corpus_root,
                    split="val",
                    count=VAL_FILES,
                )
            )

            deterministic_val = (
                evaluate_model(
                    agent=agent,
                    corpus_root=corpus_root,
                    registry=val_registry,
                    prior=prior,
                    prior_active=prior_active,
                    transition=transition,
                    dwell_slots=args.dwell_slots,
                    seed=VAL_REPLAY_SEED,
                    stochastic=False,
                )
            )

            stochastic_val = (
                evaluate_model(
                    agent=agent,
                    corpus_root=corpus_root,
                    registry=val_registry,
                    prior=prior,
                    prior_active=prior_active,
                    transition=transition,
                    dwell_slots=args.dwell_slots,
                    seed=VAL_REPLAY_SEED,
                    stochastic=True,
                )
            )

            milestone = (
                global_step
                // 1000
            )

            save_json(
                output_dir
                / f"validation_{milestone:04d}k.json",
                {
                    "deterministic":
                        deterministic_val,
                    "stochastic":
                        stochastic_val,
                    "step":
                        int(global_step),
                },
            )

            current_summary = (
                deterministic_val[
                    "summary"
                ]
            )

            current_key = ranking_key(
                current_summary
            )

            if (
                best_key is None
                or current_key > best_key
            ):

                best_key = current_key

                agent.save(
                    output_dir
                    / "best_validation.pt",
                    global_step,
                    metadata={
                        **metadata,
                        "best_validation_step":
                            int(global_step),
                        "best_validation_summary":
                            current_summary,
                    },
                )

                print(
                    "NEW BEST VALIDATION "
                    f"checkpoint at "
                    f"{global_step:,}"
                )

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    agent.save(
        output_dir
        / "final.pt",
        global_step,
        metadata={
            **metadata,
            "final_step":
                int(global_step),
        },
    )

    save_json(
        output_dir / "training_complete.json",
        {
            "total_action_steps":
                int(global_step),
            "training_worlds":
                int(train_worlds),
            "gradient_updates":
                int(update_count),
            "best_validation_key":
                best_key,
            "elapsed_s":
                time.perf_counter() - t0,
        },
    )

    print(
        "\nTRAINING COMPLETE"
    )


# ============================================================
# FINAL TEST
# ============================================================

def run_final_test(
    args: argparse.Namespace,
) -> None:

    corpus_root = Path(
        args.corpus_root
    )

    cache_root = Path(
        args.cache_root
    )

    prior_dir = Path(
        args.prior_dir
    )

    output_dir = Path(
        args.output_dir
    )

    cache = load_train500_cache(
        cache_root
    )

    prior = StructuredPrior(
        prior_dir,
        expected_fingerprint=cache[
            "fingerprint"
        ],
    )

    transition = prior.transition
    prior_active = prior.prior_active

    dummy_state = RSACState.create(
        transition,
        prior_active,
    )

    dummy_state.reset(
        prior_active
    )

    obs, _ = (
        dummy_state.observation(
            prior
        )
    )

    agent = ResidualSAC(
        obs_dim=len(obs),
        seed=args.seed,
    )

    checkpoint = (
        output_dir
        / "best_validation.pt"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing best checkpoint: "
            f"{checkpoint}"
        )

    agent.load(
        checkpoint
    )

    registry = build_replay_registry(
        corpus_root,
        split="test",
        count=TEST_FILES,
    )

    deterministic_test = (
        evaluate_model(
            agent=agent,
            corpus_root=corpus_root,
            registry=registry,
            prior=prior,
            prior_active=prior_active,
            transition=transition,
            dwell_slots=args.dwell_slots,
            seed=TEST_REPLAY_SEED,
            stochastic=False,
        )
    )

    stochastic_test = (
        evaluate_model(
            agent=agent,
            corpus_root=corpus_root,
            registry=registry,
            prior=prior,
            prior_active=prior_active,
            transition=transition,
            dwell_slots=args.dwell_slots,
            seed=TEST_REPLAY_SEED,
            stochastic=True,
        )
    )

    save_json(
        output_dir
        / "TEST_final.json",
        {
            "deterministic":
                deterministic_test,
            "stochastic":
                stochastic_test,
        },
    )

    print(
        "\n=== TEST COMPLETE ==="
    )

    print(
        json.dumps(
            deterministic_test[
                "summary"
            ],
            indent=2,
            sort_keys=True,
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--corpus-root",
        required=True,
    )

    ap.add_argument(
        "--cache-root",
        required=True,
    )

    ap.add_argument(
        "--prior-dir",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    ap.add_argument(
        "--dwell-slots",
        type=int,
        choices=(1, 2),
        default=DEFAULT_DWELL_SLOTS,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    ap.add_argument(
        "--max-action-steps",
        type=int,
        default=TOTAL_ACTION_STEPS,
    )

    ap.add_argument(
        "--val-every",
        type=int,
        default=VAL_EVERY,
    )

    ap.add_argument(
        "--skip-test",
        action="store_true",
    )

    ap.add_argument(
        "--test-only",
        action="store_true",
    )

    args = ap.parse_args()

    seed_everything(
        args.seed
    )

    if args.test_only:
        run_final_test(
            args
        )
        return

    run_training(
        args
    )

    if not args.skip_test:
        run_final_test(
            args
        )


if __name__ == "__main__":
    main()

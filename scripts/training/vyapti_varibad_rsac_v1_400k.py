from __future__ import annotations

"""
Vyapti VariBAD-style Meta-Residual Discrete SAC V1 on the existing TRAIN-500 online-world path.

Architecture
------------
1. Frozen TRAIN-500 source pool -> fresh online worlds.
2. Hidden trainer-only behavior regimes define meta-training distributions.
3. VariBAD-style recurrent Gaussian posterior q(z_t | history_<t).
4. VAE-style next-observation + receiver-signal reconstruction with
   sequential Gaussian KL.
5. Structured band prior from PO-RMAB/Whittle + UCB + causal prediction cues.
6. Residual policy learns corrections to that prior.
7. Average-twin-Q discrete SAC target, inspired by SDSAC's double-average-Q
   treatment of discrete action instability.
8. Separate conservative concentration-cost critics and adaptive Lagrange
   multiplier.
9. Automatic entropy temperature.
10. Stochastic VAL checkpoint selection with fixed 100-file TSRD replay.
11. Mandatory milestone checkpoints at 100k / 200k / 300k / 400k action steps.

This is a research hybrid. It is not a verbatim implementation of VariBAD or
SDSAC. The exact published ideas are adapted to Vyapti's causal observation
interface and 36-band discrete scheduling problem.

No hidden truth, emitter ID, or trainer-only task label enters the deployed
policy, reward, belief update, or VAL/TEST action selection.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    DEFAULT_SEED as HARNESS_DEFAULT_SEED,
    build_replay_registry,
    detector_sequence,
    save_json,
    validate_dataset,
    _seed_for_file,
)
from vyapti_train500_cache import (
    RECEIVER_DETECTION_PROBABILITY,
    RECEIVER_FALSE_ALARM_PROBABILITY,
    RECEIVER_RETUNE_TIME_MS,
    build_train_pool_from_cache,
    load_train500_cache,
    sample_fresh_training_world,
    write_runtime_manifest,
    emitter_count_distribution_from_cache,
)
from vyapti_simulator.core.metrics import TrajectoryStep
from vyapti_simulator.tsrd.benchmark_protocol import (
    score_recorded_replay,
    _assert_scorecard_consistent,
    _summarize,
)
from vyapti_varibad_rsac_components_v1 import (
    DEVICE,
    N_BANDS as COMPONENT_N_BANDS,
    StructuredPrior,
    RSACState,
    VyaptiRewardV1,
    MetaTaskSampler,
    VariBADResidualSDSAC,
    EpisodeData,
    EpisodeReplay,
    V1_WHITTLE_W,
    V1_UCB_W,
    V1_PERIODIC_W,
    V1_STALENESS_W,
    V1_AGILITY_W,
    V1_SPATIAL_W,
    V1_PRIORITY_W,
)

if COMPONENT_N_BANDS != N_BANDS:
    raise RuntimeError(
        f"N_BANDS mismatch: causal_harness={N_BANDS}, "
        f"components={COMPONENT_N_BANDS}"
    )


# ============================================================
# DEFAULTS
# ============================================================

SEED = HARNESS_DEFAULT_SEED
DWELL_SLOTS = 2
MAX_ACTION_STEPS = 400_000
VAL_EVERY = 100_000

LATENT_DIM = 64
ENCODER_HIDDEN = 256
ACTOR_HIDDEN = 256

CONTEXT_LEN = 64
TRAIN_LEN = 32
TOTAL_SEQUENCE_LEN = CONTEXT_LEN + TRAIN_LEN
BATCH_EPISODES = 8

REPLAY_EPISODES = 1000
MIN_REPLAY_EPISODES = 32

# SAC updates happen after each collected episode once replay is ready.
SAC_UPDATES_PER_EPISODE = 8

# Gamma is specified per 50 ms base slot in the frozen PPO protocol.
GAMMA_BASE_SLOT = 0.997

def action_gamma(dwell_slots: int) -> float:
    return float(GAMMA_BASE_SLOT ** int(dwell_slots))

# Automatic entropy target.
TARGET_ENTROPY_FRACTION = 0.40

# Concentration constraint target.
CONCENTRATION_TARGET = 0.05

# Residual-policy strength.
RESIDUAL_SCALE = 1.0
RESIDUAL_REG = 1e-3

# VAE/meta inference.
META_LR = 1e-4
REWARD_RECON_WEIGHT = 0.25
KL_FINAL_WEIGHT = 0.05
KL_ANNEAL_STEPS = 20_000

# SAC.
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
LR_LAMBDA = 1e-4
TAU = 0.005
INITIAL_ALPHA = 0.20
INITIAL_LAMBDA = 0.10

# Standard SAC-style initial exploration; does not affect the final policy.
WARMUP_ACTION_STEPS = 3_000

# Exact checkpointing is milestone based; an episode may end just after the
# target. We still record the actual action count and target milestone.


# ============================================================
# REPRODUCIBILITY
# ============================================================


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ============================================================
# RECEIVER
# ============================================================


def build_replay_environment(stare_file: Path):
    from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment
    from vyapti_train500_cache import (
        BAND_CENTRES_MHZ,
        RECEIVER_PROFILE,
        AMPLITUDE_MIDPOINT_DB,
        AMPLITUDE_SCALE_DB,
        MAX_OBSERVED_PDWS,
    )

    return TSRDStareEnvironment.from_stare_mode(
        stare_file=str(stare_file),
        band_centres_mhz=BAND_CENTRES_MHZ,
        receiver_profile=RECEIVER_PROFILE,
        amplitude_midpoint_db=AMPLITUDE_MIDPOINT_DB,
        amplitude_scale_db=AMPLITUDE_SCALE_DB,
        max_observed_pdws=MAX_OBSERVED_PDWS,
        detection_probability=RECEIVER_DETECTION_PROBABILITY,
        false_alarm_probability=RECEIVER_FALSE_ALARM_PROBABILITY,
        retune_time_ms=RECEIVER_RETUNE_TIME_MS,
    )


# ============================================================
# META-RL EPISODE COLLECTION
# ============================================================


def collect_episode(
    *,
    task_pool: Any,
    emitter_count: int,
    world_seed: int,
    model: VariBADResidualSDSAC,
    prior: StructuredPrior,
    reward_engine: VyaptiRewardV1,
    dwell_slots: int,
    action_budget: int,
    force_uniform_actions_before: int,
) -> EpisodeData:
    """Collect one causal episode.

    context_signal is the raw detector-positive rate per dwell. It is used by
    the VariBAD-style inference model both during training and evaluation so
    that meta-inference is invariant to our particular shaped reward.
    """
    env, _sources = task_pool.sample_world(
        int(world_seed),
        int(emitter_count),
    )

    state = RSACState.create(
        prior.transition,
        prior.prior_active,
        dwell_slots=dwell_slots,
    )
    state.reset(
        prior.prior_active
    )

    hidden = model.encoder.initial_hidden(
        1,
        DEVICE,
    )

    obs, base = state.observation(
        prior
    )

    obs_list: list[np.ndarray] = []
    base_list: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    costs: list[float] = []
    context_signal: list[float] = []
    done_list: list[bool] = []

    steps = 0

    while (
        not env.done
        and steps < int(action_budget)
    ):
        # z_t is inferred from history strictly before action t.
        with torch.no_grad():
            mu, logvar = (
                model.encoder.posterior_from_hidden(
                    hidden
                )
            )
            z = model.encoder.sample(
                mu,
                logvar,
                deterministic=False,
            )

        # Warm-up uses uniform band exploration, following standard SAC
        # practice for difficult exploration, while still preserving causal
        # context inference and R1 reward logging.
        if steps < int(force_uniform_actions_before):
            action = int(
                np.random.randint(
                    0,
                    N_BANDS,
                )
            )
        else:
            action = model.policy_action(
                obs,
                base,
                z,
                deterministic=False,
            )

        looks, _env_reward, done = (
            env.step_dwell_training(
                int(action),
                int(dwell_slots),
                reward_mode="detector_positive",
            )
        )
        looks = list(looks)

        if not looks:
            raise RuntimeError(
                "Training environment returned zero looks"
            )

        positives = detector_sequence(
            looks
        )

        bd = reward_engine.compute(
            state,
            prior,
            action,
            positives,
            len(looks),
        )

        # Raw receiver-observable context signal, not hidden truth and not the
        # shaped reward.  This is intentionally consistent train/eval.
        signal = float(
            np.mean(
                np.asarray(
                    positives,
                    dtype=np.float32,
                )
            )
        )

        obs_list.append(
            obs.copy()
        )
        base_list.append(
            base.copy()
        )
        actions.append(
            int(action)
        )
        rewards.append(
            float(bd.reward)
        )
        costs.append(
            float(bd.concentration_cost)
        )
        context_signal.append(
            signal
        )
        done_list.append(
            bool(done)
        )

        # Commit Bayesian belief FIRST, then visit/periodicity state.
        state.apply_observation(
            action,
            positives,
            len(looks),
        )

        # The context encoder consumes transition t only after action t has
        # completed, generating q_{t+1}. This preserves causal alignment.
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)
            action_t = torch.tensor(
                [int(action)],
                dtype=torch.long,
                device=DEVICE,
            )
            signal_t = torch.tensor(
                [signal],
                dtype=torch.float32,
                device=DEVICE,
            )
            _, _, hidden = model.encoder.step(
                obs_t,
                action_t,
                signal_t,
                hidden,
            )

        steps += 1

        if done:
            break

        obs, base = state.observation(
            prior
        )

    if not obs_list:
        raise RuntimeError(
            "Collected an empty episode"
        )

    final_obs, final_base = state.observation(
        prior
    )

    obs_arr = np.stack(
        obs_list + [final_obs],
        axis=0,
    ).astype(np.float32)

    base_arr = np.stack(
        base_list + [final_base],
        axis=0,
    ).astype(np.float32)

    return EpisodeData(
        obs=obs_arr,
        base_logits=base_arr,
        actions=np.asarray(
            actions,
            dtype=np.int64,
        ),
        rewards=np.asarray(
            rewards,
            dtype=np.float32,
        ),
        costs=np.asarray(
            costs,
            dtype=np.float32,
        ),
        context_signal=np.asarray(
            context_signal,
            dtype=np.float32,
        ),
        dones=np.asarray(
            done_list,
            dtype=np.float32,
        ),
        lengths=len(actions),
        task_name=task_pool.name,
    )


# ============================================================
# TRAINER
# ============================================================


class MetaSACTrainer:

    def __init__(
        self,
        model: VariBADResidualSDSAC,
        gamma_action: float,
    ):
        self.model = model
        self.gamma_action = float(gamma_action)

        self.opt_meta = torch.optim.Adam(
            list(model.encoder.parameters())
            + list(model.decoder.parameters()),
            lr=META_LR,
        )

        self.opt_actor = torch.optim.Adam(
            model.actor.parameters(),
            lr=LR_ACTOR,
        )

        self.opt_q = torch.optim.Adam(
            list(model.q1.parameters())
            + list(model.q2.parameters()),
            lr=LR_CRITIC,
        )

        self.opt_cq = torch.optim.Adam(
            list(model.cq1.parameters())
            + list(model.cq2.parameters()),
            lr=LR_CRITIC,
        )

        self.opt_alpha = torch.optim.Adam(
            [model.log_alpha],
            lr=LR_ALPHA,
        )

        self.opt_lambda = torch.optim.Adam(
            [model.log_lambda],
            lr=LR_LAMBDA,
        )

    @staticmethod
    def _set_requires_grad(
        modules: list[nn.Module],
        enabled: bool,
    ) -> None:
        for module in modules:
            for param in module.parameters():
                param.requires_grad_(enabled)

    @staticmethod
    def _soft_update(
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
                    TAU
                    * source_param.data
                )

    def meta_update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        context_signal: torch.Tensor,
        train_start: int,
        global_step: int,
    ) -> dict[str, float]:
        kl_weight = KL_FINAL_WEIGHT * min(
            1.0,
            float(global_step)
            / max(
                KL_ANNEAL_STEPS,
                1,
            ),
        )

        loss_dict = self.model.meta_loss(
            obs_state_seq=obs,
            actions=actions,
            context_signal=context_signal,
            train_start=train_start,
            kl_weight=kl_weight,
        )

        self.opt_meta.zero_grad(
            set_to_none=True
        )

        loss_dict["loss"].backward()

        nn.utils.clip_grad_norm_(
            list(
                self.model.encoder.parameters()
            )
            + list(
                self.model.decoder.parameters()
            ),
            5.0,
        )

        self.opt_meta.step()

        return {
            "meta_loss": float(
                loss_dict["loss"].detach().cpu()
            ),
            "state_recon": float(
                loss_dict["state_loss"].detach().cpu()
            ),
            "signal_recon": float(
                loss_dict["signal_loss"].detach().cpu()
            ),
            "kl": float(
                loss_dict["kl_loss"].detach().cpu()
            ),
            "kl_weight": float(
                kl_weight
            ),
        }

    def sac_update(
        self,
        obs: torch.Tensor,
        base_logits: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        costs: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
        next_base_logits: torch.Tensor,
        z: torch.Tensor,
        next_z: torch.Tensor,
    ) -> dict[str, float]:
        """One average-twin-Q discrete SAC update."""

        obs_f = obs.reshape(
            -1,
            obs.shape[-1],
        )
        base_f = base_logits.reshape(
            -1,
            base_logits.shape[-1],
        )
        act_f = actions.reshape(-1)
        reward_f = rewards.reshape(-1)
        cost_f = costs.reshape(-1)
        done_f = dones.reshape(-1)
        next_obs_f = next_obs.reshape(
            -1,
            next_obs.shape[-1],
        )
        next_base_f = next_base_logits.reshape(
            -1,
            next_base_logits.shape[-1],
        )
        z_f = z.reshape(
            -1,
            z.shape[-1],
        ).detach()
        next_z_f = next_z.reshape(
            -1,
            next_z.shape[-1],
        ).detach()

        # ----------------------------------------------------
        # Reward critic target
        # ----------------------------------------------------
        with torch.no_grad():
            _, next_probs, next_log_probs = (
                self.model.actor.distribution(
                    next_obs_f,
                    next_z_f,
                    next_base_f,
                )
            )

            q1_next = self.model.q1_target(
                next_obs_f,
                next_z_f,
            )
            q2_next = self.model.q2_target(
                next_obs_f,
                next_z_f,
            )

            # SDSAC-inspired double-average-Q target.
            q_next_avg = 0.5 * (
                q1_next
                + q2_next
            )

            next_value = torch.sum(
                next_probs
                * (
                    q_next_avg
                    - self.model.alpha.detach()
                    * next_log_probs
                ),
                dim=-1,
            )

            target_q = (
                reward_f
                + self.gamma_action
                * (1.0 - done_f)
                * next_value
            )

            # Cost critics are deliberately conservative: max(C1,C2).
            cq1_next = self.model.cq1_target(
                next_obs_f,
                next_z_f,
            )
            cq2_next = self.model.cq2_target(
                next_obs_f,
                next_z_f,
            )
            cq_next_max = torch.maximum(
                cq1_next,
                cq2_next,
            )

            next_cost_value = torch.sum(
                next_probs
                * cq_next_max,
                dim=-1,
            )

            target_cost = (
                cost_f
                + self.gamma_action
                * (1.0 - done_f)
                * next_cost_value
            )

        # ----------------------------------------------------
        # Reward critics
        # ----------------------------------------------------
        q1_selected = self.model.q1(
            obs_f,
            z_f,
        ).gather(
            1,
            act_f.unsqueeze(1),
        ).squeeze(1)

        q2_selected = self.model.q2(
            obs_f,
            z_f,
        ).gather(
            1,
            act_f.unsqueeze(1),
        ).squeeze(1)

        q_loss = (
            F.mse_loss(
                q1_selected,
                target_q,
            )
            + F.mse_loss(
                q2_selected,
                target_q,
            )
        )

        self.opt_q.zero_grad(
            set_to_none=True
        )
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            list(
                self.model.q1.parameters()
            )
            + list(
                self.model.q2.parameters()
            ),
            5.0,
        )
        self.opt_q.step()

        # ----------------------------------------------------
        # Cost critics
        # ----------------------------------------------------
        cq1_selected = self.model.cq1(
            obs_f,
            z_f,
        ).gather(
            1,
            act_f.unsqueeze(1),
        ).squeeze(1)

        cq2_selected = self.model.cq2(
            obs_f,
            z_f,
        ).gather(
            1,
            act_f.unsqueeze(1),
        ).squeeze(1)

        cost_q_loss = (
            F.mse_loss(
                cq1_selected,
                target_cost,
            )
            + F.mse_loss(
                cq2_selected,
                target_cost,
            )
        )

        self.opt_cq.zero_grad(
            set_to_none=True
        )
        cost_q_loss.backward()
        nn.utils.clip_grad_norm_(
            list(
                self.model.cq1.parameters()
            )
            + list(
                self.model.cq2.parameters()
            ),
            5.0,
        )
        self.opt_cq.step()

        # ----------------------------------------------------
        # Actor
        # ----------------------------------------------------
        # Freeze critics while differentiating actor objective.
        self._set_requires_grad(
            [
                self.model.q1,
                self.model.q2,
                self.model.cq1,
                self.model.cq2,
            ],
            False,
        )

        logits, probs, log_probs = (
            self.model.actor.distribution(
                obs_f,
                z_f,
                base_f,
            )
        )
        _ = logits

        q1_actor = self.model.q1(
            obs_f,
            z_f,
        )
        q2_actor = self.model.q2(
            obs_f,
            z_f,
        )
        q_actor_avg = 0.5 * (
            q1_actor
            + q2_actor
        )

        cq1_actor = self.model.cq1(
            obs_f,
            z_f,
        )
        cq2_actor = self.model.cq2(
            obs_f,
            z_f,
        )
        cq_actor_max = torch.maximum(
            cq1_actor,
            cq2_actor,
        )

        entropy = -torch.sum(
            probs * log_probs,
            dim=-1,
        )

        actor_integrand = (
            self.model.alpha.detach()
            * log_probs
            - q_actor_avg
            + self.model.lam.detach()
            * cq_actor_max
        )

        actor_loss = torch.sum(
            probs * actor_integrand,
            dim=-1,
        ).mean()

        residual = self.model.actor.residual_logits(
            obs_f,
            z_f,
        )

        actor_loss = (
            actor_loss
            + RESIDUAL_REG
            * residual.pow(2).mean()
        )

        self.opt_actor.zero_grad(
            set_to_none=True
        )
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.actor.parameters(),
            5.0,
        )
        self.opt_actor.step()

        self._set_requires_grad(
            [
                self.model.q1,
                self.model.q2,
                self.model.cq1,
                self.model.cq2,
            ],
            True,
        )

        # ----------------------------------------------------
        # Automatic entropy temperature
        # ----------------------------------------------------
        target_entropy = (
            TARGET_ENTROPY_FRACTION
            * math.log(N_BANDS)
        )

        # Positive-entropy convention:
        # H > target -> alpha decreases.
        alpha_loss = (
            self.model.log_alpha
            * (
                entropy.detach()
                - target_entropy
            )
        ).mean()

        self.opt_alpha.zero_grad(
            set_to_none=True
        )
        alpha_loss.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            self.model.log_alpha.clamp_(
                -8.0,
                3.0,
            )

        # ----------------------------------------------------
        # Concentration Lagrange multiplier
        # ----------------------------------------------------
        mean_cost = cost_f.detach().mean()

        # Minimize -lambda*(cost-target). If cost > target, lambda grows.
        lambda_loss = -(
            self.model.lam
            * (
                mean_cost
                - CONCENTRATION_TARGET
            )
        )

        self.opt_lambda.zero_grad(
            set_to_none=True
        )
        lambda_loss.backward()
        self.opt_lambda.step()

        with torch.no_grad():
            self.model.log_lambda.clamp_(
                -8.0,
                4.0,
            )

        # ----------------------------------------------------
        # Target updates
        # ----------------------------------------------------
        self._soft_update(
            self.model.q1,
            self.model.q1_target,
        )
        self._soft_update(
            self.model.q2,
            self.model.q2_target,
        )
        self._soft_update(
            self.model.cq1,
            self.model.cq1_target,
        )
        self._soft_update(
            self.model.cq2,
            self.model.cq2_target,
        )

        return {
            "q_loss": float(
                q_loss.detach().cpu()
            ),
            "cost_q_loss": float(
                cost_q_loss.detach().cpu()
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
                self.model.alpha.detach().cpu()
            ),
            "lambda": float(
                self.model.lam.detach().cpu()
            ),
            "entropy": float(
                entropy.mean().detach().cpu()
            ),
            "mean_cost": float(
                mean_cost.detach().cpu()
            ),
        }


# ============================================================
# BUILD TRAINING BATCH
# ============================================================


def build_training_batch(
    replay: EpisodeReplay,
) -> tuple[torch.Tensor, ...]:
    return replay.sample_window_batch(
        batch_episodes=BATCH_EPISODES,
        context_len=CONTEXT_LEN,
        train_len=TRAIN_LEN,
    )


def align_sac_window(
    obs_seq: torch.Tensor,
    base_seq: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    costs: torch.Tensor,
    dones: torch.Tensor,
    context_signal: torch.Tensor,
    train_start: int,
    train_len: int = TRAIN_LEN,
) -> tuple[torch.Tensor, ...]:
    """Create exact z_t -> z_{t+1} tensors for a selected training window."""
    train_start = int(train_start)
    train_len = int(train_len)
    if train_start < 0 or train_len <= 0:
        raise ValueError(
            "train_start must be >= 0 and train_len must be > 0"
        )

    end = train_start + train_len
    if obs_seq.shape[1] < end + 1:
        raise ValueError(
            f"obs_seq too short: need {end + 1}, got {obs_seq.shape[1]}"
        )
    if actions.shape[1] < end:
        raise ValueError(
            f"transition sequence too short: need {end}, got {actions.shape[1]}"
        )

    current_slice = slice(train_start, end)
    next_slice = slice(train_start + 1, end + 1)

    return (
        obs_seq[:, current_slice],
        obs_seq[:, next_slice],
        base_seq[:, current_slice],
        base_seq[:, next_slice],
        actions[:, current_slice],
        rewards[:, current_slice],
        costs[:, current_slice],
        dones[:, current_slice],
        context_signal[:, current_slice],
    )


# ============================================================
# VALIDATION / TEST
# ============================================================


def evaluate_model(
    model: VariBADResidualSDSAC,
    prior: StructuredPrior,
    registry: list[dict[str, Any]],
    dwell_slots: int,
    seed: int,
    stochastic: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    label = (
        "STOCHASTIC"
        if stochastic
        else "DETERMINISTIC"
    )

    bar = tqdm(
        registry,
        desc=f"{label} REPLAY VARIBAD-RSAC",
        unit="scene",
    )

    model.eval()

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
            prior.transition,
            prior.prior_active,
            dwell_slots=int(dwell_slots),
        )

        state.reset(
            prior.prior_active
        )

        hidden = model.encoder.initial_hidden(
            1,
            DEVICE,
        )

        obs, base = state.observation(
            prior
        )

        trajectory: list[TrajectoryStep] = []

        action_count = np.zeros(
            N_BANDS,
            dtype=np.int64,
        )

        switch_count = 0
        previous_action = -1
        entropies: list[float] = []
        latent_kl_to_prior: list[float] = []

        while not env.done:
            with torch.no_grad():
                mu, logvar = (
                    model.encoder.posterior_from_hidden(
                        hidden
                    )
                )

                # Deterministic mode uses posterior mean for z.
                z = model.encoder.sample(
                    mu,
                    logvar,
                    deterministic=not stochastic,
                )

                obs_t = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=DEVICE,
                ).unsqueeze(0)

                base_t = torch.as_tensor(
                    base,
                    dtype=torch.float32,
                    device=DEVICE,
                ).unsqueeze(0)

                logits, probs, log_probs = (
                    model.actor.distribution(
                        obs_t,
                        z,
                        base_t,
                    )
                )

                entropy = float(
                    (
                        -(
                            probs
                            * log_probs
                        ).sum(dim=-1)
                    ).item()
                )

                entropies.append(
                    entropy
                )

                # q(z_t) KL N(0,I) is a useful diagnostic for latent adaptation.
                q_kl = 0.5 * torch.sum(
                    torch.exp(logvar)
                    + mu.pow(2)
                    - 1.0
                    - logvar,
                    dim=-1,
                )
                latent_kl_to_prior.append(
                    float(q_kl.item())
                )

                if stochastic:
                    action = int(
                        torch.distributions.Categorical(
                            probs=probs
                        ).sample().item()
                    )
                else:
                    action = int(
                        torch.argmax(
                            logits,
                            dim=-1,
                        ).item()
                    )

            if (
                previous_action >= 0
                and action != previous_action
            ):
                switch_count += 1

            previous_action = action
            action_count[action] += 1

            looks, _env_reward, done = (
                env.step_dwell_training(
                    action,
                    int(dwell_slots),
                    reward_mode="detector_positive",
                )
            )

            looks = list(looks)

            if not looks:
                raise RuntimeError(
                    "Replay environment returned zero looks"
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

            # Same context signal as training: raw receiver hit rate.
            context_signal = float(
                np.mean(
                    np.asarray(
                        positives,
                        dtype=np.float32,
                    )
                )
            )

            with torch.no_grad():
                action_t = torch.tensor(
                    [action],
                    dtype=torch.long,
                    device=DEVICE,
                )
                signal_t = torch.tensor(
                    [context_signal],
                    dtype=torch.float32,
                    device=DEVICE,
                )

                _mu2, _lv2, hidden = (
                    model.encoder.step(
                        obs_t,
                        action_t,
                        signal_t,
                        hidden,
                    )
                )

            if done:
                break

            obs, base = state.observation(
                prior
            )

        score = score_recorded_replay(
            env,
            trajectory,
        )

        _assert_scorecard_consistent(
            score,
            trajectory,
        )

        total_actions = max(
            int(action_count.sum()),
            1,
        )

        score["switch_count"] = int(
            switch_count
        )
        score["switch_rate"] = float(
            switch_count
            / max(total_actions - 1, 1)
        )
        score["unique_bands_visited"] = int(
            np.count_nonzero(
                action_count
            )
        )
        score["max_action_share"] = float(
            np.max(action_count)
            / total_actions
        )
        score["mean_policy_entropy"] = float(
            np.mean(entropies)
            if entropies
            else 0.0
        )
        score["mean_latent_kl_to_prior"] = float(
            np.mean(latent_kl_to_prior)
            if latent_kl_to_prior
            else 0.0
        )
        score["world_id"] = int(
            recipe["world_id"]
        )
        score["source_file"] = str(
            stare_file
        )

        rows.append(
            score
        )

    summary = _summarize(
        rows
    )

    if rows:
        summary["mean_unique_bands_visited"] = float(
            np.mean(
                [
                    r[
                        "unique_bands_visited"
                    ]
                    for r in rows
                ]
            )
        )
        summary["mean_max_action_share"] = float(
            np.mean(
                [
                    r[
                        "max_action_share"
                    ]
                    for r in rows
                ]
            )
        )
        summary["mean_switch_rate"] = float(
            np.mean(
                [
                    r["switch_rate"]
                    for r in rows
                ]
            )
        )
        summary["mean_policy_entropy"] = float(
            np.mean(
                [
                    r["mean_policy_entropy"]
                    for r in rows
                ]
            )
        )
        summary["mean_latent_kl_to_prior"] = float(
            np.mean(
                [
                    r[
                        "mean_latent_kl_to_prior"
                    ]
                    for r in rows
                ]
            )
        )

    return {
        "policy": (
            "Vyapti-VariBAD-Residual-Discrete-SAC-v1"
            + (
                "-stochastic"
                if stochastic
                else "-deterministic"
            )
        ),
        "rows": rows,
        "summary": summary,
    }


# ============================================================
# CHECKPOINT SELECTION
# ============================================================


def checkpoint_key(
    summary: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float]:
    """Balanced stochastic-VAL checkpoint key.

    Feasibility is enforced first so a checkpoint that drifts materially away
    from the receiver operating point cannot win merely by improving discovery.
    Among feasible checkpoints we prioritize OIR, then unique-emitter coverage,
    then TTFI and scan efficiency.
    """

    def finite(name: str, default: float) -> float:
        value = summary.get(name, default)
        try:
            value = float(value)
        except Exception:
            return default
        return value if np.isfinite(value) else default

    pd = finite("conditional_pd", 0.0)
    pfa = finite("true_pfa", 1.0)
    oir = finite(
        "pooled_occupied_cell_detection_ratio",
        finite("opportunity_interception_ratio", float("-inf")),
    )
    eir = finite(
        "pooled_unique_emitter_interception_rate",
        float("-inf"),
    )
    ttfi = finite(
        "ttfi_km_median_ms",
        float("nan"),
    )
    if np.isfinite(ttfi):
        ttfi = ttfi / 1000.0
    else:
        ttfi = finite(
            "censored_median_ttfi_s",
            finite("median_censored_ttfi_s", float("inf")),
        )
    empty = finite("empty_scan_fraction", 1.0)

    # Receiver operating-point guard rails.
    feasible = float(
        pd >= 0.88 and pfa <= 0.06
    )

    return (
        feasible,
        oir,
        eir,
        -ttfi,
        -empty,
        pd,
        -pfa,
    )


# ============================================================
# TRAIN
# ============================================================


def train(
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

    counts = validate_dataset(
        corpus_root
    )

    cache = load_train500_cache(
        cache_root
    )

    pool, cache = build_train_pool_from_cache(
        corpus_root,
        cache_root,
    )

    write_runtime_manifest(
        cache
    )

    prior = StructuredPrior(
        prior_dir,
        expected_fingerprint=cache[
            "fingerprint"
        ],
    )

    # The descriptor task sampler is trainer-only. It does not change the
    # frozen 500-config source selection.
    task_sampler = MetaTaskSampler(
        pool,
        seed=args.seed + 17_003,
    )

    emitter_count_distribution = (
        emitter_count_distribution_from_cache(
            cache
        )
    )

    dummy_state = RSACState.create(
        prior.transition,
        prior.prior_active,
        dwell_slots=int(args.dwell_slots),
    )
    dummy_state.reset(
        prior.prior_active
    )
    dummy_obs, _ = dummy_state.observation(
        prior
    )

    obs_dim = int(
        len(dummy_obs)
    )

    model = VariBADResidualSDSAC(
        obs_dim=obs_dim,
        latent_dim=LATENT_DIM,
        enc_hidden=ENCODER_HIDDEN,
        actor_hidden=ACTOR_HIDDEN,
        residual_scale=RESIDUAL_SCALE,
    ).to(DEVICE)

    gamma_action = action_gamma(
        int(args.dwell_slots)
    )

    trainer = MetaSACTrainer(
        model,
        gamma_action=gamma_action,
    )

    replay = EpisodeReplay(
        capacity=REPLAY_EPISODES,
        seed=args.seed + 991,
    )

    reward_engine = VyaptiRewardV1()

    task_rng = np.random.default_rng(
        args.seed + 123_457
    )

    metadata = {
        "algorithm": (
            "Vyapti-VariBAD-Residual-Discrete-SAC-v1"
        ),
        "version": "v1_discovery_coverage_400k",
        "algorithm_status": (
            "research_hybrid_not_exact_published_algorithm"
        ),
        "seed": int(args.seed),
        "n_bands": N_BANDS,
        "obs_dim": obs_dim,
        "latent_dim": LATENT_DIM,
        "dwell_slots": int(
            args.dwell_slots
        ),
        "dwell_ms": int(
            50 * args.dwell_slots
        ),
        "gamma_base_slot": GAMMA_BASE_SLOT,
        "gamma_action": gamma_action,
        "train_source_pool": 500,
        "pool_fingerprint": cache[
            "fingerprint"
        ],
        "prior_dir": str(
            prior_dir
        ),
        "reward": "VyaptiRewardV1",
        "context_signal": (
            "raw_detector_positive_rate_per_dwell"
        ),
        "meta_inference": (
            "VariBAD-style causal recurrent Gaussian posterior"
        ),
        "task_decoder": False,
        "task_decoder_reason": (
            "privileged task-ID supervision is intentionally disabled"
        ),
        "task_regimes": [
            task.name
            for task in task_sampler.tasks
        ],
        "task_regime_sizes": {
            task.name: int(
                len(task.contributions)
            )
            for task in task_sampler.tasks
        },
        "task_sampler_note": (
            "trainer-only regime distributions derived from cached PDW descriptors; "
            "descriptors/task labels never enter policy, reward, evaluator, or test"
        ),
        "discrete_sac_variant": (
            "average-twin-Q target inspired by SDSAC; "
            "not a verbatim SDSAC implementation"
        ),
        "cost_critic": (
            "conservative maximum of twin cost critics"
        ),
        "target_entropy_fraction": TARGET_ENTROPY_FRACTION,
        "concentration_target": CONCENTRATION_TARGET,
        "base_prior_weights": {
            "whittle": V1_WHITTLE_W,
            "ucb": V1_UCB_W,
            "periodic": V1_PERIODIC_W,
            "staleness": V1_STALENESS_W,
            "agility": V1_AGILITY_W,
            "spatial": V1_SPATIAL_W,
            "priority": V1_PRIORITY_W,
        },
        "reward_weights": {
            "hit": VyaptiRewardV1.HIT_W,
            "information_gain": VyaptiRewardV1.IG_W,
            "coverage": VyaptiRewardV1.COVERAGE_W,
            "prediction": VyaptiRewardV1.PRED_W,
            "retune": VyaptiRewardV1.RETUNE_W,
        },
        "reward_detection_surprise": "normalized_bayesian_positive_surprise",
        "replay_capacity_episodes": REPLAY_EPISODES,
        "min_replay_episodes": MIN_REPLAY_EPISODES,
        "sac_updates_per_episode": SAC_UPDATES_PER_EPISODE,
        "checkpoint_milestones": [100000, 200000, 300000, 400000],
        "checkpoint_selection": (
            "stochastic VAL; feasible if Pd>=0.88 and Pfa<=0.06; "
            "then OIR, pooled EIR, KM/censored TTFI, empty-scan fraction"
        ),
        "context_len": CONTEXT_LEN,
        "train_len": TRAIN_LEN,
        "replay_window_sampling": (
            "random train_start >= context_len; full causal prefix retained"
        ),
        "residual_scale": RESIDUAL_SCALE,
        "warmup_action_steps": WARMUP_ACTION_STEPS,
        "hidden_truth_for_policy": False,
        "hidden_truth_for_reward": False,
        "hidden_truth_for_context_inference": False,
        "receiver_pd": float(
            RECEIVER_DETECTION_PROBABILITY
        ),
        "receiver_pfa": float(
            RECEIVER_FALSE_ALARM_PROBABILITY
        ),
        "retune_ms": float(
            RECEIVER_RETUNE_TIME_MS
        ),
    }

    save_json(
        output_dir
        / "run_manifest.json",
        metadata,
    )

    global_step = 0
    episode_count = 0
    gradient_updates = 0
    best_key: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ] | None = None
    next_validation = int(
        args.val_every
    )
    reward_window: list[float] = []
    t0 = time.perf_counter()

    print(
        "=" * 92
    )
    print(
        "VYAPTI VARIBAD-STYLE META-RESIDUAL DISCRETE SAC V1"
    )
    print(
        "=" * 92
    )
    print(
        f"TRAIN configs          : {counts['train']:,}"
    )
    print(
        "TRAIN source pool      : 500 cached configs"
    )
    print(
        f"Pool fingerprint       : {cache['fingerprint'][:16]}..."
    )
    print(
        "Meta task regimes      : "
        + ", ".join(
            task.name
            for task in task_sampler.tasks
        )
    )
    print(
        f"Observation dimension  : {obs_dim}"
    )
    print(
        f"Latent dimension       : {LATENT_DIM}"
    )
    print(
        f"Actions                : {N_BANDS} frequency bands"
    )
    print(
        f"Dwell                  : {50 * args.dwell_slots} ms"
    )
    print(
        f"Base gamma             : {GAMMA_BASE_SLOT}"
    )
    print(
        f"Action gamma           : {gamma_action:.8f}"
    )
    print(
        f"Warmup                 : {WARMUP_ACTION_STEPS:,} uniform actions"
    )
    print(
        "Hidden truth in policy : NO"
    )
    print(
        "Hidden truth in reward : NO"
    )
    print(
        "Task IDs in policy     : NO"
    )

    while global_step < int(
        args.max_action_steps
    ):
        task_pool, world_seed, emitter_count = (
            task_sampler.sample(
                emitter_count_distribution
            )
        )

        remaining = (
            int(args.max_action_steps)
            - global_step
        )

        ep = collect_episode(
            task_pool=task_pool,
            emitter_count=int(
                emitter_count
            ),
            world_seed=int(
                world_seed
            ),
            model=model,
            prior=prior,
            reward_engine=reward_engine,
            dwell_slots=int(
                args.dwell_slots
            ),
            action_budget=remaining,
            force_uniform_actions_before=max(
                0,
                WARMUP_ACTION_STEPS
                - global_step,
            ),
        )

        replay.add(ep)

        episode_count += 1
        global_step += int(
            ep.lengths
        )

        reward_window.append(
            float(
                ep.rewards.sum()
            )
        )
        if len(reward_window) > 50:
            reward_window.pop(0)

        # ----------------------------------------------------
        # META + SAC updates
        # ----------------------------------------------------
        if replay.ready(
            MIN_REPLAY_EPISODES
        ) and ep.lengths >= TOTAL_SEQUENCE_LEN:
            for _ in range(
                SAC_UPDATES_PER_EPISODE
            ):
                (
                    obs_seq,
                    base_seq,
                    actions_seq,
                    rewards_seq,
                    costs_seq,
                    signal_seq,
                    dones_seq,
                ) = replay.sample_window_batch(
                    batch_episodes=BATCH_EPISODES,
                    context_len=CONTEXT_LEN,
                    train_len=TRAIN_LEN,
                )

                # The replay sampler keeps the complete causal prefix from t=0
                # through a randomly selected later training window.
                train_start = int(obs_seq.shape[1]) - TRAIN_LEN - 1
                if train_start < CONTEXT_LEN:
                    raise RuntimeError(
                        f"sampled train_start={train_start} < context={CONTEXT_LEN}"
                    )

                # First train the variational task inference model.
                meta_stats = trainer.meta_update(
                    obs=obs_seq,
                    actions=actions_seq,
                    context_signal=signal_seq,
                    train_start=train_start,
                    global_step=global_step,
                )

                # Re-infer z after the meta update and detach it from SAC.
                with torch.no_grad():
                    z_seq, _mu, _logvar = model.infer_sequences(
                        obs_seq[:, :-1],
                        actions_seq,
                        signal_seq,
                        deterministic=False,
                    )

                (
                    current_obs,
                    next_obs,
                    current_base,
                    next_base,
                    current_actions,
                    current_rewards,
                    current_costs,
                    current_dones,
                    _current_signal,
                ) = align_sac_window(
                    obs_seq,
                    base_seq,
                    actions_seq,
                    rewards_seq,
                    costs_seq,
                    dones_seq,
                    signal_seq,
                    train_start=train_start,
                    train_len=TRAIN_LEN,
                )

                current_z = z_seq[
                    :,
                    train_start:train_start + TRAIN_LEN,
                ]

                next_z = z_seq[
                    :,
                    train_start + 1:train_start + TRAIN_LEN + 1,
                ]

                sac_stats = trainer.sac_update(
                    obs=current_obs,
                    base_logits=current_base,
                    actions=current_actions,
                    rewards=current_rewards,
                    costs=current_costs,
                    dones=current_dones,
                    next_obs=next_obs,
                    next_base_logits=next_base,
                    z=current_z,
                    next_z=next_z,
                )

                gradient_updates += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------
        if episode_count % 10 == 0:
            elapsed = (
                time.perf_counter()
                - t0
            )
            rate = (
                global_step
                / max(
                    elapsed,
                    1e-9,
                )
            )

            print(
                f"[META-RSAC] "
                f"steps={global_step:,} "
                f"episodes={episode_count:,} "
                f"updates={gradient_updates:,} "
                f"avg_ep_reward="
                f"{np.mean(reward_window):.4f} "
                f"alpha={float(model.log_alpha.exp().detach().cpu()):.5f} "
                f"lambda={float(model.log_lambda.exp().detach().cpu()):.5f} "
                f"rate={rate:.1f}/s"
            )

        # ----------------------------------------------------
        # VALIDATION MILESTONES
        # ----------------------------------------------------
        while (
            global_step >= next_validation
        ) and (
            next_validation
            <= int(args.max_action_steps)
        ):
            registry = build_replay_registry(
                corpus_root,
                split="val",
                count=VAL_FILES,
            )

            # Save training RNG state before validation.
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()

            cuda_states = None
            if torch.cuda.is_available():
                cuda_states = torch.cuda.get_rng_state_all()

            # Fixed, reproducible validation RNG.
            seed_everything(
                VAL_REPLAY_SEED
            )

            deterministic = evaluate_model(
                model,
                prior,
                registry,
                int(args.dwell_slots),
                VAL_REPLAY_SEED,
                stochastic=False,
            )

            stochastic = evaluate_model(
                model,
                prior,
                registry,
                int(args.dwell_slots),
                VAL_REPLAY_SEED,
                stochastic=True,
            )

            # Restore training RNG state after validation.
            random.setstate(
                python_state
            )
            np.random.set_state(
                numpy_state
            )
            torch.random.set_rng_state(
                torch_state
            )

            if cuda_states is not None:
                torch.cuda.set_rng_state_all(
                    cuda_states
                )

            save_json(
                output_dir
                / f"validation_{next_validation // 1000:04d}k.json",
                {
                    "target_step": int(
                        next_validation
                    ),
                    "actual_training_step": int(
                        global_step
                    ),
                    "deterministic": deterministic,
                    "stochastic": stochastic,
                },
            )

            # Always preserve the exact milestone checkpoint so 100k/200k/300k/400k
            # can be compared after training without retraining.
            milestone_path = (
                output_dir
                / f"checkpoint_{next_validation // 1000:04d}k.pt"
            )
            model.save(
                milestone_path,
                {
                    **metadata,
                    "milestone_target_step": int(next_validation),
                    "milestone_actual_step": int(global_step),
                    "milestone_deterministic_summary": deterministic["summary"],
                    "milestone_stochastic_summary": stochastic["summary"],
                },
            )

            # Model selection is aligned to the stochastic operating mode.
            current_key = checkpoint_key(
                stochastic["summary"]
            )

            print(
                "[VAL] "
                f"target={next_validation:,} "
                f"actual={global_step:,} "
                f"key={current_key}"
            )

            if (
                best_key is None
                or current_key > best_key
            ):
                best_key = current_key

                model.save(
                    output_dir
                    / "best_validation.pt",
                    {
                        **metadata,
                        "best_validation_target_step": int(
                            next_validation
                        ),
                        "best_validation_actual_step": int(
                            global_step
                        ),
                        "best_validation_key": current_key,
                        "best_validation_summary": stochastic[
                            "summary"
                        ],
                    },
                )

                print(
                    "NEW BEST VALIDATION CHECKPOINT: "
                    f"actual_step={global_step:,}"
                )

            next_validation += int(
                args.val_every
            )

    model.save(
        output_dir / "final.pt",
        {
            **metadata,
            "final_step": int(
                global_step
            ),
            "training_episodes": int(
                episode_count
            ),
            "gradient_updates": int(
                gradient_updates
            ),
            "best_validation_key": best_key,
        },
    )

    save_json(
        output_dir
        / "training_complete.json",
        {
            "total_action_steps": int(
                global_step
            ),
            "training_episodes": int(
                episode_count
            ),
            "gradient_updates": int(
                gradient_updates
            ),
            "best_validation_key": best_key,
            "runtime_s": time.perf_counter() - t0,
        },
    )

    print(
        "TRAINING COMPLETE"
    )


# ============================================================
# TEST ONLY
# ============================================================


def test_only(
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

    cache = load_train500_cache(
        cache_root
    )

    prior = StructuredPrior(
        prior_dir,
        expected_fingerprint=cache[
            "fingerprint"
        ],
    )

    dummy = RSACState.create(
        prior.transition,
        prior.prior_active,
        dwell_slots=int(args.dwell_slots),
    )
    dummy.reset(
        prior.prior_active
    )
    dummy_obs, _ = dummy.observation(
        prior
    )

    model = VariBADResidualSDSAC(
        obs_dim=len(dummy_obs),
        latent_dim=LATENT_DIM,
        enc_hidden=ENCODER_HIDDEN,
        actor_hidden=ACTOR_HIDDEN,
        residual_scale=RESIDUAL_SCALE,
    ).to(DEVICE)

    checkpoint = (
        output_dir
        / "best_validation.pt"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"best_validation.pt not found: {checkpoint}"
        )

    checkpoint_meta = model.load(
        checkpoint
    )

    registry = build_replay_registry(
        corpus_root,
        split="test",
        count=TEST_FILES,
    )

    deterministic = evaluate_model(
        model,
        prior,
        registry,
        int(args.dwell_slots),
        TEST_REPLAY_SEED,
        stochastic=False,
    )

    stochastic = evaluate_model(
        model,
        prior,
        registry,
        int(args.dwell_slots),
        TEST_REPLAY_SEED,
        stochastic=True,
    )

    result = {
        "algorithm": (
            "Vyapti-VariBAD-Residual-Discrete-SAC-v1"
        ),
        "checkpoint": str(
            checkpoint
        ),
        "checkpoint_metadata": checkpoint_meta,
        "deterministic_test": deterministic,
        "stochastic_test": stochastic,
    }

    save_json(
        output_dir / "TEST_final.json",
        result,
    )

    print(
        "=== TEST COMPLETE ==="
    )
    print(
        json.dumps(
            deterministic["summary"],
            indent=2,
            sort_keys=True,
        )
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
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
        default=DWELL_SLOTS,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    ap.add_argument(
        "--max-action-steps",
        type=int,
        default=MAX_ACTION_STEPS,
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

    if args.test_only:
        test_only(
            args
        )
        return

    train(
        args
    )

    if not args.skip_test:
        test_only(
            args
        )


if __name__ == "__main__":
    main()

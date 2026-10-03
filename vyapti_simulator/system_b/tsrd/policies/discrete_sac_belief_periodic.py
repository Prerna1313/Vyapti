#!/usr/bin/env python3
"""
Vyapti / PS26055 — TRAIN-250 Discrete SAC
==========================================

Clean Discrete Soft Actor-Critic baseline for the current Vyapti receiver
scheduler benchmark.

RESEARCH/PROTOCOL CONTRACT
---------------------------
- 36 discrete frequency-band actions.
- Current 30 s mission.
- Native per-band dwell is supplied by the environment/WorldComposer:
    * 29 bands: 50 ms
    * 7 bands: 100 ms
- The environment owns the frozen TRAIN-250 world composition and the
  authoritative reward `truth_based_intercept_utility_v2`.
- The policy sees receiver-causal information only.
- Main model uses a factorised 2-state HMM occupancy belief with Bayesian
  hit/miss filtering + staleness + current-band one-hot + remaining mission
  fraction.
- Main state includes causal dwell-level periodicity features.
- No PPO rollout, GAE, PPO optimizer epochs, BPTT, GRU, or LSTM.
- Off-policy replay buffer: 500,000 transitions.
- Warm-up: 5,000 random environment actions.
- One SAC gradient update per environment action after warm-up/replay warmup.
- Total training budget: 400,000 environment actions.
- VAL checkpoints: 100k / 200k / 300k / 400k actions.
- TEST is never used for checkpoint selection.

IMPORTANT
---------
The shared runner supplies the authoritative reward through `PublicTransition`;
the policy never accesses world truth or constructs an environment.

PROJECT-SPECIFIC ADAPTER
------------------------
The shared experiment runner owns world generation and held-out evaluation.
`create` exposes the policy through the same public transition API used by the
other registered algorithms.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..algorithm_interface import PublicTransition

# Frozen by the shared TRAIN-250 receiver contract.
N_BANDS = 36
PD = 0.90
PFA = 0.05


# =============================================================================
# FROZEN TRAINING CONFIG
# =============================================================================

MISSION_SECONDS = 30.0
BASE_SLOT_SECONDS = 0.050
BASE_SLOTS = int(round(MISSION_SECONDS / BASE_SLOT_SECONDS))  # 600
RETUNE_SECONDS = 0.0003

ACTION_DIM = int(N_BANDS)
OBS_DIM = 5 * ACTION_DIM + 1  # belief + staleness + periodicity score + periodicity confidence + current band + time

REPLAY_CAPACITY = 500_000
MIN_REPLAY_SIZE = 5_000
BATCH_SIZE = 256
TOTAL_ACTION_STEPS = 400_000
WARMUP_ACTION_STEPS = 5_000
UPDATES_PER_ACTION = 1

GAMMA_BASE = 0.997
TAU = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4

# Discrete-SAC target entropy: original discrete-SAC paper uses ~98% of max
# categorical entropy. This is a research-aligned starting point, not a
# Vyapti-specific theorem.
TARGET_ENTROPY_FRACTION = 0.98
TARGET_ENTROPY = TARGET_ENTROPY_FRACTION * math.log(ACTION_DIM)
INITIAL_ALPHA = 0.20

CHECKPOINTS = (100_000, 200_000, 300_000, 400_000)

# Fallback values from train250_observed_band_hmm.json. The supported shared
# runner injects that calibration from the versioned algorithm specification.
DEFAULT_PRIOR_ACTIVE = 0.3426563254340138
DEFAULT_P01 = 0.022195484398616423  # inactive -> active
DEFAULT_P11 = 0.9575318615765981  # active -> active

REWARD_MODE = "truth_based_intercept_utility_v2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# =============================================================================
# WORLD ADAPTER PROTOCOL
# =============================================================================

class WorldFactory(Protocol):
    """Adapter contract to the existing TRAIN-250 WorldComposer/experiment."""

    def make_train_world(self, seed: int) -> Any:
        """Return a NEW frozen TRAIN world for one episode."""
        ...

    def make_val_world(self, recipe: dict[str, Any]) -> Any:
        """Reconstruct exactly one frozen VAL world from its recipe."""
        ...

    def make_test_world(self, recipe: dict[str, Any]) -> Any:
        """Reconstruct exactly one frozen TEST world from its recipe."""
        ...

    def load_val_recipes(self) -> list[dict[str, Any]]:
        ...

    def load_test_recipes(self) -> list[dict[str, Any]]:
        ...


# =============================================================================
# CAUSAL HMM BELIEF
# =============================================================================

class TwoStateHMMBelief:
    """Factorised 2-state HMM filter with exact aggregate-dwell updates."""

    def __init__(
        self,
        transition: np.ndarray,
        pd: float = PD,
        pfa: float = PFA,
        prior_active: np.ndarray | None = None,
    ) -> None:
        transition = np.asarray(transition, dtype=np.float64)
        if transition.shape != (ACTION_DIM, 2, 2):
            raise ValueError(
                f"transition must have shape ({ACTION_DIM},2,2); got {transition.shape}"
            )
        if np.any(transition < 0.0) or not np.allclose(transition.sum(axis=-1), 1.0, atol=1e-8):
            raise ValueError("transition must contain valid row-stochastic probabilities")
        self.transition = transition
        self.pd = float(pd)
        self.pfa = float(pfa)
        self.prior = (
            np.full(ACTION_DIM, DEFAULT_PRIOR_ACTIVE, dtype=np.float64)
            if prior_active is None
            else np.asarray(prior_active, dtype=np.float64).copy()
        )
        if self.prior.shape != (ACTION_DIM,):
            raise ValueError(f"prior_active must have shape ({ACTION_DIM},)")
        self.prior = np.clip(self.prior, 1e-6, 1.0 - 1e-6)
        self.reset()

    @classmethod
    def from_default_calibration(cls) -> "TwoStateHMMBelief":
        transition = np.zeros((ACTION_DIM, 2, 2), dtype=np.float64)
        transition[:, 0, 1] = DEFAULT_P01
        transition[:, 0, 0] = 1.0 - DEFAULT_P01
        transition[:, 1, 1] = DEFAULT_P11
        transition[:, 1, 0] = 1.0 - DEFAULT_P11
        return cls(transition, prior_active=np.full(ACTION_DIM, DEFAULT_PRIOR_ACTIVE))

    @classmethod
    def from_json(cls, path: Path) -> "TwoStateHMMBelief":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") == "vyapti_band_activity_hmm_calibration_v1":
            if payload.get("source", {}).get("dataset_split") != "TRAIN only":
                raise ValueError("HMM calibration must be derived from TRAIN only")
            p01 = float(payload["inactive_to_active_probability"])
            p11 = float(payload["active_to_active_probability"])
            transition = np.zeros((ACTION_DIM, 2, 2), dtype=np.float64)
            transition[:, 0, :] = (1.0 - p01, p01)
            transition[:, 1, :] = (1.0 - p11, p11)
            prior = np.full(ACTION_DIM, float(payload["prior_active_probability"]))
            return cls(transition, prior_active=prior)
        transition = np.asarray(payload["transition"], dtype=np.float64)
        prior = payload.get("prior_active")
        prior_arr = None if prior is None else np.asarray(prior, dtype=np.float64)
        return cls(transition, prior_active=prior_arr)

    def reset(self) -> None:
        self.belief = self.prior.copy()

    def copy_reset(self) -> "TwoStateHMMBelief":
        return TwoStateHMMBelief(
            transition=self.transition.copy(),
            pd=self.pd,
            pfa=self.pfa,
            prior_active=self.prior.copy(),
        )

    @staticmethod
    def _dist(p: float) -> np.ndarray:
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        return np.asarray([1.0 - p, p], dtype=np.float64)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, None)
        total = float(x.sum())
        if total <= 1e-12:
            return np.asarray([0.5, 0.5], dtype=np.float64)
        return x / total

    def predict_band(self, band: int, slots: int) -> float:
        band = int(band)
        dist = self._dist(self.belief[band])
        if slots > 0:
            dist = dist @ np.linalg.matrix_power(self.transition[band], int(slots))
        return float(np.clip(dist[1], 1e-6, 1.0 - 1e-6))

    def predict_all(self, slots: int) -> np.ndarray:
        return np.asarray(
            [self.predict_band(b, int(slots)) for b in range(ACTION_DIM)],
            dtype=np.float64,
        )

    def aggregate_hit_probability(self, band: int, dwell_slots: int) -> float:
        """P(at least one detector-positive during the dwell | current belief)."""
        band = int(band)
        d = int(dwell_slots)
        prior = self._dist(self.belief[band])
        T = self.transition[band]
        end_prior = prior @ np.linalg.matrix_power(T, d)
        no_hit_emission = np.diag(np.asarray([1.0 - self.pfa, 1.0 - self.pd]))
        joint_no_hit = prior.copy()
        for _ in range(d):
            joint_no_hit = joint_no_hit @ no_hit_emission @ T
        return float(np.clip(1.0 - joint_no_hit.sum(), 0.0, 1.0))

    def update_aggregate(self, band: int, aggregate_hit: bool, dwell_slots: int) -> None:
        """Exact Bayesian filter for one aggregate dwell HIT/MISS."""
        band = int(band)
        d = int(dwell_slots)
        if d not in (1, 2):
            raise ValueError("Current native dwell must be 1 or 2 base slots")

        predicted = self.predict_all(d)
        prior = self._dist(self.belief[band])
        T = self.transition[band]
        end_prior = prior @ np.linalg.matrix_power(T, d)

        no_hit_emission = np.diag(np.asarray([1.0 - self.pfa, 1.0 - self.pd]))
        joint_no_hit = prior.copy()
        for _ in range(d):
            joint_no_hit = joint_no_hit @ no_hit_emission @ T

        p_no_hit = float(np.clip(joint_no_hit.sum(), 0.0, 1.0))
        if aggregate_hit:
            p_hit = max(1.0 - p_no_hit, 1e-12)
            posterior = (end_prior - joint_no_hit) / p_hit
        else:
            posterior = joint_no_hit / max(p_no_hit, 1e-12)

        posterior = self._normalize(posterior)
        self.belief = predicted
        self.belief[band] = float(np.clip(posterior[1], 1e-6, 1.0 - 1e-6))


class CausalPeriodicity:
    """Causal dwell-level periodicity cue from aggregate HIT events."""

    def __init__(self, max_events: int = 32, tolerance_fraction: float = 0.15) -> None:
        self.max_events = int(max_events)
        self.tolerance_fraction = float(tolerance_fraction)
        self.events = [[] for _ in range(ACTION_DIM)]

    def reset(self) -> None:
        self.events = [[] for _ in range(ACTION_DIM)]

    def observe(self, band: int, event_time_slot: float, aggregate_hit: bool) -> None:
        if not aggregate_hit:
            return
        hist = self.events[int(band)]
        t = float(event_time_slot)
        if hist and t <= hist[-1]:
            return
        hist.append(t)
        if len(hist) > self.max_events:
            del hist[: len(hist) - self.max_events]

    def estimate(self, band: int, now_slot: int) -> tuple[float, float]:
        hist = self.events[int(band)]
        if len(hist) < 3:
            return 0.0, 0.0
        gaps = np.diff(np.asarray(hist, dtype=np.float64))
        gaps = gaps[gaps >= 1.0]
        if len(gaps) < 2:
            return 0.0, 0.0
        period = float(np.median(gaps))
        if period <= 1.0:
            return 0.0, 0.0
        mean_gap = float(np.mean(gaps))
        if mean_gap <= 0.0:
            return 0.0, 0.0
        cv = float(np.std(gaps) / mean_gap)
        regularity = float(np.exp(-cv))
        elapsed = max(0.0, float(now_slot) - float(hist[-1]))
        remainder = elapsed % period
        phase_error = min(remainder, period - remainder)
        tol = max(self.tolerance_fraction * period, 0.5)
        phase_score = float(np.exp(-phase_error / tol))
        confidence = float(np.clip(regularity * min(1.0, len(gaps) / 6.0), 0.0, 1.0))
        score = float(np.clip(phase_score * confidence, 0.0, 1.0))
        return score, confidence

    def features(self, now_slot: int) -> tuple[np.ndarray, np.ndarray]:
        vals = [self.estimate(b, now_slot) for b in range(ACTION_DIM)]
        return (
            np.asarray([v[0] for v in vals], dtype=np.float32),
            np.asarray([v[1] for v in vals], dtype=np.float32),
        )


@dataclass
class CausalState:
    belief: TwoStateHMMBelief
    periodicity: CausalPeriodicity
    last_visit_slot: np.ndarray
    elapsed_slots: int = 0
    previous_action: int = -1

    @classmethod
    def create(cls, belief: TwoStateHMMBelief) -> "CausalState":
        return cls(
            belief=belief.copy_reset(),
            periodicity=CausalPeriodicity(),
            last_visit_slot=np.full(ACTION_DIM, -1, dtype=np.int32),
        )

    def reset(self) -> None:
        self.belief.reset()
        self.periodicity.reset()
        self.last_visit_slot.fill(-1)
        self.elapsed_slots = 0
        self.previous_action = -1

    def observe_features(self) -> np.ndarray:
        belief = self.belief.belief.astype(np.float32)
        now = int(self.elapsed_slots)
        staleness = np.where(
            self.last_visit_slot >= 0,
            (now - self.last_visit_slot) / max(BASE_SLOTS, 1),
            1.0,
        )
        staleness = np.clip(staleness, 0.0, 1.0).astype(np.float32)
        period_score, period_conf = self.periodicity.features(now)

        current_band = np.zeros(ACTION_DIM, dtype=np.float32)
        if 0 <= self.previous_action < ACTION_DIM:
            current_band[self.previous_action] = 1.0

        remaining = np.asarray(
            [np.clip(1.0 - now / max(BASE_SLOTS, 1), 0.0, 1.0)],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [belief, staleness, period_score, period_conf, current_band, remaining]
        ).astype(np.float32)

        if obs.shape != (OBS_DIM,):
            raise RuntimeError(f"Expected obs shape ({OBS_DIM},), got {obs.shape}")
        return obs

    def commit_dwell(
        self,
        action: int,
        aggregate_hit: bool,
        dwell_slots: int,
    ) -> None:
        action = int(action)
        d = int(dwell_slots)
        if d not in (1, 2):
            raise ValueError("dwell_slots must be 1 or 2")

        start_slot = int(self.elapsed_slots)
        end_slot = start_slot + d - 1

        self.belief.update_aggregate(
            band=action,
            aggregate_hit=bool(aggregate_hit),
            dwell_slots=d,
        )

        # Aggregate HIT is represented at dwell end because the exact
        # in-dwell positive timestamp is unavailable to the scheduler.
        self.periodicity.observe(
            band=action,
            event_time_slot=end_slot,
            aggregate_hit=bool(aggregate_hit),
        )

        self.last_visit_slot[action] = start_slot
        self.previous_action = action
        self.elapsed_slots += d


@dataclass(frozen=True)
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    gammas: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.rng = np.random.default_rng(int(seed))
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.gammas = np.empty(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        gamma_t: float,
    ) -> None:
        i = self.ptr
        self.obs[i] = np.asarray(obs, dtype=np.float32)
        self.next_obs[i] = np.asarray(next_obs, dtype=np.float32)
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)
        self.gammas[i] = float(gamma_t)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return int(self.size)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if self.size < batch_size:
            raise RuntimeError("Replay buffer is smaller than requested batch")
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return ReplayBatch(
            obs=torch.as_tensor(self.obs[idx], device=device),
            actions=torch.as_tensor(self.actions[idx], device=device, dtype=torch.long),
            rewards=torch.as_tensor(self.rewards[idx], device=device),
            next_obs=torch.as_tensor(self.next_obs[idx], device=device),
            dones=torch.as_tensor(self.dones[idx], device=device),
            gammas=torch.as_tensor(self.gammas[idx], device=device),
        )


# =============================================================================
# NETWORKS
# =============================================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscreteActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.backbone = MLP(obs_dim, hidden, hidden)
        self.head = nn.Linear(hidden, n_actions)

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(obs))

    def dist(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs


class DiscreteQ(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# =============================================================================
# DISCRETE SAC
# =============================================================================

class DiscreteSAC:
    def __init__(self, obs_dim: int, n_actions: int, seed: int,
                 settings: dict[str, Any] | None = None,
                 device: torch.device | None = None) -> None:
        settings = settings or {}
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.device = device or DEVICE
        hidden = int(settings.get("hidden_units", 256))
        self.tau = float(settings.get("target_update_tau", TAU))
        target_entropy_fraction = float(settings.get("target_entropy_fraction", TARGET_ENTROPY_FRACTION))
        initial_alpha = float(settings.get("initial_alpha", INITIAL_ALPHA))
        self.actor = DiscreteActor(obs_dim, n_actions, hidden).to(self.device)
        self.q1 = DiscreteQ(obs_dim, n_actions, hidden).to(self.device)
        self.q2 = DiscreteQ(obs_dim, n_actions, hidden).to(self.device)
        self.q1_target = DiscreteQ(obs_dim, n_actions, hidden).to(self.device)
        self.q2_target = DiscreteQ(obs_dim, n_actions, hidden).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(settings.get("actor_learning_rate", ACTOR_LR)))
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=float(settings.get("critic_learning_rate", CRITIC_LR)))
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=float(settings.get("critic_learning_rate", CRITIC_LR)))

        self.log_alpha = torch.tensor(
            math.log(initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(settings.get("temperature_learning_rate", ALPHA_LR)))
        self.target_entropy = target_entropy_fraction * math.log(self.n_actions)
        self.rng = np.random.default_rng(int(seed))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs, _ = self.actor.dist(x)
        if deterministic:
            return int(torch.argmax(probs, dim=-1).item())
        return int(torch.multinomial(probs, 1).item())

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        obs = batch.obs
        actions = batch.actions
        rewards = batch.rewards
        next_obs = batch.next_obs
        dones = batch.dones
        gammas = batch.gammas

        # ---------------------------------------------------------------
        # Critic target: exact expectation over all discrete actions.
        # ---------------------------------------------------------------
        with torch.no_grad():
            next_probs, next_log_probs = self.actor.dist(next_obs)
            next_q = torch.minimum(
                self.q1_target(next_obs),
                self.q2_target(next_obs),
            )
            soft_next_value = (
                next_probs * (next_q - self.alpha.detach() * next_log_probs)
            ).sum(dim=-1)
            target = rewards + (1.0 - dones) * gammas * soft_next_value

        q1_all = self.q1(obs)
        q2_all = self.q2(obs)
        q1 = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 10.0)
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 10.0)
        self.q2_opt.step()

        # ---------------------------------------------------------------
        # Actor: maximise E[alpha*H(pi) + Q].
        # ---------------------------------------------------------------
        probs, log_probs = self.actor.dist(obs)
        with torch.no_grad():
            min_q = torch.minimum(self.q1(obs), self.q2(obs))
        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q)).sum(dim=-1).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        # ---------------------------------------------------------------
        # Automatic entropy-temperature tuning.
        # ---------------------------------------------------------------
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1)
        alpha_loss = self._update_temperature(entropy)

        # ---------------------------------------------------------------
        # Polyak target update.
        # ---------------------------------------------------------------
        with torch.no_grad():
            for target_p, source_p in zip(self.q1_target.parameters(), self.q1.parameters()):
                target_p.mul_(1.0 - self.tau).add_(self.tau * source_p)
            for target_p, source_p in zip(self.q2_target.parameters(), self.q2.parameters()):
                target_p.mul_(1.0 - self.tau).add_(self.tau * source_p)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "entropy": float(entropy.mean().item()),
            "q_mean": float(torch.minimum(q1, q2).mean().item()),
            "target_mean": float(target.mean().item()),
        }

    def _update_temperature(self, entropy: torch.Tensor) -> torch.Tensor:
        """Adjust alpha toward the configured categorical entropy target."""
        alpha_loss = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 10.0)
        self.alpha_opt.step()
        return alpha_loss

    def state_dict(self, step: int) -> dict[str, Any]:
        return {
            "step": int(step),
            "obs_dim": int(self.obs_dim),
            "n_actions": int(self.n_actions),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "target_entropy": self.target_entropy,
            "tau": self.tau,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.q1.load_state_dict(payload["q1"])
        self.q2.load_state_dict(payload["q2"])
        self.q1_target.load_state_dict(payload["q1_target"])
        self.q2_target.load_state_dict(payload["q2_target"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.q1_opt.load_state_dict(payload["q1_opt"])
        self.q2_opt.load_state_dict(payload["q2_opt"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
        self.alpha_opt.load_state_dict(payload["alpha_opt"])
        self.target_entropy = float(payload.get("target_entropy", TARGET_ENTROPY))


# =============================================================================
# ENVIRONMENT HELPERS
# =============================================================================

def _resolve_dwell_slots(env: Any, action: int) -> int:
    """Read the environment's native 50/100-ms dwell for this band.

    We deliberately fail closed if the native profile cannot be found rather
    than silently changing the benchmark to a fixed dwell.
    """
    for name in ("dwell_slots_for_band", "native_dwell_slots", "get_dwell_slots"):
        fn = getattr(env, name, None)
        if callable(fn):
            value = int(fn(int(action)))
            if value in (1, 2):
                return value
    for name in ("DWELL_SLOTS", "NATIVE_DWELL_SLOTS", "dwell_slots"):
        value = getattr(env, name, None)
        if value is not None and not callable(value):
            arr = np.asarray(value).reshape(-1)
            if len(arr) == ACTION_DIM:
                d = int(arr[int(action)])
                if d in (1, 2):
                    return d
    raise RuntimeError(
        "Could not obtain the current native dwell profile from the environment. "
        "Expose dwell_slots_for_band(action) or a 36-entry dwell-slot array "
        "(29x1, 7x2) in the existing WorldComposer/environment."
    )


def _extract_positive_sequence(looks: Sequence[Any]) -> list[bool]:
    positives: list[bool] = []
    for look in looks:
        if isinstance(look, dict):
            if "hit" in look:
                positives.append(bool(look["hit"]))
            elif "detector_positive" in look:
                positives.append(bool(look["detector_positive"]))
            else:
                raise RuntimeError("Look dict contains neither 'hit' nor 'detector_positive'")
        else:
            raise TypeError(f"Unsupported look type: {type(look)!r}")
    if not positives:
        raise RuntimeError("Environment returned zero base-slot observations")
    return positives


def _step_world(env: Any, action: int, dwell_slots: int) -> tuple[list[dict], float, bool]:
    """Call the active environment reward path."""
    result = env.step_dwell_training(
        int(action),
        int(dwell_slots),
        REWARD_MODE,
    )
    if not isinstance(result, tuple) or len(result) != 3:
        raise RuntimeError(
            "Expected step_dwell_training(action, dwell_slots, reward_mode) "
            "to return (looks, reward, done)."
        )
    looks, reward, done = result
    looks = list(looks)
    return looks, float(reward), bool(done)


def _seeded_world(factory: WorldFactory, seed: int) -> Any:
    return factory.make_train_world(int(seed))


def _actual_gamma(actual_slots: int) -> float:
    """Discount by elapsed receiver slots; retune fits inside that clock."""
    slots = max(1, int(actual_slots))
    return float(GAMMA_BASE ** slots)


# =============================================================================
# VALIDATION HOOK
# =============================================================================

def evaluate_on_recipes(
    agent: DiscreteSAC,
    factory: WorldFactory,
    recipes: Sequence[dict[str, Any]],
    split: str,
    seed_offset: int,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Evaluate on frozen reconstructed worlds and collect env scorecards.

    The exact benchmark/evaluator remains in the environment. This function
    only drives the policy causally. It calls env.metrics() when available.
    """
    rows: list[dict[str, Any]] = []
    for i, recipe in enumerate(recipes):
        if split == "val":
            env = factory.make_val_world(recipe)
        elif split == "test":
            env = factory.make_test_world(recipe)
        else:
            raise ValueError(split)

        seed = int(seed_offset + i)
        if hasattr(env, "reset"):
            try:
                env.reset(seed=seed)
            except TypeError:
                env.reset(seed)

        state = CausalState.create(agent_belief_from_config())
        state.reset()

        done = False
        while not done:
            obs = state.observe_features()
            action = agent.action(obs, deterministic=deterministic)
            dwell_slots = _resolve_dwell_slots(env, action)
            looks, _, done = _step_world(env, action, dwell_slots)
            positives = _extract_positive_sequence(looks)
            state.after_action(action, positives)

        if hasattr(env, "metrics"):
            metric = dict(env.metrics())
        elif hasattr(env, "scorecard"):
            metric = dict(env.scorecard())
        else:
            metric = {"world_id": i}
        metric["world_id"] = int(recipe.get("world_id", i))
        rows.append(metric)

    return {
        "split": split,
        "n_worlds": len(rows),
        "rows": rows,
    }


# This function is replaced with the actual belief template once the adapter
# loads the transition/prior file. Keeping it global makes evaluation simple.
_ACTIVE_BELIEF: TwoStateHMMBelief | None = None


def agent_belief_from_config() -> TwoStateHMMBelief:
    if _ACTIVE_BELIEF is None:
        return TwoStateHMMBelief.from_default_calibration()
    return TwoStateHMMBelief(
        _ACTIVE_BELIEF.transition.copy(),
        pd=_ACTIVE_BELIEF.pd,
        pfa=_ACTIVE_BELIEF.pfa,
        prior_active=_ACTIVE_BELIEF.prior.copy(),
    )


# =============================================================================
# CHECKPOINT / MANIFEST
# =============================================================================

def save_checkpoint(
    path: Path,
    agent: DiscreteSAC,
    replay: ReplayBuffer,
    step: int,
    train_stats: dict[str, Any],
    transition: np.ndarray,
    prior_active: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = agent.state_dict(step)
    payload.update(
        {
            "replay_size": len(replay),
            "train_stats": train_stats,
            "transition": torch.as_tensor(transition, dtype=torch.float64),
            "prior_active": torch.as_tensor(prior_active, dtype=torch.float64),
            "protocol": {
                "algorithm": "Discrete-SAC",
                "obs_dim": OBS_DIM,
                "actions": ACTION_DIM,
                "replay_capacity": REPLAY_CAPACITY,
                "batch_size": BATCH_SIZE,
                "warmup_action_steps": WARMUP_ACTION_STEPS,
                "updates_per_action": UPDATES_PER_ACTION,
                "total_action_steps": TOTAL_ACTION_STEPS,
                "checkpoint_steps": list(CHECKPOINTS),
                "reward_mode": REWARD_MODE,
                "periodicity": True,
                "belief": "factorized 2-state HMM + exact aggregate dwell HIT/MISS filter",
                "pd": PD,
                "pfa": PFA,
                "mission_seconds": MISSION_SECONDS,
                "base_slot_seconds": BASE_SLOT_SECONDS,
                "retune_seconds": RETUNE_SECONDS,
            },
        }
    )
    torch.save(payload, path)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


# =============================================================================
# TRAINING
# =============================================================================

def train(
    factory: WorldFactory,
    output_dir: Path,
    seed: int,
    transition: np.ndarray,
    prior_active: np.ndarray,
    resume: Path | None = None,
) -> None:
    global _ACTIVE_BELIEF
    _ACTIVE_BELIEF = TwoStateHMMBelief(
        transition,
        prior_active=prior_active,
    )

    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = DiscreteSAC(OBS_DIM, ACTION_DIM, seed=seed)
    replay = ReplayBuffer(REPLAY_CAPACITY, OBS_DIM, seed=seed + 777)

    start_step = 0
    if resume is not None:
        payload = torch.load(resume, map_location=DEVICE, weights_only=False)
        agent.load_state_dict(payload)
        start_step = int(payload["step"])
        print(f"Resuming Discrete SAC from {resume} at action step {start_step:,}")

    val_recipes = factory.load_val_recipes()
    save_json(output_dir / "val_recipes.json", val_recipes)

    belief_config = agent_belief_from_config()
    manifest = {
        "algorithm": "Discrete-SAC",
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "device": str(DEVICE),
        "reward_mode": REWARD_MODE,
        "reward_owned_by": "active TRAIN-250 environment",
        "belief": "factorized two-state HMM + Bayesian hit/miss update",
        "hmm_calibration": {
            "prior_active_by_band": belief_config.prior.tolist(),
            "inactive_to_active_by_band": belief_config.transition[:, 0, 1].tolist(),
            "active_to_active_by_band": belief_config.transition[:, 1, 1].tolist(),
        },
        "periodicity": True,
        "periodicity_type": "causal dwell-level phase score + confidence",
        "periodicity_timestamp": "dwell-end interval-censored",
        "obs_components": {
            "belief": ACTION_DIM,
            "staleness": ACTION_DIM,
            "periodicity_score": ACTION_DIM,
            "periodicity_confidence": ACTION_DIM,
            "current_band_one_hot": ACTION_DIM,
            "remaining_mission": 1,
        },
        "replay_capacity": REPLAY_CAPACITY,
        "batch_size": BATCH_SIZE,
        "min_replay_size": MIN_REPLAY_SIZE,
        "updates_per_action": UPDATES_PER_ACTION,
        "warmup_action_steps": WARMUP_ACTION_STEPS,
        "total_action_steps": TOTAL_ACTION_STEPS,
        "checkpoints": list(CHECKPOINTS),
        "actor_lr": ACTOR_LR,
        "critic_lr": CRITIC_LR,
        "alpha_lr": ALPHA_LR,
        "tau": TAU,
        "target_entropy": TARGET_ENTROPY,
        "gamma_base": GAMMA_BASE,
        "mission_seconds": MISSION_SECONDS,
        "base_slot_seconds": BASE_SLOT_SECONDS,
        "retune_seconds": RETUNE_SECONDS,
        "hidden_truth_in_observation": False,
    }
    save_json(output_dir / "experiment_config.json", manifest)

    rng = np.random.default_rng(seed + 123456)
    reward_window: list[float] = []
    last_update: dict[str, float] = {}
    wall_t0 = time.perf_counter()

    milestone_set = set(CHECKPOINTS)
    next_episode = 0

    global_step = int(start_step)
    while global_step < TOTAL_ACTION_STEPS:
        # ------------------------------------------------------------------
        # One fresh frozen TRAIN world per episode.
        # ------------------------------------------------------------------
        world_seed = int(rng.integers(1, np.iinfo(np.int32).max))
        env = _seeded_world(factory, world_seed)
        if hasattr(env, "reset"):
            try:
                env.reset(seed=world_seed)
            except TypeError:
                try:
                    env.reset(world_seed)
                except TypeError:
                    env.reset()

        state = CausalState.create(agent_belief_from_config())
        state.reset()

        done = False
        episode_reward = 0.0
        episode_actions = 0

        while not done and global_step < TOTAL_ACTION_STEPS:
            obs = state.observe_features()

            if global_step < WARMUP_ACTION_STEPS:
                action = int(rng.integers(0, ACTION_DIM))
            else:
                action = agent.action(obs, deterministic=False)

            dwell_slots = _resolve_dwell_slots(env, action)
            looks, reward, done = _step_world(env, action, dwell_slots)

            # The scheduler gets ONE receiver-visible aggregate observation
            # for the dwell: HIT iff at least one detector-positive occurred.
            if isinstance(looks, dict):
                aggregate_hit = bool(
                    looks.get("hit", looks.get("detector_positive", False))
                )
            else:
                aggregate_hit = any(
                    bool(obs.get("hit", obs.get("detector_positive", False)))
                    for obs in looks
                )

            actual_slots = dwell_slots
            gamma_t = _actual_gamma(actual_slots)

            # Commit exactly one aggregate observation to belief + periodicity.
            state.commit_dwell(
                action=action,
                aggregate_hit=aggregate_hit,
                dwell_slots=actual_slots,
            )
            next_obs = state.observe_features()

            replay.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                gamma_t=gamma_t,
            )

            episode_reward += float(reward)
            episode_actions += 1
            global_step += 1
            reward_window.append(float(reward))
            if len(reward_window) > 100:
                reward_window.pop(0)

            # --------------------------------------------------------------
            # SAC update schedule.
            # --------------------------------------------------------------
            if global_step >= WARMUP_ACTION_STEPS and len(replay) >= MIN_REPLAY_SIZE:
                for _ in range(UPDATES_PER_ACTION):
                    batch = replay.sample(BATCH_SIZE, DEVICE)
                    last_update = agent.update(batch)

            # --------------------------------------------------------------
            # Milestone checkpoint exactly by environment-action budget.
            # --------------------------------------------------------------
            if global_step in milestone_set:
                ckpt = output_dir / "checkpoints" / f"sac_step_{global_step:06d}.pt"
                save_checkpoint(
                    ckpt,
                    agent,
                    replay,
                    global_step,
                    {
                        "mean_recent_reward": float(np.mean(reward_window)) if reward_window else 0.0,
                        **last_update,
                    },
                    transition,
                    prior_active,
                )
                print(
                    f"\n[CHECKPOINT] {global_step:,} actions -> {ckpt} | "
                    f"reward100={np.mean(reward_window) if reward_window else 0.0:.6f} "
                    f"alpha={last_update.get('alpha', float(agent.alpha.item())):.4f}"
                )

        next_episode += 1

        if next_episode % 25 == 0:
            elapsed = time.perf_counter() - wall_t0
            rate = global_step / max(elapsed, 1e-9)
            print(
                f"episode={next_episode:5d} actions={global_step:7d}/{TOTAL_ACTION_STEPS} "
                f"rate={rate:7.2f}/s ep_reward={episode_reward:9.5f} "
                f"replay={len(replay):7d}"
            )

    # Always save a final resumeable checkpoint, even when stopped just after a
    # milestone.
    final_step = TOTAL_ACTION_STEPS
    final_ckpt = output_dir / "checkpoints" / "sac_final.pt"
    save_checkpoint(
        final_ckpt,
        agent,
        replay,
        final_step,
        {
            "mean_recent_reward": float(np.mean(reward_window)) if reward_window else 0.0,
            **last_update,
        },
        transition,
        prior_active,
    )

    elapsed = time.perf_counter() - wall_t0
    print(
        f"\nTRAIN COMPLETE: {TOTAL_ACTION_STEPS:,} actions, "
        f"wall={elapsed/3600.0:.2f} h, final={final_ckpt}"
    )


# =============================================================================
# ADAPTER LOADER
# =============================================================================

def load_factory(spec: str) -> WorldFactory:
    """Load `module:function` or `module:ClassName` from the user's project."""
    if ":" not in spec:
        raise ValueError("--adapter must have the form module:function")
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)
    factory = obj() if isinstance(obj, type) else obj()
    required = (
        "make_train_world",
        "make_val_world",
        "make_test_world",
        "load_val_recipes",
        "load_test_recipes",
    )
    missing = [name for name in required if not hasattr(factory, name)]
    if missing:
        raise TypeError(
            f"Adapter {spec} is missing required methods: {', '.join(missing)}"
        )
    return factory


# =============================================================================
# Shared experiment-runner policy adapter
# =============================================================================

class DiscreteSACBeliefPeriodicityPolicy:
    """Discrete SAC implementing Vyapti's receiver-only transition protocol."""

    def __init__(self, bands: int, seed: int, settings: dict, checkpoint=None):
        if int(bands) != ACTION_DIM:
            raise ValueError(f"This frozen policy expects {ACTION_DIM} bands, got {bands}")
        self.bands = int(bands)
        self.settings = dict(settings)
        self.gamma_base = float(self.settings.get("gamma_per_base_slot", GAMMA_BASE))
        self.warmup_actions = int(self.settings.get("warmup_actions", WARMUP_ACTION_STEPS))
        self.min_replay_size = int(self.settings.get("min_replay_size", MIN_REPLAY_SIZE))
        self.batch_size = int(self.settings.get("batch_size", BATCH_SIZE))
        self.updates_per_action = int(self.settings.get("updates_per_action", UPDATES_PER_ACTION))
        self.update_every_actions = int(self.settings.get("update_every_actions", 1))
        self.retune_seconds = float(self.settings.get("retune_seconds", RETUNE_SECONDS))
        self.pd = float(self.settings.get("detection_probability", PD))
        self.pfa = float(self.settings.get("false_alarm_probability", PFA))
        p01 = float(self.settings.get("inactive_to_active_probability", DEFAULT_P01))
        p11 = float(self.settings.get("active_to_active_probability", DEFAULT_P11))
        prior = float(self.settings.get("prior_active_probability", DEFAULT_PRIOR_ACTIVE))
        for name, value in (("gamma_per_base_slot", self.gamma_base), ("Pd", self.pd),
                            ("Pfa", self.pfa), ("P01", p01), ("P11", p11), ("prior", prior)):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and between zero and one")
        if self.pfa >= self.pd:
            raise ValueError("Pfa must be lower than Pd")
        if min(self.warmup_actions, self.min_replay_size, self.batch_size,
               self.updates_per_action, self.update_every_actions) < 1:
            raise ValueError("SAC replay and update settings must be positive")
        if not 0 <= self.retune_seconds < BASE_SLOT_SECONDS:
            raise ValueError("retune_seconds must be in [0, one base slot)")
        device_setting = self.settings.get("device", "auto")
        if device_setting not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if device_setting == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device("cuda" if device_setting == "cuda" or
                                   (device_setting == "auto" and torch.cuda.is_available()) else "cpu")
        if self.device.type == "cpu":
            torch.set_num_threads(int(self.settings.get("cpu_threads", 1)))
        self.rng = np.random.default_rng(int(seed))
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        transition = np.zeros((self.bands, 2, 2), dtype=np.float64)
        transition[:, 0, 0], transition[:, 0, 1] = 1.0 - p01, p01
        transition[:, 1, 0], transition[:, 1, 1] = 1.0 - p11, p11
        self.belief_template = TwoStateHMMBelief(
            transition, pd=self.pd, pfa=self.pfa,
            prior_active=np.full(self.bands, prior, dtype=np.float64),
        )
        self.replay = ReplayBuffer(
            int(self.settings.get("replay_capacity", REPLAY_CAPACITY)),
            OBS_DIM, seed=int(seed) + 777,
        )
        self.agent = DiscreteSAC(OBS_DIM, self.bands, int(seed), self.settings, self.device)
        self.total_actions = 0
        self.total_updates = 0
        self.episode_decisions = 0
        self.episode_reward = 0.0
        self.band_counts = np.zeros(self.bands, dtype=np.int64)
        self._pending = None
        self._last_update = {}
        self.reset_episode(training=True)
        if checkpoint is not None:
            self._load(checkpoint)

    def reset_episode(self, *, training: bool) -> None:
        self.state = CausalState.create(self.belief_template)
        self.state.periodicity = CausalPeriodicity(
            max_events=int(self.settings.get("periodicity_history", 32)),
            tolerance_fraction=float(self.settings.get("periodicity_tolerance_fraction", 0.15)),
        )
        self._pending = None
        self.episode_decisions = 0
        self.episode_reward = 0.0
        self.band_counts = np.zeros(self.bands, dtype=np.int64)

    def select_action(self, public_state, *, training: bool) -> int:
        if self._pending is not None:
            raise RuntimeError("observe must follow every selected action")
        if int(public_state.time_slot) != int(self.state.elapsed_slots):
            raise RuntimeError("Policy belief clock diverged from the environment clock")
        features = self.state.observe_features()
        if training and self.total_actions < self.warmup_actions:
            action = int(self.rng.integers(self.bands))
        else:
            action = self.agent.action(features, deterministic=not training)
        self._pending = {"features": features, "action": action,
                         "start_slot": int(public_state.time_slot)}
        self.band_counts[action] += 1
        return action

    def observe(self, transition: PublicTransition, *, training: bool) -> None:
        if self._pending is None:
            raise RuntimeError("select_action must precede observe")
        observation = transition.next_state.previous_observation
        if observation is None:
            raise ValueError("SAC requires a receiver observation after every action")
        if int(observation.selected_band) != int(transition.action):
            raise ValueError("Action and receiver-selected band disagree")
        elapsed_slots = self._elapsed_slots(transition)
        if int(self.state.elapsed_slots) != int(transition.state.time_slot):
            raise RuntimeError("Policy belief clock diverged from the environment clock")
        self.state.commit_dwell(int(transition.action), bool(observation.hit), elapsed_slots)
        next_features = self.state.observe_features()
        # The feature layout is belief, staleness, period score, period confidence,
        # previous band, and remaining mission fraction.
        discount = self._transition_discount(transition)
        self.episode_decisions += 1
        self.episode_reward += float(transition.reward)
        if training:
            self.total_actions += 1
            self.replay.add(self._pending["features"], transition.action,
                            transition.reward, next_features,
                            transition.terminated or transition.truncated, discount)
            should_update = (self.total_actions >= self.warmup_actions
                             and len(self.replay) >= max(self.min_replay_size, self.batch_size)
                             and self.total_actions % self.update_every_actions == 0)
            if should_update:
                for _ in range(self.updates_per_action):
                    self._last_update = self.agent.update(self.replay.sample(self.batch_size, self.device))
                    self.total_updates += 1
        self._pending = None

    @staticmethod
    def _elapsed_slots(transition: PublicTransition) -> int:
        elapsed_slots = int(transition.next_state.time_slot) - int(transition.state.time_slot)
        if elapsed_slots not in (1, 2):
            raise ValueError(f"Expected a one or two slot dwell, got {elapsed_slots}")
        return elapsed_slots

    def _transition_discount(self, transition: PublicTransition) -> float:
        return float(self.gamma_base ** self._elapsed_slots(transition))

    def end_episode(self, *, training: bool) -> dict:
        if self._pending is not None:
            raise RuntimeError("Cannot end an episode before observing its final action")
        return {
            "algorithm": "discrete_sac_belief_periodic",
            "episode_decisions": self.episode_decisions,
            "episode_reward": self.episode_reward,
            "total_training_actions": self.total_actions,
            "training_updates": self.total_updates,
            "replay_size": len(self.replay),
            "band_selection_counts": self.band_counts.tolist(),
            "alpha": float(self.agent.alpha.detach().cpu()),
            **{f"mean_{key}": value for key, value in self._last_update.items()},
        }

    def save(self, path) -> None:
        torch.save({
            "format": "vyapti_discrete_sac_belief_periodic_v1",
            "bands": self.bands,
            "settings": self.settings,
            "agent": self.agent.state_dict(self.total_actions),
            "total_actions": self.total_actions,
            "total_updates": self.total_updates,
            "rng_state": self.rng.bit_generator.state,
        }, Path(path))

    def _load(self, path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if (payload.get("format") != "vyapti_discrete_sac_belief_periodic_v1"
                or payload.get("bands") != self.bands
                or int(payload["agent"].get("obs_dim", -1)) != OBS_DIM):
            raise ValueError("Checkpoint format, feature dimension, or action space does not match")
        self.agent.load_state_dict(payload["agent"])
        self.total_actions = int(payload.get("total_actions", payload["agent"].get("step", 0)))
        self.total_updates = int(payload.get("total_updates", 0))
        if "rng_state" in payload:
            self.rng.bit_generator.state = payload["rng_state"]


def create(*, bands: int, seed: int, settings: dict, checkpoint=None):
    return DiscreteSACBeliefPeriodicityPolicy(bands, seed, settings, checkpoint)


# =============================================================================
# Standalone legacy adapter CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Vyapti TRAIN-250 Discrete SAC")
    parser.add_argument("--adapter", required=True, help="module:function or module:Class returning WorldFactory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    default_calibration = (Path(__file__).resolve().parents[4]
                           / "training_setup/belief_models/train250_observed_band_hmm.json")
    parser.add_argument("--transition-file", type=Path, default=default_calibration,
                        help="TRAIN-derived HMM calibration JSON (defaults to the shared TRAIN-250 calibration).")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    global DEVICE
    if args.device == "cpu":
        DEVICE = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        DEVICE = torch.device("cuda")

    if args.transition_file is None:
        belief_template = TwoStateHMMBelief.from_default_calibration()
        print(
            "Using shared TRAIN-250 observed-occupancy fallback: "
            f"prior={DEFAULT_PRIOR_ACTIVE:.6f}, P01={DEFAULT_P01:.6f}, P11={DEFAULT_P11:.6f}."
        )
    else:
        belief_template = TwoStateHMMBelief.from_json(args.transition_file)
        print(f"Loaded HMM transition/prior from {args.transition_file}")

    global _ACTIVE_BELIEF
    _ACTIVE_BELIEF = belief_template

    factory = load_factory(args.adapter)
    train(
        factory=factory,
        output_dir=Path(args.output_dir),
        seed=int(args.seed),
        transition=belief_template.transition,
        prior_active=belief_template.prior,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

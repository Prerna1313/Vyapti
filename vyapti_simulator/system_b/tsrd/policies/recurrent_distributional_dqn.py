#!/usr/bin/env python3
"""
Vyapti / PS26055 — Recurrent Distributional Double-Dueling DQN
===============================================================

A receiver-only value-based ML scheduler for the shared Mode-B transition
runner.

Design lineage
--------------
- DRQN: recurrent Q-learning for partial observability.
- Rainbow: Double Q-learning, prioritized replay, multi-step returns,
  dueling value/advantage streams, distributional value learning, and
  parameter-noise exploration.
- R2D2: recurrent replay using contiguous sequences with burn-in.

This is NOT a distributed implementation of the original R2D2 system.
It is a single-learner, shared-runner-compatible Vyapti implementation that
uses the most relevant ideas for this 36-action receiver scheduler.

Vyapti protocol
---------------
- 36 discrete frequency-band actions.
- 30 s mission, 50 ms base clock.
- Native dwell is 1 or 2 base slots (50/100 ms).
- Policy input is receiver-causal only.
- Environment-owned reward is used ONLY as the RL learning target; reward is
  never fed back as a policy observation.
- Observation schema: 325 dimensions:
    belief(36) + staleness(36) + periodicity(36) + periodicity_confidence(36)
    + previous_action_one_hot(36) + remaining_time(1)
    + native_dwell(36) + time_since_observed_hit(36)
    + visit_count_norm(36) + recent_hit_rate(36)
- Evaluation is deterministic greedy (argmax mean Q) by default.

Important naming note
---------------------
The implementation is intentionally described as
"R2D2/Rainbow-inspired" rather than claiming paper-exact R2D2. The original
R2D2 is a distributed recurrent-replay system; this adapter is deliberately
single-process so it fits the existing Vyapti runner.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..algorithm_interface import PublicTransition


# =============================================================================
# FROZEN / DEFAULT VYAPTI SETTINGS
# =============================================================================

N_BANDS = 36
MISSION_SECONDS = 30.0
BASE_SLOT_SECONDS = 0.050
OBS_DIM = 325
HIDDEN = 256
RECENT_HIT_WINDOW = 8
MIN_REPLAY_EPISODES = 8
TRAINING_BASE_STEP_TARGET = 480_000

# RL hyperparameters: conservative starting point for a 325-D, 36-action,
# partially-observed recurrent value learner.
LEARNING_RATE = 1e-4
GAMMA_BASE = 0.997
N_STEP = 5
BATCH_SIZE = 16
REPLAY_CAPACITY_STEPS = 200_000
MIN_REPLAY_SEQUENCES = 128
UPDATES_PER_EPISODE = 8
BURN_IN = 32
LEARNING_SEQUENCE = 64
TARGET_UPDATE_INTERVAL = 1_000
MAX_GRAD_NORM = 10.0

# Prioritized replay.
PER_ALPHA = 0.9
PER_BETA_START = 0.6
PER_BETA_END = 1.0
PER_PRIORITY_ETA = 0.9
PER_EPS = 1e-3

# Distributional / quantile value function.
N_QUANTILES = 32
N_TARGET_QUANTILES = 32
QUANTILE_EMBED_DIM = 64
QUANTILE_HUBER_K = 1.0

# NoisyNet exploration.
NOISY_SIGMA0 = 0.5
WARMUP_ACTIONS = 5_000

# Deterministic evaluation default.
EVAL_MODE = "greedy"

OBSERVATION_SCHEMA = "receiver_belief_periodicity_dwell_hit_history"
CHECKPOINT_FORMAT = "vyapti_recurrent_distributional_dqn"
REWARD_MODE = "truth_based_intercept_utility_v2"


# =============================================================================
# REPLAY
# =============================================================================

@dataclass
class Episode:
    obs: np.ndarray              # [T+1, OBS_DIM], including terminal next observation
    actions: np.ndarray          # [T]
    rewards: np.ndarray          # [T]
    gammas: np.ndarray           # [T], gamma_base ** actual_slots
    dones: np.ndarray            # [T]

    def __post_init__(self) -> None:
        transitions = int(self.actions.shape[0])
        if (self.obs.ndim != 2 or self.obs.shape[0] != transitions + 1
                or self.obs.shape[1] != OBS_DIM):
            raise ValueError("Episode replay requires T+1 observations for T transitions")
        if any(array.shape != (transitions,) for array in
               (self.rewards, self.gammas, self.dones)):
            raise ValueError("Episode transition arrays must have matching length T")

    @property
    def steps(self) -> int:
        return int(self.actions.shape[0])


@dataclass
class SequenceRef:
    episode: Episode
    start: int
    priority: float = 1.0


class SequencePrioritizedReplay:
    """Sequence-prioritized replay with episode-bounded recurrent samples."""

    def __init__(self, capacity_steps: int, seed: int, sequence_stride: int = 8):
        self.capacity_steps = int(capacity_steps)
        self.sequence_stride = int(sequence_stride)
        if self.capacity_steps < 1 or self.sequence_stride < 1:
            raise ValueError("Replay capacity and sequence stride must be positive")
        self.rng = np.random.default_rng(int(seed))
        self.episodes: deque[Episode] = deque()
        self.sequence_refs: list[SequenceRef] = []
        self.total_steps = 0

    def __len__(self) -> int:
        return int(self.total_steps)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def add(self, episode: Episode) -> None:
        if episode.steps <= 0:
            return
        self.episodes.append(episode)
        self.total_steps += episode.steps
        # Reference every possible learning-window start at the configured
        # stride, including both episode boundaries for direct loss coverage.
        max_start = episode.steps - 1
        if max_start >= 0:
            starts = list(range(0, max_start + 1, self.sequence_stride))
            if not starts or starts[-1] != max_start:
                starts.append(max_start)
            for start in starts:
                self.sequence_refs.append(SequenceRef(episode, start, 1.0))
        while self.episodes and self.total_steps > self.capacity_steps:
            old = self.episodes.popleft()
            self.total_steps -= old.steps
            self.sequence_refs = [ref for ref in self.sequence_refs if ref.episode is not old]

    def sample(
        self,
        batch_size: int,
        alpha: float,
        beta: float,
    ) -> tuple[list[SequenceRef], np.ndarray]:
        if not self.sequence_refs:
            raise RuntimeError("Replay has no eligible sequences")
        priorities = np.asarray([max(float(ref.priority), PER_EPS)
                                 for ref in self.sequence_refs], dtype=np.float64)
        probs = priorities ** float(alpha)
        probs /= probs.sum()
        chosen = self.rng.choice(len(self.sequence_refs), size=int(batch_size), replace=True, p=probs)
        refs = [self.sequence_refs[int(i)] for i in chosen]
        sampled_probs = probs[chosen]
        weights = (len(self.sequence_refs) * np.maximum(sampled_probs, 1e-12)) ** (-float(beta))
        weights /= max(float(weights.max()), 1e-12)
        return refs, weights.astype(np.float32)

    def update_priorities(self, refs: list[SequenceRef], priorities: np.ndarray) -> None:
        if len(refs) != len(priorities):
            raise ValueError("Priority/ref length mismatch")
        # Sampling is with replacement; if a sequence appears multiple times,
        # keep its largest TD priority from this batch.
        merged: dict[int, tuple[SequenceRef, float]] = {}
        for ref, priority in zip(refs, priorities):
            key = id(ref)
            value = max(float(priority), PER_EPS)
            if key not in merged or value > merged[key][1]:
                merged[key] = (ref, value)
        for ref, priority in merged.values():
            ref.priority = priority


# =============================================================================
# NETWORK BUILDING BLOCKS
# =============================================================================

class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet linear layer."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = NOISY_SIGMA0):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma0 = float(sigma0)

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        sigma = self.sigma0 / math.sqrt(self.in_features)
        nn.init.constant_(self.weight_sigma, sigma)
        nn.init.constant_(self.bias_sigma, self.sigma0 / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features, self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingIQNRecurrentQ(nn.Module):
    """Recurrent dueling quantile Q-network.

    Given observation sequences, produces a distribution over returns for each
    discrete action. Quantile regression is used instead of a scalar Q value.
    """

    def __init__(
        self,
        obs_dim: int,
        actions: int,
        hidden: int = HIDDEN,
        n_quantiles: int = N_QUANTILES,
        quantile_embed_dim: int = QUANTILE_EMBED_DIM,
        noisy_sigma0: float = NOISY_SIGMA0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.actions = int(actions)
        self.hidden = int(hidden)
        self.n_quantiles = int(n_quantiles)
        self.quantile_embed_dim = int(quantile_embed_dim)

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.LayerNorm(hidden),
        )
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)

        for name, parameter in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(parameter, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(parameter, 1.0)

        # Use the frozen R3DQN experiment basis i = 1, ..., n.
        self.cos_basis = torch.arange(
            1, self.quantile_embed_dim + 1, dtype=torch.float32,
        ).mul_(math.pi)
        self.quantile_fc = nn.Linear(self.quantile_embed_dim, hidden)

        self.value_fc = NoisyLinear(hidden, 128, noisy_sigma0)
        self.value_out = NoisyLinear(128, 1, noisy_sigma0)
        self.adv_fc = NoisyLinear(hidden, 128, noisy_sigma0)
        self.adv_out = NoisyLinear(128, actions, noisy_sigma0)

        nn.init.orthogonal_(self.encoder[0].weight, math.sqrt(2.0))
        nn.init.constant_(self.encoder[0].bias, 0.0)
        nn.init.orthogonal_(self.quantile_fc.weight, math.sqrt(2.0))
        nn.init.constant_(self.quantile_fc.bias, 0.0)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def initial_state(self, batch_size: int, device: torch.device):
        return (
            torch.zeros(1, batch_size, self.hidden, device=device),
            torch.zeros(1, batch_size, self.hidden, device=device),
        )

    def _quantile_features(self, taus: torch.Tensor) -> torch.Tensor:
        # taus: [B, T, N]
        basis = self.cos_basis.to(device=taus.device, dtype=taus.dtype)
        cosines = torch.cos(taus.unsqueeze(-1) * basis)
        return F.relu(self.quantile_fc(cosines))

    @staticmethod
    def fixed_taus(batch_size: int, time_steps: int, n: int,
                   device: torch.device) -> torch.Tensor:
        taus = (torch.arange(n, dtype=torch.float32, device=device) + 0.5) / float(n)
        return taus.view(1, 1, n).expand(batch_size, time_steps, n)

    def forward_features(
        self,
        observations: torch.Tensor,
        state,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # observations: [B, T, OBS_DIM]
        encoded = self.encoder(observations)
        output, next_state = self.lstm(encoded, state)
        return output, next_state

    def quantiles_from_latent(
        self,
        latent: torch.Tensor,
        taus: torch.Tensor,
    ) -> torch.Tensor:
        """Return quantile values [B,T,N,A]."""
        phi = self._quantile_features(taus)
        z = latent.unsqueeze(2) * phi
        z = F.relu(z)

        value = F.relu(self.value_fc(z))
        value = self.value_out(value)  # [B,T,N,1]

        advantage = F.relu(self.adv_fc(z))
        advantage = self.adv_out(advantage)  # [B,T,N,A]
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def forward(
        self,
        observations: torch.Tensor,
        state,
        n_quantiles: int | None = None,
        deterministic_quantiles: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        n = int(n_quantiles or self.n_quantiles)
        latent, next_state = self.forward_features(observations, state)
        if deterministic_quantiles:
            taus = self.fixed_taus(observations.shape[0], observations.shape[1], n,
                                   observations.device)
        else:
            taus = torch.rand(observations.shape[0], observations.shape[1], n,
                              device=observations.device).clamp_(1e-4, 1.0 - 1e-4)
        quantiles = self.quantiles_from_latent(latent, taus)
        return quantiles, next_state, taus


# =============================================================================
# POLICY
# =============================================================================

class RecurrentDistributionalDQNPolicy:
    """Single-learner recurrent distributional Q-learning policy.

    The public action loop is compatible with the same PublicTransition API
    used by the existing Vyapti PPO-LSTM/SAC policies.
    """

    def __init__(self, bands: int, seed: int, settings: dict, checkpoint: Path | None = None):
        if type(bands) is not int or bands != N_BANDS:
            raise ValueError(f"Vyapti recurrent DQN expects exactly {N_BANDS} bands")

        self.bands = int(bands)
        self.settings = dict(settings)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.device = self._resolve_device(self.settings.get("device", "auto"))
        if self.device.type == "cpu":
            torch.set_num_threads(int(self.settings.get("cpu_threads", 1)))
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        self.hidden_size = int(self.settings.get("hidden_units", HIDDEN))
        self.lr = float(self.settings.get("learning_rate", LEARNING_RATE))
        self.gamma_base = float(self.settings.get("gamma_per_base_slot", GAMMA_BASE))
        self.n_step = int(self.settings.get("n_step", N_STEP))
        self.batch_size = int(self.settings.get("batch_size", BATCH_SIZE))
        self.replay_capacity_steps = int(
            self.settings.get("replay_capacity_steps", REPLAY_CAPACITY_STEPS)
        )
        self.min_replay_sequences = int(
            self.settings.get("min_replay_sequences", MIN_REPLAY_SEQUENCES)
        )
        self.min_replay_episodes = int(
            self.settings.get("min_replay_episodes", MIN_REPLAY_EPISODES)
        )
        self.updates_per_episode = int(
            self.settings.get("updates_per_episode", UPDATES_PER_EPISODE)
        )
        self.burn_in = int(self.settings.get("burn_in", BURN_IN))
        self.learning_sequence = int(
            self.settings.get("learning_sequence", LEARNING_SEQUENCE)
        )
        self.target_update_interval = int(
            self.settings.get("target_update_interval", TARGET_UPDATE_INTERVAL)
        )
        self.max_grad_norm = float(self.settings.get("max_grad_norm", MAX_GRAD_NORM))
        self.per_alpha = float(self.settings.get("per_alpha", PER_ALPHA))
        self.per_beta_start = float(self.settings.get("per_beta_start", PER_BETA_START))
        self.per_beta_end = float(self.settings.get("per_beta_end", PER_BETA_END))
        self.per_priority_eta = float(self.settings.get("per_priority_eta", PER_PRIORITY_ETA))
        self.n_quantiles = int(self.settings.get("n_quantiles", N_QUANTILES))
        self.n_target_quantiles = int(
            self.settings.get("n_target_quantiles", N_TARGET_QUANTILES)
        )
        self.quantile_embed_dim = int(self.settings.get("quantile_embed_dim", QUANTILE_EMBED_DIM))
        self.quantile_huber_k = float(self.settings.get("quantile_huber_k", QUANTILE_HUBER_K))
        self.noisy_sigma0 = float(self.settings.get("noisy_sigma0", NOISY_SIGMA0))
        self.warmup_actions = int(self.settings.get("warmup_actions", WARMUP_ACTIONS))
        self.sequence_stride = int(self.settings.get("sequence_stride", 8))
        self.recent_hit_window = int(
            self.settings.get("recent_hit_window", RECENT_HIT_WINDOW)
        )
        self.training_base_step_target = int(
            self.settings.get("training_base_step_target", TRAINING_BASE_STEP_TARGET)
        )

        self.p01 = float(self.settings.get("inactive_to_active_probability", 0.02220))
        self.p11 = float(self.settings.get("active_to_active_probability", 0.95753))
        self.prior_active = float(self.settings.get("prior_active_probability", 0.34266))
        self.pd = float(self.settings.get("detection_probability", 0.90))
        self.pfa = float(self.settings.get("false_alarm_probability", 0.05))
        self.period_history = int(self.settings.get("periodicity_history", 32))
        self.base_slot_s = float(self.settings.get("base_slot_seconds", BASE_SLOT_SECONDS))
        if "native_dwell_slots" not in self.settings:
            raise ValueError(
                "native_dwell_slots must be supplied from the frozen 36-band Mode-B profile"
            )
        self.native_dwell_slots = self._validate_dwell_profile(
            self.settings["native_dwell_slots"]
        )
        self.sweep_slots = int(np.sum(self.native_dwell_slots))
        self.native_dwell_feature = self.native_dwell_slots.astype(np.float32) / 2.0
        self.observation_dim = OBS_DIM

        if self.hidden_size < 1 or self.n_step < 1:
            raise ValueError("hidden_units and n_step must be positive")
        if (self.batch_size < 1 or self.learning_sequence < 1 or self.burn_in < 0
                or self.min_replay_sequences < 1 or self.min_replay_episodes < 1
                or self.sequence_stride < 1 or self.recent_hit_window < 1
                or self.training_base_step_target < 1
                or self.n_quantiles < 1 or self.n_target_quantiles < 1
                or self.quantile_embed_dim < 1 or not math.isfinite(self.quantile_huber_k)
                or self.quantile_huber_k <= 0 or not math.isfinite(self.noisy_sigma0)
                or self.noisy_sigma0 <= 0 or not 0 <= self.per_priority_eta <= 1
                or self.updates_per_episode < 1 or self.target_update_interval < 1
                or self.warmup_actions < 0 or self.replay_capacity_steps < 1):
            raise ValueError("Invalid replay sequence settings")
        if not math.isfinite(self.gamma_base) or not 0.0 < self.gamma_base < 1.0:
            raise ValueError("gamma_per_base_slot must be in (0,1)")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("learning_rate must be positive")
        if self.pfa >= self.pd:
            raise ValueError("Pfa must be lower than Pd")
        if not math.isclose(self.base_slot_s, BASE_SLOT_SECONDS, abs_tol=1e-12):
            raise ValueError("base_slot_seconds must be 50 ms")

        self.model = DuelingIQNRecurrentQ(
            OBS_DIM,
            self.bands,
            hidden=self.hidden_size,
            n_quantiles=self.n_quantiles,
            quantile_embed_dim=self.quantile_embed_dim,
            noisy_sigma0=self.noisy_sigma0,
        ).to(self.device)
        self.target_model = DuelingIQNRecurrentQ(
            OBS_DIM,
            self.bands,
            hidden=self.hidden_size,
            n_quantiles=self.n_quantiles,
            quantile_embed_dim=self.quantile_embed_dim,
            noisy_sigma0=self.noisy_sigma0,
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-4)
        self.replay = SequencePrioritizedReplay(
            self.replay_capacity_steps, seed=self.seed + 31337,
            sequence_stride=self.sequence_stride,
        )

        self.total_actions = 0
        self.total_receiver_base_steps = 0
        self.total_updates = 0
        self.completed_training_episodes = 0
        self.evaluation_action_mode = EVAL_MODE
        self.hidden = self.model.initial_state(1, self.device)
        self._pending: dict[str, Any] | None = None
        self._reset_episode_buffers(training=True)

        if checkpoint is not None:
            self._load(Path(checkpoint))

    @staticmethod
    def _resolve_device(value: str) -> torch.device:
        if value not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if value == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device(
            "cuda" if value == "cuda" or (value == "auto" and torch.cuda.is_available()) else "cpu"
        )

    def _validate_dwell_profile(self, profile: Any) -> np.ndarray:
        if not isinstance(profile, (list, tuple)) or len(profile) != self.bands:
            raise ValueError("native_dwell_slots must have one entry per band")
        if any(type(v) is not int or v not in (1, 2) for v in profile):
            raise ValueError("native_dwell_slots entries must be 1 or 2")
        return np.asarray(profile, dtype=np.int64)

    # -------------------------------------------------------------------------
    # Receiver-only causal state
    # -------------------------------------------------------------------------
    def _transition(self, distribution: np.ndarray) -> np.ndarray:
        inactive, active = distribution
        return np.asarray(
            (
                inactive * (1.0 - self.p01) + active * (1.0 - self.p11),
                inactive * self.p01 + active * self.p11,
            ),
            dtype=np.float64,
        )

    def _reset_receiver_state(self) -> None:
        self.belief = np.full(self.bands, self.prior_active, dtype=np.float64)
        self.last_visit_end = np.full(self.bands, -1, dtype=np.int64)
        self.last_hit_slot = np.full(self.bands, -1, dtype=np.int64)
        self.hit_times = [[] for _ in range(self.bands)]
        self.previous_action = -1
        self.elapsed_slots = 0

    def _reset_episode_buffers(self, training: bool) -> None:
        self._reset_receiver_state()
        self.hidden = self.model.initial_state(1, self.device)
        self._pending = None
        self._episode_obs: list[np.ndarray] = []
        self._episode_actions: list[int] = []
        self._episode_rewards: list[float] = []
        self._episode_gammas: list[float] = []
        self._episode_dones: list[float] = []
        self.episode_base_steps = 0
        self.episode_reward = 0.0
        self.episode_decisions = 0
        self.band_selection_counts = np.zeros(self.bands, dtype=np.int64)
        self.recent_hit_history = [
            deque(maxlen=self.recent_hit_window) for _ in range(self.bands)
        ]
        self._raw_staleness_slot_history: list[np.ndarray] = []
        self._recent_hit_rate_history: list[np.ndarray] = []
        self._training_episode = bool(training)
        self._last_losses: dict[str, float] = {}

    def reset_episode(self, *, training: bool) -> None:
        # The replay buffer deliberately survives episode boundaries during
        # training; only the current recurrent/environment state is reset.
        self._reset_episode_buffers(training)

    def _periodicity_features(self, slot: int) -> tuple[np.ndarray, np.ndarray]:
        periodicity = np.zeros(self.bands, dtype=np.float32)
        confidence = np.zeros(self.bands, dtype=np.float32)
        for band, times in enumerate(self.hit_times):
            if len(times) < 3:
                continue
            gaps = np.diff(np.asarray(times, dtype=np.float64))
            gaps = gaps[gaps > 0]
            if len(gaps) < 2:
                continue
            period = float(np.median(gaps))
            if period <= 1.0:
                continue
            cv = float(np.std(gaps) / max(float(np.mean(gaps)), 1e-12))
            certainty = float(
                np.clip(np.exp(-cv) * min(1.0, len(gaps) / 6.0), 0.0, 1.0)
            )
            phase = max(0.0, float(slot) - times[-1]) % period
            phase_error = min(phase, period - phase)
            tolerance = max(0.15 * period, 1.0)
            periodicity[band] = np.float32(
                np.clip(np.exp(-phase_error / tolerance) * certainty, 0.0, 1.0)
            )
            confidence[band] = np.float32(certainty)
        return periodicity, confidence

    def _raw_staleness_slots(self, slot: int) -> np.ndarray:
        """Unclipped diagnostic age; unseen bands age from the episode start."""
        return np.where(
            self.last_visit_end >= 0,
            np.maximum(int(slot) - self.last_visit_end, 0),
            max(int(slot), 0),
        ).astype(np.int64)

    def _features(self, slot: int) -> np.ndarray:
        raw_staleness = np.where(
            self.last_visit_end >= 0,
            slot - self.last_visit_end,
            4 * self.sweep_slots,
        )
        staleness = np.clip(
            raw_staleness / float(4 * self.sweep_slots), 0.0, 1.0,
        ).astype(np.float32)

        periodicity, confidence = self._periodicity_features(slot)

        previous = np.zeros(self.bands, dtype=np.float32)
        if 0 <= self.previous_action < self.bands:
            previous[self.previous_action] = 1.0

        remaining = np.asarray(
            [np.clip(1.0 - slot * self.base_slot_s / MISSION_SECONDS, 0.0, 1.0)],
            dtype=np.float32,
        )

        time_since_hit = np.where(
            self.last_hit_slot >= 0,
            (slot - self.last_hit_slot) / 600.0,
            1.0,
        ).astype(np.float32)
        time_since_hit = np.clip(time_since_hit, 0.0, 1.0)

        total_visits = max(int(self.episode_decisions), 1)
        visit_count_norm = np.clip(
            self.band_selection_counts.astype(np.float32) / float(total_visits),
            0.0, 1.0,
        )
        recent_hit_rate = np.zeros(self.bands, dtype=np.float32)
        for band, history in enumerate(self.recent_hit_history):
            if history:
                recent_hit_rate[band] = np.float32(np.mean(history))

        features = np.concatenate(
            (
                self.belief.astype(np.float32),
                staleness,
                periodicity,
                confidence,
                previous,
                remaining,
                self.native_dwell_feature,
                time_since_hit,
                visit_count_norm,
                recent_hit_rate,
            )
        )
        if features.shape != (OBS_DIM,) or not np.all(np.isfinite(features)):
            raise RuntimeError(f"Invalid receiver-only feature vector: {features.shape}")
        return features

    def _update_belief(self, band: int, hit: bool, slots: int) -> None:
        prior_active = self.belief.copy()
        for current_band in range(self.bands):
            distribution = np.asarray(
                (1.0 - prior_active[current_band], prior_active[current_band]),
                dtype=np.float64,
            )
            end_prior = distribution.copy()
            for _ in range(slots):
                end_prior = self._transition(end_prior)

            if current_band == band:
                no_hit = distribution.copy()
                no_hit_likelihood = np.asarray(
                    (1.0 - self.pfa, 1.0 - self.pd), dtype=np.float64
                )
                for _ in range(slots):
                    no_hit = self._transition(no_hit * no_hit_likelihood)
                posterior = end_prior - no_hit if hit else no_hit
                normalizer = float(posterior.sum())
                if not math.isfinite(normalizer) or normalizer <= 0.0:
                    raise RuntimeError("Aggregate hit/miss has zero HMM likelihood")
                end_prior = posterior / normalizer

            self.belief[current_band] = np.clip(end_prior[1], 1e-6, 1.0 - 1e-6)

    # -------------------------------------------------------------------------
    # Action loop
    # -------------------------------------------------------------------------
    def _q_values_from_current_state(self, features: np.ndarray, *, training: bool) -> np.ndarray:
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device).view(1, 1, -1)
        self.model.reset_noise()
        with torch.no_grad():
            quantiles, next_hidden, _ = self.model(
                x, self.hidden, self.n_quantiles,
                deterministic_quantiles=not training,
            )
            q = quantiles.mean(dim=2)[0, 0]
        self.hidden = tuple(part.detach() for part in next_hidden)
        return q.detach().cpu().numpy()

    def select_action(self, state, *, training: bool) -> int:
        if self._pending is not None:
            raise RuntimeError("observe must follow each selected action")

        slot = int(state.time_slot)
        if slot != int(self.elapsed_slots):
            # We only rely on receiver-causal elapsed time; fail closed on clock drift.
            raise RuntimeError(
                f"Recurrent DQN receiver clock drift: policy={self.elapsed_slots}, env={slot}"
            )

        features = self._features(slot)
        self._raw_staleness_slot_history.append(self._raw_staleness_slots(slot))
        self._recent_hit_rate_history.append(features[8 * self.bands + 1:9 * self.bands + 1].copy())
        self.model.train(bool(training))
        q_values = self._q_values_from_current_state(features, training=training)

        if training and self.total_actions < self.warmup_actions:
            action = int(self.rng.integers(self.bands))
        elif training:
            action = int(np.argmax(q_values))
        else:
            # Deterministic greedy evaluation is the primary benchmark mode.
            action = int(np.argmax(q_values))

        self._pending = {
            "features": features,
            "action": action,
            "training": bool(training),
        }
        self.band_selection_counts[action] += 1
        self.episode_decisions += 1
        if training:
            self.total_actions += 1
        return action

    def _commit_receiver_observation(
        self,
        band: int,
        hit: bool,
        slots: int,
    ) -> None:
        self._update_belief(band, hit, slots)
        end_slot = int(self.elapsed_slots + slots)

        if hit:
            times = self.hit_times[band]
            times.append(float(end_slot))
            if len(times) > self.period_history:
                del times[:-self.period_history]
            self.last_hit_slot[band] = end_slot

        self.recent_hit_history[band].append(1.0 if hit else 0.0)

        self.last_visit_end[band] = end_slot
        self.previous_action = band
        self.elapsed_slots = end_slot

    def observe(self, transition: PublicTransition, *, training: bool) -> None:
        if self._pending is None:
            raise RuntimeError("select_action must precede observe")
        if transition.action != self._pending["action"]:
            raise ValueError("Transition action differs from selected action")

        observation = transition.next_state.previous_observation
        if observation is None:
            raise ValueError("Recurrent DQN requires a receiver observation after each action")

        band = int(observation.selected_band)
        slots = int(transition.next_state.time_slot) - int(transition.state.time_slot)
        if not 0 <= band < self.bands:
            raise ValueError("Receiver observation selected an invalid band")
        if slots not in (1, 2):
            raise ValueError("Current Vyapti world must use 1- or 2-slot native dwell")

        # Commit ONLY receiver-visible information.
        self._commit_receiver_observation(band, bool(observation.hit), slots)

        # Environment reward is retained for RL training, but never becomes a
        # policy observation. This preserves strict receiver-only evaluation.
        reward = float(transition.reward)
        self.episode_reward += reward

        if training:
            self.total_receiver_base_steps += slots
            self.episode_base_steps += slots
            self._episode_obs.append(self._pending["features"])
            self._episode_actions.append(int(transition.action))
            self._episode_rewards.append(reward)
            self._episode_gammas.append(float(self.gamma_base ** slots))
            self._episode_dones.append(
                float(transition.terminated or transition.truncated)
            )
            if transition.terminated or transition.truncated:
                # Store the post-transition observation so terminal rewards
                # have their actual successor state and boundary in replay.
                self._episode_obs.append(self._features(self.elapsed_slots))

        self._pending = None

    # -------------------------------------------------------------------------
    # Recurrent prioritized replay training
    # -------------------------------------------------------------------------
    def _episode_from_buffers(self) -> Episode | None:
        if not self._episode_actions:
            return None
        if (len(self._episode_obs) != len(self._episode_actions) + 1
                or not self._episode_dones[-1]):
            raise RuntimeError("Training replay episode must include its terminal next observation")
        return Episode(
            obs=np.asarray(self._episode_obs, dtype=np.float32),
            actions=np.asarray(self._episode_actions, dtype=np.int64),
            rewards=np.asarray(self._episode_rewards, dtype=np.float32),
            gammas=np.asarray(self._episode_gammas, dtype=np.float32),
            dones=np.asarray(self._episode_dones, dtype=np.float32),
        )

    def _current_per_beta(self) -> float:
        denom = max(float(self.settings.get(
            "training_base_step_target", TRAINING_BASE_STEP_TARGET,
        )), 1.0)
        frac = np.clip(self.total_receiver_base_steps / denom, 0.0, 1.0)
        return float(self.per_beta_start + frac * (self.per_beta_end - self.per_beta_start))

    def _sample_sequences(self):
        return self.replay.sample(
            self.batch_size,
            self.per_alpha,
            self._current_per_beta(),
        )

    def _materialize_sequences(self, refs: list[SequenceRef]):
        """Build padded learning windows and variable-length burn-in prefixes."""
        batch = len(refs)
        learning_observations = self.learning_sequence + self.n_step
        learning_transitions = learning_observations - 1
        observations = np.zeros((batch, learning_observations, OBS_DIM), dtype=np.float32)
        actions = np.zeros((batch, learning_transitions), dtype=np.int64)
        rewards = np.zeros((batch, learning_transitions), dtype=np.float32)
        gammas = np.zeros((batch, learning_transitions), dtype=np.float32)
        dones = np.ones((batch, learning_transitions), dtype=np.float32)
        learning_mask = np.zeros((batch, self.learning_sequence), dtype=np.float32)
        burn_prefixes: list[np.ndarray] = []

        for row, ref in enumerate(refs):
            episode = ref.episode
            start = int(ref.start)
            burn_start = max(0, start - self.burn_in)
            burn_prefixes.append(episode.obs[burn_start:start])

            observation_count = min(learning_observations, episode.steps + 1 - start)
            observations[row, :observation_count] = episode.obs[start:start + observation_count]

            transition_count = min(learning_transitions, episode.steps - start)
            if transition_count:
                actions[row, :transition_count] = episode.actions[start:start + transition_count]
                rewards[row, :transition_count] = episode.rewards[start:start + transition_count]
                gammas[row, :transition_count] = episode.gammas[start:start + transition_count]
                dones[row, :transition_count] = episode.dones[start:start + transition_count]

            valid_learning = min(self.learning_sequence, episode.steps - start)
            learning_mask[row, :valid_learning] = 1.0

        return burn_prefixes, observations, actions, rewards, gammas, dones, learning_mask

    @staticmethod
    def _build_n_step_targets(
        rewards: torch.Tensor,
        gammas: torch.Tensor,
        dones: torch.Tensor,
        burn_in: int,
        learn_len: int,
        n_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [B,L] n-step rewards and [B,L] n-step discounts."""
        b = rewards.shape[0]
        device = rewards.device
        result = torch.zeros(b, learn_len, device=device, dtype=rewards.dtype)
        discount = torch.ones(b, learn_len, device=device, dtype=rewards.dtype)
        alive = torch.ones(b, learn_len, device=device, dtype=rewards.dtype)

        for k in range(n_step):
            rs = rewards[:, burn_in + k: burn_in + k + learn_len]
            gs = gammas[:, burn_in + k: burn_in + k + learn_len]
            ds = dones[:, burn_in + k: burn_in + k + learn_len]
            result = result + discount * alive * rs
            discount = discount * gs
            alive = alive * (1.0 - ds)

        return result, discount * alive

    @staticmethod
    def _quantile_huber_loss(
        current_quantiles: torch.Tensor,   # [B,L,N]
        target_quantiles: torch.Tensor,    # [B,L,Nt]
        current_taus: torch.Tensor,        # [B,L,N]
        huber_k: float = QUANTILE_HUBER_K,
    ) -> torch.Tensor:
        td = target_quantiles.unsqueeze(-1) - current_quantiles.unsqueeze(-2)  # [B,L,Nt,N]
        abs_td = td.abs()
        huber = torch.where(
            abs_td <= huber_k,
            0.5 * td.pow(2),
            huber_k * (abs_td - 0.5 * huber_k),
        )
        tau = current_taus.unsqueeze(-2)
        weight = torch.abs(tau - (td.detach() < 0.0).float())
        return (weight * huber / huber_k).mean(dim=(-2, -1))

    def _train_batch(self) -> dict[str, float]:
        refs, importance = self._sample_sequences()
        (burn_prefixes, obs_np, actions_np, rewards_np, gammas_np,
         dones_np, learning_mask_np) = self._materialize_sequences(refs)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions_np, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)
        gammas = torch.as_tensor(gammas_np, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones_np, dtype=torch.float32, device=self.device)
        learning_mask = torch.as_tensor(learning_mask_np, dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(importance, dtype=torch.float32, device=self.device)

        self.model.train(True)
        self.target_model.train(True)
        self.model.reset_noise()
        self.target_model.reset_noise()

        # ------------------------------------------------------------------
        # Rebuild each recurrent state from a no-gradient burn-in prefix.
        # Online network: current quantiles and next-state action selection.
        # ------------------------------------------------------------------
        online_h = []
        online_c = []
        target_h = []
        target_c = []
        with torch.no_grad():
            for prefix in burn_prefixes:
                online_state = self.model.initial_state(1, self.device)
                target_state = self.target_model.initial_state(1, self.device)
                if len(prefix):
                    prefix_tensor = torch.as_tensor(prefix, dtype=torch.float32,
                                                    device=self.device).unsqueeze(0)
                    _, online_state = self.model.forward_features(prefix_tensor, online_state)
                    _, target_state = self.target_model.forward_features(prefix_tensor, target_state)
                online_h.append(online_state[0])
                online_c.append(online_state[1])
                target_h.append(target_state[0])
                target_c.append(target_state[1])
        online_h0 = (torch.cat(online_h, dim=1), torch.cat(online_c, dim=1))
        target_h0 = (torch.cat(target_h, dim=1), torch.cat(target_c, dim=1))
        online_sequence = obs
        online_all, _, online_taus = self.model(online_sequence, online_h0, self.n_quantiles)
        current_end = self.learning_sequence
        next_start = self.n_step
        next_end = next_start + self.learning_sequence

        current_quantiles_all = online_all[:, :current_end]
        next_quantiles_all = online_all[:, next_start:next_end]
        next_mean = next_quantiles_all.mean(dim=2)
        next_actions = next_mean.argmax(dim=-1)  # [B,L]

        current_actions = actions[:, :self.learning_sequence]
        gather_index = current_actions.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, self.n_quantiles, 1
        )
        current_quantiles = current_quantiles_all.gather(-1, gather_index).squeeze(-1)
        current_taus = online_taus[:, :current_end]

        # ------------------------------------------------------------------
        # Target distribution for Double-DQN n-step bootstrap.
        # ------------------------------------------------------------------
        with torch.no_grad():
            target_all, _, _ = self.target_model(
                online_sequence, target_h0, self.n_target_quantiles
            )
            target_next_all = target_all[:, next_start:next_end]
            target_gather = next_actions.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, self.n_target_quantiles, 1
            )
            target_next_quantiles = target_next_all.gather(-1, target_gather).squeeze(-1)

            reward_seq = rewards
            gamma_seq = gammas
            done_seq = dones
            nstep_reward, nstep_discount = self._build_n_step_targets(
                reward_seq,
                gamma_seq,
                done_seq,
                0,
                self.learning_sequence,
                self.n_step,
            )
            target_quantiles = nstep_reward.unsqueeze(-1) + nstep_discount.unsqueeze(-1) * target_next_quantiles

        per_step_loss = self._quantile_huber_loss(
            current_quantiles,
            target_quantiles,
            current_taus,
            self.quantile_huber_k,
        )
        per_sample_loss = ((per_step_loss * learning_mask).sum(dim=1)
                           / learning_mask.sum(dim=1).clamp_min(1.0))
        loss = (per_sample_loss * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm).item()
        )
        self.optimizer.step()

        with torch.no_grad():
            step_td = (
                target_quantiles.mean(dim=-1) - current_quantiles.mean(dim=-1)
            ).abs()
            masked_td = step_td.masked_fill(learning_mask <= 0.0, 0.0)
            valid_count = learning_mask.sum(dim=1).clamp_min(1.0)
            priority_values = (
                self.per_priority_eta * masked_td.max(dim=-1).values
                + (1.0 - self.per_priority_eta) * masked_td.sum(dim=-1) / valid_count
                + PER_EPS
            ).detach().cpu().numpy()
        self.replay.update_priorities(refs, priority_values)

        self.total_updates += 1
        if self.total_updates % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return {
            "loss": float(loss.item()),
            "td_error": float(priority_values.mean()),
            "grad_norm": grad_norm,
            "q_value": float(current_quantiles.mean().item()),
            "target_value": float(target_quantiles.mean().item()),
            "per_beta": float(self._current_per_beta()),
        }

    def _learn_episode(self) -> dict[str, float]:
        ep = self._episode_from_buffers()
        if ep is not None:
            self.replay.add(ep)

        minimum_steps = self.burn_in + self.learning_sequence + self.n_step
        eligible_episodes = sum(
            1 for episode in self.replay.episodes if episode.steps >= minimum_steps
        )
        eligible_sequences = len(self.replay.sequence_refs)
        if (eligible_episodes < self.min_replay_episodes
                or eligible_sequences < self.min_replay_sequences):
            return {
                "loss": float("nan"),
                "td_error": float("nan"),
                "grad_norm": float("nan"),
                "q_value": float("nan"),
                "target_value": float("nan"),
                "per_beta": float(self._current_per_beta()),
            }

        stats: list[dict[str, float]] = []
        for _ in range(self.updates_per_episode):
            stats.append(self._train_batch())

        keys = stats[0].keys()
        return {k: float(np.nanmean([s[k] for s in stats])) for k in keys}

    def end_episode(self, *, training: bool) -> dict:
        if self._pending is not None:
            raise RuntimeError("Cannot end an episode before observing its final action")

        if training and self._episode_rewards:
            self._last_losses = self._learn_episode()
            self.completed_training_episodes += 1
        else:
            self._last_losses = {}

        selection_counts = self.band_selection_counts
        total_selections = int(selection_counts.sum())
        top_counts = np.sort(selection_counts)[::-1]
        staleness = (np.asarray(self._raw_staleness_slot_history, dtype=np.int64)
                     if self._raw_staleness_slot_history else np.empty((0, self.bands), dtype=np.int64))
        # Include the post-action terminal state so ages accumulated during the
        # final dwell are visible in the episode-level diagnostic.
        staleness = np.concatenate(
            (staleness, self._raw_staleness_slots(self.elapsed_slots)[None, :]), axis=0,
        )
        recent_hits = (np.asarray(self._recent_hit_rate_history, dtype=np.float32)
                       if self._recent_hit_rate_history else np.empty((0, self.bands), dtype=np.float32))
        return {
            "algorithm": "recurrent_distributional_dqn",
            "algorithm_family": "R2D2/Rainbow-inspired recurrent value-based RL",
            "training_updates": int(bool(self._last_losses and np.isfinite(next(iter(self._last_losses.values()))))) * self.updates_per_episode,
            "total_training_actions": int(self.total_actions),
            "total_policy_decisions": int(self.total_actions),
            "total_receiver_base_steps": int(self.total_receiver_base_steps),
            "completed_training_episodes": int(self.completed_training_episodes),
            "episode_decisions": int(self.episode_decisions),
            "episode_base_steps": int(self.elapsed_slots),
            "episode_reward": float(self.episode_reward),
            "replay_steps": int(len(self.replay)),
            "replay_episodes": int(self.replay.num_episodes),
            "replay_sequences": int(len(self.replay.sequence_refs)),
            "band_selection_counts": self.band_selection_counts.tolist(),
            "unique_bands_visited": int(np.count_nonzero(selection_counts)),
            "top_1_action_concentration": float(top_counts[:1].sum() / max(total_selections, 1)),
            "top_3_action_concentration": float(top_counts[:3].sum() / max(total_selections, 1)),
            "top_5_action_concentration": float(top_counts[:5].sum() / max(total_selections, 1)),
            "never_visited_bands": int(np.sum(selection_counts == 0)),
            "mean_staleness_slots": float(np.mean(staleness)),
            "maximum_staleness_slots": int(np.max(staleness)),
            "mean_recent_hit_rate": (
                float(np.mean(recent_hits)) if recent_hits.size else 0.0
            ),
            "max_recent_hit_rate": (
                float(np.max(recent_hits)) if recent_hits.size else 0.0
            ),
            "per_beta": float(self._current_per_beta()),
            **{f"mean_{k}": v for k, v in self._last_losses.items()},
        }

    # -------------------------------------------------------------------------
    # Checkpoints
    # -------------------------------------------------------------------------
    def save(self, path) -> None:
        payload = {
            "format": CHECKPOINT_FORMAT,
            "bands": self.bands,
            "observation_schema": OBSERVATION_SCHEMA,
            "obs_dim": OBS_DIM,
            "settings": self.settings,
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_actions": self.total_actions,
            "total_receiver_base_steps": self.total_receiver_base_steps,
            "total_updates": self.total_updates,
            "completed_training_episodes": self.completed_training_episodes,
            "rng_state": self.rng.bit_generator.state,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            torch.save(payload, stream)

    def _load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Checkpoint format does not match recurrent distributional DQN")
        if payload.get("bands") != self.bands or payload.get("obs_dim") != OBS_DIM:
            raise ValueError("Checkpoint action-space or observation dimension does not match")
        if payload.get("observation_schema") != OBSERVATION_SCHEMA:
            raise ValueError("Checkpoint observation schema does not match; train a fresh run")
        self.model.load_state_dict(payload["model"])
        self.target_model.load_state_dict(payload.get("target_model", payload["model"]))
        self.target_model.eval()
        self.optimizer.load_state_dict(payload["optimizer"])
        self.total_actions = int(payload.get("total_actions", 0))
        self.total_receiver_base_steps = int(payload.get("total_receiver_base_steps", 0))
        self.total_updates = int(payload.get("total_updates", 0))
        self.completed_training_episodes = int(payload.get("completed_training_episodes", 0))
        if "rng_state" in payload:
            self.rng.bit_generator.state = payload["rng_state"]

    def set_evaluation_action_mode(self, mode: str, seed: int | None = None) -> None:
        if mode != "greedy":
            raise ValueError("This value-based benchmark uses deterministic greedy evaluation")
        self.evaluation_action_mode = mode


def create(*, bands: int, seed: int, settings: dict, checkpoint=None):
    return RecurrentDistributionalDQNPolicy(bands, seed, settings, checkpoint)


def manifest() -> dict[str, Any]:
    return {
        "algorithm": "recurrent_distributional_dqn",
        "algorithm_family": "DRQN + Rainbow/R2D2-inspired",
        "obs_dim": OBS_DIM,
        "actions": N_BANDS,
        "hidden_units": HIDDEN,
        "learning_rate": LEARNING_RATE,
        "gamma_per_base_slot": GAMMA_BASE,
        "n_step": N_STEP,
        "batch_size": BATCH_SIZE,
        "replay_capacity_steps": REPLAY_CAPACITY_STEPS,
        "min_replay_sequences": MIN_REPLAY_SEQUENCES,
        "min_replay_episodes": MIN_REPLAY_EPISODES,
        "updates_per_episode": UPDATES_PER_EPISODE,
        "burn_in": BURN_IN,
        "learning_sequence": LEARNING_SEQUENCE,
        "target_update_interval": TARGET_UPDATE_INTERVAL,
        "per_alpha": PER_ALPHA,
        "per_beta_start": PER_BETA_START,
        "per_beta_end": PER_BETA_END,
        "per_priority_eta": PER_PRIORITY_ETA,
        "n_quantiles": N_QUANTILES,
        "n_target_quantiles": N_TARGET_QUANTILES,
        "noisy_exploration": True,
        "warmup_actions": WARMUP_ACTIONS,
        "training_base_step_target": TRAINING_BASE_STEP_TARGET,
        "recent_hit_window": RECENT_HIT_WINDOW,
        "observation_schema": OBSERVATION_SCHEMA,
        "reward_mode": REWARD_MODE,
        "reward_in_policy_observation": False,
        "evaluation_action_mode": "greedy",
        "distributed_r2d2": False,
    }


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2))

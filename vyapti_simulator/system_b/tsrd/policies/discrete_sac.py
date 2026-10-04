"""Discrete Soft Actor-Critic for the closed TSRD scheduler interface.

The policy consumes only PublicState/ReceiverObservation data and the scalar
reward emitted by the shared Mode-B runner. World generation, reward
calculation, scorecards, validation and test replay stay in experiment.py.
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..algorithm_interface import PublicTransition


class _Actor(nn.Module):
    def __init__(self, observation_dim: int, actions: int, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(observation_dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, actions))

    def forward(self, observation):
        logits = self.layers(observation)
        return F.softmax(logits, dim=-1), F.log_softmax(logits, dim=-1)


class _Critic(nn.Module):
    def __init__(self, observation_dim: int, actions: int, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(observation_dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, actions))

    def forward(self, observation):
        return self.layers(observation)


class _Replay:
    def __init__(self, capacity: int, observation_dim: int, rng):
        self.capacity = capacity
        self.observations = np.empty((capacity, observation_dim), np.float32)
        self.next_observations = np.empty((capacity, observation_dim), np.float32)
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.terminals = np.empty(capacity, np.float32)
        self.discounts = np.empty(capacity, np.float32)
        self.rng = rng
        self.size = self.position = 0

    def add(self, observation, action, reward, next_observation, terminal, discount):
        i = self.position
        self.observations[i] = observation
        self.next_observations[i] = next_observation
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminals[i] = terminal
        self.discounts[i] = discount
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count: int, device):
        indices = self.rng.integers(0, self.size, size=count)
        tensor = lambda values, dtype=None: torch.as_tensor(values[indices], dtype=dtype, device=device)
        return (tensor(self.observations, torch.float32), tensor(self.actions, torch.long),
                tensor(self.rewards, torch.float32), tensor(self.next_observations, torch.float32),
                tensor(self.terminals, torch.float32), tensor(self.discounts, torch.float32))


class DiscreteSACPolicy:
    """Twin-critic discrete SAC with a causal, receiver-only HMM feature state."""

    def __init__(self, bands: int, seed: int, settings: dict, checkpoint: Path | None = None):
        if bands < 2:
            raise ValueError("Discrete SAC needs at least two available bands")
        self.bands = int(bands)
        self.settings = dict(settings)
        self.base_slot_s = float(self.settings.get("base_slot_seconds", 0.05))
        self.pd = float(self.settings.get("detection_probability", 0.90))
        self.pfa = float(self.settings.get("false_alarm_probability", 0.05))
        self.p01 = float(self.settings.get("inactive_to_active_probability", 0.05))
        self.p11 = float(self.settings.get("active_to_active_probability", 0.90))
        self.gamma_base = float(self.settings.get("gamma_per_base_slot", 0.997))
        self.tau = float(self.settings.get("target_update_tau", 0.005))
        self.warmup_actions = int(self.settings.get("warmup_actions", 5000))
        self.batch_size = int(self.settings.get("batch_size", 256))
        self.update_every = int(self.settings.get("update_every_actions", 1))
        self.updates_per_action = int(self.settings.get("updates_per_action", 1))
        self.hidden = int(self.settings.get("hidden_units", 256))
        self.target_entropy_fraction = float(self.settings.get("target_entropy_fraction", 0.98))
        self.initial_alpha = float(self.settings.get("initial_alpha", 0.20))
        self.capacity = int(self.settings.get("replay_capacity", 50_000))
        self.lr_actor = float(self.settings.get("actor_learning_rate", 3e-4))
        self.lr_critic = float(self.settings.get("critic_learning_rate", 3e-4))
        self.lr_alpha = float(self.settings.get("temperature_learning_rate", 3e-4))
        for name, value in (("Pd", self.pd), ("Pfa", self.pfa), ("P01", self.p01), ("P11", self.p11),
                            ("gamma", self.gamma_base), ("tau", self.tau)):
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and strictly between zero and one")
        if self.pfa >= self.pd:
            raise ValueError("Detector Pfa must be lower than Pd")
        if min(self.capacity, self.batch_size, self.update_every, self.updates_per_action, self.hidden) < 1:
            raise ValueError("Replay, batch, update, and network sizes must be positive")
        device_setting = self.settings.get("device", "auto")
        if device_setting not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if device_setting == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Discrete SAC requested CUDA, but CUDA is unavailable")
        self.device = torch.device("cuda" if device_setting == "cuda" or
                                   (device_setting == "auto" and torch.cuda.is_available()) else "cpu")
        if self.device.type == "cpu":
            cpu_threads = int(self.settings.get("cpu_threads", 1))
            if cpu_threads < 1:
                raise ValueError("cpu_threads must be positive")
            torch.set_num_threads(cpu_threads)
        self.rng = np.random.default_rng(int(seed))
        self.evaluation_action_mode = "greedy"
        self.evaluation_rng = np.random.default_rng(int(seed))
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        observation_dim = self.bands * 3 + 1
        self.actor = _Actor(observation_dim, self.bands, self.hidden).to(self.device)
        self.q1 = _Critic(observation_dim, self.bands, self.hidden).to(self.device)
        self.q2 = _Critic(observation_dim, self.bands, self.hidden).to(self.device)
        self.target_q1 = _Critic(observation_dim, self.bands, self.hidden).to(self.device)
        self.target_q2 = _Critic(observation_dim, self.bands, self.hidden).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=self.lr_critic)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=self.lr_critic)
        self.log_alpha = torch.tensor(math.log(self.initial_alpha), dtype=torch.float32,
                                      device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)
        self.target_entropy = self.target_entropy_fraction * math.log(self.bands)
        # Evaluation policies never optimize, so do not reserve the full
        # training replay allocation for every reconstructed validation world.
        replay_capacity = self.capacity if checkpoint is None else min(self.capacity, self.batch_size)
        self.replay = _Replay(replay_capacity, observation_dim, self.rng)
        self.total_actions = self.total_updates = 0
        self._loss_window = deque(maxlen=100)
        self._recent_reward = deque(maxlen=100)
        self.reset_episode(training=True)
        if checkpoint is not None:
            self._load(checkpoint)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def reset_episode(self, *, training: bool) -> None:
        self.belief = np.full(self.bands, 0.5, dtype=np.float64)
        self.last_visit = np.full(self.bands, -1, dtype=np.int64)
        self.previous_action = -1
        self.episode_actions = 0
        self.episode_reward = 0.0
        self.episode_counts = np.zeros(self.bands, dtype=np.int64)
        self._pending_observation = None

    def _features(self, slot: int) -> np.ndarray:
        staleness = np.where(self.last_visit >= 0,
                             (slot - self.last_visit) / max(1, int(round(30.0 / self.base_slot_s))), 1.0)
        current = np.zeros(self.bands, dtype=np.float32)
        if self.previous_action >= 0:
            current[self.previous_action] = 1.0
        remaining = np.clip(1.0 - slot * self.base_slot_s / 30.0, 0.0, 1.0)
        return np.concatenate((self.belief.astype(np.float32), np.clip(staleness, 0, 1).astype(np.float32),
                               current, np.asarray([remaining], dtype=np.float32)))

    def select_action(self, state, *, training: bool) -> int:
        features = self._features(state.time_slot)
        self._pending_observation = features
        if training and self.total_actions < self.warmup_actions:
            action = int(self.rng.integers(self.bands))
        else:
            x = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                probabilities, _ = self.actor(x)
                if training:
                    action = int(torch.multinomial(probabilities, 1, generator=None).item())
                elif self.evaluation_action_mode == "sampled":
                    action = int(self.evaluation_rng.choice(self.bands, p=probabilities[0].cpu().numpy()))
                else:
                    action = int(torch.argmax(probabilities, dim=-1).item())
        self.episode_counts[action] += 1
        return action

    def set_evaluation_action_mode(self, mode: str, seed: int | None = None) -> None:
        if mode not in {"greedy", "sampled"}:
            raise ValueError("Evaluation action mode must be greedy or sampled")
        if mode == "sampled" and (type(seed) is not int or seed < 0):
            raise ValueError("Sampled evaluation requires a nonnegative integer action seed")
        self.evaluation_action_mode = mode
        if seed is not None:
            self.evaluation_rng = np.random.default_rng(seed)

    def _update_belief(self, observation, slots: int) -> None:
        band = int(observation.selected_band)
        if not 0 <= band < self.bands:
            raise ValueError("Receiver observation selected a band outside this policy's action space")
        if slots < 1:
            raise ValueError("An aggregate receiver observation must cover at least one base slot")

        def transition(distribution: np.ndarray) -> np.ndarray:
            inactive, active = distribution
            return np.asarray((inactive * (1.0 - self.p01) + active * (1.0 - self.p11),
                               inactive * self.p01 + active * self.p11), dtype=np.float64)

        for b in range(self.bands):
            prior = np.asarray((1.0 - self.belief[b], self.belief[b]), dtype=np.float64)
            end_prior = prior.copy()
            for _ in range(slots):
                end_prior = transition(end_prior)
            if b == band:
                # The runner reports only whether any of the per-slot receiver
                # outcomes was positive. Track the joint no-hit path through
                # each latent state and transition, then obtain hit paths by
                # subtracting from the unconditional end-state distribution.
                no_hit = prior.copy()
                no_hit_likelihood = np.asarray((1.0 - self.pfa, 1.0 - self.pd))
                for _ in range(slots):
                    no_hit = transition(no_hit * no_hit_likelihood)
                posterior = end_prior - no_hit if observation.hit else no_hit
                normalizer = float(posterior.sum())
                if normalizer <= 0.0 or not math.isfinite(normalizer):
                    raise RuntimeError("Aggregate receiver observation has zero HMM likelihood")
                end_prior = posterior / normalizer
            self.belief[b] = np.clip(end_prior[1], 1e-6, 1 - 1e-6)
        self.last_visit[band] = int(observation.time_slot) + 1
        self.previous_action = band

    def _elapsed_slots(self, transition: PublicTransition) -> int:
        """Use the environment's state clock as the source of elapsed time."""
        elapsed = int(transition.next_state.time_slot) - int(transition.state.time_slot)
        if elapsed < 1:
            raise ValueError("The environment state clock must advance by at least one base slot")
        return elapsed

    def _transition_discount(self, transition: PublicTransition) -> float:
        return self.gamma_base ** self._elapsed_slots(transition)

    def _update_temperature(self, entropy: torch.Tensor) -> torch.Tensor:
        """Adjust alpha in the direction that restores target policy entropy."""
        alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        nn.utils.clip_grad_norm_([self.log_alpha], 10.0)
        self.alpha_optimizer.step()
        return alpha_loss

    def observe(self, transition: PublicTransition, *, training: bool) -> None:
        if self._pending_observation is None:
            raise RuntimeError("select_action must precede observe for each transition")
        observation = transition.next_state.previous_observation
        if observation is None:
            raise ValueError("Discrete SAC requires a receiver observation after each action")
        slots = self._elapsed_slots(transition)
        if observation.dwell_elapsed_s <= 0:
            raise ValueError("Receiver observation must report positive usable dwell")
        self._update_belief(observation, slots)
        next_features = self._features(transition.next_state.time_slot)
        # The environment has already advanced the base-slot clock through
        # dwell and retune. Do not add retune_cost_s to the discount again.
        discount = self._transition_discount(transition)
        self.episode_actions += 1
        self.total_actions += 1
        self.episode_reward += float(transition.reward)
        self._recent_reward.append(float(transition.reward))
        if training:
            self.replay.add(self._pending_observation, transition.action, transition.reward,
                            next_features, transition.terminated or transition.truncated, discount)
            if (self.total_actions >= self.warmup_actions and self.replay.size >= self.batch_size
                    and self.total_actions % self.update_every == 0):
                for _ in range(self.updates_per_action):
                    self._loss_window.append(self._gradient_update())
        self._pending_observation = None

    def _gradient_update(self) -> dict:
        obs, actions, rewards, next_obs, terminals, discounts = self.replay.sample(self.batch_size, self.device)
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_obs)
            next_q = torch.minimum(self.target_q1(next_obs), self.target_q2(next_obs))
            soft_value = (next_probs * (next_q - self.alpha.detach() * next_log_probs)).sum(dim=-1)
            target = rewards + (1.0 - terminals) * discounts * soft_value
        q1_values = self.q1(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_values = self.q2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss_q1 = F.mse_loss(q1_values, target)
        loss_q2 = F.mse_loss(q2_values, target)
        self.q1_optimizer.zero_grad(set_to_none=True)
        loss_q1.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), 10.0)
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad(set_to_none=True)
        loss_q2.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), 10.0)
        self.q2_optimizer.step()
        probs, log_probs = self.actor(obs)
        with torch.no_grad():
            q_values = torch.minimum(self.q1(obs), self.q2(obs))
        actor_loss = (probs * (self.alpha.detach() * log_probs - q_values)).sum(dim=-1).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1)
        alpha_loss = self._update_temperature(entropy)
        with torch.no_grad():
            for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
                target_param.mul_(1.0 - self.tau).add_(self.tau * param)
            for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.mul_(1.0 - self.tau).add_(self.tau * param)
        self.total_updates += 1
        return {"critic1_loss": float(loss_q1.item()), "critic2_loss": float(loss_q2.item()),
                "actor_loss": float(actor_loss.item()), "temperature_loss": float(alpha_loss.item()),
                "alpha": float(self.alpha.item()), "policy_entropy": float(entropy.mean().item())}

    def end_episode(self, *, training: bool) -> dict:
        result = {"algorithm": "discrete_sac", "training_updates": self.total_updates if training else 0,
                  "total_training_actions": self.total_actions, "episode_decisions": self.episode_actions,
                  "episode_reward": self.episode_reward, "replay_size": self.replay.size,
                  "alpha": float(self.alpha.detach().cpu()), "band_selection_counts": self.episode_counts.tolist(),
                  "mean_recent_step_reward": float(np.mean(self._recent_reward)) if self._recent_reward else None}
        if self._loss_window:
            for key in self._loss_window[0]:
                result[f"mean_{key}"] = float(np.mean([row[key] for row in self._loss_window]))
        return result

    def save(self, path) -> None:
        torch.save({"format": "vyapti_discrete_sac_v1", "bands": self.bands,
                    "settings": self.settings, "actor": self.actor.state_dict(),
                    "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                    "target_q1": self.target_q1.state_dict(), "target_q2": self.target_q2.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "q1_optimizer": self.q1_optimizer.state_dict(), "q2_optimizer": self.q2_optimizer.state_dict(),
                    "log_alpha": self.log_alpha.detach().cpu(),
                    "alpha_optimizer": self.alpha_optimizer.state_dict(),
                    "total_actions": self.total_actions, "total_updates": self.total_updates,
                    "rng_state": self.rng.bit_generator.state}, Path(path))

    def _load(self, path: Path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload.get("format") != "vyapti_discrete_sac_v1" or payload.get("bands") != self.bands:
            raise ValueError("SAC checkpoint format or action-space size does not match this run")
        self.actor.load_state_dict(payload["actor"])
        self.q1.load_state_dict(payload["q1"])
        self.q2.load_state_dict(payload["q2"])
        self.target_q1.load_state_dict(payload["target_q1"])
        self.target_q2.load_state_dict(payload["target_q2"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.q1_optimizer.load_state_dict(payload["q1_optimizer"])
        self.q2_optimizer.load_state_dict(payload["q2_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        self.total_actions = int(payload.get("total_actions", 0))
        self.total_updates = int(payload.get("total_updates", 0))
        if "rng_state" in payload:
            self.rng.bit_generator.state = payload["rng_state"]


def create(*, bands: int, seed: int, settings: dict, checkpoint=None):
    return DiscreteSACPolicy(bands, seed, settings, checkpoint)

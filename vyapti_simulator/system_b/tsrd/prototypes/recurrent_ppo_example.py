"""Unselected recurrent PPO prototype, retained only as a reference.

The policy sees only the public observation history. Training consumes complete
on-policy episodes and optimizes an LSTM actor/critic with clipped PPO updates.
This is not an approved/default algorithm implementation or trained checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


SCHEMA = "vyapti_recurrent_ppo_policy_v1"
INPUT_SIZE = 44  # prior band one-hot, hit/cost/dwell, public clock encodings
BAND_COUNT = 36
HIDDEN_SIZE = 64
EXPERIMENT_SETTINGS = {
    "algorithm": "recurrent_ppo",
    "architecture": "44-feature encoder -> 64-unit LSTM -> 36-action actor and value critic",
    "optimizer": "Adam",
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ppo_clip": 0.2,
    "ppo_epochs_per_episode": 4,
    "value_coefficient": 0.01,
    "entropy_coefficient": 0.01,
    "max_gradient_norm": 0.5,
    "input_contract": (
        "past selected band, public hit/retune/dwell, and public time encodings; "
        "receiver truth and emitter identity excluded"
    ),
}
LEARNING_RATE = EXPERIMENT_SETTINGS["learning_rate"]
GAMMA = EXPERIMENT_SETTINGS["gamma"]
GAE_LAMBDA = EXPERIMENT_SETTINGS["gae_lambda"]
CLIP_RANGE = EXPERIMENT_SETTINGS["ppo_clip"]
VALUE_COEFFICIENT = EXPERIMENT_SETTINGS["value_coefficient"]
ENTROPY_COEFFICIENT = EXPERIMENT_SETTINGS["entropy_coefficient"]
PPO_EPOCHS = EXPERIMENT_SETTINGS["ppo_epochs_per_episode"]
MAX_GRAD_NORM = EXPERIMENT_SETTINGS["max_gradient_norm"]


class ActorCritic(nn.Module):
    def __init__(self, bands: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(INPUT_SIZE, HIDDEN_SIZE), nn.Tanh())
        self.memory = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True)
        self.actor = nn.Linear(HIDDEN_SIZE, bands)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, observations, state=None):
        features = self.encoder(observations)
        hidden, state = self.memory(features, state)
        return self.actor(hidden), self.critic(hidden).squeeze(-1), state


class Policy:
    """Recurrent policy implementing the shared TRAIN-250 scheduler API."""

    def __init__(self, bands: int, seed: int, checkpoint: Path | None = None):
        if bands != BAND_COUNT:
            raise ValueError(f"This checkpoint requires the frozen {BAND_COUNT}-band action space")
        torch.set_num_threads(1)
        torch.manual_seed(int(seed))
        self.bands = int(bands)
        self.rng = np.random.default_rng(int(seed))
        self.model = ActorCritic(self.bands)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.learning_enabled = checkpoint is None
        if checkpoint is not None:
            payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
            if payload.get("schema") != SCHEMA or payload.get("bands") != self.bands:
                raise ValueError("Checkpoint schema or receiver band count does not match")
            if payload.get("policy_settings") != EXPERIMENT_SETTINGS:
                raise ValueError("Checkpoint PPO settings differ from this policy implementation")
            self.model.load_state_dict(payload["state_dict"])
        self.reset_episode()

    @staticmethod
    def _features(history, time_slot: int) -> np.ndarray:
        feature = np.zeros(INPUT_SIZE, dtype=np.float32)
        if history:
            previous = history[-1]
            band = int(previous["selected_band"])
            if 0 <= band < BAND_COUNT:
                feature[band] = 1.0
            feature[36] = float(previous["hit"])
            feature[37] = np.clip(float(previous["retune_cost_s"]) / 0.05, 0.0, 1.0)
            feature[38] = np.clip(float(previous["dwell_elapsed_s"]) / 0.05, 0.0, 1.0)
        phase = 2.0 * np.pi * (int(time_slot) % 40) / 40.0
        mission_phase = 2.0 * np.pi * (int(time_slot) % 600) / 600.0
        feature[39] = np.clip(int(time_slot) / 599.0, 0.0, 1.0)
        feature[40:42] = (np.sin(phase), np.cos(phase))
        feature[42:44] = (np.sin(mission_phase), np.cos(mission_phase))
        return feature

    def reset_episode(self):
        self._state = None
        self._features_seen = []
        self._actions = []
        self._old_log_probs = []
        self._old_values = []
        self._rewards = []
        self._episode_reward = 0.0

    def select_action(self, history, time_slot):
        features = self._features(history, time_slot)
        tensor = torch.from_numpy(features).reshape(1, 1, INPUT_SIZE)
        with torch.no_grad():
            logits, value, next_state = self.model(tensor, self._state)
            distribution = Categorical(logits=logits[:, -1])
            action = distribution.sample() if self.learning_enabled else logits[:, -1].argmax(dim=-1)
            log_prob = distribution.log_prob(action)
        self._state = tuple(part.detach() for part in next_state)
        if self.learning_enabled:
            self._features_seen.append(features)
            self._actions.append(int(action.item()))
            self._old_log_probs.append(float(log_prob.item()))
            self._old_values.append(float(value[0, -1].item()))
        return int(action.item())

    def observe(self, observation):
        if self.learning_enabled:
            reward = float(bool(observation["hit"]))
            self._rewards.append(reward)
            self._episode_reward += reward

    def _advantages(self):
        rewards = np.asarray(self._rewards, dtype=np.float32)
        values = np.asarray(self._old_values, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            next_value = float(values[t + 1]) if t + 1 < len(values) else 0.0
            delta = rewards[t] + GAMMA * next_value - values[t]
            running = delta + GAMMA * GAE_LAMBDA * running
            advantages[t] = running
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def finish_training_episode(self):
        if not self.learning_enabled:
            return
        if not self._rewards or len(self._rewards) != len(self._actions):
            raise RuntimeError("PPO episode has mismatched actions and receiver rewards")
        observations = torch.from_numpy(np.asarray(self._features_seen)).unsqueeze(0)
        actions = torch.tensor(self._actions, dtype=torch.long).unsqueeze(0)
        old_log_probs = torch.tensor(self._old_log_probs, dtype=torch.float32).unsqueeze(0)
        advantages_np, returns_np = self._advantages()
        advantages = torch.from_numpy(advantages_np).unsqueeze(0)
        returns = torch.from_numpy(returns_np).unsqueeze(0)
        for _ in range(PPO_EPOCHS):
            logits, values, _ = self.model(observations)
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * torch.square(values - returns).mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + VALUE_COEFFICIENT * value_loss - ENTROPY_COEFFICIENT * entropy
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()
        self.reset_episode()

    def save(self, path: Path):
        torch.save({"schema": SCHEMA, "bands": self.bands,
                    "state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "policy_settings": EXPERIMENT_SETTINGS,
                    "algorithm": "recurrent_ppo", "hidden_size": HIDDEN_SIZE,
                    "input_size": INPUT_SIZE}, Path(path))


def create(bands: int, seed: int, checkpoint: Path | None = None):
    return Policy(bands, seed, checkpoint)

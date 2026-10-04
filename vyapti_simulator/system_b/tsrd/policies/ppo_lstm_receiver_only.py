"""Receiver-only recurrent PPO with multi-world rollouts and coverage bias.

This is a separate implementation from ``ppo_lstm``. It preserves the shared
PublicTransition contract and frozen Mode-B environment while using a fresh
325-feature observation/checkpoint schema.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..algorithm_interface import PublicTransition


NATIVE_DWELL_S = 0.05
MISSION_S = 30.0
FEATURES_PER_BAND = 9
OBSERVATION_SCHEMA = "receiver_belief_periodicity_dwell_hit_visit_coverage_v2_325d"
CHECKPOINT_FORMAT = "vyapti_ppo_lstm_v2"


def _init_linear(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class _ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, bands: int, hidden: int):
        super().__init__()
        self.encoder = nn.Sequential(
            _init_linear(nn.Linear(observation_dim, hidden), math.sqrt(2.0)),
            nn.Tanh(), nn.LayerNorm(hidden),
        )
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        for name, parameter in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(parameter, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(parameter, 1.0)
        self.actor = nn.Sequential(
            _init_linear(nn.Linear(hidden, 128), math.sqrt(2.0)), nn.Tanh(),
            _init_linear(nn.Linear(128, bands), 0.01),
        )
        self.critic = nn.Sequential(
            _init_linear(nn.Linear(hidden, 128), math.sqrt(2.0)), nn.Tanh(),
            _init_linear(nn.Linear(128, 1), 1.0),
        )

    def initial_state(self, batch_size: int, hidden: int, device):
        return (torch.zeros(1, batch_size, hidden, device=device),
                torch.zeros(1, batch_size, hidden, device=device))

    def step(self, observation, state):
        encoded = self.encoder(observation).unsqueeze(1)
        output, next_state = self.lstm(encoded, state)
        latent = output[:, 0, :]
        return self.actor(latent), self.critic(latent).squeeze(-1), next_state

    def sequence(self, observations, state):
        output, next_state = self.lstm(self.encoder(observations), state)
        return self.actor(output), self.critic(output).squeeze(-1), next_state


@dataclass
class _EpisodeRollout:
    observations: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    discounts: np.ndarray
    lambda_discounts: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    hidden_h: np.ndarray
    hidden_c: np.ndarray
    bootstrap_value: float
    receiver_base_steps: int
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None


class PPOLSTMPolicy:
    """PPO-LSTM with receiver-causal belief, visit history and coverage bias."""

    def __init__(self, bands: int, seed: int, settings: dict, checkpoint: Path | None = None):
        if type(bands) is not int or bands < 2:
            raise ValueError("PPO-LSTM-v2 needs at least two available bands")
        self.bands = bands
        self.settings = dict(settings)
        self.hidden_size = int(self.settings.get("hidden_units", 256))
        self.lr = float(self.settings.get("learning_rate", 2e-4))
        self.gamma_base = float(self.settings.get("gamma_per_base_slot", 0.997))
        self.gae_lambda_base = float(self.settings.get("gae_lambda_per_base_slot", 0.98))
        self.clip_eps = float(self.settings.get("clip_epsilon", 0.20))
        self.value_clip_eps = float(self.settings.get("value_clip_epsilon", 0.20))
        self.entropy_coef = float(self.settings.get("entropy_coefficient", 0.005))
        self.value_coef = float(self.settings.get("value_coefficient", 0.50))
        self.max_grad_norm = float(self.settings.get("max_grad_norm", 0.50))
        self.ppo_epochs = int(self.settings.get("ppo_epochs", 4))
        self.target_kl = float(self.settings.get("target_kl", 0.03))
        self.bptt_chunk = int(self.settings.get("bptt_chunk", 128))
        self.sequence_minibatch = int(self.settings.get("sequence_minibatch", 8))
        self.rollout_episodes = int(self.settings.get("rollout_episodes", 8))
        self.training_base_steps_target = int(
            self.settings.get("training_base_steps_target", 480_000)
        )
        self.coverage_kappa = float(self.settings.get("coverage_kappa", 0.10))
        self.recent_hit_window = int(self.settings.get("recent_hit_window", 8))
        self.period_history = int(self.settings.get("periodicity_history", 32))
        # Canonical TRAIN-only calibration. Normal setup injects the exact
        # calibration file values, which remain authoritative.
        self.p01 = float(self.settings.get("inactive_to_active_probability", 0.02220))
        self.p11 = float(self.settings.get("active_to_active_probability", 0.95753))
        self.prior_active = float(self.settings.get("prior_active_probability", 0.34266))
        self.pd = float(self.settings.get("detection_probability", 0.90))
        self.pfa = float(self.settings.get("false_alarm_probability", 0.05))
        self.base_slot_s = float(self.settings.get("base_slot_seconds", NATIVE_DWELL_S))
        dwell_profile = self.settings.get("native_dwell_slots")
        if (not isinstance(dwell_profile, (list, tuple)) or len(dwell_profile) != bands
                or any(type(value) is not int or value not in (1, 2) for value in dwell_profile)):
            raise ValueError("native_dwell_slots must contain one- or two-slot values per band")
        self.native_dwell_slots = np.asarray(dwell_profile, dtype=np.int64)
        self.native_dwell_feature = self.native_dwell_slots.astype(np.float32) / 2.0
        self.sweep_slots = int(np.sum(self.native_dwell_slots))
        self.visit_count_scale = max(1, math.ceil(600 / self.sweep_slots))

        positive = (self.hidden_size, self.ppo_epochs, self.bptt_chunk,
                    self.sequence_minibatch, self.rollout_episodes,
                    self.recent_hit_window, self.period_history,
                    self.training_base_steps_target)
        if any(value < 1 for value in positive):
            raise ValueError("PPO-LSTM-v2 dimensions and rollout settings must be positive")
        probabilities = (self.gamma_base, self.gae_lambda_base, self.p01, self.p11,
                         self.prior_active, self.pd, self.pfa)
        if any(not math.isfinite(value) or not 0 < value < 1 for value in probabilities):
            raise ValueError("PPO-LSTM-v2 probabilities and discounts must be in (0, 1)")
        if self.pfa >= self.pd:
            raise ValueError("Detector Pfa must be lower than Pd")
        if (not math.isclose(self.base_slot_s, NATIVE_DWELL_S, abs_tol=1e-12)
                or any(not math.isfinite(v) or v <= 0 for v in
                       (self.lr, self.clip_eps, self.value_clip_eps, self.max_grad_norm, self.target_kl))
                or not math.isfinite(self.coverage_kappa) or self.coverage_kappa < 0
                or any(not math.isfinite(v) or v < 0 for v in (self.entropy_coef, self.value_coef))):
            raise ValueError("Invalid PPO-LSTM-v2 timing, optimizer or coverage setting")

        device_setting = self.settings.get("device", "auto")
        if device_setting not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if device_setting == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PPO-LSTM-v2 requested CUDA, but CUDA is unavailable")
        self.device = torch.device("cuda" if device_setting == "cuda" or
                                   (device_setting == "auto" and torch.cuda.is_available()) else "cpu")
        if self.device.type == "cpu":
            cpu_threads = int(self.settings.get("cpu_threads", 1))
            if cpu_threads < 1:
                raise ValueError("cpu_threads must be positive")
            torch.set_num_threads(cpu_threads)

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.evaluation_action_mode = "greedy"
        self.evaluation_rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        self.observation_dim = bands * FEATURES_PER_BAND + 1
        self.model = _ActorCritic(self.observation_dim, bands, self.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
        self.total_policy_decisions = 0
        self.total_receiver_base_steps = 0
        self.total_updates = 0
        self.completed_training_episodes = 0
        self._rollout_buffer: list[_EpisodeRollout] = []
        self._rollout_episode_rewards: list[float] = []
        self._last_losses: dict[str, float] = {}
        self.reset_episode(training=True)
        if checkpoint is not None:
            self._load(checkpoint)

    @property
    def total_actions(self) -> int:
        """Compatibility diagnostic: actions are policy decisions, not slots."""
        return self.total_policy_decisions

    def _initial_recurrent_state(self):
        return self.model.initial_state(1, self.hidden_size, self.device)

    def reset_episode(self, *, training: bool) -> None:
        # Only episode-local state resets here. Batched rollout data survives
        # across worlds until the eight-world PPO update.
        self.belief = np.full(self.bands, self.prior_active, dtype=np.float64)
        self.last_visit_end = np.full(self.bands, -1, dtype=np.int64)
        self.last_hit_slot = np.full(self.bands, -1, dtype=np.int64)
        self.hit_times: list[list[float]] = [[] for _ in range(self.bands)]
        self.visit_counts = np.zeros(self.bands, dtype=np.int64)
        self.visit_hit_history: list[list[int]] = [[] for _ in range(self.bands)]
        self.previous_action = -1
        self.hidden = self._initial_recurrent_state()
        self._pending = None
        self._observations: list[np.ndarray] = []
        self._actions: list[int] = []
        self._log_probs: list[float] = []
        self._values: list[float] = []
        self._hidden_h: list[np.ndarray] = []
        self._hidden_c: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._base_steps: list[int] = []
        self._discounts: list[float] = []
        self._lambda_discounts: list[float] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._last_next_features: np.ndarray | None = None
        self._episode_slot_actions: list[int] = []
        self.episode_reward = 0.0
        self.episode_decisions = 0
        self.band_selection_counts = np.zeros(self.bands, dtype=np.int64)
        self._last_losses = {}

    def _features(self, slot: int) -> np.ndarray:
        raw_staleness = slot - self.last_visit_end
        staleness = np.where(
            self.last_visit_end >= 0,
            np.clip(raw_staleness / max(1, self.sweep_slots) / 4.0, 0.0, 1.0),
            1.0,
        )
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
            certainty = float(np.clip(np.exp(-cv) * min(1.0, len(gaps) / 6.0), 0.0, 1.0))
            phase = max(0.0, float(slot) - times[-1]) % period
            phase_error = min(phase, period - phase)
            tolerance = max(0.15 * period, 1.0)
            periodicity[band] = np.float32(np.clip(np.exp(-phase_error / tolerance) * certainty, 0, 1))
            confidence[band] = np.float32(certainty)
        previous = np.zeros(self.bands, dtype=np.float32)
        if self.previous_action >= 0:
            previous[self.previous_action] = 1.0
        time_since_hit = np.where(
            self.last_hit_slot >= 0, (slot - self.last_hit_slot) / 600.0, 1.0,
        )
        remaining = np.asarray([np.clip(1.0 - slot * self.base_slot_s / MISSION_S, 0, 1)],
                               dtype=np.float32)
        visit_norm = np.clip(self.visit_counts / float(self.visit_count_scale), 0.0, 1.0)
        recent_hit_rate = np.asarray([
            float(np.mean(history[-self.recent_hit_window:])) if history else 0.0
            for history in self.visit_hit_history
        ], dtype=np.float32)
        features = np.concatenate((
            self.belief.astype(np.float32),
            staleness.astype(np.float32),
            periodicity, confidence, previous, remaining,
            self.native_dwell_feature.astype(np.float32),
            np.clip(time_since_hit, 0, 1).astype(np.float32),
            visit_norm.astype(np.float32), recent_hit_rate,
        ))
        if features.shape != (self.observation_dim,) or not np.all(np.isfinite(features)):
            raise RuntimeError("PPO-LSTM-v2 produced invalid public observation features")
        return features

    @staticmethod
    def _coverage_urgency(features: np.ndarray) -> np.ndarray:
        bands = (len(features) - 1) // FEATURES_PER_BAND
        staleness = features[bands:2 * bands]
        visits = features[7 * bands + 1:8 * bands + 1]
        urgency = 0.7 * staleness + 0.3 * (1.0 - visits)
        return np.clip(urgency, 0.0, 1.0).astype(np.float32)

    def _adjust_logits(self, logits: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        urgency = 0.7 * observations[..., self.bands:2 * self.bands]
        urgency = urgency + 0.3 * (1.0 - observations[..., 7 * self.bands + 1:8 * self.bands + 1])
        return logits + self.coverage_kappa * torch.clamp(urgency, 0.0, 1.0)

    def select_action(self, state, *, training: bool) -> int:
        if self._pending is not None:
            raise RuntimeError("observe must follow each PPO-LSTM-v2 action")
        features = self._features(int(state.time_slot))
        h_before = self.hidden[0][:, 0, :].detach().cpu().numpy().copy()
        c_before = self.hidden[1][:, 0, :].detach().cpu().numpy().copy()
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            raw_logits, value, next_hidden = self.model.step(x, self.hidden)
            logits = self._adjust_logits(raw_logits, x)
            distribution = Categorical(logits=logits)
            probabilities = distribution.probs[0].cpu().numpy()
            if training:
                action = int(self.rng.choice(self.bands, p=probabilities))
            elif self.evaluation_action_mode == "sampled":
                action = int(self.evaluation_rng.choice(self.bands, p=probabilities))
            else:
                action = int(torch.argmax(logits, dim=-1).item())
            log_probability = float(distribution.log_prob(
                torch.as_tensor([action], dtype=torch.long, device=self.device)).item())
            value_estimate = float(value.item())
        self.hidden = tuple(part.detach() for part in next_hidden)
        self._pending = {"features": features, "action": action,
                         "log_probability": log_probability, "value": value_estimate,
                         "hidden_h": h_before, "hidden_c": c_before}
        self.band_selection_counts[action] += 1
        self.episode_decisions += 1
        if training:
            self.total_policy_decisions += 1
        return action

    def set_evaluation_action_mode(self, mode: str, seed: int | None = None) -> None:
        if mode not in {"greedy", "sampled"}:
            raise ValueError("Evaluation action mode must be greedy or sampled")
        if mode == "sampled" and (type(seed) is not int or seed < 0):
            raise ValueError("Sampled evaluation requires a nonnegative integer action seed")
        self.evaluation_action_mode = mode
        if seed is not None:
            self.evaluation_rng = np.random.default_rng(seed)

    @staticmethod
    def _transition(distribution: np.ndarray, p01: float, p11: float) -> np.ndarray:
        inactive, active = distribution
        return np.asarray((inactive * (1.0 - p01) + active * (1.0 - p11),
                           inactive * p01 + active * p11), dtype=np.float64)

    def _update_belief(self, band: int, hit: bool, slots: int) -> None:
        prior_active = self.belief.copy()
        for current_band in range(self.bands):
            distribution = np.asarray((1.0 - prior_active[current_band],
                                       prior_active[current_band]), dtype=np.float64)
            end_prior = distribution.copy()
            for _ in range(slots):
                end_prior = self._transition(end_prior, self.p01, self.p11)
            if current_band == band:
                no_hit = distribution.copy()
                no_hit_likelihood = np.asarray((1.0 - self.pfa, 1.0 - self.pd))
                for _ in range(slots):
                    no_hit = self._transition(no_hit * no_hit_likelihood, self.p01, self.p11)
                posterior = end_prior - no_hit if hit else no_hit
                normalizer = float(posterior.sum())
                if not math.isfinite(normalizer) or normalizer <= 0:
                    raise RuntimeError("Aggregate hit has zero HMM likelihood")
                end_prior = posterior / normalizer
            self.belief[current_band] = np.clip(end_prior[1], 1e-6, 1.0 - 1e-6)

    def observe(self, transition: PublicTransition, *, training: bool) -> None:
        if self._pending is None:
            raise RuntimeError("select_action must precede observe for each transition")
        if transition.action != self._pending["action"]:
            raise ValueError("Transition action differs from PPO-LSTM-v2's selected action")
        observation = transition.next_state.previous_observation
        if observation is None:
            raise ValueError("PPO-LSTM-v2 requires a receiver observation after each action")
        slots = int(transition.next_state.time_slot) - int(transition.state.time_slot)
        if slots not in (1, 2):
            raise ValueError("PPO-LSTM-v2 supports the native one- or two-slot dwell")
        band = int(observation.selected_band)
        if not 0 <= band < self.bands:
            raise ValueError("Receiver observation selected a band outside PPO-LSTM-v2 action space")
        self._update_belief(band, bool(observation.hit), slots)
        end_slot = int(transition.next_state.time_slot)
        if observation.hit:
            times = self.hit_times[band]
            times.append(float(end_slot))
            if len(times) > self.period_history:
                del times[:-self.period_history]
            self.last_hit_slot[band] = end_slot
        self.last_visit_end[band] = end_slot
        self.visit_counts[band] += 1
        history = self.visit_hit_history[band]
        history.append(int(bool(observation.hit)))
        if len(history) > self.recent_hit_window:
            del history[:-self.recent_hit_window]
        self.previous_action = band
        self._episode_slot_actions.extend([band] * slots)
        self.episode_reward += float(transition.reward)
        if training:
            item = self._pending
            self._observations.append(item["features"])
            self._actions.append(item["action"])
            self._log_probs.append(item["log_probability"])
            self._values.append(item["value"])
            self._hidden_h.append(item["hidden_h"][0])
            self._hidden_c.append(item["hidden_c"][0])
            self._rewards.append(float(transition.reward))
            self._base_steps.append(slots)
            self._discounts.append(self.gamma_base ** slots)
            self._lambda_discounts.append(self.gae_lambda_base ** slots)
            self._terminated.append(bool(transition.terminated))
            self._truncated.append(bool(transition.truncated))
            self.total_receiver_base_steps += slots
        self._last_next_features = self._features(end_slot)
        self._pending = None

    def _make_episode_rollout(self) -> _EpisodeRollout:
        if not self._rewards:
            raise RuntimeError("Cannot add an empty PPO-LSTM-v2 episode to a rollout")
        bootstrap = 0.0
        if self._truncated[-1] and not self._terminated[-1] and self._last_next_features is not None:
            features = torch.as_tensor(self._last_next_features, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, value, _ = self.model.step(features, self.hidden)
            bootstrap = float(value.item())
        return _EpisodeRollout(
            observations=np.asarray(self._observations, dtype=np.float32),
            actions=np.asarray(self._actions, dtype=np.int64),
            old_log_probs=np.asarray(self._log_probs, dtype=np.float32),
            values=np.asarray(self._values, dtype=np.float32),
            rewards=np.asarray(self._rewards, dtype=np.float32),
            discounts=np.asarray(self._discounts, dtype=np.float32),
            lambda_discounts=np.asarray(self._lambda_discounts, dtype=np.float32),
            terminated=np.asarray(self._terminated, dtype=bool),
            truncated=np.asarray(self._truncated, dtype=bool),
            hidden_h=np.asarray(self._hidden_h, dtype=np.float32),
            hidden_c=np.asarray(self._hidden_c, dtype=np.float32),
            bootstrap_value=bootstrap,
            receiver_base_steps=int(sum(self._base_steps)),
        )

    @staticmethod
    def _episode_gae(episode: _EpisodeRollout) -> None:
        rewards, values = episode.rewards, episode.values
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            if index == len(rewards) - 1:
                next_value = episode.bootstrap_value
            else:
                next_value = float(values[index + 1])
            nonterminal = 0.0 if episode.terminated[index] else 1.0
            delta = (float(rewards[index]) + float(episode.discounts[index])
                     * nonterminal * next_value - float(values[index]))
            trace = 0.0 if episode.terminated[index] else 1.0
            gae = delta + (float(episode.discounts[index])
                           * float(episode.lambda_discounts[index]) * trace * gae)
            advantages[index] = gae
        episode.advantages = advantages
        episode.returns = advantages + values

    def _learn_rollout(self) -> dict[str, float]:
        episodes = self._rollout_buffer
        if not episodes:
            return {}
        for episode in episodes:
            self._episode_gae(episode)
        all_advantages = np.concatenate([ep.advantages for ep in episodes if ep.advantages is not None])
        normalized = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        cursor = 0
        for episode in episodes:
            assert episode.advantages is not None
            length = len(episode.advantages)
            episode.advantages = normalized[cursor:cursor + length].astype(np.float32)
            cursor += length

        chunks: list[tuple[int, int, int]] = []
        for episode_index, episode in enumerate(episodes):
            chunks.extend((episode_index, start, min(start + self.bptt_chunk, len(episode.rewards)))
                          for start in range(0, len(episode.rewards), self.bptt_chunk))
        learning_rate_fraction = max(
            0.0, 1.0 - self.total_receiver_base_steps / self.training_base_steps_target,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr * learning_rate_fraction
        stats = {key: [] for key in ("loss", "policy_loss", "value_loss", "entropy", "kl", "clip_fraction")}
        for _ in range(self.ppo_epochs):
            self.rng.shuffle(chunks)
            epoch_kls = []
            for offset in range(0, len(chunks), self.sequence_minibatch):
                batch = chunks[offset:offset + self.sequence_minibatch]
                batch_size = len(batch)
                sequence_length = max(end - start for _, start, end in batch)
                obs = torch.zeros(batch_size, sequence_length, self.observation_dim, device=self.device)
                actions = torch.zeros(batch_size, sequence_length, dtype=torch.long, device=self.device)
                old_logp = torch.zeros(batch_size, sequence_length, device=self.device)
                adv = torch.zeros(batch_size, sequence_length, device=self.device)
                ret = torch.zeros(batch_size, sequence_length, device=self.device)
                old_value = torch.zeros(batch_size, sequence_length, device=self.device)
                mask = torch.zeros(batch_size, sequence_length, device=self.device)
                h0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
                c0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
                for row, (episode_index, start, end) in enumerate(batch):
                    episode = episodes[episode_index]
                    length = end - start
                    obs[row, :length] = torch.as_tensor(episode.observations[start:end],
                                                         dtype=torch.float32, device=self.device)
                    actions[row, :length] = torch.as_tensor(episode.actions[start:end],
                                                             dtype=torch.long, device=self.device)
                    old_logp[row, :length] = torch.as_tensor(episode.old_log_probs[start:end],
                                                              dtype=torch.float32, device=self.device)
                    adv[row, :length] = torch.as_tensor(episode.advantages[start:end],
                                                         dtype=torch.float32, device=self.device)
                    ret[row, :length] = torch.as_tensor(episode.returns[start:end],
                                                         dtype=torch.float32, device=self.device)
                    old_value[row, :length] = torch.as_tensor(episode.values[start:end],
                                                              dtype=torch.float32, device=self.device)
                    mask[row, :length] = 1.0
                    h0[:, row, :] = torch.as_tensor(episode.hidden_h[start],
                                                    dtype=torch.float32, device=self.device)
                    c0[:, row, :] = torch.as_tensor(episode.hidden_c[start],
                                                    dtype=torch.float32, device=self.device)

                raw_logits, predicted_value, _ = self.model.sequence(obs, (h0, c0))
                adjusted_logits = self._adjust_logits(raw_logits, obs)
                distribution = Categorical(logits=adjusted_logits)
                new_logp = distribution.log_prob(actions)
                entropy = distribution.entropy()
                ratio = torch.exp(new_logp - old_logp)
                denominator = mask.sum().clamp_min(1.0)
                policy_loss = -(torch.minimum(
                    ratio * adv,
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv,
                ) * mask).sum() / denominator
                clipped_value = old_value + torch.clamp(predicted_value - old_value,
                                                        -self.value_clip_eps, self.value_clip_eps)
                value_error = torch.maximum((predicted_value - ret) ** 2,
                                            (clipped_value - ret) ** 2)
                value_loss = 0.5 * (value_error * mask).sum() / denominator
                mean_entropy = (entropy * mask).sum() / denominator
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * mean_entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approximate_kl = ((old_logp - new_logp) * mask).sum() / denominator
                    clip_fraction = (((ratio - 1.0).abs() > self.clip_eps).float() * mask).sum() / denominator
                values_now = {
                    "loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
                    "entropy": mean_entropy, "kl": approximate_kl, "clip_fraction": clip_fraction,
                }
                for key, value in values_now.items():
                    scalar = float(value.item())
                    if not math.isfinite(scalar):
                        raise FloatingPointError(f"Non-finite PPO-LSTM-v2 {key}")
                    stats[key].append(scalar)
                epoch_kls.append(float(approximate_kl.item()))
            if epoch_kls and float(np.mean(epoch_kls)) > self.target_kl:
                break
        self.total_updates += 1
        result = {key: float(np.mean(values)) for key, values in stats.items() if values}
        result.update({
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "learning_rate_fraction": float(learning_rate_fraction),
            "total_policy_decisions": float(self.total_policy_decisions),
            "total_receiver_base_steps": float(self.total_receiver_base_steps),
            "rollout_episodes": float(len(episodes)),
            "rollout_reward_mean": float(np.mean(self._rollout_episode_rewards)),
            "rollout_receiver_steps": float(sum(ep.receiver_base_steps for ep in episodes)),
        })
        self._rollout_buffer = []
        self._rollout_episode_rewards = []
        return result

    def _policy_diagnostics(self) -> dict:
        counts = self.band_selection_counts
        total = int(counts.sum())
        sorted_counts = np.sort(counts)[::-1]
        probabilities = counts[counts > 0] / max(total, 1)
        entropy = float(-np.sum(probabilities * np.log(probabilities))) if len(probabilities) else 0.0
        normalized_entropy = entropy / math.log(self.bands) if self.bands > 1 else 0.0
        slots = np.asarray(self._episode_slot_actions, dtype=np.int64)
        rolling_coverage = None
        if len(slots) >= self.sweep_slots:
            rolling_coverage = float(np.mean([
                len(np.unique(slots[start:start + self.sweep_slots])) / self.bands
                for start in range(len(slots) - self.sweep_slots + 1)
            ]))
        last_visit = np.full(self.bands, -1, dtype=np.int64)
        age_samples = []
        for slot, selected in enumerate(slots):
            ages = np.where(last_visit >= 0, slot - last_visit, slot)
            age_samples.extend(ages.tolist())
            last_visit[selected] = slot
        return {
            "algorithm": "ppo_lstm",
            "implementation": "receiver_only_325d",
            "training_updates": int(bool(self._last_losses)),
            "total_policy_decisions": self.total_policy_decisions,
            "total_receiver_base_steps": self.total_receiver_base_steps,
            "completed_training_episodes": self.completed_training_episodes,
            "episode_decisions": self.episode_decisions,
            "episode_receiver_base_steps": len(slots),
            "episode_reward": self.episode_reward,
            "rollout_episodes_collected": int(self._last_losses.get(
                "rollout_episodes", len(self._rollout_buffer))),
            "band_selection_counts": counts.tolist(),
            "fraction_top_1_band": float(sorted_counts[:1].sum() / max(total, 1)),
            "fraction_top_3_bands": float(sorted_counts[:3].sum() / max(total, 1)),
            "fraction_top_5_bands": float(sorted_counts[:5].sum() / max(total, 1)),
            "action_entropy": entropy,
            "normalized_action_entropy": float(normalized_entropy),
            "unique_bands_visited": int(np.count_nonzero(counts)),
            "whole_mission_unique_band_fraction": float(np.count_nonzero(counts) / self.bands),
            "mean_visit_count_per_band": float(np.mean(counts)),
            "number_of_bands_never_visited": int(np.sum(counts == 0)),
            "mean_rolling_coverage": rolling_coverage,
            "max_band_staleness_slots": int(max(age_samples, default=0)),
            "mean_band_staleness_slots": float(np.mean(age_samples)) if age_samples else 0.0,
            "p95_band_staleness_slots": float(np.percentile(age_samples, 95)) if age_samples else 0.0,
            **{f"mean_{key}": value for key, value in self._last_losses.items()},
        }

    def end_episode(self, *, training: bool) -> dict:
        if self._pending is not None:
            raise RuntimeError("Cannot end PPO-LSTM-v2 episode with an unobserved action")
        if training and self._rewards:
            rollout = self._make_episode_rollout()
            self._rollout_buffer.append(rollout)
            self._rollout_episode_rewards.append(float(self.episode_reward))
            self.completed_training_episodes += 1
            self._last_losses = {}
            if len(self._rollout_buffer) == self.rollout_episodes:
                self._last_losses = self._learn_rollout()
        return self._policy_diagnostics()

    def save(self, path) -> None:
        if self._rollout_buffer:
            raise RuntimeError("PPO-LSTM-v2 checkpoints must be saved on an eight-world rollout boundary")
        payload = {
            "format": CHECKPOINT_FORMAT, "bands": self.bands,
            "observation_schema": OBSERVATION_SCHEMA,
            "observation_dim": self.observation_dim,
            "settings": self.settings, "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_policy_decisions": self.total_policy_decisions,
            "total_receiver_base_steps": self.total_receiver_base_steps,
            "total_updates": self.total_updates,
            "completed_training_episodes": self.completed_training_episodes,
            "rng_state": self.rng.bit_generator.state,
        }
        # An explicitly opened stream works consistently with Windows paths
        # and filesystems that restrict PyTorch's native path-based writer.
        with Path(path).open("wb") as stream:
            torch.save(payload, stream)

    def _load(self, path: Path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if (payload.get("format") != CHECKPOINT_FORMAT or payload.get("bands") != self.bands
                or payload.get("observation_dim") != self.observation_dim):
            raise ValueError("Checkpoint is not a compatible PPO-LSTM-v2 checkpoint")
        if payload.get("observation_schema") != OBSERVATION_SCHEMA:
            raise ValueError("PPO-LSTM-v2 checkpoint observation schema differs from this run")
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.total_policy_decisions = int(payload.get("total_policy_decisions", 0))
        self.total_receiver_base_steps = int(payload.get("total_receiver_base_steps", 0))
        self.total_updates = int(payload.get("total_updates", 0))
        self.completed_training_episodes = int(payload.get("completed_training_episodes", 0))
        if "rng_state" in payload:
            self.rng.bit_generator.state = payload["rng_state"]


def create(*, bands: int, seed: int, settings: dict, checkpoint=None):
    return PPOLSTMPolicy(bands=bands, seed=seed, settings=settings, checkpoint=checkpoint)

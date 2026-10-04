"""Recurrent PPO scheduler for the shared TSRD Mode-B transition runner."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..algorithm_interface import PublicTransition


NATIVE_DWELL_S = 0.05
MISSION_S = 30.0
FEATURES_PER_BAND = 5
OBSERVATION_SCHEMA = "receiver_belief_periodicity_dwell_hit_history_v2"


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


class PPOLSTMPolicy:
    """PPO-LSTM with receiver-only belief and causal periodicity features."""

    def __init__(self, bands: int, seed: int, settings: dict, checkpoint: Path | None = None):
        if type(bands) is not int or bands < 2:
            raise ValueError("PPO-LSTM needs at least two available bands")
        self.bands = bands
        self.settings = dict(settings)
        self.hidden_size = int(self.settings.get("hidden_units", 256))
        self.lr = float(self.settings.get("learning_rate", 2e-4))
        self.gamma_base = float(self.settings.get("gamma_per_base_slot", 0.997))
        self.gae_lambda_base = float(self.settings.get("gae_lambda_per_base_slot", 0.98))
        self.clip_eps = float(self.settings.get("clip_epsilon", 0.2))
        self.value_clip_eps = float(self.settings.get("value_clip_epsilon", 0.2))
        self.entropy_coef = float(self.settings.get("entropy_coefficient", 0.005))
        self.value_coef = float(self.settings.get("value_coefficient", 0.5))
        self.max_grad_norm = float(self.settings.get("max_grad_norm", 0.5))
        self.ppo_epochs = int(self.settings.get("ppo_epochs", 5))
        self.target_kl = float(self.settings.get("target_kl", 0.03))
        self.bptt_chunk = int(self.settings.get("bptt_chunk", 128))
        self.sequence_minibatch = int(self.settings.get("sequence_minibatch", 8))
        self.training_actions_target = int(self.settings.get("training_actions_target", 400_000))
        self.period_history = int(self.settings.get("periodicity_history", 32))
        self.p01 = float(self.settings.get("inactive_to_active_probability", 0.05))
        self.p11 = float(self.settings.get("active_to_active_probability", 0.90))
        self.prior_active = float(self.settings.get("prior_active_probability", 0.5))
        self.pd = float(self.settings.get("detection_probability", 0.90))
        self.pfa = float(self.settings.get("false_alarm_probability", 0.05))
        self.base_slot_s = float(self.settings.get("base_slot_seconds", NATIVE_DWELL_S))
        dwell_profile = self.settings.get("native_dwell_slots", [1] * bands)
        if (not isinstance(dwell_profile, (list, tuple)) or len(dwell_profile) != bands
                or any(type(value) is not int or value not in (1, 2) for value in dwell_profile)):
            raise ValueError("native_dwell_slots must contain one- or two-slot dwell values for every band")
        self.native_dwell_slots = np.asarray(dwell_profile, dtype=np.float32)
        self.native_dwell_feature = self.native_dwell_slots / 2.0
        positive = (self.hidden_size, self.ppo_epochs, self.bptt_chunk, self.sequence_minibatch,
                    self.period_history)
        if any(value < 1 for value in positive):
            raise ValueError("PPO-LSTM dimensions, epochs, chunk, and history must be positive")
        if self.training_actions_target < 1:
            raise ValueError("training_actions_target must be positive")
        probabilities = (self.gamma_base, self.gae_lambda_base, self.p01, self.p11,
                         self.prior_active, self.pd, self.pfa)
        if any(not math.isfinite(value) or not 0 < value < 1 for value in probabilities):
            raise ValueError("PPO-LSTM probabilities and discount factors must be in (0, 1)")
        if self.pfa >= self.pd:
            raise ValueError("Detector Pfa must be lower than Pd")
        if not math.isclose(self.base_slot_s, NATIVE_DWELL_S, abs_tol=1e-12):
            raise ValueError("PPO-LSTM base_slot_seconds must match the 50 ms Mode-B clock")
        if any(not math.isfinite(value) or value <= 0 for value in
               (self.lr, self.clip_eps, self.value_clip_eps, self.max_grad_norm, self.target_kl)):
            raise ValueError("PPO-LSTM optimizer and clipping settings must be finite and positive")
        if any(not math.isfinite(value) or value < 0 for value in
               (self.entropy_coef, self.value_coef)):
            raise ValueError("PPO-LSTM loss coefficients must be finite and nonnegative")

        device_setting = self.settings.get("device", "auto")
        if device_setting not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if device_setting == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PPO-LSTM requested CUDA, but CUDA is unavailable")
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
        # Existing receiver/belief/periodicity state (181 features at 36 bands)
        # plus native dwell and time since observed HIT (36 each).
        self.observation_dim = bands * (FEATURES_PER_BAND + 2) + 1
        self.model = _ActorCritic(self.observation_dim, bands, self.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
        self.total_actions = 0
        self.total_updates = 0
        self.completed_training_episodes = 0
        self.reset_episode(training=True)
        if checkpoint is not None:
            self._load(checkpoint)

    def _initial_recurrent_state(self):
        return self.model.initial_state(1, self.hidden_size, self.device)

    def reset_episode(self, *, training: bool) -> None:
        self.belief = np.full(self.bands, self.prior_active, dtype=np.float64)
        self.last_visit_end = np.full(self.bands, -1, dtype=np.int64)
        self.last_hit_slot = np.full(self.bands, -1, dtype=np.int64)
        self.hit_times: list[list[float]] = [[] for _ in range(self.bands)]
        self.previous_action = -1
        self.hidden = self._initial_recurrent_state()
        self._pending = None
        self._observations = []
        self._actions = []
        self._log_probs = []
        self._values = []
        self._hidden_h = []
        self._hidden_c = []
        self._rewards = []
        self._discounts = []
        self._lambda_discounts = []
        self._terminals = []
        self.episode_reward = 0.0
        self.episode_decisions = 0
        self.band_selection_counts = np.zeros(self.bands, dtype=np.int64)
        self._last_losses = {}

    def _features(self, slot: int) -> np.ndarray:
        staleness = np.where(self.last_visit_end >= 0,
                             (slot - self.last_visit_end) / 600.0, 1.0)
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
        features = np.concatenate((self.belief.astype(np.float32),
                                    np.clip(staleness, 0, 1).astype(np.float32),
                                    periodicity, confidence, previous, remaining,
                                    self.native_dwell_feature.astype(np.float32),
                                    np.clip(time_since_hit, 0, 1).astype(np.float32)))
        if features.shape != (self.observation_dim,) or not np.all(np.isfinite(features)):
            raise RuntimeError("PPO-LSTM produced invalid public observation features")
        return features

    def select_action(self, state, *, training: bool) -> int:
        if self._pending is not None:
            raise RuntimeError("observe must follow each PPO-LSTM action")
        features = self._features(int(state.time_slot))
        h_before = self.hidden[0][:, 0, :].detach().cpu().numpy().copy()
        c_before = self.hidden[1][:, 0, :].detach().cpu().numpy().copy()
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value, next_hidden = self.model.step(x, self.hidden)
            log_probabilities = torch.log_softmax(logits, dim=-1)[0]
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            if training:
                action = int(self.rng.choice(self.bands, p=probabilities))
            elif self.evaluation_action_mode == "sampled":
                action = int(self.evaluation_rng.choice(self.bands, p=probabilities))
            else:
                action = int(torch.argmax(logits, dim=-1).item())
            log_probability = float(log_probabilities[action].item())
            value_estimate = float(value.item())
        self.hidden = tuple(part.detach() for part in next_hidden)
        self._pending = {"features": features, "action": action,
                         "log_probability": log_probability, "value": value_estimate,
                         "hidden_h": h_before, "hidden_c": c_before,
                         "training": training}
        self.band_selection_counts[action] += 1
        self.episode_decisions += 1
        self.total_actions += int(training)
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
            raise ValueError("Transition action differs from PPO-LSTM's selected action")
        observation = transition.next_state.previous_observation
        if observation is None:
            raise ValueError("PPO-LSTM requires a receiver observation after each action")
        slots = int(transition.next_state.time_slot) - int(transition.state.time_slot)
        if slots < 1:
            raise ValueError("Environment state clock must advance by at least one base slot")
        band = int(observation.selected_band)
        if not 0 <= band < self.bands:
            raise ValueError("Receiver observation selected a band outside PPO-LSTM action space")
        if slots not in (1, 2):
            raise ValueError("PPO-LSTM supports the native one- or two-slot dwell")
        self._update_belief(band, bool(observation.hit), slots)
        end_slot = int(transition.next_state.time_slot)
        if observation.hit:
            times = self.hit_times[band]
            times.append(float(end_slot))
            if len(times) > self.period_history:
                del times[:-self.period_history]
            self.last_hit_slot[band] = end_slot
        self.last_visit_end[band] = end_slot
        self.previous_action = band
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
            self._discounts.append(self.gamma_base ** slots)
            self._lambda_discounts.append(self.gae_lambda_base ** slots)
            self._terminals.append(float(transition.terminated or transition.truncated))
        self._pending = None

    def _learn_episode(self) -> dict:
        observations = np.asarray(self._observations, dtype=np.float32)
        actions_all = np.asarray(self._actions, dtype=np.int64)
        log_probs_all = np.asarray(self._log_probs, dtype=np.float32)
        hidden_h_all = np.asarray(self._hidden_h, dtype=np.float32)
        hidden_c_all = np.asarray(self._hidden_c, dtype=np.float32)
        rewards = np.asarray(self._rewards, dtype=np.float32)
        values = np.asarray(self._values, dtype=np.float32)
        discounts = np.asarray(self._discounts, dtype=np.float32)
        lambda_discounts = np.asarray(self._lambda_discounts, dtype=np.float32)
        terminals = np.asarray(self._terminals, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            next_value = 0.0 if index == len(rewards) - 1 else float(values[index + 1])
            nonterminal = 1.0 - float(terminals[index])
            delta = float(rewards[index]) + float(discounts[index]) * nonterminal * next_value - float(values[index])
            gae = delta + float(discounts[index]) * float(lambda_discounts[index]) * nonterminal * gae
            advantages[index] = gae
        returns = advantages + values
        advantages = ((advantages - advantages.mean()) /
                      (advantages.std() + 1e-8)).astype(np.float32)

        chunks = [(start, min(start + self.bptt_chunk, len(rewards)))
                  for start in range(0, len(rewards), self.bptt_chunk)]
        # Use training actions because the number of decisions per episode is
        # policy-dependent under mixed native dwell lengths.
        learning_rate_fraction = max(
            0.0, 1.0 - self.total_actions / self.training_actions_target,
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
                sequence_length = max(end - start for start, end in batch)
                obs = torch.zeros(batch_size, sequence_length, self.observation_dim, device=self.device)
                actions = torch.zeros(batch_size, sequence_length, dtype=torch.long, device=self.device)
                old_logp = torch.zeros(batch_size, sequence_length, device=self.device)
                adv = torch.zeros(batch_size, sequence_length, device=self.device)
                ret = torch.zeros(batch_size, sequence_length, device=self.device)
                old_value = torch.zeros(batch_size, sequence_length, device=self.device)
                mask = torch.zeros(batch_size, sequence_length, device=self.device)
                h0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
                c0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
                for row, (start, end) in enumerate(batch):
                    length = end - start
                    obs[row, :length] = torch.as_tensor(observations[start:end], dtype=torch.float32,
                                                        device=self.device)
                    actions[row, :length] = torch.as_tensor(actions_all[start:end], dtype=torch.long,
                                                            device=self.device)
                    old_logp[row, :length] = torch.as_tensor(log_probs_all[start:end], dtype=torch.float32,
                                                              device=self.device)
                    adv[row, :length] = torch.as_tensor(advantages[start:end], dtype=torch.float32,
                                                        device=self.device)
                    ret[row, :length] = torch.as_tensor(returns[start:end], dtype=torch.float32,
                                                        device=self.device)
                    old_value[row, :length] = torch.as_tensor(values[start:end], dtype=torch.float32,
                                                              device=self.device)
                    mask[row, :length] = 1.0
                    h0[:, row, :] = torch.as_tensor(hidden_h_all[start], dtype=torch.float32,
                                                    device=self.device)
                    c0[:, row, :] = torch.as_tensor(hidden_c_all[start], dtype=torch.float32,
                                                    device=self.device)

                logits, predicted_value, _ = self.model.sequence(obs, (h0, c0))
                distribution = Categorical(logits=logits)
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
                    stats[key].append(float(value.item()))
                epoch_kls.append(float(approximate_kl.item()))
            if epoch_kls and float(np.mean(epoch_kls)) > self.target_kl:
                break
        self.total_updates += 1
        result = {key: float(np.mean(value)) for key, value in stats.items() if value}
        result["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        result["learning_rate_fraction"] = float(learning_rate_fraction)
        return result

    def end_episode(self, *, training: bool) -> dict:
        if self._pending is not None:
            raise RuntimeError("Cannot end PPO-LSTM episode with an unobserved action")
        if training and self._rewards:
            self._last_losses = self._learn_episode()
            self.completed_training_episodes += 1
        result = {
            "algorithm": "ppo_lstm",
            "training_updates": int(training and bool(self._last_losses)),
            "total_training_actions": self.total_actions,
            "completed_training_episodes": self.completed_training_episodes,
            "episode_decisions": self.episode_decisions,
            "episode_reward": self.episode_reward,
            "band_selection_counts": self.band_selection_counts.tolist(),
            **{f"mean_{key}": value for key, value in self._last_losses.items()},
        }
        return result

    def save(self, path) -> None:
        torch.save({
            "format": "vyapti_ppo_lstm_v1", "bands": self.bands,
            "observation_schema": OBSERVATION_SCHEMA,
            "settings": self.settings, "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), "total_actions": self.total_actions,
            "total_updates": self.total_updates,
            "completed_training_episodes": self.completed_training_episodes,
            "rng_state": self.rng.bit_generator.state,
        }, Path(path))

    def _load(self, path: Path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload.get("format") != "vyapti_ppo_lstm_v1" or payload.get("bands") != self.bands:
            raise ValueError("PPO-LSTM checkpoint format or action-space size does not match this run")
        if payload.get("observation_schema") != OBSERVATION_SCHEMA:
            raise ValueError("PPO-LSTM checkpoint uses a different observation schema; train a fresh run")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.total_actions = int(payload.get("total_actions", 0))
        self.total_updates = int(payload.get("total_updates", 0))
        self.completed_training_episodes = int(payload.get("completed_training_episodes", self.total_updates))
        if "rng_state" in payload:
            self.rng.bit_generator.state = payload["rng_state"]


def create(*, bands: int, seed: int, settings: dict, checkpoint=None):
    return PPOLSTMPolicy(bands, seed, settings, checkpoint)

from __future__ import annotations

# ============================================================
# VYAPTI — RECURRENT PPO-LSTM | TRAIN-500 ONLINE-WORLD PROTOCOL
# ============================================================
#
# RUN THIS AS ONE COLAB CELL after the shared causal_harness.py
# and vyapti_train500_cache.py files are available in the repo.
#
# PROTOCOL
# --------
# 2,500 STARE TRAIN configs
#        -> user's already-created 500-config cache
#        -> online fresh 30-s world per episode
#        -> receiver / causal observations
#        -> collect 512 ACTION transitions
#        -> 5 PPO optimizer epochs over those transitions
#        -> repeat to 200,000 ACTION steps
#        -> validate at 50k / 100k / 150k / 200k
#        -> select best validation checkpoint
#        -> TEST exactly once
#
# IMPORTANT
# ---------
# There are NO world-generation epochs here.
# PPO optimizer epochs = 5 is the only "epoch" parameter in training.
#
# Recurrent correctness:
# - exact LSTM (h,c) state BEFORE every rollout action is stored;
# - BPTT chunks initialize from that stored rollout state;
# - chunks do not silently restart from zero;
# - hidden state is reset whenever a fresh episode/world begins;
# - GAE is computed per episode and bootstraps only on rollout truncation.
#
# ============================================================

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm.auto import tqdm


# ============================================================
# 0. USER CONFIGURATION — CHANGE THESE ONLY FOR TUNING
# ============================================================

REPO_ROOT = Path("/content/Vyapti")
CORPUS_ROOT = REPO_ROOT / "tsrd_stare"
CACHE_ROOT = REPO_ROOT / "runs" / "tsrd_cache" / "train_500"
PORMAB_PRIOR_DIR = REPO_ROOT / "runs" / "pormab_500pool" / "prior"
OUTPUT_DIR = REPO_ROOT / "runs" / "ppo_lstm_500pool"

SEED = 20260929

# Training protocol
DWELL_SLOTS = 2                  # 2 x 50 ms = 100 ms action dwell
ROLLOUT_ACTIONS = 512
PPO_OPTIMIZER_EPOCHS = 5
TRAINING_ACTION_STEPS = 200_000
CHECKPOINT_STEPS = (50_000, 100_000, 150_000, 200_000)

# PPO hyperparameters
LEARNING_RATE = 2.0e-4
GAMMA = 0.997
GAE_LAMBDA = 0.98
CLIP_EPS = 0.20
ENTROPY_COEF = 0.005
VALUE_COEF = 0.50
MAX_GRAD_NORM = 0.50
TARGET_KL = 0.03

# Recurrent optimization
HIDDEN_SIZE = 256
BPTT_CHUNK = 128
SEQUENCE_MINIBATCH = 8

# Causal feature set used for the headline recurrent run
# base / belief / belief_periodic
FEATURE_MODE = "belief_periodic"

# Fixed reward for the controlled experiment
REWARD_MODE = "detector_positive"

# Final TEST is run once after best validation checkpoint selection.
# Set True for tuning/screening runs to avoid TEST leakage.
SKIP_TEST = False

# Set a smaller value such as 50_000 for a tuning/smoke run.
# None means the full 200,000-action experiment.
MAX_ACTION_STEPS = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. MAKE REPO IMPORTABLE
# ============================================================

REPO_ROOT.mkdir(parents=True, exist_ok=True)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The shared harness and cache adapter must exist in the repository.
from causal_harness import (
    N_BANDS,
    N_BASE_SLOTS,
    MISSION_S,
    BASE_SLOT_S,
    BAND_STEP_MHZ,
    RECEIVER_BW_MHZ,
    OVERLAP_FRACTION,
    PD,
    PFA,
    RETUNE_TIME_MS,
    VAL_FILES,
    TEST_FILES,
    PPO_ROLLOUT_ACTIONS,
    PPO_OPTIMIZER_EPOCHS,
    PPO_TRAINING_STEPS,
    CHECKPOINT_STEPS,
    DEFAULT_SEED,
    VAL_REPLAY_SEED,
    TEST_REPLAY_SEED,
    CausalSchedulerState,
    build_replay_registry,
    aggregate_metrics,
    detector_sequence,
    save_json,
    validate_dataset,
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


# ============================================================
# 2. GLOBAL HELPERS
# ============================================================

FEATURE_CHOICES = ("base", "belief", "belief_periodic")


def fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def compose_observation(state: CausalSchedulerState, mode: str) -> np.ndarray:
    if mode not in FEATURE_CHOICES:
        raise ValueError(f"Unknown feature_mode={mode!r}")

    f = state.pre_action_features()
    prev_onehot = np.zeros(N_BANDS, dtype=np.float32)
    previous_action = int(state.previous_action)
    if 0 <= previous_action < N_BANDS:
        prev_onehot[previous_action] = 1.0

    parts = [
        np.asarray(f["staleness"], dtype=np.float32),
        np.asarray(f["visit_count_norm"], dtype=np.float32),
        prev_onehot,
        np.asarray(
            [
                float(np.clip(state.elapsed_base_slots / max(N_BASE_SLOTS, 1), 0.0, 1.0)),
                float(state.previous_positive),
            ],
            dtype=np.float32,
        ),
    ]

    if mode in ("belief", "belief_periodic"):
        parts.extend([
            np.asarray(f["belief"], dtype=np.float32),
            np.asarray(f["belief_entropy"], dtype=np.float32),
        ])

    if mode == "belief_periodic":
        parts.extend([
            np.asarray(f["periodicity"], dtype=np.float32),
            np.asarray(f["periodicity_confidence"], dtype=np.float32),
            np.asarray(f["period_norm"], dtype=np.float32),
        ])

    return np.concatenate(parts).astype(np.float32)


# ============================================================
# 3. LSTM ACTOR-CRITIC
# ============================================================

class LSTMActorCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = HIDDEN_SIZE,
        n_actions: int = N_BANDS,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden)
        self.n_actions = int(n_actions)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.policy = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, self.n_actions),
        )

        self.value = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def initial_hidden(self, batch: int = 1):
        h = torch.zeros(1, int(batch), self.hidden_size, device=DEVICE)
        c = torch.zeros(1, int(batch), self.hidden_size, device=DEVICE)
        return h, c

    def forward_step(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ):
        x = self.encoder(obs).unsqueeze(1)
        y, next_hidden = self.lstm(x, hidden)
        z = y[:, 0, :]
        logits = self.policy(z)
        value = self.value(z).squeeze(-1)
        return logits, value, next_hidden

    def forward_sequence(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ):
        x = self.encoder(obs)
        y, next_hidden = self.lstm(x, hidden)
        logits = self.policy(y)
        values = self.value(y).squeeze(-1)
        return logits, values, next_hidden


def hidden_to_numpy(
    hidden: tuple[torch.Tensor, torch.Tensor]
) -> tuple[np.ndarray, np.ndarray]:
    h, c = hidden
    return (
        h.detach().cpu().numpy()[0, 0].astype(np.float32, copy=True),
        c.detach().cpu().numpy()[0, 0].astype(np.float32, copy=True),
    )


# ============================================================
# 4. ROLLOUT STORAGE + GAE
# ============================================================

@dataclass
class Episode:
    obs: np.ndarray
    actions: np.ndarray
    old_logp: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    dwell_slots: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    rnn_h: np.ndarray
    rnn_c: np.ndarray
    terminal: bool
    bootstrap_value: float


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dwell_slots: np.ndarray,
    gamma: float,
    gae_lambda: float,
    bootstrap_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Semi-Markov GAE within one episode.

    The function is called per episode, so no GAE term can cross an
    environment terminal boundary. For a rollout truncation, the caller
    supplies the critic bootstrap value of the post-rollout state.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    gae = 0.0

    for t in range(T - 1, -1, -1):
        d = max(1, int(dwell_slots[t]))
        gamma_d = float(gamma) ** d
        lambda_d = float(gae_lambda) ** d
        next_value = float(bootstrap_value) if t == T - 1 else float(values[t + 1])
        delta = float(rewards[t]) + gamma_d * next_value - float(values[t])
        gae = delta + gamma_d * lambda_d * gae
        advantages[t] = gae

    return advantages, advantages + values


def collect_one_episode(
    env: Any,
    model: LSTMActorCritic,
    transition: np.ndarray,
    prior_active: np.ndarray,
    feature_mode: str,
    dwell_slots: int,
    action_budget: int,
    receiver_seed: int,
) -> tuple[Episode, int]:
    """Collect one complete/partial episode from one fresh online world."""

    state = CausalSchedulerState.create(transition, prior_active)
    state.reset(prior_active)
    env.reset(seed=int(receiver_seed))

    # A NEW WORLD means a NEW LSTM state.
    hidden = model.initial_hidden(1)
    obs = compose_observation(state, feature_mode)

    obs_list: list[np.ndarray] = []
    action_list: list[int] = []
    logp_list: list[float] = []
    value_list: list[float] = []
    reward_list: list[float] = []
    dwell_list: list[int] = []
    hidden_h_list: list[np.ndarray] = []
    hidden_c_list: list[np.ndarray] = []

    terminal = False
    steps_taken = 0

    while steps_taken < int(action_budget) and not env.done:
        # EXACT recurrent state that existed immediately before this action.
        h_np, c_np = hidden_to_numpy(hidden)
        hidden_h_list.append(h_np)
        hidden_c_list.append(c_np)

        x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            logits, value, next_hidden = model.forward_step(x, hidden)
            dist = Categorical(logits=logits)
            action_t = dist.sample()
            action = int(action_t.item())
            old_logp = float(dist.log_prob(action_t).item())
            old_value = float(value.item())

        hidden = tuple(v.detach() for v in next_hidden)

        # Use the verified simulator API. Each 100-ms action produces two
        # sequential base-slot observations.
        looks, _env_reward, done = env.step_dwell_training(
            int(action),
            int(dwell_slots),
            REWARD_MODE,
        )
        looks = list(looks)

        if len(looks) == 0:
            raise RuntimeError("Environment returned zero observations for a dwell")
        if len(looks) != int(dwell_slots) and not done:
            raise RuntimeError(
                f"Expected {dwell_slots} looks, got {len(looks)} before episode termination"
            )

        positives = detector_sequence(looks)

        # Causal belief update occurs AFTER the action's observed hits.
        # The controlled reward is detector-positive, so reward is simply
        # the number of positive receiver observations in this dwell.
        state.belief.step_observations(action, positives)
        reward = float(np.sum(np.asarray(positives, dtype=np.float64)))

        obs_list.append(obs.copy())
        action_list.append(action)
        logp_list.append(old_logp)
        value_list.append(old_value)
        reward_list.append(reward)
        dwell_list.append(int(len(looks)))

        state.step(action, int(len(looks)), positives)
        steps_taken += 1
        terminal = bool(done)

        if terminal:
            break

        obs = compose_observation(state, feature_mode)

    # At a true environment terminal, V(next)=0.
    # At rollout truncation, bootstrap from the actual post-action recurrent state.
    if terminal:
        bootstrap = 0.0
    else:
        next_obs = compose_observation(state, feature_mode)
        x_next = torch.as_tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            _logits, next_value, _next_hidden = model.forward_step(x_next, hidden)
        bootstrap = float(next_value.item())

    rewards = np.asarray(reward_list, dtype=np.float64)
    values = np.asarray(value_list, dtype=np.float64)
    dwells = np.asarray(dwell_list, dtype=np.int64)

    advantages, returns = compute_gae(
        rewards,
        values,
        dwells,
        GAMMA,
        GAE_LAMBDA,
        bootstrap,
    )

    return (
        Episode(
            obs=np.asarray(obs_list, dtype=np.float32),
            actions=np.asarray(action_list, dtype=np.int64),
            old_logp=np.asarray(logp_list, dtype=np.float32),
            old_values=np.asarray(value_list, dtype=np.float32),
            rewards=rewards.astype(np.float32),
            dwell_slots=dwells,
            advantages=advantages.astype(np.float32),
            returns=returns.astype(np.float32),
            rnn_h=np.asarray(hidden_h_list, dtype=np.float32),
            rnn_c=np.asarray(hidden_c_list, dtype=np.float32),
            terminal=bool(terminal),
            bootstrap_value=float(bootstrap),
        ),
        steps_taken,
    )


# ============================================================
# 5. PPO OPTIMIZER
# ============================================================

class PPOTrainer:
    def __init__(self, model: LSTMActorCritic, seed: int):
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(LEARNING_RATE),
        )
        self.update_count = 0
        self.rng = np.random.default_rng(int(seed) + 910_003)

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        if not episodes:
            raise RuntimeError("Empty PPO rollout")

        # Normalize advantages globally across this rollout only.
        all_adv = np.concatenate([ep.advantages for ep in episodes]).astype(np.float64)
        mean_adv = float(all_adv.mean())
        std_adv = float(all_adv.std() + 1e-8)
        for ep in episodes:
            ep.advantages = ((ep.advantages - mean_adv) / std_adv).astype(np.float32)

        # Build truncated BPTT chunks. Each chunk starts from the EXACT LSTM
        # state that existed before the corresponding rollout action.
        chunks: list[tuple[Episode, int, int]] = []
        for ep in episodes:
            for start in range(0, len(ep.actions), int(BPTT_CHUNK)):
                end = min(len(ep.actions), start + int(BPTT_CHUNK))
                chunks.append((ep, start, end))

        stats: dict[str, list[float]] = {
            "loss": [],
            "kl": [],
            "clip": [],
            "entropy": [],
            "value_loss": [],
        }

        for _epoch in range(int(PPO_OPTIMIZER_EPOCHS)):
            order = self.rng.permutation(len(chunks))
            epoch_kls: list[float] = []

            for mb_start in range(0, len(order), int(SEQUENCE_MINIBATCH)):
                batch_indices = order[mb_start : mb_start + int(SEQUENCE_MINIBATCH)]
                batch = [chunks[int(i)] for i in batch_indices]

                B = len(batch)
                max_len = max(end - start for _, start, end in batch)

                obs_t = torch.zeros(
                    B, max_len, self.model.input_dim, device=DEVICE
                )
                act_t = torch.zeros(
                    B, max_len, dtype=torch.long, device=DEVICE
                )
                old_lp_t = torch.zeros(B, max_len, device=DEVICE)
                adv_t = torch.zeros(B, max_len, device=DEVICE)
                ret_t = torch.zeros(B, max_len, device=DEVICE)
                mask_t = torch.zeros(B, max_len, device=DEVICE)

                h0 = torch.zeros(
                    1, B, self.model.hidden_size, device=DEVICE
                )
                c0 = torch.zeros(
                    1, B, self.model.hidden_size, device=DEVICE
                )

                for i, (ep, start, end) in enumerate(batch):
                    n = end - start
                    obs_t[i, :n] = torch.as_tensor(
                        ep.obs[start:end], dtype=torch.float32, device=DEVICE
                    )
                    act_t[i, :n] = torch.as_tensor(
                        ep.actions[start:end], dtype=torch.long, device=DEVICE
                    )
                    old_lp_t[i, :n] = torch.as_tensor(
                        ep.old_logp[start:end], dtype=torch.float32, device=DEVICE
                    )
                    adv_t[i, :n] = torch.as_tensor(
                        ep.advantages[start:end], dtype=torch.float32, device=DEVICE
                    )
                    ret_t[i, :n] = torch.as_tensor(
                        ep.returns[start:end], dtype=torch.float32, device=DEVICE
                    )
                    mask_t[i, :n] = 1.0

                    # Exact hidden state from rollout, not zeros unless this
                    # really was the beginning of the episode.
                    h0[:, i, :] = torch.as_tensor(
                        ep.rnn_h[start], dtype=torch.float32, device=DEVICE
                    )
                    c0[:, i, :] = torch.as_tensor(
                        ep.rnn_c[start], dtype=torch.float32, device=DEVICE
                    )

                logits, values, _ = self.model.forward_sequence(
                    obs_t,
                    (h0, c0),
                )

                dist = Categorical(logits=logits)
                logp = dist.log_prob(act_t)
                entropy = dist.entropy()

                ratio = torch.exp(logp - old_lp_t)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - float(CLIP_EPS),
                    1.0 + float(CLIP_EPS),
                )

                denom = mask_t.sum().clamp_min(1.0)
                policy_loss = -(
                    torch.minimum(
                        ratio * adv_t,
                        clipped_ratio * adv_t,
                    )
                    * mask_t
                ).sum() / denom

                value_loss = (
                    ((values - ret_t) ** 2) * mask_t
                ).sum() / denom

                entropy_mean = (
                    entropy * mask_t
                ).sum() / denom

                loss = (
                    policy_loss
                    + float(VALUE_COEF) * value_loss
                    - float(ENTROPY_COEF) * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(MAX_GRAD_NORM),
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (
                        (old_lp_t - logp) * mask_t
                    ).sum() / denom
                    clip_frac = (
                        ((ratio - 1.0).abs() > float(CLIP_EPS)).float()
                        * mask_t
                    ).sum() / denom

                vals = {
                    "loss": float(loss.item()),
                    "kl": float(approx_kl.item()),
                    "clip": float(clip_frac.item()),
                    "entropy": float(entropy_mean.item()),
                    "value_loss": float(value_loss.item()),
                }

                for key, value in vals.items():
                    stats[key].append(value)

                epoch_kls.append(vals["kl"])

            # PPO early-stop safeguard.
            if epoch_kls and float(np.mean(epoch_kls)) > float(TARGET_KL):
                break

        self.update_count += 1
        return {
            key: float(np.mean(values)) if values else float("nan")
            for key, values in stats.items()
        }


# ============================================================
# 6. PO-RMAB PRIOR LOADING
# ============================================================

def load_prior(
    prior_dir: Path,
    fingerprint: str,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(prior_dir) / "pormab_transition.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing PO-RMAB prior: {path}\n"
            "Run the 500-config PO-RMAB calibration first."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("source_pool_fingerprint") != fingerprint:
        raise RuntimeError(
            "PO-RMAB prior fingerprint does not match the saved TRAIN-500 cache"
        )

    if abs(float(payload.get("model_pd", -1.0)) - PD) > 1e-12:
        raise RuntimeError("PO-RMAB prior model Pd does not match Pd=0.90")
    if abs(float(payload.get("model_pfa", -1.0)) - PFA) > 1e-12:
        raise RuntimeError("PO-RMAB prior model Pfa does not match Pfa=0.05")

    transition = np.asarray(payload["transition"], dtype=np.float64)
    prior_active = np.asarray(payload["prior_active"], dtype=np.float64)

    if transition.shape != (N_BANDS, 2, 2):
        raise RuntimeError("Malformed transition model shape")
    if prior_active.shape != (N_BANDS,):
        raise RuntimeError("Malformed prior-active shape")

    return transition, prior_active


# ============================================================
# 7. VALIDATION ENVIRONMENT
# ============================================================

def build_replay_environment(stare_file: Path):
    from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment

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


def metric_value(
    row: dict[str, Any],
    aliases: tuple[str, ...],
) -> float:
    for key in aliases:
        if key not in row or row[key] is None:
            continue
        try:
            value = float(row[key])
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return float("nan")


def summarize_validation(rows: list[dict[str, Any]]) -> dict[str, float]:
    oir = [
        metric_value(r, ("opportunity_interception_ratio", "oir", "mean_oir"))
        for r in rows
    ]
    eir = [
        metric_value(r, ("emitter_interception_ratio", "mean_emitter_interception_ratio"))
        for r in rows
    ]
    ttfi = [
        metric_value(
            r,
            ("censored_ttfi_s", "censored_ttf_s", "median_censored_ttfi_s", "censored_ttf_ms"),
        )
        for r in rows
    ]

    oir = [x for x in oir if np.isfinite(x)]
    eir = [x for x in eir if np.isfinite(x)]
    ttfi = [x / 1000.0 if x > 100.0 else x for x in ttfi if np.isfinite(x)]

    pd_values = [
        metric_value(r, ("conditional_pd", "pd", "mean_pd"))
        for r in rows
    ]
    pfa_values = [
        metric_value(r, ("true_pfa", "pfa", "mean_pfa"))
        for r in rows
    ]

    return {
        "opportunity_interception_ratio": float(np.mean(oir)) if oir else float("nan"),
        "emitter_interception_ratio": float(np.mean(eir)) if eir else float("nan"),
        "median_censored_ttfi_s": float(np.median(ttfi)) if ttfi else float("nan"),
        "conditional_pd": float(np.nanmean(pd_values)) if pd_values else float("nan"),
        "true_pfa": float(np.nanmean(pfa_values)) if pfa_values else float("nan"),
        "n_episodes": float(len(rows)),
    }


def evaluate_model(
    model: LSTMActorCritic,
    registry: list[dict[str, Any]],
    transition: np.ndarray,
    prior_active: np.ndarray,
    feature_mode: str,
    dwell_slots: int,
    seed: int,
) -> dict[str, Any]:
    from vyapti_simulator.tsrd.benchmark_protocol import (
        score_recorded_replay,
        _assert_scorecard_consistent,
        _seed_for_file,
    )
    from vyapti_simulator.core.metrics import TrajectoryStep

    model.eval()
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    bar = tqdm(
        registry,
        desc="VAL PPO-LSTM" if registry and registry[0].get("split") == "val" else "REPLAY PPO-LSTM",
        unit="scene",
    )

    try:
        for recipe in bar:
            stare_file = Path(recipe["source_file"])
            env = build_replay_environment(stare_file)
            receiver_seed = _seed_for_file(stare_file.stem, seed)
            env.reset(seed=int(receiver_seed))

            state = CausalSchedulerState.create(
                transition,
                prior_active,
            )
            state.reset(prior_active)

            # New evaluation episode -> zero recurrent state.
            hidden = model.initial_hidden(1)
            obs = compose_observation(state, feature_mode)
            trajectory: list[TrajectoryStep] = []
            actions = 0

            while not env.done:
                x = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=DEVICE,
                ).unsqueeze(0)

                with torch.no_grad():
                    logits, _value, next_hidden = model.forward_step(
                        x,
                        hidden,
                    )
                    action = int(
                        torch.argmax(logits, dim=-1).item()
                    )

                hidden = tuple(v.detach() for v in next_hidden)

                looks, _reward, done = env.step_dwell_training(
                    int(action),
                    int(dwell_slots),
                    REWARD_MODE,
                )
                looks = list(looks)

                if len(looks) == 0:
                    raise RuntimeError("Evaluation returned zero observations")
                if len(looks) != int(dwell_slots) and not done:
                    raise RuntimeError(
                        f"Expected {dwell_slots} looks, got {len(looks)}"
                    )

                positives = detector_sequence(looks)

                for look in looks:
                    trajectory.append(
                        TrajectoryStep(
                            action=action,
                            time_slot=int(look["time_slot"]),
                            observation=look,
                        )
                    )

                state.belief.step_observations(action, positives)
                state.step(action, len(looks), positives)
                actions += 1

                if not done:
                    obs = compose_observation(state, feature_mode)

            score = score_recorded_replay(env, trajectory)
            _assert_scorecard_consistent(score, trajectory)

            rows.append(
                {
                    "world_id": int(recipe["world_id"]),
                    "config": stare_file.stem,
                    "source_file": str(stare_file),
                    "ppo_action_steps": actions,
                    **score,
                }
            )

            elapsed = time.perf_counter() - t0
            rate = len(rows) / max(elapsed, 1e-9)
            eta = (len(registry) - len(rows)) / max(rate, 1e-9)
            bar.set_postfix_str(
                f"elapsed={fmt_seconds(elapsed)} ETA={fmt_seconds(eta)}"
            )
    finally:
        model.train()

    return {
        "summary": aggregate_metrics(rows),
        "rows": rows,
        "runtime_s": time.perf_counter() - t0,
    }


def checkpoint_better(
    current: dict[str, float],
    best: dict[str, float] | None,
) -> bool:
    if best is None:
        return True

    # Frozen checkpoint selection order:
    # 1) higher OIR
    # 2) lower median censored TTFI
    # 3) higher emitter interception ratio
    criteria = (
        ("opportunity_interception_ratio", True),
        ("censored_median_ttfi_s", False),
        ("pooled_unique_emitter_interception_rate", True),
    )

    for key, higher in criteria:
        a = float(current.get(key, np.nan))
        b = float(best.get(key, np.nan))

        if not np.isfinite(a):
            return False
        if not np.isfinite(b):
            return True

        if not math.isclose(
            a,
            b,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return a > b if higher else a < b

    return False


def save_checkpoint(
    path: Path,
    model: LSTMActorCritic,
    trainer: PPOTrainer,
    manifest: dict[str, Any],
    total_action_steps: int,
    milestone: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "manifest": manifest,
            "actual_action_steps": int(total_action_steps),
            "checkpoint_milestone": int(milestone),
            "training_update": int(trainer.update_count),
        },
        path,
    )


# ============================================================
# 8. MAIN TRAINING LOOP
# ============================================================

def run_training() -> None:
    if FEATURE_MODE not in FEATURE_CHOICES:
        raise ValueError(f"FEATURE_MODE must be one of {FEATURE_CHOICES}")
    if int(DWELL_SLOTS) not in (1, 2):
        raise ValueError("DWELL_SLOTS must be 1 or 2")
    if int(ROLLOUT_ACTIONS) <= 0:
        raise ValueError("ROLLOUT_ACTIONS must be positive")
    if int(PPO_OPTIMIZER_EPOCHS) <= 0:
        raise ValueError("PPO_OPTIMIZER_EPOCHS must be positive")

    set_global_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("VYAPTI — RECURRENT PPO-LSTM | TRAIN-500 ONLINE WORLD")
    print("=" * 78)
    print(f"Device                : {DEVICE}")
    print(f"2,500 TRAIN configs   : fixed source corpus")
    print(f"Selected source pool  : 500 configs (existing cache)")
    print(f"Mission               : {MISSION_S:.1f} s")
    print(f"Dwell                 : {50 * DWELL_SLOTS} ms")
    print(f"Rollout               : {ROLLOUT_ACTIONS} action transitions")
    print(f"PPO optimizer epochs  : {PPO_OPTIMIZER_EPOCHS}")
    print(f"Total action budget   : {TRAINING_ACTION_STEPS:,}")
    print(f"Feature mode          : {FEATURE_MODE}")
    print(f"Reward                : {REWARD_MODE}")
    print(f"LR / gamma / GAE      : {LEARNING_RATE} / {GAMMA} / {GAE_LAMBDA}")
    print(f"LSTM hidden           : {HIDDEN_SIZE}")
    print(f"BPTT / seq minibatch  : {BPTT_CHUNK} / {SEQUENCE_MINIBATCH}")
    print(f"Output                : {OUTPUT_DIR}")
    print("=" * 78)

    counts = validate_dataset(CORPUS_ROOT)
    print(
        f"Dataset validated: TRAIN={counts['train']}, "
        f"VAL={counts['val']}, TEST={counts['test']}"
    )

    # Existing cache only — no 2,500-file preprocessing is repeated.
    cache = load_train500_cache(CACHE_ROOT)
    pool, cache = build_train_pool_from_cache(
        CORPUS_ROOT,
        CACHE_ROOT,
    )
    write_runtime_manifest(cache)

    print("\n✅ Existing TRAIN-500 cache loaded")
    print(f"   fingerprint: {cache['fingerprint']}")
    print(f"   selected configs: {len(cache['selected_config_ids'])}")
    print(f"   emitter contributions: {len(cache['emitter_index'])}")

    transition, prior_active = load_prior(
        PORMAB_PRIOR_DIR,
        cache["fingerprint"],
    )

    state0 = CausalSchedulerState.create(
        transition,
        prior_active,
    )
    input_dim = len(
        compose_observation(
            state0,
            FEATURE_MODE,
        )
    )

    model = LSTMActorCritic(
        input_dim=input_dim,
        hidden=HIDDEN_SIZE,
    ).to(DEVICE)

    trainer = PPOTrainer(
        model,
        SEED,
    )

    val_registry = build_replay_registry(
        CORPUS_ROOT,
        "val",
        VAL_FILES,
    )

    save_json(
        OUTPUT_DIR / "val_replay_registry.json",
        val_registry,
    )

    target_steps = int(
        TRAINING_ACTION_STEPS
        if MAX_ACTION_STEPS is None
        else min(TRAINING_ACTION_STEPS, int(MAX_ACTION_STEPS))
    )

    checkpoints = [
        int(x)
        for x in CHECKPOINT_STEPS
        if int(x) <= target_steps
    ]

    manifest = {
        "algorithm": "PPO-LSTM",
        "protocol": (
            "2,500 STARE TRAIN -> saved stratified 500-config source pool -> "
            "online fresh 30-second world per episode -> 512-action rollouts -> "
            "5 PPO optimizer epochs -> repeat"
        ),
        "available_train_configs": 2500,
        "selected_train_configs": 500,
        "source_pool_fingerprint": cache["fingerprint"],
        "world_generation": (
            "online fresh world per episode; no world-generation epochs; "
            "no fixed training-world registry"
        ),
        "mission_s": MISSION_S,
        "base_slot_s": BASE_SLOT_S,
        "n_base_slots": N_BASE_SLOTS,
        "n_bands": N_BANDS,
        "band_spacing_mhz": BAND_STEP_MHZ,
        "receiver_bw_mhz": RECEIVER_BW_MHZ,
        "overlap_fraction": OVERLAP_FRACTION,
        "receiver_pd": PD,
        "receiver_pfa": PFA,
        "retune_ms": RETUNE_TIME_MS,
        "dwell_slots": DWELL_SLOTS,
        "dwell_ms": 50 * int(DWELL_SLOTS),
        "reward": REWARD_MODE,
        "feature_mode": FEATURE_MODE,
        "input_dim": input_dim,
        "lstm_hidden": HIDDEN_SIZE,
        "rollout_actions": ROLLOUT_ACTIONS,
        "optimizer_epochs": PPO_OPTIMIZER_EPOCHS,
        "training_action_target": target_steps,
        "checkpoint_milestones": checkpoints,
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "clip_eps": CLIP_EPS,
            "entropy_coef": ENTROPY_COEF,
            "value_coef": VALUE_COEF,
            "max_grad_norm": MAX_GRAD_NORM,
            "target_kl": TARGET_KL,
            "bptt_chunk": BPTT_CHUNK,
            "sequence_minibatch": SEQUENCE_MINIBATCH,
        },
        "dataset_counts": counts,
        "recurrent_correctness": {
            "stores_hidden_before_every_action": True,
            "stores_both_lstm_h_and_c": True,
            "bptt_chunk_uses_stored_rollout_hidden": True,
            "hidden_reset_at_new_world": True,
            "gae_respects_episode_boundaries": True,
        },
        "information_boundary": {
            "hidden_truth_to_policy": False,
            "hidden_truth_to_reward": False,
            "hidden_truth_to_evaluator": True,
        },
    }

    save_json(
        OUTPUT_DIR / "experiment_manifest.json",
        manifest,
    )

    best_summary: dict[str, float] | None = None
    total_steps = 0
    checkpoint_index = 0
    t0 = time.perf_counter()

    progress = tqdm(
        total=target_steps,
        desc="PPO-LSTM 500pool",
        unit="action",
    )

    last_update_stats: dict[str, float] = {}
    world_rng = np.random.default_rng(int(SEED) + 1_003_003)

    while total_steps < target_steps:
        remaining = target_steps - total_steps
        rollout_target = min(
            int(ROLLOUT_ACTIONS),
            int(remaining),
        )

        episodes: list[Episode] = []
        rollout_steps = 0

        while rollout_steps < rollout_target:
            # Each episode gets a fresh online world from the same saved
            # 500-config source pool.
            env, _sources, world_seed, _emitter_count = sample_fresh_training_world(
                pool,
                world_rng,
            )

            episode_budget = rollout_target - rollout_steps

            episode, used = collect_one_episode(
                env=env,
                model=model,
                transition=transition,
                prior_active=prior_active,
                feature_mode=FEATURE_MODE,
                dwell_slots=int(DWELL_SLOTS),
                action_budget=int(episode_budget),
                receiver_seed=int(world_seed),
            )

            if used <= 0:
                raise RuntimeError("Training episode produced zero action steps")

            episodes.append(episode)
            rollout_steps += used

        # IMPORTANT: exactly this rollout is reused for 5 PPO optimizer epochs.
        last_update_stats = trainer.update(episodes)
        total_steps += rollout_steps

        elapsed = time.perf_counter() - t0
        rate = total_steps / max(elapsed, 1e-9)
        eta = (target_steps - total_steps) / max(rate, 1e-9)

        progress.update(rollout_steps)
        progress.set_postfix_str(
            "elapsed="
            f"{fmt_seconds(elapsed)} "
            f"rate={rate:.1f}/s "
            f"ETA={fmt_seconds(eta)} "
            f"KL={last_update_stats.get('kl', float('nan')):.4f}"
        )

        while (
            checkpoint_index < len(checkpoints)
            and total_steps >= checkpoints[checkpoint_index]
        ):
            milestone = int(checkpoints[checkpoint_index])
            checkpoint_path = (
                OUTPUT_DIR
                / f"checkpoint_{milestone // 1000:04d}k.pt"
            )

            save_checkpoint(
                checkpoint_path,
                model,
                trainer,
                manifest,
                total_steps,
                milestone,
            )

            validation = evaluate_model(
                model=model,
                registry=val_registry,
                transition=transition,
                prior_active=prior_active,
                feature_mode=FEATURE_MODE,
                dwell_slots=int(DWELL_SLOTS),
                seed=int(VAL_REPLAY_SEED),
            )

            save_json(
                OUTPUT_DIR
                / f"validation_{milestone // 1000:04d}k.json",
                validation,
            )

            current = validation["summary"]

            if checkpoint_better(current, best_summary):
                best_summary = current

                torch.save(
                    {
                        "model": model.state_dict(),
                        "manifest": manifest,
                        "best_validation": best_summary,
                        "best_milestone": milestone,
                        "actual_action_steps": total_steps,
                    },
                    OUTPUT_DIR / "best_validation.pt",
                )

                save_json(
                    OUTPUT_DIR / "best_validation_summary.json",
                    {
                        "best_milestone": milestone,
                        "actual_action_steps": total_steps,
                        "summary": best_summary,
                    },
                )

            checkpoint_index += 1

    progress.close()

    # ========================================================
    # FINAL TEST — EXACTLY ONCE AFTER CHECKPOINT SELECTION
    # ========================================================

    test_result = "skipped"

    if not SKIP_TEST:
        best_path = OUTPUT_DIR / "best_validation.pt"
        if not best_path.exists():
            raise RuntimeError(
                "best_validation.pt does not exist; refusing to run TEST"
            )

        best_blob = torch.load(
            best_path,
            map_location=DEVICE,
        )
        model.load_state_dict(best_blob["model"])

        test_registry = build_replay_registry(
            CORPUS_ROOT,
            "test",
            TEST_FILES,
        )

        test = evaluate_model(
            model=model,
            registry=test_registry,
            transition=transition,
            prior_active=prior_active,
            feature_mode=FEATURE_MODE,
            dwell_slots=int(DWELL_SLOTS),
            seed=int(__import__("causal_harness").TEST_REPLAY_SEED),
        )

        save_json(
            OUTPUT_DIR / "TEST_final.json",
            test,
        )
        test_result = "completed once"

    runtime_s = time.perf_counter() - t0

    run_summary = {
        "algorithm": "PPO-LSTM",
        "total_action_steps": int(total_steps),
        "target_action_steps": int(target_steps),
        "rollout_actions": int(ROLLOUT_ACTIONS),
        "optimizer_epochs": int(PPO_OPTIMIZER_EPOCHS),
        "best_validation": best_summary,
        "test": test_result,
        "runtime_s": runtime_s,
        "runtime": fmt_seconds(runtime_s),
    }

    save_json(
        OUTPUT_DIR / "run_summary.json",
        run_summary,
    )

    print("\n" + "=" * 78)
    print("PPO-LSTM RUN COMPLETE")
    print("=" * 78)
    print(json.dumps(run_summary, indent=2, sort_keys=True, allow_nan=True))


# ============================================================
# 9. RUN
# ============================================================

run_training()

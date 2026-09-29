from __future__ import annotations

"""Vyapti recurrent PPO-GRU under the final 500-config online-world protocol.

TRAINING PROTOCOL
-----------------
2,500 STARE TRAIN
    -> user's existing deterministic stratified 500-config cache
    -> online fresh 30-second world per episode
    -> 512 action rollout target
    -> 5 PPO optimizer epochs over that rollout
    -> repeat to 200,000 action decisions
    -> checkpoints at 50k / 100k / 150k / 200k

The recurrent implementation stores the exact GRU hidden state immediately
before each rollout action. BPTT chunks initialize from that stored rollout
state; they never silently restart at zero. Rollouts may contain multiple
complete episodes. GAE is computed separately inside each episode, with the
correct terminal bootstrap handling.
"""

import argparse
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REWARD_MODE = "detector_positive"
DEFAULT_DWELL_SLOTS = 2
DEFAULT_LR = 2.0e-4
DEFAULT_GAMMA = 0.997
DEFAULT_GAE_LAMBDA = 0.98
DEFAULT_CLIP_EPS = 0.20
DEFAULT_ENTROPY_COEF = 0.005
DEFAULT_VALUE_COEF = 0.50
DEFAULT_MAX_GRAD_NORM = 0.50
DEFAULT_TARGET_KL = 0.03
DEFAULT_HIDDEN = 256
DEFAULT_BPTT = 128
DEFAULT_SEQUENCE_MINIBATCH = 8

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
    if 0 <= int(state.previous_action) < N_BANDS:
        prev_onehot[int(state.previous_action)] = 1.0

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
    terminal: bool
    bootstrap_value: float


class GRUActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden: int = DEFAULT_HIDDEN, n_actions: int = N_BANDS):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden)
        self.n_actions = int(n_actions)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Tanh(),
        )
        self.gru = nn.GRU(
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

    def initial_hidden(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(1, int(batch), self.hidden_size, device=DEVICE)

    def forward_step(self, obs: torch.Tensor, hidden: torch.Tensor):
        x = self.encoder(obs).unsqueeze(1)
        y, next_hidden = self.gru(x, hidden)
        z = y[:, 0, :]
        return self.policy(z), self.value(z).squeeze(-1), next_hidden

    def forward_sequence(self, obs: torch.Tensor, hidden: torch.Tensor):
        x = self.encoder(obs)
        y, next_hidden = self.gru(x, hidden)
        return self.policy(y), self.value(y).squeeze(-1), next_hidden


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dwell_slots: np.ndarray,
    gamma: float,
    gae_lambda: float,
    bootstrap_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        d = max(1, int(dwell_slots[t]))
        gamma_d = float(gamma) ** d
        lam_d = float(gae_lambda) ** d
        next_value = float(bootstrap_value) if t == T - 1 else float(values[t + 1])
        delta = float(rewards[t]) + gamma_d * next_value - float(values[t])
        gae = delta + gamma_d * lam_d * gae
        adv[t] = gae
    return adv, adv + values


def hidden_to_numpy(hidden: torch.Tensor) -> np.ndarray:
    return hidden.detach().cpu().numpy()[0, 0].astype(np.float32, copy=True)


def collect_one_episode(
    env: Any,
    model: GRUActorCritic,
    transition: np.ndarray,
    prior_active: np.ndarray,
    feature_mode: str,
    dwell_slots: int,
    cfg: dict[str, Any],
    action_budget: int,
    receiver_seed: int,
) -> tuple[Episode, int]:
    state = CausalSchedulerState.create(transition, prior_active)
    state.reset(prior_active)
    env.reset(seed=int(receiver_seed))

    hidden = model.initial_hidden(1)
    obs = compose_observation(state, feature_mode)

    obs_list: list[np.ndarray] = []
    action_list: list[int] = []
    logp_list: list[float] = []
    value_list: list[float] = []
    reward_list: list[float] = []
    dwell_list: list[int] = []
    hidden_list: list[np.ndarray] = []

    terminal = False
    steps_taken = 0

    while steps_taken < int(action_budget) and not env.done:
        hidden_list.append(hidden_to_numpy(hidden))
        x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, value, next_hidden = model.forward_step(x, hidden)
            dist = Categorical(logits=logits)
            action_t = dist.sample()
            action = int(action_t.item())
            logp = float(dist.log_prob(action_t).item())
            val = float(value.item())
        hidden = next_hidden.detach()

        looks, _env_reward, done = env.step_dwell_training(action, int(dwell_slots), reward_mode=REWARD_MODE)
        looks = list(looks)
        if len(looks) != int(dwell_slots) and not done:
            raise RuntimeError(f"Expected {dwell_slots} looks, got {len(looks)}")
        if len(looks) == 0:
            raise RuntimeError("Environment returned zero observations for a dwell")

        positives = detector_sequence(looks)
        prebeliefs = state.belief.step_observations(action, positives)
        # The frozen reward is detector-positive. Causal belief is updated for
        # the next action but never fed into this reward through hidden truth.
        reward = float(np.sum(np.asarray(positives, dtype=np.float64)))

        obs_list.append(obs.copy())
        action_list.append(action)
        logp_list.append(logp)
        value_list.append(val)
        reward_list.append(reward)
        dwell_list.append(int(len(looks)))

        state.step(action, int(len(looks)), positives)
        steps_taken += 1

        terminal = bool(done)
        if terminal:
            break
        obs = compose_observation(state, feature_mode)

    if terminal:
        bootstrap = 0.0
    else:
        # Rollout truncation, not environment termination. Preserve the true
        # post-action recurrent state and next observation for bootstrap.
        next_obs = compose_observation(state, feature_mode)
        x_next = torch.as_tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            _logits, next_value, _ = model.forward_step(x_next, hidden)
        bootstrap = float(next_value.item())

    rewards = np.asarray(reward_list, dtype=np.float64)
    values = np.asarray(value_list, dtype=np.float64)
    dwells = np.asarray(dwell_list, dtype=np.int64)
    advantages, returns = compute_gae(
        rewards, values, dwells,
        float(cfg["gamma"]), float(cfg["gae_lambda"]), bootstrap,
    )

    return Episode(
        obs=np.asarray(obs_list, dtype=np.float32),
        actions=np.asarray(action_list, dtype=np.int64),
        old_logp=np.asarray(logp_list, dtype=np.float32),
        old_values=np.asarray(value_list, dtype=np.float32),
        rewards=rewards.astype(np.float32),
        dwell_slots=dwells,
        advantages=advantages.astype(np.float32),
        returns=returns.astype(np.float32),
        rnn_h=np.asarray(hidden_list, dtype=np.float32),
        terminal=bool(terminal),
        bootstrap_value=float(bootstrap),
    ), steps_taken


class PPOTrainer:
    def __init__(self, model: GRUActorCritic, cfg: dict[str, Any], seed: int):
        self.model = model
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(cfg["lr"]))
        self.update_count = 0
        self.rng = np.random.default_rng(int(seed) + 910_003)

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        if not episodes:
            raise RuntimeError("Empty PPO rollout")

        all_adv = np.concatenate([ep.advantages for ep in episodes]).astype(np.float64)
        mean_adv = float(all_adv.mean())
        std_adv = float(all_adv.std() + 1e-8)
        for ep in episodes:
            ep.advantages = ((ep.advantages - mean_adv) / std_adv).astype(np.float32)

        chunks: list[tuple[Episode, int, int]] = []
        bptt = int(self.cfg["bptt"])
        for ep in episodes:
            for start in range(0, len(ep.actions), bptt):
                chunks.append((ep, start, min(len(ep.actions), start + bptt)))

        stats: dict[str, list[float]] = {k: [] for k in ("loss", "kl", "clip", "entropy", "value_loss")}
        seq_mb = int(self.cfg["sequence_minibatch"])

        for _epoch in range(PPO_OPTIMIZER_EPOCHS):
            order = self.rng.permutation(len(chunks))
            epoch_kls: list[float] = []

            for mb_start in range(0, len(order), seq_mb):
                batch = [chunks[int(i)] for i in order[mb_start: mb_start + seq_mb]]
                B = len(batch)
                max_len = max(end - start for _, start, end in batch)

                obs_t = torch.zeros(B, max_len, self.model.input_dim, device=DEVICE)
                act_t = torch.zeros(B, max_len, dtype=torch.long, device=DEVICE)
                old_lp_t = torch.zeros(B, max_len, device=DEVICE)
                adv_t = torch.zeros(B, max_len, device=DEVICE)
                ret_t = torch.zeros(B, max_len, device=DEVICE)
                mask_t = torch.zeros(B, max_len, device=DEVICE)
                h0 = torch.zeros(1, B, self.model.hidden_size, device=DEVICE)

                for i, (ep, start, end) in enumerate(batch):
                    n = end - start
                    obs_t[i, :n] = torch.as_tensor(ep.obs[start:end], dtype=torch.float32, device=DEVICE)
                    act_t[i, :n] = torch.as_tensor(ep.actions[start:end], dtype=torch.long, device=DEVICE)
                    old_lp_t[i, :n] = torch.as_tensor(ep.old_logp[start:end], dtype=torch.float32, device=DEVICE)
                    adv_t[i, :n] = torch.as_tensor(ep.advantages[start:end], dtype=torch.float32, device=DEVICE)
                    ret_t[i, :n] = torch.as_tensor(ep.returns[start:end], dtype=torch.float32, device=DEVICE)
                    mask_t[i, :n] = 1.0
                    h0[:, i, :] = torch.as_tensor(ep.rnn_h[start], dtype=torch.float32, device=DEVICE)

                logits, values, _ = self.model.forward_sequence(obs_t, h0)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act_t)
                entropy = dist.entropy()

                ratio = torch.exp(logp - old_lp_t)
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(self.cfg["clip_eps"]),
                    1.0 + float(self.cfg["clip_eps"]),
                )
                denom = mask_t.sum().clamp_min(1.0)
                policy_loss = -(torch.minimum(ratio * adv_t, clipped * adv_t) * mask_t).sum() / denom
                value_loss = (((values - ret_t) ** 2) * mask_t).sum() / denom
                entropy_mean = (entropy * mask_t).sum() / denom
                loss = (
                    policy_loss
                    + float(self.cfg["value_coef"]) * value_loss
                    - float(self.cfg["entropy_coef"]) * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg["max_grad_norm"]))
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((old_lp_t - logp) * mask_t).sum() / denom
                    clip_frac = (((ratio - 1.0).abs() > float(self.cfg["clip_eps"])).float() * mask_t).sum() / denom

                vals = {
                    "loss": float(loss.item()),
                    "kl": float(approx_kl.item()),
                    "clip": float(clip_frac.item()),
                    "entropy": float(entropy_mean.item()),
                    "value_loss": float(value_loss.item()),
                }
                for k, v in vals.items():
                    stats[k].append(v)
                epoch_kls.append(vals["kl"])

            if epoch_kls and float(np.mean(epoch_kls)) > float(self.cfg["target_kl"]):
                break

        self.update_count += 1
        return {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}


def load_prior(prior_dir: Path, fingerprint: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(prior_dir) / "pormab_transition.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing PO-RMAB prior: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_pool_fingerprint") != fingerprint:
        raise RuntimeError("PO-RMAB prior fingerprint does not match the saved TRAIN-500 cache")
    if abs(float(payload.get("model_pd", -1.0)) - PD) > 1e-12 or abs(float(payload.get("model_pfa", -1.0)) - PFA) > 1e-12:
        raise RuntimeError("PO-RMAB prior observation model does not match Pd=0.90/Pfa=0.05")
    transition = np.asarray(payload["transition"], dtype=np.float64)
    prior_active = np.asarray(payload["prior_active"], dtype=np.float64)
    if transition.shape != (N_BANDS, 2, 2) or prior_active.shape != (N_BANDS,):
        raise RuntimeError("Malformed PO-RMAB prior shapes")
    return transition, prior_active


# The user's causal harness and simulator expose the exact replay API used here.
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


def metric_value(row: dict[str, Any], aliases: tuple[str, ...]) -> float:
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
    oir = [metric_value(r, ("opportunity_interception_ratio", "oir", "mean_oir")) for r in rows]
    eir = [metric_value(r, ("emitter_interception_ratio", "mean_emitter_interception_ratio")) for r in rows]
    ttfi = [metric_value(r, ("censored_ttfi_s", "censored_ttf_s", "median_censored_ttfi_s", "censored_ttf_ms")) for r in rows]
    oir = [x for x in oir if np.isfinite(x)]
    eir = [x for x in eir if np.isfinite(x)]
    ttfi = [x / 1000.0 if x > 100.0 else x for x in ttfi if np.isfinite(x)]
    return {
        "opportunity_interception_ratio": float(np.mean(oir)) if oir else float("nan"),
        "emitter_interception_ratio": float(np.mean(eir)) if eir else float("nan"),
        "median_censored_ttfi_s": float(np.median(ttfi)) if ttfi else float("nan"),
        "conditional_pd": float(np.nanmean([metric_value(r, ("conditional_pd", "pd", "mean_pd")) for r in rows])),
        "true_pfa": float(np.nanmean([metric_value(r, ("true_pfa", "pfa", "mean_pfa")) for r in rows])),
        "n_episodes": float(len(rows)),
    }


def evaluate_model(model: GRUActorCritic, corpus_root: Path, registry: list[dict[str, Any]], transition: np.ndarray, prior_active: np.ndarray, feature_mode: str, dwell_slots: int, seed: int) -> dict[str, Any]:
    from vyapti_simulator.tsrd.benchmark_protocol import score_recorded_replay, _assert_scorecard_consistent, _seed_for_file
    from vyapti_simulator.core.metrics import TrajectoryStep

    model.eval()
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    bar = tqdm(registry, desc="VAL PPO-GRU", unit="scene")
    try:
        for recipe in bar:
            stare_file = Path(recipe["source_file"])
            env = build_replay_environment(stare_file)
            receiver_seed = _seed_for_file(stare_file.stem, seed)
            env.reset(seed=int(receiver_seed))
            state = CausalSchedulerState.create(transition, prior_active)
            state.reset(prior_active)
            hidden = model.initial_hidden(1)
            obs = compose_observation(state, feature_mode)
            trajectory: list[TrajectoryStep] = []
            actions = 0

            while not env.done:
                x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    logits, _value, next_hidden = model.forward_step(x, hidden)
                    action = int(torch.argmax(logits, dim=-1).item())
                hidden = next_hidden.detach()
                looks, _r, done = env.step_dwell_training(action, int(dwell_slots), reward_mode=REWARD_MODE)
                looks = list(looks)
                if len(looks) != dwell_slots and not done:
                    raise RuntimeError(f"Expected {dwell_slots} looks, got {len(looks)}")
                positives = detector_sequence(looks)
                for look in looks:
                    trajectory.append(TrajectoryStep(action=action, time_slot=int(look["time_slot"]), observation=look))
                state.belief.step_observations(action, positives)
                state.step(action, len(looks), positives)
                actions += 1
                if not done:
                    obs = compose_observation(state, feature_mode)

            score = score_recorded_replay(env, trajectory)
            _assert_scorecard_consistent(score, trajectory)
            rows.append({"world_id": int(recipe["world_id"]), "config": stare_file.stem, "source_file": str(stare_file), "ppo_action_steps": actions, **score})
            elapsed = time.perf_counter() - t0
            rate = len(rows) / max(elapsed, 1e-9)
            eta = (len(registry) - len(rows)) / max(rate, 1e-9)
            bar.set_postfix_str(f"elapsed={fmt_seconds(elapsed)} ETA={fmt_seconds(eta)}")
    finally:
        model.train()
    return {"summary": aggregate_metrics(rows), "rows": rows, "runtime_s": time.perf_counter() - t0}


def checkpoint_better(current: dict[str, float], best: dict[str, float] | None) -> bool:
    if best is None:
        return True
    criteria = (
        ("opportunity_interception_ratio", True),
        ("censored_median_ttfi_s", False),
        ("pooled_unique_emitter_interception_rate", True),
    )
    for key, higher in criteria:
        val_a = current.get(key)
        val_b = best.get(key)
        a = float(val_a) if val_a is not None else float("nan")
        b = float(val_b) if val_b is not None else float("nan")
        if not np.isfinite(a):
            return False
        if not np.isfinite(b):
            return True
        if not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12):
            return a > b if higher else a < b
    return False


def save_checkpoint(path: Path, model: GRUActorCritic, trainer: PPOTrainer, manifest: dict[str, Any], total_action_steps: int, milestone: int) -> None:
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


def run_training(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    corpus_root = Path(args.corpus_root)
    cache_root = Path(args.cache_root)
    prior_dir = Path(args.prior_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_mode not in FEATURE_CHOICES:
        raise ValueError(f"feature_mode must be one of {FEATURE_CHOICES}")

    counts = validate_dataset(corpus_root)
    cache = load_train500_cache(cache_root)
    pool, cache = build_train_pool_from_cache(corpus_root, cache_root)
    write_runtime_manifest(cache)
    transition, prior_active = load_prior(prior_dir, cache["fingerprint"])

    state0 = CausalSchedulerState.create(transition, prior_active)
    input_dim = len(compose_observation(state0, args.feature_mode))
    model = GRUActorCritic(input_dim=input_dim, hidden=args.hidden).to(DEVICE)
    cfg = {
        "lr": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_eps": args.clip_eps,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
        "bptt": args.bptt,
        "sequence_minibatch": args.sequence_minibatch,
    }
    trainer = PPOTrainer(model, cfg, args.seed)
    world_rng = np.random.default_rng(int(args.seed) + 1_003_003)

    val_registry = build_replay_registry(corpus_root, "val", VAL_FILES)
    save_json(output_dir / "val_replay_registry.json", val_registry)

    target_steps = PPO_TRAINING_STEPS if args.max_action_steps is None else min(PPO_TRAINING_STEPS, int(args.max_action_steps))
    checkpoints = [x for x in CHECKPOINT_STEPS if x <= target_steps]
    manifest = {
        "algorithm": "PPO-GRU",
        "protocol": "500-config saved cache; online fresh 30-second world per episode; 512-action rollouts; 5 PPO optimizer epochs",
        "available_train_configs": 2500,
        "selected_train_configs": 500,
        "source_pool_fingerprint": cache["fingerprint"],
        "world_generation": "online; no world-generation epochs; no fixed training-world registry",
        "rollout_action_target": PPO_ROLLOUT_ACTIONS,
        "optimizer_epochs": PPO_OPTIMIZER_EPOCHS,
        "training_action_target": target_steps,
        "checkpoint_milestones": checkpoints,
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
        "dwell_slots": args.dwell_slots,
        "dwell_ms": 50 * args.dwell_slots,
        "reward": REWARD_MODE,
        "feature_mode": args.feature_mode,
        "input_dim": input_dim,
        "hidden": args.hidden,
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "hyperparameters": cfg,
        "dataset_counts": counts,
        "recurrent_correctness": "store exact hidden-before-action state; BPTT chunks initialize from stored rollout hidden; no cross-episode hidden carry",
    }
    save_json(output_dir / "experiment_manifest.json", manifest)

    best_summary: dict[str, float] | None = None
    total_steps = 0
    checkpoint_index = 0
    t0 = time.perf_counter()
    progress = tqdm(total=target_steps, desc="PPO-GRU 500pool", unit="action")
    last_update_stats: dict[str, float] = {}

    while total_steps < target_steps:
        remaining = target_steps - total_steps
        rollout_target = min(PPO_ROLLOUT_ACTIONS, remaining)
        episodes: list[Episode] = []
        rollout_steps = 0

        while rollout_steps < rollout_target:
            # Every new episode is a fresh online world from the same saved 500-config pool.
            env, _sources, world_seed, emitter_count = sample_fresh_training_world(pool, world_rng)
            episode_budget = rollout_target - rollout_steps
            ep, used = collect_one_episode(
                env,
                model,
                transition,
                prior_active,
                args.feature_mode,
                args.dwell_slots,
                cfg,
                episode_budget,
                receiver_seed=world_seed,
            )
            if used <= 0:
                raise RuntimeError("Training episode produced zero action steps")
            episodes.append(ep)
            rollout_steps += used

        # Exactly this rollout's transitions are reused for 5 optimizer epochs.
        last_update_stats = trainer.update(episodes)
        total_steps += rollout_steps
        elapsed = time.perf_counter() - t0
        rate = total_steps / max(elapsed, 1e-9)
        eta = (target_steps - total_steps) / max(rate, 1e-9)
        progress.update(rollout_steps)
        progress.set_postfix_str(
            f"elapsed={fmt_seconds(elapsed)} rate={rate:.1f}/s ETA={fmt_seconds(eta)} KL={last_update_stats.get('kl', float('nan')):.4f}"
        )

        while checkpoint_index < len(checkpoints) and total_steps >= checkpoints[checkpoint_index]:
            milestone = int(checkpoints[checkpoint_index])
            ckpt = output_dir / f"checkpoint_{milestone//1000:04d}k.pt"
            save_checkpoint(ckpt, model, trainer, manifest, total_steps, milestone)

            val = evaluate_model(
                model,
                corpus_root,
                val_registry,
                transition,
                prior_active,
                args.feature_mode,
                args.dwell_slots,
                VAL_REPLAY_SEED,
            )
            save_json(output_dir / f"validation_{milestone//1000:04d}k.json", val)
            current = val["summary"]
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
                    output_dir / "best_validation.pt",
                )
                save_json(output_dir / "best_validation_summary.json", {
                    "best_milestone": milestone,
                    "actual_action_steps": total_steps,
                    "summary": best_summary,
                })
            checkpoint_index += 1

    progress.close()

    # Final TEST exactly once after best validation checkpoint selection.
    if not args.skip_test:
        best_path = output_dir / "best_validation.pt"
        if not best_path.exists():
            raise RuntimeError("No best_validation.pt exists; TEST cannot be run")
        best_blob = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(best_blob["model"])
        test_registry = build_replay_registry(corpus_root, "test", TEST_FILES)
        test = evaluate_model(
            model,
            corpus_root,
            test_registry,
            transition,
            prior_active,
            args.feature_mode,
            args.dwell_slots,
            TEST_REPLAY_SEED,
        )
        save_json(output_dir / "TEST_final.json", test)

    run_summary = {
        "algorithm": "PPO-GRU",
        "total_action_steps": total_steps,
        "target_action_steps": target_steps,
        "rollout_actions": PPO_ROLLOUT_ACTIONS,
        "optimizer_epochs": PPO_OPTIMIZER_EPOCHS,
        "best_validation": best_summary,
        "test": "completed once" if not args.skip_test else "skipped",
        "runtime_s": time.perf_counter() - t0,
        "runtime": fmt_seconds(time.perf_counter() - t0),
    }
    save_json(output_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2, sort_keys=True, allow_nan=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--cache-root", required=True, help="Existing /runs/tsrd_cache/train_500 directory")
    ap.add_argument("--prior-dir", required=True, help="PO-RMAB output/prior directory")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--feature-mode", choices=FEATURE_CHOICES, default="belief_periodic")
    ap.add_argument("--dwell-slots", type=int, choices=(1, 2), default=DEFAULT_DWELL_SLOTS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--gae-lambda", type=float, default=DEFAULT_GAE_LAMBDA)
    ap.add_argument("--clip-eps", type=float, default=DEFAULT_CLIP_EPS)
    ap.add_argument("--entropy-coef", type=float, default=DEFAULT_ENTROPY_COEF)
    ap.add_argument("--value-coef", type=float, default=DEFAULT_VALUE_COEF)
    ap.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    ap.add_argument("--target-kl", type=float, default=DEFAULT_TARGET_KL)
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--bptt", type=int, default=DEFAULT_BPTT)
    ap.add_argument("--sequence-minibatch", type=int, default=DEFAULT_SEQUENCE_MINIBATCH)
    ap.add_argument("--max-action-steps", type=int, default=None)
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

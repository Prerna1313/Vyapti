#!/usr/bin/env python3
"""
Shared training utilities for Vyapti's TSRD Stare SAC experiments.

This module is deliberately receiver-causal:
- the scheduler sees only belief/state derived from past selected-band observations;
- emitter labels are used only to construct an internal occupancy grid and are never
  exposed to the policy;
- the hidden occupancy truth is used by the simulator/evaluator, not as an observation.

TSRD timing used here:
- 0-18 GHz spectrum
- 36 bands of 500 MHz
- TSRD v2 Stare collection interval is 30 s
- 1800 decision slots => 16.6666667 ms per base slot
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Dataset / mission configuration
# -----------------------------------------------------------------------------

N_BANDS = 36
BAND_WIDTH_MHZ = 500.0
BAND_EDGES_MHZ = np.arange(N_BANDS + 1, dtype=np.float64) * BAND_WIDTH_MHZ
FREQ_MAX_MHZ = N_BANDS * BAND_WIDTH_MHZ

MISSION_US = 30_000_000.0          # TSRD v2 Stare: 30 s
N_BASE_SLOTS = 1800
BASE_SLOT_US = MISSION_US / N_BASE_SLOTS
BASE_SLOT_MS = BASE_SLOT_US / 1000.0

PD_SENSOR = 0.90
PFA_SENSOR = 0.05

FEATS_PER_BAND = 4
HIDDEN = 128

DEFAULT_DWELL_MULTIPLIERS = (1, 2, 3, 6)
DEFAULT_DWELL_TIME_PENALTY = 0.02


# -----------------------------------------------------------------------------
# Reproducibility / filesystem helpers
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_string(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(name)


def find_split_files(data_root: str | Path, split: str) -> List[Path]:
    """Find TSRD Stare H5 files for train/val/test."""
    root = Path(data_root)
    candidates = [
        root / "stare" / f"{split}_stare",
        root / "stare" / split,
        root / split / "stare",
    ]
    files: List[Path] = []
    for d in candidates:
        if d.exists():
            files.extend(sorted(d.glob("*.h5")))
            files.extend(sorted(d.glob("*.hdf5")))
    # de-duplicate while preserving order
    return list(dict.fromkeys(files))


def require_split_files(
    data_root: str | Path,
    split: str,
    expected: Optional[int] = None,
) -> List[Path]:
    files = find_split_files(data_root, split)
    if not files:
        raise FileNotFoundError(
            f"No TSRD Stare files found for split={split!r} under {Path(data_root).resolve()}. "
            f"Expected something like data/stare/{split}_stare/*.h5"
        )
    if expected is not None and len(files) != expected:
        print(
            f"[warning] expected about {expected} {split} files, found {len(files)}. "
            "Continuing with the files actually present."
        )
    return files


# -----------------------------------------------------------------------------
# TSRD -> hidden time-frequency occupancy grid
# -----------------------------------------------------------------------------


def load_truth_grid(
    h5_path: str | Path,
    n_bands: int = N_BANDS,
    n_slots: int = N_BASE_SLOTS,
    mission_us: float = MISSION_US,
    band_width_mhz: float = BAND_WIDTH_MHZ,
) -> np.ndarray:
    """
    Convert one TSRD Stare PDW stream into hidden occupancy truth [band, slot].

    Only ToA and centre frequency are needed. Emitter labels are intentionally not
    read here because the scheduler does not need emitter attribution.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        data = f["data"]
        toa = np.asarray(data[:, 0], dtype=np.float64)
        freq = np.asarray(data[:, 1], dtype=np.float64)

    valid = (
        np.isfinite(toa)
        & np.isfinite(freq)
        & (toa >= 0.0)
        & (toa < mission_us)
        & (freq >= 0.0)
        & (freq < n_bands * band_width_mhz)
    )

    toa = toa[valid]
    freq = freq[valid]

    truth = np.zeros((n_bands, n_slots), dtype=np.bool_)
    if toa.size == 0:
        return truth

    slot_idx = np.floor(toa / (mission_us / n_slots)).astype(np.int32)
    band_idx = np.floor(freq / band_width_mhz).astype(np.int32)

    valid_idx = (
        (slot_idx >= 0)
        & (slot_idx < n_slots)
        & (band_idx >= 0)
        & (band_idx < n_bands)
    )

    truth[band_idx[valid_idx], slot_idx[valid_idx]] = True
    return truth


# -----------------------------------------------------------------------------
# Receiver / belief environment
# -----------------------------------------------------------------------------


class BaseTSRDEnv:
    """Common causal belief-state receiver environment."""

    def __init__(
        self,
        truth: np.ndarray,
        pd_sensor: float = PD_SENSOR,
        pfa_sensor: float = PFA_SENSOR,
        seed: int = 0,
    ) -> None:
        truth = np.asarray(truth, dtype=np.bool_)
        if truth.ndim != 2:
            raise ValueError("truth must have shape [n_bands, n_slots]")
        self.truth = truth
        self.n_bands, self.n_slots = truth.shape
        self.pd_sensor = float(pd_sensor)
        self.pfa_sensor = float(pfa_sensor)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> Dict[str, np.ndarray | float]:
        self.t = 0
        self.belief = np.full(self.n_bands, 0.5, dtype=np.float32)
        self.last_visit = np.full(self.n_bands, -1, dtype=np.int32)
        self.hit_count = np.zeros(self.n_bands, dtype=np.int32)
        self.visit_count = np.zeros(self.n_bands, dtype=np.int32)
        # Binary latent-state transition statistics used by the existing hybrid belief model.
        self.trans_counts = np.ones((self.n_bands, 2, 2), dtype=np.float64)
        return self._obs()

    def _obs(self) -> Dict[str, np.ndarray | float]:
        staleness = np.clip(self.t - self.last_visit, 0, None).astype(np.float32)
        staleness /= max(self.n_slots, 1)
        return {
            "belief": self.belief.copy(),
            "staleness": staleness,
            "t_fraction": float(self.t / max(self.n_slots, 1)),
        }

    def transition_probs(self, band: int) -> Tuple[float, float]:
        c = self.trans_counts[band]
        p01 = float(c[0, 1] / max(c[0].sum(), 1e-12))
        p11 = float(c[1, 1] / max(c[1].sum(), 1e-12))
        return p01, p11

    def _bayes_update(self, band: int, detect: bool) -> None:
        prior = float(self.belief[band])
        like_on = self.pd_sensor if detect else (1.0 - self.pd_sensor)
        like_off = self.pfa_sensor if detect else (1.0 - self.pfa_sensor)
        denom = like_on * prior + like_off * (1.0 - prior) + 1e-12
        post = (like_on * prior) / denom
        post = float(np.clip(post, 1e-6, 1.0 - 1e-6))

        old_state = int(prior > 0.5)
        new_state = int(post > 0.5)
        self.trans_counts[band, old_state, new_state] += 1.0
        self.belief[band] = post

        # Predict one step for unselected bands as in the existing notebook architecture.
        for b in range(self.n_bands):
            p01, p11 = self.transition_probs(b)
            bel = float(self.belief[b])
            self.belief[b] = np.float32(bel * p11 + (1.0 - bel) * p01)

    def _record_visit(self, band: int, detect: bool) -> None:
        self.visit_count[band] += 1
        self.hit_count[band] += int(detect)
        self.last_visit[band] = max(self.t - 1, 0)


class FixedDwellTSRDEnv(BaseTSRDEnv):
    """
    Fixed-dwell environment.

    One scheduler decision consumes exactly one 16.6667 ms base slot.
    """

    def step(self, band: int):
        band = int(band)
        if not (0 <= band < self.n_bands):
            raise ValueError(f"band must be in [0, {self.n_bands - 1}]")
        if self.t >= self.n_slots:
            return self._obs(), 0.0, True, {"done": True}

        slot = self.t
        truth_active = bool(self.truth[band, slot])

        if truth_active:
            detect = bool(self.rng.random() < self.pd_sensor)
        else:
            detect = bool(self.rng.random() < self.pfa_sensor)

        true_detection = bool(truth_active and detect)
        false_alarm = bool((not truth_active) and detect)

        # Observation-based reward, matching the existing notebook's clean baseline.
        reward = 1.0 if detect else 0.0

        self._bayes_update(band, detect)
        self._record_visit(band, detect)
        self.t += 1
        done = self.t >= self.n_slots

        info = {
            "truth_active": truth_active,
            "detect": detect,
            "true_detection": true_detection,
            "false_alarm": false_alarm,
            "active_opportunities": int(truth_active),
            "empty_opportunities": int(not truth_active),
            "true_hits": int(true_detection),
            "false_alarms": int(false_alarm),
            "decision_time_slot": int(self.t),
            "decision_time_ms": float(self.t * BASE_SLOT_MS),
            "detect_time_slot": int(slot) if detect else None,
            "actual_dwell_slots": 1,
        }
        return self._obs(), float(reward), done, info


class DynamicDwellTSRDEnv(BaseTSRDEnv):
    """
    Dynamic-dwell environment.

    Each action chooses a (band, dwell_multiplier). The requested dwell duration is
    1/2/3/6 base slots = 16.67/33.33/50/100 ms. The clock advances by the selected
    duration (clipped at mission end), so longer dwell is never free.

    During a dwell, the detector is sampled for every base slot in the observation
    window. The scheduler receives only the aggregate detection result at the end of
    the selected dwell. This keeps the action causal without inventing intermediate
    decisions that the policy did not request.
    """

    def __init__(
        self,
        truth: np.ndarray,
        dwell_multipliers: Sequence[int] = DEFAULT_DWELL_MULTIPLIERS,
        pd_sensor: float = PD_SENSOR,
        pfa_sensor: float = PFA_SENSOR,
        time_penalty: float = DEFAULT_DWELL_TIME_PENALTY,
        seed: int = 0,
    ) -> None:
        self.dwell_multipliers = tuple(int(x) for x in dwell_multipliers)
        if not self.dwell_multipliers or any(x <= 0 for x in self.dwell_multipliers):
            raise ValueError("dwell_multipliers must be positive integers")
        self.time_penalty = float(time_penalty)
        super().__init__(truth, pd_sensor=pd_sensor, pfa_sensor=pfa_sensor, seed=seed)

    @property
    def n_actions(self) -> int:
        return self.n_bands * len(self.dwell_multipliers)

    def decode_action(self, action: int) -> Tuple[int, int]:
        action = int(action)
        if not (0 <= action < self.n_actions):
            raise ValueError(f"action must be in [0, {self.n_actions - 1}]")
        band = action // len(self.dwell_multipliers)
        dwell_idx = action % len(self.dwell_multipliers)
        return band, self.dwell_multipliers[dwell_idx]

    def encode_action(self, band: int, dwell_multiplier: int) -> int:
        try:
            dwell_idx = self.dwell_multipliers.index(int(dwell_multiplier))
        except ValueError as exc:
            raise ValueError(f"Unknown dwell multiplier {dwell_multiplier}") from exc
        return int(band) * len(self.dwell_multipliers) + dwell_idx

    def step(self, action: int):
        band, dwell_multiplier = self.decode_action(action)
        if self.t >= self.n_slots:
            return self._obs(), 0.0, True, {"done": True}

        start_t = self.t
        actual_dwell_slots = min(dwell_multiplier, self.n_slots - start_t)
        end_t = start_t + actual_dwell_slots

        active_count = 0
        empty_count = 0
        true_hits = 0
        false_alarms = 0
        first_detect_slot: Optional[int] = None

        # Detector model is applied independently per base slot inside the dwell window.
        for slot in range(start_t, end_t):
            truth_active = bool(self.truth[band, slot])
            if truth_active:
                active_count += 1
                detect = bool(self.rng.random() < self.pd_sensor)
                if detect:
                    true_hits += 1
                    if first_detect_slot is None:
                        first_detect_slot = slot
            else:
                empty_count += 1
                detect = bool(self.rng.random() < self.pfa_sensor)
                if detect:
                    false_alarms += 1
                    if first_detect_slot is None:
                        first_detect_slot = slot

        aggregate_detect = (true_hits + false_alarms) > 0
        true_detection = true_hits > 0
        false_alarm_only = (false_alarms > 0 and true_hits == 0)

        # The reward is intentionally observation-based; hidden truth remains evaluator-side.
        # A small time charge makes the dynamic-dwell experiment optimize speed as well as hits.
        reward = (1.0 if aggregate_detect else 0.0) - self.time_penalty * actual_dwell_slots

        self._bayes_update(band, aggregate_detect)
        self.visit_count[band] += 1
        self.hit_count[band] += int(aggregate_detect)
        self.last_visit[band] = end_t - 1
        self.t = end_t
        done = self.t >= self.n_slots

        info = {
            "truth_active": bool(active_count > 0),
            "detect": aggregate_detect,
            "true_detection": true_detection,
            "false_alarm": false_alarm_only,
            "active_opportunities": int(active_count),
            "empty_opportunities": int(empty_count),
            "true_hits": int(true_hits),
            "false_alarms": int(false_alarms),
            "decision_time_slot": int(self.t),
            "decision_time_ms": float(self.t * BASE_SLOT_MS),
            "detect_time_slot": int(first_detect_slot) if first_detect_slot is not None else None,
            "actual_dwell_slots": int(actual_dwell_slots),
            "requested_dwell_slots": int(dwell_multiplier),
            "band": int(band),
            "dwell_multiplier": int(dwell_multiplier),
        }
        return self._obs(), float(reward), done, info


# -----------------------------------------------------------------------------
# Causal state features
# -----------------------------------------------------------------------------


class PeriodicityTracker:
    """Past-observation-only periodicity estimate."""

    def __init__(self, n_bands: int, min_hits: int = 3, history: int = 6) -> None:
        self.n_bands = n_bands
        self.min_hits = min_hits
        self.history = history
        self.hit_times: List[List[int]] = [[] for _ in range(n_bands)]

    def reset(self) -> None:
        self.hit_times = [[] for _ in range(self.n_bands)]

    def record_hit(self, band: int, t_slot: int) -> None:
        self.hit_times[int(band)].append(int(t_slot))

    def estimate_period(self, band: int) -> Optional[float]:
        hits = self.hit_times[int(band)]
        if len(hits) < self.min_hits:
            return None
        diffs = np.diff(hits[-self.history :])
        if len(diffs) == 0:
            return None
        period = float(np.median(diffs))
        return period if period > 0 else None

    def predicted_next_hit(self, band: int, t_now: int) -> Optional[float]:
        hits = self.hit_times[int(band)]
        period = self.estimate_period(band)
        if period is None or not hits:
            return None
        last = hits[-1]
        steps = math.ceil((t_now - last) / period) if t_now > last else 1
        return float(last + max(steps, 1) * period)

    def bonus(self, band: int, t_now: int, window: float = 2.0) -> float:
        pred = self.predicted_next_hit(band, t_now)
        if pred is None:
            return 0.0
        return float(math.exp(-abs(pred - t_now) / max(window, 1e-6)))


def featurize(env: BaseTSRDEnv, tracker: PeriodicityTracker) -> np.ndarray:
    belief = np.asarray(env.belief, dtype=np.float32)
    staleness = np.clip(env.t - env.last_visit, 0, None).astype(np.float32)
    staleness /= max(env.n_slots, 1)
    hit_rate = env.hit_count.astype(np.float32) / np.maximum(env.visit_count, 1)
    periodicity = np.asarray(
        [tracker.bonus(b, env.t) for b in range(env.n_bands)], dtype=np.float32
    )
    features = np.stack([belief, staleness, hit_rate, periodicity], axis=1)
    return np.asarray(features, dtype=np.float32)


# -----------------------------------------------------------------------------
# SAC networks
# -----------------------------------------------------------------------------


class QNetSAC(nn.Module):
    def __init__(self, n_actions: int, state_dim: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualPolicyNetSAC(nn.Module):
    """
    Residual categorical policy.

    Fixed mode:
        logits = belief + 0.15*staleness + 0.6*periodicity + bounded correction.

    Dynamic mode:
        same band score plus a fixed dwell prior, then bounded correction over
        the Cartesian product (band, dwell multiplier).
    """

    def __init__(
        self,
        n_bands: int,
        state_dim: int,
        n_actions: int,
        dynamic_dwell: bool = False,
        dwell_multipliers: Sequence[int] = DEFAULT_DWELL_MULTIPLIERS,
        dwell_time_penalty: float = DEFAULT_DWELL_TIME_PENALTY,
        hidden: int = HIDDEN,
        max_correction: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_bands = int(n_bands)
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.dynamic_dwell = bool(dynamic_dwell)
        self.dwell_multipliers = tuple(int(x) for x in dwell_multipliers)
        self.max_correction = float(max_correction)

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.correction_head = nn.Linear(hidden, n_actions)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

        self.index_weights = nn.Parameter(
            torch.tensor([1.0, 0.15, 0.0, 0.6], dtype=torch.float32),
            requires_grad=False,
        )

        if dynamic_dwell:
            # Shorter dwell has a slight warm-start preference, but the learned correction
            # is free to reverse this when repeated evidence justifies longer dwell.
            dwell_bias = []
            for mult in self.dwell_multipliers:
                dwell_bias.append(-float(dwell_time_penalty) * max(mult - 1, 0))
            self.register_buffer("dwell_bias", torch.tensor(dwell_bias, dtype=torch.float32))
        else:
            self.register_buffer("dwell_bias", torch.zeros(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        x_bands = x.view(batch, self.n_bands, FEATS_PER_BAND)
        index_score = (x_bands * self.index_weights).sum(dim=-1)

        if self.dynamic_dwell:
            n_dwell = len(self.dwell_multipliers)
            base = index_score.unsqueeze(-1) + self.dwell_bias.view(1, 1, n_dwell)
            index_score_actions = base.reshape(batch, self.n_bands * n_dwell)
        else:
            index_score_actions = index_score

        h = self.trunk(x)
        correction = self.max_correction * torch.tanh(self.correction_head(h))
        logits = index_score_actions + correction
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs


# -----------------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done) -> None:
        i = self.ptr
        self.states[i] = np.asarray(state, dtype=np.float32).reshape(-1)
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_states[i] = np.asarray(next_state, dtype=np.float32).reshape(-1)
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "size": self.size,
            "ptr": self.ptr,
            "states": self.states[: self.size].copy(),
            "next_states": self.next_states[: self.size].copy(),
            "actions": self.actions[: self.size].copy(),
            "rewards": self.rewards[: self.size].copy(),
            "dones": self.dones[: self.size].copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        size = int(state["size"])
        self.size = size
        self.ptr = int(state["ptr"])
        self.states[:size] = state["states"]
        self.next_states[:size] = state["next_states"]
        self.actions[:size] = state["actions"]
        self.rewards[:size] = state["rewards"]
        self.dones[:size] = state["dones"]


# -----------------------------------------------------------------------------
# SAC helpers
# -----------------------------------------------------------------------------


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)


def make_policy_and_qs(
    *,
    dynamic_dwell: bool,
    device: torch.device,
    dwell_multipliers: Sequence[int],
    dwell_time_penalty: float,
):
    state_dim = N_BANDS * FEATS_PER_BAND
    n_actions = N_BANDS * len(dwell_multipliers) if dynamic_dwell else N_BANDS

    policy = ResidualPolicyNetSAC(
        N_BANDS,
        state_dim,
        n_actions,
        dynamic_dwell=dynamic_dwell,
        dwell_multipliers=dwell_multipliers,
        dwell_time_penalty=dwell_time_penalty,
    ).to(device)
    q1 = QNetSAC(n_actions, state_dim).to(device)
    q2 = QNetSAC(n_actions, state_dim).to(device)
    q1_target = QNetSAC(n_actions, state_dim).to(device)
    q2_target = QNetSAC(n_actions, state_dim).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    for p in q1_target.parameters():
        p.requires_grad_(False)
    for p in q2_target.parameters():
        p.requires_grad_(False)
    return policy, q1, q2, q1_target, q2_target


def sac_update(
    *,
    batch,
    policy: ResidualPolicyNetSAC,
    q1: QNetSAC,
    q2: QNetSAC,
    q1_target: QNetSAC,
    q2_target: QNetSAC,
    q_optimizer: torch.optim.Optimizer,
    pi_optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    gamma: float,
    tau: float,
    target_entropy: float,
    update_actor: bool,
    device: torch.device,
) -> Dict[str, float]:
    s, a, r, s2, d = batch
    s = s.to(device, non_blocking=True)
    a = a.to(device, non_blocking=True)
    r = r.to(device, non_blocking=True)
    s2 = s2.to(device, non_blocking=True)
    d = d.to(device, non_blocking=True)

    alpha = log_alpha.exp().detach()

    with torch.no_grad():
        next_probs, next_log_probs = policy(s2)
        min_q_next = torch.min(q1_target(s2), q2_target(s2))
        v_next = (next_probs * (min_q_next - alpha * next_log_probs)).sum(dim=1)
        target = r + gamma * (1.0 - d) * v_next

    q1_pred = q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
    q2_pred = q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
    q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

    q_optimizer.zero_grad(set_to_none=True)
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5.0)
    q_optimizer.step()

    soft_update(q1_target, q1, tau)
    soft_update(q2_target, q2, tau)

    pi_loss_value = 0.0
    alpha_loss_value = 0.0

    if update_actor:
        probs, log_probs = policy(s)
        with torch.no_grad():
            min_q_s = torch.min(q1(s), q2(s))
        pi_loss = (probs * (alpha * log_probs - min_q_s)).sum(dim=1).mean()

        pi_optimizer.zero_grad(set_to_none=True)
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.trunk.parameters()) + list(policy.correction_head.parameters()), 5.0)
        pi_optimizer.step()

        alpha_loss = (
            probs.detach()
            * (-log_alpha.exp() * (log_probs.detach() + target_entropy))
        ).sum(dim=1).mean()

        alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_optimizer.step()
        with torch.no_grad():
            log_alpha.clamp_(-10.0, 2.0)

        pi_loss_value = float(pi_loss.detach().cpu())
        alpha_loss_value = float(alpha_loss.detach().cpu())

    return {
        "q_loss": float(q_loss.detach().cpu()),
        "pi_loss": pi_loss_value,
        "alpha_loss": alpha_loss_value,
        "alpha": float(log_alpha.exp().detach().cpu()),
    }


# -----------------------------------------------------------------------------
# Action helpers
# -----------------------------------------------------------------------------


def policy_action(
    policy: ResidualPolicyNetSAC,
    state: np.ndarray,
    device: torch.device,
    deterministic: bool,
    rng: Optional[np.random.Generator] = None,
) -> int:
    x = torch.from_numpy(state.reshape(1, -1)).float().to(device)
    with torch.no_grad():
        probs, _ = policy(x)
    if deterministic:
        return int(torch.argmax(probs, dim=1).item())
    p = probs[0].detach().cpu().numpy()
    if rng is None:
        return int(np.random.choice(len(p), p=p))
    return int(rng.choice(len(p), p=p))


def decode_dynamic_action(action: int, dwell_multipliers: Sequence[int]) -> Tuple[int, int]:
    n_dwell = len(dwell_multipliers)
    band = int(action) // n_dwell
    mult = int(dwell_multipliers[int(action) % n_dwell])
    return band, mult


# -----------------------------------------------------------------------------
# Checkpoint persistence
# -----------------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    global_step: int,
    policy: nn.Module,
    q1: nn.Module,
    q2: nn.Module,
    q1_target: nn.Module,
    q2_target: nn.Module,
    q_optimizer: torch.optim.Optimizer,
    pi_optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    replay: ReplayBuffer,
    config: dict,
    history: list,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "policy_state_dict": policy.state_dict(),
        "q1_state_dict": q1.state_dict(),
        "q2_state_dict": q2.state_dict(),
        "q1_target_state_dict": q1_target.state_dict(),
        "q2_target_state_dict": q2_target.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "pi_optimizer_state_dict": pi_optimizer.state_dict(),
        "alpha_optimizer_state_dict": alpha_optimizer.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
        "replay_state": replay.state_dict(),
        "config": config,
        "history": history,
    }
    torch.save(checkpoint, path)

    # Policy-only artifact for deployment/evaluation without optimizer/replay baggage.
    policy_path = path.with_name(path.stem + "_policy.pth")
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "config": config,
        },
        policy_path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    dynamic_dwell: bool,
    dwell_multipliers: Sequence[int],
    dwell_time_penalty: float,
):
    ckpt = torch.load(path, map_location=device)
    policy, q1, q2, q1_target, q2_target = make_policy_and_qs(
        dynamic_dwell=dynamic_dwell,
        device=device,
        dwell_multipliers=dwell_multipliers,
        dwell_time_penalty=dwell_time_penalty,
    )
    policy.load_state_dict(ckpt["policy_state_dict"])
    q1.load_state_dict(ckpt["q1_state_dict"])
    q2.load_state_dict(ckpt["q2_state_dict"])
    q1_target.load_state_dict(ckpt["q1_target_state_dict"])
    q2_target.load_state_dict(ckpt["q2_target_state_dict"])
    return ckpt, policy, q1, q2, q1_target, q2_target


# -----------------------------------------------------------------------------
# Validation / metrics
# -----------------------------------------------------------------------------


def evaluate_policy(
    policy: ResidualPolicyNetSAC,
    files: Sequence[Path],
    *,
    dynamic_dwell: bool,
    dwell_multipliers: Sequence[int],
    pd_sensor: float,
    pfa_sensor: float,
    seed: int,
    device: torch.device,
    max_files: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate only against hidden truth; no truth is placed in the policy state."""
    selected_files = list(files[:max_files]) if max_files is not None else list(files)
    rng = np.random.default_rng(seed)

    total_true_hits = 0
    total_false_alarms = 0
    total_active_opportunities = 0
    total_empty_opportunities = 0
    total_truth_opportunities = 0
    total_decisions = 0
    total_elapsed_slots = 0
    first_intercepts_ms: List[float] = []
    episode_hit_rates: List[float] = []
    episode_oir: List[float] = []
    dwell_slots_accum = 0

    policy.eval()

    for i, fp in enumerate(selected_files):
        truth = load_truth_grid(fp)
        env_seed = int(rng.integers(0, 2**31 - 1))
        if dynamic_dwell:
            env = DynamicDwellTSRDEnv(
                truth,
                dwell_multipliers=dwell_multipliers,
                pd_sensor=pd_sensor,
                pfa_sensor=pfa_sensor,
                seed=env_seed,
            )
        else:
            env = FixedDwellTSRDEnv(truth, pd_sensor=pd_sensor, pfa_sensor=pfa_sensor, seed=env_seed)

        tracker = PeriodicityTracker(N_BANDS)
        env.reset()
        done = False
        ep_true_hits = 0
        ep_false_alarms = 0
        ep_active = 0
        ep_empty = 0
        ep_decisions = 0
        ep_first_intercept_slot = None

        while not done:
            state = featurize(env, tracker)
            action = policy_action(policy, state, device, deterministic=True)
            _, _, done, info = env.step(action)

            band = int(action) if not dynamic_dwell else decode_dynamic_action(action, dwell_multipliers)[0]
            if info.get("detect_time_slot") is not None:
                tracker.record_hit(band, int(info["detect_time_slot"]))

            true_hits = int(info.get("true_hits", 0))
            false_alarms = int(info.get("false_alarms", 0))
            active_ops = int(info.get("active_opportunities", 0))
            empty_ops = int(info.get("empty_opportunities", 0))
            ep_true_hits += true_hits
            ep_false_alarms += false_alarms
            ep_active += active_ops
            ep_empty += empty_ops
            ep_decisions += 1
            total_true_hits += true_hits
            total_false_alarms += false_alarms
            total_active_opportunities += active_ops
            total_empty_opportunities += empty_ops
            total_decisions += 1
            dwell_slots_accum += int(info.get("actual_dwell_slots", 1))

            if info.get("true_detection", False) and ep_first_intercept_slot is None:
                ep_first_intercept_slot = int(info["detect_time_slot"])

        total_truth_opportunities += int(truth.sum())
        ep_oir = ep_true_hits / max(int(truth.sum()), 1)
        ep_hit = (ep_true_hits + ep_false_alarms) / max(1, ep_decisions)
        episode_hit_rates.append(float(ep_hit))
        episode_oir.append(float(ep_oir))
        if ep_first_intercept_slot is not None:
            first_intercepts_ms.append(float((ep_first_intercept_slot + 1) * BASE_SLOT_MS))

        total_elapsed_slots += int(env.t)

    conditional_pd = total_true_hits / max(total_active_opportunities, 1)
    true_pfa = total_false_alarms / max(total_empty_opportunities, 1)
    opportunity_interception_ratio = total_true_hits / max(total_truth_opportunities, 1)

    return {
        "files_evaluated": float(len(selected_files)),
        "conditional_pd": float(conditional_pd),
        "true_pfa": float(true_pfa),
        "opportunity_interception_ratio": float(opportunity_interception_ratio),
        "decision_hit_rate": float((total_true_hits + total_false_alarms) / max(total_decisions, 1)),
        "true_hits": float(total_true_hits),
        "false_alarms": float(total_false_alarms),
        "active_observed_opportunities": float(total_active_opportunities),
        "empty_observed_opportunities": float(total_empty_opportunities),
        "truth_active_opportunities": float(total_truth_opportunities),
        "mean_first_intercept_ms": float(np.mean(first_intercepts_ms)) if first_intercepts_ms else float("nan"),
        "median_first_intercept_ms": float(np.median(first_intercepts_ms)) if first_intercepts_ms else float("nan"),
        "mean_episode_hit_rate": float(np.mean(episode_hit_rates)) if episode_hit_rates else 0.0,
        "mean_episode_opportunity_interception_ratio": float(np.mean(episode_oir)) if episode_oir else 0.0,
        "mean_decision_dwell_slots": float(dwell_slots_accum / max(total_decisions, 1)),
        "mission_slots_consumed": float(total_elapsed_slots),
    }


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str
    seed: int = 1
    epochs: int = 1
    expected_train_files: int = 2500
    expected_val_files: int = 250
    replay_size: int = 20000
    batch_size: int = 128
    start_steps: int = 2000
    q_warmup_steps: int = 6000
    update_every: int = 4
    updates_per_update: int = 1
    gamma: float = 0.99
    lr: float = 3e-4
    tau: float = 0.005
    target_entropy_frac: float = 0.20
    dwell_time_penalty: float = DEFAULT_DWELL_TIME_PENALTY
    max_train_files: Optional[int] = None
    max_eval_files: Optional[int] = None
    checkpoint_every_episodes: int = 100
    device: str = "auto"
    resume: Optional[str] = None


class SACTrainer:
    def __init__(self, cfg: TrainConfig, dynamic_dwell: bool, dwell_multipliers: Sequence[int]):
        self.cfg = cfg
        self.dynamic_dwell = dynamic_dwell
        self.dwell_multipliers = tuple(int(x) for x in dwell_multipliers)
        self.device = device_from_string(cfg.device)
        seed_everything(cfg.seed)

        state_dim = N_BANDS * FEATS_PER_BAND
        self.n_actions = N_BANDS * len(self.dwell_multipliers) if dynamic_dwell else N_BANDS

        (
            self.policy,
            self.q1,
            self.q2,
            self.q1_target,
            self.q2_target,
        ) = make_policy_and_qs(
            dynamic_dwell=dynamic_dwell,
            device=self.device,
            dwell_multipliers=self.dwell_multipliers,
            dwell_time_penalty=cfg.dwell_time_penalty,
        )

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr
        )
        self.pi_optimizer = torch.optim.Adam(
            list(self.policy.trunk.parameters()) + list(self.policy.correction_head.parameters()),
            lr=cfg.lr,
        )
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        self.target_entropy = cfg.target_entropy_frac * math.log(self.n_actions)

        self.replay = ReplayBuffer(cfg.replay_size, state_dim, seed=cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)

        self.epoch = 0
        self.global_step = 0
        self.episode_counter = 0
        self.history: List[dict] = []

        if cfg.resume:
            self._resume(cfg.resume)

    def _config_dict(self) -> dict:
        return {
            "dynamic_dwell": self.dynamic_dwell,
            "dwell_multipliers": list(self.dwell_multipliers),
            "n_bands": N_BANDS,
            "n_base_slots": N_BASE_SLOTS,
            "mission_us": MISSION_US,
            "base_slot_us": BASE_SLOT_US,
            "base_slot_ms": BASE_SLOT_MS,
            "band_width_mhz": BAND_WIDTH_MHZ,
            "pd_sensor": PD_SENSOR,
            "pfa_sensor": PFA_SENSOR,
            "feats_per_band": FEATS_PER_BAND,
            "architecture": "Belief+Periodicity-guided Residual SAC (Discrete)",
            "data_mode": "TSRD Stare",
            "training_split": "stare/train_stare",
            "validation_split": "stare/val_stare",
            "dwell_time_penalty": self.cfg.dwell_time_penalty,
        }

    def _resume(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.q1.load_state_dict(ckpt["q1_state_dict"])
        self.q2.load_state_dict(ckpt["q2_state_dict"])
        self.q1_target.load_state_dict(ckpt["q1_target_state_dict"])
        self.q2_target.load_state_dict(ckpt["q2_target_state_dict"])
        self.q_optimizer.load_state_dict(ckpt["q_optimizer_state_dict"])
        self.pi_optimizer.load_state_dict(ckpt["pi_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer_state_dict"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        if "replay_state" in ckpt:
            self.replay.load_state_dict(ckpt["replay_state"])
        self.epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("global_step", 0))
        self.history = list(ckpt.get("history", []))
        print(f"[resume] loaded {path}; epoch={self.epoch}, global_step={self.global_step}")

    def _save(self, filename: str) -> None:
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            out / filename,
            epoch=self.epoch,
            global_step=self.global_step,
            policy=self.policy,
            q1=self.q1,
            q2=self.q2,
            q1_target=self.q1_target,
            q2_target=self.q2_target,
            q_optimizer=self.q_optimizer,
            pi_optimizer=self.pi_optimizer,
            alpha_optimizer=self.alpha_optimizer,
            log_alpha=self.log_alpha,
            replay=self.replay,
            config=self._config_dict(),
            history=self.history,
        )

    def train(self) -> None:
        train_files = require_split_files(
            self.cfg.data_root, "train", expected=self.cfg.expected_train_files
        )
        val_files = require_split_files(
            self.cfg.data_root, "val", expected=self.cfg.expected_val_files
        )
        if self.cfg.max_train_files is not None:
            train_files = train_files[: self.cfg.max_train_files]
        if self.cfg.max_eval_files is not None:
            val_files = val_files[: self.cfg.max_eval_files]

        print("=" * 88)
        print("VYAPTI — TSRD STARE SAC TRAINING")
        print("=" * 88)
        print(f"Mode                 : {'DYNAMIC DWELL' if self.dynamic_dwell else 'FIXED DWELL'}")
        print(f"Device               : {self.device}")
        print(f"Train files          : {len(train_files)}")
        print(f"Validation files     : {len(val_files)}")
        print(f"Bands                : {N_BANDS} × {BAND_WIDTH_MHZ:.0f} MHz")
        print(f"Mission              : {MISSION_US/1e6:.1f} s")
        print(f"Base slot            : {BASE_SLOT_MS:.6f} ms")
        print(f"Actions              : {self.n_actions}")
        if self.dynamic_dwell:
            print(
                "Dwell options        : "
                + ", ".join(
                    f"{m}x ({m * BASE_SLOT_MS:.2f} ms)" for m in self.dwell_multipliers
                )
            )
        print(f"Epochs               : {self.cfg.epochs}")
        print(f"Update every         : {self.cfg.update_every} env steps")
        print(f"Q warm-up            : {self.cfg.q_warmup_steps}")
        print(f"Start random steps   : {self.cfg.start_steps}")
        print("No Stare truth enters scheduler state; truth is used only inside the simulator/evaluator.")
        print("=" * 88)

        best_oir = -np.inf
        start_wall = time.time()

        # Continue from the next epoch when resuming.
        for epoch_idx in range(self.epoch, self.cfg.epochs):
            self.epoch = epoch_idx + 1
            epoch_start = time.time()
            order = list(train_files)
            self.rng.shuffle(order)

            epoch_returns: List[float] = []
            epoch_true_hits = 0
            epoch_false_alarms = 0
            epoch_active = 0
            epoch_empty = 0
            epoch_truth_opportunities = 0

            print(f"\n--- epoch {self.epoch}/{self.cfg.epochs} ---")

            for ep_i, fp in enumerate(order, start=1):
                truth = load_truth_grid(fp)
                env_seed = int(self.rng.integers(0, 2**31 - 1))
                if self.dynamic_dwell:
                    env = DynamicDwellTSRDEnv(
                        truth,
                        dwell_multipliers=self.dwell_multipliers,
                        pd_sensor=PD_SENSOR,
                        pfa_sensor=PFA_SENSOR,
                        time_penalty=self.cfg.dwell_time_penalty,
                        seed=env_seed,
                    )
                else:
                    env = FixedDwellTSRDEnv(
                        truth,
                        pd_sensor=PD_SENSOR,
                        pfa_sensor=PFA_SENSOR,
                        seed=env_seed,
                    )

                tracker = PeriodicityTracker(N_BANDS)
                env.reset()
                state = featurize(env, tracker)
                done = False
                ep_return = 0.0
                ep_true_hits = 0
                ep_false_alarms = 0
                ep_active = 0
                ep_empty = 0
                ep_truth_ops = int(truth.sum())

                while not done:
                    if self.global_step < self.cfg.start_steps:
                        action = int(self.rng.integers(0, self.n_actions))
                    else:
                        action = policy_action(
                            self.policy,
                            state,
                            self.device,
                            deterministic=False,
                            rng=self.rng,
                        )

                    _, reward, done, info = env.step(action)

                    # Past-observation-only tracker update.
                    if info.get("detect_time_slot") is not None:
                        band = int(action)
                        if self.dynamic_dwell:
                            band, _ = decode_dynamic_action(action, self.dwell_multipliers)
                        tracker.record_hit(band, int(info["detect_time_slot"]))

                    next_state = featurize(env, tracker)
                    self.replay.add(state, action, reward, next_state, done)
                    state = next_state

                    ep_return += float(reward)
                    ep_true_hits += int(info.get("true_hits", 0))
                    ep_false_alarms += int(info.get("false_alarms", 0))
                    ep_active += int(info.get("active_opportunities", 0))
                    ep_empty += int(info.get("empty_opportunities", 0))

                    self.global_step += 1

                    if (
                        self.global_step >= self.cfg.start_steps
                        and self.replay.size >= self.cfg.batch_size
                        and self.global_step % self.cfg.update_every == 0
                    ):
                        updates = []
                        for _ in range(self.cfg.updates_per_update):
                            metrics = sac_update(
                                batch=self.replay.sample(self.cfg.batch_size),
                                policy=self.policy,
                                q1=self.q1,
                                q2=self.q2,
                                q1_target=self.q1_target,
                                q2_target=self.q2_target,
                                q_optimizer=self.q_optimizer,
                                pi_optimizer=self.pi_optimizer,
                                alpha_optimizer=self.alpha_optimizer,
                                log_alpha=self.log_alpha,
                                gamma=self.cfg.gamma,
                                tau=self.cfg.tau,
                                target_entropy=self.target_entropy,
                                update_actor=self.global_step >= self.cfg.q_warmup_steps,
                                device=self.device,
                            )
                            updates.append(metrics)

                self.episode_counter += 1
                epoch_returns.append(ep_return)
                epoch_true_hits += ep_true_hits
                epoch_false_alarms += ep_false_alarms
                epoch_active += ep_active
                epoch_empty += ep_empty
                epoch_truth_opportunities += ep_truth_ops

                if self.episode_counter % self.cfg.checkpoint_every_episodes == 0:
                    self._save("last.pth")

                if ep_i % 100 == 0 or ep_i == len(order):
                    pd_now = epoch_true_hits / max(epoch_active, 1)
                    print(
                        f"  episode {ep_i:4d}/{len(order)} | "
                        f"env_steps={self.global_step:8d} | "
                        f"condPd={pd_now:.4f} | "
                        f"return={np.mean(epoch_returns[-100:]):.3f}",
                        flush=True,
                    )

                # Release the hidden world before loading the next one.
                del env
                del truth

            val_metrics = evaluate_policy(
                self.policy,
                val_files,
                dynamic_dwell=self.dynamic_dwell,
                dwell_multipliers=self.dwell_multipliers,
                pd_sensor=PD_SENSOR,
                pfa_sensor=PFA_SENSOR,
                seed=self.cfg.seed + 100000 + self.epoch,
                device=self.device,
                max_files=self.cfg.max_eval_files,
            )

            train_pd = epoch_true_hits / max(epoch_active, 1)
            train_pfa = epoch_false_alarms / max(epoch_empty, 1)
            train_oir = epoch_true_hits / max(epoch_truth_opportunities, 1)

            record = {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "train_conditional_pd": float(train_pd),
                "train_true_pfa": float(train_pfa),
                "train_opportunity_interception_ratio": float(train_oir),
                "mean_episode_return": float(np.mean(epoch_returns) if epoch_returns else 0.0),
                "validation": val_metrics,
                "epoch_seconds": float(time.time() - epoch_start),
            }
            self.history.append(record)

            print("\nValidation:")
            for k in (
                "conditional_pd",
                "true_pfa",
                "opportunity_interception_ratio",
                "decision_hit_rate",
                "mean_first_intercept_ms",
                "median_first_intercept_ms",
                "mean_decision_dwell_slots",
            ):
                print(f"  {k:40s}: {val_metrics[k]}")

            # Save every epoch, and mark best by the PS-style mission interception ratio.
            self._save("last.pth")
            oir = float(val_metrics["opportunity_interception_ratio"])
            if oir > best_oir:
                best_oir = oir
                self._save("best.pth")
                print(f"[checkpoint] new best validation opportunity_interception_ratio={oir:.6f}")

            history_path = Path(self.cfg.output_dir) / "training_history.json"
            history_path.write_text(json.dumps(self.history, indent=2, allow_nan=True))

        elapsed = time.time() - start_wall
        self._save("final.pth")
        print("\n" + "=" * 88)
        print("TRAINING COMPLETE")
        print("=" * 88)
        print(f"Total environment steps : {self.global_step}")
        print(f"Wall time               : {elapsed/3600:.2f} h")
        print(f"Output directory        : {Path(self.cfg.output_dir).resolve()}")
        print("Saved files             : last.pth, best.pth, final.pth")
        print("Also saved              : *_policy.pth and training_history.json")
        print("=" * 88)


def load_policy_only(
    path: str | Path,
    *,
    dynamic_dwell: bool,
    dwell_multipliers: Sequence[int],
    dwell_time_penalty: float,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=device)
    config = payload.get("config", {})
    policy = ResidualPolicyNetSAC(
        N_BANDS,
        N_BANDS * FEATS_PER_BAND,
        N_BANDS * len(dwell_multipliers) if dynamic_dwell else N_BANDS,
        dynamic_dwell=dynamic_dwell,
        dwell_multipliers=dwell_multipliers,
        dwell_time_penalty=dwell_time_penalty,
    ).to(device)
    state_dict = payload.get("policy_state_dict", payload)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy, config

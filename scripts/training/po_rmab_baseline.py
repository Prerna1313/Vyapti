from __future__ import annotations

"""Vyapti PO-RMAB / belief-Whittle baseline for the final 50/100-ms benchmark.

This is a MODEL-BASED baseline, not a Neural Network. 
It computes a Whittle Index mathematically on a discretized belief grid and serves
as the strict mathematical baseline score that your PPO agent must beat.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Import from the perfectly corrected causal harness!
from causal_harness import (
    BASE_SLOT_S,
    CHECKPOINT_STEPS,
    DEFAULT_SEED,
    N_BANDS,
    N_BASE_SLOTS,
    PFA,
    PD,
    TEST_FILES,
    TRAIN_EPOCHS,
    TRAIN_FILES,
    VAL_FILES,
    VAL_REPLAY_SEED,
    CausalSchedulerState,
    aggregate_metrics,
    build_replay_registry,
    get_dwell_slots,
    detector_sequence,
    save_json,
    validate_rf_contract,
)

from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment

INDEX_GRID = 101
LAMBDA_GRID = 81
WHITTLE_HORIZON = 24
WHITTLE_GAMMA = 0.997
LAMBDA_MIN = -0.25
LAMBDA_MAX = 1.25
PERIODIC_WEIGHT = 0.20


def world_occupancy(recipe: dict) -> np.ndarray:
    """Build the exact threshold-free band-time occupancy by asking the actual physics engine."""
    band_centres = 250.0 + 500.0 * np.arange(N_BANDS)
    
    # Safely load the environment using exactly 50% overlap
    env = TSRDStareEnvironment.from_stare_mode(
        stare_file=recipe["source_file"],
        band_centres_mhz=band_centres,
        passband_halfwidth_mhz=500.0
    )
    
    # Return the exact underlying boolean occupancy grid the physics engine created
    return env._occupancy_grid


def fit_transition_model(recipes: list[dict], smoothing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the exact mathematical transition probabilities using STARE files."""
    counts = np.full((N_BANDS, 2, 2), float(smoothing), dtype=np.float64)
    prior_counts = np.full((N_BANDS, 2), float(smoothing), dtype=np.float64)

    for recipe in recipes:
        z = world_occupancy(recipe)
        prior_counts[:, 1] += z.mean(axis=1)
        prior_counts[:, 0] += 1.0 - z.mean(axis=1)
        prev = z[:, :-1]
        nxt = z[:, 1:]
        for band in range(N_BANDS):
            p = prev[band].astype(np.int8)
            q = nxt[band].astype(np.int8)
            counts[band, 0, 0] += np.count_nonzero((p == 0) & (q == 0))
            counts[band, 0, 1] += np.count_nonzero((p == 0) & (q == 1))
            counts[band, 1, 0] += np.count_nonzero((p == 1) & (q == 0))
            counts[band, 1, 1] += np.count_nonzero((p == 1) & (q == 1))

    transition = counts / counts.sum(axis=2, keepdims=True)
    prior_active = prior_counts[:, 1] / prior_counts.sum(axis=1)
    return transition, prior_active


def posterior(p: np.ndarray, y: int, pd: float = PD, pfa: float = PFA) -> np.ndarray:
    like1 = pd if y else (1.0 - pd)
    like0 = pfa if y else (1.0 - pfa)
    den = p * like1 + (1.0 - p) * like0
    return np.clip((p * like1) / np.maximum(den, 1e-12), 1e-8, 1.0 - 1e-8)


def next_belief_no_observation(p: np.ndarray, p01: float, p11: float) -> np.ndarray:
    return np.clip((1.0 - p) * p01 + p * p11, 1e-8, 1.0 - 1e-8)


def active_expected_next_beliefs(
    belief_grid: np.ndarray,
    p01: float,
    p11: float,
    pd: float,
    pfa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = belief_grid
    py1 = p * pd + (1.0 - p) * pfa
    post1 = posterior(p, 1, pd, pfa)
    post0 = posterior(p, 0, pd, pfa)
    next1 = next_belief_no_observation(post1, p01, p11)
    next0 = next_belief_no_observation(post0, p01, p11)
    return py1, next1, next0


def value_iteration_for_lambda(
    p01: float,
    p11: float,
    lam: float,
    belief_grid: np.ndarray,
    horizon: int = WHITTLE_HORIZON,
    gamma: float = WHITTLE_GAMMA,
) -> tuple[np.ndarray, np.ndarray]:
    V = np.zeros_like(belief_grid, dtype=np.float64)
    py1, next1, next0 = active_expected_next_beliefs(belief_grid, p01, p11, PD, PFA)
    passive_next = next_belief_no_observation(belief_grid, p01, p11)

    for _ in range(horizon):
        v_active_next = (
            py1 * np.interp(next1, belief_grid, V)
            + (1.0 - py1) * np.interp(next0, belief_grid, V)
        )
        v_passive_next = np.interp(passive_next, belief_grid, V)
        q_active = belief_grid * PD + gamma * v_active_next
        q_passive = lam + gamma * v_passive_next
        advantage = q_active - q_passive
        V = np.maximum(q_active, q_passive)
    return V, advantage


def compute_whittle_table(
    transition: np.ndarray,
    *,
    belief_points: int = INDEX_GRID,
    lambda_points: int = LAMBDA_GRID,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    grid = np.linspace(0.0, 1.0, belief_points, dtype=np.float64)
    lambdas = np.linspace(LAMBDA_MIN, LAMBDA_MAX, lambda_points, dtype=np.float64)
    tables = np.empty((N_BANDS, belief_points), dtype=np.float64)
    violation_counts = np.zeros(N_BANDS, dtype=np.int64)

    for band in range(N_BANDS):
        p01 = float(transition[band, 0, 1])
        p11 = float(transition[band, 1, 1])
        advantages = np.empty((belief_points, lambda_points), dtype=np.float64)
        for j, lam in enumerate(lambdas):
            _, adv = value_iteration_for_lambda(p01, p11, float(lam), grid)
            advantages[:, j] = adv

        violation_counts[band] = int(np.count_nonzero(np.diff(advantages, axis=1) > 1e-5))

        idx = np.empty(belief_points, dtype=np.float64)
        for i in range(belief_points):
            a = advantages[i]
            pos = np.where(a >= 0.0)[0]
            if len(pos) == 0:
                idx[i] = LAMBDA_MIN
                continue
            j = int(pos[-1])
            if j == lambda_points - 1:
                idx[i] = LAMBDA_MAX
                continue
            a0, a1 = float(a[j]), float(a[j + 1])
            l0, l1 = float(lambdas[j]), float(lambdas[j + 1])
            if abs(a1 - a0) < 1e-12:
                idx[i] = l0
            else:
                idx[i] = l0 + (0.0 - a0) * (l1 - l0) / (a1 - a0)
        tables[band] = idx

    return grid, tables, {"total_indexability_violations": int(violation_counts.sum())}


@dataclass
class PORMAB:
    transition: np.ndarray
    prior_active: np.ndarray
    belief_grid: np.ndarray
    index_tables: np.ndarray

    def index(self, band: int, belief: float) -> float:
        return float(np.interp(float(np.clip(belief, 0.0, 1.0)), self.belief_grid, self.index_tables[int(band)]))

    def action_scores(self, state: CausalSchedulerState, periodic: bool = False) -> np.ndarray:
        feats = state.pre_action_features()
        beliefs = feats["belief"]
        periodicity = feats["periodicity"]
        scores = np.empty(N_BANDS, dtype=np.float64)
        for b in range(N_BANDS):
            p = float(beliefs[b])
            scores[b] = self.index(b, p)
            if periodic:
                scores[b] += PERIODIC_WEIGHT * float(periodicity[b])
        return scores

    def select(self, state: CausalSchedulerState, periodic: bool = False) -> int:
        scores = self.action_scores(state, periodic=periodic)
        return int(np.argmax(scores))


def evaluate_baseline(registry: list[dict], prior: PORMAB, periodic: bool, seed: int) -> dict[str, Any]:
    """Perfect evaluation loop bypassing the old 100-ms duplication bugs."""
    rows: list[dict] = []
    band_centres = 250.0 + 500.0 * np.arange(N_BANDS)

    for i, recipe in enumerate(registry):
        # 1. Initialize strictly with 50% overlap constraint
        env = TSRDStareEnvironment.from_stare_mode(
            stare_file=recipe["source_file"],
            band_centres_mhz=band_centres,
            passband_halfwidth_mhz=500.0
        )
        
        state = CausalSchedulerState.create(prior.transition, prior.prior_active)
        state.reset(prior.prior_active)
        
        obs = env.reset()
        done = False
        
        while not done:
            action = prior.select(state, periodic=periodic)
            
            # 2. Use our fast training step directly
            obs, reward, done = env.step_training(action, reward_mode="detector_positive")
            
            # 3. Use causal_harness strict sequence extractor (NO HALLUCINATIONS)
            dwell = get_dwell_slots(env, obs, action)
            positives = detector_sequence(obs, dwell)
            
            # 4. Update memory tracking
            state.periodicity.observe_dwell(action, state.elapsed_base_slots, positives)
            state.last_visit[action] = state.elapsed_base_slots
            state.visit_counts[action] += 1
            state.previous_action = action
            state.previous_positive = any(positives)
            state.elapsed_base_slots += dwell
            
            # Advance HMM Filter
            state.belief.step_observations(action, positives)
            
        metrics = env.metrics()
        metrics["world_id"] = int(recipe.get("world_id", i))
        metrics["source_file"] = recipe.get("source_file", "")
        rows.append(metrics)
        
    return {
        "policy": "PO-RMAB+periodicity" if periodic else "PO-RMAB",
        "metrics": aggregate_metrics(rows),
        "rows": rows,
    }


def build_stare_registry(data_dir: str, max_files: int) -> list[dict]:
    """Safely build recipes directly from disk."""
    files = sorted(list(Path(data_dir).glob("*.h5")))[:max_files]
    if not files:
        raise FileNotFoundError(f"Could not find any .h5 STARE files in {data_dir}")
    return build_replay_registry(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="./runs/pormab")
    ap.add_argument("--train-dir", required=True, help="Path to STARE train files")
    ap.add_argument("--val-dir", required=True, help="Path to STARE val files")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building TRAIN registry...")
    train_registry = build_stare_registry(args.train_dir, TRAIN_FILES)

    print("[2/4] Fitting PO-RMAB prior from ground-truth physics engine...")
    transition, prior_active = fit_transition_model(train_registry)
    grid, tables, diagnostics = compute_whittle_table(transition)
    prior = PORMAB(transition, prior_active, grid, tables)

    print("[3/4] Building Validation registry...")
    val_registry = build_stare_registry(args.val_dir, VAL_FILES)

    for periodic in (False, True):
        name = "pormab_periodic" if periodic else "pormab"
        print(f"[4/4] Evaluating {name} on {len(val_registry)} validation files...")
        result = evaluate_baseline(val_registry, prior, periodic, VAL_REPLAY_SEED)
        
        save_json(out / f"{name}_validation.json", result)
        print(f"\n✅ Scorecard for {name}:")
        print(json.dumps(result["metrics"], indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

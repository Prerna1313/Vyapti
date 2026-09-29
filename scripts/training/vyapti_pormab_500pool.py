from __future__ import annotations

"""Vyapti PO-RMAB / belief-Whittle baseline for the final 500-config protocol.

Protocol
--------
2,500 STARE TRAIN configs -> user's saved 500-config cache -> online fresh worlds.
No world-generation epochs and no training-world registry are used.

PO-RMAB uses a finite ONLINE calibration sample (default 500 fresh worlds) to fit
its causal transition model. This is a calibration sample count, not an epoch.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
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
    DEFAULT_SEED,
    VAL_REPLAY_SEED,
    TEST_REPLAY_SEED,
    CausalSchedulerState,
    build_replay_registry,
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
    emitter_count_distribution_from_cache,
    sample_fresh_training_world,
    write_runtime_manifest,
)

MODEL_PD = 0.90
MODEL_PFA = 0.05
DEFAULT_DWELL_SLOTS = 2
DEFAULT_CALIBRATION_WORLDS = 500
INDEX_GRID = 101
LAMBDA_GRID = 81
WHITTLE_HORIZON = 24
WHITTLE_GAMMA = 0.997
LAMBDA_MIN = -0.25
LAMBDA_MAX = 1.25
PERIODIC_WEIGHT = 0.20


def fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fit_transition_model_online(pool: Any, worlds: int, seed: int, distribution: np.ndarray, smoothing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    counts = np.full((N_BANDS, 2, 2), float(smoothing), dtype=np.float64)
    prior_counts = np.full((N_BANDS, 2), float(smoothing), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    t0 = time.perf_counter()

    bar = tqdm(range(int(worlds)), desc="PO-RMAB calibration worlds", unit="world")
    for i in bar:
        env, _sources, world_seed, emitter_count = sample_fresh_training_world(pool, rng)
        truth = np.asarray(env.hidden_truth.grid.any(axis=0), dtype=bool)
        if truth.shape != (N_BANDS, N_BASE_SLOTS):
            raise RuntimeError(f"Unexpected truth occupancy shape {truth.shape}")

        prior_counts[:, 1] += truth.mean(axis=1)
        prior_counts[:, 0] += 1.0 - truth.mean(axis=1)
        prev = truth[:, :-1]
        nxt = truth[:, 1:]
        for band in range(N_BANDS):
            p = prev[band]
            q = nxt[band]
            counts[band, 0, 0] += np.count_nonzero((p == 0) & (q == 0))
            counts[band, 0, 1] += np.count_nonzero((p == 0) & (q == 1))
            counts[band, 1, 0] += np.count_nonzero((p == 1) & (q == 0))
            counts[band, 1, 1] += np.count_nonzero((p == 1) & (q == 1))

        elapsed = time.perf_counter() - t0
        rate = (i + 1) / max(elapsed, 1e-9)
        eta = (worlds - i - 1) / max(rate, 1e-9)
        bar.set_postfix_str(f"elapsed={fmt_seconds(elapsed)} rate={rate:.2f}/s ETA={fmt_seconds(eta)} world_seed={world_seed} N={emitter_count}")

    return counts / counts.sum(axis=2, keepdims=True), prior_counts[:, 1] / prior_counts.sum(axis=1)


def posterior(p: np.ndarray, y: int, pd: float = MODEL_PD, pfa: float = MODEL_PFA) -> np.ndarray:
    like_active = pd if y else (1.0 - pd)
    like_inactive = pfa if y else (1.0 - pfa)
    denom = p * like_active + (1.0 - p) * like_inactive
    return np.clip((p * like_active) / np.maximum(denom, 1e-12), 1e-8, 1.0 - 1e-8)


def next_belief_no_observation(p: np.ndarray, p01: float, p11: float) -> np.ndarray:
    return np.clip((1.0 - p) * p01 + p * p11, 1e-8, 1.0 - 1e-8)


def value_iteration_for_lambda(p01: float, p11: float, lam: float, belief_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    V = np.zeros_like(belief_grid, dtype=np.float64)
    py1 = belief_grid * MODEL_PD + (1.0 - belief_grid) * MODEL_PFA
    passive_next = next_belief_no_observation(belief_grid, p01, p11)
    advantage = np.zeros_like(V)
    for _ in range(WHITTLE_HORIZON):
        post1 = posterior(belief_grid, 1)
        post0 = posterior(belief_grid, 0)
        next1 = next_belief_no_observation(post1, p01, p11)
        next0 = next_belief_no_observation(post0, p01, p11)
        v_active_next = py1 * np.interp(next1, belief_grid, V) + (1.0 - py1) * np.interp(next0, belief_grid, V)
        v_passive_next = np.interp(passive_next, belief_grid, V)
        q_active = belief_grid * MODEL_PD + WHITTLE_GAMMA * v_active_next
        q_passive = lam + WHITTLE_GAMMA * v_passive_next
        advantage = q_active - q_passive
        V = np.maximum(q_active, q_passive)
    return V, advantage


def compute_whittle_table(transition: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    grid = np.linspace(0.0, 1.0, INDEX_GRID, dtype=np.float64)
    lambdas = np.linspace(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_GRID, dtype=np.float64)
    tables = np.empty((N_BANDS, INDEX_GRID), dtype=np.float64)
    violations = np.zeros(N_BANDS, dtype=np.int64)
    low = np.zeros(N_BANDS, dtype=np.int64)
    high = np.zeros(N_BANDS, dtype=np.int64)

    for band in tqdm(range(N_BANDS), desc="Whittle index table", unit="band"):
        p01 = float(transition[band, 0, 1])
        p11 = float(transition[band, 1, 1])
        advantages = np.empty((INDEX_GRID, LAMBDA_GRID), dtype=np.float64)
        for j, lam in enumerate(lambdas):
            _V, adv = value_iteration_for_lambda(p01, p11, float(lam), grid)
            advantages[:, j] = adv
        violations[band] = int(np.count_nonzero(np.diff(advantages, axis=1) > 1e-5))

        idx = np.empty(INDEX_GRID, dtype=np.float64)
        for i, a in enumerate(advantages):
            pos = np.where(a >= 0.0)[0]
            if len(pos) == 0:
                idx[i] = LAMBDA_MIN
                low[band] += 1
            else:
                j = int(pos[-1])
                if j == LAMBDA_GRID - 1:
                    idx[i] = LAMBDA_MAX
                    high[band] += 1
                else:
                    a0, a1 = float(a[j]), float(a[j + 1])
                    l0, l1 = float(lambdas[j]), float(lambdas[j + 1])
                    idx[i] = l0 if abs(a1 - a0) < 1e-12 else l0 + (0.0 - a0) * (l1 - l0) / (a1 - a0)
        tables[band] = idx

    diagnostics = {
        "belief_grid": grid.tolist(),
        "lambda_grid": lambdas.tolist(),
        "horizon": WHITTLE_HORIZON,
        "gamma": WHITTLE_GAMMA,
        "total_indexability_violations": int(violations.sum()),
        "max_band_indexability_violations": int(violations.max()),
        "boundary_low_points": low.tolist(),
        "boundary_high_points": high.tolist(),
    }
    return grid, tables, diagnostics


@dataclass
class PORMAB:
    transition: np.ndarray
    prior_active: np.ndarray
    belief_grid: np.ndarray
    index_tables: np.ndarray
    diagnostics: dict[str, Any]
    dwell_slots: int = DEFAULT_DWELL_SLOTS

    def index(self, band: int, belief: float) -> float:
        return float(np.interp(np.clip(float(belief), 0.0, 1.0), self.belief_grid, self.index_tables[int(band)]))

    def select(self, state: CausalSchedulerState, periodic: bool = False) -> int:
        feats = state.pre_action_features()
        scores = np.empty(N_BANDS, dtype=np.float64)
        for band in range(N_BANDS):
            p = float(feats["belief"][band])
            total = 0.0
            for k in range(max(self.dwell_slots, 1)):
                total += (WHITTLE_GAMMA ** k) * self.index(band, p)
                p = float(np.clip((1.0 - p) * self.transition[band, 0, 1] + p * self.transition[band, 1, 1], 1e-8, 1.0 - 1e-8))
            if periodic:
                total += PERIODIC_WEIGHT * float(feats["periodicity"][band])
            scores[band] = total
        return int(np.argmax(scores))


def build_receiver_env(stare_file: Path):
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


def evaluate_one_replay(recipe: dict[str, Any], prior: PORMAB, *, periodic: bool, seed: int, dwell_slots: int) -> dict[str, Any]:
    from vyapti_simulator.tsrd.benchmark_protocol import score_recorded_replay, _assert_scorecard_consistent, _seed_for_file
    from vyapti_simulator.core.metrics import TrajectoryStep

    stare_file = Path(recipe["source_file"])
    env = build_receiver_env(stare_file)
    receiver_seed = _seed_for_file(stare_file.stem, seed)
    env.reset(seed=int(receiver_seed))
    state = CausalSchedulerState.create(prior.transition, prior_active=prior.prior_active)
    state.reset(prior_active=prior.prior_active)
    trajectory: list[TrajectoryStep] = []

    while not env.done:
        action = prior.select(state, periodic=periodic)
        looks, _reward, done = env.step_dwell_training(action, dwell_slots, reward_mode="detector_positive")
        looks = list(looks)
        if len(looks) != dwell_slots:
            raise RuntimeError(f"Expected {dwell_slots} looks, got {len(looks)}")
        positives = detector_sequence(looks)
        for look in looks:
            trajectory.append(TrajectoryStep(action=int(action), time_slot=int(look["time_slot"]), observation=look))
        state.belief.step_observations(action, positives)
        state.step(action, dwell_slots, positives)
        if done:
            break

    score = score_recorded_replay(env, trajectory)
    _assert_scorecard_consistent(score, trajectory)
    return {"world_id": int(recipe["world_id"]), "config": stare_file.stem, "source_file": str(stare_file), **score}


def evaluate_baseline(corpus_root: Path, prior: PORMAB, *, periodic: bool, split: str, count: int, seed: int, dwell_slots: int) -> dict[str, Any]:
    from vyapti_simulator.tsrd.benchmark_protocol import _summarize
    registry = build_replay_registry(corpus_root, split, count)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    bar = tqdm(registry, desc=f"{split.upper()} {'PO-RMAB+periodicity' if periodic else 'PO-RMAB'}", unit="scene")
    for recipe in bar:
        rows.append(evaluate_one_replay(recipe, prior, periodic=periodic, seed=seed, dwell_slots=dwell_slots))
        elapsed = time.perf_counter() - t0
        rate = len(rows) / max(elapsed, 1e-9)
        eta = (len(registry) - len(rows)) / max(rate, 1e-9)
        bar.set_postfix_str(f"elapsed={fmt_seconds(elapsed)} rate={rate:.2f}/s ETA={fmt_seconds(eta)}")
    return {
        "policy": "PO-RMAB+periodicity" if periodic else "PO-RMAB",
        "split": split,
        "dwell_slots": int(dwell_slots),
        "dwell_ms": 50 * int(dwell_slots),
        "summary": _summarize(rows),
        "rows": rows,
        "runtime_s": time.perf_counter() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dwell-slots", type=int, choices=(1, 2), default=DEFAULT_DWELL_SLOTS)
    ap.add_argument("--calibration-worlds", type=int, default=DEFAULT_CALIBRATION_WORLDS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--rebuild-prior", action="store_true")
    ap.add_argument("--evaluate-test", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    corpus_root = Path(args.corpus_root)
    cache_root = Path(args.cache_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = validate_dataset(corpus_root)
    cache = load_train500_cache(cache_root)
    pool, cache = build_train_pool_from_cache(corpus_root, cache_root)
    write_runtime_manifest(cache)
    distribution = emitter_count_distribution_from_cache(cache)
    if distribution.size != 500:
        raise RuntimeError("Cached 500-config emitter-count distribution is invalid")

    print("=" * 80)
    print("VYAPTI PO-RMAB / FINAL 500-CONFIG ONLINE-WORLD PROTOCOL")
    print("=" * 80)
    print(f"TRAIN source configs   : {counts['train']:,}")
    print(f"Selected source pool  : 500 configs")
    print(f"Pool fingerprint      : {cache['fingerprint'][:16]}...")
    print(f"Calibration worlds    : {args.calibration_worlds:,} fresh online worlds")
    print(f"PPO-style optimizer   : not applicable to PO-RMAB")
    print(f"Dwell                  : {50 * args.dwell_slots} ms")
    print(f"Start                  : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    prior_dir = output_dir / "prior"
    prior_dir.mkdir(parents=True, exist_ok=True)
    transition_file = prior_dir / "pormab_transition.json"
    tables_file = prior_dir / "pormab_index_tables.npy"
    diag_file = prior_dir / "pormab_index_diagnostics.json"

    signature = {
        "source_pool_fingerprint": cache["fingerprint"],
        "calibration_worlds": int(args.calibration_worlds),
        "seed": int(args.seed),
        "model_pd": MODEL_PD,
        "model_pfa": MODEL_PFA,
        "dwell_slots": int(args.dwell_slots),
        "whittle_horizon": WHITTLE_HORIZON,
        "whittle_gamma": WHITTLE_GAMMA,
    }

    cached_ok = False
    if not args.rebuild_prior and transition_file.exists() and tables_file.exists() and diag_file.exists():
        payload = json.loads(transition_file.read_text(encoding="utf-8"))
        cached_ok = all(payload.get(k) == v for k, v in signature.items())

    if cached_ok:
        print("\nLoading cached PO-RMAB prior...")
        payload = json.loads(transition_file.read_text(encoding="utf-8"))
        transition = np.asarray(payload["transition"], dtype=np.float64)
        prior_active = np.asarray(payload["prior_active"], dtype=np.float64)
        tables = np.load(tables_file)
        diagnostics = json.loads(diag_file.read_text(encoding="utf-8"))
        grid = np.asarray(diagnostics["belief_grid"], dtype=np.float64)
    else:
        transition, prior_active = fit_transition_model_online(
            pool,
            int(args.calibration_worlds),
            int(args.seed),
            distribution,
        )
        grid, tables, diagnostics = compute_whittle_table(transition)
        diagnostics.update(signature)
        transition_file.write_text(json.dumps({**signature, "transition": transition.tolist(), "prior_active": prior_active.tolist()}, indent=2, sort_keys=True), encoding="utf-8")
        np.save(tables_file, tables)
        save_json(diag_file, diagnostics)
        save_json(prior_dir / "prior_fit_summary.json", {**signature, "runtime_s": time.perf_counter() - t0})

    prior = PORMAB(transition, prior_active, grid, tables, diagnostics, dwell_slots=int(args.dwell_slots))

    val_plain = evaluate_baseline(corpus_root, prior, periodic=False, split="val", count=VAL_FILES, seed=VAL_REPLAY_SEED, dwell_slots=args.dwell_slots)
    val_periodic = evaluate_baseline(corpus_root, prior, periodic=True, split="val", count=VAL_FILES, seed=VAL_REPLAY_SEED, dwell_slots=args.dwell_slots)
    save_json(output_dir / "VAL_PO_RMAB.json", val_plain)
    save_json(output_dir / "VAL_PO_RMAB_PERIODIC.json", val_periodic)

    result = {
        "algorithm": "PO-RMAB family",
        "protocol": "saved 500-config source pool; online fresh calibration worlds; fixed 100-file VAL replay",
        "source_pool_fingerprint": cache["fingerprint"],
        "calibration_worlds": int(args.calibration_worlds),
        "val": {"pormab": val_plain["summary"], "pormab_periodic": val_periodic["summary"]},
        "test": "skipped",
        "runtime_s": time.perf_counter() - t0,
    }

    if args.evaluate_test:
        test_plain = evaluate_baseline(corpus_root, prior, periodic=False, split="test", count=TEST_FILES, seed=TEST_REPLAY_SEED, dwell_slots=args.dwell_slots)
        test_periodic = evaluate_baseline(corpus_root, prior, periodic=True, split="test", count=TEST_FILES, seed=TEST_REPLAY_SEED, dwell_slots=args.dwell_slots)
        save_json(output_dir / "TEST_PO_RMAB.json", test_plain)
        save_json(output_dir / "TEST_PO_RMAB_PERIODIC.json", test_periodic)
        result["test"] = {"pormab": test_plain["summary"], "pormab_periodic": test_periodic["summary"]}

    result["runtime_s"] = time.perf_counter() - t0
    save_json(output_dir / "run_manifest.json", result)
    print("\nDONE")
    print(f"Total elapsed: {fmt_seconds(time.perf_counter() - t0)}")


if __name__ == "__main__":
    main()

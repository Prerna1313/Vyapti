from __future__ import annotations

"""Shared causal state, timing, reward, registry, and evaluation utilities.

This module intentionally sits above the existing Vyapti RF environment.
It assumes the RF simulator exposes the hidden-world truth only through its
internal state/evaluator and exposes causal detector results through `step_training()`.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Point to your actual Vyapti System B repository
from vyapti_simulator.tsrd import benchmark_protocol as rf


N_BANDS = 36
MISSION_S = 30.0
BASE_SLOT_S = 0.050
N_BASE_SLOTS = 600
BAND_STEP_MHZ = 500.0
RECEIVER_BW_MHZ = 1000.0
OVERLAP_FRACTION = 0.50
PD = 0.90
PFA = 0.05

TRAIN_FILES = 2500
VAL_FILES = 100
TEST_FILES = 100
TRAIN_WORLDS_PER_EPOCH = 100
TRAIN_EPOCHS = 10
TOTAL_TRAIN_WORLDS = TRAIN_WORLDS_PER_EPOCH * TRAIN_EPOCHS
CHECKPOINT_STEPS = (100_000, 200_000, 300_000, 400_000)

DEFAULT_SEED = 20260928
TRAIN_RECIPE_SEED = DEFAULT_SEED + 10_000
VAL_REPLAY_SEED = DEFAULT_SEED + 20_000
TEST_REPLAY_SEED = DEFAULT_SEED + 30_000

EPS = 1e-12


def validate_rf_contract(strict: bool = True) -> dict[str, Any]:
    """Validate the RF simulator's externally visible benchmark constants."""
    checks = {
        "n_bands": getattr(rf, "N_BANDS", None),
        "world_s": getattr(rf, "WORLD_S", None),
        "n_slots": getattr(rf, "N_SLOTS", None),
        "base_slot_s": getattr(rf, "BASE_SLOT_S", getattr(rf, "DT_S", None)),
        "band_step_mhz": getattr(rf, "BAND_STEP_MHZ", getattr(rf, "BAND_WIDTH_MHZ", None)),
        "receiver_bandwidth_mhz": getattr(rf, "RECEIVER_BANDWIDTH_MHZ", None),
        "overlap_fraction": getattr(rf, "BAND_OVERLAP_FRACTION", None),
    }

    errors: list[str] = []
    if checks["n_bands"] is not None and int(checks["n_bands"]) != N_BANDS:
        errors.append(f"N_BANDS={checks['n_bands']} != {N_BANDS}")
    if checks["world_s"] is not None and not math.isclose(float(checks["world_s"]), MISSION_S):
        errors.append(f"WORLD_S={checks['world_s']} != {MISSION_S}")
    if checks["n_slots"] is not None and int(checks["n_slots"]) != N_BASE_SLOTS:
        errors.append(f"N_SLOTS={checks['n_slots']} != {N_BASE_SLOTS}")
    if checks["base_slot_s"] is not None and not math.isclose(float(checks["base_slot_s"]), BASE_SLOT_S, rel_tol=0, abs_tol=1e-9):
        errors.append(f"BASE_SLOT_S={checks['base_slot_s']} != {BASE_SLOT_S}")
    if checks["band_step_mhz"] is not None and not math.isclose(float(checks["band_step_mhz"]), BAND_STEP_MHZ):
        errors.append(f"band step={checks['band_step_mhz']} != {BAND_STEP_MHZ}")
    
    if strict:
        missing_final = [k for k in ("receiver_bandwidth_mhz", "overlap_fraction") if checks[k] is None]
        if missing_final:
            errors.append(
                "RF environment must expose RECEIVER_BANDWIDTH_MHZ=1000 and "
                "BAND_OVERLAP_FRACTION=0.5 for the final benchmark. "
                f"Missing: {missing_final}"
            )
        
        centers = getattr(rf, "BAND_CENTRES_MHZ", None)
        if centers is not None:
            expected = [250.0 + i * 500.0 for i in range(N_BANDS)]
            if not np.allclose(centers, expected):
                errors.append("BAND_CENTRES_MHZ does not match required 250.0 + i*500.0 MHz geometry.")

    if errors:
        raise RuntimeError("Vyapti RF benchmark contract failed:\n- " + "\n- ".join(errors))
    return checks


def safe_reset(env: Any, recipe: dict, receiver_seed: int) -> np.ndarray:
    """Accept either the existing Vyapti reset API or Gymnasium-like reset."""
    try:
        out = env.reset(recipe, receiver_seed)
    except TypeError:
        try:
            out = env.reset(seed=receiver_seed, options={"recipe": recipe})
        except TypeError:
            try:
                out = env.reset(recipe=recipe, receiver_seed=receiver_seed)
            except TypeError:
                out = env.reset()
    if isinstance(out, tuple):
        out = out[0]
    return np.asarray(out, dtype=np.float32)


def safe_step(env: Any, action: int) -> tuple[np.ndarray, float, bool, dict]:
    """Preserves the ACTUAL info dictionary returned by the RF simulator."""
    out = env.step(int(action))
    
    if len(out) == 4:
        obs, reward, done, info = out
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), dict(info)
    elif len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated or truncated), dict(info)
    elif len(out) == 2:
        # Custom environment fallback
        obs, info = out
        done = getattr(env, "done", False)
        return np.asarray(obs, dtype=np.float32), 0.0, bool(done), dict(info)
        
    raise RuntimeError(f"Unexpected env.step() return length {len(out)}. Must return 4 or 5 elements.")


def get_dwell_slots(env: Any, info: dict, action: int) -> int:
    if "dwell_slots" in info:
        d = int(info["dwell_slots"])
        if d > 0:
            return d
    for name in ("last_dwell_slots", "dwell_slots"):
        value = getattr(env, name, None)
        if value is not None:
            if np.isscalar(value):
                d = int(value)
                if d > 0:
                    return d
            try:
                d = int(np.asarray(value)[int(action)])
                if d > 0:
                    return d
            except Exception:
                pass
    for name in ("dwell_slots_for_band", "get_dwell_slots"):
        fn = getattr(env, name, None)
        if callable(fn):
            d = int(fn(int(action)))
            if d > 0:
                return d
    arr = getattr(rf, "DWELL_SLOTS", None)
    if arr is not None:
        d = int(np.asarray(arr)[int(action)])
        if d > 0:
            return d
    return 1


def detector_sequence(info: dict, dwell_slots: int) -> list[bool]:
    """Extract sequence of base slot hits. FAILS LOUDLY if a 100ms aggregate is passed."""
    for key in ("detector_positive_sequence", "positive_sequence", "y_sequence", "hit_sequence"):
        if key in info:
            arr = [bool(x) for x in np.asarray(info[key]).reshape(-1).tolist()]
            if arr:
                if len(arr) != dwell_slots:
                    raise ValueError(f"Detector sequence length {len(arr)} != dwell_slots {dwell_slots}. The environment must return per-slot hits.")
                return arr
                
    for key in ("detector_positive", "y", "hit"):
        if key in info:
            if dwell_slots == 1:
                return [bool(info[key])]
            else:
                raise ValueError(f"Simulator returned single aggregate '{key}' for a {dwell_slots}-slot dwell. Do not duplicate hits.")
                
    return [False for _ in range(dwell_slots)]


def true_hit_count(info: dict, dwell_slots: int) -> int:
    for key in ("true_hit_cell_sequence", "true_detection_sequence"):
        if key in info:
            return int(np.count_nonzero(np.asarray(info[key]).reshape(-1)))
    if "true_hit_cell_count" in info:
        return int(info["true_hit_cell_count"])
    if "true_hit_cell" in info:
        return int(bool(info["true_hit_cell"]))
    return 0


def false_alarm_count(info: dict, dwell_slots: int) -> int:
    for key in ("false_alarm_sequence", "false_alarm_count_sequence"):
        if key in info:
            return int(np.count_nonzero(np.asarray(info[key]).reshape(-1)))
    if "false_alarm_count" in info:
        return int(info["false_alarm_count"])
    if "false_alarm" in info:
        return int(bool(info["false_alarm"]))
    return 0


def emitter_hits(info: dict) -> set[int]:
    arr = info.get("emitter_hits", info.get("intercepted_emitters", []))
    try:
        return {int(x) for x in np.asarray(arr, dtype=np.int64).reshape(-1).tolist()}
    except Exception:
        return set()


def build_train_recipes(pool: Any, *, seed: int = TRAIN_RECIPE_SEED) -> list[dict]:
    rng = np.random.default_rng(int(seed))
    recipes: list[dict] = []
    for epoch in range(TRAIN_EPOCHS):
        for world_in_epoch in range(TRAIN_WORLDS_PER_EPOCH):
            r = pool.sample_recipe(rng)
            recipes.append(
                {
                    "epoch": epoch + 1,
                    "world_in_epoch": world_in_epoch,
                    "global_world": len(recipes),
                    "uids": list(r["uids"]),
                    "offsets": [int(x) for x in r["offsets"]],
                    "n_emitters": int(r["n_emitters"]),
                }
            )
    return recipes


def build_replay_registry(files: list[Path]) -> list[dict]:
    """Build replay worlds DIRECTLY from their own Stare files.
    
    Guarantees no leakage by entirely avoiding the training pool structure.
    """
    registry: list[dict] = []
    for i, path in enumerate(files):
        registry.append(
            {
                "world_id": i,
                "source_file": str(path),
                "config_id": path.stem,
                "replay": True,
            }
        )
    return registry


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


class BeliefFilter:
    """Two-state causal HMM filter.
    
    NOTE: Currently uses fixed PD/PFA. Once the 50/100-ms receiver actually models
    longer integrations, this observation model should accept dwell-specific likelihoods
    (e.g., P(Y1, Y2 | occupied) for 100ms).
    """

    def __init__(self, transition: np.ndarray, pd: float = PD, pfa: float = PFA, prior_active: np.ndarray | None = None):
        transition = np.asarray(transition, dtype=np.float64)
        if transition.shape != (N_BANDS, 2, 2):
            raise ValueError(f"transition must have shape (36,2,2), got {transition.shape}")
        self.transition = transition
        self.pd = float(pd)
        self.pfa = float(pfa)
        self.prior = np.full(N_BANDS, 0.5, dtype=np.float64) if prior_active is None else np.asarray(prior_active, dtype=np.float64).copy()
        self.belief = self.prior.copy()

    def reset(self, prior_active: np.ndarray | None = None) -> None:
        self.belief = self.prior.copy() if prior_active is None else np.asarray(prior_active, dtype=np.float64).copy()

    def _predict_one(self, band: int, p: float) -> float:
        p01 = self.transition[band, 0, 1]
        p11 = self.transition[band, 1, 1]
        return float(np.clip((1.0 - p) * p01 + p * p11, 1e-6, 1.0 - 1e-6))

    def _posterior(self, p: float, y: bool) -> float:
        py1 = self.pd if y else (1.0 - self.pd)
        py0 = self.pfa if y else (1.0 - self.pfa)
        den = p * py1 + (1.0 - p) * py0
        return float(np.clip(p * py1 / max(den, EPS), 1e-6, 1.0 - 1e-6))

    @staticmethod
    def entropy(p: np.ndarray | float) -> np.ndarray | float:
        x = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0 - 1e-8)
        h = -(x * np.log(x) + (1.0 - x) * np.log(1.0 - x))
        return h / math.log(2.0)

    def step_observations(self, band: int, ys: Iterable[bool]) -> list[float]:
        band = int(band)
        pre: list[float] = []
        for y in ys:
            p_before = float(self.belief[band])
            pre.append(p_before)
            p_post = self._posterior(p_before, bool(y))
            next_belief = self.belief.copy()
            for b in range(N_BANDS):
                p0 = float(self.belief[b])
                if b == band:
                    p0 = p_post
                next_belief[b] = self._predict_one(b, p0)
            self.belief = next_belief
        return pre


class PeriodicityCue:
    """Causal dwell-level periodicity estimator (Rough Cue)."""

    def __init__(self, max_events: int = 32, tolerance_fraction: float = 0.15):
        self.max_events = int(max_events)
        self.tolerance_fraction = float(tolerance_fraction)
        self.times: list[list[float]] = [[] for _ in range(N_BANDS)]

    def reset(self) -> None:
        self.times = [[] for _ in range(N_BANDS)]

    def observe_dwell(self, band: int, start_slot: int, positives: list[bool]) -> None:
        """Properly timestamps individual positive observations within the dwell."""
        hist = self.times[int(band)]
        for i, pos in enumerate(positives):
            if pos:
                hist.append(float(start_slot + i))
                
        if len(hist) > self.max_events:
            del hist[:-self.max_events]

    def estimate(self, band: int, now_slot: int) -> tuple[float, float, float]:
        hist = self.times[int(band)]
        if len(hist) < 3:
            return 0.0, 0.0, 0.0
        gaps = np.diff(np.asarray(hist, dtype=np.float64))
        gaps = gaps[gaps > 0]
        if len(gaps) < 2:
            return 0.0, 0.0, 0.0
        period = float(np.median(gaps))
        if period <= 1.0:
            return 0.0, 0.0, 0.0
        cv = float(np.std(gaps) / max(np.mean(gaps), EPS))
        regularity = float(np.exp(-cv))
        last = float(hist[-1])
        elapsed = max(0.0, float(now_slot) - last)
        remainder = elapsed % period
        phase_error = min(remainder, period - remainder)
        tol = max(self.tolerance_fraction * period, 0.5)
        phase_score = float(np.exp(-phase_error / tol))
        confidence = float(np.clip(regularity * min(1.0, len(gaps) / 6.0), 0.0, 1.0))
        score = float(np.clip(phase_score * confidence, 0.0, 1.0))
        period_norm = float(np.clip(period / N_BASE_SLOTS, 0.0, 1.0))
        return score, confidence, period_norm

    def feature_matrix(self, now_slot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vals = [self.estimate(b, now_slot) for b in range(N_BANDS)]
        return (
            np.asarray([v[0] for v in vals], dtype=np.float32),
            np.asarray([v[1] for v in vals], dtype=np.float32),
            np.asarray([v[2] for v in vals], dtype=np.float32),
        )


@dataclass
class CausalSchedulerState:
    belief: BeliefFilter
    periodicity: PeriodicityCue
    last_visit: np.ndarray
    visit_counts: np.ndarray
    previous_action: int = -1
    previous_positive: bool = False
    elapsed_base_slots: int = 0

    @classmethod
    def create(cls, transition: np.ndarray, prior_active: np.ndarray | None = None) -> "CausalSchedulerState":
        return cls(
            belief=BeliefFilter(transition, prior_active=prior_active),
            periodicity=PeriodicityCue(),
            last_visit=np.full(N_BANDS, -1, dtype=np.int32),
            visit_counts=np.zeros(N_BANDS, dtype=np.int32),
        )

    def reset(self, prior_active: np.ndarray | None = None) -> None:
        self.belief.reset(prior_active)
        self.periodicity.reset()
        self.last_visit.fill(-1)
        self.visit_counts.fill(0)
        self.previous_action = -1
        self.previous_positive = False
        self.elapsed_base_slots = 0

    def pre_action_features(self) -> dict[str, np.ndarray]:
        t = self.elapsed_base_slots
        staleness = np.where(self.last_visit >= 0, (t - self.last_visit) / N_BASE_SLOTS, 1.0)
        staleness = np.clip(staleness, 0.0, 1.0).astype(np.float32)
        periodic, periodic_conf, period_norm = self.periodicity.feature_matrix(t)
        belief = self.belief.belief.astype(np.float32)
        entropy = np.asarray(BeliefFilter.entropy(belief), dtype=np.float32)
        return {
            "belief": belief,
            "belief_entropy": entropy,
            "staleness": staleness,
            "periodicity": periodic,
            "periodicity_confidence": periodic_conf,
            "period_norm": period_norm,
            "visit_count_norm": np.clip(self.visit_counts / 32.0, 0.0, 1.0).astype(np.float32),
        }


class RewardEngine:
    VALID = {
        "detector_positive",
        "true_detection",
        "first_intercept",
        "causal_belief_hit",
        "causal_balanced",
    }

    def __init__(self, mode: str = "causal_belief_hit"):
        if mode not in self.VALID:
            raise ValueError(f"reward mode {mode!r} not in {sorted(self.VALID)}")
        self.mode = mode
        self.seen_emitters: set[int] = set()
        self.total_base_slots = 0
        self.visit_slots = np.zeros(N_BANDS, dtype=np.float64)

    def reset(self) -> None:
        self.seen_emitters.clear()
        self.total_base_slots = 0
        self.visit_slots.fill(0.0)

    def compute(self, *, action: int, dwell_slots: int, positives: list[bool], prebeliefs: np.ndarray,
                info: dict, causal_state: CausalSchedulerState, previous_action: int) -> float:
        y = np.asarray(positives, dtype=np.float64)
        reward = 0.0
        
        if self.mode == "detector_positive":
            reward = float(y.sum())
        elif self.mode == "true_detection":
            # DIAGNOSTIC ABLATION ONLY. Simulator truth allowed.
            # Do NOT use as the causal headline reward for training.
            reward = float(true_hit_count(info, dwell_slots))
        elif self.mode == "first_intercept":
            hits = emitter_hits(info)
            new = hits.difference(self.seen_emitters)
            reward = float(len(new))
            self.seen_emitters.update(hits)
        elif self.mode == "causal_belief_hit":
            reward = float(np.sum(y * np.asarray(prebeliefs, dtype=np.float64)))
        elif self.mode == "causal_balanced":
            belief_hit = float(np.sum(y * np.asarray(prebeliefs, dtype=np.float64)))
            periodic, _, _ = causal_state.periodicity.feature_matrix(causal_state.elapsed_base_slots)
            staleness = causal_state.pre_action_features()["staleness"][int(action)]
            fair = 1.0 / N_BANDS
            share = self.visit_slots[int(action)] / max(float(self.total_base_slots), 1.0)
            concentration_penalty = max(0.0, share - fair) / max(1.0 - fair, EPS)
            
            # Use correctly cached previous_action passed from evaluate loop
            repeat_penalty = 0.0 if previous_action < 0 or previous_action != int(action) else 0.002
            
            reward = (
                belief_hit
                + 0.02 * float(staleness)
                + 0.01 * float(periodic[int(action)])
                - 0.02 * float(concentration_penalty) * float(dwell_slots)
                - repeat_penalty
            )
        self.visit_slots[int(action)] += float(dwell_slots)
        self.total_base_slots += int(dwell_slots)
        return float(reward)


def aggregate_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}

    def mean_key(key: str) -> float:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    out: dict[str, float] = {}
    for key in (
        "opportunity_interception_ratio",
        "emitter_interception_ratio",
        "conditional_pd",
        "true_pfa",
        "average_intercept_rate_hz",
        "average_seconds_per_intercept",
        "unique_band_coverage",
        "mean_censored_ttfi_s",
        "median_censored_ttfi_s",
        "p95_censored_ttfi_s",
        "mean_revisit_s",
        "p95_revisit_s",
        "max_revisit_s",
        "switch_rate",
        "normalized_action_entropy",
    ):
        out[key] = mean_key(key)
    out["n_episodes"] = float(len(rows))
    out["total_true_hits"] = float(sum(int(r.get("true_hit_cells", 0)) for r in rows))
    out["total_false_alarms"] = float(sum(int(r.get("false_alarm_events", 0)) for r in rows))
    return out


def evaluate_policy_on_registry(
    env_factory,
    registry: list[dict],
    policy_fn,
    transition: np.ndarray,
    prior_active: np.ndarray,
    reward_mode: str = "causal_belief_hit",
    deterministic_seed: int = DEFAULT_SEED,
) -> tuple[list[dict], dict[str, float]]:
    rows: list[dict] = []
    
    for i, recipe in enumerate(registry):
        env = env_factory()
        state = CausalSchedulerState.create(transition, prior_active)
        state.reset(prior_active)
        reward_engine = RewardEngine(reward_mode)
        reward_engine.reset()
        
        obs = safe_reset(env, recipe, deterministic_seed + i)
        done = False
        ppo_action_steps = 0
        
        while not done:
            action = int(policy_fn(obs, state))
            next_obs, _, done, info = safe_step(env, action)
            dwell = get_dwell_slots(env, info, action)
            positives = detector_sequence(info, dwell)
            
            pre_seq = state.belief.step_observations(action, positives)
            
            # Cache previous action BEFORE updating state for the repeat penalty
            prev_action = state.previous_action
            
            state.periodicity.observe_dwell(action, state.elapsed_base_slots, positives)
            state.last_visit[action] = state.elapsed_base_slots
            state.visit_counts[action] += 1
            state.previous_action = action
            state.previous_positive = any(positives)
            
            reward_engine.compute(
                action=action,
                dwell_slots=dwell,
                positives=positives,
                prebeliefs=np.asarray(pre_seq, dtype=np.float32),
                info=info,
                causal_state=state,
                previous_action=prev_action,  # Fixed penalty logic
            )
            
            state.elapsed_base_slots += dwell
            ppo_action_steps += 1
            obs = next_obs
            
        metrics = env.metrics() if hasattr(env, "metrics") else {}
        metrics["world_id"] = int(recipe.get("world_id", i))
        metrics["source_file"] = recipe.get("source_file", "")
        metrics["ppo_action_steps"] = ppo_action_steps
        metrics["elapsed_base_slots"] = state.elapsed_base_slots
        rows.append(metrics)
        
    return rows, aggregate_metrics(rows)

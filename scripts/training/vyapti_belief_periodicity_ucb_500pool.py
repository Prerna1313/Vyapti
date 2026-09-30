from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from causal_harness import (
    N_BANDS,
    VAL_FILES,
    VAL_REPLAY_SEED,
    DEFAULT_SEED,
    CausalSchedulerState,
    build_replay_registry,
    detector_sequence,
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
)
from vyapti_simulator.tsrd.benchmark_protocol import (
    score_recorded_replay,
    _assert_scorecard_consistent,
    _seed_for_file,
)
from vyapti_simulator.core.metrics import TrajectoryStep


DEFAULT_DWELL_SLOTS = 2
DEFAULT_UCB_C = 1.0
PERIODIC_WEIGHT = 0.20


def save_json(path: Path, obj):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def fmt_seconds(seconds: float) -> str:

    seconds = max(
        0.0,
        float(seconds),
    )

    h, rem = divmod(
        int(seconds),
        3600,
    )

    m, s = divmod(
        rem,
        60,
    )

    return f"{h:02d}:{m:02d}:{s:02d}"


def load_prior(prior_dir: Path):

    path = (
        prior_dir
        / "pormab_transition.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing PO-RMAB prior: {path}"
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    transition = np.asarray(
        payload["transition"],
        dtype=np.float64,
    )

    prior_active = np.asarray(
        payload["prior_active"],
        dtype=np.float64,
    )

    if transition.shape != (
        N_BANDS,
        2,
        2,
    ):
        raise RuntimeError(
            f"Bad transition shape: "
            f"{transition.shape}"
        )

    if prior_active.shape != (
        N_BANDS,
    ):
        raise RuntimeError(
            f"Bad prior shape: "
            f"{prior_active.shape}"
        )

    return (
        transition,
        prior_active,
    )


def build_receiver_env(stare_file: Path):

    from vyapti_simulator.tsrd.tsrd_environment import (
        TSRDStareEnvironment,
    )

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


class BeliefPeriodicityUCB:

    def __init__(
        self,
        c: float = DEFAULT_UCB_C,
        periodic_weight: float = PERIODIC_WEIGHT,
    ):

        self.c = float(c)
        self.periodic_weight = float(
            periodic_weight
        )

        self.counts = np.zeros(
            N_BANDS,
            dtype=np.int64,
        )

        self.total_actions = 0

    def reset(self):

        self.counts.fill(0)
        self.total_actions = 0

    def select(
        self,
        state: CausalSchedulerState,
    ) -> int:

        feats = state.pre_action_features()

        belief = np.asarray(
            feats["belief"],
            dtype=np.float64,
        )

        periodicity = np.asarray(
            feats["periodicity"],
            dtype=np.float64,
        )

        unseen = np.flatnonzero(
            self.counts == 0
        )

        if len(unseen) > 0:

            # Deterministic cold-start exploration.
            return int(unseen[0])

        bonus = (
            self.c
            * np.sqrt(
                np.log(
                    max(
                        self.total_actions + 1,
                        2,
                    )
                )
                / np.maximum(
                    self.counts,
                    1,
                )
            )
        )

        score = (
            belief
            + bonus
            + self.periodic_weight
            * periodicity
        )

        return int(
            np.argmax(score)
        )

    def observe(
        self,
        action: int,
    ):

        self.counts[int(action)] += 1
        self.total_actions += 1


def evaluate_one_replay(
    recipe: dict,
    *,
    transition: np.ndarray,
    prior_active: np.ndarray,
    seed: int,
    dwell_slots: int,
    ucb_c: float,
    periodic_weight: float,
) -> dict:

    stare_file = Path(
        recipe["source_file"]
    )

    env = build_receiver_env(
        stare_file
    )

    receiver_seed = _seed_for_file(
        stare_file.stem,
        seed,
    )

    env.reset(
        seed=int(receiver_seed)
    )

    state = CausalSchedulerState.create(
        transition,
        prior_active=prior_active,
    )

    state.reset(
        prior_active=prior_active
    )

    policy = BeliefPeriodicityUCB(
        c=ucb_c,
        periodic_weight=periodic_weight,
    )

    policy.reset()

    trajectory: list[TrajectoryStep] = []

    action_steps = 0

    while not env.done:

        action = policy.select(
            state
        )

        looks, _reward, done = (
            env.step_dwell_training(
                action,
                dwell_slots,
                reward_mode="detector_positive",
            )
        )

        looks = list(looks)

        if len(looks) != dwell_slots:
            raise RuntimeError(
                f"Expected {dwell_slots} looks, "
                f"got {len(looks)}"
            )

        positives = detector_sequence(
            looks
        )

        for look in looks:

            trajectory.append(
                TrajectoryStep(
                    action=int(action),
                    time_slot=int(
                        look["time_slot"]
                    ),
                    observation=look,
                )
            )

        # Causal belief update.
        state.belief.step_observations(
            action,
            positives,
        )

        # Periodicity and visit statistics.
        state.step(
            action,
            dwell_slots,
            positives,
        )

        policy.observe(
            action
        )

        action_steps += 1

        if done:
            break

    score = score_recorded_replay(
        env,
        trajectory,
    )

    _assert_scorecard_consistent(
        score,
        trajectory,
    )

    return {
        "world_id": int(
            recipe["world_id"]
        ),
        "config": stare_file.stem,
        "source_file": str(stare_file),
        "action_steps": int(
            action_steps
        ),
        **score,
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--corpus-root",
        required=True,
    )

    ap.add_argument(
        "--prior-dir",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    ap.add_argument(
        "--dwell-slots",
        type=int,
        choices=(1, 2),
        default=DEFAULT_DWELL_SLOTS,
    )

    ap.add_argument(
        "--count",
        type=int,
        default=VAL_FILES,
    )

    ap.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    ap.add_argument(
        "--ucb-c",
        type=float,
        default=DEFAULT_UCB_C,
    )

    ap.add_argument(
        "--periodic-weight",
        type=float,
        default=PERIODIC_WEIGHT,
    )

    args = ap.parse_args()

    t0 = time.perf_counter()

    corpus_root = Path(
        args.corpus_root
    )

    prior_dir = Path(
        args.prior_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts = validate_dataset(
        corpus_root
    )

    transition, prior_active = load_prior(
        prior_dir
    )

    registry = build_replay_registry(
        corpus_root,
        split=args.split,
        count=int(args.count),
    )

    print("=" * 80)
    print("VYAPTI BELIEF + PERIODICITY UCB")
    print("=" * 80)
    print(f"TRAIN configs          : {counts['train']:,}")
    print(f"VAL replay scenes      : {len(registry):,}")
    print(f"Dwell                  : {50 * args.dwell_slots} ms")
    print(f"UCB c                  : {args.ucb_c}")
    print(
        f"Periodic weight       : "
        f"{args.periodic_weight}"
    )
    print(
        "Score                  : "
        "belief + UCB bonus + periodicity"
    )
    print(
        "Hidden truth in policy : NO"
    )

    rows = []

    bar = tqdm(
        registry,
        desc="VAL Belief+Periodicity-UCB",
        unit="scene",
    )

    for recipe in bar:

        rows.append(
            evaluate_one_replay(
                recipe,
                transition=transition,
                prior_active=prior_active,
                seed=VAL_REPLAY_SEED,
                dwell_slots=args.dwell_slots,
                ucb_c=float(args.ucb_c),
                periodic_weight=float(
                    args.periodic_weight
                ),
            )
        )

    from vyapti_simulator.tsrd.benchmark_protocol import _summarize

    summary = _summarize(rows)

    result = {
        "algorithm": "Belief+Periodicity-UCB",
        "policy": (
            "belief + UCB exploration "
            "+ periodicity"
        ),
        "split": "val",
        "replay_count": len(rows),
        "dwell_slots": int(
            args.dwell_slots
        ),
        "dwell_ms": int(
            50 * args.dwell_slots
        ),
        "seed": int(
            VAL_REPLAY_SEED
        ),
        "ucb_c": float(
            args.ucb_c
        ),
        "periodic_weight": float(
            args.periodic_weight
        ),
        "prior_source": str(
            prior_dir
            / "pormab_transition.json"
        ),
        "hidden_truth_for_policy": False,
        "summary": summary,
        "rows": rows,
        "runtime_s": (
            time.perf_counter()
            - t0
        ),
    }

    save_json(
        output_dir
        / "VAL_BELIEF_PERIODICITY_UCB.json",
        result,
    )

    print("\nSUMMARY")
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        f"\nTotal elapsed: "
        f"{fmt_seconds(time.perf_counter() - t0)}"
    )


if __name__ == "__main__":
    main()


   # %cd /content/Vyapti_code

#!PYTHONPATH=/content/Vyapti_code python scripts/baselines/vyapti_belief_periodicity_ucb_500pool.py \
 # --corpus-root /content/Vyapti/tsrd_stare \
  #--prior-dir /content/Vyapti/runs/pormab_500pool/prior \
 # --output-dir /content/Vyapti/runs/belief_periodicity_ucb_500pool \
  #--dwell-slots 2 \
 # --ucb-c 1.0 \
  #--periodic-weight 0.20 \
  #--count 100


#%cd /content/Vyapti_code

#!PYTHONPATH=/content/Vyapti_code python scripts/baselines/vyapti_greedy_belief_500pool.py \
#  --corpus-root /content/Vyapti/tsrd_stare \
 # --prior-dir /content/Vyapti/runs/pormab_500pool/prior \
 # --output-dir /content/Vyapti/runs/greedy_belief_500pool \
 # --dwell-slots 2 \
 # --count 100
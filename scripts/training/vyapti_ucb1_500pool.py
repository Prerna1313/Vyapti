from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from causal_harness import (
    N_BANDS,
    VAL_FILES,
    VAL_REPLAY_SEED,
    DEFAULT_SEED,
    detector_sequence,
    build_replay_registry,
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


class UCB1Policy:

    def __init__(
        self,
        c: float = DEFAULT_UCB_C,
    ):
        self.c = float(c)
        self.counts = np.zeros(
            N_BANDS,
            dtype=np.int64,
        )
        self.reward_sum = np.zeros(
            N_BANDS,
            dtype=np.float64,
        )
        self.total_actions = 0

    def reset(self):

        self.counts.fill(0)
        self.reward_sum.fill(0.0)
        self.total_actions = 0

    def select(self) -> int:

        unseen = np.flatnonzero(
            self.counts == 0
        )

        if len(unseen) > 0:
            # Deterministic exploration of all arms.
            return int(unseen[0])

        empirical_mean = (
            self.reward_sum
            / np.maximum(
                self.counts,
                1,
            )
        )

        bonus = (
            self.c
            * np.sqrt(
                np.log(
                    max(
                        self.total_actions,
                        2,
                    )
                )
                / self.counts
            )
        )

        score = empirical_mean + bonus

        return int(np.argmax(score))

    def observe(
        self,
        positives: list[bool],
    ):

        # Normalize dwell reward to [0, 1].
        reward = float(
            np.mean(
                np.asarray(
                    positives,
                    dtype=np.float64,
                )
            )
        )

        # Action is supplied by caller through temporary field.
        raise RuntimeError(
            "Use observe_action() instead."
        )

    def observe_action(
        self,
        action: int,
        positives: list[bool],
    ):

        reward = float(
            np.mean(
                np.asarray(
                    positives,
                    dtype=np.float64,
                )
            )
        )

        self.counts[int(action)] += 1
        self.reward_sum[int(action)] += reward
        self.total_actions += 1


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

    seconds = max(0.0, float(seconds))

    h, rem = divmod(
        int(seconds),
        3600,
    )

    m, s = divmod(
        rem,
        60,
    )

    return f"{h:02d}:{m:02d}:{s:02d}"


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


def evaluate_one_replay(
    recipe: dict,
    *,
    seed: int,
    dwell_slots: int,
    ucb_c: float,
):

    stare_file = Path(
        recipe["source_file"]
    )

    receiver_seed = _seed_for_file(
        stare_file.stem,
        seed,
    )

    env = build_receiver_env(
        stare_file
    )

    env.reset(
        seed=int(receiver_seed)
    )

    policy = UCB1Policy(
        c=ucb_c
    )

    policy.reset()

    trajectory: list[TrajectoryStep] = []

    action_steps = 0

    while not env.done:

        action = policy.select()

        looks, _reward, done = env.step_dwell_training(
            action,
            dwell_slots,
            reward_mode="detector_positive",
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
                    time_slot=int(look["time_slot"]),
                    observation=look,
                )
            )

        policy.observe_action(
            action,
            positives,
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

    args = ap.parse_args()

    t0 = time.perf_counter()

    corpus_root = Path(
        args.corpus_root
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

    registry = build_replay_registry(
        corpus_root,
        split=args.split,
        count=int(args.count),
    )

    print("=" * 80)
    print("VYAPTI UCB1 BASELINE")
    print("=" * 80)
    print(f"TRAIN configs          : {counts['train']:,}")
    print(f"VAL replay scenes      : {len(registry):,}")
    print(f"Dwell                  : {50 * args.dwell_slots} ms")
    print(f"UCB c                  : {args.ucb_c}")
    print(f"Observed reward        : mean detector-positive rate")
    print(f"Hidden truth in policy : NO")

    rows = []

    bar = tqdm(
        registry,
        desc="VAL UCB1",
        unit="scene",
    )

    for recipe in bar:

        rows.append(
            evaluate_one_replay(
                recipe,
                seed=VAL_REPLAY_SEED,
                dwell_slots=args.dwell_slots,
                ucb_c=float(args.ucb_c),
            )
        )

    from vyapti_simulator.tsrd.benchmark_protocol import _summarize

    summary = _summarize(rows)

    result = {
        "algorithm": "UCB1",
        "policy": "UCB1",
        "split": "val",
        "replay_count": len(rows),
        "dwell_slots": int(args.dwell_slots),
        "dwell_ms": int(50 * args.dwell_slots),
        "seed": int(VAL_REPLAY_SEED),
        "ucb_c": float(args.ucb_c),
        "observed_reward": (
            "mean detector-positive rate over dwell"
        ),
        "hidden_truth_for_policy": False,
        "summary": summary,
        "rows": rows,
        "runtime_s": (
            time.perf_counter() - t0
        ),
    }

    save_json(
        output_dir / "VAL_UCB1.json",
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
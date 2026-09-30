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


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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


def evaluate_one_replay(
    recipe: dict,
    *,
    seed: int,
    dwell_slots: int,
) -> dict:

    stare_file = Path(recipe["source_file"])

    receiver_seed = _seed_for_file(
        stare_file.stem,
        seed,
    )

    env = build_receiver_env(stare_file)
    env.reset(seed=int(receiver_seed))

    # Separate deterministic RNG for this world.
    rng = np.random.default_rng(
        int(seed) + 1_000_003 * int(recipe["world_id"])
    )

    trajectory: list[TrajectoryStep] = []

    action_steps = 0

    while not env.done:

        action = int(
            rng.integers(
                0,
                N_BANDS,
            )
        )

        looks, _reward, done = env.step_dwell_training(
            action,
            dwell_slots,
            reward_mode="detector_positive",
        )

        looks = list(looks)

        if len(looks) != dwell_slots:
            raise RuntimeError(
                f"Expected {dwell_slots} looks, got {len(looks)}"
            )

        for look in looks:
            trajectory.append(
                TrajectoryStep(
                    action=int(action),
                    time_slot=int(look["time_slot"]),
                    observation=look,
                )
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
        "world_id": int(recipe["world_id"]),
        "config": stare_file.stem,
        "source_file": str(stare_file),
        "action_steps": int(action_steps),
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

    args = ap.parse_args()

    t0 = time.perf_counter()

    corpus_root = Path(args.corpus_root)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts = validate_dataset(corpus_root)

    registry = build_replay_registry(
        corpus_root,
        split=args.split,
        count=int(args.count),
    )

    print("=" * 80)
    print("VYAPTI RANDOM BASELINE")
    print("=" * 80)
    print(f"TRAIN configs          : {counts['train']:,}")
    print(f"VAL replay scenes      : {len(registry):,}")
    print(f"Dwell                  : {50 * args.dwell_slots} ms")
    print(f"Policy                 : seeded uniform random")
    print(f"Hidden truth in policy : NO")

    rows = []

    bar = tqdm(
        registry,
        desc="VAL Random",
        unit="scene",
    )

    for recipe in bar:

        rows.append(
            evaluate_one_replay(
                recipe,
                seed=VAL_REPLAY_SEED,
                dwell_slots=args.dwell_slots,
            )
        )

    from vyapti_simulator.tsrd.benchmark_protocol import _summarize

    summary = _summarize(rows)

    result = {
        "algorithm": "Random",
        "policy": "SeededUniformRandom",
        "split": "val",
        "replay_count": len(rows),
        "dwell_slots": int(args.dwell_slots),
        "dwell_ms": int(50 * args.dwell_slots),
        "seed": int(VAL_REPLAY_SEED),
        "summary": summary,
        "rows": rows,
        "runtime_s": time.perf_counter() - t0,
    }

    save_json(
        output_dir / "VAL_RANDOM.json",
        result,
    )

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))

    print(
        f"\nTotal elapsed: "
        f"{fmt_seconds(time.perf_counter() - t0)}"
    )


if __name__ == "__main__":
    main()
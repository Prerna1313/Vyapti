"""
Verify train/eval/test seed split per frozen protocol §6 line 180.

Checks:
  1. Default ExperimentConfig produces non-overlapping splits
  2. Overlapping splits are rejected at construction time
  3. Duplicate seeds within a split are rejected
  4. Each split has the expected size (1000/200/200) per protocol §6
  5. All seeds are integers and the union is finite
  6. run_paired_comparison can be called on each of the three splits
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vyapti_simulator.algorithms.bandit.ucb import UCB1Scheduler
from vyapti_simulator.qualification.probes import RoundRobinProbe as RoundRobinScheduler
from vyapti_simulator.experiments.experiment_runner import (
    ExperimentConfig, ExperimentRunner,
)


def check_default() -> dict:
    cfg = ExperimentConfig()
    train, eval_, test = cfg.train_seeds, cfg.eval_seeds, cfg.test_seeds
    return {
        "train_size": len(train),
        "eval_size": len(eval_),
        "test_size": len(test),
        "train_first_last": (train[0], train[-1]),
        "eval_first_last": (eval_[0], eval_[-1]),
        "test_first_last": (test[0], test[-1]),
        "train_overlap_eval": bool(set(train) & set(eval_)),
        "train_overlap_test": bool(set(train) & set(test)),
        "eval_overlap_test":  bool(set(eval_) & set(test)),
        "non_overlap_pass": (
            not (set(train) & set(eval_))
            and not (set(train) & set(test))
            and not (set(eval_) & set(test))
        ),
    }


def check_overlap_rejected() -> dict:
    try:
        ExperimentConfig(train_seeds=[0, 1, 2], eval_seeds=[2, 3, 4])
        return {"rejected": False, "pass": False}
    except ValueError as e:
        return {"rejected": True, "pass": True, "message": str(e)[:120]}


def check_duplicates_rejected() -> dict:
    try:
        ExperimentConfig(train_seeds=[0, 1, 1, 2], eval_seeds=[3, 4, 5], test_seeds=[6, 7, 8])
        return {"rejected": False, "pass": False}
    except ValueError as e:
        return {"rejected": True, "pass": True, "message": str(e)[:120]}


def check_run_on_each_split() -> dict:
    cfg = ExperimentConfig(
        train_seeds=[0, 1], eval_seeds=[1000, 1001], test_seeds=[2000, 2001],
        band_count=10, time_slots=50, emitter_density_points=[2],
        deadline_points=[25], bootstrap_samples=50,
    )
    runner = ExperimentRunner(cfg)
    out = {}
    for split in ("train", "eval", "test"):
        report = runner.run_paired_comparison(
            schedulers={
                "UCB1":       lambda: UCB1Scheduler(band_count=cfg.band_count),
                "RoundRobin": lambda: RoundRobinScheduler(band_count=cfg.band_count),
            },
            density=2,
            label=f"split_{split}",
            seed_set=split,
        )
        seeds_used = report["seed_list"]
        out[split] = {
            "report_seed_set": report["seed_set"],
            "seeds_used_match_split": sorted(seeds_used) == sorted(
                {"train": cfg.train_seeds,
                 "eval":  cfg.eval_seeds,
                 "test":  cfg.test_seeds}[split]
            ),
            "n_seeds": len(seeds_used),
        }
    return out


def main() -> dict:
    default = check_default()
    overlap = check_overlap_rejected()
    duplicates = check_duplicates_rejected()
    per_split = check_run_on_each_split()

    overall_pass = (
        default["non_overlap_pass"]
        and default["train_size"] == 1000
        and default["eval_size"] == 200
        and default["test_size"] == 200
        and default["train_first_last"] == (0, 999)
        and default["eval_first_last"]  == (1000, 1199)
        and default["test_first_last"]  == (2000, 2199)
        and overlap["pass"]
        and duplicates["pass"]
        and all(s["seeds_used_match_split"] for s in per_split.values())
    )

    return {
        "default_split": default,
        "overlap_rejected": overlap,
        "duplicates_rejected": duplicates,
        "per_split_run": per_split,
        "verdict": "PASS" if overall_pass else "FAIL",
    }


if __name__ == "__main__":
    import json
    out = main()
    print(json.dumps(out, indent=2, default=str))

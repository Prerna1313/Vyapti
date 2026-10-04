"""Replay one frozen checkpoint on fixed VAL or final TEST worlds."""

import argparse
import json
from vyapti_simulator.system_b.tsrd.experiment import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", help="Audited checkpoint filename; default final, or frozen VAL selection for TEST")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--condition", choices=("normal", "beam_periodic", "beam_stochastic", "all"),
                        default="all", help="Held-out illumination stress condition")
    parser.add_argument("--final", action="store_true", help="Required to reveal TEST results")
    parser.add_argument("--action-mode", choices=("greedy", "sampled"), default="greedy",
                        help="Greedy argmax or seeded sampling from the policy action distribution")
    parser.add_argument("--action-seed", type=int, default=20261004,
                        help="Base seed for sampled action choices (results are stored separately)")
    parser.add_argument("--data-root", help="Override the dataset root recorded in config.json")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.split, final=args.final,
                              condition=args.condition, checkpoint=args.checkpoint,
                              action_mode=args.action_mode, action_seed=args.action_seed,
                              data_root=args.data_root), indent=2))


if __name__ == "__main__":
    main()

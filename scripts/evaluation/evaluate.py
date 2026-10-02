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
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.split, final=args.final,
                              condition=args.condition, checkpoint=args.checkpoint), indent=2))


if __name__ == "__main__":
    main()

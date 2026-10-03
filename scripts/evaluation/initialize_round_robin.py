"""Create an audited zero-pretraining Round Robin baseline run."""

import argparse

from vyapti_simulator.system_b.tsrd.experiment import train
from vyapti_simulator.system_b.tsrd.training_setup import resolve_setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/round_robin",
                        help="New algorithm-specific run directory")
    parser.add_argument("--seed", type=int, default=20261003,
                        help="Seed recorded for provenance; Round Robin is deterministic")
    parser.add_argument("--environment", default="training_setup/environments/train250_composed.json")
    parser.add_argument("--algorithm", default="training_setup/algorithms/round_robin.json")
    args = parser.parse_args()
    setup = resolve_setup(args.environment, args.algorithm, seed=args.seed, episodes=0,
                          checkpoint_every=1, execution_mode="online_baseline")
    run = train(setup, args.run)
    print(f"Round Robin online baseline initialized; results will be saved under: {run}")


if __name__ == "__main__":
    main()

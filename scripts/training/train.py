"""Train a policy on fresh TRAIN-250 recorded-PDW worlds."""

import argparse
from vyapti_simulator.tsrd.experiment import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True, help="New runs/<experiment-id> directory")
    args = parser.parse_args()
    print(train(args.config, args.run))


if __name__ == "__main__":
    main()

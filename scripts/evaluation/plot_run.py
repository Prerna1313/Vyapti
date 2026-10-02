"""Regenerate a run's saved learning and evaluation figures."""

import argparse
from vyapti_simulator.system_b.tsrd.experiment import plot_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    for path in plot_run(args.run):
        print(path)


if __name__ == "__main__":
    main()

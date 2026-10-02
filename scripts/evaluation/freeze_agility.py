"""Fit a reproducible agility reference from the TRAIN cache only."""
import argparse

from vyapti_simulator.system_b.tsrd.frequency_agility import fit_train_reference
from vyapti_simulator.system_b.tsrd.experiment import _json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="Data/cache/tsrd/train_250")
    parser.add_argument("--output", required=True)
    parser.add_argument("--channel-width-mhz", type=float, default=500.0)
    parser.add_argument("--expected-configs", type=int, default=250)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    reference = fit_train_reference(args.cache, expected_configs=args.expected_configs,
                                    channel_width_mhz=args.channel_width_mhz)
    _json(output, reference)
    print(f"Frozen TRAIN agility quantiles: {reference['world_rate_quantiles_hz']} Hz; {output}")


if __name__ == "__main__":
    main()

"""Freeze a VAL selection record for a trained checkpoint or fixed baseline."""
import argparse
import json

from vyapti_simulator.system_b.tsrd.checkpoints import freeze_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--checkpoint", help="Audited checkpoint filename for a trained policy")
    choice.add_argument("--baseline", action="store_true", help="Freeze a fixed online algorithm after VAL")
    parser.add_argument("--reason", default="Selected using frozen VAL_NORMAL results")
    args = parser.parse_args()
    print(json.dumps(freeze_selection(args.run, args.checkpoint, baseline=args.baseline,
                                      reason=args.reason), indent=2))


if __name__ == "__main__":
    main()

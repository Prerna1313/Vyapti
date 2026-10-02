"""Freeze one audited checkpoint from completed VAL_NORMAL results."""
import argparse
import json

from vyapti_simulator.system_b.tsrd.checkpoints import freeze_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reason", default="Selected using frozen VAL_NORMAL results")
    args = parser.parse_args()
    print(json.dumps(freeze_selection(args.run, args.checkpoint, reason=args.reason), indent=2))


if __name__ == "__main__":
    main()

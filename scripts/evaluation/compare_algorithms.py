"""Run a 3-seed pilot / 5-seed final comparison, then explicitly reveal TEST."""
import argparse
import json

from vyapti_simulator.system_b.tsrd.algorithm_comparison import run_comparison, finalize_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Train and compare on VAL; freeze choices")
    run.add_argument("--environment", required=True)
    run.add_argument("--algorithms", nargs="+", required=True)
    run.add_argument("--stage", choices=("pilot", "final"), required=True)
    run.add_argument("--episodes", type=int, required=True)
    run.add_argument("--checkpoint-every", type=int, required=True)
    run.add_argument("--output", required=True)
    final = commands.add_parser("finalize-test", help="Reveal TEST once for the frozen final algorithm")
    final.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        report = run_comparison(args.environment, args.algorithms, args.output,
            stage=args.stage, episodes=args.episodes, checkpoint_every=args.checkpoint_every)
    else:
        report = finalize_test(args.output)
    print(json.dumps({"output": args.output, "split": report["split"],
                      "conditions": list(report["conditions"])}))


if __name__ == "__main__":
    main()

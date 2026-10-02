"""Train a policy on fresh TRAIN-250 recorded-PDW worlds."""

import argparse
from vyapti_simulator.system_b.tsrd.experiment import train
from vyapti_simulator.system_b.tsrd.training_setup import resolve_plan, resolve_setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", help="Run plan referencing environment, algorithm, budget, and checkpoint schedule")
    parser.add_argument("--environment", help="Shared environment spec")
    parser.add_argument("--algorithm", help="Algorithm implementation and settings spec")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--save-initial", action="store_true")
    parser.add_argument("--config", help="Legacy resolved config; use --plan or separate specs for new work")
    parser.add_argument("--run", required=True, help="New runs/<experiment-id> directory")
    args = parser.parse_args()
    if args.config:
        if any((args.plan, args.environment, args.algorithm, args.seed is not None, args.episodes is not None,
                args.checkpoint_every is not None, args.save_initial)):
            parser.error("--config cannot be combined with setup options")
        setup = args.config
    elif args.plan:
        if any((args.environment, args.algorithm, args.seed is not None, args.episodes is not None,
                args.checkpoint_every is not None, args.save_initial)):
            parser.error("--plan fixes the setup and cannot be combined with overrides")
        setup = resolve_plan(args.plan)
    elif args.environment and args.algorithm and args.seed is not None and args.episodes is not None and args.checkpoint_every is not None:
        setup = resolve_setup(args.environment, args.algorithm, seed=args.seed, episodes=args.episodes,
                              checkpoint_every=args.checkpoint_every, save_initial=args.save_initial)
    else:
        parser.error("Specify --plan, or --environment --algorithm --seed --episodes --checkpoint-every")
    print(train(setup, args.run))


if __name__ == "__main__":
    main()

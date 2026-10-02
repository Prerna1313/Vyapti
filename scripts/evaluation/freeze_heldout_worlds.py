"""Freeze new held-out world recipes from source labels before training."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def freeze(environment, target, *, seed=20261005):
    path = Path(environment).resolve()
    spec = json.loads(path.read_text())
    evaluation = json.loads((path.parent / spec["evaluation_spec"]).read_text())
    output = Path(target).resolve()
    if output.exists():
        raise FileExistsError(output)
    data = (path.parent / spec["data_root"]).resolve()
    evaluation["world_recipe_seed"] = seed
    evaluation["source_emitter_count_distributions"] = {}
    for split, stream in (("val", 0), ("test", 1)):
        ids = evaluation[f"{split}_config_ids"]
        counts = []
        for config_id in ids:
            with h5py.File(data / "stare" / f"{split}_stare" / f"{config_id}.h5") as source:
                counts.append(len(np.unique(source["labels"][:])))
        feasible = [n for n in counts if 1 <= n <= len(ids)]
        if not feasible:
            raise ValueError(f"No feasible emitter counts for {split}")
        evaluation["source_emitter_count_distributions"][split] = counts
        rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
        evaluation[f"{split}_composed_worlds"] = [
            {"id": f"{split}_world_{i:03d}", "emitter_count": int(rng.choice(feasible)),
             "world_seed": int(rng.integers(1, 2**31 - 1))} for i in range(len(ids))]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--target", required=True, help="New recipe file; never overwrite an existing frozen pool")
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    print(freeze(args.environment, args.target, seed=args.seed))

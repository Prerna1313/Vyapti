"""Freeze hash-only VAL/TEST recipes for deterministic TSRD world replay."""
import argparse
import json

from vyapti_simulator.system_b.tsrd.frozen_world_catalog import freeze_world_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Initialized algorithm-specific run directory")
    args = parser.parse_args()
    catalog = freeze_world_catalog(args.run, progress=lambda message: print(f"[FREEZE] {message}", flush=True))
    print(json.dumps({
        "catalog": f"{args.run}/frozen_world_catalog.json",
        "catalog_sha256": catalog["catalog_sha256"],
        "val_worlds": len(catalog["splits"]["val"]["worlds"]),
        "test_worlds": len(catalog["splits"]["test"]["worlds"]),
        "conditions": list(catalog["conditions"]),
        "reused_catalog": "reused_from_catalog_sha256" in catalog,
        "policy_evaluation_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()

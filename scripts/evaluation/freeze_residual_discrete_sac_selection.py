"""Select one residual-SAC checkpoint by highest frozen VAL_NORMAL OIR."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    target = run / "selection.json"
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads((run / "runtime_manifest.json").read_text(encoding="utf-8"))
    catalog_path = Path(manifest["world_catalog_path"])
    if sha256(catalog_path) != manifest["frozen_world_catalog_sha256"]:
        raise ValueError("The shared frozen world catalog changed since training")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    expected_identities = {
        entry["conditions"]["normal"]["identity_sha256"]
        for entry in catalog["splits"]["val"]["worlds"].values()
    }
    if len(expected_identities) != 50:
        raise ValueError("The shared catalog does not contain exactly 50 VAL_NORMAL identities")
    candidates = []
    for report_path in sorted((run / "eval" / "val" / "normal" / "checkpoints").glob("*/summary.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary = report.get("summary", {})
        if (report.get("split") != "val" or report.get("condition") != "normal"
                or report.get("frozen_world_catalog_sha256") != manifest["frozen_world_catalog_sha256"]
                or summary.get("worlds") != 50
                or summary.get("opportunity_interception_ratio") is None):
            raise ValueError(f"Incomplete or mismatched VAL_NORMAL report: {report_path}")
        rows_path = report_path.parent / "per_world.jsonl"
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != 50 or any(not row.get("frozen_world_identity_sha256") for row in rows):
            raise ValueError(f"Missing frozen-world identity records: {rows_path}")
        identities = {row["frozen_world_identity_sha256"] for row in rows}
        if identities != expected_identities:
            raise ValueError(f"VAL world identities differ from the shared frozen catalog: {rows_path}")
        candidates.append((float(summary["opportunity_interception_ratio"]), report_path, report))
    if not candidates:
        raise FileNotFoundError("Evaluate at least one checkpoint on all 50 VAL_NORMAL worlds first")
    value, report_path, report = max(candidates, key=lambda item: item[0])
    checkpoint_path = run / "checkpoints" / (report_path.parent.name + ".pt")
    # The output folder stem equals the custom checkpoint stem, but generated
    # residual checkpoints use .pt; resolve the actual indexed file explicitly.
    stem = report_path.parent.name
    matches = list((run / "checkpoints").glob(stem + ".pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not resolve uniquely the selected checkpoint {stem}")
    checkpoint_path = matches[0]
    if report.get("checkpoint_sha256") != sha256(checkpoint_path):
        raise ValueError("Selected checkpoint hash does not match its VAL report")
    selection = {
        "schema": "vyapti_residual_sac_selection_v1",
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": sha256(checkpoint_path),
        "selection_metric": "VAL_NORMAL.opportunity_interception_ratio",
        "selection_value": value,
        "validation_summary_sha256": sha256(report_path),
        "frozen_world_catalog_sha256": report["frozen_world_catalog_sha256"],
        "reason": "Highest pooled OIR among the evaluated checkpoints on the same frozen VAL_NORMAL worlds",
    }
    target.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()

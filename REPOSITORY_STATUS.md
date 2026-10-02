# Repository status

Package layout updated: 2026-10-02. The verification entries below are dated.
This page describes this checkout; it is the
starting point for future code reviews. Older `GATE0_COMPLETE.md` and
`SUMMARY_OF_CHANGES.md` are historical notes, not current certifications.

## Verified repository facts

- System B lives in `vyapti_simulator/system_b/tsrd/`; System C lives in `vyapti_simulator/system_c/emitters/`
  and `vyapti_simulator/system_c/rf/`. Shared support stays under `vyapti_simulator/`.
  The previous root `src/` and top-level RF/TSRD import shims have been removed.
  See [System B](vyapti_simulator/system_b/README.md), [System C](vyapti_simulator/system_c/README.md) and
  [the current training setup](training_setup/README.md).
- The local, ignored `Data/stare/` tree retains the selected experimental
  subset: 250 TRAIN, 50 VAL, and 50 TEST source files. `Data/scan/` retains
  29 additional SCAN files. The 6,000-file inventory below describes the
  earlier full corpus, rather than the current retained subset.
  See [TSRD_BASELINE_2026-09-28.md](TSRD_BASELINE_2026-09-28.md) for the
  repository baseline and [TSRD_REPLAY_CONTRACT.md](TSRD_REPLAY_CONTRACT.md)
  for the versioned recorded-PDW replay assumptions. The local content-hash
  inventory is `data_provenance/tsrd_corpus_manifest.json`; its upstream
  download revision remains unproven. The full-corpus gate and anomalies are
  in [TSRD_READINESS_2026-09-28.md](TSRD_READINESS_2026-09-28.md).
- Python tests live in `tests/`. The two small HDF5 integration fixtures and
  their manifest are in `tests/fixtures/tsrd/`. Local sample folders `3/` and
  `4/` are ignored by Git and are not the complete dataset.
- `TSRDCorpusLoader` selects an explicit split directory by default. Legacy
  recursive discovery requires `allow_legacy_layout=True`. With
  `require_manifest=True`, the selected files must match the manifest's
  relative paths, sizes, and SHA-256 hashes.
- `SimulationConfig` validates its basic numeric dimensions and probabilities
  when constructed. `MasterSimulationConfig` records provenance for every
  declared non-map field.

## Documentation entry points

- [README.md](README.md): installation and overview.
- [USER_GUIDE.md](USER_GUIDE.md): current import and pipeline examples.
- [API_REFERENCE.md](API_REFERENCE.md): module index; source code defines the
  actual signatures.
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md): local test commands.
- [PROVENANCE.md](PROVENANCE.md): historical and current rationale; verify
  individual claims against implementation before using them in reports.

## Verification and limits

On 2026-10-02, `python -m pytest -q --tb=short -p no:cacheprovider`
completed with **702 passed**. The Mode-B evaluator now includes TRAIN-only
agility regimes, a privileged expected-OIR scheduler with separate censored
TTFI comparisons, optional prediction scoring, and automatic three-seed pilot /
five-seed final paired reporting. The inventory remains 250 TRAIN / 50 VAL /
50 TEST source files. See [the frozen evaluation protocol](docs/protocols/mode-b-evaluation.md)
for paper-v2 provenance, receiver scope, and sealed TEST commands. Integration
tests use small fixture recordings; they are not algorithm-performance results.
Also on 2026-10-02, all 379 retained HDF5 files in `Data/` matched their
SHA-256 hashes in the historical corpus inventory. The 250/50/50 STARE file
IDs exactly matched the TRAIN-cache and held-out selections. The local report
is `results/tsrd_local_subset_verification.json`. This verifies the retained
subset's identity and integrity; the upstream download revision remains unknown.

On 2026-09-28, `python -m pytest -q --tb=line` completed with **632 passed**
using `h5py` 3.16.0 from a temporary local dependency path. The syntax check
`python -m compileall -q vyapti_simulator src tests` also passed. These are dated checkout results;
repeat the commands after any changes. Passing tests establish only the
cases they exercise; they do not certify scientific validity, dataset
representativeness, or field behavior.

`tests/test_documentation_contract.py` checks maintained Markdown links and
package imports in Python examples. `tests/test_configuration_contract.py`
checks early validation and master-config provenance coverage.

All six original HDF5 split/mode directories were content-hashed, but the
bundled CSV summaries are inconsistent and the exact source revision is
unknown. Do not infer scheduler performance from fixture tests or the earlier
exploratory validation control. A frequency-envelope audit flagged possible
PDW-label to transmitter-metadata mismatches; see the readiness report for
split-level counts and interpretation. Test HDF5 files have been opened for
inventory and data-quality checks only, with no scheduler run on test.

## Remaining documentation maintenance

Some architecture and provenance prose describes intended behavior. Review
it alongside the implementing module and a reproducible test before citing
it as a verified result. Root-level `fix_*.py` and `patch_*.py` files are
unreviewed historical helpers; do not run or delete them based on their
names alone.

# Repository status

Last reviewed: 2026-09-26. This page describes this checkout; it is the
starting point for future code reviews. Older `GATE0_COMPLETE.md` and
`SUMMARY_OF_CHANGES.md` are historical notes, not current certifications.

## Verified repository facts

- The packaged source is under `vyapti_simulator/` and `src/`.
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

On 2026-09-26, `python -m pytest -q` completed with **583 passed and 13
legacy-layout deprecation warnings**. The syntax check
`python -m compileall -q vyapti_simulator src tests` also passed. These are dated checkout results;
repeat the commands after any changes. Passing tests establish only the
cases they exercise; they do not certify scientific validity, dataset
representativeness, or field behavior.

`tests/test_documentation_contract.py` checks maintained Markdown links and
package imports in Python examples. `tests/test_configuration_contract.py`
checks early validation and master-config provenance coverage.

The complete train and validation datasets have not been downloaded into
this workspace. Do not infer full-corpus results from the local fixtures or
the sample files in `3/` and `4/`.

## Remaining documentation maintenance

Some architecture and provenance prose describes intended behavior. Review
it alongside the implementing module and a reproducible test before citing
it as a verified result. Root-level `fix_*.py` and `patch_*.py` files are
unreviewed historical helpers; do not run or delete them based on their
names alone.

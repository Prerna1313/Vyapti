# Vyapti / TSRD baseline — 2026-09-28

This is a repository and local-data baseline, not a benchmark result. The
checkout is `main` at `c253843`. No source implementation was changed during
this review.

## Repository shape

- 149 tracked files: 70 under `vyapti_simulator/`, 41 under `tests/`, 15 under
  `scripts/`, 4 under `src/`, and two tracked SAC files under `final_results/`
  that are currently deleted in the working tree.
- `core/` supplies the simulation configuration, receiver, scheduler interface,
  episode loop, mapping, and metrics. `rf/` and `src/` supply the waveform and
  synthetic RF paths. `tsrd/` supplies HDF5 ingestion, discretisation,
  recorded-PDW replay, paired-scan validation, world sampling, and scoring.
  `algorithms/` contains reference bandit and periodicity implementations;
  `experiments/`, `protocol/`, and `qualification/` provide comparisons and
  checks. `ml/env.py` is a Gymnasium wrapper for the synthetic core environment.
- `TSRDBenchmarkProtocol` accepts external `train_candidate` and
  `load_checkpoint` callbacks. The repository does not currently provide a
  complete, checked-in TSRD-trained ML scheduler and end-to-end training run.

## Local TSRD corpus

The local `dataset/` directory contains exactly 6,000 HDF5 files, totalling
65,792,130,658 bytes (61.274 GiB). Each scan/stare mode has 2,500 train, 250
validation, and 250 test files. Each split has an uninterrupted sequence of
`config_*.h5` identifiers, and every scan filename has an exact-name stare
companion. All 6,000 files are nonempty and have the HDF5 file signature.
This verifies file inventory and headers only; it does not verify their HDF5
contents or source revision.

No `dataset/corpus_manifest.json` exists, so the loader's full-corpus SHA-256
verification cannot be enabled as documented. The separate, historical
`data_provenance/manifest.json` points to
`vyapti_simulator/data/tsrd_statistics.json`, which is absent here. The local
`dataset/` tree is also absent from `.gitignore`, despite the comment about
keeping large datasets out of Git.

The bundled `data_stats.csv` and `data_stats_global.csv` files are not a
reliable inventory of this local tree. For example, scan/train has 2,500 HDF5
files but 3,245 per-file CSV rows, only 2,494 distinct IDs, and a global CSV
claim of 3,255 trains. Stare/train has 2,500 HDF5 files but 1,384 per-file CSV
rows and a global claim of 1,387 trains. Validation CSVs are likewise
inconsistent; both test split CSVs have 250 rows and agree with their global
counts. Do not use these CSVs as corpus truth until rebuilt and checked against
the HDF5 files.

HDF5 dataset shapes, checked across every train and validation file, give
233,172,417 scan/train PDWs, 3,172,039,637 stare/train PDWs, 22,693,747
scan/validation PDWs, and 316,743,173 stare/validation PDWs. These totals
match the rounded figures in the upstream challenge documentation. Eight train
files in each receiver mode have zero recorded PDWs; validation has none.

The [official dataset card](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)
describes 6,000 simulated pulse trains and five PDW measurements per pulse.
The [upstream challenge documentation](https://github.com/alan-turing-institute/turing-deinterleaving-challenge#data-description)
describes scan and stare as paired configurations with separately observed
streams; labels are only meaningful within a pulse train. TSRD was designed
principally for pulse deinterleaving. A scheduler experiment built on its
recorded stare PDWs therefore measures an approximation of scan opportunities,
not complete physical emission truth.

## Current TSRD path and scientific limits

1. `corpus_loader.py` discovers explicit split directories and pairs matching
   files. `tsrd_adapter.py` reads five PDW fields plus file-local labels; its
   real-data mode does not silently substitute synthetic data.
2. `TSRDTrainWorldPool` indexes labels from train stare recordings and resamples
   full recorded histories into training worlds. It reads full source HDF5
   arrays when composing a world, so full-corpus runtime and memory need a
   measured check before a large training run.
3. `TSRDStareEnvironment.from_stare_mode` builds a 50 ms band/slot replay using
   tune centres from the paired scan metadata. The default `binary_v1` receiver
   applies a synthetic Bernoulli detector to recorded opportunities;
   `pdw_v2` adds a recorded-PDW-derived measurement. Retune time is excluded
   from listening. A 100 ms mixed dwell is modeled as two 50 ms looks.
4. `TSRDBenchmarkProtocol` trains on train worlds, selects checkpoints on
   validation, and tests the selected checkpoint once. The scorecard reports
   observed conditional detection and false-alarm rates, interception proxies,
   time to first intercept with censoring, revisit coverage, and rewards.
   Forecast metrics are available only if a scheduler actually supplies
   predictions. Physical pulse interception remains unavailable from this
   replay. File inventory in the run manifest uses paths, sizes, and mtimes,
   not source file hashes.
5. `paired_scan_validation.py` and `audit_tsrd_metadata.py` provide development
   checks for schedule geometry and possible label/metadata inconsistencies.

## Development-corpus validation

The HDF5 metadata and array-shape inventory inspected all 2,750 train and
validation scan/stare pairs. All pairs had the expected five PDW columns and
matching label lengths, 30-second receiver collections, 36 scan tune centres,
and consistent scan/stare duration, bandwidth, and feature order. No HDF5
read or schema errors were found. The recorded scan schedule has 36 dwell
durations per cycle: 29 at 50 ms and seven at 100 ms. Nine selected pairs were
checked with `validate_pair`; all had matching receiver/transmitter metadata
and passed its 0.99 scheduled-passband gate. Their scan/stare dwell overlap is
descriptive, not physical detector probability.

The full frequency-envelope diagnostic checked all 2,500 train and 250
validation stare files using `audit_file` defaults. It flagged possible
PDW-label/transmitter-metadata joins in **458 train files (18.32%)** and zero
validation files. It found no missing nominal-frequency labels, schema issues,
or read errors. These flags do not prove a specific correction and do not
invalidate file-local PDW labels used by the replay world pool. They do make
transmitter metadata unsafe to join blindly to those labels for training.
Reports are in `results/tsrd_baseline/development_metadata_inventory.json`,
`development_sample_audit.json`, and `development_label_frequency_full.json`.
The test HDF5 split was not opened in these audits.

A one-file smoke check on validation `config_0.h5` ran 600 fixed 50 ms replay
slots at configured `Pd=0.9`, `Pfa=0.05`, and 1 ms retune. Round-robin and
reference UCB1 both completed and produced scorecards; the result is recorded
in `results/tsrd_baseline/val_config_0_smoke.json`. This checks integration,
not comparative scheduler performance or model selection.

The reproducible validation-only control in
`scripts/tsrd_reference_validation.py` then completed all 250 validation
pairs, with identical per-file receiver seeds for both policies and the same
receiver settings. Its report is
`results/tsrd_baseline/validation_reference_full.json` (342 seconds). This
run predates the full-corpus readiness gate and is retained as an exploratory
integration artifact, not an admissible scheduler comparison. It does not
establish physical pulse capture, a trained ML result, or performance on the
held-out test split.

## Verification in this environment

- `python -m compileall -q vyapti_simulator src tests`: passed.
- Focused core tests (`configuration_contract`, `mapping`, `metrics`,
  `multi_objective_reward`, `threat`): 140 passed.
- Full `python -m pytest -q --tb=line` with a temporary local `h5py` path:
  **628 passed**. The active Python environment does not normally have
  `h5py`; the dependency was installed temporarily for this review.
- `pyproject.toml` declares `h5py`. Code also imports `pydantic` and `psutil`
  while the package does not declare them as dependencies; both happen to be
  installed in the active environment.
- The temporary `.baseline_deps/` directory containing `h5py` remains in the
  workspace and is ignored by Git. Approval review rejected its cleanup.

The dated [`REPOSITORY_STATUS.md`](../../REPOSITORY_STATUS.md) statement that the train and validation
corpora are absent is stale for this checkout. Its earlier 583-test result is a
historical result and was not reproduced here.

## Next TSRD checks

1. Pin the source revision and verify file hashes before treating this corpus
   as a reproducible benchmark input. Rebuild or disregard the bundled CSVs.
2. Review the 458 diagnostic train files before using transmitter metadata as
   supervised labels. File-local PDW labels may still be used for replay.
3. Freeze receiver assumptions and metrics before candidate training. Keep
   scan-recording evaluation and stare-based counterfactual replay separate.
4. Train and compare an adaptive scheduler under the same split and
   observation contract. Keep test data untouched until selection is frozen.

No JC Wise dataset ingestion or evidence is present in the checked-in code.

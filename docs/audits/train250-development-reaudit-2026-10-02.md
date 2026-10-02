# TRAIN-250 development re-audit

This report supersedes the earlier audit's **local inventory and active training-script assumptions**. It does not supersede the earlier physics findings. The user intentionally supplied a small development pool for comparing schedulers and modes; final experiments will use larger/full pools. A small development pool is appropriate and is not itself a defect.

No raw data, cache, training implementation or folder placement was changed. The audit added this report, an inspection script and generated verification evidence. Existing working-tree deletions were left untouched.

## Verified data

- `Data/stare/train_stare`: exactly 250 H5 files.
- `Data/stare/val_stare`: exactly 50 H5 files.
- `Data/stare/test_stare`: exactly 50 H5 files.
- `runs/tsrd_cache/train_250/configs`: exactly 250 NPZ files. The actual location is `tsrd_cache`, not `train_cache`.
- The cache manifest, selection list, emitter index, summaries, raw TRAIN filenames and NPZ filenames all agree on the selected 250 configurations. No missing or extra training configs were found.
- Exactly **8,879 emitter contributions** and **290,213,219 pulses** were compared. Every cached ToA, frequency, pulse width, AoA and amplitude value matches the original H5 emitter history. Counts and label sets also match. No nonfinite PDW values or nonchronological emitter sequences were found.
- All **379 local H5 files**, including the 29 SCAN samples, were hashed. All hashes match their corresponding entries in the earlier local corpus manifest. No identical whole-file hashes cross train/val/test splits. This verifies continuity with that inventory, not the still-unpinned upstream dataset revision or absence of every possible statistical overlap.
- The full TRAIN catalog contains 2,500 configurations, including eight empty ones. It is metadata for the larger corpus, not evidence that all those H5 files are locally present.
- The cache preserves **200,242 pulses outside the nominal 30-second collection window**. These are source characteristics, not an H5-to-NPZ corruption. Keep raw histories immutable; define replay as a documented interval such as `[0, 30 seconds)` and log excluded pulses and their denominator treatment. Do not silently rescale or wrap their times.

Reproduce the data audit with `python docs/audits/inspect_train250.py`. Full results and per-file hashes are in [train250-data-verification.json](train250-data-verification.json). The original checker briefly treated the valid metadata spelling `Scanning` as different from the directory name `scan`; the checker was corrected and all 379 metadata modes rechecked. Final data errors: zero. The full pulse comparison took about 202 seconds.

## What must change before new scheduler comparisons

### 1. Provide one working training/evaluation route for this cache

The previous `scripts/training/` and `scripts/evaluation/` directories are absent from the current working tree; Git reports their former files as deleted. Their old TRAIN-500 assumptions should no longer be described as the active workflow. No replacement TRAIN-250 NPZ runtime loader or RL training entry point was found in the inspected package/scripts.

The native H5 `TSRDTrainWorldPool` still exists and is independent of the old fixed 500-config cache loader. A generic cache loader should read the new schema, validate the selection and content hashes, and accept pool sizes from configuration. Do not recreate separate `train250`, `train500`, `train1000` copies of the same algorithms.

The new index uses fields such as `source_label` and `npz_file`; its UID is `config_12::emitter::0`. This differs from the previous cache's index structure despite the shared `vyapti_tsrd_emitter_pool_v1` schema name. Document/version the index schema and validate supported fields. Qualify identities with dataset revision, original split, mode and config; local emitter labels and config numbers alone are not globally unique.

The legacy `vyapti_simulator/ml/env.py` is not a replacement: constructing `VyaptiRFEnv()` failed with `TypeError: argument of type 'SimulationConfig' is not iterable`. It passes a SimulationConfig where ScenarioRegistry expects a mapping; it also calls the underlying environment without its required emitter configuration. Use a tested Gym wrapper around the chosen recorded-PDW environment if RL libraries need one.

### 2. Repair the benchmark geometry interface

Rerun of the receiver/benchmark tests: **36 passed, 4 failed in 8.88 seconds**:

```powershell
python -m pytest -q -p no:cacheprovider --tb=short tests/test_tsrd_benchmark_contract.py tests/test_tsrd_benchmark_protocol.py
```

The four failures still arise from `passband_halfwidth_mhz` being passed by [benchmark_protocol.py](../../vyapti_simulator/tsrd/benchmark_protocol.py#L199) to a `TSRDTrainWorldPool` constructor that does not accept it. The evaluation `from_stare_mode` factory also does not accept that keyword. Align the contract across both paths. Do not mask the error by allowing arbitrary ignored keyword arguments.

Existing direct H5 replay works when constructed with explicit geometry; a smoke check created a 36-band, 600-slot binary receiver and stepped it successfully. The native train pool indexed all 8,879 emitters. Two three-emitter samples at seed 12345 had equal world signatures, and their first 12 receiver looks matched after resetting both with seed 22. That does not certify the broken end-to-end benchmark route or an NPZ runtime backend that is not yet present.

### 3. Correct stale cached H5 paths

All **8,879** emitter-index `stare_file` references point to:

```text
stare/train_stare_250/config_*.h5
```

Actual source paths relative to `Data` are:

```text
stare/train_stare/config_*.h5
```

All relative NPZ references resolve correctly. Thus the cache payload is sound but source lookup is stale. Prefer storing immutable source identity separately from a configurable filesystem root. Correct the provenance reference once, rather than renaming raw directories merely to satisfy stale export paths.

### 4. Make STARE replay independent of paired SCAN availability

There are only 10/9/10 SCAN files across train/val/test. Matching STARE/SCAN filename pairs number **1/1/4**. These are receiver-validation samples, not three complete paired corpora.

The default `iter_tsr_replay_pairs` fails for each split when it encounters a missing companion. With `allow_stare_only=True`, it enumerates 250/50/50 successfully. Its caller must supply explicit validated receiver geometry.

All inspected source receiver metadata agree on a 30-second duration and `bandwith_mhz=500`. The 29 SCAN samples agree on 36 centres, `250 + 500*b MHz` for `b=0..35`; STARE has empty centre arrays. The current replay convention treats 500 MHz as **halfwidth**, so total instantaneous bandwidth is 1,000 MHz and adjacent 500-MHz-spaced passbands overlap. Freeze centres, width meaning and boundary rules in a receiver configuration. Do not reinterpret the bandwidth while reorganizing folders.

You do not need to download 350 SCAN companions merely to compare STARE schedulers. Keep paired-SCAN validation separate and state the sample coverage.

### 5. Freeze selection manifests for all development roles

TRAIN-250 is selected by multivariate quantile-normalized KMeans centroid selection, seed 20260906, over non-empty TRAIN scenarios. The selected list is reproducible as a fixed list, but the selection/preprocessing implementation, exact feature list, transforms, cluster settings and library versions were not found in the current scripts. Save those if rebuilding the selection matters.

No equivalent saved selection-rule manifests were found for the 50 VAL or 50 TEST files. Record their exact split-qualified config IDs, content hashes, selection seeds/rules and experiment roles. Do not depend on sorted directory globbing or the first 50 filenames; adding full-corpus files later must not silently change a development experiment.

Centroid selection is a useful representative subset, not an unbiased random sample. From the supplied catalog, selected emitter-count median is 34.5 versus 37 among full non-empty TRAIN configs; selected pulse-count median is 1,028,710.5 versus 1,115,259. These are descriptive checks, not proof of representativeness on beam/PRI/agility axes. Report the sampling design and retain a few empty/sparse cases as separate sanity tests.

Source file hashes now have a fresh local audit, but the upstream dataset revision remains unpinned. The supplied SHA256SUMS files identify ZIP archives; they do not by themselves validate extracted files when the ZIP archives are absent. Retain archive hashes and the newly verified per-file hashes with explicit roles.

### 6. Protect the meaning of TEST during development

Use TRAIN-250 for learning and TRAIN-only calibration. Use VAL-50 for checkpoint selection, scheduler comparison and hyperparameter/mode decisions. Keep TEST-50 for a frozen development holdout if it has not influenced choices.

If TEST-50 results are repeatedly used to choose a scheduler, those 50 source files have become development data. Changing the local folder name does not undo this. Reserve untouched evaluation sources for the final claim; the other 200 official TEST configurations can be a clean held-out set if they remain unused. A later score over all 250 may still be reported descriptively, but it is not fully untouched if 50 influenced development. Merely auditing file integrity does not itself use scheduler performance for selection.

The user's plan to use the full pool later is compatible with this workflow. Training can expand to all available original TRAIN configurations; VAL and TEST must remain separate. Do not merge all three original splits into one training pool.

## Keep the experiment axes explicit

H5 versus NPZ is a storage/backend choice, not a physics mode. The equality check now provides strong evidence that the two contain the same source histories.

Treat these as independent settings:

- **World construction:** original source replay (Mode A) versus compositions of whole emitter histories (Mode B).
- **Receiver profile:** binary replay versus PDW measurements, with a fixed detection/false-alarm/retune contract within a comparison.
- **Physics augmentation:** no added beam process versus explicit periodic/aperiodic illumination, when implemented.
- **Scheduler:** baseline, bandit or learned policy.

For a modest first comparison, freeze the binary receiver contract already in the cache manifest: Pd 0.9, Pfa 0.05, retune 1 ms. Keep reward, observation features, dwell options, mission horizon and compute budget equal across schedulers. Do not compare a binary agent and an SNR-rich agent as though only the scheduler changed.

The native world pool is still train-only. For held-out Mode-B comparisons, add a split-aware pool/registry so composed VAL worlds use VAL histories and composed TEST worlds use TEST histories. Freeze evaluation recipes/seeds separately from training. Evaluate the same frozen policy on source-replay and composed-world suites and report them separately.

Use simple baselines first, then a small number of independent training seeds for promising policies. Pair each scheduler on the same evaluation recipes and receiver-noise seeds. Separate variance across training seeds from variance across worlds. This audit did not train policies or rank scheduler performance.

## Physics gaps that the new dataset does not fix

The prior findings still apply to the main STARE composer/receiver:

- No explicit separate hidden illumination/beam process.
- No controlled emitter-period × receiver-revisit synchronization sweep.
- No complete A–G joint frequency/spatial stress runner or validated CTMC beam process.
- No complete persisted world/receiver/agent/evaluator episode logs.
- Binary emitter attribution is an explicitly optimistic upper bound; the scorecard's illumination ratio is recorded emitter-slot interception, not proof of beam physics.
- Main replay timing is 50 ms slots; synchronization experiments need event-time overlap or convergence checks.

These need not all block an honestly labelled **recorded-PDW scheduler development benchmark**. They must be implemented before claiming the full physics/theory validation from the proposed architecture. Keep such a baseline intact while adding a separately versioned augmented profile. Do not simulate extra free-space loss on already received amplitudes without addressing double-counting.

## Recommended folder structure

Move according to ownership and lifecycle, not model names or sample counts. Keep one raw corpus tree and select subsets through manifests. The following is a proposed layout, **not a performed migration**:

```text
Vyapti/
  configs/
    experiments/
      dev250-v1.toml
      final-v1.toml
    receivers/
      binary-v1.toml
  data_provenance/
    datasets/                  # dataset revision and file hashes
    pools/
      dev250-v1.json            # exact TRAIN/VAL/TEST identities and roles
      final-v1.json
  Data/
    raw/tsrd/
      stare/{train_stare,val_stare,test_stare}/
      scan/{train_scan,val_scan,test_scan}/
    cache/tsrd/
      <pool-id>/<schema-version>/{train,val,test}/
        manifest.json
        emitter_index.json
        config_summaries.json
        configs/*.npz
  vyapti_simulator/
    tsrd/                      # source reader, generic emitter pool/cache, composer
    core/                      # receiver/observation/reward/truth contracts
    algorithms/                # reusable scheduler and policy implementations
    ml/                        # working RL wrapper and training utilities
    rf/                        # existing RF physics utilities
    experiments/               # shared train/eval, synchronization, stress runners
    visualization/             # figures generated from saved results
  scripts/
    data/                      # select, build cache, validate, freeze manifests
    training/                  # thin train.py entry point
    evaluation/                # thin evaluate.py / compare.py entry points
  runs/
    <experiment-id>/<scheduler>/<seed>/
      manifest.json
      checkpoints/
      logs/
      metrics/
      figures/
  docs/
    architecture/
    protocols/
    audits/
  archive/legacy_scripts/      # preserve superseded patches if no longer needed
  api/
  frontend/
  tests/fixtures/              # small committed test data, not research pools
```

The important move is **cache out of `runs`**: a shared immutable input cache is not the output of one scheduler run. Keep `runs` for outputs only. You can initially leave `Data/stare` and `Data/scan` in place and introduce configuration/manifests first, avoiding a disruptive all-at-once move.

Recommended code organization is incremental. Add a generic pool backend in the existing `tsrd` package, reuse `TSRDStareEnvironment` and `replay_scorecard`, and keep training orchestration separate from algorithm definitions. Do not create another duplicate simulator or move all existing `core`/`src` code during the data-layout change.

### Path and repository hygiene

- `api/tsrd_data.py:30` walks four parent directories and looks for `database/scan/train_scan`, not the supplied `Data` root. Give API, CLI and tests the same configurable data-root resolver. It is a separate dashboard path, not proof of scheduler benchmark integration.
- `.gitignore` ends in NUL-interleaved bytes spelling `runs/`; the ignore entry is malformed. `git check-ignore` did not match the current H5 or NPZ paths. Normalize the file encoding and add explicit raw-data/cache/run-output ignores before staging.
- The existing broad `*.json` and `data_provenance/*` exclusions need deliberate exceptions for tracked dataset/pool manifests. Keep the small committed test fixtures trackable.
- Centralize root paths; eliminate embedded `/content/`, `/kaggle/`, historical `database/` and per-user paths from reusable runtime code. Environment-specific launch configuration can still point to those locations.
- Keep one authoritative architecture/protocol document. Label the older audit and historical experiment summaries rather than treating multiple root-level reports as simultaneous current specifications.
- Root `patch_*.py`, plotting helpers and debug files should only move into an archive or appropriate scripts directory after checking their users. Preserve their history and do not rerun them as part of normal setup.

## Concrete migration order

1. Save the pool ID lists and hashes; assign `dev250-v1` without changing source bytes. Define TEST use and exact receiver geometry.
2. Fix the benchmark argument mismatch and provide a schema-aware generic NPZ backend plus a tested RL wrapper/trainer entry point if RL is required.
3. Fix stale source references through explicit identity/root mapping; verify H5-backed and NPZ-backed world/receiver equivalence, seeded reproducibility and policy isolation.
4. Put shared data/cache/run roots behind configuration and validate a full 30-second baseline episode through the shared runner.
5. Move raw/cache files once, validating hashes before and after. Update consumers/manifests in that same change; do not leave two competing authoritative copies. Do not delete originals until the destination is verified.
6. Compare schedulers on frozen VAL source-replay/compositional suites; record recipes, hashes, code revision, receiver settings, metrics and independent training seeds.
7. Add illumination/synchronization/stress experiments for the stronger physics claims. Expand to final source pools by changing manifests/configuration, not duplicating scripts.

**Decision:** the supplied development data and cache are internally consistent. Keep this small pool. Fix the runtime and provenance wiring, simplify folder ownership, and separate development selection from final held-out evaluation before launching new model comparisons.

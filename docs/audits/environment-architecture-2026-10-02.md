# Vyapti environment architecture audit

Audit date: 2 October 2026. Inspected checkout: `f6ee95446daa2dbd1ab6c590c470bf16273a5cfd`, including the existing working-tree files. This is an audit, not an implementation or a declaration that the proposed protocol is frozen.

## Verdict

**Partially implemented. The current training system is a STARE recorded-PDW composition benchmark, not yet the full hidden frequency-plus-spatial-illumination environment described in the request.** Preserve this useful replay baseline and introduce a separately versioned augmented-world profile. Do not relabel existing results as physical beam-interception results.

The main gaps are an explicit illumination process on the training path, controlled synchronization and A–G experiment families, held-out compositional evaluation, complete episode audit logs, and source-revision provenance. A confirmed interface regression currently breaks the general benchmark protocol.

## Evidence and verification

Selected regression command:

```powershell
python -m pytest -q -p no:cacheprovider --tb=short tests/test_tsrd_benchmark_contract.py tests/test_tsrd_benchmark_protocol.py tests/test_tsrd_corpus_loader.py tests/test_unified_observation_contract.py tests/test_data_integrity.py
```

Result with working temporary-directory access: **80 passed, 4 failed, 32.71 seconds**. Initial sandbox runs encountered Windows temporary-directory permission errors; those are not counted as implementation defects. This was a targeted suite, not the entire repository suite.

Additional live checks against the local `Data` STARE subset:

- The native train pool indexed 415 emitter contributions from 11 configurations.
- Three-emitter worlds with seed 12345 had identical signatures; seed 12346 produced a different signature.
- All five PDW columns matched the selected source rows exactly, after stable ToA ordering: config_1/label 39 (1,180 rows), config_8/label 21 (407 rows), config_6/label 69 (7,232 rows).
- The first 12 round-robin observations were identical after identical receiver resets.
- These spot checks do not establish preservation for all 606 million cached pulses or all possible seeds.

Local inventory: STARE train/val/test each contain 11 H5 files under `Data`; SCAN contains 10/9/10. The TRAIN-500 cache contains 500 NPZ files. Its manifest reports 17,826 emitter contributions and 606,499,545 raw PDWs. The older corpus manifest describes 6,000 H5 files under a different root, `D:\Vyapti\dataset`; it is historical evidence, not proof that the full corpus is currently available here. Comparing hashes listed in that manifest found zero identical-file hash groups crossing splits; the files were not all rehashed during this audit.

## Priority findings

### P1 — General benchmark protocol currently fails

[benchmark_protocol.py](../../vyapti_simulator/tsrd/benchmark_protocol.py#L199) places `passband_halfwidth_mhz` in `receiver_options`, then expands it into `TSRDTrainWorldPool` at line 225. [world_pool.py](../../vyapti_simulator/tsrd/world_pool.py#L27) does not accept that argument. This causes all four observed failures:

- `test_stare_only_requires_explicit_geometry_and_audits_metadata`
- `test_pdw_receiver_profile_flows_through_train_and_evaluation`
- `test_train_pool_and_frozen_split_protocol`
- `test_checkpoint_protocol_mixed_dwell_profile`

The evaluation call at benchmark_protocol.py:248 also expands this dictionary into `from_stare_mode`, whose signature at tsrd_environment.py:1262 likewise lacks this parameter. Resolve the geometry contract across both construction paths, rather than patching only the first exception. This finding does not establish that historical TRAIN-500 results failed: those scripts use a separate construction path.

### P1 — Hidden illumination is absent from the STARE composition path

The actual flow is cached/native per-emitter PDWs → concatenation → recorded occupancy → `TSRDStareEnvironment` → causal receiver observations. See [world_pool.py](../../vyapti_simulator/tsrd/world_pool.py#L83), [cache adapter](../../scripts/training/vyapti_train500_cache.py#L473), and [receiver step](../../vyapti_simulator/tsrd/tsrd_environment.py#L1523).

The training cache defaults to `binary_v1`: eligibility depends on pulse timing and frequency, followed by configured Pd/Pfa. AoA and amplitude are preserved but do not affect this binary detector. There is no independently represented beam state, visibility trajectory, or SNR time series in this composer.

The opt-in `pdw_v2` receiver at tsrd_environment.py:1450 already supports amplitude-dependent pulse capture and measured PDWs. This is valuable but is not a beam process, calibrated propagation model, or added measurement-error model: successful PDW fields are copied from the source recording. Source amplitudes are labelled dB in the paper; do not assume the cache's `dbm` field names establish absolute calibration.

Synthetic `ScanningEmitter` and `FrequencyAgileScanningEmitter` exist in [src/emitter_models.py](../../src/emitter_models.py#L485), including periodic visibility windows. Their generation gates away off-beam pulses, so that stream alone cannot distinguish all transmissions from receiver illumination. The separate `rf/` models are not evidence of their use in TRAIN-500.

**Recommendation:** add evaluator-only illumination trajectories between world composition and receiver eligibility/detection. Preserve a replay mode with no additional attenuation. Treat beam overlays on STARE as research augmentation because STARE already represents received, censored events; applying another propagation loss can double-count attenuation. Fully separating emitted versus received truth requires a synthetic transmission source, not recovery of missing emissions from recorded PDWs.

### P1 — Metrics must retain their physical meaning

[replay_scorecard.py](../../vyapti_simulator/tsrd/replay_scorecard.py#L61) derives emitter opportunities from recorded occupancy. Its `illumination_interception_ratio` is recorded emitter-slot interception, not a measurement of an independent beam state.

For `binary_v1`, one positive band observation credits every eligible emitter in that band/window. The code explicitly labels this `upper_bound_from_band_positive`; retain that qualification in figures and comparison tables. `pdw_v2` provides captured-PDW attribution, but remains recorded-source truth.

TTFI uses 50 ms slot boundaries, including `emitter_first_intercept_ms`, rather than the exact captured pulse ToA. This is acceptable if labelled slot-resolution timing. It is insufficient for sub-slot synchronization validation; expose physical first-capture time and distinguish it from decision/report time. The scorecard already handles censoring with Kaplan–Meier quantities and restricted mean time—reuse that strength.

### P2 — Requested experimental protocol is not frozen in code

[causal_harness.py](../../scripts/training/causal_harness.py#L66) specifies TRAIN corpus 2500, VAL evaluation 100, TEST evaluation 100. The cache loader requires exactly 500 selected train configs. Saved GRU validation registry and LSTM final test output each contain 100 recordings.

Evaluation in [benchmark_protocol.py](../../vyapti_simulator/tsrd/benchmark_protocol.py#L235) and the training scripts replays original held-out recordings. The pool implementation is train-only. Therefore full-250 held-out compositional Mode-B evaluation is missing; changing a replay file limit alone will not add it. The generic protocol can iterate all available replay files once its interface regression is fixed.

Choose and version TRAIN-250 or TRAIN-500. The pasted 250/50/50 development split is not the current 500/100/100 setup. Define DEV-EVAL explicitly from development data, not the final TEST split. Keep previous experiments under their original protocol names.

### P2 — Dataset revision and cache integrity are incomplete

[tsrd_corpus_manifest.json](../../data_provenance/tsrd_corpus_manifest.json#L6) records `dataset_revision: null` and `unknown_local_download_revision`. A paper labelled v2 does not prove that these downloaded bytes are dataset v2.

[fingerprint_cache](../../scripts/training/vyapti_train500_cache.py#L75) hashes schema, selection and emitter identities, not the NPZ payloads. It cannot detect an edited CF sequence in a cache file with the same emitter IDs. Add per-source and per-cache content hashes, dataset repository/revision, extraction version and generator version. Use `(revision, split, mode, source_config_id, local_label)` as identity; config names legitimately recur across splits.

No evidence of cross-split pool contamination was found in the inspected train-only selection path. That is narrower than an exhaustive raw/cached content-level leakage audit.

### P2 — Episode logging is incomplete

The main GRU loop [discards `_sources`](../../scripts/training/vyapti_ppo_gru_500pool.py#L700). Observations, log probabilities, value estimates and hidden states exist in rollout memory, but that does not constitute persisted episode audit logs. Saved manifests, checkpoints and aggregate validation rows are useful, but not the four complete logs requested.

Persist separate streams with episode/step IDs:

- World: world seed, receiver seed, source identities, content hashes, generator version, transforms and references to pulse/illumination truth. Store a deterministic recipe plus immutable source references rather than duplicating every source pulse in JSON.
- Receiver: action, time window, passband, retune, detector configuration, hit, measured count/PDWs and measured noise information actually supplied by the receiver.
- Agent: pre-action observation/features, action, log probability, value where applicable, latent/belief state, uncertainty, reward, terminated/truncated flags. Use null for quantities an algorithm does not have.
- Evaluator: true opportunities, captures, misses, false alarms, first-capture timing, denominator definitions, censoring, and metric-contract version.

Keep world/evaluator channels inaccessible to policy inputs. Logging hidden state must not make it an observation.

## Checklist assessment

### Source and pool

- Present: split directories, STARE/SCAN separation, train-only emitter extraction, original local labels, source paths, cache UIDs, full five-column recorded histories.
- Partial: immutable provenance and cross-split assertions; no pinned upstream revision, no payload-sensitive cache fingerprint.
- Missing from the pool record: comprehensive per-realization hop count, dwell/revisit statistics, hop span, transition summaries, PRI variation and AoA/amplitude summaries. Metadata/population extraction utilities exist, but do not supply this complete trajectory-statistics contract.

### World and emitter modes

- Present: variable emitter count; the training helper samples the empirical per-config emitter-count distribution; seeded selection; stable merged ToA ordering; preserved native CF/PW/PRI/AoA/amplitude histories.
- Missing: explicit placement offsets/cropping recipes, independently modelled illumination in composed worlds, typed mode classification for each pooled emitter, and a common emitted/received/eligible truth interface.
- Native recorded traces can contain variable frequency and PRI; preserving them is not the same as controllably selecting/calibrating all six requested modes.
- Synthetic fixed, frequency-agile, PRI-jitter, periodic scanning and joint frequency/scanning building blocks exist. `JITTERED_PERIODIC` uses uniform period perturbations; `MARKOV_HOPPER` changes frequency with a discrete per-slot probability. Neither establishes a CTMC beam process.
- The scenario registry has a default 50/30/20 mixture. Label such choices as stress distributions; they are not automatically TSRD-calibrated mode proportions. The registry's top-level hash is also explicitly a placeholder, despite real per-scenario hash support.

### Receiver, observations and truth

- Present: action-conditioned selected band, known receiver geometry, retune-window loss, 50/100 ms dwell composition, configured detection and false alarms, recorded pulse truth and emitter-slot truth.
- Present: observation whitelist and nested PDW checks, causal belief/periodicity features, hidden labels excluded from ordinary observations. Selected tests cover these contracts.
- Partial: pulse measurements through opt-in `pdw_v2`; the main binary path supplies hit/miss rather than measured CF/PW/AoA/amplitude. Do not require a physically binary receiver to expose invented measurements.
- Missing on the main path: beam visibility, calibrated received SNR, explicit detector-threshold/noise telemetry, additional PDW measurement errors, emitted-pulse truth.
- Reward: current GRU training uses detector-positive reward, so false alarms also receive positive reward. A false-alarm-aware action-outcome reward already exists. Neither implements the entire proposed miss/retune/reward decomposition. Freeze reward choices and report ablations.

### Three required automated assertions

- Leakage: useful whitelist and nested-field tests already exist. Strengthen with interventions on labels, future pulses and off-band truth, holding independent receiver noise fixed. `reset()` currently keys noise with a signature containing all PDWs and labels; changing future truth therefore changes the random field. That is a counterfactual-test/reproducibility coupling, not proof that a policy can infer the future.
- Frequency preservation: confirmed for three native-pool samples during this audit, and supported structurally by row-preserving code. Add a permanent end-to-end raw H5 → cache → world test covering all five fields, chronology, cropping, offsets and tied ToAs. Test cache corruption too.
- Reproducibility: existing same-seed tests plus successful spot checks. Always reset receiver randomness explicitly; `sample_world(seed)` selects a deterministic world but the constructor initially resets its receiver without that supplied seed. Do not universally assert that any two distinct seeds must produce different worlds; collisions are possible. Use a chosen diverse fixture and compare recipe/data hashes.

## Synchronization validation to add

This is not implemented as the requested parameter sweep. `include_periodic_synchronisation_test` is a config flag; round-robin probe documentation describes lockout, but no executable scan-period × receiver-revisit sweep or corresponding figure was found.

Start with a non-RL, one-emitter experiment and a deterministic overlap oracle. Fix pulse train, dwell, beam width/duty factor, retune and mission horizon. Vary emitter period and receiver revisit period independently, with ratio explicitly defined as `T_receiver / T_emitter`. Sample phase offsets: a ratio alone does not determine lockout.

Compare periodic scanning, randomized scanning and jittered scanning at comparable listening-time budgets. After that, add a true CTMC beam model with generator Q, non-negative off-diagonal rates, zero row sums, exponential holding times and transition probabilities derived from the exit rates. Validate holding-time and stationary-occupancy statistics. A discrete transition matrix by itself does not specify continuous-time dynamics.

Plot deadline interception probability `P(TTI <= H)` and censored TTI against ratio, plus ratio × relative-phase heatmaps. Keep this probability distinct from conditional detector Pd: a perfectly sensitive receiver can have zero interception under lockout. Show worst-phase and phase-averaged behavior. Use event-time overlap or verify convergence as time resolution improves; 50 ms quantization can create artificial resonances. Peaks and valleys are conditional on phase/window/horizon, not guaranteed at every integer ratio.

Acceptance cases: an incompatible periodic phase yields no interception, an overlapping phase yields the analytically expected first capture, and randomized visits agree with a derived opportunity-count probability within statistical uncertainty. Do not remove non-intercepts when computing TTI summaries.

## Joint stress and generalization experiments

Register A–G explicitly: fixed/static; agile/static; fixed/periodic beam; fixed/aperiodic beam; agile/periodic beam; agile/aperiodic beam; dense mixed. Existing synthetic classes cover parts of A/B/C/E, but there is no complete controlled family runner or calibrated CTMC support for D/F. A named mixed scenario alone is not G validation.

Keep frequency law, beam law, PRI law and density as independent configuration axes. Use TRAIN data only for calibrating empirical distributions. Report controlled stress mixtures separately from empirical sampling.

Generalization still needs nested train pool sizes 100/250/500/1000/2500, controlled training density, held-out agility-configuration families and separate source-replay/compositional-world results. Fix evaluation recipes and compute budgets across pool sizes; file-held-out evaluation alone does not establish unseen agility-family generalization.

`mandatory_figures.py` already has training mean/std, density, generalization and compute plotting helpers. Those are plotting capability, not evidence of the requested completed experiments. Persist per-training-seed curves, returns and wall-clock timestamps. Repeated receiver/world seeds are not substitutes for independently trained seeds. Label policy decisions and base environment slots separately: a 100 ms action consumes two 50 ms looks. Do not interpret OIR per million steps as a physical interception rate; prefer OIR versus training effort, with explicit units.

## Recommended file placement

Keep one reusable implementation in the package and thin CLI entry points in `scripts/`. Proposed paths below are recommendations, not files created by this audit:

```text
vyapti_simulator/tsrd/
  emitter_realization.py    # identity, immutable PDW history, trajectory summaries
  emitter_pool.py           # split-generic native/cached access; replace duplicated pool logic
  world_composer.py         # explicit recipe, offsets/crops, mode tags
vyapti_simulator/physics/
  illumination.py           # static, periodic, CTMC; use existing rf/ utilities where suitable
vyapti_simulator/core/
  world_truth.py            # emitted / recorded / illuminated / receiver-eligible distinctions
  episode_audit.py          # four separated log schemas and writers
vyapti_simulator/experiments/
  synchronization.py
  stress_families.py
  generalization.py
scripts/evaluation/
  run_synchronization.py
  run_stress_families.py
  run_generalization.py
configs/experiments/        # versioned protocols and recipes
tests/
  test_world_preservation.py
  test_world_reproducibility.py
  test_policy_firewall.py
  test_illumination.py
  test_synchronization.py
```

Specifically move reusable cache/world logic out of `scripts/training/vyapti_train500_cache.py` when implementing the refactor, retaining a compatibility wrapper for existing runs. Avoid adding another parallel simulator. Keep score calculations evaluator-only in `tsrd/replay_scorecard.py`; extend the existing plotting module. Do not mechanically move `src/emitter_models.py` until its imports are migrated and verified. Preserve exploratory patch scripts and historical outputs; new work should not depend on rerunning those patches.

Order of work: repair the protocol interface → freeze source/cache/split and metric contracts → add the hidden illumination layer and emitted-versus-recorded distinction → establish synchronization validation → register A–G → implement held-out compositional/generalization runs → produce independently trained multi-seed figures. Start log schemas with the contract work so subsequent experiments are reproducible.

## Literature boundary

[TSRD paper v2](https://arxiv.org/html/2602.03856v2) supports the five PDW components, separate STARE/SCAN collections and 2500/250/250 source split. Its receiver is idealized, but pulses are still dropped under stated conditions; a recording should not be treated as proof of every transmitted pulse. The manuscript version is not a substitute for pinning the dataset revision.

[Clarkson (2019)](https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/iet-rsn.2018.5668) supports simultaneous frequency/spatial interception conditions and periodic versus Markov-chain scheduling analysis. This supports the synchronization experiment; it does not validate the repository merely because code comments cite it.

[Teissier, Toumi and Khenchaf (2026)](https://researchportal.ip-paris.fr/en/publications/interception-model-of-random-scanning-strategies-against-frequenc/) explicitly considers frequency-agile emitters with rotating directional antennas and randomized receiver scanning. It supports joint stress testing, not arbitrary population proportions. Apfeld's detailed SNR model was not independently checked in this audit; do not claim an exact reproduction of that model from this report.

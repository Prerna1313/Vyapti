# System B Mode-B evaluation protocol

## Scope and frozen data inventory

This protocol evaluates recorded-TSRD scheduler development in composed
System B worlds. Current source inventory is 250 TRAIN H5 files, 50 VAL files,
and 50 TEST files. Keep the checked-in 50/50 held-out size. The protocol does
not claim a 250-world VAL or TEST set; obtain additional held-out source files
before making that claim. Do not manufacture extra source records by relabeling
or repeatedly composing the current 50 files.

All held-out source IDs, composed-world recipes, receiver seeds, and
illumination seeds are fixed in
`training_setup/evaluation/heldout_worlds.json`. Every algorithm and training
seed must replay identical world recipes and receiver seeds. VAL is available
for model selection. TEST remains sealed until algorithms, checkpoints,
hyperparameters, and analysis are frozen; then run the selected configuration
once with `--final`.

Because there are only 50 VAL source files, do not call a subset DEV-VAL and
the same overlapping set FINAL-VAL. Use the fixed 50-world VAL set for
selection and disclose its size. A separate DEV-VAL/FINAL-VAL ladder requires
new, independent VAL source data.

## Frequency agility

`frequency_agility.py` computes features from complete, unmodified source
traces. To accommodate frequency jitter and simultaneous transmissions
described in Gunn et al.'s **paper v2**, it groups exact simultaneous ToAs into
sets of 500 MHz channel IDs (`floor(frequency_mhz / 500)`). It counts changes
between these channel sets. This analysis grid is distinct from the receiver's
possibly overlapping passbands. Within-channel jitter is not a channel change;
jitter across boundaries can still cause changes. Missing pulses can hide
transitions. These are observed receiver-scale proxies, not nominal hop truth.

Per-emitter features include channel-change rate in Hz, median complete dwell
between changes (endpoint dwells censored), span in MHz, distinct recorded
frequencies/channels, and the time fraction of intervals joining changed
states. Single-time traces have unavailable rates, not zero rates.

Before training, the runner fits the 1/3 and 2/3 quantiles of mean emitter
channel-change rates across the selected 250 native TRAIN source scenarios.
It stores every fitted feature, TRAIN IDs, NPZ hashes, and cache/index hashes
in `runs/<id>/agility_reference.json`, whose hash is part of `config.json`.
Evaluation applies those boundaries to each composed world's mean emitter
rate, computed before the illumination gate. Composition can shift the source
distribution, so equal-sized evaluation strata are not assumed. Tied rates
stay together; empty strata and unmeasurable worlds remain explicit.

Cross the three regimes with the existing normal, periodic-illumination, and
stochastic-illumination conditions. Keep the same frozen world seeds for every
algorithm. Each condition report contains all three agility strata, continuous
agility/density points, and separate TTFI results. `plot_run` writes native
agility plots and Kaplan-Meier TTFI CDFs from saved reports.

## Primary outcomes and privileged comparison

Report opportunity interception ratio (OIR) as the primary scheduler outcome.
When a privileged scheduler is evaluated on the same worlds, report per-world
`OIR_privileged - OIR_policy` and its paired 95% confidence interval. Name it a
**Privileged Oracle Scheduler / Upper-Bound Scheduler**; it has hidden truth
and is not an absolute optimum unless the constrained scheduling problem is
solved exactly.

Report time-to-first-intercept (TTFI) separately: show the full empirical
distribution and censoring, plus `TTFI_policy / TTFI_privileged` only when the
ratio is defined. Do not combine TTFI and OIR into one oracle score. TTFI
observations must use the same eligible-emitter definition and mission horizon
for both schedulers.

The evaluator now runs a finite-horizon dynamic program maximizing expected
true positive band-slot looks for the active binary receiver, using known
recorded pulse windows and the same dwell/retune constraints. It never reads
future receiver-noise draws and replays the same receiver seed. Its exactness
is restricted to expected OIR under this simplified receiver; it is not a
physical optimum or an optimum for TTFI/discovery. Realized OIR gaps may be
negative because of stochastic detections and must not be clamped.

Per-world reports include the oracle scorecard, paired OIR gap, both TTFI
distributions, and a ratio of Kaplan-Meier restricted means at a common
restriction (minimum shared emitter follow-up). A zero oracle restricted mean
or absent population makes that ratio unavailable. Entirely occluded emitters
remain censored. Oracle schedules and first-intercept timelines are saved in
evaluator-only logs. Checkpoint selection remains pooled VAL_NORMAL OIR.

## Prediction outcomes

Prediction metrics are optional and depend on the algorithm interface. Report
them only when a policy explicitly emits matching predictions. For an optional
36-band next-slot activity head, report Brier score, log loss, calibration,
and per-band precision/recall/F1. Do not use accuracy as the headline for a
sparse activity target. Report intercept-time error only for a policy that
explicitly predicts intercept time. Policies without these outputs have no
prediction metric, not a synthesized score.

Adapters may implement `predict_band_activity(state)` and/or
`predict_next_intercept_slot(state)`. They receive a public state **after**
observing slot t; `state.time_slot` is target t+1. The band hook returns one
probability per band for that target. The intercept hook returns a future
absolute slot. Probabilities must be finite and in [0,1]. The evaluator clips
only log-loss calculation at 1e-12 and records reliability bins. Intercept
predictions without an eventual true intercept are reported as censored and
excluded from numerical error. Final-slot forecasts are out of horizon.

## Training seeds and uncertainty

Use three independent training seeds for development and five independent
training seeds for the final algorithm comparison. Report the five seed-level
means and standard deviation separately from world uncertainty.

For algorithm comparisons, pair results by the exact frozen world ID. For
each world compute `OIR_A - OIR_B`, then bootstrap those paired differences
with replacement for a 95% percentile interval. Do not bootstrap each
algorithm independently or replace paired differences with a difference of
pooled ratios. Record the bootstrap seed and number of resamples.

`algorithm_comparison.py` runs the fixed pilot seeds [11,22,33] or final seeds
[101,202,303,404,505]. It selects each seed's checkpoint by VAL_NORMAL OIR and
then selects the algorithm by mean seed-level pooled VAL_NORMAL OIR, using
documented deterministic ties. For paired world CIs it averages training seeds
within each world first, then resamples worlds 10,000 times with seed 20261002.
OIR and TTFI seed variation are reported separately. World identity, source
hashes, receiver/illumination seeds, and TRAIN reference hashes must agree.
Reports also identify worlds with undefined outcomes.

The suite freezes `comparison_selection.json` before TEST. Only the selected
algorithm's five checkpoints may be tested, and a started final TEST stage
cannot be rerun. Pilot runs cannot reveal TEST. Changes to selected configs,
checkpoints, selection records, or validation evidence prevent final TEST.

## Commands

Supply an explicit episode budget appropriate to the actual algorithm:

```powershell
python -m scripts.evaluation.compare_algorithms run --environment training_setup/environments/train250_composed.json --algorithms path/to/algorithm_a.json path/to/algorithm_b.json --stage pilot --episodes 1000 --checkpoint-every 100 --output runs/mode-b-pilot
python -m scripts.evaluation.compare_algorithms run --environment training_setup/environments/train250_composed.json --algorithms path/to/algorithm_a.json path/to/algorithm_b.json --stage final --episodes 1000 --checkpoint-every 100 --output runs/mode-b-final
python -m scripts.evaluation.compare_algorithms finalize-test --output runs/mode-b-final
```

The example budget is illustrative, not a justified budget for an unknown
algorithm. The chosen algorithm modules must implement the common interface.
No neural architecture or optimizer is imposed by this runner.

## Report matrix

| Dimension | Required report |
|---|---|
| Normal scheduler performance | OIR, conditional Pd, true Pfa, TTFI distribution |
| Frequency agility | OIR and TTFI against observed hop rate, dwell interval, and frequency span |
| Spatial stress | OIR and TTFI for normal, periodic, and stochastic illumination |
| Privileged comparison | Paired OIR gap and separate TTFI ratio/distribution |
| Temporal behavior | TTFI CDF, censoring, and first-intercept timeline |
| Coverage | Band coverage and revisit interval |
| Prediction | Brier, log loss, calibration, optional per-band metrics when emitted |
| Robustness | Results stratified by emitter density |
| Reliability | Five-seed variation and paired 95% world-level intervals |

Keep TEST out of checkpoint, algorithm, reward, and hyperparameter selection.
Archive the frozen config, runtime manifest, per-world metrics, checkpoint
hashes, and test report with each experiment.

## Paper and dataset provenance

The reference is Gunn et al., [arXiv:2602.03856v2](https://arxiv.org/html/2602.03856v2)
(7 April 2026). The local metadata confirms a 30 s collection and PDW units
are ToA us, frequency MHz, PW us, AoA degrees, amplitude dB. The source
download's exact repository revision is still unknown in the existing local
manifest: the paper's v2 identifier must not be relabeled as a verified dataset
download revision. Preserve both fields separately in every run.

The beam families refer to Clarkson (2019),
[DOI 10.1049/iet-rsn.2018.5668](https://doi.org/10.1049/iet-rsn.2018.5668).
Frequency/intercept-time comparison is motivated by Teissier et al. (2026),
[DOI 10.1109/TAES.2026.3660987](https://doi.org/10.1109/TAES.2026.3660987).
Numerical stress defaults and the new scheduling DP are our experiment
definitions, not claims of reproducing those papers' fitted parameters or
algorithms. System B is an adaptation of a deinterleaving dataset for recorded
PDW scheduler development; the native TSRD challenge measures clustering.

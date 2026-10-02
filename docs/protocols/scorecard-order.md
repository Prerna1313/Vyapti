# Metric names and report order

The frozen Mode-B split, seed, oracle, agility, optional-prediction, and paired
uncertainty contract is documented in
[`mode-b-evaluation.md`](mode-b-evaluation.md). That protocol distinguishes
implemented scorecard fields from planned evaluations.

Use this order when presenting results. It groups the small set of headline
metrics first, then prediction and receiver diagnostics, robustness results,
and engineering/training diagnostics. Detailed counters stay available for
audits; they do not all belong in the headline table.

## Level 1 — Primary scheduler results

Present these first:

1. `emitter_interception.unique_emitter_interception_rate`
2. `tier_a_ps26055.opportunity_interception_ratio` (OIR)
3. `cell_level.conditional_pd`
4. `cell_level.true_pfa`
5. `ttfi.km_median_ms`, `ttfi.km_p90_ms`, and `ttfi.censoring_fraction`
6. `event_level.false_alarms_per_s`
7. `revisit.mean_revisit_s` and `revisit.p95_revisit_s`
8. `decision_level.empty_scan_fraction`

These names refer to existing scorecard fields. The replay scorecard also has
`illumination.illumination_interception_ratio`; retain that name where that
specific emitter-slot denominator is used. These denominators are distinct,
so label the metric contract and units in every comparison.

## Level 2 — Prediction, receiver, and coverage diagnostics

Show prediction metrics only when the algorithm actually emits predictions.
The broader core metrics use `prediction_accuracy` and
`average_intercept_time_error_slots`; report the source field and unit:

- `prediction_accuracy`, precision, recall, F1, Brier score, and calibration
- `average_intercept_time_error_slots`, median and p95 error in slots

For receiver calibration, report curves separately from per-run scheduler
metrics: `Pd_vs_SNR`, `Pd_vs_signal_strength`, and `Pfa_vs_threshold`.
For spectrum coverage, report `unique_band_coverage` and the replay scorecard's
`revisit.band_coverage_fraction` and `revisit.time_to_90pct_band_coverage_s`.
The replay scorecard also reports `revisit.max_blind_interval_s` and
`revisit.p95_blind_interval_s`. Blind intervals count consecutive slots with
an emitter-slot opportunity somewhere in the world when the selected band had
no recorded pulse opportunity. Worlds with no blind interval report zero.

## Level 3 — Robustness results

Group results by the tested world conditions, and report curves/tables instead
of adding one scalar for every sweep:

- Frequency: `OIR_vs_hop_rate`, `TTFI_vs_hop_rate`, and `TTFI_vs_frequency_dwell`
- Illumination: `OIR_vs_beam_scan_period`, `TTFI_vs_beam_scan_period`, and
  `Pd_vs_illumination_duty`
- Combined frequency and spatial agility: `joint_frequency_spatial_OIR` and
  `joint_frequency_spatial_TTFI`

The held-out runner reports TRAIN-derived native agility strata under
`normal`, `beam_periodic`, and `beam_stochastic`, with channel-change rate,
dwell, span, density points, and censored TTFI comparisons. Arbitrarily
accelerated frequency-hop sweeps remain outside the primary benchmark.
Implemented support does not itself constitute a measured algorithm result.

## Level 4 — Policy, training, and compute diagnostics

Keep these outside operational scorecard headlines:

- Policy: `revisit.action_entropy`, `revisit.normalized_action_entropy`, and
  per-band selection distribution
- Training: episode return and algorithm-specific losses/updates
- Compute: `status.json` records `wall_clock_training_s`,
  `environment_steps`, and `steps_per_second`; device time and peak memory are
  unavailable unless the selected algorithm/device logger supplies them
- Uncertainty: seed mean, standard deviation, and confidence interval

Do not report algorithm-specific losses for algorithms that do not have those
losses. Do not infer prediction accuracy or intercept-time error from actions.

## Naming rules

- Use lowercase `snake_case` for machine fields.
- Keep units in field names (`_ms`, `_s`, `_hz`) and state the units in tables.
- Preserve existing metric contract field names. If a schema needs new names,
  version the contract and update its consumers together.
- Keep eligibility denominators explicit: the replay scorecard exposes
  `physical_emitters`, `active_emitters` (emitters with recorded PDWs in the
  mission), `eligible_intercept_emitters` (with an emitter-slot opportunity),
  and `intercepted_emitters` as separate counts. It also reports
  `missed_emitter_slot_opportunity_count`.
- `metric_contract_version` advances when the replay scorecard schema changes.
- Report unavailable metrics as unavailable with a reason; do not synthesize a
  prediction or receiver sensitivity result to fill a field.

System names also follow one order everywhere: **System A** synthetic PDW,
**System B** recorded TSRD, **System C** RF/IQ. System A's generated PDWs are
development inputs, not a fallback for the TRAIN/VAL/TEST experiment.

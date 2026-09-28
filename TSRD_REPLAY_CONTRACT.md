# TSRD-derived replay contract, version 1.1.0

This contract describes a **TSRD-derived recorded-PDW replay**, not a reconstruction
of emitted RF pulses or the original TSRD receiver. It applies to the local
snapshot inventoried in `data_provenance/tsrd_corpus_manifest.json` (inventory
SHA-256 `40d09d7cc03eae2a305314a8f8146dd2754de77d0279327dfb1952fbb7eab89e`).
The user downloaded the corpus from the original source. The exact upstream
revision of that local snapshot was not retained. The revision in
`tests/fixtures/tsrd/manifest.json` documents two fixtures only.
The local `test/config_0.h5` scan and stare hashes match those fixtures exactly;
this anchors only those two files and does not prove the revision of the other
5,998 HDF5 files.

## Data and units

| Component | Basis |
| --- | --- |
| Five PDW fields | Recorded TSRD stare `data`: ToA in microseconds, frequency in MHz, pulse width in microseconds, AoA in degrees, amplitude in dB per upstream documentation. These are processed receiver observations. |
| Emitter identity | Stare `labels`, meaningful only within each HDF5 file. Used to build offline recorded-opportunity truth and scores; never passed to a scheduler. |
| Receiver geometry | Paired scan `metadata/receiver/dwell_centres_mhz` and `bandwith_mhz` (the original misspelling is an HDF5 key). The existing adapter interprets `bandwith_mhz` as passband half-width. |
| Original receiver schedule | Paired scan `dwell_times_s` and band order, inspected for compatibility only. Scan and stare have separately censored PDW streams and are not merged. |
| Source splits | Train for development, validation for selection, test reserved for one final frozen evaluation. Test files may be opened for integrity and provenance audit only until then. |

Unit assignments follow the upstream challenge description. Numeric range checks
can find anomalies but cannot independently prove that each upstream field was
recorded with the declared unit. None of the 6,000 HDF5 files has an explicit
unit attribute on the root, `data`, or `metadata/feature_names` objects. The
amplitude field is dB; the replay does not
reinterpret it as received power in dBm.

## Fixed reference receiver profile

The validation reference profile is `binary_v1` with `fixed_50_ms` dwells,
600 decisions over each 30 s recording, 36 tune centres from its paired scan
file, passband half-width from the HDF5 receiver attribute, detection
probability 0.9, false-alarm probability 0.05, and 1 ms retune time. These
probabilities and retune time are **declared simulator assumptions**, not
measured TSRD hardware properties. A different value or `pdw_v2` profile is a
separate experiment and requires a new profile ID and report. The original
scan recording contains both 50 and 100 ms dwells; it does not validate the
replay's 50 ms decision model or any two-look 100 ms model.

The replay time origin is **assumed** to be ToA zero, with a `[0,30 s)`
analytical window aligned to the recorded scan schedule. This is a scheduler
benchmark convention, not a statement that TSRD has no later PDWs. The
[TSRD v2 paper](https://arxiv.org/html/2602.03856v2) reports stare ToA
maxima above 43 s in Table II while specifying a 30 s receiver collection;
it does not explicitly define the absolute ToA clock origin. We preserve all
source PDWs. Rows with ToA outside the replay window are counted separately
and remain available to raw-PDW/deinterleaving workflows. Do not wrap or
rescale their timestamps into the replay window.

For `binary_v1`, occupancy is derived from recorded stare PDWs in each
band-slot. The detector is a Bernoulli draw conditional on that recorded
occupancy, plus false positives on empty selected slots. Retune time reduces
the listening portion of a selected slot. The receiver's potential-detection
field is keyed by world, receiver seed, band, and slot, so action history does
not shift future noise draws. A stable per-file seed is the first 32 bits,
little-endian, of SHA-256 over `"{base_seed}:{config_stem}"`; the reference
base seed is 42. A benchmark must also record model/optimizer seeds and
software versions.

Scheduler input is limited to `scenario_descriptor`'s receiver facts and
prior `env.step` observations. The policy must not access `hidden_truth`,
emitter configurations, source labels, source HDF5 handles, or the offline
scorecard. The training callback receives a Python environment object and is
trusted code; the observation whitelist prevents accidental leakage through
the scheduler API, not deliberate inspection of that object.

## Metrics and unavailable quantities

Score only complete 600-slot trajectories. Compare event counts to the saved
trajectory before reporting. `conditional_pd` is true positives divided by
selected occupied recorded-pulse slots; `true_pfa` is false positives divided
by selected empty slots. Illumination interception uses emitter-slot recorded
opportunities; cell detection uses band-slot opportunities. Pool raw
numerators and denominators across worlds for pooled ratios and label any
mean of per-world ratios separately. A zero denominator produces JSON `null`.
First intercept times retain right censoring; observed-only means omit
censored emitters and Kaplan-Meier quantiles stay null when not estimable.

`physical_pulse_interception_ratio`, offered-pulse capture ratio, raw waveform
SNR, and the set of emitted pulses absent from the stare stream are
**unavailable**. `recorded_pulse_coverage_ratio` is a selection-window proxy,
not a capture rate. Its denominator is the recorded PDWs inside the assumed
`[0,30 s)` replay window; raw and outside-window row counts are reported
separately. This
contract changes the binary scorecard version to
`tsrd_recorded_pulse_emitter_v3` and the measured-PDW version to
`tsrd_recorded_pulse_emitter_v4`. Earlier exploratory reports use the prior
denominator and should not be mixed with these metrics. `binary_v1` credits a
band-level positive to all recorded emitters present in that slot, so emitter
attribution is an optimistic upper
bound when they coincide. Prediction accuracy and intercept-time prediction
error are unavailable for a policy that issues no predictions.

No scheduler comparison, reward tuning, or test scoring is part of this
readiness audit. The earlier validation reference run under
`results/tsrd_baseline/` predates this readiness gate and is exploratory.

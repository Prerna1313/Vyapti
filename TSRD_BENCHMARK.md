# TSRD replay benchmark

Set `corpus_root` to the Hugging Face snapshot's `data` directory. The expected
layout is `stare/{train,val,test}_stare/config_*.h5` and matching
`scan/{train,val,test}_scan/config_*.h5`. Keep the three source splits distinct.

For a **stare-only local subset**, the same split directories can be used if
receiver tune centres are supplied explicitly. The code never infers them from
stare PDWs. For the 36-centre geometry used by the bundled paired fixture:

```python
import numpy as np

centres_mhz = 250.0 + 500.0 * np.arange(36)
protocol = TSRDBenchmarkProtocol(
    corpus_root="Data", output_dir="local_benchmark_output",
    band_centres_mhz=centres_mhz,
)
```

This is suitable for a local integration run, not proof that every TSRD
configuration uses those centres. Add matching scan files for final runs; if
both scan metadata and explicit centres are present, a mismatch raises.
The report records whether geometry came from scan metadata or explicit
centres. A stare-only run retains the H5 `bandwith_mhz` attribute's existing
half-width interpretation, which should be checked against paired scan files.

Run `python -m scripts.audit_tsrd_metadata --corpus-root Data --output
tsrd_metadata_audit_local.json` before deriving transmitter properties from
labels. The chunked frequency-envelope check flags possible label-to-metadata
join errors; it does not correct labels or establish a physical pulse truth.

To check receiver geometry and the original variable-dwell scan schedule on
matching **train and validation** files, run `python -m
scripts.validate_tsrd_scan_pairs --corpus-root Data --output
tsrd_paired_scan_validation_local.json`. The command discovers exact-name
companions, checks receiver and transmitter metadata, replays the recorded
scan's band sequence and `dwell_times_s`, and compares scan/stare occupied
dwell windows. It never opens the test split. Its `passband_gate` checks scan
pulses against the scheduled passband; the positive-dwell overlap is only a
descriptive agreement measure, not `Pd`, because scan and stare recordings
have separately censored pulse streams. The 100 ms half-window counts show
whether recorded pulses occur in both halves, but the PDWs cannot establish
whether TSRD's receiver made one integrated detection decision or two 50 ms
decisions. Keep mixed-dwell scheduler results separate from this original-scan
replay until that detector assumption is independently justified.

`TSRDBenchmarkProtocol` indexes labelled histories from **train stare** files,
samples full recorded histories into train worlds, runs an algorithm-specific
training callback, scores frozen checkpoints on untouched validation stare
files, and selects by pooled illumination interception ratio. The default
development run leaves test untouched; `finalize_test` evaluates the selected
frozen checkpoint once when the protocol is locked. Scan files supply receiver geometry,
not additional pulses. This is a recorded-PDW approximation, not a physical RF
waveform reconstruction or complete emitted-pulse truth.

```python
from pathlib import Path
from vyapti_simulator.tsrd.benchmark_protocol import TSRDBenchmarkProtocol

protocol = TSRDBenchmarkProtocol(
    corpus_root=Path(snapshot_path) / "data",
    output_dir="/kaggle/working/tsrd_baseline",
    detection_probability=0.9,
    false_alarm_probability=0.05,
    retune_time_ms=1.0,
    seed=42,
)

def train_candidate(candidate_id, worlds, checkpoint_path, reward_mode):
    # Implement with your model. Iterate worlds; each item is (receiver, provenance).
    # Pass only receiver.step_training(band, reward_mode) feedback to the policy.
    # Save a frozen model at checkpoint_path before returning.
    ...

def load_checkpoint(checkpoint_path, band_count):
    # Return a fresh SchedulerInterface-compatible policy each call.
    ...

report = protocol.run(
    candidate_ids=["seed_1", "seed_2"],
    train_candidate=train_candidate,
    load_checkpoint=load_checkpoint,
    worlds_per_candidate=1000,
    emitter_count=8,
    reward_mode="detector_positive",
)

# After model, reward, receiver profile, and metrics are frozen:
final_report = protocol.finalize_test(load_checkpoint)
```

Each run writes `tsrd_run_manifest.json` before training and
`tsrd_benchmark_report.json` after development completes. The development
report has `held_out_test: null`; finalization fills that field after checking
the corpus inventory, replay settings, and selected checkpoint hash.
The manifest records
the receiver settings, seed derivation, candidate IDs, split file inventory,
and frozen checkpoint hashes. Its inventory fingerprint uses relative paths,
sizes, and modification times, **not file-content hashes**; preserve the source
dataset revision separately for stronger provenance. Training and evaluation
errors propagate and mark the manifest `failed` with the phase and exception;
an incomplete run must not be interpreted as a benchmark result. Scorecard
event totals are checked against the recorded observations before reporting.

The policy callback in `run_episode` receives a whitelisted scenario descriptor
and prior receiver observations, while the offline metrics engine receives
truth separately. The training callback is trusted code: it receives a receiver
object and provenance and could deliberately inspect hidden data. Its policy
implementation must consume only the documented observations. The whitelist
and regression tests catch accidental observation leakage, not deliberate
Python introspection by a trainer.

The default `fixed_50_ms` profile keeps 600 scheduler decisions in a 30 s
mission. Set `dwell_profile="mixed_50_100_ms"` on a **separate** protocol and
output directory to allow a policy to return `DwellAction(band, slots)` with
`slots` equal to 1 or 2. Existing integer-band policies still take one slot.
A two-slot choice yields two independent base-slot detector looks, charges a
retune only on the first look, and cannot be pre-empted. The last dwell clips
to the remaining mission slots. Mixed-dwell training callbacks can call
`receiver.step_dwell_training(band, slots, reward_mode)`; the fixed profile
continues to call `receiver.step_training(band, reward_mode)`.

Receiver noise in this replay lane is a potential-detection field keyed by
recorded world, receiver seed, band, and 50 ms base slot. Changing prior
actions does not consume a different noise draw for a later band-slot.

## Opt-in measured-PDW receiver

The default `binary_v1` profile remains the fixed band-level Bernoulli
benchmark. To test a richer but still PDW-derived receiver, construct a
**separate** protocol with `receiver_profile="pdw_v2"` and a new output
directory:

```python
protocol = TSRDBenchmarkProtocol(
    corpus_root="Data", output_dir="local_pdw_v2",
    band_centres_mhz=250.0 + 500.0 * np.arange(36),
    receiver_profile="pdw_v2",
    amplitude_midpoint_db=-90.0, amplitude_scale_db=5.0,
    detection_probability=0.9, false_alarm_probability=0.05,
)
```

For each selected dwell, `pdw_v2` applies an amplitude-dependent Bernoulli
capture to each *recorded stare PDW* in the live passband. `detection_probability`
is the **per-recorded-pulse ceiling**, not the resulting dwell-level conditional
Pd. Random potential capture is fixed by world, receiver seed, and recorded
pulse row, so policy order does not alter that pulse's draw. A positive
observation contains `receiver_measurement` with the captured-pulse count and
at most `max_observed_pdws` evenly sampled, label-free PDWs from the completed
dwell. Empty-window false alarms produce a synthetic noise-like PDW with the
same field schema; the synthetic feature distribution is a modelling
assumption, not a calibrated TSRD hardware property. A miss carries `None`.
The mixed-dwell decision observation groups the one or two completed base-look
measurements, with no mid-dwell feedback.

The amplitude midpoint and scale are experimental receiver parameters; freeze
them using training data or a predeclared sensitivity sweep, never test
outcomes. The TSRD amplitudes are already measured and the stare stream is
already censored, so this is **not** a physical emitted-pulse or I/Q
reconstruction. The `pdw_v2` scorecard uses offline captured-row labels for
exact attribution *among recorded PDWs* and reports
`recorded_pulse_detected_ratio`. Physical and offered-pulse capture ratios
remain unavailable. `binary_v1` metrics retain the older optimistic
shared-band attribution and must not be pooled with `pdw_v2` results.
As with `binary_v1`, a scheduled 100 ms dwell currently uses two independent
50 ms base looks. The paired scan PDWs cannot determine whether TSRD's
original detector used one integrated decision, so treat this as a distinct
mixed-dwell assumption rather than an exact hardware replay.

Project bins remain disjoint by default. The PDW discretiser has an opt-in
`overlap=True` mode that places a recorded pulse in every receiver passband
covering its frequency. Set the receiver instantaneous bandwidth and tune
centres explicitly for this experiment; do not mix its band-slot denominators
with disjoint-bin results. A duplicated band-slot opportunity is **not** a
second physical pulse.

Run the false-alarm-aware reward as a **separate** protocol invocation with a
fresh output directory, the same corpus and seed, and
`reward_mode="false_alarm_aware"`. Do not use test results to choose a reward
mode, model, or hyperparameters. Each trainer receives the same sampled-world
sequence for a paired comparison; its own optimizer seeds and checkpoint format
remain algorithm-specific.

The scorecard reports per-emitter first recorded pulse, first recorded
opportunity, first intercept, and right-censored TTFI. Observed-only TTFI
mean/median/p90 exclude censored emitters; Kaplan-Meier median/p90 are null
when not estimable. Illumination uses emitter-slot opportunities; cell-level
metrics use band-slot opportunities. A band-level positive is optimistically
credited to every recorded emitter in that selected passband and dwell, since
this benchmark does not perform deinterleaving. The scorecard must never be
used as a scheduler observation.

The Tier A conditional Pd is true-positive **observed occupied dwells** divided
by all observed occupied dwells. Opportunity interception ratio is true-positive
base-slot decisions divided by all occupied recorded-pulse band-slots. Neither
equals a physical pulse interception ratio. `physical_pulse_interception_ratio`
and `offered_pulse_capture_ratio` are `null` in replay: TSRD stare omits some
emitted pulses and our band-level detector does not identify individually
captured pulses. `recorded_pulse_coverage_ratio` is a separate, observable
selection-window coverage proxy over recorded PDWs inside the assumed
`[0,30 s)` replay window,
not a capture metric. The scorecard also reports all recorded rows and the
number outside that replay window; the latter are excluded from coverage and
detected-pulse denominators. Emitter-labelled
discovery and per-emitter Pd are optimistic upper bounds when multiple emitters
share a passband and base slot. Undefined zero-denominator ratios are `null`.

The `censored_mean_ttfi_s` is a Kaplan-Meier restricted mean through the
longest observed emitter follow-up; its restriction horizon is reported.
Median and p95 are Kaplan-Meier quantiles and stay `null` when not estimable.
Revisit intervals measure time between **decision starts** on the same band;
bands visited fewer than twice do not contribute, so read them alongside
unique-band coverage. Switch rate divides band changes by decision transitions.

Run `python -m scripts.pre_train_reward_screen --corpus-root <snapshot/data>
--split val --output <report.json>` before expensive model training. Optional
`--coverage-weight` and `--discovery-bonus` expose those candidate rewards
without silently choosing coefficients. The screen tests causal pathological
policies on development data only; its flags diagnose reward/metric rank
inversions, but no finite screen mathematically guarantees a reward is safe.

# TSRD corpus and replay readiness — 2026-09-28

**Updated after reviewing [TSRD v2](https://arxiv.org/html/2602.03856v2):
the local corpus passes integrity checks; the temporal mapping for a 30 s
scheduler replay remains an explicit modelling choice.** The paper itself
reports stare ToA maxima above 43 s in Table II while saying receiver
collection lasts 30 s. ToA values above 30 s are a published dataset
characteristic, not evidence of corrupt local files. The local snapshot is
content-hashed; the exact upstream download revision remains unknown. No
scheduler was run on the held-out test split during this readiness work. The
earlier validation control in `results/tsrd_baseline/` is exploratory.

## Evidence and gate status

| Gate | Result | Evidence and limit |
| --- | --- | --- |
| Source freeze | **Local snapshot passed; upstream revision unknown** | `data_provenance/tsrd_corpus_manifest.json` hashes every local HDF5 and sidecar, records the exact layout, and has inventory SHA-256 `40d09d7cc03eae2a305314a8f8146dd2754de77d0279327dfb1952fbb7eab89e`. The user identifies the download as coming from the original source; the exact source revision was not retained. |
| Corpus audit | **Passed** | All 6,000 files and 3,000 exact-name pairs were read. `results/tsrd_readiness/full_corpus_audit_v2.json` reclassifies the published ToA range as temporal context and shows zero integrity failures. The original count report is preserved as `full_corpus_audit.json`. Frequency-envelope flags remain diagnostic. No files were silently skipped. |
| Replay contract | **Versioned** | `TSRD_REPLAY_CONTRACT.md` version 1.1.0 freezes the source/simulated/unavailable boundary and the `binary_v1` fixed-50-ms reference receiver. Its detector probabilities and 1 ms retune are declared assumptions, not TSRD hardware estimates. |
| Isolation and repeatability | **Passed for the frozen reference profile** | `results/tsrd_readiness/replay_repeatability.json` and `replay_repeatability_val_1.json` repeat complete validation `config_0` and `config_1` episodes with identical action, hit, receiver-observation, and scorecard hashes at the same seed. Another seed changes the potential-detection field. Scheduler input was checked against the key whitelist with no labels or truth; runtime profiling is now confined to offline trajectories. |
| Metric accounting | **Passed for exercised edge cases** | The real validation trajectories were independently recounted against recorded PDWs: selected occupied/empty slots, true detections, false alarms, covered PDWs, and pulse denominators agree. Tests cover zero-denominator nulls, empty recordings, censored discovery, retune loss, and pooled count ratios versus means of per-world ratios. Full suite: 632 passed. This validates the implemented recorded-PDW metrics, not physical pulse interception. |
| Validation protocol | **Development ready; sensitivity check remains before final reporting** | A 30 s replay uses ToA in `[0,30 s)` against the scan schedule starting at zero. The v2 paper does not explicitly define the ToA clock origin or how its >30 s PDWs relate to the collection interval. Benchmark `run()` now leaves test untouched by default; `finalize_test()` explicitly scores the frozen validation winner. No post-gate scheduler comparison, reward selection, or test scoring has been run. |

## Frozen local inventory

| Split | Scan files / PDWs | Stare files / PDWs | Exact-name pairs | Empty recordings |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,500 / 233,172,417 | 2,500 / 3,172,039,637 | 2,500 | 8 scan + 8 stare |
| Validation | 250 / 22,693,747 | 250 / 316,743,173 | 250 | 0 |
| Test | 250 / 26,965,861 | 250 / 367,473,884 | 250 | 0 |

The 6,000 HDF5 files total 65,792,130,658 bytes; the 18 CSV/TXT sidecars
bring the hashed snapshot to 65,794,507,621 bytes. The manifest's canonical
inventory digest was recomputed successfully. The local test scan/stare
`config_0.h5` hashes equal the two checked-in fixture hashes associated with
revision `68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51`. This verifies those
**two files only**; the fixture manifest expressly does not certify the other
5,998 HDF5 files or the local download's revision.

All files have five `float32` PDW fields in the declared order, `int8` labels,
a 30 s receiver collection and a 500 MHz `bandwith_mhz` attribute. All scan
files have 36 tune centres and 502 scheduled dwells; their observed scheduled
passband fractions are at least 0.9923. All 3,000 pairs match on checked
receiver and transmitter metadata. No file had missing transmitter configs for
an observed label, nonfinite PDW values, decreasing ToA, nonpositive frequency
or pulse width, or AoA outside ±180 degrees. None of the HDF5 roots, PDW
tables, or feature-name datasets carries explicit unit attributes; units rely
on upstream documentation and receiver attribute names. See
`results/tsrd_readiness/unit_metadata.json`.

## Temporal context and diagnostic flags by split

| Split | Files with ToA ≥ 30 s | PDWs with ToA ≥ 30 s | Suspect scan label/metadata joins | Suspect stare label/metadata joins | Pair metadata failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 805 stare | 2,471,098 | 27 | 458 | 0 |
| Validation | 1 scan + 84 stare | 224,925 | 0 | 0 | 0 |
| Test | 96 stare | 265,842 | 14 | 21 | 0 |

The 2,961,865 rows with ToA ≥ 30 s are consistent with the v2 paper's Table
II maxima of 43.49/43.17/43.33 million microseconds for train/validation/test
stare data. Section II-A calls the receiver collection 30 s but does not
specify the ToA clock origin or a rule that every ToA must be below 30 s.
The paper's scan Table III also reports a validation ToA maximum of 30.14 s,
matching the local audit. The corrected integrity report has **zero timestamp
integrity failures**. These rows remain in the source HDF5; raw deinterleaving
uses the full sequences.

The Hugging Face dataset card still describes stare collection as 10 s; that
text is outdated. The latest v2 paper and all local HDF5 `collection_time_s`
values establish the 30 s collection duration for this corpus. The v2 paper
is the governing publication for this audit; the card's stale summary is not
used to set replay timing.

For our separate scheduler replay, `[0,30 s)` is a declared analytical
window aligned with the recorded scan schedule, not a claim that later PDWs
are invalid. Under that convention, later rows cannot be selected. Previously,
the pulse-coverage denominator included them while occupancy did not. The
scorecard now reports raw, in-replay-window, and outside-replay-window row
counts, and uses the in-window rows for recorded-pulse ratios. The binary
scorecard contract advanced to
`tsrd_recorded_pulse_emitter_v3` (`pdw_v2` to `v4`). Old exploratory ratios
must not be mixed with the revised ones. Validation `config_1` independently
reconciles its 3,007 ToAs ≥ 30 s between audit and replay. Overall the
after-30 s fraction is small, but individual files can be affected more:
train stare `config_405.h5` has 3,011 such rows out of 11,473 (26.2%).
Across development, 53 train stare files and 5 validation stare files have
at least 1% of PDWs at or after 30 s; 12 train files, but no validation files,
reach 5%. A predeclared sensitivity report should compare validation results
on all 250 files with the 245-file subset below the 1% threshold, without
using that slice to choose a model or adjusting the time origin after seeing
test results.

The label-frequency diagnostic flags an observed label only if at least 20
PDWs exist and at least 95% lie more than 50 MHz outside the matching
transmitter metadata's nominal frequency envelope. The scan report is
`results/tsrd_readiness/scan_label_frequency_full.json`; the stare reports are
`results/tsrd_baseline/development_label_frequency_full.json` for train and
validation and `results/tsrd_readiness/test_stare_label_frequency.json` for
test. These are **possible joins to review**, not proof that file-local labels
are wrong or a basis for automatic relabelling. The replay uses file-local
labels for offline attribution and does not infer transmitter frequency from
the flagged metadata. There are no read errors or missing nominal-frequency
labels in these diagnostics. No complete HDF5 file has been excluded.

The bundled `data_stats.csv` sidecars are not a trustworthy inventory. For
example, train scan has 3,245 CSV rows but 2,500 HDF5 files; train stare has
1,384 CSV rows; validation scan/stare have 287/508 CSV rows for 250 files
each. Test scan uses numeric CSV IDs rather than the HDF5 `config_*.h5`
names. The replay and this audit use HDF5, not those summaries.

## Software, replay boundary, and next gate

The repeatability report records Python 3.13.13, source project version 1.2.0,
NumPy 2.2.5, h5py 3.16.0, SciPy 1.17.1, pytest 9.1.1, and psutil 7.2.2.
Its receiver seed is the first 32 bits, little-endian, of SHA-256 over
`"{base_seed}:{config_stem}"`, with base seed 42 for the reference check.
The `binary_v1` detector is simulated. TSRD supplies processed, censored stare
PDWs, not raw waveforms or a complete emitted-pulse set. Physical pulse
interception and offered-pulse capture therefore remain JSON `null`.

For an internally reproducible replay, the local content hashes are a
sufficient source pin; record the original-source URL and note that an exact
upstream Git revision is unavailable. Before a publishable scheduler
comparison, explicitly retain the `[0,30 s)` time origin as a modelling
assumption, run a train/validation-only sensitivity analysis for files with
large post-30 s fractions, and state that full PDW sequences remain available
for deinterleaving. Do not wrap, rescale, or relabel later ToAs without
upstream evidence. The frequency-envelope flags can be accepted as limitations
for a replay that uses file-local labels and never joins transmitter metadata
to those labels. Preserve test for the one final evaluation after selection;
its use here was limited to provenance and data-quality inspection.

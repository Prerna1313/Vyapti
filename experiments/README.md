# TRAIN-250 recorded-PDW experiments

The active setup is **250 cached TRAIN source configurations, 50 fixed VAL
configurations, and 50 fixed TEST configurations**. TRAIN worlds are sampled
from the cached emitter pool; 250 is the source pool size, not the number of
training episodes. VAL is used for model selection. TEST is reserved for final
evaluation. The versioned config names the exact 50 IDs in each held-out split
and checks the expected counts. Composed held-out worlds are optional and are
disabled in the active config. Each of the 50 held-out source recordings can
be evaluated under NORMAL, BEAM_PERIODIC, and BEAM_STOCHASTIC conditions.

This protocol develops a band scheduler using the existing TRAIN-250 cache. It
replays recorded STARE PDWs through `TSRDStareEnvironment`; it does not claim
to simulate all emitted pulses, antenna pointing, illumination, or live RF
propagation.

Run from the repository root:

```powershell
python -m scripts.training.train --config experiments/configs/train250_binary_v1.json --run runs/ucb-prior-001
python -m scripts.evaluation.evaluate --run runs/ucb-prior-001 --split val
python -m scripts.evaluation.plot_run --run runs/ucb-prior-001
```

After choosing a checkpoint on VAL, run TEST once:

```powershell
python -m scripts.evaluation.evaluate --run runs/ucb-prior-001 --split test --final
```

The config fixes the observation, action, receiver, reward, sampling recipe,
evaluation IDs, and metric definitions. `train.py` imports `policy_module`.
Its `create(bands, seed, checkpoint=None)` returns an object with
`reset_episode()`, `select_action(history, time_slot)`, `observe(observation)`,
`finish_training_episode()`, and `save(path)`. The trainer passes only public
receiver observations. Any new policy module must use the same interface and
must not read dataset files directly.

Training creates a new run directory and refuses to overwrite it. The run
records its exact config, cache fingerprint and index hashes, episode source
identities, all actions and observations, rewards, and a checkpoint hash.
Evaluation checks the hashes, writes per-world source hashes and scorecards,
and refuses to overwrite an existing condition result. The `worlds.json` file
records the ordered fixed world list and receiver seed base. Set
`val_composed_worlds` in the config before training to add frozen mixtures:

```json
[{"id":"val_mix_01","source_config_ids":["config_0","config_1"],"world_seed":31,"emitter_count":20}]
```

The corresponding TEST recipe belongs in `test_composed_worlds`. Sources are
restricted to their own split. The selection metric is the pooled recorded
opportunity interception ratio on VAL_NORMAL; TEST results are final reporting only.
Conditional Pd, Pfa, and emitter interception are separate measures. Emitter
attribution is evaluation-side oracle attribution and can credit multiple
emitters for one band hit.

## Physics boundary

Held-out stress outputs are written under:

```
eval/val/normal/               VAL_NORMAL
eval/val/beam_periodic/        VAL_BEAM_PERIODIC
eval/val/beam_stochastic/      VAL_BEAM_STOCHASTIC
eval/test/normal/              TEST_NORMAL
eval/test/beam_periodic/       TEST_BEAM_PERIODIC
eval/test/beam_stochastic/     TEST_BEAM_STOCHASTIC
```

The CLI evaluates all configured conditions by default. Use `--condition
normal`, `--condition beam_periodic`, or `--condition beam_stochastic` to run
one. These are three conditions applied to the same 50 sources, rather than
150 independent source configurations. TRAIN has no illumination transform.

Periodic illumination uses a 2 s rotation with a 10% illuminated window and
an independently seeded phase for each emitter. Stochastic illumination uses
a two-state continuous-time Markov chain, exponential illuminated residence
of mean 0.2 s, and exponential dark residence of mean 1.8 s. Its initial state
is drawn from the stationary distribution. These defaults have the same
nominal 10% visibility and are controlled stress assumptions. They are not
measured TSRD antenna parameters.

The gate selects recorded PDWs according to their original timestamps. It
preserves their frequency, timing, PW, AoA, and amplitude values. Frequency
agility can therefore coexist with either spatial schedule. The hidden gate
intervals, per-emitter parameters, seeds, and original/visible pulse counts
are saved under `world_logs/`; these never enter receiver observations.
Metric opportunity denominators in a beam condition use visible recorded
opportunities. They must be compared alongside the offered source pulse
counts in the audit logs, not interpreted as physical emission denominators.
The report also retains the original source emitter population: emitters
that are fully dark remain missed/censored, and source-relative TTFI includes
the wait from their first recorded opportunity through beam occlusion.

Model families: [Clarkson (2019), periodic and CTMC beam schedules](https://doi.org/10.1049/iet-rsn.2018.5668)
and [Teissier et al. (2026), frequency agility and rotating directional antennas](https://doi.org/10.1109/TAES.2026.3660987).

STARE recordings contain received PDWs, not all transmitted pulses or a beam
state. An absent PDW therefore cannot prove an emitter was silent or outside
the antenna beam. The receiver uses a fixed passband and an assumed retune
cost. A full physics claim needs explicit antenna/illumination state and
independent synchronization validation against a source that records it.

## Receiver observation and detector models

TSRD receivers, the shared synthetic receiver, and the experiment policy boundary use frozen, slotted
`ReceiverObservation`, `ReceiverMetadata`, `ReceiverMeasurement`, and
`MeasuredPDW` records. Mapping-style reads remain available. Unknown fields
are rejected at both top-level and nested boundaries. Serialization creates
copies; offline profiling fields are added to those copies rather than live
observations.

The active TRAIN-250/VAL/TEST runner is **recorded STARE Bernoulli detection**
(`binary_v1`). For a selected window after retune, an eligible recorded pulse
gives a hit with probability 0.9; an empty selected window gives a false alarm
with probability 0.05. The latter is per base look, not per sample or pulse.
These probabilities are replay assumptions. Fixed world/seed noise fields
make repeated candidate evaluations deterministic. The optional `pdw_v2`
receiver uses amplitude-dependent pulse detection and a Bernoulli false-PDW
process on empty windows; it also does not run CFAR.

The **RF I/Q path** runs `CFARMatchedFilterDetector`. It applies matched
filtering, averages neighboring envelope magnitudes outside guard cells, and
uses a configured threshold gain. Noise peaks cause false alarms naturally.
The current gain convention is `10 ** (cfar_db / 10)` on magnitude, so the
offset is not a calibrated target Pfa. No equality between its empirical Pfa
and the replay's 0.05 is assumed. CFAR needs an I/Q noise process that STARE
PDWs do not contain. This resolves which model is active without claiming
that the two detector operating points are equivalent.

The RF detector returns a five-field `MeasuredPDWStream` with no emitter-ID
attribute. Legacy six-field TSRD adapters explicitly fill labels with -1
(unknown). RF closed-loop association uses measured frequency/AoA and local
track numbers. Its AoA remains a noisy simulation proxy from the emitter map,
not an independent multi-antenna AoA estimator.

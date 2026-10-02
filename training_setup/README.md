# Shared training setup

This directory contains reusable specifications, not trained models. No neural
algorithm is selected by the environment. The former `experiments/configs`
bundles have been replaced by:

```
training_setup/
  environments/train250_composed.json       receiver, actions, reward, composition
  environments/train250_source_replay.json  source-replay evaluation comparison
  evaluation/heldout_worlds.json            shared frozen VAL-50 / TEST-50 worlds
  algorithms/ucb_prior.json                 existing UCB adapter and settings
  plans/ucb_pipeline_check.json             example budget, seed, checkpoint interval
  examples/custom_algorithm.json            template for the implementation you choose
```

The composed environment uses the existing TRAIN-250 source cache. Its 50 VAL
and 50 TEST recipes and seeds were migrated without resampling. Periodic and
CTMC illumination are evaluation-only transforms of those same worlds. The
source-replay environment evaluates the 50 original recordings instead.
Both active Mode-B specs use contract v2: 300 us band-change retune, zero
same-band retune, and the TRAIN SCAN-derived 29/7 native dwell profile.

Use an explicit run plan:

```powershell
python -m scripts.training.train --plan training_setup/plans/ucb_pipeline_check.json --run runs/ucb-pipeline-check-002
```

Or select an environment and any implemented algorithm directly:

```powershell
python -m scripts.training.train --environment training_setup/environments/train250_composed.json --algorithm path/to/chosen_algorithm.json --episodes 1000 --seed 42 --checkpoint-every 100 --run runs/chosen-algorithm-seed-42
```

The numbers in the command are examples, not a convergence claim or recommended
budget for an unknown algorithm. Compare algorithms using the same world seeds,
interaction budget, receiver assumptions, validation worlds and metric. Use
several training seeds. The runner writes a fully resolved `runs/<id>/config.json`
plus hashes of the source specifications; later edits to setup files cannot alter
that run. Existing run snapshots remain readable with legacy `--config` support.

## Plugging in another algorithm

Create an algorithm spec with its importable `module`, `api`, `settings`, and
checkpoint extension. `api: public_transitions` calls:

```python
create(*, bands, seed, settings, checkpoint=None)
```

The returned adapter implements `reset_episode(training=...)`,
`select_action(public_state, training=...)`, `observe(public_transition,
training=...)`, `end_episode(training=...)`, and `save(path)`. It owns its
optimizer, architecture, update frequency, replay/rollout buffers and checkpoint
format. In evaluation it may maintain observation history but must freeze model
weights. `end_episode` may return JSON-serializable update diagnostics, which are
saved to `train/algorithm_updates.jsonl`.

`PublicState` contains only the slot and previous typed receiver observation.
`PublicTransition` contains current state, action, scalar training reward,
next state, and terminal/truncation flags. Closed observation records exclude
source IDs, emitter types, future pulses, hidden occupancy and illumination.
For recurrence, an adapter manages its own memory and resets it between worlds.
The v2 receiver contract has 36 discrete bands and a TRAIN SCAN-derived dwell
profile: 29 bands consume one 50 ms slot and 7 bands consume two. The selected
band determines its dwell duration; algorithms still return one band index.
Standard continuous-action SAC needs an appropriate adapter or an explicitly
revised action contract; discrete SAC can fit the current one.

`legacy_episode` wraps existing episode-based implementations such as UCB.
For v2, a legacy adapter must also implement `observe_reward(band, reward,
bounds)` and declare finite `reward_bounds` in its algorithm spec so it learns
from the frozen scalar reward rather than silently reverting to raw detector
hits. The supplied UCB uses the frozen v2 reward range `[-0.01, 1.0]`.
There is no PPO-specific branch in the runner. The earlier recurrent PPO example
is retained under `vyapti_simulator/system_b/tsrd/prototypes/`; it has no active spec,
trained weights or validation claim. Select the actual implementation before
writing that algorithm's settings or starting its training.

## Checkpoints and validation selection

New setups require an explicit checkpoint interval. Scheduled checkpoints are
saved after that many complete episodes; the last episode is saved once as the
final checkpoint. `--save-initial` optionally records untrained weights. Every
checkpoint has an episode count, receiver-decision count and SHA-256 in
`checkpoints/index.json`. Checkpoints and evaluation outputs are never overwritten.
The 100-episode pipeline plan with interval 25 writes weights at episodes
25, 50, 75 and 100: four checkpoints. The final checkpoint is not automatically
declared the best one.

Evaluate candidates using the same frozen VAL worlds:

```powershell
python -m scripts.evaluation.evaluate --run runs/my-run --split val --checkpoint episode_000025.json --condition all
python -m scripts.evaluation.evaluate --run runs/my-run --split val --checkpoint final.json --condition all
python -m scripts.evaluation.select_checkpoint --run runs/my-run --checkpoint final.json --reason "Selected after comparing frozen VAL_NORMAL results"
python -m scripts.evaluation.evaluate --run runs/my-run --split test --final --condition all
```

Replace the extension with the algorithm's actual format. Intermediate checkpoint
reports live under `eval/val/checkpoints/<checkpoint>/`; final reports retain
`eval/val/<condition>/`. A single `selection.json` freezes the chosen checkpoint,
VAL_NORMAL opportunity-interception metric, validation-report hash and reason.
New setups permit TEST only for that frozen choice and require `--final`.

The current corpus has 250 TRAIN, 50 VAL, and 50 TEST source files. The 50 VAL
files are the fixed checkpoint-selection set; they are not both a separate
DEV-VAL and FINAL-VAL set. TEST stays sealed until selection is frozen. See
[`docs/protocols/mode-b-evaluation.md`](../docs/protocols/mode-b-evaluation.md)
for paired world uncertainty, seed reporting, optional prediction metrics,
and the implemented oracle and frequency-agility analyses. New runs store
TRAIN-only agility thresholds and hashes before optimization. Every held-out
report includes native agility strata, a privileged expected-OIR schedule,
and separate censored TTFI comparisons.

Use `python -m scripts.evaluation.compare_algorithms run --help` for automated
three-seed pilot / five-seed final training and VAL comparison. Supply the
environment, algorithm JSON files, episode/checkpoint budgets, stage, and a
fresh output directory. Final TEST is a separate
`python -m scripts.evaluation.compare_algorithms finalize-test --output <suite>`
command for the frozen selected algorithm. The suite reports seed variation
separately from paired world CIs and prevents a second TEST run.

The existing `runs/ucb-mode-b-001` has one final checkpoint after 100 episodes
and 60,000 receiver decisions. It contains band look/hit counts, not neural
weights. Its budget was a pipeline exercise, with no intermediate checkpoint
comparison or completed VAL summary. It is not a selected scientific result.
This refactor does not manufacture missing historical checkpoints.

## Environment and evidence boundaries

`WorldComposer` selects distinct source configs uniformly, then one complete
emitter realization per source. Each trace receives one seeded integer ToA offset
uniformly in [-1,000,000,+1,000,000] us. Every pulse, relative timing, frequency
trajectory, PW, AoA and amplitude is retained. Events outside the mission remain
in the canonical stream; opportunity metrics use the mission window. Source and
offset provenance is logged per episode. Unknown type/mode/revision stays unknown.
Held-out emitter-count sampling is conditioned on counts feasible under the
50-source deduplication limit, with original distributions stored in the recipe.

The active receiver uses a Bernoulli observation: P(hit)=0.9 in a selected
window containing an in-band pulse and P(hit)=0.05 in an empty window. A band
change costs 0.3 ms; staying on the same band costs zero. The v2 scalar reward
normalizes first-intercept utility by eligible emitters, unresolved receiver
time by the 30 s mission duration, and false alarms by empty selected looks
before applying that same mission-time fraction. A positive occupied binary_v1
look credits at most the emitter with the strongest recorded in-band pulse;
one cell hit never credits all co-occurring emitters. Retune dead time is
already included in elapsed receiver resource, with no separate switch penalty.
Reward uses evaluator-side truth, but observations contain no emitter identities
or occupancy. Reward components are retained in the audit logs. Model selection
uses VAL_NORMAL pooled recorded opportunity interception ratio. The 0.3 ms
retune is the RF-switch dead time used in the cited QoS ES simulation example,
not a universal hardware specification.

Periodic visibility uses a 2 s cycle and 10% duty, with seeded emitter phases.
Stochastic visibility uses a two-state CTMC with 0.2 s mean illuminated residence
and 1.8 s mean dark residence and a stationary initial state. These are synthetic
stress assumptions. Frequency agility in the complete trace can coexist with
either visibility gate. All intervals and seeds are evaluator-only audit data.
This supports recorded-PDW scheduler development in synthetic cross-source
arrangements, not a reconstruction of global propagation or all emitted pulses.

# Training and evaluation setup

This folder contains shared experiment specifications. Results are grouped by
algorithm under `runs/`; UCB1 uses `runs/ucb1only/` and trainable Discrete SAC
uses `runs/discrete_sac/`.

## Shared TRAIN-derived HMM calibration

PPO-LSTM, Discrete SAC, periodic-belief Discrete SAC, and Belief-UCB resolve
their HMM prior and transition probabilities from
`belief_models/train250_observed_band_hmm.json`. The calibration uses only the
selected 250 TRAIN STARE files. A band-slot is active when at least one
recorded pulse falls in that 50 ms slot and within the 500 MHz half-width of a
receiver band centre. It pools adjacent-slot counts across all bands and
configs and applies Jeffreys Beta(1/2, 1/2) smoothing. Emitter labels, VAL, and
TEST are not used. These are observed recorded-pulse occupancy statistics;
they are not a claim about latent transmitter or beam-state transition laws.

Reproduce the calibration from the local TRAIN data with:

```powershell
python -m scripts.training.calibrate_train250_hmm --data-root Data --cache-root Data/cache/tsrd/train_250
```

The resolved HMM parameters are `P01=0.02219548`, `P11=0.95753186`, and
`prior_active=0.34265633`. The setup resolver fingerprints and stores the
calibration artifact in each new run's `config.json` and `runtime_manifest.json`.
Runs initialized before this calibration keep their recorded parameter values;
they must be reinitialized and retrained for a paired comparison using this
shared belief model.

```text
training_setup/
  environments/train250_composed.json       TRAIN-250 world and receiver contract
  environments/train250_source_replay.json  source-replay comparison environment
  evaluation/heldout_worlds.json            fixed 50 VAL / 50 TEST worlds
  evaluation/mode_b_protocol.json           metrics and analysis protocol
  algorithms/ucb1.json                       fixed classical UCB1 settings
  algorithms/discrete_sac.json               trainable Discrete SAC settings
```

## Run UCB1

UCB1 does not have learned weights or a training checkpoint. It selects bands
and updates its arm statistics online within each world; statistics reset for
the next world. The algorithm code and settings are fingerprinted in the run
manifest so later evaluation can verify what was run.

Initialize the algorithm-specific result directory:

```powershell
python -m scripts.evaluation.initialize_ucb1 --seed 20261003
```

The runner creates `runs/ucb1only/`. To use another independent run, pass a
different `--run runs/ucb1only-run2` directory.

Run the fixed VAL set, then freeze the baseline identity from VAL before the
sealed TEST replay:

```powershell
python -m scripts.evaluation.evaluate --run runs/ucb1only --split val --condition all
python -m scripts.evaluation.freeze_selection --run runs/ucb1only --baseline --reason "Pre-registered UCB1 baseline"
python -m scripts.evaluation.evaluate --run runs/ucb1only --split test --final --condition all
python -m scripts.evaluation.plot_run --run runs/ucb1only
```

`selection.json` is a VAL audit record. It stores the UCB1 code/settings hash
and validation report hash; it contains no weights or checkpoint.

Results are saved in separate algorithm folders:

```text
runs/ucb1only/
  config.json
  runtime_manifest.json
  status.json
  selection.json
  eval/val/<condition>/     world logs, per-world metrics, step/reward audit
  eval/test/<condition>/    same outputs for final TEST
  plots/                    summary, reward-component, and band-count figures
```

The reward used by the composed TRAIN-250 contract is:

```text
r_t = N_new/E
      - (U_before/E) * (d_t + 0.0003*s_t) / 30
      - lambda_FA * (F_t/N_empty_t) * (d_t + 0.0003*s_t) / 30
```

`d_t` is the action's actual dwell (50 or 100 ms, shortened at mission end),
and `s_t` is one for a band change. The receiver makes one binary decision per
50 ms base step, so a 100 ms dwell has two detector-evaluation opportunities.
`F_t` and `N_empty_t` count positive outcomes and truth-empty opportunities at
that granularity. The false-alarm term is zero when `N_empty_t` is zero.
`U_before` counts unresolved eligible emitters only after their first
in-scope opportunity occurred before this action began. Truth affects the
scalar reward and audit logs only; UCB1 receives the scalar reward and selected
band.

Classical UCB1 assumes stationary, equal-duration arms. Mode-B has restless
emitters and mixed dwell times, so report it as a transparent stationary
bandit control. The within-world reward and band-count figures describe online
decisions; they are not training-learning curves.

## Train an algorithm that learns weights

For a trainable algorithm, use its algorithm specification and an explicit
positive training budget. Learned algorithms save checkpoints for later VAL
selection and TEST replay; the UCB1 baseline path above does not.

```powershell
python -m scripts.training.train --environment training_setup/environments/train250_composed.json --algorithm path/to/algorithm.json --episodes 1000 --seed 42 --checkpoint-every 100 --run runs/<algorithm-name>/<run-id>
```

### Discrete SAC

Discrete SAC is implemented as a policy for Vyapti's existing public
transition interface. It does not use the standalone adapter template or
create a second world/reward/evaluation implementation. The shared runner
provides fresh TRAIN compositions, the `truth_based_intercept_utility_v2`
scalar reward, frozen VAL/TEST worlds, scorecards, oracle comparisons, and
audit logs. The SAC policy sees only its receiver-derived belief features and
the scalar reward.

Its HMM transition values (`P(inactive→active)=0.05`,
`P(active→active)=0.90`) are explicit engineering priors, not measured TSRD
statistics. The runner returns one aggregate binary result per dwell: HIT if
any 50 ms receiver evaluation was positive, otherwise NO-HIT. SAC applies an
exact two-state forward filter over each hidden base-slot transition and
conditions the end-of-dwell belief on that aggregate result. It does not
receive the internal per-slot detector outcomes. These state assumptions do
not change the reward or the evaluation scorecards.

Start a full run with the configured 400,000-action-scale settings. With the
runner's episode-based budget, 800 TRAIN episodes give approximately 400,000
band decisions; actual decisions vary with the selected bands' 50/100 ms
dwell. Checkpoints are written every 200 episodes:

```powershell
python -m scripts.training.train --environment training_setup/environments/train250_composed.json --algorithm training_setup/algorithms/discrete_sac.json --episodes 800 --seed 20261003 --checkpoint-every 200 --run runs/discrete_sac
```

This is a long CPU workload; the policy performs neural updates after 5,000
warm-up actions. After training, evaluate each saved checkpoint on VAL_NORMAL
and select the checkpoint with the highest pooled OIR. The checkpoint index
lists the exact filenames. This PowerShell loop runs VAL_NORMAL for every
candidate and stops if any evaluation fails:

```powershell
$checkpoints = (Get-Content runs/discrete_sac/checkpoints/index.json | ConvertFrom-Json).checkpoints |
    Where-Object { $_.kind -ne "initial" }
foreach ($checkpoint in $checkpoints) {
    python -m scripts.evaluation.evaluate --run runs/discrete_sac --checkpoint $checkpoint.file --split val --condition normal
    if ($LASTEXITCODE -ne 0) { throw "VAL failed for $($checkpoint.file)" }
}
```

Choose the checkpoint with the highest pooled OIR in its VAL_NORMAL
`summary.json`. Freeze that checkpoint, then run the sealed TEST set on all
three conditions. Beam VAL runs are optional diagnostics and are not required
for the current selection rule. For example, if `final.pt` wins:

```powershell
python -m scripts.evaluation.evaluate --run runs/discrete_sac --checkpoint final.pt --split val --condition normal
python -m scripts.evaluation.freeze_selection --run runs/discrete_sac --checkpoint final.pt --reason "Discrete SAC checkpoint selected after VAL"
python -m scripts.evaluation.evaluate --run runs/discrete_sac --split test --condition all --final
python -m scripts.evaluation.plot_run --run runs/discrete_sac
```

The checkpoint and all SAC results stay under `runs/discrete_sac/`. VAL and
TEST summaries, per-world scorecards, complete step logs, oracle comparisons,
and plots use the same runner layout as UCB1. This policy has no prediction
head, so prediction metrics are omitted by the existing conditional rule.

### Recurrent PPO-LSTM

PPO-LSTM uses the same public `PublicTransition` interface and shared runner as
SAC. It keeps a recurrent state within each episode and resets it at the next
world. Its 253 features retain receiver-only belief, staleness, causal
periodicity cues, previous band, and mission time, and add native dwell and
time since observed HIT per band. Reward is used only by PPO to calculate
returns and advantages; it is not included in the policy observation. The
observation contains no emitter identity or hidden truth.

The HMM prior and transitions come from the shared TRAIN-250 observed-band
calibration (`P01≈0.02220`, `P11≈0.95753`, initial activity `≈0.34266`), not
the older engineering defaults. As with SAC, one aggregate HIT/NO-HIT is
filtered over latent base-slot transitions. The learning-rate schedule uses
the configured training-action budget (400,000 by default), so mixed 50/100 ms
dwell choices set a consistent decay clock. The learning rate decays linearly
from 2e-4 to zero over that action budget.

```powershell
python -m scripts.training.train --environment training_setup/environments/train250_composed.json --algorithm training_setup/algorithms/ppo_lstm.json --episodes 800 --seed 20261003 --checkpoint-every 200 --run runs/ppo_lstm
```

Evaluate the saved checkpoints on VAL_NORMAL, select the highest pooled OIR,
freeze that checkpoint, and run all TEST conditions:

```powershell
python -m scripts.evaluation.evaluate --run runs/ppo_lstm --checkpoint final.pt --split val --condition normal
python -m scripts.evaluation.freeze_selection --run runs/ppo_lstm --checkpoint final.pt --reason "PPO-LSTM checkpoint selected by VAL_NORMAL OIR"
python -m scripts.evaluation.evaluate --run runs/ppo_lstm --split test --condition all --final
python -m scripts.evaluation.plot_run --run runs/ppo_lstm
```

For selection among scheduled checkpoints, evaluate each indexed `.pt` file
on VAL_NORMAL as shown in the Discrete SAC instructions. PPO-LSTM has no
prediction head, so its prediction metrics are omitted.

### Discrete SAC with belief and periodicity

This policy is registered in
`training_setup/algorithms/discrete_sac_belief_periodic.json` and implements
the same public transition interface as the other trainable policies. It
builds causal HMM belief, staleness, periodicity score/confidence, previous
band, and remaining-time features from receiver observations. The common
runner still owns worlds, reward, held-out evaluation, and metrics.

```powershell
python -m scripts.training.train --environment training_setup/environments/train250_composed.json --algorithm training_setup/algorithms/discrete_sac_belief_periodic.json --episodes 800 --seed 20261003 --checkpoint-every 200 --run runs/discrete_sac_belief_periodic
```

The run, checkpoints, receiver logs, per-world scorecards, and plots are saved
under `runs/discrete_sac_belief_periodic/`. Select a checkpoint using
VAL_NORMAL, freeze that selection, then run the final TEST conditions using
the same evaluation commands as the other learned policies. HMM transition
probabilities are explicit engineering priors, not measured TSRD activity
statistics.

### Belief-state MCTS

Belief-MCTS is a separate trainable algorithm. It keeps the supplied short-
horizon MCTS, UCT selection, causal HMM and periodicity state, and Bayesian
linear reward model. Hypothetical branches sample only aggregate HIT/MISS
from the receiver-only belief; the model learns from the real shared
`transition.reward`. It does not query hidden worlds or alter the reward.
The environment supplies its exact native dwell profile, and the HMM uses
the shared TRAIN-250 calibration artifact.

The reward-model input has 55 causal features: the original 46 receiver and
belief features plus candidate visit/hit history and global decision, time,
reward, and hit history. MCTS weights HIT and MISS continuation values by
their belief-derived probabilities. The selected branch is deepened while
the other receives a one-step causal rollout; per-episode planning latency is
written to the algorithm update log and evaluation world records.

The default experiment uses 128 simulations and a four-decision
horizon. Measure planning latency on the intended hardware. For a budget
ablation, compare 128, 256, 512, and 1024 simulations on the same frozen VAL
catalog before making performance claims.

```powershell
python -m scripts.training.train --environment training_setup/environments/train250_composed.json --algorithm training_setup/algorithms/belief_mcts.json --episodes 800 --seed 20261003 --checkpoint-every 200 --run runs/belief_mcts
```

Evaluate candidate checkpoints on VAL_NORMAL, freeze the selected checkpoint,
then run the final TEST conditions as for PPO-LSTM and SAC.

### Contextual Thompson Sampling

`contextual_thompson` is a separate trainable linear-Gaussian Thompson
Sampling policy. It uses a 50-feature causal context with TRAIN-250 HMM
belief, observed-HIT recurrence cues, and candidate-dependent deadline
interactions. The recurrence cue reflects receiver observations and does not
identify an emitter scan period. Its posterior learns from the
unchanged shared `transition.reward`; it does not add an intrinsic reward.
The shared setup injects calibrated HMM values and the environment's exact
native dwell vector.

```powershell
python -m scripts.training.train --environment training_setup/environments/train250_composed.json --algorithm training_setup/algorithms/contextual_thompson.json --episodes 800 --seed 20261003 --checkpoint-every 200 --run runs/contextual_thompson
```

The run follows the normal checkpoint and frozen-world VAL/TEST workflow.
Evaluation reports its per-world band counts alongside the standard metrics.
The baseline uses literal posterior sampling (`thompson_sampling_scale=1.0`);
keep that setting configurable for TRAIN/VAL-only scale comparisons.
Evaluation logs may include a causal ETA to the next observed-HIT recurrence.
That cue is not an emitter-intercept-time prediction. The shared evaluator
therefore does not score it as intercept error; a comparable intercept target
must be frozen before reporting that metric. OIR-ratio prediction is deferred
until its target and scoring rule are specified.

### Belief-UCB

Belief-UCB is a no-pretraining online baseline, separate from classical UCB1.
It chooses a band using the receiver-only predicted HIT probability plus a
visit-count exploration bonus. The first experiment uses `exploration_c=0.25`
and warms up by visiting every band once. It updates its HMM belief from the
aggregate receiver HIT/MISS and actual one- or two-slot dwell; it does not use
the scalar reward to select actions. The HMM transition probabilities are
explicit engineering assumptions, not TSRD-estimated activity statistics.

```powershell
python -m scripts.evaluation.initialize_belief_ucb --run runs/belief_ucb
python -m scripts.evaluation.evaluate --run runs/belief_ucb --split val --condition normal
python -m scripts.evaluation.freeze_selection --run runs/belief_ucb --baseline --reason "Pre-registered Belief-UCB baseline"
python -m scripts.evaluation.evaluate --run runs/belief_ucb --split test --condition all --final
python -m scripts.evaluation.plot_run --run runs/belief_ucb
```

### Round Robin

Round Robin is a fixed online control with no learned weights. It visits bands
in order (0 through 35, then wraps), restarting at band 0 for each world. The
shared runner records its scorecards, per-world selections, reward components,
and plots under `runs/round_robin/`.

It does not need VAL for training or checkpoint selection. The evaluation
protocol still requires one VAL_NORMAL replay and a frozen baseline record
before revealing TEST. This is an audit gate; it does not tune Round Robin.
Then run all three TEST conditions once:

```powershell
python -m scripts.evaluation.initialize_round_robin --run runs/round_robin
python -m scripts.evaluation.evaluate --run runs/round_robin --split val --condition normal
python -m scripts.evaluation.freeze_selection --run runs/round_robin --baseline --reason "Pre-registered Round Robin baseline"
python -m scripts.evaluation.evaluate --run runs/round_robin --split test --condition all --final
python -m scripts.evaluation.plot_run --run runs/round_robin
```

### PPO-LSTM receiver-only coverage pilot

`ppo_lstm_receiver_only.py` is a separate PPO-LSTM implementation for the receiver-only
325-feature observation and eight-world recurrent rollouts. It retains the
shared environment, receiver, scalar reward, and public transition boundary.
Reward is used only by PPO's return/advantage calculation and is not fed into
the next observation. The coverage prior is added to action logits during
both collection and PPO updates. It is based on receiver-derived staleness and
visit counts, capped at one, with `coverage_kappa` set by the algorithm file.
The learning rate decays against receiver base slots, with a 480,000-slot
full-run target. Each training episode samples a fresh TRAIN world and
discards it after the episode. Evaluation uses the existing frozen 50-world
VAL and 50-world TEST recipes.

The pilot configs compare gamma/lambda A (`0.997/0.98`), B (`0.998/0.990`),
and C (`0.999/0.995`) with `coverage_kappa=0.10`. The separate coverage screen
compares `kappa=0.05` and `0.15` against the A baseline. These are pilot
comparisons, not final multi-seed claims. Use the same training seed and 80
episodes (48,000 receiver base slots) for each candidate; evaluate VAL_NORMAL
only while choosing settings. The existing `runs/ppo_lstm_v2/` directory and
its checkpoints are a previous run and are not overwritten.

For example, run gamma A:

```powershell
python -u -m scripts.training.train --environment training_setup/environments/train250_composed.json --algorithm training_setup/algorithms/ppo_lstm_receiver_only.json --episodes 80 --seed 42 --checkpoint-every 80 --run runs/ppo_lstm_receiver_only_gamma_a_pilot
python -u -m scripts.evaluation.freeze_worlds --run runs/ppo_lstm_receiver_only_gamma_a_pilot
python -u -m scripts.evaluation.evaluate --run runs/ppo_lstm_receiver_only_gamma_a_pilot --split val --condition normal --checkpoint final.pt
```

Run gamma B/C and the two kappa screen configs with the same seed and episode
budget, using a new run directory for each. `freeze_worlds` reuses a compatible
existing recipe catalog when the world-defining setup matches. Compare
unique-emitter interception rate first, then restricted-mean TTFI and censoring,
coverage/staleness, OIR, and reward per decision. Use the recorded per-world
VAL results as paired comparisons. Do not run TEST during this screen. After
choosing a configuration and checkpoint, freeze its VAL selection; the
selection record stores the supporting VAL metrics and priority order. TEST is
then run once, using the frozen selection.

Training logs print progress every 10 episodes. `plots/` includes PPO loss,
entropy and KL curves, training reward-component curves, band-selection
histograms, and VAL checkpoint metric curves after each candidate checkpoint
has been evaluated. Evaluation reports include deadline interception rates,
rolling/whole-mission coverage, band staleness, action distribution, and
reward totals normalized by decision and receiver base step.

After selecting the pilot configuration using VAL_NORMAL, run the full budget
with a new results directory. The final run saves checkpoints every 200 worlds;
evaluate each on the same VAL_NORMAL worlds before freezing one selection:

```powershell
python -u -m scripts.training.train --environment training_setup/environments/train250_composed.json --algorithm training_setup/algorithms/ppo_lstm_receiver_only.json --episodes 800 --seed 42 --checkpoint-every 200 --run runs/ppo_lstm_receiver_only
python -u -m scripts.evaluation.freeze_worlds --run runs/ppo_lstm_receiver_only
python -u -m scripts.evaluation.evaluate --run runs/ppo_lstm_receiver_only --split val --condition normal --checkpoint episode_000200.pt
python -u -m scripts.evaluation.evaluate --run runs/ppo_lstm_receiver_only --split val --condition normal --checkpoint episode_000400.pt
python -u -m scripts.evaluation.evaluate --run runs/ppo_lstm_receiver_only --split val --condition normal --checkpoint episode_000600.pt
python -u -m scripts.evaluation.evaluate --run runs/ppo_lstm_receiver_only --split val --condition normal --checkpoint final.pt
python -m scripts.evaluation.freeze_selection --run runs/ppo_lstm_receiver_only --checkpoint episode_000400.pt --reason "Selected by the registered receiver-only VAL metric priority"
python -u -m scripts.evaluation.evaluate --run runs/ppo_lstm_receiver_only --split test --condition all --final
python -m scripts.evaluation.plot_run --run runs/ppo_lstm_receiver_only
```

`freeze_selection` verifies the stated priority against all completed greedy
VAL_NORMAL checkpoint reports, so it cannot freeze a lower-priority checkpoint
just because it has higher OIR. Replace the example `episode_000400.pt` with
the top checkpoint from that ordering. The current
local protocol has 50 VAL and 50 TEST worlds; TEST remains unopened during the
pilot and checkpoint selection.

### Recurrent Distributional DQN

`recurrent_distributional_dqn` is a trainable, single-learner recurrent
distributional Double-Dueling DQN with NoisyNet exploration. It uses the
shared 325-feature receiver-only state, calibrated TRAIN-250 HMM, native
one/two-slot dwell schedule, environment-owned reward, and common train/VAL/
TEST runner. Its prioritized replay assigns priorities to sequence starts,
stores terminal successor observations, and pads/masks shorter windows at
episode boundaries. Each sampled window uses up to 32 no-gradient recurrent
burn-in steps followed by 64 learning positions and five-step targets, so
initial and terminal decisions can receive loss. The target network updates
every 1,000 learner updates (about every 125 episodes at eight updates per
episode); monitor TD error and VAL progress before changing that interval.
Replay learning begins only after eight stored episodes each contain at least
101 decisions and at least 128 sequence starts are available. PER beta moves
from 0.6 to 1.0 across 480,000 receiver base steps. Evaluation reports
receiver-derived visit counts and recent hit rates alongside spectrum
coverage diagnostics.
It is R2D2/Rainbow-inspired, not a distributed or paper-exact R2D2
implementation. Evaluation uses fixed midpoint IQN quantiles, disables
NoisyNet noise, and selects the action with the highest mean Q value. Its IQN
cosine embedding uses frequencies 0 through 63, following the original IQN
definition. The observation normalizes both staleness and time-since-hit to
four 43-slot sweeps; diagnostics retain raw, unclipped age in receiver slots.

Train a fresh run; the non-versioned checkpoint identity and 325-D schema
reject the earlier 253-D checkpoints:

```powershell
python -m scripts.training.train --environment training_setup/environments/train250_composed.json --algorithm training_setup/algorithms/recurrent_distributional_dqn.json --episodes 800 --seed 20261003 --checkpoint-every 200 --run runs/recurrent_distributional_dqn
```

Freeze this run's recipes and evaluate saved checkpoints on the same VAL_NORMAL
worlds. Select by unique-emitter interception, restricted-mean TTFI, 90% band
coverage rate, the deadline curve, OIR, then environment reward; Pd/Pfa are
receiver sanity checks. Run TEST only after selection is frozen:

```powershell
python -m scripts.evaluation.freeze_worlds --run runs/recurrent_distributional_dqn
python -m scripts.evaluation.evaluate --run runs/recurrent_distributional_dqn --split val --condition normal --checkpoint episode_000200.pt
python -m scripts.evaluation.evaluate --run runs/recurrent_distributional_dqn --split val --condition normal --checkpoint episode_000400.pt
python -m scripts.evaluation.evaluate --run runs/recurrent_distributional_dqn --split val --condition normal --checkpoint episode_000600.pt
python -m scripts.evaluation.evaluate --run runs/recurrent_distributional_dqn --split val --condition normal --checkpoint final.pt
python -m scripts.evaluation.freeze_selection --run runs/recurrent_distributional_dqn --checkpoint episode_000400.pt --reason "Selected by the registered VAL metric priority"
python -m scripts.evaluation.evaluate --run runs/recurrent_distributional_dqn --split test --condition all --final
python -m scripts.evaluation.plot_run --run runs/recurrent_distributional_dqn
```

Replace the example checkpoint in `freeze_selection` with the checkpoint that
actually has the highest VAL_NORMAL pooled OIR. Keep TEST sealed until that
choice is frozen. Checkpoints are currently for evaluation and selection;
replay state is not saved, so interrupted training cannot be resumed faithfully.

Keep world seeds, interaction budgets, receiver settings, held-out worlds, and
metrics matched when comparing algorithms. See
[`docs/protocols/mode-b-evaluation.md`](../docs/protocols/mode-b-evaluation.md)
for paired uncertainty, optional prediction metrics, oracle comparisons, and
TRAIN-derived frequency-agility strata.
## Frozen-world recipes and combined stress reporting

Run initialization writes `runs/ucb1only/frozen_world_catalog.json`. It contains only recipes, source-file hashes, seeds, illumination-mask hashes, and replay signatures. It does not save full worlds or run the policy on TEST. TRAIN worlds are generated per episode and discarded.

Other algorithm runs with the same data path, world and receiver settings, VAL/TEST recipes, seeds, and illumination conditions reuse this verified catalog. The freeze command reports reuse immediately; when it must build a new catalog, it prints progress every 10 worlds. A run with changed world-defining settings gets its own catalog.

The conditions are `normal`, `beam_periodic`, and `beam_stochastic`. Each condition is also stratified into low, medium, and high frequency agility using thresholds fit on TRAIN. `spatial_agility_matrix.json` and `plots/{split}_spatial_agility_matrix.png` report the combined condition-by-agility results. Beam conditions are controlled visibility overlays on recorded STARE PDWs because the source data does not contain physical beam-state truth.

If evaluation stops partway through, retrying will clear and rebuild that
incomplete condition directory. A completed directory containing
`summary.json` is protected from accidental overwrite. The final TEST command
requires a completed VAL_NORMAL selection and verifies the frozen catalog,
source hashes, seeds, visibility masks, and rebuilt-world signatures.

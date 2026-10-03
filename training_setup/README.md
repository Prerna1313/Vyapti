# Training and evaluation setup

This folder contains shared experiment specifications. Results are grouped by
algorithm under `runs/`; UCB1 uses `runs/ucb1only/` and trainable Discrete SAC
uses `runs/discrete_sac/`.

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

Keep world seeds, interaction budgets, receiver settings, held-out worlds, and
metrics matched when comparing algorithms. See
[`docs/protocols/mode-b-evaluation.md`](../docs/protocols/mode-b-evaluation.md)
for paired uncertainty, optional prediction metrics, oracle comparisons, and
TRAIN-derived frequency-agility strata.
## Frozen-world recipes and combined stress reporting

Run initialization writes `runs/ucb1only/frozen_world_catalog.json`. It contains only recipes, source-file hashes, seeds, illumination-mask hashes, and replay signatures. It does not save full worlds or run the policy on TEST. TRAIN worlds are generated per episode and discarded.

The conditions are `normal`, `beam_periodic`, and `beam_stochastic`. Each condition is also stratified into low, medium, and high frequency agility using thresholds fit on TRAIN. `spatial_agility_matrix.json` and `plots/{split}_spatial_agility_matrix.png` report the combined condition-by-agility results. Beam conditions are controlled visibility overlays on recorded STARE PDWs because the source data does not contain physical beam-state truth.

If evaluation stops partway through, retrying will clear and rebuild that
incomplete condition directory. A completed directory containing
`summary.json` is protected from accidental overwrite. The final TEST command
requires a completed VAL_NORMAL selection and verifies the frozen catalog,
source hashes, seeds, visibility masks, and rebuilt-world signatures.

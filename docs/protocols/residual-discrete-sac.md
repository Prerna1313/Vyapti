# Residual Discrete SAC

The uploaded conservative residual Discrete-SAC learner is installed at
`vyapti_simulator/system_b/tsrd/policies/residual_discrete_sac.py`. Its SAC
actor, critics, replay, and SAC mission-reward target are retained. Contextual
TS feedback is explicitly the observed HIT/MISS rate, with a fresh checkpoint
schema so posteriors trained on mission reward cannot be reused. Its evaluation
posterior now updates online within each episode.

The actor evaluates exactly ten distinct candidate bands: TS sample, TS mean,
three least-visited bands, and one proposal each from TS uncertainty,
periodicity, belief, staleness and operational priority. Duplicate TS mean
proposals leave a slot that is filled by the causal fallback ranking. Equal
visit counts are ordered by oldest raw visit time. Candidate descriptors have
18 fields, including an explicit unvisited flag; the receiver state remains
325-dimensional. Old 17-field checkpoints are rejected by the candidate-schema
guard and require fresh training.

Evaluation uses actor argmax directly; critic advantages are diagnostics and
never force a TS fallback. TS remains candidate zero and adapts to receiver
HIT/MISS feedback. Candidate inclusion does not enforce coverage. There is no
external coverage floor in this configuration. The training prior mixture
decays from 0.40 to 0.05 over 25,000 actions, including warm-up; the actor's
prior-logit bias decays from 0.50 to zero over the same budget. Batch size 128
and one gradient update per action after warm-up are retained.

Use a fresh 25,000-action pilot and evaluate VAL_NORMAL before extending the
budget. The corpus contains 50 VAL and 50 TEST worlds. Reuse Round Robin's
catalog and receiver seeds for comparison.

`scripts/training/train_residual_discrete_sac.py` provides its dedicated
TRAIN-250 entry point. It samples a fresh TRAIN world per episode and uses the
shared `truth_based_intercept_utility_v2` reward for SAC learning. The
contextual Thompson component is updated separately from the mean of the
receiver-observed HIT/MISS outcomes in each action; it never receives the
truth-based mission reward. Evaluation starts each world from the learned TS
posterior and continues this causal HIT/MISS adaptation within the episode.
Evaluation policy sampling uses a fixed `--eval-seed-base` (default `420000`),
independent of the training `--seed`; the evaluator checks that it matches the
value recorded for the run. This seed controls the Thompson sampler's random
draws. Receiver randomness itself remains pinned by the shared frozen-world
catalog, because those per-world receiver seeds are part of its replay hashes.
Thus all algorithms using that catalog get identical receiver noise and the
same receiver seed for each world. Evaluation is handled by
`scripts/evaluation/evaluate_residual_discrete_sac.py`; it deterministically
rebuilds and hash-verifies each VAL/TEST world against the supplied frozen
catalog, then uses the shared recorded-replay scorecard.

This learner has its own checkpoint format and evaluation commands. Do not
pass its checkpoints to `scripts.training.train`, `scripts.evaluation.evaluate`,
or the generic `freeze_selection` command.

## Local run

From the repository root:

```powershell
python -u -m scripts.training.train_residual_discrete_sac `
  --environment training_setup/environments/train250_composed.json `
  --world-catalog runs/round_robin/frozen_world_catalog.json `
  --seed 20261004 --eval-seed-base 420000 --max-actions 400000 --device auto `
  --run runs/residual_discrete_sac
```

For Colab, use the prepared `train250_colab.json`, save `frozen_world_catalog.json`
from the Round Robin run to Drive, and set `--run` to a new Drive directory.
Training progress prints every 25 completed episodes and at checkpoint saves.

Evaluate each checkpoint on `VAL_NORMAL`, then select the highest pooled OIR:

```powershell
python -u -m scripts.evaluation.evaluate_residual_discrete_sac `
  --run runs/residual_discrete_sac --world-catalog runs/round_robin/frozen_world_catalog.json `
  --split val --condition normal --checkpoint residual_sac_step_050000.pt `
  --eval-seed-base 420000

python -m scripts.evaluation.freeze_residual_discrete_sac_selection `
  --run runs/residual_discrete_sac
```

After selection, evaluate the selected checkpoint on TEST with each frozen
condition (`normal`, `beam_periodic`, `beam_stochastic`) and `--final`. The
evaluator rejects TEST if the selected checkpoint or frozen catalog differs.

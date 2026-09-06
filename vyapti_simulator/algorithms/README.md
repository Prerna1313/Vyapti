# `algorithms/` — NOT a deliverable of the common simulator

**Ownership: algorithm team. Not maintained as part of the simulator.**

This directory is outside the boundary of the common simulator. It exists only
because the simulator needs *something* to execute during its own bring-up, and
because a worked example of the `BaseScheduler` contract is faster to read than
a specification.

Nothing in `vyapti_simulator/core/`, `config/`, `protocol/`, `tsrd/`,
`experiments/`, `visualization/` or `qualification/` imports anything from this
directory. The dependency runs one way only:

```
algorithms/*  ---imports--->  core.scheduler_interface.BaseScheduler
core/*        ---imports--->  (nothing here)
```

You can delete this entire directory and the simulator, its metrics, its
statistical protocol, its gate enforcement and its figure generation all
continue to work. Verify with:

```bash
python -m vyapti_simulator.qualification.conformance --self-test
```

## What lives here, and what to do with it

| File | Status |
|---|---|
| `baselines/fixed.py` | Superseded — use `qualification/probes.py` instead |
| `baselines/clarkson.py` | Reference only. Algorithm team owns the real version |
| `bandit/ucb.py` | Reference only |
| `bandit/thompson.py` | Reference only |
| `bandit/espe.py` | Reference only |
| `bandit/rising.py` | Reference only |
| `bandit/coverage_constrained.py` | Reference only |
| `bandit/periodicity.py` | **Validated utility** — see below |

### `bandit/periodicity.py` is worth reading before you write your own

It is the one file here with a result attached. It recovers emitter scan period
exactly (16/16 test cases: periods 8, 10, 12, 20 at dwell rates 100% down to
12%) and declines to fit on aperiodic input (0 false positives in 4 trials).

It does that by scoring candidate `(period, phase, width)` models against
**observed misses as well as hits**. This matters more than it sounds: the
obvious estimators both fail badly here, and the failure is silent.

- `median(diff(hit_slots))` is biased by the sampling. If you only dwell on a
  band every third visit, observed intervals are *multiples* of the true
  period, and the estimate is wrong by that factor. The bias is worst for the
  bands you watch least, which is exactly where you need the estimate.
- Phase folding scored on hits alone selects **divisors** of the true period.
  Folding a period-10 signal at p=5 puts every hit at phase 0 — a perfect
  score. At p=1 *every* time series scores perfectly. The usual
  "prefer the smallest high-scoring period" harmonic-suppression rule walks
  directly into this and reports p=1, "always active", for every band.

A miss refutes both. If your own estimator does not use negative evidence,
please check it against these two cases before trusting its output.

## The contract your schedulers must satisfy

Implement `vyapti_simulator.core.scheduler_interface.BaseScheduler`:

- `reset(seed, scenario_config)` — must fully clear state; use your own
  `np.random.default_rng(seed)`, never `np.random.seed()` (it is global state
  shared with the environment and will break paired comparison).
- `select_action(observation_history, current_time_slot) -> int`
- `update(action, observation)`
- `predict(current_time_slot) -> BandPrediction | None` — **optional.** Override
  it only if your policy genuinely forecasts. PS metrics 6 and 7 (percentage of
  correct predictions, average intercept time error) are computed from it and
  are reported as *unavailable* when it returns `None`. Do not return a
  placeholder to make the metric appear: an absent forecast reported as 0.0 is
  indistinguishable in a results table from a measured zero.

Before submitting results, your scheduler must pass:

```bash
python -m vyapti_simulator.qualification.conformance --scheduler your.module:YourClass
```

That checks the Gate 0 information boundary, deterministic replay under a fixed
seed, action-range validity, and reset hygiene. Results from a scheduler that
has not passed conformance are not admissible under the frozen protocol.

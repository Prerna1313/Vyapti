# Vyapti — PS26055 Electronic Warfare Scheduler Simulator

A pulse-level RF simulator for evaluating multi-armed bandit scheduling algorithms against
realistic electronic warfare (EW) emitter scenarios, calibrated against the
**Turing Synthetic Radar Dataset** (TSRD, arXiv:2602.03856, Apache-2.0).

---

## What this is

Vyapti is a **research-grade EW scheduler comparison platform** that:

1. **Simulates realistic RF emitter dynamics** — 13 emitter behaviour types (fixed, scanning,
   frequency-agile, PRI-jitter, dynamic, etc.) with validated RF parameter ranges
   (500 MHz–18 GHz, PRI 50 μs–10 ms, PW 0.2–10 μs, power −100 to −30 dBm).
2. **Implements channel physics** — free-space path loss, log-normal shadowing (σ=8 dB),
   Rayleigh fast fading, atmospheric attenuation, coherent integration gain (10·log₁₀ N dB).
3. **Shares a detection model** between synthetic (System A) and TSRD-driven (System B) paths —
   the same logistic SNR→Pd curve is applied on both, so synthetic and real-TSRD experiments
   are directly comparable.
4. **Enforces the PS26055 Frozen Protocol v1.0** — 8 gate checkpoints, mandatory result-tagging
   (7 fields per result row), oracle-free scheduler observation contract, and negative-result
   reporting discipline.
5. **Computes 7 DRDO-named figures of merit** with proper unavailable/None semantics and
   right-censored survival analysis (Kaplan-Meier).
6. **Runs fully reproducible paired comparisons** — SeedSequence tree, non-overlapping
   train/eval/test seed split, scipy-backed non-parametric statistics (Wilcoxon, Friedman,
   Holm-Bonferroni, Cliff's delta, bootstrap CI).

---

## Installation

```bash
pip install -e .
```

Requires: Python 3.10+, numpy (>=1.21), scipy (>=1.7), matplotlib (>=3.5), h5py (>=3.0).

---

## Quick start

### Certify your scheduler

```bash
python -m vyapti_simulator.qualification.conformance \
  --scheduler your.module:YourScheduler \
  --bands 10 --slots 200
```

### Run a paired comparison

```python
from vyapti_simulator.experiments import ExperimentConfig, ExperimentRunner
from vyapti_simulator.algorithms.bandit.ucb import UCB1Scheduler
from vyapti_simulator.qualification.probes import RoundRobinProbe

config = ExperimentConfig(
    train_seeds=list(range(0, 80)),   # 80 paired seeds
    band_count=10,
    time_slots=1000,
)
runner = ExperimentRunner(config)

report = runner.run_paired_comparison(
    schedulers={
        "UCB1":        lambda: UCB1Scheduler(band_count=10),
        "RoundRobin":  lambda: RoundRobinProbe(band_count=10),
    },
    density=20,
)
```

Statistical tests (Wilcoxon, Cliff's delta, bootstrap CI), Kaplan-Meier survival
curves, and all 7 PS metrics are in `report`.

### Run on real TSRD data (Kaggle)

Attach the TSRD dataset (`alan-turing-institute/turing-synthetic-radar-dataset`) as
a Kaggle input, then:

```python
from vyapti_simulator.tsrd.corpus_loader import TSRDCorpusLoader
from vyapti_simulator.tsrd.tsrd_environment import TSRDEnvironment, DetectionConfig
from vyapti_simulator.tsrd.pdw_discretiser import discretise_pdw_to_grid

loader = TSRDCorpusLoader(corpus_dir="/kaggle/input/...", require_manifest=True)
for rec in loader.iterate():
    if rec.receiver_mode.value != "scanning":
        continue  # Stare mode → oracle-only, not for scheduler path
    grid = discretise_pdw_to_grid(rec.pdw_stream, n_bands=36)
    env = TSRDEnvironment(n_bands=36, grid=grid)
    # run your scheduler ...
```

---

## Architecture overview

```
vyapti_simulator/
├── algorithms/        Bandit and reference schedulers
├── config/            Master config with provenance labels
├── core/              PS26055Environment, metrics engine, mapping
├── experiments/       ExperimentConfig, ExperimentRunner, statistics
├── protocol/          FrozenProtocolEnforcer (Gates 0–7, F, G)
├── qualification/     Conformance checks (C0–C11), probe schedulers
├── tsrd/              TSRD adapter, corpus loader, deinterleaver, discretiser
├── visualization/     12 mandatory publication figures
└── simulator.py       CLI entry point
src/
├── rf_pulse_simulator.py   System A: pulse-level synthetic RF
└── emitter_models.py        7 emitter classes + channel physics
```

---

## Receiver constraints

| | Protocol default (§3 Stage 0) | Research default (all experiments) |
|---|---|---|
| IBW | 200 MHz | 500 MHz |
| Dwell | 10 ms | 50 ms |
| Retune | 1 ms | 1 ms |
| Bands | 10 | 36 |

Both are valid per the frozen protocol's research-platform extension clause.

---

## Key constraints for algorithm teams

- Schedulers **never** see ground truth. Gate 0 (`check_hidden_state_leakage`) audits every `select_action` call.
- Use `np.random.default_rng(seed)` in `reset()`, never `np.random.seed()` or `np.random.rand()`.
- Results from uncertified schedulers are **not admissible** under the frozen protocol.
- Sub-problems F (threat prioritization) and G (multi-objective reward) are **blocked** until team memos are written.

---

## References

- **Frozen Protocol:** `PS26055_Common_Simulation_and_Evaluation_Protocol_v1.0_FROZEN.md`
- **TSRD:** arXiv:2602.03856, Apache-2.0
- **Wilcoxon / Cliff's delta:** scipy.stats, Romano et al. (2006)
- **Kaplan-Meier:** Kaplan & Meier (1958)

---

## Scientific standards enforced

1. Information boundary — scheduler never receives truth fields
2. Paired comparison — same seeds, identical truth grids
3. Deterministic replay — `reset(seed)` twice → identical decisions
4. Non-parametric statistics — Wilcoxon/Friedman (ordinal discovery metrics)
5. Multiple-testing correction — Holm-Bonferroni for pairwise comparisons
6. Right-censored survival analysis — Kaplan-Meier for first-intercept time
7. Negative-result reporting — p ≥ 0.05 is a result, not a failure
8. Provenance — every parameter tagged `[PS-DEFINED]` / `[LITERATURE-GROUNDED]` /
   `[TSRD-DERIVED]` / `[ENGINEERING-ASSUMPTION]` / `[EXPERIMENTAL-VARIABLE]`

## ?? TSRD Metrics Documentation
If you are evaluating custom schedulers against the Turing Synthetic Radar Dataset (TSRD), please see the new [TSRD Metrics Usage Guide](TSRD_METRICS_USAGE_GUIDE.md) for full instructions on imports, instantiation, and metrics extraction.

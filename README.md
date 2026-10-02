# Vyapti Simulator

## Three separate simulation systems

- **[System A](vyapti_simulator/system_a/README.md)** —
  `vyapti_simulator/system_a/synthetic_pdw/`: controlled synthetic PDW
  generation for development and examples.
- **[System B](vyapti_simulator/system_b/README.md)** — `vyapti_simulator/system_b/tsrd/`: recorded TSRD Scan/Stare
  PDWs, TRAIN-250 cache, world composition, scheduler training and frozen
  evaluation using 50 VAL and 50 TEST sources.
- **[System C](vyapti_simulator/system_c/README.md)** — `vyapti_simulator/system_c/emitters/` and `vyapti_simulator/system_c/rf/`:
  synthetic emitters, RF physics, IQ generation and CFAR detection.
- **Shared support** — `vyapti_simulator/core/`, scheduler algorithms, metrics,
  qualification and experiment utilities. Each system has its own data source
  and receiver assumptions.

New imports use `vyapti_simulator.system_a`, `vyapti_simulator.system_b`, or
`vyapti_simulator.system_c`. Shared interfaces remain under `vyapti_simulator/core/`.
See [training_setup/README.md](training_setup/README.md) for the current
model-neutral training interface. Data and run outputs keep their existing paths.
See [the scorecard naming order](docs/protocols/scorecard-order.md) for the
recommended result presentation hierarchy.

For the current code and documentation status, see [REPOSITORY_STATUS.md](REPOSITORY_STATUS.md).
Older project notes are historical snapshots; test results and implementation details in
those notes may no longer describe this checkout.

A pulse-level RF simulator for evaluating multi-armed bandit scheduling algorithms against
realistic electronic warfare (EW) emitter scenarios, calibrated against the
**Turing Synthetic Radar Dataset** (TSRD, arXiv:2602.03856, Apache-2.0).

---

## What this is

Vyapti is a research prototype with modules that:

1. **Simulates realistic RF emitter dynamics** — 13 emitter behaviour types (fixed, scanning,
   frequency-agile, PRI-jitter, dynamic, etc.) with validated RF parameter ranges
   (500 MHz–18 GHz, PRI 50 μs–10 ms, PW 0.2–10 μs, power −100 to −30 dBm).
2. **Implements channel physics** — free-space path loss, log-normal shadowing (σ=8 dB),
   Rayleigh fast fading, atmospheric attenuation, coherent integration gain (10·log₁₀ N dB).
3. **Provides receiver models for each path** — the synthetic pulse/slot path and
   TSRD grid path can share amplitude detector settings; the recorded-PDW
   training receiver and IQ CFAR detector have separate assumptions. Comparisons
   must identify the active receiver model.
4. **Implements protocol checks** — 8 gate checkpoints, result-tagging
   (7 fields per result row), oracle-free scheduler observation contract, and negative-result
   reporting discipline.
5. **Reports requested figures of merit** with unavailable/None semantics and
   right-censored survival analysis (Kaplan-Meier).
6. **Supports paired seeded comparisons** — SeedSequence tree, non-overlapping
   train/eval/test seed split, scipy-backed non-parametric statistics (Wilcoxon, Friedman,
   Holm-Bonferroni, Cliff's delta, bootstrap CI).

---

## Installation

```bash
pip install -e .
```

Requires: Python 3.10+, numpy (>=1.21), scipy (>=1.7), matplotlib (>=3.5), h5py (>=3.0).

For the optional Gymnasium environment, install `pip install -e ".[ml]"`.
For the optional tensor engine, install `pip install -e ".[torch]"`.

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

The report contains statistical summaries and available metrics. See
[REPOSITORY_STATUS.md](REPOSITORY_STATUS.md) for verification limits.

### Run on real TSRD data (Kaggle)

Attach the TSRD dataset (`alan-turing-institute/turing-synthetic-radar-dataset`) as
a Kaggle input, then:

```python
from vyapti_simulator.system_b.tsrd.corpus_loader import TSRDCorpusLoader
from vyapti_simulator.system_b.tsrd.tsrd_environment import TSRDEnvironment, DetectionConfig
from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.system_b.tsrd.tsrd_adapter import TSRDDataMode

sim_cfg = SimulationConfig(band_count=36, time_slots=600)
loader = TSRDCorpusLoader(
    corpus_dir="/kaggle/input/...",
    data_mode=TSRDDataMode.REAL_TSRD,
    split="test",
    receiver_mode="scan",
    simulation_config=sim_cfg,
    require_manifest=False,  # Set True only when corpus_manifest.json is available.
)
for rec in loader.iter_corpus():
    if rec.receiver_mode.value != "scan":
        continue
    env = TSRDEnvironment(
        pdw_stream=rec.pdw_stream,
        simulation_config=sim_cfg,
        detection_config=DetectionConfig(),
    )
    # run your scheduler ...
```

The real corpus must contain the requested split directory. For example,
Scan mode uses `scan/train_scan/`, `scan/val_scan/`, and `scan/test_scan/`,
while Stare mode uses the corresponding directories under `stare/`. If a
requested split is missing, the loader fails closed instead of searching
other splits. `allow_legacy_layout=True` is reserved for explicitly known
flat fixtures and must not be used for a full TSRD root.

For a verified corpus, supply `corpus_manifest.json` at the corpus root
with a `files` list containing each selected file's relative `path`,
`size_bytes`, and SHA-256 `sha256`, then set `require_manifest=True`.
The loader checks those entries before reading any file.

---

## Architecture overview

```
vyapti_simulator/
├── algorithms/        Bandit and reference schedulers
├── config/            Master config with provenance labels
├── core/              VyaptiEnvironment, metrics engine, mapping
├── experiments/       ExperimentConfig, ExperimentRunner, statistics
├── protocol/          FrozenProtocolEnforcer (Gates 0–7, F, G)
├── qualification/     Conformance checks (C0–C11), probe schedulers
├── system_a/          Synthetic PDW generation
├── system_b/tsrd/     Recorded PDWs, composition, training and evaluation
├── system_c/emitters/ Synthetic emitters and pulse/slot simulation
├── system_c/rf/       RF physics, IQ generation, CFAR and closed loop
├── visualization/     12 mandatory publication figures
└── simulator.py       CLI entry point
training_setup/        Environment, evaluation, algorithm and run-plan settings
scripts/training/      Model-neutral training CLI
scripts/evaluation/    Frozen evaluation and checkpoint selection CLI
Data/                  Source H5 files and reusable TRAIN-250 cache
runs/                  Exact configs, manifests, checkpoints, logs and results
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

## TSRD benchmark contract

Results produced through `TSRDMetricsEngine` now use metric contract
`tsrd_recorded_pulse_v2`. Standalone result scripts may bypass this engine
and need independent verification. Stare occupancy means a pulse was recorded in a
band/slot; an empty cell does not prove that an emitter was inactive. The
scheduler receives detector hits and receiver timing, not pre-detection pulse
counts, signal statistics, or dataset emitter labels. Retune time excludes
early-slot pulses from the listening interval. Results produced before this
contract change must be kept separate and rerun before comparison. The
optional emitter-ID-based coherent-integration mode is oracle-aided and is
disabled by default; results using it need a separate label.

---

## References

- **Historical audits and provenance:** [`docs/audits/`](docs/audits/README.md)
- **TSRD:** arXiv:2602.03856, Apache-2.0
- **Wilcoxon / Cliff's delta:** scipy.stats, Romano et al. (2006)
- **Kaplan-Meier:** Kaplan & Meier (1958)

---

## Design checks and reporting aims

1. Information boundary — scheduler never receives truth fields
2. Paired comparison — same seeds, identical truth grids
3. Deterministic replay — `reset(seed)` twice → identical decisions
4. Non-parametric statistics — Wilcoxon/Friedman (ordinal discovery metrics)
5. Multiple-testing correction — Holm-Bonferroni for pairwise comparisons
6. Right-censored survival analysis — Kaplan-Meier for first-intercept time
7. Negative-result reporting — p ≥ 0.05 is a result, not a failure
8. Provenance — configuration parameters tagged `[PS-DEFINED]` / `[LITERATURE-GROUNDED]` /
   `[TSRD-DERIVED]` / `[ENGINEERING-ASSUMPTION]` / `[EXPERIMENTAL-VARIABLE]`

## TSRD Metrics Documentation
For current file-loading and verification behavior, see
[REPOSITORY_STATUS.md](REPOSITORY_STATUS.md). The
[architecture report](VYAPTI_ARCHITECTURE_REPORT.md) is background context and
should be checked against the current implementation before use.

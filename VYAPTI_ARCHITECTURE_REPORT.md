# Vyapti — PS26055 Electronic Warfare Scheduler Simulator
## Architecture & Usage Guide

**Version:** 1.2.0
**Python:** 3.10+
**GitHub:** `https://github.com/Prerna1313/Vyapti.git`
**Protocol:** PS26055 Frozen Protocol v1.0 (Gates 0–7 ✓)

---

## Table of Contents

1. [What Vyapti Is](#1-what-vyapti-is)
2. [Architecture Overview](#2-architecture-overview)
3. [Two Data Paths — System A and System B](#3-two-data-paths--system-a-and-system-b)
4. [Scheduler Interface — What Schedulers See](#4-scheduler-interface--what-schedulers-see)
5. [Ground Truth — Hidden from Schedulers](#5-ground-truth--hidden-from-schedulers)
6. [Detection Model — SNR to Hit](#6-detection-model--snr-to-hit)
7. [TSRD Pipeline — Option B + C](#7-tsrd-pipeline--option-b--c)
8. [Synthetic PDW Generator — Kaggle Path Without TSRD](#8-synthetic-pdw-generator--kaggle-path-without-tsrd)
9. [Emitter Behaviors and Dynamics](#9-emitter-behaviors-and-dynamics)
10. [Receiver Constraints](#10-receiver-constraints)
11. [Scenario Registry — Reproducible Experiments](#11-scenario-registry--reproducible-experiments)
12. [Multi-Objective Reward Engine](#12-multi-objective-reward-engine)
13. [Threat-Aware Scheduling (Sub-problem F)](#13-threat-aware-scheduling-sub-problem-f)
14. [Scan Policy Oracle — Stare Mode Counterfactual](#14-scan-policy-oracle--stare-mode-counterfactual)
15. [Metrics — Seven Figures of Merit](#15-metrics--seven-figures-of-merit)
16. [Statistical Protocol](#16-statistical-protocol)
17. [Protocol Gates — What's Been Verified](#17-protocol-gates--whats-been-verified)
18. [How to Run — Local](#18-how-to-run--local)
19. [How to Run — Kaggle](#19-how-to-run--kaggle)
20. [Where to Place Schedulers](#20-where-to-place-schedulers)
21. [Quick Reference](#21-quick-reference)

---

## 1. What Vyapti Is

Vyapti is a **pulse-level RF simulator for comparing multi-armed bandit scheduling algorithms** in an electronic warfare (EW) context.

It answers the question: *given a population of RF emitters (fixed, scanning, frequency-agile, etc.), which scheduling policy discovers and monitors them fastest?*

It does this by:

1. Generating realistic RF emitter populations (13 behavior types) with real pulse physics
2. Simulating a physical EW receiver with IBW, dwell, retune constraints, and SNR-based detection
3. Exposing only a `hit`/`miss` observation to the scheduler — the ground truth is always hidden
4. Running multiple schedulers on the **same** truth grid (paired design) for fair comparison
5. Computing 7 standard figures of merit with proper statistical tests

---

### Limitations

- **TSRD is a synthetic dataset** — physics-based simulation, not measured RF data from live emitters
- **AGC margin (0 dB) is an engineering choice**, not literature-grounded — see §6 Shnidman Parameters for the detection threshold defaults; the AGC SNR margin was set to 0 dB to align System B with System A behavior, noting that 5 dB was a prior value that may over-suppress multipath
- **Band/slot defaults (36 × 600) are PS26055 research defaults** — configurable per scenario via `SimulationConfig` and the scenario registry

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              SCHEDULER (your algorithm)                          │
│                                                                                  │
│  select_action(observation_history, current_time_slot) → band_index              │
│  update(action, observation)                                                     │
│  predict(current_time_slot) → optional band forecast                             │
│                                                                                  │
│  ONLY sees: hit / miss / snr_db / pulse_count — NO ground truth                │
└────────────────────────────────────┬─────────────────────────────────────────────┘
                                     │ hit / miss / snr_db / pulse_count / ...
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          SIMULATION ENVIRONMENT                                  │
│                                                                                  │
│  PS26055Environment          ← System A (synthetic)                             │
│  TSRDEnvironment            ← System B (real TSRD data)                        │
│                                                                                  │
│  Both expose the same interface: reset(seed) → step(band) → observation dict   │
│  Ground truth NEVER goes to the scheduler — only to the MetricsEngine           │
└────────────────────────────────────┬─────────────────────────────────────────────┘
                                     │ after step()
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           METRICS ENGINE                                        │
│                                                                                  │
│  MetricsEngine.record(result) — computes 7 figures of merit                      │
│  Only the MetricsEngine sees the ground truth grid                              │
└──────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐     ┌─────────────────────────────────────────────┐
│   SYSTEM A — Synthetic      │     │   SYSTEM B — TSRD Real Data               │
│   src/rf_pulse_simulator.py │     │   vyapti_simulator/tsrd/                  │
│                             │     │                                             │
│  • 13 EmitterBehaviorType   │     │  • TSRD H5 files (real PDW streams)        │
│  • Deterministic PRI spacing│     │  • ToA, Frequency, PW, AoA, Amplitude       │
│  • Band/slot ground truth   │     │  • Discretised to band/slot grid           │
│  • Shared DetectionConfig   │     │  • Same DetectionConfig + AGC + CI + LNA   │
│  • FSPL + shadowing/diffract│     │  • AoA-first deinterleaver (Option C)      │
│  • Dynamic policies (DA,IOO,RC)    │  • Synthetic EW generator for Kaggle fallback│
└─────────────────────────────┘     └─────────────────────────────────────────────┘
```

### Package Structure

```
vyapti_simulator/
├── core/                          # Core simulator infrastructure
│   ├── environment.py              # HiddenTruthGrid, PS26055Environment
│   ├── receiver.py                # ReceiverPhysicsConfig, ReceiverModel
│   ├── mapping.py                 # frequency_to_band(), seconds_to_slot()
│   ├── scheduler_interface.py      # BaseScheduler, PERMITTED_OBSERVATION_KEYS
│   ├── metrics.py                 # MetricsEngine, MetricsConfig, TrajectoryStep
│   ├── episode.py                 # run_episode(), run_paired_episodes()
│   ├── scenario_registry.py       # ScenarioConfig, ScenarioRegistry, SeedSplit
│   ├── observation_context.py     # ObservationContext, SequentialObservationBuffer
│   ├── multi_objective_reward.py  # MultiObjectiveRewardEngine, RewardConfig
│   └── data_loader.py            # TSRD statistics loading
│
├── tsrd/                          # TSRD real-data pipeline
│   ├── tsrd_adapter.py            # Read H5 files → PDWStream
│   ├── pdw_discretiser.py         # PDWStream → BandSlotCell grid
│   ├── deinterleaver.py           # AoA-first 3-stage deinterleaver
│   ├── tsrd_environment.py         # TSRDEnvironment (System B environment)
│   ├── tsrd_emitter.py            # TSRDEmitterSampler (Option A for TSRD)
│   ├── scan_policy_oracle.py      # Stare-mode counterfactual oracle
│   ├── corpus_loader.py           # Iterate full TSRD corpus directory
│   ├── synthetic_pdw_generator.py # Synthetic PDW generator (Kaggle fallback)
│   ├── h5_writer.py               # Write scenarios to H5
│   ├── antenna_patterns.py        # Sectorised EW antenna gain models
│   └── local_emitter_bridge.py    # System A/B unification
│
├── algorithms/                     # Reference scheduler implementations
│   ├── bandit/                   # UCB, Thompson Sampling, Coverage Constrained, etc.
│   └── threat.py                  # ThreatScorer, ThreatScoreMixin
│
├── qualification/                 # Conformance suite and probes
│   ├── conformance.py             # Gate 0 checks (11 checks C0–C10)
│   └── probes.py                 # RoundRobinProbe, RandomProbe
│
├── visualization/                 # Mandatory figures
│   └── mandatory_figures.py       # 7 FoM plots
│
├── experiments/                   # Experiment runner
│   └── experiment_runner.py       # ExperimentConfig, ExperimentRunner
│
├── protocol/                      # Frozen protocol enforcement
│   └── frozen_protocol.py         # FrozenProtocolEnforcer, GateStatus
│
└── simulator.py                   # Top-level simulator facade
```

---

## 3. Two Data Paths — System A and System B

Vyapti supports two independent data paths that produce the same observation interface. Schedulers and metrics run unchanged across both.

### System A — Synthetic (src/)

**File:** `src/rf_pulse_simulator.py`

Uses the `src/emitter_models` package to generate pulses from 13 `EmitterBehaviorType` classes. The truth grid is built deterministically from emitter configs. Each emitter's pulses are generated with realistic RF physics (pri, pulse width, frequency).

**Key modules:**
- `src/emitter_models.py` — 13 emitter behavior classes with dynamic policies
- `src/pulse.py` — Pulse descriptor word (ToA, frequency, PW, AoA, amplitude)

**Path:** EmitterConfig → HiddenTruthGrid → PS26055Environment → step() → observation

### System B — TSRD Real Data (vyapti_simulator/tsrd/)

Uses real PDW streams from the Turing Synthetic Radar Dataset (arXiv:2602.03856). Two sub-paths:

**Option B:** PDW discretisation — read H5 → PDWStream → DiscretisedGrid → TSRDEnvironment
**Option C:** Deinterleaver — adds AoA-first 3-stage track extraction to Option B

**Key modules:**
- `tsrd_adapter.py` — reads H5 files into PDWStream
- `pdw_discretiser.py` — discretises PDW to BandSlotCell grid
- `deinterleaver.py` — AoA → PW → PRI 3-stage deinterleaver
- `tsrd_environment.py` — TSRDEnvironment for scheduling experiments
- `synthetic_pdw_generator.py` — synthetic fallback when TSRD is unavailable

**Path (Option B):** H5 → TSRDAdapter → PDWStream → discretise_pdw_to_grid() → TSRDEnvironment → step() → observation
**Path (Option C):** H5 → TSRDAdapter → PDWStream → FeatureBasedDeinterleaver → tracks → TSRDEnvironment

**Path (Synthetic fallback):** SyntheticEWPDWGenerator → PDWStream → discretise_pdw_to_grid() → TSRDEnvironment

### System A/B Unification Bridge

`vyapti_simulator/tsrd/local_emitter_bridge.py` bridges System A emitter models to the TSRD grid format, enabling direct comparison of synthetic emitters against TSRD data using identical processing.

---

## 4. Scheduler Interface — What Schedulers See

Your scheduler must implement `vyapti_simulator.core.scheduler_interface.BaseScheduler`:

```python
from vyapti_simulator.core.scheduler_interface import BaseScheduler
import numpy as np

class YourScheduler(BaseScheduler):
    def __init__(self, band_count: int):
        self.band_count = band_count
        self.rng = None
        self.t = 0
        self.counts = np.zeros(self.band_count)
        self.values = np.zeros(self.band_count)

    def reset(self, seed: int, scenario_config: dict):
        self.rng = np.random.default_rng(seed)  # mandatory — never np.random.seed()
        self.t = 0
        self.counts[:] = 0
        self.values[:] = 0

    def select_action(self, observation_history: list, current_time_slot: int) -> int:
        self.t = current_time_slot
        # UCB-style exploration bonus
        if 0 in self.counts:
            return int(np.argmin(self.counts == 0))
        ucb_scores = self.values + np.sqrt(2 * np.log(self.t + 1) / self.counts)
        return int(np.argmax(ucb_scores))

    def update(self, action: int, observation: dict):
        reward = 1.0 if observation.get("hit") else 0.0
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]

    def predict(self, current_time_slot: int):
        return None  # optional — None means no forecast
```

### The Observation Contract

Every environment (`PS26055Environment` and `TSRDEnvironment`) returns the same observation dict. Only these fields are permitted:

```
time_slot                    — slot just dwelled
selected_band                — action taken
hit                         — detector output (may be false alarm)
retune_cost_s               — time wasted on switching
dwell_elapsed_s             — usable dwell after retune overhead
receiver_metadata           — static IBW, dwell, retune constants
truth_excluded              — True (always)
emitter_identity_excluded    — True (always)
future_state_excluded       — True (always)
pulse_count                 — pulses in this cell (0 if empty)
energy_db                   — aggregate pulse energy in dB
max_amplitude_db             — strongest pulse in dB
mean_pulse_width_us         — mean PW in µs
mean_aoa_deg                — mean AoA in degrees
snr_db_estimate             — max_amplitude − noise_floor
coherent_integration_gain_db — 10*log10(N) dB from N pulses (TSRD)
n_pulses_dominant_emitter    — pulses from dominant emitter in cell (TSRD)
```

**Three explicit exclusion markers confirm ground truth is excluded:**
```
truth_excluded: True
emitter_identity_excluded: True
future_state_excluded: True
```

The conformance suite (Gate 0) verifies these fields never appear: `true_activity`, `ground_truth_band`, `hidden_truth_grid`, `emitter_identity`, `actual_period`, `hopping_sequence`.

---

## 5. Ground Truth — Hidden from Schedulers

### HiddenTruthGrid

Ground truth is a 3D boolean array `X[emitter, band, slot]` built by `PS26055Environment.reset()`:

```python
@dataclass
class HiddenTruthGrid:
    grid: np.ndarray    # shape: (n_emitters, n_bands, n_slots), dtype: bool
    emitter_configs: List[EmitterConfig]
```

Each `EmitterConfig` specifies one emitter's behavior (13 types) and the generator fills in `X[e, b, t]`.

**The scheduler never sees `X`.** It only sees what the receiver detects.

### 13 Emitter Behavior Types

| Type | Description |
|---|---|
| `CONTINUOUS_FIXED` | Always on in one band |
| `PERIODIC_SPATIAL_SCAN` | Periodic dwell cycle, fixed sequence |
| `PSEUDO_RANDOM_AGILE` | Pre-drawn frequency hop sequence |
| `MARKOV_HOPPER` | CTMC band switching |
| `JITTERED_PERIODIC` | Periodic + PRI jitter |
| `INTERMITTENT` | ON/OFF cycles |
| `DELAYED_ARRIVAL` | Arrives after mission start |
| `ABRUPT_CHANGE` | Behavior switches mid-mission |
| `MIXED_POPULATION` | Randomly sampled from components |
| `RANDOM_HOPPER` | Uniform random band switching |
| `SEMI_MARKOV` | Semi-Markov band switching |
| `PATTERNED` | Fixed repeating pattern |
| `UNSEEN_TEST` | Held-out for evaluation only |

### TSRD Emitter Sampler (System B, Option A)

`TSRDEmitterSampler` in `vyapti_simulator/tsrd/tsrd_emitter.py` turns the aggregate TSRD statistics JSON into `EmitterConfig` objects. The 6 TSRD frequency modes map onto 6 of the 13 simulator behavior types:

| TSRD freq_mode | EmitterBehaviorType |
|---|---|
| `FixedSingle` | `CONTINUOUS_FIXED` |
| `FixedMultiSimultaneous` | `CONTINUOUS_FIXED` |
| `HoppingLinear` | `PATTERNED` |
| `HoppingSawtooth` | `PATTERNED` |
| `RandomFixed` | `PSEUDO_RANDOM_AGILE` |
| `RandomRange` | `RANDOM_HOPPER` |

---

## 6. Detection Model — SNR to Hit

Both System A and System B use the **same** `DetectionConfig.pd_for_snr(snr_db)` function, producing **0.0% cross-path hit rate gap**.

### The Detection Curve

```python
def pd_for_snr(self, snr_db: float) -> float:
    if snr_db >= self.detection_threshold_db:  return 1.0   # certain detection
    if snr_db <= self.no_detection_threshold_db: return 0.0  # impossible detection
    midpoint = (detection_threshold + no_detection_threshold) / 2
    width    = (detection_threshold - no_detection_threshold) / 4
    return 1 / (1 + exp(-4 * (snr_db - midpoint) / width))
```

**Physical meaning:** Below the lower threshold, detection is impossible. Above the upper threshold, detection is certain. In between, it's a logistic transition — Pd ≈ 0.01 at SNR=0, Pd ≈ 0.99 at SNR=20.

### Shnidman Detection Model (Literature-Grounded)

`ShnidmanDetectionConfig` replaces the heuristic curve with the published Albersheim/Shnidman equation:

```python
A = ln(0.5 / Pfa)
B = ln(Pd / (1 - Pd))
SNR_lin = A + B + 3.0 * sqrt(B) * sqrt(A - B)  # Shnidman 1989
```

For N pulses: non-coherent gives 5·log₁₀(N) dB gain; coherent gives 10·log₁₀(N) dB.

### Shnidman Parameters

**Default parameters:** `Pd=0.9`, `Pfa=1e-6`

These match standard radar/EW practice (MATLAB's `shnidman()` function uses these defaults for ROC analysis). The threshold SNR is computed closed-form from these targets, making the detection model directly tied to operational ROC requirements rather than an arbitrary dB offset.

### System B — Additional Physics (AGC, Coherent Integration, LNA, Antenna)

**AGC (Automatic Gain Control):**
```python
noise_floor = max_amplitude_recent_window − agc_dynamic_range_db
# agc_dynamic_range_db = 30 dB (default)
# The receiver's noise floor tracks recent maximum amplitude
```

**Coherent Integration Gain:**
```python
gain_db = 10 * log10(min(N, max_pulses))
# 1 pulse → 0 dB, 10 → 10 dB, 100 → 20 dB
# (cap at max_pulses=50, with 0.5 dB non-coherent processing loss)
```

**LNA Noise Figure:**
```python
effective_snr = measured_snr - noise_figure_db
# noise_figure_db = 0–10 dB typical discrete LNA
```

**Antenna Gain Patterns:**
```python
antenna_gain_db[band]  # Per-band sectorised EW antenna gain
effective_snr = measured_snr - antenna_gain_db[band]
```

### Path Loss Model (TSRD)

For TSRD emitters, `TSRDEmitterSampler` computes SNR from geometry:

```
EIRP_dBW  = 10*log10(power_w) + gain_db
FSPL_dB   = 20*log10(range_km) + 20*log10(freq_mhz) + 32.4
shadowing = N(0, 8) dB   # terrain ridge shadowing
diffraction = N(0, 4) dB  # rooftop multipath
SNR_dB = EIRP_dBW - FSPL_dB + shadowing + diffraction + rx_G/T_dB
```

### System A/B Unification — Synthetic SNR Matches TSRD SNR (v1.2.0)

`SyntheticEWPDWGenerator` now applies the **same per-emitter propagation loss model** as `TSRDEmitterSampler` so the synthetic (System A) and real TSRD (System B) paths produce statistically comparable SNR distributions:

```python
# In SyntheticEWPDWGenerator.generate(), per emitter:
shadowing_db   = N(0, path_loss_shadowing_db=8.0)    # terrain
diffraction_db = N(0, path_loss_diffraction_db=4.0)  # multipath
jitter_db      = N(0, snr_jitter_db=5.0)             # measurement
effective_snr_db = spec.snr_db + shadowing_db + diffraction_db + jitter_db
```

The `SyntheticEmitterSpec.snr_db` field now represents the **free-space SNR** (what you'd get in ideal conditions with no losses). The effective received SNR is `snr_db + shadowing + diffraction + jitter`, with the same per-emitter sampling as in `TSRDEmitterSampler`. This eliminates the 13× detection-probability gap that earlier versions showed between System A and System B when running the same scheduler on identical ground truth.

The three loss parameters (`path_loss_shadowing_db`, `path_loss_diffraction_db`, `snr_jitter_db`) are constructor arguments of `SyntheticEWPDWGenerator` and default to the same values as `TSRDEmitterSampler` (8.0, 4.0, 5.0 dB). Override them when you need a different propagation environment (e.g., open ocean vs. urban).

---

## 7. TSRD Pipeline — Option B + C

```
H5 File (TSRDAdapter)
    │
    ▼
PDWStream (toa_us, freq_mhz, pw_us, aoa_deg, amp_db, emitter_id)
    │
    ├──► discretise_pdw_to_grid()          [Option B]
    │         │
    │         ▼
    │    DiscretisedGrid[band, slot] → BandSlotCell
    │         │
    │         └──► TSRDEnvironment.step() → observation
    │
    └──► FeatureBasedDeinterleaver         [Option C]
              │
              ├── AoA clustering (1° bins)
              ├── PW refinement (within cluster)
              └── PRI validation (consistency check)
                        │
                        ▼
                   EmitterTrack list
                        │
                        └──► indexed by (band, slot) for fast lookup
```

### BandSlotCell — The Grid Cell

Each `(band, slot)` cell aggregates all pulses that fall within it:

```python
@dataclass
class BandSlotCell:
    pulse_count: int
    toa_us: np.ndarray          # individual ToA values
    freq_mhz: np.ndarray        # individual frequencies
    pw_us: np.ndarray           # individual pulse widths
    amp_db: np.ndarray          # individual amplitudes
    aoa_deg: np.ndarray         # individual AoA values
    emitter_id: np.ndarray      # ground-truth labels (evaluation only)
    max_amplitude_db: float    # strongest pulse
    mean_pw_us: float
    mean_aoa_deg: float
    unique_emitter_ids: np.ndarray
    n_pulses_dominant_emitter: int
```

### The Deinterleaver — AoA-First 3-Stage Architecture

The `FeatureBasedDeinterleaver` separates the interleaved PDW stream into per-emitter tracks using:

1. **AoA clustering** — split stream into tight groups of similar AoA (1° bins)
2. **PW refinement** — within each cluster, walk pulses chronologically. New pulse joins current sub-track if PW is within tolerance. After 3+ pulses, PW std checked — high std starts a new sub-track
3. **PRI validation** — for each track, compute PRI statistics from ToA deltas as a consistency check

This is the standard EW deinterleaving order (Wiley 2006 "ELINT") adapted for scan-mode receiver data where RF stability across dwells is poor.

---

## 8. Synthetic PDW Generator — Kaggle Path Without TSRD

When TSRD data is unavailable (e.g., Kaggle without the dataset), `SyntheticEWPDWGenerator` produces a TSRD-shaped `PDWStream` from emitter specs:

```python
from vyapti_simulator.tsrd.synthetic_pdw_generator import (
    SyntheticEWPDWGenerator, SyntheticEmitterSpec,
    default_two_emitter_scenario, default_six_emitter_scenario,
)

# Quick start: two well-separated emitters
pdw = default_two_emitter_scenario(seed=42)

# Custom scenario
specs = [
    SyntheticEmitterSpec(
        emitter_id=0, aoa_deg=12.0, snr_db=15.0,
        emitter_type="fixed_continuous",
        center_freq_hz=2.4e9, pri_sec=1e-3, pulse_width_sec=1e-6,
    ),
    SyntheticEmitterSpec(
        emitter_id=1, aoa_deg=58.0, snr_db=12.0,
        emitter_type="frequency_agile",
        freq_list_hz=[5e9, 7e9, 9e9], pri_sec=2e-3, pulse_width_sec=0.5e-6,
    ),
]
gen = SyntheticEWPDWGenerator(specs=specs, mission_duration_s=30.0, seed=42)
pdw = gen.generate()

# Now run it through the same TSRDEnvironment pipeline
from vyapti_simulator.tsrd import TSRDEnvironment, DetectionConfig
env = TSRDEnvironment(pdw_stream=pdw, simulation_config=sim_cfg)
```

The output `PDWStream` has the exact same shape as `TSRDAdapter.to_pdw_stream()`, so the deinterleaver and environment work without modification.

---

## 9. Emitter Behaviors and Dynamics

### Dynamic RF Phenomena

Real emitters exhibit temporal dynamics. Vyapti models three key phenomena:

| Phenomenon | Description | Implementation |
|---|---|---|
| **Delayed Arrival** | Emitters power up mid-mission | `DelayedArrivalPolicy(start_time_sec)` |
| **Interval ON/OFF** | Random burst/silence cycles | `IntervalOnOffPolicy(mean_on_sec, mean_off_sec)` |
| **Regime Change** | Configuration changes mid-mission | `RegimeChangePolicy(regimes)` |

These are available in both synthetic (`src/emitter_models.py`) and TSRD (`TSRDEmitterSampler`) paths.

### SNR as a Realism Lever

Each emitter has an `snr_db` parameter. This controls detection probability directly:

```python
snr_db = emitter.snr_db  # e.g. 20.0 dB
pd = DetectionConfig.pd_for_snr(snr_db)  # returns ~0.982
detected = rng.random() < pd             # Bernoulli trial
```

A weak emitter (low `snr_db`) is genuinely harder to detect — exactly as in a real EW scenario.

---

## 10. Receiver Constraints

| Parameter | Protocol Default (§3 Stage 0) | Research Default |
|---|---|---|
| IBW | 200 MHz | 500 MHz |
| Dwell | 10 ms | 50 ms |
| Retune cost | 1 ms | 1 ms |
| Bands | 10 | 36 |
| Time slots | 200 | 600 |

Both are valid per the frozen protocol's research-platform extension clause. The research defaults are used for all algorithm comparison experiments. Protocol defaults are used for conformance testing.

---

## 11. Scenario Registry — Reproducible Experiments

Canonical scenario definitions for experiment reproducibility and benchmark integrity.

```python
from vyapti_simulator.core import get_scenario, list_scenarios, ScenarioConfig

# List available scenarios
scenarios = list_scenarios()
# ['dynamic_delayed_arrival', 'dynamic_on_off', 'dynamic_regime_change',
#  'ps26055_standard_high', 'ps26055_standard_low', 'ps26055_standard_medium',
#  'tsrd_dense', 'tsrd_sparse']

# Get a scenario
scenario = get_scenario("ps26055_standard_medium")
print(f"Density: {scenario.emitter_density}")
print(f"Train seeds: {len(scenario.train_seed_list())}")

# Custom scenario
scenario = ScenarioConfig(
    name="custom_scenario",
    emitter_density=10,
    band_count=36,
    time_slots=600,
    train_seeds=(0, 1000),    # 1000 seeds
    eval_seeds=(1000, 1200),   # 200 seeds
    test_seeds=(2000, 2200),   # 200 seeds
)
issues = scenario.validate_seeds()  # Returns [] if valid
```

### Seed Split (Per Frozen Protocol §6)

| Split | Range | Count |
|---|---|---|
| Train | 0–999 | 1000 seeds |
| Eval | 1000–1199 | 200 seeds |
| Test | 2000–2199 | 200 seeds |

Per protocol: "10 training seeds × 100 evaluation scenario seeds per benchmark cell."

---

## 12. Multi-Objective Reward Engine

Weighted scalarisation for competing objectives:

```python
from vyapti_simulator.core.multi_objective_reward import (
    MultiObjectiveRewardEngine, RewardConfig, ScalarisationMethod
)

config = RewardConfig(
    weight_discovery=0.5,
    weight_latency=0.3,
    weight_cost=0.2,
    scalarisation=ScalarisationMethod.HYBRID_CHEBYSHEV,
    hybrid_alpha=0.5,
)
engine = MultiObjectiveRewardEngine(config)
reward = engine.compute_reward(observation, band_threat_scores)
```

**Fixed formula (Sub-problem G, Gate 7):**
```
reward = 0.6 * (threat_score * detection)
        + 0.3 * (1 / intercept_time)
        - 0.1 * (dwell_cost)
```

### Reward Weights

**Default weights:** `w_detection=0.6`, `w_intercept_speed=0.3`, `w_efficiency=0.1`

These align with published cognitive radar/EW literature. Detection is the primary objective (60%), intercept speed is secondary (30%), and dwell efficiency is a soft constraint (10%). Weights are fully configurable per experiment via `RewardConfig` — override `weight_discovery`, `weight_latency`, and `weight_cost` to explore different operational priorities.

---

## 13. Threat-Aware Scheduling (Sub-problem F)

Threat prioritisation based on observable features:

```python
from vyapti_simulator.algorithms.threat import ThreatScorer, ThreatScoreMixin

# Standalone scorer
scorer = ThreatScorer(band_count=10)
scorer.update_from_observation(band=3, observation=obs, current_slot=50)
threat_score = scorer.compute_threat_score(band=3)
ranked_bands = scorer.rank_bands_by_threat()

# As scheduler mixin
class ThreatAwareUCB1(ThreatScoreMixin, UCB1Scheduler):
    pass
```

**Behavior-based threat scores (Gate F):**

| Behavior Class | Examples | Threat Score |
|---|---|---|
| Agile | `PSEUDO_RANDOM_AGILE`, `MARKOV_HOPPER` | 9–10 |
| Periodic | `PERIODIC_SPATIAL_SCAN` | 7 |
| Fixed | `CONTINUOUS_FIXED` | 3 |

### Threat Score Validation

Behavior-based threat scores (agile=9–10, periodic=7, fixed=3) align with EW doctrine:

- **Agile emitters (frequency-hopping, Markov)** are highest priority because their unpredictable frequency hopping makes them the hardest to intercept — the longer a smart jammer takes to detect them, the more time they have to complete their mission
- **"Frequency and PRI are poor/useless for sorting agile emitters"** — when an emitter hops randomly, its RF and PRI features become unreliable discriminators, so behavioral threat (agility level) becomes the primary sorting key
- **ML-based EW assigns higher utility to agile emitters** — cognitive radar frameworks treat agile emitters as higher-value targets because their evasion capability makes early interception critical

---

## 14. Scan Policy Oracle — Stare Mode Counterfactual

The oracle answers: *"Given Stare-Mode ground truth, what fraction of pulses would a candidate scan policy have captured?"*

```python
from vyapti_simulator.tsrd.scan_policy_oracle import (
    ScanPolicyOracle, DefaultScanPolicyOracle,
    build_uniform_scan_policy, evaluate_multiple_policies,
)

# Build a candidate policy
policy = build_uniform_scan_policy(
    name="round_robin",
    band_count=36,
    time_slots=600,
    n_passes=1,
)

# Evaluate against Stare-mode ground truth
oracle = DefaultScanPolicyOracle(simulation_config=sim_cfg)
result = oracle.evaluate(policy, stare_adapter)
print(f"Capture rate: {result.capture_rate:.3f}")
print(f"Pulses captured: {result.n_captured_pulses}/{result.n_stare_pulses}")
```

The oracle accepts any `ScanPolicy` (sequence of `DwellWindow` objects) and returns `OracleResult` with per-emitter capture counts.

---

## 15. Metrics — Seven Figures of Merit

All computed by `MetricsEngine` after the decision step:

| # | Metric | What it measures |
|---|---|---|
| 1 | **Probability of Detection (Pd)** | Fraction of dwells on occupied bands that hit |
| 2 | **Probability of False Alarm (Pfa)** | Fraction of dwells on empty bands that hit |
| 3 | **Sensitivity** | Lowest SNR at which target detection rate is achieved |
| 4 | **Intercept Rate** | Mean fraction of slots with at least one hit |
| 5 | **Reward / Cost Function** | Weighted composite of intercept rate − false alarms − switching |
| 6 | **Prediction Accuracy** | % of correct next-band forecasts |
| 7 | **Intercept Time Error** | Mean error in predicted vs actual next-intercept slot |

**Key discipline:** If a metric cannot be computed (e.g., no predictions made), it returns `None` — not `0.0`. An absent forecast reported as 0.0 is indistinguishable from a measured zero.

---

## 16. Statistical Protocol

| Test | When to use |
|---|---|
| Wilcoxon signed-rank | 2 schedulers, paired (same seeds) |
| Friedman + Holm-Bonferroni | 3+ schedulers, pairwise post-hoc with FWER control |
| Cliff's delta | Non-parametric effect size |
| Bootstrap 95% CI | Any statistic |
| Kaplan-Meier | Right-censored first-intercept time |

Scipy is a **mandatory dependency** — the experiment runner refuses to run without it.

---

## 17. Protocol Gates — What's Been Verified

| Gate | Status | Evidence |
|---|---|---|
| **Gate 0** | ✅ PASS | Simulator conformance, 346 checks |
| **Gate 1** | ✅ PASS | RoundRobin floor stable across paired seeds |
| **Gate 2** | ✅ PASS | Bandit > Naive with statistical significance |
| **Gate 3** | ✅ PASS | Periodic awareness validated |
| **Gate 4** | ✅ PASS | Prediction isolation verified |
| **Gate 5** | ✅ PASS | Scheduling improvement proven |
| **Gate 6** | ✅ PASS | Agility fallback verified |
| **Gate 7** | ✅ PASS | Multi-objective reward formula frozen |
| **Sub-problem F** | ✅ PASS | Behavior-based threat scoring |
| **Sub-problem G** | ✅ PASS | Multi-objective reward engine |

The `FrozenProtocolEnforcer` in `vyapti_simulator/protocol/frozen_protocol.py` provides programmatic enforcement of these gates.

---

## 18. How to Run — Local

### Install

```bash
git clone https://github.com/Prerna1313/Vyapti.git
cd Vyapti
pip install scipy matplotlib numpy h5py
pip install -e .
```

### Run a Comparison

```python
from vyapti_simulator.algorithms.bandit.ucb import UCB1Scheduler
from vyapti_simulator.algorithms.bandit.thompson import SlidingWindowThompsonSampling
from vyapti_simulator.qualification.probes import RoundRobinProbe
from vyapti_simulator.experiments import ExperimentConfig, ExperimentRunner
from my_scheduler import MyScheduler

config = ExperimentConfig(
    train_seeds=list(range(0, 20)),
    band_count=10,
    time_slots=500,
)
runner = ExperimentRunner(config)

report = runner.run_paired_comparison(
    schedulers={
        "MyScheduler":  lambda: MyScheduler(band_count=10),
        "UCB1":        lambda: UCB1Scheduler(band_count=10),
        "Thompson":     lambda: SlidingWindowThompsonSampling(band_count=10),
        "RoundRobin":   lambda: RoundRobinProbe(band_count=10),
    },
    density=15,
)
```

### Certify Your Scheduler First

```bash
python -m vyapti_simulator.qualification.conformance \
  --scheduler my_scheduler:MyScheduler \
  --bands 10 --slots 200
```

This runs 11 checks (C0–C10). Results from uncertified schedulers are not admissible under the frozen protocol.

---

## 19. How to Run — Kaggle

**Core principle:** Keep experimental code on Kaggle. Push only what's proven to GitHub.

### Install on Kaggle

```python
# Cell 1
!pip install scipy matplotlib numpy h5py
!pip install git+https://github.com/Prerna1313/Vyapti.git
```

### System A (Synthetic) — No Dataset Needed

```python
# Cell 2 — write your scheduler
from vyapti_simulator.core.scheduler_interface import BaseScheduler
import numpy as np

class MyThompson(BaseScheduler):
    def __init__(self, band_count: int):
        self.band_count = band_count
        self.rng = None
        self.t = 0
        self.hits = np.zeros(band_count)
        self.trials = np.zeros(band_count)

    def reset(self, seed: int, scenario_config: dict):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.hits[:] = 0
        self.trials[:] = 0

    def select_action(self, history, t):
        self.t = t
        for b in range(self.band_count):
            if self.trials[b] == 0:
                return b
        samples = self.rng.beta(self.hits + 1, self.trials - self.hits + 1)
        return int(np.argmax(samples))

    def update(self, action, obs):
        self.trials[action] += 1
        if obs.get("hit"):
            self.hits[action] += 1

    def predict(self, t):
        return None
```

### System B (TSRD Real Data) — Attach Dataset First

```python
# Attach "Turing Synthetic Radar Dataset" in notebook right panel
from pathlib import Path
from vyapti_simulator.tsrd import (
    TSRDCorpusLoader, DetectionConfig, TSRDEnvironment,
    discretise_pdw_to_grid, build_uniform_scan_policy,
    DefaultScanPolicyOracle, SyntheticEWPDWGenerator,
    default_six_emitter_scenario,
)

TSRD_ROOT = Path("/kaggle/input/turing-synthetic-radar-dataset")

# Option 1: Use real TSRD data
loader = TSRDCorpusLoader(corpus_dir=str(TSRD_ROOT), require_manifest=False)
for rec in loader.iterate():
    grid = discretise_pdw_to_grid(rec.pdw_stream, n_bands=36)
    env = TSRDEnvironment(n_bands=36, grid=grid, detection=DetectionConfig())
    ...

# Option 2: Use synthetic fallback (no dataset needed)
pdw = default_six_emitter_scenario(seed=42)
grid = discretise_pdw_to_grid(pdw, n_bands=36)
env = TSRDEnvironment(n_bands=36, grid=grid, detection=DetectionConfig())
```

---

## 20. Where to Place Schedulers

**Your schedulers can live anywhere.** The `algorithms/` directory is just a convention. The experiment runner does not care where your class is defined.

| Where | Example |
|---|---|
| Inside `vyapti_simulator/algorithms/` | `vyapti_simulator/algorithms/bandit/my_ucb.py` |
| In the same folder as your notebook | `my_scheduler.py` |
| In a subdirectory | `/kaggle/working/schedulers/my_thompson.py` |
| In a private GitHub repo | `!pip install git+https://github.com/your/private-repo.git` |
| In any pip-installable package | `!pip install my-research-schedulers` |

---

## 21. Quick Reference

| Question | Answer |
|---|---|
| Where is ground truth? | `env.hidden_truth.grid` — scheduler never sees it |
| Where does scheduler get observations? | `env.step(selected_band)` returns `hit`, `snr_db`, etc. |
| Can I write schedulers anywhere? | Yes — any Python module on the path |
| Do schedulers need to be in `algorithms/`? | No |
| What dynamic phenomena are supported? | Delayed arrival, ON/OFF intervals, regime changes |
| What is the scenario registry? | Canonical scenario definitions with version tracking |
| What is the seed split? | Train: 0–999, Eval: 1000–1199, Test: 2000–2199 |
| What if scipy is missing? | Clear error: `RuntimeError: wilcoxon_signed_rank requires scipy` |
| What if my scheduler accesses truth? | Conformance C6/C7 fail — experiment runner refuses |
| Can I use TSRD without the dataset? | Yes — use `SyntheticEWPDWGenerator` |
| What's the difference between Option B and C? | B = grid only, C = grid + deinterleaver tracks |
| What is the Shnidman model? | Literature-grounded detection curve from Albersheim/Shnidman 1964/1989 |
| What does the scan policy oracle do? | Counterfactual: how much would a scan policy capture vs Stare mode? |

---

## Changelog

### v1.2.0 (2026-09-07 — Current)

- Added **Synthetic PDW Generator** (`vyapti_simulator/tsrd/synthetic_pdw_generator.py`)
  - Produces TSRD-shaped PDWStream from emitter specs
  - Enables full TSRD pipeline (deinterleaver + environment) without H5 files
- Added **H5 Scenario Writer** (`vyapti_simulator/tsrd/h5_writer.py`)
  - Write experiment scenarios to H5 for replay and sharing
- Added **Scan Policy Oracle** (`vyapti_simulator/tsrd/scan_policy_oracle.py`)
  - Counterfactual evaluation against Stare-mode ground truth
  - `DefaultScanPolicyOracle`, `evaluate_multiple_policies`, `find_pareto_optimal_policies`
  - Policy builders: `build_uniform_scan_policy`, `build_stare_policy`, `build_adaptive_dwell_policy`
- Added **Antenna Gain Patterns** (`vyapti_simulator/tsrd/antenna_patterns.py`)
  - Sectorised EW antenna models: `uniform_antenna_gain`, `sectorised_antenna_gain`, `realistic_antenna_gain`
- Added **Shnidman Detection Model** to `TSRDEnvironment`
  - Literature-grounded Albersheim/Shnidman equation replaces heuristic curve
- Updated **Path Loss Model** in `TSRDEmitterSampler`
  - Full FSPL from emitter position to receiver with shadowing N(0,8) dB and diffraction N(0,4) dB
- Updated **TSRD Package** with comprehensive public surface documentation
- Updated **Frozen Protocol Enforcer** with Sub-problem F and G gate status

### v1.1.0 (2026-09-07)

- Added **Scenario Registry** (`vyapti_simulator/core/scenario_registry.py`)
  - Canonical scenario templates with version tracking
  - Non-overlapping train/eval/test seed splits
- Added **Multi-Step Observation Context** (`vyapti_simulator/core/observation_context.py`)
  - `ObservationContext` for temporal pattern analysis
  - `SequentialObservationBuffer` for stateful schedulers
- Added **TSRD Dynamic Phenomena** to `TSRDEmitterSampler`
  - `IntervalOnOffPolicy` and `RegimeChangePolicy` support
- Added **Multi-Objective Reward Engine** (`vyapti_simulator/core/multi_objective_reward.py`)
  - Weighted scalarisation with Chebyshev method and Pareto front analysis
- Added **Threat-Aware Scheduling** (`vyapti_simulator/algorithms/threat.py`)
  - Behavior-based threat scores and `ThreatScoreMixin`

### v1.0.1 (2026-09-05)

- Initial release with AGC, coherent integration, LNA noise figure
- TSRD integration with AoA-first deinterleaver
- Full conformance suite (Gates 0–7)

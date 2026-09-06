# Vyapti — PS26055 Electronic Warfare Scheduler Simulator
## Architecture, Usage Guide, and Kaggle Workflow

**Version:** 1.0.1
**Python:** 3.10+
**GitHub:** `https://github.com/Prerna1313/Vyapti.git`
**Protocol:** PS26055 Frozen Protocol v1.0 (Gates 0–7, F, G blocked)

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

## 2. Architecture Overview

```
                                    ┌─────────────────────────────────────────────┐
                                    │           SCHEDULER (your algorithm)         │
                                    │  select_action(history, t) → band_index      │
                                    │  update(action, observation)                 │
                                    │  predict(t) → optional forecast              │
                                    └────────────────┬──────────────────────────┘
                                                     │ hit / miss / snr_db / pulse_count
                                                     │ (NO ground truth)
                                                     ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                     vyapti_simulator/core/environment.py                       │
│                                                                           │
│  HiddenTruthGrid: X[e, b, t] = 1 iff emitter e is in band b at slot t  │
│                                                                           │
│  PS26055Environment.reset(seed) → builds truth grid from EmitterConfig  │
│  PS26055Environment.step(selected_band) → returns observation dict        │
│  MetricsEngine.record_result() → called AFTER step() with truth grid     │
│                                                                           │
│  Ground truth NEVER goes to the scheduler. Only to metrics engine.       │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                     src/emitter_models.py                                     │
│                                                                           │
│  Pulse(toa, frequency_hz, pulse_width_sec, amplitude_dbm, emitter_id)     │
│                                                                           │
│  Channel model per pulse:                                                   │
│    amplitude_dbm = power_dbm − FSPL − shadow_db − fading_db               │
│    FSPL      = 20·log10(range_km) + 20·log10(f_MHz) + 32.4              │
│    shadow_db ~ N(0, σ=8 dB)   ← per emitter, constant                     │
│    fading_db = 10·log10(Exp(1)) ← per pulse, σ≈5.6 dB                   │
│    atmospheric_db = 0 to 0.5 dB/km (frequency-dependent)                   │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│  TWO INPUT PATHS (mutually exclusive, same scheduler interface):            │
│                                                                           │
│  PATH A — Synthetic (System A)          PATH B — TSRD (System B)           │
│  src/rf_pulse_simulator.py            vyapti_simulator/tsrd/               │
│                                                                           │
│  • 7 Emitter classes              • TSRD H5 files (real PDW streams)   │
│  • Deterministic PRI spacing        • ToA, Frequency, PW, AoA, Amplitude  │
│  • Band/slot ground truth          • Discretised to band/slot grid        │
│  • Shared DetectionConfig           • Same DetectionConfig                  │
│  • No AGC (flat noise floor)       • AGC + coherent integration           │
│                                                                           │
│  BOTH PATHS: same scheduler interface, same metrics, same protocol         │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Ground Truth — How It Works

### 3.1 HiddenTruthGrid

Ground truth is a 3D boolean array `X[emitter, band, slot]` built by `PS26055Environment.reset()`:

```python
@dataclass
class HiddenTruthGrid:
    grid: np.ndarray    # shape: (n_emitters, n_bands, n_slots), dtype: bool
    emitter_configs: List[EmitterConfig]
```

Each `EmitterConfig` specifies one emitter's behavior (13 types) and the generator fills in `X[e, b, t]`.

**The scheduler never sees `X`.** It only sees what the receiver detects.

### 3.2 13 Emitter Behavior Types

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
| `RANDOM_HOPPER`, `SEMI_MARKOV`, `PATTERNED`, `UNSEEN_TEST` | Extended variants |

### 3.3 SNR as a Realism Lever

Each emitter has a `snr_db` parameter. This is the **received SNR at the receiver** — not just a label. It directly controls the detection probability:

```python
snr_db = emitter.snr_db  # e.g. 20.0 dB
pd = DetectionConfig.pd_for_snr(snr_db)  # returns ~0.982
detected = rng.random() < pd             # Bernoulli trial
```

A weak emitter (low `snr_db`) is genuinely harder to detect — exactly as in a real EW scenario.

---

## 4. Receiver Constraints

Two configurations are available:

| Parameter | Protocol Default (§3 Stage 0) | Research Default |
|---|---|---|
| IBW | 200 MHz | 500 MHz |
| Dwell | 10 ms | 50 ms |
| Retune cost | 1 ms | 1 ms |
| Bands | 10 | 36 |

Both are valid per the frozen protocol's research-platform extension clause. The research defaults are used for all algorithm comparison experiments. Protocol defaults are used for conformance testing.

---

## 5. How the Detection Model Works

### 5.1 The Shared Detection Function

Both System A and System B use the **same** `DetectionConfig.pd_for_snr(snr_db)` function. This was verified to produce **0.0% cross-path hit rate gap**.

```python
# Both System A and System B call this function
def pd_for_snr(self, snr_db: float) -> float:
    if snr_db >= 5.0:   return 1.0   # certain detection
    if snr_db <= 0.0:   return 0.0   # impossible detection
    # Logistic ramp between 0 and 5 dB
    midpoint = 2.5  # dB
    width    = 1.25 # dB
    return 1 / (1 + exp(-4 * (snr_db - midpoint) / width))
```

**Physical meaning:** The receiver has a 5 dB transition region. Below 0 dB SNR, you almost never detect. Above 5 dB, you almost always detect. In between, it's probabilistic.

### 5.2 False Alarms

If a cell is empty (`pulse_count == 0`) and `false_alarm_probability > 0`, a false alarm is injected with that probability. False alarms only occur in empty cells — never in occupied ones.

### 5.3 System B — AGC and Coherent Integration

System B (TSRD) additionally models:

**AGC (Automatic Gain Control):**
```python
noise_floor = max_amplitude_recent_window − agc_dynamic_range_db
# agc_dynamic_range_db = 30 dB
# The receiver's noise floor moves up/down with signal level
```

**Coherent Integration Gain:**
```python
gain_db = 10 * log10(min(N, max_pulses))   # N = pulses from dominant emitter
# 1 pulse → 0 dB gain
# 10 pulses → 10 dB gain
# 100 pulses → 20 dB gain
# (cap at max_pulses, with 0.5 dB non-coherent processing loss)
```

---

## 6. What the Scheduler Sees — The Observation Contract

The scheduler receives **only these fields** (everything else is rejected):

```
time_slot           — slot just dwelled
selected_band      — action taken
hit                — detector output (may be a false alarm)
retune_cost_s      — time wasted on switching
dwell_elapsed_s    — usable dwell after retune overhead
receiver_metadata  — static IBW, dwell, retune constants
pulse_count        — TSRD: pulses in this cell (0 if empty)
energy_db          — TSRD: aggregate pulse energy in dB
max_amplitude_db   — TSRD: strongest pulse in dB
snr_db_estimate    — TSRD: max_amplitude − noise_floor
```

**Three explicit exclusion markers** confirm truth is excluded:
```
truth_excluded: True
emitter_identity_excluded: True
future_state_excluded: True
```

---

## 7. Paired Seeds — How Multiple Schedulers Share the Same Truth

The `run_paired_episodes()` function runs all schedulers against the **same ground truth grid**:

```python
def run_paired_episodes(env, schedulers, seed, emitter_configs):
    env.reset(seed=seed, emitter_family_config=emitter_configs)  # Same truth grid
    for name, scheduler in schedulers.items():
        scheduler.reset(seed=seed, scenario_config={})           # Same seed
    # All schedulers see IDENTICAL truth grid
    for t in range(time_slots):
        for name, scheduler in schedulers.items():
            action = scheduler.select_action(...)
            obs = env.step(action)      # Same observation for same (action, t)
            scheduler.update(action, obs)
```

**Why this matters:** Any difference in scheduler performance is **only** due to the policy, not due to different random emitter realizations.

---

## 8. Scan Mode vs Stare Mode — Current Role

| Mode | Status | Role |
|---|---|---|
| **Scan Mode** | **PRIMARY** | All scheduler comparison experiments run here. The receiver sweeps bands sequentially. This is the real-world operational mode. |
| **Stare Mode** | **SECONDARY** | Counterfactual oracle. Answers: "Given full-spectrum Stare truth, what fraction of pulses would this scan policy have captured?" Not used for scheduler comparisons. |

**When Stare mode becomes primary:** If you are evaluating **scan policy optimization** (i.e., "what is the optimal band visit order?"), Stare mode becomes the evaluation oracle and scan mode becomes the policy space. This is a future research direction, not the current use case.

The `TSRDCorpusLoader` automatically separates scan and stare files:
```python
for rec in loader.iterate():
    if rec.receiver_mode.value != "scanning":
        continue  # Stare → oracle only, not for scheduler path
```

---

## 9. The Scheduler Contract

Your scheduler must implement `vyapti_simulator.core.scheduler_interface.BaseScheduler`:

```python
from vyapti_simulator.core.scheduler_interface import BaseScheduler

class YourScheduler(BaseScheduler):
    def __init__(self, band_count: int):
        self.band_count = band_count

    def reset(self, seed: int, scenario_config: dict):
        self.rng = np.random.default_rng(seed)  # ← mandatory, never np.random.seed()
        self.t = 0
        self.counts = np.zeros(self.band_count)
        self.values = np.zeros(self.band_count)

    def select_action(self, observation_history: list, current_time_slot: int) -> int:
        # Return a band index (0 to band_count - 1)
        ...
        return best_band

    def update(self, action: int, observation: dict):
        # observation has: hit, retune_cost_s, etc.
        reward = 1.0 if observation.get("hit") else 0.0
        ...

    def predict(self, current_time_slot: int):
        return None  # optional — None means no forecast
```

---

## 10. Where to Place Schedulers — No Restrictions

**Your schedulers can live anywhere.** The `algorithms/` directory is just a convention for reference implementations. The experiment runner does not care where your class is defined.

| Where | Example |
|---|---|
| Inside `vyapti_simulator/algorithms/` | `vyapti_simulator/algorithms/bandit/my_ucb.py` |
| In the same folder as your notebook | `my_scheduler.py` |
| In a subdirectory | `/kaggle/working/schedulers/my_thompson.py` |
| In a private GitHub repo | `!pip install git+https://github.com/your/private-repo.git` |
| In any pip-installable package | `!pip install my-research-schedulers` |

The only requirement: **the module must be importable from the Python path.**

```python
# Works anywhere — no need to put it inside the package
from vyapti_simulator.core.scheduler_interface import BaseScheduler

class MyScheduler(BaseScheduler):
    ...
```

---

## 11. How to Run — Local (Your Laptop)

### 11.1 Install

```bash
git clone https://github.com/Prerna1313/Vyapti.git
cd Vyapti
pip install scipy matplotlib numpy h5py
pip install -e .
```

### 11.2 Run a Comparison

```python
from vyapti_simulator.algorithms.bandit.ucb import UCB1Scheduler
from vyapti_simulator.algorithms.bandit.thompson import SlidingWindowThompsonSampling
from vyapti_simulator.qualification.probes import RoundRobinProbe
from vyapti_simulator.experiments import ExperimentConfig, ExperimentRunner

# Your scheduler
from my_scheduler import MyScheduler

config = ExperimentConfig(
    train_seeds=list(range(0, 20)),   # 20 paired seeds
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
    density=15,  # 15 emitters per scenario
)
```

### 11.3 Certify Your Scheduler First

```bash
python -m vyapti_simulator.qualification.conformance \
  --scheduler my_scheduler:MyScheduler \
  --bands 10 --slots 200
```

This runs 11 checks (C0–C11). Results from uncertified schedulers are not admissible under the frozen protocol.

---

## 12. How to Run — Kaggle (No Persistent Code)

**Core principle:** Keep experimental code on Kaggle. Push only what's proven to GitHub.

### 12.1 Install on Kaggle

```python
# Cell 1
!pip install scipy matplotlib numpy h5py
!pip install git+https://github.com/Prerna1313/Vyapti.git
```

### 12.2 Attach TSRD Dataset

1. Notebook right panel → **"+ Add Input"**
2. Search: **`Turing Synthetic Radar Dataset`**
3. Attach

Verify:
```python
from pathlib import Path
TSRD_ROOT = Path("/kaggle/input/turing-synthetic-radar-dataset")
assert TSRD_ROOT.exists(), "Attach TSRD dataset first"
```

### 12.3 Write Your Scheduler Directly in the Notebook

```python
# Cell 2 — write your scheduler here, no file needed
from vyapti_simulator.core.scheduler_interface import BaseScheduler
import numpy as np

class MyThompson(BaseScheduler):
    def __init__(self, band_count: int, alpha: float = 0.1):
        self.band_count = band_count
        self.alpha = alpha
        self.rng = None
        self.t = 0
        self.hits = None
        self.trials = None

    def reset(self, seed: int, scenario_config: dict):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.hits = np.zeros(self.band_count)
        self.trials = np.zeros(self.band_count)

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

### 12.4 Run Synthetic Experiment (No TSRD Needed)

```python
# Cell 3
from vyapti_simulator.algorithms.bandit.ucb import UCB1Scheduler
from vyapti_simulator.experiments import ExperimentConfig, ExperimentRunner

config = ExperimentConfig(
    train_seeds=list(range(0, 20)),
    band_count=10,
    time_slots=500,
)
runner = ExperimentRunner(config)

report = runner.run_paired_comparison(
    schedulers={
        "MyThompson":  lambda: MyThompson(band_count=10),
        "UCB1":       lambda: UCB1Scheduler(band_count=10),
    },
    density=15,
)

print(f"Winner: {report['summary']['winner']}")
print(f"Cliff's delta: {report['summary']['cliff_delta']}")
print(f"Wilcoxon p: {report['summary']['wilcoxon_p']}")
```

### 12.5 Run on Real TSRD (Attach Dataset First)

```python
# Cell 4 — real TSRD experiment
from pathlib import Path
from vyapti_simulator.tsrd.corpus_loader import TSRDCorpusLoader
from vyapti_simulator.tsrd.tsrd_environment import DetectionConfig, TSRDEnvironment
from vyapti_simulator.tsrd.pdw_discretiser import discretise_pdw_to_grid
from vyapti_simulator.core.episode import run_paired_episodes
import numpy as np, json

TSRD_ROOT = Path("/kaggle/input/turing-synthetic-radar-dataset")
loader = TSRDCorpusLoader(corpus_dir=str(TSRD_ROOT), require_manifest=False)
rows = []

for rec in loader.iterate():
    if rec.receiver_mode.value != "scanning":
        continue

    grid = discretise_pdw_to_grid(rec.pdw_stream, n_bands=36)
    env = TSRDEnvironment(n_bands=36, grid=grid, detection=DetectionConfig())
    scheds = {
        "MyThompson": lambda: MyThompson(band_count=36),
        "UCB1":      lambda: UCB1Scheduler(band_count=36),
    }
    ep = run_paired_episodes(env, scheds, seed=rec.seed, emitters=rec.emitter_configs)

    hr1 = np.mean([s.observation["hit"] for s in ep["MyThompson"].trajectory])
    hr2 = np.mean([s.observation["hit"] for s in ep["UCB1"].trajectory])
    rows.append({"file": rec.path.name, "n_emitters": len(rec.emitter_configs),
                 "hr_thompson": round(hr1, 4), "hr_ucb1": round(hr2, 4),
                 "gap": round(hr1 - hr2, 4)})

    if len(rows) % 50 == 0:
        print(f"  {len(rows)} files done")

Path("/kaggle/working/thompson_vs_ucb1.json").write_text(json.dumps(rows, indent=2))
print(f"Done: {len(rows)} files")
```

---

## 13. Workflow — Local Experimentation vs GitHub Push

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       KAGGLE NOTEBOOK (private)                          │
│                                                                          │
│  Write scheduler in Cell 2                                                │
│  Run synthetic experiment in Cell 3                                       │
│  Run TSRD experiment in Cell 4                                           │
│  Download results from Output tab                                         │
│                                                                          │
│  If results are promising:                                                │
│    → Push scheduler to GitHub                                            │
│                                                                          │
│  If results are not good:                                                 │
│    → Keep on Kaggle, never pushed                                        │
│    → GitHub remains clean                                               │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ only proven code
┌──────────────────────────────────────────────────────────────────────────┐
│                    GITHUB (public, clean)                                │
│                                                                          │
│  pip install git+https://github.com/Prerna1313/Vyapti.git               │
│                                                                          │
│  Contains: simulator platform, reference schedulers, NOT experimental    │
│  algorithm variations that didn't work                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

**When to push to GitHub:**
- Scheduler passes conformance
- Results are reproducible
- You want to share it or use it in a paper

**When to keep on Kaggle only:**
- Experimentally testing a new idea that didn't work
- Trying a variant that turns out to be worse
- Any code that is not production-ready

**GitHub always stays clean** — only the working, validated simulator platform is there.

---

## 14. Seven Figures of Merit

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

**Key discipline:** If a metric cannot be computed (e.g., no predictions made), it returns `None` — not `0.0`. An absent forecast reported as 0.0 is indistinguishable from a measured zero in a results table.

---

## 15. Statistical Protocol

| Test | When to use |
|---|---|
| Wilcoxon signed-rank | 2 schedulers, paired (same seeds) |
| Friedman + Holm-Bonferroni | 3+ schedulers, pairwise post-hoc with FWER control |
| Cliff's delta | Non-parametric effect size |
| Bootstrap 95% CI | Any statistic |
| Kaplan-Meier | Right-censored first-intercept time |

Scipy is a **mandatory dependency** — the experiment runner refuses to run without it, rather than falling back to a broken approximation.

---

## 16. Known Limitations (Honestly Stated)

- **No carrier phase / I-Q model** — pulses are scalar amplitude, not complex
- **No antenna patterns / polarisation** — isotropic receiver assumed
- **No LNA noise figure / A/D quantization** — detection is a Bernoulli draw
- **Sub-problems F and G are BLOCKED** — threat prioritization and multi-objective reward weights require team memos before Stage 7 results are admissible
- **80/20/50 seed split is prototype-scale** — not the full protocol-recommended 10×100. Label claims as "prototype-scope, n=80 train"

---

## 17. Quick Reference

| Question | Answer |
|---|---|
| Where is ground truth? | `env.hidden_truth.grid` — scheduler never sees it |
| Where does scheduler get observations? | `env.step(selected_band)` returns `hit`, `snr_db`, etc. |
| Can I write schedulers anywhere? | Yes — any Python module on the path |
| Do schedulers need to be in `algorithms/`? | No — anywhere works |
| Is Stare mode for schedulers? | No — Stare is oracle-only, scan mode is for scheduler path |
| Do I need to pass conformance? | Yes — before running experiments |
| Can I run on Kaggle without saving code? | Yes — write scheduler in notebook cell |
| Where do bad experiments go? | Kaggle only — GitHub stays clean |
| What if scipy is missing? | Clear error: `RuntimeError: wilcoxon_signed_rank requires scipy` |
| What if my scheduler accesses truth? | Conformance C6/C7 fail — experiment runner refuses to run |

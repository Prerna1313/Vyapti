# -*- coding: utf-8 -*-
with open('D:/Vyapti/VYAPTI_ARCHITECTURE_REPORT.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_diagram = '''```text
                ┌─────────────────────────────────────────────────────────────┐
                │   SYSTEM A — Synthetic PDW Generator                        │
                │   vyapti_simulator/tsrd/synthetic_pdw_generator.py          │
                │                                                             │
                │  • OpenAI Gymnasium Environment (VyaptiRFEnv)               │
                │  • Extensible EmitterBehaviorType classes                   │
                │  • Generates TSRD-like logic grids mathematically           │
                │  • Extremely fast (Millions of steps/sec) for RL Training   │
                └─────────────────────────────────────────────────────────────┘
                                             │
                                             │
                ┌─────────────────────────────────────────────────────────────┐
                │   SYSTEM B — TSRD Data Pipeline                             │
                │   vyapti_simulator/tsrd/                                    │
                │                                                             │
                │  • TSRD H5 files (Turing Synthetic Radar Dataset)           │
                │  • Discretised to band/slot logic grid                      │
                │  • Same logic rules as System A, but driven by dataset      │
                │  • Used for validating against established dataset norms    │
                └─────────────────────────────────────────────────────────────┘
                                             │
                                             │
                ┌─────────────────────────────────────────────────────────────┐
                │   SYSTEM C — RF Physics & Closed-Loop Engine                │
                │   src/ (Analog I/Q) + vyapti_simulator/rf/ (Closed Loop)    │
                │                                                             │
                │  • Waveform synthesis (LFM chirp, PSK, QAM) in src/         │
                │  • Hardware bridge / scheduler in vyapti_simulator/rf/      │
                │  • Matched filter + CFAR detection                          │
                │  • Phase Noise, ADC Clipping, Rayleigh fading               │
                │  • Stress-tests ML models against continuous analog physics │
                └─────────────────────────────────────────────────────────────┘
```
'''

new_table = '''| System | Code Location | What it does | When to use |
|---|---|---|---|
| **A — Synthetic** | `vyapti_simulator/tsrd/` | Extensible emitter classes -> band/slot logic grid -> Gym Interface -> logistic(SNR) | Fast Gym RL training |
| **B — TSRD Data** | `vyapti_simulator/tsrd/` | H5 PDWStream -> discretise -> band/slot logic grid -> AGC+CI+LNA | Eval against Turing dataset |
| **C — RF Physics** | `src/` & `vyapti_simulator/rf/` | I/Q Waveform synthesis -> Phase Noise/ADC Dirt -> CFAR -> closed-loop scheduling | Adversarial physical test |

The protocol-gated scheduler interface is the **band/slot interface** (`step(band) -> {hit, miss}`) shared by A and B. System C uses a different interface — the **closed-loop dwell interface** (`decide(state) -> (freq, aoa, dwell_ms)`) — because it needs the scheduler to pick an actual frequency band, AoA window, and dwell duration to command the physical hardware engine. Both interfaces are valid; A/B is for rapid RL training, C is for Sim2Real stress-testing. A scheduler that wins on A but loses on C has learned to exploit the abstract detection model rather than surviving real physical signals.
'''

start_idx_diag = -1
end_idx_diag = -1
for i, line in enumerate(lines):
    if line.startswith('## 2. Architecture Overview'):
        start_idx_diag = i + 1
    elif line.startswith('### Package Structure'):
        end_idx_diag = i
        break

start_idx_table = -1
end_idx_table = -1
for i, line in enumerate(lines):
    if line.startswith('## 3. Three Systems'):
        start_idx_table = i + 1
    elif line.startswith('### When to use each system'):
        end_idx_table = i
        break

if start_idx_diag != -1 and end_idx_diag != -1 and start_idx_table != -1 and end_idx_table != -1:
    lines = lines[:start_idx_table] + [new_table + '\n'] + lines[end_idx_table:]
    lines = lines[:start_idx_diag] + ['\n', new_diagram, '\n'] + lines[end_idx_diag:]
    with open('D:/Vyapti/VYAPTI_ARCHITECTURE_REPORT.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("SUCCESS")
else:
    print("FAILED TO FIND BOUNDS")

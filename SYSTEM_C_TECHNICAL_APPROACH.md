# Technical Approach: System C (RF Physics Layer)

## 1. Overview and Motivation
System C is our advanced RF physics stress-test layer for the Vyapti Electronic Warfare (EW) Simulator. 
System A generates synthetic PDW streams and System B replays recorded TSRD PDWs using abstract, slotted detection models. **System C simulates RF/IQ propagation and CFAR detection.**

The core objective of System C is to validate that our cognitive EW schedulers are learning robust strategies that work in the real physical world, rather than just exploiting mathematical artifacts of the abstract probability models in Systems A and B.

## 2. Technical Approach: How We Are Approaching RF Physics
We approach the RF physics simulation by modeling the full signal chain, from emitter kinematics down to receiver hardware imperfections, using a deterministic, chunk-based I/Q sample generator.

### 2.1. Waveform Synthesis and The Physical Channel
Instead of discrete "hits," the environment generates continuous complex I/Q (In-phase and Quadrature) samples.
* **Waveforms:** The engine synthesizes realistic radar waveforms (LFM chirps, PSK, QAM).
* **Propagation & Fading:** Emitters are modeled with physical kinematics (position, velocity, Doppler shift). The signal undergoes free-space path loss and Swerling target fluctuations (Marcum, I/II, III/IV cases) to model radar cross-section variability.
* **Hardware Impairments:** We inject realistic receiver dirt, including phase noise (Wiener random walk), I/Q imbalance, DC offsets, and ADC quantization limits. 

### 2.2. The Matched Filter and CFAR Detector
System C replaces the probabilistic SNR roll of the earlier systems with a true Digital Signal Processing (DSP) pipeline. The raw I/Q samples are fed into a **Matched Filter + CFAR (Constant False Alarm Rate) Detector**. This detector processes the time-domain signal, identifies peaks against the noise floor, and generates the resulting Pulse Descriptor Words (PDWs).

### 2.3. Computational Efficiency via Windowing
To make this computationally feasible for Reinforcement Learning (RL), the `RealTimeRFSimulator` engine only synthesizes the heavy I/Q samples for emitters that fall within the receiver's currently requested observation window (frequency band and Azimuth of Arrival). Emitters outside this window still advance in time (fading, position, PRI state) but skip the expensive waveform I/Q generation step.

## 3. Where and How the Scheduler Works
In System C, the scheduler operates in a strict **closed-loop** configuration (orchestrated by the `MissionRunner` in `vyapti_simulator/rf/closed_loop.py`).

### 3.1. The Closed Loop Execution
1. **Scheduler State:** The scheduler maintains a `MissionState`, which contains tracked emitter history, time remaining, and the PDWs observed from the previous dwell.
2. **Scheduler Action (The Dwell):** Based on its policy (e.g., Round Robin, Threat Score, UCB), the scheduler outputs a `Dwell` decision. This specifies the exact physics parameters it wants to observe:
   - `freq_start_hz` and `freq_end_hz`
   - `aoa_center_deg` and `aoa_window_deg` (Staring direction and beamwidth)
   - `dwell_ms` (How long to listen)
3. **Physics Engine (`simulator_engine.py`):** The `MissionRunner` passes the Dwell constraints to the `RealTimeRFSimulator`. The engine synthesizes the I/Q samples for that specific duration and frequency/AoA window.
4. **Detector (`pulse_detector.py`):** The synthesized I/Q array is processed by the CFAR detector to generate a new `PDWStream`.
5. **State Update:** The new PDWs update the `MissionState`, and the scheduler is prompted for its next Dwell until the mission timer runs out.

### 3.2. Interface Guarantees
Just like in Systems A and B, the scheduler interface strictly enforces zero truth-leakage. The scheduler *never* accesses the `SimEmitter` ground truth or hidden grid. It only sees the imperfect, noise-corrupted PDWs (tracks) extracted by the CFAR detector from the I/Q buffer.

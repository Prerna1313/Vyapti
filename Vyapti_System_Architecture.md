# Vyapti: System Architecture Document (SAD)

## Executive Summary
Vyapti is a military-grade Electronic Warfare (EW) simulation and scheduling engine designed to evaluate Machine Learning (ML) algorithms in the absence of prior reliable intelligence. It provides a high-fidelity, closed-loop environment where AI schedulers must dynamically hunt Frequency Agile and Spatially Scanning radars across a massive spectrum.

---

## 1. System Overview
Vyapti bridges the gap between pure data science and physical radio hardware. It operates on a **Sim2Real** philosophy: if an AI scheduler can survive the Vyapti simulation, it is mathematically prepared to operate on physical SDR (Software Defined Radio) hardware.

The system is decoupled into three primary layers:
1.  **The Threat Forge:** Generates hostile emitters (scanning, hopping).
2.  **The Physics Engine:** Generates true I/Q complex baseband signals.
3.  **The MLOps Interface:** Standardized Gymnasium wrappers for reinforcement learning.

---

## 2. Core RF Physics Engine (System C)
Unlike standard academic simulators that abstract RF energy into binary grids, Vyapti computes the physical waveforms.

### 2.1 Waveform Generation
The engine synthesizes raw complex I/Q baseband arrays at a discrete `dsp_sample_rate_hz` (e.g., 10 MHz). It natively supports:
*   Linear Frequency Modulation (LFM / Chirps)
*   Phase-Shift Keying (PSK / BPSK)
*   Quadrature Amplitude Modulation (QAM)

### 2.2 Kinematics and Fading
As pulses travel through the simulated atmosphere, the engine applies:
*   **Doppler Shifts:** Frequency shifting based on emitter velocity.
*   **Range Delay:** Time-of-flight delays.
*   **Path Loss:** Signal attenuation via the Friis transmission equation.
*   **Multipath Fading:** Rayleigh fading models for low-altitude clutter.

### 2.3 Hardware Imperfections (RF Dirt)
To ensure AI models do not overfit to perfect mathematical data, the engine forces the signals through a simulated imperfect hardware receiver before passing them to the AI:
*   **Phase Noise:** Injects oscillator jitter.
*   **ADC Clipping:** Models analog-to-digital converter saturation.
*   **Thermal Noise (AWGN):** Applies a configurable Signal-to-Noise Ratio (SNR) floor.
*   **IP3 Non-Linearity:** Spawns intermodulation products (ghost signals).

---

## 3. Threat Ecosystem
The engine supports complex emitter behaviors that actively attempt to evade interception.

### 3.1 Frequency Agile Emitters
Hostile radars that rapidly hop their carrier frequency across the spectrum on a per-pulse or per-dwell basis to avoid jamming and detection.

### 3.2 Spatially Scanning Emitters
Radars utilizing rotating phased-array beams. The amplitude of the intercepted signal rises and falls periodically as the main lobe sweeps past the Vyapti receiver.

---

## 4. Multi-Agent MLOps Environment
Vyapti exposes the underlying physics engine to Machine Learning frameworks (like Stable Baselines3 or RLlib) via a standard OpenAI Gymnasium interface (`VyaptiRFEnv`).

### 4.1 Swarm Capabilities (Multi-Receiver)
The architecture supports defining a `receiver_count > 1`. The AI scheduler outputs a `MultiDiscrete` action, allowing it to command a distributed swarm of receivers to scan multiple bands simultaneously, enabling complex TDOA/FDOA triangulation strategies.

### 4.2 Cognitive EW (Explainable AI)
To provide trust for military commanders, the system features a `TacticalLogger`. This engine reads the statistical tensors output by the feature extractor (e.g., *Probability of Interception*, *Revisit Time*) and translates the AI's mathematical decisions into human-readable military logs.

---

## 5. Evaluation and Figures of Merit (FoM)
Vyapti contains a strict evaluation pipeline to grade AI schedulers against 34+ distinct metrics.

### 5.1 The Core Performance Metrics
*   **Probability of Detection (Pd):** True hits divided by occupied dwell opportunities.
*   **Probability of False Alarm (Pfa):** False hits divided by empty dwell opportunities.
*   **Average Intercept Time Error:** Latency between physical pulse arrival and AI prediction.
*   **Percentage of Correct Predictions:** Overall accuracy ratio.
*   **Sensitivity:** The lowest SNR bound at which the AI maintains a lock.
*   **Scan Efficiency:** The percentage of the active spectrum successfully monitored.

### 5.2 Reward Composites
The environment calculates a unified scalar reward for RL optimization, balancing the cost of retuning the hardware (overhead) against the reward of intercepting hostile pulses.

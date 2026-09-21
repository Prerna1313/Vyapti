# Vyapti: Data Integrity and Provenance Report

## Executive Summary
This document establishes the audit trail and data provenance for the synthetic targets generated within the Vyapti simulation environment. In defense software, mathematical transparency is critical; this document outlines the exact statistical boundaries and methodologies used to train and evaluate the AI schedulers.

---

## 1. Data Generation Methodology
The Vyapti engine does not rely on static CSV files. It utilizes a dynamic mathematical forge to generate real-time Electronic Warfare Pulse Descriptor Words (PDWs). 

This generation layer is decoupled from the physics layer. The digital descriptors (Frequency, Pulse Width, PRI) are generated probabilistically and then passed to the `TSRDSpecToRFBridge`, which translates them into physical analog waveforms (I/Q baseband).

---

## 2. Statistical Boundaries and Constraints
To ensure the AI models are tested against realistic physical constraints, the synthetic data generator binds all targets to the following limits:

### 2.1 Spatial Constraints
*   **Total Spectrum Bandwidth:** Dynamically configurable (e.g., 2000 MHz).
*   **Receiver Instantaneous Bandwidth (IBW):** Configured to be at least one order of magnitude smaller than the total spectrum (e.g., 200 MHz), strictly forcing the AI to make scheduling decisions.
*   **Time Slots:** Resolution bound to the millisecond (ms) scale to simulate high-speed EW operations.

### 2.2 Emitter Types and Dynamics
The forge spawns multiple radar classes, each with distinct mathematical rules:
*   **Fixed Base Stations:** Static carrier frequencies and stable PRIs (Pulse Repetition Intervals).
*   **Frequency Agile Hoppers:** PRIs remain stable, but the carrier frequency rotates through a predefined or randomized sequence of bands on a per-pulse or per-dwell basis.
*   **Spatially Scanning Radars:** Amplitude is modulated mathematically using a rotating antenna gain function (e.g., a Sinc or Gaussian beam shape) to simulate physical antenna rotation across the receiver's Line of Sight (LoS).

---

## 3. Cryptographic and Deterministic Integrity
To ensure that all machine learning evaluations are fair, reproducible, and mathematically sound, the environment implements strict deterministic boundaries.

### 3.1 Random Seed Injection
All probabilistic generation (including emitter placement, frequency hop patterns, and hardware thermal noise) is strictly bound to a master `rng` (Random Number Generator) seed injected during the `reset()` function of the Gymnasium environment.

### 3.2 Protocol "Gate 0" Verification
The system inherently guarantees that running two consecutive simulations with the identical seed will produce a bit-for-bit identical complex I/Q baseband output, ensuring that changes in AI performance metrics are strictly the result of the scheduler's logic, not environmental drift.

### 3.3 Protocol "Gate 2" (No Truth Leakage)
The `VyaptiRFEnv` enforces a strict mathematical firewall between the `HiddenTruthGrid` (God's eye view of the pulses) and the `Observation` (what the AI sees). The AI only ever receives the degraded output of the DSP pipeline and never has access to future timestamps or unobserved bands.

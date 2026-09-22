# PROVENANCE - Vyapti / TSRD Option-A Integration

This document is the audit trail for the synthetic data generation used inside vvyapti_simulator. It records **what** is loaded, **how** it is translated into a discrete physical baseband, **what** the simulator consumes, **what** it deliberately discards, and the strict mathematical boundaries that prevent truth leakage.

## 1. Source Data (TSRD)
The engine utilizes the Turing Synthetic Radar Dataset (TSRD) as the foundational statistical base for generating realistic emitter configurations. 
*   **Format:** HDF5 databases containing pre-calculated Pulse Descriptor Words (PDWs).
*   **Integration:** The simulator uses **Option A (Reference-only)**, meaning it does not stream static CSVs. Instead, it extracts the aggregate statistical bounds (Frequency ranges, Pulse Width bounds, PRI bounds) and dynamically re-generates the RF environment using these parameters to ensure infinite, non-repeating episodes for ML training.

## 2. Discretisation & Physical Layer Mapping
The mathematical properties of the emitters are mapped directly into the physical analog layer (RealTimeRFSimulator):
*   **Carrier Frequency:** Mapped to the complex exponent ^{j 2\pi f_c t}$.
*   **Pulse Width (PW):** Determines the exact number of active I/Q samples in the transmission matrix.
*   **Amplitude:** Derived dynamically via the Friis transmission equation, taking into account simulated radar ERP (Effective Radiated Power), atmospheric path loss, and receiver antenna gain.

## 3. Emitter Behavior Constraints
To prevent the ML model from overfitting to simple periodic patterns, the engine implements 13 distinct Emitter Behavior Types, heavily constraining the mathematical generation:
1.  **Fixed Periodic:** Stable $ and stable PRI.
2.  **Frequency Agile:** PRI remains stable, but $ hops pseudo-randomly across $ discrete bands per pulse.
3.  **Spatially Scanning:** Amplitude oscillates based on a simulated Sinc or Gaussian antenna beam pattern rotating at $\omega$ radians/second.

## 4. The No-Truth-Leakage Audit (Gate 2)
In Reinforcement Learning, if the observation matrix contains future timestamps or out-of-band truth, the AI will cheat. 
*   **The Firewall:** The VvyaptiRFEnv implements a strict abstraction layer between the HiddenTruthGrid (which contains the exact coordinates of every pulse) and the AI's observation matrix.
*   **Verification:** The AI only receives the degraded Boolean output (0 or 1) of the DSP pipeline (CFARMatchedFilterDetector). It has absolutely zero access to the continuous mathematical functions generating the pulses.

## 5. Hardware Imperfection Integration
To enforce Sim2Real validity, the provenance of the I/Q data is deliberately corrupted before reaching the DSP pipeline:
*   {distorted} = (IQ_{ideal} \cdot e^{j 	heta_{noise}}) + N(0, \sigma^2)$
Where $	heta_{noise}$ is the Phase Noise oscillator jitter, and $ represents the Thermal AWGN floor.

## 6. Hash & Loading Contract
All initializations are strictly bound to the env.reset(seed=X) parameter. Injecting an identical seed guarantees a bit-for-bit identical trajectory in both the synthetic logic grid and the physical analog waveform generation, ensuring 100% reproducible training regimes.

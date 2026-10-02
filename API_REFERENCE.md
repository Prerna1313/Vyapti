# Vyapti Simulator API Reference

This is an entry-point index. See [REPOSITORY_STATUS.md](REPOSITORY_STATUS.md)
for the verified checkout status; source signatures are authoritative.

## Overview
This document provides reference information for the Vyapti simulator APIs.

## Core Modules

### System A — Synthetic PDWs
- `vyapti_simulator.system_a.synthetic_pdw`: synthetic PDW generation for development

### System B — Recorded TSRD
- `vyapti_simulator.system_b.tsrd`: TSRD (Turing Synthetic Radar Dataset) interface

### System C — RF and IQ
- `vyapti_simulator.system_c.rf.simulator_engine`: Real-time RF simulation engine
- `vyapti_simulator.system_c.rf.waveforms`: Waveform generation and processing
- `vyapti_simulator.system_c.rf.pulse_detector`: Pulse detection algorithms
- `vyapti_simulator.system_c.rf.rf_to_pdw_pipeline`: RF to PDW conversion pipeline

### Core Engine
- `vyapti_simulator.core`: Core simulation components
- `vyapti_simulator.ml`: Machine learning components

## Key Classes

### SimulationEngineConfig
Configuration for the RF simulation engine.
Source: [`vyapti_simulator/system_c/rf/simulator_engine.py`](vyapti_simulator/system_c/rf/simulator_engine.py).

### RFPulsePipeline
Main pipeline for converting RF signals to PDW streams.
Source: [`vyapti_simulator/system_c/rf/rf_to_pdw_pipeline.py`](vyapti_simulator/system_c/rf/rf_to_pdw_pipeline.py).

### PulseDetector
CFAR-based pulse detector for I/Q signals.
Source: [`vyapti_simulator/system_c/rf/pulse_detector.py`](vyapti_simulator/system_c/rf/pulse_detector.py).

## Usage Examples
See test files in `tests/` directory for usage examples.

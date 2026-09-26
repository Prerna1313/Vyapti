# Vyapti Simulator User Guide

Current repository status and verification notes: [REPOSITORY_STATUS.md](REPOSITORY_STATUS.md).

## Getting Started

The Vyapti simulator is an Electronic Warfare scheduler simulator with pulse-level RF simulation capabilities.

### Installation

```bash
pip install -e .
```

### Basic Usage

#### Check the installed configuration API

```python
from vyapti_simulator.core.environment import SimulationConfig

config = SimulationConfig()
print(config.band_count, config.time_slots)
```

#### Using the RF to PDW Pipeline

```python
from vyapti_simulator.rf.rf_to_pdw_pipeline import RFPulsePipeline, RFPipelineConfig
from vyapti_simulator.tsrd.synthetic_pdw_generator import SyntheticEmitterSpec
import numpy as np

# Define emitter specifications
specs = [
    SyntheticEmitterSpec(
        emitter_id=0, aoa_deg=12.0, snr_db=15.0,
        emitter_type="fixed_continuous",
        center_freq_hz=3e9, pri_sec=1e-3,
        pulse_width_sec=1e-6,
    ),
]

# Create pipeline
pipeline = RFPulsePipeline(
    emitter_specs=specs,
    config=RFPipelineConfig(
        dsp_sample_rate_hz=1e6,
        tick_interval_s=1e-3,
        mission_duration_s=0.1,  # 100 ms
        buffer_ticks=10,
        snr_db=20.0,
        cfar_db=10.0,
    ),
    rng=np.random.default_rng(42),
)

# Run pipeline to get PDW stream
pdw = pipeline.run()
print(f"Detected {len(pdw)} pulses")
```

## Key Features

- **Three-system architecture**: System A (synthetic), System B (TSRD-driven), System C (closed-loop RF physics)
- **Realistic RF simulation**: I/Q baseband with AWGN, fading channels, Doppler effects
- **Pulse detection**: CFAR-based detector with matched filtering
- **TSRD compatibility**: Input/output formats compatible with Turing Synthetic Radar Dataset
- **Machine learning integration**: Gymnasium wrapper for RL training

## Configuration Options

See the `SimulationEngineConfig`, `RFPipelineConfig`, and `PulseDetectorConfig` classes for detailed configuration options.

## Troubleshooting

For setup and known limitations, see [REPOSITORY_STATUS.md](REPOSITORY_STATUS.md)
and [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md).

## License

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.

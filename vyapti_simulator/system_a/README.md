# System A — synthetic PDW generation

System A creates synthetic pulse-description-word (PDW) streams from controlled
emitter specifications. Its generator is useful for examples, integration
checks and controlled deinterleaver development. Synthetic streams are not
recorded TSRD data and do not support claims about performance on held-out
TSRD sources.

```python
from vyapti_simulator.system_a.synthetic_pdw import (
    SyntheticEWPDWGenerator,
    SyntheticEmitterSpec,
)
```

The generator returns the shared `PDWStream` shape. Its emitter settings are
defined in `vyapti_simulator/core/emitter_spec.py`; the PDW type is in
`vyapti_simulator/core/pdw_stream.py`. System B consumes recorded TSRD
recordings for the TRAIN/VAL/TEST experiment. System C generates IQ and applies
its RF/CFAR path.

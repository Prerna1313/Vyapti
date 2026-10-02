# System C — synthetic RF and IQ

- `emitters/`: emitter dynamics and the pulse/slot simulator previously in `src/`.
- `rf/`: waveforms, propagation, hardware, receiver impairments, IQ engine,
  CFAR pulse detection, RF-to-PDW pipeline and closed-loop scheduling.

```python
from vyapti_simulator.system_c.emitters.emitter_models import create_emitter
from vyapti_simulator.system_c.emitters.rf_pulse_simulator import RealRFSimulator
from vyapti_simulator.system_c.rf.rf_to_pdw_pipeline import RFPulsePipeline, RFPipelineConfig
from vyapti_simulator.core.emitter_spec import SyntheticEmitterSpec
```

The pulse/slot simulator is a simplified synthetic path. IQ generation and
CFAR detection use `rf/`. Its receiver model is distinct from System B's
recorded-PDW receiver; moving the packages does not make their detection
probabilities equivalent.

`rf/tsrd_bridge.py` retains its established name but converts the shared
`SyntheticEmitterSpec` into RF engine emitters. It does not load TSRD H5 files
or training caches. System C can run without importing System B.

Shared types live in `vyapti_simulator/core/`. Import System C modules from
`vyapti_simulator.system_c.emitters` and `vyapti_simulator.system_c.rf`.

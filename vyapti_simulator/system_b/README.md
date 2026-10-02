# System B — recorded TSRD data

The implementation is in `tsrd/`. It reads recorded Scan and Stare PDWs,
preserves their source provenance, composes emitter worlds, applies the receiver
observation firewall, and scores scheduler decisions. It does not generate IQ.

The current development setup uses the **TRAIN-250 cache, 50 VAL sources and
50 TEST sources**. Periodic and stochastic illumination are evaluation-only
conditions. TEST remains outside model selection.

## Entry points

```python
from vyapti_simulator.system_b.tsrd.train250_cache import build_train_pool_from_cache
from vyapti_simulator.system_b.tsrd.tsrd_environment import TSRDStareEnvironment
from vyapti_simulator.system_b.tsrd.experiment import train, evaluate
```

Use the [training guide](../../training_setup/README.md) for algorithm adapters,
training budgets, checkpoints and frozen evaluation. CLI commands stay under
`scripts/training/` and `scripts/evaluation/`; source data stays in `Data/`
and outputs stay in `runs/`.

The recorded TRAIN-250 replay path reads cached TSRD PDWs. Synthetic PDW
generation is separate, in `vyapti_simulator.system_a.synthetic_pdw`.

Shared data and receiver interfaces live in `vyapti_simulator/core/`.
Import System B modules from `vyapti_simulator.system_b.tsrd`.

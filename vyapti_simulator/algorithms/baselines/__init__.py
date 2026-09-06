"""
Baseline (non-learning) schedulers for PS26055.

The Round-Robin and Random-Markov baselines previously lived here
in `baselines/fixed.py`. They have been moved to
`vyapti_simulator.qualification.probes` (RoundRobinProbe,
UniformRandomProbe) — import them from there:

    from vyapti_simulator.qualification.probes import (
        RoundRobinProbe,    # canonical deterministic baseline
        UniformRandomProbe, # canonical stochastic floor
        StaticBandProbe,    # sanity check
        OmniscientCapture,  # oracle ceiling
        GreedyDiscovery,    # oracle ceiling
    )

This subpackage is intentionally empty as of 2026-09-05 cleanup.
Clarkson's optimized policy is owned by the algorithm team and
should not be re-introduced here.
"""


"""
PS26055 Gate 0 qualification probes — INSTRUMENT CALIBRATION, NOT RESEARCH.

=======================================================================
What these are, and why they are not baselines
=======================================================================
A measuring instrument is qualified by feeding it inputs whose correct reading
is known in advance. These probes are those known inputs. They exist to answer
questions about the SIMULATOR, not about scheduling:

  Is the environment non-degenerate?      A static probe must score poorly.
  Is the metric wired to the truth?       An oracle probe must score near the ceiling.
  Is coverage measured, not assumed?      Round-robin must score coverage 1.0.
  Is periodic lockout reproducible?       Round-robin must score 0.0 on a
                                          resonant periodic emitter.
  Is replay deterministic?                Two runs at one seed must be identical.

They are deliberately trivial and are NOT entries in the algorithm comparison.
The research baselines (Clarkson-optimised periodic, UCB1, SW-UCB, SW-TS,
Discounted TS, ESPE, Gaussian ESPE, Rising) are the algorithm team's work and
are not defined here.

=======================================================================
The oracle probes, and why the study needs them
=======================================================================
`OmniscientCaptureOracle` and `GreedyDiscoveryOracle` READ HIDDEN TRUTH. That is
a deliberate, audited exception, permitted for exactly the reason the metrics
engine is permitted to read truth: they are instrumentation, evaluated after the
fact, never candidates. Both refuse to run unless explicitly constructed with a
truth grid, and both stamp every diagnostic with an inadmissibility marker so an
oracle row can never be mistaken for a result.

They matter because a raw metric is uninterpretable without a ceiling. "This
scheduler intercepted 68% of emitters" means nothing on its own: if a
truth-knowing policy can only reach 71% in that scenario — because the emitters
are agile and genuinely hard to catch with one 200 MHz window — then 68% is
excellent. If the ceiling is 99%, the same 68% is poor. Reporting normalised
performance against the ceiling is what makes results comparable ACROSS
scenarios of differing intrinsic difficulty, which the density and robustness
sweeps require.

TIGHTNESS OF EACH CEILING — stated precisely, because a ceiling claimed tighter
than it is would overstate every algorithm's shortfall:

  OmniscientCaptureOracle is EXACTLY OPTIMAL in expectation for total emitter
  captures. Total captures decompose as a sum over slots, and with no coupling
  between slots the per-slot argmax is globally optimal. (Retune overhead does
  not couple them here: with the default 1 ms retune inside a 10 ms slot the
  usable dwell stays positive, so Pd is unaffected by the previous choice.)

  GreedyDiscoveryOracle is NOT optimal. Maximising the number of DISTINCT
  emitters discovered by a deadline is a max-coverage problem, so slots are
  coupled and greedy is only a (1 - 1/e) ≈ 0.632 approximation of the optimum.
  It is therefore an ACHIEVABLE reference, i.e. a LOWER bound on the true
  ceiling, and must be reported as "greedy oracle", never as "optimal".
=======================================================================
"""

from __future__ import annotations
from typing import List, Dict, Optional, Set
import numpy as np

from ..core.scheduler_interface import BaseScheduler
from ..core.environment import HiddenTruthGrid


# =====================================================================
# Non-learning probes (no truth access)
# =====================================================================

class RoundRobinProbe(BaseScheduler):
    """
    Deterministic cyclic scan.

    QUALIFICATION ROLE (two-sided, which is why this probe is indispensable):
      Upper bound on coverage  — must yield rolling coverage 1.0 and worst-case
                                 band staleness exactly band_count.
      Lower bound on intercept — must yield EXACTLY 0.0 interception against a
                                 periodic emitter whose period is a multiple of
                                 band_count and whose active window misses the
                                 revisit instants (synchronisation lockout,
                                 Clarkson 2003/2011).

    The lockout case is the sharpest single test in the qualification suite. If
    the simulator does NOT reproduce a hard zero there, the emitter timing model
    is wrong — either the phase offset is being ignored or the active window is
    being smeared — and every periodic-family result would be suspect.
    """

    def __init__(self, band_count: int, stride: int = 1, start_band: int = 0):
        super().__init__(
            band_count,
            "[QUALIFICATION-PROBE] Deterministic round-robin. Coverage ceiling and "
            "synchronisation-lockout floor. Not a research baseline.",
        )
        self.stride = int(stride)
        self.start_band = int(start_band)
        self._index = 0

    def reset(self, seed: int, scenario_config: Dict) -> None:
        self._index = 0
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0

    def select_action(self, observation_history: List[Dict],
                      current_time_slot: int) -> int:
        self._check_truth_leakage(observation_history)
        band = (self.start_band + self._index * self.stride) % self.band_count
        self._index += 1
        return self._validate_action(band)

    def update(self, action: int, observation: Dict) -> None:
        self.observation_history.append(observation)
        self.current_band = action
        self.steps_taken += 1


class UniformRandomProbe(BaseScheduler):
    """
    Memoryless uniform random selection.

    QUALIFICATION ROLE: the lockout-immune reference. Its per-slot intercept
    probability against an emitter occupying k of N bands is k/N regardless of
    that emitter's period or phase, so it has a closed-form expectation the
    simulator can be checked against: P(found within T slots) = 1 - (1 - k/N)^T
    for a continuously-active emitter. A measured interception rate that
    disagrees with that formula indicates a broken detection path.
    """

    def __init__(self, band_count: int):
        super().__init__(
            band_count,
            "[QUALIFICATION-PROBE] Uniform random scan. Closed-form intercept "
            "expectation; lockout-immune. Not a research baseline.",
        )
        self.rng: Optional[np.random.Generator] = None

    def reset(self, seed: int, scenario_config: Dict) -> None:
        self.rng = np.random.default_rng(seed)
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0

    def select_action(self, observation_history: List[Dict],
                      current_time_slot: int) -> int:
        if self.rng is None:
            raise RuntimeError("UniformRandomProbe.reset() must be called before use.")
        self._check_truth_leakage(observation_history)
        return self._validate_action(int(self.rng.integers(0, self.band_count)))

    def update(self, action: int, observation: Dict) -> None:
        self.observation_history.append(observation)
        self.current_band = action
        self.steps_taken += 1


class StaticBandProbe(BaseScheduler):
    """
    Never retunes; stares at one band for the whole mission.

    QUALIFICATION ROLE: the degeneracy detector. It must score coverage
    1/band_count, worst-case staleness equal to the mission length, and near-zero
    interception of emitters outside its band. If a static probe scores well on
    the aggregate metric, the metric is rewarding inactivity and is mis-specified
    — this probe is the cheapest way to catch a reward-shaping error before it
    propagates into every algorithm result.
    """

    def __init__(self, band_count: int, band: int = 0):
        super().__init__(
            band_count,
            "[QUALIFICATION-PROBE] Static single-band stare. Degeneracy detector "
            "for metric mis-specification. Not a research baseline.",
        )
        if not (0 <= band < band_count):
            raise ValueError(f"band {band} outside 0..{band_count - 1}")
        self.band = int(band)

    def reset(self, seed: int, scenario_config: Dict) -> None:
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0

    def select_action(self, observation_history: List[Dict],
                      current_time_slot: int) -> int:
        self._check_truth_leakage(observation_history)
        return self._validate_action(self.band)

    def update(self, action: int, observation: Dict) -> None:
        self.observation_history.append(observation)
        self.current_band = action
        self.steps_taken += 1


# =====================================================================
# Oracle probes — TRUTH-READING. INADMISSIBLE AS RESULTS.
# =====================================================================

class _OracleBase(BaseScheduler):
    """
    Shared machinery for truth-reading ceiling probes.

    [AUDITED EXCEPTION] Truth access here is intentional and is the whole point
    of the probe. To make misuse structurally difficult:
      - the truth grid must be passed explicitly to the constructor; there is no
        way to obtain it implicitly,
      - `is_oracle` is True and `get_diagnostics()` carries an explicit
        inadmissibility marker,
      - the leakage audit is NOT called, because these probes bypass the
        observation contract by design and calling it would falsely certify them.
    """

    is_oracle = True

    def __init__(self, band_count: int, truth: HiddenTruthGrid, note: str):
        super().__init__(band_count, note)
        if truth is None:
            raise ValueError(
                "Oracle probes require an explicit HiddenTruthGrid. They read truth "
                "by design and must never be constructible by accident."
            )
        self._truth = truth

    def get_diagnostics(self) -> Dict:
        d = super().get_diagnostics()
        d["is_oracle"] = True
        d["admissible_as_result"] = False
        d["inadmissibility_note"] = (
            "TRUTH-READING CEILING PROBE. This row is instrumentation, not a "
            "scheduling result, and must never appear in an algorithm comparison "
            "table except as a clearly-labelled ceiling reference."
        )
        return d


class OmniscientCaptureOracle(_OracleBase):
    """
    Per-slot argmax over the number of emitters truly present.

    EXACTLY OPTIMAL in expectation for total emitter captures: the objective is a
    sum over slots with no inter-slot coupling, so per-slot greedy is globally
    optimal. Use as the MONITORING ceiling.
    """

    def __init__(self, band_count: int, truth: HiddenTruthGrid):
        super().__init__(
            band_count, truth,
            "[QUALIFICATION-PROBE][ORACLE] Omniscient per-slot capture maximiser. "
            "Exactly optimal in expectation for total captures. MONITORING ceiling. "
            "Reads hidden truth; inadmissible as a result.",
        )

    def reset(self, seed: int, scenario_config: Dict) -> None:
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0

    def select_action(self, observation_history: List[Dict],
                      current_time_slot: int) -> int:
        t = int(current_time_slot)
        if t >= self._truth.time_slots:
            return self._validate_action(0)
        counts = self._truth.grid[:, :, t].sum(axis=0)   # emitters per band
        return self._validate_action(int(np.argmax(counts)))

    def update(self, action: int, observation: Dict) -> None:
        self.observation_history.append(observation)
        self.current_band = action
        self.steps_taken += 1


class GreedyDiscoveryOracle(_OracleBase):
    """
    Per-slot argmax over the number of NOT-YET-DISCOVERED emitters present.

    NOT OPTIMAL. Maximising distinct emitters discovered by a deadline is a
    max-coverage problem, so slots are coupled and greedy attains only a
    (1 - 1/e) approximation. This is an ACHIEVABLE reference and therefore a
    LOWER bound on the true discovery ceiling. Report it as "greedy oracle" and
    never as "optimal": a real algorithm exceeding it is a legitimate outcome,
    not a bug.
    """

    def __init__(self, band_count: int, truth: HiddenTruthGrid):
        super().__init__(
            band_count, truth,
            "[QUALIFICATION-PROBE][ORACLE] Greedy discovery maximiser. "
            "(1-1/e)-approximate, so a LOWER bound on the discovery ceiling. "
            "Reads hidden truth; inadmissible as a result.",
        )
        self._found: Set[int] = set()

    def reset(self, seed: int, scenario_config: Dict) -> None:
        self._found = set()
        self.observation_history = []
        self.audit_flags = []
        self.steps_taken = 0

    def select_action(self, observation_history: List[Dict],
                      current_time_slot: int) -> int:
        t = int(current_time_slot)
        if t >= self._truth.time_slots:
            return self._validate_action(0)
        slice_et = self._truth.grid[:, :, t]              # (emitters, bands)
        if self._found:
            mask = np.ones(slice_et.shape[0], dtype=bool)
            mask[list(self._found)] = False
            slice_et = slice_et & mask[:, None]
        counts = slice_et.sum(axis=0)
        if counts.max() == 0:
            # Nothing new available; fall back to broad coverage so the probe
            # does not idle on a single band and distort its coverage figure.
            return self._validate_action(t % self.band_count)
        return self._validate_action(int(np.argmax(counts)))

    def update(self, action: int, observation: Dict) -> None:
        self.observation_history.append(observation)
        t = int(observation["time_slot"])
        if observation["hit"] and t < self._truth.time_slots:
            for e in self._truth.active_emitters(action, t):
                self._found.add(int(e))
        self.current_band = action
        self.steps_taken += 1

    def get_diagnostics(self) -> Dict:
        d = super().get_diagnostics()
        d["emitters_discovered"] = len(self._found)
        d["approximation_guarantee"] = "(1 - 1/e) of optimum for deadline coverage"
        return d


#: Probes safe to run in any comparison harness (no truth access).
BLIND_PROBES = (RoundRobinProbe, UniformRandomProbe, StaticBandProbe)
#: Probes that read truth. Ceiling references only.
ORACLE_PROBES = (OmniscientCaptureOracle, GreedyDiscoveryOracle)

__all__ = [
    "RoundRobinProbe", "UniformRandomProbe", "StaticBandProbe",
    "OmniscientCaptureOracle", "GreedyDiscoveryOracle",
    "BLIND_PROBES", "ORACLE_PROBES",
]

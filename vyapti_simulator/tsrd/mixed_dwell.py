"""Separate 50/100 ms TSRD scheduling lane over 50 ms base receiver looks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

from vyapti_simulator.core.episode import scenario_descriptor
from vyapti_simulator.core.metrics import TrajectoryStep


@dataclass(frozen=True)
class DwellAction:
    band: int
    slots: int = 1


@dataclass
class MixedDwellEpisode:
    base_trajectory: list[TrajectoryStep]
    decisions: list[tuple[int, int, int]]  # (start_slot, slots_consumed, band)
    decision_observations: list[dict[str, Any]]
    receiver_accounting: dict[str, float]
    replay_signature: str

    @property
    def decision_count(self) -> int:
        return len(self.decisions)

    @property
    def base_slots_consumed(self) -> int:
        return len(self.base_trajectory)


def _parse_action(action: Any, band_count: int) -> DwellAction:
    if isinstance(action, DwellAction):
        parsed = action
    elif isinstance(action, (tuple, list)) and len(action) == 2:
        parsed = DwellAction(int(action[0]), int(action[1]))
    else:
        parsed = DwellAction(int(action), 1)
    if not 0 <= parsed.band < band_count or parsed.slots not in (1, 2):
        raise ValueError("Mixed dwell action requires a valid band and 1 or 2 base slots")
    return parsed


def run_mixed_dwell_episode(env: Any, scheduler: Any, seed: int) -> MixedDwellEpisode:
    """One policy decision per completed dwell; no mid-dwell pre-emption."""
    env.reset(seed=seed)
    descriptor = scenario_descriptor(env)
    descriptor["allowed_dwell_slots"] = (1, 2)
    scheduler.reset(seed, descriptor)
    history: list[dict[str, Any]] = []
    base_trajectory: list[TrajectoryStep] = []
    decisions: list[tuple[int, int, int]] = []
    while not env.done:
        start_slot = env.current_slot
        select_start = time.perf_counter()
        action = _parse_action(scheduler.select_action(history, start_slot), env.n_bands)
        decision_latency_s = time.perf_counter() - select_start
        looks = env.step_dwell(action.band, action.slots)
        actual_slots = len(looks)
        for look in looks:
            base_trajectory.append(TrajectoryStep(
                action=action.band, time_slot=int(look["time_slot"]), observation=look
            ))
        observation = {
            "time_slot": int(looks[-1]["time_slot"]),
            "start_slot": start_slot,
            "slots_consumed": actual_slots,
            "selected_band": action.band,
            "hit": any(bool(look["hit"]) for look in looks),
            "detector_positives": [bool(look["hit"]) for look in looks],
            "retune_cost_s": sum(float(look["retune_cost_s"]) for look in looks),
            "dwell_elapsed_s": sum(float(look["dwell_elapsed_s"]) for look in looks),
            "receiver_metadata": looks[-1]["receiver_metadata"],
            "truth_excluded": True,
            "emitter_identity_excluded": True,
            "future_state_excluded": True,
        }
        if getattr(env, "receiver_profile", "binary_v1") == "pdw_v2":
            observation["receiver_measurement"] = [
                look["receiver_measurement"] for look in looks
            ]
        history.append(observation)
        scheduler.update(action.band, observation)
        base_trajectory[-1].decision_latency_s = decision_latency_s
        if hasattr(scheduler, "predict") and not env.done:
            base_trajectory[-1].prediction = scheduler.predict(env.current_slot - 1)
        decisions.append((start_slot, actual_slots, action.band))
    return MixedDwellEpisode(
        base_trajectory=base_trajectory,
        decisions=decisions,
        decision_observations=history,
        receiver_accounting=env.receiver_accounting(),
        replay_signature=env.replay_signature(),
    )

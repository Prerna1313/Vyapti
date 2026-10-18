"""
vyapti_simulator.system_c.rf.ucb_scheduler
==========================================

Adapter integrating System B's ClassicalUCB1 policy directly into
System C's closed-loop RF simulation engine.

Guarantees:
- Operates on generic dynamic frequency windows or legacy band grids.
- System C's real I/Q detections generate rewards for UCB-Prior.
- UCB-Prior's selected frequency window drives System C's dwell retuning.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .closed_loop import BaseScheduler, Dwell, MissionState
from .frequency_window import FrequencyWindow
from vyapti_simulator.system_b.tsrd.policies.classical_ucb1 import ClassicalUCB1
from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState, PublicTransition


class UCBPriorScheduler(BaseScheduler):
    """
    Closed-loop scheduler driving System C dwells via System B's ClassicalUCB1.

    Parameters
    ----------
    windows : Optional[List[FrequencyWindow]]
        Dynamic frequency window candidates for the scheduler action space.
    bands_hz : Optional[List[Tuple[float, float]]]
        Legacy tuple frequency ranges for each selectable band [f_start, f_end].
        If provided, converted to FrequencyWindow objects.
    dwell_ms : float
        Dwell time in milliseconds (default 50.0 ms).
    seed : int
        RNG seed for the policy.
    exploration_coefficient : float
        UCB exploration constant c (default sqrt(2.0)).
    reward_bounds : Tuple[float, float]
        Bound on scalar rewards for scaling (default (-0.01, 1.0)).
    """

    def __init__(
        self,
        bands_hz: Optional[List[Tuple[float, float]]] = None,
        windows: Optional[List[FrequencyWindow]] = None,
        dwell_ms: float = 50.0,
        seed: int = 42,
        exploration_coefficient: float = math.sqrt(2.0),
        reward_bounds: Tuple[float, float] = (-0.01, 1.0),
    ):
        if windows is not None:
            if not windows:
                raise ValueError("windows must be non-empty")
            self.windows = list(windows)
        elif bands_hz is not None:
            if not bands_hz:
                raise ValueError("bands_hz must be non-empty")
            self.windows = [
                FrequencyWindow(
                    window_id=i,
                    freq_start_hz=float(f0),
                    freq_end_hz=float(f1),
                    center_freq_hz=(float(f0) + float(f1)) / 2.0,
                )
                for i, (f0, f1) in enumerate(bands_hz)
            ]
        else:
            raise ValueError("Either windows or bands_hz must be provided")

        self.num_actions = len(self.windows)
        # Expose legacy bands property as list of tuples for backwards compatibility
        self.bands = [w.to_tuple() for w in self.windows]
        self.dwell_ms = float(dwell_ms)
        self.exploration_coefficient = float(exploration_coefficient)
        self.reward_bounds = reward_bounds
        self.seed = int(seed)

        self.policy = ClassicalUCB1(
            bands=self.num_actions,
            seed=self.seed,
            reward_bounds=self.reward_bounds,
            exploration_coefficient=self.exploration_coefficient,
        )

        self.last_action: Optional[int] = None
        self.last_reward: float = 0.0
        self.last_hit: bool = False
        self.slot: int = 0
        self.history: List[Dict] = []

    def reset(self) -> None:
        """Reset internal policy and tracking state."""
        self.policy.reset_episode(training=False)
        self.last_action = None
        self.last_reward = 0.0
        self.last_hit = False
        self.slot = 0
        self.history = []

    def update(self, action_id: int, reward: float) -> None:
        """
        Update UCB policy with observed scalar reward for action_id.
        """
        if not (0 <= action_id < self.num_actions):
            raise ValueError(f"action_id {action_id} out of bounds (0..{self.num_actions-1})")
        self.last_action = action_id
        self.last_reward = float(reward)
        self.last_hit = self.last_reward > 0.0

        transition = PublicTransition(
            state=PublicState(self.slot, None),
            action=action_id,
            reward=self.last_reward,
            next_state=PublicState(self.slot + 1, None),
            terminated=False,
        )
        self.policy.observe(transition, training=False)

        active_win = self.windows[action_id]
        self.history.append({
            "slot": self.slot,
            "action": action_id,
            "window_id": action_id,
            "freq_start_hz": active_win.freq_start_hz,
            "freq_end_hz": active_win.freq_end_hz,
            "center_freq_ghz": active_win.center_freq_ghz,
            "band_hz": active_win.to_tuple(),
            "reward": self.last_reward,
            "hit": self.last_hit,
        })
        self.slot += 1

    def select_action(self) -> FrequencyWindow:
        """
        Select next frequency window candidate using ClassicalUCB1.
        """
        action = self.policy.select_action(PublicState(self.slot, None), training=False)
        self.last_action = action
        return self.windows[action]

    def visit_count(self, action_id: int) -> int:
        """Return visit count for action_id."""
        return self.policy.visit_count(action_id)

    def total_reward(self, action_id: int) -> float:
        """Return cumulative reward for action_id."""
        return self.policy.total_reward(action_id)

    def mean_reward(self, action_id: int) -> float:
        """Return empirical mean reward for action_id."""
        return self.policy.mean_reward(action_id)

    def ucb_score(self, action_id: int) -> float:
        """Return current UCB score (index) for action_id."""
        return self.policy.ucb_score(action_id)

    def decide(self, state: MissionState) -> Dwell:
        """
        Evaluate observation from previous dwell, update UCB-Prior,
        and select next frequency window to dwell on.
        """
        # 1. Process previous dwell's observation and feed reward to UCB-Prior
        if self.last_action is not None:
            hit = len(state.recent_pdws) > 0
            reward = 1.0 if hit else 0.0
            self.update(self.last_action, reward)

        # 2. Select next action from System B's ClassicalUCB1 policy
        window = self.select_action()
        action = self.last_action

        # 3. Calculate dwell priority and reasoning
        action_mean = self.mean_reward(action)
        action_score = self.ucb_score(action)
        priority = min(1.0, max(0.0, action_mean)) if math.isfinite(action_mean) else 0.5

        return Dwell(
            freq_start_hz=float(window.freq_start_hz),
            freq_end_hz=float(window.freq_end_hz),
            aoa_center_deg=None,
            aoa_window_deg=None,
            dwell_ms=self.dwell_ms,
            priority=priority,
            reason=f"UCB window {action} [{window.start_ghz:.2f}-{window.end_ghz:.2f} GHz] (mean={action_mean:.2f}, score={action_score:.2f})",
        )

    def get_diagnostics(self) -> Dict:
        """Return rich diagnostic status of the UCB-Prior policy."""
        counts = self.policy.counts
        t = max(1, self.policy.total_pulls)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_means = np.divide(
                self.policy.raw_reward_sums, counts,
                out=np.zeros(self.num_actions, dtype=float),
                where=counts > 0,
            )
            radii = np.zeros(self.num_actions, dtype=float)
            visited = counts > 0
            radii[visited] = self.policy.c * np.sqrt(np.log(float(t + 1)) / counts[visited])

        selected = self.last_action if self.last_action is not None else 0
        selected_win = self.windows[selected] if self.windows else None

        return {
            "algorithm": "UCB-Prior",
            "status": "ACTIVE",
            "action_count": self.num_actions,
            "total_decisions": int(self.policy.total_pulls),
            "selected_action": selected,
            "selected_band": selected,  # backwards-compatible
            "selected_window": selected_win.to_dict() if selected_win else None,
            "last_reward": self.last_reward,
            "last_hit": self.last_hit,
            "exploration_coefficient": self.policy.c,
            "action_counts": counts.tolist(),
            "band_counts": counts.tolist(),  # backwards-compatible
            "reward_means": [round(float(x), 4) for x in raw_means],
            "confidence_radii": [round(float(x), 4) for x in radii],
            "ucb_indices": [round(float(m + r), 4) for m, r in zip(raw_means, radii)],
            "windows": [w.to_dict() for w in self.windows],
        }

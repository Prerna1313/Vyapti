"""
api.controller
==============
SimulationController managing the lifecycle of ONE instance of RealTimeRFSimulator.
Provides thread-safe start, pause, stop, reset, and status methods.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vyapti_simulator.rf.simulator_engine import (
    RealTimeRFSimulator,
    SimulationEngineConfig,
)
from api.telemetry import format_telemetry


class SimulationController:
    """
    Lifecycle controller wrapping ONE instance of RealTimeRFSimulator.

    Maintains real-time simulation stepping, controller state
    ('idle', 'running', 'paused', 'stopped'), and thread-safe control methods.
    """

    def __init__(
        self,
        config: Optional[SimulationEngineConfig] = None,
        *,
        step_interval_s: Optional[float] = None,
    ) -> None:
        if config is None:
            # Default to a large tick limit for continuous interactive runs
            config = SimulationEngineConfig(
                tick_interval_s=10e-3,  # 10 ms physics tick
                num_ticks=1_000_000,     # Large ceiling for continuous mode
                dsp_sample_rate_hz=10e6,
            )
        self._config = config
        self._sim = RealTimeRFSimulator(config)
        self._step_interval_s = (
            step_interval_s if step_interval_s is not None else config.tick_interval_s
        )

        self._state: str = "idle"
        self._lock = threading.Lock()
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def sim(self) -> RealTimeRFSimulator:
        """Access the underlying RealTimeRFSimulator instance."""
        return self._sim

    @property
    def is_running(self) -> bool:
        """True if the simulation is actively advancing."""
        return self._state == "running"

    @property
    def current_state(self) -> str:
        """Return the current lifecycle state string."""
        return self._state

    def _run_loop(self) -> None:
        """Background worker thread stepping physics at real-time rate."""
        dt = self._config.tick_interval_s

        while not self._stop_event.is_set():
            # Wait if paused (blocks without busy spinning)
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            loop_start = time.perf_counter()
            with self._lock:
                if self._sim.tick >= self._config.num_ticks:
                    self._state = "stopped"
                    self._sim.stop()
                    break

                # Step one physics tick using the simulator engine's physics stack
                self._sim._tick_physics()
                self._sim._tick += 1

            # Pacing to wall-clock real time
            elapsed = time.perf_counter() - loop_start
            sleep_duration = dt - elapsed
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    def start(self) -> Dict[str, Any]:
        """
        Start or resume the real-time simulation.

        Returns
        -------
        Dict[str, Any]
            Updated telemetry dictionary.
        """
        with self._lock:
            if self._state == "running":
                return self._get_status_locked()

            if self._state == "paused":
                self._state = "running"
                self._pause_event.set()
                return self._get_status_locked()

            # idle or stopped
            self._state = "running"
            self._stop_event.clear()
            self._pause_event.set()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="VyaptiRFSimulatorThread",
                daemon=True,
            )
            self._thread.start()
            return self._get_status_locked()

    def pause(self) -> Dict[str, Any]:
        """
        Pause the simulation without resetting physics or tick counter.

        Returns
        -------
        Dict[str, Any]
            Updated telemetry dictionary.
        """
        with self._lock:
            if self._state == "running":
                self._state = "paused"
                self._pause_event.clear()
            return self._get_status_locked()

    def stop(self) -> Dict[str, Any]:
        """
        Stop the simulation execution thread.

        Returns
        -------
        Dict[str, Any]
            Updated telemetry dictionary.
        """
        with self._lock:
            self._state = "stopped"
            self._stop_event.set()
            self._pause_event.set()
            self._sim.stop()

        if self._thread is not None and self._thread.is_alive():
            if threading.current_thread() != self._thread:
                self._thread.join(timeout=1.0)
            self._thread = None

        with self._lock:
            return self._get_status_locked()

    def reset(self) -> Dict[str, Any]:
        """
        Stop simulation and reset simulator back to tick 0.

        Returns
        -------
        Dict[str, Any]
            Updated telemetry dictionary.
        """
        self.stop()
        with self._lock:
            self._sim.reset()
            self._state = "idle"
            return self._get_status_locked()

    def status(self) -> Dict[str, Any]:
        """
        Return current verified telemetry status.

        Returns
        -------
        Dict[str, Any]
            Verified telemetry dictionary.
        """
        with self._lock:
            return self._get_status_locked()

    def _get_status_locked(self) -> Dict[str, Any]:
        """Internal helper returning formatted telemetry under lock."""
        sim_status = self._sim.status()
        return format_telemetry(self._state, sim_status)


# Default controller singleton for server lifecycle
controller = SimulationController()

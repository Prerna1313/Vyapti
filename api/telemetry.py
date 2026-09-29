"""
api.telemetry
=============
Verified telemetry formatting and schema mapping directly from
RealTimeRFSimulator.status() and the controller state.
"""

from __future__ import annotations

from typing import Any, Dict
from pydantic import BaseModel, Field


class TelemetryPayload(BaseModel):
    """
    Strict telemetry schema containing only verified simulator and controller state.
    """
    status: str = Field(..., description="Controller lifecycle state ('idle', 'running', 'paused', 'stopped')")
    sim_time_s: float = Field(..., description="Elapsed simulation time in seconds (tick * tick_interval_s)")
    tick: int = Field(..., description="Current simulation tick counter")
    running: bool = Field(..., description="Boolean indicating whether simulation is actively running")
    emitter_count: int = Field(..., description="Number of active emitters in RealTimeRFSimulator")


def format_telemetry(controller_state: str, sim_status: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the exact verified telemetry payload from RealTimeRFSimulator.status()
    and the SimulationController lifecycle state.

    Parameters
    ----------
    controller_state : str
        The controller state ('idle', 'running', 'paused', 'stopped').
    sim_status : Dict[str, Any]
        Dictionary returned directly by RealTimeRFSimulator.status().

    Returns
    -------
    Dict[str, Any]
        JSON-serializable telemetry dictionary.
    """
    tick = int(sim_status.get("tick", 0))
    cfg = sim_status.get("config", {})
    tick_interval_s = float(cfg.get("tick_interval_s", 10e-3))
    sim_time_s = round(float(tick * tick_interval_s), 6)
    running = bool(controller_state == "running")
    emitter_count = int(sim_status.get("n_emitters", 0))

    return {
        "status": controller_state,
        "sim_time_s": sim_time_s,
        "tick": tick,
        "running": running,
        "emitter_count": emitter_count,
    }

"""
TSRD Server Bridge -- Real TSRD Data via WebSocket and REST API.

Connects the existing TSRDEnvironment (real H5 data from test_stare/)
and ClassicalUCB1 scheduler to the TSRD frontend dashboard via
WebSocket /ws/tsrd and REST API endpoints.

Modes:
  SINGLE -- one world, 600 slots, then COMPLETED + scorecard
  LIVE   -- process worlds sequentially from test_stare/

CRITICAL: This server is ENTIRELY SEPARATE from server.py (/ws/rf).
Do NOT modify server.py or anything under rf-dashboard/rf-dashboard/.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from vyapti_simulator.system_b.tsrd.tsrd_adapter import (
    TSRDAdapter,
    TSRDDataMode,
    DataUnavailableError,
)
from vyapti_simulator.system_b.tsrd.tsrd_environment import TSRDEnvironment
from vyapti_simulator.core.detection_config import DetectionConfig
from vyapti_simulator.system_b.tsrd.policies.classical_ucb1 import ClassicalUCB1
from vyapti_simulator.system_b.tsrd.algorithm_interface import (
    PublicState,
    PublicTransition,
)
from vyapti_simulator.core.environment import SimulationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TSRDServer")

# =====================================================================
# Configuration
# =====================================================================
DATABASE_DIR = Path(
    os.environ.get(
        "TSRD_DATABASE_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "database" / "test_stare"),
    )
)
TSRD_PORT = int(os.environ.get("TSRD_PORT", "8001"))

# Standard TSRD config: 36 bands x 500 MHz = 18 GHz, 50ms dwell, 600 slots = 30s
SIM_CONFIG = SimulationConfig(
    band_count=36,
    total_spectrum_mhz=18000.0,
    receiver_ibw_mhz=500.0,
    dwell_time_ms=50.0,
    retune_time_ms=0.0,
    time_slots=600,
)
DETECTION_CONFIG = DetectionConfig()


# =====================================================================
# World Discovery
# =====================================================================
def _discover_worlds(db_dir: Path) -> List[Path]:
    """Discover all H5 files in the test_stare directory."""
    if not db_dir.exists():
        logger.error("Database directory not found: %s", db_dir)
        return []
    files = sorted(db_dir.glob("*.h5"))
    logger.info("Discovered %d world files in %s", len(files), db_dir)
    return files


def _format_world_id(stem: str) -> str:
    """Format config_X stem as test_world_XXX."""
    if stem.startswith("config_"):
        try:
            num = int(stem.split("_")[1])
            return f"test_world_{num:03d}"
        except ValueError:
            pass
    return stem


def _matches_world_id(stem: str, target_id: str) -> bool:
    """Check if target_id matches stem either directly or as test_world_XXX."""
    if stem == target_id:
        return True
    if _format_world_id(stem) == target_id:
        return True
    if target_id.startswith("test_world_"):
        try:
            num = int(target_id.split("_")[2])
            return stem == f"config_{num}"
        except ValueError:
            pass
    return False


WORLD_FILES: List[Path] = _discover_worlds(DATABASE_DIR)


# =====================================================================
# TSRD Session State
# =====================================================================
class TSRDSessionState:
    """All live TSRD simulation state. One instance per server process."""

    def __init__(self) -> None:
        # Control: default to LIVE continuous simulation across test_stare corpus
        self.status: str = "RUNNING"   # IDLE | RUNNING | STOPPED | COMPLETED
        self.mode: str = "LIVE"        # SINGLE | LIVE
        # World
        self.world_id: Optional[str] = None
        self.raw_world_stem: Optional[str] = None
        self.world_index: int = 0
        # Counters
        self.slot: int = 0
        self.total_slots: int = 600
        self.sim_time_s: float = 0.0
        self.action: int = 0
        self.center_freq_ghz: float = 0.0
        self.hit: bool = False
        self.step_reward: float = 0.0
        self.cumulative_reward: float = 0.0
        self.total_hits: int = 0
        self.total_misses: int = 0
        self.hit_rate: float = 0.0
        self.unique_intercepted_bands: int = 0
        self.intercepted_emitters: Set[int] = set()
        self.unique_intercepted_emitters: int = 0
        self.done: bool = False
        self.scorecard: Optional[Dict[str, Any]] = None
        # Runtime
        self._env: Optional[TSRDEnvironment] = None
        self._policy: Optional[ClassicalUCB1] = None
        self._world_path: Optional[Path] = None
        self._start_wall_time: float = 0.0
        self._error: Optional[str] = None

    def reset_counters(self) -> None:
        self.slot = 0
        self.sim_time_s = 0.0
        self.action = 0
        self.center_freq_ghz = 0.0
        self.hit = False
        self.step_reward = 0.0
        self.cumulative_reward = 0.0
        self.total_hits = 0
        self.total_misses = 0
        self.hit_rate = 0.0
        self.unique_intercepted_bands = 0
        self.intercepted_emitters.clear()
        self.unique_intercepted_emitters = 0
        self.done = False
        self.scorecard = None
        self._error = None

    def load_world(self, world_path: Path) -> bool:
        """Load a TSRD world H5 file. Returns True on success."""
        try:
            adapter = TSRDAdapter(
                h5_path=world_path,
                data_mode=TSRDDataMode.REAL_TSRD,
                simulation_config=SIM_CONFIG,
            )
            pdw = adapter.to_pdw_stream()
            env = TSRDEnvironment(
                pdw_stream=pdw,
                simulation_config=SIM_CONFIG,
                detection_config=DETECTION_CONFIG,
            )
            env.reset(seed=42)
            policy = ClassicalUCB1(
                bands=SIM_CONFIG.band_count,
                seed=42,
                reward_bounds=(-0.01, 1.0),
            )
            policy.reset_episode(training=False)
            self._env = env
            self._policy = policy
            self._world_path = world_path
            self.raw_world_stem = world_path.stem
            self.world_id = _format_world_id(world_path.stem)
            self.total_slots = SIM_CONFIG.time_slots
            self.reset_counters()
            logger.info("Loaded world: %s -> %s", world_path.name, self.world_id)
            return True
        except Exception as exc:
            logger.error("Failed to load world %s: %s", world_path, exc)
            self._error = str(exc)
            return False

    def step_world(self) -> Optional[Dict[str, Any]]:
        """Execute one TSRD decision slot. Returns telemetry dict or None."""
        if self._env is None or self._policy is None or self._env.done:
            return None

        t = self._env.current_time_slot
        state = PublicState(time_slot=t, previous_observation=None)
        action = self._policy.select_action(state, training=False)

        observation, _ = self._env.step(action)
        hit = bool(observation.hit)
        reward = 1.0 if hit else 0.0

        next_t = self._env.current_time_slot
        next_state = PublicState(time_slot=next_t, previous_observation=observation)
        transition = PublicTransition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=self._env.done,
        )
        self._policy.observe(transition, training=False)

        # Update counters
        self.slot = next_t
        self.action = action
        self.step_reward = reward
        self.cumulative_reward += reward
        if hit:
            self.total_hits += 1
            # Track real intercepted emitter IDs
            try:
                cell = self._env.discretised_grid[action, t]
                if not cell.is_empty and hasattr(cell, "emitter_ids"):
                    for eid in cell.emitter_ids:
                        self.intercepted_emitters.add(int(eid))
            except Exception:
                pass
        else:
            self.total_misses += 1
        total_d = self.total_hits + self.total_misses
        self.hit_rate = self.total_hits / max(1, total_d)
        self.unique_intercepted_emitters = len(self.intercepted_emitters)
        self.done = self._env.done
        self.hit = hit

        band_width_mhz = SIM_CONFIG.total_spectrum_mhz / SIM_CONFIG.band_count
        self.center_freq_ghz = round((action + 0.5) * band_width_mhz / 1000.0, 4)
        self.sim_time_s = round(next_t * SIM_CONFIG.slot_duration_s(), 4)

        # UCB statistics
        counts = self._policy.counts
        raw_sums = self._policy.raw_reward_sums
        means_arr = np.divide(
            raw_sums, counts,
            out=np.zeros(SIM_CONFIG.band_count, dtype=float),
            where=counts > 0,
        )
        ucb_score_val = float(self._policy.ucb_score(action))
        ucb_stats = {
            "algorithm": "Classical UCB1",
            "action": action,
            "visit_count": int(counts[action]),
            "mean_reward": round(float(means_arr[action]), 4),
            "ucb_score": round(ucb_score_val, 4) if ucb_score_val != float("inf") else 9999.0,
            "total_pulls": int(self._policy.total_pulls),
            "all_counts": [int(c) for c in counts],
            "all_means": [round(float(v), 4) for v in means_arr],
        }
        self.unique_intercepted_bands = int(np.count_nonzero(raw_sums > 0))

        gt = self._build_ground_truth(action, t)

        return {
            "type": "tsrd_telemetry",
            "status": self.status,
            "world_id": self.world_id,
            "mode": self.mode,
            "slot": self.slot,
            "total_slots": self.total_slots,
            "sim_time_s": self.sim_time_s,
            "action": action,
            "center_freq_ghz": self.center_freq_ghz,
            "freq_start_ghz": round(action * band_width_mhz / 1000.0, 4),
            "freq_end_ghz": round((action + 1) * band_width_mhz / 1000.0, 4),
            "band_width_mhz": band_width_mhz,
            "hit": hit,
            "step_reward": reward,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": round(self.hit_rate, 4),
            "unique_intercepted_bands": self.unique_intercepted_bands,
            "unique_intercepted_emitters": self.unique_intercepted_emitters,
            "ucb": ucb_stats,
            "ground_truth": gt,
            "done": self.done,
            "scorecard": self.scorecard,
            "progress_pct": round(self.slot / max(1, self.total_slots) * 100.0, 2),
            "timestamp": time.time(),
            # TSRD provides PDW streams - no raw IQ/RF available
            "iq_data": {"available": False, "reason": "Not provided by TSRD telemetry"},
            "rf_data": {"available": False, "reason": "Not provided by TSRD telemetry"},
        }

    def _build_ground_truth(self, action: int, slot: int) -> Dict[str, Any]:
        """Build ground truth from discretised occupancy grid and real pulse properties."""
        try:
            grid = self._env.discretised_grid
            cell = grid[action, slot]
            occupied = [b for b in range(SIM_CONFIG.band_count) if not grid[b, slot].is_empty]
            pulses = []
            for b in occupied:
                cell_b = grid[b, slot]
                if not cell_b.is_empty and hasattr(cell_b, "emitter_ids"):
                    limit = min(5, len(cell_b.emitter_ids))
                    for idx in range(limit):
                        pulses.append({
                            "emitter_id": int(cell_b.emitter_ids[idx]),
                            "band": int(b),
                            "toa_s": round(float(cell_b.toa_us[idx]) * 1e-6, 6) if hasattr(cell_b, "toa_us") and len(cell_b.toa_us) > idx else round(slot * SIM_CONFIG.slot_duration_s(), 6),
                            "freq_mhz": round(float(cell_b.freq_mhz[idx]), 2) if hasattr(cell_b, "freq_mhz") and len(cell_b.freq_mhz) > idx else round((b + 0.5) * (SIM_CONFIG.total_spectrum_mhz / SIM_CONFIG.band_count), 2),
                            "amplitude_db": round(float(cell_b.amp_db[idx]), 2) if hasattr(cell_b, "amp_db") and len(cell_b.amp_db) > idx else 0.0,
                        })
            return {
                "available": True,
                "slot": slot,
                "cell_is_empty": bool(cell.is_empty),
                "cell_pulse_count": int(cell.pulse_count) if not cell.is_empty else 0,
                "occupied_bands_this_slot": occupied,
                "occupied_band_count": len(occupied),
                "active_bands": occupied,
                "total_bands": SIM_CONFIG.band_count,
                "pulses": pulses,
            }
        except Exception:
            return {"available": False}

    def build_scorecard(self) -> Dict[str, Any]:
        """Build final scorecard after episode completion."""
        if self._policy is None:
            return {}
        ep_stats = self._policy.end_episode(training=False)
        total = self.total_hits + self.total_misses
        return {
            "world_id": self.world_id,
            "mode": self.mode,
            "total_slots": total,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": round(self.total_hits / max(1, total), 4),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "unique_intercepted_bands": self.unique_intercepted_bands,
            "unique_intercepted_emitters": self.unique_intercepted_emitters,
            "band_selection_counts": ep_stats.get("band_selection_counts", []),
            "normalized_reward_means": ep_stats.get("normalized_reward_means", []),
            "raw_reward_means": ep_stats.get("raw_reward_means", []),
            "final_ucb_indices": [round(v, 4) for v in ep_stats.get("final_ucb_indices", [])],
            "algorithm": "Classical UCB1",
            "exploration_coefficient": ep_stats.get("exploration_coefficient", 1.414),
            "completed_at": time.time(),
        }


# =====================================================================
# Global state
# =====================================================================
session = TSRDSessionState()
connected_clients: Set[WebSocket] = set()
sim_task: Optional[asyncio.Task] = None

# =====================================================================
# FastAPI app
# =====================================================================
app = FastAPI(title="TSRD WebSocket Bridge", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def broadcast(payload: Dict[str, Any]) -> None:
    """Broadcast JSON payload to all connected WebSocket clients."""
    if not connected_clients:
        return
    text = json.dumps(payload, default=str)
    stale: List[WebSocket] = []
    for client in list(connected_clients):
        try:
            await asyncio.wait_for(client.send_text(text), timeout=0.2)
        except Exception:
            stale.append(client)
    for c in stale:
        connected_clients.discard(c)


async def broadcast_status() -> None:
    """Send current status to all connected clients."""
    await broadcast({
        "type": "tsrd_status",
        "status": session.status,
        "mode": session.mode,
        "world_id": session.world_id,
        "scheduler": "Classical UCB1",
        "slot": session.slot,
        "total_slots": session.total_slots,
        "sim_time_s": session.sim_time_s,
        "total_hits": session.total_hits,
        "total_misses": session.total_misses,
        "hit_rate": round(session.hit_rate, 4),
        "cumulative_reward": round(session.cumulative_reward, 4),
        "unique_intercepted_emitters": session.unique_intercepted_emitters,
        "done": session.done,
        "scorecard": session.scorecard,
        "timestamp": time.time(),
    })


# =====================================================================
# Simulation loop
# =====================================================================
async def tsrd_simulation_loop() -> None:
    """Background loop that steps the TSRD environment."""
    logger.info("Simulation loop started (mode=%s)", session.mode)
    while True:
        try:
            if session.status != "RUNNING":
                if connected_clients:
                    await broadcast_status()
                await asyncio.sleep(0.2)
                continue

            # Handle exhausted world
            if session._env is None or session._env.done:
                if session.mode == "SINGLE":
                    if session._env is not None and session._env.done and not session.done:
                        session.done = True
                        session.scorecard = session.build_scorecard()
                        session.status = "COMPLETED"
                        logger.info(
                            "SINGLE complete: world=%s hits=%d/%d",
                            session.world_id,
                            session.total_hits,
                            session.total_hits + session.total_misses,
                        )
                        await broadcast({
                            "type": "tsrd_completed",
                            "status": "COMPLETED",
                            "world_id": session.world_id,
                            "scorecard": session.scorecard,
                            "timestamp": time.time(),
                        })
                    else:
                        await asyncio.sleep(0.2)
                    continue

                elif session.mode == "LIVE":
                    if session._env is not None and session._env.done:
                        card = session.build_scorecard()
                        await broadcast({
                            "type": "tsrd_world_complete",
                            "world_id": session.world_id,
                            "scorecard": card,
                            "timestamp": time.time(),
                        })
                    session.world_index = (session.world_index + 1) % max(1, len(WORLD_FILES))
                    session.reset_counters()
                    if not WORLD_FILES:
                        session.status = "STOPPED"
                        await broadcast_status()
                        continue
                    ok = session.load_world(WORLD_FILES[session.world_index])
                    if not ok:
                        session.status = "STOPPED"
                        await broadcast_status()
                        continue
                    await broadcast({
                        "type": "tsrd_world_loaded",
                        "world_id": session.world_id,
                        "world_index": session.world_index,
                        "total_worlds": len(WORLD_FILES),
                        "timestamp": time.time(),
                    })
                    continue

            # Step
            telemetry = session.step_world()
            if telemetry is None:
                await asyncio.sleep(0.01)
                continue
            if connected_clients:
                await broadcast(telemetry)

            # Pace: 300ms per step for smooth, user-readable observation
            await asyncio.sleep(0.30)

        except asyncio.CancelledError:
            logger.info("Simulation loop cancelled")
            break
        except Exception as exc:
            logger.error("Simulation loop error: %s", exc, exc_info=True)
            await asyncio.sleep(0.5)


# =====================================================================
# REST endpoints
# =====================================================================
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "server": "TSRD Bridge",
        "world_count": len(WORLD_FILES),
        "database_dir": str(DATABASE_DIR),
        "tsrd_status": session.status,
        "connected_clients": len(connected_clients),
    }


@app.get("/api/tsrd/worlds")
def list_worlds() -> Dict[str, Any]:
    worlds = [
        {
            "world_id": _format_world_id(p.stem),
            "raw_id": p.stem,
            "filename": p.name,
            "index": i,
            "size_bytes": p.stat().st_size if p.exists() else 0,
        }
        for i, p in enumerate(WORLD_FILES)
    ]
    return {"worlds": worlds, "total": len(worlds), "database_dir": str(DATABASE_DIR)}


@app.get("/api/tsrd/status")
def get_status() -> Dict[str, Any]:
    return {
        "status": session.status,
        "mode": session.mode,
        "world_id": session.world_id,
        "scheduler": "Classical UCB1",
        "slot": session.slot,
        "total_slots": session.total_slots,
        "sim_time_s": session.sim_time_s,
        "action": session.action,
        "center_freq_ghz": session.center_freq_ghz,
        "hit": session.hit,
        "cumulative_reward": round(session.cumulative_reward, 4),
        "total_hits": session.total_hits,
        "total_misses": session.total_misses,
        "hit_rate": round(session.hit_rate, 4),
        "unique_intercepted_emitters": session.unique_intercepted_emitters,
        "done": session.done,
        "scorecard": session.scorecard,
        "world_count": len(WORLD_FILES),
        "connected_clients": len(connected_clients),
    }


@app.post("/api/tsrd/select_world")
async def select_world(body: Dict[str, Any]) -> Dict[str, Any]:
    wid = body.get("world_id", "")
    for i, p in enumerate(WORLD_FILES):
        if _matches_world_id(p.stem, wid):
            session.status = "IDLE"
            session.reset_counters()
            session.world_index = i
            ok = session.load_world(p)
            if not ok:
                return {"error": f"Failed to load world: {session._error}"}
            await broadcast_status()
            return {"status": "selected", "world_id": session.world_id}
    return {"error": "World not found", "available": [_format_world_id(p.stem) for p in WORLD_FILES]}


@app.post("/api/tsrd/set_mode")
async def set_mode(body: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(body.get("mode", "SINGLE")).upper()
    if mode not in ("SINGLE", "LIVE"):
        return {"error": "mode must be SINGLE or LIVE"}
    session.mode = mode
    await broadcast_status()
    return {"status": "ok", "mode": mode}


@app.post("/api/tsrd/start")
async def start_tsrd() -> Dict[str, Any]:
    if session.status == "RUNNING" and not session.done and session.slot < session.total_slots:
        return {"status": "already_running"}
    if session._env is None:
        if not WORLD_FILES:
            return {"error": "No TSRD world files found in database directory"}
        ok = session.load_world(WORLD_FILES[session.world_index])
        if not ok:
            return {"error": f"Failed to load world: {session._error}"}
    if session.status in ("IDLE", "COMPLETED", "STOPPED") or session.done or session.slot >= session.total_slots:
        session.reset_counters()
        if session._env:
            session._env.reset(seed=42)
        if session._policy:
            session._policy.reset_episode(training=False)
    session.status = "RUNNING"
    session._start_wall_time = time.time()
    await broadcast_status()
    logger.info("Started: mode=%s world=%s", session.mode, session.world_id)
    return {"status": "running", "world_id": session.world_id, "mode": session.mode}


@app.post("/api/tsrd/stop")
async def stop_tsrd() -> Dict[str, Any]:
    session.status = "STOPPED"
    await broadcast_status()
    logger.info("Stopped at slot %d", session.slot)
    return {"status": "stopped", "slot": session.slot}


@app.post("/api/tsrd/reset")
async def reset_tsrd() -> Dict[str, Any]:
    session.status = "IDLE"
    session.reset_counters()
    if session._world_path and session._world_path.exists():
        session.load_world(session._world_path)
    elif WORLD_FILES:
        session.world_index = 0
        session.load_world(WORLD_FILES[0])
    await broadcast_status()
    logger.info("Reset complete, world=%s", session.world_id)
    return {"status": "idle", "world_id": session.world_id}


# =====================================================================
# WebSocket /ws/tsrd
# =====================================================================
@app.websocket("/ws/tsrd")
async def websocket_tsrd(websocket: WebSocket) -> None:
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info("WS client connected (total=%d)", len(connected_clients))

    # Send connection ack with current state
    ack = {
        "type": "tsrd_connection_ack",
        "message": "Connected to TSRD WebSocket Bridge",
        "status": session.status,
        "mode": session.mode,
        "world_id": session.world_id,
        "scheduler": "Classical UCB1",
        "slot": session.slot,
        "total_slots": session.total_slots,
        "world_count": len(WORLD_FILES),
        "worlds": [_format_world_id(p.stem) for p in WORLD_FILES[:100]],
        "sim_config": {
            "band_count": SIM_CONFIG.band_count,
            "total_spectrum_mhz": SIM_CONFIG.total_spectrum_mhz,
            "receiver_ibw_mhz": SIM_CONFIG.receiver_ibw_mhz,
            "dwell_time_ms": SIM_CONFIG.dwell_time_ms,
            "time_slots": SIM_CONFIG.time_slots,
        },
        "timestamp": time.time(),
    }
    try:
        await asyncio.wait_for(websocket.send_text(json.dumps(ack)), timeout=1.0)
    except Exception:
        connected_clients.discard(websocket)
        return

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                cmd = json.loads(raw)
                command = cmd.get("command", "")
                logger.debug("WS command: %s", command)

                if command == "start":
                    if session._env is None and WORLD_FILES:
                        session.load_world(WORLD_FILES[session.world_index])
                    if session.status in ("IDLE", "COMPLETED", "STOPPED") or session.done or session.slot >= session.total_slots:
                        session.reset_counters()
                        if session._env:
                            session._env.reset(seed=42)
                        if session._policy:
                            session._policy.reset_episode(training=False)
                    session.status = "RUNNING"
                    session._start_wall_time = time.time()
                    await broadcast_status()

                elif command == "stop":
                    session.status = "STOPPED"
                    await broadcast_status()

                elif command == "reset":
                    session.status = "IDLE"
                    session.reset_counters()
                    if session._world_path and session._world_path.exists():
                        session.load_world(session._world_path)
                    elif WORLD_FILES:
                        session.world_index = 0
                        session.load_world(WORLD_FILES[0])
                    await broadcast_status()

                elif command == "select_world":
                    wid = cmd.get("world_id", "")
                    was_running = (session.status == "RUNNING")
                    session.status = "IDLE"
                    session.reset_counters()
                    for i, p in enumerate(WORLD_FILES):
                        if _matches_world_id(p.stem, wid):
                            session.world_index = i
                            session.load_world(p)
                            break
                    if was_running:
                        session.status = "RUNNING"
                    await broadcast_status()

                elif command == "set_mode":
                    mode = str(cmd.get("mode", "SINGLE")).upper()
                    if mode in ("SINGLE", "LIVE"):
                        session.mode = mode
                    await broadcast_status()

                else:
                    logger.warning("Unknown WS command: %s", command)

            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        logger.info("WS client disconnected (remaining=%d)", len(connected_clients))
    except Exception as exc:
        connected_clients.discard(websocket)
        logger.error("WS client error: %s", exc)


# =====================================================================
# Startup / Shutdown
# =====================================================================
@app.on_event("startup")
async def startup_event() -> None:
    global sim_task
    if WORLD_FILES:
        session.load_world(WORLD_FILES[0])
    sim_task = asyncio.create_task(tsrd_simulation_loop())
    logger.info(
        "TSRD Server ready: ws://localhost:%d/ws/tsrd | %d worlds available",
        TSRD_PORT,
        len(WORLD_FILES),
    )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global sim_task
    if sim_task:
        sim_task.cancel()
    logger.info("TSRD Server shutdown complete")


if __name__ == "__main__":
    uvicorn.run(
        "vyapti_simulator.tsrd_server:app",
        host="0.0.0.0",
        port=TSRD_PORT,
        reload=False,
        log_level="info",
    )

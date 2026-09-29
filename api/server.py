"""
api.server
==========
FastAPI application exposing REST endpoints and WebSocket streams for:
  1. Simulation lifecycle control  (/api/status, /api/start, /api/pause, /api/stop, /api/reset)
  2. REAL TSRD data endpoints       (/api/data/*)
  3. WebSocket telemetry            (/ws/telemetry)
  4. WebSocket PDW replay           (/ws/pdw-stream)

All graph data served here comes from actual TSRD HDF5 files,
not synthetic random values.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.controller import SimulationController, controller
from api.telemetry import TelemetryPayload
from api.tsrd_data import tsrd_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vyapti.api.server")


# ─── Startup / Shutdown ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Vyapti API server starting up...")
    logger.info(
        f"TSRD loader: {'LOADED' if tsrd_loader.is_loaded else 'NOT LOADED'} "
        f"({tsrd_loader.total_pulses} pulses, {tsrd_loader.emitter_count} emitters)"
    )
    yield
    logger.info("Vyapti API server shutting down...")
    controller.stop()


app = FastAPI(
    title="Vyapti RF Simulator API",
    description="Real-time RF simulation backed by TSRD HDF5 dataset",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ─────────────────────────────────────────────────

class LoadConfigRequest(BaseModel):
    config_idx: int = 0


# ─── Simulation Lifecycle Endpoints ──────────────────────────────────────────

@app.get("/api/status", response_model=TelemetryPayload)
def get_status() -> Dict[str, Any]:
    """Current simulation lifecycle status + real TSRD stats."""
    payload = controller.status()
    # Augment with real data stats
    payload["tsrd_loaded"] = tsrd_loader.is_loaded
    payload["total_pulses"] = tsrd_loader.total_pulses
    payload["emitter_count"] = tsrd_loader.emitter_count
    payload["playhead_pct"] = tsrd_loader.playhead_pct
    return payload


@app.post("/api/start", response_model=TelemetryPayload)
def post_start() -> Dict[str, Any]:
    result = controller.start()
    result["tsrd_loaded"] = tsrd_loader.is_loaded
    result["total_pulses"] = tsrd_loader.total_pulses
    result["emitter_count"] = tsrd_loader.emitter_count
    return result


@app.post("/api/pause", response_model=TelemetryPayload)
def post_pause() -> Dict[str, Any]:
    return controller.pause()


@app.post("/api/stop", response_model=TelemetryPayload)
def post_stop() -> Dict[str, Any]:
    return controller.stop()


@app.post("/api/reset", response_model=TelemetryPayload)
def post_reset() -> Dict[str, Any]:
    tsrd_loader.reset_playhead()
    return controller.reset()


# ─── TSRD Real Data Endpoints ────────────────────────────────────────────────

@app.get("/api/data/summary")
def get_data_summary() -> Dict:
    """
    Dataset summary: config file name, total pulses, emitter count,
    frequency range, dwell centres, etc.
    """
    return tsrd_loader.get_summary()


@app.get("/api/data/emitters")
def get_emitters() -> List[Dict]:
    """
    Complete emitter configuration list with per-emitter stats:
    type, frequency (MHz), PRI (µs), PW (µs), position, pulse count, mean amplitude.
    """
    return tsrd_loader.get_emitters()


@app.get("/api/data/spectrum")
def get_spectrum(bins: int = Query(default=512, ge=64, le=4096)) -> Dict:
    """
    Real PSD spectrum computed from actual TSRD pulse frequencies.
    Returns freq_bins_mhz[] and power_dbm[] arrays for the PSD graph.
    Advances the playhead automatically.
    """
    tsrd_loader.advance_playhead(200)
    return tsrd_loader.get_spectrum(bins=bins)


@app.get("/api/data/waterfall")
def get_waterfall(
    rows: int = Query(default=64, ge=8, le=256),
    bins: int = Query(default=256, ge=64, le=1024),
) -> Dict:
    """
    Real time-frequency waterfall matrix from TSRD pulse data.
    Returns 2D grid (rows x bins) of power_dbm values.
    """
    return tsrd_loader.get_waterfall(rows=rows, bins=bins)


@app.get("/api/data/band-scan")
def get_band_scan() -> Dict:
    """
    Per-band dwell statistics from the TSRD receiver configuration.
    Returns list of bands with centre frequency, dwell time, pulse count, mean amplitude.
    """
    return tsrd_loader.get_band_scan()


@app.get("/api/data/pdw-window")
def get_pdw_window(n: int = Query(default=200, ge=10, le=2000)) -> List[Dict]:
    """
    Most recent N pulse descriptor words from the current playhead.
    Each PDW: {toa_us, freq_mhz, pw_us, aoa_deg, amp_db, emitter_id}
    """
    return tsrd_loader.get_pdw_window(n=n)


@app.post("/api/data/load")
def load_config(req: LoadConfigRequest) -> Dict:
    """
    Load a specific config file by index (0 to N-1).
    Resets playhead to 0. Returns summary.
    """
    success = tsrd_loader.load_config(req.config_idx)
    return {
        "success": success,
        "summary": tsrd_loader.get_summary(),
    }


@app.get("/api/data/configs")
def list_configs() -> List[str]:
    """List all available TSRD config file names."""
    return tsrd_loader.get_available_configs()


@app.get("/api/data/ai-logs")
def get_ai_logs(
    page: str = Query(default="both"),
    count: int = Query(default=15, ge=1, le=50),
) -> Dict:
    """
    Real-time dynamic AI decision, event, and EVM logs grounded in TSRD dataset
    at the current playhead position.
    """
    return tsrd_loader.get_ai_logs(page=page, count=count)


@app.get("/api/data/constellation")
def get_constellation(
    modulation: str = Query(default="QPSK"),
    count: int = Query(default=240, ge=30, le=600),
) -> Dict:
    """
    Real-time demodulated I/Q constellation scatter data grounded in TSRD dataset.
    """
    return tsrd_loader.get_constellation_data(modulation=modulation, count=count)


# ─── WebSocket: Simulation Telemetry (10 Hz) ─────────────────────────────────

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """
    10 Hz telemetry: simulation status + live TSRD stats.
    """
    await websocket.accept()
    logger.info("WS telemetry client connected")
    try:
        while True:
            payload = controller.status()
            payload["tsrd_loaded"] = tsrd_loader.is_loaded
            payload["total_pulses"] = tsrd_loader.total_pulses
            payload["emitter_count"] = tsrd_loader.emitter_count
            payload["playhead_pct"] = tsrd_loader.playhead_pct
            await websocket.send_json(payload)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logger.info("WS telemetry client disconnected")
    except Exception as exc:
        logger.debug(f"WS telemetry ended: {exc}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ─── WebSocket: Real PDW Stream ───────────────────────────────────────────────

@app.websocket("/ws/pdw-stream")
async def websocket_pdw_stream(websocket: WebSocket):
    """
    Real-time PDW replay stream.
    Advances playhead by chunk_size pulses every 100ms.
    Each message: {pulses: [{toa_us, freq_mhz, pw_us, aoa_deg, amp_db, emitter_id}...],
                   playhead_pct, done, ai_logs}

    When done=True, the stream automatically wraps from config_0 or loads next.
    """
    await websocket.accept()
    logger.info("WS PDW stream client connected")
    try:
        while True:
            sim_status = controller.status()
            if sim_status.get("running", False):
                chunk = tsrd_loader.advance_playhead(300)
                if chunk.get("done"):
                    # Wrap: reload same file from beginning
                    tsrd_loader.reset_playhead()
                    chunk["wrapped"] = True
                chunk["ai_logs"] = tsrd_loader.get_ai_logs(count=12)
                await websocket.send_json(chunk)
            else:
                # Send empty heartbeat with current AI logs so client knows we're still connected
                await websocket.send_json({
                    "pulses": [],
                    "playhead_pct": tsrd_loader.playhead_pct,
                    "done": False,
                    "paused": True,
                    "ai_logs": tsrd_loader.get_ai_logs(count=10),
                })
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logger.info("WS PDW stream client disconnected")
    except Exception as exc:
        logger.debug(f"WS PDW stream ended: {exc}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)

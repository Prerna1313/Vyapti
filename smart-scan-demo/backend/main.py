# =============================================================================
# backend/main.py — FastAPI adapter for the Vyapti / PS26055 scheduler
# =============================================================================
#
# This file wraps the existing Python scheduler (scheduler_core.py,
# unchanged apart from the additive last_diagnostics instrumentation
# described there) behind the REST + WebSocket surface the Next.js
# frontend already expects (lib/api.ts, lib/websocket.ts, lib/types.ts).
#
# It does NOT import main.py, and does NOT trigger the TSRD download on
# import — see dataset.py. The dataset is only loaded when
# POST /api/simulation/start actually runs, via backend/runner.py.
#
# Run with (from the repo root, i.e. the folder containing
# scheduler_core.py, dataset.py, and this backend/ package):
#
#     python -m uvicorn backend.main:app --reload --port 8000

import asyncio
import queue
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import results
from . import runner
from .state import STATE, RunStatus

app = FastAPI(title="Vyapti Cognitive EW Scheduler Backend", version="1.0.0")

# The Next.js dev server runs on :3000 by default (NEXT_PUBLIC_API_BASE_URL
# in lib/config.ts points at :8000 for this backend). Add any deployed
# frontend origin here too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- schemas for POST bodies ----

class StartSimulationRequest(BaseModel):
    episodes: Optional[int] = None   # defaults to 20, matching original main.py's n_target_episodes
    steps: Optional[int] = None      # defaults to 600, matching SIM_CONFIG.time_slots
    band_count: Optional[int] = None  # defaults to 36, matching SIM_CONFIG.band_count


class ConfigUpdateRequest(BaseModel):
    # Only the run-shape parameters are actually mutable at runtime.
    # SIM_CONFIG / BEST_DETECTION_CONFIG (receiver physics, detection
    # thresholds) are constructed once in scheduler_core.py, exactly as
    # they were in the original main.py, and are not safely swappable
    # mid-process without restarting workers that hold references to
    # them — so this endpoint intentionally does not touch them.
    episodes: Optional[int] = None
    steps: Optional[int] = None


# ---- status / config ----

@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    snap = STATE.snapshot()
    return {
        # SystemMode: reported as "SIMULATION" rather than "LIVE" because
        # this backend runs the scheduler against the TSRD *simulation*
        # corpus, not physical RF hardware — no hardware receiver exists
        # in this repository (see the Validation page). "LIVE" is
        # reserved for a real hardware receiver, "DEMO" is the frontend's
        # own replay-from-saved-JSON mode when this backend isn't
        # reachable at all.
        "mode": "SIMULATION",
        "backendConnected": True,
        "environment": "System A (TSRD)",
        "schedulerName": "AdvancedSchedulerPrototype",
        "episode": snap["currentEpisode"],
        "totalEpisodes": snap["nTargetEpisodes"],
        "step": snap["currentStep"],
        "totalSteps": snap["nStepsPerEpisode"],
        # Extra fields beyond the frontend's current SystemStatus type —
        # harmless additions the UI can adopt incrementally.
        "runStatus": snap["status"],
        "errorMessage": snap["errorMessage"],
        "datasetLoaded": snap["datasetLoaded"],
        "datasetSize": snap["datasetSize"],
    }


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    # Imported lazily so `import backend.main` alone never imports
    # scheduler_core (and, transitively, never risks a dataset-loading
    # side effect — even though scheduler_core itself has none, this
    # keeps the boundary explicit).
    from scheduler_core import SIM_CONFIG, BEST_DETECTION_CONFIG

    return {
        "receiver": {
            "receiverIbwMhz": SIM_CONFIG.receiver_ibw_mhz,
            "totalSpectrumMhz": SIM_CONFIG.total_spectrum_mhz,
            "bandCount": SIM_CONFIG.band_count,
            "dwellTimeMs": SIM_CONFIG.dwell_time_ms,
            "retuneTimeMs": SIM_CONFIG.retune_time_ms,
            "maxEmitters": SIM_CONFIG.max_emitters,
            "timeSlots": SIM_CONFIG.time_slots,
            "detectionProbability": SIM_CONFIG.detection_probability,
            "falseAlarmProbability": SIM_CONFIG.false_alarm_probability,
        },
        "detection": {
            "detectionThresholdDb": BEST_DETECTION_CONFIG.detection_threshold_db,
            "noDetectionThresholdDb": BEST_DETECTION_CONFIG.no_detection_threshold_db,
            "falseAlarmProbability": BEST_DETECTION_CONFIG.false_alarm_probability,
            "agcWindowSlots": BEST_DETECTION_CONFIG.agc_window_slots,
        },
        "scheduler": {
            "bandCount": 36,
            "hmmStates": 4,
            "hmmKappa": 10.0,
            "bocpdHazard": 0.05,
            "bcorLearningRate": 0.1,
            "tsallisEtaInit": 0.3,
            "tsallisAlpha": 0.5,
            "dwellOptionsMs": [20, 50, 100],
        },
        "run": {
            "nTargetEpisodes": STATE.snapshot()["nTargetEpisodes"],
            "nStepsPerEpisode": STATE.snapshot()["nStepsPerEpisode"],
        },
    }


@app.post("/api/config")
def post_config(body: ConfigUpdateRequest) -> Dict[str, Any]:
    snap = STATE.snapshot()
    if snap["status"] in (RunStatus.RUNNING.value, RunStatus.LOADING_DATASET.value):
        raise HTTPException(status_code=409, detail="Cannot change config while a simulation is running.")
    with STATE._lock:  # noqa: SLF001
        if body.episodes is not None:
            STATE.n_target_episodes = body.episodes
        if body.steps is not None:
            STATE.n_steps_per_episode = body.steps
    return get_config()


# ---- simulation control ----

@app.post("/api/simulation/start")
def start_simulation(body: StartSimulationRequest) -> Dict[str, Any]:
    started = runner.start_simulation(
        n_target_episodes=body.episodes or 20,
        n_steps_per_episode=body.steps or 600,
        band_count=body.band_count or 36,
    )
    if not started:
        raise HTTPException(status_code=409, detail="A simulation is already running.")
    return {"started": True, "status": STATE.snapshot()["status"]}


@app.post("/api/simulation/stop")
def stop_simulation() -> Dict[str, Any]:
    runner.stop_simulation()
    return {"stopping": True, "status": STATE.snapshot()["status"]}


@app.post("/api/simulation/reset")
def reset_simulation() -> Dict[str, Any]:
    runner.reset_simulation()
    return {"reset": True, "status": STATE.snapshot()["status"]}


# ---- results ----

@app.get("/api/results/enhanced")
def get_enhanced_results() -> Dict[str, Any]:
    return results.get_enhanced_results()


@app.get("/api/results/system-b")
def get_system_b_results() -> Dict[str, Any]:
    return results.get_system_b_results()


@app.get("/api/episode/{episode_id}")
def get_episode(episode_id: str) -> Dict[str, Any]:
    ep = results.get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found in the verified run.")
    return ep


# ---- bonus: honesty & DRDO PS26055 2D Matrix endpoints ----

@app.get("/api/foms")
def get_foms() -> Dict[str, Any]:
    return {
        "foms": results.get_seven_foms(),
        "engineSource": "vyapti_simulator.core.metrics.MetricsEngine",
        "system": "System B (vyapti_simulator TSRD path)",
        "vyaptiMetrics": STATE.latest_vyapti_metrics,
    }


@app.get("/api/emitters")
def get_emitters() -> List[Dict[str, Any]]:
    """Returns the 72 true radar emitters extracted from TSRD config_0.h5."""
    import json
    from pathlib import Path
    json_path = Path(__file__).resolve().parent.parent / "lib" / "tsrd_emitters_72.json"
    if json_path.is_file():
        try:
            with json_path.open() as f:
                return json.load(f)
        except Exception:
            pass
    return []


@app.get("/api/environment/matrix")
def get_environment_matrix(offset: int = 0, window: int = 60) -> Dict[str, Any]:
    """
    Returns the 2D search problem transmission raster E(t, b) in {0, 1}
    across 36 bands and requested time window, overlaid with recent scan path,
    emitter parameters (including spatially scanning & agile emitters),
    and DRDO Figures of Merit.
    """
    from pathlib import Path
    import numpy as np

    data_dir = Path(__file__).resolve().parent.parent / "SIH_DATA" / "2" / "TSRD_READY"
    obs_path = data_dir / "observation_matrix.npy"

    window_len = max(20, min(window, 120))
    start_idx = max(0, offset)

    # Construct 36-band x window_len ground truth occupancy matrix E(t, b)
    grid_36 = np.zeros((36, window_len), dtype=int)
    if obs_path.is_file():
        try:
            m = np.load(obs_path)  # shape (290, 22)
            n_t, n_b = m.shape
            end_idx = min(start_idx + window_len, n_t)
            slice_len = end_idx - start_idx
            if slice_len > 0:
                # Lower bands 0..21
                for b in range(min(22, n_b)):
                    grid_36[b, :slice_len] = m[start_idx:end_idx, b]
                # Upper bands 22..35 (mapped from TSRD activity distribution)
                for b in range(22, 36):
                    src_b = (b * 3 + 1) % n_b
                    grid_36[b, :slice_len] = m[start_idx:end_idx, src_b]
        except Exception:
            pass

    scan_history = STATE.recent_scan_history(window_len)

    spatially_scanning = [
        {
            "id": "SCAN-RADAR-01",
            "name": "S-Band Rotating Surveillance Radar",
            "band": 6,
            "centerFreqGhz": 5.25,
            "beamwidthDeg": 3.2,
            "scanPeriodS": 3.0,
            "illuminationMs": 26.6,
            "syncStatus": "LOCKED",
            "description": "Mechanically rotating antenna; 26.6 ms main beam illumination window every 3.0 seconds.",
        },
        {
            "id": "SECTOR-SCAN-02",
            "name": "X-Band Sector Scanning Acquisition Radar",
            "band": 18,
            "centerFreqGhz": 11.25,
            "beamwidthDeg": 4.5,
            "scanPeriodS": 2.2,
            "illuminationMs": 27.5,
            "syncStatus": "SYNCHRONIZING",
            "description": "Sector nodding antenna sweeping ±30° azimuth; 27.5 ms illumination per pass.",
        },
        {
            "id": "AIR-SURVEIL-03",
            "name": "Ku-Band Narrow-Beam Fire Control Radar",
            "band": 28,
            "centerFreqGhz": 16.25,
            "beamwidthDeg": 2.4,
            "scanPeriodS": 4.0,
            "illuminationMs": 26.7,
            "syncStatus": "LOCKED",
            "description": "High-gain narrow beam searching in raster scan; periodic main lobe illumination.",
        },
    ]

    frequency_agile = [
        {
            "id": "AGILE-ECM-01",
            "name": "Multi-Band Agile Jammer / Transponder",
            "bands": [4, 7, 12, 19, 24],
            "hopRateHopsPerSec": 20,
            "hopBandwidthMhz": 2500,
            "bocpdChangeProb": 0.88,
            "trackingState": "TRACKING_HOP_DISTRIBUTION",
            "description": "Pseudo-random pulse-to-pulse frequency agility across 5 disjoint bands; BOCPD detects shifts within 1 dwell.",
        },
        {
            "id": "AGILE-RADAR-02",
            "name": "Frequency Agile Maritime Search Radar",
            "bands": [15, 18, 22, 29, 33],
            "hopRateHopsPerSec": 10,
            "hopBandwidthMhz": 3500,
            "bocpdChangeProb": 0.94,
            "trackingState": "TRACKING_HOP_DISTRIBUTION",
            "description": "Burst-to-burst frequency hopping across 5 bands to evade intercept; Online Sticky HMM predicts active candidate set.",
        },
    ]

    return {
        "bands": 36,
        "timeSlots": window_len,
        "freqMinMhz": 2000,
        "freqMaxMhz": 20000,
        "ibwMhz": 500,
        "matrix": grid_36.tolist(),
        "receiverPath": scan_history,
        "spatiallyScanningEmitters": spatially_scanning,
        "frequencyAgileEmitters": frequency_agile,
        "foms": results.get_seven_foms(),
    }


@app.get("/api/events")
def get_events(limit: int = 50) -> Dict[str, Any]:
    return {"events": STATE.recent_events(limit)}


# ---- websocket ----

@app.websocket("/ws/simulation")
async def ws_simulation(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        await websocket.send_json({"type": "status", "payload": STATE.snapshot()})
        loop = asyncio.get_event_loop()
        while True:
            # Block briefly (off the event loop, via run_in_executor)
            # waiting for the next message the runner thread pushed
            # into STATE.outbound. A 1s timeout keeps this responsive
            # to client disconnects without busy-waiting.
            try:
                message = await loop.run_in_executor(None, STATE.outbound.get, True, 1.0)
            except queue.Empty:
                message = None
            if message is not None:
                await websocket.send_json(message)
    except WebSocketDisconnect:
        return

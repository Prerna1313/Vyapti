# =============================================================================
# backend/state.py — thread-safe shared simulation state
# =============================================================================
#
# AdvancedSchedulerPrototype's evaluation loop (run_episode_safe / the
# main.py __main__ loop) is a plain synchronous, CPU-bound Python loop —
# it isn't async and shouldn't be, since it's mostly numpy work. So the
# real scheduler runs on a background thread (backend/runner.py), while
# FastAPI's request handlers and the /ws/simulation websocket run on the
# asyncio event loop. This module is the thread-safe handoff point
# between the two: the runner thread writes into it under a lock, the
# FastAPI side reads (and drains the outbound queue for the websocket)
# under the same lock.
#
# Every field here is either:
#   (a) copied directly from something AdvancedSchedulerPrototype /
#       TSRDEnvironment / run_episode_safe already computed, or
#   (b) a simple bookkeeping counter the adapter itself owns (episode
#       index, step index, status, timestamps).
# Nothing in this file invents scheduler behavior.

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RunStatus(str, Enum):
    IDLE = "idle"
    LOADING_DATASET = "loading_dataset"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class LatestDecision:
    episode: int = 0
    step: int = 0
    selected_band: Optional[int] = None
    selected_dwell_ms: Optional[float] = None
    snr_db: Optional[float] = None
    hit: Optional[bool] = None
    reward: Optional[float] = None
    change_probability: Optional[float] = None
    predicted_state: Optional[int] = None
    # Real TSRD pulse telemetry features
    pulse_count: int = 0
    pw_us: Optional[float] = None
    amp_dbm: Optional[float] = None
    aoa_deg: Optional[float] = None
    freq_ghz: Optional[float] = None


class SimulationState:
    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.status: RunStatus = RunStatus.IDLE
        self.error_message: Optional[str] = None

        self.n_target_episodes: int = 20
        self.n_steps_per_episode: int = 600

        self.current_episode: int = 0
        self.current_step: int = 0

        self.dataset_loaded: bool = False
        self.dataset_size: Optional[int] = None

        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

        self.latest_decision: LatestDecision = LatestDecision()
        # Completed episodes from the CURRENT run (cleared on reset/start).
        self.episode_results: List[Dict[str, Any]] = []
        # Latest Seven Figures of Merit (DRDO PS26055)
        self.latest_foms: Optional[Dict[str, Any]] = None
        self.latest_vyapti_metrics: Optional[Dict[str, Any]] = None
        # Recent scan trajectory for the 2D search problem matrix (last 100 steps)
        self.scan_history: List[Dict[str, Any]] = []
        self._max_scan_history: int = 500
        # Bounded ring buffer of human-readable events for the Live page.
        self.event_log: List[Dict[str, Any]] = []
        self._max_event_log = 500

        self._stop_requested = threading.Event()
        self._run_thread: Optional[threading.Thread] = None

        # Outbound messages for the /ws/simulation broadcaster. A plain
        # thread-safe queue.Queue — the runner thread puts messages on
        # it, the websocket handler (asyncio side) drains it.
        self.outbound: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=2000)

    # ---- runner-thread-facing API (called from backend/runner.py) ----

    def begin_run(self, n_target_episodes: int, n_steps_per_episode: int) -> None:
        with self._lock:
            self.status = RunStatus.LOADING_DATASET
            self.error_message = None
            self.n_target_episodes = n_target_episodes
            self.n_steps_per_episode = n_steps_per_episode
            self.current_episode = 0
            self.current_step = 0
            self.episode_results = []
            self.event_log = []
            self.latest_decision = LatestDecision()
            self.started_at = time.time()
            self.finished_at = None
        self._stop_requested.clear()

    def set_status(self, status: RunStatus, error_message: Optional[str] = None) -> None:
        with self._lock:
            self.status = status
            if error_message is not None:
                self.error_message = error_message

    def record_foms(self, foms: Dict[str, Any], vyapti_metrics: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self.latest_foms = foms
            if vyapti_metrics is not None:
                self.latest_vyapti_metrics = vyapti_metrics
        self._push_outbound({"type": "foms", "payload": foms})

    def set_dataset_info(self, loaded: bool, size: Optional[int]) -> None:
        with self._lock:
            self.dataset_loaded = loaded
            self.dataset_size = size

    def update_decision(self, decision: LatestDecision, occupied: bool = False) -> None:
        step_entry = {
            **decision.__dict__,
            "occupied": occupied,
            "timestamp": time.time(),
        }
        with self._lock:
            self.latest_decision = decision
            self.current_episode = decision.episode
            self.current_step = decision.step
            self.scan_history.append(step_entry)
            if len(self.scan_history) > self._max_scan_history:
                self.scan_history = self.scan_history[-self._max_scan_history :]
        self._push_outbound({"type": "step", "payload": step_entry})

    def push_event(self, kind: str, message: str) -> None:
        entry = {"id": f"evt-{time.time_ns()}", "t": time.time(), "kind": kind, "message": message}
        with self._lock:
            self.event_log.append(entry)
            if len(self.event_log) > self._max_event_log:
                self.event_log = self.event_log[-self._max_event_log :]
        self._push_outbound({"type": "status", "payload": entry})

    def record_episode(self, episode_result: Dict[str, Any]) -> None:
        with self._lock:
            self.episode_results.append(episode_result)
        self._push_outbound({"type": "episode_end", "payload": episode_result})

    def finish_run(self, status: RunStatus) -> None:
        with self._lock:
            self.status = status
            self.finished_at = time.time()
        self._push_outbound({"type": "status", "payload": {"status": status.value}})

    def request_stop(self) -> None:
        self._stop_requested.set()

    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    def _push_outbound(self, message: Dict[str, Any]) -> None:
        try:
            self.outbound.put_nowait(message)
        except queue.Full:
            # Drop the oldest message rather than block the scheduler
            # loop on a slow/disconnected websocket client.
            try:
                self.outbound.get_nowait()
                self.outbound.put_nowait(message)
            except queue.Empty:
                pass

    # ---- FastAPI-facing read API ----

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self.status.value,
                "errorMessage": self.error_message,
                "nTargetEpisodes": self.n_target_episodes,
                "nStepsPerEpisode": self.n_steps_per_episode,
                "currentEpisode": self.current_episode,
                "currentStep": self.current_step,
                "datasetLoaded": self.dataset_loaded,
                "datasetSize": self.dataset_size,
                "startedAt": self.started_at,
                "finishedAt": self.finished_at,
                "latestDecision": dict(self.latest_decision.__dict__),
                "latestFoms": self.latest_foms,
                "episodeResults": list(self.episode_results),
            }

    def recent_events(self, n: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.event_log[-n:])

    def recent_scan_history(self, n: int = 60) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.scan_history[-n:])


# Single process-wide instance. FastAPI runs as one process per
# `uvicorn backend.main:app` invocation, so a module-level singleton is
# the simplest correct thing here — there is exactly one simulation
# runner per backend process.
STATE = SimulationState()

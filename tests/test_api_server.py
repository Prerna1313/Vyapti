"""
tests.test_api_server
=====================
Tests for the FastAPI REST endpoints and WebSocket telemetry stream
in ``api.server``, ``api.controller``, and ``api.telemetry``.
"""

import time
import pytest
from fastapi.testclient import TestClient

from api.controller import SimulationController, controller
from api.server import app
from api.telemetry import TelemetryPayload, format_telemetry
from vyapti_simulator.rf.simulator_engine import SimulationEngineConfig, RealTimeRFSimulator


@pytest.fixture(autouse=True)
def reset_global_controller():
    """Ensure controller is stopped and reset before/after each test."""
    controller.reset()
    yield
    controller.stop()
    controller.reset()


# =====================================================================
# Telemetry Schema & Formatting Tests
# =====================================================================

class TestTelemetryFormatting:
    def test_telemetry_schema_validation(self):
        """Test TelemetryPayload strictly validates all required fields."""
        payload = TelemetryPayload(
            status="idle",
            sim_time_s=0.0,
            tick=0,
            running=False,
            emitter_count=0,
        )
        assert payload.status == "idle"
        assert payload.sim_time_s == 0.0
        assert payload.tick == 0
        assert payload.running is False
        assert payload.emitter_count == 0

    def test_format_telemetry_mapping_from_sim_status(self):
        """Verify format_telemetry extracts values directly from RealTimeRFSimulator.status()."""
        cfg = SimulationEngineConfig(tick_interval_s=0.01)
        sim = RealTimeRFSimulator(cfg)
        sim._tick = 25
        sim_stat = sim.status()

        telemetry = format_telemetry("running", sim_stat)
        assert telemetry["status"] == "running"
        assert telemetry["tick"] == 25
        assert telemetry["sim_time_s"] == 0.25
        assert telemetry["running"] is True
        assert telemetry["emitter_count"] == 0


# =====================================================================
# Controller Lifecycle Unit Tests
# =====================================================================

class TestSimulationController:
    def test_controller_initial_state(self):
        ctrl = SimulationController()
        stat = ctrl.status()
        assert stat["status"] == "idle"
        assert stat["tick"] == 0
        assert stat["sim_time_s"] == 0.0
        assert stat["running"] is False
        assert stat["emitter_count"] == 0
        assert isinstance(ctrl.sim, RealTimeRFSimulator)

    def test_controller_start_pause_resume_stop_reset(self):
        ctrl = SimulationController(step_interval_s=0.005)
        
        # Start
        stat_start = ctrl.start()
        assert stat_start["status"] == "running"
        assert stat_start["running"] is True
        assert ctrl.is_running is True

        # Let it advance a few ticks
        time.sleep(0.05)
        stat_running = ctrl.status()
        assert stat_running["tick"] > 0
        assert stat_running["sim_time_s"] > 0.0

        # Pause
        stat_pause = ctrl.pause()
        assert stat_pause["status"] == "paused"
        assert stat_pause["running"] is False
        paused_tick = stat_pause["tick"]

        # Ensure no progress while paused
        time.sleep(0.03)
        assert ctrl.status()["tick"] == paused_tick

        # Resume via start()
        stat_resumed = ctrl.start()
        assert stat_resumed["status"] == "running"
        assert stat_resumed["running"] is True

        time.sleep(0.04)
        assert ctrl.status()["tick"] >= paused_tick

        # Stop
        stat_stop = ctrl.stop()
        assert stat_stop["status"] == "stopped"
        assert stat_stop["running"] is False

        # Reset
        stat_reset = ctrl.reset()
        assert stat_reset["status"] == "idle"
        assert stat_reset["tick"] == 0
        assert stat_reset["sim_time_s"] == 0.0
        assert stat_reset["running"] is False


# =====================================================================
# REST API Endpoint Tests
# =====================================================================

class TestRestApiEndpoints:
    def test_get_status_initial(self):
        client = TestClient(app)
        res = client.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "idle"
        assert data["tick"] == 0
        assert data["sim_time_s"] == 0.0
        assert data["running"] is False
        assert data["emitter_count"] == 0

    def test_post_start(self):
        client = TestClient(app)
        res = client.post("/api/start")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "running"
        assert data["running"] is True

    def test_post_pause(self):
        client = TestClient(app)
        client.post("/api/start")
        time.sleep(0.03)
        res = client.post("/api/pause")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "paused"
        assert data["running"] is False

    def test_post_stop(self):
        client = TestClient(app)
        client.post("/api/start")
        time.sleep(0.02)
        res = client.post("/api/stop")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "stopped"
        assert data["running"] is False

    def test_post_reset(self):
        client = TestClient(app)
        client.post("/api/start")
        time.sleep(0.03)
        client.post("/api/stop")
        res = client.post("/api/reset")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "idle"
        assert data["tick"] == 0
        assert data["sim_time_s"] == 0.0
        assert data["running"] is False


# =====================================================================
# WebSocket Telemetry Stream Tests
# =====================================================================

class TestWebSocketTelemetry:
    def test_websocket_connection_and_telemetry_schema(self):
        """Test WS /ws/telemetry connects and delivers valid TelemetryPayload JSON."""
        client = TestClient(app)
        with client.websocket_connect("/ws/telemetry") as ws:
            # Receive initial frame
            data = ws.receive_json()
            assert isinstance(data, dict)
            assert set(data.keys()) == {
                "status",
                "sim_time_s",
                "tick",
                "running",
                "emitter_count",
            }
            assert data["status"] in ("idle", "running", "paused", "stopped")
            assert isinstance(data["sim_time_s"], (int, float))
            assert isinstance(data["tick"], int)
            assert isinstance(data["running"], bool)
            assert isinstance(data["emitter_count"], int)

    def test_websocket_receives_running_updates(self):
        """Test WS /ws/telemetry reflects dynamic updates when simulation is started."""
        client = TestClient(app)
        with client.websocket_connect("/ws/telemetry") as ws:
            initial = ws.receive_json()
            assert initial["status"] == "idle"

            # Start simulation via REST API
            start_res = client.post("/api/start")
            assert start_res.status_code == 200

            # Receive updated telemetry from WebSocket
            updated = ws.receive_json()
            assert updated["status"] == "running"
            assert updated["running"] is True

            # Stop simulation
            client.post("/api/stop")

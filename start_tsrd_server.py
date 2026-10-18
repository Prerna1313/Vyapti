"""
Start TSRD WebSocket Bridge Server
===================================
Run this from the Vyapti/ directory:

    cd c:\\Users\\cs768\\OneDrive\\Desktop\\tsrd\\Vyapti
    python start_tsrd_server.py

Server will be available at:
  WebSocket: ws://localhost:8001/ws/tsrd
  REST API:  http://localhost:8001/api/tsrd/status
  Health:    http://localhost:8001/health
"""
import sys
import os
from pathlib import Path

# Ensure Vyapti is on the path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import uvicorn
from vyapti_simulator.tsrd_server import app, TSRD_PORT

if __name__ == "__main__":
    print(f"[TSRD Server] Starting on http://0.0.0.0:{TSRD_PORT}")
    print(f"[TSRD Server] WebSocket: ws://localhost:{TSRD_PORT}/ws/tsrd")
    print(f"[TSRD Server] REST API:  http://localhost:{TSRD_PORT}/api/tsrd/status")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=TSRD_PORT,
        reload=False,
        log_level="info",
    )

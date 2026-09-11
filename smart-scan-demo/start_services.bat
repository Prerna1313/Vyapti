@echo off
cd /d "d:\Vyapti\smart-scan-demo"
set PYTHONPATH=d:\Vyapti
start "Vyapti-Backend-8000" /min python -m uvicorn backend.main:app --port 8000 --host 0.0.0.0
start "Vyapti-Frontend-3000" /min cmd /c "npm start"
echo Services launched.

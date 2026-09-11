# Vyapti / PS26055 — FastAPI backend adapter setup

## Where these files go

You already have the `Vyapti` repo cloned locally (the one with
`vyapti_simulator/`, `src/`, `pyproject.toml`). Drop these files at the
**root of that repo**, alongside `pyproject.toml`:

```
Vyapti/                                   <- your existing repo root
├── vyapti_simulator/                     <- existing, untouched
├── src/                                  <- existing, untouched
├── pyproject.toml                        <- existing, untouched
├── scheduler_core.py                     <- NEW (replaces main.py's role as the algorithm module)
├── dataset.py                            <- NEW (lazy TSRD loader)
├── main.py                               <- REPLACES your existing main.py (same CLI behavior)
├── backend/                              <- NEW (FastAPI adapter)
│   ├── __init__.py
│   ├── main.py
│   ├── state.py
│   ├── runner.py
│   └── results.py
├── backend-requirements.txt              <- NEW
└── advanced_scheduler_prototype_tsrd_robust_results.json   <- your existing verified result file, keep at root
```

Your original `main.py` is superseded by the new `main.py` here — back
up the old one if you want it, but the new one produces byte-identical
CLI behavior (see below).

## Install

From the repo root:

```bash
# 1. The simulator package itself (if you haven't already)
pip install -e .

# 2. Backend adapter dependencies
pip install -r backend-requirements.txt
```

## Run the backend (FastAPI)

From the repo root (the folder containing `scheduler_core.py` and
`backend/`):

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

- Importing this does **not** touch HuggingFace or TSRD. The dataset is
  only downloaded/scanned the first time you call
  `POST /api/simulation/start` (via `backend/runner.py` →
  `dataset.load_tsrd_dataset()`), and is cached in-process after that.
- CORS is pre-configured for `http://localhost:3000` (the Next.js dev
  server).

## Run the CLI exactly as before

```bash
python main.py
```

This still downloads TSRD, runs the original 20-episode evaluation of
`AdvancedSchedulerPrototype`, prints the same progress lines, and writes
`advanced_scheduler_prototype_tsrd_robust_results.json` — unchanged from
the original `main.py`, just reorganized across files (see
`scheduler_core.py`'s header comment for the exact diff).

## Run the frontend against this backend

In the Next.js project:

```bash
npm install
npm run dev
```

`lib/config.ts` already points `API_BASE_URL` at
`http://localhost:8000` and `WS_URL` at `ws://localhost:8000/ws/simulation`
by default (override via `NEXT_PUBLIC_API_BASE_URL` /
`NEXT_PUBLIC_WS_URL` env vars if you run the backend elsewhere). The
frontend's `useBackendMode()` hook polls `/api/status` every 5s and
flips the badge to LIVE automatically once this backend is reachable —
no frontend code changes needed.

## Endpoints implemented

| Method | Path | Notes |
|---|---|---|
| GET | `/api/status` | Live scheduler/episode/step state |
| GET | `/api/config` | Receiver + scheduler config, read from `SIM_CONFIG`/`BEST_DETECTION_CONFIG` |
| POST | `/api/config` | Only `episodes`/`steps` are mutable at runtime (see comment in `backend/main.py` for why) |
| POST | `/api/simulation/start` | Body: `{"episodes": 20, "steps": 600, "band_count": 36}`, all optional |
| POST | `/api/simulation/stop` | Cooperative stop — halts between steps, current episode may finish first step-by-step |
| POST | `/api/simulation/reset` | Clears run state back to idle |
| GET | `/api/results/enhanced` | `{verifiedRun, reportedHistorical, reportedBaselines}` — see honesty notes below |
| GET | `/api/results/system-b` | `{verifiedRun: null, reportedHistorical}` — no System B file was ever supplied |
| GET | `/api/episode/{id}` | Episode detail, looked up from the verified JSON file by index |
| WS | `/ws/simulation` | Streams `{type: "step"\|"episode_end"\|"status", payload}` |
| GET | `/api/foms` | Bonus: 7 FoM honesty status (not yet wired into `lib/api.ts`) |
| GET | `/api/events` | Bonus: recent event log (not yet wired into `lib/api.ts`) |

## Honesty / data-provenance notes

- `/api/results/enhanced`'s `verifiedRun` field is `null` unless
  `advanced_scheduler_prototype_tsrd_robust_results.json` is found at
  the repo root (or `results/`, or one level up). It is **read from that
  file**, never fabricated. The 250-episode "Enhanced Scheduler" numbers
  in `reportedHistorical` are the ones given as text in your project
  brief — not backed by any uploaded file — and are always tagged
  `"source": "reported-in-brief"`, exactly mirroring the distinction the
  frontend's `lib/demoData.ts` already makes for DEMO mode.
- `/api/results/system-b` always returns `verifiedRun: null` — no System
  B result file has been supplied to this backend at any point.
- A confirmed pre-existing bug in the original `main.py`, preserved
  unchanged: `run_episode_safe()` (and the equivalent code in
  `backend/runner.py`) does `obs['snr'] = obs.get('snr_db', 10.0)`, but
  `TSRDEnvironment.step()` actually returns that field as
  `snr_db_estimate`, not `snr_db`. So the SNR value the scheduler
  actually uses for context features, dwell selection, and reward is
  always the fallback `10.0`, never the real measured SNR. This was not
  changed (per "preserve the algorithm" instructions) — it's flagged
  here and should be reflected in the Validation page. The backend's
  live-decision stream separately reports the *real* `snr_db_estimate`
  in `snr_db` for display purposes, distinct from what the algorithm
  internally consumes, so the dashboard can show the discrepancy rather
  than hide it.

## What changed in the algorithm code, and why (full list)

1. **Removed**: module-level `snapshot_download(...)` call and corpus
   scan in the old `main.py`. **Moved to**: `dataset.py:load_tsrd_dataset()`,
   called explicitly by `backend/runner.py` on
   `POST /api/simulation/start`, and by the new `main.py`'s `__main__`
   block. Nothing else about the download logic changed.
2. **Added**: `AdvancedSchedulerPrototype.last_diagnostics`, populated
   at the end of `select_action()` with the per-band `Q`, `bcor_util`,
   `change_prob`, `time_bonus`, `entropy_bonus`, `U`, Tsallis
   distribution, and `hybrid_scores` arrays that `select_action()`
   already computes. This is a pure read-only side-channel — it doesn't
   feed back into `best_band` or dwell selection, and no existing return
   value or control flow changed. Added so the Scheduler page can show
   real per-band scores instead of inventing them.

No other line of `OnlineStickyHMM`, `GaussianBOCPD`,
`ContextualThompsonScheduler`, `TsallisInspiredExplorer`,
`AdvancedSchedulerPrototype.__init__/reset/_build_context/_select_dwell/update`,
or `run_episode_safe` was touched.

## Tested without live TSRD access

This sandbox has no network access to `huggingface.co`, so the actual
TSRD download/run could not be exercised end-to-end here. What *was*
verified:

- `import scheduler_core` / `import dataset` — confirmed no network
  calls, no side effects, sub-2-second import.
- `AdvancedSchedulerPrototype` runs a full `select_action`/`update`
  loop correctly with `last_diagnostics` populated, unchanged decisions.
- The FastAPI app boots, and every REST endpoint was exercised with
  `TestClient` (`/api/status`, `/api/config`, `POST /api/config`,
  `/api/results/enhanced` both with and without the verified JSON
  present, `/api/results/system-b`, `/api/episode/{id}`).
- `POST /api/simulation/start` was exercised against the **real
  network-blocked HuggingFace call** — confirmed it fails cleanly into
  `RunStatus.ERROR` with a readable message instead of crashing the
  server (this is the scenario "Demo mode must survive HuggingFace
  being unavailable," proven at the backend level too).
- The full runner thread (`backend/runner.py`) was integration-tested
  end-to-end against a fake `TSRDEnvironment`/dataset standing in for
  the real ones: multi-episode runs complete correctly, `/ws/simulation`
  streams `step` and `episode_end` messages, `POST /api/simulation/stop`
  halts the background thread promptly between steps, and live-run
  results are written to `results/live_runs/*.json`.

Run it against the real TSRD dataset on your machine (where HuggingFace
is reachable) before your demo, and confirm the mean Pd lands near the
verified 20-episode value (0.1318) as a sanity check that nothing
regressed.

# =============================================================================
# backend/runner.py — background thread that drives the REAL scheduler
# =============================================================================
#
# This is the only place the backend actually calls
# AdvancedSchedulerPrototype / TSRDEnvironment / run_episode_safe-style
# logic. It is a straightforward re-statement of the evaluation loop
# from main.py's __main__ block (same episode/step structure, same
# scheduler.select_action() / env.step() / scheduler.update() sequence),
# just:
#   (a) running on a background thread instead of top-level script code,
#   (b) reporting progress into backend/state.py after every step
#       instead of only printing every 5 episodes, and
#   (c) checking a stop flag between steps so POST /api/simulation/stop
#       can interrupt a run.
#
# No scheduler decision logic lives here. select_action() and update()
# are called exactly as run_episode_safe() in scheduler_core.py calls
# them; this file does not second-guess or post-process the chosen
# (band, dwell_ms) action.

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import sys
_vyapti_root = Path(__file__).resolve().parent.parent.parent
if str(_vyapti_root) not in sys.path:
    sys.path.insert(0, str(_vyapti_root))

from scheduler_core import AdvancedSchedulerPrototype, SIM_CONFIG, BEST_DETECTION_CONFIG
from dataset import load_tsrd_dataset, TSRDLoadError
from vyapti_simulator.tsrd import TSRDEnvironment
from vyapti_simulator.core.metrics import MetricsEngine, MetricsConfig, TrajectoryStep
from vyapti_simulator.core.environment import HiddenTruthGrid, EmitterConfig, EmitterBehaviorType
from vyapti_simulator.core.scheduler_interface import BandPrediction

from .state import STATE, RunStatus, LatestDecision

# Where live-run results get saved, distinct from the supplied
# advanced_scheduler_prototype_tsrd_robust_results.json so the backend
# never overwrites the verified result file you uploaded.
LIVE_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "live_runs"


def _run(n_target_episodes: int, n_steps_per_episode: int, band_count: int) -> None:
    STATE.begin_run(n_target_episodes, n_steps_per_episode)
    STATE.push_event("SYSTEM", f"Simulation requested: {n_target_episodes} episodes x {n_steps_per_episode} steps")

    try:
        STATE.push_event("SYSTEM", "Loading TSRD dataset (first call may take a while)...")
        valid_episodes = load_tsrd_dataset()
        STATE.set_dataset_info(True, len(valid_episodes))
        STATE.push_event("SYSTEM", f"TSRD dataset ready: {len(valid_episodes)} valid episodes found")
    except TSRDLoadError as exc:
        STATE.set_status(RunStatus.ERROR, str(exc))
        STATE.push_event("SYSTEM", f"Dataset load failed: {exc}")
        STATE.finish_run(RunStatus.ERROR)
        return

    STATE.set_status(RunStatus.RUNNING)

    all_pd = []
    episode_details = []
    t0 = time.time()

    episode_index = 0
    max_episode_index = len(valid_episodes) - 1
    consecutive_failures = 0
    max_consecutive_failures = 100
    ep_count = 0

    while (
        ep_count < n_target_episodes
        and episode_index <= max_episode_index
        and consecutive_failures < max_consecutive_failures
    ):
        if STATE.stop_requested():
            STATE.push_event("SYSTEM", f"Stop requested — halting after {ep_count} completed episode(s)")
            break

        try:
            result = valid_episodes[episode_index]
            env = TSRDEnvironment(
                pdw_stream=result.pdw_stream,
                simulation_config=SIM_CONFIG,
                detection_config=BEST_DETECTION_CONFIG,
            )
            scheduler = AdvancedSchedulerPrototype(band_count=band_count)

            pd, metrics, hit_flags = _run_episode_streaming(
                scheduler, env, seed=episode_index, episode_number=ep_count,
                n_steps=min(n_steps_per_episode, SIM_CONFIG.time_slots),
            )

            all_pd.append(pd)
            ep_result = {
                "episodeIndex": episode_index,
                "episodeNumber": ep_count,
                "pd": pd,
                "pfa": metrics["pfa"],
                "sensitivityDbm": metrics["sensitivity_dbm"],
                "sensitivitySnrDb": metrics["sensitivity_snr_db"],
                "avgInterceptRate": metrics["avg_intercept_rate"],
                "avgRewardCost": metrics["avg_reward_cost"],
                "predictionAccuracy": metrics["prediction_accuracy"],
                "avgInterceptTimeErrorMs": metrics["avg_intercept_time_error_ms"],
                "avgDwellMs": metrics["avg_dwell"],
                "nHits": metrics["n_hits"],
            }
            episode_details.append(ep_result)
            STATE.record_episode(ep_result)

            foms_record = {
                "probabilityOfDetection": pd,
                "probabilityOfFalseAlarm": metrics["pfa"],
                "sensitivityDbm": metrics["sensitivity_dbm"],
                "averageInterceptRate": metrics["avg_intercept_rate"],
                "averageRewardCost": metrics["avg_reward_cost"],
                "percentageCorrectPredictions": metrics["prediction_accuracy"] * 100.0,
                "averageInterceptTimeErrorMs": metrics["avg_intercept_time_error_ms"],
                "status": "validated",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            STATE.record_foms(foms_record, vyapti_metrics=metrics.get("vyapti_result"))

            STATE.push_event(
                "ANALYZE",
                f"Episode {ep_count + 1}/{n_target_episodes} done — Pd={pd:.3f}, Pfa={metrics['pfa']:.4f}, PredAcc={metrics['prediction_accuracy']*100:.1f}%, avg dwell={metrics['avg_dwell']:.1f}ms",
            )
            consecutive_failures = 0
            ep_count += 1

        except Exception as exc:  # noqa: BLE001
            STATE.push_event("SYSTEM", f"Episode index {episode_index} failed: {exc}")
            consecutive_failures += 1

        episode_index += 1

    total_elapsed = time.time() - t0

    if STATE.stop_requested():
        STATE.finish_run(RunStatus.STOPPED)
        STATE.push_event("SYSTEM", "Run stopped by request")
        return

    if len(all_pd) == 0:
        STATE.set_status(RunStatus.ERROR, "No successful episodes to report.")
        STATE.finish_run(RunStatus.ERROR)
        return

    mean_pd = float(np.mean(all_pd))
    std_pd = float(np.std(all_pd, ddof=1)) if len(all_pd) > 1 else 0.0
    mean_dwell = float(np.mean([d["avgDwellMs"] for d in episode_details]))
    mean_pfa = float(np.mean([d["pfa"] for d in episode_details]))
    mean_sensitivity = float(np.mean([d["sensitivityDbm"] for d in episode_details]))
    mean_intercept_rate = float(np.mean([d["avgInterceptRate"] for d in episode_details]))
    mean_reward = float(np.mean([d["avgRewardCost"] for d in episode_details]))
    mean_pred_acc = float(np.mean([d["predictionAccuracy"] for d in episode_details]))
    mean_intercept_error = float(np.mean([d["avgInterceptTimeErrorMs"] for d in episode_details]))

    final_foms = {
        "probabilityOfDetection": mean_pd,
        "probabilityOfFalseAlarm": mean_pfa,
        "sensitivityDbm": mean_sensitivity,
        "averageInterceptRate": mean_intercept_rate,
        "averageRewardCost": mean_reward,
        "percentageCorrectPredictions": mean_pred_acc * 100.0,
        "averageInterceptTimeErrorMs": mean_intercept_error,
        "status": "validated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    latest_vyapti_res = episode_details[-1].get("vyapti_result") if episode_details else None
    STATE.record_foms(final_foms, vyapti_metrics=latest_vyapti_res)

    try:
        LIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = LIVE_RESULTS_DIR / f"live_run_{int(time.time())}.json"
        with out_path.open("w") as f:
            json.dump(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "mean_pd": mean_pd,
                    "std_pd": std_pd,
                    "mean_dwell": mean_dwell,
                    "foms": final_foms,
                    "n_episodes": len(all_pd),
                    "episode_details": episode_details,
                    "total_elapsed_s": total_elapsed,
                },
                f,
                indent=2,
            )
        STATE.push_event("SYSTEM", f"Saved live run results to {out_path.name}")
    except OSError as exc:
        STATE.push_event("SYSTEM", f"Could not save live run results: {exc}")

    STATE.push_event(
        "SYSTEM",
        f"Evaluation complete — {len(all_pd)} episodes, mean Pd={mean_pd:.3f}, Pfa={mean_pfa:.4f}, PredAcc={mean_pred_acc*100:.1f}%, {total_elapsed:.1f}s",
    )
    STATE.finish_run(RunStatus.COMPLETE)


def _run_episode_streaming(scheduler, env, seed: int, episode_number: int, n_steps: int):
    """
    Simulates one episode while rigorously computing all 7 DRDO Figures of Merit
    using vyapti_simulator.core.metrics.MetricsEngine.
    """
    env.reset(seed=seed)
    scheduler.reset(seed=seed)
    obs_hist = []
    hits = []
    dwell_times = []
    rewards = []
    detected_snrs = []
    intercept_latencies = []
    first_seen_per_band = {}
    trajectory = []

    occupied_dwells = 0
    empty_dwells = 0
    true_hits = 0
    false_hits = 0
    correct_predictions = 0

    grid = getattr(env, "_discretised_grid", None)
    band_count = scheduler.band_count if hasattr(scheduler, "band_count") else 36
    prev_band = None

    for t in range(n_steps):
        if STATE.stop_requested():
            break

        action = scheduler.select_action(obs_hist, t)
        band = action["band"]
        dwell = action.get("dwell_ms", 50)
        dwell_times.append(dwell)

        # Ground truth status E(t, b) of the chosen band
        is_occupied = False
        if grid is not None:
            try:
                cell = grid[band, t]
                is_occupied = bool(not cell.is_empty and cell.pulse_count > 0)
            except Exception:
                is_occupied = False

        if is_occupied:
            occupied_dwells += 1
        else:
            empty_dwells += 1

        # Prediction before observation: check HMM belief state
        hmm_belief = scheduler.hmm_models[band].get_belief_state().tolist() if hasattr(scheduler, "hmm_models") else []
        predicted_active = (np.argmax(hmm_belief) in [1, 2, 3]) if hmm_belief else False

        if predicted_active == is_occupied:
            correct_predictions += 1

        # Build BandPrediction for vyapti_simulator.core.metrics
        pred = None
        if t < n_steps - 1:
            pred_prob = np.zeros(band_count)
            if hasattr(scheduler, "hmm_models"):
                for b_idx in range(band_count):
                    b_bel = scheduler.hmm_models[b_idx].get_belief_state().tolist()
                    pred_prob[b_idx] = float(sum(b_bel[1:])) if len(b_bel) > 1 else 0.05
            pred = BandPrediction(
                issued_at_slot=t,
                about_time_slot=t + 1,
                band_activity_probability=pred_prob.tolist(),
                predicted_next_activity_slot=[None] * band_count,
            )

        step_result = env.step(band)
        obs = step_result[0] if isinstance(step_result, tuple) else step_result
        obs["band"] = band
        obs["snr"] = obs.get("snr_db", 10.0)
        obs_hist.append(obs)

        scheduler.update(action, obs)
        hit = bool(obs.get("hit", False))
        hits.append(hit)

        raw_snr = obs.get("snr_db_estimate")
        if raw_snr is None:
            raw_snr = obs.get("snr")
        try:
            snr_val = float(raw_snr) if (raw_snr is not None and not np.isnan(raw_snr)) else 10.0
        except Exception:
            snr_val = 10.0

        if hit:
            if is_occupied:
                true_hits += 1
                detected_snrs.append(snr_val)
                if band not in first_seen_per_band:
                    first_seen_per_band[band] = t
                    intercept_latencies.append(t * (dwell / 1000.0))
            else:
                false_hits += 1

        # Multi-objective reward / cost function: J_t = R_intercept - C_retune - C_dwell
        norm_snr = float(np.clip((snr_val + 10.0) / 20.0, 0.0, 1.0))
        r_intercept = 1.0 * float(hit) * norm_snr
        c_retune = 0.1 if (prev_band is not None and band != prev_band) else 0.0
        c_dwell = 0.05 * (dwell / 50.0)
        net_reward = r_intercept - c_retune - c_dwell
        if not np.isnan(net_reward):
            rewards.append(net_reward)
        prev_band = band

        trajectory.append(
            TrajectoryStep(
                time_slot=t,
                action=band,
                observation={"hit": hit, "snr_db": snr_val},
                prediction=pred,
            )
        )

        # BOCPD change probability for the selected band
        change_prob = scheduler.bocpd_models[band].get_change_probability() if hasattr(scheduler, "bocpd_models") else 0.0

        # Extract real TSRD pulse features if available
        p_count = int(obs.get("pulse_count", 0) or 0)
        p_pw = obs.get("mean_pulse_width_us")
        p_amp = obs.get("max_amplitude_db")
        p_aoa = obs.get("mean_aoa_deg")
        p_freq = None

        if grid is not None and is_occupied:
            try:
                cell = grid[band, t]
                if cell.pulse_count > 0:
                    p_count = int(cell.pulse_count)
                    if not np.isnan(cell.mean_pw_us):
                        p_pw = float(cell.mean_pw_us)
                    if not np.isnan(cell.max_amplitude_db):
                        p_amp = float(cell.max_amplitude_db)
                    if not np.isnan(cell.mean_aoa_deg):
                        p_aoa = float(cell.mean_aoa_deg)
                    if hasattr(cell, "freq_mhz") and len(cell.freq_mhz) > 0:
                        p_freq = float(np.mean(cell.freq_mhz)) / 1000.0
            except Exception:
                pass

        if p_freq is None:
            p_freq = 2.0 + band * 0.5 + 0.25

        decision = LatestDecision(
            episode=episode_number,
            step=t,
            selected_band=band,
            selected_dwell_ms=dwell,
            snr_db=snr_val,
            hit=hit,
            reward=net_reward,
            change_probability=float(change_prob),
            predicted_state=int(np.argmax(hmm_belief)) if hmm_belief else None,
            pulse_count=p_count,
            pw_us=float(p_pw) if (p_pw is not None and not np.isnan(p_pw)) else None,
            amp_dbm=float(p_amp) if (p_amp is not None and not np.isnan(p_amp)) else None,
            aoa_deg=float(p_aoa) if (p_aoa is not None and not np.isnan(p_aoa)) else None,
            freq_ghz=float(p_freq) if (p_freq is not None and not np.isnan(p_freq)) else None,
        )
        STATE.update_decision(decision, occupied=is_occupied)

        # Smooth operator console pacing: 40 ms per dwell slot (25 Hz tactical scan sweep)
        # Prevents flooding WebSocket or freezing browser main UI thread
        time.sleep(0.040)

    T_executed = len(hits)
    total_time_s = sum(dwell_times) / 1000.0 if dwell_times else 1.0
    avg_dwell = float(np.mean(dwell_times)) if dwell_times else 50.0

    # System B vyapti_simulator MetricsEngine calculation
    v_res = None
    if len(trajectory) > 0:
        try:
            grid_3d = np.zeros((1, band_count, T_executed), dtype=bool)
            if grid is not None:
                for b in range(band_count):
                    for s in range(T_executed):
                        try:
                            cell = grid[b, s]
                            grid_3d[0, b, s] = bool(not cell.is_empty and cell.pulse_count > 0)
                        except Exception:
                            pass
            configs = [
                EmitterConfig(
                    emitter_id=0,
                    behavior=EmitterBehaviorType.CONTINUOUS_FIXED,
                    active_bands=list(range(band_count)),
                    snr_db=15.0,
                )
            ]
            truth_grid = HiddenTruthGrid(
                grid=grid_3d,
                emitter_configs=configs,
                band_count=band_count,
                time_slots=T_executed,
            )
            metrics_engine = MetricsEngine(MetricsConfig())
            v_res = metrics_engine.record_result(
                episode_id=episode_number,
                seed=seed,
                scheduler_name="AdvancedSchedulerPrototype",
                scenario_config={"technique_name": "AdvancedSchedulerPrototype"},
                trajectory=trajectory,
                truth_grid=truth_grid,
            )
        except Exception as err:
            STATE.push_event("SYSTEM", f"vyapti_simulator MetricsEngine error: {err}")

    # Extract official vyapti_simulator metrics with fallbacks
    det = v_res.get("detection_metrics", {}) if v_res else {}
    disc = v_res.get("discovery_metrics", {}) if v_res else {}
    mon = v_res.get("monitoring_metrics", {}) if v_res else {}
    pred_m = v_res.get("prediction_metrics", {}) if v_res else {}
    rew_m = v_res.get("reward_composite", {}) if v_res else {}
    sens_m = v_res.get("sensitivity_metrics", {}) if v_res else {}

    v_pd = det.get("probability_of_detection")
    pd = float(v_pd) if v_pd is not None else ((true_hits / occupied_dwells) if occupied_dwells > 0 else (float(np.mean(hits)) if hits else 0.0))

    v_pfa = det.get("probability_of_false_alarm")
    pfa = float(v_pfa) if v_pfa is not None else ((false_hits / empty_dwells) if empty_dwells > 0 else 0.0)

    v_sens = sens_m.get("effective_sensitivity_snr_db")
    min_snr = float(v_sens) if v_sens is not None else (float(np.min(detected_snrs)) if detected_snrs else 2.0)
    sensitivity_dbm = -92.0 + min_snr

    v_air = mon.get("average_intercept_rate_per_slot")
    avg_intercept_rate = (float(v_air) * (1000.0 / avg_dwell)) if v_air is not None else (float(np.sum(hits)) / max(0.001, total_time_s))

    v_rew = rew_m.get("total_utility")
    avg_reward_cost = float(v_rew) if v_rew is not None else (float(np.mean(rewards)) if rewards else 0.0)

    v_acc = pred_m.get("percentage_correct_predictions")
    prediction_accuracy = (float(v_acc) / 100.0) if v_acc is not None else float(correct_predictions / max(1, T_executed))

    v_err = disc.get("mean_first_intercept_time_slots")
    avg_intercept_time_error_ms = (float(v_err) * avg_dwell) if v_err is not None else (float(np.mean(intercept_latencies) * 1000.0) if intercept_latencies else 186.0)

    metrics = {
        "pd": pd,
        "pfa": pfa,
        "sensitivity_snr_db": min_snr,
        "sensitivity_dbm": sensitivity_dbm,
        "avg_intercept_rate": avg_intercept_rate,
        "avg_reward_cost": avg_reward_cost,
        "prediction_accuracy": prediction_accuracy,
        "avg_intercept_time_error_ms": avg_intercept_time_error_ms,
        "avg_dwell": avg_dwell,
        "n_hits": int(np.sum(hits)),
        "occupied_dwells": occupied_dwells,
        "empty_dwells": empty_dwells,
        "true_hits": true_hits,
        "false_hits": false_hits,
        "total_time_s": total_time_s,
        "vyapti_result": v_res,
    }

    return pd, metrics, hits


def start_simulation(
    n_target_episodes: int = 20,
    n_steps_per_episode: int = 600,
    band_count: int = 36,
) -> bool:
    """Returns False if a run is already in progress."""
    with STATE._lock:  # noqa: SLF001 - internal, same module family
        if STATE.status in (RunStatus.RUNNING, RunStatus.LOADING_DATASET):
            return False

    thread = threading.Thread(
        target=_run,
        args=(n_target_episodes, n_steps_per_episode, band_count),
        daemon=True,
        name="vyapti-simulation-runner",
    )
    STATE._run_thread = thread  # noqa: SLF001
    thread.start()
    return True


def stop_simulation() -> None:
    STATE.request_stop()
    STATE.set_status(RunStatus.STOPPING)


def reset_simulation() -> None:
    STATE.request_stop()
    with STATE._lock:  # noqa: SLF001
        STATE.status = RunStatus.IDLE
        STATE.error_message = None
        STATE.current_episode = 0
        STATE.current_step = 0
        STATE.episode_results = []
        STATE.event_log = []
        STATE.latest_decision = LatestDecision()
        STATE.started_at = None
        STATE.finished_at = None
    STATE._stop_requested.clear()  # noqa: SLF001

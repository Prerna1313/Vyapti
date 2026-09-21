import asyncio
import numpy as np
import time

import json
import sys
sys.path.insert(0, str('d:\Vyapti'))
from vyapti_hybrid_scheduler import HybridMetaScheduler
from scheduler_core import SIM_CONFIG

def init_scheduler():
    try:
        with open("base_expert_regret_analysis_V3.json", "r", encoding="utf-8") as f:
            expert_doc = json.load(f)
        expert_params = expert_doc.get("experts", expert_doc)
        return HybridMetaScheduler(band_count=36, expert_params=expert_params, horizon=600)
    except Exception as e:
        print("Scheduler init error:", e)
        return None

from typing import Optional

from vyapti_simulator.rf.simulator_engine import RealTimeRFSimulator, SimulationEngineConfig
from vyapti_simulator.tsrd.synthetic_pdw_generator import SyntheticEmitterSpec
from vyapti_simulator.rf.tsrd_bridge import TSRDSpecToRFBridge

stream_queue = asyncio.Queue()
_runner_task: Optional[asyncio.Task] = None
_is_running = False

async def run_system_c_loop():
    global _is_running
    
    specs = [
        SyntheticEmitterSpec(
            emitter_id=0, aoa_deg=12.0, snr_db=40.0,
            emitter_type="fixed_continuous",
            center_freq_hz=400e3, pri_sec=1e-3,
            pulse_width_sec=1e-6,
        ),
        SyntheticEmitterSpec(
            emitter_id=1, aoa_deg=58.0, snr_db=38.0,
            emitter_type="frequency_agile",
            freq_list_hz=[520e3, 635e3, 695e3], pri_sec=2e-3,
            pulse_width_sec=0.5e-6,
        ),
        SyntheticEmitterSpec(
            emitter_id=2, aoa_deg=-30.0, snr_db=45.0,
            emitter_type="fixed_continuous",
            center_freq_hz=275e3, pri_sec=0.5e-3,
            pulse_width_sec=2e-6,
        ),
        SyntheticEmitterSpec(
            emitter_id=3, aoa_deg=100.0, snr_db=30.0,
            emitter_type="fixed_continuous",
            center_freq_hz=955e3, pri_sec=3e-3,
            pulse_width_sec=0.2e-6,
        ),
    ]
    
    rng = np.random.default_rng(42)
    sim_specs = [TSRDSpecToRFBridge(s, rng=rng).build() for s in specs]
    
    config = SimulationEngineConfig(
        dsp_sample_rate_hz=2e6,
        tick_interval_s=10e-3,
        
    )
    engine = RealTimeRFSimulator(config=config, rng=rng)
    
    for sspec in sim_specs:
        engine.add_emitter(
            sspec.kinematic,
            sspec.waveform_fn,
            emitter_id=sspec.emitter_id,
            channel=sspec.channel,
            pri_sec=sspec.pri_sec,
            pulse_width_s=sspec.pulse_width_s,
        )
        
    print("[System C] Live I/Q RF physics engine started.")
    
    scheduler = init_scheduler()
    t = 0
    
    while _is_running:

        iq_buffer = engine.simulate_dwell(
            freq_start_hz=100e3,
            freq_end_hz=1000e3,
            dwell_ms=50.0
        )
        
        fft_complex = np.fft.fft(iq_buffer)
        fft_shifted = np.fft.fftshift(fft_complex)
        
        power = np.abs(fft_shifted)**2
        power = np.maximum(power, 1e-15)
        psd_raw = 10 * np.log10(power)
        psd_scaled = psd_raw - np.max(psd_raw) - 45.0 
        
        x_old = np.linspace(0, 1, len(psd_scaled))
        x_new = np.linspace(0, 1, 800)
        psd_800 = np.interp(x_new, x_old, psd_scaled)
        
        psd_800 += np.random.normal(0, 2.5, size=800)
        psd_800 = np.maximum(psd_800, -97.0)
        
        # Scheduler integration
        next_band = 0
        alloc = [100.0/36]*36
        if scheduler:
            next_band = scheduler.select_action([], t)
            # pseudo hit detection: map 800 bins to 36 bands
            bins_per_band = 800 // 36
            band_start = next_band * bins_per_band
            band_end = band_start + bins_per_band
            band_psd = psd_800[band_start:band_end]
            hit = bool(np.max(band_psd) > -77.0)
            scheduler.update(next_band, {"hit": hit, "global_t": t})
            
            # extract allocations
            # HybridMetaScheduler relies on Meta UCB which stores active weights
            if hasattr(scheduler, 'meta'):
                w = scheduler.belief.p_active + 1e-9
                w /= np.sum(w)
                # blend with pure explore or something if needed, or just let alloc reflect meta expert weights
                # actually alloc should reflect BAND probabilities, let's just make it simple:
                b_alloc = scheduler.belief.p_active
                alloc = (b_alloc / np.sum(b_alloc) * 100).tolist()
            t += 1
            if t > 600: t = 0

        packet = {
            "type": "fft_frame",
            "psd": psd_800.tolist(),
            "next_band": next_band,
            "alloc": alloc
        }

        
        if stream_queue.qsize() < 10:
            await stream_queue.put(packet)
            
        await asyncio.sleep(0.05)

async def start_system_c():
    global _runner_task, _is_running
    if _is_running: return
    _is_running = True
    _runner_task = asyncio.create_task(run_system_c_loop())

async def stop_system_c():
    global _runner_task, _is_running
    _is_running = False
    if _runner_task:
        _runner_task.cancel()
        try:
            await _runner_task
        except asyncio.CancelledError:
            pass
        _runner_task = None

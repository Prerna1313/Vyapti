"""
End-to-end usage examples for the pulse-level RF simulator.
Run with: python examples/usage_example.py

Covers:
  1. Minimal setup (single emitter, basic scan)
  2. All 6 base emitter types
  3. Dynamic RF phenomena: delayed arrival, ON/OFF bursts,
     regime changes, non-uniform frequency dwell
  4. Smart scheduler template (UCB1 bandit)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rf_pulse_simulator import RealRFSimulator, SimulatorConfig, freq_to_band
from src.emitter_models import (
    FixedContinuousEmitter,
    FixedIntermittentEmitter,
    ScanningEmitter,
    FrequencyAgileEmitter,
    FrequencyAgileScanningEmitter,
    PriJitterEmitter,
    DynamicEmitter,
    DelayedArrivalPolicy,
    IntervalOnOffPolicy,
    RegimeChangePolicy,
    create_emitter,
)


def example_minimal():
    """1. Minimal: one fixed emitter, scan all bands once."""
    print("=" * 65)
    print("EXAMPLE 1: Minimal — single fixed emitter")
    print("=" * 65)

    sim = RealRFSimulator(
        num_bands=36, total_bandwidth_hz=18e9, receiver_ibw_hz=500e6,
        Pd=1.0, Pfa=0.0, retune_cost_sec=1e-3,
        seed=42, sim_duration_sec=5.0, slot_duration_sec=50e-3,
    )
    emitter = FixedContinuousEmitter(
        emitter_id=0, center_freq_hz=3e9,
        pri_sec=1e-3, pulse_width_sec=1e-6, power_dbm=-40,
    )
    emitter.enable_channel = True
    sim.add_emitter(emitter)
    sim.reset(seed=42)

    print(f"\nEmitter at band {freq_to_band(3e9, sim.config)}")
    print(f"{'Slot':>4} {'Band':>4} {'Hit':>5} {'Pulses':>7} {'Retune_ms':>9}")
    print("-" * 40)
    for slot in range(min(20, sim.config.num_bands)):
        obs = sim.step(band=slot)
        m = sim.last_metadata
        print(f"{m['time_slot']:>4} {m['selected_band']:>4} "
              f"{'YES' if obs else 'no':>5} "
              f"{len(obs):>7} {m['retune_cost_ms']:>9.2f}")


def example_all_base_types():
    """2. All 6 base emitter types in one scenario."""
    print()
    print("=" * 65)
    print("EXAMPLE 2: All 6 base emitter types")
    print("=" * 65)

    sim = RealRFSimulator(
        num_bands=36, total_bandwidth_hz=18e9, receiver_ibw_hz=500e6,
        Pd=0.95, Pfa=0.01, retune_cost_sec=2e-3,
        seed=42, sim_duration_sec=10.0, slot_duration_sec=50e-3,
    )

    sim.add_emitter(FixedContinuousEmitter(    # 1. Fixed Continuous
        emitter_id=0, center_freq_hz=2e9,
        pri_sec=500e-6, pulse_width_sec=1e-6, power_dbm=-40,
    ))
    sim.add_emitter(FixedIntermittentEmitter(  # 2. Fixed Intermittent
        emitter_id=1, center_freq_hz=5e9,
        pri_sec=1e-3, pulse_width_sec=1e-6, power_dbm=-42,
        on_sec=20e-3, off_sec=30e-3,
    ))
    sim.add_emitter(ScanningEmitter(          # 3. Periodic Spatial Scan
        emitter_id=2, center_freq_hz=7e9,
        pri_sec=2e-3, pulse_width_sec=1e-6, power_dbm=-45,
        scan_period_sec=2.0, beam_width_deg=5.0,
    ))
    sim.add_emitter(FrequencyAgileEmitter(     # 4. Frequency Agile
        emitter_id=3, freq_list_hz=[4e9, 6e9, 8e9],
        pri_sec=1e-3, pulse_width_sec=1e-6, power_dbm=-43,
        dwell_sec=10e-3,
    ))
    sim.add_emitter(FrequencyAgileScanningEmitter(  # 5. Agile + Scan
        emitter_id=4, freq_list_hz=[10e9, 15e9],
        pri_sec=1e-3, pulse_width_sec=1e-6, power_dbm=-44,
        dwell_sec=20e-3, scan_period_sec=1.0, beam_width_deg=20.0,
    ))
    sim.add_emitter(PriJitterEmitter(         # 6. PRI Jitter
        emitter_id=5, center_freq_hz=12e9,
        nominal_pri_sec=1e-3, pulse_width_sec=1e-6, power_dbm=-41,
        jitter_fraction=0.15,
    ))

    # Enable realistic channel model (FSPL + Rayleigh fading + shadowing)
    # on every emitter in this scenario.
    for emitter in sim.emitters:
        emitter.enable_channel = True

    sim.reset(seed=42)
    truth = sim.get_ground_truth()
    print(f"\nTotal pulses in sim: {truth['total_pulses']}")
    for e_stat in truth['emitters']:
        print(f"  E{e_stat['emitter_id']}: {e_stat['type']:25s} {e_stat['n_pulses']:6d} pulses")


def example_dynamic_phenomena():
    """3. The 4 dynamic RF phenomena — what makes the simulation non-static."""
    print()
    print("=" * 65)
    print("EXAMPLE 3: Dynamic RF phenomena (what makes it non-static)")
    print("=" * 65)

    sim = RealRFSimulator(
        num_bands=36, total_bandwidth_hz=18e9, receiver_ibw_hz=500e6,
        Pd=1.0, Pfa=0.0, retune_cost_sec=0.0,
        seed=42, sim_duration_sec=30.0, slot_duration_sec=5.0,
    )

    # ---- D1: Delayed Arrival (emitter appears at t=8s) ----
    print("\n[D1] Delayed Arrival — emitter appears at t=8s")
    base_d1 = FixedContinuousEmitter(
        emitter_id=0, center_freq_hz=2e9,
        pri_sec=1e-3, pulse_width_sec=1e-6, power_dbm=-40,
    )
    base_d1.enable_channel = True
    sim.add_emitter(DynamicEmitter(
        emitter_id=0,
        base_emitter=base_d1,
        policy=DelayedArrivalPolicy(start_time_sec=8.0),
    ))

    # ---- D2: Random ON/OFF Bursts ----
    print("[D2] Interval ON/OFF — random bursts at 5 GHz")
    base_d2 = FixedContinuousEmitter(
        emitter_id=1, center_freq_hz=5e9,
        pri_sec=1e-3, pulse_width_sec=1e-6, power_dbm=-40,
    )
    base_d2.enable_channel = True
    sim.add_emitter(DynamicEmitter(
        emitter_id=1,
        base_emitter=base_d2,
        policy=IntervalOnOffPolicy(
            mean_on_sec=3.0, mean_off_sec=4.0, seed=42,
        ),
    ))

    # ---- D3: Regime Change (freq switch at t=15s) ----
    print("[D3] Regime Change — freq switches from 8 GHz to 12 GHz at t=15s")
    base_d3 = FixedContinuousEmitter(
        emitter_id=2, center_freq_hz=8e9,
        pri_sec=1e-3, pulse_width_sec=1e-6, power_dbm=-40,
    )
    base_d3.enable_channel = True
    sim.add_emitter(DynamicEmitter(
        emitter_id=2,
        base_emitter=base_d3,
        policy=RegimeChangePolicy(regimes=[
            RegimeChangePolicy.Regime(
                change_time_sec=15.0,
                freq_override_hz=12e9,
            ),
        ]),
    ))

    # ---- D4: Non-uniform dwell (Frequency Agile) ----
    print("[D4] Non-uniform Dwell — [5ms, 15ms, 5ms, 15ms] at 15 GHz / 16 GHz")
    agile = FrequencyAgileEmitter(
        emitter_id=3,
        freq_list_hz=[15e9, 16e9],
        pri_sec=1e-3, pulse_width_sec=1e-6, power_dbm=-40,
        dwell_schedule_sec=[5e-3, 15e-3],  # non-uniform!
    )
    agile.enable_channel = True
    sim.add_emitter(agile)

    sim.reset(seed=42)

    print(f"\nTotal pulses: {sim.get_ground_truth()['total_pulses']}")
    print(f"\n{'Slot':>4} {'t(s)':>5} {'Band':>4} {'Hit':>5} {'Pulses':>7} {'Note'}")
    print("-" * 55)
    for slot in range(6):
        obs = sim.step(band=slot)
        t = sim.last_metadata['time']
        notes = []
        if t < 8 and slot == 0:
            notes.append("D1: not started")
        elif 8 <= t < 15 and slot == 1:
            notes.append("D1: active!")
        elif t >= 15 and slot == 3:
            notes.append("D3: regime changed!")
        note_str = " | ".join(notes) if notes else ""
        print(f"{slot:>4} {t:>5.0f} {sim.last_metadata['selected_band']:>4} "
              f"{'YES' if obs else 'no':>5} "
              f"{len(obs):>7} {note_str}")

    # Show dwell asymmetry for D4
    pulses = [p for p in sim.get_all_pulses() if p.emitter_id == 3]
    if pulses:
        from collections import Counter
        freq_counts = Counter(p.frequency_hz for p in pulses)
        print(f"\n  D4 (non-uniform dwell) pulse counts:")
        for freq, count in sorted(freq_counts.items()):
            print(f"    {freq/1e9:.1f} GHz: {count} pulses")


def example_scheduler_template():
    """4. Replace _select_band with your bandit algorithm."""
    print()
    print("=" * 65)
    print("EXAMPLE 4: UCB1 scheduler template (replace _select_band)")
    print("=" * 65)

    sim = RealRFSimulator(
        num_bands=36, total_bandwidth_hz=18e9, receiver_ibw_hz=500e6,
        Pd=0.9, Pfa=0.01, retune_cost_sec=1e-3,
        seed=42, sim_duration_sec=5.0, slot_duration_sec=50e-3,
    )
    e1 = FixedContinuousEmitter(
        emitter_id=0, center_freq_hz=4e9,
        pri_sec=1e-3, pulse_width_sec=1e-6, power_dbm=-40,
    )
    e1.enable_channel = True
    sim.add_emitter(e1)
    e2 = FixedContinuousEmitter(
        emitter_id=1, center_freq_hz=10e9,
        pri_sec=1e-3, pulse_width_sec=1e-6, power_dbm=-40,
    )
    e2.enable_channel = True
    sim.add_emitter(e2)
    sim.reset(seed=42)

    class UCB1Scheduler:
        def __init__(self, n_bands):
            self.n_bands = n_bands
            self.hits = [0] * n_bands
            self.tries = [0] * n_bands

        def select_band(self):
            import math
            total = sum(self.tries) + 1
            scores = [
                (self.hits[i] / max(1, self.tries[i])) +
                math.sqrt(2 * math.log(total) / max(1, self.tries[i]))
                for i in range(self.n_bands)
            ]
            return scores.index(max(scores))

        def update(self, band, hit):
            self.tries[band] += 1
            if hit:
                self.hits[band] += 1

    sched = UCB1Scheduler(sim.config.num_bands)
    total_hits = 0
    for slot in range(100):
        band = sched.select_band()
        obs = sim.step(band=band)
        hit = bool(obs)
        sched.update(band, hit)
        if hit:
            total_hits += 1

    print(f"\nUCB1 over 100 slots: {total_hits} hits ({total_hits}%)")
    print(f"Best bands: top-5 by hit rate:")
    rates = [(i, sched.hits[i] / max(1, sched.tries[i])) for i in range(sim.config.num_bands)]
    for band, rate in sorted(rates, key=lambda x: -x[1])[:5]:
        print(f"  Band {band:2d}: {rate:.2%} ({sched.hits[band]}/{sched.tries[band]})")


if __name__ == "__main__":
    example_minimal()
    example_all_base_types()
    example_dynamic_phenomena()
    example_scheduler_template()
    print("\nAll examples complete.")

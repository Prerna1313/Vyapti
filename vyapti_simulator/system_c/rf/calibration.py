"""Reproducible Monte Carlo calibration for the System C pulse detector."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from .pulse_detector import CFARMatchedFilterDetector, PulseDetectorConfig
from .waveforms import add_awgn, generate_lfm_chirp


SNR_SWEEP_DB = tuple(float(value) for value in range(0, 36, 5))
CFAR_OFFSET_SWEEP_DB = (3.0, 6.0, 9.0, 10.0, 12.0, 15.0, 18.0, 21.0)


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    if trials <= 0:
        return None, None
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    radius = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius))


def _pulse_trial(config: PulseDetectorConfig, snr_db: float, seed: int) -> tuple[np.ndarray, float]:
    n = config.samples_per_tick
    pulse_samples = max(2, int(round(config.pulse_width_s * config.dsp_sample_rate_hz)))
    time_s = np.arange(pulse_samples, dtype=np.float64) / config.dsp_sample_rate_hz
    pulse = generate_lfm_chirp(
        time_s,
        f0=-0.5 * config.chirp_bandwidth_hz,
        f1=0.5 * config.chirp_bandwidth_hz,
        peak_power_w=1.0,
    )
    start = (n - pulse_samples) // 2
    clean = np.zeros(n, dtype=np.complex128)
    clean[start:start + pulse_samples] = pulse
    noise_power = float(np.mean(np.abs(pulse) ** 2) / (10.0 ** (snr_db / 10.0)))
    noise = add_awgn(np.zeros(n, dtype=np.complex128), noise_floor_w=noise_power,
                     rng=np.random.default_rng(seed))
    expected_toa_us = start / config.dsp_sample_rate_hz * 1e6
    return clean + noise, expected_toa_us


def run_receiver_calibration(
    *,
    trials: int = 1000,
    seed: int = 20261002,
    snr_grid_db: Sequence[float] = SNR_SWEEP_DB,
    cfar_offset_grid_db: Sequence[float] = CFAR_OFFSET_SWEEP_DB,
    config: PulseDetectorConfig | None = None,
) -> dict:
    """Measure detection vs SNR and noise-only false alarms vs CFAR offset.

    Detection probability counts a detection whose measured ToA is within
    two pulse widths of the injected pulse. False-alarm probability is the
    fraction of noise-only ticks that produce one or more PDWs; this is a
    frame-level operational probability, not per-cell CA-CFAR Pfa.
    """
    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be a positive integer")
    snr_values = tuple(float(value) for value in snr_grid_db)
    cfar_values = tuple(float(value) for value in cfar_offset_grid_db)
    if not snr_values or not cfar_values or not np.all(np.isfinite(snr_values + cfar_values)):
        raise ValueError("SNR and CFAR-offset sweeps must be nonempty and finite")
    base = config or PulseDetectorConfig(
        dsp_sample_rate_hz=10e6,
        tick_interval_s=1e-3,
        cfar_db=10.0,
        cfar_train_cells=20,
        cfar_guard_cells=16,
        pulse_width_s=2e-6,
        chirp_bandwidth_hz=1e6,
        carrier_freq_hz=3e9,
    )
    pulse_samples = max(2, int(round(base.pulse_width_s * base.dsp_sample_rate_hz)))
    if base.samples_per_tick <= pulse_samples + 2 * (base.cfar_train_cells + base.cfar_guard_cells):
        raise ValueError("tick must leave room for the pulse and CFAR training cells")

    pd_rows = []
    for snr_index, snr_db in enumerate(snr_values):
        detector = CFARMatchedFilterDetector(replace(base), emitter_map={}, rng=np.random.default_rng(seed))
        detected = 0
        for trial in range(trials):
            iq, expected_toa_us = _pulse_trial(
                base, snr_db, int(np.random.SeedSequence([seed, snr_index, trial]).generate_state(1)[0])
            )
            pdws = detector.detect(iq)
            tolerance_us = 2.0 * base.pulse_width_s * 1e6
            if np.any(np.abs(pdws.toa_us.astype(float) - expected_toa_us) <= tolerance_us):
                detected += 1
        low, high = _wilson_interval(detected, trials)
        pd_rows.append({
            "snr_db": snr_db,
            "true_detections": detected,
            "trials": trials,
            "pd": detected / trials,
            "pd_ci95_wilson": [low, high],
            "cfar_db": float(base.cfar_db),
        })

    pfa_rows = []
    for offset_index, cfar_db in enumerate(cfar_values):
        detector_config = replace(base, cfar_db=cfar_db)
        detector = CFARMatchedFilterDetector(
            detector_config, emitter_map={}, rng=np.random.default_rng(seed)
        )
        false_alarm_ticks = 0
        false_pdw_count = 0
        for trial in range(trials):
            noise = add_awgn(
                np.zeros(base.samples_per_tick, dtype=np.complex128),
                noise_floor_w=1.0,
                rng=np.random.default_rng(
                    int(np.random.SeedSequence([seed, 0xFA, offset_index, trial]).generate_state(1)[0])
                ),
            )
            pdws = detector.detect(noise)
            false_pdw_count += len(pdws)
            false_alarm_ticks += int(len(pdws) > 0)
        low, high = _wilson_interval(false_alarm_ticks, trials)
        pfa_rows.append({
            "cfar_db": cfar_db,
            "false_alarm_ticks": false_alarm_ticks,
            "trials": trials,
            "false_alarm_probability_per_tick": false_alarm_ticks / trials,
            "pfa_per_tick_ci95_wilson": [low, high],
            "false_pdws_per_tick": false_pdw_count / trials,
        })

    return {
        "protocol_version": "system_c_iq_receiver_calibration_v1",
        "seed": int(seed),
        "trials_per_point": trials,
        "snr_grid_db": list(snr_values),
        "cfar_offset_grid_db": list(cfar_values),
        "detector": {
            "kind": "one_dimensional_matched_filter_ca_cfar",
            "cfar_db": float(base.cfar_db),
            "cfar_train_cells_per_side": int(base.cfar_train_cells),
            "cfar_guard_cells_per_side": int(base.cfar_guard_cells),
            "dsp_sample_rate_hz": float(base.dsp_sample_rate_hz),
            "tick_interval_s": float(base.tick_interval_s),
            "pulse_width_s": float(base.pulse_width_s),
            "chirp_bandwidth_hz": float(base.chirp_bandwidth_hz),
            "noise_model": "continuous circular complex AWGN across each receiver tick",
        },
        "pd_vs_snr": pd_rows,
        "false_alarm_vs_cfar_offset": pfa_rows,
        "notes": [
            "SNR grid follows published CA-CFAR comparisons over 0 to 35 dB.",
            "The paper's Pfa target is not assumed to hold for this implementation; false alarms are measured empirically.",
            "False-alarm probability is per tick and must not be reported as per-cell Pfa.",
        ],
    }

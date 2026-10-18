"""
vyapti_simulator.system_c.rf.frequency_window
=============================================

Dynamic frequency-window abstractions and candidate window generator
for continuous RF spectrum observation and closed-loop UCB tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class FrequencyWindow:
    """
    A continuous RF frequency observation window for System C dwell tuning.

    Parameters
    ----------
    window_id : int
        Unique integer action index for this frequency window.
    freq_start_hz : float
        Lower frequency bound in Hz.
    freq_end_hz : float
        Upper frequency bound in Hz.
    center_freq_hz : float
        Center frequency in Hz.
    """
    window_id: int
    freq_start_hz: float
    freq_end_hz: float
    center_freq_hz: float

    @property
    def action_id(self) -> int:
        return self.window_id

    @property
    def start_hz(self) -> float:
        return self.freq_start_hz

    @property
    def end_hz(self) -> float:
        return self.freq_end_hz

    @property
    def center_hz(self) -> float:
        return self.center_freq_hz

    @property
    def bandwidth_hz(self) -> float:
        return self.freq_end_hz - self.freq_start_hz

    @property
    def center_freq_ghz(self) -> float:
        return self.center_freq_hz / 1e9

    @property
    def start_ghz(self) -> float:
        return self.freq_start_hz / 1e9

    @property
    def end_ghz(self) -> float:
        return self.freq_end_hz / 1e9

    def to_tuple(self) -> Tuple[float, float]:
        return (self.freq_start_hz, self.freq_end_hz)

    def contains(self, freq_hz: float) -> bool:
        return self.freq_start_hz <= float(freq_hz) <= self.freq_end_hz

    def to_dict(self) -> Dict:
        return {
            "action_id": self.action_id,
            "window_id": self.window_id,
            "start_hz": self.freq_start_hz,
            "end_hz": self.freq_end_hz,
            "center_hz": self.center_freq_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "freq_start_hz": self.freq_start_hz,
            "freq_end_hz": self.freq_end_hz,
            "center_freq_hz": self.center_freq_hz,
            "center_freq_ghz": round(self.center_freq_ghz, 4),
            "bandwidth_mhz": round(self.bandwidth_hz / 1e6, 2),
        }


def generate_frequency_windows(
    rf_start_hz: float = 0.0,
    rf_end_hz: float = 18.0e9,
    window_bandwidth_hz: float = 500.0e6,
    tuning_step_hz: Optional[float] = None,
) -> List[FrequencyWindow]:
    """
    Generate dynamic candidate frequency windows from continuous RF capabilities.

    Parameters
    ----------
    rf_start_hz : float
        Lower bound of tunable RF spectrum in Hz (default 0.0 Hz).
    rf_end_hz : float
        Upper bound of tunable RF spectrum in Hz (default 18.0 GHz).
    window_bandwidth_hz : float
        Bandwidth of each observation window in Hz (e.g. 500 MHz, 250 MHz, etc.).
    tuning_step_hz : float, optional
        Tuning step size in Hz. If None, contiguous non-overlapping windows
        (step = window_bandwidth_hz) are generated. If smaller than window_bandwidth_hz,
        overlapping/sliding candidate windows are generated.

    Returns
    -------
    List[FrequencyWindow]
        Ordered list of candidate frequency windows representing the scheduler action space.
    """
    if window_bandwidth_hz <= 0:
        raise ValueError("window_bandwidth_hz must be positive")
    if rf_end_hz <= rf_start_hz:
        raise ValueError("rf_end_hz must be strictly greater than rf_start_hz")

    step = float(tuning_step_hz) if tuning_step_hz is not None else float(window_bandwidth_hz)
    if step <= 0:
        raise ValueError("tuning_step_hz must be positive")

    windows: List[FrequencyWindow] = []
    current_start = float(rf_start_hz)
    wid = 0

    while current_start + window_bandwidth_hz <= rf_end_hz + 1e-6:
        w_start = current_start
        w_end = min(current_start + window_bandwidth_hz, float(rf_end_hz))
        w_center = (w_start + w_end) / 2.0
        windows.append(FrequencyWindow(
            window_id=wid,
            freq_start_hz=w_start,
            freq_end_hz=w_end,
            center_freq_hz=w_center,
        ))
        wid += 1
        current_start += step
        if current_start >= rf_end_hz:
            break

    # If the end of spectrum wasn't reached and there is a remainder:
    if not windows or windows[-1].freq_end_hz < rf_end_hz - 1e-6:
        w_start = max(float(rf_start_hz), float(rf_end_hz) - window_bandwidth_hz)
        w_end = float(rf_end_hz)
        windows.append(FrequencyWindow(
            window_id=wid,
            freq_start_hz=w_start,
            freq_end_hz=w_end,
            center_freq_hz=(w_start + w_end) / 2.0,
        ))

    return windows

"""
Vyapti 3.0: GPU Tensorized RF Physics Engine (PyTorch Prototype)

This module provides a hardware-accelerated alternative to the standard RealTimeRFSimulator.
By utilizing PyTorch, it can distribute the massive I/Q complex math array generation across
GPU cores, enabling the simulation of thousands of simultaneous emitters in real-time.
"""

import numpy as np
import time
import logging
from typing import Callable, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not installed. TorchRFSimulator will fall back to CPU or fail.")

class TorchRFSimulator:
    """
    GPU-accelerated RF Physics Engine.
    Mirrors the API of RealTimeRFSimulator but executes math on PyTorch tensors.
    """
    def __init__(self, sample_rate_hz: float = 100e6, tick_interval_s: float = 1e-3, device: str = None):
        self.sample_rate = sample_rate_hz
        self.tick_interval = tick_interval_s
        self.num_samples = int(self.sample_rate * self.tick_interval)

        if device is None:
            self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # The master buffer holding the simulated analog waves
        self.iq_buffer = torch.zeros(self.num_samples, dtype=torch.complex64, device=self.device)
        self.emitters = []

    def add_emitter(
        self,
        kinematic: "KinematicEmitter",          # noqa: F821
        waveform_fn: Callable,
        *,
        emitter_id: Optional[int] = None,
        channel: Optional["RayleighFadingChannel"] = None,  # noqa: F821
        pri_sec: float = 1e-3,
        pulse_width_s: float = 1e-6,
        carrier_freq_hz: float = 0.0,
        aoa_deg: float = 0.0,
    ) -> int:
        """Register an emitter with the tensor engine (matches RealTimeRFSimulator API)."""
        if emitter_id is None:
            emitter_id = len(self.emitters)
        # For the tensor prototype, we store the minimum needed for the current simulate_dwell.
        # We approximate bandwidth as inverse pulse width and set a default power.
        self.emitters.append({
            "freq": carrier_freq_hz,
            "bw": 1.0 / max(pulse_width_s, 1e-9),  # Hz, avoid division by zero
            "power": 0.0,   # dBm, placeholder since power info not in new signature
        })
        return emitter_id

    def simulate_dwell(self, thermal_noise_dbm: float = -100.0) -> np.ndarray:
        """
        Generates the complex baseband I/Q buffer using PyTorch tensor math.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to run the TorchRFSimulator.")

        t = torch.arange(0, self.num_samples, device=self.device, dtype=torch.float32) / self.sample_rate

        # 1. Reset buffer to zero
        self.iq_buffer.zero_()

        # 2. Vectorized Emitter Generation
        # For prototype, we generate sine waves for each emitter.
        # In a full migration, the waveform_fns would be rewritten as torch operations.
        for em in self.emitters:
            # Convert dBm to linear amplitude
            amp = 10 ** ((em["power"] - 30) / 20)
            phase = 2.0 * np.pi * em["freq"] * t

            # Complex exponential: cos + j*sin
            wave = amp * torch.polar(torch.ones_like(phase), phase)
            self.iq_buffer += wave

        # 3. Add Thermal Noise (AWGN)
        noise_amp = 10 ** ((thermal_noise_dbm - 30) / 20)
        noise_i = torch.randn(self.num_samples, device=self.device, dtype=torch.float32) * noise_amp
        noise_q = torch.randn(self.num_samples, device=self.device, dtype=torch.float32) * noise_amp
        noise = torch.complex(noise_i, noise_q)

        self.iq_buffer += noise

        # 4. Return to CPU numpy array for legacy DSP processing
        return self.iq_buffer.cpu().numpy()

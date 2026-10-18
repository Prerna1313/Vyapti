"""
vyapti_simulator.system_c.rf.scenario
=====================================

RF Scenario configuration decoupling the physical RF environment
from the scheduler's candidate action space.

Guarantees:
- Complete parameterization of RF frequency range, emitters, receiver constraints, and scheduler settings.
- Generates dynamic FrequencyWindow action candidates.
- The scheduler does NOT receive the emitter list or ground truth as answers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from vyapti_simulator.core.emitter_spec import SyntheticEmitterSpec
from .frequency_window import FrequencyWindow, generate_frequency_windows


@dataclass
class ReceiverConfig:
    """
    Physical receiver hardware & signal processing parameters.
    """
    observation_bandwidth_hz: float = 500.0e6
    tuning_step_hz: float = 500.0e6
    dwell_time_ms: float = 50.0
    sample_rate_hz: float = 10.0e6
    cfar_threshold_db: float = 10.0
    min_pulse_samples: int = 3
    noise_floor_w: Optional[float] = None
    snr_db: float = 20.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_bandwidth_hz": self.observation_bandwidth_hz,
            "tuning_step_hz": self.tuning_step_hz,
            "dwell_time_ms": self.dwell_time_ms,
            "sample_rate_hz": self.sample_rate_hz,
            "cfar_threshold_db": self.cfar_threshold_db,
            "min_pulse_samples": self.min_pulse_samples,
            "snr_db": self.snr_db,
        }


@dataclass
class SchedulerConfig:
    """
    Online multi-armed bandit scheduler parameters.
    """
    algorithm: str = "UCB1"
    exploration_parameter: float = math.sqrt(2.0)
    window_bandwidth_hz: float = 500.0e6
    tuning_step_hz: Optional[float] = None
    seed: int = 42
    reward_bounds: Tuple[float, float] = (-0.01, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "exploration_parameter": self.exploration_parameter,
            "window_bandwidth_hz": self.window_bandwidth_hz,
            "tuning_step_hz": self.tuning_step_hz,
            "seed": self.seed,
            "reward_bounds": list(self.reward_bounds),
        }


@dataclass
class RFScenario:
    """
    Complete specification of the physical RF environment and receiver/scheduler setup.

    The scheduler only receives observations from System C (I/Q, PSD, CFAR detections);
    it never receives the emitter list.
    """
    rf_start_hz: float = 2.0e9
    rf_end_hz: float = 18.0e9
    emitters: List[SyntheticEmitterSpec] = field(default_factory=list)
    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @property
    def emitter_count(self) -> int:
        return len(self.emitters)

    @property
    def total_bandwidth_hz(self) -> float:
        return max(1.0, self.rf_end_hz - self.rf_start_hz)

    def generate_candidate_windows(self) -> List[FrequencyWindow]:
        """
        Generate candidate FrequencyWindow actions dynamically from the scenario parameters.
        """
        step = self.scheduler.tuning_step_hz or self.receiver.tuning_step_hz
        bw = self.scheduler.window_bandwidth_hz or self.receiver.observation_bandwidth_hz
        return generate_frequency_windows(
            rf_start_hz=self.rf_start_hz,
            rf_end_hz=self.rf_end_hz,
            window_bandwidth_hz=bw,
            tuning_step_hz=step,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rf_start_hz": self.rf_start_hz,
            "rf_end_hz": self.rf_end_hz,
            "rf_start_ghz": self.rf_start_hz / 1e9,
            "rf_end_ghz": self.rf_end_hz / 1e9,
            "total_bandwidth_ghz": self.total_bandwidth_hz / 1e9,
            "emitter_count": self.emitter_count,
            "receiver": self.receiver.to_dict(),
            "scheduler": self.scheduler.to_dict(),
        }

    @classmethod
    def scenario_a(cls) -> "RFScenario":
        """
        Scenario A: Broad-spectrum 2.0 - 18.0 GHz with 8 diverse emitters.
        """
        emitters = [
            SyntheticEmitterSpec(
                emitter_id=1,
                center_freq_hz=2.78e9,
                pri_sec=1.5e-3,
                pulse_width_sec=12.0e-6,
                chirp_bandwidth_hz=1.0e6,
                aoa_deg=20.0,
                snr_db=18.0,
                emitter_type="fixed_continuous",
            ),
            SyntheticEmitterSpec(
                emitter_id=2,
                center_freq_hz=3.25e9,
                pri_sec=1.0e-3,
                pulse_width_sec=10.0e-6,
                chirp_bandwidth_hz=2.0e6,
                aoa_deg=45.0,
                snr_db=22.0,
                emitter_type="fixed_continuous",
            ),
            SyntheticEmitterSpec(
                emitter_id=3,
                center_freq_hz=7.86e9,
                pri_sec=0.8e-3,
                pulse_width_sec=5.0e-6,
                chirp_bandwidth_hz=1.5e6,
                aoa_deg=30.0,
                snr_db=19.0,
                emitter_type="fixed_continuous",
            ),
            SyntheticEmitterSpec(
                emitter_id=4,
                center_freq_hz=8.45e9,
                pri_sec=0.5e-3,
                pulse_width_sec=2.5e-6,
                chirp_bandwidth_hz=3.0e6,
                aoa_deg=60.0,
                snr_db=24.0,
                emitter_type="fixed_continuous",
            ),
            SyntheticEmitterSpec(
                emitter_id=5,
                center_freq_hz=9.25e9,
                pri_sec=0.7e-3,
                pulse_width_sec=4.0e-6,
                chirp_bandwidth_hz=2.0e6,
                aoa_deg=75.0,
                snr_db=17.5,
                emitter_type="fixed_continuous",
            ),
            SyntheticEmitterSpec(
                emitter_id=6,
                center_freq_hz=11.42e9,
                pri_sec=1.2e-3,
                pulse_width_sec=8.0e-6,
                chirp_bandwidth_hz=2.5e6,
                aoa_deg=120.0,
                snr_db=21.0,
                emitter_type="fixed_continuous",
            ),
            SyntheticEmitterSpec(
                emitter_id=7,
                center_freq_hz=14.20e9,
                pri_sec=0.9e-3,
                pulse_width_sec=6.0e-6,
                chirp_bandwidth_hz=3.5e6,
                aoa_deg=105.0,
                snr_db=18.5,
                emitter_type="fixed_continuous",
            ),
            SyntheticEmitterSpec(
                emitter_id=8,
                center_freq_hz=15.23e9,
                pri_sec=0.6e-3,
                pulse_width_sec=3.0e-6,
                chirp_bandwidth_hz=4.0e6,
                aoa_deg=90.0,
                snr_db=20.0,
                emitter_type="fixed_continuous",
            ),
        ]
        return cls(
            rf_start_hz=2.0e9,
            rf_end_hz=18.0e9,
            emitters=emitters,
            receiver=ReceiverConfig(
                observation_bandwidth_hz=500.0e6,
                tuning_step_hz=500.0e6,
                dwell_time_ms=50.0,
            ),
            scheduler=SchedulerConfig(
                algorithm="UCB1",
                exploration_parameter=math.sqrt(2.0),
                window_bandwidth_hz=500.0e6,
                tuning_step_hz=500.0e6,
            ),
        )

    @classmethod
    def scenario_b(cls) -> "RFScenario":
        """
        Scenario B: Narrower 4.0 - 12.0 GHz band with 3 emitters (8 GHz total range).
        With 500 MHz windows, this creates 16 actions instead of 32 or 36.
        """
        emitters = [
            SyntheticEmitterSpec(
                emitter_id=101,
                center_freq_hz=5.25e9,  # Inside [5.0 - 5.5 GHz]
                pri_sec=1.1e-3,
                pulse_width_sec=8.0e-6,
                chirp_bandwidth_hz=1.2e6,
                aoa_deg=35.0,
                snr_db=20.0,
                emitter_type="fixed_continuous",
            ),
            SyntheticEmitterSpec(
                emitter_id=102,
                center_freq_hz=7.80e9,  # Inside [7.5 - 8.0 GHz]
                pri_sec=0.75e-3,
                pulse_width_sec=4.0e-6,
                chirp_bandwidth_hz=2.2e6,
                aoa_deg=65.0,
                snr_db=23.0,
                emitter_type="fixed_continuous",
            ),
            SyntheticEmitterSpec(
                emitter_id=103,
                center_freq_hz=10.60e9,  # Inside [10.5 - 11.0 GHz]
                pri_sec=1.4e-3,
                pulse_width_sec=10.0e-6,
                chirp_bandwidth_hz=1.8e6,
                aoa_deg=110.0,
                snr_db=19.5,
                emitter_type="fixed_continuous",
            ),
        ]
        return cls(
            rf_start_hz=4.0e9,
            rf_end_hz=12.0e9,
            emitters=emitters,
            receiver=ReceiverConfig(
                observation_bandwidth_hz=500.0e6,
                tuning_step_hz=500.0e6,
                dwell_time_ms=50.0,
            ),
            scheduler=SchedulerConfig(
                algorithm="UCB1",
                exploration_parameter=math.sqrt(2.0),
                window_bandwidth_hz=500.0e6,
                tuning_step_hz=500.0e6,
            ),
        )

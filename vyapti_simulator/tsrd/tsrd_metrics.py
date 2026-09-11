import numpy as np
from typing import Dict, Any, Optional
from vyapti_simulator.core.metrics import MetricsEngine, MetricsConfig
from vyapti_simulator.tsrd.tsrd_adapter import load_stare_mode_as_occupancy_grid, extract_emitter_metadata

class TSRDMetricsEngine(MetricsEngine):
    """
    Extended MetricsEngine for computing comprehensive TSRD-specific metrics
    using Stare Mode ground truth.
    """
    def __init__(self, config: MetricsConfig, stare_mode_file: Optional[str] = None, scan_mode_file: Optional[str] = None):
        super().__init__(config)
        self.stare_mode_file = stare_mode_file
        self.scan_mode_file = scan_mode_file
        
        self.stare_occupancy_grid = None
        self.emitter_data = None
        
        if self.config.use_tsr_stare_mode and self.stare_mode_file:
            self.stare_occupancy_grid = load_stare_mode_as_occupancy_grid(self.stare_mode_file)
            if self.config.compute_emitter_population_metrics:
                self.emitter_data = extract_emitter_metadata(self.stare_mode_file)

    def record_result(self, episode_id, seed, scheduler_name, scenario_config, trajectory, truth_grid, receiver_accounting=None) -> Dict[str, Any]:
        base_result = super().record_result(
            episode_id=episode_id,
            seed=seed,
            scheduler_name=scheduler_name,
            scenario_config=scenario_config,
            trajectory=trajectory,
            truth_grid=truth_grid,
            receiver_accounting=receiver_accounting
        )
        
        if self.config.compute_comprehensive_metrics and self.stare_occupancy_grid is not None:
            comp_metrics = self.compute_comprehensive_metrics(trajectory, self.stare_occupancy_grid)
            base_result.update(comp_metrics)
            
        return base_result

    def compute_comprehensive_metrics(self, trajectory, occupancy_grid: np.ndarray) -> Dict[str, Any]:
        if not trajectory:
            return {}
            
        hits = np.array([step.observation.get('hit', False) for step in trajectory], dtype=int)
        actions = np.array([step.action for step in trajectory], dtype=int)
        slots = np.array([step.time_slot for step in trajectory], dtype=int)
        
        # Guard against index out of bounds
        max_b, max_s = occupancy_grid.shape
        valid_idx = (actions >= 0) & (actions < max_b) & (slots >= 0) & (slots < max_s)
        
        occupied = np.zeros(len(trajectory), dtype=int)
        occupied[valid_idx] = occupancy_grid[actions[valid_idx], slots[valid_idx]]
        empty = 1 - occupied
        
        true_hits = np.sum(hits & occupied)
        false_alarms = np.sum(hits & empty)
        n_occupied = np.sum(occupied)
        n_empty = np.sum(empty)
        
        pd = true_hits / n_occupied if n_occupied > 0 else 0.0
        pfa = false_alarms / n_empty if n_empty > 0 else 0.0
        
        missed_detections = n_occupied - true_hits
        correct_rejections = n_empty - false_alarms
        precision = true_hits / (true_hits + false_alarms) if (true_hits + false_alarms) > 0 else 0.0
        recall = pd
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Coverage
        total_dwells = len(trajectory)
        occupancy_rate = n_occupied / total_dwells if total_dwells > 0 else 0.0
        unique_occupied_bands = len(set(np.where(occupancy_grid.any(axis=1))[0]))
        band_utilization = unique_occupied_bands / occupancy_grid.shape[0] if occupancy_grid.shape[0] > 0 else 0.0
        
        # Scheduler
        total_hits = np.sum(hits)
        unique_bands_visited = len(set(actions))
        exploration_rate = unique_bands_visited / occupancy_grid.shape[0] if occupancy_grid.shape[0] > 0 else 0.0
        
        # Efficiency
        dwell_efficiency = true_hits / total_dwells if total_dwells > 0 else 0.0
        waste_rate = np.sum(empty) / total_dwells if total_dwells > 0 else 0.0
        
        # Latency (using the existing compute_latency_metrics from parent if latencies were collected)
        latency_metrics = self.compute_latency_metrics() if hasattr(self, 'compute_latency_metrics') else {}
        
        # Per-band
        per_band = {}
        if self.config.compute_per_band_metrics:
            for b in range(occupancy_grid.shape[0]):
                b_visits = np.sum(actions == b)
                b_hits = np.sum((actions == b) & (hits == 1))
                b_occ = np.sum(occupancy_grid[b, :])
                per_band[str(b)] = {
                    "visits": int(b_visits),
                    "hits": int(b_hits),
                    "occupied_cells": int(b_occ),
                    "pd": float(b_hits / b_occ) if b_occ > 0 else 0.0
                }
                
        # Temporal
        temporal = {}
        if self.config.compute_temporal_metrics:
            pd_cum = []
            for t in range(1, len(trajectory) + 1):
                h_t = hits[:t]
                o_t = occupied[:t]
                th_t = np.sum(h_t & o_t)
                no_t = np.sum(o_t)
                pd_cum.append(float(th_t / no_t if no_t > 0 else 0.0))
            temporal["pd_cumulative"] = pd_cum
            
        return {
            "comprehensive_detection": {
                "true_pd": float(pd),
                "true_pfa": float(pfa),
                "true_hits": int(true_hits),
                "false_alarms": int(false_alarms),
                "missed_detections": int(missed_detections),
                "correct_rejections": int(correct_rejections),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            },
            "comprehensive_coverage": {
                "occupancy_rate": float(occupancy_rate),
                "unique_occupied_bands": int(unique_occupied_bands),
                "band_utilization": float(band_utilization),
                "time_utilization": float(np.sum(occupancy_grid.any(axis=0)) / occupancy_grid.shape[1] if occupancy_grid.shape[1] > 0 else 0.0),
            },
            "comprehensive_scheduler": {
                "observed_hit_rate": float(total_hits / total_dwells if total_dwells > 0 else 0.0),
                "total_hits": int(total_hits),
                "total_dwells": int(total_dwells),
                "unique_bands_visited": int(unique_bands_visited),
                "exploration_rate": float(exploration_rate),
            },
            "comprehensive_efficiency": {
                "dwell_efficiency": float(dwell_efficiency),
                "waste_rate": float(waste_rate),
                "useful_dwells": int(true_hits),
                "wasted_dwells": int(np.sum(empty)),
            },
            "comprehensive_latency": latency_metrics,
            "per_band_metrics": per_band,
            "temporal_metrics": temporal,
            "emitter_population": self.emitter_data if self.emitter_data else {},
            "spectral_environment": {
                "total_spectrum_mhz": occupancy_grid.shape[0] * 500,
                "occupied_spectrum_mhz": unique_occupied_bands * 500,
                "spectral_occupancy_percent": float(band_utilization * 100),
                "congestion_index": float(occupancy_rate),
                "interference_potential": "high" if occupancy_rate > 0.5 else "medium" if occupancy_rate > 0.2 else "low"
            },
            "assumptions_and_limitations": {
                "noise_floor_dbm": -130,
                "antenna_gain_dbi": 0,
                "shadowing_fading": "Not modeled",
                "jammer_info": "Not available",
                "weather_effects": "Not modeled"
            }
        }

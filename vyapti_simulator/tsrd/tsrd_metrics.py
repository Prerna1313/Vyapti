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
            "threat_assessment": self._calculate_threat_assessment(self.emitter_data) if getattr(self, "emitter_data", None) else {},
            "spatial_analysis": self._calculate_spatial_metrics(self.emitter_data) if getattr(self, "emitter_data", None) else {},
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

    def _calculate_threat_assessment(self, emitter_data: Dict[str, Any]) -> Dict[str, Any]:
        if not emitter_data or 'emitters' not in emitter_data or not emitter_data['emitters']:
            return {}
            
        threat_distribution = {"High": 0, "Medium": 0, "Low": 0}
        per_emitter_threat = []
        
        for e in emitter_data['emitters']:
            score = 0
            
            # PRI logic
            min_pri = min(e.get('pris_us', [100])) if e.get('pris_us') else 100
            if min_pri < 10:
                score += 5
            elif min_pri < 50:
                score += 3
            else:
                score += 1
                
            # Frequency logic
            max_freq = max(e.get('frequencies_mhz', [1000])) if e.get('frequencies_mhz') else 1000
            if max_freq > 10000:
                score += 2
            elif max_freq > 5000:
                score += 1
                
            # Pulse Width logic
            min_pw = min(e.get('pws_us', [2.0])) if e.get('pws_us') else 2.0
            if min_pw < 0.5:
                score += 2
            elif min_pw < 1.0:
                score += 1
                
            # Type logic
            e_type = str(e.get('type', '')).lower()
            if 'fire' in e_type or 'missile' in e_type:
                score += 3
            elif 'search' in e_type:
                score += 1
                
            # Classification
            if score >= 7:
                level = "High"
            elif score >= 4:
                level = "Medium"
            else:
                level = "Low"
                
            threat_distribution[level] += 1
            per_emitter_threat.append({
                "emitter_id": e.get('emitter_id'),
                "threat_level": level,
                "threat_score": score
            })
            
        total = len(per_emitter_threat)
        threat_percentage = {k: (v / total * 100) for k, v in threat_distribution.items()}
        dominant = max(threat_distribution, key=threat_distribution.get)
        
        return {
            "threat_distribution": threat_distribution,
            "threat_percentage": threat_percentage,
            "dominant_threat_level": dominant,
            "per_emitter_threat": per_emitter_threat
        }

    def _calculate_spatial_metrics(self, emitter_data: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np
        import math
        
        if not emitter_data or 'emitters' not in emitter_data or not emitter_data['emitters']:
            return {}
            
        ranges = []
        aoas = []
        
        for e in emitter_data['emitters']:
            pos = e.get('position_km')
            if pos and len(pos) >= 2:
                x, y = pos[0], pos[1]
                z = pos[2] if len(pos) >= 3 else 0.0
                r = math.sqrt(x**2 + y**2 + z**2)
                aoa = math.degrees(math.atan2(y, x))
                if aoa < 0:
                    aoa += 360.0
                ranges.append(r)
                aoas.append(aoa)
                
        if not ranges:
            return {}
            
        ranges = np.array(ranges)
        aoas = np.array(aoas)
        
        # Clustering
        q1 = int(np.sum((aoas >= 0) & (aoas < 90)))
        q2 = int(np.sum((aoas >= 90) & (aoas < 180)))
        q3 = int(np.sum((aoas >= 180) & (aoas < 270)))
        q4 = int(np.sum((aoas >= 270) & (aoas <= 360)))
        
        r_close = int(np.sum(ranges < 50))
        r_med = int(np.sum((ranges >= 50) & (ranges <= 200)))
        r_long = int(np.sum(ranges > 200))
        
        return {
            "aoa_statistics": {
                "mean_deg": float(np.mean(aoas)),
                "std_deg": float(np.std(aoas)),
                "min_deg": float(np.min(aoas)),
                "max_deg": float(np.max(aoas))
            },
            "aoa_clustering": {
                "0_to_90": q1,
                "90_to_180": q2,
                "180_to_270": q3,
                "270_to_360": q4
            },
            "range_statistics": {
                "mean_km": float(np.mean(ranges)),
                "std_km": float(np.std(ranges)),
                "min_km": float(np.min(ranges)),
                "max_km": float(np.max(ranges))
            },
            "range_distribution": {
                "close_under_50km": r_close,
                "medium_50_to_200km": r_med,
                "long_over_200km": r_long
            }
        }

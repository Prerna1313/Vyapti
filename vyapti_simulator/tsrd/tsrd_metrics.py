import numpy as np
from typing import Dict, Any, Optional
from vyapti_simulator.core.metrics import MetricsEngine, MetricsConfig, compute_exploration_metrics
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
            comp_metrics = self.compute_comprehensive_metrics(
                trajectory, self.stare_occupancy_grid, truth_grid=truth_grid,
                base_result=base_result)
            base_result.update(comp_metrics)

        return base_result

    def compute_comprehensive_metrics(self, trajectory, occupancy_grid: np.ndarray,
                                      truth_grid=None, base_result=None) -> Dict[str, Any]:
        if not trajectory:
            return {}

        hits = np.array([step.observation.get('hit', False) for step in trajectory], dtype=int)
        actions = np.array([step.action for step in trajectory], dtype=int)
        slots = np.array([step.time_slot for step in trajectory], dtype=int)

        max_b, max_s = occupancy_grid.shape
        valid_idx = (actions >= 0) & (actions < max_b) & (slots >= 0) & (slots < max_s)

        occupied = np.zeros(len(trajectory), dtype=int)
        occupied[valid_idx] = occupancy_grid[actions[valid_idx], slots[valid_idx]]
        empty = 1 - occupied

        true_hits = np.sum(hits & occupied)
        false_alarms = np.sum(hits & empty)
        n_occupied = np.sum(occupied)
        n_empty = np.sum(empty)
        total_truth_opportunities = np.sum(occupancy_grid)

        # Tier A: PS26055
        conditional_pd = float(true_hits / n_occupied if n_occupied > 0 else 0.0)
        true_pfa = float(false_alarms / n_empty if n_empty > 0 else 0.0)
        opportunity_interception_ratio = float(true_hits / total_truth_opportunities if total_truth_opportunities > 0 else 0.0)
        
        dwell_ms = scenario_config.get('dwell_time_ms', 10.0)
        retune_ms = scenario_config.get('retune_time_ms', 1.0)
        mission_duration_s = len(trajectory) * (dwell_ms + retune_ms) / 1000.0
        
        average_intercept_rate_hz = float(true_hits / mission_duration_s) if mission_duration_s > 0 else 0.0
        average_seconds_per_intercept = float(mission_duration_s / true_hits) if true_hits > 0 else None
        
        # Tier B: Scheduler
        total_dwells = len(trajectory)
        true_detection_yield = float(true_hits / total_dwells if total_dwells > 0 else 0.0)
        empty_scan_fraction = float(np.sum(empty) / total_dwells if total_dwells > 0 else 0.0)

        # Tier C: Prediction
        prediction_metrics = (base_result or {}).get("prediction_metrics", {})
        preds = [s.prediction for s in trajectory if getattr(s, 'prediction', None) is not None]
        tier_c = {
            "prediction_accuracy": prediction_metrics.get("percentage_correct_predictions"),
            "valid_prediction_count": prediction_metrics.get("predictions_scored", 0),
            "prediction_count": len(preds),
            "average_intercept_time_error": prediction_metrics.get("average_intercept_time_error_slots"),
            "median_intercept_time_error": prediction_metrics.get("median_intercept_time_error_slots"),
            "p95_intercept_time_error": None,
            "brier_score": prediction_metrics.get("brier_score"),
            "unavailable_reason": prediction_metrics.get("unavailable_reason"),
        }

        # Tier D: Environment descriptors
        unique_occupied_bands = int(np.sum(np.any(occupancy_grid, axis=1)))
        environment_band_occupancy = float(unique_occupied_bands / max_b)
        environment_time_occupancy = float(np.sum(occupancy_grid.any(axis=0)) / max_s if max_s > 0 else 0.0)

        # Per-band
        per_band = {}
        if self.config.compute_per_band_metrics:
            for b in range(max_b):
                b_visits = np.sum(actions == b)
                b_hits = np.sum((actions == b) & (hits == 1))
                b_occ = np.sum(occupancy_grid[b, :])
                per_band[str(b)] = {
                    "visits": int(b_visits),
                    "true_hits": int(b_hits),
                    "active_opportunities": int(b_occ),
                    "conditional_pd": float(b_hits / b_occ) if b_occ > 0 else 0.0
                }

        # Temporal cumulative PD
        temporal = {}
        if self.config.compute_temporal_metrics:
            pd_cum = []
            cum_hits = 0
            cum_occ = 0
            for t in range(len(trajectory)):
                if occupied[t]:
                    cum_occ += 1
                    if hits[t]:
                        cum_hits += 1
                pd_cum.append(float(cum_hits / cum_occ if cum_occ > 0 else 0.0))
            temporal["pd_cumulative"] = pd_cum

        # Latency
        latency_metrics = self.compute_latency_metrics() if hasattr(self, 'compute_latency_metrics') else {}
        exploration = compute_exploration_metrics(trajectory, max_b)

        tier_d = {
            "active_band_time_opportunities": int(total_truth_opportunities),
            "environment_band_occupancy": environment_band_occupancy,
            "environment_time_occupancy": environment_time_occupancy,
            "unique_occupied_bands": unique_occupied_bands,
            "emitter_population": self.emitter_data if self.emitter_data else {}
        }

        if getattr(self, "emitter_data", None):
            tier_d["spatial_analysis"] = self._calculate_spatial_metrics(self.emitter_data)

        # Plugs for Emitter Interception (To be wired via full HiddenTruthGrid)
        discovery = (base_result or {}).get("discovery_metrics", {})
        emitter_ratio = discovery.get("emitter_interception_ratio")
        emitter_audit = {
            "eligible_emitter_count": None,
            "intercepted_emitter_count": None,
            "emitter_interception_ratio": emitter_ratio,
            "emitters_never_intercepted": []
        }

        return {
            "tier_a_ps26055": {
                "conditional_pd": conditional_pd,
                "true_pfa": true_pfa,
                "opportunity_interception_ratio": opportunity_interception_ratio,
                "emitter_interception_ratio": emitter_ratio,
                "average_intercept_rate_hz": average_intercept_rate_hz,
                "average_seconds_per_intercept": average_seconds_per_intercept,
                "emitter_audit": emitter_audit
            },
            "tier_b_scheduler": {
                "true_detection_yield_per_dwell": true_detection_yield,
                "empty_scan_fraction": empty_scan_fraction,
                "total_dwells": total_dwells,
                "action_entropy": exploration.get("action_entropy", 0.0),
                "normalized_action_entropy": exploration.get("normalized_action_entropy", 0.0),
                "unique_band_coverage": exploration.get("unique_band_coverage", 0.0),
                "exploration_fraction": exploration.get("exploration_fraction", 0.0),
            },
            "tier_c_prediction": tier_c,
            "tier_d_environment": tier_d,
            "tier_e_implementation": latency_metrics,
            "per_band_metrics": per_band,
            "temporal_metrics": temporal
        }

    def _calculate_spatial_metrics(self, emitter_data: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np
        import math

        if not emitter_data or 'emitters' not in emitter_data or not emitter_data['emitters']:
            return {}

        aoas = []

        for e in emitter_data['emitters']:
            pos = e.get('position_km')
            if pos and len(pos) >= 2:
                x, y = pos[0], pos[1]
                aoa = math.degrees(math.atan2(y, x))
                if aoa < 0:
                    aoa += 360.0
                aoas.append(aoa)

        if not aoas:
            return {}

        aoas = np.array(aoas)

        # Clustering
        q1 = int(np.sum((aoas >= 0) & (aoas < 90)))
        q2 = int(np.sum((aoas >= 90) & (aoas < 180)))
        q3 = int(np.sum((aoas >= 180) & (aoas < 270)))
        q4 = int(np.sum((aoas >= 270) & (aoas <= 360)))

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
        }

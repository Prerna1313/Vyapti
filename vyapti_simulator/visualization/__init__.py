"""
Vyapti visualization — the 12 mandatory figures.

Per frozen protocol line 181-182, every result submission must include all 12
figures. This package provides the generation functions.

Usage
-----
    from vyapti_simulator.visualization import plot_discovery_survival, plot_density_curve

    # Or generate all 12 from one experiment report:
    from vyapti_simulator.visualization import generate_all_figures
    paths = generate_all_figures(experiment_report, output_dir="figures/")
"""

from .mandatory_figures import (
    plot_discovery_survival,
    plot_deadline_curve,
    plot_density_curve,
    plot_scalability,
    plot_coverage_curve,
    plot_robustness_heatmap,
    plot_pareto_front,
    plot_generalization,
    plot_compute_time,
    plot_training_stability,
    plot_calibration,
    plot_ablation,
    generate_all_figures,
)

__all__ = [
    "plot_discovery_survival",
    "plot_deadline_curve",
    "plot_density_curve",
    "plot_scalability",
    "plot_coverage_curve",
    "plot_robustness_heatmap",
    "plot_pareto_front",
    "plot_generalization",
    "plot_compute_time",
    "plot_training_stability",
    "plot_calibration",
    "plot_ablation",
    "generate_all_figures",
]

"""PS26055 Experiments — paired-trial execution and the statistical protocol."""

from .experiment_runner import (
    ExperimentConfig,
    ExperimentRunner,
    wilcoxon_signed_rank,
    friedman_test,
    holm_bonferroni_posthoc,
    bootstrap_ci,
    kaplan_meier_curve,
    cliffs_delta,
)

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "wilcoxon_signed_rank",
    "friedman_test",
    "holm_bonferroni_posthoc",
    "bootstrap_ci",
    "kaplan_meier_curve",
    "cliffs_delta",
]

"""Evaluation metrics, baselines, and subgroup analysis."""

from .metrics import (
    exact_match_accuracy,
    mean_absolute_error,
    negative_log_likelihood,
    jensen_shannon_divergence,
    wasserstein_distance_1d,
    compute_metrics,
    compute_metrics_by_subgroup,
    fidelity_gap,
    delta_gap,
)

from .baselines import (
    UniformRandomBaseline,
    NationalMarginalBaseline,
    DemographicCellLookup,
    SupervisedClassifierBaseline,
    evaluate_all_baselines,
)

__all__ = [
    "exact_match_accuracy",
    "mean_absolute_error",
    "negative_log_likelihood",
    "jensen_shannon_divergence",
    "wasserstein_distance_1d",
    "compute_metrics",
    "compute_metrics_by_subgroup",
    "fidelity_gap",
    "delta_gap",
    "UniformRandomBaseline",
    "NationalMarginalBaseline",
    "DemographicCellLookup",
    "SupervisedClassifierBaseline",
    "evaluate_all_baselines",
]

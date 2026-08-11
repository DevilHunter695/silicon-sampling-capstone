"""Evaluation metrics for silicon sampling fidelity."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)


def exact_match_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Exact match accuracy (strict agreement)."""
    return np.mean(y_true == y_pred)


def mean_absolute_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """MAE on ordinal scale (respects ordering)."""
    return np.mean(np.abs(y_true - y_pred))


def negative_log_likelihood(
    y_true: np.ndarray, pred_probs: np.ndarray
) -> float:
    """
    Negative log-likelihood of true answers.

    Args:
        y_true: True answers (0-indexed into pred_probs columns)
        pred_probs: NxK predicted probabilities (N samples, K answer options)

    Returns: NLL (lower is better)
    """
    assert len(y_true) == len(pred_probs)
    assert np.all(y_true >= 0) and np.all(y_true < pred_probs.shape[1])

    # Extract probability of true answer for each sample
    true_probs = pred_probs[np.arange(len(y_true)), y_true]

    # Clip to avoid log(0)
    true_probs = np.clip(true_probs, 1e-9, 1.0)

    # NLL
    nll = -np.mean(np.log(true_probs))

    return nll


def jensen_shannon_divergence(
    true_dist: np.ndarray, pred_dist: np.ndarray
) -> float:
    """
    Jensen-Shannon divergence between true and predicted answer distributions.

    Lower is better. Symmetric, well-defined, and comparable to literature.
    """
    # Ensure distributions sum to 1
    true_dist = true_dist / true_dist.sum()
    pred_dist = pred_dist / pred_dist.sum()

    return jensenshannon(true_dist, pred_dist)


def wasserstein_distance_1d(
    y_true_values: np.ndarray, true_dist: np.ndarray, pred_dist: np.ndarray
) -> float:
    """
    Wasserstein-1 distance between true and predicted distributions.

    Respects ordinal ordering (unlike JSD).

    Args:
        y_true_values: The ordinal values (e.g., [1, 2, 3, 4] for 4-point Likert)
        true_dist: True probability distribution
        pred_dist: Predicted probability distribution
    """
    # Ensure distributions sum to 1
    true_dist = true_dist / true_dist.sum()
    pred_dist = pred_dist / pred_dist.sum()

    return wasserstein_distance(y_true_values, y_true_values, u_weights=true_dist, v_weights=pred_dist)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_probs: np.ndarray = None,
    answer_values: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute all fidelity metrics.

    Args:
        y_true: True answers (0-indexed)
        y_pred: Predicted answers (0-indexed)
        pred_probs: Predicted probabilities (NxK), optional for NLL
        answer_values: Ordinal values for W1, optional

    Returns: Dict with keys: accuracy, mae, nll, jsd, w1 (if applicable)
    """
    assert len(y_true) == len(y_pred)

    metrics = {}

    # Individual-level metrics
    metrics["accuracy"] = exact_match_accuracy(y_true, y_pred)
    metrics["mae"] = mean_absolute_error(y_true, y_pred)

    if pred_probs is not None:
        metrics["nll"] = negative_log_likelihood(y_true, pred_probs)

    # Distributional metrics (if we can compute)
    if pred_probs is not None:
        # True distribution
        true_counts = np.bincount(y_true, minlength=pred_probs.shape[1])
        true_dist = true_counts / true_counts.sum()

        # Predicted distribution (average of predicted probs)
        pred_dist = pred_probs.mean(axis=0)

        metrics["jsd"] = jensen_shannon_divergence(true_dist, pred_dist)

        if answer_values is not None:
            metrics["w1"] = wasserstein_distance_1d(answer_values, true_dist, pred_dist)

    return metrics


def compute_metrics_by_subgroup(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    pred_probs_col: str = None,
    subgroup_col: str = "subgroup",
) -> pd.DataFrame:
    """
    Compute metrics stratified by subgroup.

    Args:
        df: DataFrame with predictions and subgroup assignments
        y_true_col: Column name with true answers
        y_pred_col: Column name with predicted answers
        pred_probs_col: Column name with predicted probabilities (optional)
        subgroup_col: Column name with subgroup labels

    Returns: DataFrame with metrics per subgroup and sample sizes
    """
    results = []

    for subgroup_name in df[subgroup_col].unique():
        df_sub = df[df[subgroup_col] == subgroup_name]

        y_true = df_sub[y_true_col].values
        y_pred = df_sub[y_pred_col].values

        pred_probs = None
        if pred_probs_col and pred_probs_col in df_sub.columns:
            pred_probs = np.array(df_sub[pred_probs_col].tolist())

        metrics = compute_metrics(y_true, y_pred, pred_probs)
        metrics["subgroup"] = subgroup_name
        metrics["n"] = len(df_sub)

        results.append(metrics)

    return pd.DataFrame(results)


def fidelity_gap(metrics_by_subgroup: pd.DataFrame, metric_name: str = "accuracy") -> float:
    """
    Compute fidelity gap: (best) - (worst) for a metric across subgroups.

    Higher gap indicates more inequality.
    """
    return metrics_by_subgroup[metric_name].max() - metrics_by_subgroup[metric_name].min()


def delta_gap(gap_zero_shot: float, gap_fine_tuned: float) -> float:
    """
    Compute Δ_gap: gap(fine_tuned) - gap(zero_shot).

    Positive Δ_gap means fine-tuning widened the gap (hurt fairness).
    Negative Δ_gap means fine-tuning narrowed the gap (improved fairness).
    """
    return gap_fine_tuned - gap_zero_shot

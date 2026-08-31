"""
Industry-standard credit-scorecard evaluation metrics.

Generic ML metrics (AUC-ROC, RMSE, R²) tell you whether a model discriminates
at all. Real credit-risk teams also expect three metrics specific to
scorecard evaluation, none of which existed in this codebase before:

- KS-statistic: the maximum separation between the cumulative "good" and
  "bad" distributions across the score range — the traditional headline
  metric for a credit scorecard's discriminatory power.
- Gini coefficient: 2×AUC−1, rescaled to the [-1, 1] range risk teams use.
- Population Stability Index (PSI): measures how much a score's distribution
  has drifted between two samples (e.g. training vs. a later production
  batch) — the standard early-warning signal for model/feature drift.
"""

from __future__ import annotations

import numpy as np


def ks_statistic(y_true, y_score) -> float:
    """
    Kolmogorov-Smirnov statistic for a binary classifier.

    Max distance between the cumulative distribution of scores for the
    positive class and the negative class. Ranges [0, 1] — higher is more
    discriminative. Returns 0.0 if either class is empty.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) == 0:
        return 0.0

    order = np.argsort(y_score)
    y_true_sorted = y_true[order]

    n_pos = y_true_sorted.sum()
    n_neg = len(y_true_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    cum_pos = np.cumsum(y_true_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_true_sorted) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


def gini_coefficient(auc_roc: float) -> float:
    """Gini coefficient, the credit-scorecard-standard rescaling of AUC-ROC to [-1, 1]."""
    return float(2 * auc_roc - 1)


def population_stability_index(expected, actual, bins: int = 10) -> float:
    """
    Population Stability Index between an "expected" distribution (e.g. the
    training-time score distribution) and an "actual" one (e.g. a later
    production batch). Bin edges are the expected distribution's deciles by
    default.

    Conventional interpretation: PSI < 0.1 no significant shift,
    0.1-0.25 moderate shift worth investigating, > 0.25 significant drift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(expected, breakpoints)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)  # degenerate/duplicate edges would collapse bins

    if len(bin_edges) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Floor at a small epsilon so an empty bin doesn't produce log(0)/divide-by-zero.
    eps = 1e-4
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def psi_interpretation(psi: float) -> str:
    """Map a PSI value to the conventional stability label."""
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate_shift"
    return "significant_drift"

"""
Tests for the credit-scorecard-standard evaluation metrics (KS, Gini, PSI).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from app.evaluation.metrics import (
    ks_statistic,
    gini_coefficient,
    population_stability_index,
    psi_interpretation,
)


class TestKsStatistic:
    def test_perfect_separation_is_one(self):
        """A score that perfectly ranks positives above negatives has KS = 1.0."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        assert ks_statistic(y_true, y_score) == pytest.approx(1.0)

    def test_random_score_is_low(self):
        """A score uncorrelated with the label should have low KS, not near 1."""
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=2000)
        y_score = rng.random(2000)
        assert ks_statistic(y_true, y_score) < 0.15

    def test_empty_input_returns_zero(self):
        assert ks_statistic([], []) == 0.0

    def test_single_class_returns_zero(self):
        """No negatives (or no positives) present — KS is undefined, must not crash."""
        assert ks_statistic([1, 1, 1], [0.2, 0.5, 0.9]) == 0.0
        assert ks_statistic([0, 0, 0], [0.2, 0.5, 0.9]) == 0.0


class TestGiniCoefficient:
    def test_matches_auc_formula(self):
        assert gini_coefficient(0.75) == pytest.approx(0.5)
        assert gini_coefficient(0.5) == pytest.approx(0.0)
        assert gini_coefficient(1.0) == pytest.approx(1.0)


class TestPopulationStabilityIndex:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(1)
        sample = rng.normal(size=5000)
        psi = population_stability_index(sample, sample.copy())
        assert psi < 0.02

    def test_shifted_distribution_is_higher(self):
        rng = np.random.default_rng(1)
        expected = rng.normal(loc=0, size=5000)
        actual_same = rng.normal(loc=0, size=5000)
        actual_shifted = rng.normal(loc=3, size=5000)

        psi_same = population_stability_index(expected, actual_same)
        psi_shifted = population_stability_index(expected, actual_shifted)
        assert psi_shifted > psi_same

    def test_empty_input_returns_zero(self):
        assert population_stability_index([], [1, 2, 3]) == 0.0
        assert population_stability_index([1, 2, 3], []) == 0.0


class TestPsiInterpretation:
    def test_bands(self):
        assert psi_interpretation(0.05) == "stable"
        assert psi_interpretation(0.15) == "moderate_shift"
        assert psi_interpretation(0.30) == "significant_drift"

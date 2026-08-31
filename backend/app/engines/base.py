"""
Base classes for CreditSetu scoring engines.

Two shapes of engine exist in this codebase:

- TrainableEngine: wraps a fitted model (Capacity, Guardrail). Subclasses
  scikit-learn's BaseEstimator so it gets get_params()/set_params()/repr for
  free and composes naturally with sklearn tooling (GridSearchCV, pipelines,
  etc.) — the same convention any Python ML practitioner already knows,
  instead of a bespoke method-naming scheme. The fit/predict contract is the
  sklearn one; save()/load() are the model-registry hooks each subclass
  implements.

- ScoringComponent: a stateless, rule/signal-based scorer (Intent Engine).
  There's no model to fit, so forcing the sklearn fit/predict shape onto it
  would be interface conformity for its own sake — it only needs score().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


class TrainableEngine(BaseEstimator, ABC):
    """Base class for engines that fit a LightGBM model and persist it."""

    @abstractmethod
    def fit(self, X, y, **kwargs) -> dict:
        """Fit the underlying model. Returns a dict of training metrics."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the fitted model (and its metadata) to path."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously persisted model from path."""


class ScoringComponent(ABC):
    """Base class for stateless rule/signal-based scoring components."""

    @abstractmethod
    def score(self, features: dict) -> dict:
        """Score a single customer's feature dict."""

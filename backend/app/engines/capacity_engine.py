"""
Capacity Scoring Engine for CreditSetu.

Estimates repayment capacity for customers — including those with thin/no bureau file —
using behavioural transaction data features and a LightGBM regressor.

Key design decision: LightGBM natively handles missing values (NaN), so customers
without a bureau_score still get scored. This is the core value proposition —
NTC and gig-worker segments that traditional credit scoring systems can't evaluate.

The training target (true_repayment_capacity) is a known synthetic function defined
in the data generator by default. When a deployer has real historical repayment data,
fit() accepts any customers_df carrying that target column — swap the synthetic
target for actual observed repayment behaviour and retrain with no code changes.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

from .base import TrainableEngine
from .. import model_registry
from ..features.feature_engineering import ML_FEATURE_NAMES
from ..scoring_profile import ScoringProfile, default_profile


class CapacityEngine(TrainableEngine):
    """
    Predicts safe monthly repayment capacity using behavioural features.
    """

    def __init__(self, profile: Optional[ScoringProfile] = None, model_path: Optional[str] = None):
        """
        Args:
            profile: ScoringProfile supplying capacity_max_amount for score
                normalization. Defaults to the built-in demo profile.
            model_path: Path to a saved model file. If None, model must be trained.
        """
        self.profile = profile or default_profile()
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names = ML_FEATURE_NAMES
        self.model_path = model_path
        self.metadata: dict = {}
        self._is_trained = False

        if model_path and (Path(model_path).exists() or Path(model_path).with_suffix(".latest.json").exists()):
            self.load(model_path)

    def fit(
        self,
        features_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> dict:
        """
        Fit the capacity model.

        Args:
            features_df: DataFrame with engineered features (one row per customer)
            customers_df: DataFrame with customer profiles (has true_repayment_capacity)
            test_size: Fraction held out for evaluation
            seed: Random seed

        Returns:
            Dictionary of training metrics
        """
        # Merge features with ground-truth target
        merged = features_df.merge(
            customers_df[["customer_id", "true_repayment_capacity"]],
            on="customer_id",
        )

        X = merged[self.feature_names].copy()
        y = merged["true_repayment_capacity"].values.astype(float)

        # Convert boolean to int for LightGBM
        if "has_bureau_score" in X.columns:
            X["has_bureau_score"] = X["has_bureau_score"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        self.model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )

        self._is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        # AUC-ROC for "good capacity" classification (threshold at median)
        median_capacity = float(np.median(y))
        y_test_binary = (y_test > median_capacity).astype(int)
        y_pred_proba = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-10)
        try:
            auc = float(roc_auc_score(y_test_binary, y_pred_proba))
        except ValueError:
            auc = 0.5

        # Classification precision/recall on median threshold
        from sklearn.metrics import precision_score, recall_score
        y_pred_binary = (y_pred > median_capacity).astype(int)
        precision = float(precision_score(y_test_binary, y_pred_binary, zero_division=0))
        recall = float(recall_score(y_test_binary, y_pred_binary, zero_division=0))

        self._train_metrics = {
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "auc_roc": round(auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        # Kept for evaluation/metrics.py (KS-statistic, PSI) — not needed for
        # normal scoring, only for benchmark_runner.py's deeper credit-scorecard
        # metrics.
        self._last_eval = {
            "y_test_binary": y_test_binary,
            "y_test_pred_proba": y_pred_proba,
            "train_predictions": self.model.predict(X_train),
            "test_predictions": y_pred,
        }

        return self._train_metrics

    @property
    def last_eval(self) -> dict:
        """Raw arrays from the most recent fit() call's held-out evaluation."""
        if not hasattr(self, "_last_eval"):
            raise RuntimeError("No evaluation data available — call fit() first.")
        return self._last_eval

    def predict(self, features: dict) -> dict:
        """
        Predict repayment capacity for a single customer.

        Args:
            features: Feature dictionary from feature_engineering.py

        Returns:
            Dictionary with:
            - capacity_amount: predicted safe monthly repayment
            - capacity_score: normalized to [0, 1]
            - capacity_confidence: confidence band width
        """
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")

        X = self._prepare_features(features)
        prediction = float(self.model.predict(X)[0])

        # Ensure non-negative
        prediction = max(prediction, 0.0)

        # Get prediction spread from individual trees for confidence estimate
        tree_preds = []
        for tree_idx in range(self.model.n_estimators_):
            try:
                pred = self.model.predict(X, start_iteration=tree_idx, num_iteration=1)
                tree_preds.append(float(pred[0]))
            except Exception:
                break

        if tree_preds:
            confidence = float(np.std(tree_preds))
        else:
            confidence = prediction * 0.15

        capacity_score = min(prediction / self.profile.capacity_max_amount, 1.0)

        return {
            "capacity_amount": round(prediction, 0),
            "capacity_score": round(capacity_score, 4),
            "capacity_confidence": round(confidence, 0),
        }

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict for multiple customers in one vectorized pass — builds the
        feature matrix once and calls the model once per tree-slice instead of
        once per customer, which is what makes this viable at real volume
        (hundreds of thousands of rows) rather than one Python-level model
        call per row.
        """
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")

        X = self._prepare_features_batch(features_df)
        predictions = np.clip(self.model.predict(X), 0.0, None)

        tree_preds = []
        for tree_idx in range(self.model.n_estimators_):
            try:
                tree_preds.append(self.model.predict(X, start_iteration=tree_idx, num_iteration=1))
            except Exception:
                break

        if tree_preds:
            confidence = np.std(np.vstack(tree_preds), axis=0)
        else:
            confidence = predictions * 0.15

        capacity_score = np.minimum(predictions / self.profile.capacity_max_amount, 1.0)

        return pd.DataFrame({
            "customer_id": features_df["customer_id"].values,
            "capacity_amount": np.round(predictions, 0),
            "capacity_score": np.round(capacity_score, 4),
            "capacity_confidence": np.round(confidence, 0),
        })

    def save(self, path: str) -> None:
        """Save the fitted model to the versioned model registry."""
        if self.model is None:
            raise RuntimeError("No model to save")
        metadata = {
            "engine": "CapacityEngine",
            "profile_name": self.profile.name,
            "feature_names": self.feature_names,
            "metrics": getattr(self, "_train_metrics", {}),
        }
        model_registry.save(self.model, path, metadata)

    def load(self, path: str) -> None:
        """Load the latest registered model version from path."""
        self.model, self.metadata = model_registry.load(path)
        self._is_trained = True

    def _prepare_features(self, features: dict) -> pd.DataFrame:
        """Convert feature dict to model input format."""
        row = {}
        for name in self.feature_names:
            val = features.get(name)
            if name == "has_bureau_score":
                row[name] = int(bool(val))
            elif val is None:
                row[name] = np.nan
            else:
                row[name] = float(val)
        return pd.DataFrame([row])

    def _prepare_features_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized equivalent of _prepare_features for a whole DataFrame."""
        X = pd.DataFrame(index=features_df.index)
        for name in self.feature_names:
            col = features_df[name] if name in features_df.columns else pd.Series(np.nan, index=features_df.index)
            if name == "has_bureau_score":
                X[name] = col.fillna(False).astype(bool).astype(int)
            else:
                X[name] = pd.to_numeric(col, errors="coerce")
        return X

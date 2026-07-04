"""
Capacity Scoring Engine for CreditSetu.

Estimates repayment capacity for customers — including those with thin/no bureau file —
using behavioural transaction data features and a LightGBM regressor.

Key design decision: LightGBM natively handles missing values (NaN), so customers
without a bureau_score still get scored. This is the core value proposition —
NTC and gig-worker segments that traditional credit scoring systems can't evaluate.

The training target (true_repayment_capacity) is a known synthetic function defined
in the data generator. In production, this would be replaced by actual observed
repayment behaviour (e.g., on-time repayment rates on existing loans).
"""

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

from ..features.feature_engineering import ML_FEATURE_NAMES


class CapacityEngine:
    """
    Predicts safe monthly repayment capacity using behavioural features.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to a saved model file. If None, model must be trained.
        """
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names = ML_FEATURE_NAMES
        self.model_path = model_path
        self._is_trained = False

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(
        self,
        features_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> dict:
        """
        Train the capacity model on synthetic data.

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

        self._train_metrics = {
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "auc_roc": round(auc, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        return self._train_metrics

    def predict(self, features: dict) -> dict:
        """
        Predict repayment capacity for a single customer.

        Args:
            features: Feature dictionary from feature_engineering.py

        Returns:
            Dictionary with:
            - capacity_amount: predicted safe monthly repayment in INR
            - capacity_score: normalized to [0, 1]
            - capacity_confidence: confidence band width
        """
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X = self._prepare_features(features)
        prediction = float(self.model.predict(X)[0])

        # Ensure non-negative
        prediction = max(prediction, 0.0)

        # Get prediction spread from individual trees for confidence estimate
        # LightGBM's raw predictions from each tree
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

        # Normalize to [0, 1] score
        # Use empirical max from training distribution
        max_capacity = 50000  # reasonable max for Indian retail lending context
        capacity_score = min(prediction / max_capacity, 1.0)

        return {
            "capacity_amount": round(prediction, 0),
            "capacity_score": round(capacity_score, 4),
            "capacity_confidence": round(confidence, 0),
        }

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict for multiple customers."""
        results = []
        for _, row in features_df.iterrows():
            result = self.predict(row.to_dict())
            result["customer_id"] = row["customer_id"]
            results.append(result)
        return pd.DataFrame(results)

    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        """Load trained model from disk."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
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

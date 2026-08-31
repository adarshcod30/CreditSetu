"""
Guardrail Engine for CreditSetu.

Flags and suppresses over-leveraged or repayment-stressed customers before they're
shown as leads. Implements a hybrid approach:

1. Hard rules: Clear-cut over-leverage indicators that ALWAYS trigger, regardless
   of what the ML model says. Thresholds live on the ScoringProfile passed in at
   construction time, not as module-level constants — a deployer changes policy
   by swapping the profile, not by editing this file.
2. Soft ML classifier: LightGBM classifier trained to catch borderline/softer
   cases that the hard rules miss.

Output: Risk tier (Safe / Watch / Suppress) with specific triggered reasons.

Validation caveat: the default fit() target ("is_stressed") is synthetically
generated from the same features used to train on, so its reported AUC partly
reflects the model re-learning its own generator rather than real-world
predictive power. fit() accepts any customers_df carrying that target column —
a deployer with real historical default/delinquency outcomes retrains against
those instead, with no code changes.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
)

from .base import TrainableEngine
from .. import model_registry
from ..features.feature_engineering import ML_FEATURE_NAMES
from ..scoring_profile import ScoringProfile, default_profile


class GuardrailEngine(TrainableEngine):
    """
    Risk assessment engine that determines whether a customer should be
    surfaced as a lead or suppressed due to over-leverage risk.
    """

    def __init__(self, profile: Optional[ScoringProfile] = None, model_path: Optional[str] = None):
        self.profile = profile or default_profile()
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names = ML_FEATURE_NAMES
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
        Fit the guardrail classifier.

        By default the training target is a synthetic "is_stressed" label
        derived from persona type and features (see module docstring). Pass a
        customers_df with a real "is_stressed" column (derived from actual
        default/delinquency outcomes) to train on real data instead.
        """
        merged = features_df.merge(
            customers_df[["customer_id", "persona_type"]],
            on="customer_id",
        )

        if "is_stressed" in customers_df.columns:
            merged = merged.merge(customers_df[["customer_id", "is_stressed"]], on="customer_id")
        else:
            # Logit-based synthetic credit stress risk probability formula.
            # Coefficients are calibrated against real elasticities measured on
            # Kaggle's "Give Me Some Credit" dataset (150K real borrowers, real
            # 2-year default outcomes: https://www.openml.org/search?type=data&id=45577) —
            # not hand-guessed. On that dataset: overall default rate is ~7%;
            # any delinquency history raises the default rate ~7.5x (3.0% -> 22.1%);
            # debt-ratio deciles show default rate roughly doubling from the
            # lowest to highest bucket (5.5% -> 13.1%), a much more moderate
            # effect than a naive assumption would suggest.
            logit = -2.6  # baseline logit for ~7% default rate, matching the real anchor
            logit += merged["concurrent_lender_count"] * 0.7
            logit += (merged["emi_to_inflow_ratio"] - 0.15) * 2.2  # softened: real debt-ratio effect is ~2x, not 10x+
            logit += merged["nach_bounce_count_6m"] * 2.1  # any delinquency ~7.5x's real odds; strengthened from prior estimate
            logit += (1.0 - merged["rent_consistency"]) * 0.8
            logit += merged["income_cv"] * 1.5
            logit = np.where(merged["persona_type"] == "over_leveraged", logit + 2.5, logit)

            prob = 1.0 / (1.0 + np.exp(-logit))
            rng = np.random.default_rng(seed)
            merged["is_stressed"] = (rng.random(size=len(prob)) < prob).astype(int)

        X = merged[self.feature_names].copy()
        y = merged["is_stressed"].values

        if "has_bureau_score" in X.columns:
            X["has_bureau_score"] = X["has_bureau_score"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y,
        )

        self.model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=20,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,  # handle class imbalance
            random_state=seed,
            verbose=-1,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )

        self._is_trained = True

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        try:
            auc = float(roc_auc_score(y_test, y_pred_proba))
        except ValueError:
            auc = 0.5

        self._train_metrics = {
            "auc_roc": round(auc, 4),
            "false_positive_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
            "false_negative_rate": round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4),
            "precision": round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4),
            "recall": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
            "n_stressed": int(y.sum()),
            "n_safe": int(len(y) - y.sum()),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        # Kept for evaluation/metrics.py (KS-statistic, PSI) — not needed for
        # normal scoring, only for benchmark_runner.py's deeper credit-scorecard
        # metrics.
        self._last_eval = {
            "y_test": y_test,
            "y_test_pred_proba": y_pred_proba,
            "train_pred_proba": self.model.predict_proba(X_train)[:, 1],
            "test_pred_proba": y_pred_proba,
        }

        return self._train_metrics

    @property
    def last_eval(self) -> dict:
        """Raw arrays from the most recent fit() call's held-out evaluation."""
        if not hasattr(self, "_last_eval"):
            raise RuntimeError("No evaluation data available — call fit() first.")
        return self._last_eval

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Sklearn-style raw stress probability for a prepared feature matrix."""
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, features: dict, ml_proba: Optional[float] = None) -> dict:
        """
        Evaluate a single customer through the guardrail.

        Hard rules are checked FIRST — they override the ML model.
        The ML model catches softer/borderline cases.

        Args:
            features: Feature dictionary
            ml_proba: Precomputed stress probability (used by evaluate_batch
                to avoid re-running the model once per row). Computed here if
                omitted.

        Returns:
            Dictionary with guardrail_tier, guardrail_score, guardrail_reasons.
        """
        hard_suppress, reasons = self._check_hard_rules(features)
        if hard_suppress:
            return {
                "guardrail_tier": "Suppress",
                "guardrail_score": 1.0,
                "guardrail_reasons": reasons,
            }

        if ml_proba is None:
            if self._is_trained and self.model is not None:
                X = self._prepare_features(features)
                ml_proba = float(self.predict_proba(X)[0])
            else:
                ml_proba = self._heuristic_risk_score(features)

        g = self.profile.guardrail
        if ml_proba > g.suppress_threshold:
            tier = "Suppress"
            reasons.append(f"ML risk model flagged elevated stress risk (score: {ml_proba:.2f})")
        elif ml_proba > g.watch_threshold:
            tier = "Watch"
            if features.get("emi_to_inflow_ratio", 0) > 0.40:
                reasons.append(f"Moderate EMI burden: {features['emi_to_inflow_ratio']:.1%}")
            if features.get("emi_to_inflow_trend", 0) > 0.05:
                reasons.append("Rising EMI-to-income trend over last 3 months")
            if features.get("nach_bounce_count_6m", 0) > 0:
                reasons.append(f"NACH bounce(s) in trailing 6 months: {features['nach_bounce_count_6m']}")
            if not reasons:
                reasons.append(f"Borderline risk indicators (ML score: {ml_proba:.2f})")
        else:
            tier = "Safe"

        return {
            "guardrail_tier": tier,
            "guardrail_score": round(ml_proba, 4),
            "guardrail_reasons": reasons,
        }

    def evaluate_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate multiple customers. The model is called once, vectorized
        across the whole batch — the per-row loop below only builds tier/
        reason dictionaries from that precomputed probability, not a model
        call per customer, which is what makes this viable at real volume.
        """
        if self._is_trained and self.model is not None:
            X = self._prepare_features_batch(features_df)
            ml_probas = self.predict_proba(X)
        else:
            ml_probas = [None] * len(features_df)

        results = []
        for i, (_, row) in enumerate(features_df.iterrows()):
            features = row.to_dict()
            proba = float(ml_probas[i]) if ml_probas[i] is not None else None
            result = self.evaluate(features, ml_proba=proba)
            result["customer_id"] = features["customer_id"]
            results.append(result)
        return pd.DataFrame(results)

    def save(self, path: str) -> None:
        """Save the fitted model to the versioned model registry."""
        if self.model is None:
            raise RuntimeError("No model to save")
        metadata = {
            "engine": "GuardrailEngine",
            "profile_name": self.profile.name,
            "feature_names": self.feature_names,
            "metrics": getattr(self, "_train_metrics", {}),
        }
        model_registry.save(self.model, path, metadata)

    def load(self, path: str) -> None:
        """Load the latest registered model version from path."""
        self.model, self.metadata = model_registry.load(path)
        self._is_trained = True

    def _check_hard_rules(self, features: dict) -> tuple[bool, list[str]]:
        """Evaluate the non-negotiable hard rules. Returns (suppress?, reasons)."""
        g = self.profile.guardrail
        reasons = []

        concurrent_lenders = features.get("concurrent_lender_count", 0) or 0
        if concurrent_lenders >= g.max_concurrent_lenders:
            reasons.append(
                f"High concurrent lender count: {concurrent_lenders} active EMIs "
                f"(threshold: {g.max_concurrent_lenders})"
            )

        emi_ratio = features.get("emi_to_inflow_ratio", 0) or 0
        if emi_ratio > g.max_emi_to_inflow_ratio:
            reasons.append(
                f"EMI-to-income ratio too high: {emi_ratio:.1%} "
                f"(threshold: {g.max_emi_to_inflow_ratio:.0%})"
            )

        bounces_3m = features.get("nach_bounce_count_3m", 0) or 0
        if bounces_3m >= g.max_nach_bounces_3m:
            reasons.append(
                f"NACH bounce detected in trailing 3 months: {bounces_3m} bounce(s)"
            )

        return (len(reasons) > 0, reasons)

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

    def _heuristic_risk_score(self, features: dict) -> float:
        """Fallback heuristic when ML model isn't available."""
        score = 0.0
        score += min(features.get("concurrent_lender_count", 0) / 6, 1.0) * 0.3
        score += min(features.get("emi_to_inflow_ratio", 0) / 0.8, 1.0) * 0.3
        score += min(features.get("nach_bounce_count_6m", 0) / 4, 1.0) * 0.25
        if features.get("emi_to_inflow_trend", 0) > 0:
            score += 0.15
        return min(score, 1.0)

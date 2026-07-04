"""
Guardrail Engine for CreditSetu.

Flags and suppresses over-leveraged or repayment-stressed customers before they're
shown as leads. Implements a hybrid approach:

1. Hard rules: Clear-cut over-leverage indicators that ALWAYS trigger, regardless
   of what the ML model says. These represent non-negotiable risk thresholds.
2. Soft ML classifier: LightGBM classifier trained to catch borderline/softer
   cases that the hard rules miss.

Output: Risk tier (Safe / Watch / Suppress) with specific triggered reasons.
"""

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from ..features.feature_engineering import ML_FEATURE_NAMES


# ─── Hard Rule Thresholds ────────────────────────────────────────────────────────
# These are non-negotiable. A customer matching ANY of these is Suppressed.
HARD_RULES = {
    "max_concurrent_lenders": 5,          # >= 5 concurrent EMI counterparties
    "max_emi_to_inflow_ratio": 0.60,      # EMI burden exceeds 60% of income
    "max_nach_bounces_3m": 1,             # Any bounce in trailing 3 months
}

# Soft ML thresholds
WATCH_THRESHOLD = 0.3    # ML probability above this → Watch tier
SUPPRESS_THRESHOLD = 0.6  # ML probability above this → Suppress tier


class GuardrailEngine:
    """
    Risk assessment engine that determines whether a customer should be
    surfaced as a lead or suppressed due to over-leverage risk.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names = ML_FEATURE_NAMES
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
        Train the guardrail classifier.

        The training target is a synthetic "is_stressed" label derived from
        persona type and features. In production, this would be derived from
        actual default/delinquency data.
        """
        # Merge features with customer persona details
        merged = features_df.merge(
            customers_df[["customer_id", "persona_type"]],
            on="customer_id",
        )

        # Logit-based credit stress risk probability formula
        # Baseline stress logit represents ~3% default rate
        logit = -3.2
        logit += merged["concurrent_lender_count"] * 0.7
        logit += (merged["emi_to_inflow_ratio"] - 0.15) * 6.5
        logit += merged["nach_bounce_count_6m"] * 1.6
        # Inconsistent rent increases default odds
        logit += (1.0 - merged["rent_consistency"]) * 0.8
        # High income variability increases default odds
        logit += merged["income_cv"] * 1.5

        # Over-leveraged persona gets a heavy default odds multiplier
        logit = np.where(merged["persona_type"] == "over_leveraged", logit + 2.5, logit)

        # Calculate sigmoid probability
        prob = 1.0 / (1.0 + np.exp(-logit))

        # Bernoulli trial with random generator to draw stress label (adds realistic noise)
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

        # Evaluate
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
        }

        return self._train_metrics

    def evaluate(self, features: dict) -> dict:
        """
        Evaluate a single customer through the guardrail.

        Hard rules are checked FIRST — they override the ML model.
        The ML model catches softer/borderline cases.

        Args:
            features: Feature dictionary

        Returns:
            Dictionary with:
            - guardrail_tier: "Safe" | "Watch" | "Suppress"
            - guardrail_score: float [0, 1] (higher = more risky)
            - guardrail_reasons: list of triggered reason strings
        """
        reasons = []
        hard_suppress = False

        # ─── Hard Rules (always checked, override ML) ────────────────
        concurrent_lenders = features.get("concurrent_lender_count", 0)
        if concurrent_lenders >= HARD_RULES["max_concurrent_lenders"]:
            reasons.append(
                f"High concurrent lender count: {concurrent_lenders} active EMIs "
                f"(threshold: {HARD_RULES['max_concurrent_lenders']})"
            )
            hard_suppress = True

        emi_ratio = features.get("emi_to_inflow_ratio", 0)
        if emi_ratio > HARD_RULES["max_emi_to_inflow_ratio"]:
            reasons.append(
                f"EMI-to-income ratio too high: {emi_ratio:.1%} "
                f"(threshold: {HARD_RULES['max_emi_to_inflow_ratio']:.0%})"
            )
            hard_suppress = True

        bounces_3m = features.get("nach_bounce_count_3m", 0)
        if bounces_3m >= HARD_RULES["max_nach_bounces_3m"]:
            reasons.append(
                f"NACH bounce detected in trailing 3 months: {bounces_3m} bounce(s)"
            )
            hard_suppress = True

        if hard_suppress:
            return {
                "guardrail_tier": "Suppress",
                "guardrail_score": 1.0,
                "guardrail_reasons": reasons,
            }

        # ─── Soft ML Classification ──────────────────────────────────
        if self._is_trained and self.model is not None:
            X = self._prepare_features(features)
            ml_proba = float(self.model.predict_proba(X)[0, 1])
        else:
            # Fallback heuristic if model not trained
            ml_proba = self._heuristic_risk_score(features)

        # Determine tier based on ML probability
        if ml_proba > SUPPRESS_THRESHOLD:
            tier = "Suppress"
            reasons.append(
                f"ML risk model flagged elevated stress risk (score: {ml_proba:.2f})"
            )
        elif ml_proba > WATCH_THRESHOLD:
            tier = "Watch"
            # Add specific borderline reasons
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
        """Evaluate multiple customers."""
        results = []
        for _, row in features_df.iterrows():
            result = self.evaluate(row.to_dict())
            result["customer_id"] = row["customer_id"]
            results.append(result)
        return pd.DataFrame(results)

    def save(self, path: str) -> None:
        """Save trained model."""
        if self.model is None:
            raise RuntimeError("No model to save")
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        """Load trained model."""
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

    def _heuristic_risk_score(self, features: dict) -> float:
        """Fallback heuristic when ML model isn't available."""
        score = 0.0
        score += min(features.get("concurrent_lender_count", 0) / 6, 1.0) * 0.3
        score += min(features.get("emi_to_inflow_ratio", 0) / 0.8, 1.0) * 0.3
        score += min(features.get("nach_bounce_count_6m", 0) / 4, 1.0) * 0.25
        if features.get("emi_to_inflow_trend", 0) > 0:
            score += 0.15
        return min(score, 1.0)

"""
Composite Scorer for CreditSetu.

Combines all three engines (Intent, Capacity, Guardrail) into a single
ranked lead score with product suggestion and plain-language explanation.

A lead only surfaces to the dashboard if Guardrail tier != "Suppress".

Weights, currency, and the product decision table all come from the
ScoringProfile passed at construction time rather than being hardcoded here —
a deployer changes lending policy by swapping the profile, not by editing
this file. Defaults preserve the original demo behavior (Intent 0.4 /
Capacity 0.6, ₹ currency, the original Home/Auto/Personal Loan tiers).
"""

from typing import Optional

import numpy as np
import pandas as pd

from .intent_engine import IntentEngine
from .capacity_engine import CapacityEngine
from .guardrail_engine import GuardrailEngine
from ..scoring_profile import ScoringProfile, default_profile


class CompositeScorer:
    """
    Combines Intent, Capacity, and Guardrail engines into a final lead score.
    """

    def __init__(
        self,
        intent_engine: IntentEngine,
        capacity_engine: CapacityEngine,
        guardrail_engine: GuardrailEngine,
        profile: Optional[ScoringProfile] = None,
        intent_weight: Optional[float] = None,
        capacity_weight: Optional[float] = None,
    ):
        self.intent_engine = intent_engine
        self.capacity_engine = capacity_engine
        self.guardrail_engine = guardrail_engine
        self.profile = profile or default_profile()
        self.intent_weight = intent_weight if intent_weight is not None else self.profile.intent_weight
        self.capacity_weight = capacity_weight if capacity_weight is not None else self.profile.capacity_weight

    def score(self, features: dict) -> dict:
        """
        Compute composite score for a single customer.

        Args:
            features: Feature dictionary from feature_engineering.py

        Returns:
            Dictionary with all sub-scores, composite score, tier, explanation,
            and suggested product.
        """
        # Run all three engines
        intent_result = self.intent_engine.score(features)
        capacity_result = self.capacity_engine.predict(features)
        guardrail_result = self.guardrail_engine.evaluate(features)

        # Composite score = weighted combination of intent and capacity
        intent_score = intent_result["intent_score"]
        capacity_score = capacity_result["capacity_score"]
        composite = (
            self.intent_weight * intent_score +
            self.capacity_weight * capacity_score
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        # If suppressed, composite stays for ranking transparency but lead is excluded
        is_qualified_lead = guardrail_result["guardrail_tier"] != "Suppress"

        # Suggest product based on capacity, income, and persona features
        suggested_product = self.profile.select_product(
            capacity_amount=capacity_result["capacity_amount"],
            income_mean=features.get("income_mean", 0) or 0,
            has_bureau_score=bool(features.get("has_bureau_score", True)),
            gig_pattern_score=features.get("gig_pattern_score", 0.0) or 0.0,
        )

        # Build explanation (will be enhanced by SHAP in shap_explainer.py)
        explanation = self._build_explanation(
            intent_result, capacity_result, guardrail_result, features
        )

        return {
            "composite_score": round(composite, 4),
            "intent_score": intent_result["intent_score"],
            "intent_event_type": intent_result["intent_event_type"],
            "intent_event_recency_days": intent_result["intent_event_recency_days"],
            "intent_details": intent_result["intent_details"],
            "capacity_score": capacity_result["capacity_score"],
            "capacity_amount": capacity_result["capacity_amount"],
            "capacity_confidence": capacity_result["capacity_confidence"],
            "guardrail_score": guardrail_result["guardrail_score"],
            "guardrail_tier": guardrail_result["guardrail_tier"],
            "guardrail_reasons": guardrail_result["guardrail_reasons"],
            "is_qualified_lead": is_qualified_lead,
            "suggested_product": suggested_product,
            "explanation": explanation,
        }

    def score_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score multiple customers and return ranked results.

        Runs each sub-engine's own vectorized batch method once across the
        whole DataFrame, then combines results row-wise — avoids re-running
        the intent/capacity/guardrail engines' expensive parts once per row.
        """
        intent_df = self.intent_engine.score_batch(features_df)
        capacity_df = self.capacity_engine.predict_batch(features_df)
        guardrail_df = self.guardrail_engine.evaluate_batch(features_df)

        merged = features_df[["customer_id"]].merge(
            intent_df, on="customer_id"
        ).merge(
            capacity_df, on="customer_id"
        ).merge(
            guardrail_df, on="customer_id"
        )

        composite = (
            self.intent_weight * merged["intent_score"] +
            self.capacity_weight * merged["capacity_score"]
        ).clip(0.0, 1.0)
        merged["composite_score"] = composite.round(4)
        merged["is_qualified_lead"] = merged["guardrail_tier"] != "Suppress"

        features_by_id = features_df.set_index("customer_id")
        suggested_products = []
        explanations = []
        for _, row in merged.iterrows():
            features = features_by_id.loc[row["customer_id"]].to_dict()
            features["customer_id"] = row["customer_id"]
            suggested_products.append(self.profile.select_product(
                capacity_amount=row["capacity_amount"],
                income_mean=features.get("income_mean", 0) or 0,
                has_bureau_score=bool(features.get("has_bureau_score", True)),
                gig_pattern_score=features.get("gig_pattern_score", 0.0) or 0.0,
            ))
            explanations.append(self._build_explanation(
                {
                    "intent_score": row["intent_score"],
                    "intent_event_type": row["intent_event_type"],
                    "intent_event_recency_days": row["intent_event_recency_days"],
                },
                {"capacity_amount": row["capacity_amount"], "capacity_score": row["capacity_score"]},
                {"guardrail_tier": row["guardrail_tier"], "guardrail_reasons": row["guardrail_reasons"]},
                features,
            ))
        merged["suggested_product"] = suggested_products
        merged["explanation"] = explanations

        return merged.sort_values("composite_score", ascending=False).reset_index(drop=True)

    def _build_explanation(
        self,
        intent_result: dict,
        capacity_result: dict,
        guardrail_result: dict,
        features: dict,
    ) -> str:
        """
        Build a plain-language explanation of the score.

        This is a basic template-based approach. The SHAP explainer
        enriches this with feature-level attribution in shap_explainer.py.
        """
        parts = []
        currency = self.profile.currency_symbol

        # Capacity explanation
        cap_amount = capacity_result["capacity_amount"]
        if capacity_result["capacity_score"] > 0.6:
            parts.append(f"Strong repayment capacity ({currency}{cap_amount:,.0f}/month estimated)")
        elif capacity_result["capacity_score"] > 0.3:
            parts.append(f"Moderate repayment capacity ({currency}{cap_amount:,.0f}/month estimated)")
        else:
            parts.append(f"Limited repayment capacity ({currency}{cap_amount:,.0f}/month estimated)")

        # Key drivers
        drivers = []
        income_cv = features.get("income_cv", 1.0)
        if income_cv < 0.15:
            drivers.append("stable income pattern")
        elif income_cv > 0.5:
            drivers.append("variable income pattern")

        rent_consistency = features.get("rent_consistency", 0)
        if rent_consistency > 0.7:
            drivers.append("consistent rent payments")

        gig_score = features.get("gig_pattern_score", 0)
        if gig_score > 0.6:
            drivers.append("gig-economy income profile")

        if not features.get("has_bureau_score"):
            drivers.append("thin/no bureau file — scored via transaction behaviour")

        if drivers:
            parts.append(f"Driven by: {', '.join(drivers)}")

        # Intent explanation
        if intent_result["intent_score"] > 0:
            event_type = intent_result["intent_event_type"]
            days = intent_result["intent_event_recency_days"]
            event_descriptions = {
                "emi_closure": "Recent EMI closure",
                "income_step_up": "Recent income increase",
                "positive_shift": "Positive cash flow shift",
                "new_commitment": "New financial commitment",
                "negative_shift": "Negative cash flow shift",
            }
            event_desc = event_descriptions.get(event_type, event_type)
            parts.append(f"{event_desc} detected {days} days ago")

        # Guardrail explanation
        if guardrail_result["guardrail_tier"] == "Suppress":
            parts.append(f"⚠ SUPPRESSED: {'; '.join(guardrail_result['guardrail_reasons'])}")
        elif guardrail_result["guardrail_tier"] == "Watch":
            parts.append(f"⚡ Watch: {'; '.join(guardrail_result['guardrail_reasons'])}")

        return ". ".join(parts) + "."

"""
Composite Scorer for CreditSetu.

Combines all three engines (Intent, Capacity, Guardrail) into a single
ranked lead score with product suggestion and plain-language explanation.

A lead only surfaces to the dashboard if Guardrail tier != "Suppress".

Default weights: Intent (0.4) + Capacity (0.6). These are configurable
because different bank strategies might weight intent-readiness vs.
capacity differently. The 60/40 default reflects the idea that repayment
capacity is a more fundamental lending signal, but intent-timing matters
for conversion optimization.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .intent_engine import IntentEngine
from .capacity_engine import CapacityEngine
from .guardrail_engine import GuardrailEngine


# Default scoring weights — configurable
DEFAULT_INTENT_WEIGHT = 0.40
DEFAULT_CAPACITY_WEIGHT = 0.60

# Product suggestion thresholds
PRODUCT_RULES = {
    "home_loan": {"min_capacity_amount": 25000, "min_income_mean": 60000},
    "auto_loan": {"min_capacity_amount": 15000, "min_income_mean": 40000},
    "personal_loan": {"min_capacity_amount": 5000, "min_income_mean": 15000},
}


class CompositeScorer:
    """
    Combines Intent, Capacity, and Guardrail engines into a final lead score.
    """

    def __init__(
        self,
        intent_engine: IntentEngine,
        capacity_engine: CapacityEngine,
        guardrail_engine: GuardrailEngine,
        intent_weight: float = DEFAULT_INTENT_WEIGHT,
        capacity_weight: float = DEFAULT_CAPACITY_WEIGHT,
    ):
        self.intent_engine = intent_engine
        self.capacity_engine = capacity_engine
        self.guardrail_engine = guardrail_engine
        self.intent_weight = intent_weight
        self.capacity_weight = capacity_weight

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
        suggested_product = self._suggest_product(
            capacity_result["capacity_amount"],
            features.get("income_mean", 0),
            features,
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
        """Score multiple customers and return ranked results."""
        results = []
        for _, row in features_df.iterrows():
            result = self.score(row.to_dict())
            result["customer_id"] = row["customer_id"]
            results.append(result)

        df = pd.DataFrame(results)
        # Sort by composite score descending
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        return df

    def _suggest_product(self, capacity_amount: float, income_mean: float, features: dict) -> str:
        """Suggest the most appropriate loan product based on capacity, income, and persona."""
        has_bureau = features.get("has_bureau_score", True)
        gig_score = features.get("gig_pattern_score", 0.0)

        # Segment 1: Gig Workers & New-to-Credit (micro products)
        if not has_bureau or gig_score > 0.40:
            if capacity_amount >= 12000:
                return "Personal Loan"
            elif capacity_amount >= 5000:
                return "Micro-Credit Line"
            else:
                return "Retail Credit Card"

        # Segment 2: Standard Borrowers (Auto / Home / Personal Loan)
        if capacity_amount >= 32000 and income_mean >= 75000:
            return "Home Loan"

        if capacity_amount >= 16000 and income_mean >= 40000:
            return "Auto Loan"

        if capacity_amount >= 5000:
            return "Personal Loan"

        return "Retail Credit Card"  # low capacity default

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

        # Capacity explanation
        cap_amount = capacity_result["capacity_amount"]
        if capacity_result["capacity_score"] > 0.6:
            parts.append(f"Strong repayment capacity (₹{cap_amount:,.0f}/month estimated)")
        elif capacity_result["capacity_score"] > 0.3:
            parts.append(f"Moderate repayment capacity (₹{cap_amount:,.0f}/month estimated)")
        else:
            parts.append(f"Limited repayment capacity (₹{cap_amount:,.0f}/month estimated)")

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

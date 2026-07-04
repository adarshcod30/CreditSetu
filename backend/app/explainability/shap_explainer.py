"""
SHAP-based explainability module for CreditSetu.

Uses SHAP TreeExplainer on the Capacity and Guardrail LightGBM models to provide
per-customer feature attribution. Converts top-3 SHAP features into
plain-language explanations using template-based NLG.

Every API response for a scored customer includes this explainability data.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not installed — explainability will use fallback descriptions")

from ..features.feature_engineering import ML_FEATURE_NAMES


# Human-readable feature name mappings
FEATURE_DISPLAY_NAMES = {
    "income_mean": "Monthly Income",
    "income_cv": "Income Stability",
    "income_timing_cv": "Income Regularity",
    "gig_pattern_score": "Gig Payout Pattern",
    "emi_to_inflow_ratio": "EMI-to-Income Ratio",
    "emi_to_inflow_trend": "EMI Trend (3m)",
    "concurrent_lender_count": "Active Lenders",
    "nach_bounce_count_6m": "NACH Bounces (6m)",
    "nach_bounce_count_3m": "NACH Bounces (3m)",
    "rent_consistency": "Rent Consistency",
    "merchant_category_entropy": "Spending Diversity",
    "monthly_surplus": "Cash Surplus",
    "has_bureau_score": "Bureau File Status",
    "bureau_score": "Bureau Score",
}

# Templates for positive/negative contribution
CONTRIBUTION_TEMPLATES = {
    "income_mean": {
        "positive": "strong monthly income of ₹{value:,.0f}",
        "negative": "limited monthly income of ₹{value:,.0f}",
    },
    "income_cv": {
        "positive": "highly stable income pattern (CV: {value:.2f})",
        "negative": "variable income pattern (CV: {value:.2f})",
    },
    "income_timing_cv": {
        "positive": "regular income timing",
        "negative": "irregular income timing",
    },
    "gig_pattern_score": {
        "positive": "strong gig-economy income pattern",
        "negative": "limited gig income pattern",
    },
    "emi_to_inflow_ratio": {
        "positive": "low EMI burden ({value:.0%} of income)",
        "negative": "high EMI burden ({value:.0%} of income)",
    },
    "emi_to_inflow_trend": {
        "positive": "declining EMI burden trend",
        "negative": "rising EMI burden trend",
    },
    "concurrent_lender_count": {
        "positive": "few active lenders ({value:.0f})",
        "negative": "multiple active lenders ({value:.0f})",
    },
    "nach_bounce_count_6m": {
        "positive": "no NACH bounces",
        "negative": "{value:.0f} NACH bounce(s) in 6 months",
    },
    "nach_bounce_count_3m": {
        "positive": "clean repayment in recent 3 months",
        "negative": "{value:.0f} NACH bounce(s) in 3 months",
    },
    "rent_consistency": {
        "positive": "consistent rent payments (score: {value:.2f})",
        "negative": "inconsistent or no rent payments",
    },
    "merchant_category_entropy": {
        "positive": "diverse spending across categories",
        "negative": "concentrated spending pattern",
    },
    "monthly_surplus": {
        "positive": "healthy monthly surplus of ₹{value:,.0f}",
        "negative": "limited monthly surplus of ₹{value:,.0f}",
    },
    "has_bureau_score": {
        "positive": "bureau credit history available",
        "negative": "no bureau history (thin file — scored via transaction data)",
    },
    "bureau_score": {
        "positive": "good bureau score ({value:.0f})",
        "negative": "low bureau score ({value:.0f})",
    },
}


class ShapExplainer:
    """
    Generates SHAP-based feature explanations for Capacity and Guardrail models.
    """

    def __init__(
        self,
        capacity_model=None,
        guardrail_model=None,
    ):
        self.capacity_model = capacity_model
        self.guardrail_model = guardrail_model
        self.capacity_explainer = None
        self.guardrail_explainer = None

        if HAS_SHAP:
            self._init_explainers()

    def _init_explainers(self):
        """Initialize SHAP TreeExplainers for both models."""
        if self.capacity_model is not None:
            try:
                self.capacity_explainer = shap.TreeExplainer(self.capacity_model)
            except Exception as e:
                warnings.warn(f"Could not create capacity SHAP explainer: {e}")

        if self.guardrail_model is not None:
            try:
                self.guardrail_explainer = shap.TreeExplainer(self.guardrail_model)
            except Exception as e:
                warnings.warn(f"Could not create guardrail SHAP explainer: {e}")

    def explain(self, features: dict, model_type: str = "capacity") -> dict:
        """
        Generate SHAP explanation for a single customer.

        Args:
            features: Feature dictionary
            model_type: "capacity" or "guardrail"

        Returns:
            Dictionary with:
            - shap_contributions: list of {feature, display_name, value, contribution}
            - top_features: top 3 contributing features
            - explanation_text: plain-language explanation
        """
        explainer = (
            self.capacity_explainer if model_type == "capacity"
            else self.guardrail_explainer
        )

        if explainer is None or not HAS_SHAP:
            return self._fallback_explanation(features, model_type)

        try:
            # Prepare features
            X = self._prepare_features(features)
            shap_values = explainer.shap_values(X)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classifier returns [class_0_values, class_1_values]
                shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values

            # Calculate sum of absolute contributions to normalize
            total_abs_contrib = sum(abs(float(shap_vals[i])) for i in range(len(self.feature_names)) if i < len(shap_vals))

            # Build contribution list with relative percentage impact
            contributions = []
            for i, fname in enumerate(self.feature_names):
                if i < len(shap_vals):
                    raw_val = float(shap_vals[i])
                    # Calculate percentage impact relative to total feature shift
                    pct_impact = (raw_val / total_abs_contrib * 100) if total_abs_contrib > 0 else 0.0
                    
                    feat_val = features.get(fname)
                    if feat_val is not None and pd.isna(feat_val):
                        feat_val = None

                    contributions.append({
                        "feature": fname,
                        "display_name": FEATURE_DISPLAY_NAMES.get(fname, fname),
                        "value": feat_val,
                        "contribution": round(pct_impact, 2),  # percentage impact
                    })

            # Sort by absolute contribution descending
            contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

            # Top 3 features
            top_features = contributions[:3]

            # Generate explanation text using the templates
            explanation_text = self._generate_explanation_text(top_features, features)

            return {
                "shap_contributions": contributions,
                "top_features": top_features,
                "explanation_text": explanation_text,
            }

        except Exception as e:
            warnings.warn(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(features, model_type)

    def explain_batch(
        self, features_df: pd.DataFrame, model_type: str = "capacity"
    ) -> list[dict]:
        """Generate explanations for multiple customers."""
        results = []
        for _, row in features_df.iterrows():
            result = self.explain(row.to_dict(), model_type)
            result["customer_id"] = row["customer_id"]
            results.append(result)
        return results

    @property
    def feature_names(self):
        return ML_FEATURE_NAMES

    def _prepare_features(self, features: dict) -> pd.DataFrame:
        """Convert feature dict to model input format."""
        row = {}
        for name in self.feature_names:
            val = features.get(name)
            if name == "has_bureau_score":
                row[name] = int(bool(val))
            elif val is None:
                row[name] = float('nan')
            else:
                row[name] = float(val)
        return pd.DataFrame([row])

    def _generate_explanation_text(self, top_features: list[dict], features: dict) -> str:
        """Convert top SHAP features into a natural-language sentence."""
        parts = []
        for feat in top_features:
            fname = feat["feature"]
            value = feat["value"]
            contribution = feat["contribution"]

            templates = CONTRIBUTION_TEMPLATES.get(fname)
            if templates:
                direction = "positive" if contribution > 0 else "negative"
                template = templates[direction]
                try:
                    if value is not None:
                        text = template.format(value=value)
                    else:
                        text = template.replace("({value:.0f})", "").replace("(CV: {value:.2f})", "")
                        text = template.format(value=0) if "{value" in template else template
                except (ValueError, KeyError):
                    text = f"{FEATURE_DISPLAY_NAMES.get(fname, fname)}"
            else:
                display_name = FEATURE_DISPLAY_NAMES.get(fname, fname)
                text = f"{display_name} ({'positive' if contribution > 0 else 'negative'} impact)"

            parts.append(text)

        if parts:
            return f"Score driven by {parts[0]}; {'; '.join(parts[1:])}" if len(parts) > 1 else f"Score driven by {parts[0]}"
        return "Insufficient data for detailed explanation"

    def _fallback_explanation(self, features: dict, model_type: str) -> dict:
        """Fallback when SHAP is unavailable — use feature-value-based explanation."""
        raw_heurs = []
        
        # Calculate raw heuristic contributions based on logical credit rules
        for fname in self.feature_names:
            val = features.get(fname)
            if val is not None and pd.isna(val):
                val = None
            contrib = 0.0
            
            if val is not None:
                if fname == "income_mean":
                    contrib = min(val / 80000.0, 1.5) * 15.0
                elif fname == "bureau_score":
                    contrib = ((val - 650) / 100.0) * 12.0
                elif fname == "monthly_surplus":
                    contrib = min(val / 30000.0, 1.5) * 10.0
                elif fname == "rent_consistency":
                    contrib = val * 8.0
                elif fname == "gig_pattern_score":
                    contrib = -val * 3.0  # gig income has slightly higher risk discount
                elif fname == "emi_to_inflow_ratio":
                    contrib = -val * 25.0
                elif fname == "concurrent_lender_count":
                    contrib = -val * 4.0
                elif fname == "nach_bounce_count_6m":
                    contrib = -val * 18.0
                elif fname == "nach_bounce_count_3m":
                    contrib = -val * 25.0
                elif fname == "income_cv":
                    contrib = -val * 10.0
            
            raw_heurs.append({
                "feature": fname,
                "display_name": FEATURE_DISPLAY_NAMES.get(fname, fname),
                "value": val,
                "contribution": contrib,
            })
            
        # Normalize heuristic contributions to sum to 100% relative impact
        total_abs_contrib = sum(abs(h["contribution"]) for h in raw_heurs)
        
        contributions = []
        for h in raw_heurs:
            pct_impact = (h["contribution"] / total_abs_contrib * 100) if total_abs_contrib > 0 else 0.0
            contributions.append({
                "feature": h["feature"],
                "display_name": h["display_name"],
                "value": h["value"],
                "contribution": round(pct_impact, 2),
            })

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        top_features = contributions[:3]

        return {
            "shap_contributions": contributions,
            "top_features": top_features,
            "explanation_text": self._generate_explanation_text(top_features, features),
        }

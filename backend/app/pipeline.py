"""
CreditIntelligencePipeline — the single high-level entry point for using
CreditSetu as a library inside another application, instead of wiring feature
engineering + the three engines + SHAP explainability together by hand.

    from app.pipeline import CreditIntelligencePipeline
    from app.scoring_profile import ScoringProfile

    pipeline = CreditIntelligencePipeline(profile=ScoringProfile.from_yaml("my_profile.yaml"))
    pipeline.fit(customers_df, transactions_df)   # or pipeline.load_models(cap_path, guard_path)
    result = pipeline.score_customer(customer, customer_transactions_df)

Data contract
-------------
`customer` (dict) must include:
    - customer_id: str
    - bureau_score: float | None   (None => thin/no-file, handled natively — this
      is the core value proposition, not an edge case to work around)

`transactions` (pandas.DataFrame, one row per transaction) must include columns:
    - date: parseable to a datetime
    - amount: float, >= 0
    - type: "credit" | "debit"
    - category: str — at minimum distinguish income categories used for
      capacity/intent detection ("salary", "gig_payout", "merchant_collection"),
      "emi", "rent", and everything else as discretionary spend categories
    - counterparty: str
    - is_bounce: bool — True for a returned/bounced debit (e.g. NACH)

See features/feature_engineering.py for exactly how these columns are turned
into the model-facing feature set (ML_FEATURE_NAMES).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .engines.intent_engine import IntentEngine
from .engines.capacity_engine import CapacityEngine
from .engines.guardrail_engine import GuardrailEngine
from .engines.composite_scorer import CompositeScorer
from .explainability.shap_explainer import ShapExplainer
from .features.feature_engineering import engineer_features, engineer_features_batch
from .scoring_profile import ScoringProfile, default_profile


class CreditIntelligencePipeline:
    """
    One-call wrapper around feature engineering + Intent/Capacity/Guardrail
    engines + SHAP explainability + the composite scorer.
    """

    def __init__(
        self,
        profile: Optional[ScoringProfile] = None,
        capacity_engine: Optional[CapacityEngine] = None,
        guardrail_engine: Optional[GuardrailEngine] = None,
        intent_engine: Optional[IntentEngine] = None,
    ):
        self.profile = profile or default_profile()
        self.capacity_engine = capacity_engine or CapacityEngine(profile=self.profile)
        self.guardrail_engine = guardrail_engine or GuardrailEngine(profile=self.profile)
        self.intent_engine = intent_engine or IntentEngine(profile=self.profile)
        self.scorer = CompositeScorer(
            self.intent_engine, self.capacity_engine, self.guardrail_engine, profile=self.profile,
        )
        self._refresh_explainer()

    def fit(
        self,
        customers_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> dict:
        """
        Engineer features from raw transactions and fit the Capacity +
        Guardrail engines. `customers_df` needs `true_repayment_capacity`
        (Capacity target) and, optionally, a real `is_stressed` column
        (Guardrail target) — see the engines' own docstrings for the
        synthetic-vs-real-data distinction. Returns both engines' metrics.

        For a bulk seed/backfill workflow that also calls score_batch()
        against the same data right after fitting, compute features once with
        engineer_features_batch() and call fit_from_features() /
        score_batch(..., features_df=...) instead — avoids running feature
        engineering twice over a large transaction volume.
        """
        features_df = engineer_features_batch(customers_df, transactions_df)
        return self.fit_from_features(features_df, customers_df, test_size=test_size, seed=seed)

    def fit_from_features(
        self,
        features_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> dict:
        """Same as fit(), but takes an already-computed features_df."""
        capacity_metrics = self.capacity_engine.fit(features_df, customers_df, test_size=test_size, seed=seed)
        guardrail_metrics = self.guardrail_engine.fit(features_df, customers_df, test_size=test_size, seed=seed)
        self._refresh_explainer()
        return {"capacity_engine": capacity_metrics, "guardrail_engine": guardrail_metrics}

    def load_models(self, capacity_model_path: str, guardrail_model_path: str) -> None:
        """Load previously-fitted models from the model registry instead of retraining."""
        self.capacity_engine.load(capacity_model_path)
        self.guardrail_engine.load(guardrail_model_path)
        self._refresh_explainer()

    def score_customer(self, customer: dict, transactions: pd.DataFrame) -> dict:
        """Score one customer given their raw transaction history."""
        features = engineer_features(customer, transactions)
        result = self.scorer.score(features)

        shap_result = self.explainer.explain(features, model_type="capacity")
        result["shap_contributions"] = shap_result["shap_contributions"]
        result["top_features"] = shap_result["top_features"]
        if shap_result.get("explanation_text"):
            result["explanation"] = shap_result["explanation_text"] + ". " + result["explanation"]

        result["adverse_action_reasons"] = (
            self.explainer.get_adverse_action_reasons(shap_result["shap_contributions"])
            if not result["is_qualified_lead"] else []
        )
        result["customer_id"] = customer.get("customer_id")
        return result

    def score_batch(
        self,
        customers_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Score many customers at once. Every stage — feature engineering,
        engine inference, and SHAP explanation — runs vectorized across the
        whole batch rather than once per customer; this is the path to use
        at real volume. Pass a precomputed `features_df` (e.g. one you also
        passed to fit_from_features()) to skip recomputing it.
        """
        if features_df is None:
            features_df = engineer_features_batch(customers_df, transactions_df)
        scores_df = self.scorer.score_batch(features_df)

        shap_results = self.explainer.explain_batch(features_df, model_type="capacity")
        shap_by_id = {r["customer_id"]: r for r in shap_results}

        shap_contributions, top_features, explanations, adverse_reasons = [], [], [], []
        for _, row in scores_df.iterrows():
            shap_result = shap_by_id.get(row["customer_id"], {})
            contributions = shap_result.get("shap_contributions", [])
            shap_contributions.append(contributions)
            top_features.append(shap_result.get("top_features", []))

            explanation = row["explanation"]
            if shap_result.get("explanation_text"):
                explanation = shap_result["explanation_text"] + ". " + explanation
            explanations.append(explanation)

            adverse_reasons.append(
                self.explainer.get_adverse_action_reasons(contributions) if not row["is_qualified_lead"] else []
            )

        scores_df["shap_contributions"] = shap_contributions
        scores_df["top_features"] = top_features
        scores_df["explanation"] = explanations
        scores_df["adverse_action_reasons"] = adverse_reasons
        return scores_df

    def _refresh_explainer(self) -> None:
        self.explainer = ShapExplainer(
            capacity_model=self.capacity_engine.model,
            guardrail_model=self.guardrail_engine.model,
            currency_symbol=self.profile.currency_symbol,
        )

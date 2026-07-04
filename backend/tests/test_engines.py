"""
Tests for scoring engines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd

from app.data_generation.synthetic_customer_generator import generate_customers
from app.data_generation.synthetic_transaction_generator import generate_all_transactions
from app.features.feature_engineering import engineer_features_batch
from app.engines.intent_engine import IntentEngine
from app.engines.capacity_engine import CapacityEngine
from app.engines.guardrail_engine import GuardrailEngine
from app.engines.composite_scorer import CompositeScorer


@pytest.fixture(scope="module")
def test_data():
    """Generate a small test dataset for engine tests."""
    customers_df = generate_customers(n_customers=50, seed=99)
    transactions_df = generate_all_transactions(customers_df, seed=99)
    features_df = engineer_features_batch(customers_df, transactions_df)
    return customers_df, transactions_df, features_df


@pytest.fixture(scope="module")
def trained_engines(test_data):
    """Train engines on test data."""
    customers_df, _, features_df = test_data

    capacity = CapacityEngine()
    capacity.train(features_df, customers_df)

    guardrail = GuardrailEngine()
    guardrail.train(features_df, customers_df)

    intent = IntentEngine()

    return intent, capacity, guardrail


class TestIntentEngine:
    """Test the Intent Signal Engine."""

    def test_score_range(self, test_data, trained_engines):
        """Intent scores must be in [0, 1]."""
        _, _, features_df = test_data
        intent, _, _ = trained_engines

        for _, row in features_df.iterrows():
            result = intent.score(row.to_dict())
            assert 0.0 <= result["intent_score"] <= 1.0, \
                f"Intent score {result['intent_score']} out of range"

    def test_returns_required_fields(self, test_data, trained_engines):
        """Must return all required fields."""
        _, _, features_df = test_data
        intent, _, _ = trained_engines

        result = intent.score(features_df.iloc[0].to_dict())
        required = ["intent_score", "intent_event_type", "intent_event_recency_days", "intent_details"]
        for field in required:
            assert field in result, f"Missing field: {field}"


class TestCapacityEngine:
    """Test the Capacity Scoring Engine."""

    def test_score_range(self, test_data, trained_engines):
        """Capacity scores must be in [0, 1]."""
        _, _, features_df = test_data
        _, capacity, _ = trained_engines

        for _, row in features_df.iterrows():
            result = capacity.predict(row.to_dict())
            assert 0.0 <= result["capacity_score"] <= 1.0, \
                f"Capacity score {result['capacity_score']} out of range"

    def test_capacity_amount_non_negative(self, test_data, trained_engines):
        """Predicted capacity amount must be non-negative."""
        _, _, features_df = test_data
        _, capacity, _ = trained_engines

        for _, row in features_df.iterrows():
            result = capacity.predict(row.to_dict())
            assert result["capacity_amount"] >= 0, "Capacity amount must be non-negative"

    def test_ntc_customers_get_scored(self, test_data, trained_engines):
        """NTC/gig customers with null bureau_score MUST get valid scores."""
        customers_df, _, features_df = test_data
        _, capacity, _ = trained_engines

        ntc_ids = customers_df[
            customers_df["persona_type"].isin(["new_to_credit", "gig_worker"])
        ]["customer_id"].tolist()

        for cust_id in ntc_ids[:5]:  # Test first 5
            row = features_df[features_df["customer_id"] == cust_id].iloc[0]
            result = capacity.predict(row.to_dict())
            assert result["capacity_score"] is not None
            assert not pd.isna(result["capacity_score"]), \
                f"NTC customer {cust_id} got NaN capacity score"


class TestGuardrailEngine:
    """Test the Guardrail Engine."""

    def test_tier_values(self, test_data, trained_engines):
        """Guardrail tier must be one of Safe/Watch/Suppress."""
        _, _, features_df = test_data
        _, _, guardrail = trained_engines

        valid_tiers = {"Safe", "Watch", "Suppress"}
        for _, row in features_df.iterrows():
            result = guardrail.evaluate(row.to_dict())
            assert result["guardrail_tier"] in valid_tiers, \
                f"Invalid tier: {result['guardrail_tier']}"

    def test_over_leveraged_suppressed(self, test_data, trained_engines):
        """Over-leveraged customers with high EMI count should be Suppressed."""
        customers_df, _, features_df = test_data
        _, _, guardrail = trained_engines

        overlev_ids = customers_df[
            customers_df["persona_type"] == "over_leveraged"
        ]["customer_id"].tolist()

        suppressed_count = 0
        for cust_id in overlev_ids:
            matching = features_df[features_df["customer_id"] == cust_id]
            if matching.empty:
                continue
            row = matching.iloc[0]
            result = guardrail.evaluate(row.to_dict())
            if result["guardrail_tier"] == "Suppress":
                suppressed_count += 1

        # Most over-leveraged should be suppressed
        suppression_rate = suppressed_count / len(overlev_ids) if overlev_ids else 0
        assert suppression_rate > 0.5, \
            f"Expected >50% over-leveraged suppressed, got {suppression_rate:.0%}"

    def test_reasons_provided_for_suppress(self, test_data, trained_engines):
        """Suppressed customers must have at least one reason."""
        _, _, features_df = test_data
        _, _, guardrail = trained_engines

        for _, row in features_df.iterrows():
            result = guardrail.evaluate(row.to_dict())
            if result["guardrail_tier"] == "Suppress":
                assert len(result["guardrail_reasons"]) > 0, \
                    "Suppressed customer must have reasons"


class TestCompositeScorer:
    """Test the Composite Scorer."""

    def test_score_range(self, test_data, trained_engines):
        """Composite scores must be in [0, 1]."""
        _, _, features_df = test_data
        intent, capacity, guardrail = trained_engines
        scorer = CompositeScorer(intent, capacity, guardrail)

        for _, row in features_df.head(10).iterrows():
            result = scorer.score(row.to_dict())
            assert 0.0 <= result["composite_score"] <= 1.0

    def test_suppressed_not_qualified(self, test_data, trained_engines):
        """Suppressed customers must not be qualified leads."""
        _, _, features_df = test_data
        intent, capacity, guardrail = trained_engines
        scorer = CompositeScorer(intent, capacity, guardrail)

        for _, row in features_df.head(20).iterrows():
            result = scorer.score(row.to_dict())
            if result["guardrail_tier"] == "Suppress":
                assert result["is_qualified_lead"] is False, \
                    "Suppressed customer should not be a qualified lead"

    def test_has_product_suggestion(self, test_data, trained_engines):
        """All scored customers should have a product suggestion."""
        _, _, features_df = test_data
        intent, capacity, guardrail = trained_engines
        scorer = CompositeScorer(intent, capacity, guardrail)

        result = scorer.score(features_df.iloc[0].to_dict())
        assert result["suggested_product"] in {"Personal Loan", "Auto Loan", "Home Loan"}

    def test_has_explanation(self, test_data, trained_engines):
        """All scored customers should have an explanation."""
        _, _, features_df = test_data
        intent, capacity, guardrail = trained_engines
        scorer = CompositeScorer(intent, capacity, guardrail)

        result = scorer.score(features_df.iloc[0].to_dict())
        assert result["explanation"] is not None
        assert len(result["explanation"]) > 10

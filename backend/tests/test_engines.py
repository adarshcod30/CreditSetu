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
from app.scoring_profile import ScoringProfile, default_profile


@pytest.fixture(scope="module")
def test_data():
    """Generate a small test dataset for engine tests."""
    customers_df = generate_customers(n_customers=50, seed=99)
    transactions_df = generate_all_transactions(customers_df, seed=99)
    features_df = engineer_features_batch(customers_df, transactions_df)
    return customers_df, transactions_df, features_df


@pytest.fixture(scope="module")
def trained_engines(test_data):
    """Fit engines on test data."""
    customers_df, _, features_df = test_data

    capacity = CapacityEngine()
    capacity.fit(features_df, customers_df)

    guardrail = GuardrailEngine()
    guardrail.fit(features_df, customers_df)

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
        """All scored customers should have a product suggestion from the default profile's catalog."""
        _, _, features_df = test_data
        intent, capacity, guardrail = trained_engines
        scorer = CompositeScorer(intent, capacity, guardrail)

        valid_products = {
            "Home Loan", "Auto Loan", "Personal Loan",
            "Micro-Credit Line", "Retail Credit Card",
        }
        for _, row in features_df.head(10).iterrows():
            result = scorer.score(row.to_dict())
            assert result["suggested_product"] in valid_products

    def test_has_explanation(self, test_data, trained_engines):
        """All scored customers should have an explanation."""
        _, _, features_df = test_data
        intent, capacity, guardrail = trained_engines
        scorer = CompositeScorer(intent, capacity, guardrail)

        result = scorer.score(features_df.iloc[0].to_dict())
        assert result["explanation"] is not None
        assert len(result["explanation"]) > 10

    def test_batch_matches_single_row_scoring(self, test_data, trained_engines):
        """
        score_batch()'s vectorized path must produce the same results as
        calling score() once per row — this is the parity check for the
        capacity/guardrail engines' vectorized predict_batch/evaluate_batch.
        """
        _, _, features_df = test_data
        intent, capacity, guardrail = trained_engines
        scorer = CompositeScorer(intent, capacity, guardrail)

        subset = features_df.head(15).reset_index(drop=True)
        batch_results = scorer.score_batch(subset).set_index("customer_id")

        for _, row in subset.iterrows():
            single_result = scorer.score(row.to_dict())
            batch_row = batch_results.loc[row["customer_id"]]
            assert batch_row["composite_score"] == pytest.approx(single_result["composite_score"], abs=1e-6)
            assert batch_row["capacity_amount"] == pytest.approx(single_result["capacity_amount"], abs=1e-6)
            assert batch_row["guardrail_tier"] == single_result["guardrail_tier"]
            assert batch_row["suggested_product"] == single_result["suggested_product"]


class TestScoringProfile:
    """Test that swapping a ScoringProfile actually changes engine behavior."""

    def test_custom_currency_appears_in_explanation(self, test_data, trained_engines):
        _, _, features_df = test_data
        intent, capacity, guardrail = trained_engines

        custom_profile = default_profile().model_copy(deep=True)
        custom_profile.currency_symbol = "$"
        scorer = CompositeScorer(intent, capacity, guardrail, profile=custom_profile)

        result = scorer.score(features_df.iloc[0].to_dict())
        assert "$" in result["explanation"]
        assert "₹" not in result["explanation"]

    def test_tighter_guardrail_thresholds_suppress_more(self, test_data):
        """A tighter profile's guardrail thresholds should be at least as strict."""
        customers_df, _, features_df = test_data

        loose_profile = ScoringProfile(name="loose")
        tight_profile = ScoringProfile(name="tight")
        tight_profile.guardrail.max_concurrent_lenders = 1
        tight_profile.guardrail.suppress_threshold = 0.01

        loose_guardrail = GuardrailEngine(profile=loose_profile)
        loose_guardrail.fit(features_df, customers_df)
        tight_guardrail = GuardrailEngine(profile=tight_profile)
        tight_guardrail.fit(features_df, customers_df)

        loose_suppressed = sum(
            loose_guardrail.evaluate(row.to_dict())["guardrail_tier"] == "Suppress"
            for _, row in features_df.iterrows()
        )
        tight_suppressed = sum(
            tight_guardrail.evaluate(row.to_dict())["guardrail_tier"] == "Suppress"
            for _, row in features_df.iterrows()
        )
        assert tight_suppressed >= loose_suppressed

    def test_product_catalog_is_profile_driven(self):
        """select_product must use the profile's own rules, not a hardcoded table."""
        profile = ScoringProfile(
            name="single-product",
            standard_product_rules=[{"name": "Only Product", "min_capacity_amount": 0, "min_income_mean": 0}],
            thin_file_product_rules=[{"name": "Only Product", "min_capacity_amount": 0, "min_income_mean": 0}],
            fallback_product="Only Product",
        )
        product = profile.select_product(
            capacity_amount=1.0, income_mean=1.0, has_bureau_score=True, gig_pattern_score=0.0,
        )
        assert product == "Only Product"

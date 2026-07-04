"""
Tests for synthetic data generation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd

from app.data_generation.personas import ALL_PERSONAS, validate_proportions
from app.data_generation.synthetic_customer_generator import generate_customers
from app.data_generation.synthetic_transaction_generator import (
    generate_transactions,
    generate_all_transactions,
)


class TestPersonas:
    """Test persona configuration."""

    def test_proportions_sum_to_one(self):
        """Persona proportions must sum to 1.0."""
        validate_proportions()  # Will assert internally

    def test_five_personas_defined(self):
        """Must have exactly 5 persona types."""
        assert len(ALL_PERSONAS) == 5

    def test_persona_names_unique(self):
        """All persona names must be unique."""
        names = [p.name for p in ALL_PERSONAS]
        assert len(names) == len(set(names))


class TestCustomerGenerator:
    """Test synthetic customer generation."""

    @pytest.fixture
    def customers_df(self):
        return generate_customers(n_customers=100, seed=42)

    def test_correct_count(self, customers_df):
        """Should generate the requested number of customers."""
        assert len(customers_df) == 100

    def test_deterministic(self):
        """Same seed should produce identical results."""
        df1 = generate_customers(n_customers=50, seed=42)
        df2 = generate_customers(n_customers=50, seed=42)
        assert df1["customer_id"].tolist() == df2["customer_id"].tolist()

    def test_persona_distribution(self, customers_df):
        """Persona distribution should be approximately correct (±10% tolerance for small N)."""
        counts = customers_df["persona_type"].value_counts(normalize=True)
        assert abs(counts.get("salaried_stable", 0) - 0.35) < 0.10
        assert abs(counts.get("gig_worker", 0) - 0.20) < 0.10
        assert abs(counts.get("over_leveraged", 0) - 0.15) < 0.10

    def test_bureau_score_null_for_ntc(self, customers_df):
        """NTC personas should have null bureau scores."""
        ntc = customers_df[customers_df["persona_type"] == "new_to_credit"]
        assert ntc["bureau_score"].isna().all(), "NTC customers should have no bureau score"

    def test_bureau_score_populated_for_salaried(self, customers_df):
        """Most salaried customers should have bureau scores."""
        salaried = customers_df[customers_df["persona_type"] == "salaried_stable"]
        has_score = salaried["bureau_score"].notna().mean()
        assert has_score > 0.80, f"Expected >80% salaried with bureau score, got {has_score:.0%}"

    def test_over_leveraged_have_emis(self, customers_df):
        """Over-leveraged customers should have 4+ EMIs."""
        overlev = customers_df[customers_df["persona_type"] == "over_leveraged"]
        assert (overlev["emi_count"] >= 4).all(), "Over-leveraged must have 4+ EMIs"

    def test_repayment_capacity_non_negative(self, customers_df):
        """True repayment capacity should never be negative."""
        assert (customers_df["true_repayment_capacity"] >= 0).all()

    def test_life_events_exist(self, customers_df):
        """Some customers should have life events."""
        has_events = customers_df["life_events"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        assert has_events.any(), "At least some customers should have life events"

    def test_required_columns_present(self, customers_df):
        """All required columns must be present."""
        required = ["customer_id", "name", "age", "gender", "occupation",
                     "persona_type", "bureau_score", "city", "monthly_income",
                     "true_repayment_capacity", "life_events"]
        for col in required:
            assert col in customers_df.columns, f"Missing column: {col}"


class TestTransactionGenerator:
    """Test synthetic transaction generation."""

    @pytest.fixture
    def sample_customer(self):
        df = generate_customers(n_customers=10, seed=42)
        return df.iloc[0].to_dict()

    def test_generates_transactions(self, sample_customer):
        """Should generate a non-empty set of transactions."""
        txns = generate_transactions(sample_customer, seed=42)
        assert len(txns) > 0

    def test_has_required_columns(self, sample_customer):
        """Transactions must have required columns."""
        txns = generate_transactions(sample_customer, seed=42)
        required = ["date", "amount", "type", "category", "counterparty"]
        for col in required:
            assert col in txns.columns, f"Missing column: {col}"

    def test_amounts_positive(self, sample_customer):
        """All transaction amounts must be positive."""
        txns = generate_transactions(sample_customer, seed=42)
        assert (txns["amount"] > 0).all()

    def test_types_valid(self, sample_customer):
        """Transaction types must be 'credit' or 'debit'."""
        txns = generate_transactions(sample_customer, seed=42)
        assert set(txns["type"].unique()).issubset({"credit", "debit"})

    def test_over_leveraged_has_bounces(self):
        """Over-leveraged customers should have NACH bounce events."""
        df = generate_customers(n_customers=100, seed=42)
        overlev = df[df["persona_type"] == "over_leveraged"]
        if len(overlev) == 0:
            pytest.skip("No over-leveraged customers in sample")

        found_bounce = False
        for _, row in overlev.iterrows():
            txns = generate_transactions(row.to_dict(), seed=42)
            if txns["is_bounce"].any():
                found_bounce = True
                break

        assert found_bounce, "At least one over-leveraged customer should have NACH bounces"

    def test_batch_generation(self):
        """Batch generation should work for all customers."""
        customers = generate_customers(n_customers=20, seed=42)
        all_txns = generate_all_transactions(customers, seed=42)
        assert len(all_txns) > 0
        assert all_txns["customer_id"].nunique() == 20

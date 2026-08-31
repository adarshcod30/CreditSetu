"""
Tests for CreditIntelligencePipeline (the library entry point), the model
registry, and feature-engineering edge cases in real/messy transaction data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from app.data_generation.synthetic_customer_generator import generate_customers
from app.data_generation.synthetic_transaction_generator import generate_all_transactions
from app.pipeline import CreditIntelligencePipeline
from app.scoring_profile import default_profile
from app.engines.capacity_engine import CapacityEngine
from app import model_registry
from app.features.feature_engineering import engineer_features


@pytest.fixture(scope="module")
def fitted_pipeline():
    customers_df = generate_customers(n_customers=60, seed=7)
    transactions_df = generate_all_transactions(customers_df, seed=7)
    pipeline = CreditIntelligencePipeline(profile=default_profile())
    pipeline.fit(customers_df, transactions_df)
    return pipeline, customers_df, transactions_df


class TestPipelineScoring:
    def test_score_customer_matches_score_batch(self, fitted_pipeline):
        """The single-customer and batch entry points must agree."""
        pipeline, customers_df, transactions_df = fitted_pipeline
        sample = customers_df.iloc[0].to_dict()
        cust_txns = transactions_df[transactions_df["customer_id"] == sample["customer_id"]]

        single_result = pipeline.score_customer(sample, cust_txns)
        batch_result = pipeline.score_batch(
            customers_df.head(10), transactions_df[transactions_df["customer_id"].isin(customers_df.head(10)["customer_id"])]
        )
        batch_row = batch_result.set_index("customer_id").loc[sample["customer_id"]]

        assert single_result["composite_score"] == pytest.approx(batch_row["composite_score"], abs=1e-6)
        assert single_result["guardrail_tier"] == batch_row["guardrail_tier"]

    def test_adverse_action_reasons_present_when_not_qualified(self, fitted_pipeline):
        """Suppressed customers must get non-empty adverse action reasons; qualified ones shouldn't need them."""
        pipeline, customers_df, transactions_df = fitted_pipeline
        results = pipeline.score_batch(customers_df, transactions_df)

        suppressed = results[~results["is_qualified_lead"]]
        if len(suppressed) == 0:
            pytest.skip("No suppressed customers in this sample")
        assert suppressed["adverse_action_reasons"].apply(len).gt(0).all()

    def test_load_models_reuses_fitted_engines(self, fitted_pipeline, tmp_path):
        pipeline, customers_df, transactions_df = fitted_pipeline
        cap_path = str(tmp_path / "capacity_model.pkl")
        guard_path = str(tmp_path / "guardrail_model.pkl")
        pipeline.capacity_engine.save(cap_path)
        pipeline.guardrail_engine.save(guard_path)

        fresh_pipeline = CreditIntelligencePipeline(profile=default_profile())
        fresh_pipeline.load_models(cap_path, guard_path)

        sample = customers_df.iloc[0].to_dict()
        cust_txns = transactions_df[transactions_df["customer_id"] == sample["customer_id"]]
        result = fresh_pipeline.score_customer(sample, cust_txns)
        assert 0.0 <= result["composite_score"] <= 1.0


class TestModelRegistry:
    def test_save_creates_versions_and_pointer(self, tmp_path):
        path = tmp_path / "model.pkl"
        model_registry.save({"a": 1}, path, metadata={"note": "v1"})
        model_registry.save({"a": 2}, path, metadata={"note": "v2"})

        loaded, meta = model_registry.load(path)
        assert loaded == {"a": 2}
        assert meta["note"] == "v2"
        assert meta["version"] == 2

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            model_registry.load(tmp_path / "does_not_exist.pkl")

    def test_capacity_engine_save_load_roundtrip(self, fitted_pipeline, tmp_path):
        pipeline, _, _ = fitted_pipeline
        path = str(tmp_path / "capacity_model.pkl")
        pipeline.capacity_engine.save(path)

        reloaded = CapacityEngine()
        reloaded.load(path)
        assert reloaded.metadata["engine"] == "CapacityEngine"
        assert reloaded.metadata["version"] == 1


class TestFeatureEngineeringEdgeCases:
    """Real transaction feeds are messy — these must not crash or silently misbehave."""

    def _base_txns(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"txn_id": "T1", "date": "2024-01-05", "amount": 50000, "type": "credit",
             "category": "salary", "counterparty": "employer", "is_bounce": False},
            {"txn_id": "T2", "date": "2024-02-05", "amount": 50000, "type": "credit",
             "category": "salary", "counterparty": "employer", "is_bounce": False},
        ])

    def test_empty_transactions_returns_default_features(self):
        features = engineer_features({"customer_id": "C1", "bureau_score": None}, pd.DataFrame())
        assert features["income_mean"] == 0.0
        assert features["has_bureau_score"] is False

    def test_single_transaction_does_not_crash(self):
        txns = self._base_txns().iloc[:1]
        features = engineer_features({"customer_id": "C1", "bureau_score": 700}, txns)
        assert features["income_mean"] >= 0

    def test_all_debit_no_income_customer(self):
        txns = pd.DataFrame([
            {"txn_id": "T1", "date": "2024-01-05", "amount": 500, "type": "debit",
             "category": "groceries", "counterparty": "store", "is_bounce": False},
            {"txn_id": "T2", "date": "2024-01-10", "amount": 300, "type": "debit",
             "category": "groceries", "counterparty": "store", "is_bounce": False},
        ])
        features = engineer_features({"customer_id": "C1", "bureau_score": None}, txns)
        assert features["income_mean"] == 0.0
        assert features["income_cv"] == 1.0  # fully unstable/unknown, not a crash or negative value

    def test_duplicate_txn_id_deduplicated(self):
        txns = pd.concat([self._base_txns(), self._base_txns().iloc[:1]], ignore_index=True)
        features_with_dupe = engineer_features({"customer_id": "C1", "bureau_score": None}, txns)
        features_clean = engineer_features({"customer_id": "C1", "bureau_score": None}, self._base_txns())
        assert features_with_dupe["income_mean"] == pytest.approx(features_clean["income_mean"])

    def test_negative_amount_dropped_not_crashing(self):
        txns = self._base_txns().copy()
        txns.loc[len(txns)] = {"txn_id": "T3", "date": "2024-03-05", "amount": -1000,
                                "type": "credit", "category": "salary", "counterparty": "employer", "is_bounce": False}
        features = engineer_features({"customer_id": "C1", "bureau_score": None}, txns)
        assert features["income_mean"] >= 0

    def test_unparsed_string_dates_and_amounts_are_coerced(self):
        """engineer_features() must be usable directly with un-pre-converted API-supplied data."""
        txns = pd.DataFrame([
            {"date": "2024-01-05", "amount": "50000", "type": "CREDIT",
             "category": "Salary", "counterparty": "employer", "is_bounce": False},
            {"date": "2024-02-05", "amount": "50000", "type": "Credit",
             "category": "salary", "counterparty": "employer", "is_bounce": False},
        ])
        features = engineer_features({"customer_id": "C1", "bureau_score": None}, txns)
        assert features["income_mean"] > 0  # case-insensitive type/category matching worked

    def test_unknown_category_does_not_crash(self):
        txns = self._base_txns().copy()
        txns.loc[len(txns), :] = {"txn_id": "T3", "date": "2024-03-05", "amount": 200,
                                   "type": "debit", "category": "some_new_category_2026",
                                   "counterparty": "x", "is_bounce": False}
        features = engineer_features({"customer_id": "C1", "bureau_score": None}, txns)
        assert 0.0 <= features["merchant_category_entropy"] <= 1.0

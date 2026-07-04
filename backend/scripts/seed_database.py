#!/usr/bin/env python3
"""
Seed database script for CreditSetu.

Generates ~1000 synthetic customers with transactions, trains ML models,
scores all customers, and stores everything in SQLite.

Usage:
    python scripts/seed_database.py [--n_customers 1000] [--seed 42]

Deterministic via seed parameter for reproducible demo results.
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

# Add backend root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.database import init_db, SessionLocal, engine, Base
from app.models.customer import Customer
from app.models.transaction import Transaction
from app.models.score import Score
from app.data_generation.synthetic_customer_generator import generate_customers
from app.data_generation.synthetic_transaction_generator import generate_all_transactions
from app.features.feature_engineering import engineer_features_batch
from app.engines.intent_engine import IntentEngine
from app.engines.capacity_engine import CapacityEngine
from app.engines.guardrail_engine import GuardrailEngine
from app.engines.composite_scorer import CompositeScorer
from app.explainability.shap_explainer import ShapExplainer


def seed_database(n_customers: int = 1000, seed: int = 42):
    """
    Full pipeline: generate data → engineer features → train models →
    score customers → store everything in SQLite.
    """
    total_start = time.time()

    print("=" * 60)
    print("CreditSetu — Database Seeding")
    print("=" * 60)

    # ─── Initialize Database ─────────────────────────────────────────
    print("\n[1/7] Initializing database...")
    os.makedirs("data/models", exist_ok=True)

    # Drop and recreate all tables
    Base.metadata.drop_all(bind=engine)
    init_db()
    print("  ✅ Database tables created")

    # ─── Generate Customers ──────────────────────────────────────────
    print(f"\n[2/7] Generating {n_customers} synthetic customers...")
    t0 = time.time()
    customers_df = generate_customers(n_customers=n_customers, seed=seed)

    # Log persona distribution
    persona_counts = customers_df["persona_type"].value_counts()
    print(f"  Persona distribution:")
    for persona, count in persona_counts.items():
        print(f"    {persona}: {count} ({count/n_customers:.1%})")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ─── Generate Transactions ───────────────────────────────────────
    print(f"\n[3/7] Generating transactions for all customers...")
    t0 = time.time()
    transactions_df = generate_all_transactions(customers_df, seed=seed)
    print(f"  Generated {len(transactions_df):,} transactions")
    print(f"  Avg per customer: {len(transactions_df) / n_customers:.0f}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ─── Store Customers in Database ─────────────────────────────────
    print("\n[4/7] Storing customers and transactions in database...")
    t0 = time.time()
    db = SessionLocal()

    try:
        # Store customers
        for _, row in customers_df.iterrows():
            customer = Customer(
                customer_id=row["customer_id"],
                name=row["name"],
                age=int(row["age"]),
                gender=row["gender"],
                occupation=row["occupation"],
                persona_type=row["persona_type"],
                bureau_score=row["bureau_score"] if row["bureau_score"] is not None else None,
                city=row["city"],
                account_open_date=row["account_open_date"],
                monthly_income=float(row["monthly_income"]),
                emi_count=int(row["emi_count"]),
                total_emi=float(row["total_emi"]),
                true_repayment_capacity=float(row["true_repayment_capacity"]),
                life_events=json.dumps(row["life_events"]),
                observation_months=int(row["observation_months"]),
            )
            db.add(customer)
        db.commit()
        print(f"  ✅ {n_customers} customers stored")

        # Store transactions in batches using bulk insert
        batch_size = 50000
        txn_records = []
        for _, row in transactions_df.iterrows():
            txn_records.append({
                "txn_id": row["txn_id"],
                "customer_id": row["customer_id"],
                "date": row["date"],
                "amount": float(row["amount"]),
                "type": row["type"],
                "category": row.get("category"),
                "counterparty": row.get("counterparty"),
                "channel": row.get("channel"),
                "narration": row.get("narration"),
                "is_bounce": bool(row.get("is_bounce", False)),
            })
        
        for i in range(0, len(txn_records), batch_size):
            batch = txn_records[i:i + batch_size]
            db.bulk_insert_mappings(Transaction, batch)
            db.commit()
            print(f"  Stored {min(i + batch_size, len(txn_records)):,}/{len(txn_records):,} transactions")

        print(f"  Time: {time.time() - t0:.1f}s")

        # ─── Feature Engineering ─────────────────────────────────────
        print("\n[5/7] Computing features...")
        t0 = time.time()
        features_df = engineer_features_batch(customers_df, transactions_df)
        print(f"  Computed {len(features_df.columns)} features")
        print(f"  Time: {time.time() - t0:.1f}s")

        # ─── Train Models ────────────────────────────────────────────
        print("\n[6/7] Training ML models...")
        t0 = time.time()

        capacity_engine = CapacityEngine()
        cap_metrics = capacity_engine.train(features_df, customers_df)
        capacity_engine.save("data/models/capacity_model.pkl")
        print(f"  Capacity Engine — AUC-ROC: {cap_metrics['auc_roc']}, RMSE: ₹{cap_metrics['rmse']:,.0f}")

        guardrail_engine = GuardrailEngine()
        guard_metrics = guardrail_engine.train(features_df, customers_df)
        guardrail_engine.save("data/models/guardrail_model.pkl")
        print(f"  Guardrail Engine — AUC-ROC: {guard_metrics['auc_roc']}, FPR: {guard_metrics['false_positive_rate']}")

        print(f"  Time: {time.time() - t0:.1f}s")

        # ─── Score All Customers ─────────────────────────────────────
        print("\n[7/7] Scoring all customers...")
        t0 = time.time()

        intent_engine = IntentEngine()
        scorer = CompositeScorer(intent_engine, capacity_engine, guardrail_engine)

        # SHAP explainer
        explainer = ShapExplainer(
            capacity_model=capacity_engine.model,
            guardrail_model=guardrail_engine.model,
        )

        scored = 0
        tier_counts = {"Safe": 0, "Watch": 0, "Suppress": 0}

        for _, row in features_df.iterrows():
            features = row.to_dict()
            score_result = scorer.score(features)

            # Get SHAP explanation
            shap_result = explainer.explain(features, model_type="capacity")

            # Combine explanations
            explanation = score_result["explanation"]
            if shap_result.get("explanation_text"):
                explanation = shap_result["explanation_text"] + ". " + explanation

            score_record = Score(
                customer_id=row["customer_id"],
                intent_score=score_result["intent_score"],
                intent_event_type=score_result["intent_event_type"],
                intent_event_recency_days=score_result["intent_event_recency_days"],
                capacity_score=score_result["capacity_score"],
                capacity_amount=score_result["capacity_amount"],
                capacity_confidence=score_result["capacity_confidence"],
                guardrail_score=score_result["guardrail_score"],
                guardrail_tier=score_result["guardrail_tier"],
                guardrail_reasons=json.dumps(score_result["guardrail_reasons"]),
                composite_score=score_result["composite_score"],
                is_qualified_lead=score_result["is_qualified_lead"],
                suggested_product=score_result["suggested_product"],
                explanation=explanation,
                shap_contributions=json.dumps(shap_result.get("shap_contributions", [])),
                top_features=json.dumps(shap_result.get("top_features", [])),
            )
            db.add(score_record)

            tier_counts[score_result["guardrail_tier"]] += 1
            scored += 1

            if scored % 200 == 0:
                db.commit()
                print(f"  Scored {scored}/{n_customers}")

        db.commit()

        print(f"\n  ✅ All {scored} customers scored")
        print(f"  Risk Tiers: Safe={tier_counts['Safe']}, Watch={tier_counts['Watch']}, Suppress={tier_counts['Suppress']}")
        print(f"  Time: {time.time() - t0:.1f}s")

    finally:
        db.close()

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"✅ Database seeding complete in {total_time:.1f}s")
    print(f"   {n_customers} customers, {len(transactions_df):,} transactions")
    print(f"   DB file: data/creditsetu.db")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed CreditSetu database")
    parser.add_argument("--n_customers", type=int, default=1000, help="Number of customers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    seed_database(n_customers=args.n_customers, seed=args.seed)

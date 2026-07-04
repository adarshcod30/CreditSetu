"""
Score and data generation API routes for CreditSetu.
"""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from ..database import get_db, init_db
from ..models.customer import Customer
from ..models.transaction import Transaction
from ..models.score import Score
from ..schemas.schemas import ScoreResponse, ShapFeature, GenerateDataRequest, GenerateDataResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Scoring"])


@router.get("/score/{customer_id}", response_model=ScoreResponse)
def get_score(customer_id: str, db: Session = Depends(get_db)):
    """Get full score breakdown with explainability for one customer."""
    # Get latest score
    score = (
        db.query(Score)
        .filter(Score.customer_id == customer_id)
        .order_by(Score.id.desc())
        .first()
    )

    if not score:
        raise HTTPException(status_code=404, detail=f"No score found for customer {customer_id}")

    # Parse JSON fields
    guardrail_reasons = []
    if score.guardrail_reasons:
        try:
            guardrail_reasons = json.loads(score.guardrail_reasons)
        except (json.JSONDecodeError, TypeError):
            guardrail_reasons = [score.guardrail_reasons]

    import math
    def sanitize_shap_features(raw_list):
        sanitized = []
        for item in raw_list:
            val = item.get("value")
            if val is not None and isinstance(val, float) and math.isnan(val):
                item["value"] = None
            sanitized.append(ShapFeature(**item))
        return sanitized

    shap_contributions = []
    if score.shap_contributions:
        try:
            raw = json.loads(score.shap_contributions)
            shap_contributions = sanitize_shap_features(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    top_features = []
    if score.top_features:
        try:
            raw = json.loads(score.top_features)
            top_features = sanitize_shap_features(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    return ScoreResponse(
        customer_id=score.customer_id,
        composite_score=score.composite_score,
        intent_score=score.intent_score,
        intent_event_type=score.intent_event_type,
        intent_event_recency_days=score.intent_event_recency_days,
        capacity_score=score.capacity_score,
        capacity_amount=score.capacity_amount,
        capacity_confidence=score.capacity_confidence,
        guardrail_score=score.guardrail_score,
        guardrail_tier=score.guardrail_tier,
        guardrail_reasons=guardrail_reasons,
        is_qualified_lead=score.is_qualified_lead,
        suggested_product=score.suggested_product,
        explanation=score.explanation or "",
        shap_contributions=shap_contributions,
        top_features=top_features,
        scored_at=score.scored_at,
    )


@router.post("/data/generate", response_model=GenerateDataResponse)
def generate_data(
    request: GenerateDataRequest,
    db: Session = Depends(get_db),
):
    """
    Regenerate synthetic dataset.

    WARNING: This wipes all existing data and repopulates.
    """
    from ..data_generation.synthetic_customer_generator import generate_customers
    from ..data_generation.synthetic_transaction_generator import generate_all_transactions
    from ..features.feature_engineering import engineer_features_batch
    from ..engines.intent_engine import IntentEngine
    from ..engines.capacity_engine import CapacityEngine
    from ..engines.guardrail_engine import GuardrailEngine
    from ..engines.composite_scorer import CompositeScorer
    from ..explainability.shap_explainer import ShapExplainer
    import os

    logger.info(f"Generating {request.n_customers} customers with seed={request.seed}")

    # Wipe existing data
    db.query(Score).delete()
    db.query(Transaction).delete()
    db.query(Customer).delete()
    db.commit()

    # Generate customers
    customers_df = generate_customers(n_customers=request.n_customers, seed=request.seed)

    # Generate transactions
    transactions_df = generate_all_transactions(customers_df, seed=request.seed)

    # Store customers using bulk insert
    customer_records = []
    for _, row in customers_df.iterrows():
        customer_records.append({
            "customer_id": row["customer_id"],
            "name": row["name"],
            "age": int(row["age"]),
            "gender": row["gender"],
            "occupation": row["occupation"],
            "persona_type": row["persona_type"],
            "bureau_score": row["bureau_score"] if row["bureau_score"] is not None else None,
            "city": row["city"],
            "account_open_date": row["account_open_date"],
            "monthly_income": float(row["monthly_income"]),
            "emi_count": int(row["emi_count"]),
            "total_emi": float(row["total_emi"]),
            "true_repayment_capacity": float(row["true_repayment_capacity"]),
            "life_events": json.dumps(row["life_events"]),
            "observation_months": int(row["observation_months"]),
        })
    db.bulk_insert_mappings(Customer, customer_records)
    db.commit()

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

    # Feature engineering
    features_df = engineer_features_batch(customers_df, transactions_df)

    # Train engines
    capacity_engine = CapacityEngine()
    capacity_engine.train(features_df, customers_df)

    guardrail_engine = GuardrailEngine()
    guardrail_engine.train(features_df, customers_df)

    # Save models
    os.makedirs("data/models", exist_ok=True)
    capacity_engine.save("data/models/capacity_model.pkl")
    guardrail_engine.save("data/models/guardrail_model.pkl")

    # Score all customers
    intent_engine = IntentEngine()
    scorer = CompositeScorer(intent_engine, capacity_engine, guardrail_engine)

    # SHAP explainer
    explainer = ShapExplainer(
        capacity_model=capacity_engine.model,
        guardrail_model=guardrail_engine.model,
    )

    for _, row in features_df.iterrows():
        features = row.to_dict()
        score_result = scorer.score(features)

        # Get SHAP explanation
        shap_result = explainer.explain(features, model_type="capacity")

        # Enhance explanation with SHAP text
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
    db.commit()

    return GenerateDataResponse(
        message=f"Successfully generated {request.n_customers} customers with {len(transactions_df)} transactions",
        n_customers=request.n_customers,
        n_transactions=len(transactions_df),
        seed=request.seed,
    )

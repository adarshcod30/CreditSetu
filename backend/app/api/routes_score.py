"""
Score and data generation API routes for CreditSetu.
"""

import json
import logging
import math
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.customer import Customer
from ..models.transaction import Transaction
from ..models.score import Score
from ..config import settings
from ..scoring_profile import profile_from_path_or_default
from ..schemas.schemas import (
    ScoreResponse,
    ShapFeature,
    GenerateDataRequest,
    GenerateDataResponse,
    AdhocScoreRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Scoring"])


def _sanitize_shap_features(raw_list: list[dict]) -> list[ShapFeature]:
    """SHAP contributions can carry NaN for unavailable features — JSON can't
    represent NaN, so normalize it to None before building the response model."""
    sanitized = []
    for item in raw_list:
        val = item.get("value")
        if val is not None and isinstance(val, float) and math.isnan(val):
            item = {**item, "value": None}
        sanitized.append(ShapFeature(**item))
    return sanitized


def _parse_json_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [raw]


@router.get("/score/{customer_id}", response_model=ScoreResponse)
def get_score(customer_id: str, db: Session = Depends(get_db)):
    """Get full score breakdown with explainability for one customer."""
    score = (
        db.query(Score)
        .filter(Score.customer_id == customer_id)
        .order_by(Score.id.desc())
        .first()
    )

    if not score:
        raise HTTPException(status_code=404, detail=f"No score found for customer {customer_id}")

    guardrail_reasons = _parse_json_list(score.guardrail_reasons)
    adverse_action_reasons = _parse_json_list(score.adverse_action_reasons)

    shap_contributions: list[ShapFeature] = []
    if score.shap_contributions:
        try:
            shap_contributions = _sanitize_shap_features(json.loads(score.shap_contributions))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    top_features: list[ShapFeature] = []
    if score.top_features:
        try:
            top_features = _sanitize_shap_features(json.loads(score.top_features))
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
        adverse_action_reasons=adverse_action_reasons,
        scored_at=score.scored_at,
    )


@router.post("/score/adhoc", response_model=ScoreResponse)
def score_adhoc(request: AdhocScoreRequest):
    """
    Score a customer's transactions directly from the request body — the
    bring-your-own-data integration path. No demo database or prior seeding
    required, only a customer + their transaction history matching the data
    contract documented in app/pipeline.py. Uses whatever Capacity/Guardrail
    models are currently registered (see POST /api/data/generate, or fit your
    own via CreditIntelligencePipeline.fit()).
    """
    import pandas as pd
    from ..pipeline import CreditIntelligencePipeline

    profile = profile_from_path_or_default(settings.SCORING_PROFILE_PATH)
    pipeline = CreditIntelligencePipeline(profile=profile)

    try:
        pipeline.load_models(settings.CAPACITY_MODEL_PATH, settings.GUARDRAIL_MODEL_PATH)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained models found. Call POST /api/data/generate first, "
                "or train your own via CreditIntelligencePipeline.fit()."
            ),
        )

    customer = request.customer.model_dump()
    transactions = pd.DataFrame([t.model_dump() for t in request.transactions])

    try:
        result = pipeline.score_customer(customer, transactions)
    except Exception as e:
        logger.exception("Adhoc scoring failed for customer %s", customer.get("customer_id"))
        raise HTTPException(status_code=422, detail=f"Could not score the supplied data: {e}")

    return ScoreResponse(
        customer_id=result["customer_id"],
        composite_score=result["composite_score"],
        intent_score=result["intent_score"],
        intent_event_type=result["intent_event_type"],
        intent_event_recency_days=result["intent_event_recency_days"],
        capacity_score=result["capacity_score"],
        capacity_amount=result["capacity_amount"],
        capacity_confidence=result["capacity_confidence"],
        guardrail_score=result["guardrail_score"],
        guardrail_tier=result["guardrail_tier"],
        guardrail_reasons=result["guardrail_reasons"],
        is_qualified_lead=result["is_qualified_lead"],
        suggested_product=result["suggested_product"],
        explanation=result["explanation"],
        shap_contributions=_sanitize_shap_features(result["shap_contributions"]),
        top_features=_sanitize_shap_features(result["top_features"]),
        adverse_action_reasons=result["adverse_action_reasons"],
        scored_at=None,
    )


@router.post("/data/generate", response_model=GenerateDataResponse)
def generate_data(
    request: GenerateDataRequest,
    db: Session = Depends(get_db),
):
    """
    Regenerate synthetic demo dataset and (re)fit the Capacity/Guardrail
    models on it.

    WARNING: This wipes all existing data and repopulates.
    """
    import os
    from ..data_generation.synthetic_customer_generator import generate_customers
    from ..data_generation.synthetic_transaction_generator import generate_all_transactions
    from ..features.feature_engineering import engineer_features_batch
    from ..pipeline import CreditIntelligencePipeline

    logger.info(f"Generating {request.n_customers} customers with seed={request.seed}")

    # Wipe existing data
    db.query(Score).delete()
    db.query(Transaction).delete()
    db.query(Customer).delete()
    db.commit()

    customers_df = generate_customers(n_customers=request.n_customers, seed=request.seed)
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

    # Feature engineering — computed once, reused for both fit and score_batch below.
    features_df = engineer_features_batch(customers_df, transactions_df)

    profile = profile_from_path_or_default(settings.SCORING_PROFILE_PATH)
    pipeline = CreditIntelligencePipeline(profile=profile)
    pipeline.fit_from_features(features_df, customers_df)

    os.makedirs("data/models", exist_ok=True)
    pipeline.capacity_engine.save(settings.CAPACITY_MODEL_PATH)
    pipeline.guardrail_engine.save(settings.GUARDRAIL_MODEL_PATH)

    # Score all customers — vectorized across the whole batch, not once per row.
    scores_df = pipeline.score_batch(customers_df, transactions_df, features_df=features_df)

    for _, row in scores_df.iterrows():
        score_record = Score(
            customer_id=row["customer_id"],
            intent_score=row["intent_score"],
            intent_event_type=row["intent_event_type"],
            intent_event_recency_days=row["intent_event_recency_days"],
            capacity_score=row["capacity_score"],
            capacity_amount=row["capacity_amount"],
            capacity_confidence=row["capacity_confidence"],
            guardrail_score=row["guardrail_score"],
            guardrail_tier=row["guardrail_tier"],
            guardrail_reasons=json.dumps(row["guardrail_reasons"]),
            composite_score=row["composite_score"],
            is_qualified_lead=bool(row["is_qualified_lead"]),
            suggested_product=row["suggested_product"],
            explanation=row["explanation"],
            shap_contributions=json.dumps(row["shap_contributions"]),
            top_features=json.dumps(row["top_features"]),
            adverse_action_reasons=json.dumps(row["adverse_action_reasons"]),
        )
        db.add(score_record)
    db.commit()

    # Free up memory explicitly
    n_txns = len(transactions_df)
    del transactions_df
    del customers_df
    del features_df
    import gc
    gc.collect()

    return GenerateDataResponse(
        message=f"Successfully generated {request.n_customers} customers with {n_txns} transactions",
        n_customers=request.n_customers,
        n_transactions=n_txns,
        seed=request.seed,
    )

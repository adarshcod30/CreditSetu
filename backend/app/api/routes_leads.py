"""
Lead API routes for CreditSetu.
"""

import json
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..database import get_db
from ..models.customer import Customer
from ..models.score import Score
from ..schemas.schemas import LeadResponse, LeadsListResponse, DashboardStats

router = APIRouter(prefix="/api/leads", tags=["Leads"])


@router.get("", response_model=LeadsListResponse)
def get_leads(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    product_type: str = Query(None),
    exclude_suppressed: bool = Query(True),
    db: Session = Depends(get_db),
):
    """
    Get ranked, filterable list of qualified leads.

    Leads are ranked by composite score descending. By default, suppressed
    customers are excluded. Set exclude_suppressed=false to show them
    (useful for demo transparency).
    """
    # Join Customer and their latest Score
    query = (
        db.query(Customer, Score)
        .join(Score, Customer.customer_id == Score.customer_id)
        .filter(Score.composite_score >= min_score)
    )

    if exclude_suppressed:
        query = query.filter(Score.guardrail_tier != "Suppress")

    if product_type:
        query = query.filter(Score.suggested_product == product_type)

    # Get only the latest score per customer using a subquery
    from sqlalchemy import func
    latest_scores = (
        db.query(
            Score.customer_id,
            func.max(Score.id).label("max_id"),
        )
        .group_by(Score.customer_id)
        .subquery()
    )

    query = (
        db.query(Customer, Score)
        .join(latest_scores, Customer.customer_id == latest_scores.c.customer_id)
        .join(Score, Score.id == latest_scores.c.max_id)
        .filter(Score.composite_score >= min_score)
    )

    if exclude_suppressed:
        query = query.filter(Score.guardrail_tier != "Suppress")

    if product_type:
        query = query.filter(Score.suggested_product == product_type)

    total = query.count()

    results = (
        query
        .order_by(desc(Score.composite_score))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    leads = []
    for customer, score in results:
        guardrail_reasons = []
        if score.guardrail_reasons:
            try:
                guardrail_reasons = json.loads(score.guardrail_reasons)
            except (json.JSONDecodeError, TypeError):
                guardrail_reasons = [score.guardrail_reasons]

        leads.append(LeadResponse(
            customer_id=customer.customer_id,
            name=customer.name,
            age=customer.age,
            occupation=customer.occupation,
            persona_type=customer.persona_type,
            city=customer.city,
            bureau_score=customer.bureau_score,
            composite_score=score.composite_score,
            intent_score=score.intent_score,
            capacity_score=score.capacity_score,
            guardrail_tier=score.guardrail_tier,
            guardrail_reasons=guardrail_reasons,
            is_qualified_lead=score.is_qualified_lead,
            suggested_product=score.suggested_product,
            explanation=score.explanation or "",
            scored_at=score.scored_at,
        ))

    return LeadsListResponse(
        leads=leads,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/stats", response_model=DashboardStats)
def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get summary statistics for the dashboard header."""
    from sqlalchemy import func

    # Latest scores subquery
    latest_scores = (
        db.query(
            Score.customer_id,
            func.max(Score.id).label("max_id"),
        )
        .group_by(Score.customer_id)
        .subquery()
    )

    scores = (
        db.query(Score)
        .join(latest_scores, Score.id == latest_scores.c.max_id)
        .all()
    )

    total_customers = len(scores)
    total_leads = sum(1 for s in scores if s.is_qualified_lead)
    suppressed = sum(1 for s in scores if s.guardrail_tier == "Suppress")
    watch = sum(1 for s in scores if s.guardrail_tier == "Watch")
    safe = sum(1 for s in scores if s.guardrail_tier == "Safe")
    avg_score = sum(s.composite_score for s in scores) / total_customers if total_customers > 0 else 0

    product_dist = {}
    for s in scores:
        if s.is_qualified_lead and s.suggested_product:
            product_dist[s.suggested_product] = product_dist.get(s.suggested_product, 0) + 1

    return DashboardStats(
        total_customers=total_customers,
        total_leads=total_leads,
        suppressed_count=suppressed,
        watch_count=watch,
        safe_count=safe,
        avg_composite_score=round(avg_score, 4),
        product_distribution=product_dist,
    )

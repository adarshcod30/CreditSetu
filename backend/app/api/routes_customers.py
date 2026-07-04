"""
Customer API routes for CreditSetu.
"""

import json
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.customer import Customer
from ..models.transaction import Transaction
from ..schemas.schemas import (
    CustomerSummary,
    CustomerDetail,
    CustomerListResponse,
    TransactionResponse,
)

router = APIRouter(prefix="/api/customers", tags=["Customers"])


@router.get("", response_model=CustomerListResponse)
def list_customers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    persona_type: str = Query(None),
    db: Session = Depends(get_db),
):
    """List customers with pagination and optional persona filter."""
    query = db.query(Customer)

    if persona_type:
        query = query.filter(Customer.persona_type == persona_type)

    total = query.count()
    customers = query.offset((page - 1) * page_size).limit(page_size).all()

    return CustomerListResponse(
        customers=[
            CustomerSummary(
                customer_id=c.customer_id,
                name=c.name,
                age=c.age,
                gender=c.gender,
                occupation=c.occupation,
                persona_type=c.persona_type,
                bureau_score=c.bureau_score,
                city=c.city,
                monthly_income=c.monthly_income,
            )
            for c in customers
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{customer_id}", response_model=CustomerDetail)
def get_customer(customer_id: str, db: Session = Depends(get_db)):
    """Get full customer profile including transaction history."""
    customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    transactions = (
        db.query(Transaction)
        .filter(Transaction.customer_id == customer_id)
        .order_by(Transaction.date.desc())
        .all()
    )

    return CustomerDetail(
        customer_id=customer.customer_id,
        name=customer.name,
        age=customer.age,
        gender=customer.gender,
        occupation=customer.occupation,
        persona_type=customer.persona_type,
        bureau_score=customer.bureau_score,
        city=customer.city,
        account_open_date=customer.account_open_date,
        monthly_income=customer.monthly_income,
        emi_count=customer.emi_count,
        total_emi=customer.total_emi,
        observation_months=customer.observation_months,
        transactions=[
            TransactionResponse(
                txn_id=t.txn_id,
                customer_id=t.customer_id,
                date=t.date,
                amount=t.amount,
                type=t.type,
                category=t.category,
                counterparty=t.counterparty,
                channel=t.channel,
                narration=t.narration,
                is_bounce=t.is_bounce,
            )
            for t in transactions
        ],
    )

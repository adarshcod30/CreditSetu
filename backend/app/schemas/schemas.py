"""
Pydantic v2 request/response schemas for CreditSetu API.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ─── Request Schemas ─────────────────────────────────────────────────────────────

class GenerateDataRequest(BaseModel):
    """Request body for POST /api/data/generate"""
    n_customers: int = Field(default=1000, ge=10, le=5000, description="Number of customers to generate")
    seed: int = Field(default=42, description="Random seed for reproducibility")


class BenchmarkRunRequest(BaseModel):
    """Request body for POST /api/benchmark/run"""
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)


# ─── Response Schemas ────────────────────────────────────────────────────────────

class CustomerSummary(BaseModel):
    """Brief customer info for list views."""
    customer_id: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    persona_type: str
    bureau_score: Optional[float] = None
    city: Optional[str] = None
    monthly_income: Optional[float] = None

    model_config = {"from_attributes": True}


class TransactionResponse(BaseModel):
    """Single transaction record."""
    txn_id: str
    customer_id: str
    date: str
    amount: float
    type: str
    category: Optional[str] = None
    counterparty: Optional[str] = None
    channel: Optional[str] = None
    narration: Optional[str] = None
    is_bounce: bool = False

    model_config = {"from_attributes": True}


class CustomerDetail(BaseModel):
    """Full customer profile with transactions."""
    customer_id: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    persona_type: str
    bureau_score: Optional[float] = None
    city: Optional[str] = None
    account_open_date: Optional[str] = None
    monthly_income: Optional[float] = None
    emi_count: Optional[int] = None
    total_emi: Optional[float] = None
    observation_months: Optional[int] = None
    transactions: list[TransactionResponse] = []

    model_config = {"from_attributes": True}


class ShapFeature(BaseModel):
    """Single SHAP feature contribution."""
    feature: str
    display_name: str
    value: Optional[float] = None
    contribution: float


class ScoreResponse(BaseModel):
    """Full score breakdown with explainability."""
    customer_id: str
    composite_score: float
    intent_score: float
    intent_event_type: Optional[str] = None
    intent_event_recency_days: Optional[int] = None
    capacity_score: float
    capacity_amount: float
    capacity_confidence: float
    guardrail_score: float
    guardrail_tier: str
    guardrail_reasons: list[str] = []
    is_qualified_lead: bool
    suggested_product: str
    explanation: str
    shap_contributions: list[ShapFeature] = []
    top_features: list[ShapFeature] = []
    scored_at: Optional[str] = None

    model_config = {"from_attributes": True}


class LeadResponse(BaseModel):
    """Lead entry for the dashboard table."""
    customer_id: str
    name: str
    age: Optional[int] = None
    occupation: Optional[str] = None
    persona_type: str
    city: Optional[str] = None
    bureau_score: Optional[float] = None
    composite_score: float
    intent_score: float
    capacity_score: float
    guardrail_tier: str
    guardrail_reasons: list[str] = []
    is_qualified_lead: bool
    suggested_product: str
    explanation: str
    scored_at: Optional[str] = None


class LeadsListResponse(BaseModel):
    """Paginated lead list response."""
    leads: list[LeadResponse]
    total: int
    page: int
    page_size: int


class CustomerListResponse(BaseModel):
    """Paginated customer list response."""
    customers: list[CustomerSummary]
    total: int
    page: int
    page_size: int


class BenchmarkReport(BaseModel):
    """Benchmark evaluation report."""
    capacity_engine: dict = {}
    intent_engine: dict = {}
    guardrail_engine: dict = {}
    composite: dict = {}
    latency: dict = {}
    generated_at: Optional[str] = None


class GenerateDataResponse(BaseModel):
    """Response for data generation endpoint."""
    message: str
    n_customers: int
    n_transactions: int
    seed: int


class DashboardStats(BaseModel):
    """Summary statistics for dashboard."""
    total_customers: int
    total_leads: int
    suppressed_count: int
    watch_count: int
    safe_count: int
    avg_composite_score: float
    product_distribution: dict = {}

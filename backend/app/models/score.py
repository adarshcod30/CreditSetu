"""
Score ORM model for CreditSetu.
"""

from sqlalchemy import Column, Integer, String, Float, Text, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from ..database import Base


class Score(Base):
    """
    Score record model.

    Stores the composite score and all sub-scores for a customer.
    One-to-many from Customer — allows tracking score history over time.
    """

    __tablename__ = "scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(20), ForeignKey("customers.customer_id"), nullable=False, index=True)
    scored_at = Column(String(30), default=lambda: datetime.utcnow().isoformat())

    # Sub-scores
    intent_score = Column(Float)
    intent_event_type = Column(String(50), nullable=True)
    intent_event_recency_days = Column(Integer, nullable=True)

    capacity_score = Column(Float)
    capacity_amount = Column(Float)
    capacity_confidence = Column(Float)

    guardrail_score = Column(Float)
    guardrail_tier = Column(String(20))  # Safe, Watch, Suppress
    guardrail_reasons = Column(Text)  # JSON string

    # Composite
    composite_score = Column(Float)
    is_qualified_lead = Column(Boolean)
    suggested_product = Column(String(50))

    # Explainability
    explanation = Column(Text)
    shap_contributions = Column(Text)  # JSON string
    top_features = Column(Text)  # JSON string

    # Relationship
    customer = relationship("Customer", back_populates="scores")

    def __repr__(self):
        return f"<Score {self.customer_id}: {self.composite_score:.3f} ({self.guardrail_tier})>"

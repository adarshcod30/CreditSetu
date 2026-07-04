"""
Customer ORM model for CreditSetu.
"""

from sqlalchemy import Column, Integer, String, Float, Date, Text, Boolean
from sqlalchemy.orm import relationship

from ..database import Base


class Customer(Base):
    """
    Customer profile model.

    Stores demographic and financial profile data.
    One-to-many relationship with Transaction and Score.
    """

    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer)
    gender = Column(String(1))
    occupation = Column(String(100))
    persona_type = Column(String(50), nullable=False)
    bureau_score = Column(Float, nullable=True)  # Nullable for NTC/gig personas
    city = Column(String(50))
    account_open_date = Column(String(20))
    monthly_income = Column(Float)
    emi_count = Column(Integer, default=0)
    total_emi = Column(Float, default=0)
    true_repayment_capacity = Column(Float)  # Ground-truth target for ML
    life_events = Column(Text)  # JSON string of life-event labels
    observation_months = Column(Integer)

    # Relationships
    transactions = relationship("Transaction", back_populates="customer", cascade="all, delete-orphan")
    scores = relationship("Score", back_populates="customer", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Customer {self.customer_id}: {self.name} ({self.persona_type})>"

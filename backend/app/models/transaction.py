"""
Transaction ORM model for CreditSetu.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship

from ..database import Base


class Transaction(Base):
    """
    Transaction record model.

    Structurally mirrors Account Aggregator Deposit FI type transaction data.
    In production, these would come from AA-integrated FIP data.
    """

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    txn_id = Column(String(30), unique=True, nullable=False, index=True)
    customer_id = Column(String(20), ForeignKey("customers.customer_id"), nullable=False, index=True)
    date = Column(String(20), nullable=False)  # ISO format date string
    amount = Column(Float, nullable=False)
    type = Column(String(10), nullable=False)  # 'credit' or 'debit'
    category = Column(String(50))  # salary, emi, rent, groceries, etc.
    counterparty = Column(String(100))
    channel = Column(String(10))  # UPI, NEFT, NACH
    narration = Column(String(200))
    is_bounce = Column(Boolean, default=False)

    # Relationship
    customer = relationship("Customer", back_populates="transactions")

    def __repr__(self):
        return f"<Transaction {self.txn_id}: {self.type} ₹{self.amount}>"

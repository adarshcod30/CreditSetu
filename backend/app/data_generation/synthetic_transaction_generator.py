"""
Synthetic transaction generator for CreditSetu.

Generates 6-12 months of daily-level transaction data per customer that structurally
mirrors Account Aggregator / UPI transaction data.

IMPORTANT: All transactions are SYNTHETIC. In production, this data would come from
IDBI Bank's Account Aggregator integration (AA framework by ReBIT/Sahamati).
The transaction schema is designed to match the AA Deposit FI type schema:
- transaction_type (Debit/Credit)
- amount
- narration (mirrors the free-text narration field in AA data)
- counterparty (extracted from narration in production)
- category (would be derived from narration parsing in production)

Deterministic via numpy random seed for reproducible results.
"""

import uuid
from datetime import date, timedelta, datetime
from typing import Optional

import numpy as np
import pandas as pd

from .personas import MERCHANT_CATEGORIES


def generate_transactions(
    customer_row: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic transactions for a single customer.

    Args:
        customer_row: Dictionary with customer profile data from customer generator.
        seed: Random seed (combined with customer index for uniqueness).

    Returns:
        DataFrame of transactions with columns: txn_id, customer_id, date, amount,
        type, category, counterparty, channel, narration, is_bounce
    """
    # Create a unique seed per customer by combining base seed with customer ID
    customer_index = int(customer_row["customer_id"].split("-")[1])
    rng = np.random.default_rng(seed + customer_index)

    obs_months = customer_row["observation_months"]
    start_date = date.fromisoformat(customer_row["account_open_date"])
    end_date = date.today()
    total_days = (end_date - start_date).days

    if total_days <= 0:
        total_days = obs_months * 30
        start_date = end_date - timedelta(days=total_days)

    transactions = []

    # Generate each transaction type
    transactions.extend(_generate_income_credits(customer_row, start_date, total_days, rng))
    transactions.extend(_generate_upi_spend(customer_row, start_date, total_days, rng))
    transactions.extend(_generate_rent_debits(customer_row, start_date, total_days, rng))
    transactions.extend(_generate_emi_debits(customer_row, start_date, total_days, rng))
    transactions.extend(_generate_nach_bounces(customer_row, start_date, total_days, rng))

    # Apply life events (modify transaction stream based on ground-truth events)
    transactions = _apply_life_events(transactions, customer_row, start_date, rng)

    # Sort by date
    transactions.sort(key=lambda x: x["date"])

    # Assign transaction IDs
    for i, txn in enumerate(transactions):
        txn["txn_id"] = f"TXN-{customer_index:05d}-{i + 1:06d}"
        txn["customer_id"] = customer_row["customer_id"]

    return pd.DataFrame(transactions)


def _generate_income_credits(
    customer: dict, start_date: date, total_days: int, rng: np.random.Generator
) -> list[dict]:
    """Generate income credit transactions based on persona type."""
    transactions = []
    monthly_income = customer["monthly_income"]
    variability = customer["income_variability"]
    persona = customer["persona_type"]

    if persona in ("salaried_stable", "new_to_credit", "over_leveraged"):
        # Monthly salary credits — on 1st or 28th with small variation
        salary_day = int(rng.choice([1, 28]))
        for month_offset in range(int(customer["observation_months"])):
            txn_date = start_date + timedelta(days=int(month_offset * 30 + salary_day))
            if txn_date > date.today():
                break

            # Small variation in salary amount
            amount = int(monthly_income * (1 + rng.normal(0, variability)))
            amount = max(amount, int(monthly_income * 0.8))  # floor

            source = rng.choice(["salary.employer@icici", "payroll.company@hdfcbank",
                                 "salary.corp@sbi", "hr.payroll@axisbank"])
            transactions.append({
                "date": txn_date.isoformat(),
                "amount": amount,
                "type": "credit",
                "category": "salary",
                "counterparty": source,
                "channel": "NEFT",
                "narration": f"NEFT-SALARY-{txn_date.strftime('%b').upper()}-{source.split('.')[0].upper()}",
                "is_bounce": False,
            })

    elif persona == "gig_worker":
        # Weekly/bi-weekly variable payouts from platform VPAs
        platforms = rng.choice(
            ["Swiggy Payments", "Zomato Pay", "Uber India", "Ola Money", "Dunzo", "Rapido"],
            size=int(rng.integers(2, 4)),
            replace=False,
        ).tolist()

        weekly_income = monthly_income / 4.0
        for day in range(0, total_days, int(rng.integers(3, 8))):
            txn_date = start_date + timedelta(days=int(day))
            if txn_date > date.today():
                break

            platform = rng.choice(platforms)
            # Highly variable amount
            amount = int(weekly_income / 2 * (1 + rng.normal(0, variability)))
            amount = max(amount, 500)  # minimum payout

            vpa = platform.lower().replace(" ", ".") + "@icici"
            transactions.append({
                "date": txn_date.isoformat(),
                "amount": amount,
                "type": "credit",
                "category": "gig_payout",
                "counterparty": vpa,
                "channel": "UPI",
                "narration": f"UPI-{platform.upper()}-PAYOUT-{txn_date.strftime('%d%b')}",
                "is_bounce": False,
            })

    elif persona == "self_employed":
        # Irregular merchant collection inflows — multiple small credits
        for day in range(total_days):
            txn_date = start_date + timedelta(days=int(day))
            if txn_date > date.today():
                break

            # 40-70% chance of receiving payment on any given day
            if rng.random() < 0.55:
                daily_avg = monthly_income / 25  # ~25 working days
                amount = int(daily_avg * rng.lognormal(0, 0.6))
                amount = max(amount, 100)

                source = rng.choice([
                    "customer.upi@ybl", "payment@icici", "merchant.settle@hdfcbank",
                    "gpay.collect@sbi", "phonepe.settle@ybl",
                ])
                transactions.append({
                    "date": txn_date.isoformat(),
                    "amount": amount,
                    "type": "credit",
                    "category": "merchant_collection",
                    "counterparty": source,
                    "channel": "UPI",
                    "narration": f"UPI-P2M-COLLECTION-{txn_date.strftime('%d%b')}",
                    "is_bounce": False,
                })

    return transactions


def _generate_upi_spend(
    customer: dict, start_date: date, total_days: int, rng: np.random.Generator
) -> list[dict]:
    """Generate UPI P2M spending transactions across realistic Indian merchant categories."""
    transactions = []
    monthly_income = customer["monthly_income"]
    expense_fraction = customer["expense_fraction"]
    monthly_spend = monthly_income * expense_fraction

    # Distribute spend across categories with realistic proportions
    category_weights = {
        "groceries": 0.25,
        "utilities": 0.12,
        "food_delivery": 0.15,
        "transport": 0.10,
        "entertainment": 0.08,
        "healthcare": 0.05,
        "education": 0.05,
        "shopping": 0.20,
    }

    for day in range(total_days):
        txn_date = start_date + timedelta(days=int(day))
        if txn_date > date.today():
            break

        # 2-5 transactions per day on average
        n_txns = rng.poisson(3)
        for _ in range(n_txns):
            # Pick category based on weights
            categories = list(category_weights.keys())
            weights = list(category_weights.values())
            category = rng.choice(categories, p=weights)

            merchants = MERCHANT_CATEGORIES.get(category, ["Unknown Merchant"])
            merchant = rng.choice(merchants)

            # Transaction amount varies by category
            daily_budget = monthly_spend / 30
            category_daily = daily_budget * category_weights[category] * 30 / max(n_txns, 1)

            if category in ("groceries", "shopping"):
                amount = int(rng.lognormal(np.log(category_daily * 0.8), 0.5))
            elif category in ("utilities",):
                # Utilities are less frequent but higher amount — handle monthly
                if rng.random() < 0.03:  # ~once/month
                    amount = int(monthly_spend * category_weights[category] * rng.uniform(0.8, 1.2))
                else:
                    continue
            else:
                amount = int(rng.lognormal(np.log(max(category_daily * 0.5, 50)), 0.4))

            amount = max(amount, 10)
            amount = min(amount, int(monthly_spend * 0.3))  # cap individual txn

            transactions.append({
                "date": txn_date.isoformat(),
                "amount": amount,
                "type": "debit",
                "category": category,
                "counterparty": merchant,
                "channel": "UPI",
                "narration": f"UPI-P2M-{merchant.upper().replace(' ', '-')[:20]}",
                "is_bounce": False,
            })

    return transactions


def _generate_rent_debits(
    customer: dict, start_date: date, total_days: int, rng: np.random.Generator
) -> list[dict]:
    """Generate monthly rent payments for applicable customers."""
    transactions = []
    if not customer["has_rent"] or customer["rent_amount"] <= 0:
        return transactions

    rent_amount = customer["rent_amount"]
    rent_day = int(rng.integers(1, 6))  # rent typically paid 1st-5th

    for month_offset in range(int(customer["observation_months"])):
        txn_date = start_date + timedelta(days=int(month_offset * 30 + rent_day))
        if txn_date > date.today():
            break

        # Small variation in rent (usually very consistent)
        amount = int(rent_amount * rng.uniform(0.98, 1.02))

        transactions.append({
            "date": txn_date.isoformat(),
            "amount": amount,
            "type": "debit",
            "category": "rent",
            "counterparty": "landlord.rent@ybl",
            "channel": "UPI",
            "narration": f"UPI-RENT-{txn_date.strftime('%b').upper()}-MONTHLY",
            "is_bounce": False,
        })

    return transactions


def _generate_emi_debits(
    customer: dict, start_date: date, total_days: int, rng: np.random.Generator
) -> list[dict]:
    """Generate EMI/loan debit transactions."""
    transactions = []
    emi_count = customer["emi_count"]
    if emi_count == 0:
        return transactions

    emi_amounts = customer["emi_amounts"]
    lenders = customer["selected_lenders"]

    for emi_idx in range(emi_count):
        emi_amount = emi_amounts[emi_idx]
        lender = lenders[emi_idx]
        emi_day = int(rng.integers(5, 15))  # EMI dates typically 5th-15th

        for month_offset in range(int(customer["observation_months"])):
            txn_date = start_date + timedelta(days=int(month_offset * 30 + emi_day))
            if txn_date > date.today():
                break

            # EMI amounts are very consistent
            amount = int(emi_amount * rng.uniform(0.99, 1.01))

            transactions.append({
                "date": txn_date.isoformat(),
                "amount": amount,
                "type": "debit",
                "category": "emi",
                "counterparty": lender,
                "channel": "NACH",
                "narration": f"NACH-EMI-{lender.replace(' ', '-').upper()}-{txn_date.strftime('%b%y')}",
                "is_bounce": False,
            })

    return transactions


def _generate_nach_bounces(
    customer: dict, start_date: date, total_days: int, rng: np.random.Generator
) -> list[dict]:
    """Generate NACH bounce events (failed debit returns) for stressed customers."""
    transactions = []
    bounce_prob = customer["bounce_probability"]
    if bounce_prob <= 0 or customer["emi_count"] == 0:
        return transactions

    lenders = customer["selected_lenders"]

    for month_offset in range(int(customer["observation_months"])):
        if rng.random() < bounce_prob:
            # Bounce on one of the EMI dates
            lender = rng.choice(lenders)
            bounce_day = int(rng.integers(5, 15))
            txn_date = start_date + timedelta(days=int(month_offset * 30 + bounce_day))
            if txn_date > date.today():
                break

            # Bounce amount matches the EMI
            lender_idx = lenders.index(lender) if lender in lenders else 0
            amount = customer["emi_amounts"][lender_idx] if lender_idx < len(customer["emi_amounts"]) else 5000

            transactions.append({
                "date": txn_date.isoformat(),
                "amount": amount,
                "type": "debit",
                "category": "nach_bounce",
                "counterparty": lender,
                "channel": "NACH",
                "narration": f"NACH-RETURN-INSUFFICIENT-FUNDS-{lender.replace(' ', '-').upper()}",
                "is_bounce": True,
            })

    return transactions


def _apply_life_events(
    transactions: list[dict],
    customer: dict,
    start_date: date,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Modify the transaction stream to inject life events.

    This creates the actual observable pattern in transaction data that the
    Intent Engine's change-point detection should be able to find.
    """
    life_events = customer.get("life_events", [])
    if not life_events:
        return transactions

    for event in life_events:
        event_type = event["event_type"]
        day_offset = event["day_offset"]
        event_date = start_date + timedelta(days=int(day_offset))
        event_date_str = event_date.isoformat()

        if event_type == "emi_closure":
            # Remove EMI debits after the event date (simulate EMI being paid off)
            if customer["emi_count"] > 0:
                # Close the first EMI
                closed_lender = customer["selected_lenders"][0]
                transactions = [
                    t for t in transactions
                    if not (
                        t["category"] == "emi"
                        and t["counterparty"] == closed_lender
                        and t["date"] >= event_date_str
                    )
                ]

        elif event_type == "income_step_up":
            # Increase income credits after the event date by 20-40%
            step_up_factor = rng.uniform(1.20, 1.40)
            for txn in transactions:
                if (txn["type"] == "credit"
                        and txn["category"] in ("salary", "gig_payout", "merchant_collection")
                        and txn["date"] >= event_date_str):
                    txn["amount"] = int(txn["amount"] * step_up_factor)

        elif event_type == "new_commitment":
            # Add a new regular debit after the event date
            commitment_amount = int(customer["monthly_income"] * rng.uniform(0.05, 0.12))
            for month_offset in range(int(customer["observation_months"])):
                commit_date = start_date + timedelta(days=int(month_offset * 30 + 10))
                if commit_date >= event_date and commit_date <= date.today():
                    transactions.append({
                        "date": commit_date.isoformat(),
                        "amount": commitment_amount,
                        "type": "debit",
                        "category": "emi",
                        "counterparty": "New Finance Ltd EMI",
                        "channel": "NACH",
                        "narration": f"NACH-EMI-NEW-FINANCE-{commit_date.strftime('%b%y')}",
                        "is_bounce": False,
                    })

    return transactions


def generate_all_transactions(
    customers_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate transactions for all customers.

    Args:
        customers_df: DataFrame from generate_customers()
        seed: Base random seed

    Returns:
        DataFrame of all transactions across all customers
    """
    all_txns = []
    for _, row in customers_df.iterrows():
        customer_txns = generate_transactions(row.to_dict(), seed=seed)
        all_txns.append(customer_txns)

    if all_txns:
        return pd.concat(all_txns, ignore_index=True)
    return pd.DataFrame()

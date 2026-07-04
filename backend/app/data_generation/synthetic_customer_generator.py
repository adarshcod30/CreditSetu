"""
Synthetic customer generator for CreditSetu.

Generates realistic customer profiles across 5 persona archetypes that structurally
mirror what an Indian bank would see via Account Aggregator integration.

IMPORTANT: All data generated here is SYNTHETIC. In production, this module would be
replaced by actual AA-integrated customer data from IDBI Bank's systems. The synthetic
data is designed to be structurally equivalent for prototype demonstration purposes.

Deterministic via numpy random seed for reproducible demo results.
"""

import uuid
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .personas import (
    ALL_PERSONAS,
    INDIAN_CITIES,
    PersonaConfig,
)

# Indian first names and last names for realistic customer names
FIRST_NAMES_MALE = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan",
    "Krishna", "Ishaan", "Shaurya", "Atharva", "Advait", "Dhruv", "Kabir",
    "Ritvik", "Aarush", "Kian", "Darsh", "Laksh", "Pranav", "Rohan", "Nikhil",
    "Rahul", "Amit", "Vikram", "Suresh", "Rajesh", "Manoj", "Deepak",
    "Sunil", "Arun", "Vijay", "Prakash", "Sandeep", "Ravi", "Ajay",
    "Anand", "Gaurav", "Harsh", "Kunal", "Mohit", "Naveen", "Pankaj",
]

FIRST_NAMES_FEMALE = [
    "Aadhya", "Diya", "Saanvi", "Ananya", "Myra", "Aaradhya", "Anika",
    "Pari", "Kiara", "Isha", "Kavya", "Riya", "Sneha", "Priya", "Neha",
    "Pooja", "Swati", "Anjali", "Divya", "Shruti", "Meera", "Nisha",
    "Pallavi", "Rashmi", "Tanvi", "Sakshi", "Komal", "Mansi", "Nikita",
    "Sonia", "Aishwarya", "Lakshmi", "Deepika", "Sunita", "Jyoti",
]

LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Gupta", "Singh", "Kumar", "Joshi",
    "Reddy", "Nair", "Iyer", "Pillai", "Rao", "Desai", "Shah",
    "Mehta", "Chopra", "Malhotra", "Bhat", "Hegde", "Shetty",
    "Mishra", "Pandey", "Saxena", "Agarwal", "Banerjee", "Mukherjee",
    "Chatterjee", "Das", "Bose", "Sen", "Kulkarni", "Patil", "Naik",
    "Kaur", "Gill", "Sethi", "Kapoor", "Bajaj", "Tiwari", "Dubey",
]


def generate_customers(
    n_customers: int = 1000,
    seed: int = 42,
    observation_months: int = 9,
) -> pd.DataFrame:
    """
    Generate synthetic customer profiles.

    Args:
        n_customers: Number of customers to generate.
        seed: Random seed for reproducibility.
        observation_months: Average months of transaction history (6-12 range).

    Returns:
        DataFrame with columns: customer_id, name, age, gender, occupation,
        persona_type, bureau_score, city, account_open_date,
        monthly_income, true_repayment_capacity, life_events,
        observation_months
    """
    rng = np.random.default_rng(seed)
    customers = []

    # Assign personas according to proportions
    persona_assignments = _assign_personas(n_customers, rng)

    for i, persona in enumerate(persona_assignments):
        customer = _generate_single_customer(i, persona, rng, observation_months)
        customers.append(customer)

    df = pd.DataFrame(customers)
    return df


def _assign_personas(n_customers: int, rng: np.random.Generator) -> list[PersonaConfig]:
    """Assign personas to customers according to configured proportions."""
    assignments = []
    remaining = n_customers

    for j, persona in enumerate(ALL_PERSONAS):
        if j == len(ALL_PERSONAS) - 1:
            # Last persona gets the remainder to avoid rounding issues
            count = remaining
        else:
            count = int(round(persona.proportion * n_customers))
            remaining -= count
        assignments.extend([persona] * count)

    # Shuffle so personas aren't grouped together
    rng.shuffle(assignments)
    return assignments


def _generate_single_customer(
    index: int,
    persona: PersonaConfig,
    rng: np.random.Generator,
    avg_observation_months: int,
) -> dict:
    """Generate a single customer profile based on persona config."""
    # Basic demographics
    gender = rng.choice(["M", "F"], p=[0.55, 0.45])
    if gender == "M":
        first_name = rng.choice(FIRST_NAMES_MALE)
    else:
        first_name = rng.choice(FIRST_NAMES_FEMALE)
    last_name = rng.choice(LAST_NAMES)
    name = f"{first_name} {last_name}"

    age = int(rng.integers(persona.age_range[0], persona.age_range[1] + 1))
    occupation = rng.choice(persona.occupations)
    city = rng.choice(INDIAN_CITIES)

    # Bureau score
    bureau_score = None
    if rng.random() < persona.bureau_score_probability and persona.bureau_score_range:
        bureau_score = int(rng.integers(
            persona.bureau_score_range[0],
            persona.bureau_score_range[1] + 1,
        ))

    # Monthly income (drawn from range with some noise)
    monthly_income = int(rng.integers(
        persona.monthly_income_range[0],
        persona.monthly_income_range[1] + 1,
    ))

    # EMI count for this customer
    emi_count = int(rng.integers(
        persona.emi_count_range[0],
        persona.emi_count_range[1] + 1,
    ))

    # Per-EMI amount
    emi_amounts = []
    selected_lenders = []
    if emi_count > 0:
        available_lenders = list(persona.lender_names)
        rng.shuffle(available_lenders)
        selected_lenders = available_lenders[:emi_count]
        for _ in range(emi_count):
            fraction = rng.uniform(*persona.emi_amount_fraction_range)
            emi_amounts.append(int(monthly_income * fraction))

    total_emi = sum(emi_amounts)

    # Observation period (6-12 months)
    obs_months = int(rng.integers(6, 13))
    account_open_date = date.today() - timedelta(days=obs_months * 30)

    # True repayment capacity — the ground-truth target for the Capacity Engine.
    # In production, this would be replaced by actual observed repayment behavior.
    # Formula: fraction of (income - existing obligations) with stability factor + noise
    stability_factor = 1.0 - persona.income_variability  # more stable = higher capacity
    disposable = max(monthly_income - total_emi, 0)
    # Safe repayment = 30-40% of disposable income, adjusted by stability
    safe_fraction = rng.uniform(0.25, 0.40)
    capacity_base = disposable * safe_fraction * stability_factor
    noise = rng.normal(0, capacity_base * 0.05)  # ±5% noise
    true_repayment_capacity = max(int(capacity_base + noise), 0)

    # Life events (ground-truth labels for Intent Engine validation)
    life_events = _generate_life_events(persona, obs_months, rng)

    # Customer ID
    customer_id = f"CUST-{index + 1:05d}"

    return {
        "customer_id": customer_id,
        "name": name,
        "age": age,
        "gender": gender,
        "occupation": occupation,
        "persona_type": persona.name,
        "bureau_score": bureau_score,
        "city": city,
        "account_open_date": account_open_date.isoformat(),
        "monthly_income": monthly_income,
        "emi_count": emi_count,
        "emi_amounts": emi_amounts,
        "selected_lenders": selected_lenders,
        "total_emi": total_emi,
        "true_repayment_capacity": true_repayment_capacity,
        "life_events": life_events,
        "observation_months": obs_months,
        "has_rent": bool(rng.random() < persona.has_rent_probability),
        "rent_amount": int(monthly_income * rng.uniform(*persona.rent_fraction_range)) if rng.random() < persona.has_rent_probability else 0,
        "bounce_probability": persona.bounce_probability_per_month,
        "income_variability": persona.income_variability,
        "expense_fraction": float(rng.uniform(*persona.monthly_expense_fraction)),
    }


def _generate_life_events(
    persona: PersonaConfig,
    obs_months: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Generate ground-truth life-event labels for a customer.

    These are used both to inject realistic events into the transaction stream
    AND to validate the Intent Engine's detection accuracy.
    """
    events = []
    total_days = obs_months * 30

    for event_type, probability in persona.life_event_probabilities.items():
        if probability > 0 and rng.random() < probability:
            # Place the event somewhere in the observation window
            # Bias toward more recent events (more interesting for demo)
            day_offset = int(rng.beta(2, 5) * total_days)
            # Ensure it's not on the very first or last day
            day_offset = max(14, min(day_offset, total_days - 14))

            events.append({
                "event_type": event_type,
                "day_offset": day_offset,
                "description": _event_description(event_type),
            })

    return events


def _event_description(event_type: str) -> str:
    """Human-readable description of a life event."""
    descriptions = {
        "emi_closure": "Customer closed an existing EMI — freed up repayment capacity",
        "income_step_up": "Significant sustained increase in income detected",
        "new_commitment": "New regular financial commitment detected (e.g., new EMI or subscription)",
    }
    return descriptions.get(event_type, f"Unknown event: {event_type}")

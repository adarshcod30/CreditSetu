"""
Persona definitions for synthetic customer generation.

Each persona represents a distinct customer archetype commonly seen in Indian retail
banking. The proportions reflect realistic population distributions — not an even split.

NOTE: These personas and their parameters are entirely synthetic, designed to
structurally mirror the kinds of customers IDBI Bank would serve. In production,
persona segmentation would be derived from actual Account Aggregator data.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PersonaConfig:
    """Configuration for a single customer persona archetype."""

    name: str
    proportion: float  # fraction of total population
    description: str

    # Income parameters
    monthly_income_range: tuple[int, int]  # (min, max) in INR
    income_frequency: str  # 'monthly', 'weekly', 'biweekly', 'irregular'
    income_variability: float  # CV of income amount (0 = perfectly stable)
    income_sources: list[str] = field(default_factory=list)

    # Bureau parameters
    bureau_score_range: Optional[tuple[int, int]] = None  # None = thin/no file
    bureau_score_probability: float = 0.0  # probability of having a bureau score

    # Demographics
    age_range: tuple[int, int] = (25, 55)
    occupations: list[str] = field(default_factory=list)

    # Loan/EMI parameters
    emi_count_range: tuple[int, int] = (0, 0)  # (min, max) concurrent EMIs
    emi_amount_fraction_range: tuple[float, float] = (0.05, 0.15)  # fraction of income per EMI
    lender_names: list[str] = field(default_factory=list)

    # NACH bounce parameters
    bounce_probability_per_month: float = 0.0  # probability of a bounce event per month

    # Rent parameters
    has_rent_probability: float = 0.5
    rent_fraction_range: tuple[float, float] = (0.15, 0.35)  # fraction of income

    # Life event parameters
    life_event_probabilities: dict[str, float] = field(default_factory=dict)

    # Spend profile
    monthly_expense_fraction: tuple[float, float] = (0.4, 0.7)  # fraction of income on UPI spend


# ─── Shared Constants ───────────────────────────────────────────────────────────

PLATFORM_VPAS = [
    "swiggy.payments@icici",
    "zomato.pay@hdfcbank",
    "uber.india@ybl",
    "ola.money@paytm",
    "dunzo.delivery@icici",
    "urbancompany@ybl",
    "rapido.bike@icici",
    "bigbasket@hdfcbank",
    "blinkit.quick@icici",
    "zepto.grocery@ybl",
]

EMPLOYER_VPAS = [
    "salary.tcs@icici",
    "payroll.infosys@hdfcbank",
    "salary.wipro@sbi",
    "hr.hcl@axisbank",
    "payroll.cognizant@icici",
    "salary.accenture@hdfcbank",
    "techm.payroll@sbi",
    "capgemini.salary@axisbank",
    "deloitte.in@hdfcbank",
    "ey.india.pay@icici",
]

LENDER_COUNTERPARTIES = [
    "HDFC Ltd EMI",
    "Bajaj Finance EMI",
    "ICICI Bank EMI",
    "SBI Card EMI",
    "Tata Capital EMI",
    "Muthoot Finance EMI",
    "L&T Finance EMI",
    "Fullerton India EMI",
    "IIFL Finance EMI",
    "Manappuram EMI",
    "Home First EMI",
    "Piramal Finance EMI",
]

MERCHANT_CATEGORIES = {
    "groceries": [
        "DMart Supermarket", "BigBasket", "More Megastore", "Reliance Fresh",
        "Star Bazaar", "Spencer's Retail", "Nature's Basket",
    ],
    "utilities": [
        "BESCOM Electricity", "BWSSB Water", "Jio Recharge", "Airtel Recharge",
        "Tata Power", "Mahanagar Gas", "MSEB Electricity", "CESC Bill",
    ],
    "food_delivery": [
        "Swiggy Order", "Zomato Order", "EatSure", "Box8 Order",
        "Dominos Online", "McDonalds App",
    ],
    "transport": [
        "Uber Trip", "Ola Ride", "Rapido Auto", "Metro Card Recharge",
        "BMTC Bus Pass", "Parking MCD",
    ],
    "entertainment": [
        "BookMyShow Tickets", "PVR Cinemas", "Netflix Subscription",
        "Hotstar Premium", "Spotify India",
    ],
    "healthcare": [
        "Apollo Pharmacy", "1mg Order", "Practo Consult", "MedPlus Pharmacy",
        "Netmeds Order",
    ],
    "education": [
        "Byju's Subscription", "Unacademy Plus", "Coursera Payment",
    ],
    "shopping": [
        "Amazon India", "Flipkart", "Myntra", "Ajio", "Nykaa",
        "Croma Electronics", "Reliance Digital",
    ],
}

INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Kochi", "Chandigarh", "Indore", "Bhopal", "Nagpur",
    "Coimbatore", "Visakhapatnam", "Vadodara", "Surat", "Noida",
]

# ─── Persona Definitions ────────────────────────────────────────────────────────

SALARIED_STABLE = PersonaConfig(
    name="salaried_stable",
    proportion=0.35,
    description="Salaried professionals with stable income and existing bureau history",
    monthly_income_range=(45000, 160000),
    income_frequency="monthly",
    income_variability=0.04,  # ±4% variation — very stable
    income_sources=EMPLOYER_VPAS,
    bureau_score_range=(670, 830),
    bureau_score_probability=0.98,
    age_range=(26, 52),
    occupations=["Software Engineer", "Analyst", "Manager", "Accountant", "HR Executive",
                 "Bank Officer", "Government Employee", "Teacher", "Consultant"],
    emi_count_range=(0, 3),
    emi_amount_fraction_range=(0.06, 0.14),
    lender_names=LENDER_COUNTERPARTIES[:6],
    bounce_probability_per_month=0.02,  # occasional technical bounces
    has_rent_probability=0.60,
    rent_fraction_range=(0.15, 0.28),
    life_event_probabilities={
        "emi_closure": 0.30,
        "income_step_up": 0.25,
        "new_commitment": 0.12,
    },
    monthly_expense_fraction=(0.30, 0.50),
)

GIG_WORKER = PersonaConfig(
    name="gig_worker",
    proportion=0.20,
    description="Gig economy workers (delivery, ride-sharing) with frequent variable payouts and thin/no bureau file",
    monthly_income_range=(20000, 50000),
    income_frequency="weekly",
    income_variability=0.30,  # high variability
    income_sources=PLATFORM_VPAS[:8],
    bureau_score_range=(580, 660),
    bureau_score_probability=0.20,  # some have credit cards or small loans
    age_range=(19, 36),
    occupations=["Delivery Partner", "Ride-share Driver", "Freelance Courier",
                 "Gig Worker", "Platform Partner"],
    emi_count_range=(0, 2),
    emi_amount_fraction_range=(0.04, 0.10),
    lender_names=LENDER_COUNTERPARTIES[2:6],
    bounce_probability_per_month=0.06,  # occasional cash flow mismatch bounces
    has_rent_probability=0.50,
    rent_fraction_range=(0.18, 0.35),
    life_event_probabilities={
        "income_step_up": 0.35,
        "emi_closure": 0.10,
        "new_commitment": 0.15,
    },
    monthly_expense_fraction=(0.45, 0.70),
)

NEW_TO_CREDIT = PersonaConfig(
    name="new_to_credit",
    proportion=0.15,
    description="Young professionals with steady salary but zero credit history — NTC segment",
    monthly_income_range=(30000, 75000),
    income_frequency="monthly",
    income_variability=0.06,
    income_sources=EMPLOYER_VPAS,
    bureau_score_range=None,
    bureau_score_probability=0.0,  # deliberately zero — this is the NTC point
    age_range=(21, 29),
    occupations=["Junior Developer", "Graduate Trainee", "Associate Analyst",
                 "Junior Consultant", "Teaching Assistant", "Intern (Full-time)"],
    emi_count_range=(0, 1),
    emi_amount_fraction_range=(0.03, 0.08),
    lender_names=LENDER_COUNTERPARTIES[4:8],
    bounce_probability_per_month=0.02,
    has_rent_probability=0.70,
    rent_fraction_range=(0.20, 0.35),
    life_event_probabilities={
        "income_step_up": 0.40,
        "emi_closure": 0.05,
        "new_commitment": 0.15,
    },
    monthly_expense_fraction=(0.40, 0.60),
)

SELF_EMPLOYED = PersonaConfig(
    name="self_employed",
    proportion=0.15,
    description="Small shopkeepers and self-employed individuals with irregular but detectable UPI merchant inflows",
    monthly_income_range=(25000, 90000),
    income_frequency="irregular",
    income_variability=0.40,  # high variability
    income_sources=[
        "customer.upi@ybl", "payment.received@icici", "merchant.settle@hdfcbank",
        "gpay.merchant@sbi", "phonepe.merchant@ybl", "paytm.merchant@paytm",
    ],
    bureau_score_range=(590, 680),
    bureau_score_probability=0.40,  # thin file — some have score, most don't
    age_range=(25, 58),
    occupations=["Shopkeeper", "Small Business Owner", "Kirana Store Owner",
                 "Auto Repair Shop", "Tailor", "Street Food Vendor", "Salon Owner"],
    emi_count_range=(0, 3),
    emi_amount_fraction_range=(0.05, 0.12),
    lender_names=LENDER_COUNTERPARTIES[:6],
    bounce_probability_per_month=0.05,
    has_rent_probability=0.35,
    rent_fraction_range=(0.10, 0.22),
    life_event_probabilities={
        "income_step_up": 0.30,
        "emi_closure": 0.20,
        "new_commitment": 0.15,
    },
    monthly_expense_fraction=(0.35, 0.55),
)

OVER_LEVERAGED = PersonaConfig(
    name="over_leveraged",
    proportion=0.15,
    description="Customers with multiple concurrent loans, rising EMI burden, and NACH bounce events — must be caught by Guardrail Engine",
    monthly_income_range=(35000, 110000),
    income_frequency="monthly",
    income_variability=0.10,
    income_sources=EMPLOYER_VPAS,
    bureau_score_range=(500, 660),
    bureau_score_probability=0.95,
    age_range=(28, 52),
    occupations=["Sales Executive", "Supervisor", "Clerk", "Field Officer",
                 "Warehouse Manager", "Driver (Corporate)", "Technician"],
    emi_count_range=(4, 7),
    emi_amount_fraction_range=(0.12, 0.20),
    lender_names=LENDER_COUNTERPARTIES,  # all lenders available
    bounce_probability_per_month=0.25,  # frequent bounces
    has_rent_probability=0.55,
    rent_fraction_range=(0.15, 0.25),
    life_event_probabilities={
        "income_step_up": 0.08,
        "emi_closure": 0.12,
        "new_commitment": 0.35,  # taking on more debt
    },
    monthly_expense_fraction=(0.25, 0.40),  # lower discretionary spend (squeezed by EMIs)
)

# ─── All Personas ────────────────────────────────────────────────────────────────

ALL_PERSONAS: list[PersonaConfig] = [
    SALARIED_STABLE,
    GIG_WORKER,
    NEW_TO_CREDIT,
    SELF_EMPLOYED,
    OVER_LEVERAGED,
]


def validate_proportions() -> None:
    """Verify persona proportions sum to 1.0."""
    total = sum(p.proportion for p in ALL_PERSONAS)
    assert abs(total - 1.0) < 1e-6, f"Persona proportions sum to {total}, expected 1.0"


# Run validation on import
validate_proportions()

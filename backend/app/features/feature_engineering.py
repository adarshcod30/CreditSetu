"""
Feature engineering pipeline for CreditSetu.

Computes per-customer behavioural features from raw transaction data.
Uses ruptures (PELT algorithm) for change-point detection on daily net cash flow
to identify life-event triggers algorithmically — not from ground-truth labels.

Features are designed to work even when bureau_score is missing (null),
which is the core value proposition for NTC and gig-worker segments.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    warnings.warn("ruptures not installed — change-point detection will be skipped")


def engineer_features(
    customer: dict,
    transactions_df: pd.DataFrame,
) -> dict:
    """
    Compute all features for a single customer from their transactions.

    Args:
        customer: Customer profile dictionary
        transactions_df: DataFrame of transactions for this customer only (pre-converted dates/amounts)

    Returns:
        Dictionary of computed features
    """
    if transactions_df.empty:
        return _empty_features(customer)

    # Use the pre-converted transactions dataframe
    txns = transactions_df

    # Separate credits and debits
    credits = txns[txns["type"] == "credit"]
    debits = txns[txns["type"] == "debit"]

    features = {}

    # ─── Income Features ─────────────────────────────────────────────
    features.update(_compute_income_features(credits, customer))

    # ─── Gig Pattern Score ───────────────────────────────────────────
    features["gig_pattern_score"] = _compute_gig_score(credits)

    # ─── EMI & Leverage Features ─────────────────────────────────────
    features.update(_compute_emi_features(txns, credits))

    # ─── Rent Consistency ────────────────────────────────────────────
    features["rent_consistency"] = _compute_rent_consistency(debits)

    # ─── Spending Diversity ──────────────────────────────────────────
    features["merchant_category_entropy"] = _compute_category_entropy(debits)

    # ─── Surplus ─────────────────────────────────────────────────────
    monthly_credits = credits.set_index("date").resample("ME")["amount"].sum()
    monthly_debits = debits.set_index("date").resample("ME")["amount"].sum()
    monthly_surplus = monthly_credits.subtract(monthly_debits, fill_value=0)
    features["monthly_surplus"] = float(monthly_surplus.mean()) if len(monthly_surplus) > 0 else 0.0

    # ─── Bureau Features ─────────────────────────────────────────────
    features["has_bureau_score"] = customer.get("bureau_score") is not None
    features["bureau_score"] = customer.get("bureau_score")  # Can be None — LightGBM handles this

    # ─── Change-Point Detection (Life Events) ────────────────────────
    features.update(_detect_life_events(txns))

    return features


def engineer_features_batch(
    customers_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute features for all customers.

    Args:
        customers_df: DataFrame from customer generator
        transactions_df: DataFrame of all transactions

    Returns:
        DataFrame with one row per customer, columns are feature names
    """
    all_features = []

    # Pre-convert date and amount to avoid doing it 5000 times
    txns = transactions_df.copy()
    txns["date"] = pd.to_datetime(txns["date"])
    txns["amount"] = txns["amount"].astype(float)

    # Group transactions by customer_id to avoid O(N*M) filtering overhead
    grouped_txns = {cust_id: group for cust_id, group in txns.groupby("customer_id")}

    for _, customer in customers_df.iterrows():
        cust_id = customer["customer_id"]
        cust_txns = grouped_txns.get(cust_id, pd.DataFrame())
        features = engineer_features(customer.to_dict(), cust_txns)
        features["customer_id"] = cust_id
        all_features.append(features)

    return pd.DataFrame(all_features)


def _compute_income_features(credits: pd.DataFrame, customer: dict) -> dict:
    """Compute income regularity and stability features."""
    if credits.empty:
        return {
            "income_mean": 0.0,
            "income_cv": 1.0,
            "income_timing_cv": 1.0,
        }

    # Filter to income-type credits (salary, gig_payout, merchant_collection)
    income_categories = {"salary", "gig_payout", "merchant_collection"}
    income_txns = credits[credits["category"].isin(income_categories)]

    if income_txns.empty:
        return {
            "income_mean": 0.0,
            "income_cv": 1.0,
            "income_timing_cv": 1.0,
        }

    # Monthly aggregated income
    monthly_income = income_txns.set_index("date").resample("ME")["amount"].sum()

    income_mean = float(monthly_income.mean())
    income_std = float(monthly_income.std()) if len(monthly_income) > 1 else 0.0
    income_cv = income_std / income_mean if income_mean > 0 else 1.0

    # Timing regularity: CV of days between income credits
    income_dates = income_txns["date"].sort_values()
    if len(income_dates) > 1:
        intervals = income_dates.diff().dropna().dt.days
        timing_cv = float(intervals.std() / intervals.mean()) if intervals.mean() > 0 else 1.0
    else:
        timing_cv = 1.0

    return {
        "income_mean": income_mean,
        "income_cv": min(income_cv, 2.0),  # cap at 2.0
        "income_timing_cv": min(timing_cv, 2.0),
    }


def _compute_gig_score(credits: pd.DataFrame) -> float:
    """
    Compute gig-pattern score.

    Distinguishes gig payout patterns from salaried income using:
    - Payment frequency (gig = many small, frequent payments)
    - Counterparty diversity (gig = multiple platform VPAs)
    - Amount variability (gig = high CV)
    """
    if credits.empty:
        return 0.0

    income_txns = credits[credits["category"].isin({"salary", "gig_payout", "merchant_collection"})]
    if income_txns.empty:
        return 0.0

    # Frequency: normalized count per month
    date_range = (income_txns["date"].max() - income_txns["date"].min()).days
    months = max(date_range / 30, 1)
    freq_per_month = len(income_txns) / months
    freq_score = min(freq_per_month / 15, 1.0)  # 15+ credits/month = max gig score

    # Counterparty diversity
    n_counterparties = income_txns["counterparty"].nunique()
    diversity_score = min(n_counterparties / 5, 1.0)  # 5+ counterparties = max

    # Amount variability
    amount_cv = income_txns["amount"].std() / income_txns["amount"].mean() if income_txns["amount"].mean() > 0 else 0
    variability_score = min(amount_cv / 0.5, 1.0)  # CV of 0.5+ = max

    # Composite gig score
    return float(0.35 * freq_score + 0.35 * diversity_score + 0.30 * variability_score)


def _compute_emi_features(txns: pd.DataFrame, credits: pd.DataFrame) -> dict:
    """Compute EMI burden and lender count features."""
    emi_txns = txns[(txns["category"] == "emi") & (txns["type"] == "debit")]
    bounce_txns = txns[txns["is_bounce"] == True]

    # Concurrent lender count (distinct EMI counterparties in last 6 months)
    if not emi_txns.empty:
        six_months_ago = txns["date"].max() - pd.Timedelta(days=180)
        recent_emi = emi_txns[emi_txns["date"] >= six_months_ago]
        concurrent_lender_count = int(recent_emi["counterparty"].nunique())

        # Monthly EMI total
        monthly_emi = emi_txns.set_index("date").resample("ME")["amount"].sum()
        monthly_income = credits.set_index("date").resample("ME")["amount"].sum()

        # EMI-to-inflow ratio
        aligned = pd.DataFrame({"emi": monthly_emi, "income": monthly_income}).fillna(0)
        aligned["ratio"] = aligned["emi"] / aligned["income"].replace(0, np.nan)
        aligned["ratio"] = aligned["ratio"].fillna(0)

        emi_to_inflow_ratio = float(aligned["ratio"].iloc[-1]) if len(aligned) > 0 else 0.0

        # 3-month trend direction
        if len(aligned) >= 3:
            recent_ratios = aligned["ratio"].iloc[-3:]
            emi_to_inflow_trend = float(recent_ratios.iloc[-1] - recent_ratios.iloc[0])
        else:
            emi_to_inflow_trend = 0.0
    else:
        concurrent_lender_count = 0
        emi_to_inflow_ratio = 0.0
        emi_to_inflow_trend = 0.0

    # NACH bounce count (trailing 6 months)
    if not bounce_txns.empty:
        three_months_ago = txns["date"].max() - pd.Timedelta(days=90)
        six_months_ago = txns["date"].max() - pd.Timedelta(days=180)
        nach_bounce_count_6m = int(len(bounce_txns[bounce_txns["date"] >= six_months_ago]))
        nach_bounce_count_3m = int(len(bounce_txns[bounce_txns["date"] >= three_months_ago]))
    else:
        nach_bounce_count_6m = 0
        nach_bounce_count_3m = 0

    return {
        "concurrent_lender_count": concurrent_lender_count,
        "emi_to_inflow_ratio": min(emi_to_inflow_ratio, 1.0),
        "emi_to_inflow_trend": emi_to_inflow_trend,
        "nach_bounce_count_6m": nach_bounce_count_6m,
        "nach_bounce_count_3m": nach_bounce_count_3m,
    }


def _compute_rent_consistency(debits: pd.DataFrame) -> float:
    """
    Compute rent payment consistency score.

    High consistency (low std dev of amount and timing) indicates financial discipline.
    Returns 0 if no rent payments found.
    """
    rent_txns = debits[debits["category"] == "rent"]
    if len(rent_txns) < 3:
        return 0.0

    # Amount consistency
    amount_std = rent_txns["amount"].std()
    amount_mean = rent_txns["amount"].mean()
    amount_cv = amount_std / amount_mean if amount_mean > 0 else 1.0

    # Timing consistency (std of day-of-month)
    rent_days = rent_txns["date"].dt.day
    day_std = rent_days.std()
    day_cv = day_std / 15  # normalize against half-month

    # Consistency score: higher = more consistent
    consistency = 1.0 - min((amount_cv + day_cv) / 2, 1.0)
    return float(max(consistency, 0.0))


def _compute_category_entropy(debits: pd.DataFrame) -> float:
    """
    Compute Shannon entropy of spending across merchant categories.

    Higher entropy = more diverse spending (positive signal for capacity).
    """
    if debits.empty:
        return 0.0

    spend_cats = debits[~debits["category"].isin({"emi", "rent", "nach_bounce"})]
    if spend_cats.empty:
        return 0.0

    cat_counts = spend_cats["category"].value_counts(normalize=True)
    entropy = -float((cat_counts * np.log2(cat_counts + 1e-10)).sum())

    # Normalize to [0, 1] — max entropy for 8 categories = log2(8) ≈ 3
    return min(entropy / 3.0, 1.0)


def _detect_life_events(txns: pd.DataFrame) -> dict:
    """
    Use ruptures PELT algorithm to detect structural breaks in rolling 30-day net cash flow.

    This is the algorithmic detection — NOT using ground-truth labels.
    The ground-truth labels are only used for validation (in benchmark_runner.py).
    """
    default = {
        "detected_event_type": None,
        "life_event_recency_days": None,
        "life_event_magnitude": 0.0,
        "change_points": [],
    }

    if not HAS_RUPTURES or txns.empty:
        return default

    try:
        # Build daily net cash flow series
        txns_dated = txns.copy()
        txns_dated["signed_amount"] = np.where(txns_dated["type"] == "credit", txns_dated["amount"], -txns_dated["amount"])

        daily_flow = txns_dated.set_index("date").resample("D")["signed_amount"].sum().fillna(0)

        if len(daily_flow) < 30:
            return default

        # Smooth the daily net cash flow with a 30-day rolling average
        # This removes regular monthly cycles (salary credit day spikes)
        smoothed_flow = daily_flow.rolling(window=30, min_periods=1).mean()
        signal = smoothed_flow.values.reshape(-1, 1).astype(float)

        # Run PELT change-point detection on smoothed signal
        # Lower penalty (pen=1.5) works best on smoothed, low-variance series
        algo = rpt.Pelt(model="rbf", min_size=15, jump=2).fit(signal)
        breakpoints = algo.predict(pen=1.5)

        # Remove the last breakpoint (always = len(signal) in ruptures)
        breakpoints = [bp for bp in breakpoints if bp < len(signal)]

        if not breakpoints:
            return default

        # Classify each breakpoint
        events = []
        for bp in breakpoints:
            bp_date = smoothed_flow.index[bp]
            days_ago = (smoothed_flow.index[-1] - bp_date).days

            # Compute magnitude on the smoothed signal: before vs after
            window = min(15, bp, len(signal) - bp)
            if window < 3:
                continue
            before_mean = float(np.mean(signal[max(0, bp - window):bp]))
            after_mean = float(np.mean(signal[bp:min(len(signal), bp + window)]))
            magnitude = after_mean - before_mean

            # Classify the event type based on the change pattern
            event_type = _classify_breakpoint(txns_dated, bp_date, magnitude, smoothed_flow)

            events.append({
                "breakpoint_index": bp,
                "date": bp_date.isoformat(),
                "days_ago": days_ago,
                "magnitude": magnitude,
                "event_type": event_type,
            })

        if not events:
            return default

        # Pick the most recent significant event
        events.sort(key=lambda e: e["days_ago"])
        
        # Prefer actual events over positive/negative shifts if available
        specific_events = [e for e in events if e["event_type"] in ("emi_closure", "income_step_up", "new_commitment")]
        most_recent = specific_events[0] if specific_events else events[0]

        return {
            "detected_event_type": most_recent["event_type"],
            "life_event_recency_days": most_recent["days_ago"],
            "life_event_magnitude": most_recent["magnitude"],
            "change_points": events,
        }

    except Exception:
        # Don't let change-point detection crash the pipeline
        return default


def _classify_breakpoint(
    txns: pd.DataFrame,
    bp_date: pd.Timestamp,
    magnitude: float,
    smoothed_flow: pd.Series,
) -> str:
    """
    Classify a detected breakpoint into an event type using transaction-level rules.

    This uses heuristics on the transaction stream around the breakpoint,
    NOT ground-truth labels.
    """
    window = pd.Timedelta(days=30)

    before = txns[(txns["date"] >= bp_date - window) & (txns["date"] < bp_date)]
    after = txns[(txns["date"] >= bp_date) & (txns["date"] < bp_date + window)]

    # Check for EMI closure: regular debit disappears after breakpoint
    emi_before = before[before["category"] == "emi"]["counterparty"].unique()
    emi_after = after[after["category"] == "emi"]["counterparty"].unique()
    closed_emis = set(emi_before) - set(emi_after)

    if closed_emis:
        return "emi_closure"

    # Check for income step-up: significant positive shift in credits
    income_cats = {"salary", "gig_payout", "merchant_collection"}
    income_before = before[before["category"].isin(income_cats)]["amount"].mean()
    income_after = after[after["category"].isin(income_cats)]["amount"].mean()

    if not np.isnan(income_before) and not np.isnan(income_after):
        if income_after > income_before * 1.12:
            return "income_step_up"

    # Check for new commitment: new regular debit appears
    new_emis = set(emi_after) - set(emi_before)
    if new_emis:
        return "new_commitment"

    # Default: classify by direction
    if magnitude > 0:
        return "positive_shift"
    else:
        return "negative_shift"


def _empty_features(customer: dict) -> dict:
    """Return default features when no transactions exist."""
    return {
        "income_mean": 0.0,
        "income_cv": 1.0,
        "income_timing_cv": 1.0,
        "gig_pattern_score": 0.0,
        "concurrent_lender_count": 0,
        "emi_to_inflow_ratio": 0.0,
        "emi_to_inflow_trend": 0.0,
        "nach_bounce_count_6m": 0,
        "nach_bounce_count_3m": 0,
        "rent_consistency": 0.0,
        "merchant_category_entropy": 0.0,
        "monthly_surplus": 0.0,
        "has_bureau_score": customer.get("bureau_score") is not None,
        "bureau_score": customer.get("bureau_score"),
        "detected_event_type": None,
        "life_event_recency_days": None,
        "life_event_magnitude": 0.0,
        "change_points": [],
    }


# Feature names used by ML models (in consistent order)
ML_FEATURE_NAMES = [
    "income_mean",
    "income_cv",
    "income_timing_cv",
    "gig_pattern_score",
    "emi_to_inflow_ratio",
    "emi_to_inflow_trend",
    "concurrent_lender_count",
    "nach_bounce_count_6m",
    "nach_bounce_count_3m",
    "rent_consistency",
    "merchant_category_entropy",
    "monthly_surplus",
    "has_bureau_score",
    "bureau_score",
]

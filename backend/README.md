# CreditSetu

General-purpose, explainable credit intelligence engine for thin-file and alternative-data lending. Score customers on behavioral transaction data — including customers with no bureau score at all — using three composable engines (Intent, Capacity, Guardrail), SHAP-backed explanations, and regulator-style adverse action reason codes.

This is the library package. For the full project — a reference FastAPI service, a React dashboard, synthetic demo data, and deployment docs — see the [GitHub repository](https://github.com/adarshcod30/CreditSetu).

## Install

```bash
pip install creditsetu
```

That pulls in only `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `pydantic`, and `pyyaml` — enough to fit models and score customers. SHAP explainability and ruptures-based life-event detection are opt-in and degrade gracefully if skipped:

```bash
pip install "creditsetu[explain]"   # + SHAP feature attributions / adverse action reasons
pip install "creditsetu[intent]"    # + ruptures change-point life-event detection
pip install "creditsetu[full]"      # both
```

## Usage

```python
from app.pipeline import CreditIntelligencePipeline
from app.scoring_profile import ScoringProfile

# Currency, guardrail thresholds, product catalog, weights — all policy,
# no code changes needed to adapt this to a different institution/market.
profile = ScoringProfile(name="acme", org_name="Acme Lending", currency_symbol="$")
pipeline = CreditIntelligencePipeline(profile=profile)

# Fit on your own historical data — customers_df needs a
# `true_repayment_capacity` column, ideally a real `is_stressed` column too.
pipeline.fit(customers_df, transactions_df)

# Score one customer...
result = pipeline.score_customer(customer, customer_transactions_df)

# ...or a whole batch, vectorized end-to-end.
scores_df = pipeline.score_batch(customers_df, transactions_df)
```

**Data contract** — `customer` is a dict needing at least `customer_id` and `bureau_score` (`None` is a first-class value, not an edge case — that's the point). `transactions` is a DataFrame with `date`, `amount`, `type` (`credit`/`debit`), `category`, `counterparty`, `is_bounce`.

Full details live in the package's `pipeline.py` module docstring, and the complete architecture/evaluation writeup is in the [GitHub README](https://github.com/adarshcod30/CreditSetu#readme).

## License

MIT

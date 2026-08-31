"""
ScoringProfile — the single configuration object that encodes one institution's
(or one loan product's) lending policy.

Everything that used to be a hardcoded module-level constant in the engines —
guardrail thresholds, product catalog, currency, composite weights, intent event
weights — lives here instead, with the original values preserved as defaults so
behavior is unchanged out of the box. A host application swaps in its own
ScoringProfile (built in code or loaded from YAML) to reuse the same engine code
for a different institution, market, or product line — including running several
profiles side by side for multi-tenant / multi-product deployments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class GuardrailThresholds(BaseModel):
    """Risk tiering thresholds for the Guardrail Engine."""

    max_concurrent_lenders: int = 5
    max_emi_to_inflow_ratio: float = 0.60
    max_nach_bounces_3m: int = 1
    watch_threshold: float = 0.30
    suppress_threshold: float = 0.60


class ProductRule(BaseModel):
    """
    One tier in a product decision table. Rules in a list are evaluated in
    order; the first rule whose thresholds are met wins.
    """

    name: str
    min_capacity_amount: float = 0.0
    min_income_mean: float = 0.0


class ScoringProfile(BaseModel):
    """Bundles every tunable that encodes one institution's/product's policy."""

    name: str = "default"
    description: str = ""

    org_name: str = "CreditSetu"
    currency_code: str = "INR"
    currency_symbol: str = "₹"

    # Composite scorer weights (must sum to ~1.0)
    intent_weight: float = 0.40
    capacity_weight: float = 0.60

    # Intent engine
    intent_event_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "emi_closure": 0.90,
            "income_step_up": 0.85,
            "positive_shift": 0.60,
            "new_commitment": -0.30,
            "negative_shift": -0.20,
        }
    )
    intent_decay_rate: float = 0.015

    # Guardrail engine
    guardrail: GuardrailThresholds = Field(default_factory=GuardrailThresholds)

    # Capacity engine — used to normalize a raw predicted amount into [0, 1]
    capacity_max_amount: float = 50000.0

    # Product decision table, evaluated top-down, first match wins.
    # "thin_file" applies to customers with no bureau score or a strong gig
    # income pattern; "standard" applies to everyone else.
    thin_file_gig_score_threshold: float = 0.40
    standard_product_rules: list[ProductRule] = Field(
        default_factory=lambda: [
            ProductRule(name="Home Loan", min_capacity_amount=32000, min_income_mean=75000),
            ProductRule(name="Auto Loan", min_capacity_amount=16000, min_income_mean=40000),
            ProductRule(name="Personal Loan", min_capacity_amount=5000),
        ]
    )
    thin_file_product_rules: list[ProductRule] = Field(
        default_factory=lambda: [
            ProductRule(name="Personal Loan", min_capacity_amount=12000),
            ProductRule(name="Micro-Credit Line", min_capacity_amount=5000),
        ]
    )
    fallback_product: str = "Retail Credit Card"

    model_config = {"validate_assignment": True}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScoringProfile":
        """Load a profile from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Write this profile to a YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)

    def select_product(
        self,
        capacity_amount: float,
        income_mean: float,
        has_bureau_score: bool,
        gig_pattern_score: float,
    ) -> str:
        """Apply the product decision table to pick a suggested product."""
        is_thin_file = (not has_bureau_score) or gig_pattern_score > self.thin_file_gig_score_threshold
        rules = self.thin_file_product_rules if is_thin_file else self.standard_product_rules

        for rule in rules:
            if capacity_amount >= rule.min_capacity_amount and income_mean >= rule.min_income_mean:
                return rule.name

        return self.fallback_product


_DEFAULT_PROFILE: Optional[ScoringProfile] = None


def default_profile() -> ScoringProfile:
    """
    Return the built-in default profile (India retail-lending demo values).

    Cached as a singleton so repeated calls don't reconstruct the object;
    callers that need to mutate should copy it (`profile.model_copy(deep=True)`)
    rather than mutate the shared instance.
    """
    global _DEFAULT_PROFILE
    if _DEFAULT_PROFILE is None:
        _DEFAULT_PROFILE = ScoringProfile(
            name="default",
            description="Original CreditSetu demo policy — India retail lending, INR.",
        )
    return _DEFAULT_PROFILE


def profile_from_path_or_default(path: Optional[str]) -> ScoringProfile:
    """Load a profile from `path` if given, else return the built-in default."""
    return ScoringProfile.from_yaml(path) if path else default_profile()

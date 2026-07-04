"""
Intent Signal Engine for CreditSetu.

Detects real-time life-event triggers indicating active loan-readiness using
ruptures change-point detection on daily net cash flow series.

Scoring approach:
- Each detected life event is scored by recency (exponential decay)
- Event type weights reflect loan-readiness signal strength
- Final score is the max across all recent positive events, clipped to [0, 1]

Validation: compare detected events against ground-truth labels from the
synthetic data generator to compute precision/recall (done in benchmark_runner.py).
"""

import numpy as np
import pandas as pd
from typing import Optional


# Event type weights — how strongly each event type indicates loan-readiness
EVENT_WEIGHTS = {
    "emi_closure": 0.90,       # Very strong signal: freed-up repayment capacity
    "income_step_up": 0.85,    # Strong signal: increased ability to service new debt
    "positive_shift": 0.60,    # Moderate signal: general improvement in cash flow
    "new_commitment": -0.30,   # Negative signal: reduced capacity
    "negative_shift": -0.20,   # Weak negative signal
}

# Decay rate for recency weighting (per day)
RECENCY_DECAY_RATE = 0.015  # score halves roughly every 46 days


class IntentEngine:
    """
    Scores customers based on detected life-event signals from their
    transaction cash flow patterns.
    """

    def __init__(
        self,
        event_weights: Optional[dict[str, float]] = None,
        decay_rate: float = RECENCY_DECAY_RATE,
    ):
        """
        Args:
            event_weights: Override default event type weights.
            decay_rate: Exponential decay rate for recency scoring.
        """
        self.event_weights = event_weights or EVENT_WEIGHTS
        self.decay_rate = decay_rate

    def score(self, features: dict) -> dict:
        """
        Compute intent score for a single customer.

        Args:
            features: Feature dictionary from feature_engineering.py,
                      must include change_points, detected_event_type, etc.

        Returns:
            Dictionary with:
            - intent_score: float [0, 1]
            - intent_event_type: str or None
            - intent_event_recency_days: int or None
            - intent_details: list of event dicts
        """
        change_points = features.get("change_points", [])

        if not change_points:
            return {
                "intent_score": 0.0,
                "intent_event_type": None,
                "intent_event_recency_days": None,
                "intent_details": [],
            }

        # Score each detected event
        scored_events = []
        for event in change_points:
            event_type = event.get("event_type", "unknown")
            days_ago = event.get("days_ago", 999)
            weight = self.event_weights.get(event_type, 0.0)

            # Recency-weighted score: higher weight for more recent events
            recency_factor = np.exp(-self.decay_rate * days_ago)
            event_score = weight * recency_factor

            scored_events.append({
                "event_type": event_type,
                "days_ago": days_ago,
                "weight": weight,
                "recency_factor": round(recency_factor, 3),
                "event_score": round(event_score, 3),
                "date": event.get("date"),
                "magnitude": event.get("magnitude", 0),
            })

        # Final score = max positive event score, clipped to [0, 1]
        positive_scores = [e["event_score"] for e in scored_events if e["event_score"] > 0]
        intent_score = max(positive_scores) if positive_scores else 0.0
        intent_score = float(np.clip(intent_score, 0.0, 1.0))

        # Best event for explanation
        best_event = max(scored_events, key=lambda e: e["event_score"])

        return {
            "intent_score": round(intent_score, 4),
            "intent_event_type": best_event["event_type"] if intent_score > 0 else None,
            "intent_event_recency_days": best_event["days_ago"] if intent_score > 0 else None,
            "intent_details": scored_events,
        }

    def score_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Score multiple customers."""
        results = []
        for _, row in features_df.iterrows():
            result = self.score(row.to_dict())
            result["customer_id"] = row["customer_id"]
            results.append(result)
        return pd.DataFrame(results)

    def validate_against_ground_truth(
        self,
        features_df: pd.DataFrame,
        customers_df: pd.DataFrame,
    ) -> dict:
        """
        Compare detected events against ground-truth life-event labels.

        Returns precision, recall, and per-event-type metrics.
        Used by benchmark_runner.py for the evaluation slide.
        """
        tp = 0  # true positives
        fp = 0  # false positives
        fn = 0  # false negatives
        y_true = []
        y_scores = []

        for _, features_row in features_df.iterrows():
            cust_id = features_row["customer_id"]
            cust_row = customers_df[customers_df["customer_id"] == cust_id]
            if cust_row.empty:
                continue

            ground_truth_events = cust_row.iloc[0].get("life_events", [])
            if isinstance(ground_truth_events, str):
                import json
                try:
                    ground_truth_events = json.loads(ground_truth_events)
                except (json.JSONDecodeError, TypeError):
                    ground_truth_events = []

            gt_types = {e["event_type"] for e in ground_truth_events} if ground_truth_events else set()
            detected_type = features_row.get("detected_event_type")
            detected_set = {detected_type} if detected_type and detected_type not in ("positive_shift", "negative_shift") else set()

            # Match logic: event type match (ignoring exact timing for now)
            matched = gt_types & detected_set
            tp += len(matched)
            fp += len(detected_set - gt_types)
            fn += len(gt_types - detected_set)

            # AUC metrics gathering
            y_true.append(1 if any(e in ("emi_closure", "income_step_up") for e in gt_types) else 0)
            scored_res = self.score(features_row.to_dict())
            y_scores.append(scored_res["intent_score"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            auc = 0.5

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "auc_roc": round(auc, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

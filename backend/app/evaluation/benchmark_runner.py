"""
Benchmark runner for CreditSetu.

Computes evaluation metrics for all three engines and the composite scorer.
These numbers are the exact values that should be pasted into the pitch deck's
benchmarking slide — pull them from the output, never hand-write a number.

Outputs: benchmark_report.json and benchmark_report.md
"""

import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    r2_score,
)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.data_generation.synthetic_customer_generator import generate_customers
from app.data_generation.synthetic_transaction_generator import generate_all_transactions
from app.features.feature_engineering import engineer_features_batch
from app.engines.intent_engine import IntentEngine
from app.engines.capacity_engine import CapacityEngine
from app.engines.guardrail_engine import GuardrailEngine
from app.engines.composite_scorer import CompositeScorer


def run_full_benchmark(
    n_customers: int = 500,
    seed: int = 42,
    test_size: float = 0.2,
) -> dict:
    """
    Run the complete benchmark suite.

    Uses a fresh dataset (not the demo database) to ensure unbiased evaluation.
    All metrics are computed on held-out test splits.
    """
    print("=" * 60)
    print("CreditSetu Benchmark Runner")
    print("=" * 60)

    report = {}

    # ─── Generate Fresh Data ─────────────────────────────────────────
    print("\n[1/6] Generating benchmark dataset...")
    t0 = time.time()
    customers_df = generate_customers(n_customers=n_customers, seed=seed + 100)
    transactions_df = generate_all_transactions(customers_df, seed=seed + 100)
    print(f"  Generated {len(customers_df)} customers, {len(transactions_df)} transactions")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ─── Feature Engineering ─────────────────────────────────────────
    print("\n[2/6] Computing features...")
    t0 = time.time()
    features_df = engineer_features_batch(customers_df, transactions_df)
    print(f"  Computed {len(features_df.columns)} features for {len(features_df)} customers")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ─── Capacity Engine Benchmark ───────────────────────────────────
    print("\n[3/6] Benchmarking Capacity Engine...")
    capacity_engine = CapacityEngine()
    capacity_metrics = capacity_engine.train(features_df, customers_df, test_size=test_size)
    report["capacity_engine"] = {
        "auc_roc": capacity_metrics["auc_roc"],
        "rmse": capacity_metrics["rmse"],
        "r2": capacity_metrics["r2"],
        "n_train": capacity_metrics["n_train"],
        "n_test": capacity_metrics["n_test"],
        "description": "LightGBM regressor predicting safe monthly repayment capacity",
    }
    print(f"  AUC-ROC: {capacity_metrics['auc_roc']}")
    print(f"  RMSE: ₹{capacity_metrics['rmse']:,.0f}")
    print(f"  R²: {capacity_metrics['r2']}")

    # ─── Intent Engine Benchmark ─────────────────────────────────────
    print("\n[4/6] Benchmarking Intent Engine...")
    intent_engine = IntentEngine()
    intent_metrics = intent_engine.validate_against_ground_truth(features_df, customers_df)
    report["intent_engine"] = {
        "precision": intent_metrics["precision"],
        "recall": intent_metrics["recall"],
        "f1": intent_metrics["f1"],
        "true_positives": intent_metrics["true_positives"],
        "false_positives": intent_metrics["false_positives"],
        "false_negatives": intent_metrics["false_negatives"],
        "description": "Change-point detection (ruptures PELT) on daily net cash flow",
    }
    print(f"  Precision: {intent_metrics['precision']}")
    print(f"  Recall: {intent_metrics['recall']}")
    print(f"  F1: {intent_metrics['f1']}")

    # ─── Guardrail Engine Benchmark ──────────────────────────────────
    print("\n[5/6] Benchmarking Guardrail Engine...")
    guardrail_engine = GuardrailEngine()
    guardrail_metrics = guardrail_engine.train(features_df, customers_df, test_size=test_size)
    report["guardrail_engine"] = {
        "auc_roc": guardrail_metrics["auc_roc"],
        "false_positive_rate": guardrail_metrics["false_positive_rate"],
        "false_negative_rate": guardrail_metrics["false_negative_rate"],
        "precision": guardrail_metrics["precision"],
        "recall": guardrail_metrics["recall"],
        "n_stressed": guardrail_metrics["n_stressed"],
        "n_safe": guardrail_metrics["n_safe"],
        "description": "Hybrid hard-rules + LightGBM classifier for over-leverage detection",
    }
    print(f"  AUC-ROC: {guardrail_metrics['auc_roc']}")
    print(f"  False Positive Rate: {guardrail_metrics['false_positive_rate']}")
    print(f"  False Negative Rate: {guardrail_metrics['false_negative_rate']}")

    # ─── Composite Score Benchmark ───────────────────────────────────
    print("\n[6/6] Benchmarking Composite Scorer + Latency...")
    scorer = CompositeScorer(intent_engine, capacity_engine, guardrail_engine)

    # Measure latency
    latencies = []
    all_scores = []
    for _, row in features_df.iterrows():
        t_start = time.time()
        result = scorer.score(row.to_dict())
        t_end = time.time()
        latencies.append((t_end - t_start) * 1000)  # ms
        result["customer_id"] = row["customer_id"]
        all_scores.append(result)

    scores_df = pd.DataFrame(all_scores)

    # Precision@Top20%: of the top 20% ranked leads, what fraction are "good"?
    # "Good" = customer with true_repayment_capacity above median AND not over-leveraged
    merged = scores_df.merge(
        customers_df[["customer_id", "true_repayment_capacity", "persona_type"]],
        on="customer_id",
    )
    merged["is_good_lead"] = (
        (merged["true_repayment_capacity"] > merged["true_repayment_capacity"].median()) &
        (merged["persona_type"] != "over_leveraged")
    )

    # Sort by composite score, take top 20%
    merged_sorted = merged.sort_values("composite_score", ascending=False)
    top_20_pct = merged_sorted.head(int(len(merged_sorted) * 0.2))
    precision_at_20 = float(top_20_pct["is_good_lead"].mean()) if len(top_20_pct) > 0 else 0.0

    report["composite"] = {
        "precision_at_top_20_pct": round(precision_at_20, 4),
        "total_qualified_leads": int(scores_df["is_qualified_lead"].sum()),
        "total_suppressed": int((~scores_df["is_qualified_lead"]).sum()),
        "qualification_rate": round(float(scores_df["is_qualified_lead"].mean()), 4),
        "description": "Weighted combination of Intent (0.4) and Capacity (0.6) scores",
    }

    report["latency"] = {
        "avg_ms": round(float(np.mean(latencies)), 2),
        "p50_ms": round(float(np.median(latencies)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "total_customers": len(latencies),
        "description": "End-to-end scoring latency per customer (measured, not estimated)",
    }

    print(f"  Precision@Top20%: {precision_at_20}")
    print(f"  Avg Latency: {np.mean(latencies):.2f}ms")
    print(f"  P95 Latency: {np.percentile(latencies, 95):.2f}ms")

    report["generated_at"] = datetime.utcnow().isoformat()

    # ─── Save Reports ────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)

    json_path = "data/benchmark_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ JSON report saved: {json_path}")

    md_path = "data/benchmark_report.md"
    _write_markdown_report(report, md_path)
    print(f"✅ Markdown report saved: {md_path}")

    return report


def _write_markdown_report(report: dict, path: str):
    """Generate a human-readable benchmark report in Markdown."""
    lines = [
        "# CreditSetu — Benchmark Report",
        "",
        f"*Generated: {report.get('generated_at', 'N/A')}*",
        "",
        "> **Note**: All metrics are computed on synthetic data. In production, these would",
        "> be validated against actual loan repayment outcomes.",
        "",
        "## Capacity Engine",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| AUC-ROC | {report['capacity_engine']['auc_roc']} |",
        f"| RMSE | ₹{report['capacity_engine']['rmse']:,.0f} |",
        f"| R² | {report['capacity_engine']['r2']} |",
        f"| Train/Test Split | {report['capacity_engine']['n_train']}/{report['capacity_engine']['n_test']} |",
        "",
        "## Intent Engine",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Precision | {report['intent_engine']['precision']} |",
        f"| Recall | {report['intent_engine']['recall']} |",
        f"| F1 Score | {report['intent_engine']['f1']} |",
        f"| True Positives | {report['intent_engine']['true_positives']} |",
        f"| False Positives | {report['intent_engine']['false_positives']} |",
        f"| False Negatives | {report['intent_engine']['false_negatives']} |",
        "",
        "## Guardrail Engine",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| AUC-ROC | {report['guardrail_engine']['auc_roc']} |",
        f"| False Positive Rate | {report['guardrail_engine']['false_positive_rate']} |",
        f"| False Negative Rate | {report['guardrail_engine']['false_negative_rate']} |",
        f"| Precision | {report['guardrail_engine']['precision']} |",
        f"| Recall | {report['guardrail_engine']['recall']} |",
        "",
        "## Composite Score",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Precision@Top20% | {report['composite']['precision_at_top_20_pct']} |",
        f"| Qualified Leads | {report['composite']['total_qualified_leads']} |",
        f"| Suppressed | {report['composite']['total_suppressed']} |",
        f"| Qualification Rate | {report['composite']['qualification_rate']:.1%} |",
        "",
        "## Scoring Latency",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Average | {report['latency']['avg_ms']}ms |",
        f"| P50 (Median) | {report['latency']['p50_ms']}ms |",
        f"| P95 | {report['latency']['p95_ms']}ms |",
        f"| P99 | {report['latency']['p99_ms']}ms |",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))

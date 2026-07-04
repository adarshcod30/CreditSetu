#!/usr/bin/env python3
"""
Run CreditSetu benchmark suite.

Usage:
    python scripts/run_benchmark.py [--n_customers 500] [--seed 42]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.benchmark_runner import run_full_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CreditSetu benchmarks")
    parser.add_argument("--n_customers", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    report = run_full_benchmark(
        n_customers=args.n_customers,
        seed=args.seed,
        test_size=args.test_size,
    )

    print("\n📊 Final Report Summary:")
    print(f"  Capacity AUC-ROC: {report['capacity_engine']['auc_roc']}")
    print(f"  Intent Precision: {report['intent_engine']['precision']}")
    print(f"  Intent Recall: {report['intent_engine']['recall']}")
    print(f"  Guardrail FPR: {report['guardrail_engine']['false_positive_rate']}")
    print(f"  Precision@Top20%: {report['composite']['precision_at_top_20_pct']}")
    print(f"  Avg Latency: {report['latency']['avg_ms']}ms")

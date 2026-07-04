# CreditSetu — Benchmark Report

*Generated: 2026-07-04T13:58:39.702120*

> **Note**: All metrics are computed on synthetic data. In production, these would
> be validated against actual loan repayment outcomes.

## Capacity Engine

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9918 |
| RMSE | ₹2,967 |
| R² | 0.9279 |
| Train/Test Split | 800/200 |

## Intent Engine

| Metric | Value |
|--------|-------|
| Precision | 0.3394 |
| Recall | 0.4526 |
| F1 Score | 0.3879 |
| True Positives | 282 |
| False Positives | 549 |
| False Negatives | 341 |

## Guardrail Engine

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.7962 |
| False Positive Rate | 0.1151 |
| False Negative Rate | 0.3443 |
| Precision | 0.7143 |
| Recall | 0.6557 |

## Composite Score

| Metric | Value |
|--------|-------|
| Precision@Top20% | 0.995 |
| Qualified Leads | 722 |
| Suppressed | 278 |
| Qualification Rate | 72.2% |

## Scoring Latency

| Metric | Value |
|--------|-------|
| Average | 21.19ms |
| P50 (Median) | 21.05ms |
| P95 | 21.72ms |
| P99 | 26.49ms |

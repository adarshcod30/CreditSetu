"""
Benchmark API routes for CreditSetu.
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

from ..config import settings
from ..schemas.schemas import BenchmarkReport

router = APIRouter(prefix="/api/benchmark", tags=["Benchmarking"])


@router.post("/run")
def run_benchmark():
    """
    Trigger benchmark evaluation run.

    Runs the full benchmark suite and saves results to benchmark_report.json.
    """
    try:
        from ..evaluation.benchmark_runner import run_full_benchmark
        report = run_full_benchmark()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.get("/latest", response_model=BenchmarkReport)
def get_latest_benchmark():
    """Return the last saved benchmark report."""
    report_path = Path(settings.BENCHMARK_REPORT_PATH)
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No benchmark report found. Run POST /api/benchmark/run first.",
        )

    with open(report_path) as f:
        report = json.load(f)

    return BenchmarkReport(**report)

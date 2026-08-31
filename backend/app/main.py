"""
CreditSetu — general-purpose, explainable credit intelligence engine for
thin-file and alternative-data lending.

FastAPI reference/demo application entrypoint — a thin API layer over the
CreditIntelligencePipeline library (see app/pipeline.py). A host application
embedding CreditSetu as a library doesn't need this file at all.

IMPORTANT: All data in this demo is SYNTHETIC. In a real deployment, customer
data would come from a bank's own Account Aggregator or core banking
integration — see app/pipeline.py's data contract for what's expected instead.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: create DB tables on startup."""
    logger.info("Starting CreditSetu backend...")
    init_db()
    logger.info("Database tables initialized")
    yield
    logger.info("Shutting down CreditSetu backend")


app = FastAPI(
    title="CreditSetu API",
    description=(
        "General-purpose credit intelligence engine for thin-file and alternative-data "
        "lending. Generates ranked, explainable loan leads using Intent, Capacity, and "
        "Guardrail scoring engines with SHAP-backed explainability and adverse action "
        "reason codes. "
        "\n\n⚠️ This demo is seeded with synthetically generated data. In a real "
        "deployment, connect your own transaction data via the /api/score/adhoc "
        "endpoint or the CreditIntelligencePipeline library — see app/pipeline.py."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — allow frontend origin
origins = settings.cors_origins_list
allow_all = "*" in origins or (len(origins) == 1 and origins[0] == "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else origins,
    allow_credentials=not allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
from .api.routes_customers import router as customers_router
from .api.routes_leads import router as leads_router
from .api.routes_score import router as score_router
from .api.routes_benchmark import router as benchmark_router

app.include_router(customers_router)
app.include_router(leads_router)
app.include_router(score_router)
app.include_router(benchmark_router)


@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "service": "CreditSetu API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "note": "All data is synthetic — for prototype demonstration only",
    }


@app.get("/api/health", tags=["Health"])
def health_check():
    """Detailed health check."""
    from .database import SessionLocal
    try:
        db = SessionLocal()
        from .models.customer import Customer
        count = db.query(Customer).count()
        db.close()
        return {
            "status": "healthy",
            "database": "connected",
            "customers_loaded": count,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }

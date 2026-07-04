"""
Application configuration for CreditSetu.

Uses Pydantic Settings for environment variable management.
All settings can be overridden via .env file or environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str = "sqlite:///data/creditsetu.db"

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # Data generation defaults
    DEFAULT_SEED: int = 42
    N_CUSTOMERS: int = 1000

    # API
    API_PREFIX: str = "/api"

    # Model paths
    CAPACITY_MODEL_PATH: str = "data/models/capacity_model.pkl"
    GUARDRAIL_MODEL_PATH: str = "data/models/guardrail_model.pkl"

    # Benchmark
    BENCHMARK_REPORT_PATH: str = "data/benchmark_report.json"
    BENCHMARK_REPORT_MD_PATH: str = "data/benchmark_report.md"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


# Singleton
settings = Settings()

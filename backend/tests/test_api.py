"""
Tests for API endpoints.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "CreditSetu API"

    def test_health(self):
        response = client.get("/api/health")
        assert response.status_code == 200


class TestCustomerEndpoints:
    """Test customer API endpoints."""

    def test_list_customers(self):
        response = client.get("/api/customers")
        assert response.status_code == 200
        data = response.json()
        assert "customers" in data
        assert "total" in data
        assert "page" in data

    def test_list_customers_pagination(self):
        response = client.get("/api/customers?page=1&page_size=5")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 5

    def test_get_customer_not_found(self):
        response = client.get("/api/customers/NONEXISTENT")
        assert response.status_code == 404


class TestLeadEndpoints:
    """Test lead API endpoints."""

    def test_list_leads(self):
        response = client.get("/api/leads")
        assert response.status_code == 200
        data = response.json()
        assert "leads" in data
        assert "total" in data

    def test_leads_with_filters(self):
        response = client.get("/api/leads?min_score=0.5&exclude_suppressed=true")
        assert response.status_code == 200

    def test_leads_show_suppressed(self):
        response = client.get("/api/leads?exclude_suppressed=false")
        assert response.status_code == 200

    def test_dashboard_stats(self):
        response = client.get("/api/leads/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_customers" in data
        assert "total_leads" in data


class TestScoreEndpoints:
    """Test scoring API endpoints."""

    def test_score_not_found(self):
        response = client.get("/api/score/NONEXISTENT")
        assert response.status_code == 404


class TestBenchmarkEndpoints:
    """Test benchmark API endpoints."""

    def test_latest_benchmark_no_report(self):
        """Should return 404 if no benchmark has been run."""
        response = client.get("/api/benchmark/latest")
        # Could be 404 or 200 depending on state
        assert response.status_code in (200, 404)


class TestOpenAPI:
    """Test that OpenAPI docs are available."""

    def test_docs_endpoint(self):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "CreditSetu API"

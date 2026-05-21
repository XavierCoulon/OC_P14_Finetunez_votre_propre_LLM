"""Tests de l'API FastAPI CHSA Triage (sans GPU — vLLM mocké)."""
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import Response

os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("VLLM_URL", "http://localhost:8000")

from src.api.main import app  # noqa: E402

client = TestClient(app)

MOCK_VLLM_RESPONSE = {
    "choices": [{"text": "Priorité P1 — Urgence absolue. Suspicion SCA. ECG immédiat."}]
}


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_metrics() -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert "error_rate" in data


def test_triage_no_api_key() -> None:
    resp = client.post("/v1/triage", json={"patient_description": "fièvre"})
    assert resp.status_code == 401


def test_triage_empty_description() -> None:
    resp = client.post(
        "/v1/triage",
        json={"patient_description": ""},
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 422


def test_triage_vllm_unavailable_returns_500() -> None:
    # Sans vLLM démarré, l'API doit retourner 500 avec un message d'erreur JSON
    resp = client.post(
        "/v1/triage",
        json={"patient_description": "Homme 52 ans, douleur thoracique.", "think": False},
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 500
    assert "detail" in resp.json()

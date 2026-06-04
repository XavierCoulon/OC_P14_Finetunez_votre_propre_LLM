"""Tests de l'API FastAPI CHSA Triage (sans GPU — vLLM mocké)."""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("VLLM_URL", "http://localhost:8000")

from src.api.main import (  # noqa: E402
    _detect_language,
    _ensure_priority_first,
    app,
)

client = TestClient(app)

# Réponse vLLM au format chat/completions correct (choices[0].message.content)
def _mock_vllm_response(content: str, reasoning: str | None = None):
    message = MagicMock()
    message.get = lambda key, default=None: {
        "content": content,
        "reasoning": reasoning,
    }.get(key, default)
    return {"choices": [{"message": message}]}


# ── Endpoints infrastructure ──────────────────────────────────────────────────

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "model" in resp.json()


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_triage_no_api_key():
    resp = client.post("/v1/triage", json={"patient_description": "fièvre"})
    assert resp.status_code == 401


def test_triage_wrong_api_key():
    resp = client.post(
        "/v1/triage",
        json={"patient_description": "fièvre"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


# ── Validation ────────────────────────────────────────────────────────────────

def test_triage_empty_description():
    resp = client.post(
        "/v1/triage",
        json={"patient_description": ""},
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 422


def test_triage_vllm_unavailable_returns_500():
    resp = client.post(
        "/v1/triage",
        json={"patient_description": "Homme 52 ans, douleur thoracique.", "think": False},
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 500
    assert "detail" in resp.json()


# ── Réponse nominale (vLLM mocké) ─────────────────────────────────────────────

def test_triage_nominal_response():
    mock_content = "**P1 – Absolute Emergency**\n\nSuspected ACS. Immediate ECG required."
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": mock_content,
                "reasoning": None,
            }
        }]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("src.api.main.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)
        resp = client.post(
            "/v1/triage",
            json={"patient_description": "58-year-old male with chest pain.", "think": False},
            headers={"X-API-Key": "test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "triage_response" in data
    assert "request_id" in data
    assert "latency_ms" in data
    assert data["triage_response"].startswith("**P1")


def test_triage_thinking_null_when_no_reasoning():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "**P2 – Relative Emergency**\n\nStable.", "reasoning": None}}]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("src.api.main.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)
        resp = client.post(
            "/v1/triage",
            json={"patient_description": "Patient avec fièvre légère.", "think": True},
            headers={"X-API-Key": "test-key"},
        )

    assert resp.status_code == 200
    assert resp.json()["thinking"] is None


# ── _detect_language ──────────────────────────────────────────────────────────

class TestDetectLanguage:

    def test_french_detected(self):
        assert _detect_language("J'ai mal aux dents depuis hier.") == "fr"

    def test_english_detected(self):
        assert _detect_language("58-year-old male with chest pain and shortness of breath.") == "en"

    def test_french_medical(self):
        assert _detect_language("Homme de 52 ans avec douleur thoracique irradiant au bras gauche.") == "fr"

    def test_english_medical(self):
        assert _detect_language("Female patient, history of hypertension, presenting with headache.") == "en"

    def test_empty_defaults_to_french(self):
        assert _detect_language("") == "fr"


# ── _ensure_priority_first ────────────────────────────────────────────────────

class TestEnsurePriorityFirst:

    def test_already_starts_with_p1(self):
        text = "**P1 – Absolute Emergency**\n\nImmediate action required."
        assert _ensure_priority_first(text).startswith("**P1")

    def test_priority_in_middle_moved_to_top(self):
        text = "The patient presents with severe symptoms.\n\n**P1 – Absolute Emergency**\n\nECG immediately."
        result = _ensure_priority_first(text)
        assert result.startswith("**P1")

    def test_p2_moved_to_top(self):
        text = "Assessment complete. **P2 – Relative Emergency** noted. Monitor closely."
        result = _ensure_priority_first(text)
        assert result.startswith("**P2")

    def test_p3_moved_to_top(self):
        text = "Non-urgent. Classification: **P3 – Deferred Emergency**."
        result = _ensure_priority_first(text)
        assert result.startswith("**P3")

    def test_no_priority_returns_unchanged(self):
        text = "Some response without any priority level."
        result = _ensure_priority_first(text)
        assert result == text

    def test_empty_string_returns_empty(self):
        assert _ensure_priority_first("") == ""

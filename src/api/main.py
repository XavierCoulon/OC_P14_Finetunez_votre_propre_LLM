import os
import re
import time
from collections import deque
from statistics import median, quantiles

import httpx
from fastapi import Depends, FastAPI, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from .middleware import audit_log, new_request_id, verify_api_key
from .models import TriageRequest, TriageResponse

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "XavierCoulon/qwen3-1.7b-chsa-sft-merged")
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # requis pour HF Inference Endpoints
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT_FR = """Tu es un agent de triage médical pour le Centre Hospitalier Saint-Aurélien (CHSA).
À partir de la description du patient, tu dois :
1. Classer le cas selon le niveau de priorité :
   - **P1 – Urgence absolue** : pronostic vital engagé, prise en charge immédiate (< 5 min)
   - **P2 – Urgence relative** : situation grave mais stable, prise en charge rapide (< 20 min)
   - **P3 – Urgence différée** : situation non critique, peut attendre (< 2h)
2. Justifier brièvement la classification
3. Indiquer les premiers gestes ou examens prioritaires

Réponds en français en commençant toujours par le niveau de priorité en gras."""

SYSTEM_PROMPT_EN = """You are a medical triage agent for the Centre Hospitalier Saint-Aurélien (CHSA).
Based on the patient description, you must:
1. Classify the case by priority level:
   - **P1 – Absolute Emergency**: vital prognosis at risk, immediate care (< 5 min)
   - **P2 – Relative Emergency**: serious but stable, urgent care (< 20 min)
   - **P3 – Deferred Emergency**: non-critical, can wait (< 2h)
2. Briefly justify the classification
3. Indicate the first actions or priority examinations

Answer in English, always starting with the priority level in bold."""

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(
    title="CHSA Triage API",
    version="1.0.0",
    swagger_ui_parameters={"persistAuthorization": True},
)

# Rolling window pour les métriques (1 000 dernières requêtes)
_latencies: deque[float] = deque(maxlen=1000)
_errors: int = 0
_total: int = 0
_start_time = time.time()


def _detect_language(text: str) -> str:
    """Détection légère basée sur les mots fréquents FR/EN — sans dépendance externe."""
    fr_markers = {"le", "la", "les", "de", "du", "des", "un", "une", "je", "j'ai",
                  "et", "en", "au", "aux", "est", "avec", "depuis", "par", "sur"}
    en_markers = {"the", "a", "an", "of", "with", "and", "in", "is", "for", "my",
                  "have", "i", "pain", "since", "years", "old", "history"}
    words = set(text.lower().split())
    fr_score = len(words & fr_markers)
    en_score = len(words & en_markers)
    return "en" if en_score > fr_score else "fr"


def _build_user_content(description: str, think: bool) -> str:
    tag = "/think" if think else "/no_think"
    return f"{tag}\n{description}"


def _extract_thinking(text: str) -> tuple[str, str | None]:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    thinking = match.group(1).strip() if match else None
    response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return response, thinking


@app.post("/v1/triage", response_model=TriageResponse, dependencies=[Depends(verify_api_key), Security(api_key_header)])
async def triage(body: TriageRequest, request: Request) -> TriageResponse:
    global _errors, _total
    request_id = new_request_id()
    t0 = time.monotonic()
    status_code = 200

    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        async with httpx.AsyncClient(timeout=60.0, headers=headers) as client:
            resp = await client.post(
                f"{VLLM_URL}/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_EN if _detect_language(body.patient_description) == "en" else SYSTEM_PROMPT_FR},
                        {"role": "user", "content": _build_user_content(body.patient_description, body.think)}
                    ],
                    "max_tokens": MAX_NEW_TOKENS,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "repetition_penalty": 1.2,
                    "chat_template_kwargs": {"enable_thinking": body.think},
                },
            )
            resp.raise_for_status()

        raw = resp.json()["choices"][0]["message"]["content"]
        response_text, thinking = _extract_thinking(raw)
        _total += 1
    except Exception as exc:
        _errors += 1
        _total += 1
        status_code = 500
        latency_ms = (time.monotonic() - t0) * 1000
        audit_log(request_id, body.patient_description, latency_ms, status_code)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    latency_ms = (time.monotonic() - t0) * 1000
    _latencies.append(latency_ms)
    audit_log(request_id, body.patient_description, latency_ms, status_code)

    return TriageResponse(
        request_id=request_id,
        triage_response=response_text,
        thinking=thinking,
        latency_ms=round(latency_ms, 1),
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/metrics")
async def metrics() -> dict:
    lats = list(_latencies)
    p50 = round(median(lats), 1) if lats else None
    p95 = round(quantiles(lats, n=20)[18], 1) if len(lats) >= 20 else None
    p99 = round(quantiles(lats, n=100)[98], 1) if len(lats) >= 100 else None
    return {
        "uptime_s": round(time.time() - _start_time),
        "total_requests": _total,
        "errors": _errors,
        "error_rate": round(_errors / _total, 4) if _total else 0,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
    }

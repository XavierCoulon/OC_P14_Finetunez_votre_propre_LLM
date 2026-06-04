import os
import re
import time

import httpx
from fastapi import Depends, FastAPI, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from .middleware import audit_log, new_request_id, verify_api_key
from .models import TriageRequest, TriageResponse

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "XavierCoulon/qwen3-1.7b-chsa-sft-merged")
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # requis pour HF Inference Endpoints
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT_FR = """Tu es un agent de triage médical pour le Centre Hospitalier Saint-Aurélien (CHSA).
À partir de la description du patient, tu dois :
1. Classer le cas selon le niveau de priorité :
   - **P1 – Urgence absolue** : pronostic vital engagé, prise en charge immédiate (< 5 min)
   - **P2 – Urgence relative** : situation grave mais stable, prise en charge rapide (< 20 min)
   - **P3 – Urgence différée** : situation non critique, peut attendre (< 2h)
2. Justifier brièvement la classification
3. Indiquer les premiers gestes ou examens prioritaires

IMPORTANT : Ta réponse DOIT commencer par le niveau de priorité en gras, par exemple : **P1 – Urgence absolue**"""

SYSTEM_PROMPT_EN = """You are a medical triage agent for the Centre Hospitalier Saint-Aurélien (CHSA).
Based on the patient description, you must:
1. Classify the case by priority level:
   - **P1 – Absolute Emergency**: vital prognosis at risk, immediate care (< 5 min)
   - **P2 – Relative Emergency**: serious but stable, urgent care (< 20 min)
   - **P3 – Deferred Emergency**: non-critical, can wait (< 2h)
2. Briefly justify the classification
3. Indicate the first actions or priority examinations

IMPORTANT: Your response MUST start with the priority level in bold, e.g.: **P1 – Absolute Emergency**"""

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(
    title="CHSA Triage API",
    version="1.0.0",
    swagger_ui_parameters={"persistAuthorization": True},
)


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


def _ensure_priority_first(text: str) -> str:
    """Déplace le niveau de priorité (P1/P2/P3) en tête de réponse s'il n'y est pas déjà."""
    if not text:
        return text
    first_line = text.split("\n")[0]
    if re.match(r"^\*?\*?P[123]", first_line):
        return text  # déjà en tête
    match = re.search(r"(\*{0,2}P[123][^*\n]*\*{0,2})", text)
    if match:
        priority = match.group(1).strip("* ").strip()
        body = text[:match.start()].strip() + "\n\n" + text[match.end():].strip()
        return f"**{priority}**\n\n{body.strip()}"
    return text


def _extract_thinking(text: str) -> tuple[str, str | None]:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    thinking = match.group(1).strip() if match else None
    response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return response, thinking


@app.post("/v1/triage", response_model=TriageResponse, dependencies=[Depends(verify_api_key), Security(api_key_header)])
async def triage(body: TriageRequest, request: Request) -> TriageResponse:
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
                    "chat_template_kwargs": {"enable_thinking": bool(body.think)},
                },
            )
            resp.raise_for_status()

        message = resp.json()["choices"][0]["message"]
        content = message.get("content") or ""
        # vLLM 0.18.x avec --reasoning-parser qwen3 : thinking dans "reasoning"
        # Le modèle 1.7B place parfois toute la réponse dans reasoning sans content final
        reasoning = (message.get("reasoning") or "").strip()
        if reasoning and not content:
            response_text, thinking = _extract_thinking(reasoning)
            if not thinking:
                response_text = reasoning
        elif reasoning:
            thinking = reasoning or None
            response_text = content.strip()
        else:
            response_text, thinking = _extract_thinking(content)
        response_text = _ensure_priority_first(response_text)
    except Exception as exc:
        status_code = 500
        latency_ms = (time.monotonic() - t0) * 1000
        audit_log(request_id, body.patient_description, latency_ms, status_code)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    latency_ms = (time.monotonic() - t0) * 1000
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

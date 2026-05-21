import os
import re
import time
from collections import deque
from statistics import median, quantiles

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from .middleware import audit_log, new_request_id, verify_api_key
from .models import TriageRequest, TriageResponse

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "XavierCoulon/qwen3-1.7b-chsa-sft-merged")
MAX_NEW_TOKENS = 512

app = FastAPI(title="CHSA Triage API", version="1.0.0")

# Rolling window pour les métriques (1 000 dernières requêtes)
_latencies: deque[float] = deque(maxlen=1000)
_errors: int = 0
_total: int = 0
_start_time = time.time()


def _build_prompt(description: str, think: bool) -> str:
    tag = "/think" if think else "/no_think"
    return (
        f"<|im_start|>user\n{tag}\n{description}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _extract_thinking(text: str) -> tuple[str, str | None]:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    thinking = match.group(1).strip() if match else None
    response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return response, thinking


@app.post("/v1/triage", response_model=TriageResponse, dependencies=[Depends(verify_api_key)])
async def triage(body: TriageRequest, request: Request) -> TriageResponse:
    global _errors, _total
    request_id = new_request_id()
    t0 = time.monotonic()
    status_code = 200

    try:
        prompt = _build_prompt(body.patient_description, body.think)
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{VLLM_URL}/v1/completions",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "max_tokens": MAX_NEW_TOKENS,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "repetition_penalty": 1.2,
                },
            )
            resp.raise_for_status()

        raw = resp.json()["choices"][0]["text"]
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

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import HTTPException, Request, status

AUDIT_LOG = Path("audit/api_log.jsonl")
AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

_API_KEY = os.environ.get("API_KEY", "")


def verify_api_key(request: Request) -> None:
    key = request.headers.get("X-API-Key", "")
    if not _API_KEY or key != _API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def audit_log(request_id: str, patient_text: str, latency_ms: float, status_code: int) -> None:
    # Ne jamais logger le texte patient brut (RGPD) — uniquement le hash SHA-256
    patient_hash = hashlib.sha256(patient_text.encode()).hexdigest()
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request_id": request_id,
        "patient_hash": patient_hash,
        "latency_ms": round(latency_ms, 1),
        "status": status_code,
    }
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def new_request_id() -> str:
    return str(uuid.uuid4())

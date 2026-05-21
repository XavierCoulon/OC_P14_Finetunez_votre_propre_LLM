from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    patient_description: str = Field(..., min_length=1, max_length=4000)
    think: bool = Field(True, description="Active le mode raisonnement Qwen3 (/think)")


class TriageResponse(BaseModel):
    request_id: str
    triage_response: str
    thinking: str | None = None
    latency_ms: float

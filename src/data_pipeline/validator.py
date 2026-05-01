"""
Validation qualité du dataset final.
"""
import json
import re
from pathlib import Path

PII_PATTERNS = [
    re.compile(r"\b\d{10,}\b"),                         # numéros longs (SS, tél)
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.\w+"),  # emails
    re.compile(r"\b0[0-9]{9}\b"),                        # tél FR
]

MIN_INSTRUCTION_TOKENS = 3
MAX_INSTRUCTION_TOKENS = 2000
MIN_RESPONSE_TOKENS = 5
MAX_RESPONSE_TOKENS = 1500


def _approx_tokens(text: str) -> int:
    return len(text.split())


def validate_sft_record(record: dict) -> list[str]:
    errors = []
    for field in ["id", "source", "language", "instruction", "response"]:
        if not record.get(field):
            errors.append(f"Champ manquant : {field}")
    instr_len = _approx_tokens(record.get("instruction", ""))
    resp_len = _approx_tokens(record.get("response", ""))
    if instr_len < MIN_INSTRUCTION_TOKENS:
        errors.append(f"instruction trop courte ({instr_len} mots)")
    if instr_len > MAX_INSTRUCTION_TOKENS:
        errors.append(f"instruction trop longue ({instr_len} mots)")
    if resp_len < MIN_RESPONSE_TOKENS:
        errors.append(f"response trop courte ({resp_len} mots)")
    for pat in PII_PATTERNS:
        for field in ["instruction", "response"]:
            if pat.search(record.get(field, "")):
                errors.append(f"PII potentiel détecté dans {field}")
                break
    return errors


def validate_dpo_record(record: dict) -> list[str]:
    errors = []
    for field in ["id", "source", "prompt", "chosen", "rejected"]:
        if not record.get(field):
            errors.append(f"Champ manquant : {field}")
    chosen_content = (record.get("chosen") or {}).get("content", "")
    rejected_content = (record.get("rejected") or {}).get("content", "")
    if chosen_content == rejected_content:
        errors.append("chosen == rejected")
    if not chosen_content:
        errors.append("chosen.content vide")
    if not rejected_content:
        errors.append("rejected.content vide")
    return errors


def validate_file(file_path: Path, dataset_type: str = "sft") -> dict:
    errors_by_record = {}
    total = 0
    validator = validate_sft_record if dataset_type == "sft" else validate_dpo_record

    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            errors = validator(record)
            if errors:
                errors_by_record[record.get("id", str(i))] = errors
            total += 1

    return {"total": total, "errors": errors_by_record, "valid": total - len(errors_by_record)}

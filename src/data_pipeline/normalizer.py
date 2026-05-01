"""
Normalisation des données brutes vers le schéma unifié SFT/DPO.
Entrée  : data/raw/<source>/<source>_raw.jsonl
Sortie  : data/interim/normalized/<source>_normalized.jsonl
"""
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone


def _log_entry(transformation_id: str, source_file: str, output_file: str, record_id: str) -> dict:
    return {
        "transformation_id": transformation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_record_id": record_id,
        "operation": "normalization",
        "source_file": source_file,
        "output_file": output_file,
    }


def normalize_sft_file(input_file: Path, output_file: Path, audit_log: Path) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    source_name = input_file.stem.replace("_raw", "")

    count = 0
    log_entries = []

    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            raw = json.loads(line)
            transformation_id = f"norm_{source_name}_{i:06d}"
            record_id = f"sft_{source_name}_{i:06d}"

            record = {
                "id": record_id,
                "source": raw.get("source", source_name),
                "language": raw.get("language", "en"),
                "task_type": "medical_qa",
                "instruction": raw.get("question", "").strip(),
                "response": raw.get("answer", "").strip(),
                "metadata": {
                    "symptoms": [],
                    "antecedents": [],
                    "constantes": {},
                    "confidence_level": "medium",
                    "original_source_id": raw.get("original_id", str(i)),
                    "transformation_ids": [transformation_id],
                    "split": None,
                },
            }

            if not record["instruction"] or not record["response"]:
                continue
            # Filtrer les réponses trop courtes (ex: QCM lettre seule)
            if len(record["response"].split()) < 5:
                continue
            # Filtrer les instructions trop longues (textes cliniques >2000 mots)
            if len(record["instruction"].split()) > 2000:
                continue

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_entries.append(_log_entry(transformation_id, str(input_file), str(output_file), record_id))
            count += 1

    _append_audit(audit_log, log_entries)
    return count


def normalize_dpo_file(input_file: Path, output_file: Path, audit_log: Path) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    log_entries = []

    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            raw = json.loads(line)
            transformation_id = f"norm_dpo_{i:06d}"
            record_id = f"dpo_ultramedical_{i:06d}"

            record = {
                "id": record_id,
                "source": raw.get("source", "ultramedical_preference"),
                "language": raw.get("language", "en"),
                "task_type": "medical_qa",
                "prompt": raw.get("prompt", "").strip(),
                "chosen": {"role": "assistant", "content": raw.get("chosen", "").strip()},
                "rejected": {"role": "assistant", "content": raw.get("rejected", "").strip()},
                "metadata": {
                    "symptoms": [],
                    "antecedents": [],
                    "constantes": {},
                    "confidence_level": "high",
                    "original_source_id": raw.get("original_id", str(i)),
                    "chosen_score": raw.get("chosen_score", 1.0),
                    "rejected_score": raw.get("rejected_score", 0.0),
                    "transformation_ids": [transformation_id],
                    "split": None,
                },
            }

            if not record["prompt"] or not record["chosen"]["content"] or not record["rejected"]["content"]:
                continue

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_entries.append(_log_entry(transformation_id, str(input_file), str(output_file), record_id))
            count += 1

    _append_audit(audit_log, log_entries)
    return count


def _append_audit(audit_log: Path, entries: list[dict]) -> None:
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_log, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

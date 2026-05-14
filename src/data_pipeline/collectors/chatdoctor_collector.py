"""
Collecteur ChatDoctor (HealthcareMagic) — lavita/medical-qa-datasets, config chatdoctor_healthcaremagic.

Structure HF :
  instruction : consigne générique ("If you are a doctor...") — ignorée
  input       : vraie question patient avec description des symptômes
  output      : réponse du médecin

Mapping → schéma unifié :
  input  → question
  output → answer
"""
from datasets import load_dataset
from pathlib import Path
import json


def collect(target_pairs: int, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chatdoctor_raw.jsonl"

    dataset = load_dataset(
        "lavita/medical-qa-datasets",
        "chatdoctor_healthcaremagic",
        split="train",
    )

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for row in dataset:
            if count >= target_pairs:
                break
            # "instruction" est une consigne générique → on prend "input" comme question patient
            question = (row.get("input") or "").strip()
            answer = (row.get("output") or "").strip()

            # Filtres qualité minimaux
            if len(question) < 30 or len(answer) < 50:
                continue

            record = {
                "source": "chatdoctor",
                "language": "en",
                "question": question,
                "answer": answer,
                "original_id": str(row.get("__index_level_0__", count)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count

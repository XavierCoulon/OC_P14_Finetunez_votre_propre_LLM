from datasets import load_dataset
from pathlib import Path
import json


def collect(target_pairs: int, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mediqa_raw.jsonl"

    dataset = load_dataset("lavita/medical-qa-datasets", "medical_meadow_mediqa", split="train")

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for row in dataset:
            if count >= target_pairs:
                break
            question = (row.get("input") or row.get("question") or "").strip()
            answer = (row.get("output") or row.get("answer") or "").strip()
            if not question or not answer:
                continue
            record = {
                "source": "mediqa",
                "language": "en",
                "question": question,
                "answer": answer,
                "original_id": str(row.get("id", count)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count

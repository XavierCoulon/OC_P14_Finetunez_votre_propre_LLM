from datasets import load_dataset
from pathlib import Path
import json


def collect(target_pairs: int, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "medquad_raw.jsonl"

    dataset = load_dataset("lavita/MedQuAD", split="train")

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for row in dataset:
            if count >= target_pairs:
                break
            question = row.get("question", "").strip()
            answer = row.get("answer", "").strip()
            if not question or not answer:
                continue
            record = {
                "source": "medquad",
                "language": "en",
                "question": question,
                "answer": answer,
                "original_id": str(row.get("qid", count)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count

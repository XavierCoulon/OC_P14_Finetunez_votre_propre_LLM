from datasets import load_dataset
from pathlib import Path
import json


def collect(target_pairs: int, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "frenchmedmcqa_raw.jsonl"

    # Format déjà SFT avec prompt (messages) et completion (assistant)
    dataset = load_dataset("PARTAGES-dev/frenchmedmcqa-sft", split="train")

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for row in dataset:
            if count >= target_pairs:
                break
            # Extraire le contenu user (dernière entrée role=user dans prompt)
            prompt_messages = row.get("prompt", [])
            user_content = next(
                (m["content"].strip() for m in reversed(prompt_messages) if m["role"] == "user"),
                "",
            )
            # Extraire la réponse assistant
            completion = row.get("completion", [])
            answer = next(
                (m["content"].strip() for m in completion if m["role"] == "assistant"),
                "",
            )
            if not user_content or not answer:
                continue
            record = {
                "source": "frenchmedmcqa",
                "language": "fr",
                "question": user_content,
                "answer": answer,
                "original_id": str(row.get("doc_id", count)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count

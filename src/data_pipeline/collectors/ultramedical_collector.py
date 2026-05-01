from datasets import load_dataset
from pathlib import Path
import json


def collect_sft(target_pairs: int, output_dir: Path) -> int:
    """Extrait des paires SFT depuis UltraMedical (conversations human → gpt)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ultramedical_sft_raw.jsonl"

    dataset = load_dataset("TsinghuaC3I/UltraMedical", split="train")

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for row in dataset:
            if count >= target_pairs:
                break
            convs = row.get("conversations", [])
            question = next((c["value"].strip() for c in convs if c["from"] == "human"), "")
            answer = next((c["value"].strip() for c in convs if c["from"] == "gpt"), "")
            if not question or not answer:
                continue
            record = {
                "source": "ultramedical",
                "language": "en",
                "question": question,
                "answer": answer,
                "original_id": str(row.get("id", count)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def collect_dpo(target_pairs: int, score_gap_min: float, output_dir: Path) -> int:
    """Extrait des paires DPO (chosen/rejected) depuis UltraMedical-Preference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ultramedical_dpo_raw.jsonl"

    dataset = load_dataset("TsinghuaC3I/UltraMedical-Preference", split="train")

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for row in dataset:
            if count >= target_pairs:
                break
            prompt = row.get("prompt", "").strip()
            chosen_msgs = row.get("chosen", [])
            rejected_msgs = row.get("rejected", [])

            chosen = next((m["content"].strip() for m in chosen_msgs if m["role"] == "assistant"), "")
            rejected = next((m["content"].strip() for m in rejected_msgs if m["role"] == "assistant"), "")

            metadata = row.get("metadata", {})
            chosen_score = float((metadata.get("chosen") or {}).get("score") or 1.0)
            rejected_score = float((metadata.get("rejected") or {}).get("score") or 0.0)

            if not prompt or not chosen or not rejected:
                continue
            if chosen == rejected:
                continue
            if (chosen_score - rejected_score) < score_gap_min:
                continue

            record = {
                "source": "ultramedical_preference",
                "language": "en",
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "original_id": str(row.get("prompt_id", count)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count

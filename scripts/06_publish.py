"""
Publication du dataset final : dataset_card.json + résumé des stats.
"""
import json
from pathlib import Path
from datetime import date

SFT_DIR = Path("data/processed/sft")
DPO_DIR = Path("data/processed/dpo")
EVAL_DIR = Path("data/processed/eval_clinique")
META_DIR = Path("data/metadata")
META_DIR.mkdir(parents=True, exist_ok=True)


def count_lines(path: Path) -> int:
    return sum(1 for _ in open(path, encoding="utf-8"))


def main():
    stats = {
        "dataset_name": "chsa-medical-triage",
        "description": "Corpus médical bilingue FR/EN pour fine-tuning SFT+DPO — POC CHSA",
        "created": str(date.today()),
        "anonymization": "Presidio (replace strategy)",
        "rgpd_compliant": True,
        "sft": {split: count_lines(SFT_DIR / f"{split}.jsonl") for split in ["train", "val", "test"]},
        "dpo": {split: count_lines(DPO_DIR / f"{split}.jsonl") for split in ["train", "val", "test"]},
        "eval_clinique": count_lines(EVAL_DIR / "eval_clinique.jsonl"),
        "sources": ["mediqa", "frenchmedmcqa", "medquad", "ultramedical", "ultramedical_preference"],
        "languages": ["en", "fr"],
    }

    out = META_DIR / "dataset_card.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=== Dataset Card ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n✓ Sauvegardé dans {out}")


if __name__ == "__main__":
    main()

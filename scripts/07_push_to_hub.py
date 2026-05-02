"""
Push du dataset final vers HuggingFace Hub.
Nécessite un token HF avec droits write dans .env (HF_TOKEN + HF_DATASET_REPO).
"""
import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "XavierCoulon/oc-p14-dataset")
HF_PRIVATE = os.getenv("HF_PRIVATE", "true").lower() == "true"

SFT_DIR = Path("data/processed/sft")
DPO_DIR = Path("data/processed/dpo")
EVAL_DIR = Path("data/processed/eval_clinique")


def load_jsonl(path: Path) -> list[dict]:
    return [flatten(json.loads(line)) for line in open(path, encoding="utf-8")]


def flatten(record: dict) -> dict:
    """Sérialise les champs imbriqués en JSON string (Parquet n'accepte pas les structs vides)."""
    for key in ("metadata", "chosen", "rejected"):
        if key in record and isinstance(record[key], dict):
            record[key] = json.dumps(record[key], ensure_ascii=False)
    return record


def main():
    if not HF_TOKEN:
        print("Erreur : HF_TOKEN manquant dans .env", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", private=HF_PRIVATE, exist_ok=True)
    print(f"Repo : https://huggingface.co/datasets/{HF_DATASET_REPO} (private={HF_PRIVATE})")

    configs = {
        "sft": DatasetDict({
            "train": Dataset.from_list(load_jsonl(SFT_DIR / "train.jsonl")),
            "val":   Dataset.from_list(load_jsonl(SFT_DIR / "val.jsonl")),
            "test":  Dataset.from_list(load_jsonl(SFT_DIR / "test.jsonl")),
        }),
        "dpo": DatasetDict({
            "train": Dataset.from_list(load_jsonl(DPO_DIR / "train.jsonl")),
            "val":   Dataset.from_list(load_jsonl(DPO_DIR / "val.jsonl")),
            "test":  Dataset.from_list(load_jsonl(DPO_DIR / "test.jsonl")),
        }),
        "eval_clinique": DatasetDict({
            "eval": Dataset.from_list(load_jsonl(EVAL_DIR / "eval_clinique.jsonl")),
        }),
    }

    for config_name, dataset in configs.items():
        print(f"Push {config_name}...", end=" ", flush=True)
        dataset.push_to_hub(HF_DATASET_REPO, config_name=config_name, private=HF_PRIVATE, token=HF_TOKEN)
        total = sum(len(v) for v in dataset.values())
        print(f"{total} enregistrements")

    print(f"\n✓ Dataset disponible : https://huggingface.co/datasets/{HF_DATASET_REPO}")


if __name__ == "__main__":
    main()

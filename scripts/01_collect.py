"""
Script de collecte des sources brutes depuis HuggingFace.
Produit : data/raw/<source>/<source>_raw.jsonl
"""
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.collectors import (
    mediqa_collector,
    frenchmedmcqa_collector,
    medquad_collector,
    ultramedical_collector,
)

CONFIG_PATH = Path("configs/sources.yaml")
RAW_DIR = Path("data/raw")


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    sources = config["sources"]
    total = 0

    print("=== Collecte MediQA ===")
    n = mediqa_collector.collect(
        target_pairs=sources["mediqa"]["target_pairs"],
        output_dir=RAW_DIR / "mediqa",
    )
    print(f"  → {n} paires collectées")
    total += n

    print("=== Collecte FrenchMedMCQA ===")
    n = frenchmedmcqa_collector.collect(
        target_pairs=sources["frenchmedmcqa"]["target_pairs"],
        output_dir=RAW_DIR / "frenchmedmcqa",
    )
    print(f"  → {n} paires collectées")
    total += n

    print("=== Collecte MedQuAD ===")
    n = medquad_collector.collect(
        target_pairs=sources["medquad"]["target_pairs"],
        output_dir=RAW_DIR / "medquad",
    )
    print(f"  → {n} paires collectées")
    total += n

    print("=== Collecte UltraMedical (SFT) ===")
    n = ultramedical_collector.collect_sft(
        target_pairs=sources["ultramedical"]["target_pairs_sft"],
        output_dir=RAW_DIR / "ultramedical_preference",
    )
    print(f"  → {n} paires SFT collectées")
    total += n

    print("=== Collecte UltraMedical (DPO) ===")
    n = ultramedical_collector.collect_dpo(
        target_pairs=sources["ultramedical"]["target_pairs_dpo"],
        score_gap_min=sources["ultramedical"]["dpo_score_gap_min"],
        output_dir=RAW_DIR / "ultramedical_preference",
    )
    print(f"  → {n} paires DPO collectées")

    print(f"\n✓ Total SFT brut : {total} paires")
    print(f"✓ Données dans : {RAW_DIR}/")


if __name__ == "__main__":
    main()

"""
Normalisation de toutes les sources brutes vers le schéma unifié.
Entrée  : data/raw/
Sortie  : data/interim/normalized/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.normalizer import normalize_sft_file, normalize_dpo_file

RAW_DIR = Path("data/raw")
NORM_DIR = Path("data/interim/normalized")
AUDIT_LOG = Path("audit/transformation_log.jsonl")

SFT_FILES = [
    RAW_DIR / "mediqa" / "mediqa_raw.jsonl",
    RAW_DIR / "frenchmedmcqa" / "frenchmedmcqa_raw.jsonl",
    RAW_DIR / "medquad" / "medquad_raw.jsonl",
    RAW_DIR / "ultramedical_preference" / "ultramedical_sft_raw.jsonl",
]

DPO_FILES = [
    RAW_DIR / "ultramedical_preference" / "ultramedical_dpo_raw.jsonl",
]


def main():
    total_sft = 0
    for f in SFT_FILES:
        out = NORM_DIR / f.name.replace("_raw", "_normalized")
        n = normalize_sft_file(f, out, AUDIT_LOG)
        print(f"  SFT {f.name} → {n} enregistrements")
        total_sft += n

    total_dpo = 0
    for f in DPO_FILES:
        out = NORM_DIR / f.name.replace("_raw", "_normalized")
        n = normalize_dpo_file(f, out, AUDIT_LOG)
        print(f"  DPO {f.name} → {n} enregistrements")
        total_dpo += n

    print(f"\n✓ SFT normalisé : {total_sft} paires")
    print(f"✓ DPO normalisé : {total_dpo} paires")
    print(f"✓ Log RGPD      : {AUDIT_LOG}")


if __name__ == "__main__":
    main()

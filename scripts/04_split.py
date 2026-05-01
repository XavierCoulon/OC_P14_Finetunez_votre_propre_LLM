"""
Création des splits train/val/test + eval_clinique.
"""
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.splitter import split_sft, split_dpo

CONFIG_PATH = Path("configs/split.yaml")
ANON_DIR = Path("data/interim/anonymized")
SFT_OUT = Path("data/processed/sft")
DPO_OUT = Path("data/processed/dpo")
EVAL_OUT = Path("data/processed/eval_clinique")


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    print("=== Split SFT ===")
    sft_stats = split_sft(
        anon_dir=ANON_DIR,
        output_dir=SFT_OUT,
        eval_dir=EVAL_OUT,
        train_ratio=cfg["sft"]["train"],
        val_ratio=cfg["sft"]["val"],
        seed=cfg["seed"],
        eval_size=cfg["eval_clinique"]["size"],
    )
    for k, v in sft_stats.items():
        print(f"  {k:6s} : {v}")

    print("=== Split DPO ===")
    dpo_stats = split_dpo(
        anon_dir=ANON_DIR,
        output_dir=DPO_OUT,
        train_ratio=cfg["dpo"]["train"],
        val_ratio=cfg["dpo"]["val"],
        seed=cfg["seed"],
    )
    for k, v in dpo_stats.items():
        print(f"  {k:6s} : {v}")


if __name__ == "__main__":
    main()

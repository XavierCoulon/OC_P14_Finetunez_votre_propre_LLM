"""
Validation qualité de tous les splits finaux.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.validator import validate_file

SFT_DIR = Path("data/processed/sft")
DPO_DIR = Path("data/processed/dpo")
EVAL_DIR = Path("data/processed/eval_clinique")

EXIT_CODE = 0


def check(path, dtype):
    global EXIT_CODE
    result = validate_file(path, dataset_type=dtype)
    n_errors = len(result["errors"])
    status = "✓" if n_errors == 0 else "✗"
    print(f"  {status} {path.name}: {result['valid']}/{result['total']} valides, {n_errors} erreurs")
    if n_errors > 0:
        EXIT_CODE = 1
        for rec_id, errs in list(result["errors"].items())[:3]:
            print(f"      [{rec_id}] {errs}")


def main():
    print("=== Validation SFT ===")
    for f in sorted(SFT_DIR.glob("*.jsonl")):
        check(f, "sft")

    print("=== Validation DPO ===")
    for f in sorted(DPO_DIR.glob("*.jsonl")):
        check(f, "dpo")

    print("=== Validation eval_clinique ===")
    for f in sorted(EVAL_DIR.glob("*.jsonl")):
        check(f, "sft")

    if EXIT_CODE == 0:
        print("\n✓ Tous les splits sont valides.")
    else:
        print("\n✗ Des erreurs ont été détectées.")
    sys.exit(EXIT_CODE)


if __name__ == "__main__":
    main()

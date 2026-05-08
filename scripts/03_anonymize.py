"""
Anonymisation de masse avec Presidio.
Entrée  : data/interim/normalized/
Sortie  : data/interim/anonymized/
"""
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.anonymizer import anonymize_file

NORM_DIR = Path("data/interim/normalized")
ANON_DIR = Path("data/interim/anonymized")
AUDIT_LOG = Path("audit/transformation_log.jsonl")

with open("configs/sources.yaml") as f:
    _sources = yaml.safe_load(f)["sources"]

_disabled = {k for k, v in _sources.items() if not v.get("enabled", True)}
_skip_anonymization = {k for k, v in _sources.items() if v.get("skip_anonymization", False)}


def main():
    sft_files = [
        f for f in NORM_DIR.glob("*_normalized.jsonl")
        if "dpo" not in f.name
        and not any(d in f.name for d in _disabled)
    ]
    dpo_files = [f for f in NORM_DIR.glob("*dpo*_normalized.jsonl")]

    total_records, total_entities = 0, 0

    print("=== Anonymisation SFT ===")
    if _skip_anonymization:
        print(f"  Sources sans anonymisation (pas de PII patient) : {', '.join(_skip_anonymization)}")
    for f in sft_files:
        out = ANON_DIR / f.name.replace("_normalized", "_anonymized")
        n, e = anonymize_file(f, out, AUDIT_LOG, dataset_type="sft", skip_sources=_skip_anonymization)
        print(f"  {f.name} → {n} enregistrements, {e} entités masquées")
        total_records += n
        total_entities += e

    print("=== Anonymisation DPO ===")
    for f in dpo_files:
        out = ANON_DIR / f.name.replace("_normalized", "_anonymized")
        n, e = anonymize_file(f, out, AUDIT_LOG, dataset_type="dpo", skip_sources=_skip_anonymization)
        print(f"  {f.name} → {n} enregistrements, {e} entités masquées")
        total_records += n
        total_entities += e

    print(f"\n✓ Total : {total_records} enregistrements, {total_entities} entités anonymisées")


if __name__ == "__main__":
    main()

"""
Anonymisation des données avec Microsoft Presidio.
Stratégie : replace → <PERSON>, <DATE_TIME>, etc.

Améliorations vs v1 :
- Filtre longueur minimale : entités < 3 chars ignorées (évite A/B/C/D/E des QCM)
- Allowlist médicale : termes médicaux latins/grecs fréquemment mal taguées
- Seuil NRP relevé à 0.85 : réduit les faux positifs sur organisations médicales
"""
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


ENTITIES = ["PERSON", "LOCATION", "DATE_TIME", "PHONE_NUMBER", "EMAIL_ADDRESS", "NRP"]
SCORE_THRESHOLD = 0.70
NRP_SCORE_THRESHOLD = 0.85   # plus strict pour éviter les orgas médicales
MIN_ENTITY_LENGTH = 3        # ignore les entités < 3 chars (lettres QCM A/B/C...)

# Termes médicaux fréquemment mal taguées comme PERSON/LOCATION
MEDICAL_ALLOWLIST = {
    # Termes latins / grecs
    "primigravidas", "multigravidas", "primigravida", "multigravida",
    "cholesteatoma", "hypotympanum", "epitympanum", "tympanum",
    "ataxia", "dysarthria", "dysphagia", "dysnomia", "dyscalculia",
    "myopathy", "neuropathy", "nephropathy", "cardiomyopathy",
    "lymphedema", "papillomas", "lymphoma", "amyloidosis",
    "myeloma", "sarcoma", "melanoma", "adenoma", "carcinoma",
    "fibroma", "fibrothecoma",
    # Syndromes nommés (noms propres médicaux)
    "sjogren", "guillain", "barre", "charcot", "marie", "tooth",
    "alzheimer", "parkinson", "huntington", "wilson", "fabry",
    "marfan", "turner", "down", "klinefelter", "bartter", "milroy",
    # Organisations médicales
    "acog", "who", "nih", "cdc", "fda", "ema",
    "gynecologists", "obstetricians", "pediatricians",
}

# Champs de texte à anonymiser par type de dataset
SFT_TEXT_FIELDS = ["instruction", "response"]
DPO_TEXT_FIELDS = ["prompt", "chosen.content", "rejected.content"]


def _build_engines(language: str):
    model_name = "fr_core_news_md" if language == "fr" else "en_core_web_lg"
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": language, "model_name": model_name}],
    })
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=[language])
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def _filter_results(results: list, text: str) -> list:
    """
    Filtre les faux positifs :
    - Entités trop courtes (< MIN_ENTITY_LENGTH chars)
    - Termes dans la MEDICAL_ALLOWLIST
    - NRP sous le seuil strict
    """
    filtered = []
    for r in results:
        entity_text = text[r.start:r.end].strip().lower()

        # Filtre 1 : longueur minimale
        if len(entity_text) < MIN_ENTITY_LENGTH:
            continue

        # Filtre 2 : allowlist médicale
        if entity_text in MEDICAL_ALLOWLIST:
            continue

        # Filtre 3 : NRP avec seuil plus strict
        if r.entity_type == "NRP" and r.score < NRP_SCORE_THRESHOLD:
            continue

        filtered.append(r)
    return filtered


def _anonymize_text(text: str, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine, language: str) -> tuple[str, int]:
    if not text:
        return text, 0
    results = analyzer.analyze(text=text, language=language, entities=ENTITIES, score_threshold=SCORE_THRESHOLD)
    if not results:
        return text, 0

    results = _filter_results(results, text)
    if not results:
        return text, 0

    operators = {entity: OperatorConfig("replace", {"new_value": f"<{entity}>"}) for entity in ENTITIES}
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
    return anonymized.text, len(results)


def _checksum(text: str) -> str:
    return sha256(text.encode()).hexdigest()[:16]


def anonymize_file(input_file: Path, output_file: Path, audit_log: Path, dataset_type: str = "sft", skip_sources: set | None = None) -> tuple[int, int]:
    """
    skip_sources : sources à laisser intactes (pas de PII patient — ex: QCM d'examen).
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if skip_sources is None:
        skip_sources = set()

    engines: dict[str, tuple] = {}
    total_records = 0
    total_entities = 0
    log_entries = []

    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            record = json.loads(line)
            language = record.get("language", "en")
            source = record.get("source", "")

            # Source sans PII patient → on écrit le record tel quel
            if source in skip_sources:
                anon_id = f"anon_{i:06d}_skipped"
                record.get("metadata", {}).get("transformation_ids", []).append(anon_id)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_records += 1
                log_entries.append({
                    "transformation_id": anon_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "input_record_id": record.get("id", str(i)),
                    "operation": "anonymization_skipped",
                    "reason": "source flagged skip_anonymization (no patient PII)",
                    "source_file": str(input_file),
                    "output_file": str(output_file),
                })
                continue

            if language not in engines:
                engines[language] = _build_engines(language)
            analyzer, anonymizer_engine = engines[language]

            entities_count = 0

            if dataset_type == "sft":
                for field in SFT_TEXT_FIELDS:
                    original = record.get(field, "")
                    anonymized, n = _anonymize_text(original, analyzer, anonymizer_engine, language)
                    record[field] = anonymized
                    entities_count += n
            else:
                for field_path in DPO_TEXT_FIELDS:
                    parts = field_path.split(".")
                    if len(parts) == 1:
                        original = record.get(parts[0], "")
                        anonymized, n = _anonymize_text(original, analyzer, anonymizer_engine, language)
                        record[parts[0]] = anonymized
                    else:
                        obj = record.get(parts[0], {})
                        original = obj.get(parts[1], "") if isinstance(obj, dict) else ""
                        anonymized, n = _anonymize_text(original, analyzer, anonymizer_engine, language)
                        if isinstance(obj, dict):
                            obj[parts[1]] = anonymized
                    entities_count += n

            anon_id = f"anon_{i:06d}"
            record.get("metadata", {}).get("transformation_ids", []).append(anon_id)

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_entities += entities_count
            total_records += 1

            log_entries.append({
                "transformation_id": anon_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_record_id": record.get("id", str(i)),
                "operation": "anonymization",
                "tool": "presidio",
                "strategy": "replace",
                "filters": f"min_length={MIN_ENTITY_LENGTH}, allowlist={len(MEDICAL_ALLOWLIST)} terms, nrp_threshold={NRP_SCORE_THRESHOLD}",
                "entities_modified": entities_count,
                "source_file": str(input_file),
                "output_file": str(output_file),
            })

    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_log, "a", encoding="utf-8") as f:
        for entry in log_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return total_records, total_entities

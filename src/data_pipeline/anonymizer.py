"""
Anonymisation des données avec Microsoft Presidio.
Stratégie : replace → <PERSON>, <DATE_TIME>, etc.
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


ENTITIES = ["PERSON", "LOCATION", "DATE_TIME", "PHONE_NUMBER", "EMAIL_ADDRESS", "NRP"]
SCORE_THRESHOLD = 0.70

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


def _anonymize_text(text: str, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine, language: str) -> tuple[str, int]:
    if not text:
        return text, 0
    results = analyzer.analyze(text=text, language=language, entities=ENTITIES, score_threshold=SCORE_THRESHOLD)
    if not results:
        return text, 0
    operators = {entity: OperatorConfig("replace", {"new_value": f"<{entity}>"}) for entity in ENTITIES}
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
    return anonymized.text, len(results)


def _checksum(text: str) -> str:
    return sha256(text.encode()).hexdigest()[:16]


def anonymize_file(input_file: Path, output_file: Path, audit_log: Path, dataset_type: str = "sft") -> tuple[int, int]:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Pré-charger les deux moteurs
    engines: dict[str, tuple] = {}

    total_records = 0
    total_entities = 0
    log_entries = []

    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            record = json.loads(line)
            language = record.get("language", "en")

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

            # Mettre à jour les transformation_ids
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
                "entities_modified": entities_count,
                "source_file": str(input_file),
                "output_file": str(output_file),
            })

    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_log, "a", encoding="utf-8") as f:
        for entry in log_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return total_records, total_entities

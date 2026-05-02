# POC Agent IA Triage Médical – CHSA

Fine-tuning supervisé (SFT + LoRA) et alignement par préférences (DPO) du modèle Qwen3-1.7B pour l'assistance au triage médical des urgences.

## Pipeline

```
Qwen3-1.7B-Base → SFT (LoRA) → DPO → Endpoint vLLM
```

## Structure

```
data/           Données (raw hors git, processed versionné)
src/            Code source du pipeline
scripts/        Scripts d'orchestration (01_collect → 06_publish)
configs/        Paramètres (sources, presidio, split)
audit/          Logs RGPD (transformation_log.jsonl)
notebooks/      EDA et expérimentations
```

## Installation

```bash
uv sync
uv sync --extra dev
uv run python -m spacy download fr_core_news_md
uv run python -m spacy download en_core_web_lg
```

## Étape 1 – Collecte des données

```bash
uv run python scripts/01_collect.py
uv run python scripts/02_normalize.py
uv run python scripts/03_anonymize.py
uv run python scripts/04_split.py
uv run python scripts/05_validate.py
```

## Sources de données

| Source | HuggingFace | Langue | Type | Licence | Paires retenues |
|---|---|---|---|---|---|
| MediQA | [lavita/medical-qa-datasets](https://huggingface.co/datasets/lavita/medical-qa-datasets) | EN | QA médical (SFT) | Apache 2.0 | 948 |
| FrenchMedMCQA | [PARTAGES-dev/frenchmedmcqa-sft](https://huggingface.co/datasets/PARTAGES-dev/frenchmedmcqa-sft) | FR | QCM médical (SFT) | Apache 2.0 | 1 500 |
| MedQuAD | [lavita/MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) | EN | QA médical (SFT) | CC BY 4.0 | 3 500 |
| UltraMedical | [TsinghuaC3I/UltraMedical](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical) | EN | QA médical (SFT) | MIT | 500 |
| UltraMedical-Preference | [TsinghuaC3I/UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | EN | Préférences chosen/rejected (DPO) | MIT | 2 000 |

**Toutes les sources sont publiques et sous licence ouverte.** Elles ne contiennent pas de données patients réelles.

### Dataset final (après déduplication MinHash + anonymisation Presidio)

| Split | SFT | DPO |
|---|---|---|
| train | 4 324 | 1 600 |
| val | 539 | 200 |
| test | 545 | 200 |
| eval_clinique | 100 | — |

Voir [`audit/rgpd_report.md`](audit/rgpd_report.md) pour le détail du processus d'anonymisation et la conformité RGPD.

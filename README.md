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

| Source | HuggingFace | Langue | Type | Licence | Paires retenues | Statut |
|---|---|---|---|---|---|---|
| ~~MediQA~~ | [lavita/medical-qa-datasets](https://huggingface.co/datasets/lavita/medical-qa-datasets) | EN | QA médical (SFT) | Apache 2.0 | — | ⚠️ Désactivé |
| FrenchMedMCQA | [PARTAGES-dev/frenchmedmcqa-sft](https://huggingface.co/datasets/PARTAGES-dev/frenchmedmcqa-sft) | FR | QCM médical (SFT) | Apache 2.0 | 1 500 | ✅ Actif |
| MedQuAD | [lavita/MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) | EN | QA médical (SFT) | CC BY 4.0 | 3 500 | ✅ Actif |
| ChatDoctor | [lavita/medical-qa-datasets](https://huggingface.co/datasets/lavita/medical-qa-datasets/viewer/chatdoctor_healthcaremagic) | EN | Consultations patient-médecin réelles (SFT) | MIT | 1 000 | ✅ Actif |
| UltraMedical | [TsinghuaC3I/UltraMedical](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical) | EN | Vignettes cliniques synthétiques (SFT) | MIT | 500 | ✅ Actif |
| UltraMedical-Preference | [TsinghuaC3I/UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | EN | Préférences chosen/rejected (DPO) | MIT | 2 000 | ✅ Actif |

**Toutes les sources sont publiques et sous licence ouverte.**

> **Pourquoi MediQA est désactivé ?** Remplacé par ChatDoctor (HealthcareMagic), plus adapté à l'agent de triage : vraies questions de patients avec vraies réponses de médecins. Le code de collecte MediQA reste disponible — réactiver via `enabled: true` dans `configs/sources.yaml`.

### Dataset final (après déduplication MinHash + anonymisation Presidio)

| Split | SFT | DPO |
|---|---|---|
| train | 5 033 | 1 600 |
| val | 628 | 200 |
| test | 631 | 200 |
| eval_clinique | 100 | — |

Voir [`audit/rgpd_report.md`](audit/rgpd_report.md) pour le détail du processus d'anonymisation et la conformité RGPD.

---

## Étape 2 – Fine-tuning SFT (LoRA)

Fine-tuning supervisé de **Qwen3-1.7B** avec LoRA, exécuté sur Google Colab (GPU requis).

**Notebook :** [`notebooks/04_sft_qwen3_14b_alpaca.ipynb`](notebooks/04_sft_qwen3_14b_alpaca.ipynb) — ouvrir depuis GitHub dans Colab.

### Stack

- [Unsloth](https://unsloth.ai) + `trl` SFTTrainer + `peft` LoRA
- Tracking : [Weights & Biases](https://wandb.ai) (project `chsa-sft-qwen3`)

### Configuration LoRA

| Paramètre | Valeur |
|---|---|
| Modèle base | `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` |
| Rank (r) | 8 |
| LoRA alpha | 16 |
| Dropout | 0.05 |
| Modules | q/k/v/o_proj + gate/up/down_proj |
| Paramètres entraînables | ~1 % |

### Hyperparamètres

| Paramètre | Valeur |
|---|---|
| Epochs | 3 |
| Learning rate | 2e-4 (cosine) |
| Batch effectif | 8 (1 × 8 grad. accum.) |
| Warmup ratio | 0.1 |
| Optimizer | adamw_8bit |
| Seed | 42 |
| Checkpoints | tous les 200 steps |

### Thinking Mode Qwen3

Qwen3 supporte un mode raisonnement (`<think>...</think>`) activé via `/think` dans le prompt. Pour préserver cette capacité après SFT, le dataset est mixé : **75 % `/think`** + **25 % `/no_think`**.

Paramètres d'inférence recommandés : `temperature=0.6`, `top_p=0.95`, `top_k=20`.

### Modèle publié

[`XavierCoulon/qwen3-1.7b-chsa-sft-lora`](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-sft-lora) — adapters LoRA sur HuggingFace Hub.

### Workflow Colab ↔ GitHub

```
Problème ou amélioration identifié dans Colab
     ↓
Partager le traceback / output ici
     ↓
Édition locale du notebook + commit + push
     ↓
Colab : File > Open notebook from GitHub → re-run
```

---

## Schéma des métadonnées

Chaque enregistrement normalisé contient un bloc `metadata` commun :

```json
{
  "metadata": {
    "symptoms": [],
    "antecedents": [],
    "constantes": {},
    "confidence_level": "medium",
    "original_source_id": "abc123",
    "transformation_ids": ["norm_chatdoctor_000042"],
    "split": "train"
  }
}
```

| Champ | Type | Description | État |
|---|---|---|---|
| `symptoms` | `list[str]` | Symptômes extraits du texte (ex : `["fièvre", "douleur thoracique"]`) | Vide — extraction NLP hors scope étape 1 |
| `antecedents` | `list[str]` | Antécédents médicaux du patient (ex : `["diabète", "hypertension"]`) | Vide — idem |
| `constantes` | `dict` | Signes vitaux structurés (ex : `{"temperature": 38.5, "heart_rate": 97}`) | Vide — idem |
| `confidence_level` | `str` | Qualité estimée de la paire : `"high"` (DPO avec scores), `"medium"` (SFT public) | Hardcodé par source |
| `original_source_id` | `str` | Identifiant dans le dataset HuggingFace d'origine | ✅ Renseigné |
| `transformation_ids` | `list[str]` | Traçabilité RGPD : liste des opérations appliquées (`norm_*`, `anon_*`) | ✅ Renseigné |
| `split` | `str \| null` | Split d'appartenance (`"train"`, `"val"`, `"test"`) — renseigné par `04_split.py` | ✅ Renseigné après split |

### Pourquoi `symptoms`, `antecedents`, `constantes` sont vides

Ces champs ont été définis pour une extraction NLP ultérieure (NER médical, parsing de vignettes cliniques). Ils seraient pertinents pour :
- **Stratifier les splits** par type de pathologie ou niveau de complexité
- **Filtrer** les cas selon les constantes vitales dans un contexte de triage réel
- **Évaluer** la couverture clinique du corpus

Leur extraction automatique (ex. via un modèle NER médical comme `en_core_med7` ou `fr_core_news_md`) est une amélioration possible à l'étape 2 si la qualité du fine-tuning s'avère insuffisante.

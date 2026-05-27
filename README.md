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

### Normalisation (`02_normalize.py`)

Convertit chaque fichier brut (`data/raw/`) vers un schéma unifié dans `data/interim/normalized/`.

**Schéma SFT :**

```json
{ "id": "sft_medquad_000042", "source": "medquad", "language": "en",
  "task_type": "medical_qa", "instruction": "...", "response": "...",
  "metadata": { "original_source_id": "...", "transformation_ids": ["norm_medquad_000042"], "split": null } }
```

**Schéma DPO :**

```json
{ "id": "dpo_ultramedical_000001", "source": "ultramedical_preference", "language": "en",
  "task_type": "medical_qa", "prompt": "...",
  "chosen": { "role": "assistant", "content": "..." },
  "rejected": { "role": "assistant", "content": "..." },
  "metadata": { "chosen_score": 1.0, "rejected_score": 0.0, "transformation_ids": ["norm_dpo_000001"], "split": null } }
```

**Filtres appliqués :**

| Filtre | Règle |
|---|---|
| Paires incomplètes | Supprimées si `instruction` ou `response` est vide |
| Réponses trop courtes | Supprimées si < 5 mots (ex : réponses QCM lettre seule) |
| Instructions trop longues | Supprimées si > 2 000 mots (textes cliniques hors contexte) |

Chaque enregistrement produit une entrée dans `audit/transformation_log.jsonl` pour la traçabilité RGPD.

---

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

Fine-tuning supervisé de **Qwen3-1.7B** avec LoRA, exécuté sur Kaggle ou Google Colab (GPU requis).

**Notebook :** [`notebooks/04_sft_qwen3_14b_alpaca.ipynb`](notebooks/04_sft_qwen3_14b_alpaca.ipynb) — compatible Kaggle et Colab (détection automatique).

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

### Workflow Kaggle/Colab ↔ GitHub

```
Problème ou amélioration identifié dans le notebook
     ↓
Édition locale + commit + push
     ↓
Kaggle/Colab : importer le notebook depuis GitHub → re-run
```

---

## Étape 2b – Alignement DPO

Alignement par préférences sur **1 600 paires** chosen/rejected (UltraMedical-Preference), exécuté sur Kaggle ou Colab (GPU requis).

**Notebook :** [`notebooks/05_dpo_qwen3_kaggle.ipynb`](notebooks/05_dpo_qwen3_kaggle.ipynb)

### Configuration DPO

| Paramètre | Valeur |
|---|---|
| Beta | 0.1 |
| Epochs | 1 |
| Learning rate | 5e-5 (cosine) |
| Batch effectif | 4 (1 × 4 grad. accum.) |
| Max length | 1 024 tokens |

### Modèles publiés

| Modèle | Description |
|---|---|
| [`XavierCoulon/qwen3-1.7b-chsa-dpo`](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-dpo) | Adapters LoRA DPO |
| [`XavierCoulon/qwen3-1.7b-chsa-dpo-merged`](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-dpo-merged) | Modèle fusionné 16bit pour vLLM |

---

## Étape 3 – Déploiement vLLM + FastAPI + CI/CD

### Stack

- **vLLM** : serveur d'inférence OpenAI-compatible (batching dynamique, PagedAttention)
- **HuggingFace Inference Endpoints** : hébergement GPU managé (T4, scale-to-zero) — [endpoint déployé](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-dpo-merged)
- **FastAPI** : proxy API avec authentification par clé et audit RGPD
- **Docker / docker-compose** : conteneurisation et reproductibilité (développement local)
- **GitHub Actions** : CI (lint + tests + build) / CD (push image GHCR sur tag)

### Prérequis

1. Le modèle merged `XavierCoulon/qwen3-1.7b-chsa-dpo-merged` est déjà publié sur HF Hub (généré en fin de notebook DPO, cellule `cell-16`).
2. Configurer `.env` (copier depuis `.env.example`) :
   ```bash
   cp .env.example .env
   # renseigner HF_TOKEN, API_KEY (openssl rand -hex 32), VLLM_URL, MODEL_NAME
   ```

### Lancement production (HF Inference Endpoints)

```bash
make api-prod
```

L'API FastAPI se connecte à l'endpoint vLLM hébergé sur HuggingFace. Le `VLLM_URL` dans `.env` pointe vers l'endpoint HF.

Tester :

```bash
curl -X POST http://localhost:8080/v1/triage \
     -H "X-API-Key: $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"patient_description": "Homme 52 ans, douleur thoracique irradiant bras gauche, sueurs froides.", "think": false}'
```

### Lancement local (Docker)

```bash
docker compose up
```

### Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/triage` | Analyse triage médical (auth requise) |
| `GET /health` | Status + modèle chargé |
| `GET /metrics` | Latence P50/P95/P99, compteurs requêtes/erreurs |

### Benchmark de latence

```bash
make api-bench-prod
```

Résultats sauvegardés dans `audit/benchmark_results.json`.

### Métriques de performance (GPU T4 — HF Inference Endpoints)

| Métrique | Valeur |
|---|---|
| Latence P50 | ~9 800 ms |
| Latence P95 | — |
| Tokens/s | — |
| Taux d'erreur | 0 % |

> Latence mesurée avec `think=false` sur T4. Élevée pour un POC ; acceptable en contexte hospitalier non-urgent. Le mode `think=true` (chaîne de raisonnement) augmente la latence mais améliore la qualité de réponse.

### Configuration vLLM (HF Inference Endpoints)

Arguments de démarrage requis pour Qwen3 sur T4 :

```
--max-model-len 2048 --dtype float16 --enforce-eager
```

### CI/CD

- **CI** (chaque push sur `main`) : lint ruff + pytest + docker build → [GitHub Actions](https://github.com/XavierCoulon/OC_P14_Finetunez_votre_propre_LLM/actions)
- **CD** (tag `v*`) : push image sur `ghcr.io/xaviercoulon/chsa-triage-api`

### Limites d'usage

- POC **non validé cliniquement** — usage expérimental uniquement, sous supervision médicale
- Contexte maximum : **2 048 tokens** (tronqué silencieusement au-delà)
- Langues supportées : **FR** (prioritaire) et **EN**
- Pas d'accès aux constantes vitales en temps réel
- Les réponses tendent vers des conseils médicaux généraux plutôt qu'une classification P1/P2/P3 stricte (limite du corpus d'entraînement)

### Checklist go/no-go

Voir [`docs/go_no_go_checklist.md`](docs/go_no_go_checklist.md) — à compléter avant la soutenance.

### Roadmap déploiement production

| Étape | Action |
|---|---|
| Validation clinique | Revue par médecins urgentistes sur 500 cas réels |
| Corpus triage | Enrichir avec des paires P1/P2/P3 annotées pour améliorer la classification |
| Infrastructure | GPU dédié (A10G) pour latence < 2 s |
| Sécurité | Audit de sécurité, rotation des clés, rate limiting |
| Conformité | Validation RGPD complète, DPA avec établissement |

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

| Champ | Type | Description | État actuel | Amélioration possible (hors POC) |
|---|---|---|---|---|
| `symptoms` | `list[str]` | Symptômes extraits du texte (ex : `["fièvre", "douleur thoracique"]`) | Toujours `[]` | NER médical : `en_core_med7` (EN) ou `DrBenchmark/DrNER` (FR) |
| `antecedents` | `list[str]` | Antécédents médicaux du patient (ex : `["diabète", "hypertension"]`) | Toujours `[]` | Idem — même pipeline NER |
| `constantes` | `dict` | Signes vitaux structurés (ex : `{"temperature": 38.5, "heart_rate": 97}`) | Toujours `{}` | Parser regex ciblé (`T° 38.5`, `FC 97 bpm`, `TA 120/80`, `SpO2 98%`) |
| `confidence_level` | `str` | Qualité estimée de la paire | Dérivé de la source via `SOURCE_CONFIDENCE` | Score dynamique : longueur réponse + entités NER détectées + `chosen_score` DPO |
| `original_source_id` | `str` | Identifiant dans le dataset HuggingFace d'origine | Lu depuis `original_id` dans le raw, ou index de ligne | — |
| `transformation_ids` | `list[str]` | Traçabilité RGPD : liste des opérations appliquées (`norm_*`, `anon_*`) | Généré : `norm_{source}_{i:06d}` | — |
| `split` | `str \| null` | Split d'appartenance (`"train"`, `"val"`, `"test"`) | `null` à la normalisation — renseigné par `04_split.py` | Stratification par `symptoms` / `confidence_level` une fois les champs remplis |

**Niveaux de confiance par source (`SOURCE_CONFIDENCE` dans `src/data_pipeline/normalizer.py`) :**

| Source | `confidence_level` | Justification |
|---|---|---|
| ChatDoctor | `"high"` | Vraies consultations patient/médecin |
| MedQuAD | `"high"` | QA officiel NIH |
| FrenchMedMCQA | `"medium"` | QCM avec explication |
| UltraMedical SFT | `"low"` | Vignettes synthétiques |
| UltraMedical-Preference (DPO) | `"high"` | Paires scorées avec `chosen_score` / `rejected_score` |

### Pourquoi `symptoms`, `antecedents`, `constantes` sont vides

Ces champs ont été définis pour une extraction NLP ultérieure (NER médical, parsing de vignettes cliniques). Ils seraient pertinents pour :
- **Stratifier les splits** par type de pathologie ou niveau de complexité
- **Filtrer** les cas selon les constantes vitales dans un contexte de triage réel
- **Évaluer** la couverture clinique du corpus

Leur extraction automatique (ex. via un modèle NER médical comme `en_core_med7` ou `fr_core_news_md`) est une amélioration possible à l'étape 2 si la qualité du fine-tuning s'avère insuffisante.

# POC Agent IA Triage Médical – CHSA

Fine-tuning supervisé (SFT + LoRA) et alignement par préférences (DPO) du modèle Qwen3-1.7B pour l'assistance au triage médical des urgences du Centre Hospitalier Saint-Aurélien.

## Pipeline

```
Qwen3-1.7B-Base → SFT (LoRA) → DPO → Endpoint vLLM → FastAPI
```

## Structure

```
data/           Données (raw/interim hors git, processed versionné)
src/            Code source (pipeline data + API FastAPI)
scripts/        Scripts d'orchestration (01_collect → 08_export_wandb)
configs/        Paramètres (sources, presidio, split)
audit/          Logs RGPD, W&B exports, benchmark
notebooks/      SFT (04) et DPO (05) — exécutés sur Kaggle
```

## Installation

```bash
uv sync
uv sync --extra dev
uv run python -m spacy download fr_core_news_md
uv run python -m spacy download en_core_web_lg
```

---

## Étape 1 – Collecte des données

```bash
uv run python scripts/01_collect.py
uv run python scripts/02_normalize.py
uv run python scripts/03_anonymize.py
uv run python scripts/04_split.py
uv run python scripts/05_validate.py
# ou en une fois :
make pipeline
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
| Réponses trop courtes | Supprimées si < 5 mots |
| Instructions trop longues | Supprimées si > 2 000 mots |

Chaque enregistrement produit une entrée dans `audit/transformation_log.jsonl` pour la traçabilité RGPD.

---

## Sources de données

| Source | HuggingFace | Langue | Type | Licence | Paires retenues | Statut |
|---|---|---|---|---|---|---|
| ~~MediQA~~ | [lavita/medical-qa-datasets](https://huggingface.co/datasets/lavita/medical-qa-datasets) | EN | QA médical (SFT) | Apache 2.0 | — | ⚠️ Désactivé |
| FrenchMedMCQA | [PARTAGES-dev/frenchmedmcqa-sft](https://huggingface.co/datasets/PARTAGES-dev/frenchmedmcqa-sft) | FR | QCM médical (SFT) | Apache 2.0 | 1 184 | ✅ Actif |
| MedQuAD | [lavita/MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) | EN | QA médical (SFT) | CC BY 4.0 | 2 664 | ✅ Actif |
| ChatDoctor | [lavita/medical-qa-datasets](https://huggingface.co/datasets/lavita/medical-qa-datasets/viewer/chatdoctor_healthcaremagic) | EN | Consultations patient-médecin (SFT) | MIT | 794 | ✅ Actif |
| UltraMedical | [TsinghuaC3I/UltraMedical](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical) | EN | Vignettes cliniques synthétiques (SFT) | MIT | 391 | ✅ Actif |
| UltraMedical-Preference | [TsinghuaC3I/UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | EN | Préférences chosen/rejected (DPO) | MIT | 1 600 | ✅ Actif |

**Toutes les sources sont publiques et sous licence ouverte.**

> **Pourquoi MediQA est désactivé ?** Les outputs sont des extraits tronqués des inputs (fiches Mayo Clinic), pas de vraies paires Q/R. Réactiver via `enabled: true` dans `configs/sources.yaml`.

### Dataset final

| Split | SFT | DPO |
|---|---|---|
| train | 5 033 | 1 600 |
| val | 628 | 200 |
| test | 631 | 200 |
| eval_clinique | 100 | — |

Distribution linguistique : **76 % anglais** (MedQuAD + ChatDoctor + UltraMedical), **24 % français** (FrenchMedMCQA).
Le dataset DPO est **99 % anglais** (UltraMedical-Preference uniquement).

Voir [`audit/rgpd_report.md`](audit/rgpd_report.md) pour le détail de l'anonymisation et la conformité RGPD.

---

## Étape 2 – Fine-tuning SFT (LoRA)

Fine-tuning supervisé de **Qwen3-1.7B** avec LoRA — exécuté sur Kaggle ou Google Colab (GPU T4 requis).

**Notebook :** [`notebooks/04_sft_qwen3_14b_alpaca.ipynb`](notebooks/04_sft_qwen3_14b_alpaca.ipynb)

### Stack

- [Unsloth](https://unsloth.ai) + `trl` SFTTrainer + `peft` LoRA
- Tracking : [Weights & Biases](https://wandb.ai) — projet [`chsa-sft-qwen3`](https://wandb.ai/xcoulon/chsa-sft-qwen3)

### Configuration LoRA

| Paramètre | Valeur |
|---|---|
| Modèle base | `unsloth/Qwen3-1.7B-bnb-4bit` |
| Rank (r) | 8 |
| LoRA alpha | 16 |
| Dropout | 0.05 |
| Modules | q/k/v/o_proj + gate/up/down_proj |
| Paramètres entraînables | ~1 % |

### Hyperparamètres SFT

| Paramètre | Valeur | Justification |
|---|---|---|
| Epochs | 2 | Plateau eval loss à ~epoch 1.9 — epoch 3 = +0.003 loss pour +50 min GPU |
| Learning rate | 2e-4 (cosine) | |
| Batch effectif | 8 (1 × 8 grad. accum.) | |
| Warmup ratio | 0.1 | |
| Optimizer | adamw_8bit | |
| Seed | 42 | |

### Résultats SFT (`run-20260601-1239`)

| Métrique | Valeur |
|---|---|
| Eval loss finale | 1.311 (step 1200, epoch 1.91) |
| Train loss finale | 1.202 |
| Durée | ~100 min (T4) |
| Grad norm | stable 0.6–0.9 |

### Thinking Mode Qwen3

Qwen3 supporte un mode raisonnement (`<think>...</think>`) activé via `/think` dans le prompt. Pour préserver cette capacité après SFT, le dataset est mixé : **75 % `/think`** + **25 % `/no_think`**.

Paramètres d'inférence thinking : `temperature=0.6`, `top_p=0.95`, `top_k=20`. Ne jamais utiliser le greedy decoding.

> **Note déploiement :** Le champ `thinking` retourne `null` sur l'endpoint actuel. Cette limitation n'a pas été investiguée exhaustivement — elle peut résulter d'une contrainte de configuration vLLM sur HF Endpoints ou d'une dégradation du thinking mode lors du fine-tuning. Le toggle `think: true/false` reste fonctionnel (impact sur la latence).

### Compatibilité `transformers==4.56.2`

`transformers 4.56.2` introduit `list_repo_templates()` qui lève une `RemoteEntryNotFoundError` sur les repos unsloth sans dossier `additional_chat_templates`. Un patch est appliqué en cellule 3 du notebook avant le chargement du modèle.

### Modèle publié

[`XavierCoulon/qwen3-1.7b-chsa-sft-lora`](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-sft-lora) — adapters LoRA
[`XavierCoulon/qwen3-1.7b-chsa-sft-lora-merged`](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-sft-lora-merged) — modèle fusionné 16-bit (base DPO)

---

## Étape 2b – Alignement DPO

Alignement par préférences sur **1 600 paires** chosen/rejected (UltraMedical-Preference).

**Notebook :** [`notebooks/05_dpo_qwen3_kaggle.ipynb`](notebooks/05_dpo_qwen3_kaggle.ipynb)

### Hyperparamètres DPO

| Paramètre | Valeur | Justification |
|---|---|---|
| Beta | 0.02 | Conservateur — peu de contrainte KL, exploration libre des préférences |
| Epochs | 2 | Convergence margin à epoch ~1.75 |
| Learning rate | 1e-5 (cosine) | Grad norms élevées (1.2–3.1), clipping actif |
| Batch effectif | 8 (1 × 8 grad. accum.) | |
| Max length | 1 024 tokens | |

### Résultats DPO (`run-20260601-1636`)

| Métrique | Valeur |
|---|---|
| Eval loss finale | 0.598 |
| Reward margin finale (eval) | 0.395 (0 → 0.395) |
| Reward accuracy (eval) | 75.5 % |
| Durée | ~94 min (T4) |

La reward margin croît régulièrement et converge — le modèle a appris à distinguer chosen/rejected. L'accuracy DPO plafonne à 75.5 % : cohérent avec la capacité d'un 1.7B sur des paires médicales complexes.

### Modèles publiés

| Modèle | Description |
|---|---|
| [`XavierCoulon/qwen3-1.7b-chsa-dpo`](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-dpo) | Adapters LoRA DPO |
| [`XavierCoulon/qwen3-1.7b-chsa-dpo-merged`](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-dpo-merged) | Modèle fusionné 16-bit — servi en production |

---

## Évaluation — eval_clinique (RAGAS)

Évaluation sur **100 cas cliniques isolés** (aucun overlap entraînement) après SFT et DPO. Framework **RAGAS** avec Mistral comme juge LLM et `sentence-transformers/all-MiniLM-L6-v2` pour les embeddings.

Prérequis : secret `MISTRAL_API_KEY` dans Kaggle Secrets.

### Métriques

| Métrique | Description |
|---|---|
| **FactualCorrectness** | Exactitude factuelle vs référence (LLM-as-judge Mistral) |
| **ResponseRelevancy** | La réponse adresse-t-elle la question ? (Mistral + embeddings) |
| **SemanticSimilarity** | Proximité sémantique avec la référence (embeddings MiniLM) |

### Résultats

| Modèle | FactualCorrectness | ResponseRelevancy | SemanticSimilarity |
|---|---|---|---|
| **SFT** (`run-20260601-1239`) | 0.358 | 0.569 | 0.731 |
| **DPO** (`run-20260601-1636`) | 0.336 | **0.680** | 0.662 |
| Δ SFT → DPO | -0.022 | **+0.111** | -0.069 |

**Interprétation :**
- Le DPO améliore significativement la pertinence des réponses (+11 pts `answer_relevancy`) — objectif atteint.
- La baisse de similarité sémantique est attendue : le modèle aligné produit des réponses différentes de la référence mais plus pertinentes.
- Les niveaux absolus (factual_correctness ~0.35) reflètent la capacité d'un 1.7B quantisé 4-bit sur du médical — pas un problème de pipeline.

Métriques et tables (100 exemples + breakdown par source) loggés dans W&B :
- SFT : [`chsa-sft-qwen3`](https://wandb.ai/xcoulon/chsa-sft-qwen3) → `eval_clinique/ragas_*`
- DPO : [`chsa-dpo-qwen3`](https://wandb.ai/xcoulon/chsa-dpo-qwen3) → `eval_clinique/ragas_*`

```bash
make export-wandb   # met à jour audit/wandb_sft_last_run.json et wandb_dpo_last_run.json
```

---

## Étape 3 – Déploiement vLLM + FastAPI + CI/CD

### Architecture

```
User → POST /v1/triage
         ↓
   FastAPI (local, make api-prod)
         ↓  HTTP + Bearer token
   HF Inference Endpoint (vLLM 0.18.x, GPU T4)
         ↓
   Qwen3-1.7B-chsa-dpo-merged
```

**vLLM** est un moteur de serving LLM haute performance (PagedAttention + continuous batching) exposant une API compatible OpenAI. Le FastAPI agit comme proxy avec authentification par clé et audit RGPD.

### Stack

- **vLLM** sur [HuggingFace Inference Endpoints](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-dpo-merged) (T4, scale-to-zero)
- **FastAPI** : proxy API, détection de langue, audit
- **Docker / docker-compose** : conteneurisation (prod Linux GPU + local Mac CPU)
- **GitHub Actions** : CI lint + tests + build / CD push image GHCR sur tag

### Prérequis

Configurer `.env` (copier depuis `.env.example`) :

```bash
cp .env.example .env
# renseigner : HF_TOKEN, API_KEY, VLLM_URL, MODEL_NAME
```

### Lancement

**Mode production (HF Inference Endpoint) :**

```bash
make api-prod    # FastAPI local → HF Endpoint vLLM
```

**Mode Docker local Linux (GPU NVIDIA requis) :**

```bash
make api-up      # docker compose (vLLM + FastAPI)
make api-down    # arrêt
```

**Mode Docker local Mac Apple Silicon (CPU, ~20-60s/requête) :**

```bash
make api-up-mac  # docker-compose.mac.yml override (ARM64, --device cpu)
```

> L'image vLLM GPU pèse ~24 GB. L'image ARM64 CPU est ~8-12 GB. Pour un usage local sur Mac, `make api-prod` + HF Endpoint reste recommandé.

### Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/triage` | Analyse triage médical (auth requise) |
| `GET /health` | Status + modèle chargé |
| `GET /metrics` | Latence P50/P95/P99, compteurs requêtes/erreurs |

### Exemple de requête

```bash
curl -X POST http://localhost:8080/v1/triage \
     -H "X-API-Key: $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "patient_description": "58-year-old male, crushing chest pain radiating to left arm for 45 minutes, cold sweats, nausea.",
       "think": false
     }'
```

Réponse :

```json
{
  "request_id": "...",
  "triage_response": "**P1 – Absolute Emergency**\n\nACS presentation...",
  "thinking": null,
  "latency_ms": 4518.2
}
```

### Comportement de l'API

**Détection de langue automatique** : l'API détecte si la description est en français ou en anglais et adapte le system prompt. La réponse est dans la même langue que l'input.

**Priorité en tête garantie** : un post-traitement `_ensure_priority_first()` déplace le niveau `P1/P2/P3` en première ligne si le modèle ne le place pas correctement.

**Toggle thinking** : `think: true` active le mode raisonnement Qwen3 (`/think` tag). Le champ `thinking` retourne actuellement `null` — limitation non investiguée exhaustivement (configuration vLLM ou modèle). Le toggle impacte la latence (+1-2s).

### Configuration vLLM (HF Inference Endpoint)

Arguments Container :

```
--max-model-len 2048 --dtype float16 --enforce-eager
```

### Benchmark de latence

```bash
make api-bench-prod
```

### Métriques de performance (GPU T4 — HF Inference Endpoints)

| Métrique | Valeur | Seuil prod |
|---|---|---|
| Latence P50 | 5 699 ms | < 3 s ⚠️ |
| Latence P95 | 7 159 ms | < 5 s ⚠️ |
| Latence min | 4 057 ms | — |
| 5 requêtes concurrentes (wall) | 8 455 ms | — ✅ |
| Taux d'erreur | 0 % | < 1 % ✅ |

> Latence mesurée avec `think=false` sur T4 HF Endpoint (free tier). Le continuous batching vLLM est actif : 5 requêtes simultanées en 8.5s vs ~28s séquentiel. Production : GPU A10G requis pour P50 < 2s.

### CI/CD

- **CI** (chaque push `main`) : lint ruff + pytest + docker build → [GitHub Actions](https://github.com/XavierCoulon/OC_P14_Finetunez_votre_propre_LLM/actions)
- **CD** (tag `v*`) : push image sur `ghcr.io/xaviercoulon/chsa-triage-api`

### Limites d'usage

- POC **non validé cliniquement** — usage expérimental uniquement, sous supervision médicale
- Contexte maximum : **2 048 tokens**
- Langues : **FR** et **EN** (détection automatique)
- Qualité de réponse dépendante de la richesse de l'input : les descriptions courtes (`"j'ai mal aux dents"`) induisent des hallucinations ; les descriptions multi-symptômes avec contexte donnent de meilleurs résultats
- Hallucinations résiduelles sur le contenu médical : cohérent avec `factual_correctness=0.336` — inhérent à un modèle 1.7B
- `thinking: null` : le mode raisonnement ne produit pas de chaîne de pensée visible sur le déploiement actuel

### Checklist go/no-go

Voir [`docs/go_no_go_checklist.md`](docs/go_no_go_checklist.md).

### Roadmap déploiement production

| Étape | Action |
|---|---|
| Validation clinique | Revue par médecins urgentistes sur 500 cas réels |
| Corpus triage | Enrichir avec des paires P1/P2/P3 annotées (classification stricte) |
| Infrastructure | GPU dédié (A10G) pour latence < 2 s |
| Thinking mode | Investiguer `--reasoning-parser qwen3` sur vLLM ≥ 0.18.x |
| Sécurité | Audit sécurité, rotation des clés, rate limiting |
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

| Champ | État actuel | Amélioration possible |
|---|---|---|
| `symptoms` | Toujours `[]` | NER médical : `en_core_med7` (EN) ou `DrBenchmark/DrNER` (FR) |
| `antecedents` | Toujours `[]` | Idem |
| `constantes` | Toujours `{}` | Parser regex ciblé (T°, FC, TA, SpO2) |
| `confidence_level` | Dérivé de la source | Score dynamique basé sur longueur + NER + `chosen_score` DPO |

**Niveaux de confiance par source :**

| Source | `confidence_level` |
|---|---|
| ChatDoctor | `"high"` |
| MedQuAD | `"high"` |
| FrenchMedMCQA | `"medium"` |
| UltraMedical SFT | `"low"` |
| UltraMedical-Preference (DPO) | `"high"` |

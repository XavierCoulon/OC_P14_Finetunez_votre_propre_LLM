# POC Agent IA Triage Médical – CHSA

Agent de triage médical P1/P2/P3 fondé sur **Qwen3-1.7B** fine-tuné (SFT + LoRA + DPO) et déployé via vLLM.

```
Qwen3-1.7B-Base → SFT (LoRA) → DPO → Endpoint vLLM → FastAPI
```

## Installation

```bash
uv sync && uv sync --extra dev
uv run python -m spacy download fr_core_news_md
uv run python -m spacy download en_core_web_lg
```

## Structure

```
data/        Données (raw/interim hors git)
src/         Code source (pipeline + API FastAPI)
scripts/     01_collect → 08_export_wandb
notebooks/   04_sft (Kaggle) · 05_dpo (Kaggle)
docs/        Rapport technique · Décisions · Checklist
audit/       Logs RGPD · W&B exports · Benchmark
```

## Pipeline données

```bash
make pipeline        # collect → normalize → anonymize → split → validate → publish
make export-wandb    # exporte les métriques W&B vers audit/
```

## API

```bash
make api-prod        # FastAPI local → HF Inference Endpoint (vLLM)
make api-bench-prod  # benchmark latence → audit/benchmark_results.json
```

```bash
curl -X POST http://localhost:8080/v1/triage \
     -H "X-API-Key: $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"patient_description": "Homme 58 ans, douleur thoracique irradiant bras gauche, sueurs froides depuis 45 min.", "think": false}'
```

Réponse : `triage_response` commence toujours par `**P1/P2/P3 – ...**`. Langue détectée automatiquement (FR/EN).

## Résultats clés

| Étape | Métrique | Valeur |
|---|---|---|
| SFT | Eval loss | 1.311 (plateau epoch 1.91) |
| DPO | Reward margin | 0.395 · Accuracy 75.5 % |
| RAGAS SFT | FactualCorrectness / ResponseRelevancy | 0.358 / 0.569 |
| RAGAS DPO | FactualCorrectness / ResponseRelevancy | 0.336 / **0.680** (+11 pts) |
| API | Latence P50 / P95 | 5.7s / 7.2s (T4 HF free tier) |

## Modèles publiés

| Modèle | Usage |
|---|---|
| [qwen3-1.7b-chsa-sft-lora](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-sft-lora) | Adapters LoRA SFT |
| [qwen3-1.7b-chsa-dpo-merged](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-dpo-merged) | Modèle production (vLLM) |

W&B : [chsa-sft-qwen3](https://wandb.ai/xcoulon/chsa-sft-qwen3) · [chsa-dpo-qwen3](https://wandb.ai/xcoulon/chsa-dpo-qwen3)

## Limites

- **Non validé cliniquement** — POC expérimental, supervision médicale requise
- Hallucinations sur inputs courts ; classification correcte sur descriptions riches
- Latence P50 = 5.7s (GPU dédié A10G requis pour < 2s en production)
- `thinking: null` — limitation vLLM/modèle non résolue
- Contexte max : 2 048 tokens

## Documentation

| Document | Contenu |
|---|---|
| [`docs/rapport_technique.md`](docs/rapport_technique.md) | Rapport complet ≤ 20 pages |
| [`docs/finetuning_decisions.md`](docs/finetuning_decisions.md) | Historique hyperparamètres SFT & DPO |
| [`docs/go_no_go_checklist.md`](docs/go_no_go_checklist.md) | Checklist soutenance — **GO** |
| [`docs/monitoring.md`](docs/monitoring.md) | Procédures de surveillance post-déploiement |
| [`audit/rgpd_report.md`](audit/rgpd_report.md) | Audit anonymisation RGPD |

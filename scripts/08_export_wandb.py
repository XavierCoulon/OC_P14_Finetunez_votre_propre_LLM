"""
Export du dernier run W&B pour les projets SFT et DPO.
Produit : audit/wandb_<projet>_last_run.json

L'historique est nettoyé et séparé en deux séries :
- train_history : steps avec métriques d'entraînement
- eval_history  : steps avec métriques d'évaluation

Les NaN sont convertis en null (JSON valide).
"""
import json
import math
from pathlib import Path

import wandb

PROJECTS = {
    "sft": "xcoulon/chsa-sft-qwen3",
    "dpo": "xcoulon/chsa-dpo-qwen3",
}

TRAIN_KEYS_SFT = {"train/loss", "train/learning_rate", "train/grad_norm", "train/epoch", "train/global_step"}
EVAL_KEYS_SFT  = {"eval/loss", "eval/runtime", "eval/samples_per_second", "train/epoch", "train/global_step"}

TRAIN_KEYS_DPO = {
    "train/loss", "train/learning_rate", "train/grad_norm", "train/epoch", "train/global_step",
    "train/rewards/chosen", "train/rewards/rejected", "train/rewards/margins", "train/rewards/accuracies",
}
EVAL_KEYS_DPO = {
    "eval/loss", "eval/runtime", "eval/samples_per_second", "train/epoch", "train/global_step",
    "eval/rewards/chosen", "eval/rewards/rejected", "eval/rewards/margins", "eval/rewards/accuracies",
}

TRAIN_KEYS = {"sft": TRAIN_KEYS_SFT, "dpo": TRAIN_KEYS_DPO}
EVAL_KEYS  = {"sft": EVAL_KEYS_SFT,  "dpo": EVAL_KEYS_DPO}

OUTPUT_DIR = Path("audit")
OUTPUT_DIR.mkdir(exist_ok=True)


def nan_to_null(obj):
    """Convertit récursivement les NaN/Inf en None (→ null JSON), et force les types W&B en primitives."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_null(v) for v in obj]
    if hasattr(obj, "items"):
        return {k: nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def split_history(rows: list[dict], train_keys: set, eval_keys: set):
    """Sépare les rows train (eval/loss absent) des rows eval (train/loss absent)."""
    train, evl = [], []
    for row in rows:
        if row.get("train/loss") is not None:
            train.append({k: row[k] for k in train_keys if k in row})
        elif row.get("eval/loss") is not None:
            evl.append({k: row[k] for k in eval_keys if k in row})
    return train, evl


api = wandb.Api()

for label, project in PROJECTS.items():
    runs = api.runs(project, order="-created_at")
    if not runs:
        print(f"[{label}] Aucun run trouvé dans {project}")
        continue

    run = runs[0]
    print(f"[{label}] Dernier run : {run.name} ({run.id}) — état : {run.state}")

    raw_history = nan_to_null(run.history(samples=10000).to_dict(orient="records"))
    train_history, eval_history = split_history(raw_history, TRAIN_KEYS[label], EVAL_KEYS[label])

    payload = {
        "project": project,
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "config": nan_to_null(dict(run.config)),
        "summary": nan_to_null(dict(run.summary)),
        "train_history": train_history,
        "eval_history": eval_history,
    }

    out_path = OUTPUT_DIR / f"wandb_{label}_last_run.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[{label}] Exporté → {out_path} ({len(train_history)} train steps, {len(eval_history)} eval steps)")

"""
Export du dernier run W&B pour les projets SFT et DPO.
Produit : audit/wandb_<projet>_last_run.json
"""
import json
from pathlib import Path

import wandb

PROJECTS = {
    "sft": "xcoulon/chsa-sft-qwen3",
    "dpo": "xcoulon/chsa-dpo-qwen3",
}

OUTPUT_DIR = Path("audit")
OUTPUT_DIR.mkdir(exist_ok=True)

api = wandb.Api()

for label, project in PROJECTS.items():
    runs = api.runs(project, order="-created_at")
    if not runs:
        print(f"[{label}] Aucun run trouvé dans {project}")
        continue

    run = runs[0]
    print(f"[{label}] Dernier run : {run.name} ({run.id}) — état : {run.state}")

    history = run.history(samples=10000).to_dict(orient="records")

    payload = {
        "project": project,
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "config": dict(run.config),
        "summary": dict(run.summary),
        "history": history,
    }

    out_path = OUTPUT_DIR / f"wandb_{label}_last_run.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    print(f"[{label}] Exporté → {out_path} ({len(history)} steps)")

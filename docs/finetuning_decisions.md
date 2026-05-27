# Décisions d'hyperparamètres — SFT & DPO

Historique des décisions prises sur les hyperparamètres d'entraînement, justifiées par l'analyse des runs W&B.

---

## SFT — Qwen3-1.7B LoRA

**Notebook :** `notebooks/04_sft_qwen3_14b_alpaca.ipynb`  
**Run de référence :** [`run-20260523-0701`](https://wandb.ai/xcoulon/chsa-sft-qwen3) (id: `f6gh2gd9`)  
**Dataset :** `XavierCoulon/oc-p14-dataset` (split `sft`, ~4 324 paires au moment du run)

### Paramètres actuels

| Paramètre | Valeur |
|---|---|
| `num_train_epochs` | **2** |
| `learning_rate` | 2e-4 |
| `lr_scheduler_type` | cosine |
| `warmup_ratio` | 0.1 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 (batch effectif = 8) |
| `optim` | adamw_8bit |
| `weight_decay` | 0.01 |
| `max_seq_length` | 2048 |
| LoRA `r` | 8 |
| LoRA `lora_alpha` | 16 |
| LoRA `lora_dropout` | 0.05 |

### Décisions et justifications

#### `num_train_epochs` : 3 → **2**
**Observation W&B :** L'eval loss atteint un plateau à partir de l'epoch ~1.85 (step 1000/1623). Les 4 dernières évaluations sont quasi identiques :

| Step | Epoch | Eval loss |
|---|---|---|
| 1000 | 1.85 | 1.1267 |
| 1200 | 2.22 | 1.1275 |
| 1400 | 2.59 | 1.1274 |
| 1600 | 2.96 | **1.1257** |

L'epoch 3 complète (steps 1200–1623) n'apporte aucune amélioration mesurable et représente ~43 minutes de calcul inutiles sur GPU T4 Kaggle.

**Décision :** passer à 2 epochs. Si le dataset est agrandi (push-hub en attente), re-vérifier la courbe avant de conclure.

### Métriques finales du run de référence

| Métrique | Valeur |
|---|---|
| Train loss (moyenne run) | 1.166 |
| Train loss (dernière valeur) | 1.03 |
| Eval loss finale | 1.126 |
| Grad norm (finale) | 0.97 |
| Durée | 127.8 min |

---

## DPO — Qwen3-1.7B LoRA (depuis SFT merged)

**Notebook :** `notebooks/05_dpo_qwen3_kaggle.ipynb`  
**Run de référence :** [`run-20260524-1138`](https://wandb.ai/xcoulon/chsa-dpo-qwen3) (id: `uxos1lue`)  
**Modèle de départ :** `XavierCoulon/qwen3-1.7b-chsa-sft-lora-merged`  
**Dataset :** `XavierCoulon/oc-p14-dataset` (split `dpo`, 1 600 paires chosen/rejected)

### Paramètres actuels

| Paramètre | Valeur |
|---|---|
| `beta` | **0.05** |
| `num_train_epochs` | 1 |
| `learning_rate` | **1e-5** |
| `lr_scheduler_type` | cosine |
| `warmup_ratio` | 0.1 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 (batch effectif = 8) |
| `optim` | adamw_8bit |
| `max_length` | 1024 |
| `max_prompt_length` | 512 |
| `eval_steps` / `save_steps` | **50** |
| LoRA `r` | 8 |
| LoRA `lora_alpha` | 16 |

### Décisions et justifications

#### `learning_rate` : 2e-5 → **1e-5**

**Observation W&B :** Les grad norms sont entre **5 et 13** sur tout le run (pic à 13.6 au step 60), alors que `max_grad_norm=1`. Cela signifie que le clipping est actif à chaque step, signe d'un LR trop élevé. En comparaison, le SFT avait des grad norms entre 0.43 et 1.10.

```
Step  50 : grad_norm = 10.1
Step  60 : grad_norm = 13.6  ← maximum
Step 130 : grad_norm = 11.3  ← instabilité (loss remonte à 0.618)
```

**Décision :** diviser le LR par 2 pour réduire l'amplitude des mises à jour et limiter les instabilités.

#### `beta` : 0.1 → **0.05**

**Observation W&B :** En fin de run, les rewards/chosen restent négatifs sur l'eval (−0.137), tandis que les rewards/rejected sont très négatifs (−0.901). Le modèle apprend surtout à pénaliser les mauvaises réponses, pas à valoriser les bonnes.

| Métrique | Eval step 100 | Eval step 200 |
|---|---|---|
| rewards/chosen | −0.219 | **−0.137** |
| rewards/rejected | −0.898 | **−0.901** |
| rewards/margin | 0.679 | 0.764 |
| rewards/accuracy | 74.0% | **75.0%** |

Un `beta` plus faible réduit la contrainte KL par rapport au modèle de référence SFT, donnant plus de liberté au modèle pour s'aligner sur les préférences — et donc potentiellement remonter les rewards/chosen.

**Décision :** passer beta à 0.05. Si les rewards/chosen deviennent positifs, le signal d'alignement est sain.

#### `eval_steps` / `save_steps` : 100 → **50**

**Observation W&B :** Avec 200 steps au total (1 epoch), on n'a obtenu que **2 points d'évaluation** (step 100 et step 200). Impossible de voir la dynamique d'apprentissage ni de détecter un éventuel overfitting précoce.

**Décision :** passer à 50 steps pour avoir 4 points d'évaluation, ce qui suffit à voir la courbe sur 1 epoch.

### Métriques finales du run de référence

| Métrique | Valeur |
|---|---|
| Train loss (moyenne run) | 0.556 |
| Train loss (finale) | 0.451 |
| Eval loss finale | 0.536 |
| Eval rewards/accuracy | 75.0% |
| Eval rewards/margin | 0.764 |
| Durée | 45.3 min |

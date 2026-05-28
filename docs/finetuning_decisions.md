# Décisions d'hyperparamètres — SFT & DPO

Historique des décisions prises sur les hyperparamètres d'entraînement, justifiées par l'analyse des runs W&B.

---

## SFT — Qwen3-1.7B LoRA

**Notebook :** `notebooks/04_sft_qwen3_14b_alpaca.ipynb`  
**Runs de référence :**
- [`run-20260523-0701`](https://wandb.ai/xcoulon/chsa-sft-qwen3) (id: `f6gh2gd9`) — 4 324 paires, 3 epochs
- [`run-20260527-1118`](https://wandb.ai/xcoulon/chsa-sft-qwen3) (id: `kmrozvf9`) — 5 033 paires, 2 epochs

**Dataset :** `XavierCoulon/oc-p14-dataset` (split `sft`, 5 033 paires actuellement)

### Paramètres actuels

| Paramètre | Valeur |
|---|---|
| `num_train_epochs` | **3** |
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

### Historique des décisions — `num_train_epochs`

#### Itération 1 : 3 → **2** *(run-20260523-0701, 4 324 paires)*
**Observation :** Plateau à partir de l'epoch ~1.85 (step 1000/1623). Les 4 dernières évaluations sont quasi identiques :

| Step | Epoch | Eval loss |
|---|---|---|
| 1000 | 1.85 | 1.1267 |
| 1200 | 2.22 | 1.1275 |
| 1400 | 2.59 | 1.1274 |
| 1600 | 2.96 | **1.1257** |

**Décision :** passer à 2 epochs — ~43 min économisées. Note : « re-vérifier si le dataset grossit ».

#### Itération 2 : 2 → 3 *(run-20260527-1118, 5 033 paires)* — temporaire
**Observation :** Eval loss à 1.310 à epoch 1.91 (fin du run à 2 epochs), semblait encore en descente.  
**Décision :** passage à 3 epochs — mais infirmée par l'itération suivante.

#### Itération 3 : 3 → **2** *(run-20260528-1124, 5 033 paires)*
**Observation :** Run complet à 3 epochs — le minimum d'eval loss est à step 1200 / epoch 1.91 (1.2957), identique au run précédent. L'epoch 3 n'améliore que de 0.011 (1.2957 → 1.2989 final) pour ~50 min de GPU supplémentaires.

| Step | Epoch | Eval loss |
|---|---|---|
| 200  | 0.32 | 1.4473 |
| 600  | 0.95 | 1.3469 |
| 1000 | 1.59 | 1.3120 |
| **1200** | **1.91** | **1.2957** ← minimum |
| 1400 | 2.22 | 1.3033 ← légère remontée |
| 1800 | 2.86 | 1.2989 |

**Conclusion définitive :** le plateau à ~epoch 1.9 est une constante de ce dataset, indépendante de sa taille (4 324 ou 5 033 paires). 2 epochs est le meilleur compromis qualité/GPU.

**Nouvelle métrique :** ROUGE-L = **0.394** sur eval_clinique (100 cas) — baseline de référence pour comparaison post-DPO.

### Métriques des runs de référence SFT

| Run | Dataset | Epochs | Train loss moy. | Eval loss finale | Durée |
|---|---|---|---|---|---|
| run-20260523-0701 | 4 324 | 3 | 1.166 | 1.126 | 127.8 min |
| run-20260527-1118 | 5 033 | 2 | 1.430 | 1.310 | 96.8 min |

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

# Rapport Technique — POC Agent IA de Triage Médical (CHSA)

**Auteur :** Xavier Coulon  
**Date :** Juin 2026  
**Projet :** OC P14 — Fine-tuner votre propre LLM  
**Établissement fictif :** Centre Hospitalier Saint-Aurélien (CHSA)

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Données — Collecte, structuration et conformité RGPD](#2-données)
3. [Fine-tuning supervisé — SFT + LoRA](#3-sft)
4. [Alignement par préférences — DPO](#4-dpo)
5. [Évaluation clinique — RAGAS](#5-évaluation)
6. [Déploiement — Architecture et performance](#6-déploiement)
7. [Analyse des résultats](#7-analyse)
8. [Roadmap production](#8-roadmap)
9. [Conclusion](#9-conclusion)

---

## 1. Introduction

### Contexte

Le Centre Hospitalier Saint-Aurélien (CHSA) reçoit plusieurs centaines de passages aux urgences chaque semaine. Le tri initial des patients — distinguer une urgence vitale d'une urgence différée — est une tâche chronophage qui mobilise des soignants qualifiés dès l'accueil. Une erreur de classification peut retarder une prise en charge critique ou engorger inutilement le circuit d'urgence.

### Objectif du POC

Ce projet démontre la faisabilité technique d'un **agent IA de triage médical** fondé sur un LLM open-source de petite taille (Qwen3-1.7B), entraîné sur des données médicales publiques puis aligné par apprentissage par préférences (DPO). L'agent classifie chaque description patient en trois niveaux de priorité :

| Niveau | Signification | Délai |
|---|---|---|
| **P1 – Urgence absolue** | Pronostic vital engagé | < 5 min |
| **P2 – Urgence relative** | Situation grave mais stable | < 20 min |
| **P3 – Urgence différée** | Non critique | < 2h |

### Pipeline technique

```
Qwen3-1.7B-Base
      ↓ SFT (LoRA) — 5 033 paires médicales bilingues
Qwen3-1.7B-SFT-LoRA
      ↓ DPO — 1 600 paires chosen/rejected
Qwen3-1.7B-DPO-Merged
      ↓ vLLM (HF Inference Endpoint, T4)
API FastAPI — POST /v1/triage
```

---

## 2. Données

### 2.1 Sources

Cinq sources publiques ont été retenues après évaluation :

| Source | Type | Langue | Paires SFT | Paires DPO | Licence |
|---|---|---|---|---|---|
| MedQuAD (NIH) | Q&A médical | EN | 2 664 | — | CC BY 4.0 |
| ChatDoctor (HealthcareMagic) | Consultations patient/médecin | EN | 794 | — | MIT |
| FrenchMedMCQA | QCM médical | FR | 1 184 | — | Apache 2.0 |
| UltraMedical (SFT) | Vignettes cliniques synthétiques | EN | 391 | — | MIT |
| UltraMedical-Preference | Paires chosen/rejected scorées | EN | — | 1 600 | MIT |

> **MediQA exclu** : les outputs sont des extraits tronqués des inputs (fiches Mayo Clinic), pas de vraies paires Q/R.

**Distribution linguistique :** 76 % anglais / 24 % français (SFT) ; 99 % anglais (DPO).

### 2.2 Pipeline de collecte et normalisation

Chaque source est collectée, normalisée vers un schéma unifié, anonymisée, puis splitée. Les opérations sont enregistrées dans `audit/transformation_log.jsonl` (71 MB, append-only) pour la traçabilité RGPD.

**Filtres appliqués lors de la normalisation :**
- Suppression des paires avec `instruction` ou `response` vide
- Suppression des réponses < 5 mots (réponses QCM lettre seule)
- Suppression des instructions > 2 000 mots (textes cliniques hors contexte)

### 2.3 Anonymisation RGPD (Presidio)

| Source | Stratégie | Résultat |
|---|---|---|
| ChatDoctor | Presidio ACTIF | 3 052 entités masquées (PERSON, EMAIL, PHONE...) |
| FrenchMedMCQA | skip (48 % faux positifs, aucune donnée patient réelle) | — |
| MedQuAD | skip (46 % faux positifs, Q&A officiel NIH) | — |
| UltraMedical SFT | skip (98 % faux positifs, données synthétiques) | — |
| UltraMedical DPO | skip (84 % faux positifs, données synthétiques) | — |

Un post-traitement regex a corrigé un cas de PII manqué (prénom "Amber", score < 0.70). Vérification finale : zéro tag PII dans les splits traités.

Voir [`audit/rgpd_report.md`](rgpd_report.md) pour le détail complet.

### 2.4 Splits finaux

| Split | SFT | DPO |
|---|---|---|
| train | 5 033 | 1 600 |
| val | 628 | 200 |
| test | 631 | 200 |
| **eval_clinique** | **100** | — |

L'`eval_clinique` est un split isolé sans aucun overlap avec les données d'entraînement — il sert exclusivement à l'évaluation RAGAS post-entraînement.

---

## 3. Fine-tuning supervisé (SFT + LoRA)

### 3.1 Architecture

Le modèle de base est **Qwen3-1.7B** (quantisé 4-bit NF4 via unsloth). LoRA permet d'entraîner ~1 % des paramètres seulement, rendant le fine-tuning réalisable sur GPU T4 (16 GB).

| Composant | Valeur |
|---|---|
| Modèle base | `unsloth/Qwen3-1.7B-bnb-4bit` |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| Dropout | 0.05 |
| Modules ciblés | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Paramètres entraînables | ~1 % |

### 3.2 Hyperparamètres

| Paramètre | Valeur | Justification |
|---|---|---|
| Epochs | **2** | Plateau d'eval loss systématique à ~epoch 1.9 (observé sur 4 324 et 5 033 paires) |
| Learning rate | 2e-4 (cosine) | Standard unsloth/LoRA |
| Batch effectif | 8 (1 × 8 grad. accum.) | Contrainte VRAM T4 |
| Warmup ratio | 0.1 | |
| Optimizer | adamw_8bit | Efficacité mémoire |
| Seed | 42 | Reproductibilité |

L'itération des epochs (3 → 2 → 3 → 2) est documentée dans [`docs/finetuning_decisions.md`](finetuning_decisions.md). La décision finale de 2 epochs repose sur une observation constante : l'epoch 3 n'apporte que −0.003 de gain sur l'eval loss pour +50 min de GPU.

### 3.3 Thinking Mode Qwen3

Qwen3 supporte un mode raisonnement (`<think>...</think>`) activé par le tag `/think` dans le prompt. Pour préserver cette capacité après SFT, le dataset est mixé :
- **75 % des exemples** avec `/think`
- **25 % des exemples** avec `/no_think`

Paramètres d'inférence requis en mode thinking : `temperature=0.6`, `top_p=0.95`, `top_k=20`.

### 3.4 Résultats (`run-20260601-1239`)

**Progression de l'eval loss :**

| Step | Epoch | Eval loss |
|---|---|---|
| 200 | 0.32 | 1.434 |
| 400 | 0.64 | 1.375 |
| 600 | 0.95 | 1.342 |
| 800 | 1.27 | 1.327 |
| 1 000 | 1.59 | 1.314 |
| **1 200** | **1.91** | **1.311** ← plateau |

**Métriques finales :**

| Métrique | Valeur |
|---|---|
| Train loss finale | 1.202 |
| Eval loss finale | 1.311 |
| Grad norm (stable) | 0.6 – 0.9 |
| Durée | 100 min (T4) |
| Étapes | 1 260 |

**Modèles publiés :**
- [`XavierCoulon/qwen3-1.7b-chsa-sft-lora`](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-sft-lora) — adapters LoRA
- [`XavierCoulon/qwen3-1.7b-chsa-sft-lora-merged`](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-sft-lora-merged) — modèle fusionné 16-bit (base pour DPO)

---

## 4. Alignement par préférences (DPO)

### 4.1 Principe

Le DPO (Direct Preference Optimization) aligne le modèle sur des préférences humaines sans modèle de récompense séparé. Chaque paire d'entraînement contient une réponse `chosen` (préférée) et une réponse `rejected` (moins bonne). L'objectif est d'augmenter la probabilité des réponses choisies relative au modèle de référence (SFT).

La `reward margin` (chosen − rejected) est le signal clé : une margin croissante indique un alignement sain.

### 4.2 Hyperparamètres

| Paramètre | Valeur | Justification |
|---|---|---|
| Beta | **0.02** | Faible contrainte KL — liberté d'exploration des préférences. Itéré depuis 0.1 → 0.05 → 0.02 (voir `docs/finetuning_decisions.md`) |
| Learning rate | 1e-5 (cosine) | Grad norms DPO élevées (1.2–3.1), clipping actif → LR conservateur |
| Epochs | 2 | Eval loss en descente à epoch 1.0 lors des runs précédents |
| Batch effectif | 8 (1 × 8 grad. accum.) | |
| Max length | 1 024 tokens | |

### 4.3 Résultats (`run-20260601-1636`)

**Progression DPO loss et reward margin :**

| Step | Epoch | DPO loss | Margin (eval) | Accuracy (eval) |
|---|---|---|---|---|
| 0 | 0 | **0.693** (= log(2), baseline) | 0.027 | 75.0 % |
| 100 | 0.5 | 0.658 | 0.098 | 75.0 % |
| 200 | 1.0 | 0.616 | 0.263 | 75.5 % |
| 300 | 1.5 | 0.600 | 0.381 | 75.5 % |
| 350 | 1.75 | 0.598 | 0.394 | 75.5 % |
| **400** | **2.0** | **0.598** | **0.395** ← convergence | **75.5 %** |

La loss démarre à 0.693 (valeur théorique log(2) quand le modèle n'a aucune préférence) — signe d'une initialisation correcte depuis le modèle SFT. La convergence est confirmée : margin quasi-identique entre step 350 et 400.

**Métriques finales :**

| Métrique | Valeur |
|---|---|
| DPO eval loss | 0.598 |
| Reward margin (eval) | 0.395 |
| Reward accuracy (eval) | 75.5 % |
| Durée | 94 min (T4) |

**Modèles publiés :**
- [`XavierCoulon/qwen3-1.7b-chsa-dpo`](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-dpo) — adapters LoRA DPO
- [`XavierCoulon/qwen3-1.7b-chsa-dpo-merged`](https://huggingface.co/XavierCoulon/qwen3-1.7b-chsa-dpo-merged) — modèle fusionné 16-bit (production)

---

## 5. Évaluation clinique (RAGAS)

### 5.1 Protocole

Après chaque étape d'entraînement, le modèle est évalué sur **100 cas cliniques isolés** (split `eval_clinique`) sans aucun overlap avec les données d'entraînement SFT ou DPO.

**Framework :** [RAGAS](https://docs.ragas.io) avec :
- **Mistral-small-latest** comme juge LLM (FactualCorrectness, ResponseRelevancy)
- **sentence-transformers/all-MiniLM-L6-v2** pour les embeddings (SemanticSimilarity)

**Configuration :** `RunConfig(max_workers=8, timeout=120, max_retries=5)` pour respecter les rate limits Mistral.

### 5.2 Métriques

| Métrique | Description |
|---|---|
| **FactualCorrectness** | Exactitude factuelle de la réponse vs référence (LLM-as-judge) |
| **ResponseRelevancy** | La réponse adresse-t-elle la question posée ? (LLM + embeddings) |
| **SemanticSimilarity** | Proximité sémantique avec la réponse de référence (embeddings uniquement) |

Toutes les métriques ∈ [0, 1].

### 5.3 Résultats comparatifs

| Métrique | SFT | DPO | Δ | Interprétation |
|---|---|---|---|---|
| FactualCorrectness | 0.358 | 0.336 | −0.022 | Dans le bruit de mesure |
| **ResponseRelevancy** | 0.569 | **0.680** | **+0.111** | ✅ Objectif DPO atteint |
| SemanticSimilarity | 0.731 | 0.662 | −0.069 | Attendu (voir §7) |

Les métriques et tableaux complets (100 exemples + breakdown par source) sont loggés dans W&B :
- SFT : [xcoulon/chsa-sft-qwen3](https://wandb.ai/xcoulon/chsa-sft-qwen3) — run `9hazkdo4`
- DPO : [xcoulon/chsa-dpo-qwen3](https://wandb.ai/xcoulon/chsa-dpo-qwen3) — run `uii3sq0c`

---

## 6. Déploiement

### 6.1 Architecture

```
Utilisateur
    ↓ POST /v1/triage  {patient_description, think}
FastAPI (local — make api-prod)
    ↓ HTTP + Bearer token
HF Inference Endpoint (vLLM 0.18.x, GPU T4)
    ↓
Qwen3-1.7B-chsa-dpo-merged
```

**vLLM** expose une API OpenAI-compatible. Ses deux technologies clés :
- **PagedAttention** : gestion paginée du KV-cache, quasi-zéro gaspillage VRAM
- **Continuous Batching** : les nouvelles requêtes sont traitées pendant la génération en cours

**FastAPI** agit comme proxy avec :
- Authentification par clé (`X-API-Key`)
- Détection automatique de langue (FR/EN) avec system prompt adapté
- Post-traitement garantissant le niveau de priorité en tête de réponse
- Audit log RGPD (hash patient SHA-256, timestamp, latence)

### 6.2 Configuration vLLM (HF Endpoint)

```
--max-model-len 2048 --dtype float16 --enforce-eager
```

### 6.3 Benchmark de performance

Benchmark effectué avec 10 requêtes séquentielles + 5 concurrentes (`think=false`) depuis `scripts/benchmark_latency.py`.

**Latence nominale (10 requêtes séquentielles) :**

| P50 | P95 | Min | Max |
|---|---|---|---|
| 5 699 ms | 7 159 ms | 4 057 ms | 7 159 ms |

**Charge concurrente (5 requêtes simultanées) :**

| Wall time | Temps séquentiel équivalent | Gain vLLM |
|---|---|---|
| 8 455 ms | ~28 500 ms | **×3.4** |

Le continuous batching réduit le wall time de 3.4× par rapport à un traitement séquentiel — validation directe de l'efficacité de vLLM.

**Cas limites :**

| Cas | Résultat |
|---|---|
| Prompt vide | 422 Unprocessable Entity ✅ |
| Texte 2 000 mots | 4 664 ms, pas de crash ✅ |
| Injection HTML/XSS | 4 448 ms, pas de crash ✅ |

### 6.4 CI/CD

- **CI** (chaque push sur `main`) : lint ruff + pytest + docker build — [GitHub Actions](https://github.com/XavierCoulon/OC_P14_Finetunez_votre_propre_LLM/actions)
- **CD** (tag `v*`) : push image sur `ghcr.io/xaviercoulon/chsa-triage-api`

### 6.5 Checklist Go / No-Go

| Critère | Verdict |
|---|---|
| RAGAS FactualCorrectness ≥ 0.20 | ✅ 0.336 |
| RAGAS ResponseRelevancy | ✅ 0.680 |
| Taux d'erreur API < 1 % | ✅ 0 % |
| 5 requêtes concurrentes sans crash | ✅ |
| Auth API key fonctionnelle | ✅ |
| Audit log RGPD | ✅ |
| CI/CD opérationnel | ✅ |
| Latence P50 < 3s | ⚠️ 5.7s (HF free tier T4) |
| Revue médicale 10 cas | ⚠️ Partielle — P1/P2/P3 correct sur inputs riches |

**Verdict : GO pour la soutenance POC — NO-GO pour la production.**

---

## 7. Analyse des résultats

### Le DPO améliore la pertinence, pas la fidélité

La baisse de `SemanticSimilarity` (0.731 → 0.662) n'est **pas un échec** : le modèle DPO produit des réponses différentes de la référence, mais plus pertinentes par rapport à la question (+11 pts `ResponseRelevancy`). C'est le comportement attendu d'un modèle aligné sur des préférences — il optimise la pertinence plutôt que la reproduction de la référence.

### Qualité variable selon la richesse de l'input

L'analyse manuelle de 10 cas révèle un comportement bi-modal :

| Type d'input | Résultat |
|---|---|
| Description riche (âge, symptômes multiples, antécédents, durée) | Bonne classification P1/P2/P3, peu d'hallucinations |
| Description courte ("j'ai mal aux dents") | Sur-triage (P2 au lieu de P3), hallucinations pour "remplir" la réponse |

Ce comportement est une conséquence directe du corpus d'entraînement : les sources (MedQuAD, ChatDoctor, UltraMedical) sont des Q&A médicaux complexes, sans cas simples bien représentés.

### Limite structurelle : absence de ground truth triage

**Aucune des 5 sources ne contient de label P1/P2/P3.** Le modèle a appris à produire du vocabulaire médical pertinent, mais la classification de priorité repose entièrement sur l'instruction dans le system prompt — pas sur des exemples annotés de triage. C'est la limite principale du corpus pour cet usage.

Conséquence directe : la cohérence de classification n'est pas garantie. Un même tableau clinique peut recevoir P1 ou P2 selon la formulation de l'input, sans que le modèle ait appris de règle de décision stable.

### Évaluation triage-spécifique absente

RAGAS mesure la qualité générale de la réponse mais pas ce qui compte cliniquement. Deux métriques critiques sont manquantes dans ce POC :

| Métrique | Définition | Risque clinique |
|---|---|---|
| **Taux de sous-triage** | P1 classé P2 ou P3 | Élevé — retard de prise en charge vitale |
| **Taux de sur-triage** | P3 classé P1 ou P2 | Modéré — engorgement des urgences |

Pour un déploiement réel, le taux de sous-triage doit être 0 % — c'est un critère bloquant non mesuré dans ce POC.

### Capacités multilingues

Le modèle répond dans la langue de l'input grâce à la détection de langue côté API (system prompt bilingue). Cette capacité est fonctionnelle mais limitée aux inputs clairement FR ou EN.

### Thinking mode non fonctionnel en production

Le champ `thinking` retourne systématiquement `null`. Deux hypothèses non discriminées :
1. Le fine-tuning a dégradé la capacité de génération de blocs `<think>...</think>`
2. La configuration vLLM sur HF Endpoint (manque de `--reasoning-parser qwen3`) empêche le parsing des balises

Le toggle `think: true/false` reste fonctionnel pour la latence (+1-2s en mode thinking).

### Continuous batching vLLM validé

Le gain ×3.4 sur 5 requêtes simultanées (8.5s vs 28.5s séquentiel) confirme que vLLM est un choix justifié pour un contexte hospitalier avec flux continu de patients.

---

## 8. Roadmap production

### Court terme (0–3 mois)

| Action | Impact |
|---|---|
| Constituer un dataset triage annoté (500–1 000 cas P1/P2/P3 validés par urgentistes) | Correction de la limite structurelle principale — le modèle apprendra des règles de décision stables |
| Construire des paires DPO ciblées triage (chosen = bonne priorité, rejected = mauvaise priorité) | DPO actuel est généraliste — des paires sur des erreurs de classification ont un impact direct |
| Mesurer les taux de sous-triage et sur-triage sur eval_clinique | Métriques cliniquement pertinentes absentes du POC |
| Implémenter un format de sortie structuré (JSON : `priority`, `reasoning`, `actions`) | Classification exploitable programmatiquement, réduction des hallucinations de remplissage |
| Investiguer le thinking mode (`--reasoning-parser qwen3` sur vLLM ≥ 0.18.x) | Activer la chaîne de raisonnement visible |
| Validation clinique formelle sur 500 cas réels | Prérequis déploiement |

### Moyen terme (3–12 mois)

| Action | Impact |
|---|---|
| GPU dédié A10G (24 GB) | P50 < 2s, capacité > 50 req/min |
| Passage à un modèle 7B (Qwen3-7B) | FactualCorrectness > 0.50 estimé |
| RAG sur guidelines médicales (SFMU, HAS) | Ancrage factuel validé, réduction des hallucinations |
| Questionnaire adaptatif multi-tours | L'agent pose des questions de clarification sur les inputs courts → meilleure classification |
| Mécanisme de feedback médecin | Amélioration continue des paires DPO depuis les corrections en production |

### Long terme (> 12 mois)

| Action | Impact |
|---|---|
| DPA avec le CHSA, certification RGPD complète | Conformité légale déploiement |
| Audit de sécurité (OWASP, pentest API) | Conformité sécurité |
| Intégration SIH (dossier patient) | Contexte patient pour meilleure classification |
| Validation CE marquage dispositif médical | Usage clinique réglementaire |

---

## 9. Conclusion

Ce POC démontre la **faisabilité technique complète** de la chaîne SFT → DPO → déploiement vLLM pour un agent de triage médical fondé sur un LLM open-source de petite taille.

**Ce qui fonctionne :**
- Pipeline de collecte, normalisation et anonymisation RGPD reproductible
- Fine-tuning SFT convergent en 2 epochs (~100 min T4)
- Alignement DPO mesurable : +11 pts de pertinence des réponses
- Endpoint vLLM opérationnel, CI/CD automatisé, API robuste (0 % d'erreurs)
- Support bilingue FR/EN fonctionnel

**Ce qui nécessite un investissement pour la production :**
- Latence P50 de 5.7s (vs seuil < 3s) — nécessite un GPU dédié
- Qualité factuelle modeste (0.336) — inhérente au 1.7B, à corriger avec un modèle plus grand ou du RAG
- Validation clinique formelle — prerequis réglementaire non négociable

Ce travail constitue une base technique solide et documentée pour engager la phase de validation clinique et de montée en charge au sein du CHSA.

---

*Rapport généré depuis les données des runs W&B `run-20260601-1239` (SFT) et `run-20260601-1636` (DPO). Métriques archivées dans `audit/wandb_sft_last_run.json` et `audit/wandb_dpo_last_run.json`.*

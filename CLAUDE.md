# Projet P14 – POC Agent IA Triage Médical (CHSA)

## Contexte
POC d'un agent IA de triage médical pour le Centre Hospitalier Saint-Aurélien (CHSA).
Modèle : Qwen3-1.7B-Base → SFT (LoRA) → DPO → déploiement vLLM.

## Repo GitHub
https://github.com/XavierCoulon/OC_P14_Finetunez_votre_propre_LLM

## Issues de référence
- #1 : Consignes complètes du projet
- #2 : Bases théoriques SFT & DPO
- #3 : Cours Post-training 101 (Han Fang)
- #4 : Cours HuggingFace LLM Course Chapter 11 (SFT, LoRA, évaluation)
- #5 : Étape 0 – Comprendre SFT et DPO
- #6 : Étape 1 – Plan collecte et structuration des données

## Stack technique
- Python 3.11, uv (jamais pip directement)
- Fine-tuning : trl (SFTTrainer semaine 2, DPOTrainer semaine 3) + peft (LoRA)
- Anonymisation : Presidio (presidio-analyzer + presidio-anonymizer)
- Modèles spaCy : fr_core_news_md (FR), en_core_web_lg (EN)
- Évaluation : lighteval (MMLU médical)
- Déploiement : vLLM + GitHub Actions CI/CD

## Feuille de route
- Semaine 1 : Collecte et structuration des données (étape 1 en cours)
- Semaine 2 : SFT + LoRA
- Semaine 3 : DPO
- Semaine 4 : Déploiement vLLM + évaluation + rapport

## Règles importantes
- data/raw/ et data/interim/ sont hors git (.gitignore)
- audit/transformation_log.jsonl : append only, ne jamais modifier une entrée
- eval_clinique séparé strict : aucun overlap avec les splits SFT/DPO
- Toujours uv add pour ajouter des dépendances, uv sync pour installer

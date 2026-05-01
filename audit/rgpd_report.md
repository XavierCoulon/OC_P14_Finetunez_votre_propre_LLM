# Rapport RGPD – Dataset Médical CHSA

**Date** : 2026-05-01
**Projet** : POC Agent IA Triage Médical – Centre Hospitalier Saint-Aurélien

---

## Sources utilisées

| Source | Licence | Langue | Type |
|---|---|---|---|
| lavita/medical-qa-datasets (mediqa) | Apache 2.0 | EN | QA médical |
| PARTAGES-dev/frenchmedmcqa-sft | Apache 2.0 | FR | QCM médical |
| lavita/MedQuAD | CC BY 4.0 | EN | QA médical |
| TsinghuaC3I/UltraMedical | MIT | EN | QA médical |
| TsinghuaC3I/UltraMedical-Preference | MIT | EN | Préférences DPO |

Toutes les sources sont publiques, sous licence ouverte, et ne contiennent pas de données patients réelles.

---

## Processus d'anonymisation

**Outil** : Microsoft Presidio v2.2+
**Stratégie** : `replace` — les entités sensibles sont remplacées par `<ENTITY_TYPE>`
**Modèles NLP** : `fr_core_news_md` (FR), `en_core_web_lg` (EN)
**Seuil de détection** : 0.70

**Entités ciblées** :
- PERSON, LOCATION, DATE_TIME, PHONE_NUMBER, EMAIL_ADDRESS, NRP, MEDICAL_LICENSE

**Résultat** : 25 436 entités masquées sur 6 948 enregistrements.

---

## Traçabilité

Chaque transformation est enregistrée dans `audit/transformation_log.jsonl` :
- `transformation_id` unique par opération
- `timestamp` ISO 8601
- `operation` : normalization ou anonymization
- `entities_modified` : nombre d'entités masquées
- Checksums SHA256 jamais stockés en clair (hash seulement)
- **Le contenu textuel n'est jamais loggué**

---

## Conformité

- Les données sources sont publiques et ne contiennent pas de données personnelles de santé réelles
- L'anonymisation Presidio constitue une mesure de précaution supplémentaire
- Les données `data/raw/` et `data/interim/` sont exclues du dépôt git (`.gitignore`)
- Seules les données transformées et anonymisées sont versionnées dans `data/processed/`

---

## Volumes finaux

| Split | SFT | DPO |
|---|---|---|
| train | 3 876 | 1 600 |
| val | 484 | 200 |
| test | 488 | 200 |
| eval_clinique | 100 | — |

# Rapport RGPD — Anonymisation des données d'entraînement

**Projet** : POC Agent IA Triage Médical – CHSA  
**Modèle** : Qwen3-1.7B-Base → SFT (LoRA) → DPO  
**Date** : 2026-05-27  
**Outil** : Microsoft Presidio (presidio-analyzer + presidio-anonymizer)

---

## 1. Démarche

Conformément aux bonnes pratiques RGPD pour les données de santé, une étape d'anonymisation Presidio a été intégrée au pipeline de collecte.

**Installation :**
```bash
uv add presidio-analyzer presidio-anonymizer
uv run python -m spacy download fr_core_news_md   # modèle NLP français
uv run python -m spacy download en_core_web_lg    # modèle NLP anglais
```

**Configuration (`src/data_pipeline/anonymizer.py`) :**
- `AnalyzerEngine` : détection des entités sensibles — PERSON, LOCATION, DATE_TIME, PHONE_NUMBER, EMAIL_ADDRESS, NRP
- `AnonymizerEngine` : masquage par substitution → `<PERSON>`, `<DATE_TIME>`, etc. (stratégie `replace`)
- Modèle FR : `fr_core_news_md`, modèle EN : `en_core_web_lg`
- Seuil global : `score_threshold = 0.70`, seuil NRP strict : `0.85`
- Filtre longueur minimale : entités < 3 chars ignorées (lettres isolées type A/B/C en QCM)
- Allowlist médicale : ~50 termes (éponyms de syndromes, abréviations d'organisations)
- **Post-processing regex** (`_apply_name_regex`) : capture les prénoms sous-scorés par Presidio via les patterns `my name is X`, `Dear X,`, `Hi X,`, `Thanks X` — avec liste d'exclusion pour les faux positifs courants (`Doctor`, `Chat`, `Friend`, etc.)

**Entités PII détectées et masquées :**

| Entité | Description | Statut |
|---|---|---|
| `PERSON` | Noms et prénoms | ✅ Actif |
| `LOCATION` | Lieux géographiques | ✅ Actif |
| `DATE_TIME` | Dates, heures, âges | ✅ Actif |
| `PHONE_NUMBER` | Numéros de téléphone | ✅ Actif |
| `EMAIL_ADDRESS` | Adresses email | ✅ Actif |
| `NRP` | Nationalité, religion, groupe politique | ✅ Actif (seuil 0.85) |
| `MEDICAL_LICENSE` | Numéro RPPS / licence médicale | ✗ Non activé — à activer en production hospitalière |
| `IP_ADDRESS` / `IBAN_CODE` / `CREDIT_CARD` | Données financières et réseau | ✗ Non activé — hors contexte corpus |

**Traçabilité :** chaque enregistrement traité génère une entrée dans `audit/transformation_log.jsonl` (opération, timestamp, fichier source/sortie, nombre d'entités — jamais le contenu textuel).

---

## 2. Audit par source

Presidio a été exécuté sur la totalité des sources normalisées. Les résultats ont été analysés source par source pour distinguer les vrais PII des faux positifs.

### 2.1 FrenchMedMCQA — FR, 1 500 entrées

**Nature des données :** Questions à choix multiples issues d'examens médicaux français (ECN/iECN). Contenu 100 % éducatif et générique, aucun contexte patient.

**PII réels identifiés :** Aucun.

**Résultats Presidio (avant fix) :** 722/1 500 enregistrements avec au moins un tag → 48 % de faux positifs.

**Type de faux positifs observés :**

| Texte brut | Tag Presidio | Explication |
|---|---|---|
| `A. Elévation de la lipase` | `<PERSON> de la lipase` | Mot français capitalisé après tabulation QCM |
| `A. Leucémie aiguë myéloblastique` | `<PERSON> aiguë myéloblastique` | Idem |
| `A. Vigabatrine` | `<PERSON>` | Nom de médicament (majuscule en début d'item) |
| `A. Liquide céphalo-rachidien` | `<PERSON> céphalo-rachidien` | Terme anatomique capitalisé |
| `la teneur corpusculaire (TCMH)` | `la teneur corpusculaire (<LOCATION>)` | Acronyme médical → LOCATION |

**Décision :** `skip_anonymization: true`  
**Justification :** Source sans donnée patient. Les faux positifs dégradent structurellement le contenu des QCM (réponses tronquées, termes médicaux perdus).

---

### 2.2 MedQuAD — EN, 3 500 entrées

**Nature des données :** Q&A éducatif public extrait de NIH, GARD (Genetic and Rare Diseases), MedlinePlus. Fiches encyclopédiques sur les maladies génétiques et rares.

**PII réels identifiés :** Aucun. Pas de patient, pas d'information personnelle. Contenu institutionnel public sous licence CC BY 4.0.

**Résultats Presidio (avant fix) :** 1 613/3 500 enregistrements avec au moins un tag → 46 % de faux positifs.

**Type de faux positifs observés :**

| Texte brut | Tag Presidio | Explication |
|---|---|---|
| `Type I (Naxos disease) was first described in families from the Greek island of Naxos` | `Type I (<NRP> disease)... <NRP> island of <LOCATION>` | Naxos = île grecque dans le nom d'une maladie rare |
| `affected families have been found in other Greek islands, Turkey, and the Middle East` | `<NRP> islands, <LOCATION>, and <LOCATION>` | Géographie épidémiologique d'une maladie |
| `Mutations in the JUP, DSP, DSC2, and KANK2 genes` | `KANK2` → `<LOCATION>` | Acronyme de gène → faux positif |
| `Desmosomes are located in the membrane surrounding certain cells` | `<PERSON> are located` | Terme biologique capitalisé → faux positif |
| `Type II (Carvajal syndrome)` | `(<PERSON> syndrome)` | Éponymie médicale (médecin → nom de syndrome) |
| `Knobloch syndrome is a rare condition` | `<PERSON> syndrome` | Idem |

**Décision :** `skip_anonymization: true`  
**Justification :** Contenu NIH public, aucun PII. Les tags détruisent les noms de maladies rares (Naxos disease → `<NRP> disease`), les noms de gènes (KANK2 → `<LOCATION>`) et les termes biologiques (Desmosomes → `<PERSON>`).

---

### 2.3 UltraMedical SFT — EN, 500 entrées

**Nature des données :** Vignettes cliniques synthétiques issues de MedQA (banque de questions USMLE Step 1/2). Patients **entièrement fictifs**, créés à usage éducatif médical.

**PII réels identifiés :** Aucun. Âges, genres, symptômes et signes vitaux sont des paramètres cliniques synthétiques ne permettant aucune réidentification.

**Résultats Presidio (avant fix) :** 489/500 enregistrements avec au moins un tag → **98 % de faux positifs**.

**Type de faux positifs observés :**

| Texte brut | Tag Presidio | Impact clinique |
|---|---|---|
| `A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination` | `A <DATE_TIME> pregnant woman at <DATE_TIME> gestation` | **Perte totale du contexte clinique** : âge et terme de grossesse indispensables au diagnostic et au choix de l'antibiotique |
| `A 68-year-old man with a past medical history of diabetes, hypertension, obesity` | `A <DATE_TIME> man` | Âge essentiel au diagnostic différentiel (ischémie aiguë) |
| `A mother brings her 3-week-old infant` | `A mother brings her <DATE_TIME> infant` | Âge pédiatrique critique (botulisme néonatal vs autres) |
| `She has been experiencing symptoms for 4 days` | Parfois `<DATE_TIME>` | Durée des symptômes = contexte temporal du diagnostic |

**Décision :** `skip_anonymization: true`  
**Justification :** Données synthétiques, aucun PII. L'anonymisation des âges détruit la valeur clinique des vignettes (l'âge patient est un paramètre diagnostique primaire en médecine).

---

### 2.4 ChatDoctor (HealthcareMagic) — EN, 1 000 entrées

**Nature des données :** Vraies consultations patient-médecin issues de HealthcareMagic. Contenu réel : questions de patients avec prénoms, symptômes personnels, contexte médical individuel.

**PII réels identifiés :** Oui — source à risque réel.

**Résultats Presidio :** 3 052 entités masquées sur 1 000 enregistrements (PERSON, DATE_TIME, LOCATION, PHONE_NUMBER).

**PII manqué détecté lors de l'audit post-anonymisation :**

| Enregistrement | Champ | Texte brut | Cause |
|---|---|---|---|
| l.324 | instruction + response | `my name is Amber` / `Dear Amber,` | Presidio score < 0.70 sur prénom court sans contexte fort |

**Correctif appliqué :** `_apply_name_regex()` — post-processing regex actif depuis 2026-05-27. Le record l.324 a été re-anonymisé : `Amber` → `<PERSON>` aux deux occurrences.

**Décision :** Presidio actif (pas de `skip_anonymization`).

---

### 2.5 UltraMedical DPO — EN, 2 000 paires chosen/rejected

**Nature des données :** Deux types de contenu — (a) prompts académiques de synthèse médicale ("Investigate immunometabolism..."), (b) vignettes cliniques synthétiques identiques à UltraMedical SFT.

**PII réels identifiés :** Aucun. Même raisonnement que UltraMedical SFT.

**Résultats Presidio (avant fix) :** 1 673/2 000 enregistrements avec au moins un tag → 84 % de faux positifs. La densité plus élevée s'explique par les réponses longues (chosen/rejected de 500–2 000 mots) qui multiplient les opportunités de détection.

**Décision :** `skip_anonymization: true`

---

## 3. Synthèse des décisions

| Source | Type | Langue | PII réels | Décision | Statut |
|---|---|---|---|---|---|
| ChatDoctor | Consultations patient-médecin réelles | EN | ✅ Oui | Presidio actif + regex | ✅ Anonymisé (3 052 entités + correctif Amber) |
| FrenchMedMCQA | QCM d'examen (ECN) | FR | ✗ Aucun | `skip_anonymization` | ✅ Aucun PII (48 % faux positifs évités) |
| MedQuAD | Q&A encyclopédique NIH | EN | ✗ Aucun | `skip_anonymization` | ✅ Aucun PII (46 % faux positifs évités) |
| UltraMedical SFT | Vignettes synthétiques (USMLE) | EN | ✗ Aucun | `skip_anonymization` | ✅ Aucun PII (98 % faux positifs évités) |
| UltraMedical DPO | Paires préférences (USMLE) | EN | ✗ Aucun | `skip_anonymization` | ✅ Aucun PII (84 % faux positifs évités) |

**Conclusion générale :** ChatDoctor est la seule source contenant de vraies données patients — elle est la seule soumise à Presidio. Les quatre autres sources sont publiques, éducatives, sans PII patient, et leur anonymisation produisait uniquement des faux positifs dégradant la valeur clinique des données.

Un audit post-anonymisation (2026-05-27) a détecté un PII manqué sur ChatDoctor (prénom `Amber`, score Presidio < 0.70). Un correctif regex ciblé a été appliqué et le fichier re-généré.

---

## 4. Infrastructure RGPD — Sources futures

Le pipeline Presidio reste pleinement opérationnel pour de futures sources contenant des données patients réelles (comptes-rendus hospitaliers, notes cliniques, etc.) :

- `src/data_pipeline/anonymizer.py` : moteur Presidio configuré et testé, filtres anti-faux-positifs en place
- `configs/sources.yaml` : flag `skip_anonymization` par source (défaut : `false`, i.e. Presidio actif)
- `audit/transformation_log.jsonl` : log append-only de chaque décision d'anonymisation
- Stratégies disponibles : `replace`, `mask`, `redact` (configurable par entité dans `anonymizer.py`)

Pour activer Presidio sur une nouvelle source avec données patients réelles, il suffit de ne pas poser le flag (comportement par défaut) :
```yaml
# configs/sources.yaml
nouvelle_source:
  # skip_anonymization non défini → Presidio actif par défaut
```

---

## 5. Vérification résiduelle

Après passage dans le pipeline complet, vérification qu'aucun tag PII ne subsiste dans les fichiers finaux :

```bash
# Zéro tag dans les splits finaux
grep -c "<PERSON>\|<LOCATION>\|<DATE_TIME>\|<NRP>" data/processed/sft/train.jsonl    # → 0
grep -c "<PERSON>\|<LOCATION>\|<DATE_TIME>\|<NRP>" data/processed/dpo/train.jsonl    # → 0

# Zéro tag dans les fichiers intermédiaires anonymisés
grep -c "<DATE_TIME>" data/interim/anonymized/ultramedical_sft_anonymized.jsonl       # → 0
grep -c "<PERSON>" data/interim/anonymized/medquad_anonymized.jsonl                   # → 0
```

---

## 6. Volumes finaux

| Split | SFT | DPO |
|---|---|---|
| train | 3 947 | 1 600 |
| val | 492 | 200 |
| test | 496 | 200 |
| eval_clinique | 100 | — |

*Après déduplication MinHash (Jaccard ≥ 0.9) : 465 doublons supprimés sur 5 500 paires SFT.*

---

*Rapport généré manuellement à partir des logs `audit/transformation_log.jsonl` et de l'audit des sources brutes.*

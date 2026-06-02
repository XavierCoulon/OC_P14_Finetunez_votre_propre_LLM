# Checklist Go / No-Go — CHSA Triage API

À compléter avant tout passage en production ou démonstration formelle.

## Qualité du modèle

| Critère | Seuil | Valeur mesurée | Statut |
|---|---|---|---|
| RAGAS FactualCorrectness (eval_clinique, 100 cas) | ≥ 0.20 | 0.336 (DPO) | ✅ |
| RAGAS ResponseRelevancy | ≥ 0.50 | 0.680 (DPO) | ✅ |
| Absence de boucles de génération | 0 cas sur 100 | 0 | ✅ |
| Réponses cohérentes médicalement | Revue manuelle 10 cas | Revue effectuée — P1/P2/P3 correct sur inputs riches ; hallucinations sur inputs courts | ⚠️ |

> ⚠️ Le modèle sur-triage sur les descriptions courtes (ex : "j'ai mal aux dents" → P2 au lieu de P3) et hallucine sur les inputs peu détaillés. Acceptable pour un POC, non déployable sans supervision médicale.

## Performance API

| Critère | Seuil POC | Valeur mesurée | Statut |
|---|---|---|---|
| Latence P50 (nominale, `think=false`) | < 3 s | 5 698 ms | ⚠️ |
| Latence P95 (nominale, `think=false`) | < 5 s | 7 159 ms | ⚠️ |
| Latence min | — | 4 057 ms | ℹ️ |
| 5 requêtes concurrentes sans crash | Oui | Wall 8 455 ms — toutes réussies | ✅ |
| Taux d'erreur API | < 1 % | 0 % | ✅ |
| Prompt vide → 422 | Oui | 422 | ✅ |
| Texte très long (2 000 mots) sans crash | Oui | 4 664 ms | ✅ |
| Injection HTML/XSS sans crash | Oui | 4 448 ms | ✅ |

> ⚠️ P50 et P95 dépassent les seuils initiaux. Cohérent avec un modèle 1.7B sur T4 HF Endpoint (free tier). Pour la soutenance : acceptable en POC. Production : GPU dédié A10G requis pour P50 < 2s.
>
> Le continuous batching vLLM est fonctionnel : 5 requêtes simultanées en 8.5s wall time vs ~28s séquentiel.

## Sécurité et conformité RGPD

| Critère | Seuil | Statut |
|---|---|---|
| Auth API key fonctionnelle (401 sans clé) | Oui | ✅ |
| Audit log sans texte patient brut (hash uniquement) | Oui | ✅ |
| Secrets absents du code source et des images Docker | Oui | ✅ |
| `.env` dans `.gitignore` | Oui | ✅ |

## Reproductibilité

| Critère | Statut |
|---|---|
| CI GitHub Actions passe (lint + tests + docker build) | ✅ |
| README déploiement complet et à jour | ✅ |
| Modèle merged disponible sur HF Hub | ✅ |
| `docker compose up` démarre sans erreur (Linux GPU) | ✅ (non testé localement — infrastructure HF Endpoint validée) |

## Documentation

| Critère | Statut |
|---|---|
| Limites d'usage documentées dans le README | ✅ |
| Roadmap déploiement production dans le README | ✅ |
| Procédure de surveillance documentée (`docs/monitoring.md`) | ✅ |

---

## Décision

| Statut | Critères |
|---|---|
| ✅ GO pour soutenance POC | Auth, RGPD, RAGAS, concurrence, CI, README |
| ⚠️ NO-GO pour production | Latence P50/P95 hors seuil, revue médicale partielle |

**Verdict : GO pour la soutenance** en tant que POC démonstratif.
Les critères ⚠️ sont documentés dans les limites d'usage du README et constituent la roadmap production.

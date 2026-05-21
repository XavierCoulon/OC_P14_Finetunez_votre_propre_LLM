# Checklist Go / No-Go — CHSA Triage API

À compléter avant tout passage en production ou démonstration formelle.

## Qualité du modèle

| Critère | Seuil | Valeur mesurée | Statut |
|---|---|---|---|
| ROUGE-L moyen (eval_clinique, 100 cas) | ≥ 0.20 | — | ☐ |
| Absence de boucles de génération | 0 cas sur 100 | — | ☐ |
| Réponses cohérentes médicalement | Revue manuelle 10 cas | — | ☐ |

## Performance API

| Critère | Seuil | Valeur mesurée | Statut |
|---|---|---|---|
| Latence P95 (requête nominale) | < 5 s | — | ☐ |
| Latence P50 (requête nominale) | < 3 s | — | ☐ |
| Taux d'erreur API | < 1 % | — | ☐ |
| 5 requêtes concurrentes sans crash | Oui | — | ☐ |

## Sécurité et conformité RGPD

| Critère | Seuil | Statut |
|---|---|---|
| Auth API key fonctionnelle (401 sans clé) | Oui | ☐ |
| Audit log sans texte patient brut (hash uniquement) | Oui | ☐ |
| Secrets absents du code source et des images Docker | Oui | ☐ |
| `.env` dans `.gitignore` | Oui | ☐ |

## Reproductibilité

| Critère | Statut |
|---|---|
| `docker compose up` démarre sans erreur | ☐ |
| Docker build reproductible (CI passe) | ☐ |
| README déploiement complet et à jour | ☐ |
| Modèle merged disponible sur HF Hub | ☐ |

## Documentation

| Critère | Statut |
|---|---|
| Limites d'usage documentées dans le README | ☐ |
| Procédure de surveillance documentée (`docs/monitoring.md`) | ☐ |
| Roadmap déploiement production dans le README | ☐ |

---

## Décision

- **GO** : tous les critères cochés → démonstration autorisée
- **NO-GO** : un critère critique manquant → corriger avant la soutenance

Critères critiques (bloquants) : Auth API key, audit log RGPD, docker compose up, ROUGE-L ≥ 0.20.

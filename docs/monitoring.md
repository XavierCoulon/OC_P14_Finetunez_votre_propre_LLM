# Procédures de surveillance — CHSA Triage API

## Surveillance quotidienne

- Vérifier `GET /health` → status `ok`

```bash
curl -s http://localhost:8080/health
```

## Surveillance hebdomadaire

- Vérifier le volume de `audit/api_log.jsonl` (croissance anormale = usage non prévu)
- Contrôler que le champ `patient_hash` est bien un hash SHA-256 (64 caractères hex) et non du texte brut

## Alertes

| Condition | Action |
|---|---|
| Taux d'erreurs élevé sur `/health` ou logs vLLM | Redémarrer le service API, vérifier les logs vLLM |
| Latence élevée constatée via benchmark | Vérifier la charge GPU, redémarrer vLLM si nécessaire |
| Conteneur vLLM OOM | Réduire `--max-model-len` ou passer en quantization 4bit |

## Rotation des clés API

- Fréquence : trimestrielle (ou immédiatement si suspicion de fuite)
- Procédure :
  1. Générer une nouvelle clé : `openssl rand -hex 32`
  2. Mettre à jour `API_KEY` dans `.env` (local) et les GitHub Secrets
  3. Redémarrer le conteneur API : `docker compose restart api`
  4. Vérifier que l'ancienne clé retourne bien 401

## Revue de l'audit log

- Fréquence : mensuelle
- Vérification : aucun champ ne contient de données patient en clair
- Archivage : compresser et archiver les logs > 30 jours

```bash
# Vérifier le format des entrées
tail -5 audit/api_log.jsonl | python -m json.tool

# Compter les requêtes par jour
grep -o '"ts":"[^"]*"' audit/api_log.jsonl | cut -c7-16 | sort | uniq -c
```

## Limites d'usage à communiquer aux utilisateurs

- Ce modèle est un **POC expérimental**, non validé cliniquement
- Ne pas utiliser pour des décisions médicales sans supervision humaine
- Contexte maximum : 2 048 tokens (les textes plus longs sont tronqués)
- Langues supportées : français (prioritaire) et anglais
- Pas d'accès aux constantes vitales en temps réel

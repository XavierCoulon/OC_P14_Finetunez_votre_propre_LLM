.PHONY: collect normalize anonymize split validate publish push-hub pipeline \
        api-up api-down api-logs api-test api-bench lint test

collect:
	uv run python scripts/01_collect.py

normalize:
	uv run python scripts/02_normalize.py

anonymize:
	uv run python scripts/03_anonymize.py

split:
	uv run python scripts/04_split.py

validate:
	uv run python scripts/05_validate.py

publish:
	uv run python scripts/06_publish.py

push-hub:
	uv run python scripts/07_push_to_hub.py

pipeline: collect normalize anonymize split validate publish

# --- API & Déploiement ---

api-up:
	docker compose up --build -d

api-down:
	docker compose down

api-logs:
	docker compose logs -f api

api-test:
	curl -s -X POST http://localhost:8080/v1/triage \
		-H "X-API-Key: $${API_KEY}" \
		-H "Content-Type: application/json" \
		-d '{"patient_description": "Homme 52 ans, douleur thoracique irradiant bras gauche, sueurs froides.", "think": false}' \
		| python -m json.tool

api-bench:
	uv run python scripts/benchmark_latency.py --url http://localhost:8080 --key $${API_KEY}

# --- Qualité du code ---

lint:
	uv run ruff check src/

test:
	uv run pytest tests/ -v

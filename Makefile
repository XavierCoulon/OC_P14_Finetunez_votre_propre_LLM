.PHONY: collect normalize anonymize split validate publish push-hub pipeline

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

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install fastapi "uvicorn[standard]" httpx

COPY src/api/ ./src/api/

EXPOSE 8080
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

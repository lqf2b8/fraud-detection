FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
# Shared library (feature pipeline must match training exactly)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY services/api/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY services/api/app ./app
RUN useradd --create-home --uid 10001 app
USER app

# Locally the model comes from ./models; on Cloud Run, from a Cloud Storage volume at /models.
ENV MODEL_DIR=/models PORT=8000
EXPOSE 8000
HEALTHCHECK CMD curl -fs http://localhost:${PORT}/health || exit 1
# Cloud Run sets PORT; exec so uvicorn receives shutdown signals directly.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

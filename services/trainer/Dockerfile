FROM python:3.11-slim

# libgomp is required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV PROJECT_ROOT=/app \
    RAW_DATA_DIR=/data/raw \
    MODEL_DIR=/models

ENTRYPOINT ["python", "-m", "fraud_detection.train"]

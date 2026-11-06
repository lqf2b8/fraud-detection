FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY services/dashboard/requirements.txt .
RUN pip install -r requirements.txt
COPY services/dashboard/app.py .
COPY services/dashboard/assets ./assets
RUN useradd --create-home --uid 10001 app
USER app

ENV API_URL=http://api:8000 PORT=8050
EXPOSE 8050
# Cloud Run sets PORT; long timeout covers the API's cold start (it loads the model on boot).
CMD ["sh", "-c", "exec gunicorn app:server --bind 0.0.0.0:${PORT} --workers 2 --timeout 120"]

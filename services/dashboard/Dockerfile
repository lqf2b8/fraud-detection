FROM python:3.11-slim

WORKDIR /app
COPY services/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/dashboard/app.py .
COPY services/dashboard/assets ./assets

ENV API_URL=http://api:8000
EXPOSE 8050
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050", "--workers", "2"]

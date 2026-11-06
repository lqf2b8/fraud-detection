FEATURE_GROUPS ?= none
SEED ?= 42

.PHONY: install data train train-synthetic up down logs test lint gcp-bootstrap publish-model

install:          ## local dev environment
	pip install -e ".[dev,notebook]" -r services/api/requirements.txt -r services/dashboard/requirements.txt

data:             ## download IEEE-CIS data (requires Kaggle API token + joined competition)
	python scripts/download_data.py

train:            ## train on real data in Docker (ablation: make train FEATURE_GROUPS=network,anomaly SEED=7)
	docker compose --profile train run --build --rm -e FEATURE_GROUPS=$(FEATURE_GROUPS) -e SEED=$(SEED) trainer

train-synthetic:  ## train on synthetic data in Docker (no download)
	docker compose --profile train run --build --rm trainer --synthetic

up:               ## start API (:8000) and dashboard (:8050)
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

test:
	pytest -q

lint:
	ruff check src services tests

gcp-bootstrap:    ## one-time Google Cloud setup (needs GCP_PROJECT in .env)
	docker compose --profile gcp run --rm gcloud /setup/bootstrap.sh

publish-model:    ## upload ./models to Cloud Storage and roll the live API onto it
	docker compose --profile gcp run --rm gcloud /setup/publish_model.sh

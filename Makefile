.PHONY: install data train train-synthetic up down logs test lint

install:          ## local dev environment
	pip install -e ".[dev]" -r services/api/requirements.txt -r services/dashboard/requirements.txt

data:             ## download IEEE-CIS data (requires Kaggle API token + joined competition)
	bash scripts/download_data.sh

train:            ## train on real data in Docker
	docker compose --profile train run --rm trainer

train-synthetic:  ## train on synthetic data in Docker (no download)
	docker compose --profile train run --rm trainer --synthetic

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

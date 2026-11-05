# Fraud Detection

End-to-end fraud and anomaly detection on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset
(~590k card-not-present e-commerce transactions from Vesta Corporation, ~3.5% fraud).

The project is a small set of Dockerized services:

| Service | Tech | Role |
|---|---|---|
| `trainer` | LightGBM, pandas | One-off job: builds features, trains with a **time-based** split, writes model + metadata |
| `api` | FastAPI | Serves the model — `POST /predict`, `GET /model-info`, `GET /health` |
| `dashboard` | Plotly Dash | Score transactions and review model performance; talks to the model only through the API |

```
            ┌──────────┐   models/ (volume)   ┌───────┐   HTTP   ┌───────────┐
 data/raw ─▶│ trainer  │ ───────────────────▶ │  api  │ ◀─────── │ dashboard │ ◀── browser
            └──────────┘                      └───────┘          └───────────┘
                                                  ▲
                                                  └── any other client (curl, batch jobs)
```

## Quick start

```bash
# 1. Train (synthetic data, no download, ~30s)
make train-synthetic

# 2. Start the API and dashboard
make up
#   API docs:  http://localhost:8000/docs
#   Dashboard: http://localhost:8050
```

To use the real data, set up a Kaggle API token, join the competition and accept its rules, then:

```bash
make data    # downloads to data/raw (git-ignored)
make train
curl -X POST localhost:8000/reload   # API picks up the new model
```

## Example request

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "transactions": [{"TransactionID": 1, "TransactionAmt": 249.95, "ProductCD": "W",
                    "card4": "visa", "P_emaildomain": "anonymous.com", "DeviceType": "mobile"}]
}'
```

Any IEEE-CIS column can be sent. Columns you leave out are treated as missing values.

## Repo layout

```
src/fraud_detection/   shared library: config, data, features, train, evaluate
services/trainer/      training job image
services/api/          FastAPI service
services/dashboard/    Plotly Dash app
notebooks/             EDA and experiments
scripts/               data download
tests/                 unit + API tests (run on synthetic data)
```

## Modeling approach

- **Validation:** train on the earliest 80% of transactions by time and validate on the latest 20%. This mirrors deployment and exposes drift.
- **Metrics:** PR AUC and precision at the top 1% and 5% of scores (a fixed review capacity), not just ROC AUC.
- **Threshold:** stored with the model, and it can be overridden per request (`/predict?threshold=0.3`).

## Roadmap

- [ ] EDA notebook: fraud rate over time, missingness patterns, amount distributions
- [ ] Entity resolution: build a client UID from card/address/email and add UID-level aggregates
- [ ] Velocity and behavioral features (counts and amounts per entity over time windows)
- [ ] Graph features: shared cards, devices and emails linking entities (fraud-ring detection)
- [ ] Unsupervised anomaly score (Isolation Forest) as a complementary signal
- [ ] Cost-based threshold selection (expected loss vs. review cost)
- [ ] SHAP explanations returned by the API and shown in the dashboard
- [ ] Drift monitoring on score and feature distributions
- [ ] Model card documenting assumptions, limitations and performance

## Development

```bash
make install   # local editable install
make test
make lint
```

Data: IEEE-CIS Fraud Detection, Kaggle. Not redistributed here; see the competition rules.

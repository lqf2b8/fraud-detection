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

To use the real data, join the competition and accept its rules, then add your Kaggle credentials:

```bash
cp .env.example .env   # set KAGGLE_API_TOKEN (or legacy KAGGLE_USERNAME + KAGGLE_KEY)
pip install kaggle
make data    # downloads to data/raw (git-ignored); same as: python scripts/download_data.py
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
notebooks/             EDA and experiments (01_eda: drift, signals, missingness, entity resolution)
scripts/               data download (cross-platform Python)
tests/                 unit + API tests (run on synthetic data)
```

## Modeling approach

- **Validation:** train on the earliest 80% of transactions by time and validate on the latest 20%. This mirrors deployment and exposes drift.
- **Metrics:** PR AUC and precision at the top 1% and 5% of scores (a fixed review capacity), not just ROC AUC.
- **Entity features are past-only:** each transaction sees only earlier transactions, both in training and at inference, so validation reflects what a live system would know.
- **Known-fraud history with a reporting delay:** for each customer, card, email + address and device, the model sees how many of its earlier transactions were confirmed fraud. It only counts labels at least `LABEL_DELAY_DAYS` (default 30) old, which mimics chargeback lag. A transaction never sees its own label or any label too recent to have been reported yet. Tests check this.
- **Segments:** metrics are also reported for customers seen in training vs. new customers. The EDA found new customers 2.3× riskier.
- **Threshold:** stored with the model, and it can be overridden per request (`/predict?threshold=0.3`).

## Results

All results come from the same time-based holdout: the latest 20% of training transactions, 118,108 rows, 3.44% fraud.
"Fraud caught" is recall at that review budget.

| Version | Change | PR AUC | ROC AUC | Precision @ top 1% | Precision @ top 5% | Fraud caught @ 5% |
|---|---|---|---|---|---|---|
| v0 | Baseline: raw features, LightGBM | 0.588 | 0.923 | 90.6% | 43.4% | ~63% |
| v1 | + past-only customer (UID) & card history features | 0.592 | 0.925 | 90.4% | 44.0% | ~64% |
| v2 | + known-fraud history (customer, card, email + address, device), 30-day label delay | **0.659** | **0.941** | **96.3%** | **46.7%** | **~68%** |

Random scoring would give a PR AUC of 0.034 (the fraud rate). With v2, reviewing the top 5% of
transactions by model score catches about 68% of all fraud, and 96% of the top 1% are fraud.

**By segment:**

| Segment | Share of validation | Fraud rate | v1 PR AUC | v2 PR AUC |
|---|---|---|---|---|
| Customer seen in training | 39.9% | 1.98% | 0.722 | **0.919** |
| New customer | 60.1% | 4.41% | 0.570 | 0.560 |

Known-fraud history is the strongest signal for returning customers. It doesn't help new customers:
device and email + address history added nothing measurable. New customers make up most of the
volume and the remaining headroom.

**Caveat:** Vesta labels an account's later transactions as fraud once a fraud is reported. Some of
the v2 gain on seen customers is the model learning that labeling rule. In production, a confirmed
fraudulent account would usually be blocked, so the real-world lift on returning customers would
likely be smaller than shown here.

## Roadmap

- [x] EDA notebook: fraud rate over time, missingness patterns, amount distributions, pseudo-customer IDs
- [x] Entity resolution: pseudo-customer UID (card1 + addr1 + email + account start day) with past-only aggregates (prior count, typical amount, amount ratio, time since last transaction)
- [x] Known-fraud history per customer / card / email + address / device, with a 30-day label delay
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

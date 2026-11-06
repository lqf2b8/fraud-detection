# Fraud Detection

**Live demo:** [dashboard](https://fraud-dashboard-73tqrlqnaq-uc.a.run.app) · [API docs](https://fraud-api-73tqrlqnaq-uc.a.run.app/docs)
(hosted on Google Cloud Run; scales to zero, so the first load after a quiet period takes a few seconds)

End-to-end fraud and anomaly detection on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset
(~590k card-not-present e-commerce transactions from Vesta Corporation, ~3.5% fraud).

The project is a small set of Dockerized services:

| Service | Tech | Role |
|---|---|---|
| `trainer` | LightGBM, pandas | One-off job: builds features, trains with a **time-based** split, writes model + metadata |
| `api` | FastAPI | Serves the model: scoring with SHAP explanations, model metadata, example transactions |
| `dashboard` | Plotly Dash | Review queue with explanations, threshold and cost tradeoff, model performance; reaches the model only through the API |

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

## Deploy to Google Cloud Run

The API and dashboard run on Cloud Run, deployed from GitHub Actions with keyless Workload Identity
Federation. The model is trained locally (the Kaggle data never leaves your machine) and published to a
private Cloud Storage bucket that the API mounts read-only.

```
 your machine                     GitHub Actions                  Google Cloud
 make train ──▶ ./models ──make publish-model──────────────────▶ gs://<project>-fraud-models
                                                                       │ (read-only volume)
 git push main ──▶ CI ──▶ deploy ──▶ Artifact Registry ──▶ Cloud Run: fraud-api ◀── fraud-dashboard
```

**One-time setup** (the `gcloud` CLI runs in a container, so nothing to install):

1. Set `GCP_PROJECT` (and optionally `GCP_REGION`) in `.env`. The project needs billing enabled.
2. `make gcp-bootstrap`: sign in, then it creates the image registry, the model bucket, least-privilege
   service accounts, and Workload Identity Federation scoped to this repository only.
3. Add the three values it prints as GitHub repository **variables**
   (Settings → Secrets and variables → Actions → Variables): `GCP_PROJECT`, `GCP_REGION`, `WIF_PROVIDER`.
4. `make publish-model` uploads your trained model.
5. Run the **deploy** workflow (Actions → deploy → Run workflow). After that, every push to `main` that
   passes CI redeploys automatically. The workflow summary prints the dashboard and API URLs.

**Refreshing the model** after an experiment needs no code deploy: `make train`, then
`make publish-model`, which uploads the new artifacts and rolls the API onto them.

Both services scale to zero when idle, so the first request after a quiet period takes a few seconds
while the API loads the model.

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict?explain=true` | Score a batch of transactions; optionally return the top 5 SHAP reasons for each |
| `GET /examples`, `GET /examples/{id}` | Sample of real validation transactions (caught frauds, missed frauds, false alarms, cleared legit) with scores and reasons |
| `GET /model-info` | Metrics, seen vs. new customer segments, threshold table, cost-optimal operating point, SHAP importance |
| `POST /reload`, `GET /health` | Load a newly trained model without restarting; liveness check |

```bash
curl -X POST "localhost:8000/predict?explain=true" -H 'Content-Type: application/json' -d '{
  "transactions": [{"TransactionID": 1, "TransactionAmt": 249.95, "ProductCD": "W",
                    "card4": "visa", "P_emaildomain": "anonymous.com", "DeviceType": "mobile"}]
}'
```

Any IEEE-CIS column can be sent. Columns you leave out are treated as missing values. Each reason includes a plain-English
description, for example *"Confirmed prior frauds on this customer = 2 (+2.1)"*.

## Business layer

- **Cost-based threshold:** cost = review cost × transactions flagged + customer friction × false alarms + fraud
  dollars missed. Training picks the cheapest threshold for the assumed `REVIEW_COST_USD` (default $10) and
  `FALSE_POSITIVE_COST_USD` (default $0), and the API uses it by default.
- **Review capacity:** in the dashboard's *Threshold & cost* tab, set the review cost and the team's capacity
  (max % flagged), plus the friction cost of a false alarm. It shows the cheapest feasible threshold, with frauds caught, fraud dollars caught and savings
  compared with no model.
- **Explanations:** exact TreeSHAP values computed natively by LightGBM (`pred_contrib`), with no extra dependency.
  For each transaction they add up to the model's score. Shown in the review queue and returned by the API.

## Repo layout

```
src/fraud_detection/   shared library: config, data, features, network, anomaly, explain, train, evaluate
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
- **Shared-attribute network (v3):** customers link to the devices, email + address pairs and card fingerprints they use. For each transaction, the model sees how many *other* customers used the same attribute before, and how many of them were already confirmed fraudsters (same 30-day label delay; a customer's own fraud is excluded). A brand-new customer can still be linked to a known fraud ring through a shared device or card.
- **Unsupervised anomaly score (v3):** an Isolation Forest fit on all training transactions without labels. (Fitting on legit transactions only would leak labels into the training set.) It's used as a model feature and also reported on its own.
- **Segments:** metrics are also reported for customers seen in training vs. new customers. The EDA found new customers 2.3× riskier.
- **No validation leakage in model selection:** early stopping uses the last 10% of the *training* period, so the validation set is only ever used for reporting. The raw timestamp is excluded as a feature; hour and day of week are kept.
- **Threshold:** the cost-optimal threshold is stored with the model, and it can be overridden per request (`/predict?threshold=0.3`).

## Results

v2 was slightly optimistic: early stopping picked the number of trees on the validation set. From v2.1 on,
validation is used only for reporting. Note: ROC AUC here isn't comparable to the Kaggle leaderboard, which
scored a hidden test set. These results come from a time split of the training data, with stricter rules.

All results come from the same time-based holdout: the latest 20% of training transactions, 118,108 rows, 3.44% fraud.
"Fraud caught" is recall at that review budget.

| Version | Change | PR AUC | ROC AUC | Precision @ top 1% | Precision @ top 5% | Fraud caught @ 5% |
|---|---|---|---|---|---|---|
| v0 | Baseline: raw features, LightGBM | 0.588 | 0.923 | 90.6% | 43.4% | ~63% |
| v1 | + past-only customer (UID) & card history features | 0.592 | 0.925 | 90.4% | 44.0% | ~64% |
| v2 | + known-fraud history (customer, card, email + address, device), 30-day label delay | 0.659 | 0.941 | 96.3% | 46.7% | ~68% |
| v2.1 | Stricter evaluation: early stopping off validation, raw timestamp dropped | 0.634 | 0.933 | 94.6% | 44.7% | ~65% |
| v2.2 | Refit on the full training period after early stopping | **0.660** | **0.940** | **95.9%** | **46.9%** | **~68%** |
| v3 | + network features + Isolation Forest score | 0.640 | 0.937 | 95.9% | 44.4% | ~65% |
| v3.1 | Ablation: network + anomaly off by default (= v2.2 features) | **0.660** | **0.940** | **95.9%** | **46.9%** | **~68%** |

Random scoring would give a PR AUC of 0.034 (the fraud rate). With v2.2, reviewing the top 5% of
transactions by model score catches about 68% of all fraud, and 96% of the top 1% are fraud. v2.2 matches v2
under the stricter evaluation.

**Operating point (v2.2, assuming $10 per manual review):** flag 9.9% of transactions → catch **80% of fraud
dollars** at 27% precision. Total cost (reviews + fraud missed) is **$239k vs. $610k with no model, 61% lower**,
over the ~1-month validation period. The review cost is an illustrative assumption, not from the data; the
dashboard lets you change it, and the best threshold moves accordingly.

**By segment:**

| Segment | Share of validation | Fraud rate | v1 PR AUC | v2.2 PR AUC |
|---|---|---|---|---|
| Customer seen in training | 39.9% | 1.98% | 0.722 | **0.922** |
| New customer | 60.1% | 4.41% | 0.570 | 0.558 |

**v3 was a negative result.** Adding network features and an Isolation Forest score lowered PR AUC
(0.660 → 0.640). An ablation with a seed-noise check located the problem:

| Run | Feature groups | Seed | PR AUC | Seen customers | New customers |
|---|---|---|---|---|---|
| v2.2 baseline | none | 42 | 0.660 | 0.922 | 0.558 |
| Seed check | none | 7 | 0.657 | 0.914 | 0.557 |
| + anomaly | anomaly | 42 | 0.659 | 0.923 | 0.558 |
| + network | network | 42 | 0.656 | 0.923 | 0.553 |
| v3 | network + anomaly | 42 | 0.640 | 0.884 | 0.554 |

- **Noise floor:** changing the seed moves PR AUC by about 0.003 overall and 0.008 for seen customers.
- **Each group alone:** within noise. Neither helps, including for new customers, the segment they targeted.
- **Combined:** clearly worse for seen customers (−0.038). Likely cause: crowding. Both groups add features
  that overlap with the strong known-fraud signals, so column subsampling picks those signals less often.
  This is a hypothesis, not proven.
- **Anomaly score alone:** finds fraud among new customers with no labels at all (PR AUC 0.153 vs. a 0.044
  base rate; ROC AUC 0.73), but adds nothing once the supervised model sees the same C/D columns.

**Decision:** both groups stay in the code (`FEATURE_GROUPS=network,anomaly`) but are **off by default**.
The served model is v2.2's feature set (v3.1 below).

**v4: velocity features, multi-seed evaluation and diagnostics.** Single runs turned out to be too coarse,
so comparisons now use `make experiment` (features built once, 5 seeds, mean ± sd) and `make diagnose`
(per-feature drift, adversarial validation, segment-level permutation).

| Feature set (5 seeds) | Overall PR AUC | Seen customers | New customers |
|---|---|---|---|
| Baseline (v2.2 features) | 0.658 ± 0.002 | 0.917 ± 0.004 | 0.559 ± 0.002 |
| + velocity (card, region, email + address, device) | 0.640 ± 0.008 | 0.884 ± 0.013 | 0.554 ± 0.002 |
| + velocity without device | 0.648 ± 0.009 | 0.896 ± 0.020 | 0.558 ± 0.002 |

- The baseline is stable (seen-customer sd 0.004), so the earlier single-run drops were real, not noise.
- **Diagnosed:** device-keyed velocity drifted between late training and validation (PSI up to 0.75; the
  1-hour count rose from 5 to 29 on average). `DeviceInfo` values like "Windows" are shared by thousands of
  customers, so their counts track site traffic, not behavior. The same flaw likely sank v3's device and
  card-fingerprint network counts.
- **After removing device:** every feature is stable (PSI ≤ 0.08), but velocity still adds nothing for new
  customers and makes the seen-customer model less stable. The C/D columns already carry this signal.
- **Decision:** velocity stays in the code (`FEATURE_GROUPS=velocity`), off by default.

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
- [x] Velocity features: transactions and dollars per card / region / email + address in the last 1h, 1d, 7d (tested over 5 seeds, no lift; off by default)
- [x] Multi-seed experiments and drift / adversarial / permutation diagnostics (`make experiment`, `make diagnose`)
- [x] Unsupervised anomaly score (Isolation Forest) as a complementary signal, aimed at new customers (tested, no lift; off by default)
- [x] Network features: other customers and known fraudsters sharing a device, email + address or card fingerprint (tested, no lift; off by default)
- [ ] Multi-hop graph features (connected components, fraud-ring size) and graph embeddings
- [ ] Conformal prediction: approve / decline / manual review based on prediction-set uncertainty
- [x] Cost-based threshold selection (expected loss vs. review cost) with review-capacity constraint
- [x] SHAP explanations returned by the API and shown in the dashboard
- [ ] Drift monitoring on score and feature distributions
- [ ] Model card documenting assumptions, limitations and performance

## Development

```bash
make install   # local editable install
make test
make lint
```

Data: IEEE-CIS Fraud Detection, Kaggle. Not redistributed here; see the competition rules.

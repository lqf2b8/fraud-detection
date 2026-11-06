#!/usr/bin/env bash
# Upload the locally trained model to Cloud Storage and roll the API onto it.
#
#   make train && make publish-model
#
# The model is trained on your machine (the Kaggle data never leaves it). This uploads only the
# artifacts in ./models, then starts a new API revision so the live site picks them up.
# No code deploy is needed to refresh the model.
set -euo pipefail

: "${GCP_PROJECT:?Set GCP_PROJECT in .env}"
REGION="${GCP_REGION:-us-central1}"
BUCKET="${GCP_PROJECT}-fraud-models"
MODELS=/models

for f in model.joblib metadata.json examples.parquet; do
  [ -f "$MODELS/$f" ] || { echo "Missing $MODELS/$f. Run 'make train' first."; exit 1; }
done

gcloud config set project "$GCP_PROJECT" --quiet >/dev/null
VERSION="$(python3 -c "import json; m=json.load(open('$MODELS/metadata.json')); print(m['feature_set'] + ' ' + m['trained_at'])" 2>/dev/null || date -u +%FT%TZ)"
echo "Publishing model: $VERSION"

gcloud storage cp "$MODELS/model.joblib" "$MODELS/metadata.json" "$MODELS/examples.parquet" \
  "gs://$BUCKET/" --quiet

if gcloud run services describe fraud-api --region "$REGION" >/dev/null 2>&1; then
  # A new revision re-reads the bucket on startup. MODEL_VERSION is only a label for that.
  gcloud run services update fraud-api --region "$REGION" --quiet \
    --update-env-vars "MODEL_VERSION=$(date -u +%Y%m%d%H%M%S)"
  url="$(gcloud run services describe fraud-api --region "$REGION" --format='value(status.url)')"
  echo "API rolled to the new model: $url/health"
else
  echo "Uploaded. The API isn't deployed yet: push to main (or run the deploy workflow) to create it."
fi

#!/usr/bin/env bash
# One-time Google Cloud setup for running the API and dashboard on Cloud Run, deployed from
# GitHub Actions. Safe to rerun: every step skips or updates what already exists.
#
#   make gcp-bootstrap      (runs this inside the gcloud container; needs GCP_PROJECT in .env)
#
# Creates:
#   - an Artifact Registry repository for the images (keeps the last 3 versions of each)
#   - a Cloud Storage bucket for model artifacts (trained locally, uploaded by publish_model.sh)
#   - service accounts, each with only what it needs:
#       fraud-api        runs the API: reads the model bucket
#       fraud-dashboard  runs the dashboard: no Google Cloud permissions at all
#       fraud-deployer   used by GitHub Actions to push images and deploy the two services
#   - Workload Identity Federation so GitHub Actions in lqf2b8/fraud-detection (and no other
#     repository) can act as fraud-deployer without any stored key
#
# Uses its own names throughout, so it can share a project with other deployments.
set -euo pipefail

: "${GCP_PROJECT:?Set GCP_PROJECT in .env}"
REGION="${GCP_REGION:-us-central1}"
REPO_SLUG="lqf2b8/fraud-detection"
AR_REPO="fraud-detection"
BUCKET="${GCP_PROJECT}-fraud-models"
POOL="github"
PROVIDER="fraud-detection-oidc"
P="$GCP_PROJECT"
sa() { echo "$1@$P.iam.gserviceaccount.com"; }

step() { printf '\n== %s\n' "$*"; }

step "Sign in"
if [ -z "$(gcloud auth list --filter=status:ACTIVE --format='value(account)')" ]; then
  gcloud auth login --no-launch-browser
fi
gcloud config set project "$P" --quiet >/dev/null
NUMBER="$(gcloud projects describe "$P" --format='value(projectNumber)')"

step "Billing"
gcloud services enable cloudbilling.googleapis.com --quiet
if [ "$(gcloud billing projects describe "$P" --format='value(billingEnabled)')" != "True" ]; then
  echo "Billing is not enabled for $P. Link a billing account at"
  echo "  https://console.cloud.google.com/billing/linkedaccount?project=$P"
  echo "then rerun."
  exit 1
fi
echo "billing enabled"

step "APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  iamcredentials.googleapis.com sts.googleapis.com storage.googleapis.com iam.googleapis.com \
  cloudresourcemanager.googleapis.com --quiet

step "Artifact Registry ($REGION)"
if ! gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$AR_REPO" --repository-format=docker \
    --location="$REGION" --description="Fraud detection images" --quiet
fi
cat > /tmp/cleanup.json <<'JSON'
[{"name": "keep-recent", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 3}},
 {"name": "delete-old", "action": {"type": "Delete"}, "condition": {"olderThan": "604800s"}}]
JSON
gcloud artifacts repositories set-cleanup-policies "$AR_REPO" --location="$REGION" \
  --policy=/tmp/cleanup.json --no-dry-run --quiet >/dev/null
echo "images: $REGION-docker.pkg.dev/$P/$AR_REPO"

step "Service accounts"
for name in fraud-api fraud-dashboard fraud-deployer; do
  if ! gcloud iam service-accounts describe "$(sa $name)" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$name" --quiet
  fi
  echo "ok: $(sa $name)"
done
sleep 10  # new accounts take a moment to appear in policies

step "Model bucket"
if ! gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$BUCKET" --location="$REGION" \
    --uniform-bucket-level-access --public-access-prevention --quiet
fi
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:$(sa fraud-api)" --role=roles/storage.objectViewer --quiet >/dev/null
echo "models: gs://$BUCKET (private; only fraud-api reads it)"

step "Deployer permissions"
for role in roles/run.admin roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "$P" --member="serviceAccount:$(sa fraud-deployer)" \
    --role="$role" --condition=None --quiet >/dev/null
done
for name in fraud-api fraud-dashboard; do
  gcloud iam service-accounts add-iam-policy-binding "$(sa $name)" \
    --member="serviceAccount:$(sa fraud-deployer)" --role=roles/iam.serviceAccountUser \
    --quiet >/dev/null
done
echo "fraud-deployer: pushes images and deploys the two services"

step "Workload Identity Federation for $REPO_SLUG"
if ! gcloud iam workload-identity-pools describe "$POOL" --location=global >/dev/null 2>&1; then
  gcloud iam workload-identity-pools create "$POOL" --location=global \
    --display-name="GitHub Actions" --quiet
fi
if ! gcloud iam workload-identity-pools providers describe "$PROVIDER" --location=global \
    --workload-identity-pool="$POOL" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location=global \
    --workload-identity-pool="$POOL" --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository == '$REPO_SLUG'" --quiet
fi
gcloud iam service-accounts add-iam-policy-binding "$(sa fraud-deployer)" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$NUMBER/locations/global/workloadIdentityPools/$POOL/attribute.repository/$REPO_SLUG" \
  --quiet >/dev/null
echo "only $REPO_SLUG can act as fraud-deployer"

step "Done. Add these as GitHub repository variables (Settings > Secrets and variables >"
echo "Actions > Variables). None of them is a secret."
echo "  GCP_PROJECT   $P"
echo "  GCP_REGION    $REGION"
echo "  WIF_PROVIDER  projects/$NUMBER/locations/global/workloadIdentityPools/$POOL/providers/$PROVIDER"
echo
echo "Then upload a trained model:  make publish-model"

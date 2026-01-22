#!/usr/bin/env bash
set -euo pipefail

REGION="europe-west1"
FRONTEND_SERVICE="alcohol-frontend"
BACKEND_SERVICE="alcohol-api"
REPO="containers"

PROJECT_ID="$(gcloud config get-value project)"
if [ -z "$PROJECT_ID" ]; then
  echo "No active gcloud project. Run: gcloud config set project <PROJECT_ID>"
  exit 1
fi

# Get backend URL
BACKEND_URL="$(gcloud run services describe ${BACKEND_SERVICE} --region ${REGION} --format='value(status.url)')"
echo "Backend URL: ${BACKEND_URL}"

# Ensure services enabled
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# Ensure artifact registry exists
if ! gcloud artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" --repository-format=docker --location="${REGION}"
fi

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${FRONTEND_SERVICE}:v1"

# Build with Cloud Build using a direct docker build (no cloudbuild.yaml dependency)
gcloud builds submit \
  --tag "${IMAGE}" \
  --region "${REGION}" \
  --timeout "1200s" \
  --config /dev/null \
  -- \
  .

# Deploy
gcloud run deploy "${FRONTEND_SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --set-env-vars "BACKEND=${BACKEND_URL}"

echo "Frontend deployed:"
gcloud run services describe "${FRONTEND_SERVICE}" --region "${REGION}" --format='value(status.url)'


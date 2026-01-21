#!/usr/bin/env bash
set -euo pipefail

REGION="europe-west1"
SERVICE="alcohol-api"
REPO="containers"
DOCKERFILE="dockerfiles/api.Dockerfile"

PROJECT_ID="$(gcloud config get-value project)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:v1"

# Enable required APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com >/dev/null

# Create Artifact Registry repo if it doesn't exist
if ! gcloud artifacts repositories describe "${REPO}" --location "${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Docker images"
fi

echo "Building image with Cloud Build using cloudbuild.yaml: ${IMAGE}"
gcloud builds submit --config cloudbuild.yaml .

echo "Deploying to Cloud Run: ${SERVICE}"
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --allow-unauthenticated


echo "Service URL:"
gcloud run services describe "${SERVICE}" \
  --region "${REGION}" \
  --format="value(status.url)"


#!/usr/bin/env bash
# ==============================================================================
# Google Cloud Platform Production Deployment Script
# Multi-Document AI Research Assistant
# ==============================================================================

set -euo pipefail

# Configurable variables
GCP_PROJECT_ID="${GCP_PROJECT_ID:-my-rag-project}"
GCP_REGION="${GCP_REGION:-us-central1}"
BACKEND_SERVICE_NAME="rag-backend"
FRONTEND_SERVICE_NAME="rag-frontend"
DB_INSTANCE_NAME="rag-postgres-db"
GCS_BUCKET_NAME="${GCP_PROJECT_ID}-rag-uploads"

echo "=== 🚀 Starting GCP Production Deployment ==="
echo "Project ID: ${GCP_PROJECT_ID}"
echo "Region:     ${GCP_REGION}"

# 1. Set gcloud project
gcloud config set project "${GCP_PROJECT_ID}"

# 2. Enable GCP APIs
echo "=== Enabling GCP Services APIs ==="
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    sqladmin.googleapis.com \
    secretmanager.googleapis.com \
    storage.googleapis.com \
    cloudbuild.googleapis.com

# 3. Create Artifact Registry repository if not exists
echo "=== Setting up Artifact Registry ==="
if ! gcloud artifacts repositories describe rag-repo --location="${GCP_REGION}" >/dev/null 2>&1; then
    gcloud artifacts repositories create rag-repo \
        --repository-format=docker \
        --location="${GCP_REGION}" \
        --description="Docker repository for RAG application"
fi

REGISTRY_URL="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/rag-repo"

# 4. Build and Push Container Images
echo "=== Building Backend Image ==="
gcloud builds submit --tag "${REGISTRY_URL}/${BACKEND_SERVICE_NAME}:latest" -f backend/Dockerfile .

echo "=== Building Frontend Image ==="
gcloud builds submit --tag "${REGISTRY_URL}/${FRONTEND_SERVICE_NAME}:latest" -f frontend/Dockerfile .

# 5. Deploy Backend to Cloud Run
echo "=== Deploying Backend to Cloud Run ==="
gcloud run deploy "${BACKEND_SERVICE_NAME}" \
    --image="${REGISTRY_URL}/${BACKEND_SERVICE_NAME}:latest" \
    --platform=managed \
    --region="${GCP_REGION}" \
    --allow-unauthenticated \
    --min-instances=0 \
    --max-instances=5 \
    --memory=2Gi \
    --cpu=2 \
    --set-env-vars="LOG_FORMAT=json,LOG_LEVEL=INFO"

BACKEND_URL=$(gcloud run services describe "${BACKEND_SERVICE_NAME}" --platform=managed --region="${GCP_REGION}" --format="value(status.url)")
echo "Backend URL: ${BACKEND_URL}"

# 6. Deploy Frontend to Cloud Run
echo "=== Deploying Frontend to Cloud Run ==="
gcloud run deploy "${FRONTEND_SERVICE_NAME}" \
    --image="${REGISTRY_URL}/${FRONTEND_SERVICE_NAME}:latest" \
    --platform=managed \
    --region="${GCP_REGION}" \
    --allow-unauthenticated \
    --min-instances=0 \
    --max-instances=3 \
    --memory=1Gi \
    --set-env-vars="BACKEND_URL=${BACKEND_URL}"

FRONTEND_URL=$(gcloud run services describe "${FRONTEND_SERVICE_NAME}" --platform=managed --region="${GCP_REGION}" --format="value(status.url)")

echo "=== ✅ GCP Deployment Completed Successfully! ==="
echo "Backend API:  ${BACKEND_URL}"
echo "Frontend UI:  ${FRONTEND_URL}"

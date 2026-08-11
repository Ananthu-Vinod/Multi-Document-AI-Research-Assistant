# Production Deployment Manual

This directory contains the production deployment assets and scripts for deploying **Multi-Document AI Research Assistant** to Google Cloud Platform (GCP) or Docker Compose.

---

## Option 1: Docker Compose Deployment (Self-Hosted / VPS)

To run the complete stack locally or on a VPS (Linux/Ubuntu):

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Fill in your LLM API Keys in .env
# GEMINI_API_KEY=your_key_here
# JWT_SECRET=your_random_secret_string

# 3. Start all services (PostgreSQL + FastAPI + Streamlit)
docker-compose up -d --build

# 4. Check container health
docker-compose ps
```

- **Frontend UI**: `http://localhost:8501`
- **Backend REST API & Swagger Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/healthz`
- **Metrics**: `http://localhost:8000/metrics`

---

## Option 2: Google Cloud Platform (Cloud Run + Cloud SQL)

### Prerequisites
- Install [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install)
- Authenticate gcloud: `gcloud auth login`
- Set your target project: `gcloud config set project YOUR_PROJECT_ID`

### One-Command Deployment

```bash
# Make script executable
chmod +x deploy/deploy_gcp.sh

# Run GCP deployment
GCP_PROJECT_ID="your-gcp-project-id" GCP_REGION="us-central1" ./deploy/deploy_gcp.sh
```

### Environment Variables & Secrets in Cloud Run
Set your API credentials in Google Secret Manager or directly in Cloud Run:

```bash
gcloud run services update rag-backend \
    --region us-central1 \
    --update-env-vars GEMINI_API_KEY="your-gemini-api-key",JWT_SECRET="your-jwt-secret-key"
```

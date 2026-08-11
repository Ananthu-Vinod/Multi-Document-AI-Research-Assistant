# Multi-stage production Dockerfile for FastAPI backend (Optimized for 512MB RAM limits)
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy source context and locate requirements.txt dynamically
COPY . /tmp/src
RUN if [ -f /tmp/src/requirements.txt ]; then \
        cp /tmp/src/requirements.txt /app/requirements.txt; \
    elif [ -f /tmp/src/backend/requirements.txt ]; then \
        cp /tmp/src/backend/requirements.txt /app/requirements.txt; \
    fi

RUN pip install --prefix=/install --no-cache-dir -r /app/requirements.txt

# Runner stage
FROM python:3.11-slim AS runner

WORKDIR /app

# Restrict multi-threading memory overhead for low-RAM containers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/backend:/app \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Copy application files dynamically based on build context
COPY . /tmp/src
RUN mkdir -p /app/backend && \
    if [ -f /tmp/src/app.py ]; then \
        cp -r /tmp/src/* /app/backend/; \
    elif [ -d /tmp/src/backend ]; then \
        cp -r /tmp/src/backend/* /app/backend/; \
    fi

# Create runtime directories & non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /app/uploads /app/chroma_db /app/data && \
    chown -R appuser:appuser /app

USER appuser

WORKDIR /app/backend

EXPOSE 8000 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/healthz || exit 1

# Bind dynamically to $PORT and use 1 worker to stay under 512MB RAM
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

# Production Dockerfile for Streamlit Frontend
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
    elif [ -f /tmp/src/frontend/requirements.txt ]; then \
        cp /tmp/src/frontend/requirements.txt /app/requirements.txt; \
    fi

RUN pip install --prefix=/install --no-cache-dir -r /app/requirements.txt

FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/frontend:/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Copy application files dynamically based on build context
COPY . /tmp/src
RUN mkdir -p /app/frontend && \
    if [ -f /tmp/src/streamlit_app.py ]; then \
        cp -r /tmp/src/* /app/frontend/; \
    elif [ -d /tmp/src/frontend ]; then \
        cp -r /tmp/src/frontend/* /app/frontend/; \
    fi

RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

USER appuser

WORKDIR /app/frontend

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]

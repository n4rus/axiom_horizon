# Multi-stage build for production
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Production stage
FROM python:3.12-slim AS production

# Create non-root user
RUN groupadd -r kai && useradd -r -g kai -d /app -s /sbin/nologin kai

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /install /usr/local

# Copy core Kai files
COPY kai_mind.py .
COPY kai_bridge.py .
COPY physics_core.py .
COPY rust/ rust/

# Copy product layer
COPY products/kai-insight/ products/kai-insight/

# Copy monitoring configs
COPY products/kai-insight/monitoring/ monitoring/

# Copy nginx configs
COPY products/kai-insight/nginx/ nginx/

# Set ownership
RUN chown -R kai:kai /app

# Switch to non-root user
USER kai

ENV PYTHONPATH=/app
ENV KAI_PORT=8000
ENV KAI_DAEMON_INTERVAL=60
ENV KAI_SAVE_EVERY=5

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python3", "products/kai-insight/api/rest_server_final.py"]

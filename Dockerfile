# Multi-stage build for minimal production image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt requirements-security.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -r requirements-security.txt

# Production stage
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -g 1000 antbot && \
    useradd -u 1000 -g antbot -m -s /bin/bash antbot

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/config /app/models && \
    chown -R antbot:antbot /app

# Copy source code with optimized layer caching
COPY --chown=antbot:antbot src/ /app/src/
COPY --chown=antbot:antbot config/ /app/config/
COPY --chown=antbot:antbot scripts/ /app/scripts/
COPY --chown=antbot:antbot main.py env.template /app/

# Set permissions
RUN chmod -R 755 /app/src /app/config && \
    chmod -R 777 /app/data /app/logs && \
    chmod +x /app/main.py

# Switch to non-root user
USER antbot

# Environment variables for production
ENV SIMULATION_MODE=true \
    INITIAL_CAPITAL=0.1 \
    LOG_LEVEL=INFO \
    ENABLE_MONITORING=true

# Expose monitoring port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.path.append('.'); from src.core.system_metrics import SystemMetrics; print('Health check passed')" || exit 1

# Run the Enhanced Ant Bot
CMD ["python", "main.py"] 
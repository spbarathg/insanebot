FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    POETRY_VERSION=1.4.2 \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libssl-dev \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/config && \
    chown -R appuser:appuser /app

# Copy application code
COPY src /app/src
COPY scripts /app/scripts

# Set permissions
RUN chmod -R 755 /app/src && \
    chmod -R 777 /app/data /app/logs /app/config

# Switch to non-root user
USER appuser

# Set environment variables for Solana
ENV SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
ENV SIMULATION_MODE=True
ENV SIMULATION_CAPITAL=0.1

# Run the application
CMD ["python", "src/main.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
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
RUN mkdir -p /app/data /app/logs /app/config /app/models && \
    chown -R appuser:appuser /app

# Copy Enhanced Ant Bot application code
COPY src /app/src
COPY config /app/config
COPY scripts /app/scripts
COPY enhanced_main_entry.py /app/
COPY start_without_grok.py /app/
COPY run_enhanced_ant_bot.py /app/
COPY config.py /app/
COPY config.json /app/

# Set permissions
RUN chmod -R 755 /app/src /app/config && \
    chmod -R 777 /app/data /app/logs && \
    chmod +x /app/enhanced_main_entry.py /app/start_without_grok.py /app/run_enhanced_ant_bot.py

# Switch to non-root user
USER appuser

# Set environment variables for Enhanced Ant Bot
ENV SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
ENV SIMULATION_MODE=True
ENV INITIAL_CAPITAL=0.1
ENV USE_MOCK_GROK=true

# Run the Enhanced Ant Bot
CMD ["python", "start_without_grok.py"]

# Health check - using simple script check since no web server by default
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import sys; sys.path.append('.'); from src.core.quicknode_service import QuickNodeService; print('Health check passed')" || exit 1
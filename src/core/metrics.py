"""
Performance metrics collection and exposition using Prometheus.
"""
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import logging
import threading

logger = logging.getLogger(__name__)

# Metrics definitions
REQUEST_LATENCY = Histogram('bot_request_latency_seconds', 'Request latency', ['endpoint'])
TRADE_SUCCESS = Counter('bot_trade_success_total', 'Number of successful trades')
TRADE_FAILURE = Counter('bot_trade_failure_total', 'Number of failed trades')
DROPPED_TRADES = Counter('bot_dropped_trades_total', 'Number of dropped trades')
TPS = Gauge('bot_trades_per_second', 'Trades per second')

# Example usage wrappers
def track_latency(endpoint: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        return wrapper
    return decorator

# Start Prometheus metrics server in a background thread
def start_metrics_server(port: int = 8001):
    def run():
        logger.info(f"Starting Prometheus metrics server on port {port}")
        start_http_server(port)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

# Example: call start_metrics_server() at app startup 
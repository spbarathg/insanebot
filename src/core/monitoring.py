"""
Monitoring and observability module for the trading bot.
"""
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import time
from typing import Dict, Any

# Initialize OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Initialize Prometheus metrics
TRADE_COUNTER = Counter('trading_bot_trades_total', 'Total trades executed', ['side', 'status'])
POSITION_SIZE = Gauge('trading_bot_position_size', 'Current position size', ['token'])
TOKEN_PRICE = Gauge('trading_bot_token_price', 'Token price', ['token'])
TRADE_LATENCY = Histogram('trading_bot_trade_latency_seconds', 'Trade execution latency')
ERROR_COUNTER = Counter('trading_bot_errors_total', 'Error count', ['type'])

class MonitoringManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_logging()
        self._setup_tracing()
        self._setup_metrics()
        
    def _setup_logging(self):
        """Configure structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        LoggingInstrumentor().instrument()
        
    def _setup_tracing(self):
        """Configure OpenTelemetry tracing."""
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.config.get('otlp_endpoint', 'localhost:4317')
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
    def _setup_metrics(self):
        """Start Prometheus metrics server."""
        start_http_server(self.config.get('metrics_port', 8000))
        
    @tracer.start_as_current_span("execute_trade")
    async def record_trade(self, token: str, side: str, amount: float, price: float, status: str):
        """Record trade metrics and traces."""
        TRADE_COUNTER.labels(side=side, status=status).inc()
        POSITION_SIZE.labels(token=token).set(amount)
        TOKEN_PRICE.labels(token=token).set(price)
        
    @tracer.start_as_current_span("measure_latency")
    def record_latency(self, start_time: float):
        """Record trade execution latency."""
        latency = time.time() - start_time
        TRADE_LATENCY.observe(latency)
        
    def record_error(self, error_type: str):
        """Record error metrics."""
        ERROR_COUNTER.labels(type=error_type).inc()
        
    @tracer.start_as_current_span("update_position")
    def update_position(self, token: str, size: float):
        """Update position metrics."""
        POSITION_SIZE.labels(token=token).set(size)
        
    @tracer.start_as_current_span("update_price")
    def update_price(self, token: str, price: float):
        """Update price metrics."""
        TOKEN_PRICE.labels(token=token).set(price) 
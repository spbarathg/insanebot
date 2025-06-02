"""
Prometheus Metrics Integration

Comprehensive metrics collection for Enhanced Ant Bot system monitoring.
Tracks trading performance, system health, defense effectiveness, and more.
"""

import time
import logging
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metric types for categorization"""
    TRADING = "trading"
    SYSTEM = "system"
    DEFENSE = "defense"
    PERFORMANCE = "performance"
    NETWORK = "network"

@dataclass
class MetricConfig:
    """Configuration for metrics collection"""
    enabled: bool = True
    collection_interval: int = 10  # seconds
    retention_hours: int = 24
    export_port: int = 8000

class EnhancedAntBotMetrics:
    """
    Comprehensive metrics collection for Enhanced Ant Bot
    
    Provides real-time monitoring of:
    - Trading performance and P&L
    - System health and uptime
    - Defense system effectiveness
    - Network and execution performance
    - Agent lifecycle and scaling
    """
    
    def __init__(self, config: MetricConfig = None):
        self.config = config or MetricConfig()
        self.registry = CollectorRegistry()
        
        # Initialize all metrics
        self._init_trading_metrics()
        self._init_system_metrics()
        self._init_defense_metrics()
        self._init_performance_metrics()
        self._init_network_metrics()
        
        logger.info("📊 Prometheus metrics initialized")
    
    def _init_trading_metrics(self):
        """Initialize trading-related metrics"""
        # Trading counters
        self.trades_total = Counter(
            'antbot_trades_total',
            'Total number of trades executed',
            ['strategy', 'outcome', 'token_type'],
            registry=self.registry
        )
        
        self.profit_loss_total = Counter(
            'antbot_profit_loss_sol_total',
            'Total profit/loss in SOL',
            ['strategy', 'outcome'],
            registry=self.registry
        )
        
        # Trading gauges
        self.active_positions = Gauge(
            'antbot_active_positions',
            'Number of active trading positions',
            ['strategy'],
            registry=self.registry
        )
        
        self.portfolio_value_sol = Gauge(
            'antbot_portfolio_value_sol',
            'Current portfolio value in SOL',
            registry=self.registry
        )
        
        self.available_capital_sol = Gauge(
            'antbot_available_capital_sol',
            'Available capital for trading in SOL',
            registry=self.registry
        )
        
        # Trading histograms
        self.trade_duration_seconds = Histogram(
            'antbot_trade_duration_seconds',
            'Duration of trades in seconds',
            ['strategy', 'outcome'],
            buckets=[60, 300, 900, 1800, 3600, 7200, 14400, 28800],
            registry=self.registry
        )
        
        self.trade_profit_percentage = Histogram(
            'antbot_trade_profit_percentage',
            'Trade profit/loss percentage',
            ['strategy'],
            buckets=[-50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 200, 500, 1000],
            registry=self.registry
        )
    
    def _init_system_metrics(self):
        """Initialize system health metrics"""
        # System info
        self.system_info = Info(
            'antbot_system_info',
            'System information',
            registry=self.registry
        )
        
        # System gauges
        self.system_uptime_seconds = Gauge(
            'antbot_system_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
        
        self.active_ants_total = Gauge(
            'antbot_active_ants_total',
            'Total number of active ants',
            ['ant_type'],
            registry=self.registry
        )
        
        self.memory_usage_bytes = Gauge(
            'antbot_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage_percentage = Gauge(
            'antbot_cpu_usage_percentage',
            'CPU usage percentage',
            registry=self.registry
        )
        
        # System counters
        self.system_errors_total = Counter(
            'antbot_system_errors_total',
            'Total system errors',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        self.system_restarts_total = Counter(
            'antbot_system_restarts_total',
            'Total system restarts',
            ['reason'],
            registry=self.registry
        )
    
    def _init_defense_metrics(self):
        """Initialize defense system metrics"""
        # Defense counters
        self.defense_activations_total = Counter(
            'antbot_defense_activations_total',
            'Total defense system activations',
            ['layer', 'threat_type'],
            registry=self.registry
        )
        
        self.threats_detected_total = Counter(
            'antbot_threats_detected_total',
            'Total threats detected',
            ['threat_type', 'severity'],
            registry=self.registry
        )
        
        self.trades_blocked_total = Counter(
            'antbot_trades_blocked_total',
            'Total trades blocked by defense systems',
            ['layer', 'reason'],
            registry=self.registry
        )
        
        # Defense gauges
        self.survival_score = Gauge(
            'antbot_survival_score',
            'Current system survival score (0-100)',
            registry=self.registry
        )
        
        self.defense_mode = Gauge(
            'antbot_defense_mode',
            'Current defense mode (0=normal, 1=elevated, 2=high, 3=critical, 4=lockdown)',
            registry=self.registry
        )
        
        self.active_threats = Gauge(
            'antbot_active_threats',
            'Number of active threats',
            ['threat_type'],
            registry=self.registry
        )
    
    def _init_performance_metrics(self):
        """Initialize performance metrics"""
        # Execution performance
        self.transaction_execution_seconds = Histogram(
            'antbot_transaction_execution_seconds',
            'Transaction execution time in seconds',
            ['transaction_type', 'outcome'],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.sub_100ms_executions_total = Counter(
            'antbot_sub_100ms_executions_total',
            'Total sub-100ms executions',
            ['transaction_type'],
            registry=self.registry
        )
        
        # Signal processing
        self.signal_processing_seconds = Histogram(
            'antbot_signal_processing_seconds',
            'Signal processing time in seconds',
            ['signal_type'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        self.signals_generated_total = Counter(
            'antbot_signals_generated_total',
            'Total signals generated',
            ['signal_type', 'confidence_level'],
            registry=self.registry
        )
    
    def _init_network_metrics(self):
        """Initialize network and RPC metrics"""
        # RPC performance
        self.rpc_request_duration_seconds = Histogram(
            'antbot_rpc_request_duration_seconds',
            'RPC request duration in seconds',
            ['rpc_provider', 'method'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.rpc_requests_total = Counter(
            'antbot_rpc_requests_total',
            'Total RPC requests',
            ['rpc_provider', 'method', 'status'],
            registry=self.registry
        )
        
        # Network conditions
        self.network_congestion_score = Gauge(
            'antbot_network_congestion_score',
            'Network congestion score (0-100)',
            registry=self.registry
        )
        
        self.priority_fee_lamports = Gauge(
            'antbot_priority_fee_lamports',
            'Current priority fee in lamports',
            registry=self.registry
        )
    
    # Trading metrics methods
    def record_trade(self, strategy: str, outcome: str, token_type: str, 
                    profit_sol: float, profit_pct: float, duration_seconds: float):
        """Record a completed trade"""
        self.trades_total.labels(strategy=strategy, outcome=outcome, token_type=token_type).inc()
        self.profit_loss_total.labels(strategy=strategy, outcome=outcome).inc(profit_sol)
        self.trade_duration_seconds.labels(strategy=strategy, outcome=outcome).observe(duration_seconds)
        self.trade_profit_percentage.labels(strategy=strategy).observe(profit_pct)
    
    def update_portfolio_metrics(self, portfolio_value: float, available_capital: float, 
                               active_positions_by_strategy: Dict[str, int]):
        """Update portfolio-related metrics"""
        self.portfolio_value_sol.set(portfolio_value)
        self.available_capital_sol.set(available_capital)
        
        for strategy, count in active_positions_by_strategy.items():
            self.active_positions.labels(strategy=strategy).set(count)
    
    # System metrics methods
    def update_system_health(self, uptime_seconds: float, memory_bytes: int, 
                           cpu_percentage: float, active_ants: Dict[str, int]):
        """Update system health metrics"""
        self.system_uptime_seconds.set(uptime_seconds)
        self.memory_usage_bytes.set(memory_bytes)
        self.cpu_usage_percentage.set(cpu_percentage)
        
        for ant_type, count in active_ants.items():
            self.active_ants_total.labels(ant_type=ant_type).set(count)
    
    def record_system_error(self, component: str, error_type: str):
        """Record a system error"""
        self.system_errors_total.labels(component=component, error_type=error_type).inc()
    
    def record_system_restart(self, reason: str):
        """Record a system restart"""
        self.system_restarts_total.labels(reason=reason).inc()
    
    # Defense metrics methods
    def record_defense_activation(self, layer: str, threat_type: str):
        """Record defense system activation"""
        self.defense_activations_total.labels(layer=layer, threat_type=threat_type).inc()
    
    def record_threat_detection(self, threat_type: str, severity: str):
        """Record threat detection"""
        self.threats_detected_total.labels(threat_type=threat_type, severity=severity).inc()
    
    def record_trade_blocked(self, layer: str, reason: str):
        """Record trade blocked by defense system"""
        self.trades_blocked_total.labels(layer=layer, reason=reason).inc()
    
    def update_defense_status(self, survival_score: float, defense_mode: int, 
                            active_threats_by_type: Dict[str, int]):
        """Update defense system status"""
        self.survival_score.set(survival_score)
        self.defense_mode.set(defense_mode)
        
        for threat_type, count in active_threats_by_type.items():
            self.active_threats.labels(threat_type=threat_type).set(count)
    
    # Performance metrics methods
    def record_transaction_execution(self, transaction_type: str, outcome: str, 
                                   execution_time_seconds: float):
        """Record transaction execution performance"""
        self.transaction_execution_seconds.labels(
            transaction_type=transaction_type, 
            outcome=outcome
        ).observe(execution_time_seconds)
        
        if execution_time_seconds < 0.1:  # Sub-100ms
            self.sub_100ms_executions_total.labels(transaction_type=transaction_type).inc()
    
    def record_signal_processing(self, signal_type: str, processing_time_seconds: float, 
                               confidence_level: str):
        """Record signal processing performance"""
        self.signal_processing_seconds.labels(signal_type=signal_type).observe(processing_time_seconds)
        self.signals_generated_total.labels(
            signal_type=signal_type, 
            confidence_level=confidence_level
        ).inc()
    
    # Network metrics methods
    def record_rpc_request(self, rpc_provider: str, method: str, status: str, 
                          duration_seconds: float):
        """Record RPC request performance"""
        self.rpc_requests_total.labels(
            rpc_provider=rpc_provider, 
            method=method, 
            status=status
        ).inc()
        self.rpc_request_duration_seconds.labels(
            rpc_provider=rpc_provider, 
            method=method
        ).observe(duration_seconds)
    
    def update_network_conditions(self, congestion_score: float, priority_fee_lamports: int):
        """Update network condition metrics"""
        self.network_congestion_score.set(congestion_score)
        self.priority_fee_lamports.set(priority_fee_lamports)
    
    def get_metrics_output(self) -> str:
        """Get Prometheus metrics output"""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_content_type(self) -> str:
        """Get Prometheus content type"""
        return CONTENT_TYPE_LATEST

# Global metrics instance
_metrics_instance: Optional[EnhancedAntBotMetrics] = None

def get_metrics() -> EnhancedAntBotMetrics:
    """Get global metrics instance"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = EnhancedAntBotMetrics()
    return _metrics_instance

def initialize_metrics(config: MetricConfig = None) -> EnhancedAntBotMetrics:
    """Initialize global metrics instance"""
    global _metrics_instance
    _metrics_instance = EnhancedAntBotMetrics(config)
    return _metrics_instance 
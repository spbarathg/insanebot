"""
Production Safeguards and Safety Mechanisms
Implements comprehensive safety features for production deployment.
"""

import asyncio
import time
import logging
import json
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from datetime import datetime, timedelta
import psutil
import signal
import sys

from src.core.titan_shield_coordinator import TitanShieldCoordinator, DefenseMode


class SafeguardLevel(Enum):
    """Safeguard severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SafeguardType(Enum):
    """Types of safeguards"""
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    OPERATIONAL = "operational"
    SECURITY = "security"
    COMPLIANCE = "compliance"


@dataclass
class SafeguardRule:
    """Individual safeguard rule definition"""
    rule_id: str
    name: str
    description: str
    safeguard_type: SafeguardType
    level: SafeguardLevel
    threshold: float
    action: str
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class SystemHealth:
    """System health metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_latency: float = 0.0
    active_connections: int = 0
    error_rate: float = 0.0
    uptime: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class FinancialMetrics:
    """Financial safety metrics"""
    total_capital: float = 0.0
    available_capital: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    risk_exposure: float = 0.0
    leverage_ratio: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


class ProductionSafeguards:
    """
    Comprehensive production safeguards system
    Implements multiple layers of safety mechanisms
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.safeguard_rules: Dict[str, SafeguardRule] = {}
        self.system_health = SystemHealth()
        self.financial_metrics = FinancialMetrics()
        
        # Safety state
        self.is_initialized = False
        self.emergency_mode = False
        self.shutdown_requested = False
        self.last_health_check = datetime.now()
        
        # Monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.health_check_interval = config.get("health_check_interval", 30)  # seconds
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, bool] = {}
        self.emergency_callbacks: List[Callable] = []
        
        # Logging
        self.logger = self._setup_logging()
        
        # Initialize safeguard rules
        self._initialize_safeguard_rules()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup production-grade logging"""
        logger = logging.getLogger("production_safeguards")
        logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler for critical events
        try:
            file_handler = logging.FileHandler('logs/production_safeguards.log')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.WARNING)
            logger.addHandler(file_handler)
        except Exception:
            pass  # Continue without file logging if directory doesn't exist
        
        return logger
    
    def _initialize_safeguard_rules(self):
        """Initialize all safeguard rules"""
        
        # Financial safeguards
        financial_rules = [
            SafeguardRule(
                rule_id="daily_loss_limit",
                name="Daily Loss Limit",
                description="Halt trading if daily losses exceed threshold",
                safeguard_type=SafeguardType.FINANCIAL,
                level=SafeguardLevel.HIGH,
                threshold=0.05,  # 5% daily loss limit
                action="halt_trading"
            ),
            SafeguardRule(
                rule_id="max_drawdown_limit",
                name="Maximum Drawdown Limit",
                description="Emergency stop if drawdown exceeds limit",
                safeguard_type=SafeguardType.FINANCIAL,
                level=SafeguardLevel.CRITICAL,
                threshold=0.20,  # 20% max drawdown
                action="emergency_stop"
            ),
            SafeguardRule(
                rule_id="position_size_limit",
                name="Position Size Limit",
                description="Limit individual position sizes",
                safeguard_type=SafeguardType.FINANCIAL,
                level=SafeguardLevel.MEDIUM,
                threshold=0.10,  # 10% max position size
                action="reject_trade"
            ),
            SafeguardRule(
                rule_id="leverage_limit",
                name="Leverage Limit",
                description="Limit total leverage exposure",
                safeguard_type=SafeguardType.FINANCIAL,
                level=SafeguardLevel.HIGH,
                threshold=2.0,  # 2x max leverage
                action="reduce_positions"
            ),
            SafeguardRule(
                rule_id="liquidity_reserve",
                name="Liquidity Reserve",
                description="Maintain minimum liquidity reserves",
                safeguard_type=SafeguardType.FINANCIAL,
                level=SafeguardLevel.HIGH,
                threshold=0.20,  # 20% minimum liquidity
                action="halt_new_positions"
            )
        ]
        
        # Technical safeguards
        technical_rules = [
            SafeguardRule(
                rule_id="memory_usage_limit",
                name="Memory Usage Limit",
                description="Alert when memory usage is high",
                safeguard_type=SafeguardType.TECHNICAL,
                level=SafeguardLevel.HIGH,
                threshold=85.0,  # 85% memory usage
                action="force_gc_and_alert"
            ),
            SafeguardRule(
                rule_id="cpu_usage_limit",
                name="CPU Usage Limit",
                description="Alert when CPU usage is sustained high",
                safeguard_type=SafeguardType.TECHNICAL,
                level=SafeguardLevel.MEDIUM,
                threshold=90.0,  # 90% CPU usage
                action="throttle_operations"
            ),
            SafeguardRule(
                rule_id="error_rate_limit",
                name="Error Rate Limit",
                description="Circuit breaker for high error rates",
                safeguard_type=SafeguardType.TECHNICAL,
                level=SafeguardLevel.HIGH,
                threshold=0.10,  # 10% error rate
                action="enable_circuit_breaker"
            ),
            SafeguardRule(
                rule_id="network_latency_limit",
                name="Network Latency Limit",
                description="Alert on high network latency",
                safeguard_type=SafeguardType.TECHNICAL,
                level=SafeguardLevel.MEDIUM,
                threshold=500.0,  # 500ms latency
                action="switch_to_backup_endpoints"
            )
        ]
        
        # Operational safeguards
        operational_rules = [
            SafeguardRule(
                rule_id="uptime_limit",
                name="Continuous Uptime Limit",
                description="Force restart after extended uptime",
                safeguard_type=SafeguardType.OPERATIONAL,
                level=SafeguardLevel.MEDIUM,
                threshold=86400.0,  # 24 hours
                action="schedule_restart"
            ),
            SafeguardRule(
                rule_id="trade_frequency_limit",
                name="Trade Frequency Limit",
                description="Limit trades per time period",
                safeguard_type=SafeguardType.OPERATIONAL,
                level=SafeguardLevel.MEDIUM,
                threshold=100.0,  # 100 trades per hour
                action="throttle_trading"
            )
        ]
        
        # Add all rules
        all_rules = financial_rules + technical_rules + operational_rules
        for rule in all_rules:
            self.safeguard_rules[rule.rule_id] = rule
    
    async def initialize(self) -> bool:
        """Initialize production safeguards"""
        try:
            self.logger.info("Initializing production safeguards")
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Start monitoring thread
            self._start_monitoring()
            
            # Initial health check
            await self._update_system_health()
            
            self.is_initialized = True
            self.logger.info("Production safeguards initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize production safeguards: {e}")
            return False
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ProductionSafeguardsMonitor"
        )
        self.monitoring_thread.start()
        self.logger.info("Started production safeguards monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.shutdown_requested:
            try:
                # Update system health
                asyncio.run(self._update_system_health())
                
                # Check all safeguard rules
                asyncio.run(self._check_all_safeguards())
                
                # Sleep until next check
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Short sleep on error
    
    async def _update_system_health(self):
        """Update system health metrics"""
        try:
            # CPU and memory usage
            self.system_health.cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            self.system_health.memory_usage = memory_info.percent
            
            # Disk usage
            disk_info = psutil.disk_usage('/')
            self.system_health.disk_usage = (disk_info.used / disk_info.total) * 100
            
            # Network (simplified)
            self.system_health.network_latency = 0.0  # Would implement actual network checks
            
            # Uptime
            self.system_health.uptime = time.time() - psutil.boot_time()
            
            self.system_health.last_update = datetime.now()
            self.last_health_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to update system health: {e}")
    
    async def _check_all_safeguards(self):
        """Check all active safeguard rules"""
        for rule_id, rule in self.safeguard_rules.items():
            if not rule.enabled:
                continue
            
            try:
                await self._check_safeguard_rule(rule)
            except Exception as e:
                self.logger.error(f"Error checking safeguard rule {rule_id}: {e}")
    
    async def _check_safeguard_rule(self, rule: SafeguardRule):
        """Check individual safeguard rule"""
        current_value = await self._get_current_value(rule)
        
        if current_value is None:
            return
        
        # Check if threshold is exceeded
        threshold_exceeded = False
        
        if rule.rule_id in ["daily_loss_limit", "max_drawdown_limit"]:
            # For loss/drawdown rules, trigger when value exceeds threshold
            threshold_exceeded = current_value >= rule.threshold
        elif rule.rule_id in ["liquidity_reserve"]:
            # For reserve rules, trigger when value falls below threshold
            threshold_exceeded = current_value <= rule.threshold
        else:
            # For most rules, trigger when value exceeds threshold
            threshold_exceeded = current_value >= rule.threshold
        
        if threshold_exceeded:
            await self._trigger_safeguard_action(rule, current_value)
    
    async def _get_current_value(self, rule: SafeguardRule) -> Optional[float]:
        """Get current value for safeguard rule"""
        if rule.rule_id == "daily_loss_limit":
            return abs(self.financial_metrics.daily_pnl) / self.financial_metrics.total_capital
        elif rule.rule_id == "max_drawdown_limit":
            return self.financial_metrics.current_drawdown
        elif rule.rule_id == "position_size_limit":
            return self.financial_metrics.risk_exposure
        elif rule.rule_id == "leverage_limit":
            return self.financial_metrics.leverage_ratio
        elif rule.rule_id == "liquidity_reserve":
            return self.financial_metrics.available_capital / self.financial_metrics.total_capital
        elif rule.rule_id == "memory_usage_limit":
            return self.system_health.memory_usage
        elif rule.rule_id == "cpu_usage_limit":
            return self.system_health.cpu_usage
        elif rule.rule_id == "error_rate_limit":
            return self.system_health.error_rate
        elif rule.rule_id == "network_latency_limit":
            return self.system_health.network_latency
        elif rule.rule_id == "uptime_limit":
            return self.system_health.uptime
        
        return None
    
    async def _trigger_safeguard_action(self, rule: SafeguardRule, current_value: float):
        """Trigger safeguard action"""
        rule.trigger_count += 1
        rule.last_triggered = datetime.now()
        
        self.logger.warning(
            f"Safeguard triggered: {rule.name} "
            f"(Current: {current_value:.3f}, Threshold: {rule.threshold:.3f}, "
            f"Action: {rule.action})"
        )
        
        # Execute action based on rule
        if rule.action == "halt_trading":
            await self._halt_trading(rule.level)
        elif rule.action == "emergency_stop":
            await self._emergency_stop(rule.level)
        elif rule.action == "reject_trade":
            await self._enable_trade_rejection(rule.rule_id)
        elif rule.action == "reduce_positions":
            await self._reduce_positions(rule.level)
        elif rule.action == "halt_new_positions":
            await self._halt_new_positions(rule.level)
        elif rule.action == "force_gc_and_alert":
            await self._force_garbage_collection()
        elif rule.action == "throttle_operations":
            await self._throttle_operations(rule.level)
        elif rule.action == "enable_circuit_breaker":
            await self._enable_circuit_breaker(rule.rule_id)
        elif rule.action == "switch_to_backup_endpoints":
            await self._switch_to_backup_endpoints()
        elif rule.action == "schedule_restart":
            await self._schedule_restart(rule.level)
        elif rule.action == "throttle_trading":
            await self._throttle_trading(rule.level)
        
        # Execute emergency callbacks if critical
        if rule.level in [SafeguardLevel.CRITICAL, SafeguardLevel.EMERGENCY]:
            await self._execute_emergency_callbacks(rule, current_value)
    
    async def _halt_trading(self, level: SafeguardLevel):
        """Halt all trading operations"""
        self.circuit_breakers["trading_halted"] = True
        self.logger.critical(f"TRADING HALTED - Level: {level.value}")
        
        # Notify external systems
        await self._send_alert({
            "type": "trading_halt",
            "level": level.value,
            "timestamp": datetime.now().isoformat(),
            "message": "Trading operations have been halted due to safeguard trigger"
        })
    
    async def _emergency_stop(self, level: SafeguardLevel):
        """Emergency stop all operations"""
        self.emergency_mode = True
        self.circuit_breakers["emergency_stop"] = True
        
        self.logger.critical(f"EMERGENCY STOP ACTIVATED - Level: {level.value}")
        
        # Attempt to liquidate positions safely
        await self._emergency_liquidation()
        
        # Notify all stakeholders
        await self._send_critical_alert({
            "type": "emergency_stop",
            "level": level.value,
            "timestamp": datetime.now().isoformat(),
            "message": "EMERGENCY STOP: All operations halted due to critical safeguard trigger"
        })
        
        # Schedule system shutdown
        self.shutdown_requested = True
    
    async def _enable_trade_rejection(self, rule_id: str):
        """Enable trade rejection for specific rule"""
        self.circuit_breakers[f"reject_trades_{rule_id}"] = True
        self.logger.warning(f"Trade rejection enabled for rule: {rule_id}")
    
    async def _reduce_positions(self, level: SafeguardLevel):
        """Reduce position sizes to manage risk"""
        self.circuit_breakers["position_reduction"] = True
        self.logger.warning(f"Position reduction triggered - Level: {level.value}")
        
        # Would implement actual position reduction logic
        # This is a placeholder for the interface
    
    async def _halt_new_positions(self, level: SafeguardLevel):
        """Halt opening new positions"""
        self.circuit_breakers["new_positions_halted"] = True
        self.logger.warning(f"New positions halted - Level: {level.value}")
    
    async def _force_garbage_collection(self):
        """Force garbage collection to free memory"""
        import gc
        collected = gc.collect()
        self.logger.info(f"Forced garbage collection: {collected} objects collected")
    
    async def _throttle_operations(self, level: SafeguardLevel):
        """Throttle system operations"""
        self.circuit_breakers["operations_throttled"] = True
        self.logger.warning(f"Operations throttling enabled - Level: {level.value}")
    
    async def _enable_circuit_breaker(self, rule_id: str):
        """Enable circuit breaker for high error rates"""
        self.circuit_breakers[f"circuit_breaker_{rule_id}"] = True
        self.logger.warning(f"Circuit breaker enabled for: {rule_id}")
    
    async def _switch_to_backup_endpoints(self):
        """Switch to backup network endpoints"""
        self.logger.warning("Switching to backup network endpoints")
        # Would implement actual endpoint switching logic
    
    async def _schedule_restart(self, level: SafeguardLevel):
        """Schedule system restart"""
        self.logger.warning(f"System restart scheduled - Level: {level.value}")
        # Would implement graceful restart scheduling
    
    async def _throttle_trading(self, level: SafeguardLevel):
        """Throttle trading frequency"""
        self.circuit_breakers["trading_throttled"] = True
        self.logger.warning(f"Trading throttling enabled - Level: {level.value}")
    
    async def _emergency_liquidation(self):
        """Perform emergency liquidation of positions"""
        self.logger.critical("EMERGENCY LIQUIDATION INITIATED")
        
        # This would implement actual emergency liquidation logic
        # Key considerations:
        # 1. Prioritize liquid positions first
        # 2. Use market orders for speed
        # 3. Accept slippage to ensure execution
        # 4. Log all liquidation attempts
        
        await self._send_critical_alert({
            "type": "emergency_liquidation",
            "timestamp": datetime.now().isoformat(),
            "message": "Emergency liquidation of positions initiated"
        })
    
    async def _execute_emergency_callbacks(self, rule: SafeguardRule, current_value: float):
        """Execute emergency callback functions"""
        for callback in self.emergency_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(rule, current_value)
                else:
                    callback(rule, current_value)
            except Exception as e:
                self.logger.error(f"Error executing emergency callback: {e}")
    
    async def _send_alert(self, alert_data: Dict[str, Any]):
        """Send standard alert"""
        self.logger.warning(f"ALERT: {json.dumps(alert_data, indent=2)}")
        
        # Would implement actual alerting system (email, Slack, etc.)
        # For now, just log the alert
    
    async def _send_critical_alert(self, alert_data: Dict[str, Any]):
        """Send critical alert through all channels"""
        self.logger.critical(f"CRITICAL ALERT: {json.dumps(alert_data, indent=2)}")
        
        # Would implement critical alerting (SMS, phone calls, etc.)
        # For now, just log as critical
    
    # Public interface methods
    
    def update_financial_metrics(self, metrics: Dict[str, float]):
        """Update financial metrics for safeguard monitoring"""
        if "total_capital" in metrics:
            self.financial_metrics.total_capital = metrics["total_capital"]
        if "available_capital" in metrics:
            self.financial_metrics.available_capital = metrics["available_capital"]
        if "daily_pnl" in metrics:
            self.financial_metrics.daily_pnl = metrics["daily_pnl"]
        if "total_pnl" in metrics:
            self.financial_metrics.total_pnl = metrics["total_pnl"]
        if "max_drawdown" in metrics:
            self.financial_metrics.max_drawdown = metrics["max_drawdown"]
        if "current_drawdown" in metrics:
            self.financial_metrics.current_drawdown = metrics["current_drawdown"]
        if "risk_exposure" in metrics:
            self.financial_metrics.risk_exposure = metrics["risk_exposure"]
        if "leverage_ratio" in metrics:
            self.financial_metrics.leverage_ratio = metrics["leverage_ratio"]
        
        self.financial_metrics.last_update = datetime.now()
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed"""
        return not any([
            self.circuit_breakers.get("trading_halted", False),
            self.circuit_breakers.get("emergency_stop", False),
            self.emergency_mode
        ])
    
    def is_new_position_allowed(self) -> bool:
        """Check if new positions are allowed"""
        return not any([
            self.circuit_breakers.get("new_positions_halted", False),
            self.circuit_breakers.get("trading_halted", False),
            self.circuit_breakers.get("emergency_stop", False),
            self.emergency_mode
        ])
    
    def should_reject_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Check if a specific trade should be rejected"""
        # Check general circuit breakers
        if not self.is_trading_allowed():
            return True
        
        # Check position size limits
        if self.circuit_breakers.get("reject_trades_position_size_limit", False):
            position_size = trade_data.get("position_size", 0)
            if position_size > self.safeguard_rules["position_size_limit"].threshold:
                return True
        
        return False
    
    def register_emergency_callback(self, callback: Callable):
        """Register callback for emergency situations"""
        self.emergency_callbacks.append(callback)
    
    def get_safeguard_status(self) -> Dict[str, Any]:
        """Get comprehensive safeguard status"""
        return {
            "is_initialized": self.is_initialized,
            "emergency_mode": self.emergency_mode,
            "circuit_breakers": self.circuit_breakers.copy(),
            "active_rules": len([r for r in self.safeguard_rules.values() if r.enabled]),
            "triggered_rules": len([r for r in self.safeguard_rules.values() if r.trigger_count > 0]),
            "system_health": {
                "cpu_usage": self.system_health.cpu_usage,
                "memory_usage": self.system_health.memory_usage,
                "disk_usage": self.system_health.disk_usage,
                "uptime": self.system_health.uptime,
                "last_update": self.system_health.last_update.isoformat()
            },
            "financial_health": {
                "total_capital": self.financial_metrics.total_capital,
                "current_drawdown": self.financial_metrics.current_drawdown,
                "daily_pnl": self.financial_metrics.daily_pnl,
                "risk_exposure": self.financial_metrics.risk_exposure,
                "last_update": self.financial_metrics.last_update.isoformat()
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown of safeguards"""
        self.logger.info("Shutting down production safeguards")
        self.shutdown_requested = True
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Production safeguards shutdown complete") 
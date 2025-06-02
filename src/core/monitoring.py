"""
Production-Ready Monitoring and Alerting System

This module provides comprehensive monitoring, metrics collection, alerting,
and observability features for the trading bot.
"""

import asyncio
import logging
import time
import threading
import json
import os
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, deque
import psutil

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Alert:
    """Alert definition"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    component: str
    timestamp: float
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class Metric:
    """Metric data point"""
    name: str
    value: float
    timestamp: float
    type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None

@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    component: str
    check_function: Callable
    interval: float = 60.0
    timeout: float = 10.0
    enabled: bool = True
    critical: bool = False

class MetricsCollector:
    """
    Comprehensive metrics collection system
    
    Features:
    - Counter, Gauge, Histogram, and Timer metrics
    - Time-series storage with configurable retention
    - Statistical aggregations
    - Memory-efficient circular buffers
    """
    
    def __init__(self, retention_minutes: int = 1440):  # 24 hours default
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=retention_minutes))
        self.retention_minutes = retention_minutes
        self._lock = threading.RLock()
        
        # Built-in system metrics
        self._start_time = time.time()
        self._setup_system_metrics()
    
    def _setup_system_metrics(self):
        """Setup automatic system metrics collection"""
        # Start background thread for system metrics
        self._system_metrics_thread = threading.Thread(
            target=self._collect_system_metrics, 
            daemon=True
        )
        self._system_metrics_thread.start()
    
    def _collect_system_metrics(self):
        """Collect system metrics in background"""
        while True:
            try:
                current_time = time.time()
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_gauge("system.cpu_usage", cpu_percent, {"unit": "percent"})
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.record_gauge("system.memory_usage", memory.percent, {"unit": "percent"})
                self.record_gauge("system.memory_available", memory.available / (1024**3), {"unit": "GB"})
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.record_gauge("system.disk_usage", disk.percent, {"unit": "percent"})
                
                # Network I/O
                net_io = psutil.net_io_counters()
                self.record_counter("system.network_bytes_sent", net_io.bytes_sent, {"unit": "bytes"})
                self.record_counter("system.network_bytes_recv", net_io.bytes_recv, {"unit": "bytes"})
                
                # Process-specific metrics
                process = psutil.Process()
                self.record_gauge("process.memory_usage", process.memory_info().rss / (1024**2), {"unit": "MB"})
                self.record_gauge("process.cpu_percent", process.cpu_percent(), {"unit": "percent"})
                self.record_gauge("process.open_files", len(process.open_files()), {"unit": "count"})
                
                # Application uptime
                uptime = current_time - self._start_time
                self.record_gauge("app.uptime", uptime, {"unit": "seconds"})
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            time.sleep(60)  # Collect every minute
    
    def record_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Record a counter metric (monotonically increasing)"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            type=MetricType.COUNTER,
            tags=tags or {}
        )
        
        with self._lock:
            self.metrics[name].append(metric)
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric (can go up or down)"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            type=MetricType.GAUGE,
            tags=tags or {}
        )
        
        with self._lock:
            self.metrics[name].append(metric)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric (for distributions)"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            type=MetricType.HISTOGRAM,
            tags=tags or {}
        )
        
        with self._lock:
            self.metrics[name].append(metric)
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric"""
        metric = Metric(
            name=name,
            value=duration,
            timestamp=time.time(),
            type=MetricType.TIMER,
            tags=tags or {},
            unit="seconds"
        )
        
        with self._lock:
            self.metrics[name].append(metric)
    
    def get_metric_stats(self, name: str, minutes: int = 60) -> Dict[str, Any]:
        """Get statistical summary of a metric over time window"""
        with self._lock:
            if name not in self.metrics:
                return {"error": f"Metric {name} not found"}
            
            current_time = time.time()
            cutoff_time = current_time - (minutes * 60)
            
            # Filter metrics within time window
            recent_metrics = [
                m for m in self.metrics[name] 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {"error": f"No data for {name} in last {minutes} minutes"}
            
            values = [m.value for m in recent_metrics]
            
            return {
                "metric": name,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": statistics.mean(values),
                "median": statistics.median(values),
                "stddev": statistics.stdev(values) if len(values) > 1 else 0,
                "percentile_95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                "first_value": values[0],
                "last_value": values[-1],
                "time_window_minutes": minutes,
                "timestamp": current_time
            }
    
    def get_all_metrics(self, minutes: int = 5) -> Dict[str, Dict[str, Any]]:
        """Get all metrics data"""
        with self._lock:
            result = {}
            for name in self.metrics.keys():
                result[name] = self.get_metric_stats(name, minutes)
            return result

class HealthMonitor:
    """
    Health monitoring system for components and services
    """
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Setup default health checks
        self._setup_default_health_checks()
    
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        # System health check
        self.register_health_check(
            "system_resources",
            "system",
            self._check_system_resources,
            interval=30.0,
            critical=True
        )
        
        # Database connectivity check (placeholder)
        self.register_health_check(
            "database_connection",
            "database",
            self._check_database_connection,
            interval=60.0,
            critical=True
        )
        
        # External API checks
        self.register_health_check(
            "solana_rpc",
            "network",
            self._check_solana_rpc,
            interval=30.0,
            critical=False
        )
    
    def register_health_check(self, name: str, component: str, check_function: Callable,
                            interval: float = 60.0, timeout: float = 10.0, 
                            enabled: bool = True, critical: bool = False):
        """Register a health check"""
        health_check = HealthCheck(
            name=name,
            component=component,
            check_function=check_function,
            interval=interval,
            timeout=timeout,
            enabled=enabled,
            critical=critical
        )
        
        with self._lock:
            self.health_checks[name] = health_check
            logger.info(f"Registered health check: {name}")
    
    def start_monitoring(self):
        """Start health monitoring in background"""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                current_time = time.time()
                
                for name, check in self.health_checks.items():
                    if not check.enabled:
                        continue
                    
                    # Check if it's time to run this health check
                    last_check = self.health_status.get(name, {}).get('last_check', 0)
                    if current_time - last_check < check.interval:
                        continue
                    
                    # Run health check
                    asyncio.create_task(self._run_health_check(name, check))
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
            
            time.sleep(5)  # Check every 5 seconds
    
    async def _run_health_check(self, name: str, check: HealthCheck):
        """Run a single health check"""
        try:
            start_time = time.time()
            
            # Run check with timeout
            if asyncio.iscoroutinefunction(check.check_function):
                result = await asyncio.wait_for(
                    check.check_function(), 
                    timeout=check.timeout
                )
            else:
                result = check.check_function()
            
            duration = time.time() - start_time
            
            # Update status
            with self._lock:
                self.health_status[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "last_check": time.time(),
                    "duration": duration,
                    "result": result,
                    "error": None,
                    "component": check.component,
                    "critical": check.critical
                }
            
        except asyncio.TimeoutError:
            with self._lock:
                self.health_status[name] = {
                    "status": "timeout",
                    "last_check": time.time(),
                    "duration": check.timeout,
                    "result": False,
                    "error": "Health check timed out",
                    "component": check.component,
                    "critical": check.critical
                }
        except Exception as e:
            with self._lock:
                self.health_status[name] = {
                    "status": "error",
                    "last_check": time.time(),
                    "duration": time.time() - start_time,
                    "result": False,
                    "error": str(e),
                    "component": check.component,
                    "critical": check.critical
                }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        with self._lock:
            healthy_count = sum(1 for status in self.health_status.values() if status["status"] == "healthy")
            total_count = len(self.health_status)
            
            critical_issues = [
                name for name, status in self.health_status.items()
                if status["critical"] and status["status"] != "healthy"
            ]
            
            overall_status = "healthy"
            if critical_issues:
                overall_status = "critical"
            elif healthy_count < total_count:
                overall_status = "degraded"
            
            return {
                "overall_status": overall_status,
                "healthy_checks": healthy_count,
                "total_checks": total_count,
                "critical_issues": critical_issues,
                "timestamp": time.time(),
                "details": self.health_status.copy()
            }
    
    def _check_system_resources(self) -> bool:
        """Check if system resources are within acceptable limits"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return False
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _check_database_connection(self) -> bool:
        """Check database connectivity (placeholder)"""
        # This would typically test actual database connection
        # For now, just return True as placeholder
        return True
    
    def _check_solana_rpc(self) -> bool:
        """Check Solana RPC connectivity (placeholder)"""
        # This would typically test actual RPC connection
        # For now, just return True as placeholder
        return True

class AlertManager:
    """
    Comprehensive alerting system
    """
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_channels: List[Callable] = []
        self._lock = threading.RLock()
        
        # Setup default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alerting rules"""
        # High CPU usage
        self.add_alert_rule(
            "high_cpu_usage",
            "system.cpu_usage",
            threshold=80.0,
            comparison="greater_than",
            severity=AlertSeverity.WARNING,
            component="system"
        )
        
        # Critical CPU usage
        self.add_alert_rule(
            "critical_cpu_usage",
            "system.cpu_usage",
            threshold=95.0,
            comparison="greater_than",
            severity=AlertSeverity.CRITICAL,
            component="system"
        )
        
        # High memory usage
        self.add_alert_rule(
            "high_memory_usage",
            "system.memory_usage",
            threshold=85.0,
            comparison="greater_than",
            severity=AlertSeverity.WARNING,
            component="system"
        )
        
        # Trading errors
        self.add_alert_rule(
            "trading_error_rate",
            "trading.error_rate",
            threshold=0.1,  # 10% error rate
            comparison="greater_than",
            severity=AlertSeverity.ERROR,
            component="trading"
        )
    
    def add_alert_rule(self, name: str, metric_name: str, threshold: float,
                      comparison: str = "greater_than", severity: AlertSeverity = AlertSeverity.WARNING,
                      component: str = "unknown", window_minutes: int = 5):
        """Add an alerting rule"""
        rule = {
            "name": name,
            "metric_name": metric_name,
            "threshold": threshold,
            "comparison": comparison,
            "severity": severity,
            "component": component,
            "window_minutes": window_minutes,
            "enabled": True
        }
        
        with self._lock:
            self.alert_rules.append(rule)
            logger.info(f"Added alert rule: {name}")
    
    def add_notification_channel(self, channel: Callable):
        """Add a notification channel (function to call when alert is triggered)"""
        self.notification_channels.append(channel)
    
    def evaluate_rules(self, metrics_collector: MetricsCollector):
        """Evaluate all alert rules against current metrics"""
        current_time = time.time()
        
        for rule in self.alert_rules:
            if not rule["enabled"]:
                continue
            
            try:
                # Get metric stats for the time window
                stats = metrics_collector.get_metric_stats(
                    rule["metric_name"], 
                    rule["window_minutes"]
                )
                
                if "error" in stats:
                    continue
                
                # Check if threshold is breached
                metric_value = stats["avg"]  # Use average as default
                threshold_breached = self._check_threshold(
                    metric_value, 
                    rule["threshold"], 
                    rule["comparison"]
                )
                
                if threshold_breached:
                    # Check if we already have an active alert for this rule
                    existing_alert = self._find_active_alert(rule["name"])
                    
                    if not existing_alert:
                        # Create new alert
                        alert = Alert(
                            id=f"{rule['name']}_{int(current_time)}",
                            title=f"{rule['name']} threshold breached",
                            description=f"Metric {rule['metric_name']} value {metric_value:.2f} {rule['comparison']} threshold {rule['threshold']}",
                            severity=rule["severity"],
                            component=rule["component"],
                            timestamp=current_time,
                            metric_name=rule["metric_name"],
                            metric_value=metric_value,
                            threshold=rule["threshold"]
                        )
                        
                        self._trigger_alert(alert)
                
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule['name']}: {e}")
    
    def _check_threshold(self, value: float, threshold: float, comparison: str) -> bool:
        """Check if value breaches threshold based on comparison type"""
        if comparison == "greater_than":
            return value > threshold
        elif comparison == "less_than":
            return value < threshold
        elif comparison == "equal":
            return abs(value - threshold) < 0.01
        elif comparison == "not_equal":
            return abs(value - threshold) >= 0.01
        else:
            return False
    
    def _find_active_alert(self, rule_name: str) -> Optional[Alert]:
        """Find an active alert for a given rule"""
        with self._lock:
            for alert in self.alerts:
                if (alert.title.startswith(rule_name) and 
                    not alert.resolved and 
                    time.time() - alert.timestamp < 3600):  # Within last hour
                    return alert
        return None
    
    def _trigger_alert(self, alert: Alert):
        """Trigger an alert and send notifications"""
        with self._lock:
            self.alerts.append(alert)
        
        logger.warning(f"Alert triggered: {alert.title} - {alert.description}")
        
        # Send notifications
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logger.error(f"Error sending alert notification: {e}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"Alert acknowledged: {alert_id}")
                    return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    logger.info(f"Alert resolved: {alert_id}")
                    return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        with self._lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        with self._lock:
            active_alerts = self.get_active_alerts()
            
            severity_counts = defaultdict(int)
            component_counts = defaultdict(int)
            
            for alert in active_alerts:
                severity_counts[alert.severity.value] += 1
                component_counts[alert.component] += 1
            
            return {
                "total_active_alerts": len(active_alerts),
                "total_alerts": len(self.alerts),
                "severity_breakdown": dict(severity_counts),
                "component_breakdown": dict(component_counts),
                "timestamp": time.time()
            }

class ProductionMonitor:
    """
    Complete production monitoring system combining metrics, health checks, and alerts
    """
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.health_monitor = HealthMonitor()
        self.alert_manager = AlertManager()
        
        # Setup alert monitoring
        self._setup_alert_monitoring()
        
        # Setup default notification channels
        self._setup_notification_channels()
    
    def _setup_alert_monitoring(self):
        """Setup automatic alert rule evaluation"""
        def evaluate_alerts():
            while True:
                try:
                    self.alert_manager.evaluate_rules(self.metrics)
                except Exception as e:
                    logger.error(f"Error evaluating alerts: {e}")
                time.sleep(30)  # Evaluate every 30 seconds
        
        alert_thread = threading.Thread(target=evaluate_alerts, daemon=True)
        alert_thread.start()
    
    def _setup_notification_channels(self):
        """Setup default notification channels"""
        # Log-based notification
        def log_notification(alert: Alert):
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }[alert.severity]
            
            logger.log(log_level, f"ALERT: {alert.title} - {alert.description}")
        
        self.alert_manager.add_notification_channel(log_notification)
        
        # File-based notification for critical alerts
        def file_notification(alert: Alert):
            if alert.severity == AlertSeverity.CRITICAL:
                try:
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/critical_alerts.log", "a") as f:
                        f.write(f"{datetime.now().isoformat()} - {json.dumps(asdict(alert))}\n")
                except Exception as e:
                    logger.error(f"Failed to write critical alert to file: {e}")
        
        self.alert_manager.add_notification_channel(file_notification)
    
    def start(self):
        """Start all monitoring components"""
        self.health_monitor.start_monitoring()
        logger.info("Production monitoring started")
    
    def stop(self):
        """Stop all monitoring components"""
        self.health_monitor.stop_monitoring()
        logger.info("Production monitoring stopped")
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        return {
            "health": self.health_monitor.get_health_summary(),
            "alerts": self.alert_manager.get_alert_summary(),
            "key_metrics": {
                "cpu_usage": self.metrics.get_metric_stats("system.cpu_usage", 5),
                "memory_usage": self.metrics.get_metric_stats("system.memory_usage", 5),
                "app_uptime": self.metrics.get_metric_stats("app.uptime", 1)
            },
            "timestamp": time.time()
        }

# Global monitoring instance
production_monitor = ProductionMonitor()

# Convenience functions
def record_metric(name: str, value: float, metric_type: str = "gauge", tags: Optional[Dict[str, str]] = None):
    """Record a metric"""
    if metric_type == "counter":
        production_monitor.metrics.record_counter(name, value, tags)
    elif metric_type == "gauge":
        production_monitor.metrics.record_gauge(name, value, tags)
    elif metric_type == "histogram":
        production_monitor.metrics.record_histogram(name, value, tags)
    elif metric_type == "timer":
        production_monitor.metrics.record_timer(name, value, tags)

def get_system_health() -> Dict[str, Any]:
    """Get current system health"""
    return production_monitor.get_system_overview() 
"""
SystemMetrics - Comprehensive system performance monitoring for Ant Bot

Provides real-time metrics collection, performance analysis, resource monitoring,
and system health tracking with Prometheus integration and custom analytics.
"""

import time
import asyncio
import psutil
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
import json
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

@dataclass
class MetricSnapshot:
    """Snapshot of system metrics at a point in time"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    system_load: float
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    alert_id: str
    metric_name: str
    threshold_value: float
    comparison: str  # 'greater', 'less', 'equal'
    severity: str   # 'low', 'medium', 'high', 'critical'
    callback: Optional[Callable] = None
    is_active: bool = True
    last_triggered: float = 0.0

@dataclass
class ComponentMetrics:
    """Metrics for a specific system component"""
    component_name: str
    operations_count: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    total_processing_time: float = 0.0
    last_activity: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)

class SystemMetrics:
    """Comprehensive system metrics and monitoring"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Core metrics storage
        self.metric_history: deque = deque(maxlen=self.config.get('history_size', 1000))
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.custom_metrics: Dict[str, Any] = {}
        
        # Performance monitoring
        self.performance_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_callbacks: List[Callable] = []
        self.metrics_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # System monitoring
        self.monitoring_enabled = True
        self.monitoring_task = None
        self.monitoring_interval = self.config.get('monitoring_interval', 30)  # seconds
        
        # Performance tracking
        self.operation_timers: Dict[str, float] = {}
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Resource baselines
        self.baseline_metrics = {}
        self.anomaly_detection_enabled = self.config.get('anomaly_detection', True)
        
        # Thread safety
        self.metrics_lock = threading.RLock()
        
        self._initialized = False
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default metrics configuration"""
        return {
            "monitoring_interval": 30,
            "history_size": 1000,
            "prometheus_port": 8001,
            "enable_prometheus": True,
            "anomaly_detection": True,
            "resource_thresholds": {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "disk_usage": 90.0
            },
            "performance_thresholds": {
                "response_time": 5.0,
                "error_rate": 0.05
            }
        }
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # System metrics
        self.prom_cpu_usage = Gauge('ant_bot_cpu_usage_percent', 'CPU usage percentage')
        self.prom_memory_usage = Gauge('ant_bot_memory_usage_percent', 'Memory usage percentage')
        self.prom_disk_usage = Gauge('ant_bot_disk_usage_percent', 'Disk usage percentage')
        self.prom_network_sent = Gauge('ant_bot_network_bytes_sent', 'Network bytes sent')
        self.prom_network_recv = Gauge('ant_bot_network_bytes_recv', 'Network bytes received')
        
        # Application metrics
        self.prom_operations_total = Counter('ant_bot_operations_total', 'Total operations', ['component', 'operation'])
        self.prom_operation_duration = Histogram('ant_bot_operation_duration_seconds', 'Operation duration', ['component', 'operation'])
        self.prom_errors_total = Counter('ant_bot_errors_total', 'Total errors', ['component', 'error_type'])
        self.prom_active_components = Gauge('ant_bot_active_components', 'Number of active components')
        
        # Trading metrics
        self.prom_trades_total = Counter('ant_bot_trades_total', 'Total trades executed', ['side', 'status'])
        self.prom_capital_amount = Gauge('ant_bot_capital_amount', 'Current capital amount')
        self.prom_profit_loss = Gauge('ant_bot_profit_loss', 'Current profit/loss')
        
        # Compounding metrics
        self.prom_compound_rate = Gauge('ant_bot_compound_rate', 'Current compound rate', ['layer'])
        self.prom_efficiency_score = Gauge('ant_bot_efficiency_score', 'System efficiency score', ['component'])
        
        # Worker metrics
        self.prom_active_workers = Gauge('ant_bot_active_workers', 'Number of active worker ants')
        self.prom_worker_efficiency = Gauge('ant_bot_worker_efficiency', 'Worker efficiency score')
    
    async def initialize(self) -> bool:
        """Initialize the metrics system"""
        try:
            # Start Prometheus server if enabled
            if self.config.get('enable_prometheus', True):
                await self._start_prometheus_server()
            
            # Initialize baseline metrics
            await self._collect_baseline_metrics()
            
            # Setup default alerts
            await self._setup_default_alerts()
            
            # Start monitoring
            await self._start_monitoring()
            
            self._initialized = True
            logger.info("SystemMetrics initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SystemMetrics: {e}")
            return False
    
    async def _start_prometheus_server(self):
        """Start Prometheus metrics server"""
        try:
            port = self.config.get('prometheus_port', 8001)
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.warning(f"Could not start Prometheus server: {e}")
    
    async def _collect_baseline_metrics(self):
        """Collect baseline system metrics for comparison"""
        try:
            snapshot = await self._collect_system_snapshot()
            self.baseline_metrics = {
                'cpu_usage': snapshot.cpu_usage,
                'memory_usage': snapshot.memory_usage,
                'disk_usage': snapshot.disk_usage,
                'timestamp': snapshot.timestamp
            }
            logger.info("Baseline metrics collected")
        except Exception as e:
            logger.error(f"Error collecting baseline metrics: {e}")
    
    async def _setup_default_alerts(self):
        """Setup default system alerts"""
        thresholds = self.config.get('resource_thresholds', {})
        
        # CPU usage alert
        await self.add_alert(
            alert_id="high_cpu_usage",
            metric_name="cpu_usage",
            threshold_value=thresholds.get('cpu_usage', 80.0),
            comparison="greater",
            severity="high"
        )
        
        # Memory usage alert
        await self.add_alert(
            alert_id="high_memory_usage",
            metric_name="memory_usage",
            threshold_value=thresholds.get('memory_usage', 85.0),
            comparison="greater",
            severity="high"
        )
        
        # Disk usage alert
        await self.add_alert(
            alert_id="high_disk_usage",
            metric_name="disk_usage",
            threshold_value=thresholds.get('disk_usage', 90.0),
            comparison="greater",
            severity="critical"
        )
    
    async def _start_monitoring(self):
        """Start background monitoring task"""
        self.monitoring_enabled = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_enabled:
            try:
                # Collect system metrics
                snapshot = await self._collect_system_snapshot()
                
                with self.metrics_lock:
                    self.metric_history.append(snapshot)
                    
                # Update Prometheus metrics
                await self._update_prometheus_metrics(snapshot)
                
                # Check alerts
                await self._check_alerts(snapshot)
                
                # Detect anomalies if enabled
                if self.anomaly_detection_enabled:
                    await self._detect_anomalies(snapshot)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _collect_system_snapshot(self) -> MetricSnapshot:
        """Collect comprehensive system metrics snapshot"""
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network metrics
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # System load
            try:
                system_load = psutil.getloadavg()[0]  # 1-minute load average
            except AttributeError:
                # Windows doesn't have getloadavg
                system_load = cpu_usage / 100.0
            
            # Active connections (simplified)
            try:
                connections = len(psutil.net_connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                connections = 0
            
            return MetricSnapshot(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                active_connections=connections,
                system_load=system_load,
                custom_metrics=self.custom_metrics.copy()
            )
            
        except Exception as e:
            logger.error(f"Error collecting system snapshot: {e}")
            # Return minimal snapshot on error
            return MetricSnapshot(
                timestamp=time.time(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                active_connections=0,
                system_load=0.0
            )
    
    async def _update_prometheus_metrics(self, snapshot: MetricSnapshot):
        """Update Prometheus metrics with snapshot data"""
        try:
            self.prom_cpu_usage.set(snapshot.cpu_usage)
            self.prom_memory_usage.set(snapshot.memory_usage)
            self.prom_disk_usage.set(snapshot.disk_usage)
            
            if 'bytes_sent' in snapshot.network_io:
                self.prom_network_sent.set(snapshot.network_io['bytes_sent'])
            if 'bytes_recv' in snapshot.network_io:
                self.prom_network_recv.set(snapshot.network_io['bytes_recv'])
            
            self.prom_active_components.set(len(self.component_metrics))
            
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    async def record_operation(
        self, 
        component: str, 
        operation: str, 
        duration: float = None,
        success: bool = True,
        **kwargs
    ):
        """Record an operation for a component"""
        try:
            with self.metrics_lock:
                # Initialize component metrics if not exists
                if component not in self.component_metrics:
                    self.component_metrics[component] = ComponentMetrics(component_name=component)
                
                comp_metrics = self.component_metrics[component]
                
                # Update operation counts
                comp_metrics.operations_count += 1
                comp_metrics.last_activity = time.time()
                
                if success:
                    comp_metrics.success_count += 1
                else:
                    comp_metrics.error_count += 1
                
                # Update response times
                if duration is not None:
                    comp_metrics.total_processing_time += duration
                    comp_metrics.avg_response_time = (
                        comp_metrics.total_processing_time / comp_metrics.operations_count
                    )
                    
                    # Store for trend analysis
                    self.response_times[f"{component}_{operation}"].append(duration)
                
                # Update Prometheus metrics
                self.prom_operations_total.labels(component=component, operation=operation).inc()
                if duration is not None:
                    self.prom_operation_duration.labels(component=component, operation=operation).observe(duration)
                if not success:
                    self.prom_errors_total.labels(component=component, error_type="operation_failure").inc()
            
        except Exception as e:
            logger.error(f"Error recording operation: {e}")
    
    async def record_trade(
        self,
        side: str,
        status: str,
        amount: float = None,
        profit_loss: float = None
    ):
        """Record trading metrics"""
        try:
            self.prom_trades_total.labels(side=side, status=status).inc()
            
            if amount is not None:
                current_capital = self.get_custom_metric('capital_amount', 0.0)
                if side == 'buy':
                    new_capital = current_capital - amount
                else:
                    new_capital = current_capital + amount
                self.set_custom_metric('capital_amount', new_capital)
                self.prom_capital_amount.set(new_capital)
            
            if profit_loss is not None:
                current_pl = self.get_custom_metric('total_profit_loss', 0.0)
                new_pl = current_pl + profit_loss
                self.set_custom_metric('total_profit_loss', new_pl)
                self.prom_profit_loss.set(new_pl)
                
        except Exception as e:
            logger.error(f"Error recording trade metrics: {e}")
    
    async def record_compounding_metrics(
        self,
        layer: str,
        compound_rate: float,
        efficiency_score: float = None
    ):
        """Record compounding layer metrics"""
        try:
            self.prom_compound_rate.labels(layer=layer).set(compound_rate)
            
            if efficiency_score is not None:
                self.prom_efficiency_score.labels(component=layer).set(efficiency_score)
                
        except Exception as e:
            logger.error(f"Error recording compounding metrics: {e}")
    
    async def record_worker_metrics(
        self,
        active_workers: int,
        efficiency_score: float
    ):
        """Record worker ant metrics"""
        try:
            self.prom_active_workers.set(active_workers)
            self.prom_worker_efficiency.set(efficiency_score)
            
        except Exception as e:
            logger.error(f"Error recording worker metrics: {e}")
    
    def start_operation_timer(self, operation_id: str) -> str:
        """Start timing an operation"""
        timer_id = f"{operation_id}_{int(time.time() * 1000)}"
        self.operation_timers[timer_id] = time.time()
        return timer_id
    
    def end_operation_timer(self, timer_id: str) -> float:
        """End timing an operation and return duration"""
        if timer_id in self.operation_timers:
            duration = time.time() - self.operation_timers[timer_id]
            del self.operation_timers[timer_id]
            return duration
        return 0.0
    
    def set_custom_metric(self, name: str, value: Any):
        """Set a custom metric value"""
        with self.metrics_lock:
            self.custom_metrics[name] = {
                'value': value,
                'timestamp': time.time()
            }
    
    def get_custom_metric(self, name: str, default: Any = None) -> Any:
        """Get a custom metric value"""
        with self.metrics_lock:
            metric = self.custom_metrics.get(name)
            if metric:
                return metric['value']
            return default
    
    async def add_alert(
        self,
        alert_id: str,
        metric_name: str,
        threshold_value: float,
        comparison: str = "greater",
        severity: str = "medium",
        callback: Optional[Callable] = None
    ) -> bool:
        """Add a performance alert"""
        try:
            alert = PerformanceAlert(
                alert_id=alert_id,
                metric_name=metric_name,
                threshold_value=threshold_value,
                comparison=comparison,
                severity=severity,
                callback=callback
            )
            
            self.performance_alerts[alert_id] = alert
            logger.info(f"Added alert: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding alert {alert_id}: {e}")
            return False
    
    async def _check_alerts(self, snapshot: MetricSnapshot):
        """Check all alerts against current metrics"""
        for alert_id, alert in self.performance_alerts.items():
            if not alert.is_active:
                continue
            
            try:
                # Get metric value
                metric_value = getattr(snapshot, alert.metric_name, None)
                if metric_value is None:
                    continue
                
                # Check threshold
                triggered = False
                if alert.comparison == "greater" and metric_value > alert.threshold_value:
                    triggered = True
                elif alert.comparison == "less" and metric_value < alert.threshold_value:
                    triggered = True
                elif alert.comparison == "equal" and metric_value == alert.threshold_value:
                    triggered = True
                
                if triggered:
                    # Avoid spam - only trigger once per minute
                    if time.time() - alert.last_triggered > 60:
                        await self._trigger_alert(alert, metric_value)
                        alert.last_triggered = time.time()
                        
            except Exception as e:
                logger.error(f"Error checking alert {alert_id}: {e}")
    
    async def _trigger_alert(self, alert: PerformanceAlert, current_value: float):
        """Trigger an alert"""
        alert_data = {
            'alert_id': alert.alert_id,
            'metric_name': alert.metric_name,
            'threshold_value': alert.threshold_value,
            'current_value': current_value,
            'severity': alert.severity,
            'timestamp': time.time()
        }
        
        logger.warning(f"Alert triggered: {alert.alert_id} - {alert.metric_name} = {current_value}")
        
        # Call custom callback if provided
        if alert.callback:
            try:
                await alert.callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # Call global alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                logger.error(f"Error in global alert callback: {e}")
    
    async def _detect_anomalies(self, snapshot: MetricSnapshot):
        """Detect anomalies in system metrics"""
        if not self.baseline_metrics or len(self.metric_history) < 10:
            return
        
        try:
            # Simple anomaly detection based on deviation from baseline
            cpu_deviation = abs(snapshot.cpu_usage - self.baseline_metrics['cpu_usage'])
            memory_deviation = abs(snapshot.memory_usage - self.baseline_metrics['memory_usage'])
            
            # Alert if metrics deviate significantly from baseline
            if cpu_deviation > 50:  # 50% deviation
                logger.warning(f"CPU usage anomaly detected: {snapshot.cpu_usage}% (baseline: {self.baseline_metrics['cpu_usage']}%)")
            
            if memory_deviation > 30:  # 30% deviation
                logger.warning(f"Memory usage anomaly detected: {snapshot.memory_usage}% (baseline: {self.baseline_metrics['memory_usage']}%)")
                
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add global alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        if not self.metric_history:
            return {"status": "no_data", "components": {}}
        
        latest = self.metric_history[-1]
        
        # Determine overall health status
        status = "healthy"
        if latest.cpu_usage > 90 or latest.memory_usage > 95:
            status = "critical"
        elif latest.cpu_usage > 70 or latest.memory_usage > 80:
            status = "warning"
        
        return {
            "status": status,
            "timestamp": latest.timestamp,
            "system_metrics": {
                "cpu_usage": latest.cpu_usage,
                "memory_usage": latest.memory_usage,
                "disk_usage": latest.disk_usage,
                "system_load": latest.system_load,
                "active_connections": latest.active_connections
            },
            "components": {
                name: {
                    "operations_count": metrics.operations_count,
                    "success_rate": metrics.success_count / max(1, metrics.operations_count),
                    "error_rate": metrics.error_count / max(1, metrics.operations_count),
                    "avg_response_time": metrics.avg_response_time,
                    "last_activity": metrics.last_activity
                }
                for name, metrics in self.component_metrics.items()
            },
            "active_alerts": len([a for a in self.performance_alerts.values() if a.is_active]),
            "custom_metrics_count": len(self.custom_metrics)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metric_history:
            return {}
        
        recent_snapshots = list(self.metric_history)[-10:]  # Last 10 snapshots
        
        if not recent_snapshots:
            return {}
        
        return {
            "avg_cpu_usage": sum(s.cpu_usage for s in recent_snapshots) / len(recent_snapshots),
            "avg_memory_usage": sum(s.memory_usage for s in recent_snapshots) / len(recent_snapshots),
            "avg_disk_usage": sum(s.disk_usage for s in recent_snapshots) / len(recent_snapshots),
            "total_operations": sum(m.operations_count for m in self.component_metrics.values()),
            "total_errors": sum(m.error_count for m in self.component_metrics.values()),
            "active_components": len(self.component_metrics),
            "monitoring_uptime": time.time() - (self.metric_history[0].timestamp if self.metric_history else time.time())
        }
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        try:
            return generate_latest()
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return ""
    
    async def cleanup(self):
        """Cleanup metrics system"""
        try:
            self.monitoring_enabled = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Clear data structures
            self.metric_history.clear()
            self.component_metrics.clear()
            self.custom_metrics.clear()
            self.operation_timers.clear()
            self.response_times.clear()
            
            logger.info("SystemMetrics cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during SystemMetrics cleanup: {e}") 
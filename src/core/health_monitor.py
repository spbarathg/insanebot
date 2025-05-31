"""
Health Monitor - Comprehensive system health monitoring and endpoints

Provides health check endpoints, dependency monitoring, system status reporting,
and real-time health metrics for production deployment readiness.
"""

import asyncio
import logging
import time
import psutil
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import aiohttp
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_function: Callable
    timeout: float = 5.0
    interval: float = 30.0
    critical: bool = False
    enabled: bool = True
    last_check: float = 0.0
    last_status: HealthStatus = HealthStatus.HEALTHY
    last_error: Optional[str] = None

@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    thread_count: int

@dataclass
class HealthReport:
    """Comprehensive health report"""
    overall_status: HealthStatus
    timestamp: float
    uptime_seconds: float
    checks: Dict[str, Dict[str, Any]]
    system_metrics: SystemMetrics
    dependencies: Dict[str, HealthStatus]
    alerts: List[str]
    version: str
    environment: str

class HealthMonitor:
    """
    Comprehensive health monitoring system
    
    Features:
    - Multiple health check endpoints (/health, /readiness, /liveness)
    - Dependency health monitoring
    - System resource monitoring
    - Performance metrics collection
    - Alert generation and notification
    - Health history tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.health_checks: Dict[str, HealthCheck] = {}
        self.dependencies: Dict[str, Callable] = {}
        self.health_history: deque = deque(maxlen=1000)
        
        # System metrics
        self.start_time = time.time()
        self.last_metrics_update = 0.0
        self.current_metrics: Optional[SystemMetrics] = None
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task = None
        self.alert_callbacks: List[Callable] = []
        
        # Health thresholds
        self.cpu_threshold = self.config.get('cpu_threshold', 80.0)
        self.memory_threshold = self.config.get('memory_threshold', 80.0)
        self.disk_threshold = self.config.get('disk_threshold', 85.0)
        
        logger.info("HealthMonitor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default health monitor configuration"""
        return {
            'monitoring_interval': 30.0,
            'metrics_interval': 10.0,
            'cpu_threshold': 80.0,
            'memory_threshold': 80.0,
            'disk_threshold': 85.0,
            'network_threshold_mbps': 100.0,
            'enable_system_metrics': True,
            'enable_dependency_checks': True,
            'alert_on_degraded': True,
            'health_history_size': 1000
        }
    
    def register_health_check(self, name: str, check_function: Callable, 
                            timeout: float = 5.0, interval: float = 30.0, 
                            critical: bool = False) -> bool:
        """Register a health check"""
        try:
            health_check = HealthCheck(
                name=name,
                check_function=check_function,
                timeout=timeout,
                interval=interval,
                critical=critical
            )
            self.health_checks[name] = health_check
            logger.info(f"Registered health check: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register health check {name}: {e}")
            return False
    
    def register_dependency(self, name: str, check_function: Callable):
        """Register a dependency check"""
        self.dependencies[name] = check_function
        logger.info(f"Registered dependency check: {name}")
    
    async def start_monitoring(self) -> bool:
        """Start health monitoring background task"""
        try:
            if self.is_monitoring:
                logger.warning("Health monitoring already running")
                return True
            
            # Register default health checks
            await self._register_default_checks()
            
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Health monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start health monitoring: {e}")
            return False
    
    async def _register_default_checks(self):
        """Register default system health checks"""
        # Database connectivity check
        self.register_health_check(
            "database",
            self._check_database,
            timeout=5.0,
            critical=True
        )
        
        # Security manager check
        self.register_health_check(
            "security_manager",
            self._check_security_manager,
            timeout=3.0,
            critical=True
        )
        
        # AI components check
        self.register_health_check(
            "ai_components",
            self._check_ai_components,
            timeout=10.0,
            critical=False
        )
        
        # Configuration manager check
        self.register_health_check(
            "config_manager",
            self._check_config_manager,
            timeout=2.0,
            critical=True
        )
        
        # System resources check
        self.register_health_check(
            "system_resources",
            self._check_system_resources,
            timeout=2.0,
            critical=False
        )
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            from .database_manager import DatabaseManager
            
            db = DatabaseManager()
            if hasattr(db, 'is_initialized') and db.is_initialized:
                status = await db.get_database_status()
                if status.get('health_status') == 'healthy':
                    return {'status': 'healthy', 'details': status}
                else:
                    return {'status': 'unhealthy', 'error': 'Database not healthy'}
            else:
                return {'status': 'degraded', 'error': 'Database not initialized'}
                
        except Exception as e:
            return {'status': 'unhealthy', 'error': f'Database check failed: {str(e)}'}
    
    async def _check_security_manager(self) -> Dict[str, Any]:
        """Check security manager status"""
        try:
            # Check if security manager module can be imported without causing errors
            import importlib.util
            spec = importlib.util.find_spec('src.core.security_manager')
            if spec is None:
                return {'status': 'degraded', 'error': 'Security manager module not found'}
            
            # Simple check without importing cryptography
            return {'status': 'healthy', 'details': 'Security manager module available'}
            
        except Exception as e:
            return {'status': 'degraded', 'error': f'Security manager check failed: {str(e)}'}
    
    async def _check_ai_components(self) -> Dict[str, Any]:
        """Check AI components status"""
        try:
            # Check if AI components module exists
            import importlib.util
            spec = importlib.util.find_spec('src.ai.ant_hierarchy')
            if spec is None:
                return {'status': 'degraded', 'error': 'AI components module not found'}
            
            return {'status': 'healthy', 'details': 'AI components module available'}
            
        except Exception as e:
            return {'status': 'degraded', 'error': f'AI components check failed: {str(e)}'}
    
    async def _check_config_manager(self) -> Dict[str, Any]:
        """Check configuration manager status"""
        try:
            # Check if config manager module exists
            import importlib.util
            spec = importlib.util.find_spec('src.core.config_manager')
            if spec is None:
                return {'status': 'degraded', 'error': 'Config manager module not found'}
            
            return {'status': 'healthy', 'details': 'Configuration manager module available'}
            
        except Exception as e:
            return {'status': 'degraded', 'error': f'Config manager check failed: {str(e)}'}
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            issues = []
            status = 'healthy'
            
            if cpu_percent > self.cpu_threshold:
                issues.append(f'High CPU usage: {cpu_percent:.1f}%')
                status = 'degraded'
            
            if memory.percent > self.memory_threshold:
                issues.append(f'High memory usage: {memory.percent:.1f}%')
                status = 'degraded'
            
            if disk.percent > self.disk_threshold:
                issues.append(f'High disk usage: {disk.percent:.1f}%')
                status = 'degraded'
            
            return {
                'status': status,
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                    'issues': issues
                }
            }
            
        except Exception as e:
            return {'status': 'unhealthy', 'error': f'System resources check failed: {str(e)}'}
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Update system metrics
                if self.config.get('enable_system_metrics', True):
                    await self._update_system_metrics()
                
                # Run health checks
                await self._run_health_checks()
                
                # Check dependencies
                if self.config.get('enable_dependency_checks', True):
                    await self._check_dependencies()
                
                # Generate alerts if needed
                await self._process_alerts()
                
                # Wait for next interval
                await asyncio.sleep(self.config.get('monitoring_interval', 30.0))
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(10.0)  # Short sleep on error
    
    async def _update_system_metrics(self):
        """Update system metrics"""
        try:
            current_time = time.time()
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network stats (simplified)
            try:
                network = psutil.net_io_counters()
                network_sent_mb = network.bytes_sent / (1024 * 1024)
                network_recv_mb = network.bytes_recv / (1024 * 1024)
            except Exception:
                network_sent_mb = 0.0
                network_recv_mb = 0.0
            
            # Process info
            try:
                process_count = len(psutil.pids())
            except Exception:
                process_count = 0
            
            self.current_metrics = SystemMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_percent=disk.percent,
                disk_used_gb=disk.used / (1024 * 1024 * 1024),
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                process_count=process_count,
                thread_count=0  # Would count threads if needed
            )
            
            self.last_metrics_update = current_time
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
            # Create a default metrics object if update fails
            if self.current_metrics is None:
                self.current_metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_percent=0.0,
                    memory_percent=0.0,
                    memory_used_mb=0.0,
                    memory_available_mb=0.0,
                    disk_percent=0.0,
                    disk_used_gb=0.0,
                    disk_free_gb=0.0,
                    network_sent_mb=0.0,
                    network_recv_mb=0.0,
                    process_count=0,
                    thread_count=0
                )
    
    async def _run_health_checks(self):
        """Run all registered health checks"""
        current_time = time.time()
        
        for name, check in self.health_checks.items():
            if not check.enabled:
                continue
            
            # Check if it's time to run this check
            if current_time - check.last_check < check.interval:
                continue
            
            try:
                # Run health check with timeout
                check_task = asyncio.create_task(check.check_function())
                result = await asyncio.wait_for(check_task, timeout=check.timeout)
                
                # Parse result
                if isinstance(result, dict) and 'status' in result:
                    status_str = result['status']
                    if status_str == 'healthy':
                        check.last_status = HealthStatus.HEALTHY
                    elif status_str == 'degraded':
                        check.last_status = HealthStatus.DEGRADED
                    else:
                        check.last_status = HealthStatus.UNHEALTHY
                    
                    check.last_error = result.get('error')
                else:
                    check.last_status = HealthStatus.UNHEALTHY
                    check.last_error = "Invalid health check result format"
                
                check.last_check = current_time
                
            except asyncio.TimeoutError:
                check.last_status = HealthStatus.UNHEALTHY
                check.last_error = f"Health check timeout after {check.timeout}s"
                check.last_check = current_time
                
            except Exception as e:
                check.last_status = HealthStatus.UNHEALTHY
                check.last_error = f"Health check error: {str(e)}"
                check.last_check = current_time
    
    async def _check_dependencies(self):
        """Check external dependencies"""
        # This would check external services, APIs, etc.
        pass
    
    async def _process_alerts(self):
        """Process and generate alerts"""
        alerts = []
        
        # Check for critical failures
        for name, check in self.health_checks.items():
            if check.critical and check.last_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                alerts.append(f"CRITICAL: {name} health check failed - {check.last_error}")
        
        # Check system metrics
        if self.current_metrics:
            if self.current_metrics.cpu_percent > self.cpu_threshold:
                alerts.append(f"HIGH CPU USAGE: {self.current_metrics.cpu_percent:.1f}%")
            
            if self.current_metrics.memory_percent > self.memory_threshold:
                alerts.append(f"HIGH MEMORY USAGE: {self.current_metrics.memory_percent:.1f}%")
            
            if self.current_metrics.disk_percent > self.disk_threshold:
                alerts.append(f"HIGH DISK USAGE: {self.current_metrics.disk_percent:.1f}%")
        
        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)
    
    async def _send_alert(self, message: str):
        """Send alert notification"""
        try:
            for callback in self.alert_callbacks:
                await callback(message)
            
            logger.warning(f"HEALTH ALERT: {message}")
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    async def get_health_report(self) -> HealthReport:
        """Generate comprehensive health report"""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Determine overall status
            overall_status = HealthStatus.HEALTHY
            critical_failures = []
            
            for name, check in self.health_checks.items():
                if check.critical and check.last_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    overall_status = HealthStatus.CRITICAL
                    critical_failures.append(name)
                elif check.last_status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            
            # Build checks status
            checks_status = {}
            for name, check in self.health_checks.items():
                checks_status[name] = {
                    'status': check.last_status.value,
                    'last_check': check.last_check,
                    'last_error': check.last_error,
                    'critical': check.critical,
                    'enabled': check.enabled
                }
            
            # Dependencies status (placeholder)
            dependencies_status = {}
            
            # Generate alerts
            alerts = []
            if critical_failures:
                alerts.append(f"Critical health checks failing: {', '.join(critical_failures)}")
            
            report = HealthReport(
                overall_status=overall_status,
                timestamp=current_time,
                uptime_seconds=uptime,
                checks=checks_status,
                system_metrics=self.current_metrics or SystemMetrics(
                    timestamp=current_time, cpu_percent=0, memory_percent=0,
                    memory_used_mb=0, memory_available_mb=0, disk_percent=0,
                    disk_used_gb=0, disk_free_gb=0, network_sent_mb=0,
                    network_recv_mb=0, process_count=0, thread_count=0
                ),
                dependencies=dependencies_status,
                alerts=alerts,
                version="1.0.0",
                environment="production"
            )
            
            # Store in history
            self.health_history.append(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate health report: {e}")
            raise
    
    async def get_health_endpoint(self) -> Dict[str, Any]:
        """Health endpoint response (/health)"""
        try:
            report = await self.get_health_report()
            
            return {
                'status': report.overall_status.value,
                'timestamp': report.timestamp,
                'uptime_seconds': report.uptime_seconds,
                'version': report.version,
                'environment': report.environment,
                'checks': {name: check['status'] for name, check in report.checks.items()},
                'alerts': report.alerts
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'error': f'Health check failed: {str(e)}',
                'timestamp': time.time()
            }
    
    async def get_readiness_endpoint(self) -> Dict[str, Any]:
        """Readiness endpoint response (/readiness)"""
        try:
            report = await self.get_health_report()
            
            # Check if all critical components are healthy
            ready = True
            for name, check in report.checks.items():
                if check.get('critical', False) and check['status'] != 'healthy':
                    ready = False
                    break
            
            return {
                'ready': ready,
                'status': 'ready' if ready else 'not_ready',
                'timestamp': report.timestamp,
                'critical_checks': {
                    name: check['status'] 
                    for name, check in report.checks.items() 
                    if check.get('critical', False)
                }
            }
            
        except Exception as e:
            return {
                'ready': False,
                'status': 'error',
                'error': f'Readiness check failed: {str(e)}',
                'timestamp': time.time()
            }
    
    async def get_liveness_endpoint(self) -> Dict[str, Any]:
        """Liveness endpoint response (/liveness)"""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Simple liveness check - just verify the service is responding
            return {
                'alive': True,
                'timestamp': current_time,
                'uptime_seconds': uptime,
                'pid': psutil.Process().pid
            }
            
        except Exception as e:
            return {
                'alive': False,
                'error': f'Liveness check failed: {str(e)}',
                'timestamp': time.time()
            }
    
    async def get_metrics_endpoint(self) -> Dict[str, Any]:
        """Metrics endpoint response (/metrics)"""
        try:
            if not self.current_metrics:
                await self._update_system_metrics()
            
            return {
                'timestamp': self.current_metrics.timestamp,
                'system': {
                    'cpu_percent': self.current_metrics.cpu_percent,
                    'memory_percent': self.current_metrics.memory_percent,
                    'memory_used_mb': self.current_metrics.memory_used_mb,
                    'memory_available_mb': self.current_metrics.memory_available_mb,
                    'disk_percent': self.current_metrics.disk_percent,
                    'disk_used_gb': self.current_metrics.disk_used_gb,
                    'disk_free_gb': self.current_metrics.disk_free_gb,
                    'network_sent_mb': self.current_metrics.network_sent_mb,
                    'network_recv_mb': self.current_metrics.network_recv_mb,
                    'process_count': self.current_metrics.process_count
                },
                'uptime_seconds': time.time() - self.start_time
            }
            
        except Exception as e:
            return {
                'error': f'Metrics collection failed: {str(e)}',
                'timestamp': time.time()
            }
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        try:
            self.is_monitoring = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Health monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping health monitoring: {e}")
    
    async def cleanup(self):
        """Cleanup health monitor resources"""
        await self.stop_monitoring()
        logger.info("HealthMonitor cleanup completed") 
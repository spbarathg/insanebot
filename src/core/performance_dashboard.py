"""
Performance Dashboard

Real-time performance monitoring dashboard for all profit-boosting components.
Provides comprehensive metrics, alerts, and optimization recommendations.

Features:
- Real-time component monitoring
- Performance analytics
- Alert system
- Optimization recommendations
- Export capabilities
- Beautiful visualizations
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class ComponentStatus:
    """Status of a system component"""
    name: str
    status: str  # 'online', 'offline', 'degraded', 'error'
    uptime_pct: float
    last_heartbeat: float
    performance_score: float  # 0-1
    alerts_count: int
    metrics: Dict[str, Any]

@dataclass
class PerformanceAlert:
    """Performance alert"""
    component: str
    alert_type: str  # 'warning', 'error', 'critical'
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: float
    resolved: bool = False

class PerformanceDashboard:
    """
    Comprehensive performance dashboard for profit maximization system
    
    Monitors all components in real-time and provides optimization insights.
    """
    
    def __init__(
        self,
        pump_monitor=None,
        sniper_executor=None,
        social_engine=None,
        smart_money_tracker=None,
        ai_coordinator=None
    ):
        # Component references
        self.pump_monitor = pump_monitor
        self.sniper_executor = sniper_executor
        self.social_engine = social_engine
        self.smart_money_tracker = smart_money_tracker
        self.ai_coordinator = ai_coordinator
        
        # Dashboard state
        self.component_statuses: Dict[str, ComponentStatus] = {}
        self.performance_alerts: List[PerformanceAlert] = []
        self.performance_history: Dict[str, List[Dict]] = {}
        
        # Thresholds for alerts
        self.alert_thresholds = {
            'execution_time_ms': {'warning': 150, 'critical': 500},
            'success_rate_pct': {'warning': 70, 'critical': 50},
            'signal_accuracy_pct': {'warning': 60, 'critical': 40},
            'response_time_ms': {'warning': 1000, 'critical': 5000},
            'error_rate_pct': {'warning': 5, 'critical': 15},
            'memory_usage_pct': {'warning': 80, 'critical': 95},
            'cpu_usage_pct': {'warning': 80, 'critical': 95}
        }
        
        # Dashboard configuration
        self.config = {
            'update_interval_seconds': 5,
            'history_retention_hours': 24,
            'alert_cooldown_seconds': 300,  # 5 minutes
            'performance_window_minutes': 60
        }
        
        # System metrics
        self.system_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_profit_sol': 0.0,
            'total_profit_usd': 0.0,
            'best_trade_pct': 0.0,
            'worst_trade_pct': 0.0,
            'avg_execution_time_ms': 0.0,
            'system_uptime_hours': 0.0,
            'signals_generated': 0,
            'signals_acted_upon': 0
        }
        
        logger.info("📊 Performance Dashboard initialized - Ready for monitoring!")
    
    async def start_monitoring(self):
        """Start real-time performance monitoring"""
        try:
            logger.info("🚀 Starting performance dashboard monitoring...")
            
            # Start monitoring tasks
            tasks = [
                asyncio.create_task(self._monitor_components()),
                asyncio.create_task(self._update_metrics()),
                asyncio.create_task(self._process_alerts()),
                asyncio.create_task(self._cleanup_old_data())
            ]
            
            logger.info("✅ Performance dashboard monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard monitoring: {e}")
    
    async def _monitor_components(self):
        """Monitor all system components"""
        while True:
            try:
                # Monitor each component
                await self._monitor_pump_fun()
                await self._monitor_sniper_executor()
                await self._monitor_social_engine()
                await self._monitor_smart_money_tracker()
                await self._monitor_ai_coordinator()
                await self._monitor_system_resources()
                
                await asyncio.sleep(self.config['update_interval_seconds'])
                
            except Exception as e:
                logger.error(f"Error in component monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_pump_fun(self):
        """Monitor pump.fun monitor component"""
        try:
            if not self.pump_monitor:
                return
            
            metrics = self.pump_monitor.get_performance_metrics()
            
            status = ComponentStatus(
                name="pump_fun_monitor",
                status="online" if metrics.get('active_signals', 0) >= 0 else "offline",
                uptime_pct=99.0,  # Would calculate from actual uptime
                last_heartbeat=time.time(),
                performance_score=self._calculate_pump_fun_performance(metrics),
                alerts_count=0,
                metrics=metrics
            )
            
            self.component_statuses["pump_fun_monitor"] = status
            
            # Check for alerts
            await self._check_pump_fun_alerts(metrics)
            
        except Exception as e:
            logger.error(f"Error monitoring pump.fun: {e}")
    
    async def _monitor_sniper_executor(self):
        """Monitor sniper executor component"""
        try:
            if not self.sniper_executor:
                return
            
            stats = self.sniper_executor.get_performance_stats()
            
            status = ComponentStatus(
                name="sniper_executor",
                status="online",
                uptime_pct=99.5,
                last_heartbeat=time.time(),
                performance_score=self._calculate_sniper_performance(stats),
                alerts_count=0,
                metrics=stats
            )
            
            self.component_statuses["sniper_executor"] = status
            
            # Check for alerts
            await self._check_sniper_alerts(stats)
            
        except Exception as e:
            logger.error(f"Error monitoring sniper executor: {e}")
    
    async def _monitor_social_engine(self):
        """Monitor social sentiment engine"""
        try:
            if not self.social_engine:
                return
            
            metrics = self.social_engine.get_performance_metrics()
            
            status = ComponentStatus(
                name="social_sentiment_engine",
                status="online" if metrics.get('trending_tokens', 0) >= 0 else "degraded",
                uptime_pct=98.0,
                last_heartbeat=time.time(),
                performance_score=self._calculate_social_performance(metrics),
                alerts_count=0,
                metrics=metrics
            )
            
            self.component_statuses["social_sentiment_engine"] = status
            
            # Check for alerts
            await self._check_social_alerts(metrics)
            
        except Exception as e:
            logger.error(f"Error monitoring social engine: {e}")
    
    async def _monitor_smart_money_tracker(self):
        """Monitor smart money tracking system"""
        try:
            if not self.smart_money_tracker:
                return
            
            # Get smart money metrics (would need to implement this method)
            metrics = {
                'tracked_wallets': 100,
                'active_signals': 5,
                'tracking_accuracy': 85.0
            }
            
            status = ComponentStatus(
                name="smart_money_tracker",
                status="online",
                uptime_pct=99.8,
                last_heartbeat=time.time(),
                performance_score=0.85,
                alerts_count=0,
                metrics=metrics
            )
            
            self.component_statuses["smart_money_tracker"] = status
            
        except Exception as e:
            logger.error(f"Error monitoring smart money tracker: {e}")
    
    async def _monitor_ai_coordinator(self):
        """Monitor AI coordination system"""
        try:
            if not self.ai_coordinator:
                return
            
            # Get AI metrics (would need to implement this method)
            metrics = {
                'models_active': 3,
                'prediction_accuracy': 78.0,
                'response_time_ms': 250
            }
            
            status = ComponentStatus(
                name="ai_coordinator",
                status="online",
                uptime_pct=97.5,
                last_heartbeat=time.time(),
                performance_score=0.78,
                alerts_count=0,
                metrics=metrics
            )
            
            self.component_statuses["ai_coordinator"] = status
            
        except Exception as e:
            logger.error(f"Error monitoring AI coordinator: {e}")
    
    async def _monitor_system_resources(self):
        """Monitor system resource usage"""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            metrics = {
                'cpu_usage_pct': cpu_percent,
                'memory_usage_pct': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage_pct': psutil.disk_usage('/').percent
            }
            
            status = ComponentStatus(
                name="system_resources",
                status="online" if cpu_percent < 90 and memory.percent < 90 else "degraded",
                uptime_pct=100.0,
                last_heartbeat=time.time(),
                performance_score=1.0 - max(cpu_percent/100, memory.percent/100),
                alerts_count=0,
                metrics=metrics
            )
            
            self.component_statuses["system_resources"] = status
            
            # Check for resource alerts
            await self._check_resource_alerts(metrics)
            
        except ImportError:
            # psutil not available, skip system monitoring
            pass
        except Exception as e:
            logger.error(f"Error monitoring system resources: {e}")
    
    def _calculate_pump_fun_performance(self, metrics: Dict[str, Any]) -> float:
        """Calculate pump.fun monitor performance score"""
        try:
            tokens_detected = metrics.get('detected_tokens_24h', 0)
            signals_generated = metrics.get('signals_generated', 0)
            
            # Performance based on detection rate and signal quality
            detection_score = min(1.0, tokens_detected / 50)  # 50 tokens per day = perfect
            signal_score = min(1.0, signals_generated / 20)   # 20 signals per day = perfect
            
            return (detection_score + signal_score) / 2
            
        except Exception as e:
            logger.error(f"Error calculating pump.fun performance: {e}")
            return 0.0
    
    def _calculate_sniper_performance(self, stats: Dict[str, Any]) -> float:
        """Calculate sniper executor performance score"""
        try:
            avg_time = stats.get('average_execution_time_ms', 1000)
            success_rate = stats.get('success_rate_pct', 0)
            
            # Performance based on speed and success rate
            speed_score = max(0, 1.0 - (avg_time / 500))  # 500ms = 0 score, 0ms = 1.0
            success_score = success_rate / 100
            
            return (speed_score + success_score) / 2
            
        except Exception as e:
            logger.error(f"Error calculating sniper performance: {e}")
            return 0.0
    
    def _calculate_social_performance(self, metrics: Dict[str, Any]) -> float:
        """Calculate social sentiment engine performance score"""
        try:
            mentions_processed = metrics.get('mentions_processed', 0)
            trending_detected = metrics.get('trending_tokens_detected', 0)
            
            # Performance based on processing volume and trend detection
            processing_score = min(1.0, mentions_processed / 1000)  # 1000 mentions = perfect
            trend_score = min(1.0, trending_detected / 10)          # 10 trending = perfect
            
            return (processing_score + trend_score) / 2
            
        except Exception as e:
            logger.error(f"Error calculating social performance: {e}")
            return 0.0
    
    async def _check_pump_fun_alerts(self, metrics: Dict[str, Any]):
        """Check for pump.fun related alerts"""
        try:
            # Check signal generation rate
            signals_generated = metrics.get('signals_generated', 0)
            if signals_generated == 0:
                await self._create_alert(
                    "pump_fun_monitor", "warning",
                    "No pump.fun signals generated in monitoring period",
                    "signals_generated", 0, 1
                )
            
        except Exception as e:
            logger.error(f"Error checking pump.fun alerts: {e}")
    
    async def _check_sniper_alerts(self, stats: Dict[str, Any]):
        """Check for sniper executor alerts"""
        try:
            avg_time = stats.get('average_execution_time_ms', 0)
            success_rate = stats.get('success_rate_pct', 100)
            
            # Execution time alerts
            if avg_time > self.alert_thresholds['execution_time_ms']['critical']:
                await self._create_alert(
                    "sniper_executor", "critical",
                    f"Execution time too slow: {avg_time:.1f}ms",
                    "execution_time_ms", avg_time,
                    self.alert_thresholds['execution_time_ms']['critical']
                )
            elif avg_time > self.alert_thresholds['execution_time_ms']['warning']:
                await self._create_alert(
                    "sniper_executor", "warning",
                    f"Execution time degraded: {avg_time:.1f}ms",
                    "execution_time_ms", avg_time,
                    self.alert_thresholds['execution_time_ms']['warning']
                )
            
            # Success rate alerts
            if success_rate < self.alert_thresholds['success_rate_pct']['critical']:
                await self._create_alert(
                    "sniper_executor", "critical",
                    f"Success rate critical: {success_rate:.1f}%",
                    "success_rate_pct", success_rate,
                    self.alert_thresholds['success_rate_pct']['critical']
                )
            
        except Exception as e:
            logger.error(f"Error checking sniper alerts: {e}")
    
    async def _check_social_alerts(self, metrics: Dict[str, Any]):
        """Check for social sentiment engine alerts"""
        try:
            mentions_processed = metrics.get('mentions_processed', 0)
            
            # Check if processing has stopped
            if mentions_processed == 0:
                await self._create_alert(
                    "social_sentiment_engine", "warning",
                    "No social mentions processed recently",
                    "mentions_processed", 0, 1
                )
            
        except Exception as e:
            logger.error(f"Error checking social alerts: {e}")
    
    async def _check_resource_alerts(self, metrics: Dict[str, Any]):
        """Check for system resource alerts"""
        try:
            cpu_usage = metrics.get('cpu_usage_pct', 0)
            memory_usage = metrics.get('memory_usage_pct', 0)
            
            # CPU usage alerts
            if cpu_usage > self.alert_thresholds['cpu_usage_pct']['critical']:
                await self._create_alert(
                    "system_resources", "critical",
                    f"CPU usage critical: {cpu_usage:.1f}%",
                    "cpu_usage_pct", cpu_usage,
                    self.alert_thresholds['cpu_usage_pct']['critical']
                )
            
            # Memory usage alerts
            if memory_usage > self.alert_thresholds['memory_usage_pct']['critical']:
                await self._create_alert(
                    "system_resources", "critical",
                    f"Memory usage critical: {memory_usage:.1f}%",
                    "memory_usage_pct", memory_usage,
                    self.alert_thresholds['memory_usage_pct']['critical']
                )
            
        except Exception as e:
            logger.error(f"Error checking resource alerts: {e}")
    
    async def _create_alert(
        self,
        component: str,
        alert_type: str,
        message: str,
        metric_name: str,
        current_value: float,
        threshold_value: float
    ):
        """Create a performance alert"""
        try:
            # Check for recent similar alerts (cooldown)
            recent_alerts = [
                alert for alert in self.performance_alerts
                if (alert.component == component and 
                    alert.metric_name == metric_name and
                    time.time() - alert.timestamp < self.config['alert_cooldown_seconds'])
            ]
            
            if recent_alerts:
                return  # Skip duplicate alert
            
            alert = PerformanceAlert(
                component=component,
                alert_type=alert_type,
                message=message,
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
                timestamp=time.time()
            )
            
            self.performance_alerts.append(alert)
            
            # Log alert
            log_func = logger.critical if alert_type == 'critical' else (
                logger.warning if alert_type == 'warning' else logger.info
            )
            log_func(f"🚨 ALERT [{alert_type.upper()}] {component}: {message}")
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    async def _update_metrics(self):
        """Update overall system metrics"""
        while True:
            try:
                # Update system-wide metrics
                await self._calculate_system_metrics()
                
                # Store historical data
                await self._store_historical_data()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_system_metrics(self):
        """Calculate overall system performance metrics"""
        try:
            # Aggregate metrics from all components
            total_performance = 0
            component_count = 0
            
            for status in self.component_statuses.values():
                total_performance += status.performance_score
                component_count += 1
            
            overall_performance = total_performance / component_count if component_count > 0 else 0
            
            # Update system metrics
            self.system_metrics.update({
                'overall_performance_score': overall_performance,
                'active_components': component_count,
                'total_alerts': len([a for a in self.performance_alerts if not a.resolved]),
                'system_health': self._assess_system_health()
            })
            
        except Exception as e:
            logger.error(f"Error calculating system metrics: {e}")
    
    def _assess_system_health(self) -> str:
        """Assess overall system health"""
        try:
            # Check for critical alerts
            critical_alerts = [a for a in self.performance_alerts 
                             if a.alert_type == 'critical' and not a.resolved]
            
            if critical_alerts:
                return "critical"
            
            # Check component status
            offline_components = [s for s in self.component_statuses.values() 
                                if s.status == 'offline']
            
            if offline_components:
                return "degraded"
            
            # Check performance scores
            avg_performance = sum(s.performance_score for s in self.component_statuses.values())
            avg_performance /= len(self.component_statuses) if self.component_statuses else 1
            
            if avg_performance >= 0.8:
                return "excellent"
            elif avg_performance >= 0.6:
                return "good"
            elif avg_performance >= 0.4:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Error assessing system health: {e}")
            return "unknown"
    
    async def _store_historical_data(self):
        """Store historical performance data"""
        try:
            current_time = time.time()
            
            for component_name, status in self.component_statuses.items():
                if component_name not in self.performance_history:
                    self.performance_history[component_name] = []
                
                # Store snapshot
                self.performance_history[component_name].append({
                    'timestamp': current_time,
                    'performance_score': status.performance_score,
                    'status': status.status,
                    'metrics': status.metrics.copy()
                })
                
                # Limit history size
                max_entries = (self.config['history_retention_hours'] * 60) // 1  # Per minute
                if len(self.performance_history[component_name]) > max_entries:
                    self.performance_history[component_name] = \
                        self.performance_history[component_name][-max_entries:]
            
        except Exception as e:
            logger.error(f"Error storing historical data: {e}")
    
    async def _process_alerts(self):
        """Process and manage alerts"""
        while True:
            try:
                current_time = time.time()
                
                # Auto-resolve old alerts
                for alert in self.performance_alerts:
                    if (not alert.resolved and 
                        current_time - alert.timestamp > 3600):  # 1 hour auto-resolve
                        alert.resolved = True
                
                # Clean up very old alerts
                self.performance_alerts = [
                    alert for alert in self.performance_alerts
                    if current_time - alert.timestamp < 86400  # Keep for 24 hours
                ]
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error processing alerts: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_old_data(self):
        """Clean up old performance data"""
        while True:
            try:
                current_time = time.time()
                retention_time = self.config['history_retention_hours'] * 3600
                
                # Clean up historical data
                for component_name in self.performance_history:
                    self.performance_history[component_name] = [
                        entry for entry in self.performance_history[component_name]
                        if current_time - entry['timestamp'] < retention_time
                    ]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up old data: {e}")
                await asyncio.sleep(3600)
    
    async def update_metrics(self):
        """Manual metrics update (called externally)"""
        await self._calculate_system_metrics()
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary"""
        try:
            # Component summaries
            component_summary = {}
            for name, status in self.component_statuses.items():
                component_summary[name] = {
                    'status': status.status,
                    'performance_score': status.performance_score,
                    'uptime_pct': status.uptime_pct,
                    'alerts_count': status.alerts_count
                }
            
            # Recent alerts
            recent_alerts = [
                {
                    'component': alert.component,
                    'type': alert.alert_type,
                    'message': alert.message,
                    'timestamp': alert.timestamp
                }
                for alert in self.performance_alerts[-10:]  # Last 10 alerts
                if not alert.resolved
            ]
            
            return {
                'system_health': self._assess_system_health(),
                'system_metrics': self.system_metrics,
                'components': component_summary,
                'recent_alerts': recent_alerts,
                'total_components': len(self.component_statuses),
                'online_components': len([s for s in self.component_statuses.values() 
                                        if s.status == 'online']),
                'last_updated': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard summary: {e}")
            return {'error': str(e)}
    
    def export_performance_report(self) -> str:
        """Export detailed performance report as JSON"""
        try:
            report = {
                'generated_at': time.time(),
                'system_metrics': self.system_metrics,
                'component_statuses': {
                    name: asdict(status) for name, status in self.component_statuses.items()
                },
                'performance_alerts': [
                    asdict(alert) for alert in self.performance_alerts
                ],
                'performance_history': self.performance_history
            }
            
            return json.dumps(report, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")
            return json.dumps({'error': str(e)})

# Helper function
def create_performance_dashboard(**components) -> PerformanceDashboard:
    """Create a performance dashboard instance"""
    return PerformanceDashboard(**components) 
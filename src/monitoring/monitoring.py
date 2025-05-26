import asyncio
import json
import os
import psutil
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .alerts import AlertSystem
import logging

logger = logging.getLogger(__name__)

class MonitoringError(Exception):
    """Custom exception for monitoring errors."""
    pass

class MonitoringSystem:
    def __init__(self):
        self.alert_system = AlertSystem()
        self.metrics_history: List[Dict] = []
        self.last_metrics_save = 0
        self.last_backup = 0
        self.start_time = asyncio.get_event_loop().time()
        self.metrics_file = "metrics.json"
        self.backup_dir = "backups"
        self.alert_thresholds = {
            'cpu': 80.0,  # Default CPU warning threshold
            'memory': 80.0,  # Default memory warning threshold
            'disk': 90.0,  # Default disk warning threshold
            'win_rate': 40.0,  # Default win rate warning threshold
            'profit': -10.0  # Default profit warning threshold
        }
        logger.info("MonitoringSystem instance initialized", extra={
            'alert_thresholds': self.alert_thresholds
        })

    def get_system_metrics(self) -> Dict:
        """Get current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'uptime': asyncio.get_event_loop().time() - self.start_time
            }
            
            logger.debug("System metrics collected", extra={'metrics': metrics})
            return metrics

        except Exception as e:
            error_msg = f"Error getting system metrics: {str(e)}"
            logger.error(error_msg)
            raise MonitoringError(error_msg)

    def get_trading_metrics(self, trade_history: List[Dict]) -> Dict:
        """Calculate trading performance metrics."""
        try:
            if not trade_history:
                logger.warning("No trade history available for metrics calculation")
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_profit': 0,
                    'avg_profit_per_trade': 0,
                    'trades_per_hour': 0
                }

            total_trades = len(trade_history)
            winning_trades = sum(1 for trade in trade_history if trade.get('profit', 0) > 0)
            total_profit = sum(trade.get('profit', 0) for trade in trade_history)
            
            # Calculate trades per hour
            first_trade_time = datetime.fromisoformat(trade_history[0]['timestamp'])
            hours_running = (datetime.now() - first_trade_time).total_seconds() / 3600
            trades_per_hour = total_trades / max(hours_running, 1)

            metrics = {
                'total_trades': total_trades,
                'win_rate': (winning_trades / total_trades) * 100 if total_trades > 0 else 0,
                'total_profit': total_profit,
                'avg_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0,
                'trades_per_hour': trades_per_hour
            }
            
            logger.info("Trading metrics calculated", extra={'metrics': metrics})
            return metrics

        except Exception as e:
            error_msg = f"Error calculating trading metrics: {str(e)}"
            logger.error(error_msg)
            raise MonitoringError(error_msg)

    def check_health(self, trade_history: List[Dict]) -> Dict:
        """Check system and trading health"""
        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'system': self._check_system_health(),
                'trading': self._check_trading_health(trade_history),
                'alerts': []
            }

            # Check for alerts
            if health_status['system']['cpu_percent'] > self.alert_thresholds['cpu']:
                alert = {
                    'type': 'system',
                    'level': 'warning',
                    'message': f"High CPU usage: {health_status['system']['cpu_percent']}%"
                }
                health_status['alerts'].append(alert)
                logger.warning(alert['message'], extra={'alert': alert})

            if health_status['system']['memory_percent'] > self.alert_thresholds['memory']:
                alert = {
                    'type': 'system',
                    'level': 'warning',
                    'message': f"High memory usage: {health_status['system']['memory_percent']}%"
                }
                health_status['alerts'].append(alert)
                logger.warning(alert['message'], extra={'alert': alert})

            if health_status['system']['disk_percent'] > self.alert_thresholds['disk']:
                alert = {
                    'type': 'system',
                    'level': 'warning',
                    'message': f"Low disk space: {health_status['system']['disk_percent']}% used"
                }
                health_status['alerts'].append(alert)
                logger.warning(alert['message'], extra={'alert': alert})

            if health_status['trading']['win_rate'] < self.alert_thresholds['win_rate']:
                alert = {
                    'type': 'trading',
                    'level': 'warning',
                    'message': f"Low win rate: {health_status['trading']['win_rate']}%"
                }
                health_status['alerts'].append(alert)
                logger.warning(alert['message'], extra={'alert': alert})

            if health_status['trading']['profit'] < self.alert_thresholds['profit']:
                alert = {
                    'type': 'trading',
                    'level': 'warning',
                    'message': f"Low profit: {health_status['trading']['profit']}%"
                }
                health_status['alerts'].append(alert)
                logger.warning(alert['message'], extra={'alert': alert})

            # Log alerts
            if health_status['alerts']:
                self._log_alerts(health_status['alerts'])

            logger.info("Health check completed", extra={'health_status': health_status})
            return health_status

        except Exception as e:
            error_msg = f"Error checking health: {str(e)}"
            logger.error(error_msg)
            raise MonitoringError(error_msg)

    def _check_system_health(self) -> Dict:
        """Check system resource usage"""
        try:
            metrics = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'network_io': psutil.net_io_counters()._asdict()
            }
            logger.debug("System health metrics collected", extra={'metrics': metrics})
            return metrics
        except Exception as e:
            error_msg = f"Error checking system health: {str(e)}"
            logger.error(error_msg)
            raise MonitoringError(error_msg)

    def _check_trading_health(self, trade_history: List[Dict]) -> Dict:
        """Check trading performance metrics"""
        try:
            if not trade_history:
                logger.warning("No trade history available for health check")
                return {
                    'win_rate': 0,
                    'profit': 0,
                    'trades_per_hour': 0,
                    'avg_profit_per_trade': 0
                }

            # Calculate metrics
            total_trades = len(trade_history)
            winning_trades = sum(1 for trade in trade_history if trade['profit'] > 0)
            total_profit = sum(trade['profit'] for trade in trade_history)
            
            # Calculate trades per hour
            if total_trades > 1:
                first_trade = datetime.fromisoformat(trade_history[0]['timestamp'])
                last_trade = datetime.fromisoformat(trade_history[-1]['timestamp'])
                hours = (last_trade - first_trade).total_seconds() / 3600
                trades_per_hour = total_trades / hours if hours > 0 else 0
            else:
                trades_per_hour = 0

            metrics = {
                'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                'profit': total_profit,
                'trades_per_hour': trades_per_hour,
                'avg_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0
            }
            
            logger.debug("Trading health metrics calculated", extra={'metrics': metrics})
            return metrics

        except Exception as e:
            error_msg = f"Error checking trading health: {str(e)}"
            logger.error(error_msg)
            raise MonitoringError(error_msg)

    def save_metrics(self, trade_history: List[Dict]) -> None:
        """Save current metrics to file"""
        try:
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_metrics_save < 300:  # Default 5 minutes interval
                logger.debug("Skipping metrics save - too soon since last save")
                return

            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system': self._check_system_health(),
                'trading': self._check_trading_health(trade_history)
            }

            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            self.last_metrics_save = current_time
            logger.info("Metrics saved successfully", extra={'metrics': metrics})

        except Exception as e:
            error_msg = f"Error saving metrics: {str(e)}"
            logger.error(error_msg)
            raise MonitoringError(error_msg)

    def create_backup(self) -> None:
        """Create backup of important files"""
        try:
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_backup < 3600:  # Default 1 hour interval
                logger.debug("Skipping backup - too soon since last backup")
                return

            # Create backup directory if it doesn't exist
            if not os.path.exists(self.backup_dir):
                os.makedirs(self.backup_dir)
                logger.debug(f"Created backup directory: {self.backup_dir}")

            # Backup files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")

            # Create backup directory
            os.makedirs(backup_path)
            logger.debug(f"Created backup directory: {backup_path}")

            # Copy files that exist
            files_to_backup = [
                "trade_log.json",
                "whale_log.json", 
                "debug.log",
                "alert.log",
                self.metrics_file
            ]

            for file in files_to_backup:
                if os.path.exists(file):
                    shutil.copy2(file, os.path.join(backup_path, os.path.basename(file)))
                    logger.debug(f"Backed up file: {file}")

            self.last_backup = current_time
            logger.info(f"Backup created successfully", extra={'backup_path': backup_path})

        except Exception as e:
            error_msg = f"Error creating backup: {str(e)}"
            logger.error(error_msg)
            raise MonitoringError(error_msg)

    def _log_alerts(self, alerts: List[Dict]) -> None:
        """Log alerts to file"""
        try:
            alert_file = "alert.log"
            with open(alert_file, 'a') as f:
                for alert in alerts:
                    log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'type': alert['type'],
                        'level': alert['level'],
                        'message': alert['message']
                    }
                    f.write(json.dumps(log_entry) + '\n')
                    logger.info(f"Alert logged", extra={'alert': alert})

        except Exception as e:
            error_msg = f"Error logging alerts: {str(e)}"
            logger.error(error_msg)
            raise MonitoringError(error_msg)

    def get_metrics(self) -> Optional[Dict]:
        """Get current metrics"""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    metrics = json.load(f)
                logger.debug("Retrieved current metrics", extra={'metrics': metrics})
                return metrics
            logger.warning("No metrics file found")
            return None

        except Exception as e:
            error_msg = f"Error getting metrics: {str(e)}"
            logger.error(error_msg)
            raise MonitoringError(error_msg)

    def get_metrics_summary(self, hours: int = 24) -> Dict:
        """Get summary of metrics for the last N hours."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [
                m for m in self.metrics_history
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]

            if not recent_metrics:
                logger.warning(f"No metrics available for the last {hours} hours")
                return {}

            # Calculate averages
            avg_cpu = sum(m['system']['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m['system']['memory_percent'] for m in recent_metrics) / len(recent_metrics)
            avg_win_rate = sum(m['trading']['win_rate'] for m in recent_metrics) / len(recent_metrics)
            avg_profit = sum(m['trading']['total_profit'] for m in recent_metrics) / len(recent_metrics)

            summary = {
                'period_hours': hours,
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory,
                'avg_win_rate': avg_win_rate,
                'avg_profit': avg_profit,
                'total_trades': recent_metrics[-1]['trading']['total_trades'],
                'trades_per_hour': recent_metrics[-1]['trading']['trades_per_hour']
            }
            
            logger.info(f"Metrics summary calculated", extra={
                'period_hours': hours,
                'summary': summary
            })
            return summary

        except Exception as e:
            error_msg = f"Error getting metrics summary: {str(e)}"
            logger.error(error_msg)
            raise MonitoringError(error_msg) 
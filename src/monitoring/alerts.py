import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

logger = logging.getLogger(__name__)

class AlertSystem:
    def __init__(self):
        self.alert_history = []
        self.alert_log_file = "alert.log"
        self._load_alert_history()

    def _load_alert_history(self):
        """Load alert history from file"""
        try:
            with open(self.alert_log_file, 'r') as f:
                self.alert_history = json.load(f)
        except FileNotFoundError:
            self.alert_history = []

    def _save_alert_history(self):
        """Save alert history to file"""
        with open(self.alert_log_file, 'w') as f:
            json.dump(self.alert_history, f)

    def alert(self, level: str, subject: str, message: str):
        """Record alert in history"""
        try:
            # Record alert
            alert_record = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'subject': subject,
                'message': message
            }
            self.alert_history.append(alert_record)
            self._save_alert_history()

            # Log alert
            if level in ['critical', 'error']:
                logger.error(f"🚨 {subject} - {message}")
            elif level == 'warning':
                logger.warning(f"⚠️ {subject} - {message}")
            else:
                logger.info(f"Alert: {subject} - {message}")

        except Exception as e:
            logger.error(f"Alert error: {e}")

    def check_system_health(self, metrics: dict):
        """Check system health and record alerts if needed"""
        try:
            # Check CPU usage
            if metrics['cpu_percent'] > 80.0:  # Default CPU warning threshold
                self.alert(
                    'warning',
                    'High CPU Usage',
                    f"CPU usage is at {metrics['cpu_percent']}%"
                )

            # Check memory usage
            if metrics['memory_percent'] > 80.0:  # Default memory warning threshold
                self.alert(
                    'warning',
                    'High Memory Usage',
                    f"Memory usage is at {metrics['memory_percent']}%"
                )

            # Check disk usage
            if metrics['disk_percent'] > 90.0:  # Default disk warning threshold
                self.alert(
                    'critical',
                    'Low Disk Space',
                    f"Disk usage is at {metrics['disk_percent']}%"
                )

        except Exception as e:
            logger.error(f"Check system health error: {e}")

    def check_trading_health(self, metrics: dict):
        """Check trading health and record alerts if needed"""
        try:
            # Check win rate
            if metrics['win_rate'] < 40.0:  # Default win rate warning threshold
                self.alert(
                    'warning',
                    'Low Win Rate',
                    f"Win rate is at {metrics['win_rate']:.2f}%"
                )

            # Check total profit
            if metrics['total_profit'] < -10.0:  # Default profit warning threshold
                self.alert(
                    'error',
                    'Significant Loss',
                    f"Total profit is at {metrics['total_profit']:.2%}"
                )

            # Check trade frequency
            if metrics['total_trades'] < 1:  # Default minimum trades per hour
                self.alert(
                    'warning',
                    'Low Trade Frequency',
                    f"Only {metrics['total_trades']} trades in the last hour"
                )

        except Exception as e:
            logger.error(f"Check trading health error: {e}")

    def get_alert_history(self, hours: int = 24) -> list:
        """Get alert history for the last N hours"""
        try:
            cutoff = datetime.now().timestamp() - (hours * 3600)
            return [
                alert for alert in self.alert_history
                if datetime.fromisoformat(alert['timestamp']).timestamp() > cutoff
            ]
        except Exception as e:
            logger.error(f"Get alert history error: {e}")
            return [] 
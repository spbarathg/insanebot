import json
from datetime import datetime
from loguru import logger
from ..utils.config import settings

class AlertSystem:
    def __init__(self):
        self.alert_history = []
        self._load_alert_history()

    def _load_alert_history(self):
        """Load alert history from file"""
        try:
            with open(settings.ALERT_LOG_FILE, 'r') as f:
                self.alert_history = json.load(f)
        except FileNotFoundError:
            self.alert_history = []

    def _save_alert_history(self):
        """Save alert history to file"""
        with open(settings.ALERT_LOG_FILE, 'w') as f:
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
            if metrics['cpu_percent'] > settings.CPU_WARNING_THRESHOLD:
                self.alert(
                    'warning',
                    'High CPU Usage',
                    f"CPU usage is at {metrics['cpu_percent']}%"
                )

            # Check memory usage
            if metrics['memory_percent'] > settings.MEMORY_WARNING_THRESHOLD:
                self.alert(
                    'warning',
                    'High Memory Usage',
                    f"Memory usage is at {metrics['memory_percent']}%"
                )

            # Check disk usage
            if metrics['disk_percent'] > settings.DISK_WARNING_THRESHOLD:
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
            if metrics['win_rate'] < settings.WIN_RATE_WARNING_THRESHOLD:
                self.alert(
                    'warning',
                    'Low Win Rate',
                    f"Win rate is at {metrics['win_rate']:.2f}%"
                )

            # Check total profit
            if metrics['total_profit'] < settings.PROFIT_WARNING_THRESHOLD:
                self.alert(
                    'error',
                    'Significant Loss',
                    f"Total profit is at {metrics['total_profit']:.2%}"
                )

            # Check trade frequency
            if metrics['total_trades'] < settings.MIN_TRADES_PER_HOUR:
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
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
            with open('alerts.json', 'r') as f:
                self.alert_history = json.load(f)
        except FileNotFoundError:
            self.alert_history = []

    def _save_alert_history(self):
        """Save alert history to file"""
        with open('alerts.json', 'w') as f:
            json.dump(self.alert_history, f)

    def send_email_alert(self, subject: str, message: str):
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = settings.ALERT_EMAIL
            msg['To'] = settings.ALERT_EMAIL
            msg['Subject'] = f"[Meme Bot Alert] {subject}"

            msg.attach(MIMEText(message, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(settings.ALERT_EMAIL, settings.ALERT_EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()

            logger.info(f"Email alert sent: {subject}")

        except Exception as e:
            logger.error(f"Send email alert error: {e}")

    def send_telegram_alert(self, message: str):
        """Send Telegram alert"""
        try:
            url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": settings.TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            response.raise_for_status()

            logger.info(f"Telegram alert sent: {message}")

        except Exception as e:
            logger.error(f"Send Telegram alert error: {e}")

    def alert(self, level: str, subject: str, message: str):
        """Send alert through all channels"""
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

            # Send alerts based on level
            if level in ['critical', 'error']:
                self.send_email_alert(subject, message)
                self.send_telegram_alert(f"🚨 {subject}\n\n{message}")
            elif level == 'warning':
                self.send_telegram_alert(f"⚠️ {subject}\n\n{message}")
            else:
                logger.info(f"Alert: {subject} - {message}")

        except Exception as e:
            logger.error(f"Alert error: {e}")

    def check_system_health(self, metrics: dict):
        """Check system health and send alerts if needed"""
        try:
            # Check CPU usage
            if metrics['cpu_percent'] > 90:
                self.alert(
                    'warning',
                    'High CPU Usage',
                    f"CPU usage is at {metrics['cpu_percent']}%"
                )

            # Check memory usage
            if metrics['memory_percent'] > 90:
                self.alert(
                    'warning',
                    'High Memory Usage',
                    f"Memory usage is at {metrics['memory_percent']}%"
                )

            # Check disk usage
            if metrics['disk_percent'] > 90:
                self.alert(
                    'critical',
                    'Low Disk Space',
                    f"Disk usage is at {metrics['disk_percent']}%"
                )

        except Exception as e:
            logger.error(f"Check system health error: {e}")

    def check_trading_health(self, metrics: dict):
        """Check trading health and send alerts if needed"""
        try:
            # Check win rate
            if metrics['win_rate'] < 50:
                self.alert(
                    'warning',
                    'Low Win Rate',
                    f"Win rate is at {metrics['win_rate']:.2f}%"
                )

            # Check total profit
            if metrics['total_profit'] < -0.1:  # -10%
                self.alert(
                    'error',
                    'Significant Loss',
                    f"Total profit is at {metrics['total_profit']:.2%}"
                )

            # Check trade frequency
            if metrics['total_trades'] < 10:  # Less than 10 trades per hour
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
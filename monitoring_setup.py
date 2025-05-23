#!/usr/bin/env python3
"""
Advanced monitoring and alerting setup for Solana trading bot.
"""
import os
import json
import asyncio
import aiohttp
from datetime import datetime
import smtplib
from email.mime.text import MimeText
from typing import Dict, Any

class TradingBotMonitor:
    """Advanced monitoring system for trading bot."""
    
    def __init__(self):
        self.config = self.load_config()
        self.performance_data = {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "max_drawdown": 0.0,
            "start_balance": 0.0,
            "current_balance": 0.0
        }
    
    def load_config(self):
        """Load monitoring configuration."""
        try:
            with open('config/monitoring_config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.create_default_config()
    
    def create_default_config(self):
        """Create default monitoring configuration."""
        config = {
            "telegram": {
                "enabled": False,
                "bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
                "chat_id": "YOUR_CHAT_ID"
            },
            "discord": {
                "enabled": False,
                "webhook_url": "YOUR_DISCORD_WEBHOOK_URL"
            },
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "your-email@gmail.com",
                "password": "your-app-password",
                "to_email": "alerts@yourdomain.com"
            },
            "alerts": {
                "profit_threshold": 0.1,
                "loss_threshold": -0.05,
                "large_trade_threshold": 0.01,
                "error_notifications": True,
                "daily_summary": True
            },
            "performance_tracking": {
                "save_trades": True,
                "calculate_metrics": True,
                "benchmark_comparison": True,
                "export_csv": True
            }
        }
        
        os.makedirs('config', exist_ok=True)
        with open('config/monitoring_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("📝 Created default monitoring config at config/monitoring_config.json")
        print("✏️  Please edit the file with your real credentials")
        return config
    
    async def send_telegram_alert(self, message: str):
        """Send alert via Telegram."""
        if not self.config['telegram']['enabled']:
            return
        
        bot_token = self.config['telegram']['bot_token']
        chat_id = self.config['telegram']['chat_id']
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": f"🤖 Trading Bot Alert\n\n{message}",
            "parse_mode": "HTML"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        print("✅ Telegram alert sent")
                    else:
                        print(f"❌ Telegram alert failed: {response.status}")
        except Exception as e:
            print(f"❌ Telegram error: {e}")
    
    async def send_discord_alert(self, message: str):
        """Send alert via Discord webhook."""
        if not self.config['discord']['enabled']:
            return
        
        webhook_url = self.config['discord']['webhook_url']
        data = {
            "content": f"🤖 **Trading Bot Alert**\n\n{message}"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=data) as response:
                    if response.status == 204:
                        print("✅ Discord alert sent")
                    else:
                        print(f"❌ Discord alert failed: {response.status}")
        except Exception as e:
            print(f"❌ Discord error: {e}")
    
    def send_email_alert(self, subject: str, message: str):
        """Send alert via email."""
        if not self.config['email']['enabled']:
            return
        
        try:
            msg = MimeText(message)
            msg['Subject'] = f"Trading Bot: {subject}"
            msg['From'] = self.config['email']['username']
            msg['To'] = self.config['email']['to_email']
            
            server = smtplib.SMTP(
                self.config['email']['smtp_server'], 
                self.config['email']['smtp_port']
            )
            server.starttls()
            server.login(
                self.config['email']['username'], 
                self.config['email']['password']
            )
            server.send_message(msg)
            server.quit()
            
            print("✅ Email alert sent")
        except Exception as e:
            print(f"❌ Email error: {e}")
    
    async def alert_trade_executed(self, trade_data: Dict[str, Any]):
        """Alert when a trade is executed."""
        action = trade_data.get('action', 'UNKNOWN')
        token = trade_data.get('token_symbol', 'UNKNOWN')
        amount = trade_data.get('amount', 0)
        price = trade_data.get('price', 0)
        pnl = trade_data.get('pnl', 0)
        
        emoji = "📈" if action == "BUY" else "📉"
        pnl_emoji = "💰" if pnl > 0 else "💸" if pnl < 0 else "🔄"
        
        message = f"""
{emoji} <b>Trade Executed</b>
Token: {token}
Action: {action}
Amount: {amount:.4f} SOL
Price: ${price:.6f}
{pnl_emoji} PnL: {pnl:+.4f} SOL
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        await self.send_telegram_alert(message)
        await self.send_discord_alert(message)
        self.send_email_alert(f"Trade: {action} {token}", message)
    
    async def alert_profit_milestone(self, total_pnl: float, daily_pnl: float):
        """Alert when profit milestones are reached."""
        message = f"""
🎉 <b>Profit Milestone Reached!</b>
Total PnL: {total_pnl:+.4f} SOL
Daily PnL: {daily_pnl:+.4f} SOL
Win Rate: {self.calculate_win_rate():.1f}%
Total Trades: {self.performance_data['total_trades']}
"""
        
        await self.send_telegram_alert(message)
        await self.send_discord_alert(message)
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.performance_data['total_trades'] == 0:
            return 0.0
        return (self.performance_data['winning_trades'] / self.performance_data['total_trades']) * 100
    
    def generate_daily_report(self) -> str:
        """Generate daily performance report."""
        win_rate = self.calculate_win_rate()
        
        return f"""
📊 <b>Daily Trading Report</b>
Date: {datetime.now().strftime('%Y-%m-%d')}

📈 Performance:
• Daily PnL: {self.performance_data['daily_pnl']:+.4f} SOL
• Total PnL: {self.performance_data['total_pnl']:+.4f} SOL
• Win Rate: {win_rate:.1f}%
• Total Trades: {self.performance_data['total_trades']}
• Winning Trades: {self.performance_data['winning_trades']}

💰 Portfolio:
• Start Balance: {self.performance_data['start_balance']:.4f} SOL
• Current Balance: {self.performance_data['current_balance']:.4f} SOL
• Max Drawdown: {self.performance_data['max_drawdown']:.4f} SOL

🎯 Status: {"🟢 Profitable" if self.performance_data['total_pnl'] > 0 else "🔴 In Loss"}
"""

def setup_monitoring():
    """Setup monitoring system."""
    print("🔧 Setting up advanced monitoring system...")
    
    monitor = TradingBotMonitor()
    
    print("✅ Monitoring system initialized")
    print("📝 Configuration file created at: config/monitoring_config.json")
    print("\n📋 Next steps:")
    print("1. Edit config/monitoring_config.json with your credentials")
    print("2. Enable desired alert channels (Telegram/Discord/Email)")
    print("3. Set up webhook URLs and API tokens")
    print("4. Test alerts before going live")
    
    return monitor

if __name__ == "__main__":
    setup_monitoring() 
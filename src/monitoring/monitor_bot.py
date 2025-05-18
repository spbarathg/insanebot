import json
import time
from datetime import datetime
import psutil
import requests
from loguru import logger

def get_bot_status():
    """Get bot service status"""
    try:
        result = requests.get('http://localhost:8080/status')
        return result.json()
    except:
        return None

def get_system_metrics():
    """Get system resource usage"""
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent
    }

def analyze_trades():
    """Analyze recent trades"""
    try:
        with open('trades.json', 'r') as f:
            trades = json.load(f)
        
        if not trades:
            return None
            
        recent_trades = [
            t for t in trades 
            if time.time() - t['timestamp'] < 3600  # Last hour
        ]
        
        if not recent_trades:
            return None
            
        return {
            'total_trades': len(recent_trades),
            'profitable_trades': sum(1 for t in recent_trades if t['profit'] > 0),
            'total_profit': sum(t['profit'] for t in recent_trades),
            'avg_profit': sum(t['profit'] for t in recent_trades) / len(recent_trades)
        }
    except:
        return None

def monitor_bot():
    """Main monitoring loop"""
    logger.add(
        f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        rotation="100 MB"
    )
    
    while True:
        try:
            # Get metrics
            bot_status = get_bot_status()
            system_metrics = get_system_metrics()
            trade_metrics = analyze_trades()
            
            # Log metrics
            logger.info("=== Bot Status ===")
            if bot_status:
                logger.info(f"Bot Status: {bot_status['status']}")
                logger.info(f"Active Trades: {len(bot_status['active_trades'])}")
            
            logger.info("\n=== System Metrics ===")
            logger.info(f"CPU Usage: {system_metrics['cpu_percent']}%")
            logger.info(f"Memory Usage: {system_metrics['memory_percent']}%")
            logger.info(f"Disk Usage: {system_metrics['disk_percent']}%")
            
            if trade_metrics:
                logger.info("\n=== Trading Metrics (Last Hour) ===")
                logger.info(f"Total Trades: {trade_metrics['total_trades']}")
                logger.info(f"Profitable Trades: {trade_metrics['profitable_trades']}")
                logger.info(f"Win Rate: {(trade_metrics['profitable_trades']/trade_metrics['total_trades'])*100:.2f}%")
                logger.info(f"Total Profit: {trade_metrics['total_profit']:.2%}")
                logger.info(f"Average Profit: {trade_metrics['avg_profit']:.2%}")
            
            logger.info("\n" + "="*50 + "\n")
            
            # Wait before next check
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_bot() 
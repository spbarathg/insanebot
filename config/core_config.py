"""
Core configuration settings for the trading bot.
"""
from typing import Dict, Any
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create necessary directories
for directory in [CONFIG_DIR, DATA_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Core configuration
CORE_CONFIG: Dict[str, Any] = {
    "trading": {
        "min_liquidity": float(os.getenv("MIN_LIQUIDITY", "10000")),  # Minimum liquidity in SOL
        "max_slippage": float(os.getenv("MAX_SLIPPAGE", "0.02")),  # 2% maximum slippage
        "min_profit_threshold": float(os.getenv("MIN_PROFIT", "0.05")),  # 5% minimum profit target
        "max_position_size": float(os.getenv("MAX_POSITION", "0.1")),  # Maximum position size in SOL
        "cooldown_period": int(os.getenv("COOLDOWN", "300")),  # 5 minutes between trades
    },
    "monitoring": {
        "check_interval": int(os.getenv("CHECK_INTERVAL", "60")),  # Check every minute
        "max_retries": int(os.getenv("MAX_RETRIES", "3")),  # Maximum retries for failed operations
        "retry_delay": int(os.getenv("RETRY_DELAY", "5")),  # Delay between retries in seconds
    },
    "error_handling": {
        "max_consecutive_errors": int(os.getenv("MAX_ERRORS", "5")),
        "error_cooldown": int(os.getenv("ERROR_COOLDOWN", "300")),  # 5 minutes cooldown after max errors
    }
}

# Market configuration
MARKET_CONFIG: Dict[str, Any] = {
    "min_liquidity": float(os.getenv("MIN_LIQUIDITY", "10000")),  # Minimum liquidity in SOL
    "volatility_threshold": float(os.getenv("VOLATILITY", "0.1")),  # 10% price change threshold
    "min_holders": int(os.getenv("MIN_HOLDERS", "100")),  # Minimum number of token holders
    "max_token_age": int(os.getenv("MAX_TOKEN_AGE", "30")),  # Maximum token age in days
}

# Trading configuration
TRADING_CONFIG: Dict[str, Any] = {
    "max_position_size": float(os.getenv("MAX_POSITION", "0.1")),  # Maximum position size in SOL
    "min_position_size": float(os.getenv("MIN_POSITION", "0.01")),  # Minimum position size in SOL
    "stop_loss": float(os.getenv("STOP_LOSS", "0.05")),  # 5% stop loss
    "take_profit": float(os.getenv("TAKE_PROFIT", "0.1")),  # 10% take profit
    "max_concurrent_trades": int(os.getenv("MAX_TRADES", "5")),  # Maximum number of concurrent trades
}

# Export all configurations
__all__ = [
    'CORE_CONFIG',
    'MARKET_CONFIG',
    'TRADING_CONFIG'
] 
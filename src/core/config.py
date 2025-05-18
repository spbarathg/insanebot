"""
Configuration module for the core components.
"""
import os
import sys
from pathlib import Path

# Add config directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "config"))

try:
    from core_config import CORE_CONFIG, MARKET_CONFIG, TRADING_CONFIG
except ImportError:
    # Fallback configuration for tests
    CORE_CONFIG = {
        "trading": {
            "min_liquidity": 10000.0,
            "max_slippage": 0.02,
            "min_profit_threshold": 0.05,
            "max_position_size": 0.1,
            "cooldown_period": 300,
        },
        "monitoring": {
            "check_interval": 60,
            "max_retries": 3,
            "retry_delay": 5,
        },
        "error_handling": {
            "max_consecutive_errors": 5,
            "error_cooldown": 300,
            "critical_errors": ["ValueError", "ConnectionError"]
        }
    }

    MARKET_CONFIG = {
        "min_liquidity": 10000.0,
        "volatility_threshold": 0.1,
        "min_holders": 100,
        "max_token_age": 30,
    }

    TRADING_CONFIG = {
        "max_position_size": 0.1,
        "min_position_size": 0.01,
        "stop_loss": 0.05,
        "take_profit": 0.1,
        "max_concurrent_trades": 5,
    }

__all__ = [
    'CORE_CONFIG',
    'MARKET_CONFIG',
    'TRADING_CONFIG'
] 
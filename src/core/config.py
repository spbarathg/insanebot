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
    from ant_princess_config import ANT_PRINCESS_CONFIG, QUEEN_CONFIG as ANT_QUEEN_CONFIG
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
    
    # Fallback ANT_PRINCESS_CONFIG if not available from config directory
    ANT_PRINCESS_CONFIG = {
        "market_weight": 0.6,
        "sentiment_weight": 0.3,
        "wallet_weight": 0.1,
        "buy_threshold": 0.7,
        "sell_threshold": -0.3,
        "base_position_size": 0.01,
        "max_position_size": 0.1,
        "multiplication_thresholds": {
            "performance_score": 0.8,
            "experience_threshold": 10
        }
    }
    
    # Fallback ANT_QUEEN_CONFIG if not available from config directory
    ANT_QUEEN_CONFIG = {
        "optimization_frequency": 86400,  # seconds
        "performance_metrics": {
            "score_weights": {
                "profit": 0.4,
                "risk_management": 0.3,
                "execution_speed": 0.2,
                "adaptability": 0.1
            },
            "adaptation_threshold": 0.7
        },
        "colony_management": {
            "max_princesses": 10,
            "pruning_threshold": 0.5,
            "experience_retention_period": 2592000  # 30 days in seconds
        }
    }

# AI configuration settings
AI_CONFIG = {
    "grok_api_key": os.getenv("GROK_API_KEY", ""),
    "grok_api_url": os.getenv("GROK_API_URL", "https://api.grok.ai/v1"),
    "analysis_timeframe": "24h",
    "sentiment_timeframe": "7d",
    "model_parameters": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 500
    }
}

__all__ = [
    'CORE_CONFIG',
    'MARKET_CONFIG',
    'TRADING_CONFIG',
    'ANT_PRINCESS_CONFIG',
    'ANT_QUEEN_CONFIG',
    'AI_CONFIG'
] 
"""
Configuration package for Enhanced Ant Bot
"""

from .ant_princess_config import ANT_PRINCESS_CONFIG, QUEEN_CONFIG, SYSTEM_CONSTANTS
from .core_config import CORE_CONFIG, MARKET_CONFIG, TRADING_CONFIG

# Create a unified settings object for backward compatibility
class Settings:
    def __init__(self):
        # Core settings
        self.MIN_LIQUIDITY = CORE_CONFIG["trading"]["min_liquidity"]
        self.MAX_SLIPPAGE = CORE_CONFIG["trading"]["max_slippage"]
        self.MIN_PROFIT_THRESHOLD = CORE_CONFIG["trading"]["min_profit_threshold"]
        self.MAX_POSITION_SIZE = CORE_CONFIG["trading"]["max_position_size"]
        self.COOLDOWN_PERIOD = CORE_CONFIG["trading"]["cooldown_period"]
        
        # Monitoring settings
        self.CHECK_INTERVAL = CORE_CONFIG["monitoring"]["check_interval"]
        self.HEARTBEAT_INTERVAL = CORE_CONFIG["monitoring"]["heartbeat_interval"]
        self.PERFORMANCE_TRACKING = CORE_CONFIG["monitoring"]["performance_tracking"]
        
        # Market settings
        self.TARGET_POOLS = MARKET_CONFIG["target_pools"]
        self.EXCLUDED_TOKENS = MARKET_CONFIG["excluded_tokens"]
        self.PRICE_IMPACT_THRESHOLD = MARKET_CONFIG["price_impact_threshold"]
        
        # Risk management settings
        self.RISK_LIMITS = {
            "max_exposure": 0.8,  # Maximum 80% of capital exposed
            "max_token_exposure": 0.2,  # Maximum 20% in any single token
            "max_portfolio_exposure": 1.0,  # Maximum 100% portfolio exposure
            "max_daily_loss": 0.05,  # Maximum 5% daily loss
            "stop_loss_threshold": 0.1,  # 10% stop loss
            "max_consecutive_losses": 3
        }
        
        # Position limits
        self.POSITION_LIMITS = {
            "max_position_size": 0.1,  # Maximum 0.1 SOL per position
            "min_position_size": 0.001,  # Minimum 0.001 SOL per position
            "max_positions": 10,  # Maximum 10 concurrent positions
            "position_timeout": 3600  # 1 hour position timeout
        }
        
        # File paths
        self.PORTFOLIO_FILE = "data/portfolio.json"
        self.PERFORMANCE_FILE = "data/performance.json"
        self.TRADE_LOG_FILE = "data/trades.json"
        
        # API settings
        self.API_TIMEOUT = 30
        self.MAX_RETRIES = 3
        self.RATE_LIMIT_DELAY = 0.1

# Create the settings instance
settings = Settings()

# Export everything for backward compatibility
__all__ = [
    'ANT_PRINCESS_CONFIG',
    'QUEEN_CONFIG', 
    'SYSTEM_CONSTANTS',
    'CORE_CONFIG',
    'MARKET_CONFIG',
    'TRADING_CONFIG',
    'Settings',
    'settings'
] 
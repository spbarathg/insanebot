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
        self.MAX_RETRIES = CORE_CONFIG["monitoring"]["max_retries"]
        self.RETRY_DELAY = CORE_CONFIG["monitoring"]["retry_delay"]
        
        # Error handling
        self.MAX_CONSECUTIVE_ERRORS = CORE_CONFIG["error_handling"]["max_consecutive_errors"]
        self.ERROR_COOLDOWN = CORE_CONFIG["error_handling"]["error_cooldown"]
        
        # Market settings
        self.VOLATILITY_THRESHOLD = MARKET_CONFIG["volatility_threshold"]
        self.MIN_HOLDERS = MARKET_CONFIG["min_holders"]
        self.MAX_TOKEN_AGE = MARKET_CONFIG["max_token_age"]
        
        # Trading settings
        self.MIN_POSITION_SIZE = TRADING_CONFIG["min_position_size"]
        self.STOP_LOSS = TRADING_CONFIG["stop_loss"]
        self.TAKE_PROFIT = TRADING_CONFIG["take_profit"]
        self.MAX_CONCURRENT_TRADES = TRADING_CONFIG["max_concurrent_trades"]
        
        # System monitoring thresholds
        self.CPU_WARNING_THRESHOLD = 80.0
        self.MEMORY_WARNING_THRESHOLD = 85.0
        self.DISK_WARNING_THRESHOLD = 90.0
        self.WIN_RATE_WARNING_THRESHOLD = 0.3
        self.PROFIT_WARNING_THRESHOLD = -10.0

# Create settings instance
settings = Settings()

__all__ = [
    'ANT_PRINCESS_CONFIG',
    'QUEEN_CONFIG',
    'SYSTEM_CONSTANTS',
    'CORE_CONFIG',
    'MARKET_CONFIG',
    'TRADING_CONFIG',
    'settings'
] 
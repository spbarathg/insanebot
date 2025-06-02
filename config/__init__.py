"""
Configuration package for Enhanced Ant Bot
"""

from .core_config import CORE_CONFIG, MARKET_CONFIG, TRADING_CONFIG

# Create a unified settings object for backward compatibility
class Settings:
    def __init__(self):
        # Core trading settings - use actual structure
        trading_config = CORE_CONFIG.get("trading", {})
        self.MIN_LIQUIDITY = trading_config.get("min_liquidity", 10000)
        self.MAX_SLIPPAGE = trading_config.get("max_slippage", 0.05)
        self.MIN_PROFIT_THRESHOLD = trading_config.get("min_profit_threshold", 0.05)
        self.MAX_POSITION_SIZE = trading_config.get("max_position_size", 0.02)
        self.COOLDOWN_PERIOD = trading_config.get("cooldown_period", 300)
        
        # Monitoring settings (provide fallback values)
        self.CHECK_INTERVAL = 30  # seconds
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 1  # seconds
        
        # Error handling settings (provide fallback values)
        self.MAX_CONSECUTIVE_ERRORS = 5
        self.ERROR_COOLDOWN = 60  # seconds
        
        # Market settings (provide fallback values)
        self.VOLATILITY_THRESHOLD = 0.1
        self.MIN_HOLDERS = 100
        self.MAX_TOKEN_AGE = 86400  # 24 hours
        
        # Trading settings (provide fallback values)
        self.MIN_POSITION_SIZE = 0.001
        self.STOP_LOSS = 0.1
        self.TAKE_PROFIT = 0.2
        self.MAX_CONCURRENT_TRADES = 3
        
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
        self.RATE_LIMIT_DELAY = 0.1

# Create the settings instance
settings = Settings()

# Export everything for backward compatibility
__all__ = [
    'CORE_CONFIG',
    'MARKET_CONFIG',
    'TRADING_CONFIG',
    'Settings',
    'settings'
] 
"""
Configuration settings for Enhanced Ant Bot
"""
import os
from pathlib import Path

class Settings:
    """Application settings"""
    
    # API Keys
    QUICKNODE_ENDPOINT_URL = os.getenv("QUICKNODE_ENDPOINT_URL", "")
    HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
    JUPITER_API_KEY = os.getenv("JUPITER_API_KEY", "")
    GROK_API_KEY = os.getenv("GROK_API_KEY", "")
    X_API_KEY = os.getenv("X_API_KEY", "")
    
    # Wallet configuration
    PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
    WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "")
    
    # Trading configuration
    INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "0.1"))
    MIN_LIQUIDITY = float(os.getenv("MIN_LIQUIDITY", "10000"))
    MAX_SLIPPAGE = float(os.getenv("MAX_SLIPPAGE", "0.02"))
    
    # File paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    CONFIG_DIR = BASE_DIR / "config"
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, LOGS_DIR]:
        directory.mkdir(exist_ok=True)
    
    # Data files
    WHALE_LOG_FILE = DATA_DIR / "whale_activity.json"
    TRADING_LOG_FILE = DATA_DIR / "trading_history.json"
    PERFORMANCE_LOG_FILE = DATA_DIR / "performance_metrics.json"
    
    # Network configuration
    SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    NETWORK = os.getenv("NETWORK", "mainnet-beta")
    
    # Monitoring configuration
    ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Trading safety limits
    MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "5"))
    MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
    MIN_POSITION_SIZE = float(os.getenv("MIN_POSITION_SIZE", "0.01"))
    
    # Risk management
    STOP_LOSS_PERCENTAGE = float(os.getenv("STOP_LOSS_PERCENTAGE", "0.05"))
    TAKE_PROFIT_PERCENTAGE = float(os.getenv("TAKE_PROFIT_PERCENTAGE", "0.1"))
    
    # Database (if using)
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    
    # Rate limiting
    API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "100"))
    
    # Webhook URLs (for notifications)
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# Create settings instance
settings = Settings() 
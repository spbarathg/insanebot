from pydantic import BaseSettings
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import json
from pathlib import Path

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    SOLANA_PRIVATE_KEY: str = os.getenv("SOLANA_PRIVATE_KEY", "")
    QUICKNODE_API_KEY: str = os.getenv("QUICKNODE_API_KEY", "")
    X_API_KEY: str = os.getenv("X_API_KEY", "")
    X_API_SECRET: str = os.getenv("X_API_SECRET", "")
    GROK_API_KEY: str = os.getenv("GROK_API_KEY", "")
    AWS_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY", "")
    AWS_SECRET_KEY: str = os.getenv("AWS_SECRET_KEY", "")
    BIRDEYE_API_KEY: str = os.getenv("BIRDEYE_API_KEY", "")
    SOLANA_RPC_URL: str = os.getenv("SOLANA_RPC_URL", "")

    # Alert Settings
    ALERT_EMAIL: str = os.getenv("ALERT_EMAIL", "")
    ALERT_EMAIL_PASSWORD: str = os.getenv("ALERT_EMAIL_PASSWORD", "")

    # Trading Parameters
    STARTING_CAPITAL: float = 1.0  # Initial capital in SOL
    MAX_TRADE_SIZE: float = 1.0   # Maximum trade size in SOL
    MIN_TRADE_SIZE: float = 0.01  # Minimum trade size in SOL
    MAX_POSITION_SIZE: float = 0.2  # Maximum position size as percentage of capital
    DAILY_LOSS_LIMIT: float = 0.1  # Maximum daily loss as percentage of capital
    MIN_PROFIT_TARGET: float = 0.05 # Minimum profit target
    MAX_PROFIT_TARGET: float = 0.2  # Maximum profit target
    CONFIDENCE_THRESHOLD: float = 0.7  # Minimum confidence to execute trade

    # Risk Management
    TIERED_EXITS: Dict[float, float] = {
        0.05: 0.3,  # Sell 30% at 5% profit
        0.1: 0.3,   # Sell 30% at 10% profit
        0.2: 0.4    # Sell 40% at 20% profit
    }
    LOSS_RECOVERY_MULTIPLIER: float = 1.5
    MAX_RECOVERY_TRADES: int = 3
    KILL_SWITCH_THRESHOLD: float = -0.1  # -10% loss triggers kill switch
    STOP_LOSS_PERCENTAGE: float = 0.05  # Stop loss percentage
    TAKE_PROFIT_PERCENTAGE: float = 0.1  # Take profit percentage

    # Network and API Settings
    RPC_ENDPOINTS: List[str] = [
        "https://api.mainnet-beta.solana.com",
        "https://solana-api.projectserum.com"
    ]
    GROK_API_LATENCY: float = 0.42  # seconds
    GROK_API_TIMEOUT: float = 0.5   # seconds
    GROK_API_RETRY_DELAY: float = 0.1  # seconds
    GROK_API_MAX_RETRIES: int = 3
    DATA_REFRESH_INTERVAL: int = 5  # Data refresh interval in seconds
    MAX_RETRIES: int = 3
    RETRY_INTERVAL: int = 10  # Retry interval in seconds
    PRIORITY_FEE_MULTIPLIER: float = 1.5  # Priority fee multiplier
    RPC_FAILOVER_TIMEOUT: float = 1.0  # seconds
    MAX_RPC_RETRIES: int = 3

    # DEX Settings
    JUPITER_API_URL: str = "https://quote-api.jup.ag/v6"
    DEXSCREENER_API_URL: str = "https://api.dexscreener.com/latest/dex"
    BIRDEYE_API_URL: str = "https://public-api.birdeye.so"
    MIN_SLIPPAGE: float = 0.01  # 1% minimum slippage
    MAX_SLIPPAGE: float = 0.05  # 5% maximum slippage
    SWAP_TIMEOUT: float = 30.0  # seconds

    # Volatility Management
    VOLATILITY_THRESHOLD: float = 0.15  # 15% price movement
    MAX_PRICE_IMPACT: float = 0.05  # Maximum price impact
    MIN_LIQUIDITY_RATIO: float = 0.1  # Minimum liquidity ratio

    # Rug Pull Protection
    MIN_LIQUIDITY_LOCK_TIME: int = 86400  # 24 hours
    MIN_HOLDER_COUNT: int = 100
    MAX_OWNER_CONCENTRATION: float = 0.2  # 20% maximum owner concentration
    MIN_LIQUIDITY: float = 1.0  # Minimum liquidity in SOL
    MAX_TOKEN_AGE: int = 3600  # Maximum token age in seconds

    # Capital Efficiency
    CAPITAL_EFFICIENCY_THRESHOLD: float = 0.8  # 80% minimum capital efficiency
    MIN_WIN_RATE: float = 0.65  # Minimum win rate for strategy
    MIN_PROFIT_PER_TRADE: float = 0.02  # Minimum profit per trade

    # Logging
    LOG_LEVEL: str = "INFO"
    TRADE_LOG_FILE: str = "trades.json"
    WHALE_LOG_FILE: str = "whale_trades.json"
    DEBUG_LOG_FILE: str = "debug.log"
    ALERT_LOG_FILE: str = "alerts.json"

    # Simulation Mode
    SIMULATION_MODE: bool = True
    SIMULATION_CAPITAL: float = 0.1  # SOL

    # Feature Weights
    FEATURE_WEIGHTS: Dict[str, float] = {
        'sentiment': 0.3,
        'whale_activity': 0.2,
        'price_momentum': 0.2,
        'liquidity': 0.15,
        'volume': 0.15
    }

    # Monitoring
    HEALTH_CHECK_INTERVAL: int = 60  # seconds
    METRICS_SAVE_INTERVAL: int = 300  # 5 minutes
    BACKUP_INTERVAL: int = 3600  # 1 hour
    MAX_LOG_SIZE: int = 100 * 1024 * 1024  # 100 MB
    MAX_LOG_FILES: int = 10

    # Alert Thresholds
    CPU_WARNING_THRESHOLD: int = 90  # percent
    MEMORY_WARNING_THRESHOLD: int = 90  # percent
    DISK_WARNING_THRESHOLD: int = 90  # percent
    WIN_RATE_WARNING_THRESHOLD: int = 50  # percent
    PROFIT_WARNING_THRESHOLD: float = -0.1  # -10%
    MIN_TRADES_PER_HOUR: int = 10

    # Local LLM Settings
    LLM_MODEL_PATH: str = "mistralai/Mistral-7B-v0.1"
    LLM_DEVICE: str = "cuda"
    LLM_BATCH_SIZE: int = 1
    LLM_MAX_LENGTH: int = 512
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.9
    LLM_CACHE_SIZE: int = 100
    LLM_CACHE_TTL: int = 30
    
    # Learning Parameters
    LEARNING_RATE: float = 0.001
    MIN_SAMPLES_FOR_LEARNING: int = 10
    MAX_SAMPLES_FOR_LEARNING: int = 1000
    
    # File Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOG_DIR: Path = BASE_DIR / "logs"
    MODEL_DIR: Path = BASE_DIR / "models"
    
    PORTFOLIO_FILE: Path = DATA_DIR / "portfolio.json"
    TRADE_LOG_FILE: Path = DATA_DIR / "trades.json"
    ACTIVE_TRADES_FILE: Path = DATA_DIR / "active_trades.json"
    TRAINING_DATA_FILE: Path = DATA_DIR / "training_data.json"
    DEBUG_LOG_FILE: Path = LOG_DIR / "debug.log"
    
    # Risk Limits
    RISK_LIMITS: Dict = {
        'max_exposure': 100.0,  # Maximum portfolio exposure in SOL
        'max_token_exposure': 10.0,  # Maximum exposure per token in SOL
        'max_portfolio_exposure': 50.0,  # Maximum total portfolio exposure in SOL
        'kill_switch_threshold': -0.15,  # Kill switch threshold (-15%)
        'min_sentiment_score': 0.6,  # Minimum sentiment score
        'sentiment_spike_threshold': 0.2,  # Sentiment spike threshold
    }
    
    # Position Limits
    POSITION_LIMITS: Dict = {
        'max_position_size': 5.0,  # Maximum position size in SOL
        'min_adjustment': 0.1,  # Minimum position size adjustment in SOL
    }

    def __init__(self, **data):
        super().__init__(**data)
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories"""
        self.DATA_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
        self.MODEL_DIR.mkdir(exist_ok=True)

    class Config:
        env_file = ".env"

settings = Settings() 
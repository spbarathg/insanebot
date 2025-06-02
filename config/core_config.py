"""
Core configuration settings for the trading bot.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import os
from dotenv import load_dotenv
import json
from pydantic import BaseModel, field_validator, Field
import logging

logger = logging.getLogger(__name__)

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

class MEVProtectionConfig(BaseModel):
    """Enhanced MEV protection configuration with Jito bundle support"""
    protection_level: str = Field(default="advanced", pattern="^(basic|advanced|jito_bundle|private_pool|maximum)$")
    jito_tip_lamports: int = Field(default=50000, ge=1000, le=1000000, description="Jito tip amount")
    enable_jito_bundles: bool = Field(default=True, description="Enable Jito bundle submissions")
    randomize_timing: bool = Field(default=True, description="Enable timing randomization")
    timing_variance_ms: int = Field(default=500, ge=100, le=2000, description="Timing variance in milliseconds")
    bundle_max_wait_ms: int = Field(default=2000, ge=500, le=10000, description="Max bundle wait time")
    sandwich_detection_enabled: bool = Field(default=True, description="Enable sandwich attack detection")
    front_run_protection: bool = Field(default=True, description="Enable front-running protection")
    threat_confidence_threshold: float = Field(default=0.7, ge=0.1, le=1.0, description="Threat detection threshold")
    emergency_protection_threshold: float = Field(default=0.9, ge=0.5, le=1.0, description="Emergency protection threshold")

class TradingConfig(BaseModel):
    """Enhanced trading configuration with audit recommendations"""
    # Improved risk parameters based on audit
    min_liquidity: float = Field(default=10000, ge=1000, le=1000000, description="Minimum liquidity in SOL")
    max_slippage: float = Field(default=0.05, ge=0.001, le=0.15, description="Maximum slippage - REDUCED to 5%")
    min_profit_threshold: float = Field(default=0.05, ge=0.01, le=1.0, description="Minimum profit target")
    max_position_size: float = Field(default=0.02, ge=0.005, le=0.05, description="Maximum position size - REDUCED to 2%")
    cooldown_period: int = Field(default=300, ge=10, le=3600, description="Cooldown between trades in seconds")
    
    # Enhanced technical analysis parameters
    min_confidence_threshold: float = Field(default=0.7, ge=0.5, le=1.0, description="Minimum signal confidence")
    require_volume_confirmation: bool = Field(default=True, description="Require volume confirmation for trades")
    min_volume_ratio: float = Field(default=1.5, ge=1.0, le=5.0, description="Minimum volume ratio for trades")
    
    # Enhanced execution parameters
    sub_100ms_execution_enabled: bool = Field(default=True, description="Enable sub-100ms execution targeting")
    parallel_rpc_count: int = Field(default=3, ge=1, le=5, description="Number of parallel RPC endpoints")
    execution_timeout_ms: int = Field(default=50, ge=10, le=1000, description="Execution timeout in milliseconds")
    
    # Signal processing weights (rebalanced based on audit)
    signal_weights: Dict[str, float] = Field(default={
        'technical_analysis': 0.30,    # NEW: Highest weight
        'pump_fun': 0.25,             # Reduced from 35%
        'smart_money': 0.25,          # Keep existing
        'social_sentiment': 0.15,      # Reduced from 25%
        'ai_analysis': 0.05           # Minimal weight
    })
    
    @field_validator('max_slippage')
    def validate_slippage(cls, v):
        if v > 0.1:  # 10% slippage warning
            logger.warning(f"High slippage tolerance configured: {v*100:.1f}%")
        return v
    
    @field_validator('signal_weights')
    def validate_signal_weights(cls, v):
        total_weight = sum(v.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow 1% tolerance
            raise ValueError(f"Signal weights must sum to 1.0, got {total_weight}")
        return v

class SecurityConfig(BaseModel):
    """Enhanced security and risk management validation"""
    private_key: str = Field(min_length=32, description="Solana private key")
    wallet_address: str = Field(min_length=32, max_length=50, description="Solana wallet address")
    
    # Enhanced risk management based on audit
    max_daily_loss: float = Field(default=0.05, ge=0.01, le=0.15, description="Maximum daily loss - REDUCED to 5%")
    emergency_stop_loss: float = Field(default=0.15, ge=0.05, le=0.25, description="Emergency stop loss - REDUCED to 15%")
    max_portfolio_exposure: float = Field(default=0.15, ge=0.05, le=0.30, description="Max meme coin exposure - 15%")
    
    # Enhanced security parameters
    wallet_balance_threshold_sol: float = Field(default=0.01, ge=0.001, le=1.0, description="Minimum wallet balance")
    max_concurrent_trades: int = Field(default=3, ge=1, le=10, description="Maximum concurrent trades - REDUCED")
    rate_limit_requests_per_minute: int = Field(default=60, ge=10, le=200, description="API rate limit - REDUCED")
    
    # Position sizing controls
    kelly_criterion_enabled: bool = Field(default=True, description="Enable Kelly Criterion position sizing")
    max_kelly_fraction: float = Field(default=0.25, ge=0.1, le=0.5, description="Maximum Kelly fraction")
    
    @field_validator('private_key')
    def validate_private_key(cls, v):
        if len(v) < 64:  # Solana private keys should be longer
            raise ValueError("Private key appears too short for Solana")
        return v

class APIConfig(BaseModel):
    """Enhanced API endpoints and keys validation"""
    helius_api_key: Optional[str] = Field(min_length=10, description="Helius API key")
    quicknode_endpoint: Optional[str] = Field(description="QuickNode RPC endpoint")
    grok_api_key: Optional[str] = Field(description="Grok API key (optional)")
    
    # Enhanced API configuration
    backup_rpc_endpoints: List[str] = Field(default_factory=list, description="Backup RPC endpoints")
    api_timeout_seconds: int = Field(default=5, ge=1, le=30, description="API timeout - REDUCED to 5s")
    max_retries: int = Field(default=2, ge=1, le=5, description="Maximum API retries - REDUCED")
    
    # Rate limiting
    requests_per_second: float = Field(default=10.0, ge=1.0, le=50.0, description="Requests per second limit")
    burst_limit: int = Field(default=20, ge=5, le=100, description="Burst request limit")
    
    @field_validator('quicknode_endpoint')
    def validate_quicknode(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("QuickNode endpoint must be a valid URL")
        return v

class ExitStrategyConfig(BaseModel):
    """Exit strategy configuration based on audit recommendations"""
    # Trailing stop configuration
    enable_trailing_stops: bool = Field(default=True, description="Enable trailing stop losses")
    min_profit_for_trail: float = Field(default=0.05, ge=0.01, le=0.20, description="Min profit before trailing")
    trail_step_pct: float = Field(default=0.02, ge=0.01, le=0.10, description="Trailing step percentage")
    
    # Partial profit taking
    enable_partial_exits: bool = Field(default=True, description="Enable partial profit taking")
    profit_targets: List[float] = Field(default=[0.15, 0.30, 0.60, 1.00], description="Profit target levels")
    exit_percentages: List[float] = Field(default=[0.25, 0.25, 0.25, 0.25], description="Exit percentages")
    
    # Advanced exit triggers
    enable_volume_exhaustion: bool = Field(default=True, description="Enable volume exhaustion exits")
    enable_rsi_divergence: bool = Field(default=True, description="Enable RSI divergence exits")
    enable_time_based_exits: bool = Field(default=True, description="Enable time-based exits")
    max_hold_time_hours: int = Field(default=24, ge=1, le=168, description="Maximum hold time in hours")
    
    @field_validator('exit_percentages')
    def validate_exit_percentages(cls, v):
        if abs(sum(v) - 1.0) > 0.01:
            raise ValueError("Exit percentages must sum to 1.0")
        return v

class AdvancedRiskConfig(BaseModel):
    """Advanced risk management configuration"""
    # Portfolio risk management
    enable_portfolio_var: bool = Field(default=True, description="Enable portfolio VaR calculation")
    max_portfolio_var_1d: float = Field(default=0.05, ge=0.01, le=0.15, description="Max 1-day portfolio VaR")
    max_correlation: float = Field(default=0.7, ge=0.3, le=1.0, description="Maximum position correlation")
    min_diversification_score: float = Field(default=0.5, ge=0.1, le=1.0, description="Minimum diversification score")
    
    # Dynamic risk adjustment
    enable_dynamic_sizing: bool = Field(default=True, description="Enable dynamic position sizing")
    volatility_adjustment_factor: float = Field(default=0.5, ge=0.1, le=2.0, description="Volatility adjustment factor")
    confidence_scaling_factor: float = Field(default=1.0, ge=0.5, le=2.0, description="Confidence scaling factor")
    
    # Emergency protocols
    enable_circuit_breakers: bool = Field(default=True, description="Enable trading circuit breakers")
    max_drawdown_threshold: float = Field(default=0.20, ge=0.05, le=0.50, description="Max drawdown before halt")
    rapid_loss_threshold: float = Field(default=0.10, ge=0.02, le=0.25, description="Rapid loss threshold")
    rapid_loss_timeframe_minutes: int = Field(default=15, ge=5, le=60, description="Rapid loss timeframe")

def load_validated_config() -> Dict[str, Any]:
    """Load and validate all configuration with audit improvements"""
    try:
        # Collect configuration from environment with improved defaults
        trading_config = TradingConfig(
            min_liquidity=float(os.getenv("MIN_LIQUIDITY", "10000")),
            max_slippage=float(os.getenv("MAX_SLIPPAGE", "0.05")),  # REDUCED to 5%
            min_profit_threshold=float(os.getenv("MIN_PROFIT", "0.05")),
            max_position_size=float(os.getenv("MAX_POSITION", "0.02")),  # REDUCED to 2%
            cooldown_period=int(os.getenv("COOLDOWN", "300")),
            min_confidence_threshold=float(os.getenv("MIN_CONFIDENCE", "0.7")),
            sub_100ms_execution_enabled=os.getenv("SUB_100MS_EXECUTION", "true").lower() == "true",
            parallel_rpc_count=int(os.getenv("PARALLEL_RPC_COUNT", "3")),
            execution_timeout_ms=int(os.getenv("EXECUTION_TIMEOUT_MS", "50"))
        )
        
        security_config = SecurityConfig(
            private_key=os.getenv("PRIVATE_KEY", ""),
            wallet_address=os.getenv("WALLET_ADDRESS", ""),
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "0.05")),  # REDUCED to 5%
            emergency_stop_loss=float(os.getenv("EMERGENCY_STOP", "0.15")),  # REDUCED to 15%
            max_portfolio_exposure=float(os.getenv("MAX_MEME_EXPOSURE", "0.15")),  # 15% max meme exposure
            wallet_balance_threshold_sol=float(os.getenv("WALLET_BALANCE_THRESHOLD", "0.01")),
            max_concurrent_trades=int(os.getenv("MAX_CONCURRENT_TRADES", "3")),  # REDUCED
            rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60"))  # REDUCED
        )
        
        api_config = APIConfig(
            helius_api_key=os.getenv("HELIUS_API_KEY"),
            quicknode_endpoint=os.getenv("QUICKNODE_ENDPOINT"),
            grok_api_key=os.getenv("GROK_API_KEY"),
            backup_rpc_endpoints=os.getenv("BACKUP_RPC_ENDPOINTS", "").split(",") if os.getenv("BACKUP_RPC_ENDPOINTS") else [],
            api_timeout_seconds=int(os.getenv("API_TIMEOUT", "5")),  # REDUCED
            max_retries=int(os.getenv("MAX_RETRIES", "2")),  # REDUCED
            requests_per_second=float(os.getenv("REQUESTS_PER_SECOND", "10.0"))
        )
        
        mev_config = MEVProtectionConfig(
            protection_level=os.getenv("MEV_PROTECTION_LEVEL", "advanced"),
            jito_tip_lamports=int(os.getenv("JITO_TIP_LAMPORTS", "50000")),
            enable_jito_bundles=os.getenv("ENABLE_JITO_BUNDLES", "true").lower() == "true",
            randomize_timing=os.getenv("RANDOMIZE_TIMING", "true").lower() == "true",
            timing_variance_ms=int(os.getenv("TIMING_VARIANCE_MS", "500")),
            bundle_max_wait_ms=int(os.getenv("BUNDLE_MAX_WAIT_MS", "2000")),
            sandwich_detection_enabled=os.getenv("SANDWICH_DETECTION", "true").lower() == "true",
            front_run_protection=os.getenv("FRONT_RUN_PROTECTION", "true").lower() == "true"
        )
        
        exit_strategy_config = ExitStrategyConfig(
            enable_trailing_stops=os.getenv("ENABLE_TRAILING_STOPS", "true").lower() == "true",
            min_profit_for_trail=float(os.getenv("MIN_PROFIT_FOR_TRAIL", "0.05")),
            trail_step_pct=float(os.getenv("TRAIL_STEP_PCT", "0.02")),
            enable_partial_exits=os.getenv("ENABLE_PARTIAL_EXITS", "true").lower() == "true",
            max_hold_time_hours=int(os.getenv("MAX_HOLD_TIME_HOURS", "24"))
        )
        
        advanced_risk_config = AdvancedRiskConfig(
            enable_portfolio_var=os.getenv("ENABLE_PORTFOLIO_VAR", "true").lower() == "true",
            max_portfolio_var_1d=float(os.getenv("MAX_PORTFOLIO_VAR", "0.05")),
            max_correlation=float(os.getenv("MAX_CORRELATION", "0.7")),
            enable_dynamic_sizing=os.getenv("ENABLE_DYNAMIC_SIZING", "true").lower() == "true",
            enable_circuit_breakers=os.getenv("ENABLE_CIRCUIT_BREAKERS", "true").lower() == "true",
            max_drawdown_threshold=float(os.getenv("MAX_DRAWDOWN_THRESHOLD", "0.20"))
        )
        
        logger.info("✅ Enhanced configuration validation passed with audit improvements")
        
        return {
            "trading": trading_config.dict(),
            "security": security_config.dict(),
            "api": api_config.dict(),
            "mev_protection": mev_config.dict(),
            "exit_strategy": exit_strategy_config.dict(),
            "advanced_risk": advanced_risk_config.dict()
        }
        
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {str(e)}")
        raise ValueError(f"Invalid configuration: {str(e)}")

# Enhanced constants with audit recommendations
ENHANCED_TRADING_CONSTANTS = {
    # Core trading parameters (improved)
    "MAX_SLIPPAGE": 0.05,           # REDUCED from 15% to 5%
    "MAX_POSITION_SIZE": 0.02,      # REDUCED from 10% to 2%
    "MIN_CONFIDENCE": 0.7,          # Minimum 70% confidence
    "MAX_DAILY_LOSS": 0.05,         # REDUCED to 5% max daily loss
    "MAX_MEME_EXPOSURE": 0.15,      # 15% max exposure to meme coins
    
    # Signal processing weights (rebalanced)
    "SIGNAL_WEIGHTS": {
        "technical_analysis": 0.30,   # NEW: Highest priority
        "pump_fun": 0.25,            # Reduced from 35%
        "smart_money": 0.25,         # Keep existing
        "social_sentiment": 0.15,     # Reduced from 25%
        "ai_analysis": 0.05          # Minimal weight
    },
    
    # Exit strategy parameters
    "ENABLE_TRAILING_STOPS": True,
    "ENABLE_PARTIAL_EXITS": True,
    "PROFIT_TARGETS": [0.15, 0.30, 0.60, 1.00],  # 15%, 30%, 60%, 100%
    "EXIT_PERCENTAGES": [0.25, 0.25, 0.25, 0.25],  # Scale out equally
    
    # MEV protection
    "ENABLE_MEV_PROTECTION": True,
    "MEV_PROTECTION_LEVEL": "advanced",
    "ENABLE_JITO_BUNDLES": True,
    "JITO_TIP_LAMPORTS": 50000,
    
    # Performance optimization
    "SUB_100MS_EXECUTION": True,
    "PARALLEL_RPC_COUNT": 3,
    "EXECUTION_TIMEOUT_MS": 50,
    
    # Risk management
    "ENABLE_CIRCUIT_BREAKERS": True,
    "MAX_DRAWDOWN_THRESHOLD": 0.20,   # 20% max drawdown
    "EMERGENCY_STOP_THRESHOLD": 0.15,  # 15% emergency stop
}

# Legacy support with improved defaults
try:
    validated_config = load_validated_config()
    CORE_CONFIG = validated_config["trading"]
    MARKET_CONFIG = CORE_CONFIG.copy()
    TRADING_CONFIG = CORE_CONFIG.copy()
    
    # Add enhanced constants
    CORE_CONFIG.update(ENHANCED_TRADING_CONSTANTS)
    
except Exception as e:
    logger.error(f"❌ Using fallback configuration with enhanced defaults: {str(e)}")
    
    # Enhanced fallback configuration
    CORE_CONFIG: Dict[str, Any] = {
        "trading": {
            "min_liquidity": float(os.getenv("MIN_LIQUIDITY", "10000")),
            "max_slippage": 0.05,  # IMPROVED: 5% instead of 15%
            "min_profit_threshold": float(os.getenv("MIN_PROFIT", "0.05")),
            "max_position_size": 0.02,  # IMPROVED: 2% instead of 10%
            "cooldown_period": int(os.getenv("COOLDOWN", "300")),
            "min_confidence_threshold": 0.7,  # NEW: Minimum confidence
        }
    }
    
    # Apply enhanced constants
    CORE_CONFIG.update(ENHANCED_TRADING_CONSTANTS)
    MARKET_CONFIG = CORE_CONFIG.copy()
    TRADING_CONFIG = CORE_CONFIG.copy()

def load_config() -> Dict[str, Any]:
    """Load configuration from config files with enhanced validation"""
    try:
        config_dir = Path(__file__).parent
        
        # Load main config
        config_file = config_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Load risk management config
        risk_file = config_dir / "risk_management.json"
        if risk_file.exists():
            with open(risk_file, 'r') as f:
                risk_config = json.load(f)
                config.update(risk_config)
        
        # Apply enhanced defaults
        enhanced_defaults = {
            "max_slippage": 0.05,
            "max_position_size": 0.02,
            "max_daily_loss": 0.05,
            "max_meme_exposure": 0.15,
            "min_confidence": 0.7,
            "enable_trailing_stops": True,
            "enable_partial_exits": True,
            "enable_mev_protection": True
        }
        
        for key, value in enhanced_defaults.items():
            if key not in config:
                config[key] = value
        
        return config
        
    except Exception as e:
        logger.error(f"Error loading enhanced config: {str(e)}")
        return ENHANCED_TRADING_CONSTANTS

# Export all configurations with enhanced features
__all__ = [
    'CORE_CONFIG',
    'MARKET_CONFIG', 
    'TRADING_CONFIG',
    'ENHANCED_TRADING_CONSTANTS',
    'load_validated_config',
    'TradingConfig',
    'SecurityConfig',
    'MEVProtectionConfig',
    'ExitStrategyConfig',
    'AdvancedRiskConfig'
] 
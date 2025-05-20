"""
Configuration settings for Solana trading bot.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, LOG_DIR, MODEL_DIR]:
    if not directory.exists():
        directory.mkdir(parents=True)

# API Keys (with fallbacks for development environment)
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "abc123example_replace_with_real_api_key")
SOLANA_PRIVATE_KEY = os.getenv("SOLANA_PRIVATE_KEY", "0000000000000000000000000000000000000000000000000000000000000000")

# Trading settings
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "True").lower() in ["true", "1", "yes"]
SIMULATION_CAPITAL = float(os.getenv("SIMULATION_CAPITAL", "0.1"))  # SOL
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "True").lower() in ["true", "1", "yes"]

# Jupiter API settings
JUPITER_API_URL = "https://quote-api.jup.ag/v6"

# Network settings
RPC_ENDPOINTS = [
    f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}",
    "https://api.mainnet-beta.solana.com"  # Fallback
]
HELIUS_WEBSOCKET_URL = f"wss://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"

# Market parameters
MIN_LIQUIDITY = float(os.getenv("MIN_LIQUIDITY", "1.0"))  # Minimum liquidity in SOL
MAX_PRICE_IMPACT = float(os.getenv("MAX_PRICE_IMPACT", "0.05"))  # 5% max price impact
TARGET_PROFIT = float(os.getenv("TARGET_PROFIT", "10.0"))  # 10% target profit
STOP_LOSS = float(os.getenv("STOP_LOSS", "5.0"))  # 5% stop loss
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "1.0"))  # SOL

# Position sizing
DEFAULT_POSITION_SIZE = float(os.getenv("DEFAULT_POSITION_SIZE", "0.01"))  # SOL
MIN_POSITION_SIZE = float(os.getenv("MIN_POSITION_SIZE", "0.001"))  # SOL
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.1"))  # SOL

# Trading bot parameters
LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL", "30"))  # seconds
TRADE_COOLDOWN = int(os.getenv("TRADE_COOLDOWN", "300"))  # seconds
MAX_RPC_RETRIES = int(os.getenv("MAX_RPC_RETRIES", "3"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.7"))  # Minimum LLM confidence

# LLM settings
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-v0.1")
MIN_TRAINING_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", "100"))

# Risk parameters
MAX_TOKEN_AGE = int(os.getenv("MAX_TOKEN_AGE", "3600"))  # seconds
MIN_HOLDER_COUNT = int(os.getenv("MIN_HOLDER_COUNT", "100"))

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "trading_bot.log"
ERROR_LOG_FILE = LOG_DIR / "errors.log"

# File paths
TRADE_HISTORY_FILE = DATA_DIR / "trade_history.json"
TOKEN_WATCHLIST_FILE = DATA_DIR / "token_watchlist.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
TRAINING_DATA_FILE = DATA_DIR / "training_data.json"

# Initialize with default settings
settings_dict = {
    # Directories
    "BASE_DIR": BASE_DIR,
    "DATA_DIR": DATA_DIR,
    "LOG_DIR": LOG_DIR,
    "MODEL_DIR": MODEL_DIR,
    
    # API Keys
    "HELIUS_API_KEY": HELIUS_API_KEY,
    "SOLANA_PRIVATE_KEY": SOLANA_PRIVATE_KEY,
    
    # Trading settings
    "SIMULATION_MODE": SIMULATION_MODE,
    "SIMULATION_CAPITAL": SIMULATION_CAPITAL,
    "USE_LOCAL_LLM": USE_LOCAL_LLM,
    
    # Jupiter API settings
    "JUPITER_API_URL": JUPITER_API_URL,
    
    # Network settings
    "RPC_ENDPOINTS": RPC_ENDPOINTS,
    "HELIUS_WEBSOCKET_URL": HELIUS_WEBSOCKET_URL,
    
    # Market parameters
    "MIN_LIQUIDITY": MIN_LIQUIDITY,
    "MAX_PRICE_IMPACT": MAX_PRICE_IMPACT,
    "TARGET_PROFIT": TARGET_PROFIT,
    "STOP_LOSS": STOP_LOSS,
    "DAILY_LOSS_LIMIT": DAILY_LOSS_LIMIT,
    
    # Position sizing
    "DEFAULT_POSITION_SIZE": DEFAULT_POSITION_SIZE,
    "MIN_POSITION_SIZE": MIN_POSITION_SIZE,
    "MAX_POSITION_SIZE": MAX_POSITION_SIZE,
    
    # Trading bot parameters
    "LOOP_INTERVAL": LOOP_INTERVAL,
    "TRADE_COOLDOWN": TRADE_COOLDOWN,
    "MAX_RPC_RETRIES": MAX_RPC_RETRIES,
    "MIN_CONFIDENCE": MIN_CONFIDENCE,
    
    # LLM settings
    "LLM_MODEL": LLM_MODEL,
    "MIN_TRAINING_SAMPLES": MIN_TRAINING_SAMPLES,
    
    # Risk parameters
    "MAX_TOKEN_AGE": MAX_TOKEN_AGE,
    "MIN_HOLDER_COUNT": MIN_HOLDER_COUNT,
    
    # Logging settings
    "LOG_LEVEL": LOG_LEVEL,
    "LOG_FILE": LOG_FILE,
    "ERROR_LOG_FILE": ERROR_LOG_FILE,
    
    # File paths
    "TRADE_HISTORY_FILE": TRADE_HISTORY_FILE,
    "TOKEN_WATCHLIST_FILE": TOKEN_WATCHLIST_FILE,
    "PORTFOLIO_FILE": PORTFOLIO_FILE,
    "TRAINING_DATA_FILE": TRAINING_DATA_FILE,
}

# Load custom settings from config file if it exists
CONFIG_FILE = BASE_DIR / "config.json"
if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE, "r") as f:
            custom_settings = json.load(f)
        settings_dict.update(custom_settings)
        print(f"Loaded custom settings from {CONFIG_FILE}")
    except Exception as e:
        print(f"Error loading custom settings: {e}")

# Create a module-level 'settings' variable accessible as settings.SETTING_NAME
class Settings:
    def __init__(self, settings_dict):
        for key, value in settings_dict.items():
            setattr(self, key, value)

settings = Settings(settings_dict) 
"""
Core configuration settings for the Enhanced Ant Bot System.
"""

import os
from typing import Dict, Any, Optional

# Default configuration values
DEFAULT_CONFIG = {
    'trading': {
        'enabled': True,
        'max_slippage': 0.02,
        'min_liquidity': 10000,
        'position_size_pct': 0.02,
        'stop_loss_pct': 0.15,
        'target_profit_pct': 3.0
    },
    'risk_management': {
        'max_position_size': 1000,
        'stop_loss_percentage': 0.05,
        'max_daily_loss': 0.1,
        'emergency_stop_loss': 0.25
    },
    'monitoring': {
        'discord_notifications': True,
        'log_level': 'INFO',
        'metrics_interval': 60,
        'health_check_interval': 30
    },
    'ai': {
        'model': 'enhanced',
        'confidence_threshold': 0.7,
        'learning_rate': 0.001,
        'update_interval': 300
    },
    'services': {
        'helius_enabled': True,
        'quicknode_enabled': True,
        'pump_fun_monitoring': True,
        'smart_money_tracking': True
    }
}

def get_config() -> Dict[str, Any]:
    """Get the current configuration."""
    return DEFAULT_CONFIG.copy()

def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values."""
    config = get_config()
    _deep_update(config, updates)
    return config

def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Deep update a dictionary."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value

# Environment-specific overrides
def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables."""
    env_config = {}
    
    # Trading configuration
    if os.getenv('TRADING_ENABLED'):
        env_config.setdefault('trading', {})['enabled'] = os.getenv('TRADING_ENABLED').lower() == 'true'
    
    if os.getenv('MAX_SLIPPAGE'):
        env_config.setdefault('trading', {})['max_slippage'] = float(os.getenv('MAX_SLIPPAGE'))
    
    # Risk management configuration
    if os.getenv('MAX_POSITION_SIZE'):
        env_config.setdefault('risk_management', {})['max_position_size'] = float(os.getenv('MAX_POSITION_SIZE'))
    
    if os.getenv('STOP_LOSS_PERCENTAGE'):
        env_config.setdefault('risk_management', {})['stop_loss_percentage'] = float(os.getenv('STOP_LOSS_PERCENTAGE'))
    
    # Monitoring configuration
    if os.getenv('LOG_LEVEL'):
        env_config.setdefault('monitoring', {})['log_level'] = os.getenv('LOG_LEVEL')
    
    if os.getenv('DISCORD_NOTIFICATIONS'):
        env_config.setdefault('monitoring', {})['discord_notifications'] = os.getenv('DISCORD_NOTIFICATIONS').lower() == 'true'
    
    return env_config

# Export commonly used values
TRADING_ENABLED = DEFAULT_CONFIG['trading']['enabled']
MAX_SLIPPAGE = DEFAULT_CONFIG['trading']['max_slippage']
MIN_LIQUIDITY = DEFAULT_CONFIG['trading']['min_liquidity']
MAX_POSITION_SIZE = DEFAULT_CONFIG['risk_management']['max_position_size']
STOP_LOSS_PERCENTAGE = DEFAULT_CONFIG['risk_management']['stop_loss_percentage']
LOG_LEVEL = DEFAULT_CONFIG['monitoring']['log_level']

# Additional config exports for core module
CORE_CONFIG = DEFAULT_CONFIG
MARKET_CONFIG = DEFAULT_CONFIG['trading']
TRADING_CONFIG = DEFAULT_CONFIG['trading'] 
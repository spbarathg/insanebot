#!/usr/bin/env python3
"""
Setup script for Solana Trading Bot environment
"""
import os
import sys
import json
import logging
from pathlib import Path

def setup_directories():
    """Create required directories"""
    directories = [
        "data",
        "logs",
        "config",
        "models",
        "data/market",
        "data/trades",
        "data/wallets",
        "logs/api",
        "logs/trading",
        "logs/monitoring"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def setup_logging():
    """Set up logging configuration"""
    log_files = [
        "logs/api.log",
        "logs/errors.log",
        "logs/trades.log",
        "logs/debug.log",
        "logs/trading_bot.log",
        "logs/whale_tracker.log",
        "logs/alert.log",
        "logs/llm.log",
        "logs/market.log",
        "logs/monitoring.log",
        "logs/system.log",
        "logs/wallet.log"
    ]
    
    for log_file in log_files:
        Path(log_file).touch()
        print(f"Created log file: {log_file}")

def setup_config():
    """Create default configuration files"""
    config = {
        "trading": {
            "min_liquidity": 1.0,
            "max_price_impact": 0.05,
            "target_profit": 10.0,
            "stop_loss": 5.0,
            "daily_loss_limit": 1.0,
            "default_position_size": 0.01,
            "min_position_size": 0.001,
            "max_position_size": 0.1,
            "loop_interval": 30,
            "trade_cooldown": 300,
            "max_rpc_retries": 3,
            "min_confidence": 0.7
        },
        "system": {
            "max_system_load": 80,
            "throttle_trades_at_load": 70,
            "llm_inference_timeout": 5,
            "batch_size": 10,
            "scheduled_training_hour": 2
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    with open("config/config.json", "w") as f:
        json.dump(config, f, indent=4)
    print("Created config/config.json")

def main():
    """Main setup function"""
    print("Setting up Solana Trading Bot environment...")
    
    # Create directories
    setup_directories()
    
    # Set up logging
    setup_logging()
    
    # Create config files
    setup_config()
    
    print("\nEnvironment setup complete!")
    print("You can now run the bot with: python run_bot.py")

if __name__ == "__main__":
    main() 
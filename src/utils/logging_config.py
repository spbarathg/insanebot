"""
Logging configuration for the Solana trading bot.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import json
from datetime import datetime

# Default logging configuration
DEFAULT_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": "logs/trading_bot.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        },
        "error_file": {
            "level": "ERROR",
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": "logs/errors.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 3
        }
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file", "error_file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

class LoggingConfig:
    """Manage logging configuration for the trading bot."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize logging configuration."""
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self, level: str = "INFO", enable_json: bool = False) -> None:
        """
        Setup logging configuration.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_json: Whether to enable JSON logging format
        """
        # Remove default loguru handler
        logger.remove()
        
        # Setup console logging
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        
        logger.add(
            sys.stderr,
            format=console_format,
            level=level.upper(),
            colorize=True
        )
        
        # Setup file logging
        if enable_json:
            file_format = self._json_formatter
        else:
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
            )
        
        # Main log file
        logger.add(
            self.log_dir / "trading_bot.log",
            format=file_format,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
        
        # Error log file
        logger.add(
            self.log_dir / "errors.log",
            format=file_format,
            level="ERROR",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
        
        # Trade log file
        logger.add(
            self.log_dir / "trades.log",
            format=file_format,
            level="INFO",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            filter=lambda record: "TRADE" in record["extra"]
        )
        
    def _json_formatter(self, record):
        """Custom JSON formatter for structured logging."""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "module": record["module"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
            "extra": record["extra"]
        }
        
        if record["exception"]:
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback
            }
            
        return json.dumps(log_entry)
    
    def configure_external_loggers(self, level: str = "WARNING") -> None:
        """Configure external library loggers to reduce noise."""
        # Reduce noise from external libraries
        external_loggers = [
            "aiohttp",
            "urllib3",
            "asyncio",
            "solana",
            "websockets",
            "httpx"
        ]
        
        for logger_name in external_loggers:
            logging.getLogger(logger_name).setLevel(getattr(logging, level.upper()))
    
    def get_trade_logger(self):
        """Get a specialized logger for trade events."""
        trade_logger = logger.bind(TRADE=True)
        return trade_logger

# Global logging setup function
def setup_logging(
    level: str = "INFO", 
    log_dir: Optional[Path] = None,
    enable_json: bool = False
) -> LoggingConfig:
    """
    Setup global logging configuration.
    
    Args:
        level: Logging level
        log_dir: Directory for log files
        enable_json: Enable JSON logging format
        
    Returns:
        LoggingConfig instance
    """
    config = LoggingConfig(log_dir)
    config.setup_logging(level, enable_json)
    config.configure_external_loggers()
    
    logger.info(f"Logging initialized with level: {level}")
    return config

# For backwards compatibility
def init_logging(level: str = "INFO") -> None:
    """Initialize logging with basic configuration."""
    setup_logging(level) 
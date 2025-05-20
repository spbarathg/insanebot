"""
Logging configuration for Solana trading bot.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
import traceback
import functools
import time
from typing import Callable, Any
from pathlib import Path
from .config import settings

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Create formatters
formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

# Create handlers
def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create loggers for different components
trade_logger = setup_logger('trade', 'logs/trade.log')
error_logger = setup_logger('error', 'logs/error.log', level=logging.ERROR)
system_logger = setup_logger('system', 'logs/system.log')
market_logger = setup_logger('market', 'logs/market.log')
wallet_logger = setup_logger('wallet', 'logs/wallet.log')
llm_logger = setup_logger('llm', 'logs/llm.log')
monitoring_logger = setup_logger('monitoring', 'logs/monitoring.log')
alert_logger = setup_logger('alert', 'logs/alert.log')
whale_logger = setup_logger('whale_tracker', 'logs/whale_tracker.log')
debug_logger = setup_logger('debug', 'logs/debug.log', level=logging.DEBUG)

# Custom exception classes
class TradingError(Exception):
    """Base exception for trading-related errors"""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details
        error_logger.error(f"Trading Error: {message}", extra={
            'details': details,
            'traceback': traceback.format_exc()
        })

class MarketDataError(Exception):
    """Exception for market data related errors"""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details
        error_logger.error(f"Market Data Error: {message}", extra={
            'details': details,
            'traceback': traceback.format_exc()
        })

class WalletError(Exception):
    """Exception for wallet-related errors"""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details
        error_logger.error(f"Wallet Error: {message}", extra={
            'details': details,
            'traceback': traceback.format_exc()
        })

class LLMError(Exception):
    """Exception for LLM-related errors"""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details
        error_logger.error(f"LLM Error: {message}", extra={
            'details': details,
            'traceback': traceback.format_exc()
        })

class NetworkError(Exception):
    """Custom exception for network-related errors"""
    pass

class MonitoringError(Exception):
    """Exception for monitoring-related errors"""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details
        error_logger.error(f"Monitoring Error: {message}", extra={
            'details': details,
            'traceback': traceback.format_exc()
        })

# Error handling decorator
def handle_errors(logger: logging.Logger) -> Callable:
    """Decorator to handle errors and log them"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator

# Logging decorator for performance monitoring
def log_performance(logger: logging.Logger) -> Callable:
    """Decorator to log function performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            logger.debug(
                f"Function {func.__name__} took {end_time - start_time:.2f} seconds",
                extra={
                    'function': func.__name__,
                    'execution_time': end_time - start_time,
                    'timestamp': datetime.now().isoformat()
                }
            )
            return result
        return wrapper
    return decorator

# Logging decorator for async functions
def log_async_performance(logger: logging.Logger) -> Callable:
    """Decorator to log async function performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            
            logger.debug(
                f"Async function {func.__name__} took {end_time - start_time:.2f} seconds",
                extra={
                    'function': func.__name__,
                    'execution_time': end_time - start_time,
                    'timestamp': datetime.now().isoformat()
                }
            )
            return result
        return wrapper
    return decorator

# Initialize log directories
def initialize_logging():
    """Initialize logging directories and files."""
    try:
        # Create log directory if it doesn't exist
        if not settings.LOG_DIR.exists():
            settings.LOG_DIR.mkdir(parents=True)
            
        # Create an empty log file if it doesn't exist
        if not settings.LOG_FILE.exists():
            with open(settings.LOG_FILE, "w") as f:
                f.write("")
                
        if not settings.ERROR_LOG_FILE.exists():
            with open(settings.ERROR_LOG_FILE, "w") as f:
                f.write("")
                
        # Create other log files
        for log_file in ["wallet.log", "trades.log", "api.log"]:
            log_path = settings.LOG_DIR / log_file
            if not log_path.exists():
                with open(log_path, "w") as f:
                    f.write("")
                    
        logging.info("Logging initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing logging: {str(e)}")
        return False

# Initialize logging when module is imported
initialize_logging() 
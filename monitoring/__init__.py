"""
Trading Bot Monitoring Package

This package provides comprehensive monitoring and analysis tools for trading bots.
"""

__version__ = "1.0.0"
__author__ = "Trading Bot Monitor"

# Import main components for easy access
try:
    from .trading_bot_monitor import TradingBotMonitor
    from .log_analyzer import LogAnalyzer
    from .integrate_with_bot import (
        TradingBotLogger,
        get_trading_logger,
        log_decision,
        log_execution,
        log_performance,
        init_monitoring
    )
except ImportError:
    # Handle case where dependencies might not be installed
    pass

__all__ = [
    'TradingBotMonitor',
    'LogAnalyzer', 
    'TradingBotLogger',
    'get_trading_logger',
    'log_decision',
    'log_execution',
    'log_performance',
    'init_monitoring'
] 
"""
Trading Engine Module

This module implements the core trading functionality for the Ant Bot Ultimate Bot.
Provides trading execution, compounding logic, risk management, and market analysis.
"""

from .compounding_logic import CompoundingLogic

# Import other modules if they exist, otherwise create stubs
try:
    from .trade_executor import TradeExecutor
except ImportError:
    class TradeExecutor:
        """Stub TradeExecutor for testing"""
        pass

try:
    from .risk_manager import RiskManager
except ImportError:
    class RiskManager:
        """Stub RiskManager for testing"""
        pass

try:
    from .market_analyzer import MarketAnalyzer
except ImportError:
    class MarketAnalyzer:
        """Stub MarketAnalyzer for testing"""
        pass

__all__ = [
    'TradeExecutor',
    'CompoundingLogic', 
    'RiskManager',
    'MarketAnalyzer'
] 
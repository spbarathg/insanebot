"""
Advanced risk management module for trading operations.
"""

from .position_sizer import PositionSizer, PositionSizeResult
from .stop_loss_manager import StopLossManager, StopLossOrder

class RiskViolationError(Exception):
    """Raised when a risk management rule is violated."""
    pass

class RiskManager:
    """Basic risk manager for testing and simple risk management."""
    
    def __init__(self, max_position_size: float = 0.1, max_daily_loss: float = 0.05):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.daily_losses = 0.0
        self.position_sizer = PositionSizer()
        self.stop_loss_manager = StopLossManager()
    
    def validate_trade(self, amount: float, current_balance: float) -> bool:
        """Validate if a trade meets risk management criteria."""
        # Check position size limit
        position_ratio = amount / current_balance
        if position_ratio > self.max_position_size:
            return False
        
        # Check daily loss limit
        if self.daily_losses > self.max_daily_loss:
            return False
        
        return True
    
    def record_loss(self, loss_amount: float):
        """Record a trading loss."""
        self.daily_losses += loss_amount
    
    def reset_daily_losses(self):
        """Reset daily loss counter."""
        self.daily_losses = 0.0
    
    def check_risk_violation(self, amount: float, current_balance: float):
        """Check for risk violations and raise exception if found."""
        if not self.validate_trade(amount, current_balance):
            raise RiskViolationError(f"Trade amount {amount} violates risk limits")

__all__ = [
    'PositionSizer',
    'PositionSizeResult', 
    'StopLossManager',
    'StopLossOrder',
    'RiskManager',
    'RiskViolationError'
] 
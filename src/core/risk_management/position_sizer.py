"""
Position sizing module for calculating optimal trade sizes based on risk parameters.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum
from loguru import logger

class SizingMethod(Enum):
    """Position sizing methods"""
    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"
    ATR_BASED = "atr_based"

@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    recommended_size_sol: float
    recommended_size_usd: float
    risk_percentage: float
    max_loss_sol: float
    max_loss_usd: float
    sizing_method: SizingMethod
    confidence_score: float
    reasoning: str
    warnings: list
    
    @property
    def is_valid(self) -> bool:
        """Check if position size is valid"""
        return (
            self.recommended_size_sol > 0 and
            self.risk_percentage <= 10.0 and  # Max 10% risk
            len(self.warnings) == 0
        )

class PositionSizer:
    """
    Advanced position sizing calculator that determines optimal trade sizes
    based on various risk management strategies.
    """
    
    def __init__(self):
        """Initialize position sizer with default parameters."""
        # Risk parameters
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.max_portfolio_risk = 0.20  # 20% max total portfolio risk
        self.min_position_size = 0.001  # 0.001 SOL minimum
        self.max_position_size = 1.0    # 1 SOL maximum
        
        # Kelly Criterion parameters
        self.kelly_multiplier = 0.25  # Conservative Kelly (25% of full Kelly)
        self.min_win_rate = 0.35      # Minimum win rate to use Kelly
        
        # Volatility parameters
        self.volatility_lookback = 14  # Days for volatility calculation
        self.volatility_multiplier = 2.0  # Volatility adjustment factor
        
        logger.info("PositionSizer initialized")
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        token_price: float,
        stop_loss_percentage: float,
        confidence: float,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        method: SizingMethod = SizingMethod.VOLATILITY_ADJUSTED
    ) -> PositionSizeResult:
        """
        Calculate optimal position size based on specified method.
        
        Args:
            portfolio_value: Current portfolio value in SOL
            token_price: Token price in USD
            stop_loss_percentage: Stop loss as percentage (e.g., 0.05 for 5%)
            confidence: Trading confidence (0-1)
            volatility: Token volatility (optional)
            win_rate: Historical win rate (optional)
            avg_win: Average win amount (optional)
            avg_loss: Average loss amount (optional)
            method: Position sizing method to use
        """
        try:
            warnings = []
            
            # Validate inputs
            if portfolio_value <= 0:
                return self._create_invalid_result("Invalid portfolio value", method)
            
            if stop_loss_percentage <= 0 or stop_loss_percentage >= 1:
                warnings.append("Stop loss percentage should be between 0 and 1")
            
            # Calculate base position size using selected method
            if method == SizingMethod.FIXED_PERCENTAGE:
                size_sol = self._calculate_fixed_percentage(portfolio_value, confidence)
                reasoning = f"Fixed percentage sizing: {self.max_risk_per_trade*100:.1f}% of portfolio"
                
            elif method == SizingMethod.KELLY_CRITERION:
                if win_rate is None or avg_win is None or avg_loss is None:
                    # Fall back to fixed percentage
                    size_sol = self._calculate_fixed_percentage(portfolio_value, confidence)
                    reasoning = "Kelly Criterion fallback to fixed percentage (insufficient data)"
                    warnings.append("Insufficient data for Kelly Criterion, using fixed percentage")
                else:
                    size_sol = self._calculate_kelly_criterion(
                        portfolio_value, win_rate, avg_win, avg_loss, confidence
                    )
                    reasoning = f"Kelly Criterion: win_rate={win_rate:.2f}, avg_win={avg_win:.4f}, avg_loss={avg_loss:.4f}"
                    
            elif method == SizingMethod.VOLATILITY_ADJUSTED:
                if volatility is None:
                    volatility = 0.05  # Default 5% volatility
                    warnings.append("Using default volatility (5%)")
                
                size_sol = self._calculate_volatility_adjusted(
                    portfolio_value, volatility, confidence, stop_loss_percentage
                )
                reasoning = f"Volatility adjusted: vol={volatility:.2f}, confidence={confidence:.2f}"
                
            elif method == SizingMethod.RISK_PARITY:
                size_sol = self._calculate_risk_parity(
                    portfolio_value, stop_loss_percentage, confidence
                )
                reasoning = f"Risk parity: equal risk allocation with {stop_loss_percentage:.1%} stop loss"
                
            else:
                # Default to fixed percentage
                size_sol = self._calculate_fixed_percentage(portfolio_value, confidence)
                reasoning = "Default fixed percentage sizing"
            
            # Apply position size limits
            size_sol = max(self.min_position_size, min(size_sol, self.max_position_size))
            
            # Calculate risk metrics
            max_loss_sol = size_sol * stop_loss_percentage
            risk_percentage = (max_loss_sol / portfolio_value) * 100
            
            # Check risk limits
            if risk_percentage > self.max_risk_per_trade * 100:
                # Reduce position size to meet risk limit
                size_sol = (portfolio_value * self.max_risk_per_trade) / stop_loss_percentage
                max_loss_sol = size_sol * stop_loss_percentage
                risk_percentage = (max_loss_sol / portfolio_value) * 100
                warnings.append(f"Position size reduced to meet {self.max_risk_per_trade*100:.1f}% risk limit")
            
            # Convert to USD
            sol_price_usd = 100.0  # Approximate SOL price
            size_usd = size_sol * sol_price_usd
            max_loss_usd = max_loss_sol * sol_price_usd
            
            # Calculate confidence score for the sizing
            confidence_score = self._calculate_sizing_confidence(
                size_sol, portfolio_value, risk_percentage, confidence, len(warnings)
            )
            
            return PositionSizeResult(
                recommended_size_sol=size_sol,
                recommended_size_usd=size_usd,
                risk_percentage=risk_percentage,
                max_loss_sol=max_loss_sol,
                max_loss_usd=max_loss_usd,
                sizing_method=method,
                confidence_score=confidence_score,
                reasoning=reasoning,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return self._create_invalid_result(f"Calculation error: {str(e)}", method)
    
    def _calculate_fixed_percentage(self, portfolio_value: float, confidence: float) -> float:
        """Calculate position size using fixed percentage method."""
        # Adjust risk based on confidence
        adjusted_risk = self.max_risk_per_trade * confidence
        return portfolio_value * adjusted_risk
    
    def _calculate_kelly_criterion(
        self, 
        portfolio_value: float, 
        win_rate: float, 
        avg_win: float, 
        avg_loss: float,
        confidence: float
    ) -> float:
        """Calculate position size using Kelly Criterion."""
        if win_rate < self.min_win_rate:
            # Not enough edge, use conservative sizing
            return self._calculate_fixed_percentage(portfolio_value, confidence)
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        if avg_loss <= 0:
            return self._calculate_fixed_percentage(portfolio_value, confidence)
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply Kelly multiplier for safety
        kelly_fraction *= self.kelly_multiplier
        
        # Adjust for confidence
        kelly_fraction *= confidence
        
        # Ensure reasonable bounds
        kelly_fraction = max(0.001, min(kelly_fraction, self.max_risk_per_trade))
        
        return portfolio_value * kelly_fraction
    
    def _calculate_volatility_adjusted(
        self, 
        portfolio_value: float, 
        volatility: float, 
        confidence: float,
        stop_loss_percentage: float
    ) -> float:
        """Calculate position size adjusted for volatility."""
        # Base risk allocation
        base_risk = self.max_risk_per_trade * confidence
        
        # Adjust for volatility (higher volatility = smaller position)
        volatility_adjustment = 1 / (1 + volatility * self.volatility_multiplier)
        
        # Adjust for stop loss distance (tighter stop = larger position)
        stop_loss_adjustment = 0.02 / max(stop_loss_percentage, 0.01)  # 2% reference
        stop_loss_adjustment = min(stop_loss_adjustment, 2.0)  # Cap at 2x
        
        adjusted_risk = base_risk * volatility_adjustment * stop_loss_adjustment
        adjusted_risk = min(adjusted_risk, self.max_risk_per_trade)
        
        return portfolio_value * adjusted_risk
    
    def _calculate_risk_parity(
        self, 
        portfolio_value: float, 
        stop_loss_percentage: float,
        confidence: float
    ) -> float:
        """Calculate position size for equal risk allocation."""
        # Target risk amount
        target_risk = portfolio_value * self.max_risk_per_trade * confidence
        
        # Position size to achieve target risk with given stop loss
        position_size = target_risk / stop_loss_percentage
        
        return position_size
    
    def _calculate_sizing_confidence(
        self, 
        position_size: float, 
        portfolio_value: float, 
        risk_percentage: float,
        trade_confidence: float,
        warning_count: int
    ) -> float:
        """Calculate confidence score for the position sizing."""
        base_confidence = 0.8
        
        # Adjust for trade confidence
        base_confidence *= trade_confidence
        
        # Adjust for risk level (lower risk = higher confidence)
        risk_factor = 1 - (risk_percentage / (self.max_risk_per_trade * 100))
        base_confidence *= (0.5 + 0.5 * risk_factor)
        
        # Adjust for position size relative to portfolio
        size_ratio = position_size / portfolio_value
        if size_ratio < 0.001:  # Very small position
            base_confidence *= 0.7
        elif size_ratio > 0.1:  # Large position
            base_confidence *= 0.8
        
        # Reduce confidence for warnings
        base_confidence *= (1 - 0.1 * warning_count)
        
        return max(0.0, min(1.0, base_confidence))
    
    def _create_invalid_result(self, reason: str, method: SizingMethod) -> PositionSizeResult:
        """Create an invalid position size result."""
        return PositionSizeResult(
            recommended_size_sol=0.0,
            recommended_size_usd=0.0,
            risk_percentage=0.0,
            max_loss_sol=0.0,
            max_loss_usd=0.0,
            sizing_method=method,
            confidence_score=0.0,
            reasoning=reason,
            warnings=[reason]
        )
    
    def update_parameters(
        self,
        max_risk_per_trade: Optional[float] = None,
        max_portfolio_risk: Optional[float] = None,
        kelly_multiplier: Optional[float] = None
    ) -> None:
        """Update position sizing parameters."""
        if max_risk_per_trade is not None:
            self.max_risk_per_trade = max_risk_per_trade
            logger.info(f"Updated max risk per trade to {max_risk_per_trade:.2%}")
        
        if max_portfolio_risk is not None:
            self.max_portfolio_risk = max_portfolio_risk
            logger.info(f"Updated max portfolio risk to {max_portfolio_risk:.2%}")
        
        if kelly_multiplier is not None:
            self.kelly_multiplier = kelly_multiplier
            logger.info(f"Updated Kelly multiplier to {kelly_multiplier:.2f}")
    
    def get_recommended_method(
        self, 
        has_historical_data: bool, 
        volatility_available: bool,
        confidence: float
    ) -> SizingMethod:
        """Get recommended sizing method based on available data."""
        if has_historical_data and confidence > 0.7:
            return SizingMethod.KELLY_CRITERION
        elif volatility_available:
            return SizingMethod.VOLATILITY_ADJUSTED
        else:
            return SizingMethod.FIXED_PERCENTAGE 
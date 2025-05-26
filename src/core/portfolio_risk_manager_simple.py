"""
Simplified Portfolio Risk Management System for Solana Trading Bot
This version avoids numpy dependencies to prevent system crashes

Key features:
- Basic portfolio risk monitoring
- Position size validation
- Simple diversification checks
- Drawdown protection
- Compatible interface with existing system
"""

import asyncio
import time
import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level categories"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class SimpleRiskMetrics:
    """Simplified portfolio risk metrics without numpy dependencies"""
    total_value: float
    total_positions: int
    max_drawdown: float
    current_drawdown: float
    portfolio_concentration: float  # Simplified concentration measure
    largest_position_weight: float
    total_unrealized_pnl: float
    average_position_size: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def overall_risk_level(self) -> str:
        """Determine overall portfolio risk level using simple rules"""
        risk_score = 0
        
        # Check concentration risk
        if self.largest_position_weight > 0.5:  # >50% in single position
            risk_score += 3
        elif self.largest_position_weight > 0.3:  # >30% in single position
            risk_score += 2
        elif self.largest_position_weight > 0.2:  # >20% in single position
            risk_score += 1
        
        # Check drawdown
        if self.max_drawdown > 0.3:  # >30% drawdown
            risk_score += 3
        elif self.max_drawdown > 0.2:  # >20% drawdown
            risk_score += 2
        elif self.max_drawdown > 0.1:  # >10% drawdown
            risk_score += 1
        
        # Check portfolio concentration
        if self.portfolio_concentration > 0.8:  # High concentration
            risk_score += 2
        elif self.portfolio_concentration > 0.6:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 6:
            return RiskLevel.EXTREME.value
        elif risk_score >= 4:
            return RiskLevel.HIGH.value
        elif risk_score >= 2:
            return RiskLevel.MEDIUM.value
        else:
            return RiskLevel.LOW.value

@dataclass
class RiskLimits:
    """Simple risk limits without complex calculations"""
    max_single_position: float = 0.25    # 25% max single position
    max_drawdown_limit: float = 0.2      # 20% max drawdown
    min_positions: int = 3               # Minimum diversification
    max_concentration: float = 0.7       # Portfolio concentration limit

class SimplePortfolioRiskManager:
    """Simplified portfolio risk manager without numpy dependencies"""
    
    def __init__(self, portfolio_manager):
        self.portfolio_manager = portfolio_manager
        self.risk_limits = RiskLimits()
        self.risk_history = []
        self.last_assessment = None
        self.performance_history = []
        
        # Initialize tracking variables
        self.peak_value = 0.0
        self.max_historical_drawdown = 0.0
        
        logger.info("Simple Portfolio Risk Manager initialized (numpy-free)")
    
    async def initialize(self) -> bool:
        """Initialize the risk manager"""
        try:
            # Get initial portfolio state
            summary = self.portfolio_manager.get_portfolio_summary()
            self.peak_value = summary.get('current_value', 0.0)
            
            logger.info("✅ Simple Risk Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing Simple Risk Manager: {str(e)}")
            return False
    
    async def assess_portfolio_risk(self, token_prices: Dict[str, float] = None) -> SimpleRiskMetrics:
        """Assess portfolio risk using simple calculations"""
        try:
            # Get current portfolio data
            summary = self.portfolio_manager.get_portfolio_summary()
            holdings = self.portfolio_manager.get_holdings()
            
            total_value = summary.get('current_value', 0.0)
            total_positions = len(holdings)
            
            # Calculate drawdown
            if total_value > self.peak_value:
                self.peak_value = total_value
            
            current_drawdown = 0.0
            if self.peak_value > 0:
                current_drawdown = (self.peak_value - total_value) / self.peak_value
            
            max_drawdown = max(self.max_historical_drawdown, current_drawdown)
            self.max_historical_drawdown = max_drawdown
            
            # Calculate position weights and concentration
            largest_position_weight = 0.0
            position_weights = []
            
            if total_value > 0 and holdings:
                for holding in holdings:
                    position_value = holding.get('current_value_sol', 0)
                    weight = position_value / total_value
                    position_weights.append(weight)
                    largest_position_weight = max(largest_position_weight, weight)
            
            # Simple concentration calculation (Herfindahl-like index)
            portfolio_concentration = sum(w * w for w in position_weights) if position_weights else 1.0
            
            # Average position size
            average_position_size = sum(position_weights) / len(position_weights) if position_weights else 0.0
            
            # Create risk metrics
            risk_metrics = SimpleRiskMetrics(
                total_value=total_value,
                total_positions=total_positions,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                portfolio_concentration=portfolio_concentration,
                largest_position_weight=largest_position_weight,
                total_unrealized_pnl=summary.get('unrealized_profit', 0.0),
                average_position_size=average_position_size
            )
            
            # Store assessment
            self.last_assessment = risk_metrics
            self.risk_history.append({
                'timestamp': time.time(),
                'risk_level': risk_metrics.overall_risk_level,
                'total_value': total_value,
                'drawdown': current_drawdown
            })
            
            # Keep only last 100 assessments
            if len(self.risk_history) > 100:
                self.risk_history = self.risk_history[-100:]
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {str(e)}")
            # Return safe default metrics
            return SimpleRiskMetrics(
                total_value=0.0,
                total_positions=0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                portfolio_concentration=0.0,
                largest_position_weight=0.0,
                total_unrealized_pnl=0.0,
                average_position_size=0.0
            )
    
    async def check_risk_violations(self, risk_metrics: SimpleRiskMetrics) -> List[str]:
        """Check for risk limit violations"""
        violations = []
        
        try:
            # Check single position concentration
            if risk_metrics.largest_position_weight > self.risk_limits.max_single_position:
                violations.append(
                    f"Single position too large: {risk_metrics.largest_position_weight:.1%} "
                    f"(limit: {self.risk_limits.max_single_position:.1%})"
                )
            
            # Check portfolio drawdown
            if risk_metrics.max_drawdown > self.risk_limits.max_drawdown_limit:
                violations.append(
                    f"Maximum drawdown exceeded: {risk_metrics.max_drawdown:.1%} "
                    f"(limit: {self.risk_limits.max_drawdown_limit:.1%})"
                )
            
            # Check minimum diversification
            if risk_metrics.total_positions < self.risk_limits.min_positions and risk_metrics.total_positions > 0:
                violations.append(
                    f"Insufficient diversification: {risk_metrics.total_positions} positions "
                    f"(minimum: {self.risk_limits.min_positions})"
                )
            
            # Check portfolio concentration
            if risk_metrics.portfolio_concentration > self.risk_limits.max_concentration:
                violations.append(
                    f"Portfolio too concentrated: {risk_metrics.portfolio_concentration:.2f} "
                    f"(limit: {self.risk_limits.max_concentration:.2f})"
                )
            
        except Exception as e:
            logger.error(f"Error checking risk violations: {str(e)}")
            violations.append(f"Error in risk check: {str(e)}")
        
        return violations
    
    def get_risk_stats(self) -> Dict[str, Any]:
        """Get current risk statistics"""
        try:
            stats = {
                'last_assessment': self.last_assessment.__dict__ if self.last_assessment else None,
                'peak_value': self.peak_value,
                'max_historical_drawdown': self.max_historical_drawdown,
                'risk_history_length': len(self.risk_history),
                'risk_limits': {
                    'max_single_position': self.risk_limits.max_single_position,
                    'max_drawdown_limit': self.risk_limits.max_drawdown_limit,
                    'min_positions': self.risk_limits.min_positions,
                    'max_concentration': self.risk_limits.max_concentration
                }
            }
            
            # Add recent risk levels if available
            if self.risk_history:
                recent_risks = [entry['risk_level'] for entry in self.risk_history[-10:]]
                stats['recent_risk_levels'] = recent_risks
                stats['current_risk_level'] = recent_risks[-1] if recent_risks else 'unknown'
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting risk stats: {str(e)}")
            return {'error': str(e)}
    
    def should_reduce_risk(self) -> bool:
        """Simple check if risk should be reduced"""
        if not self.last_assessment:
            return False
        
        return (
            self.last_assessment.overall_risk_level in [RiskLevel.HIGH.value, RiskLevel.EXTREME.value] or
            self.last_assessment.max_drawdown > 0.15 or  # >15% drawdown
            self.last_assessment.largest_position_weight > 0.4  # >40% in single position
        )
    
    def get_position_size_recommendation(self, token_address: str, target_allocation: float = 0.1) -> float:
        """Get recommended position size based on current portfolio"""
        try:
            summary = self.portfolio_manager.get_portfolio_summary()
            total_value = summary.get('current_value', 0.0)
            
            if total_value <= 0:
                return 0.0
            
            # Limit position size based on risk limits
            max_position_value = total_value * self.risk_limits.max_single_position
            target_position_value = total_value * target_allocation
            
            recommended_value = min(max_position_value, target_position_value)
            return recommended_value
            
        except Exception as e:
            logger.error(f"Error calculating position size recommendation: {str(e)}")
            return 0.0
    
    async def close(self):
        """Clean up resources"""
        try:
            # Save risk history if needed
            if self.risk_history:
                logger.info(f"Simple Risk Manager closing - {len(self.risk_history)} risk assessments completed")
            
        except Exception as e:
            logger.error(f"Error closing Simple Risk Manager: {str(e)}")

# Alias for compatibility with existing code
PortfolioRiskManager = SimplePortfolioRiskManager 
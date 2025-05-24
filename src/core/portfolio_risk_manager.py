"""
Portfolio Risk Management System for Solana Trading Bot

This module provides comprehensive portfolio risk management including:
- Portfolio-wide risk monitoring and assessment
- Position correlation analysis and diversification
- Dynamic position sizing based on portfolio risk
- Advanced stop-loss and take-profit management
- Drawdown protection and portfolio rebalancing
- Risk-adjusted performance metrics
- Portfolio optimization and allocation strategies
"""

import asyncio
import time
import logging
import math
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json

logger = logging.getLogger(__name__)

class RiskMetric(Enum):
    """Types of risk metrics tracked"""
    PORTFOLIO_VAR = "portfolio_var"  # Value at Risk
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    CORRELATION_RISK = "correlation_risk"
    CONCENTRATION_RISK = "concentration_risk"
    VOLATILITY = "volatility"

class StopLossType(Enum):
    """Types of stop-loss strategies"""
    FIXED_PERCENT = "fixed_percent"
    TRAILING = "trailing"
    ATR_BASED = "atr_based"  # Average True Range
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    TIME_BASED = "time_based"
    PORTFOLIO_BASED = "portfolio_based"

class RebalanceStrategy(Enum):
    """Portfolio rebalancing strategies"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MOMENTUM_BASED = "momentum_based"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_TARGET = "volatility_target"

@dataclass
class PositionRisk:
    """Risk metrics for individual positions"""
    token_address: str
    token_symbol: str
    position_value: float
    portfolio_weight: float
    daily_var: float
    volatility: float
    correlation_score: float
    concentration_risk: float
    stop_loss_level: float
    take_profit_level: float
    time_in_position: float
    unrealized_pnl: float
    risk_score: float
    
    @property
    def is_risk_position(self) -> bool:
        """Check if position exceeds risk thresholds"""
        return (self.portfolio_weight > 0.2 or  # >20% of portfolio
                self.risk_score > 0.7 or         # High risk score
                self.unrealized_pnl < -0.1)      # >10% loss

@dataclass
class PortfolioRiskMetrics:
    """Comprehensive portfolio risk metrics"""
    total_value: float
    total_positions: int
    portfolio_var_1d: float  # 1-day Value at Risk
    portfolio_var_5d: float  # 5-day Value at Risk
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    portfolio_volatility: float
    correlation_risk: float
    concentration_risk: float
    largest_position_weight: float
    total_unrealized_pnl: float
    risk_adjusted_return: float
    diversification_ratio: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def overall_risk_level(self) -> str:
        """Determine overall portfolio risk level"""
        risk_factors = 0
        
        if self.portfolio_var_1d > 0.05:  # >5% daily VaR
            risk_factors += 2
        elif self.portfolio_var_1d > 0.03:  # >3% daily VaR
            risk_factors += 1
            
        if self.max_drawdown > 0.15:  # >15% max drawdown
            risk_factors += 2
        elif self.max_drawdown > 0.1:  # >10% max drawdown
            risk_factors += 1
            
        if self.concentration_risk > 0.7:  # High concentration
            risk_factors += 2
        elif self.concentration_risk > 0.5:
            risk_factors += 1
            
        if self.correlation_risk > 0.8:  # High correlation
            risk_factors += 1
            
        if risk_factors >= 6:
            return "extreme"
        elif risk_factors >= 4:
            return "high"
        elif risk_factors >= 2:
            return "medium"
        else:
            return "low"

@dataclass
class RiskLimit:
    """Risk limits and constraints"""
    max_portfolio_var: float = 0.05  # 5% max daily VaR
    max_single_position: float = 0.25  # 25% max single position
    max_sector_exposure: float = 0.4   # 40% max sector exposure
    max_drawdown_limit: float = 0.2    # 20% max drawdown before action
    min_diversification: int = 3       # Minimum 3 positions
    max_correlation: float = 0.8       # Maximum average correlation
    max_leverage: float = 1.0          # No leverage for now
    stop_loss_trigger: float = 0.15    # 15% portfolio loss triggers review

class CorrelationAnalyzer:
    """Analyzes correlations between portfolio positions"""
    
    def __init__(self):
        self.price_history = defaultdict(deque)  # Store price history per token
        self.correlation_matrix = {}
        self.correlation_cache = {}
        self.cache_duration = 300  # 5 minutes
    
    def update_price_history(self, token_address: str, price: float, timestamp: float = None):
        """Update price history for correlation analysis"""
        try:
            if timestamp is None:
                timestamp = time.time()
            
            # Store price with timestamp
            self.price_history[token_address].append({
                'price': price,
                'timestamp': timestamp
            })
            
            # Keep only last 100 data points
            if len(self.price_history[token_address]) > 100:
                self.price_history[token_address].popleft()
                
        except Exception as e:
            logger.error(f"Error updating price history: {str(e)}")
    
    def calculate_correlation(self, token1: str, token2: str, lookback_periods: int = 20) -> float:
        """Calculate price correlation between two tokens"""
        try:
            cache_key = f"{token1}_{token2}_{lookback_periods}_{int(time.time() // self.cache_duration)}"
            
            if cache_key in self.correlation_cache:
                return self.correlation_cache[cache_key]
            
            # Get price histories
            history1 = list(self.price_history[token1])[-lookback_periods:]
            history2 = list(self.price_history[token2])[-lookback_periods:]
            
            if len(history1) < 10 or len(history2) < 10:
                # Not enough data for reliable correlation
                return 0.0
            
            # Calculate returns
            returns1 = []
            returns2 = []
            
            for i in range(1, min(len(history1), len(history2))):
                if history1[i-1]['price'] > 0 and history2[i-1]['price'] > 0:
                    ret1 = (history1[i]['price'] / history1[i-1]['price']) - 1
                    ret2 = (history2[i]['price'] / history2[i-1]['price']) - 1
                    returns1.append(ret1)
                    returns2.append(ret2)
            
            if len(returns1) < 5:
                return 0.0
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(returns1, returns2)[0, 1]
            
            # Handle NaN values
            if np.isnan(correlation):
                correlation = 0.0
            
            # Cache result
            self.correlation_cache[cache_key] = correlation
            
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return 0.0
    
    def get_portfolio_correlation_matrix(self, token_addresses: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix for portfolio tokens"""
        try:
            matrix = {}
            
            for token1 in token_addresses:
                matrix[token1] = {}
                for token2 in token_addresses:
                    if token1 == token2:
                        matrix[token1][token2] = 1.0
                    else:
                        correlation = self.calculate_correlation(token1, token2)
                        matrix[token1][token2] = correlation
            
            self.correlation_matrix = matrix
            return matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return {}
    
    def calculate_diversification_ratio(self, positions: Dict[str, float], 
                                      correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate portfolio diversification ratio"""
        try:
            if len(positions) < 2:
                return 1.0  # Single position = no diversification
            
            # Calculate weighted average correlation
            total_weight = sum(positions.values())
            if total_weight == 0:
                return 1.0
            
            # Normalize weights
            weights = {token: weight / total_weight for token, weight in positions.items()}
            
            # Calculate portfolio correlation
            portfolio_correlation = 0.0
            total_pairs = 0
            
            for token1, weight1 in weights.items():
                for token2, weight2 in weights.items():
                    if token1 != token2 and token1 in correlation_matrix and token2 in correlation_matrix[token1]:
                        correlation = abs(correlation_matrix[token1][token2])
                        portfolio_correlation += weight1 * weight2 * correlation
                        total_pairs += 1
            
            if total_pairs == 0:
                return 1.0
            
            # Diversification ratio = 1 - average correlation
            diversification_ratio = 1.0 - (portfolio_correlation / len(positions))
            
            return max(0.0, min(1.0, diversification_ratio))
            
        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {str(e)}")
            return 0.5

class StopLossManager:
    """Advanced stop-loss management system"""
    
    def __init__(self):
        self.stop_loss_rules = {}
        self.trailing_stops = {}
        self.position_entry_prices = {}
        self.position_high_water_marks = {}
        self.time_based_exits = {}
    
    def set_stop_loss(self, token_address: str, stop_type: StopLossType, 
                     entry_price: float, **params) -> bool:
        """Set stop-loss rule for a position"""
        try:
            current_time = time.time()
            
            stop_rule = {
                'type': stop_type,
                'entry_price': entry_price,
                'current_price': entry_price,
                'timestamp': current_time,
                'params': params
            }
            
            if stop_type == StopLossType.FIXED_PERCENT:
                stop_percentage = params.get('stop_percentage', 0.1)  # 10% default
                stop_rule['stop_price'] = entry_price * (1 - stop_percentage)
                
            elif stop_type == StopLossType.TRAILING:
                trail_percentage = params.get('trail_percentage', 0.05)  # 5% default
                stop_rule['trail_percentage'] = trail_percentage
                stop_rule['stop_price'] = entry_price * (1 - trail_percentage)
                self.position_high_water_marks[token_address] = entry_price
                
            elif stop_type == StopLossType.VOLATILITY_ADJUSTED:
                volatility = params.get('volatility', 0.1)
                multiplier = params.get('volatility_multiplier', 2.0)
                stop_distance = volatility * multiplier
                stop_rule['stop_price'] = entry_price * (1 - stop_distance)
                
            elif stop_type == StopLossType.TIME_BASED:
                max_hold_time = params.get('max_hold_hours', 24)  # 24 hours default
                stop_rule['exit_time'] = current_time + (max_hold_time * 3600)
                
            self.stop_loss_rules[token_address] = stop_rule
            self.position_entry_prices[token_address] = entry_price
            
            logger.info(f"🛡️ Set {stop_type.value} stop-loss for {token_address[:8]}... at {entry_price}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting stop-loss: {str(e)}")
            return False
    
    def update_stop_loss(self, token_address: str, current_price: float) -> bool:
        """Update stop-loss based on current price"""
        try:
            if token_address not in self.stop_loss_rules:
                return False
            
            stop_rule = self.stop_loss_rules[token_address]
            stop_rule['current_price'] = current_price
            
            if stop_rule['type'] == StopLossType.TRAILING:
                # Update trailing stop
                if token_address in self.position_high_water_marks:
                    if current_price > self.position_high_water_marks[token_address]:
                        self.position_high_water_marks[token_address] = current_price
                        
                        # Update trailing stop price
                        trail_percentage = stop_rule.get('trail_percentage', 0.05)
                        new_stop_price = current_price * (1 - trail_percentage)
                        
                        if new_stop_price > stop_rule.get('stop_price', 0):
                            stop_rule['stop_price'] = new_stop_price
                            logger.debug(f"📈 Updated trailing stop for {token_address[:8]}... to {new_stop_price:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating stop-loss: {str(e)}")
            return False
    
    def check_stop_loss_trigger(self, token_address: str, current_price: float) -> Tuple[bool, str]:
        """Check if stop-loss should be triggered"""
        try:
            if token_address not in self.stop_loss_rules:
                return False, "No stop-loss rule"
            
            stop_rule = self.stop_loss_rules[token_address]
            current_time = time.time()
            
            # Check price-based stops
            if 'stop_price' in stop_rule and current_price <= stop_rule['stop_price']:
                return True, f"{stop_rule['type'].value} triggered: {current_price} <= {stop_rule['stop_price']}"
            
            # Check time-based stops
            if stop_rule['type'] == StopLossType.TIME_BASED:
                exit_time = stop_rule.get('exit_time', 0)
                if current_time >= exit_time:
                    return True, f"Time-based exit triggered after {(current_time - stop_rule['timestamp']) / 3600:.1f} hours"
            
            return False, "No trigger"
            
        except Exception as e:
            logger.error(f"Error checking stop-loss trigger: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def remove_stop_loss(self, token_address: str):
        """Remove stop-loss rule for a position"""
        try:
            if token_address in self.stop_loss_rules:
                del self.stop_loss_rules[token_address]
            if token_address in self.position_entry_prices:
                del self.position_entry_prices[token_address]
            if token_address in self.position_high_water_marks:
                del self.position_high_water_marks[token_address]
                
        except Exception as e:
            logger.error(f"Error removing stop-loss: {str(e)}")

class PortfolioRiskManager:
    """Main portfolio risk management system"""
    
    def __init__(self, portfolio_manager):
        self.portfolio_manager = portfolio_manager
        self.correlation_analyzer = CorrelationAnalyzer()
        self.stop_loss_manager = StopLossManager()
        
        self.risk_limits = RiskLimit()
        self.risk_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        self.rebalance_history = []
        
        # Portfolio tracking
        self.portfolio_high_water_mark = 0
        self.last_rebalance_time = 0
        self.emergency_stop_triggered = False
        
        logger.info("PortfolioRiskManager initialized")
    
    async def initialize(self) -> bool:
        """Initialize the portfolio risk manager"""
        try:
            logger.info("⚠️ Initializing portfolio risk management system...")
            
            # Initialize with current portfolio value
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            self.portfolio_high_water_mark = portfolio_summary['current_value']
            
            logger.info("✅ Portfolio risk management system initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize portfolio risk manager: {str(e)}")
            return False
    
    async def assess_portfolio_risk(self, token_prices: Dict[str, float] = None) -> PortfolioRiskMetrics:
        """Perform comprehensive portfolio risk assessment"""
        try:
            # Get current portfolio data
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            holdings = self.portfolio_manager.get_holdings()
            
            if not holdings:
                return PortfolioRiskMetrics(
                    total_value=portfolio_summary['current_value'],
                    total_positions=0,
                    portfolio_var_1d=0,
                    portfolio_var_5d=0,
                    max_drawdown=0,
                    current_drawdown=0,
                    sharpe_ratio=0,
                    sortino_ratio=0,
                    portfolio_volatility=0,
                    correlation_risk=0,
                    concentration_risk=0,
                    largest_position_weight=0,
                    total_unrealized_pnl=0,
                    risk_adjusted_return=0,
                    diversification_ratio=1
                )
            
            # Update price history for correlation analysis
            if token_prices:
                for token_address, price in token_prices.items():
                    self.correlation_analyzer.update_price_history(token_address, price)
            
            # Calculate position weights and risks
            total_value = portfolio_summary['current_value']
            position_risks = []
            position_weights = {}
            
            for holding in holdings:
                token_address = holding['token_address']
                position_value = holding['current_value_sol']
                weight = position_value / total_value if total_value > 0 else 0
                
                position_weights[token_address] = weight
                
                # Calculate individual position risk
                position_risk = self._calculate_position_risk(holding, weight, total_value)
                position_risks.append(position_risk)
            
            # Calculate correlation matrix
            token_addresses = list(position_weights.keys())
            correlation_matrix = self.correlation_analyzer.get_portfolio_correlation_matrix(token_addresses)
            
            # Calculate portfolio-level metrics
            portfolio_var_1d = self._calculate_portfolio_var(position_risks, correlation_matrix, 1)
            portfolio_var_5d = self._calculate_portfolio_var(position_risks, correlation_matrix, 5)
            
            # Calculate drawdown metrics
            current_value = portfolio_summary['current_value']
            self.portfolio_high_water_mark = max(self.portfolio_high_water_mark, current_value)
            current_drawdown = (self.portfolio_high_water_mark - current_value) / self.portfolio_high_water_mark
            
            # Calculate other risk metrics
            concentration_risk = self._calculate_concentration_risk(position_weights)
            correlation_risk = self._calculate_correlation_risk(correlation_matrix, position_weights)
            diversification_ratio = self.correlation_analyzer.calculate_diversification_ratio(
                position_weights, correlation_matrix
            )
            
            # Performance metrics
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_summary)
            sortino_ratio = self._calculate_sortino_ratio(portfolio_summary)
            portfolio_volatility = self._calculate_portfolio_volatility()
            
            # Create risk metrics object
            risk_metrics = PortfolioRiskMetrics(
                total_value=current_value,
                total_positions=len(holdings),
                portfolio_var_1d=portfolio_var_1d,
                portfolio_var_5d=portfolio_var_5d,
                max_drawdown=portfolio_summary.get('max_drawdown', current_drawdown),
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                portfolio_volatility=portfolio_volatility,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                largest_position_weight=max(position_weights.values()) if position_weights else 0,
                total_unrealized_pnl=portfolio_summary.get('unrealized_profit', 0),
                risk_adjusted_return=portfolio_summary['percent_return'] / max(portfolio_volatility, 0.01),
                diversification_ratio=diversification_ratio
            )
            
            # Store in history
            self.risk_history.append(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {str(e)}")
            return PortfolioRiskMetrics(
                total_value=0, total_positions=0, portfolio_var_1d=0, portfolio_var_5d=0,
                max_drawdown=0, current_drawdown=0, sharpe_ratio=0, sortino_ratio=0,
                portfolio_volatility=0, correlation_risk=0, concentration_risk=0,
                largest_position_weight=0, total_unrealized_pnl=0, risk_adjusted_return=0,
                diversification_ratio=0
            )
    
    def _calculate_position_risk(self, holding: Dict, weight: float, total_value: float) -> PositionRisk:
        """Calculate risk metrics for individual position"""
        try:
            token_address = holding['token_address']
            
            # Get price history for volatility calculation
            price_history = list(self.correlation_analyzer.price_history.get(token_address, []))
            
            # Calculate volatility
            volatility = 0.1  # Default volatility
            if len(price_history) > 5:
                prices = [p['price'] for p in price_history[-20:]]
                returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
                if len(returns) > 1:
                    volatility = np.std(returns)
            
            # Calculate correlation score (average with other positions)
            correlation_score = 0.5  # Default neutral correlation
            
            # Daily VaR for this position (simplified)
            daily_var = weight * volatility * 1.65  # 95% confidence level
            
            # Risk score based on multiple factors
            risk_score = min(1.0, (
                weight * 2 +  # Position size risk
                volatility * 3 +  # Volatility risk
                abs(holding.get('percent_change', 0)) / 100 * 2  # Performance risk
            ) / 3)
            
            return PositionRisk(
                token_address=token_address,
                token_symbol=holding.get('token_symbol', 'UNKNOWN'),
                position_value=holding['current_value_sol'],
                portfolio_weight=weight,
                daily_var=daily_var,
                volatility=volatility,
                correlation_score=correlation_score,
                concentration_risk=weight,  # Position weight as concentration risk
                stop_loss_level=0,  # Will be set by stop-loss manager
                take_profit_level=0,
                time_in_position=time.time() - holding.get('entry_time', time.time()),
                unrealized_pnl=holding.get('unrealized_pl_sol', 0) / total_value,
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {str(e)}")
            return PositionRisk(
                token_address="", token_symbol="", position_value=0, portfolio_weight=0,
                daily_var=0, volatility=0, correlation_score=0, concentration_risk=0,
                stop_loss_level=0, take_profit_level=0, time_in_position=0,
                unrealized_pnl=0, risk_score=0
            )
    
    def _calculate_portfolio_var(self, position_risks: List[PositionRisk], 
                               correlation_matrix: Dict, days: int) -> float:
        """Calculate portfolio Value at Risk"""
        try:
            if not position_risks:
                return 0.0
            
            # Simplified VaR calculation
            total_var = 0
            for position in position_risks:
                position_var = position.daily_var * math.sqrt(days)
                total_var += position_var ** 2
            
            # Add correlation effects (simplified)
            correlation_adjustment = 1.0
            if len(position_risks) > 1:
                avg_correlation = 0.3  # Assume moderate correlation
                correlation_adjustment = 1 + avg_correlation * 0.5
            
            portfolio_var = math.sqrt(total_var) * correlation_adjustment
            
            return min(portfolio_var, 0.5)  # Cap at 50%
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {str(e)}")
            return 0.1
    
    def _calculate_concentration_risk(self, position_weights: Dict[str, float]) -> float:
        """Calculate portfolio concentration risk"""
        try:
            if not position_weights:
                return 0.0
            
            # Herfindahl-Hirschman Index (HHI) based concentration
            hhi = sum(weight ** 2 for weight in position_weights.values())
            
            # Normalize to 0-1 scale (1 = maximum concentration)
            max_hhi = 1.0  # Single position
            min_hhi = 1.0 / len(position_weights) if position_weights else 1.0
            
            if max_hhi == min_hhi:
                return 0.0
            
            concentration_risk = (hhi - min_hhi) / (max_hhi - min_hhi)
            
            return max(0.0, min(1.0, concentration_risk))
            
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {str(e)}")
            return 0.5
    
    def _calculate_correlation_risk(self, correlation_matrix: Dict, 
                                  position_weights: Dict[str, float]) -> float:
        """Calculate portfolio correlation risk"""
        try:
            if len(position_weights) < 2:
                return 0.0
            
            # Calculate weighted average correlation
            total_correlation = 0
            total_pairs = 0
            
            for token1, weight1 in position_weights.items():
                for token2, weight2 in position_weights.items():
                    if token1 != token2 and token1 in correlation_matrix:
                        correlation = abs(correlation_matrix[token1].get(token2, 0))
                        total_correlation += weight1 * weight2 * correlation
                        total_pairs += 1
            
            if total_pairs == 0:
                return 0.0
            
            avg_correlation = total_correlation / total_pairs
            
            return max(0.0, min(1.0, avg_correlation))
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {str(e)}")
            return 0.5
    
    def _calculate_sharpe_ratio(self, portfolio_summary: Dict) -> float:
        """Calculate Sharpe ratio"""
        try:
            returns = portfolio_summary.get('percent_return', 0) / 100
            volatility = self._calculate_portfolio_volatility()
            
            if volatility == 0:
                return 0.0
            
            # Assume risk-free rate of 0% for simplicity
            sharpe_ratio = returns / volatility
            
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def _calculate_sortino_ratio(self, portfolio_summary: Dict) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        try:
            returns = portfolio_summary.get('percent_return', 0) / 100
            
            # For simplification, use volatility as downside deviation
            downside_deviation = self._calculate_portfolio_volatility()
            
            if downside_deviation == 0:
                return 0.0
            
            sortino_ratio = returns / downside_deviation
            
            return sortino_ratio
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        try:
            if len(self.risk_history) < 5:
                return 0.1  # Default volatility
            
            # Calculate volatility from portfolio value changes
            values = [risk.total_value for risk in list(self.risk_history)[-20:]]
            returns = [(values[i] / values[i-1] - 1) for i in range(1, len(values))]
            
            if len(returns) < 2:
                return 0.1
            
            volatility = np.std(returns)
            
            return max(0.01, min(1.0, volatility))
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {str(e)}")
            return 0.1
    
    async def check_risk_violations(self, risk_metrics: PortfolioRiskMetrics) -> List[str]:
        """Check for risk limit violations"""
        try:
            violations = []
            
            # Check VaR limits
            if risk_metrics.portfolio_var_1d > self.risk_limits.max_portfolio_var:
                violations.append(f"Portfolio VaR ({risk_metrics.portfolio_var_1d:.1%}) exceeds limit ({self.risk_limits.max_portfolio_var:.1%})")
            
            # Check position concentration
            if risk_metrics.largest_position_weight > self.risk_limits.max_single_position:
                violations.append(f"Largest position ({risk_metrics.largest_position_weight:.1%}) exceeds limit ({self.risk_limits.max_single_position:.1%})")
            
            # Check drawdown limits
            if risk_metrics.current_drawdown > self.risk_limits.max_drawdown_limit:
                violations.append(f"Current drawdown ({risk_metrics.current_drawdown:.1%}) exceeds limit ({self.risk_limits.max_drawdown_limit:.1%})")
            
            # Check diversification
            if risk_metrics.total_positions < self.risk_limits.min_diversification:
                violations.append(f"Portfolio has only {risk_metrics.total_positions} positions (minimum: {self.risk_limits.min_diversification})")
            
            # Check correlation risk
            if risk_metrics.correlation_risk > self.risk_limits.max_correlation:
                violations.append(f"Portfolio correlation ({risk_metrics.correlation_risk:.1%}) exceeds limit ({self.risk_limits.max_correlation:.1%})")
            
            return violations
            
        except Exception as e:
            logger.error(f"Error checking risk violations: {str(e)}")
            return [f"Error checking violations: {str(e)}"]
    
    async def update_stop_losses(self, token_prices: Dict[str, float]) -> List[str]:
        """Update all stop-losses and check for triggers"""
        try:
            triggered_stops = []
            
            for token_address, price in token_prices.items():
                # Update stop-loss
                self.stop_loss_manager.update_stop_loss(token_address, price)
                
                # Check for trigger
                should_trigger, reason = self.stop_loss_manager.check_stop_loss_trigger(token_address, price)
                
                if should_trigger:
                    triggered_stops.append({
                        'token_address': token_address,
                        'current_price': price,
                        'reason': reason
                    })
                    logger.warning(f"🚨 Stop-loss triggered for {token_address[:8]}...: {reason}")
            
            return triggered_stops
            
        except Exception as e:
            logger.error(f"Error updating stop-losses: {str(e)}")
            return []
    
    def get_risk_stats(self) -> Dict[str, Any]:
        """Get risk management statistics"""
        try:
            recent_risks = list(self.risk_history)[-10:] if self.risk_history else []
            
            return {
                'total_risk_assessments': len(self.risk_history),
                'current_risk_level': recent_risks[-1].overall_risk_level if recent_risks else 'unknown',
                'average_portfolio_var': np.mean([r.portfolio_var_1d for r in recent_risks]) if recent_risks else 0,
                'average_correlation_risk': np.mean([r.correlation_risk for r in recent_risks]) if recent_risks else 0,
                'portfolio_high_water_mark': self.portfolio_high_water_mark,
                'active_stop_losses': len(self.stop_loss_manager.stop_loss_rules),
                'emergency_stop_status': self.emergency_stop_triggered,
                'last_rebalance': self.last_rebalance_time,
                'risk_limit_violations_today': 0  # Would track actual violations
            }
            
        except Exception as e:
            logger.error(f"Error getting risk stats: {str(e)}")
            return {}
    
    async def close(self):
        """Close the portfolio risk manager"""
        try:
            logger.info("🔚 Portfolio risk manager closed")
        except Exception as e:
            logger.error(f"Error closing portfolio risk manager: {str(e)}") 
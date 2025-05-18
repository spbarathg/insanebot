import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import json
from loguru import logger
from ..utils.config import settings

class PortfolioManager:
    def __init__(self):
        self.positions: Dict[str, Dict] = {}
        self.risk_limits = settings.RISK_LIMITS
        self.position_limits = settings.POSITION_LIMITS
        self.daily_stats = {
            'trades': 0,
            'profit_loss': 0.0,
            'max_drawdown': 0.0,
            'start_balance': 0.0,
            'current_balance': 0.0
        }
        self._load_portfolio()

    def _load_portfolio(self):
        """Load portfolio data from file"""
        try:
            with open(settings.PORTFOLIO_FILE, 'r') as f:
                data = json.load(f)
                self.positions = data.get('positions', {})
                self.daily_stats = data.get('daily_stats', self.daily_stats)
        except FileNotFoundError:
            logger.info("No existing portfolio data found")
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")

    def _save_portfolio(self):
        """Save portfolio data to file"""
        try:
            with open(settings.PORTFOLIO_FILE, 'w') as f:
                json.dump({
                    'positions': self.positions,
                    'daily_stats': self.daily_stats
                }, f)
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")

    async def initialize(self, initial_balance: float):
        """Initialize portfolio with initial balance"""
        try:
            self.daily_stats['start_balance'] = initial_balance
            self.daily_stats['current_balance'] = initial_balance
            self._save_portfolio()
            logger.info(f"Portfolio initialized with {initial_balance} SOL")
        except Exception as e:
            logger.error(f"Error initializing portfolio: {e}")

    async def add_position(self, token: str, amount: float, price: float):
        """Add new position to portfolio"""
        try:
            if token in self.positions:
                logger.warning(f"Position for {token} already exists")
                return False

            # Check position limits
            if not self._check_position_limits(token, amount, price):
                return False

            # Add position
            self.positions[token] = {
                'amount': amount,
                'entry_price': price,
                'current_price': price,
                'entry_time': datetime.now().isoformat(),
                'pnl': 0.0,
                'pnl_percentage': 0.0
            }

            # Update daily stats
            self.daily_stats['trades'] += 1

            self._save_portfolio()
            return True

        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False

    async def update_position(self, token: str, current_price: float):
        """Update position with current price"""
        try:
            if token not in self.positions:
                return False

            position = self.positions[token]
            position['current_price'] = current_price
            
            # Calculate PnL
            pnl = (current_price - position['entry_price']) * position['amount']
            position['pnl'] = pnl
            position['pnl_percentage'] = (current_price - position['entry_price']) / position['entry_price']

            # Update daily stats
            self.daily_stats['profit_loss'] += pnl
            self.daily_stats['current_balance'] += pnl

            # Update max drawdown
            current_drawdown = (self.daily_stats['start_balance'] - self.daily_stats['current_balance']) / self.daily_stats['start_balance']
            self.daily_stats['max_drawdown'] = max(self.daily_stats['max_drawdown'], current_drawdown)

            self._save_portfolio()
            return True

        except Exception as e:
            logger.error(f"Error updating position: {e}")
            return False

    async def remove_position(self, token: str):
        """Remove position from portfolio"""
        try:
            if token not in self.positions:
                return False

            # Update daily stats
            position = self.positions[token]
            self.daily_stats['profit_loss'] += position['pnl']
            self.daily_stats['current_balance'] += position['pnl']

            # Remove position
            del self.positions[token]

            self._save_portfolio()
            return True

        except Exception as e:
            logger.error(f"Error removing position: {e}")
            return False

    def _check_position_limits(self, token: str, amount: float, price: float) -> bool:
        """Check if position meets risk limits"""
        try:
            position_value = amount * price
            
            # Check maximum position size
            if position_value > self.position_limits['max_position_size']:
                logger.warning(f"Position size {position_value} exceeds maximum {self.position_limits['max_position_size']}")
                return False

            # Check maximum exposure per token
            if position_value > self.risk_limits['max_token_exposure']:
                logger.warning(f"Token exposure {position_value} exceeds maximum {self.risk_limits['max_token_exposure']}")
                return False

            # Check total portfolio exposure
            total_exposure = sum(p['amount'] * p['current_price'] for p in self.positions.values())
            if total_exposure + position_value > self.risk_limits['max_portfolio_exposure']:
                logger.warning(f"Total exposure {total_exposure + position_value} exceeds maximum {self.risk_limits['max_portfolio_exposure']}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return False

    async def rebalance_portfolio(self):
        """Rebalance portfolio based on risk metrics"""
        try:
            # Calculate current risk exposure
            risk_exposure = self._calculate_risk_exposure()
            
            # Check if rebalancing is needed
            if risk_exposure > self.risk_limits['max_exposure']:
                await self._reduce_exposure()
                
            # Optimize position sizes
            await self._optimize_position_sizes()
            
            self._save_portfolio()
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {e}")

    def _calculate_risk_exposure(self) -> float:
        """Calculate current risk exposure"""
        try:
            total_exposure = 0.0
            for position in self.positions.values():
                position_value = position['amount'] * position['current_price']
                total_exposure += position_value
            return total_exposure
        except Exception as e:
            logger.error(f"Error calculating risk exposure: {e}")
            return 0.0

    async def _reduce_exposure(self):
        """Reduce portfolio exposure"""
        try:
            # Sort positions by PnL
            sorted_positions = sorted(
                self.positions.items(),
                key=lambda x: x[1]['pnl_percentage']
            )
            
            # Reduce exposure by closing worst performing positions
            for token, position in sorted_positions:
                if self._calculate_risk_exposure() <= self.risk_limits['max_exposure']:
                    break
                    
                await self.remove_position(token)
                
        except Exception as e:
            logger.error(f"Error reducing exposure: {e}")

    async def _optimize_position_sizes(self):
        """Optimize position sizes based on risk metrics"""
        try:
            total_balance = self.daily_stats['current_balance']
            
            for token, position in self.positions.items():
                # Calculate optimal position size
                optimal_size = self._calculate_optimal_position_size(
                    token,
                    position['current_price'],
                    total_balance
                )
                
                # Adjust position if needed
                if abs(optimal_size - position['amount']) > self.position_limits['min_adjustment']:
                    await self._adjust_position_size(token, optimal_size)
                    
        except Exception as e:
            logger.error(f"Error optimizing position sizes: {e}")

    def _calculate_optimal_position_size(self, token: str, price: float, total_balance: float) -> float:
        """Calculate optimal position size based on risk metrics"""
        try:
            # Get token risk metrics
            risk_metrics = self._get_token_risk_metrics(token)
            
            # Calculate position size based on risk-adjusted return
            risk_adjusted_return = risk_metrics['expected_return'] / risk_metrics['volatility']
            position_size = (total_balance * risk_adjusted_return) / price
            
            # Apply position limits
            position_size = min(
                position_size,
                self.position_limits['max_position_size'] / price
            )
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating optimal position size: {e}")
            return 0.0

    def _get_token_risk_metrics(self, token: str) -> Dict:
        """Get risk metrics for a token"""
        try:
            # This would typically come from market data analysis
            # For now, return default values
            return {
                'expected_return': 0.1,  # 10% expected return
                'volatility': 0.2,  # 20% volatility
                'correlation': 0.0  # No correlation with other positions
            }
        except Exception as e:
            logger.error(f"Error getting token risk metrics: {e}")
            return {
                'expected_return': 0.0,
                'volatility': 1.0,
                'correlation': 0.0
            }

    async def _adjust_position_size(self, token: str, new_size: float):
        """Adjust position size"""
        try:
            if token not in self.positions:
                return False
                
            position = self.positions[token]
            current_size = position['amount']
            
            if abs(new_size - current_size) < self.position_limits['min_adjustment']:
                return False
                
            # Update position size
            position['amount'] = new_size
            
            self._save_portfolio()
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting position size: {e}")
            return False

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        try:
            total_value = sum(p['amount'] * p['current_price'] for p in self.positions.values())
            total_pnl = sum(p['pnl'] for p in self.positions.values())
            
            return {
                'total_positions': len(self.positions),
                'total_value': total_value,
                'total_pnl': total_pnl,
                'daily_stats': self.daily_stats,
                'positions': self.positions
            }
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {} 
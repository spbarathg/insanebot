import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import json
from loguru import logger
import time
import math

class PortfolioManager:
    def __init__(self):
        self.positions: Dict[str, Dict] = {}
        # AGGRESSIVE HIGH-RISK, HIGH-REWARD SETTINGS
        self.risk_limits = {
            'max_exposure': 0.95,  # 95% max exposure for high-risk
            'max_token_exposure': 0.4,  # 40% single token exposure for moonshots
            'max_portfolio_exposure': 1.0  # Full portfolio exposure
        }
        self.position_limits = {
            'max_position_size': 0.4,  # 40% max position for high-reward plays
            'min_adjustment': 0.01,
            'profit_multiplier': 2.5,  # 2.5x profit reinvestment multiplier
            'compound_rate': 1.15  # 15% compounding boost
        }
        self.daily_stats = {
            'trades': 0,
            'profit_loss': 0.0,
            'max_drawdown': 0.0,
            'start_balance': 0.0,
            'current_balance': 0.0,
            'win_streak': 0,
            'total_wins': 0,
            'compound_multiplier': 1.0
        }
        self.portfolio_file = "portfolio.json"
        self.profit_threshold = 0.20  # 20% profit target for high-risk trades
        self.stop_loss = -0.15  # 15% stop loss for risk management
        self._load_portfolio()

    def _load_portfolio(self):
        """Load portfolio data from file"""
        try:
            with open(self.portfolio_file, 'r') as f:
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
            with open(self.portfolio_file, 'w') as f:
                json.dump({
                    'positions': self.positions,
                    'daily_stats': self.daily_stats
                }, f)
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")

    def initialize(self, starting_balance_sol: float = 0.1):
        """Initialize portfolio with starting balance"""
        try:
            # Set starting balance if portfolio is empty
            if not self.positions and self.daily_stats.get('current_balance', 0) == 0:
                self.daily_stats['start_balance'] = starting_balance_sol
                self.daily_stats['current_balance'] = starting_balance_sol
                self.daily_stats['compound_multiplier'] = 1.0
                logger.info(f"Portfolio initialized with {starting_balance_sol} SOL (HIGH-RISK MODE)")
                self._save_portfolio()
            
            return True
        except Exception as e:
            logger.error(f"Error initializing portfolio: {e}")
            return False

    async def add_position(self, token: str, amount: float, price: float):
        """Add new position with aggressive sizing for high-reward potential"""
        try:
            if token in self.positions:
                logger.warning(f"Position for {token} already exists")
                return False

            # Apply compound multiplier to position size for exponential growth
            compound_multiplier = self.daily_stats.get('compound_multiplier', 1.0)
            adjusted_amount = amount * compound_multiplier
            
            # Check position limits
            if not self._check_position_limits_aggressive(token, adjusted_amount, price):
                return False

            # Add position with profit targets and stop losses
            self.positions[token] = {
                'amount': adjusted_amount,
                'entry_price': price,
                'current_price': price,
                'entry_time': datetime.now().isoformat(),
                'pnl': 0.0,
                'pnl_percentage': 0.0,
                'profit_target': price * (1 + self.profit_threshold),  # 20% profit target
                'stop_loss': price * (1 + self.stop_loss),  # 15% stop loss
                'max_profit': 0.0,
                'risk_level': 'HIGH'
            }

            # Update daily stats
            self.daily_stats['trades'] += 1

            self._save_portfolio()
            logger.info(f"AGGRESSIVE POSITION: {adjusted_amount:.6f} {token} at ${price:.6f} (Target: +{self.profit_threshold*100}%)")
            return True

        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False

    async def update_position(self, token: str, current_price: float):
        """Update position with aggressive profit tracking and auto-exit logic"""
        try:
            if token not in self.positions:
                return False

            position = self.positions[token]
            position['current_price'] = current_price
            
            # Calculate PnL with controlled compounding
            entry_price = position['entry_price']
            amount = position['amount']
            raw_pnl = (current_price - entry_price) * amount
            
            # Apply profit multiplier for exponential gains (but controlled)
            if raw_pnl > 0:
                # Controlled profit multiplier - scales down with larger profits
                profit_multiplier = self.position_limits['profit_multiplier']
                if raw_pnl > 0.1:  # If profit > 0.1 SOL, reduce multiplier
                    profit_multiplier = min(profit_multiplier, 1.5)
                compounded_pnl = raw_pnl * profit_multiplier
            else:
                compounded_pnl = raw_pnl  # No multiplier for losses
            
            position['pnl'] = compounded_pnl
            position['pnl_percentage'] = (current_price - entry_price) / entry_price
            
            # Track maximum profit for trailing stops
            if compounded_pnl > position.get('max_profit', 0):
                position['max_profit'] = compounded_pnl

            # Auto-exit logic for profit taking and loss cutting
            should_exit, exit_reason = self._should_exit_position(position, current_price)
            if should_exit:
                await self._execute_exit(token, exit_reason, compounded_pnl)
                return True

            # Update compound multiplier based on success (controlled growth)
            if compounded_pnl > 0:
                self.daily_stats['win_streak'] += 1
                self.daily_stats['total_wins'] += 1
                # Controlled compound multiplier increase
                streak_bonus = 1 + min(self.daily_stats['win_streak'] * 0.02, 0.2)  # Max 20% bonus, 2% per win
                self.daily_stats['compound_multiplier'] = min(2.0, streak_bonus)  # Max 2x multiplier
            else:
                self.daily_stats['win_streak'] = 0
                # Reduce multiplier slightly on losses
                self.daily_stats['compound_multiplier'] = max(0.9, self.daily_stats['compound_multiplier'] * 0.98)

            self._save_portfolio()
            return True

        except Exception as e:
            logger.error(f"Error updating position: {e}")
            return False

    def _should_exit_position(self, position: Dict, current_price: float) -> tuple[bool, str]:
        """Determine if position should be exited based on profit targets or stop losses"""
        pnl_pct = position['pnl_percentage']
        
        # Take profit at target
        if pnl_pct >= self.profit_threshold:
            return True, f"PROFIT_TARGET_HIT_{pnl_pct*100:.1f}%"
        
        # Stop loss protection
        if pnl_pct <= self.stop_loss:
            return True, f"STOP_LOSS_HIT_{pnl_pct*100:.1f}%"
        
        # Trailing stop for high profits (lock in gains above 30%)
        if pnl_pct > 0.30:
            max_profit_pct = position.get('max_profit', 0) / (position['amount'] * position['entry_price'])
            if pnl_pct < (max_profit_pct * 0.8):  # 20% pullback from max
                return True, f"TRAILING_STOP_{pnl_pct*100:.1f}%"
        
        return False, ""

    async def _execute_exit(self, token: str, reason: str, final_pnl: float):
        """Execute position exit and update compound metrics"""
        try:
            position = self.positions[token]
            
            # Log the exit
            logger.info(f"POSITION EXIT: {token} - {reason} - P&L: {final_pnl:.6f} SOL")
            
            # Update compound multiplier based on successful exit (controlled)
            if final_pnl > 0:
                # Successful trade - controlled multiplier boost
                current_multiplier = self.daily_stats.get('compound_multiplier', 1.0)
                boost_rate = min(self.position_limits['compound_rate'], 1.05)  # Max 5% boost per trade
                new_multiplier = min(2.0, current_multiplier * boost_rate)  # Cap at 2x
                self.daily_stats['compound_multiplier'] = new_multiplier
                logger.info(f"COMPOUND BOOST: {current_multiplier:.2f} → {new_multiplier:.2f}")
            
            # Remove position
            del self.positions[token]
            
            # Update final stats with controlled growth
            profit_contribution = min(final_pnl, 1.0)  # Cap individual trade impact
            self.daily_stats['profit_loss'] += profit_contribution
            self.daily_stats['current_balance'] += profit_contribution
            
            self._save_portfolio()
            
        except Exception as e:
            logger.error(f"Error executing exit: {e}")

    def _check_position_limits_aggressive(self, token: str, amount: float, price: float) -> bool:
        """Check if position meets aggressive high-risk limits"""
        try:
            position_value = amount * price
            current_balance = self.daily_stats.get('current_balance', 0.1)
            
            # Allow larger positions for high-reward potential
            max_position_sol = current_balance * self.position_limits['max_position_size']
            if position_value > max_position_sol:
                logger.warning(f"Position size {position_value:.6f} SOL exceeds max {max_position_sol:.6f} SOL")
                return False

            # Check maximum exposure per token (more aggressive)
            max_token_exposure = current_balance * self.risk_limits['max_token_exposure']
            if position_value > max_token_exposure:
                logger.warning(f"Token exposure {position_value:.6f} exceeds max {max_token_exposure:.6f}")
                return False

            # Check total portfolio exposure (near 100% for high-risk)
            total_exposure = sum(p['amount'] * p['current_price'] for p in self.positions.values())
            max_total_exposure = current_balance * self.risk_limits['max_exposure']
            if total_exposure + position_value > max_total_exposure:
                logger.warning(f"Total exposure would exceed {self.risk_limits['max_exposure']*100}%")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return False

    def get_aggressive_position_size(self, token: str, confidence: float, market_momentum: float) -> float:
        """Calculate aggressive position size based on confidence and momentum"""
        try:
            current_balance = self.daily_stats.get('current_balance', 0.1)
            compound_multiplier = self.daily_stats.get('compound_multiplier', 1.0)
            
            # Base allocation (controlled)
            base_allocation = min(self.position_limits['max_position_size'], 0.25)  # Cap at 25%
            
            # Confidence multiplier (high confidence = larger position)
            confidence_multiplier = 0.8 + (confidence * 0.8)  # 0.8x to 1.6x based on confidence
            
            # Momentum multiplier (strong momentum = larger position)
            momentum_multiplier = 0.9 + (market_momentum * 0.2)  # 0.9x to 1.1x based on momentum
            
            # Win streak bonus (controlled)
            win_streak = self.daily_stats.get('win_streak', 0)
            streak_multiplier = 1.0 + min(win_streak * 0.05, 0.25)  # Up to 25% bonus for streaks
            
            # Calculate final position size with controlled growth
            position_percentage = (base_allocation * confidence_multiplier * 
                                 momentum_multiplier * streak_multiplier)
            
            # Apply compound multiplier but cap it
            position_percentage *= min(compound_multiplier, 1.5)  # Cap compound effect
            
            # Cap at maximum exposure
            position_percentage = min(position_percentage, self.risk_limits['max_token_exposure'])
            
            position_size_sol = current_balance * position_percentage
            
            logger.info(f"CONTROLLED SIZING: {position_percentage*100:.1f}% allocation "
                       f"(Confidence: {confidence:.2f}, Momentum: {market_momentum:.2f}, "
                       f"Streak: {win_streak}, Compound: {compound_multiplier:.2f})")
            
            return position_size_sol
            
        except Exception as e:
            logger.error(f"Error calculating aggressive position size: {e}")
            return current_balance * 0.1  # Fallback to 10%

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary with profit metrics"""
        try:
            total_value = sum(p['amount'] * p['current_price'] for p in self.positions.values())
            total_pnl = sum(p['pnl'] for p in self.positions.values())
            current_balance = self.daily_stats.get('current_balance', 0.0)
            start_balance = self.daily_stats.get('start_balance', 0.1)
            
            # Calculate returns
            total_return = ((current_balance - start_balance) / start_balance) * 100 if start_balance > 0 else 0
            
            return {
                'total_positions': len(self.positions),
                'total_value': total_value,
                'current_value': total_value,
                'current_balance': current_balance,
                'total_pnl': total_pnl,
                'unrealized_profit': total_pnl,
                'total_return_percent': total_return,
                'compound_multiplier': self.daily_stats.get('compound_multiplier', 1.0),
                'win_streak': self.daily_stats.get('win_streak', 0),
                'total_wins': self.daily_stats.get('total_wins', 0),
                'total_trades': self.daily_stats.get('trades', 0),
                'win_rate': (self.daily_stats.get('total_wins', 0) / max(self.daily_stats.get('trades', 1), 1)) * 100,
                'max_drawdown': self.daily_stats.get('max_drawdown', 0.0),
                'daily_stats': self.daily_stats,
                'positions': self.positions
            }
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {
                'total_positions': 0,
                'total_value': 0.0,
                'current_value': 0.0,
                'current_balance': 0.0,
                'total_pnl': 0.0,
                'unrealized_profit': 0.0,
                'total_return_percent': 0.0,
                'compound_multiplier': 1.0,
                'win_streak': 0,
                'total_wins': 0,
                'total_trades': 0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'daily_stats': self.daily_stats,
                'positions': {}
            }

    def get_holdings(self) -> List[Dict]:
        """Get holdings in the format expected by risk manager"""
        try:
            holdings = []
            current_time = time.time()
            
            for token_address, position in self.positions.items():
                holding = {
                    'token_address': token_address,
                    'token_symbol': position.get('symbol', 'UNKNOWN'),
                    'amount': position.get('amount', 0),
                    'current_price': position.get('current_price', 0),
                    'current_value_sol': position.get('amount', 0) * position.get('current_price', 0),
                    'entry_price': position.get('entry_price', 0),
                    'entry_time': position.get('entry_time', current_time),
                    'pnl': position.get('pnl', 0),
                    'unrealized_pl_sol': position.get('pnl', 0),
                    'percent_change': position.get('pnl_percentage', 0),
                    'profit_target': position.get('profit_target', 0),
                    'stop_loss': position.get('stop_loss', 0),
                    'max_profit': position.get('max_profit', 0),
                    'risk_level': position.get('risk_level', 'MEDIUM')
                }
                holdings.append(holding)
            
            return holdings
            
        except Exception as e:
            logger.error(f"Error getting holdings: {e}")
            return [] 
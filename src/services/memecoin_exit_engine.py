"""
Memecoin Exit Strategy Engine

Advanced exit strategy system specifically designed for memecoin trading
with dynamic profit-taking based on social momentum, volume, and viral indicators.

Features:
- Social momentum trailing stops
- Volume-based profit taking
- Viral peak detection
- Multi-tier exit strategies
- Risk-adjusted position management
- Dynamic position sizing based on signal confidence
- Multi-signal ensemble scoring
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import secrets

def secure_choice(seq):
    """Secure random choice from sequence"""
    return seq[secrets.randbelow(len(seq))]

def secure_uniform(a: float, b: float) -> float:
    """Secure uniform random float between a and b"""
    return a + (b - a) * (secrets.randbelow(10000) / 10000.0)

logger = logging.getLogger(__name__)

@dataclass
class SignalEnsemble:
    """Ensemble of trading signals for position sizing"""
    pump_fun_signal: float = 0.0  # 0.0 to 1.0
    smart_money_signal: float = 0.0
    social_sentiment_signal: float = 0.0
    on_chain_signal: float = 0.0
    narrative_signal: float = 0.0  # From LLM analysis
    technical_signal: float = 0.0
    
    @property
    def ensemble_score(self) -> float:
        """Calculate weighted ensemble score"""
        weights = {
            'pump_fun': 0.25,
            'smart_money': 0.25,
            'social_sentiment': 0.20,
            'on_chain': 0.15,
            'narrative': 0.10,
            'technical': 0.05
        }
        
        return (
            self.pump_fun_signal * weights['pump_fun'] +
            self.smart_money_signal * weights['smart_money'] +
            self.social_sentiment_signal * weights['social_sentiment'] +
            self.on_chain_signal * weights['on_chain'] +
            self.narrative_signal * weights['narrative'] +
            self.technical_signal * weights['technical']
        )
    
    @property
    def confidence_level(self) -> str:
        """Get confidence level based on ensemble score"""
        score = self.ensemble_score
        if score >= 0.8:
            return "very_high"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "very_low"

@dataclass
class DynamicPositionConfig:
    """Dynamic position sizing configuration"""
    base_position_size: float = 0.01  # Base position in SOL
    max_position_multiplier: float = 5.0  # Max 5x base position
    confidence_scaling: bool = True
    signal_threshold: float = 0.3  # Minimum ensemble score to trade
    
    def calculate_position_size(self, ensemble: SignalEnsemble, available_capital: float) -> float:
        """Calculate position size based on signal ensemble"""
        if ensemble.ensemble_score < self.signal_threshold:
            return 0.0
        
        if not self.confidence_scaling:
            return min(self.base_position_size, available_capital)
        
        # Scale position size by confidence
        multiplier = 1.0 + (ensemble.ensemble_score - 0.5) * (self.max_position_multiplier - 1.0)
        multiplier = max(0.5, min(multiplier, self.max_position_multiplier))
        
        position_size = self.base_position_size * multiplier
        return min(position_size, available_capital * 0.2)  # Max 20% of capital per trade

@dataclass
class MemecoinPosition:
    """Active memecoin position with enhanced tracking"""
    token_address: str
    token_symbol: str
    entry_price: float
    entry_time: float
    amount: float
    strategy_type: str  # 'pump_fun_snipe', 'social_momentum', 'ensemble_decision'
    signal_ensemble: SignalEnsemble = None  # Signal scores at entry
    current_price: float = 0.0
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    peak_price: float = 0.0
    peak_pnl_pct: float = 0.0
    social_momentum: float = 0.0
    viral_indicators: List[str] = None
    exit_triggers: List[str] = None
    
    def __post_init__(self):
        if self.viral_indicators is None:
            self.viral_indicators = []
        if self.exit_triggers is None:
            self.exit_triggers = []
        if self.signal_ensemble is None:
            self.signal_ensemble = SignalEnsemble()

@dataclass
class ExitSignal:
    """Exit signal for a position"""
    token_address: str
    token_symbol: str
    exit_reason: str
    exit_urgency: str  # 'low', 'medium', 'high', 'critical'
    position_size: float
    current_price: float
    entry_value: float
    current_value: float
    profit_pct: float
    confidence: float
    timestamp: float

class MemecoinExitEngine:
    """
    Advanced exit strategy engine for memecoin positions
    
    Uses multiple signals including social momentum, volume analysis,
    viral indicators, and technical analysis to optimize exit timing.
    """
    
    def __init__(self, callback_handler: Optional[Callable] = None):
        self.callback_handler = callback_handler
        
        # Active positions
        self.positions: Dict[str, MemecoinPosition] = {}
        
        # Exit strategy configuration
        self.exit_config = {
            # Profit taking levels
            'profit_targets': [0.5, 1.0, 2.0, 5.0, 10.0],  # 50%, 100%, 200%, 500%, 1000%
            'profit_take_amounts': [0.2, 0.3, 0.3, 0.15, 0.05],  # % to sell at each level
            
            # Trailing stop configuration
            'trailing_stop_activation': 0.3,  # Start trailing at 30% profit
            'trailing_stop_distance': 0.15,   # 15% from peak
            'social_trailing_multiplier': 1.5, # Tighter stops when social momentum drops
            
            # Time-based exits
            'max_hold_time_hours': 24,        # Max 24 hours
            'momentum_decay_hours': 2,        # Exit if no momentum for 2 hours
            
            # Risk management
            'stop_loss_pct': 0.20,           # 20% stop loss
            'emergency_exit_pct': 0.35,      # 35% emergency exit
            
            # Volume-based exits
            'volume_decay_threshold': 0.5,   # Exit if volume drops 50%
            'whale_dump_threshold': 0.1,     # Exit if 10%+ single sell
        }
        
        # Performance tracking
        self.exit_stats = {
            'total_exits': 0,
            'profitable_exits': 0,
            'total_profit': 0.0,
            'best_exit_pct': 0.0,
            'worst_exit_pct': 0.0,
            'avg_hold_time_hours': 0.0,
            'strategy_performance': {}
        }
        
        # Price and social data tracking
        self.price_history: Dict[str, deque] = {}
        self.social_history: Dict[str, deque] = {}
        
        logger.info("💰 Memecoin Exit Engine initialized - Ready for profit optimization!")
    
    async def add_position(
        self, 
        token_address: str, 
        token_symbol: str,
        entry_price: float,
        amount: float,
        strategy_type: str
    ):
        """Add a new position to track"""
        try:
            position = MemecoinPosition(
                token_address=token_address,
                token_symbol=token_symbol,
                entry_price=entry_price,
                entry_time=time.time(),
                amount=amount,
                strategy_type=strategy_type,
                current_price=entry_price,
                peak_price=entry_price
            )
            
            self.positions[token_address] = position
            
            # Initialize tracking data
            self.price_history[token_address] = deque(maxlen=1000)  # Last 1000 price points
            self.social_history[token_address] = deque(maxlen=100)   # Last 100 social data points
            
            logger.info(f"📈 POSITION ADDED: {token_symbol} | {amount:.4f} @ ${entry_price:.6f} | {strategy_type}")
            
            # Start monitoring this position
            asyncio.create_task(self._monitor_position(token_address))
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
    
    async def _monitor_position(self, token_address: str):
        """Monitor a specific position for exit conditions"""
        try:
            while token_address in self.positions:
                try:
                    position = self.positions[token_address]
                    
                    # Update position data
                    await self._update_position_data(position)
                    
                    # Check for exit conditions
                    exit_signal = await self._check_exit_conditions(position)
                    
                    if exit_signal:
                        # Generate exit signal
                        if self.callback_handler:
                            await self.callback_handler(exit_signal)
                        
                        # Remove position after exit
                        await self._process_exit(position, exit_signal)
                        break
                    
                    # Update position in storage
                    self.positions[token_address] = position
                    
                    await asyncio.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Error monitoring position {token_address}: {e}")
                    await asyncio.sleep(10)
            
        except Exception as e:
            logger.error(f"Error in position monitoring: {e}")
    
    async def _update_position_data(self, position: MemecoinPosition):
        """Update position with current market data"""
        try:
            # Get current price (this would integrate with your price feed)
            current_price = await self._get_current_price(position.token_address)
            
            if current_price > 0:
                position.current_price = current_price
                position.current_value = position.amount * current_price
                
                # Calculate P&L
                entry_value = position.amount * position.entry_price
                position.unrealized_pnl = position.current_value - entry_value
                position.unrealized_pnl_pct = (position.unrealized_pnl / entry_value) * 100
                
                # Track peak
                if current_price > position.peak_price:
                    position.peak_price = current_price
                    position.peak_pnl_pct = position.unrealized_pnl_pct
                
                # Store price history
                self.price_history[position.token_address].append({
                    'price': current_price,
                    'timestamp': time.time()
                })
                
                # Update social momentum (placeholder - would integrate with social engine)
                position.social_momentum = await self._get_social_momentum(position.token_address)
                
                # Store social history
                self.social_history[position.token_address].append({
                    'momentum': position.social_momentum,
                    'timestamp': time.time()
                })
                
        except Exception as e:
            logger.error(f"Error updating position data: {e}")
    
    async def _get_current_price(self, token_address: str) -> float:
        """Get current token price (placeholder)"""
        try:
            # This would integrate with your price feed service
            # For now, simulate price movement
            base_price = 0.001  # Placeholder base price
            volatility = secure_uniform(0.9, 1.1)  # ±10% volatility
            return base_price * volatility
            
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return 0.0
    
    async def _get_social_momentum(self, token_address: str) -> float:
        """Get current social momentum score"""
        try:
            # This would integrate with your social sentiment engine
            # For now, simulate momentum decay
            return secure_uniform(0.3, 0.9)  # Random momentum score
            
        except Exception as e:
            logger.error(f"Error getting social momentum: {e}")
            return 0.0
    
    async def _check_exit_conditions(self, position: MemecoinPosition) -> Optional[ExitSignal]:
        """Check all exit conditions for a position"""
        try:
            current_time = time.time()
            hold_time_hours = (current_time - position.entry_time) / 3600
            
            # 1. PROFIT TARGET EXITS
            for i, target_pct in enumerate(self.exit_config['profit_targets']):
                if position.unrealized_pnl_pct >= target_pct * 100:
                    sell_amount = position.amount * self.exit_config['profit_take_amounts'][i]
                    
                    return ExitSignal(
                        token_address=position.token_address,
                        token_symbol=position.token_symbol,
                        exit_reason=f"Profit target {target_pct*100:.0f}% reached",
                        exit_urgency="medium",
                        position_size=sell_amount,
                        current_price=position.current_price,
                        entry_value=position.amount * position.entry_price,
                        current_value=position.current_value,
                        profit_pct=position.unrealized_pnl_pct,
                        confidence=0.8,
                        timestamp=current_time
                    )
            
            # 2. TRAILING STOP EXITS
            if position.peak_pnl_pct >= self.exit_config['trailing_stop_activation'] * 100:
                # Calculate trailing stop based on social momentum
                base_distance = self.exit_config['trailing_stop_distance']
                if position.social_momentum < 0.5:  # Low momentum = tighter stop
                    stop_distance = base_distance * self.exit_config['social_trailing_multiplier']
                else:
                    stop_distance = base_distance
                
                stop_price = position.peak_price * (1 - stop_distance)
                
                if position.current_price <= stop_price:
                    return ExitSignal(
                        token_address=position.token_address,
                        token_symbol=position.token_symbol,
                        exit_reason=f"Trailing stop triggered (from peak)",
                        exit_urgency="high",
                        position_size=position.amount,
                        current_price=position.current_price,
                        entry_value=position.amount * position.entry_price,
                        current_value=position.current_value,
                        profit_pct=position.unrealized_pnl_pct,
                        confidence=0.9,
                        timestamp=current_time
                    )
            
            # 3. STOP LOSS EXITS
            if position.unrealized_pnl_pct <= -self.exit_config['stop_loss_pct'] * 100:
                return ExitSignal(
                    token_address=position.token_address,
                    token_symbol=position.token_symbol,
                    exit_reason="Stop loss triggered",
                    exit_urgency="critical",
                    position_size=position.amount,
                    current_price=position.current_price,
                    entry_value=position.amount * position.entry_price,
                    current_value=position.current_value,
                    profit_pct=position.unrealized_pnl_pct,
                    confidence=1.0,
                    timestamp=current_time
                )
            
            # 4. TIME-BASED EXITS
            if hold_time_hours >= self.exit_config['max_hold_time_hours']:
                return ExitSignal(
                    token_address=position.token_address,
                    token_symbol=position.token_symbol,
                    exit_reason=f"Max hold time reached ({hold_time_hours:.1f}h)",
                    exit_urgency="medium",
                    position_size=position.amount,
                    current_price=position.current_price,
                    entry_value=position.amount * position.entry_price,
                    current_value=position.current_value,
                    profit_pct=position.unrealized_pnl_pct,
                    confidence=0.6,
                    timestamp=current_time
                )
            
            # 5. SOCIAL MOMENTUM DECAY
            if await self._check_momentum_decay(position):
                return ExitSignal(
                    token_address=position.token_address,
                    token_symbol=position.token_symbol,
                    exit_reason="Social momentum decay detected",
                    exit_urgency="high",
                    position_size=position.amount * 0.7,  # Partial exit
                    current_price=position.current_price,
                    entry_value=position.amount * position.entry_price,
                    current_value=position.current_value,
                    profit_pct=position.unrealized_pnl_pct,
                    confidence=0.7,
                    timestamp=current_time
                )
            
            # 6. VOLUME ANALYSIS
            if await self._check_volume_conditions(position):
                return ExitSignal(
                    token_address=position.token_address,
                    token_symbol=position.token_symbol,
                    exit_reason="Volume analysis triggered exit",
                    exit_urgency="high",
                    position_size=position.amount * 0.8,
                    current_price=position.current_price,
                    entry_value=position.amount * position.entry_price,
                    current_value=position.current_value,
                    profit_pct=position.unrealized_pnl_pct,
                    confidence=0.8,
                    timestamp=current_time
                )
            
            return None  # No exit conditions met
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return None
    
    async def _check_momentum_decay(self, position: MemecoinPosition) -> bool:
        """Check for social momentum decay"""
        try:
            if len(self.social_history[position.token_address]) < 10:
                return False
            
            recent_momentum = list(self.social_history[position.token_address])[-10:]
            
            # Check if momentum has been declining
            momentum_values = [m['momentum'] for m in recent_momentum]
            
            # Simple momentum decay detection
            if all(momentum_values[i] >= momentum_values[i+1] for i in range(len(momentum_values)-1)):
                return True  # Consistent decline
            
            # Check if current momentum is much lower than peak
            if position.social_momentum < 0.3 and max(momentum_values) > 0.7:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking momentum decay: {e}")
            return False
    
    async def _check_volume_conditions(self, position: MemecoinPosition) -> bool:
        """Check volume-based exit conditions"""
        try:
            # This would integrate with volume analysis
            # For now, return False (no volume exit)
            return False
            
        except Exception as e:
            logger.error(f"Error checking volume conditions: {e}")
            return False
    
    async def _process_exit(self, position: MemecoinPosition, exit_signal: ExitSignal):
        """Process the exit and update statistics"""
        try:
            # Calculate final metrics
            hold_time_hours = (exit_signal.timestamp - position.entry_time) / 3600
            
            # Update statistics
            self.exit_stats['total_exits'] += 1
            
            if exit_signal.profit_pct > 0:
                self.exit_stats['profitable_exits'] += 1
            
            self.exit_stats['total_profit'] += exit_signal.profit_pct
            
            if exit_signal.profit_pct > self.exit_stats['best_exit_pct']:
                self.exit_stats['best_exit_pct'] = exit_signal.profit_pct
            
            if exit_signal.profit_pct < self.exit_stats['worst_exit_pct']:
                self.exit_stats['worst_exit_pct'] = exit_signal.profit_pct
            
            # Update average hold time
            total_exits = self.exit_stats['total_exits']
            current_avg = self.exit_stats['avg_hold_time_hours']
            new_avg = ((current_avg * (total_exits - 1)) + hold_time_hours) / total_exits
            self.exit_stats['avg_hold_time_hours'] = new_avg
            
            # Update strategy performance
            strategy = position.strategy_type
            if strategy not in self.exit_stats['strategy_performance']:
                self.exit_stats['strategy_performance'][strategy] = {
                    'exits': 0, 'total_profit': 0.0, 'avg_profit': 0.0
                }
            
            strategy_stats = self.exit_stats['strategy_performance'][strategy]
            strategy_stats['exits'] += 1
            strategy_stats['total_profit'] += exit_signal.profit_pct
            strategy_stats['avg_profit'] = strategy_stats['total_profit'] / strategy_stats['exits']
            
            logger.info(f"📤 EXIT PROCESSED: {position.token_symbol} | "
                       f"Profit: {exit_signal.profit_pct:.1f}% | "
                       f"Hold: {hold_time_hours:.1f}h | "
                       f"Reason: {exit_signal.exit_reason}")
            
            # Remove position
            del self.positions[position.token_address]
            
        except Exception as e:
            logger.error(f"Error processing exit: {e}")
    
    def get_active_positions(self) -> List[MemecoinPosition]:
        """Get all active positions"""
        return list(self.positions.values())
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions"""
        try:
            if not self.positions:
                return {'total_positions': 0, 'total_value': 0, 'total_pnl': 0}
            
            positions = list(self.positions.values())
            
            total_value = sum(p.current_value for p in positions)
            total_pnl = sum(p.unrealized_pnl for p in positions)
            total_entry_value = sum(p.amount * p.entry_price for p in positions)
            total_pnl_pct = (total_pnl / total_entry_value) * 100 if total_entry_value > 0 else 0
            
            best_performer = max(positions, key=lambda p: p.unrealized_pnl_pct)
            worst_performer = min(positions, key=lambda p: p.unrealized_pnl_pct)
            
            return {
                'total_positions': len(positions),
                'total_value': total_value,
                'total_pnl': total_pnl,
                'total_pnl_pct': total_pnl_pct,
                'best_performer': {
                    'symbol': best_performer.token_symbol,
                    'pnl_pct': best_performer.unrealized_pnl_pct
                },
                'worst_performer': {
                    'symbol': worst_performer.token_symbol,
                    'pnl_pct': worst_performer.unrealized_pnl_pct
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting position summary: {e}")
            return {'error': str(e)}
    
    def get_exit_statistics(self) -> Dict[str, Any]:
        """Get exit strategy performance statistics"""
        return {
            **self.exit_stats,
            'win_rate_pct': (
                (self.exit_stats['profitable_exits'] / self.exit_stats['total_exits']) * 100
                if self.exit_stats['total_exits'] > 0 else 0
            ),
            'avg_profit_pct': (
                self.exit_stats['total_profit'] / self.exit_stats['total_exits']
                if self.exit_stats['total_exits'] > 0 else 0
            )
        }

# Helper function
def create_memecoin_exit_engine(callback_handler: Optional[Callable] = None) -> MemecoinExitEngine:
    """Create a memecoin exit engine instance"""
    return MemecoinExitEngine(callback_handler) 
"""
Advanced Exit Management System

Implements sophisticated exit strategies for maximum profit optimization:
- Trailing stop losses with volatility adjustment
- Partial profit taking at multiple levels
- Volume exhaustion detection
- RSI divergence exits
- Time-based exits
- Emergency liquidation protocols
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class ExitTrigger(Enum):
    """Types of exit triggers"""
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    PROFIT_TARGET = "profit_target"
    PARTIAL_PROFIT = "partial_profit"
    VOLUME_EXHAUSTION = "volume_exhaustion"
    RSI_DIVERGENCE = "rsi_divergence"
    TIME_BASED = "time_based"
    EMERGENCY = "emergency"
    MANUAL = "manual"

class ExitStrategy(Enum):
    """Exit strategy types"""
    CONSERVATIVE = "conservative"  # Quick profits, tight stops
    AGGRESSIVE = "aggressive"     # Higher targets, wider stops
    SCALPING = "scalping"        # Very quick exits
    SWING = "swing"              # Longer term holds
    MOMENTUM = "momentum"        # Ride the momentum

@dataclass
class ExitRule:
    """Individual exit rule configuration"""
    trigger_type: ExitTrigger
    percentage_to_exit: float  # 0.0 to 1.0
    trigger_value: float
    is_active: bool = True
    priority: int = 1  # Higher priority = executes first
    description: str = ""

@dataclass
class PositionExit:
    """Exit execution record"""
    position_id: str
    token_address: str
    exit_trigger: ExitTrigger
    exit_price: float
    exit_amount: float
    exit_percentage: float
    profit_loss: float
    timestamp: float = field(default_factory=time.time)
    execution_time_ms: float = 0
    slippage: float = 0
    success: bool = True
    notes: str = ""

@dataclass
class PositionState:
    """Current state of a position for exit management"""
    position_id: str
    token_address: str
    entry_price: float
    current_price: float
    initial_amount: float
    remaining_amount: float
    entry_time: float
    last_update: float
    
    # Exit rules
    exit_rules: List[ExitRule] = field(default_factory=list)
    exit_history: List[PositionExit] = field(default_factory=list)
    
    # Tracking data
    highest_price: float = 0
    lowest_price: float = float('inf')
    price_history: deque = field(default_factory=lambda: deque(maxlen=100))
    volume_history: deque = field(default_factory=lambda: deque(maxlen=50))
    rsi_history: deque = field(default_factory=lambda: deque(maxlen=20))
    
    # Dynamic exit parameters
    trailing_stop_price: float = 0
    last_volume_spike: float = 0
    divergence_signals: int = 0
    
    @property
    def current_profit_pct(self) -> float:
        """Current profit/loss percentage"""
        if self.entry_price <= 0:
            return 0
        return (self.current_price / self.entry_price - 1) * 100
    
    @property
    def total_profit_loss(self) -> float:
        """Total realized + unrealized P&L"""
        unrealized = (self.current_price - self.entry_price) * self.remaining_amount
        realized = sum(exit.profit_loss for exit in self.exit_history)
        return realized + unrealized
    
    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable"""
        return self.current_profit_pct > 0

class AdvancedExitManager:
    """
    Sophisticated exit management system for maximizing profits
    
    Features:
    - Multiple exit strategies based on market conditions
    - Trailing stops with volatility adjustment
    - Partial profit taking at key levels
    - Technical indicator based exits
    - Emergency liquidation protocols
    """
    
    def __init__(self):
        self.positions: Dict[str, PositionState] = {}
        self.exit_templates = self._initialize_exit_templates()
        self.performance_metrics = {
            'total_exits': 0,
            'profitable_exits': 0,
            'average_hold_time': 0,
            'best_exit_profit': 0,
            'worst_exit_loss': 0,
            'total_profit_from_exits': 0
        }
        
        # Configuration
        self.config = {
            'min_profit_for_trail': 0.05,    # 5% profit before trailing
            'trail_step_pct': 0.02,          # 2% trailing step
            'max_hold_time_hours': 24,       # 24 hour max hold
            'volume_exhaustion_threshold': 0.3,  # 70% volume drop
            'rsi_divergence_threshold': 3,   # 3 divergence signals
            'emergency_loss_threshold': 0.25, # 25% emergency stop
        }
        
        logger.info("🚪 Advanced Exit Manager initialized")
    
    def _initialize_exit_templates(self) -> Dict[ExitStrategy, List[ExitRule]]:
        """Initialize predefined exit strategy templates"""
        templates = {
            ExitStrategy.CONSERVATIVE: [
                ExitRule(ExitTrigger.PROFIT_TARGET, 0.5, 0.15, description="15% profit - 50% exit"),
                ExitRule(ExitTrigger.PROFIT_TARGET, 0.3, 0.25, description="25% profit - 30% exit"),
                ExitRule(ExitTrigger.PROFIT_TARGET, 0.2, 0.40, description="40% profit - 20% exit"),
                ExitRule(ExitTrigger.TRAILING_STOP, 1.0, 0.08, description="8% trailing stop"),
                ExitRule(ExitTrigger.STOP_LOSS, 1.0, 0.10, description="10% stop loss"),
            ],
            
            ExitStrategy.AGGRESSIVE: [
                ExitRule(ExitTrigger.PROFIT_TARGET, 0.25, 0.30, description="30% profit - 25% exit"),
                ExitRule(ExitTrigger.PROFIT_TARGET, 0.25, 0.60, description="60% profit - 25% exit"),
                ExitRule(ExitTrigger.PROFIT_TARGET, 0.25, 1.00, description="100% profit - 25% exit"),
                ExitRule(ExitTrigger.PROFIT_TARGET, 0.25, 2.00, description="200% profit - 25% exit"),
                ExitRule(ExitTrigger.TRAILING_STOP, 1.0, 0.15, description="15% trailing stop"),
                ExitRule(ExitTrigger.STOP_LOSS, 1.0, 0.15, description="15% stop loss"),
            ],
            
            ExitStrategy.SCALPING: [
                ExitRule(ExitTrigger.PROFIT_TARGET, 0.8, 0.05, description="5% profit - 80% exit"),
                ExitRule(ExitTrigger.PROFIT_TARGET, 0.2, 0.10, description="10% profit - 20% exit"),
                ExitRule(ExitTrigger.TRAILING_STOP, 1.0, 0.03, description="3% trailing stop"),
                ExitRule(ExitTrigger.STOP_LOSS, 1.0, 0.05, description="5% stop loss"),
                ExitRule(ExitTrigger.TIME_BASED, 1.0, 1800, description="30 min time exit"), # 30 minutes
            ],
            
            ExitStrategy.MOMENTUM: [
                ExitRule(ExitTrigger.VOLUME_EXHAUSTION, 0.6, 0.5, description="Volume exhaustion - 60% exit"),
                ExitRule(ExitTrigger.RSI_DIVERGENCE, 0.4, 3, description="RSI divergence - 40% exit"),
                ExitRule(ExitTrigger.TRAILING_STOP, 1.0, 0.12, description="12% trailing stop"),
                ExitRule(ExitTrigger.STOP_LOSS, 1.0, 0.12, description="12% stop loss"),
            ]
        }
        
        # Add emergency exits to all strategies
        for strategy_rules in templates.values():
            strategy_rules.append(
                ExitRule(ExitTrigger.EMERGENCY, 1.0, 0.25, priority=10, description="25% emergency stop")
            )
        
        return templates
    
    async def add_position(
        self, 
        position_id: str,
        token_address: str,
        entry_price: float,
        amount: float,
        exit_strategy: ExitStrategy = ExitStrategy.CONSERVATIVE
    ) -> bool:
        """Add new position to exit management"""
        try:
            # Create position state
            position = PositionState(
                position_id=position_id,
                token_address=token_address,
                entry_price=entry_price,
                current_price=entry_price,
                initial_amount=amount,
                remaining_amount=amount,
                entry_time=time.time(),
                last_update=time.time(),
                highest_price=entry_price,
                lowest_price=entry_price
            )
            
            # Apply exit strategy template
            if exit_strategy in self.exit_templates:
                position.exit_rules = [rule for rule in self.exit_templates[exit_strategy]]
            
            # Initialize trailing stop
            position.trailing_stop_price = entry_price * (1 - 0.10)  # 10% initial trailing stop
            
            self.positions[position_id] = position
            
            logger.info(f"📈 Position added to exit manager: {position_id} | {exit_strategy.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False
    
    async def update_position_price(
        self, 
        position_id: str, 
        current_price: float, 
        volume: float = 0,
        rsi: float = 50
    ):
        """Update position with current market data"""
        try:
            if position_id not in self.positions:
                return
            
            position = self.positions[position_id]
            
            # Update price data
            old_price = position.current_price
            position.current_price = current_price
            position.last_update = time.time()
            
            # Track price extremes
            position.highest_price = max(position.highest_price, current_price)
            position.lowest_price = min(position.lowest_price, current_price)
            
            # Store history for analysis
            position.price_history.append({
                'price': current_price,
                'timestamp': time.time()
            })
            
            if volume > 0:
                position.volume_history.append({
                    'volume': volume,
                    'timestamp': time.time()
                })
            
            if rsi > 0:
                position.rsi_history.append({
                    'rsi': rsi,
                    'timestamp': time.time()
                })
            
            # Update trailing stop
            await self._update_trailing_stop(position)
            
            # Check for exit triggers
            await self._check_exit_triggers(position)
            
        except Exception as e:
            logger.error(f"Error updating position price: {e}")
    
    async def _update_trailing_stop(self, position: PositionState):
        """Update trailing stop loss level"""
        try:
            current_profit_pct = position.current_profit_pct / 100
            
            # Only start trailing if we have minimum profit
            if current_profit_pct < self.config['min_profit_for_trail']:
                return
            
            # Calculate new trailing stop
            trail_pct = self.config['trail_step_pct']
            
            # Adjust trailing distance based on volatility
            if len(position.price_history) >= 10:
                recent_prices = [p['price'] for p in list(position.price_history)[-10:]]
                volatility = np.std(recent_prices) / np.mean(recent_prices)
                
                # Wider trailing stop in high volatility
                if volatility > 0.05:  # 5% volatility
                    trail_pct = min(0.08, trail_pct * 2)  # Double trail distance, max 8%
            
            # Update trailing stop (only moves up for long positions)
            new_trail_stop = position.current_price * (1 - trail_pct)
            if new_trail_stop > position.trailing_stop_price:
                position.trailing_stop_price = new_trail_stop
                
                logger.debug(f"📈 Trailing stop updated: {position.position_id} -> {new_trail_stop:.6f}")
            
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
    
    async def _check_exit_triggers(self, position: PositionState):
        """Check all exit triggers for a position"""
        try:
            current_profit_pct = position.current_profit_pct / 100
            time_held = (time.time() - position.entry_time) / 3600  # hours
            
            # Sort exit rules by priority
            active_rules = [rule for rule in position.exit_rules if rule.is_active]
            active_rules.sort(key=lambda x: x.priority, reverse=True)
            
            for rule in active_rules:
                should_exit, exit_amount = await self._evaluate_exit_rule(position, rule)
                
                if should_exit and exit_amount > 0:
                    # Execute exit
                    await self._execute_exit(
                        position=position,
                        exit_trigger=rule.trigger_type,
                        exit_amount=exit_amount,
                        notes=rule.description
                    )
                    
                    # Deactivate rule if full exit
                    if rule.percentage_to_exit >= 1.0:
                        rule.is_active = False
                    
                    break  # Only execute highest priority trigger
            
        except Exception as e:
            logger.error(f"Error checking exit triggers: {e}")
    
    async def _evaluate_exit_rule(self, position: PositionState, rule: ExitRule) -> Tuple[bool, float]:
        """Evaluate if an exit rule should trigger"""
        try:
            current_profit_pct = position.current_profit_pct / 100
            time_held = (time.time() - position.entry_time)
            
            should_exit = False
            exit_amount = 0
            
            if rule.trigger_type == ExitTrigger.PROFIT_TARGET:
                if current_profit_pct >= rule.trigger_value:
                    should_exit = True
                    exit_amount = position.remaining_amount * rule.percentage_to_exit
            
            elif rule.trigger_type == ExitTrigger.STOP_LOSS:
                if current_profit_pct <= -rule.trigger_value:
                    should_exit = True
                    exit_amount = position.remaining_amount * rule.percentage_to_exit
            
            elif rule.trigger_type == ExitTrigger.TRAILING_STOP:
                if position.current_price <= position.trailing_stop_price:
                    should_exit = True
                    exit_amount = position.remaining_amount * rule.percentage_to_exit
            
            elif rule.trigger_type == ExitTrigger.TIME_BASED:
                if time_held >= rule.trigger_value:  # rule.trigger_value in seconds
                    should_exit = True
                    exit_amount = position.remaining_amount * rule.percentage_to_exit
            
            elif rule.trigger_type == ExitTrigger.VOLUME_EXHAUSTION:
                volume_exhaustion = await self._detect_volume_exhaustion(position)
                if volume_exhaustion >= rule.trigger_value:
                    should_exit = True
                    exit_amount = position.remaining_amount * rule.percentage_to_exit
            
            elif rule.trigger_type == ExitTrigger.RSI_DIVERGENCE:
                divergence_count = await self._detect_rsi_divergence(position)
                if divergence_count >= rule.trigger_value:
                    should_exit = True
                    exit_amount = position.remaining_amount * rule.percentage_to_exit
            
            elif rule.trigger_type == ExitTrigger.EMERGENCY:
                if current_profit_pct <= -rule.trigger_value:
                    should_exit = True
                    exit_amount = position.remaining_amount  # Emergency = full exit
            
            return should_exit, exit_amount
            
        except Exception as e:
            logger.error(f"Error evaluating exit rule: {e}")
            return False, 0
    
    async def _detect_volume_exhaustion(self, position: PositionState) -> float:
        """Detect volume exhaustion (momentum ending)"""
        try:
            if len(position.volume_history) < 10:
                return 0
            
            volumes = [v['volume'] for v in list(position.volume_history)]
            
            # Compare recent volume to peak volume
            recent_volume = np.mean(volumes[-5:])  # Last 5 periods
            peak_volume = max(volumes)
            
            if peak_volume > 0:
                volume_drop = 1 - (recent_volume / peak_volume)
                return volume_drop
            
            return 0
            
        except Exception as e:
            logger.error(f"Error detecting volume exhaustion: {e}")
            return 0
    
    async def _detect_rsi_divergence(self, position: PositionState) -> int:
        """Detect bearish RSI divergence"""
        try:
            if len(position.rsi_history) < 5 or len(position.price_history) < 5:
                return 0
            
            rsi_values = [r['rsi'] for r in list(position.rsi_history)[-5:]]
            price_values = [p['price'] for p in list(position.price_history)[-5:]]
            
            # Simple divergence detection: price making higher highs, RSI making lower highs
            divergence_signals = 0
            
            for i in range(1, len(rsi_values)):
                if price_values[i] > price_values[i-1] and rsi_values[i] < rsi_values[i-1]:
                    divergence_signals += 1
            
            return divergence_signals
            
        except Exception as e:
            logger.error(f"Error detecting RSI divergence: {e}")
            return 0
    
    async def _execute_exit(
        self, 
        position: PositionState, 
        exit_trigger: ExitTrigger,
        exit_amount: float,
        notes: str = ""
    ):
        """Execute position exit"""
        try:
            start_time = time.perf_counter()
            
            # Calculate exit details
            exit_price = position.current_price
            exit_percentage = exit_amount / position.initial_amount
            profit_loss = (exit_price - position.entry_price) * exit_amount
            
            # Create exit record
            exit_record = PositionExit(
                position_id=position.position_id,
                token_address=position.token_address,
                exit_trigger=exit_trigger,
                exit_price=exit_price,
                exit_amount=exit_amount,
                exit_percentage=exit_percentage,
                profit_loss=profit_loss,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                notes=notes
            )
            
            # Update position
            position.remaining_amount -= exit_amount
            position.exit_history.append(exit_record)
            
            # Update performance metrics
            self._update_performance_metrics(exit_record)
            
            logger.info(
                f"🚪 EXIT EXECUTED: {position.position_id} | "
                f"Trigger: {exit_trigger.value} | "
                f"Amount: {exit_amount:.4f} | "
                f"Price: {exit_price:.6f} | "
                f"P&L: {profit_loss:.4f} | "
                f"Notes: {notes}"
            )
            
            # Remove position if fully exited
            if position.remaining_amount <= 0.001:  # Small threshold for rounding
                del self.positions[position.position_id]
                logger.info(f"📊 Position fully closed: {position.position_id}")
            
            return exit_record
            
        except Exception as e:
            logger.error(f"Error executing exit: {e}")
            return None
    
    def _update_performance_metrics(self, exit_record: PositionExit):
        """Update exit performance metrics"""
        try:
            self.performance_metrics['total_exits'] += 1
            
            if exit_record.profit_loss > 0:
                self.performance_metrics['profitable_exits'] += 1
            
            self.performance_metrics['total_profit_from_exits'] += exit_record.profit_loss
            
            # Update best/worst records
            if exit_record.profit_loss > self.performance_metrics['best_exit_profit']:
                self.performance_metrics['best_exit_profit'] = exit_record.profit_loss
            
            if exit_record.profit_loss < self.performance_metrics['worst_exit_loss']:
                self.performance_metrics['worst_exit_loss'] = exit_record.profit_loss
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def force_exit_position(self, position_id: str, reason: str = "Manual exit") -> bool:
        """Force immediate exit of entire position"""
        try:
            if position_id not in self.positions:
                logger.warning(f"Position not found for force exit: {position_id}")
                return False
            
            position = self.positions[position_id]
            
            await self._execute_exit(
                position=position,
                exit_trigger=ExitTrigger.MANUAL,
                exit_amount=position.remaining_amount,
                notes=reason
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error force exiting position: {e}")
            return False
    
    async def update_exit_strategy(self, position_id: str, new_strategy: ExitStrategy) -> bool:
        """Update exit strategy for existing position"""
        try:
            if position_id not in self.positions:
                return False
            
            position = self.positions[position_id]
            
            # Apply new exit rules
            if new_strategy in self.exit_templates:
                position.exit_rules = [rule for rule in self.exit_templates[new_strategy]]
                logger.info(f"📋 Exit strategy updated: {position_id} -> {new_strategy.value}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating exit strategy: {e}")
            return False
    
    def get_position_status(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get current position status"""
        try:
            if position_id not in self.positions:
                return None
            
            position = self.positions[position_id]
            
            return {
                'position_id': position.position_id,
                'token_address': position.token_address,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'current_profit_pct': position.current_profit_pct,
                'remaining_amount': position.remaining_amount,
                'total_profit_loss': position.total_profit_loss,
                'time_held_hours': (time.time() - position.entry_time) / 3600,
                'trailing_stop_price': position.trailing_stop_price,
                'active_exit_rules': len([r for r in position.exit_rules if r.is_active]),
                'exit_count': len(position.exit_history),
                'is_profitable': position.is_profitable
            }
            
        except Exception as e:
            logger.error(f"Error getting position status: {e}")
            return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get exit management performance summary"""
        total_exits = self.performance_metrics['total_exits']
        
        return {
            'total_exits': total_exits,
            'profitable_exits': self.performance_metrics['profitable_exits'],
            'win_rate': (self.performance_metrics['profitable_exits'] / total_exits * 100) if total_exits > 0 else 0,
            'total_profit': self.performance_metrics['total_profit_from_exits'],
            'best_exit': self.performance_metrics['best_exit_profit'],
            'worst_exit': self.performance_metrics['worst_exit_loss'],
            'active_positions': len(self.positions),
            'avg_profit_per_exit': (self.performance_metrics['total_profit_from_exits'] / total_exits) if total_exits > 0 else 0
        } 
"""
Stop loss and take profit management system.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum
from loguru import logger

class StopLossType(Enum):
    """Types of stop loss orders"""
    FIXED = "fixed"
    TRAILING = "trailing"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    TIME_BASED = "time_based"

class OrderStatus(Enum):
    """Status of stop loss orders"""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

@dataclass
class StopLossOrder:
    """Represents a stop loss order"""
    id: str
    token_address: str
    token_symbol: str
    order_type: StopLossType
    entry_price: float
    current_price: float
    stop_price: float
    take_profit_price: Optional[float]
    position_size: float
    created_at: float
    updated_at: float
    status: OrderStatus = OrderStatus.ACTIVE
    
    # Trailing stop parameters
    trail_amount: Optional[float] = None
    trail_percentage: Optional[float] = None
    highest_price: Optional[float] = None
    
    # Time-based parameters
    max_hold_time: Optional[float] = None  # seconds
    
    # ATR parameters
    atr_multiplier: Optional[float] = None
    atr_value: Optional[float] = None
    
    # Callbacks
    on_trigger: Optional[Callable] = field(default=None, repr=False)
    
    @property
    def is_expired(self) -> bool:
        """Check if order is expired"""
        if self.max_hold_time is None:
            return False
        return time.time() - self.created_at > self.max_hold_time
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        return (self.current_price - self.entry_price) * self.position_size
    
    @property
    def unrealized_pnl_percentage(self) -> float:
        """Calculate unrealized P&L percentage"""
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100

class StopLossManager:
    """
    Manages stop loss and take profit orders with various strategies.
    """
    
    def __init__(self):
        """Initialize stop loss manager."""
        self.active_orders: Dict[str, StopLossOrder] = {}
        self.triggered_orders: List[StopLossOrder] = []
        self.order_counter = 0
        
        # Default parameters
        self.default_stop_loss_percentage = 0.05  # 5%
        self.default_take_profit_percentage = 0.15  # 15%
        self.default_trail_percentage = 0.03  # 3%
        self.default_atr_multiplier = 2.0
        self.max_hold_time_hours = 24  # 24 hours default
        
        logger.info("StopLossManager initialized")
    
    def create_stop_loss_order(
        self,
        token_address: str,
        token_symbol: str,
        entry_price: float,
        position_size: float,
        stop_loss_type: StopLossType = StopLossType.FIXED,
        stop_loss_percentage: Optional[float] = None,
        take_profit_percentage: Optional[float] = None,
        trail_percentage: Optional[float] = None,
        atr_multiplier: Optional[float] = None,
        atr_value: Optional[float] = None,
        max_hold_hours: Optional[float] = None,
        on_trigger_callback: Optional[Callable] = None
    ) -> str:
        """
        Create a new stop loss order.
        
        Returns:
            Order ID
        """
        try:
            self.order_counter += 1
            order_id = f"SL_{self.order_counter}_{int(time.time())}"
            
            # Set default values
            stop_loss_pct = stop_loss_percentage or self.default_stop_loss_percentage
            take_profit_pct = take_profit_percentage or self.default_take_profit_percentage
            
            # Calculate stop loss price
            stop_price = self._calculate_initial_stop_price(
                entry_price, stop_loss_type, stop_loss_pct, atr_multiplier, atr_value
            )
            
            # Calculate take profit price
            take_profit_price = entry_price * (1 + take_profit_pct) if take_profit_pct else None
            
            # Calculate max hold time
            max_hold_time = None
            if max_hold_hours:
                max_hold_time = max_hold_hours * 3600  # Convert to seconds
            elif stop_loss_type == StopLossType.TIME_BASED:
                max_hold_time = self.max_hold_time_hours * 3600
            
            order = StopLossOrder(
                id=order_id,
                token_address=token_address,
                token_symbol=token_symbol,
                order_type=stop_loss_type,
                entry_price=entry_price,
                current_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                position_size=position_size,
                created_at=time.time(),
                updated_at=time.time(),
                trail_amount=None,
                trail_percentage=trail_percentage or self.default_trail_percentage,
                highest_price=entry_price,
                max_hold_time=max_hold_time,
                atr_multiplier=atr_multiplier or self.default_atr_multiplier,
                atr_value=atr_value,
                on_trigger=on_trigger_callback
            )
            
            self.active_orders[order_id] = order
            
            logger.bind(RISK=True).info(
                f"📋 Created {stop_loss_type.value} stop loss for {token_symbol}: "
                f"Entry=${entry_price:.6f}, Stop=${stop_price:.6f}, "
                f"Size={position_size:.4f} SOL"
            )
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error creating stop loss order: {str(e)}")
            return ""
    
    def update_prices(self, price_updates: Dict[str, float]) -> List[str]:
        """
        Update prices for all active orders and check for triggers.
        
        Args:
            price_updates: Dict of token_address -> current_price
            
        Returns:
            List of triggered order IDs
        """
        triggered_orders = []
        
        for order_id, order in list(self.active_orders.items()):
            if order.token_address in price_updates:
                new_price = price_updates[order.token_address]
                
                # Update order with new price
                order.current_price = new_price
                order.updated_at = time.time()
                
                # Update trailing stop if applicable
                if order.order_type == StopLossType.TRAILING:
                    self._update_trailing_stop(order, new_price)
                
                # Check for triggers
                if self._check_order_trigger(order):
                    triggered_orders.append(order_id)
                    self._trigger_order(order)
        
        return triggered_orders
    
    def _calculate_initial_stop_price(
        self,
        entry_price: float,
        stop_type: StopLossType,
        stop_percentage: float,
        atr_multiplier: Optional[float],
        atr_value: Optional[float]
    ) -> float:
        """Calculate initial stop loss price based on type."""
        if stop_type == StopLossType.ATR_BASED and atr_value and atr_multiplier:
            return entry_price - (atr_value * atr_multiplier)
        else:
            # Default to percentage-based
            return entry_price * (1 - stop_percentage)
    
    def _update_trailing_stop(self, order: StopLossOrder, new_price: float) -> None:
        """Update trailing stop loss price."""
        if order.highest_price is None:
            order.highest_price = new_price
        
        # Update highest price if new high
        if new_price > order.highest_price:
            order.highest_price = new_price
            
            # Update trailing stop price
            if order.trail_percentage:
                new_stop_price = order.highest_price * (1 - order.trail_percentage)
                
                # Only move stop loss up, never down
                if new_stop_price > order.stop_price:
                    old_stop = order.stop_price
                    order.stop_price = new_stop_price
                    
                    logger.bind(RISK=True).debug(
                        f"📈 Updated trailing stop for {order.token_symbol}: "
                        f"${old_stop:.6f} → ${new_stop_price:.6f} "
                        f"(High: ${order.highest_price:.6f})"
                    )
    
    def _check_order_trigger(self, order: StopLossOrder) -> bool:
        """Check if an order should be triggered."""
        # Check time-based expiration
        if order.is_expired:
            logger.bind(RISK=True).info(
                f"⏰ Time-based stop triggered for {order.token_symbol} "
                f"(held for {(time.time() - order.created_at)/3600:.1f} hours)"
            )
            return True
        
        # Check stop loss trigger
        if order.current_price <= order.stop_price:
            logger.bind(RISK=True).warning(
                f"🛑 Stop loss triggered for {order.token_symbol}: "
                f"${order.current_price:.6f} <= ${order.stop_price:.6f} "
                f"(Loss: {order.unrealized_pnl_percentage:.2f}%)"
            )
            return True
        
        # Check take profit trigger
        if order.take_profit_price and order.current_price >= order.take_profit_price:
            logger.bind(RISK=True).success(
                f"🎯 Take profit triggered for {order.token_symbol}: "
                f"${order.current_price:.6f} >= ${order.take_profit_price:.6f} "
                f"(Profit: {order.unrealized_pnl_percentage:.2f}%)"
            )
            return True
        
        return False
    
    def _trigger_order(self, order: StopLossOrder) -> None:
        """Trigger an order and move it to triggered list."""
        order.status = OrderStatus.TRIGGERED
        
        # Remove from active orders
        if order.id in self.active_orders:
            del self.active_orders[order.id]
        
        # Add to triggered orders
        self.triggered_orders.append(order)
        
        # Call trigger callback if provided
        if order.on_trigger:
            try:
                order.on_trigger(order)
            except Exception as e:
                logger.error(f"Error in stop loss trigger callback: {str(e)}")
        
        logger.bind(RISK=True).info(
            f"⚡ Order {order.id} triggered for {order.token_symbol} "
            f"(P&L: {order.unrealized_pnl_percentage:+.2f}%)"
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.active_orders[order_id]
            
            logger.bind(RISK=True).info(f"❌ Cancelled stop loss order {order_id} for {order.token_symbol}")
            return True
        
        return False
    
    def get_order(self, order_id: str) -> Optional[StopLossOrder]:
        """Get an order by ID."""
        return self.active_orders.get(order_id)
    
    def get_orders_for_token(self, token_address: str) -> List[StopLossOrder]:
        """Get all active orders for a specific token."""
        return [order for order in self.active_orders.values() 
                if order.token_address == token_address]
    
    def get_portfolio_risk_summary(self) -> Dict:
        """Get summary of portfolio risk from active orders."""
        total_position_value = 0
        total_max_loss = 0
        orders_by_type = {}
        
        for order in self.active_orders.values():
            position_value = order.position_size * order.current_price
            max_loss = order.position_size * (order.entry_price - order.stop_price)
            
            total_position_value += position_value
            total_max_loss += max_loss
            
            order_type = order.order_type.value
            if order_type not in orders_by_type:
                orders_by_type[order_type] = {"count": 0, "value": 0}
            
            orders_by_type[order_type]["count"] += 1
            orders_by_type[order_type]["value"] += position_value
        
        return {
            "active_orders": len(self.active_orders),
            "total_position_value": total_position_value,
            "total_max_loss": total_max_loss,
            "max_loss_percentage": (total_max_loss / total_position_value * 100) if total_position_value > 0 else 0,
            "orders_by_type": orders_by_type,
            "triggered_orders_today": len([o for o in self.triggered_orders 
                                         if time.time() - o.updated_at < 86400])
        }
    
    def cleanup_old_orders(self, max_age_hours: float = 168) -> int:
        """Clean up old triggered orders (default: 1 week)."""
        max_age_seconds = max_age_hours * 3600
        current_time = time.time()
        
        initial_count = len(self.triggered_orders)
        self.triggered_orders = [
            order for order in self.triggered_orders
            if current_time - order.updated_at < max_age_seconds
        ]
        
        cleaned_count = initial_count - len(self.triggered_orders)
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old triggered orders")
        
        return cleaned_count
    
    def update_order_parameters(
        self,
        order_id: str,
        stop_loss_percentage: Optional[float] = None,
        take_profit_percentage: Optional[float] = None,
        trail_percentage: Optional[float] = None
    ) -> bool:
        """Update parameters of an existing order."""
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        
        if stop_loss_percentage is not None:
            order.stop_price = order.entry_price * (1 - stop_loss_percentage)
            
        if take_profit_percentage is not None:
            order.take_profit_price = order.entry_price * (1 + take_profit_percentage)
            
        if trail_percentage is not None:
            order.trail_percentage = trail_percentage
            
        order.updated_at = time.time()
        
        logger.bind(RISK=True).info(f"📝 Updated parameters for order {order_id}")
        return True 
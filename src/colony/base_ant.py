"""
Base Ant Class - Core functionality shared across all ant types
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

class AntRole(Enum):
    """Hierarchy roles in the Ant system"""
    FOUNDING_QUEEN = "founding_queen"
    QUEEN = "queen"
    WORKER = "worker"
    DRONE = "drone"
    ACCOUNTING = "accounting"
    PRINCESS = "princess"

class AntStatus(Enum):
    """Operational status of Ant agents"""
    ACTIVE = "active"
    SPLITTING = "splitting"
    MERGING = "merging"
    RETIRING = "retiring"
    DORMANT = "dormant"
    ERROR = "error"

@dataclass
class AntCapital:
    """Capital management for Ant agents"""
    current_balance: float = 0.0
    allocated_capital: float = 0.0
    available_capital: float = 0.0
    total_trades: int = 0
    profit_loss: float = 0.0
    hedged_amount: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    def update_balance(self, new_balance: float):
        """Update capital balance and derived metrics"""
        self.profit_loss += (new_balance - self.current_balance)
        self.current_balance = new_balance
        self.available_capital = max(0, new_balance - self.allocated_capital)
        self.last_updated = time.time()
    
    def allocate_capital(self, amount: float) -> bool:
        """Allocate capital for trading operations"""
        if self.available_capital >= amount:
            self.allocated_capital += amount
            self.available_capital -= amount
            return True
        return False
    
    def release_capital(self, amount: float):
        """Release allocated capital back to available pool"""
        self.allocated_capital = max(0, self.allocated_capital - amount)
        self.available_capital += amount

@dataclass
class AntPerformance:
    """Performance tracking for Ant agents"""
    total_trades: int = 0
    successful_trades: int = 0
    total_profit: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    average_trade_time: float = 0.0
    risk_score: float = 0.5
    efficiency_score: float = 0.0
    last_trade_time: float = 0.0
    compound_factor: float = 1.0
    trades_this_cycle: int = 0
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        return (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
    
    @property
    def profit_per_trade(self) -> float:
        """Calculate average profit per trade"""
        return self.total_profit / self.total_trades if self.total_trades > 0 else 0.0
    
    def update_trade_result(self, profit: float, trade_time: float, success: bool):
        """Update performance metrics with new trade result"""
        self.total_trades += 1
        self.trades_this_cycle += 1
        self.total_profit += profit
        self.last_trade_time = time.time()
        
        if success:
            self.successful_trades += 1
            # Update compound factor for successful trades
            if profit > 0:
                self.compound_factor *= (1 + profit / 100)  # Assuming profit in percentage
        
        # Update best/worst trade
        if profit > self.best_trade:
            self.best_trade = profit
        if profit < self.worst_trade:
            self.worst_trade = profit
        
        # Update average trade time
        if self.total_trades == 1:
            self.average_trade_time = trade_time
        else:
            self.average_trade_time = (self.average_trade_time * (self.total_trades - 1) + trade_time) / self.total_trades

class BaseAnt(ABC):
    """Base class for all Ant agents in the hierarchy"""
    
    def __init__(self, ant_id: str, role: AntRole, parent_id: Optional[str] = None):
        self.ant_id = ant_id
        self.role = role
        self.parent_id = parent_id
        self.status = AntStatus.ACTIVE
        self.created_at = time.time()
        self.last_activity = time.time()
        
        # Core components
        self.capital = AntCapital()
        self.performance = AntPerformance()
        self.children: List[str] = []
        self.metadata: Dict[str, Any] = {}
        
        # Configuration based on role
        self.config = self._get_role_config()
        
        logger.info(f"Created {role.value} ant: {ant_id}")
        
    def _get_role_config(self) -> Dict[str, Any]:
        """Get configuration based on ant role"""
        configs = {
            AntRole.FOUNDING_QUEEN: {
                "max_children": 10,
                "split_threshold": 20.0,  # 20 SOL to create new Queen
                "merge_threshold": 1.0,   # Merge when below 1 SOL
                "max_trades": float('inf'),
                "retirement_trades": None,
                "compound_layers": ["monetary", "worker", "carwash", "intelligence", "data"]
            },
            AntRole.QUEEN: {
                "max_children": 50,
                "split_threshold": 2.0,   # 2 SOL to create new Worker
                "merge_threshold": 0.5,   # Merge when below 0.5 SOL
                "max_trades": float('inf'),
                "retirement_trades": None,
                "target_split_amount": 1500.0  # $1500 for Queen splitting
            },
            AntRole.WORKER: {
                "max_children": 0,
                "split_threshold": 2.0,   # 2 SOL creates 5 workers
                "merge_threshold": 0.1,   # Merge when below 0.1 SOL
                "max_trades": 10,
                "retirement_trades": (5, 10),  # Retire after 5-10 trades
                "target_return_range": (1.03, 1.50),  # 1.03x-1.50x returns
                "trades_per_coin": (5, 10)
            },
            AntRole.DRONE: {
                "max_children": 0,
                "split_threshold": None,
                "merge_threshold": None,
                "max_trades": float('inf'),
                "retirement_trades": None,
                "ai_sync_interval": 60  # Sync between Grok and LLM every 60 seconds
            },
            AntRole.ACCOUNTING: {
                "max_children": 0,
                "split_threshold": None,
                "merge_threshold": None,
                "max_trades": float('inf'),
                "retirement_trades": None,
                "hedge_percentage": 0.1  # Hedge 10% of capital
            },
            AntRole.PRINCESS: {
                "max_children": 0,
                "split_threshold": None,
                "merge_threshold": None,
                "max_trades": float('inf'),
                "retirement_trades": None,
                "accumulation_threshold": 10.0  # Accumulate up to 10 SOL before action
            }
        }
        return configs[self.role]
    
    def should_split(self) -> bool:
        """Determine if this ant should split based on capital and performance"""
        if not self.config["split_threshold"]:
            return False
        
        # Check capital threshold
        capital_ready = self.capital.available_capital >= self.config["split_threshold"]
        
        # Check performance threshold (must be profitable for workers)
        if self.role == AntRole.WORKER:
            performance_ready = self.performance.total_profit > 0 and self.performance.win_rate > 50.0
        else:
            performance_ready = True  # Queens can split based on capital alone
        
        # Check children limit
        children_limit_ok = len(self.children) < self.config["max_children"]
        
        return capital_ready and performance_ready and children_limit_ok
    
    def should_merge(self) -> bool:
        """Determine if this ant should be merged due to poor performance"""
        if not self.config["merge_threshold"]:
            return False
        
        # Check if capital is below merge threshold
        capital_low = self.capital.current_balance < self.config["merge_threshold"]
        
        # Check if performance is poor (for workers)
        if self.role == AntRole.WORKER:
            performance_poor = (
                self.performance.total_trades >= 5 and 
                (self.performance.win_rate < 30.0 or self.performance.total_profit < -0.1)
            )
        else:
            performance_poor = False
        
        return capital_low or performance_poor
    
    def should_retire(self) -> bool:
        """Determine if this ant should retire based on trade count"""
        if not self.config["retirement_trades"]:
            return False
        
        min_trades, max_trades = self.config["retirement_trades"]
        
        # Retire if reached max trades or exceeded performance expectations
        if self.performance.total_trades >= max_trades:
            return True
        
        # Early retirement if performance is exceptional and min trades reached
        if (self.performance.total_trades >= min_trades and 
            self.performance.win_rate > 80.0 and 
            self.performance.total_profit > 0.5):
            return True
        
        return False
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""
        return {
            "ant_id": self.ant_id,
            "role": self.role.value,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "children_count": len(self.children),
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "last_activity": datetime.fromtimestamp(self.last_activity).isoformat(),
            "capital": {
                "current_balance": self.capital.current_balance,
                "available_capital": self.capital.available_capital,
                "allocated_capital": self.capital.allocated_capital,
                "profit_loss": self.capital.profit_loss,
                "hedged_amount": self.capital.hedged_amount
            },
            "performance": {
                "total_trades": self.performance.total_trades,
                "win_rate": self.performance.win_rate,
                "total_profit": self.performance.total_profit,
                "profit_per_trade": self.performance.profit_per_trade,
                "compound_factor": self.performance.compound_factor,
                "trades_this_cycle": self.performance.trades_this_cycle
            },
            "flags": {
                "should_split": self.should_split(),
                "should_merge": self.should_merge(),
                "should_retire": self.should_retire()
            }
        }
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the ant agent"""
        pass
    
    @abstractmethod
    async def execute_cycle(self) -> Dict[str, Any]:
        """Execute one operational cycle"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup resources when retiring/merging"""
        pass 
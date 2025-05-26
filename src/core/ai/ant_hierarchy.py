"""
Ant Hierarchy System - Core implementation of the Ant Bot architecture

This module implements the hierarchical structure:
- Founding Ant Queen: Manages multiple Ant Queens, handles system-wide coordination
- Ant Queen: Manages Worker Ants (Princesses), handles 2+ SOL capital management
- Ant Princess: Individual trading agents, retire after 5-10 trades

Key Features:
- Automatic capital-based splitting and merging
- Performance-based worker lifecycle management
- Hierarchical decision making and coordination
- Dynamic resource allocation
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from .grok_engine import GrokEngine
from ..local_llm import LocalLLM
from ...services.wallet_manager import WalletManager
from ..portfolio_risk_manager import PortfolioRiskManager

logger = logging.getLogger(__name__)

class AntRole(Enum):
    """Hierarchy roles in the Ant system"""
    FOUNDING_QUEEN = "founding_queen"
    QUEEN = "queen"
    PRINCESS = "princess"

class AntStatus(Enum):
    """Operational status of Ant agents"""
    ACTIVE = "active"
    SPLITTING = "splitting"
    MERGING = "merging"
    RETIRING = "retiring"
    DORMANT = "dormant"

@dataclass
class AntCapital:
    """Capital management for Ant agents"""
    current_balance: float = 0.0
    allocated_capital: float = 0.0
    available_capital: float = 0.0
    total_trades: int = 0
    profit_loss: float = 0.0
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
        self.total_profit += profit
        self.last_trade_time = time.time()
        
        if success:
            self.successful_trades += 1
        
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

class BaseAnt:
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
        
        # Configuration based on role
        self.config = self._get_role_config()
        
    def _get_role_config(self) -> Dict[str, Any]:
        """Get configuration based on ant role"""
        configs = {
            AntRole.FOUNDING_QUEEN: {
                "max_children": 10,
                "split_threshold": 20.0,  # 20 SOL to create new Queen
                "merge_threshold": 0.5,   # Merge when below 0.5 SOL
                "max_trades": float('inf'),
                "retirement_trades": None
            },
            AntRole.QUEEN: {
                "max_children": 50,
                "split_threshold": 2.0,   # 2 SOL to create new Princess
                "merge_threshold": 0.1,   # Merge when below 0.1 SOL
                "max_trades": float('inf'),
                "retirement_trades": None
            },
            AntRole.PRINCESS: {
                "max_children": 0,
                "split_threshold": None,
                "merge_threshold": None,
                "max_trades": 10,
                "retirement_trades": (5, 10)  # Retire after 5-10 trades
            }
        }
        return configs[self.role]
    
    def should_split(self) -> bool:
        """Determine if this ant should split based on capital and performance"""
        if not self.config["split_threshold"]:
            return False
        
        # Check capital threshold
        capital_ready = self.capital.available_capital >= self.config["split_threshold"]
        
        # Check performance threshold (must be profitable)
        performance_ready = self.performance.total_profit > 0 and self.performance.win_rate > 50.0
        
        # Check children limit
        children_limit_ok = len(self.children) < self.config["max_children"]
        
        return capital_ready and performance_ready and children_limit_ok
    
    def should_merge(self) -> bool:
        """Determine if this ant should be merged due to poor performance"""
        if not self.config["merge_threshold"]:
            return False
        
        # Check if capital is below merge threshold
        capital_low = self.capital.current_balance < self.config["merge_threshold"]
        
        # Check if performance is poor
        performance_poor = (
            self.performance.total_trades >= 5 and 
            (self.performance.win_rate < 30.0 or self.performance.total_profit < -0.1)
        )
        
        return capital_low or performance_poor
    
    def should_retire(self) -> bool:
        """Determine if this ant should retire (mainly for Princesses)"""
        if not self.config["retirement_trades"]:
            return False
        
        min_trades, max_trades = self.config["retirement_trades"]
        
        # Retire if reached max trades
        if self.performance.total_trades >= max_trades:
            return True
        
        # Retire if reached min trades and performance criteria met
        if self.performance.total_trades >= min_trades:
            # Retire if profitable or risk is too high
            return (
                self.performance.total_profit > 0.01 or  # Made profit
                self.performance.win_rate < 20.0 or      # Very poor performance
                self.performance.risk_score > 0.8        # Too risky
            )
        
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
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "capital": {
                "current_balance": self.capital.current_balance,
                "allocated_capital": self.capital.allocated_capital,
                "available_capital": self.capital.available_capital,
                "profit_loss": self.capital.profit_loss
            },
            "performance": {
                "total_trades": self.performance.total_trades,
                "win_rate": self.performance.win_rate,
                "total_profit": self.performance.total_profit,
                "profit_per_trade": self.performance.profit_per_trade,
                "risk_score": self.performance.risk_score
            },
            "should_split": self.should_split(),
            "should_merge": self.should_merge(),
            "should_retire": self.should_retire()
        }

class AntPrincess(BaseAnt):
    """Individual trading agent (Worker Ant) with 5-10 trade lifecycle"""
    
    def __init__(self, ant_id: str, parent_id: str, initial_capital: float = 0.5):
        super().__init__(ant_id, AntRole.PRINCESS, parent_id)
        self.capital.current_balance = initial_capital
        self.capital.available_capital = initial_capital
        
        # Trading components
        self.grok_engine = None
        self.local_llm = None
        self.active_positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        
        # Princess-specific tracking
        self.target_trades = None
        self.specialization = None  # Can specialize in certain token types
        
    async def initialize(self, grok_engine: GrokEngine, local_llm: LocalLLM):
        """Initialize the Princess with AI components"""
        self.grok_engine = grok_engine
        self.local_llm = local_llm
        
        # Determine target trades (5-10 range)
        import random
        self.target_trades = random.randint(5, 10)
        
        logger.info(f"Princess {self.ant_id} initialized with {self.capital.current_balance} SOL, target: {self.target_trades} trades")
        
    async def analyze_opportunity(self, token_address: str, market_data: Dict) -> Optional[Dict]:
        """Analyze trading opportunity using AI components"""
        try:
            self.update_activity()
            
            # Check if we should retire
            if self.should_retire():
                logger.info(f"Princess {self.ant_id} should retire after {self.performance.total_trades} trades")
                return None
            
            # Get sentiment from Grok
            sentiment_analysis = await self.grok_engine.analyze_market(market_data)
            if not sentiment_analysis or "error" in sentiment_analysis:
                return None
            
            # Get technical analysis from Local LLM
            technical_analysis = await self.local_llm.analyze_market(market_data)
            if not technical_analysis or "error" in technical_analysis:
                return None
            
            # Combine analyses for decision
            decision = self._make_trading_decision(sentiment_analysis, technical_analysis, market_data)
            
            if decision["action"] != "hold":
                logger.info(f"Princess {self.ant_id} found opportunity: {decision['action']} {token_address}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Princess {self.ant_id} analysis error: {str(e)}")
            return None
    
    def _make_trading_decision(self, sentiment: Dict, technical: Dict, market_data: Dict) -> Dict:
        """Combine AI analyses to make trading decision"""
        try:
            # Extract confidence scores
            sentiment_score = sentiment.get("confidence", 0.0)
            technical_score = technical.get("confidence", 0.0)
            
            # Weight the analyses (Grok for sentiment, LLM for technical)
            sentiment_weight = 0.4
            technical_weight = 0.6
            
            combined_score = (sentiment_score * sentiment_weight) + (technical_score * technical_weight)
            
            # Determine action based on combined score
            if combined_score >= 0.7:
                action = "buy"
            elif combined_score <= -0.7:
                action = "sell"
            else:
                action = "hold"
            
            # Calculate position size based on confidence and available capital
            max_position_size = min(self.capital.available_capital * 0.8, 0.1)  # Max 80% of capital or 0.1 SOL
            position_size = max_position_size * abs(combined_score)
            
            return {
                "action": action,
                "position_size": position_size,
                "confidence": abs(combined_score),
                "sentiment_score": sentiment_score,
                "technical_score": technical_score,
                "reasoning": f"Sentiment: {sentiment_score:.2f}, Technical: {technical_score:.2f}, Combined: {combined_score:.2f}",
                "token_address": market_data.get("token_address"),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Princess {self.ant_id} decision error: {str(e)}")
            return {"action": "hold", "position_size": 0, "confidence": 0}
    
    async def execute_trade(self, decision: Dict, wallet_manager: WalletManager) -> Dict:
        """Execute trading decision"""
        try:
            if decision["action"] == "hold" or decision["position_size"] <= 0:
                return {"success": False, "message": "No trade to execute"}
            
            # Allocate capital for trade
            if not self.capital.allocate_capital(decision["position_size"]):
                return {"success": False, "message": "Insufficient capital"}
            
            # Simulate trade execution (replace with real trading logic)
            trade_start = time.time()
            
            # For simulation - random profit/loss
            import random
            success_probability = decision["confidence"]
            success = random.random() < success_probability
            
            if success:
                profit = decision["position_size"] * random.uniform(0.02, 0.15)  # 2-15% profit
            else:
                profit = -decision["position_size"] * random.uniform(0.01, 0.08)  # 1-8% loss
            
            trade_time = time.time() - trade_start
            
            # Update capital and performance
            self.capital.release_capital(decision["position_size"])
            self.capital.update_balance(self.capital.current_balance + profit)
            self.performance.update_trade_result(profit, trade_time, success)
            
            # Record trade
            trade_record = {
                "trade_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "token_address": decision["token_address"],
                "action": decision["action"],
                "position_size": decision["position_size"],
                "profit": profit,
                "success": success,
                "confidence": decision["confidence"],
                "reasoning": decision["reasoning"]
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"Princess {self.ant_id} executed trade: {profit:.4f} SOL {'profit' if profit > 0 else 'loss'}")
            
            return {
                "success": True,
                "trade_record": trade_record,
                "new_balance": self.capital.current_balance,
                "trades_completed": self.performance.total_trades
            }
            
        except Exception as e:
            logger.error(f"Princess {self.ant_id} trade execution error: {str(e)}")
            return {"success": False, "message": str(e)}

class AntQueen(BaseAnt):
    """Manages multiple Princesses, handles 2+ SOL operations"""
    
    def __init__(self, ant_id: str, parent_id: Optional[str] = None, initial_capital: float = 2.0):
        super().__init__(ant_id, AntRole.QUEEN, parent_id)
        self.capital.current_balance = initial_capital
        self.capital.available_capital = initial_capital
        
        # Queen-specific attributes
        self.princesses: Dict[str, AntPrincess] = {}
        self.retired_princesses: List[str] = []
        self.ai_components_initialized = False
        
    async def initialize_ai_components(self) -> bool:
        """Initialize AI components for the Queen's operations"""
        try:
            self.grok_engine = GrokEngine()
            self.local_llm = LocalLLM()
            
            await self.grok_engine.initialize()
            await self.local_llm.initialize()
            
            self.ai_components_initialized = True
            logger.info(f"Queen {self.ant_id} AI components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Queen {self.ant_id} AI initialization error: {str(e)}")
            return False
    
    async def create_princess(self, initial_capital: float = 0.5) -> Optional[str]:
        """Create a new Princess with allocated capital"""
        try:
            if not self.capital.allocate_capital(initial_capital):
                logger.warning(f"Queen {self.ant_id} insufficient capital to create Princess")
                return None
            
            princess_id = f"princess_{int(time.time())}_{len(self.princesses)}"
            princess = AntPrincess(princess_id, self.ant_id, initial_capital)
            
            if self.ai_components_initialized:
                await princess.initialize(self.grok_engine, self.local_llm)
            
            self.princesses[princess_id] = princess
            self.children.append(princess_id)
            
            logger.info(f"Queen {self.ant_id} created Princess {princess_id} with {initial_capital} SOL")
            return princess_id
            
        except Exception as e:
            logger.error(f"Queen {self.ant_id} Princess creation error: {str(e)}")
            return None
    
    async def manage_princesses(self, market_opportunities: List[Dict]) -> List[Dict]:
        """Manage Princess lifecycle and distribute opportunities"""
        try:
            results = []
            
            # Check for retiring Princesses
            retiring_princesses = []
            for princess_id, princess in self.princesses.items():
                if princess.should_retire():
                    retiring_princesses.append(princess_id)
            
            # Process retiring Princesses
            for princess_id in retiring_princesses:
                await self.retire_princess(princess_id)
            
            # Create new Princesses if we have capital and opportunities
            while (self.should_split() and 
                   len(self.princesses) < self.config["max_children"] and 
                   len(market_opportunities) > len(self.princesses)):
                await self.create_princess()
            
            # Distribute opportunities to active Princesses
            active_princesses = list(self.princesses.values())
            for i, opportunity in enumerate(market_opportunities):
                if i < len(active_princesses):
                    princess = active_princesses[i]
                    decision = await princess.analyze_opportunity(
                        opportunity["token_address"], 
                        opportunity
                    )
                    if decision and decision["action"] != "hold":
                        results.append({
                            "princess_id": princess.ant_id,
                            "decision": decision
                        })
            
            self.update_activity()
            return results
            
        except Exception as e:
            logger.error(f"Queen {self.ant_id} Princess management error: {str(e)}")
            return []
    
    async def retire_princess(self, princess_id: str) -> bool:
        """Retire a Princess and reclaim capital"""
        try:
            if princess_id not in self.princesses:
                return False
            
            princess = self.princesses[princess_id]
            
            # Reclaim capital
            reclaimed_capital = princess.capital.current_balance
            self.capital.update_balance(self.capital.current_balance + reclaimed_capital)
            
            # Archive Princess data
            retirement_record = {
                "princess_id": princess_id,
                "retirement_time": time.time(),
                "final_balance": reclaimed_capital,
                "total_trades": princess.performance.total_trades,
                "total_profit": princess.performance.total_profit,
                "win_rate": princess.performance.win_rate,
                "trade_history": princess.trade_history
            }
            
            # Move to retired list
            self.retired_princesses.append(retirement_record)
            del self.princesses[princess_id]
            self.children.remove(princess_id)
            
            logger.info(f"Queen {self.ant_id} retired Princess {princess_id}: {reclaimed_capital:.4f} SOL reclaimed, {princess.performance.total_trades} trades completed")
            return True
            
        except Exception as e:
            logger.error(f"Queen {self.ant_id} Princess retirement error: {str(e)}")
            return False
    
    def get_queen_status(self) -> Dict[str, Any]:
        """Get comprehensive Queen status including all Princesses"""
        status = self.get_status_summary()
        
        # Add Princess details
        princess_status = []
        for princess_id, princess in self.princesses.items():
            princess_status.append(princess.get_status_summary())
        
        status.update({
            "active_princesses": len(self.princesses),
            "retired_princesses": len(self.retired_princesses),
            "princess_details": princess_status,
            "total_princess_capital": sum(p.capital.current_balance for p in self.princesses.values()),
            "total_princess_trades": sum(p.performance.total_trades for p in self.princesses.values()),
            "average_princess_performance": {
                "avg_win_rate": sum(p.performance.win_rate for p in self.princesses.values()) / len(self.princesses) if self.princesses else 0,
                "avg_profit": sum(p.performance.total_profit for p in self.princesses.values()) / len(self.princesses) if self.princesses else 0
            }
        })
        
        return status

class FoundingAntQueen(BaseAnt):
    """Top-level coordinator managing multiple Queens"""
    
    def __init__(self, ant_id: str = "founding_queen_0", initial_capital: float = 20.0):
        super().__init__(ant_id, AntRole.FOUNDING_QUEEN)
        self.capital.current_balance = initial_capital
        self.capital.available_capital = initial_capital
        
        # Founding Queen specific attributes
        self.queens: Dict[str, AntQueen] = {}
        self.system_metrics = {
            "total_ants": 0,
            "total_capital": initial_capital,
            "total_trades": 0,
            "system_profit": 0.0,
            "system_start_time": time.time()
        }
        
    async def initialize(self) -> bool:
        """Initialize the Founding Queen and create initial Queen"""
        try:
            # Create initial Queen
            await self.create_queen()
            
            logger.info(f"Founding Queen {self.ant_id} initialized with {self.capital.current_balance} SOL")
            return True
            
        except Exception as e:
            logger.error(f"Founding Queen initialization error: {str(e)}")
            return False
    
    async def create_queen(self, initial_capital: float = 2.0) -> Optional[str]:
        """Create a new Queen with allocated capital"""
        try:
            if not self.capital.allocate_capital(initial_capital):
                logger.warning(f"Founding Queen insufficient capital to create Queen")
                return None
            
            queen_id = f"queen_{int(time.time())}_{len(self.queens)}"
            queen = AntQueen(queen_id, self.ant_id, initial_capital)
            
            # Initialize Queen's AI components
            await queen.initialize_ai_components()
            
            self.queens[queen_id] = queen
            self.children.append(queen_id)
            
            logger.info(f"Founding Queen created Queen {queen_id} with {initial_capital} SOL")
            return queen_id
            
        except Exception as e:
            logger.error(f"Founding Queen Queen creation error: {str(e)}")
            return None
    
    async def coordinate_system(self, market_opportunities: List[Dict]) -> Dict[str, Any]:
        """Coordinate the entire Ant system"""
        try:
            results = {
                "decisions": [],
                "system_actions": [],
                "metrics": {}
            }
            
            # Distribute opportunities across Queens
            opportunities_per_queen = len(market_opportunities) // max(1, len(self.queens))
            
            for i, (queen_id, queen) in enumerate(self.queens.items()):
                start_idx = i * opportunities_per_queen
                end_idx = start_idx + opportunities_per_queen if i < len(self.queens) - 1 else len(market_opportunities)
                queen_opportunities = market_opportunities[start_idx:end_idx]
                
                queen_results = await queen.manage_princesses(queen_opportunities)
                results["decisions"].extend(queen_results)
            
            # Check for system-level actions
            if self.should_split():
                new_queen_id = await self.create_queen()
                if new_queen_id:
                    results["system_actions"].append(f"Created new Queen: {new_queen_id}")
            
            # Update system metrics
            await self.update_system_metrics()
            results["metrics"] = self.system_metrics
            
            self.update_activity()
            return results
            
        except Exception as e:
            logger.error(f"Founding Queen coordination error: {str(e)}")
            return {"decisions": [], "system_actions": [], "metrics": {}}
    
    async def update_system_metrics(self):
        """Update system-wide metrics"""
        try:
            total_capital = self.capital.current_balance
            total_trades = 0
            total_profit = 0.0
            total_ants = 1  # Founding Queen
            
            # Aggregate metrics from all Queens
            for queen in self.queens.values():
                total_capital += queen.capital.current_balance
                total_trades += queen.performance.total_trades
                total_profit += queen.performance.total_profit
                total_ants += 1  # Queen
                
                # Add Princess metrics
                for princess in queen.princesses.values():
                    total_capital += princess.capital.current_balance
                    total_trades += princess.performance.total_trades
                    total_profit += princess.performance.total_profit
                    total_ants += 1  # Princess
            
            self.system_metrics.update({
                "total_ants": total_ants,
                "total_capital": total_capital,
                "total_trades": total_trades,
                "system_profit": total_profit,
                "runtime_hours": (time.time() - self.system_metrics["system_start_time"]) / 3600
            })
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {str(e)}")
    
    async def process_operations(self):
        """Process all Founding Queen operations"""
        try:
            # Update system metrics
            await self.update_system_metrics()
            
            # Check if we need to create more Queens
            if self.should_split() and len(self.queens) < self.config["max_children"]:
                await self.create_queen()
            
            # Check for underperforming Queens that should be merged
            queens_to_merge = []
            for queen_id, queen in self.queens.items():
                if queen.should_merge():
                    queens_to_merge.append(queen_id)
            
            # Process Queen merging (simplified - just reclaim capital)
            for queen_id in queens_to_merge:
                await self.merge_queen(queen_id)
            
            self.update_activity()
            
        except Exception as e:
            logger.error(f"Error processing Founding Queen operations: {str(e)}")
    
    async def merge_queen(self, queen_id: str) -> bool:
        """Merge an underperforming Queen and redistribute assets"""
        try:
            if queen_id not in self.queens:
                return False
            
            queen = self.queens[queen_id]
            
            # Retire all Princesses first
            princess_ids = list(queen.princesses.keys())
            for princess_id in princess_ids:
                await queen.retire_princess(princess_id)
            
            # Reclaim Queen capital
            reclaimed_capital = queen.capital.current_balance
            self.capital.update_balance(self.capital.current_balance + reclaimed_capital)
            
            # Remove Queen
            del self.queens[queen_id]
            self.children.remove(queen_id)
            
            logger.info(f"Founding Queen merged Queen {queen_id}: {reclaimed_capital:.4f} SOL reclaimed")
            return True
            
        except Exception as e:
            logger.error(f"Error merging Queen {queen_id}: {str(e)}")
            return False
    
    async def apply_ai_insights(self, insights: Dict[str, Any]):
        """Apply AI insights to the system"""
        try:
            # Process insights for system-level decisions
            if "system_recommendations" in insights:
                recommendations = insights["system_recommendations"]
                
                # Handle expansion recommendations
                if recommendations.get("expand_queens", False):
                    await self.create_queen()
                
                # Handle capital reallocation
                if "capital_reallocation" in recommendations:
                    await self._reallocate_capital(recommendations["capital_reallocation"])
            
            # Distribute insights to Queens
            for queen in self.queens.values():
                if hasattr(queen, 'apply_ai_insights'):
                    await queen.apply_ai_insights(insights)
            
        except Exception as e:
            logger.error(f"Error applying AI insights: {str(e)}")
    
    async def _reallocate_capital(self, reallocation_plan: Dict[str, float]):
        """Reallocate capital based on AI recommendations"""
        try:
            for queen_id, target_capital in reallocation_plan.items():
                if queen_id in self.queens:
                    queen = self.queens[queen_id]
                    current_capital = queen.capital.current_balance
                    
                    if target_capital > current_capital:
                        # Allocate more capital
                        additional = target_capital - current_capital
                        if self.capital.allocate_capital(additional):
                            queen.capital.update_balance(target_capital)
                    elif target_capital < current_capital:
                        # Reclaim excess capital
                        excess = current_capital - target_capital
                        queen.capital.current_balance = target_capital
                        queen.capital.available_capital = max(0, target_capital - queen.capital.allocated_capital)
                        self.capital.update_balance(self.capital.current_balance + excess)
            
        except Exception as e:
            logger.error(f"Error reallocating capital: {str(e)}")
    
    async def save_state(self):
        """Save system state for persistence"""
        try:
            state = {
                "founding_queen": self.get_status_summary(),
                "queens": {qid: queen.get_queen_status() for qid, queen in self.queens.items()},
                "system_metrics": self.system_metrics,
                "timestamp": time.time()
            }
            
            # Save to file (simplified implementation)
            import json
            with open("ant_system_state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info("System state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = self.get_status_summary()
        
        # Add detailed Queen information
        queen_details = []
        for queen_id, queen in self.queens.items():
            queen_details.append(queen.get_queen_status())
        
        status.update({
            "system_metrics": self.system_metrics,
            "active_queens": len(self.queens),
            "queen_details": queen_details
        })
        
        return status 
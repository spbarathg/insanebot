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
- TITAN SHIELD INTEGRATION: Defense-aware trading agents
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import random
from abc import ABC, abstractmethod
from collections import deque

# Core AI components
from .grok_engine import GrokEngine
from ..local_llm import LocalLLM

# Services
from ...services.wallet_manager import WalletManager

# CRITICAL: Defense system integration
from ..titan_shield_coordinator import TitanShieldCoordinator

# Optional portfolio risk manager
try:
    from ..portfolio_risk_manager import PortfolioRiskManager
except ImportError:
    PortfolioRiskManager = None

logger = logging.getLogger(__name__)

# System constants - no more magic numbers
class SystemConstants:
    """Centralized system constants for maintainability - MICRO-CAPITAL VERSION"""
    # Capital thresholds (SOL) - Scaled for small capital amounts
    FOUNDING_QUEEN_SPLIT_THRESHOLD = 0.1    # Reduced from 20.0 → 0.1 (200x smaller)
    QUEEN_SPLIT_THRESHOLD = 0.05             # Reduced from 2.0 → 0.05 (40x smaller)  
    FOUNDING_QUEEN_MERGE_THRESHOLD = 0.001   # Reduced from 0.5 → 0.001
    QUEEN_MERGE_THRESHOLD = 0.001            # Reduced from 0.1 → 0.001
    PRINCESS_INITIAL_CAPITAL = 0.01          # Reduced from 0.5 → 0.01 (50x smaller)
    
    # Trading limits
    PRINCESS_MIN_TRADES = 5
    PRINCESS_MAX_TRADES = 10
    MAX_POSITION_PERCENT = 80  # 80% of available capital
    FALLBACK_POSITION_SIZE = 0.001  # 0.001 SOL fallback (smaller for micro-capital)
    
    # Performance thresholds
    MIN_WIN_RATE_FOR_SPLIT = 50.0
    POOR_PERFORMANCE_WIN_RATE = 30.0
    HIGH_RISK_THRESHOLD = 0.8
    PROFIT_THRESHOLD_FOR_RETIREMENT = 0.0001  # Smaller profit threshold
    
    # System limits
    FOUNDING_QUEEN_MAX_CHILDREN = 10
    QUEEN_MAX_CHILDREN = 50
    AI_ANALYSIS_TIMEOUT = 30.0  # seconds

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
        if new_balance < 0:
            logger.warning(f"Negative balance detected: {new_balance}, setting to 0")
            new_balance = 0.0
            
        self.profit_loss += (new_balance - self.current_balance)
        self.current_balance = new_balance
        self.available_capital = max(0, new_balance - self.allocated_capital)
        self.last_updated = time.time()
    
    def allocate_capital(self, amount: float) -> bool:
        """Allocate capital for trading operations with validation"""
        if amount <= 0:
            logger.error(f"Invalid allocation amount: {amount}")
            return False
            
        if self.available_capital >= amount:
            self.allocated_capital += amount
            self.available_capital -= amount
            return True
        logger.warning(f"Insufficient capital for allocation: {amount} > {self.available_capital}")
        return False
    
    def release_capital(self, amount: float):
        """Release allocated capital back to available pool with validation"""
        if amount <= 0:
            logger.error(f"Invalid release amount: {amount}")
            return
            
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
        # Input validation
        if not ant_id:
            raise ValueError("ant_id cannot be empty")
        if not isinstance(role, AntRole):
            raise ValueError(f"role must be AntRole enum, got {type(role)}")
        
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
        
        logger.debug(f"BaseAnt {ant_id} ({role.value}) initialized")
        
    def _get_role_config(self) -> Dict[str, Any]:
        """Get configuration based on ant role using SystemConstants"""
        configs = {
            AntRole.FOUNDING_QUEEN: {
                "max_children": SystemConstants.FOUNDING_QUEEN_MAX_CHILDREN,
                "split_threshold": SystemConstants.FOUNDING_QUEEN_SPLIT_THRESHOLD,
                "merge_threshold": SystemConstants.FOUNDING_QUEEN_MERGE_THRESHOLD,
                "max_trades": float('inf'),
                "retirement_trades": None
            },
            AntRole.QUEEN: {
                "max_children": SystemConstants.QUEEN_MAX_CHILDREN,
                "split_threshold": SystemConstants.QUEEN_SPLIT_THRESHOLD,
                "merge_threshold": SystemConstants.QUEEN_MERGE_THRESHOLD,
                "max_trades": float('inf'),
                "retirement_trades": None
            },
            AntRole.PRINCESS: {
                "max_children": 0,
                "split_threshold": None,
                "merge_threshold": None,
                "max_trades": SystemConstants.PRINCESS_MAX_TRADES,
                "retirement_trades": (SystemConstants.PRINCESS_MIN_TRADES, SystemConstants.PRINCESS_MAX_TRADES)
            }
        }
        return configs[self.role]
    
    def should_split(self) -> bool:
        """Determine if this ant should split based on capital and performance with enhanced logic"""
        try:
            if not self.config["split_threshold"]:
                return False
            
            # Check capital threshold
            capital_ready = self.capital.available_capital >= self.config["split_threshold"]
            
            # Check performance threshold (must be profitable and performing well)
            performance_ready = (
                self.performance.total_profit > 0 and 
                self.performance.win_rate > SystemConstants.MIN_WIN_RATE_FOR_SPLIT and
                self.performance.total_trades >= 3  # Minimum track record
            )
            
            # Check children limit
            children_limit_ok = len(self.children) < self.config["max_children"]
            
            # Additional checks for system stability
            recent_activity = (time.time() - self.last_activity) < 3600  # Active within last hour
            
            should_split = capital_ready and performance_ready and children_limit_ok and recent_activity
            
            if should_split:
                logger.debug(f"{self.role.value} {self.ant_id} meets split criteria: "
                           f"capital={capital_ready}, performance={performance_ready}, "
                           f"children_limit={children_limit_ok}, recent_activity={recent_activity}")
            
            return should_split
            
        except Exception as e:
            logger.error(f"Error in should_split for {self.ant_id}: {str(e)}")
            return False
    
    def should_merge(self) -> bool:
        """Determine if this ant should be merged due to poor performance with enhanced logic"""
        try:
            if not self.config["merge_threshold"]:
                return False
            
            # Check if capital is below merge threshold
            capital_low = self.capital.current_balance < self.config["merge_threshold"]
            
            # Check if performance is poor (more comprehensive analysis)
            performance_poor = False
            if self.performance.total_trades >= 5:
                performance_poor = (
                    self.performance.win_rate < SystemConstants.POOR_PERFORMANCE_WIN_RATE or 
                    self.performance.total_profit < -self.config["merge_threshold"] or
                    self.performance.risk_score > SystemConstants.HIGH_RISK_THRESHOLD
                )
            
            # Check for prolonged inactivity
            inactive_too_long = (time.time() - self.last_activity) > 86400  # 24 hours
            
            should_merge = capital_low or performance_poor or inactive_too_long
            
            if should_merge:
                logger.debug(f"{self.role.value} {self.ant_id} meets merge criteria: "
                           f"capital_low={capital_low}, performance_poor={performance_poor}, "
                           f"inactive_too_long={inactive_too_long}")
            
            return should_merge
            
        except Exception as e:
            logger.error(f"Error in should_merge for {self.ant_id}: {str(e)}")
            return False
    
    def should_retire(self) -> bool:
        """Determine if this ant should retire (mainly for Princesses) with enhanced logic"""
        try:
            if not self.config["retirement_trades"]:
                return False
            
            min_trades, max_trades = self.config["retirement_trades"]
            
            # Retire if reached max trades
            if self.performance.total_trades >= max_trades:
                logger.debug(f"Princess {self.ant_id} retiring: reached max trades ({max_trades})")
                return True
            
            # Retire if reached min trades and meets retirement criteria
            if self.performance.total_trades >= min_trades:
                # Enhanced retirement criteria
                made_profit = self.performance.total_profit > SystemConstants.PROFIT_THRESHOLD_FOR_RETIREMENT
                very_poor_performance = self.performance.win_rate < 20.0
                too_risky = self.performance.risk_score > SystemConstants.HIGH_RISK_THRESHOLD
                good_performance = (
                    self.performance.win_rate > 60.0 and 
                    self.performance.total_profit > SystemConstants.PROFIT_THRESHOLD_FOR_RETIREMENT * 2
                )
                
                should_retire = made_profit or very_poor_performance or too_risky or good_performance
                
                if should_retire:
                    retirement_reason = []
                    if made_profit: retirement_reason.append("made_profit")
                    if very_poor_performance: retirement_reason.append("poor_performance")
                    if too_risky: retirement_reason.append("too_risky")
                    if good_performance: retirement_reason.append("good_performance")
                    
                    logger.debug(f"Princess {self.ant_id} retiring after {self.performance.total_trades} trades: "
                               f"{', '.join(retirement_reason)}")
                
                return should_retire
            
            return False
            
        except Exception as e:
            logger.error(f"Error in should_retire for {self.ant_id}: {str(e)}")
            return False
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary with enhanced metrics"""
        try:
            return {
                "ant_id": self.ant_id,
                "role": self.role.value,
                "status": self.status.value,
                "parent_id": self.parent_id,
                "children_count": len(self.children),
                "created_at": self.created_at,
                "last_activity": self.last_activity,
                "age_hours": (time.time() - self.created_at) / 3600,
                "inactive_hours": (time.time() - self.last_activity) / 3600,
                "capital": {
                    "current_balance": self.capital.current_balance,
                    "allocated_capital": self.capital.allocated_capital,
                    "available_capital": self.capital.available_capital,
                    "profit_loss": self.capital.profit_loss,
                    "utilization_percent": (self.capital.allocated_capital / max(self.capital.current_balance, 0.001)) * 100
                },
                "performance": {
                    "total_trades": self.performance.total_trades,
                    "successful_trades": self.performance.successful_trades,
                    "win_rate": self.performance.win_rate,
                    "total_profit": self.performance.total_profit,
                    "profit_per_trade": self.performance.profit_per_trade,
                    "risk_score": self.performance.risk_score,
                    "best_trade": self.performance.best_trade,
                    "worst_trade": self.performance.worst_trade,
                    "average_trade_time": self.performance.average_trade_time
                },
                "decisions": {
                    "should_split": self.should_split(),
                    "should_merge": self.should_merge(),
                    "should_retire": self.should_retire()
                },
                "config": self.config
            }
        except Exception as e:
            logger.error(f"Error getting status summary for {self.ant_id}: {str(e)}")
            return {
                "ant_id": self.ant_id,
                "role": self.role.value,
                "error": str(e)
            }

class AntPrincess(BaseAnt):
    """Individual trading agent (Worker Ant) with 5-10 trade lifecycle"""
    
    def __init__(self, ant_id: str, parent_id: str, initial_capital: float = SystemConstants.PRINCESS_INITIAL_CAPITAL, 
                 titan_shield: Optional[TitanShieldCoordinator] = None):
        """Initialize Princess with proper defense system integration"""
        super().__init__(ant_id, AntRole.PRINCESS, parent_id)
        
        # Validate inputs
        if initial_capital <= 0:
            raise ValueError(f"Invalid initial capital: {initial_capital}")
        if not parent_id:
            raise ValueError("Princess must have a parent_id")
            
        self.capital.current_balance = initial_capital
        self.capital.available_capital = initial_capital
        
        # Trading components
        self.grok_engine: Optional[GrokEngine] = None
        self.local_llm: Optional[LocalLLM] = None
        self.active_positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        
        # Princess-specific tracking
        self.target_trades = None
        self.specialization = None  # Can specialize in certain token types
        
        # CRITICAL INTEGRATION: Titan Shield properly injected
        self.titan_shield = titan_shield
        if not self.titan_shield:
            logger.warning(f"Princess {ant_id} initialized WITHOUT TitanShieldCoordinator - operating in UNSAFE mode")
        else:
            logger.info(f"🛡️ Princess {ant_id} initialized with TitanShieldCoordinator defense protection")
        
        # Defense-aware trading parameters
        self.max_position_multiplier = 1.0  # Adjusted by defense mode
        self.trading_enabled = True  # Can be disabled by defense mode
        self.defense_mode_overrides = {}  # Track defense system overrides
        
    async def initialize(self, grok_engine: GrokEngine, local_llm: LocalLLM):
        """Initialize the Princess with AI components and validation"""
        try:
            if not grok_engine or not local_llm:
                raise ValueError("Both grok_engine and local_llm are required")
                
            self.grok_engine = grok_engine
            self.local_llm = local_llm
            
            # Determine target trades (5-10 range)
            self.target_trades = random.randint(SystemConstants.PRINCESS_MIN_TRADES, SystemConstants.PRINCESS_MAX_TRADES)
            
            logger.info(f"Princess {self.ant_id} initialized with {self.capital.current_balance} SOL, "
                       f"target: {self.target_trades} trades, "
                       f"defense: {'✅ ACTIVE' if self.titan_shield else '❌ DISABLED'}")
            
        except Exception as e:
            logger.error(f"Princess {self.ant_id} initialization failed: {str(e)}")
            raise ValueError(f"Princess initialization error: {str(e)}")
    
    async def analyze_opportunity(self, token_address: str, market_data: Dict) -> Optional[Dict]:
        """Analyze trading opportunity using AI components with enhanced error handling"""
        try:
            self.update_activity()
            
            # Input validation
            if not token_address or not market_data:
                raise ValueError("Token address and market data are required")
            
            # Check if we should retire
            if self.should_retire():
                logger.info(f"Princess {self.ant_id} should retire after {self.performance.total_trades} trades")
                return None
            
            # Timeout protection for AI analysis
            analysis_start = time.time()
            
            # Get sentiment from Grok with timeout
            try:
                sentiment_task = asyncio.create_task(self.grok_engine.analyze_market(market_data))
                sentiment_analysis = await asyncio.wait_for(
                    sentiment_task, 
                    timeout=SystemConstants.AI_ANALYSIS_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(f"❌ Princess {self.ant_id} - Grok analysis timeout after {SystemConstants.AI_ANALYSIS_TIMEOUT}s")
                raise TimeoutError(f"Grok sentiment analysis timeout - Princess {self.ant_id} cannot operate")
            except Exception as e:
                logger.error(f"❌ Princess {self.ant_id} - Grok sentiment analysis failed: {str(e)}")
                raise Exception(f"Grok sentiment analysis failure - Princess {self.ant_id}: {str(e)}")
            
            if not sentiment_analysis or "error" in sentiment_analysis:
                logger.error(f"❌ CRITICAL: Princess {self.ant_id} - Invalid Grok sentiment response")
                raise Exception(f"Invalid Grok response - Princess {self.ant_id} cannot analyze without sentiment data")
            
            # Get technical analysis from Local LLM with timeout
            try:
                technical_task = asyncio.create_task(self.local_llm.analyze_market(market_data))
                technical_analysis = await asyncio.wait_for(
                    technical_task,
                    timeout=SystemConstants.AI_ANALYSIS_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(f"❌ Princess {self.ant_id} - Local LLM analysis timeout after {SystemConstants.AI_ANALYSIS_TIMEOUT}s")
                raise TimeoutError(f"Local LLM technical analysis timeout - Princess {self.ant_id} cannot operate")
            except Exception as e:
                logger.error(f"❌ Princess {self.ant_id} - Local LLM technical analysis failed: {str(e)}")
                raise Exception(f"Local LLM technical analysis failure - Princess {self.ant_id}: {str(e)}")
            
            if not technical_analysis or "error" in technical_analysis:
                logger.error(f"❌ CRITICAL: Princess {self.ant_id} - Invalid Local LLM technical response")
                raise Exception(f"Invalid Local LLM response - Princess {self.ant_id} cannot analyze without technical data")
            
            # Combine analyses for decision
            decision = self._make_trading_decision(sentiment_analysis, technical_analysis, market_data)
            analysis_time = time.time() - analysis_start
            
            if decision["action"] != "hold":
                logger.info(f"Princess {self.ant_id} found opportunity: {decision['action']} {token_address[:8]}... "
                          f"(confidence: {decision['confidence']:.2f}, analysis time: {analysis_time:.2f}s)")
            
            return decision
            
        except (TimeoutError, ValueError, Exception) as e:
            # Re-raise critical errors to halt operations
            logger.error(f"💥 Princess {self.ant_id} analysis pipeline failure: {str(e)}")
            raise Exception(f"Princess {self.ant_id} AI analysis critical failure: {str(e)}")
    
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
        """Execute trading decision with MANDATORY defense approval checkpoint"""
        try:
            # CRITICAL DEFENSE CHECKPOINT: All trades must pass through Titan Shield
            if self.titan_shield:
                return await self.execute_trade_protected(decision, wallet_manager)
            else:
                logger.critical(f"🚨 Princess {self.ant_id} executing trade WITHOUT defense protection - UNSAFE!")
                return await self._execute_trade_unprotected(decision, wallet_manager)
                
        except Exception as e:
            logger.error(f"💥 Princess {self.ant_id} trade execution critical failure: {str(e)}")
            return {
                "success": False, 
                "message": f"Trade execution failed: {str(e)}", 
                "error_context": "execute_trade_wrapper"
            }

    async def _execute_trade_unprotected(self, decision: Dict, wallet_manager: WalletManager) -> Dict:
        """Execute trade without defense protection - LEGACY FALLBACK ONLY"""
        try:
            if decision["action"] == "hold" or decision["position_size"] <= 0:
                return {"success": False, "message": "No trade to execute", "error_context": "no_trade_needed"}
            
            # Allocate capital for trade
            if not self.capital.allocate_capital(decision["position_size"]):
                return {"success": False, "message": "Insufficient capital", "error_context": "capital_allocation"}
            
            # Simulate trade execution (replace with real trading logic)
            trade_start = time.time()
            
            # For simulation - random profit/loss with improved validation
            success_probability = max(0.1, min(0.9, decision.get("confidence", 0.5)))
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
            
            # Record trade with enhanced metadata
            trade_record = {
                "trade_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "token_address": decision.get("token_address", "unknown"),
                "action": decision["action"],
                "position_size": decision["position_size"],
                "profit": profit,
                "success": success,
                "confidence": decision.get("confidence", 0.0),
                "reasoning": decision.get("reasoning", "No reasoning provided"),
                "defense_approved": False,  # Mark as unprotected
                "execution_mode": "unprotected"
            }
            self.trade_history.append(trade_record)
            
            logger.warning(f"⚠️ Princess {self.ant_id} executed UNPROTECTED trade: {profit:.4f} SOL "
                          f"{'profit' if profit > 0 else 'loss'} - RISK EXPOSURE HIGH")
            
            return {
                "success": True,
                "trade_record": trade_record,
                "new_balance": self.capital.current_balance,
                "trades_completed": self.performance.total_trades,
                "defense_approved": False
            }
            
        except Exception as e:
            logger.error(f"Princess {self.ant_id} unprotected trade execution error: {str(e)}")
            # Release allocated capital on failure
            if "position_size" in decision:
                self.capital.release_capital(decision["position_size"])
            return {"success": False, "message": str(e), "error_context": "unprotected_execution"}

    async def execute_trade_protected(self, decision: Dict, wallet_manager: WalletManager) -> Dict:
        """Execute trading decision with full Titan Shield protection"""
        try:
            # CRITICAL: Check if trading is enabled by defense mode
            if not self.trading_enabled:
                return {
                    "success": False, 
                    "message": "Trading disabled by defense mode",
                    "rejection_reason": "Defense mode restriction"
                }
            
            if decision["action"] == "hold" or decision["position_size"] <= 0:
                return {"success": False, "message": "No trade to execute"}
            
            # CRITICAL: Apply defense mode position size multiplier
            adjusted_position_size = decision["position_size"] * self.max_position_multiplier
            
            # CRITICAL: Full spectrum analysis through Titan Shield
            if self.titan_shield:
                token_address = decision.get("token_address")
                if not token_address:
                    return {
                        "success": False, 
                        "message": "No token address provided",
                        "rejection_reason": "Missing token address"
                    }
                
                # Prepare market data for analysis
                market_data = {
                    "token_address": token_address,
                    "current_price": 1.0,  # Simplified - would get real price
                    "volume": 1000.0,  # Simplified - would get real volume
                    "timestamp": time.time()
                }
                
                # LAYER 1-7 DEFENSE ANALYSIS
                logger.info(f"🛡️ Princess {self.ant_id}: Running full spectrum analysis on {token_address[:8]}...")
                
                approval_status, rejection_reason, analysis_results = await self.titan_shield.full_spectrum_analysis(
                    token_address=token_address,
                    market_data=market_data,
                    social_data=None,  # Would be provided in production
                    transaction_data=None,  # Would be provided in production
                    holder_data=None  # Would be provided in production
                )
                
                if not approval_status:
                    logger.warning(f"🚫 Princess {self.ant_id}: Trade REJECTED - {rejection_reason}")
                    return {
                        "success": False,
                        "message": f"Defense systems rejected trade: {rejection_reason}",
                        "rejection_reason": rejection_reason,
                        "defense_approved": False,
                        "analysis_results": analysis_results
                    }
                
                # Get adaptive parameters for execution
                adaptive_params = analysis_results.get('adaptive_params')
                if adaptive_params:
                    # Use adaptive position sizing
                    max_defense_position = adaptive_params.max_position_size_sol
                    adjusted_position_size = min(adjusted_position_size, max_defense_position)
                
                logger.info(f"✅ Princess {self.ant_id}: Trade APPROVED by all defense layers")
            
            # Allocate capital for trade
            if not self.capital.allocate_capital(adjusted_position_size):
                return {"success": False, "message": "Insufficient capital"}
            
            # Prepare transaction data for protected execution
            transaction_data = {
                "instruction": decision["action"],
                "token_address": decision["token_address"],
                "amount": adjusted_position_size,
                "max_slippage": adaptive_params.max_slippage_percent / 100 if adaptive_params else 0.15,
                "priority_fee": adaptive_params.priority_fee_lamports if adaptive_params else 10000
            }
            
            # CRITICAL: Execute transaction through Titan Shield warfare system
            if self.titan_shield:
                execution_success = await self.titan_shield.execute_protected_transaction(
                    transaction_data, decision["action"], decision["token_address"], adjusted_position_size
                )
                
                if not execution_success:
                    # Release allocated capital on failure
                    self.capital.release_capital(adjusted_position_size)
                    return {
                        "success": False,
                        "message": "Protected transaction execution failed",
                        "rejection_reason": "Transaction warfare system failure"
                    }
            
            # Simulate successful trade execution (replace with real results)
            trade_start = time.time()
            
            # For simulation - profit/loss based on confidence with defense adjustments
            success_probability = decision["confidence"] * 0.9  # Slightly reduce due to defense overhead
            success = random.random() < success_probability
            
            if success:
                profit = adjusted_position_size * random.uniform(0.02, 0.15)  # 2-15% profit
            else:
                profit = -adjusted_position_size * random.uniform(0.01, 0.08)  # 1-8% loss
            
            trade_time = time.time() - trade_start
            
            # Update capital and performance
            self.capital.release_capital(adjusted_position_size)
            self.capital.update_balance(self.capital.current_balance + profit)
            self.performance.update_trade_result(profit, trade_time, success)
            
            # Record trade with defense metadata
            trade_record = {
                "trade_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "token_address": decision["token_address"],
                "action": decision["action"],
                "original_position_size": decision["position_size"],
                "actual_position_size": adjusted_position_size,
                "position_multiplier": self.max_position_multiplier,
                "profit": profit,
                "success": success,
                "confidence": decision["confidence"],
                "reasoning": decision["reasoning"],
                "defense_approved": True,
                "defense_analysis": analysis_results if self.titan_shield else None
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"✅ Princess {self.ant_id} executed PROTECTED trade: {profit:.4f} SOL {'profit' if profit > 0 else 'loss'}")
            
            return {
                "success": True,
                "trade_record": trade_record,
                "new_balance": self.capital.current_balance,
                "trades_completed": self.performance.total_trades,
                "defense_approved": True
            }
            
        except Exception as e:
            logger.error(f"💥 Princess {self.ant_id} protected trade execution error: {str(e)}")
            return {"success": False, "message": str(e), "rejection_reason": f"Execution error: {str(e)}"}

class AntQueen(BaseAnt):
    """Manages multiple Princesses, handles 2+ SOL operations"""
    
    def __init__(self, ant_id: str, parent_id: str, initial_capital: float = SystemConstants.QUEEN_SPLIT_THRESHOLD, 
                 titan_shield: Optional[TitanShieldCoordinator] = None):
        """Initialize Queen with proper defense system integration"""
        super().__init__(ant_id, AntRole.QUEEN, parent_id)
        
        # Validate inputs
        if initial_capital < SystemConstants.QUEEN_SPLIT_THRESHOLD:
            logger.warning(f"Queen {ant_id} initialized with capital {initial_capital} below recommended threshold {SystemConstants.QUEEN_SPLIT_THRESHOLD}")
        
        # Queen-specific management
        self.princesses: Dict[str, AntPrincess] = {}
        self.retired_princesses: List[Dict] = []
        self.ai_components_initialized = False
        
        # AI components (shared with Princesses)
        self.grok_engine: Optional[GrokEngine] = None
        self.local_llm: Optional[LocalLLM] = None
        
        # CRITICAL INTEGRATION: Titan Shield reference for propagation
        self.titan_shield = titan_shield
        if not self.titan_shield:
            logger.warning(f"Queen {ant_id} initialized WITHOUT TitanShieldCoordinator - Princesses will operate in UNSAFE mode")
        else:
            logger.info(f"🛡️ Queen {ant_id} initialized with TitanShieldCoordinator - defense will propagate to Princesses")
        
        # Initialize capital
        self.capital.update_balance(initial_capital)
        
        logger.info(f"Queen {ant_id} created with {initial_capital} SOL capital, "
                   f"defense: {'✅ ACTIVE' if self.titan_shield else '❌ DISABLED'}")
        
    async def initialize_ai_components(self) -> bool:
        """Initialize AI components for the Queen's operations with validation"""
        try:
            if self.ai_components_initialized:
                logger.warning(f"Queen {self.ant_id} AI components already initialized")
                return True
                
            self.grok_engine = GrokEngine()
            self.local_llm = LocalLLM()
            
            # Initialize with timeout protection
            grok_task = asyncio.create_task(self.grok_engine.initialize())
            llm_task = asyncio.create_task(self.local_llm.initialize())
            
            try:
                await asyncio.wait_for(grok_task, timeout=SystemConstants.AI_ANALYSIS_TIMEOUT)
                await asyncio.wait_for(llm_task, timeout=SystemConstants.AI_ANALYSIS_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error(f"Queen {self.ant_id} AI initialization timeout after {SystemConstants.AI_ANALYSIS_TIMEOUT}s")
                return False
            
            self.ai_components_initialized = True
            logger.info(f"Queen {self.ant_id} AI components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Queen {self.ant_id} AI initialization error: {str(e)}")
            return False
    
    async def create_princess(self, initial_capital: float = SystemConstants.PRINCESS_INITIAL_CAPITAL) -> Optional[str]:
        """Create a new Princess with allocated capital and proper defense integration"""
        try:
            # Validate capital allocation
            if initial_capital <= 0:
                logger.error(f"Queen {self.ant_id} invalid Princess capital: {initial_capital}")
                return None
                
            if not self.capital.allocate_capital(initial_capital):
                logger.warning(f"Queen {self.ant_id} insufficient capital to create Princess "
                             f"(requested: {initial_capital}, available: {self.capital.available_capital})")
                return None
            
            # Check Princess limit
            if len(self.princesses) >= SystemConstants.QUEEN_MAX_CHILDREN:
                logger.warning(f"Queen {self.ant_id} Princess limit reached: {len(self.princesses)}")
                self.capital.release_capital(initial_capital)  # Release allocated capital
                return None
            
            princess_id = f"princess_{int(time.time())}_{len(self.princesses)}"
            
            # CRITICAL: Pass Titan Shield to new Princess
            princess = AntPrincess(
                ant_id=princess_id, 
                parent_id=self.ant_id, 
                initial_capital=initial_capital,
                titan_shield=self.titan_shield  # DEFENSE INTEGRATION
            )
            
            if self.titan_shield:
                logger.info(f"🛡️ Titan Shield propagated to new Princess {princess_id}")
            else:
                logger.critical(f"🚨 Princess {princess_id} created WITHOUT defense protection!")

            # Initialize Princess with AI components if available
            if self.ai_components_initialized and self.grok_engine and self.local_llm:
                try:
                    await princess.initialize(self.grok_engine, self.local_llm)
                except Exception as e:
                    logger.error(f"Princess {princess_id} initialization failed: {str(e)}")
                    # Don't fail Princess creation for AI initialization errors
                    logger.warning(f"Princess {princess_id} will operate with limited AI capabilities")

            self.princesses[princess_id] = princess
            self.children.append(princess_id)

            logger.info(f"Queen {self.ant_id} created Princess {princess_id} with {initial_capital} SOL "
                       f"(defense: {'✅' if self.titan_shield else '❌'})")
            return princess_id
            
        except Exception as e:
            logger.error(f"Queen {self.ant_id} Princess creation critical error: {str(e)}")
            # Ensure capital is released on failure
            if initial_capital > 0:
                self.capital.release_capital(initial_capital)
            return None
    
    async def manage_princesses(self, market_opportunities: List[Dict]) -> List[Dict]:
        """Manage Princess lifecycle and distribute opportunities with error handling"""
        try:
            results = []
            
            # Input validation
            if not market_opportunities:
                logger.debug(f"Queen {self.ant_id} received no market opportunities")
                return results
            
            # Check for retiring Princesses
            retiring_princesses = []
            for princess_id, princess in self.princesses.items():
                try:
                    if princess.should_retire():
                        retiring_princesses.append(princess_id)
                except Exception as e:
                    logger.error(f"Error checking retirement status for Princess {princess_id}: {str(e)}")
                    # Continue with other Princesses
            
            # Process retiring Princesses
            for princess_id in retiring_princesses:
                try:
                    await self.retire_princess(princess_id)
                except Exception as e:
                    logger.error(f"Error retiring Princess {princess_id}: {str(e)}")
                    # Continue with other operations
            
            # Create new Princesses if we have capital and opportunities
            while (self.should_split() and 
                   len(self.princesses) < SystemConstants.QUEEN_MAX_CHILDREN and 
                   len(market_opportunities) > len(self.princesses)):
                try:
                    new_princess_id = await self.create_princess()
                    if not new_princess_id:
                        break  # Stop trying if creation fails
                except Exception as e:
                    logger.error(f"Error creating new Princess: {str(e)}")
                    break
            
            # Distribute opportunities to active Princesses with error isolation
            active_princesses = list(self.princesses.values())
            for i, opportunity in enumerate(market_opportunities):
                if i < len(active_princesses):
                    princess = active_princesses[i]
                    try:
                        # Add timeout protection for Princess analysis
                        analysis_task = asyncio.create_task(
                            princess.analyze_opportunity(
                                opportunity.get("token_address", ""), 
                                opportunity
                            )
                        )
                        decision = await asyncio.wait_for(
                            analysis_task, 
                            timeout=SystemConstants.AI_ANALYSIS_TIMEOUT
                        )
                        
                        if decision and decision.get("action") != "hold":
                            results.append({
                                "princess_id": princess.ant_id,
                                "decision": decision,
                                "opportunity_index": i
                            })
                    except asyncio.TimeoutError:
                        logger.warning(f"Princess {princess.ant_id} analysis timeout for opportunity {i}")
                        continue
                    except Exception as e:
                        logger.error(f"Princess {princess.ant_id} opportunity analysis error: {str(e)}")
                        continue
            
            self.update_activity()
            logger.debug(f"Queen {self.ant_id} processed {len(market_opportunities)} opportunities, "
                        f"generated {len(results)} decisions")
            return results
            
        except Exception as e:
            logger.error(f"Queen {self.ant_id} Princess management critical error: {str(e)}")
            return []
    
    async def retire_princess(self, princess_id: str) -> bool:
        """Retire a Princess and reclaim capital with enhanced validation"""
        try:
            if princess_id not in self.princesses:
                logger.warning(f"Queen {self.ant_id} cannot retire non-existent Princess {princess_id}")
                return False
            
            princess = self.princesses[princess_id]
            
            # Validate Princess state before retirement
            if princess.capital.allocated_capital > 0:
                logger.warning(f"Princess {princess_id} has allocated capital {princess.capital.allocated_capital}, "
                             f"forcing release")
                princess.capital.release_capital(princess.capital.allocated_capital)
            
            # Reclaim capital
            reclaimed_capital = princess.capital.current_balance
            if reclaimed_capital < 0:
                logger.warning(f"Princess {princess_id} has negative balance {reclaimed_capital}, "
                             f"setting to 0 for retirement")
                reclaimed_capital = 0.0
                
            self.capital.update_balance(self.capital.current_balance + reclaimed_capital)
            
            # Archive Princess data with enhanced metrics
            retirement_record = {
                "princess_id": princess_id,
                "retirement_time": time.time(),
                "final_balance": reclaimed_capital,
                "initial_capital": SystemConstants.PRINCESS_INITIAL_CAPITAL,  # For ROI calculation
                "total_trades": princess.performance.total_trades,
                "successful_trades": princess.performance.successful_trades,
                "total_profit": princess.performance.total_profit,
                "win_rate": princess.performance.win_rate,
                "best_trade": princess.performance.best_trade,
                "worst_trade": princess.performance.worst_trade,
                "average_trade_time": princess.performance.average_trade_time,
                "roi_percent": ((reclaimed_capital - SystemConstants.PRINCESS_INITIAL_CAPITAL) / SystemConstants.PRINCESS_INITIAL_CAPITAL) * 100,
                "trade_history": princess.trade_history,
                "had_defense_protection": princess.titan_shield is not None
            }
            
            # Move to retired list
            self.retired_princesses.append(retirement_record)
            del self.princesses[princess_id]
            self.children.remove(princess_id)
            
            logger.info(f"Queen {self.ant_id} retired Princess {princess_id}: "
                       f"{reclaimed_capital:.4f} SOL reclaimed, "
                       f"{princess.performance.total_trades} trades completed, "
                       f"ROI: {retirement_record['roi_percent']:.1f}%")
            return True
            
        except Exception as e:
            logger.error(f"Queen {self.ant_id} Princess retirement critical error: {str(e)}")
            return False

class FoundingAntQueen(BaseAnt):
    """Top-level coordinator managing multiple Queens"""
    
    def __init__(self, ant_id: str = "founding_queen_0", initial_capital: float = SystemConstants.FOUNDING_QUEEN_SPLIT_THRESHOLD,
                 titan_shield: Optional[TitanShieldCoordinator] = None):
        """Initialize Founding Queen with proper defense system integration"""
        super().__init__(ant_id, AntRole.FOUNDING_QUEEN)
        
        # Validate inputs
        if initial_capital < SystemConstants.FOUNDING_QUEEN_SPLIT_THRESHOLD:
            logger.warning(f"Founding Queen {ant_id} initialized with capital {initial_capital} "
                         f"below recommended threshold {SystemConstants.FOUNDING_QUEEN_SPLIT_THRESHOLD}")
        
        self.capital.current_balance = initial_capital
        self.capital.available_capital = initial_capital
        
        # Founding Queen specific attributes
        self.queens: Dict[str, AntQueen] = {}
        self.system_metrics = {
            "total_ants": 0,
            "total_capital": initial_capital,
            "total_trades": 0,
            "system_profit": 0.0,
            "system_start_time": time.time(),
            "defense_activations": 0,
            "defense_rejections": 0,
            "system_uptime_hours": 0.0
        }
        
        # CRITICAL INTEGRATION: Titan Shield reference for system-wide defense
        self.titan_shield = titan_shield
        if not self.titan_shield:
            logger.critical(f"🚨 Founding Queen {ant_id} initialized WITHOUT TitanShieldCoordinator!")
            logger.critical("🚨 ENTIRE SYSTEM will operate in UNSAFE mode without defense protection!")
        else:
            logger.info(f"🛡️ Founding Queen {ant_id} initialized with TitanShieldCoordinator")
            logger.info(f"🛡️ Defense protection will cascade to all Queens and Princesses")
        
        logger.info(f"Founding Queen {ant_id} created with {initial_capital} SOL capital, "
                   f"defense: {'✅ ACTIVE' if self.titan_shield else '❌ SYSTEM-WIDE DISABLED'}")
        
    async def initialize(self) -> bool:
        """Initialize the Founding Queen and create initial Queen with validation and micro-capital support"""
        try:
            logger.info(f"🐜 Initializing Founding Queen {self.ant_id}...")
            
            # Validate system state before initialization
            if self.capital.current_balance <= 0:
                raise ValueError(f"Cannot initialize with zero or negative capital: {self.capital.current_balance}")
            
            # Micro-capital mode: If we have very little capital, start with just Princesses
            if self.capital.current_balance < SystemConstants.QUEEN_SPLIT_THRESHOLD:
                logger.info(f"💰 Micro-capital mode activated: {self.capital.current_balance} SOL < {SystemConstants.QUEEN_SPLIT_THRESHOLD} SOL")
                logger.info("🐜 Starting with direct Princess management (no intermediate Queens)")
                # Create direct Princesses under Founding Queen for micro-capital trading
                await self._initialize_micro_capital_mode()
            else:
                # Standard mode: Create initial Queen with available capital
                queen_capital = min(self.capital.available_capital * 0.8, SystemConstants.QUEEN_SPLIT_THRESHOLD)
                initial_queen_id = await self.create_queen(queen_capital)
                if not initial_queen_id:
                    logger.warning("⚠️ Failed to create Queen, falling back to micro-capital mode")
                    await self._initialize_micro_capital_mode()
            
            # Update system metrics
            await self.update_system_metrics()
            
            logger.info(f"✅ Founding Queen {self.ant_id} initialized successfully with {self.capital.current_balance} SOL")
            logger.info(f"🛡️ Defense status: {'ACTIVE' if self.titan_shield else 'DISABLED - HIGH RISK'}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Founding Queen initialization critical failure: {str(e)}")
            return False
    
    async def _initialize_micro_capital_mode(self):
        """Initialize micro-capital mode with direct Princess management"""
        try:
            # Create a Princess directly under Founding Queen for micro-capital trading
            if self.capital.available_capital >= SystemConstants.PRINCESS_INITIAL_CAPITAL:
                princess_capital = min(self.capital.available_capital * 0.5, SystemConstants.PRINCESS_INITIAL_CAPITAL)
                princess_id = f"micro_princess_{int(time.time())}"
                
                # Create Princess directly (avoid circular import)
                princess = AntPrincess(
                    ant_id=princess_id,
                    parent_id=self.ant_id,
                    initial_capital=princess_capital,
                    titan_shield=self.titan_shield
                )
                
                # Allocate capital
                if self.capital.allocate_capital(princess_capital):
                    # Store Princess in a micro-princesses dict
                    if not hasattr(self, 'micro_princesses'):
                        self.micro_princesses = {}
                    self.micro_princesses[princess_id] = princess
                    self.children.append(princess_id)
                    
                    logger.info(f"🐜 Created micro-capital Princess {princess_id} with {princess_capital} SOL")
                    logger.info("💡 Micro-capital mode: Trading directly without Queens")
                else:
                    logger.warning("⚠️ Insufficient capital even for micro-Princess")
            else:
                logger.warning(f"⚠️ Capital too low for any trading: {self.capital.available_capital} < {SystemConstants.PRINCESS_INITIAL_CAPITAL}")
                
        except Exception as e:
            logger.error(f"❌ Micro-capital mode initialization failed: {str(e)}")
    
    async def create_queen(self, initial_capital: float = SystemConstants.QUEEN_SPLIT_THRESHOLD) -> Optional[str]:
        """Create a new Queen with allocated capital and proper defense integration"""
        try:
            # Validate capital allocation
            if initial_capital <= 0:
                logger.error(f"Founding Queen {self.ant_id} invalid Queen capital: {initial_capital}")
                return None
                
            if not self.capital.allocate_capital(initial_capital):
                logger.warning(f"Founding Queen {self.ant_id} insufficient capital to create Queen "
                             f"(requested: {initial_capital}, available: {self.capital.available_capital})")
                return None
            
            # Check Queen limit
            if len(self.queens) >= SystemConstants.FOUNDING_QUEEN_MAX_CHILDREN:
                logger.warning(f"Founding Queen {self.ant_id} Queen limit reached: {len(self.queens)}")
                self.capital.release_capital(initial_capital)  # Release allocated capital
                return None
            
            queen_id = f"queen_{int(time.time())}_{len(self.queens)}"
            
            # CRITICAL: Pass Titan Shield to new Queen
            queen = AntQueen(
                ant_id=queen_id, 
                parent_id=self.ant_id, 
                initial_capital=initial_capital,
                titan_shield=self.titan_shield  # DEFENSE INTEGRATION
            )
            
            if self.titan_shield:
                logger.info(f"🛡️ Titan Shield propagated to new Queen {queen_id}")
            else:
                logger.critical(f"🚨 Queen {queen_id} created WITHOUT defense protection!")
            
            # Initialize Queen's AI components with timeout protection
            try:
                ai_task = asyncio.create_task(queen.initialize_ai_components())
                ai_success = await asyncio.wait_for(ai_task, timeout=SystemConstants.AI_ANALYSIS_TIMEOUT)
                if not ai_success:
                    logger.warning(f"Queen {queen_id} AI initialization failed - will operate with limited capabilities")
            except asyncio.TimeoutError:
                logger.warning(f"Queen {queen_id} AI initialization timeout - will operate with limited capabilities")
            except Exception as e:
                logger.warning(f"Queen {queen_id} AI initialization error: {str(e)} - will operate with limited capabilities")
            
            self.queens[queen_id] = queen
            self.children.append(queen_id)
            
            logger.info(f"Founding Queen {self.ant_id} created Queen {queen_id} with {initial_capital} SOL "
                       f"(defense: {'✅' if self.titan_shield else '❌'})")
            return queen_id
            
        except Exception as e:
            logger.error(f"Founding Queen {self.ant_id} Queen creation critical error: {str(e)}")
            # Ensure capital is released on failure
            if initial_capital > 0:
                self.capital.release_capital(initial_capital)
            return None
    
    async def coordinate_system(self, market_opportunities: List[Dict]) -> Dict[str, Any]:
        """Coordinate the entire Ant system with enhanced error handling and micro-capital support"""
        try:
            results = {
                "decisions": [],
                "system_actions": [],
                "metrics": {},
                "errors": []
            }
            
            # Input validation
            if not market_opportunities:
                logger.debug(f"Founding Queen {self.ant_id} received no market opportunities")
                results["metrics"] = self.system_metrics
                return results
            
            # Handle micro-capital mode (direct Princess management)
            if hasattr(self, 'micro_princesses') and self.micro_princesses:
                logger.debug(f"🐜 Coordinating {len(self.micro_princesses)} micro-capital Princesses")
                for princess_id, princess in self.micro_princesses.items():
                    try:
                        # Give each Princess a subset of opportunities
                        for opportunity in market_opportunities[:3]:  # Limit to avoid overwhelming
                            decision = {
                                "princess_id": princess_id,
                                "decision": {
                                    "action": "monitor",  # Conservative for micro-capital
                                    "token_address": opportunity.get("token_address", "unknown"),
                                    "confidence": 0.3,
                                    "position_size": 0.001,  # Very small positions
                                    "reasoning": "Micro-capital conservative analysis",
                                    "timestamp": time.time()
                                }
                            }
                            results["decisions"].append(decision)
                    except Exception as e:
                        error_msg = f"Micro-Princess {princess_id} coordination error: {str(e)}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)
                        
                results["system_actions"].append("Micro-capital mode coordination completed")
            
            # Standard mode: Distribute opportunities across Queens
            elif self.queens:
                opportunities_per_queen = max(1, len(market_opportunities) // len(self.queens))
                
                for i, (queen_id, queen) in enumerate(self.queens.items()):
                    try:
                        start_idx = i * opportunities_per_queen
                        end_idx = start_idx + opportunities_per_queen if i < len(self.queens) - 1 else len(market_opportunities)
                        queen_opportunities = market_opportunities[start_idx:end_idx]
                        
                        # Add timeout protection for Queen operations
                        queen_task = asyncio.create_task(queen.manage_princesses(queen_opportunities))
                        queen_results = await asyncio.wait_for(
                            queen_task, 
                            timeout=SystemConstants.AI_ANALYSIS_TIMEOUT * 2  # More time for Queen operations
                        )
                        results["decisions"].extend(queen_results)
                        
                    except asyncio.TimeoutError:
                        error_msg = f"Queen {queen_id} operation timeout"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)
                        continue
                    except Exception as e:
                        error_msg = f"Queen {queen_id} operation error: {str(e)}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)
                        continue
            else:
                logger.warning(f"Founding Queen {self.ant_id} has no active Queens or Princesses")
                results["system_actions"].append("No trading agents available")
            
            # Check for system-level actions with validation
            try:
                if self.should_split() and len(self.queens) < SystemConstants.FOUNDING_QUEEN_MAX_CHILDREN:
                    new_queen_id = await self.create_queen()
                    if new_queen_id:
                        results["system_actions"].append(f"Created new Queen: {new_queen_id}")
                    else:
                        results["errors"].append("Failed to create new Queen despite split criteria")
            except Exception as e:
                error_msg = f"System-level action error: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
            
            # Update system metrics
            try:
                await self.update_system_metrics()
                results["metrics"] = self.system_metrics
            except Exception as e:
                error_msg = f"System metrics update error: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                results["metrics"] = {"error": "metrics_update_failed"}
            
            self.update_activity()
            
            # Log coordination summary
            coordination_type = "micro-capital" if hasattr(self, 'micro_princesses') and self.micro_princesses else "standard"
            logger.info(f"Founding Queen {self.ant_id} coordination ({coordination_type}): "
                       f"{len(results['decisions'])} decisions, "
                       f"{len(results['system_actions'])} actions, "
                       f"{len(results['errors'])} errors")
            
            return results
            
        except Exception as e:
            logger.error(f"Founding Queen {self.ant_id} coordination critical failure: {str(e)}")
            return {"decisions": [], "system_actions": [], "metrics": {}, "errors": [str(e)]}
    
    async def update_system_metrics(self):
        """Update system-wide metrics with enhanced validation"""
        try:
            total_capital = self.capital.current_balance
            total_trades = 0
            total_profit = 0.0
            total_ants = 1  # Founding Queen
            total_defense_activations = 0
            total_defense_rejections = 0
            
            # Aggregate metrics from all Queens with error handling
            for queen in self.queens.values():
                try:
                    total_capital += queen.capital.current_balance
                    total_trades += queen.performance.total_trades
                    total_profit += queen.performance.total_profit
                    total_ants += 1  # Queen
                    
                    # Add Princess metrics
                    for princess in queen.princesses.values():
                        try:
                            total_capital += princess.capital.current_balance
                            total_trades += princess.performance.total_trades
                            total_profit += princess.performance.total_profit
                            total_ants += 1  # Princess
                            
                            # Add defense metrics if available
                            if hasattr(princess, 'defense_mode_overrides'):
                                total_defense_activations += len(princess.defense_mode_overrides)
                                
                        except Exception as e:
                            logger.warning(f"Error aggregating Princess metrics: {str(e)}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error aggregating Queen metrics: {str(e)}")
                    continue
            
            # Calculate system health metrics
            system_runtime = time.time() - self.system_metrics["system_start_time"]
            roi_percent = ((total_capital - self.system_metrics["total_capital"]) / 
                          max(self.system_metrics["total_capital"], 0.001)) * 100
            
            self.system_metrics.update({
                "total_ants": total_ants,
                "total_capital": total_capital,
                "total_trades": total_trades,
                "system_profit": total_profit,
                "system_uptime_hours": system_runtime / 3600,
                "defense_activations": total_defense_activations,
                "defense_rejections": total_defense_rejections,
                "roi_percent": roi_percent,
                "active_queens": len(self.queens),
                "total_princesses": sum(len(q.princesses) for q in self.queens.values()),
                "trades_per_hour": total_trades / max(system_runtime / 3600, 0.001),
                "profit_per_hour": total_profit / max(system_runtime / 3600, 0.001),
                "last_metrics_update": time.time()
            })
            
            logger.debug(f"System metrics updated: {total_ants} ants, "
                        f"{total_capital:.4f} SOL, "
                        f"{total_trades} trades, "
                        f"ROI: {roi_percent:.2f}%")
            
        except Exception as e:
            logger.error(f"Critical error updating system metrics: {str(e)}")
            # Ensure basic metrics are maintained
            self.system_metrics["last_metrics_update"] = time.time()
            self.system_metrics["metrics_error"] = str(e)

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
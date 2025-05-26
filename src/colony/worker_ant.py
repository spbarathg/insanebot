"""
Worker Ant - Individual trading agent with compounding behavior

Executes meme coin buy/sell strategies with:
- 5-10 trades per coin targeting 1.03x-1.50x returns
- Splits into 5 Worker Ants when reaching 2 SOL
- Merges with others when underperforming
"""

import asyncio
import logging
import random
import time
from typing import Dict, List, Optional, Any, Tuple

from .base_ant import BaseAnt, AntRole, AntStatus
from ..intelligence.ai_coordinator import AICoordinator
from ..trading.trade_executor import TradeExecutor
from ..trading.compounding_logic import CompoundingLogic

logger = logging.getLogger(__name__)

class WorkerAnt(BaseAnt):
    """Individual trading agent with compounding behavior"""
    
    def __init__(self, ant_id: str, parent_id: str, initial_capital: float = 0.4):
        super().__init__(ant_id, AntRole.WORKER, parent_id)
        
        # Worker-specific attributes
        self.current_coin: Optional[str] = None
        self.trades_on_current_coin: int = 0
        self.position_entry_price: float = 0.0
        self.position_size: float = 0.0
        self.target_trades_this_coin: int = 0
        
        # Performance tracking
        self.coins_traded: List[str] = []
        self.coin_performance: Dict[str, Dict] = {}
        
        # AI and trading components (initialized later)
        self.ai_coordinator: Optional[AICoordinator] = None
        self.trade_executor: Optional[TradeExecutor] = None
        self.compounding_logic: Optional[CompoundingLogic] = None
        
        # Initialize capital
        self.capital.update_balance(initial_capital)
        
        logger.info(f"WorkerAnt {ant_id} created with {initial_capital} SOL capital")
    
    async def initialize(self) -> bool:
        """Initialize the worker ant with AI and trading components"""
        try:
            # Initialize AI coordinator for decision making
            self.ai_coordinator = AICoordinator()
            await self.ai_coordinator.initialize()
            
            # Initialize trade executor
            self.trade_executor = TradeExecutor()
            await self.trade_executor.initialize()
            
            # Initialize compounding logic
            self.compounding_logic = CompoundingLogic()
            
            logger.info(f"WorkerAnt {self.ant_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize WorkerAnt {self.ant_id}: {e}")
            self.status = AntStatus.ERROR
            return False
    
    async def execute_cycle(self) -> Dict[str, Any]:
        """Execute one trading cycle"""
        if self.status != AntStatus.ACTIVE:
            return {"status": "inactive", "reason": f"Ant status: {self.status.value}"}
        
        try:
            self.update_activity()
            
            # Check if we should retire first
            if self.should_retire():
                await self._prepare_retirement()
                return {"status": "retiring", "reason": "Reached retirement criteria"}
            
            # Check if we should split
            if self.should_split():
                split_request = await self._prepare_split()
                return {"status": "splitting", "split_request": split_request}
            
            # Check if we should merge
            if self.should_merge():
                merge_request = await self._prepare_merge()
                return {"status": "merging", "merge_request": merge_request}
            
            # Execute trading logic
            trading_result = await self._execute_trading_logic()
            
            # Update compounding metrics
            if trading_result.get("trade_executed"):
                await self._update_compounding_metrics(trading_result)
            
            return {
                "status": "active",
                "trading_result": trading_result,
                "performance": self.get_performance_summary(),
                "flags": {
                    "should_split": self.should_split(),
                    "should_merge": self.should_merge(),
                    "should_retire": self.should_retire()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in WorkerAnt {self.ant_id} cycle: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _execute_trading_logic(self) -> Dict[str, Any]:
        """Execute the core trading logic for this worker"""
        try:
            # If we don't have a current coin, find one
            if not self.current_coin:
                coin_selection = await self._select_coin_to_trade()
                if not coin_selection["success"]:
                    return coin_selection
                
                self.current_coin = coin_selection["coin_address"]
                self.trades_on_current_coin = 0
                self.target_trades_this_coin = random.randint(*self.config["trades_per_coin"])
                
                logger.info(f"WorkerAnt {self.ant_id} selected coin {self.current_coin} "
                           f"for {self.target_trades_this_coin} trades")
            
            # Execute trade on current coin
            trade_result = await self._execute_trade_on_coin()
            
            # Check if we've completed our trades on this coin
            if self.trades_on_current_coin >= self.target_trades_this_coin:
                await self._complete_coin_trading()
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error in trading logic for WorkerAnt {self.ant_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _select_coin_to_trade(self) -> Dict[str, Any]:
        """Select a meme coin to trade based on AI analysis"""
        try:
            # Get coin recommendations from AI coordinator
            recommendations = await self.ai_coordinator.get_trading_recommendations(
                capital_available=self.capital.available_capital,
                risk_tolerance=self.performance.risk_score,
                worker_experience=self.performance.total_trades
            )
            
            if not recommendations or not recommendations.get("coins"):
                return {"success": False, "reason": "No suitable coins found"}
            
            # Select the top recommended coin
            selected_coin = recommendations["coins"][0]
            
            return {
                "success": True,
                "coin_address": selected_coin["address"],
                "coin_symbol": selected_coin["symbol"],
                "confidence_score": selected_coin["confidence"],
                "recommendation_data": selected_coin
            }
            
        except Exception as e:
            logger.error(f"Error selecting coin for WorkerAnt {self.ant_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_trade_on_coin(self) -> Dict[str, Any]:
        """Execute a buy or sell trade on the current coin"""
        try:
            # Determine trade direction and size
            trade_decision = await self._make_trade_decision()
            
            if not trade_decision["should_trade"]:
                return {
                    "trade_executed": False,
                    "reason": trade_decision["reason"],
                    "waiting_for_signal": True
                }
            
            # Calculate position size with compounding
            position_size = await self._calculate_position_size(trade_decision)
            
            # Execute the trade
            trade_result = await self.trade_executor.execute_trade(
                coin_address=self.current_coin,
                trade_type=trade_decision["direction"],
                amount=position_size,
                max_slippage=0.02,
                worker_id=self.ant_id
            )
            
            if trade_result["success"]:
                # Update our position tracking
                await self._update_position_tracking(trade_result, trade_decision)
                
                # Update performance metrics
                profit = trade_result.get("profit", 0.0)
                trade_time = trade_result.get("execution_time", 0.0)
                success = profit > 0
                
                self.performance.update_trade_result(profit, trade_time, success)
                self.trades_on_current_coin += 1
                
                # Update capital
                new_balance = self.capital.current_balance + profit
                self.capital.update_balance(new_balance)
                
                logger.info(f"WorkerAnt {self.ant_id} executed {trade_decision['direction']} "
                           f"on {self.current_coin}, profit: {profit}")
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error executing trade for WorkerAnt {self.ant_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _make_trade_decision(self) -> Dict[str, Any]:
        """Make a buy/sell decision based on AI analysis and current position"""
        try:
            # Get current market analysis for the coin
            market_analysis = await self.ai_coordinator.analyze_coin_opportunity(
                coin_address=self.current_coin,
                current_position_size=self.position_size,
                trades_on_coin=self.trades_on_current_coin
            )
            
            # Determine if we should buy or sell
            if self.position_size == 0:
                # No position, consider buying
                if market_analysis["buy_signal"] and market_analysis["confidence"] > 0.6:
                    return {
                        "should_trade": True,
                        "direction": "buy",
                        "confidence": market_analysis["confidence"],
                        "reason": "Strong buy signal detected"
                    }
            else:
                # Have position, consider selling
                current_return = await self._calculate_current_return()
                target_min, target_max = self.config["target_return_range"]
                
                # Sell if we've hit our target return or stop loss
                if current_return >= target_min or current_return <= -0.05:  # 5% stop loss
                    return {
                        "should_trade": True,
                        "direction": "sell",
                        "confidence": 0.8,
                        "reason": f"Target return achieved: {current_return:.3f}"
                    }
                
                # Sell if market conditions deteriorated
                if market_analysis["sell_signal"] and market_analysis["confidence"] > 0.7:
                    return {
                        "should_trade": True,
                        "direction": "sell",
                        "confidence": market_analysis["confidence"],
                        "reason": "Strong sell signal detected"
                    }
            
            return {
                "should_trade": False,
                "reason": "No clear trading signal",
                "market_analysis": market_analysis
            }
            
        except Exception as e:
            logger.error(f"Error making trade decision for WorkerAnt {self.ant_id}: {e}")
            return {"should_trade": False, "reason": f"Decision error: {e}"}
    
    async def _calculate_position_size(self, trade_decision: Dict) -> float:
        """Calculate position size with compounding logic"""
        try:
            base_size = self.capital.available_capital * 0.2  # Base 20% of available capital
            
            # Apply compounding multiplier based on performance
            compound_multiplier = await self.compounding_logic.calculate_multiplier(
                win_rate=self.performance.win_rate,
                profit_factor=self.performance.compound_factor,
                trades_completed=self.performance.total_trades,
                confidence=trade_decision["confidence"]
            )
            
            position_size = base_size * compound_multiplier
            
            # Cap position size to available capital
            max_position = self.capital.available_capital * 0.8  # Max 80% of capital
            position_size = min(position_size, max_position)
            
            logger.debug(f"WorkerAnt {self.ant_id} position size: {position_size} "
                        f"(base: {base_size}, multiplier: {compound_multiplier})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for WorkerAnt {self.ant_id}: {e}")
            return self.capital.available_capital * 0.1  # Fallback to 10%
    
    async def _calculate_current_return(self) -> float:
        """Calculate current return on position"""
        try:
            if self.position_size == 0 or self.position_entry_price == 0:
                return 0.0
            
            # Get current price for the coin
            current_price = await self.trade_executor.get_current_price(self.current_coin)
            
            if current_price > 0:
                return (current_price - self.position_entry_price) / self.position_entry_price
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating return for WorkerAnt {self.ant_id}: {e}")
            return 0.0
    
    async def _update_position_tracking(self, trade_result: Dict, trade_decision: Dict):
        """Update position tracking after trade execution"""
        if trade_decision["direction"] == "buy":
            self.position_entry_price = trade_result["execution_price"]
            self.position_size = trade_result["amount"]
        elif trade_decision["direction"] == "sell":
            self.position_entry_price = 0.0
            self.position_size = 0.0
    
    async def _complete_coin_trading(self):
        """Complete trading on current coin and prepare for next"""
        # Record performance for this coin
        if self.current_coin:
            coin_performance = {
                "trades_executed": self.trades_on_current_coin,
                "target_trades": self.target_trades_this_coin,
                "final_return": await self._calculate_current_return(),
                "completed_at": time.time()
            }
            self.coin_performance[self.current_coin] = coin_performance
            self.coins_traded.append(self.current_coin)
        
        # Reset for next coin
        self.current_coin = None
        self.trades_on_current_coin = 0
        self.position_entry_price = 0.0
        self.position_size = 0.0
        self.target_trades_this_coin = 0
        
        logger.info(f"WorkerAnt {self.ant_id} completed trading cycle on coin")
    
    async def _update_compounding_metrics(self, trading_result: Dict):
        """Update compounding metrics after successful trade"""
        if self.compounding_logic:
            await self.compounding_logic.update_metrics(
                ant_id=self.ant_id,
                trade_result=trading_result,
                current_performance=self.performance
            )
    
    async def _prepare_split(self) -> Dict[str, Any]:
        """Prepare to split into 5 worker ants"""
        self.status = AntStatus.SPLITTING
        
        # Calculate capital distribution for 5 new workers
        total_capital = self.capital.current_balance
        capital_per_worker = total_capital / 5.0
        
        return {
            "type": "worker_split",
            "parent_id": self.ant_id,
            "new_workers_count": 5,
            "capital_per_worker": capital_per_worker,
            "inheritance_data": {
                "learned_strategies": await self._export_learned_strategies(),
                "coin_performance": self.coin_performance,
                "risk_preferences": self.performance.risk_score
            }
        }
    
    async def _prepare_retirement(self) -> Dict[str, Any]:
        """Prepare for retirement"""
        self.status = AntStatus.RETIRING
        
        # Close any open positions
        if self.position_size > 0:
            await self._close_all_positions()
        
        return {
            "type": "worker_retirement",
            "ant_id": self.ant_id,
            "final_capital": self.capital.current_balance,
            "final_performance": self.get_performance_summary(),
            "learned_data": await self._export_learned_strategies()
        }
    
    async def _prepare_merge(self) -> Dict[str, Any]:
        """Prepare to merge with better performing worker"""
        self.status = AntStatus.MERGING
        
        return {
            "type": "worker_merge",
            "ant_id": self.ant_id,
            "capital_to_transfer": self.capital.current_balance,
            "performance_data": self.get_performance_summary(),
            "merge_reason": "underperforming"
        }
    
    async def _export_learned_strategies(self) -> Dict[str, Any]:
        """Export learned strategies for inheritance"""
        return {
            "successful_patterns": self.coin_performance,
            "trade_preferences": {
                "preferred_trade_count": self.target_trades_this_coin,
                "risk_tolerance": self.performance.risk_score,
                "compound_factor": self.performance.compound_factor
            },
            "ai_learning_data": await self.ai_coordinator.export_learning_data() if self.ai_coordinator else {}
        }
    
    async def _close_all_positions(self):
        """Close all open positions before retirement/merge"""
        if self.position_size > 0 and self.current_coin:
            await self.trade_executor.execute_trade(
                coin_address=self.current_coin,
                trade_type="sell",
                amount=self.position_size,
                max_slippage=0.05,  # Higher slippage tolerance for closing
                worker_id=self.ant_id
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get detailed performance summary"""
        base_summary = self.get_status_summary()
        base_summary.update({
            "worker_specific": {
                "current_coin": self.current_coin,
                "trades_on_current_coin": self.trades_on_current_coin,
                "position_size": self.position_size,
                "position_entry_price": self.position_entry_price,
                "coins_traded_count": len(self.coins_traded),
                "coin_performance": self.coin_performance
            }
        })
        return base_summary
    
    async def cleanup(self):
        """Cleanup resources when retiring/merging"""
        try:
            # Close positions
            await self._close_all_positions()
            
            # Cleanup AI coordinator
            if self.ai_coordinator:
                await self.ai_coordinator.cleanup()
            
            # Cleanup trade executor
            if self.trade_executor:
                await self.trade_executor.cleanup()
            
            logger.info(f"WorkerAnt {self.ant_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during WorkerAnt {self.ant_id} cleanup: {e}") 
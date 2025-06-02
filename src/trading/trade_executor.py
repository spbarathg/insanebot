"""
Trade Executor - Wrapper for the TradeExecution class

This module provides a TradeExecutor class that wraps the core TradeExecution
functionality to provide a consistent interface for the worker ants.
"""

import asyncio
import logging
from typing import Dict, Optional, Any
from ..core.trade_execution import TradeExecution

logger = logging.getLogger(__name__)

class TradeExecutor:
    """Trade execution service for worker ants"""
    
    def __init__(self):
        self.trade_execution = TradeExecution()
        self.initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the trade executor"""
        try:
            self.initialized = await self.trade_execution.initialize()
            if self.initialized:
                logger.info("TradeExecutor initialized successfully")
            else:
                logger.error("Failed to initialize TradeExecutor")
            return self.initialized
        except Exception as e:
            logger.error(f"Error initializing TradeExecutor: {e}")
            return False
    
    async def execute_trade(self, coin_address: str, trade_type: str, amount: float, 
                           max_slippage: float = 0.02, worker_id: str = None) -> Dict[str, Any]:
        """Execute a trade for a worker ant"""
        if not self.initialized:
            return {
                "success": False,
                "error": "TradeExecutor not initialized"
            }
        
        try:
            # Convert trade_type to the appropriate method call
            if trade_type.lower() in ['buy', 'long']:
                result = await self.trade_execution.execute_buy(coin_address, amount)
            elif trade_type.lower() in ['sell', 'short']:
                result = await self.trade_execution.execute_sell(coin_address, amount)
            else:
                return {
                    "success": False,
                    "error": f"Invalid trade type: {trade_type}"
                }
            
            if result:
                # Format the result for worker ant consumption
                return {
                    "success": True,
                    "transaction": result.get("transaction"),
                    "amount": result.get("amount_sol", result.get("sol_amount", 0)),
                    "token_amount": result.get("token_amount", result.get("amount_tokens", 0)),
                    "price": result.get("price", 0),
                    "profit": self._calculate_profit(result, trade_type),
                    "execution_time": result.get("execution_time", 0),
                    "timestamp": result.get("timestamp", 0)
                }
            else:
                return {
                    "success": False,
                    "error": "Trade execution failed"
                }
                
        except Exception as e:
            logger.error(f"Error executing trade for worker {worker_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _calculate_profit(self, result: Dict, trade_type: str) -> float:
        """Calculate profit from trade result"""
        try:
            # This is a simplified profit calculation
            # In a real implementation, this would be more sophisticated
            if trade_type.lower() in ['sell', 'short']:
                # For sells, profit is the SOL received minus what was paid
                return result.get("sol_amount", 0) - result.get("cost_basis", 0)
            else:
                # For buys, profit is negative (cost)
                return -result.get("amount_sol", 0)
        except:
            return 0.0
    
    async def get_balance(self) -> Optional[Dict]:
        """Get wallet balance"""
        if not self.initialized:
            return None
        return await self.trade_execution.get_wallet_balance()
    
    async def close(self):
        """Close the trade executor"""
        if self.initialized:
            await self.trade_execution.close()
            self.initialized = False
            logger.info("TradeExecutor closed") 
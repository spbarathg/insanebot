"""
Trading simulator for backtesting and simulation.
"""
from typing import Dict, Optional, List
import json
import time
from loguru import logger
from solana.rpc.async_api import AsyncClient

class TradingSimulator:
    def __init__(self, client: AsyncClient):
        self.client = client
        self.state: Dict = {
            "balance": 10.0,  # Start with 10 SOL
            "trades": [],
            "holdings": {}
        }
        
    def load_state(self) -> None:
        """Load saved simulation state or create a new one."""
        try:
            with open("simulation_state.json", "r") as f:
                self.state = json.load(f)
        except FileNotFoundError:
            # Use default state if file not found
            pass
        
    def save_state(self) -> None:
        """Save the current simulation state."""
        with open("simulation_state.json", "w") as f:
            json.dump(self.state, f, indent=2)
            
    def get_balance(self) -> float:
        """Get simulated SOL balance."""
        return self.state["balance"]
        
    def get_token_balance(self, token_address: str) -> float:
        """Get simulated token balance."""
        return self.state["holdings"].get(token_address, {}).get("amount", 0.0)
        
    async def simulate_trade(
        self,
        token_address: str,
        amount: float,
        is_buy: bool
    ) -> bool:
        """Simulate a trade execution."""
        try:
            # Simulate some basic market effects
            slippage = 0.01  # 1% slippage
            price = 1.0  # Placeholder price - in a real implementation this would be queried
            
            if is_buy:
                # Check if we have enough SOL
                if amount > self.state["balance"]:
                    logger.warning(f"Insufficient SOL balance: {self.state['balance']} < {amount}")
                    return False
                    
                # Execute buy
                cost = amount * (1 + slippage)
                token_amount = amount / price
                
                # Update state
                self.state["balance"] -= cost
                
                if token_address not in self.state["holdings"]:
                    self.state["holdings"][token_address] = {
                        "amount": 0.0,
                        "cost_basis": 0.0
                    }
                    
                # Update holdings
                old_amount = self.state["holdings"][token_address]["amount"]
                old_cost = self.state["holdings"][token_address]["cost_basis"]
                
                # Calculate new cost basis
                total_cost = old_cost + cost
                total_amount = old_amount + token_amount
                
                self.state["holdings"][token_address]["amount"] = total_amount
                self.state["holdings"][token_address]["cost_basis"] = total_cost
                
            else:
                # Sell operation
                token_balance = self.get_token_balance(token_address)
                
                if token_balance == 0:
                    logger.warning(f"No token balance for {token_address}")
                    return False
                    
                # Calculate sell amount (either the requested amount or all available)
                sell_amount = min(token_balance, amount / price)
                
                # Execute sell
                sol_received = (sell_amount * price) * (1 - slippage)
                
                # Update state
                self.state["balance"] += sol_received
                self.state["holdings"][token_address]["amount"] -= sell_amount
                
                # If all tokens are sold, calculate profit/loss
                if self.state["holdings"][token_address]["amount"] == 0:
                    cost_basis = self.state["holdings"][token_address]["cost_basis"]
                    profit = sol_received - cost_basis
                    profit_percent = (profit / cost_basis) * 100
                    
                    logger.info(f"Sold all tokens with profit: {profit:.4f} SOL ({profit_percent:.2f}%)")
                    
                    # Reset cost basis
                    self.state["holdings"][token_address]["cost_basis"] = 0.0
            
            # Record the trade
            self.state["trades"].append({
                "token": token_address,
                "is_buy": is_buy,
                "amount": amount,
                "timestamp": time.time()
            })
            
            # Save updated state
            self.save_state()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in trade simulation: {str(e)}")
            return False 
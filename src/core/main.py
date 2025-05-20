"""
Main bot implementation for Solana trading bot.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from ..utils.config import settings
from .market_data import MarketData
from .trade_execution import TradeExecution
from .wallet_manager import WalletManager
from .local_llm import LocalLLM
from .helius_service import HeliusService
from .jupiter_service import JupiterService

logger = logging.getLogger(__name__)

class MemeCoinBot:
    """
    Main Solana trading bot implementation that combines market data,
    trade execution, local LLM for trading decisions, and monitoring.
    """
    
    def __init__(self):
        self.market_data = MarketData()
        self.trade_execution = TradeExecution()
        self.wallet_manager = WalletManager()
        self.local_llm = LocalLLM()
        self.helius = HeliusService()
        self.jupiter = JupiterService()
        self.trade_history = []
        self.active_trades = {}
        self.running = False
        self.daily_loss = 0
        self.last_reset = datetime.now()
        self.token_watchlist = []
        
    async def initialize(self) -> bool:
        """Initialize all bot components."""
        try:
            logger.info("Initializing MemeCoin trading bot...")
            
            # Initialize services
            if not await self.helius.initialize():
                logger.error("Failed to initialize Helius service")
                return False
                
            if not await self.jupiter.initialize():
                logger.error("Failed to initialize Jupiter service")
                return False
                
            if not await self.market_data.initialize():
                logger.error("Failed to initialize market data service")
                return False
                
            if not await self.wallet_manager.initialize():
                logger.error("Failed to initialize wallet manager")
                return False
                
            if not await self.trade_execution.initialize():
                logger.error("Failed to initialize trade execution")
                return False
                
            # Initialize LLM for trading decisions if enabled
            if settings.USE_LOCAL_LLM:
                if not await self.local_llm.initialize():
                    logger.error("Failed to initialize local LLM")
                    # Continue without LLM if it fails
            
            # Load trade history
            self._load_trade_history()
            
            # Load token watchlist
            self._load_token_watchlist()
            
            logger.info("MemeCoin trading bot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing bot: {str(e)}")
            return False
            
    def _load_trade_history(self) -> None:
        """Load trade history from file."""
        try:
            if not settings.DATA_DIR.exists():
                settings.DATA_DIR.mkdir(parents=True)
                
            history_file = settings.DATA_DIR / "trade_history.json"
            if history_file.exists():
                with open(history_file, "r") as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded {len(self.trade_history)} trades from history")
            else:
                logger.info("No trade history found, starting fresh")
                self.trade_history = []
        except Exception as e:
            logger.error(f"Error loading trade history: {str(e)}")
            self.trade_history = []
            
    def _save_trade_history(self) -> None:
        """Save trade history to file."""
        try:
            if not settings.DATA_DIR.exists():
                settings.DATA_DIR.mkdir(parents=True)
                
            history_file = settings.DATA_DIR / "trade_history.json"
            with open(history_file, "w") as f:
                json.dump(self.trade_history, f, indent=2)
            logger.info(f"Saved {len(self.trade_history)} trades to history")
        except Exception as e:
            logger.error(f"Error saving trade history: {str(e)}")
            
    def _load_token_watchlist(self) -> None:
        """Load token watchlist from file."""
        try:
            if not settings.DATA_DIR.exists():
                settings.DATA_DIR.mkdir(parents=True)
                
            watchlist_file = settings.DATA_DIR / "token_watchlist.json"
            if watchlist_file.exists():
                with open(watchlist_file, "r") as f:
                    self.token_watchlist = json.load(f)
                logger.info(f"Loaded {len(self.token_watchlist)} tokens from watchlist")
            else:
                logger.info("No token watchlist found, starting fresh")
                self.token_watchlist = []
        except Exception as e:
            logger.error(f"Error loading token watchlist: {str(e)}")
            self.token_watchlist = []
            
    def _save_token_watchlist(self) -> None:
        """Save token watchlist to file."""
        try:
            if not settings.DATA_DIR.exists():
                settings.DATA_DIR.mkdir(parents=True)
                
            watchlist_file = settings.DATA_DIR / "token_watchlist.json"
            with open(watchlist_file, "w") as f:
                json.dump(self.token_watchlist, f, indent=2)
            logger.info(f"Saved {len(self.token_watchlist)} tokens to watchlist")
        except Exception as e:
            logger.error(f"Error saving token watchlist: {str(e)}")
    
    async def close(self) -> None:
        """Close all services."""
        try:
            # Save data
            self._save_trade_history()
            self._save_token_watchlist()
            
            # Close services
            await self.helius.close()
            await self.jupiter.close()
            await self.market_data.close()
            await self.wallet_manager.close()
            await self.trade_execution.close()
            
            if settings.USE_LOCAL_LLM:
                await self.local_llm.close()
                
            logger.info("All services closed successfully")
        except Exception as e:
            logger.error(f"Error closing services: {str(e)}")
            
    async def add_token_to_watchlist(self, token_address: str) -> bool:
        """Add a token to the watchlist."""
        try:
            # Check if token is valid
            token_info = await self.market_data.get_token_metadata(token_address)
            if not token_info:
                logger.warning(f"Invalid token: {token_address}")
                return False
                
            # Add to watchlist if not already there
            if token_address not in self.token_watchlist:
                self.token_watchlist.append(token_address)
                self._save_token_watchlist()
                logger.info(f"Added {token_address} to watchlist")
                return True
            else:
                logger.info(f"Token {token_address} already in watchlist")
                return True
                
        except Exception as e:
            logger.error(f"Error adding token to watchlist: {str(e)}")
            return False
            
    async def remove_token_from_watchlist(self, token_address: str) -> bool:
        """Remove a token from the watchlist."""
        try:
            if token_address in self.token_watchlist:
                self.token_watchlist.remove(token_address)
                self._save_token_watchlist()
                logger.info(f"Removed {token_address} from watchlist")
                return True
            else:
                logger.info(f"Token {token_address} not in watchlist")
                return False
                
        except Exception as e:
            logger.error(f"Error removing token from watchlist: {str(e)}")
            return False
            
    async def process_token(self, token_address: str) -> Optional[Dict]:
        """Process a token for potential trading."""
        try:
            # Get token data
            token_data = await self.market_data.get_token_full_data(token_address)
            if not token_data:
                logger.warning(f"No data for token: {token_address}")
                return None
                
            # Analyze using LLM if available
            llm_analysis = None
            if settings.USE_LOCAL_LLM:
                llm_analysis = await self.local_llm.analyze_market(token_data)
                
            # Combine data
            result = {
                "token": token_data,
                "analysis": llm_analysis,
                "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing token: {str(e)}")
            return None
            
    async def execute_buy(self, token_address: str, amount_sol: float) -> Optional[Dict]:
        """Execute a buy trade."""
        try:
            # Process token first
            token_data = await self.process_token(token_address)
            if not token_data:
                logger.warning(f"Cannot buy token without data: {token_address}")
                return None
                
            # Check daily loss limit
            if self.daily_loss >= settings.DAILY_LOSS_LIMIT:
                logger.warning(f"Daily loss limit reached: {self.daily_loss} >= {settings.DAILY_LOSS_LIMIT}")
                return None
                
            # Execute trade
            result = await self.trade_execution.execute_buy(token_address, amount_sol)
            if result:
                # Add to active trades
                self.active_trades[token_address] = {
                    "type": "buy",
                    "amount_sol": amount_sol,
                    "amount_tokens": result["token_amount"],
                    "price": result["price"],
                    "timestamp": result["timestamp"],
                }
                
                # Add to trade history
                self.trade_history.append(result)
                self._save_trade_history()
                
                # If using LLM, have it learn from the trade
                if settings.USE_LOCAL_LLM:
                    self.local_llm.learn_from_trade({
                        "token": token_address,
                        "market_state": token_data["token"],
                        "decision": "buy",
                        "amount": amount_sol,
                        "price": result["price"],
                        "profit": 0,  # No profit yet
                        "timestamp": result["timestamp"]
                    })
                    
                logger.info(f"Buy executed: {amount_sol} SOL → {result['token_amount']} tokens at {result['price']}")
                return result
            else:
                logger.warning(f"Buy failed for {token_address}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing buy: {str(e)}")
            return None
            
    async def execute_sell(self, token_address: str, amount_tokens: float) -> Optional[Dict]:
        """Execute a sell trade."""
        try:
            # Check if we have active trade
            if token_address not in self.active_trades:
                logger.warning(f"No active trade for token: {token_address}")
                return None
                
            # Execute trade
            result = await self.trade_execution.execute_sell(token_address, amount_tokens)
            if result:
                # Calculate profit/loss
                buy_data = self.active_trades[token_address]
                buy_value = buy_data["amount_sol"]
                sell_value = result["sol_amount"]
                profit = sell_value - buy_value
                profit_percentage = (profit / buy_value) * 100 if buy_value > 0 else 0
                
                # Update result with profit info
                result["buy_value"] = buy_value
                result["profit"] = profit
                result["profit_percentage"] = profit_percentage
                
                # Update daily loss if negative
                if profit < 0:
                    self.daily_loss += abs(profit)
                    
                # Remove from active trades
                del self.active_trades[token_address]
                
                # Add to trade history
                self.trade_history.append(result)
                self._save_trade_history()
                
                # If using LLM, have it learn from the trade
                if settings.USE_LOCAL_LLM:
                    token_data = await self.market_data.get_token_full_data(token_address)
                    if token_data:
                        self.local_llm.learn_from_trade({
                            "token": token_address,
                            "market_state": token_data,
                            "decision": "sell",
                            "amount": amount_tokens,
                            "price": result["price"],
                            "profit": profit,
                            "timestamp": result["timestamp"]
                        })
                        
                logger.info(f"Sell executed: {amount_tokens} tokens → {result['sol_amount']} SOL, Profit: {profit} SOL ({profit_percentage:.2f}%)")
                return result
            else:
                logger.warning(f"Sell failed for {token_address}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing sell: {str(e)}")
            return None
            
    async def start(self) -> None:
        """Start the trading bot."""
        try:
            self.running = True
            logger.info("Starting MemeCoin trading bot...")
            
            # Reset daily loss if it's a new day
            self._check_daily_reset()
            
            # Main bot loop
            while self.running:
                try:
                    # Process watchlist tokens
                    for token_address in self.token_watchlist:
                        await self._process_watchlist_token(token_address)
                        
                    # Check active trades for potential sells
                    for token_address, trade_data in list(self.active_trades.items()):
                        await self._check_active_trade(token_address, trade_data)
                        
                    # Sleep to avoid excessive API calls
                    await asyncio.sleep(settings.LOOP_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"Error in bot loop: {str(e)}")
                    await asyncio.sleep(10)  # Sleep longer on error
                    
            logger.info("Bot stopped")
            
        except Exception as e:
            logger.error(f"Error starting bot: {str(e)}")
            self.running = False
            
    def stop(self) -> None:
        """Stop the trading bot."""
        self.running = False
        logger.info("Bot scheduled to stop")
        
    def _check_daily_reset(self) -> None:
        """Check and reset daily loss counter if it's a new day."""
        now = datetime.now()
        if now.date() > self.last_reset.date():
            logger.info(f"Resetting daily loss from {self.daily_loss} to 0")
            self.daily_loss = 0
            self.last_reset = now
            
    async def _process_watchlist_token(self, token_address: str) -> None:
        """Process a token from the watchlist."""
        try:
            # Get token data
            token_data = await self.process_token(token_address)
            if not token_data:
                return
                
            # Skip if already in active trades
            if token_address in self.active_trades:
                return
                
            # Check if this token should be bought based on analysis
            should_buy = False
            amount_sol = 0
            
            if settings.USE_LOCAL_LLM:
                # Use LLM recommendation if available
                if token_data.get("analysis"):
                    action = token_data["analysis"].get("action")
                    confidence = token_data["analysis"].get("confidence", 0)
                    if action == "buy" and confidence >= settings.MIN_CONFIDENCE:
                        should_buy = True
                        amount_sol = token_data["analysis"].get("position_size", settings.DEFAULT_POSITION_SIZE)
            else:
                # Use basic rules if no LLM
                token_info = token_data["token"]
                liquidity = token_info.get("liquidity_usd", 0)
                if liquidity >= settings.MIN_LIQUIDITY:
                    should_buy = True
                    amount_sol = settings.DEFAULT_POSITION_SIZE
                    
            # Execute buy if conditions met
            if should_buy and amount_sol > 0:
                await self.execute_buy(token_address, amount_sol)
                
        except Exception as e:
            logger.error(f"Error processing watchlist token {token_address}: {str(e)}")
            
    async def _check_active_trade(self, token_address: str, trade_data: Dict) -> None:
        """Check an active trade for potential sell."""
        try:
            # Get current token data
            token_data = await self.process_token(token_address)
            if not token_data:
                return
                
            # Get current price
            current_price_usd = token_data["token"].get("price_usd", 0)
            buy_price = trade_data.get("price", 0)
            
            # Calculate current profit/loss
            profit_percentage = 0
            if buy_price > 0:
                profit_percentage = ((current_price_usd / buy_price) - 1) * 100
                
            # Determine if we should sell
            should_sell = False
            
            if settings.USE_LOCAL_LLM:
                # Use LLM recommendation if available
                if token_data.get("analysis"):
                    action = token_data["analysis"].get("action")
                    confidence = token_data["analysis"].get("confidence", 0)
                    if action == "sell" and confidence >= settings.MIN_CONFIDENCE:
                        should_sell = True
            else:
                # Use basic rules if no LLM
                # Sell if reached target profit or stop loss
                if profit_percentage >= settings.TARGET_PROFIT or profit_percentage <= -settings.STOP_LOSS:
                    should_sell = True
                    
            # Execute sell if conditions met
            if should_sell:
                amount_tokens = trade_data.get("amount_tokens", 0)
                if amount_tokens > 0:
                    await self.execute_sell(token_address, amount_tokens)
                    
        except Exception as e:
            logger.error(f"Error checking active trade {token_address}: {str(e)}")
            
    async def get_portfolio_status(self) -> Dict:
        """Get current portfolio status."""
        try:
            # Get wallet balance
            sol_balance = await self.wallet_manager.check_balance()
            
            # Calculate total investment and current value
            total_investment = 0
            current_value = 0
            
            for token_address, trade_data in self.active_trades.items():
                token_data = await self.process_token(token_address)
                if token_data:
                    # Calculate investment value
                    investment = trade_data.get("amount_sol", 0)
                    total_investment += investment
                    
                    # Calculate current value
                    amount_tokens = trade_data.get("amount_tokens", 0)
                    current_price = token_data["token"].get("price_sol", 0)
                    if current_price > 0:
                        token_value = amount_tokens * current_price
                        current_value += token_value
                        
            # Calculate overall profit/loss
            overall_profit = current_value - total_investment
            overall_profit_percentage = (overall_profit / total_investment) * 100 if total_investment > 0 else 0
            
            return {
                "sol_balance": sol_balance,
                "total_investment": total_investment,
                "current_value": current_value,
                "overall_profit": overall_profit,
                "overall_profit_percentage": overall_profit_percentage,
                "active_trades": len(self.active_trades),
                "total_trades": len(self.trade_history),
                "daily_loss": self.daily_loss,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio status: {str(e)}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }

if __name__ == "__main__":
    # Configure logging
    logger.add(
        settings.DEBUG_LOG_FILE,
        rotation=settings.MAX_LOG_SIZE,
        retention=settings.MAX_LOG_FILES
    )
    
    # Create and run bot
    bot = MemeCoinBot()
    asyncio.run(bot.start()) 
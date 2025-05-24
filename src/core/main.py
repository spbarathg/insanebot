"""
Main bot implementation for Solana trading bot.
"""
from dotenv import load_dotenv
load_dotenv()
import asyncio
import json
import logging
import time
import random
import math
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import settings
from src.core.wallet_manager import WalletManager
from src.core.local_llm import LocalLLM
from src.core.helius_service import HeliusService
from src.core.jupiter_service import JupiterService

# Import arbitrage scanner
from .arbitrage import CrossDEXScanner

# Configure logging
logger.add("logs/portfolio.log", rotation="5 MB", level="INFO", 
          filter=lambda record: "PORTFOLIO" in record.get("extra", {}))
logger.add("logs/scanner.log", rotation="5 MB", level="DEBUG", 
          filter=lambda record: "SCANNER" in record.get("extra", {}))
logger.add("logs/trades.log", rotation="5 MB", level="INFO", 
          filter=lambda record: "TRADE" in record.get("extra", {}))

# Ensure required directories exist
for directory in ['logs', 'data', 'models']:
    os.makedirs(directory, exist_ok=True)

class PortfolioManager:
    """Manages the trading portfolio and tracks performance"""
    
    def __init__(self):
        """Initialize portfolio manager"""
        self.holdings = {}
        self.trade_history = []
        self.starting_balance = 0
        self.current_balance = 0
        self.total_profit_loss = 0
        self.total_trades = 0
        self.successful_trades = 0
        self.realized_profit = 0
        self.daily_stats = {}
        self.last_updated = time.time()
        
    async def initialize(self, initial_balance: float) -> bool:
        """Initialize with starting balance"""
        self.starting_balance = initial_balance
        self.current_balance = initial_balance
        self.last_updated = time.time()
        logger.info(f"Portfolio initialized with {initial_balance} SOL")
        return True
        
    def add_trade(self, trade_data: Dict) -> None:
        """Add a trade to history and update portfolio"""
        self.trade_history.append(trade_data)
        self.total_trades += 1
        
        # Update trade date in daily stats
        trade_date = datetime.fromtimestamp(trade_data["timestamp"]).strftime("%Y-%m-%d")
        if trade_date not in self.daily_stats:
            self.daily_stats[trade_date] = {
                "trades": 0,
                "volume": 0,
                "profit_loss": 0
            }
        
        self.daily_stats[trade_date]["trades"] += 1
        
        # Handle buy trades
        if trade_data["action"] == "buy":
            token = trade_data["token_address"]
            amount_sol = trade_data["amount_sol"]
            price_usd = trade_data["price_usd"]
            
            # Calculate token amount based on SOL spent
            token_amount = amount_sol * trade_data.get("sol_price", 100) / price_usd
            
            # Update holdings
            if token not in self.holdings:
                self.holdings[token] = {
                    "amount": token_amount,
                    "cost_basis": amount_sol,
                    "avg_price_usd": price_usd,
                    "last_price_usd": price_usd
                }
            else:
                # Average down/up the cost basis
                total_amount = self.holdings[token]["amount"] + token_amount
                total_cost = self.holdings[token]["cost_basis"] + amount_sol
                
                self.holdings[token]["amount"] = total_amount
                self.holdings[token]["cost_basis"] = total_cost
                self.holdings[token]["avg_price_usd"] = total_cost * trade_data.get("sol_price", 100) / total_amount
                self.holdings[token]["last_price_usd"] = price_usd
            
            # Update current balance
            self.current_balance -= amount_sol
            self.daily_stats[trade_date]["volume"] += amount_sol
        
        # Handle sell trades
        elif trade_data["action"] == "sell":
            token = trade_data["token_address"]
            amount_sol = trade_data["amount_sol"]
            price_usd = trade_data["price_usd"]
            
            # Update balance
            self.current_balance += amount_sol
            self.daily_stats[trade_date]["volume"] += amount_sol
            
            # Calculate profit/loss if we have this token
            if token in self.holdings and self.holdings[token]["amount"] > 0:
                # Calculate token amount based on SOL received
                token_amount = amount_sol * trade_data.get("sol_price", 100) / price_usd
                
                # Calculate profit/loss
                token_cost_basis_per_unit = self.holdings[token]["cost_basis"] / self.holdings[token]["amount"]
                cost_of_sold = token_cost_basis_per_unit * token_amount
                profit_loss = amount_sol - cost_of_sold
                
                # Update stats
                self.realized_profit += profit_loss
                self.total_profit_loss += profit_loss
                self.daily_stats[trade_date]["profit_loss"] += profit_loss
                
                if profit_loss > 0:
                    self.successful_trades += 1
                
                # Update holdings
                if token_amount >= self.holdings[token]["amount"]:
                    # Sold all tokens
                    del self.holdings[token]
                else:
                    # Sold partial position
                    remaining_amount = self.holdings[token]["amount"] - token_amount
                    remaining_cost = self.holdings[token]["cost_basis"] - cost_of_sold
                    
                    self.holdings[token]["amount"] = remaining_amount
                    self.holdings[token]["cost_basis"] = remaining_cost
                    self.holdings[token]["last_price_usd"] = price_usd
                
                # Log success
                logger.info(f"Trade profit/loss: {profit_loss:.6f} SOL")
                trade_data["profit_loss"] = profit_loss
                
                if profit_loss > 0:
                    logger.info(f"Successful trade: {token_amount:.2f} {trade_data['token']} sold with {profit_loss:.6f} SOL profit")
            
        self.last_updated = time.time()
    
    def update_prices(self, token_prices: Dict[str, float]) -> None:
        """Update current prices of holdings"""
        for token, price in token_prices.items():
            if token in self.holdings:
                self.holdings[token]["last_price_usd"] = price
        
        self.last_updated = time.time()
    
    def get_portfolio_value(self, sol_price: float = 100.0) -> float:
        """Calculate total portfolio value in SOL"""
        holdings_value = 0
        
        for token, data in self.holdings.items():
            token_value_usd = data["amount"] * data["last_price_usd"]
            token_value_sol = token_value_usd / sol_price
            holdings_value += token_value_sol
        
        return self.current_balance + holdings_value
    
    def get_portfolio_summary(self) -> Dict:
        """Get summary of portfolio performance"""
        current_value = self.get_portfolio_value()
        total_return = current_value - self.starting_balance
        percent_return = (total_return / self.starting_balance) * 100 if self.starting_balance > 0 else 0
        
        # Calculate win rate
        win_rate = (self.successful_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        
        # Calculate daily performance
        daily_performance = []
        for date, stats in sorted(self.daily_stats.items()):
            daily_performance.append({
                "date": date,
                "trades": stats["trades"],
                "volume": stats["volume"],
                "profit_loss": stats["profit_loss"],
                "profit_percentage": (stats["profit_loss"] / stats["volume"]) * 100 if stats["volume"] > 0 else 0
            })
        
        return {
            "current_value": current_value,
            "initial_investment": self.starting_balance,
            "total_return": total_return,
            "percent_return": percent_return,
            "realized_profit": self.realized_profit,
            "unrealized_profit": total_return - self.realized_profit,
            "total_trades": self.total_trades,
            "successful_trades": self.successful_trades,
            "win_rate": win_rate,
            "holdings_count": len(self.holdings),
            "last_updated": self.last_updated,
            "daily_performance": daily_performance
        }
    
    def get_holdings(self) -> List[Dict]:
        """Get detailed holdings information"""
        holdings_list = []
        
        for token_address, data in self.holdings.items():
            current_value_usd = data["amount"] * data["last_price_usd"]
            cost_basis_usd = data["cost_basis"] * 100  # Approximate conversion
            unrealized_pl = current_value_usd - cost_basis_usd
            percent_change = (data["last_price_usd"] / data["avg_price_usd"] - 1) * 100
            
            holdings_list.append({
                "token_address": token_address,
                "amount": data["amount"],
                "current_price_usd": data["last_price_usd"],
                "avg_price_usd": data["avg_price_usd"],
                "cost_basis_sol": data["cost_basis"],
                "current_value_usd": current_value_usd,
                "current_value_sol": current_value_usd / 100,  # Approximate conversion
                "unrealized_pl_usd": unrealized_pl,
                "unrealized_pl_sol": unrealized_pl / 100,  # Approximate conversion
                "percent_change": percent_change
            })
        
        # Sort by value (descending)
        holdings_list.sort(key=lambda x: x["current_value_usd"], reverse=True)
        
        return holdings_list

class MarketScanner:
    """Scans for trading opportunities and new tokens"""
    
    def __init__(self, helius_service: HeliusService, jupiter_service: JupiterService):
        """Initialize market scanner"""
        self.helius_service = helius_service
        self.jupiter_service = jupiter_service
        self.token_watchlist = []
        self.new_token_candidates = []
        self.last_scan = 0
        self.scan_interval = 300  # 5 minutes between scans for more frequent discovery (was 600)
        self.max_watchlist_size = 25  # Allow many more tokens on watchlist (was 15)
        self.max_candidates = 50  # Track many more candidates (was 25)
    
    async def initialize(self) -> bool:
        """Initialize the market scanner"""
        # Add base tokens to watchlist
        self.token_watchlist = [
            "So11111111111111111111111111111111111111112",  # SOL
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # BONK
            "JTO9c5fHf2xHjdJwEiXBXJ4DFXm7nDY7ix6Esw4qGAiA",  # JTO
            "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3",  # PYTH
        ]
        logger.info(f"Market scanner initialized with {len(self.token_watchlist)} tokens")
        return True
    
    async def scan_for_new_tokens(self) -> List[str]:
        """Scan for new promising tokens to add to watchlist"""
        now = time.time()
        
        # Only scan periodically
        if now - self.last_scan < self.scan_interval:
            return []
        
        self.last_scan = now
        logger.info("Scanning for new trading opportunities...")
        
        # Get real tokens from Jupiter API
        try:
            # Get random tokens from Jupiter
            random_tokens = await self.jupiter_service.get_random_tokens(count=5)
            
            new_tokens = []
            for token_data in random_tokens:
                token_address = token_data.get('address')
                if not token_address:
                    continue
                    
                # Skip if already in watchlist or candidates
                if token_address in self.token_watchlist or token_address in self.new_token_candidates:
                    continue
                
                # Skip common stable coins and major tokens
                symbol = token_data.get('symbol', '').upper()
                if symbol in {'USDC', 'USDT', 'SOL', 'BTC', 'ETH', 'WBTC', 'WETH'}:
                    continue
                    
                new_tokens.append(token_address)
                self.new_token_candidates.append(token_address)
                
                logger.info(f"Found new token candidate: {symbol} ({token_address[:8]}...)")
                
            if new_tokens:
                logger.info(f"Found {len(new_tokens)} new potential tokens")
            else:
                logger.info("No new tokens found in this scan")
                
            return new_tokens
            
        except Exception as e:
            logger.error(f"Error scanning for new tokens: {str(e)}")
            return []
    
    async def evaluate_token(self, token_address: str) -> Dict:
        """Evaluate if a token is worth adding to the main watchlist"""
        try:
            # Get token metadata
            metadata = await self.helius_service.get_token_metadata(token_address)
            if not metadata:
                return {"score": 0, "reason": "Failed to get metadata"}
                
            # Get liquidity data
            liquidity_data = await self.helius_service.get_token_liquidity(token_address)
            if not liquidity_data:
                return {"score": 0, "reason": "Failed to get liquidity data"}
                
            # Get holders data
            holders_data = await self.helius_service.get_token_holders(token_address, limit=10)
            if not holders_data:
                return {"score": 0, "reason": "Failed to get holders data"}
                
            # Extract key metrics
            liquidity = liquidity_data.get("liquidity", 0)
            holders_count = metadata.get("holders", 0)
            volume = metadata.get("volumeUsd24h", 0)
            
            # Debug logging for discovered tokens
            logger.debug(f"Evaluating {metadata.get('symbol', 'UNKNOWN')}: Liquidity=${liquidity:.0f}, Holders={holders_count}, Volume=${volume:.0f}")
            
            # Check minimum criteria - Much lower thresholds for better token discovery
            if liquidity < 1000:  # $1K minimum liquidity (was $10K)
                return {"score": 0.1, "reason": "Insufficient liquidity"}
                
            if holders_count < 10:  # Much lower holder count (was 50)
                return {"score": 0.2, "reason": "Too few holders"}
                
            if volume < 100:  # $100 minimum daily volume (was $1K)
                return {"score": 0.3, "reason": "Low trading volume"}
                
            # Calculate concentration score (lower is better)
            top_holder_percentage = holders_data[0]["percentage"] if holders_data else 0.5
            concentration_score = 1 - top_holder_percentage
            
            # Calculate final score (0 to 1) - More generous scoring
            score = (
                min(liquidity / 100000, 1) * 0.4 +  # Liquidity score (reduced from 1M to 100K)
                min(volume / 10000, 1) * 0.3 +  # Volume score (reduced from 100K to 10K)
                min(holders_count / 100, 1) * 0.2 +  # Holders score (reduced from 1000 to 100)
                concentration_score * 0.1  # Concentration score
            )
            
            return {
                "token_address": token_address,
                "name": metadata.get("name", "Unknown"),
                "symbol": metadata.get("symbol", "UNKNOWN"),
                "score": score,
                "liquidity": liquidity,
                "volume": volume,
                "holders": holders_count,
                "top_holder_percentage": top_holder_percentage,
                "evaluation_time": time.time()
            }
                
        except Exception as e:
            logger.error(f"Error evaluating token {token_address}: {str(e)}")
            return {"score": 0, "reason": f"Evaluation error: {str(e)}"}
    
    async def update_watchlist(self) -> None:
        """Update watchlist with promising new tokens"""
        # First scan for new tokens
        new_tokens = await self.scan_for_new_tokens()
        
        # Evaluate new token candidates
        for token_address in list(self.new_token_candidates):
            evaluation = await self.evaluate_token(token_address)
            
            # Much lower score threshold to add more tokens (from 0.25 to 0.15)
            if evaluation.get("score", 0) > 0.15:
                if token_address not in self.token_watchlist:
                    # Add to watchlist if there's room
                    if len(self.token_watchlist) < self.max_watchlist_size:
                        self.token_watchlist.append(token_address)
                        logger.info(f"✅ Added {evaluation.get('symbol', 'Unknown')} to watchlist with score {evaluation.get('score', 0):.2f}")
                    else:
                        # Remove a less promising token if watchlist is full
                        # For now, remove the last token (oldest)
                        removed_token = self.token_watchlist.pop()
                        self.token_watchlist.append(token_address)
                        logger.info(f"✅ Added {evaluation.get('symbol', 'Unknown')} to watchlist (replaced older token)")
                
                # Remove from candidates
                if token_address in self.new_token_candidates:
                    self.new_token_candidates.remove(token_address)
            elif evaluation.get("score", 0) < 0.05:
                # Remove tokens with extremely low scores immediately (reduced from 0.2 to 0.05)
                if token_address in self.new_token_candidates:
                    self.new_token_candidates.remove(token_address)
                    logger.debug(f"❌ Removed extremely low-scoring token {token_address[:8]}... (score: {evaluation.get('score', 0):.2f})")
            elif time.time() - evaluation.get("evaluation_time", 0) > 86400:
                # Remove old candidates after 24 hours
                if token_address in self.new_token_candidates:
                    self.new_token_candidates.remove(token_address)
                    
        # Limit the number of candidates we track
        if len(self.new_token_candidates) > self.max_candidates:
            # Remove oldest candidates
            self.new_token_candidates = self.new_token_candidates[-self.max_candidates:]
    
    def get_current_watchlist(self) -> List[str]:
        """Get the current token watchlist."""
        return self.token_watchlist.copy()

class MemeCoinBot:
    """
    Main Solana trading bot implementation that combines market data,
    trade execution, local LLM for trading decisions, and monitoring.
    """
    
    def __init__(self):
        """Initialize the trading bot"""
        self.helius_service = None
        self.jupiter_service = None
        self.local_llm = None
        self.portfolio = None
        self.market_scanner = None
        self.arbitrage_scanner = None  # Add arbitrage scanner
        
        # Trading state
        self.running = False
        self.active_trades = 0
        self.max_concurrent_trades = 5
        self.min_trade_interval = 300  # 5 minutes between trades on same token
        self.trade_cooldowns = {}  # Track when we last traded each token
        self.market_data_cache = {}  # Cache market data for tokens
        
        # Performance tracking
        self.total_scans = 0
        self.opportunities_analyzed = 0
        self.trades_executed = 0
        self.arbitrage_opportunities_found = 0  # Track arbitrage opportunities
        
        logger.info("MemeCoinBot initialized")
        
    async def initialize(self) -> bool:
        """Initialize all bot components with proper error handling."""
        try:
            logger.info("🚀 Initializing MemeCoinBot...")
            
            # Initialize core services
            self.helius_service = HeliusService()
            self.jupiter_service = JupiterService() 
            self.local_llm = LocalLLM()
            self.portfolio = PortfolioManager()
            
            # Initialize services
            logger.info("🔧 Initializing Helius service...")
            if not await self.helius_service.initialize():
                logger.error("❌ Failed to initialize Helius service")
                return False
            
            logger.info("🔧 Initializing Jupiter service...")
            if not await self.jupiter_service.initialize():
                logger.error("❌ Failed to initialize Jupiter service")
                return False
            
            logger.info("🔧 Initializing Local LLM...")
            if not await self.local_llm.initialize():
                logger.error("❌ Failed to initialize Local LLM")
                return False
            
            logger.info("🔧 Initializing Portfolio Manager...")
            starting_balance = 0.1  # 0.1 SOL for simulation
            if not await self.portfolio.initialize(starting_balance):
                logger.error("❌ Failed to initialize Portfolio Manager")
                return False
            
            # Initialize Market Scanner
            logger.info("🔧 Initializing Market Scanner...")
            self.market_scanner = MarketScanner(self.helius_service, self.jupiter_service)
            
            # Initialize Arbitrage Scanner
            logger.info("🔧 Initializing Cross-DEX Arbitrage Scanner...")
            self.arbitrage_scanner = CrossDEXScanner(self.jupiter_service, self.helius_service)
            if not await self.arbitrage_scanner.initialize():
                logger.error("❌ Failed to initialize Arbitrage Scanner")
                return False
            
            logger.success("✅ All services initialized successfully!")
            logger.info("MemeCoinBot initialization complete")
            
            return True
            
        except Exception as e:
            logger.error(f"💥 Failed to initialize MemeCoinBot: {str(e)}")
            logger.exception("Full initialization error:")
            return False
            
    async def start(self):
        """Start the bot trading loop."""
        logger.bind(ACTIVITY="BOT_START").debug("🚀 MemeCoinBot.start() called")
        
        if not all([self.helius_service, self.jupiter_service, self.local_llm, self.portfolio, self.arbitrage_scanner]):
            logger.bind(ACTIVITY="BOT_START").error("❌ Bot not initialized. Call initialize() first.")
            return
        
        self.running = True
        logger.bind(ACTIVITY="BOT_START").success("🎯 MemeCoinBot trading loop started!")
        
        try:
            # Main trading loop
            while self.running:
                loop_start_time = time.time()
                
                try:
                    # Phase 1: Update watchlist and scan for new tokens
                    logger.bind(SCANNER=True).debug("🔄 Updating token watchlist...")
                    await self.market_scanner.update_watchlist()
                    
                    # Phase 2: Scan for arbitrage opportunities
                    logger.bind(ARBITRAGE=True).debug("🔍 Scanning for arbitrage opportunities...")
                    arbitrage_opportunities = await self.arbitrage_scanner.scan_arbitrage_opportunities()
                    
                    if arbitrage_opportunities:
                        self.arbitrage_opportunities_found += len(arbitrage_opportunities)
                        logger.bind(ARBITRAGE=True).info(f"💰 Found {len(arbitrage_opportunities)} arbitrage opportunities!")
                        
                        # Execute the most profitable arbitrage opportunity
                        best_opportunity = max(arbitrage_opportunities, key=lambda x: x.profit_percentage)
                        if best_opportunity.profit_percentage > 0.5:  # Only execute if >0.5% profit
                            logger.bind(ARBITRAGE=True).info(
                                f"⚡ Executing best arbitrage: {best_opportunity.token_symbol} "
                                f"({best_opportunity.profit_percentage:.2f}% profit)"
                            )
                            # For now, just log - execution disabled by default for safety
                            # result = await self.arbitrage_scanner.execute_arbitrage(best_opportunity)
                    
                    # Phase 3: Regular token analysis and trading
                    current_watchlist = self.market_scanner.get_current_watchlist()
                    logger.bind(SCANNER=True).info(f"📊 Analyzing {len(current_watchlist)} tokens on watchlist")
                    
                    # Analyze each token on the watchlist
                    for token_address in current_watchlist:
                        if not self.running:
                            break
                        
                        self.total_scans += 1
                        await self.check_token_opportunity(token_address)
                    
                    # Phase 4: Display summary statistics
                    await self._log_performance_summary(arbitrage_opportunities)
                    
                except Exception as e:
                    logger.bind(ACTIVITY="BOT_LOOP").error(f"💥 Error in main loop: {str(e)}")
                    logger.bind(ACTIVITY="BOT_LOOP").exception("Full loop error traceback:")
                
                # Rate limiting - ensure we don't run too frequently
                loop_duration = time.time() - loop_start_time
                min_loop_time = 30  # Minimum 30 seconds between loops
                if loop_duration < min_loop_time:
                    sleep_time = min_loop_time - loop_duration
                    logger.bind(SCANNER=True).debug(f"⏱️ Sleeping for {sleep_time:.1f} seconds...")
                    await asyncio.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.bind(ACTIVITY="BOT_STOP").info("🛑 Received stop signal")
        except Exception as e:
            logger.bind(ACTIVITY="BOT_STOP").error(f"💥 Critical error in trading loop: {str(e)}")
            logger.bind(ACTIVITY="BOT_STOP").exception("Critical error traceback:")
        finally:
            await self.stop()
    
    async def _log_performance_summary(self, arbitrage_opportunities: List = None):
        """Log performance summary including arbitrage statistics."""
        try:
            # Portfolio summary
            portfolio_summary = self.portfolio.get_portfolio_summary()
            
            # Arbitrage scanner stats
            arbitrage_stats = self.arbitrage_scanner.get_scanner_stats() if self.arbitrage_scanner else {}
            
            logger.bind(SUMMARY=True).info(
                f"📈 Portfolio: {portfolio_summary['current_value']:.3f} SOL "
                f"({portfolio_summary['percent_return']:+.1f}%) | "
                f"Trades: {portfolio_summary['total_trades']} "
                f"({portfolio_summary['win_rate']:.1f}% win rate)"
            )
            
            if arbitrage_stats:
                logger.bind(ARBITRAGE=True).info(
                    f"🔄 Arbitrage: {arbitrage_stats.get('opportunities_found', 0)} opportunities found, "
                    f"{arbitrage_stats.get('successful_executions', 0)} executed, "
                    f"{arbitrage_stats.get('total_profit_made', 0):.4f} SOL profit"
                )
            
            # Current arbitrage opportunities
            if arbitrage_opportunities:
                for opp in arbitrage_opportunities[:3]:  # Show top 3
                    logger.bind(ARBITRAGE=True).info(
                        f"💎 {opp.token_symbol}: {opp.profit_percentage:.2f}% profit "
                        f"({opp.buy_dex.value}→{opp.sell_dex.value}) "
                        f"Risk: {opp.risk_level} | Time left: {opp.time_remaining:.0f}s"
                    )
                    
        except Exception as e:
            logger.error(f"Error logging performance summary: {str(e)}")
    
    async def check_token_opportunity(self, token_address: str):
        """Check a specific token for trading opportunities."""
        try:
            # Check cooldown period
            now = time.time()
            if token_address in self.trade_cooldowns:
                time_since_last_trade = now - self.trade_cooldowns[token_address]
                if time_since_last_trade < self.min_trade_interval:
                    logger.bind(SCANNER=True).debug(f"⏸️ {token_address[:8]}... still in cooldown ({self.min_trade_interval - time_since_last_trade:.0f}s remaining)")
                    return
            
            # Skip if we're at max concurrent trades
            if self.active_trades >= self.max_concurrent_trades:
                logger.bind(SCANNER=True).debug(f"⏸️ Max concurrent trades reached ({self.active_trades}/{self.max_concurrent_trades})")
                return
            
            # Get token metadata
            logger.bind(SCANNER=True).debug(f"📡 Getting metadata for token {token_address[:8]}...")
            metadata = await self.helius_service.get_token_metadata(token_address)
            if not metadata:
                logger.bind(SCANNER=True).warning(f"❌ No metadata for token {token_address[:8]}...")
                return
            
            # Get price information
            logger.bind(SCANNER=True).debug(f"💰 Getting price for {metadata.get('symbol', 'UNKNOWN')}...")
            price_info = await self.helius_service.get_token_price(token_address)
            if not price_info:
                logger.bind(SCANNER=True).warning(f"❌ No price data for token {token_address[:8]}...")
                return
                
            # Get liquidity data
            liquidity_data = await self.helius_service.get_token_liquidity(token_address)
            if not liquidity_data:
                logger.bind(SCANNER=True).warning(f"❌ No liquidity data for token {token_address[:8]}...")
                return
                
            # Combine data for analysis
            token_data = {
                "address": token_address,
                "name": metadata.get("name", "Unknown"),
                "symbol": metadata.get("symbol", "UNKNOWN"),
                "price_usd": price_info.get("price", 0),
                "price_sol": price_info.get("pricePerSol", 0),
                "liquidity_usd": liquidity_data.get("liquidity", 0),
                "liquidity_sol": liquidity_data.get("liquidity_sol", 0),
                "volumeUsd24h": metadata.get("volumeUsd24h", 0),
                "holders": metadata.get("holders", 0),
                "market_cap": metadata.get("market_cap", 0),
                "price_history": price_info.get("price_history", []),
                "timestamp": now
            }
            
            logger.bind(SCANNER=True).debug(f"✅ Data collected for {metadata.get('symbol', 'UNKNOWN')} ({token_address[:8]}...)")
            
            # Cache the market data
            self.market_data_cache[token_address] = token_data
            
            # Update token prices in portfolio
            token_prices = {token_address: token_data["price_usd"]}
            self.portfolio.update_prices(token_prices)
            
            # Get trading recommendation from LLM
            logger.bind(SCANNER=True).debug(f"🤖 Getting LLM analysis for {metadata.get('symbol', 'UNKNOWN')}...")
            recommendation = await self.local_llm.analyze_market(token_data)
            
            if not recommendation:
                logger.bind(SCANNER=True).warning(f"❌ No recommendation for {metadata.get('symbol', 'UNKNOWN')}")
                return
            
            action = recommendation.get("action", "hold").lower()
            confidence = recommendation.get("confidence", 0)
            reasoning = recommendation.get("reasoning", "No reasoning provided")
            
            # Log the analysis result
            action_emoji = "🟢" if action == "buy" else "🔴" if action == "sell" else "🟡"
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            
            logger.bind(ANALYSIS=True).info(
                f"{action_emoji} {metadata.get('symbol', 'UNKNOWN')} | "
                f"{action.upper()} | "
                f"Confidence: {confidence:.2f} [{confidence_bar}] | "
                f"Price: ${token_data['price_usd']:.6f}"
            )
            
            logger.bind(ANALYSIS=True).debug(f"💭 Reasoning: {reasoning}")
            
            # Determine if we should trade
            min_confidence = 0.6 if action in ["buy", "sell"] else 0.5
            should_trade = (
                action in ["buy", "sell"] and
                confidence >= min_confidence and
                self.active_trades < self.max_concurrent_trades
            )
            
            # Execute trade if conditions are met
            if should_trade:
                logger.bind(TRADE=True).info(f"🚀 Executing {action.upper()} trade for {metadata.get('symbol', 'UNKNOWN')}")
                await self.execute_trade(token_data, action, recommendation.get("position_size", 0.01), recommendation.get("reasoning", ""))
                # Set cooldown for this token
                self.trade_cooldowns[token_address] = now
                logger.bind(SCANNER=True).info(f"⏰ Set cooldown for {metadata.get('symbol', 'UNKNOWN')} until {datetime.fromtimestamp(now + self.min_trade_interval).strftime('%H:%M:%S')}")
                
        except Exception as e:
            logger.bind(SCANNER=True).error(f"💥 Error checking token {token_address[:8]}...: {str(e)}")
            logger.bind(SCANNER=True).exception("Full error traceback:")
            
    async def execute_trade(self, token_data: Dict, action: str, position_size: float, reasoning: str):
        """Execute a trade (simulated)."""
        try:
            token_symbol = token_data.get("symbol", "UNKNOWN")
            token_address = token_data.get("address", "")
            
            # Calculate trade details
            price = token_data.get("price_usd", 0)
            sol_price = 100.0  # Approximate SOL price in USD
            amount = position_size  # In SOL
            
            if action == "buy":
                logger.info(f"Executing BUY: {amount} SOL of {token_symbol} at ${price}")
                logger.info(f"Reasoning: {reasoning}")
                
                # Simulate a successful buy
                trade_data = {
                    "action": "buy",
                    "token": token_symbol,
                    "token_address": token_address,
                    "amount_sol": amount,
                    "price_usd": price,
                    "sol_price": sol_price,
                    "timestamp": time.time(),
                    "status": "success",
                    "transaction_id": f"SIM_BUY_{int(time.time())}"
                }
                
                # Update portfolio
                self.portfolio.add_trade(trade_data)
                self.active_trades += 1
                
            elif action == "sell":
                logger.info(f"Executing SELL: {amount} SOL of {token_symbol} at ${price}")
                logger.info(f"Reasoning: {reasoning}")
                
                # Simulate a successful sell
                trade_data = {
                    "action": "sell",
                    "token": token_symbol,
                    "token_address": token_address,
                    "amount_sol": amount,
                    "price_usd": price,
                    "sol_price": sol_price,
                    "timestamp": time.time(),
                    "status": "success",
                    "transaction_id": f"SIM_SELL_{int(time.time())}"
                }
                
                # Update portfolio
                self.portfolio.add_trade(trade_data)
                self.active_trades = max(0, self.active_trades - 1)
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            
    async def stop(self):
        """Stop the bot and cleanup resources."""
        logger.bind(ACTIVITY="BOT_STOP").info("🛑 Stopping MemeCoinBot...")
        
        self.running = False
        
        try:
            # Close all services
            if self.arbitrage_scanner:
                await self.arbitrage_scanner.close()
                
            if self.helius_service:
                await self.helius_service.close()
                
            if self.jupiter_service:
                await self.jupiter_service.close()
                
            if self.local_llm:
                await self.local_llm.close()
                
            logger.bind(ACTIVITY="BOT_STOP").success("✅ MemeCoinBot stopped successfully")
            
        except Exception as e:
            logger.bind(ACTIVITY="BOT_STOP").error(f"💥 Error during shutdown: {str(e)}")
            logger.bind(ACTIVITY="BOT_STOP").exception("Shutdown error traceback:")

if __name__ == "__main__":
    # Create and run bot
    bot = MemeCoinBot()
    
    try:
        # Initialize bot
        if asyncio.run(bot.initialize()):
            # Start bot
            asyncio.run(bot.start())
        else:
            logger.error("Failed to initialize bot")
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        # Ensure proper cleanup
        asyncio.run(bot.stop()) 
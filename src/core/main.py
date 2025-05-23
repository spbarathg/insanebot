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
        self.scan_interval = 600  # 10 minutes between scans for more frequent discovery
        self.max_watchlist_size = 15  # Allow more tokens on watchlist
        self.max_candidates = 25  # Track more candidates
    
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
            
            # Check minimum criteria - Lower thresholds for better token discovery
            if liquidity < 10000:  # $10K minimum liquidity (was $50K)
                return {"score": 0.1, "reason": "Insufficient liquidity"}
                
            if holders_count < 50:  # Minimum holder count (was 100)
                return {"score": 0.2, "reason": "Too few holders"}
                
            if volume < 1000:  # $1K minimum daily volume (was $10K)
                return {"score": 0.3, "reason": "Low trading volume"}
                
            # Calculate concentration score (lower is better)
            top_holder_percentage = holders_data[0]["percentage"] if holders_data else 0.5
            concentration_score = 1 - top_holder_percentage
            
            # Calculate final score (0 to 1)
            score = (
                min(liquidity / 1000000, 1) * 0.4 +  # Liquidity score
                min(volume / 100000, 1) * 0.3 +  # Volume score
                min(holders_count / 1000, 1) * 0.2 +  # Holders score
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
            
            # Lower the score threshold to add more tokens (from 0.4 to 0.25)
            if evaluation.get("score", 0) > 0.25:
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
            elif evaluation.get("score", 0) < 0.2:
                # Remove tokens with very low scores immediately
                if token_address in self.new_token_candidates:
                    self.new_token_candidates.remove(token_address)
                    logger.debug(f"❌ Removed low-scoring token {token_address[:8]}... (score: {evaluation.get('score', 0):.2f})")
            elif time.time() - evaluation.get("evaluation_time", 0) > 86400:
                # Remove old candidates after 24 hours
                if token_address in self.new_token_candidates:
                    self.new_token_candidates.remove(token_address)
                    
        # Limit the number of candidates we track
        if len(self.new_token_candidates) > self.max_candidates:
            # Remove oldest candidates
            self.new_token_candidates = self.new_token_candidates[-self.max_candidates:]

class MemeCoinBot:
    """
    Main Solana trading bot implementation that combines market data,
    trade execution, local LLM for trading decisions, and monitoring.
    """
    
    def __init__(self):
        """Initialize bot components."""
        self.wallet_manager = WalletManager()
        self.local_llm = LocalLLM()
        self.helius_service = HeliusService()
        self.jupiter_service = JupiterService()
        self.running = False
        self.portfolio = PortfolioManager()
        self.market_scanner = None
        self.last_check = {}
        self.trade_cooldowns = {}
        self.min_trade_interval = 600  # 10 minutes between trades for the same token
        self.market_data_cache = {}
        self.max_concurrent_trades = 5
        self.active_trades = 0
        self.risk_limit = 0.1  # Maximum percentage of portfolio to risk at once
        self.sentiment_threshold = 0.3  # Minimum sentiment score to consider buying
        self.initialized = False
        
    async def initialize(self) -> bool:
        """Initialize all bot components with proper error handling."""
        if self.initialized:
            logger.warning("Bot already initialized")
            return True
            
        logger.debug("MemeCoinBot.initialize() called")
        try:
            logger.info("Initializing MemeCoinBot components...")
            
            # Check environment variables
            required_env_vars = ['HELIUS_API_KEY', 'JUPITER_API_KEY']
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
            # Initialize services with retries
            max_retries = 3
            for service_name, service in [
                ("Wallet Manager", self.wallet_manager),
                ("Local LLM", self.local_llm),
                ("Helius Service", self.helius_service),
                ("Jupiter Service", self.jupiter_service)
            ]:
                for attempt in range(max_retries):
                    try:
                        if await service.initialize():
                            logger.info(f"{service_name} initialized successfully")
                            break
                        else:
                            raise Exception(f"{service_name} initialization failed")
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise Exception(f"Failed to initialize {service_name} after {max_retries} attempts: {str(e)}")
                        logger.warning(f"Retrying {service_name} initialization (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(1)
            
            # Initialize portfolio with wallet balance
            balance = await self.wallet_manager.check_balance()
            if balance is None:
                raise ValueError("Failed to get wallet balance")
            await self.portfolio.initialize(balance)
            
            # Initialize market scanner
            self.market_scanner = MarketScanner(self.helius_service, self.jupiter_service)
            if not await self.market_scanner.initialize():
                raise ValueError("Failed to initialize market scanner")
            
            logger.info(f"Monitoring {len(self.market_scanner.token_watchlist)} tokens")
            logger.info("MemeCoinBot initialization complete")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MemeCoinBot: {str(e)}")
            await self.close()  # Clean up any partially initialized services
            return False
            
    async def start(self):
        """Start the bot trading loop."""
        logger.bind(ACTIVITY="BOT_START").debug("🚀 MemeCoinBot.start() called")
        
        if not self.initialized:
            logger.bind(ACTIVITY="BOT_START").error("❌ Bot not initialized. Call initialize() first.")
            return
            
        if self.running:
            logger.bind(ACTIVITY="BOT_START").warning("⚠️ Bot already running")
            return
            
        self.running = True
        logger.bind(ACTIVITY="BOT_START").success("✅ MemeCoinBot started - Entering main trading loop")
        
        try:
            loop_count = 0
            while self.running:
                try:
                    loop_count += 1
                    logger.bind(ACTIVITY="MAIN_LOOP").info(f"🔄 Starting trading cycle #{loop_count}")
                    
                    # Update market scanner watchlist
                    logger.bind(SCANNER=True).debug("📡 Updating market scanner watchlist...")
                    await self.market_scanner.update_watchlist()
                    watchlist_size = len(self.market_scanner.token_watchlist)
                    logger.bind(SCANNER=True).info(f"👁️ Monitoring {watchlist_size} tokens on watchlist")
                    
                    # Check all monitored tokens
                    tokens_checked = 0
                    for token in self.market_scanner.token_watchlist:
                        logger.bind(SCANNER=True).debug(f"🔍 Checking token: {token[:8]}...")
                        await self.check_token(token)
                        tokens_checked += 1
                    
                    logger.bind(SCANNER=True).info(f"✅ Completed analysis of {tokens_checked} tokens")
                    
                    # Get portfolio summary
                    summary = self.portfolio.get_portfolio_summary()
                    holdings = self.portfolio.get_holdings()
                    
                    # Log portfolio status every few iterations
                    if loop_count % 5 == 0 or random.random() < 0.2:  # Every 5 cycles or ~20% chance
                        logger.bind(PORTFOLIO=True).info(f"💰 Portfolio value: {summary['current_value']:.4f} SOL")
                        logger.bind(PORTFOLIO=True).info(f"📈 Profit: {summary['total_return']:.4f} SOL ({summary['percent_return']:.2f}%)")
                        logger.bind(PORTFOLIO=True).info(f"🎯 Win rate: {summary['win_rate']:.1f}% ({summary['successful_trades']} of {summary['total_trades']} trades)")
                        logger.bind(PORTFOLIO=True).info(f"🏃 Active trades: {self.active_trades}/{self.max_concurrent_trades}")
                        
                        if holdings:
                            logger.bind(PORTFOLIO=True).info(f"🏷️ Holding {len(holdings)} different tokens")
                            top_holdings = holdings[:3] if len(holdings) > 3 else holdings
                            for holding in top_holdings:
                                profit_emoji = "📈" if holding['percent_change'] > 0 else "📉" if holding['percent_change'] < 0 else "➡️"
                                logger.bind(PORTFOLIO=True).info(f"{profit_emoji} {holding['symbol']}: {holding['amount']:.2f} @ ${holding['current_price_usd']:.6f} = {holding['current_value_sol']:.4f} SOL ({holding['percent_change']:+.2f}%)")
                        else:
                            logger.bind(PORTFOLIO=True).info("📭 No current holdings")
                    
                    # Log cycle completion
                    logger.bind(ACTIVITY="MAIN_LOOP").debug(f"⏰ Trading cycle #{loop_count} completed, waiting 30s...")
                    
                    # Sleep between cycles
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    logger.bind(ACTIVITY="SHUTDOWN").info("⏹️ Bot operation was cancelled")
                    break
                except Exception as e:
                    logger.bind(ACTIVITY="ERROR").error(f"💥 Error in main loop: {str(e)}")
                    logger.bind(ACTIVITY="ERROR").exception("Full error traceback:")
                    await asyncio.sleep(10)
        finally:
            self.running = False
            logger.bind(ACTIVITY="SHUTDOWN").info("🛑 Main trading loop ended")
            
    async def check_token(self, token_address: str):
        """Check a token for trading opportunities."""
        try:
            # Skip if in cooldown period
            now = time.time()
            if token_address in self.trade_cooldowns and now - self.trade_cooldowns[token_address] < self.min_trade_interval:
                logger.bind(SCANNER=True).debug(f"⏳ Token {token_address[:8]}... in cooldown")
                return
                
            # Skip if checked very recently
            if token_address in self.last_check and now - self.last_check[token_address] < 30:
                logger.bind(SCANNER=True).debug(f"🕐 Token {token_address[:8]}... checked recently")
                return
                
            # Update last check time
            self.last_check[token_address] = now
            
            logger.bind(SCANNER=True).debug(f"📊 Fetching data for token {token_address[:8]}...")
            
            # Get token metadata
            metadata = await self.helius_service.get_token_metadata(token_address)
            if not metadata:
                logger.bind(SCANNER=True).warning(f"❌ No metadata for token {token_address[:8]}...")
                return
                
            # Get token price
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
                logger.bind(SCANNER=True).warning(f"❌ No LLM recommendation for {metadata.get('symbol', 'UNKNOWN')}")
                return
                
            # Get sentiment if we're considering a buy
            sentiment = None
            if recommendation.get("action", "hold") == "buy" and recommendation.get("confidence", 0) > 0.6:
                logger.bind(SCANNER=True).debug(f"😊 Getting sentiment analysis for {metadata.get('symbol', 'UNKNOWN')}...")
                sentiment = await self.local_llm.get_market_sentiment(token_address)
            
            # Get risk assessment
            logger.bind(SCANNER=True).debug(f"⚖️ Getting risk assessment for {metadata.get('symbol', 'UNKNOWN')}...")
            risk = await self.local_llm.get_risk_assessment(token_data)
            
            # Log the analysis
            action = recommendation.get("action", "hold")
            confidence = recommendation.get("confidence", 0)
            
            action_emoji = "🟢" if action == "buy" else "🔴" if action == "sell" else "🟡"
            confidence_emoji = "🔥" if confidence > 0.8 else "👍" if confidence > 0.6 else "😐" if confidence > 0.4 else "👎"
            
            logger.bind(SCANNER=True).info(f"📊 Analysis: {metadata.get('symbol', 'UNKNOWN')} ({token_address[:8]}...)")
            logger.bind(SCANNER=True).info(f"💰 Price: ${token_data['price_usd']:.6f} | 💧 Liquidity: ${token_data['liquidity_usd']:.2f}")
            logger.bind(SCANNER=True).info(f"{action_emoji} Action: {action.upper()} | {confidence_emoji} Confidence: {confidence:.2f}")
            
            if risk:
                risk_level = risk.get("risk_category", "unknown")
                risk_emoji = "🔴" if risk_level.lower() == "high" else "🟡" if risk_level.lower() == "medium" else "🟢"
                logger.bind(SCANNER=True).info(f"{risk_emoji} Risk: {risk_level.upper()} ({risk.get('overall_risk', 0):.2f})")
            
            if sentiment:
                sentiment_score = sentiment.get("score", 0)
                sentiment_direction = "positive" if sentiment_score > 0.2 else "negative" if sentiment_score < -0.2 else "neutral"
                sentiment_emoji = "😊" if sentiment_direction == "positive" else "😞" if sentiment_direction == "negative" else "😐"
                logger.bind(SCANNER=True).info(f"{sentiment_emoji} Sentiment: {sentiment_direction.upper()} ({sentiment_score:.2f})")
            
            # Determine if we should execute the trade
            should_trade = False
            trade_reason = ""
            
            if action == "buy":
                # Check if we should buy
                if (confidence > 0.7 and 
                    self.active_trades < self.max_concurrent_trades and
                    (sentiment is None or sentiment.get("score", 0) > -self.sentiment_threshold)):
                    
                    # Check risk limits
                    max_position = recommendation.get("position_size", 0.01)
                    portfolio_value = self.portfolio.get_portfolio_value()
                    position_percent = max_position / portfolio_value if portfolio_value > 0 else 0
                    
                    if position_percent <= self.risk_limit:
                        should_trade = True
                        trade_reason = f"High confidence ({confidence:.2f}), within risk limits ({position_percent:.2%})"
                    else:
                        trade_reason = f"Position too large ({position_percent:.2%} > {self.risk_limit:.2%})"
                else:
                    reasons = []
                    if confidence <= 0.7:
                        reasons.append(f"low confidence ({confidence:.2f})")
                    if self.active_trades >= self.max_concurrent_trades:
                        reasons.append(f"max trades reached ({self.active_trades}/{self.max_concurrent_trades})")
                    if sentiment and sentiment.get("score", 0) <= -self.sentiment_threshold:
                        reasons.append(f"negative sentiment ({sentiment.get('score', 0):.2f})")
                    trade_reason = f"Buy criteria not met: {', '.join(reasons)}"
            
            elif action == "sell":
                # Check if we should sell
                if confidence > 0.6:
                    # Check if we own this token
                    holdings = self.portfolio.get_holdings()
                    for holding in holdings:
                        if holding["token_address"] == token_address:
                            should_trade = True
                            trade_reason = f"Sell signal with high confidence ({confidence:.2f}), currently holding {holding['amount']:.2f}"
                            break
                    if not should_trade:
                        trade_reason = f"Sell signal but no holdings in {metadata.get('symbol', 'UNKNOWN')}"
                else:
                    trade_reason = f"Sell confidence too low ({confidence:.2f})"
            else:
                trade_reason = f"Hold recommendation ({confidence:.2f})"
            
            logger.bind(SCANNER=True).info(f"🎯 Decision: {trade_reason}")
            
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
            
    async def close(self):
        """Close all services."""
        logger.info("Shutting down MemeCoinBot...")
        self.running = False
        
        try:
            # Log final portfolio status
            summary = self.portfolio.get_portfolio_summary()
            logger.info(f"Final portfolio value: {summary['current_value']:.4f} SOL")
            logger.info(f"Total profit/loss: {summary['total_return']:.4f} SOL ({summary['percent_return']:.2f}%)")
            logger.info(f"Total trades: {summary['total_trades']}")
            
            await self.wallet_manager.close()
            await self.local_llm.close()
            await self.helius_service.close()
            await self.jupiter_service.close()
            logger.info("All services closed")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

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
        asyncio.run(bot.close()) 
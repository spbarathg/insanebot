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

# Import ML engine components
from .ml_engine import (
    PricePredictor, 
    PatternRecognizer, 
    SentimentAnalyzer, 
    RiskScorer,
    MLSignal,
    PredictionResult,
    PatternType,
    SentimentResult,
    RiskScore
)

# Import execution engine
from .execution_engine import (
    ExecutionEngine,
    OrderType,
    ExecutionStrategy,
    ExecutionParams,
    ExecutionResult
)

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
        
        # Add ML engine components
        self.price_predictor = None
        self.pattern_recognizer = None
        self.sentiment_analyzer = None
        self.risk_scorer = None
        
        # Add execution engine
        self.execution_engine = None
        
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
        self.ml_predictions_made = 0  # Track ML predictions
        self.patterns_detected = 0  # Track pattern detections
        self.advanced_executions = 0  # Track advanced execution engine usage
        
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
            
            # Initialize ML Engine Components
            logger.info("🧠 Initializing ML Engine...")
            
            # Price Predictor
            logger.info("🔧 Initializing Price Predictor...")
            self.price_predictor = PricePredictor()
            if not await self.price_predictor.initialize():
                logger.error("❌ Failed to initialize Price Predictor")
                return False
            
            # Pattern Recognizer
            logger.info("🔧 Initializing Pattern Recognizer...")
            self.pattern_recognizer = PatternRecognizer()
            if not await self.pattern_recognizer.initialize():
                logger.error("❌ Failed to initialize Pattern Recognizer")
                return False
            
            # Sentiment Analyzer
            logger.info("🔧 Initializing Sentiment Analyzer...")
            self.sentiment_analyzer = SentimentAnalyzer()
            if not await self.sentiment_analyzer.initialize():
                logger.error("❌ Failed to initialize Sentiment Analyzer")
                return False
            
            # Risk Scorer
            logger.info("🔧 Initializing Risk Scorer...")
            self.risk_scorer = RiskScorer()
            if not await self.risk_scorer.initialize():
                logger.error("❌ Failed to initialize Risk Scorer")
                return False
            
            # Advanced Execution Engine
            logger.info("🔧 Initializing Advanced Execution Engine...")
            self.execution_engine = ExecutionEngine(self.jupiter_service, self.helius_service)
            if not await self.execution_engine.initialize():
                logger.error("❌ Failed to initialize Execution Engine")
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
        """Log performance summary including arbitrage and execution statistics."""
        try:
            # Portfolio summary
            portfolio_summary = self.portfolio.get_portfolio_summary()
            
            # Arbitrage scanner stats
            arbitrage_stats = self.arbitrage_scanner.get_scanner_stats() if self.arbitrage_scanner else {}
            
            # Execution engine stats
            execution_stats = self.execution_engine.get_performance_stats() if self.execution_engine else {}
            
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
            
            if execution_stats:
                logger.bind(EXECUTION=True).info(
                    f"⚡ Advanced Execution: {execution_stats.get('total_executions', 0)} total, "
                    f"{execution_stats.get('success_rate', 0):.1%} success rate, "
                    f"avg slippage: {execution_stats.get('average_slippage', 0):.2%}, "
                    f"avg time: {execution_stats.get('average_execution_time', 0):.1f}s"
                )
            
            # ML Engine summary
            logger.bind(ML=True).info(
                f"🧠 ML Analysis: {self.ml_predictions_made} predictions, "
                f"{self.patterns_detected} patterns detected, "
                f"{self.advanced_executions} advanced executions"
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
        """Check a specific token for trading opportunities with ML analysis."""
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
                
            # Get holders data for ML analysis
            holders_data = await self.helius_service.get_token_holders(token_address, limit=20)
                
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
            
            # ===== ML ANALYSIS PIPELINE =====
            
            # Phase 1: Price Prediction
            logger.bind(ML=True).debug(f"🔮 Running price prediction for {metadata.get('symbol', 'UNKNOWN')}...")
            price_prediction = None
            try:
                price_prediction = await self.price_predictor.predict_price(
                    token_address, token_data, token_data["price_history"]
                )
                if price_prediction:
                    self.ml_predictions_made += 1
                    logger.bind(ML=True).info(
                        f"🔮 Price prediction for {metadata.get('symbol', 'UNKNOWN')}: "
                        f"1h: ${price_prediction.predicted_price_1h:.6f} ({price_prediction.expected_return_1h:+.1%}), "
                        f"24h: ${price_prediction.predicted_price_24h:.6f} ({price_prediction.expected_return_24h:+.1%}), "
                        f"Confidence: {price_prediction.weighted_confidence:.2f}"
                    )
            except Exception as e:
                logger.bind(ML=True).error(f"Error in price prediction: {str(e)}")
            
            # Phase 2: Pattern Recognition
            logger.bind(ML=True).debug(f"📊 Running pattern recognition for {metadata.get('symbol', 'UNKNOWN')}...")
            patterns = []
            try:
                patterns = await self.pattern_recognizer.recognize_patterns(
                    token_address, token_data, token_data["price_history"]
                )
                if patterns:
                    self.patterns_detected += len(patterns)
                    for pattern in patterns[:3]:  # Show top 3 patterns
                        logger.bind(ML=True).info(
                            f"📈 Pattern detected: {pattern.pattern_type.value} "
                            f"(Confidence: {pattern.confidence:.2f}, "
                            f"Expected: {pattern.expected_direction} {pattern.expected_move:+.1%})"
                        )
            except Exception as e:
                logger.bind(ML=True).error(f"Error in pattern recognition: {str(e)}")
            
            # Phase 3: Sentiment Analysis
            logger.bind(ML=True).debug(f"😊 Running sentiment analysis for {metadata.get('symbol', 'UNKNOWN')}...")
            sentiment_result = None
            try:
                sentiment_result = await self.sentiment_analyzer.analyze_sentiment(
                    token_address, token_data, token_data["price_history"], holders_data
                )
                if sentiment_result:
                    logger.bind(ML=True).info(
                        f"😊 Sentiment for {metadata.get('symbol', 'UNKNOWN')}: "
                        f"{sentiment_result.overall_sentiment.value} "
                        f"(Score: {sentiment_result.sentiment_score:+.2f}, "
                        f"Fear/Greed: {sentiment_result.fear_greed_index:.0f})"
                    )
            except Exception as e:
                logger.bind(ML=True).error(f"Error in sentiment analysis: {str(e)}")
            
            # Phase 4: Risk Assessment
            logger.bind(ML=True).debug(f"⚠️ Running risk assessment for {metadata.get('symbol', 'UNKNOWN')}...")
            risk_score = None
            try:
                risk_score = await self.risk_scorer.calculate_risk_score(
                    token_address, token_data, token_data["price_history"], holders_data
                )
                if risk_score:
                    logger.bind(ML=True).info(
                        f"⚠️ Risk assessment for {metadata.get('symbol', 'UNKNOWN')}: "
                        f"{risk_score.risk_category.upper()} risk "
                        f"(Score: {risk_score.overall_risk_score:.2f}, "
                        f"Max position: {risk_score.recommended_position_size:.1%})"
                    )
            except Exception as e:
                logger.bind(ML=True).error(f"Error in risk assessment: {str(e)}")
            
            # Phase 5: Generate ML Trading Signal
            ml_signal = self._generate_ml_signal(
                token_address, token_data, price_prediction, patterns, sentiment_result, risk_score
            )
            
            # Store ML results in cache for execution engine optimization
            if ml_signal:
                self.market_data_cache[f"{token_address}_ml_signal"] = ml_signal
            if risk_score:
                self.market_data_cache[f"{token_address}_risk_score"] = risk_score
            
            # Get traditional LLM recommendation for comparison
            logger.bind(SCANNER=True).debug(f"🤖 Getting LLM analysis for {metadata.get('symbol', 'UNKNOWN')}...")
            llm_recommendation = await self.local_llm.analyze_market(token_data)
            
            if not llm_recommendation:
                logger.bind(SCANNER=True).warning(f"❌ No LLM recommendation for {metadata.get('symbol', 'UNKNOWN')}")
                return
            
            # Combine ML signal with LLM recommendation
            final_decision = self._combine_ml_and_llm_decisions(ml_signal, llm_recommendation, token_data)
            
            action = final_decision.get("action", "hold").lower()
            confidence = final_decision.get("confidence", 0)
            reasoning = final_decision.get("reasoning", "No reasoning provided")
            
            # Log the combined analysis result
            action_emoji = "🟢" if action == "buy" else "🔴" if action == "sell" else "🟡"
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            
            logger.bind(ANALYSIS=True).info(
                f"{action_emoji} {metadata.get('symbol', 'UNKNOWN')} | "
                f"{action.upper()} | "
                f"Confidence: {confidence:.2f} [{confidence_bar}] | "
                f"Price: ${token_data['price_usd']:.6f}"
            )
            
            if ml_signal:
                logger.bind(ML=True).info(
                    f"🧠 ML Signal: {ml_signal.signal_type.upper()} "
                    f"(Strength: {ml_signal.signal_strength:.2f}, "
                    f"Quality: {ml_signal.quality_score:.2f})"
                )
            
            logger.bind(ANALYSIS=True).debug(f"💭 Reasoning: {reasoning}")
            
            # Determine if we should trade
            min_confidence = 0.65 if action in ["buy", "sell"] else 0.5  # Higher threshold with ML
            should_trade = (
                action in ["buy", "sell"] and
                confidence >= min_confidence and
                self.active_trades < self.max_concurrent_trades and
                (not risk_score or risk_score.is_safe_to_trade)  # Check ML risk assessment
            )
            
            # Execute trade if conditions are met
            if should_trade:
                # Use ML-recommended position size if available
                position_size = final_decision.get("position_size", 0.01)
                if risk_score and risk_score.recommended_position_size < position_size:
                    position_size = risk_score.recommended_position_size
                    logger.bind(ML=True).info(f"🎯 Position size adjusted by risk analysis: {position_size:.1%}")
                
                logger.bind(TRADE=True).info(f"🚀 Executing {action.upper()} trade for {metadata.get('symbol', 'UNKNOWN')}")
                await self.execute_trade(token_data, action, position_size, reasoning)
                # Set cooldown for this token
                self.trade_cooldowns[token_address] = now
                logger.bind(SCANNER=True).info(f"⏰ Set cooldown for {metadata.get('symbol', 'UNKNOWN')} until {datetime.fromtimestamp(now + self.min_trade_interval).strftime('%H:%M:%S')}")
                
        except Exception as e:
            logger.bind(SCANNER=True).error(f"💥 Error checking token {token_address[:8]}...: {str(e)}")
            logger.bind(SCANNER=True).exception("Full error traceback:")
    
    def _generate_ml_signal(self, token_address: str, token_data: Dict, 
                           price_prediction: PredictionResult = None, 
                           patterns: List[PatternType] = None, 
                           sentiment: SentimentResult = None, 
                           risk: RiskScore = None) -> MLSignal:
        """Generate ML trading signal from analysis components"""
        try:
            if not any([price_prediction, patterns, sentiment, risk]):
                return None
            
            current_time = time.time()
            current_price = token_data.get('price_usd', 0)
            
            # Determine signal type and strength
            signal_strength = 0.0
            signal_type = "hold"
            reasoning = []
            
            # Price prediction influence
            if price_prediction:
                expected_return = price_prediction.expected_return_24h
                confidence = price_prediction.weighted_confidence
                
                if expected_return > 0.1 and confidence > 0.6:  # >10% expected return with good confidence
                    signal_strength += 0.4
                    signal_type = "buy"
                    reasoning.append(f"ML predicts {expected_return:+.1%} return (confidence: {confidence:.2f})")
                elif expected_return < -0.05 and confidence > 0.6:  # <-5% expected return
                    signal_strength += 0.3
                    signal_type = "sell"
                    reasoning.append(f"ML predicts {expected_return:+.1%} decline (confidence: {confidence:.2f})")
            
            # Pattern analysis influence
            if patterns:
                bullish_patterns = [p for p in patterns if p.expected_direction == "up"]
                bearish_patterns = [p for p in patterns if p.expected_direction == "down"]
                
                bullish_strength = sum(p.reliability_score for p in bullish_patterns)
                bearish_strength = sum(p.reliability_score for p in bearish_patterns)
                
                if bullish_strength > bearish_strength and bullish_strength > 0.5:
                    signal_strength += min(0.3, bullish_strength * 0.3)
                    if signal_type != "sell":
                        signal_type = "buy"
                    reasoning.append(f"Bullish patterns detected (strength: {bullish_strength:.2f})")
                elif bearish_strength > bullish_strength and bearish_strength > 0.5:
                    signal_strength += min(0.3, bearish_strength * 0.3)
                    if signal_type != "buy":
                        signal_type = "sell"
                    reasoning.append(f"Bearish patterns detected (strength: {bearish_strength:.2f})")
            
            # Sentiment influence
            if sentiment:
                sentiment_score = sentiment.sentiment_score
                confidence = sentiment.confidence
                
                if sentiment_score > 0.3 and confidence > 0.6:  # Positive sentiment
                    signal_strength += min(0.2, sentiment_score * confidence * 0.3)
                    if signal_type != "sell":
                        signal_type = "buy"
                    reasoning.append(f"Positive sentiment ({sentiment.overall_sentiment.value})")
                elif sentiment_score < -0.3 and confidence > 0.6:  # Negative sentiment
                    signal_strength += min(0.2, abs(sentiment_score) * confidence * 0.3)
                    if signal_type != "buy":
                        signal_type = "sell"
                    reasoning.append(f"Negative sentiment ({sentiment.overall_sentiment.value})")
            
            # Risk assessment influence
            if risk:
                if risk.overall_risk_score > 0.8:  # Very high risk
                    signal_strength *= 0.3  # Drastically reduce signal strength
                    reasoning.append(f"High risk reduces signal strength ({risk.risk_category})")
                elif risk.overall_risk_score > 0.6:  # High risk
                    signal_strength *= 0.6  # Moderately reduce signal strength
                    reasoning.append(f"Elevated risk ({risk.risk_category})")
                elif risk.overall_risk_score < 0.3:  # Low risk
                    signal_strength *= 1.2  # Slightly boost signal strength
                    reasoning.append(f"Low risk environment ({risk.risk_category})")
            
            # Ensure signal strength is within bounds
            signal_strength = min(1.0, signal_strength)
            
            # Calculate confidence based on data availability
            confidence = 0.5  # Base confidence
            if price_prediction:
                confidence += 0.2
            if patterns:
                confidence += 0.1
            if sentiment:
                confidence += 0.1
            if risk:
                confidence += 0.1
            confidence = min(0.95, confidence)
            
            # Calculate target and stop loss prices
            if signal_type == "buy":
                target_price = current_price * 1.05  # 5% target
                stop_loss_price = current_price * 0.95  # 5% stop loss
                if risk and risk.stop_loss_level:
                    stop_loss_price = min(stop_loss_price, risk.stop_loss_level)
            elif signal_type == "sell":
                target_price = current_price * 0.95  # 5% target (for short)
                stop_loss_price = current_price * 1.05  # 5% stop loss (for short)
            else:
                target_price = current_price
                stop_loss_price = current_price * 0.9
            
            # Position size recommendation
            position_size = 0.02  # Default 2%
            if risk:
                position_size = risk.recommended_position_size
            
            # Create ML signal
            ml_signal = MLSignal(
                token_address=token_address,
                token_symbol=token_data.get('symbol', 'UNKNOWN'),
                signal_type=signal_type,
                signal_strength=signal_strength,
                confidence=confidence,
                predicted_return=price_prediction.expected_return_24h if price_prediction else 0.0,
                risk_score=risk.overall_risk_score if risk else 0.5,
                sentiment_score=sentiment.sentiment_score if sentiment else 0.0,
                pattern_signals=patterns or [],
                price_prediction=price_prediction,
                timeframe="24h",
                entry_price=current_price,
                target_price=target_price,
                stop_loss_price=stop_loss_price,
                position_size_recommendation=position_size,
                reasoning=reasoning,
                signal_timestamp=current_time,
                expires_at=current_time + 3600  # 1 hour expiration
            )
            
            return ml_signal
            
        except Exception as e:
            logger.error(f"Error generating ML signal: {str(e)}")
            return None
    
    def _combine_ml_and_llm_decisions(self, ml_signal: MLSignal, llm_recommendation: Dict, token_data: Dict) -> Dict:
        """Combine ML signal with LLM recommendation for final decision"""
        try:
            # Extract LLM recommendation
            llm_action = llm_recommendation.get("action", "hold").lower()
            llm_confidence = llm_recommendation.get("confidence", 0)
            llm_reasoning = llm_recommendation.get("reasoning", "")
            
            # If no ML signal, use LLM recommendation
            if not ml_signal:
                return {
                    "action": llm_action,
                    "confidence": llm_confidence,
                    "reasoning": f"LLM only: {llm_reasoning}",
                    "position_size": llm_recommendation.get("position_size", 0.01)
                }
            
            # Combine signals
            combined_reasoning = []
            
            # Agreement boost
            if ml_signal.signal_type == llm_action:
                # Both agree - boost confidence
                combined_confidence = min(0.95, (ml_signal.confidence + llm_confidence) * 0.7)
                combined_action = ml_signal.signal_type
                combined_reasoning.append(f"ML and LLM agree on {combined_action.upper()}")
            else:
                # Disagreement - be more conservative
                if ml_signal.signal_strength > llm_confidence:
                    combined_action = ml_signal.signal_type
                    combined_confidence = ml_signal.confidence * 0.8
                    combined_reasoning.append(f"ML signal stronger: {ml_signal.signal_type.upper()}")
                else:
                    combined_action = llm_action
                    combined_confidence = llm_confidence * 0.8
                    combined_reasoning.append(f"LLM signal stronger: {llm_action.upper()}")
                
                combined_reasoning.append("Signals disagree - reduced confidence")
            
            # Add specific reasoning from both
            combined_reasoning.extend(ml_signal.reasoning)
            combined_reasoning.append(f"LLM: {llm_reasoning}")
            
            # Position sizing - use more conservative approach
            ml_position_size = ml_signal.position_size_recommendation if ml_signal else 0.01
            llm_position_size = llm_recommendation.get("position_size", 0.01)
            combined_position_size = min(ml_position_size, llm_position_size)
            
            return {
                "action": combined_action,
                "confidence": combined_confidence,
                "reasoning": " | ".join(combined_reasoning),
                "position_size": combined_position_size
            }
            
        except Exception as e:
            logger.error(f"Error combining ML and LLM decisions: {str(e)}")
            return llm_recommendation
            
    async def execute_trade(self, token_data: Dict, action: str, position_size: float, reasoning: str):
        """Execute a trade using the advanced execution engine."""
        try:
            token_symbol = token_data.get("symbol", "UNKNOWN")
            token_address = token_data.get("address", "")
            
            # Calculate trade details
            price = token_data.get("price_usd", 0)
            sol_price = 100.0  # Approximate SOL price in USD
            amount = position_size  # In SOL
            
            # Determine execution strategy based on market conditions and ML analysis
            execution_strategy = ExecutionStrategy.SMART  # Default to smart strategy
            
            # Get ML analysis for execution optimization
            ml_signal = self.market_data_cache.get(f"{token_address}_ml_signal")
            risk_score = self.market_data_cache.get(f"{token_address}_risk_score")
            
            # Adjust execution strategy based on ML insights
            if risk_score:
                if risk_score.overall_risk_score > 0.7:
                    execution_strategy = ExecutionStrategy.STEALTH  # High risk = stealth
                elif risk_score.overall_risk_score < 0.3:
                    execution_strategy = ExecutionStrategy.AGGRESSIVE  # Low risk = aggressive
            
            # Set execution parameters based on risk and market conditions
            execution_params = ExecutionParams(
                max_slippage=0.015,  # 1.5% max slippage
                max_price_impact=0.02,  # 2% max price impact
                execution_timeout=60.0,  # 60 seconds timeout
                split_threshold=500.0,  # Split orders >$500
                max_order_chunks=8,
                mev_protection=True,
                gas_optimization=True
            )
            
            # Adjust parameters based on risk assessment
            if risk_score:
                if risk_score.overall_risk_score > 0.6:
                    execution_params.max_slippage *= 0.8  # Tighter slippage for risky tokens
                    execution_params.split_threshold *= 0.5  # Split smaller amounts
                elif risk_score.overall_risk_score < 0.3:
                    execution_params.max_slippage *= 1.2  # Allow more slippage for low risk
            
            if action == "buy":
                logger.info(f"🚀 Executing ADVANCED BUY: {amount} SOL of {token_symbol} using {execution_strategy.value} strategy")
                logger.info(f"💭 Reasoning: {reasoning}")
                
                # Execute buy using execution engine
                result = await self.execution_engine.execute_trade(
                    order_type=OrderType.MARKET,
                    input_token="So11111111111111111111111111111111111111112",  # SOL
                    output_token=token_address,
                    amount=amount,
                    strategy=execution_strategy,
                    execution_params=execution_params
                )
                
                if result.success:
                    # Create trade data for portfolio tracking
                    trade_data = {
                        "action": "buy",
                        "token": token_symbol,
                        "token_address": token_address,
                        "amount_sol": result.executed_amount,
                        "received_amount": result.received_amount,
                        "price_usd": result.actual_price,
                        "sol_price": sol_price,
                        "slippage": result.slippage,
                        "gas_used": result.gas_used,
                        "execution_time": result.execution_time,
                        "execution_strategy": execution_strategy.value,
                        "timestamp": time.time(),
                        "status": "success",
                        "transaction_id": result.transaction_id
                    }
                    
                    # Update portfolio
                    self.portfolio.add_trade(trade_data)
                    self.active_trades += 1
                    self.advanced_executions += 1
                    
                    logger.info(f"✅ Advanced BUY execution successful: {result.executed_amount:.6f} SOL → {result.received_amount:.2f} {token_symbol}")
                    logger.info(f"📊 Execution stats: {result.slippage:.2%} slippage, {result.execution_time:.1f}s, {len(result.routes_used)} routes")
                    
                else:
                    logger.error(f"❌ Advanced BUY execution failed: {', '.join(result.errors)}")
                
            elif action == "sell":
                logger.info(f"🚀 Executing ADVANCED SELL: {amount} SOL worth of {token_symbol} using {execution_strategy.value} strategy")
                logger.info(f"💭 Reasoning: {reasoning}")
                
                # Execute sell using execution engine
                result = await self.execution_engine.execute_trade(
                    order_type=OrderType.MARKET,
                    input_token=token_address,
                    output_token="So11111111111111111111111111111111111111112",  # SOL
                    amount=amount,
                    strategy=execution_strategy,
                    execution_params=execution_params
                )
                
                if result.success:
                    # Create trade data for portfolio tracking
                    trade_data = {
                        "action": "sell",
                        "token": token_symbol,
                        "token_address": token_address,
                        "amount_sol": result.received_amount,  # SOL received
                        "sold_amount": result.executed_amount,  # Token amount sold
                        "price_usd": result.actual_price,
                        "sol_price": sol_price,
                        "slippage": result.slippage,
                        "gas_used": result.gas_used,
                        "execution_time": result.execution_time,
                        "execution_strategy": execution_strategy.value,
                        "timestamp": time.time(),
                        "status": "success",
                        "transaction_id": result.transaction_id
                    }
                    
                    # Update portfolio
                    self.portfolio.add_trade(trade_data)
                    self.active_trades = max(0, self.active_trades - 1)
                    self.advanced_executions += 1
                    
                    logger.info(f"✅ Advanced SELL execution successful: {result.executed_amount:.2f} {token_symbol} → {result.received_amount:.6f} SOL")
                    logger.info(f"📊 Execution stats: {result.slippage:.2%} slippage, {result.execution_time:.1f}s, {len(result.routes_used)} routes")
                    
                else:
                    logger.error(f"❌ Advanced SELL execution failed: {', '.join(result.errors)}")
                
        except Exception as e:
            logger.error(f"💥 Error in advanced trade execution: {str(e)}")
            logger.exception("Full execution error traceback:")
    
    async def stop(self):
        """Stop the bot and cleanup resources."""
        logger.bind(ACTIVITY="BOT_STOP").info("🛑 Stopping MemeCoinBot...")
        
        self.running = False
        
        try:
            # Close all services
            if self.arbitrage_scanner:
                await self.arbitrage_scanner.close()
                
            if self.execution_engine:
                await self.execution_engine.close()
                
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
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
from ..utils.config import settings
from .data_ingestion import DataIngestion
from .grok_engine import GrokEngine
from .trade_execution import TradeExecution
from ..monitoring.monitoring import MonitoringSystem
from .market_data import MarketData
from .wallet_manager import WalletManager
from .local_llm import LocalLLM
from .middleware import handle_errors, rate_limit, error_handler
from .cache import market_data_cache, token_cache, price_cache

logger = logging.getLogger(__name__)

class MemeCoinBot:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.grok_engine = GrokEngine()
        self.trade_execution = TradeExecution()
        self.monitoring = MonitoringSystem()
        self.market_data = MarketData()
        self.wallet_manager = WalletManager()
        self.local_llm = LocalLLM()
        self.trade_history: List[Dict] = []
        self.feature_weights = settings.FEATURE_WEIGHTS.copy()
        self.running = False
        self.daily_loss_limit = 1000  # $1000 daily loss limit
        self.daily_loss = 0
        self.last_reset = datetime.now()
        self.load_trade_history()

    def load_trade_history(self) -> None:
        """Load trade history from file."""
        try:
            with open(settings.TRADE_LOG_FILE, 'r') as f:
                self.trade_history = json.load(f)
            logger.info(f"Loaded {len(self.trade_history)} trades from history")
        except FileNotFoundError:
            logger.info("No trade history found, starting fresh")
            self.trade_history = []
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
            self.trade_history = []

    def save_trade_history(self) -> None:
        """Save trade history to file."""
        try:
            with open(settings.TRADE_LOG_FILE, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
            logger.info(f"Saved {len(self.trade_history)} trades to history")
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")

    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize components
            await self.data_ingestion.start()
            await self.trade_execution.initialize()
            await self.market_data.initialize()
            await self.wallet_manager.initialize()
            await self.grok_engine.initialize()
            await self.local_llm.initialize()

            logger.info("Bot initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize bot: {str(e)}")
            return False

    async def close(self):
        """Close all components"""
        try:
            self.running = False
            logger.info("Stopping MemeCoinBot...")
            
            # Save trade history
            self.save_trade_history()
            
            # Close components
            await self.data_ingestion.close()
            await self.trade_execution.close()
            await self.market_data.close()
            await self.wallet_manager.close()
            await self.grok_engine.close()
            await self.local_llm.close()
            
            logger.info("MemeCoinBot stopped")
            
        except Exception as e:
            logger.error(f"Error closing bot: {str(e)}")

    @handle_errors
    async def process_token(self, token_address: str) -> Optional[Dict]:
        """Process a token for potential trading"""
        try:
            # Get token data with caching
            token_data = await self.market_data.get_token_data(token_address)
            if not token_data:
                logger.warning(f"No data available for token {token_address}")
                return None

            # Get price data with caching
            price_data = await self.market_data.get_price_data(token_address)
            if not price_data:
                logger.warning(f"No price data available for token {token_address}")
                return None

            # Combine data
            token_data.update(price_data)
            return token_data

        except Exception as e:
            await error_handler.handle_error(e, "process_token")
            return None

    @handle_errors
    @rate_limit(max_requests=10, time_window=60)  # 10 trades per minute
    async def execute_trade(self, token_data: Dict) -> bool:
        """Execute a trade based on token data"""
        try:
            # Check daily loss limit
            if self.daily_loss >= self.daily_loss_limit:
                logger.warning("Daily loss limit reached, skipping trade")
                return False

            # Implement your trading logic here
            # This is a placeholder for the actual trading implementation
            logger.info(f"Executing trade for token {token_data['symbol']}")
            return True

        except Exception as e:
            await error_handler.handle_error(e, "execute_trade")
            return False

    @handle_errors
    async def start(self) -> None:
        """Start the trading bot"""
        try:
            if not await self.initialize():
                logger.error("Failed to initialize bot")
                return

            self.running = True
            logger.info("Starting MemeCoinBot...")

            while self.running:
                try:
                    # Reset daily loss if needed
                    if datetime.now() - self.last_reset > timedelta(days=1):
                        self.daily_loss = 0
                        self.last_reset = datetime.now()
                        logger.info("Daily loss counter reset")

                    # Process new tokens
                    new_tokens = await self.data_ingestion.get_new_tokens()
                    for token in new_tokens:
                        if not self.running:
                            break
                        token_data = await self.process_token(token['address'])
                        if token_data:
                            success = await self.execute_trade(token_data)
                            if not success:
                                self.daily_loss += 100  # Example loss amount

                    # Update feature weights
                    self.update_feature_weights()

                    # Sleep to prevent excessive API calls
                    await asyncio.sleep(settings.TRADING_INTERVAL)

                except Exception as e:
                    await error_handler.handle_error(e, "main_loop")
                    if not error_handler.should_continue():
                        logger.error("Too many errors, stopping bot")
                        break
                    await asyncio.sleep(5)

        except Exception as e:
            await error_handler.handle_error(e, "start")
        finally:
            await self.close()

    async def monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.running:
            try:
                # Check system and trading health
                self.monitoring.check_health(self.trade_history)
                
                # Save metrics
                self.monitoring.save_metrics(self.trade_history)
                
                # Create backup if needed
                self.monitoring.create_backup()
                
                await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(settings.RETRY_INTERVAL)

    async def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit reached"""
        try:
            balance = await self.wallet_manager.get_balance()
            if balance is None:
                return True

            initial_balance = settings.STARTING_CAPITAL
            self.daily_loss = (initial_balance - balance) / initial_balance
            
            if self.daily_loss >= settings.DAILY_LOSS_LIMIT:
                logger.warning(f"Daily loss limit reached: {self.daily_loss:.2%}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error checking daily loss: {str(e)}")
            return True

    def update_feature_weights(self) -> None:
        """Update feature weights based on recent trade performance."""
        try:
            # Get recent trades
            cutoff_time = datetime.now() - timedelta(days=settings.LEARNING_WINDOW)
            recent_trades = [
                trade for trade in self.trade_history
                if datetime.fromisoformat(trade['timestamp']) > cutoff_time
            ]
            
            if not recent_trades:
                return
            
            # Calculate win rates for each feature
            feature_wins = {feature: 0 for feature in self.feature_weights}
            feature_trades = {feature: 0 for feature in self.feature_weights}
            
            for trade in recent_trades:
                for feature in self.feature_weights:
                    if trade.get(feature, False):
                        feature_trades[feature] += 1
                        if trade['profit'] > 0:
                            feature_wins[feature] += 1
            
            # Update weights based on win rates
            for feature in self.feature_weights:
                if feature_trades[feature] > 0:
                    win_rate = feature_wins[feature] / feature_trades[feature]
                    self.feature_weights[feature] = win_rate
            
            # Normalize weights
            total_weight = sum(self.feature_weights.values())
            if total_weight > 0:
                for feature in self.feature_weights:
                    self.feature_weights[feature] /= total_weight
            
            logger.info(f"Updated feature weights: {self.feature_weights}")
            
        except Exception as e:
            logger.error(f"Error updating feature weights: {e}")

    def is_new_token(self, token: Dict) -> bool:
        """Check if token is new enough to trade."""
        try:
            token_age = time.time() - token['created_at']
            return token_age <= settings.MAX_TOKEN_AGE
        except Exception as e:
            logger.error(f"Error checking token age: {e}")
            return False

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
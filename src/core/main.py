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
        self.daily_loss = 0.0
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

    async def process_token(self, token: Dict) -> Optional[Dict]:
        """Process a single token for trading"""
        try:
            # Get market data
            market_data = await self.market_data.get_token_data(token['address'])
            if not market_data:
                return None

            # Get sentiment from Grok
            sentiment = await self.grok_engine.analyze_sentiment(token)
            
            # Prepare data for local LLM
            llm_data = {
                'token_address': token['address'],
                'price': market_data['price'],
                'volume_24h': market_data['volume_24h'],
                'liquidity': market_data['liquidity'],
                'market_cap': market_data['market_cap'],
                'sentiment': sentiment,
                'recent_trades': self.trade_history[-10:]  # Last 10 trades
            }

            # Get trading decision from local LLM
            decision = await self.local_llm.analyze_market(llm_data)
            if not decision or decision['confidence'] < settings.CONFIDENCE_THRESHOLD:
                return None

            # Execute trade
            trade_result = await self.execute_trade(token, decision)
            if trade_result:
                # Learn from trade result
                self.local_llm.learn_from_trade(trade_result)
                
                # Update trade history
                self.trade_history.append(trade_result)
                self.save_trade_history()

            return trade_result

        except Exception as e:
            logger.error(f"Error processing token: {str(e)}")
            return None

    async def execute_trade(self, token: Dict, decision: Dict) -> Optional[Dict]:
        """Execute a trade based on the decision"""
        try:
            # Calculate trade size
            trade_size = min(
                settings.MAX_TRADE_SIZE,
                self.trade_execution.get_available_balance() * decision['position_size']
            )
            
            if trade_size < settings.MIN_TRADE_SIZE:
                return None
            
            # Execute buy order
            buy_result = await self.trade_execution.execute_buy(
                token['address'],
                trade_size
            )
            
            if not buy_result:
                return None
            
            # Wait for price movement
            await asyncio.sleep(5)  # Adjust based on market conditions
            
            # Execute sell order
            sell_result = await self.trade_execution.execute_sell(
                token['address'],
                buy_result['amount']
            )
            
            if not sell_result:
                return None
            
            # Calculate profit
            profit = sell_result['amount'] - buy_result['amount']
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'token_address': token['address'],
                'token_symbol': token['symbol'],
                'buy_amount': buy_result['amount'],
                'sell_amount': sell_result['amount'],
                'profit': profit,
                'confidence': decision['confidence'],
                'reasoning': decision['reasoning'],
                'market_state': {
                    'price': buy_result['price'],
                    'volume': buy_result['volume'],
                    'liquidity': buy_result['liquidity']
                }
            }
            
            logger.info(f"Trade executed: {trade_record}")
            return trade_record
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None

    async def start(self) -> None:
        """Start the trading bot"""
        try:
            if not await self.initialize():
                return

            self.running = True
            logger.info("Starting MemeCoinBot...")
            
            # Start monitoring
            asyncio.create_task(self.monitor_loop())
            
            # Main trading loop
            while self.running:
                try:
                    # Check daily loss limit
                    if await self.check_daily_loss_limit():
                        logger.warning("Stopping bot due to daily loss limit")
                        break

                    # Get market data
                    market_data = await self.data_ingestion.get_market_data()
                    if not market_data:
                        await asyncio.sleep(settings.DATA_REFRESH_INTERVAL)
                        continue

                    # Filter for new tokens
                    new_tokens = [
                        token for token in market_data
                        if self.is_new_token(token)
                    ]

                    # Process each token
                    for token in new_tokens:
                        if not self.running:
                            break
                        await self.process_token(token)

                    await asyncio.sleep(settings.DATA_REFRESH_INTERVAL)

                except Exception as e:
                    logger.error(f"Error in trading loop: {str(e)}")
                    await asyncio.sleep(settings.RETRY_INTERVAL)

        except Exception as e:
            logger.error(f"Error starting bot: {str(e)}")
        finally:
            await self.close()
            self.running = False

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
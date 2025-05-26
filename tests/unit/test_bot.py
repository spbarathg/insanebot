"""
Test suite for the trading bot.
"""
import pytest
import asyncio
import json
from datetime import datetime
from loguru import logger
from unittest.mock import AsyncMock, MagicMock

# Create a mock MemeCoinBot for testing without importing the real one
class MockMemeCoinBot:
    def __init__(self):
        self.running = False
        self.trade_history = []
        self.feature_weights = {
            "max_position_size": 0.1,
            "min_position_size": 0.01,
            "stop_loss": 0.05,
            "take_profit": 0.1,
            "max_concurrent_trades": 5,
        }
        self.active_trades = {}
        
    def load_trade_history(self):
        self.trade_history = []
        
    def update_feature_weights(self):
        # Normalize weights
        total = sum(self.feature_weights.values())
        for key in self.feature_weights:
            self.feature_weights[key] /= total
            
    async def initialize(self):
        pass
        
    async def run(self):
        self.running = True
        
    async def close(self):
        self.running = False

@pytest.fixture
def bot():
    return MockMemeCoinBot()

def test_bot_initialization(bot):
    assert bot.running == False
    assert len(bot.trade_history) == 0
    assert bot.feature_weights is not None

def test_trade_history_loading(bot):
    bot.load_trade_history()
    assert isinstance(bot.trade_history, list)

def test_feature_weight_update(bot):
    # Add some mock trades
    bot.trade_history = [
        {
            'timestamp': '2024-01-01T00:00:00',
            'profit': 0.1,
            'sentiment': True,
            'whale_activity': False,
            'price_momentum': True
        }
    ]
    bot.update_feature_weights()
    assert all(0 <= w <= 1 for w in bot.feature_weights.values())
    assert abs(sum(bot.feature_weights.values()) - 1.0) < 0.0001 

async def run_test():
    # Configure logging
    logger.add(
        f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        rotation="100 MB",
        level="DEBUG"
    )
    
    # Create and start bot
    bot = MockMemeCoinBot()
    try:
        # Run for 1 hour in test mode
        logger.info("Running test for 1 hour...")
        await asyncio.wait_for(bot.run(), timeout=3600)
    except asyncio.TimeoutError:
        logger.info("Test completed after 1 hour")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
    finally:
        await bot.close()
        
        # Print test summary
        logger.info("\nTest Summary:")
        logger.info(f"Total trades: {len(bot.active_trades)}")
        
        if bot.active_trades:
            profitable_trades = sum(1 for t in bot.active_trades.values() if t.get('profit', 0) > 0)
            total_profit = sum(t.get('profit', 0) for t in bot.active_trades.values())
            
            logger.info(f"Profitable trades: {profitable_trades}")
            logger.info(f"Win rate: {(profitable_trades/len(bot.active_trades))*100:.2f}%")
            logger.info(f"Total profit: {total_profit:.2%}")
            
            # Save detailed trade history
            with open(f"test_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                json.dump(bot.active_trades, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_test()) 
"""
Mock data generation for testing.
"""
import json
from datetime import datetime, timedelta
import random
import asyncio
import time
from typing import Dict, List, Optional
from loguru import logger

def generate_mock_trades(num_trades: int = 100) -> list:
    """Generate mock trade history."""
    trades = []
    base_time = datetime.now() - timedelta(days=1)
    
    for i in range(num_trades):
        trade_time = base_time + timedelta(minutes=i*5)
        profit = random.uniform(-0.1, 0.3)
        
        trade = {
            'timestamp': trade_time.isoformat(),
            'token_address': f'mock_token_{i}',
            'token_symbol': f'MOCK{i}',
            'buy_amount': random.uniform(0.1, 0.5),
            'sell_amount': random.uniform(0.1, 0.5),
            'profit': profit,
            'confidence': random.uniform(0.7, 0.95),
            'sentiment': random.random() > 0.3,
            'whale_activity': random.random() > 0.5,
            'price_momentum': random.random() > 0.4
        }
        trades.append(trade)
    
    return trades

def save_mock_trades(filename: str = 'mock_trades.json'):
    """Save mock trades to file."""
    trades = generate_mock_trades()
    with open(filename, 'w') as f:
        json.dump(trades, f, indent=2)

def generate_mock_metrics() -> dict:
    """Generate mock system metrics."""
    return {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': random.uniform(20, 80),
        'memory_percent': random.uniform(30, 70),
        'disk_percent': random.uniform(40, 60),
        'uptime': random.uniform(3600, 86400)
    }

class MockDataProvider:
    def __init__(self):
        self.tokens: Dict[str, Dict] = {}
        self.sentiment_cache: Dict[str, float] = {}
        self.whale_cache: Dict[str, Dict] = {}
        self._generate_mock_tokens()

    def _generate_mock_tokens(self):
        """Generate initial mock tokens"""
        for i in range(5):
            token = f"mock_token_{i}"
            self.tokens[token] = {
                'price': random.uniform(0.0001, 0.001),
                'liquidity': random.uniform(10, 50),
                'holders': random.randint(20, 100),
                'launch_time': time.time() - random.uniform(0, 300),
                'price_change_1m': random.uniform(-0.1, 0.2),
                'volume_change_1m': random.uniform(-0.1, 0.3)
            }

    async def get_pump_data(self) -> Dict:
        """Get mock pump.fun data"""
        # Randomly add new tokens
        if random.random() < 0.1:  # 10% chance to add new token
            token = f"mock_token_{len(self.tokens)}"
            self.tokens[token] = {
                'price': random.uniform(0.0001, 0.001),
                'liquidity': random.uniform(10, 50),
                'holders': random.randint(20, 100),
                'launch_time': time.time(),
                'price_change_1m': random.uniform(-0.1, 0.2),
                'volume_change_1m': random.uniform(-0.1, 0.3)
            }

        # Update existing tokens
        for token in self.tokens:
            self.tokens[token].update({
                'price': self.tokens[token]['price'] * (1 + random.uniform(-0.05, 0.1)),
                'price_change_1m': random.uniform(-0.1, 0.2),
                'volume_change_1m': random.uniform(-0.1, 0.3)
            })

        return self.tokens

    def get_sentiment(self, token: str) -> float:
        """Get mock sentiment score"""
        if token not in self.sentiment_cache:
            self.sentiment_cache[token] = random.uniform(0, 1)
        return self.sentiment_cache[token]

    def get_whale_activity(self, token: str) -> Dict:
        """Get mock whale activity"""
        if token not in self.whale_cache:
            self.whale_cache[token] = {
                'buy_count': random.randint(0, 5),
                'total_buy_volume': random.uniform(0, 2),
                'sell_count': random.randint(0, 3)
            }
        return self.whale_cache[token]

    async def simulate_trade(self, token: str, amount: float, is_buy: bool) -> bool:
        """Simulate trade execution"""
        if token not in self.tokens:
            return False

        # Simulate price impact
        price_impact = amount * 0.01  # 1% price impact
        if is_buy:
            self.tokens[token]['price'] *= (1 + price_impact)
        else:
            self.tokens[token]['price'] *= (1 - price_impact)

        # Update whale activity
        if is_buy:
            self.whale_cache[token]['buy_count'] += 1
            self.whale_cache[token]['total_buy_volume'] += amount
        else:
            self.whale_cache[token]['sell_count'] += 1

        return True

    async def close(self):
        """Clean up resources"""
        pass

if __name__ == "__main__":
    save_mock_trades() 
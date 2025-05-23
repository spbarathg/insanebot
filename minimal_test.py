#!/usr/bin/env python3
"""
Minimal test script for Solana bot components.
"""
import asyncio
import logging
import time
import random
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("minimal_test")

class MinimalHeliusService:
    """Minimal Helius service simulation"""
    
    async def initialize(self):
        logger.info("Initialized HeliusService")
        return True
    
    async def get_token_price(self, token_address):
        """Simulate getting token price"""
        if token_address == "So11111111111111111111111111111111111111112":  # SOL
            price = 100.0 + random.uniform(-5, 5)
        else:
            price = random.uniform(0.00001, 0.1)
        
        logger.info(f"Token {token_address[:8]}... price: ${price:.6f}")
        return {"price": price, "pricePerSol": price / 100.0}
    
    async def get_token_metadata(self, token_address):
        """Simulate getting token metadata"""
        if token_address == "So11111111111111111111111111111111111111112":
            metadata = {
                "name": "Wrapped SOL",
                "symbol": "SOL",
                "holders": 300000,
                "volumeUsd24h": 500000000
            }
        elif token_address == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v":
            metadata = {
                "name": "USD Coin",
                "symbol": "USDC",
                "holders": 500000,
                "volumeUsd24h": 1000000000
            }
        else:
            metadata = {
                "name": f"Token {token_address[:8]}",
                "symbol": "UNKNOWN",
                "holders": 1000,
                "volumeUsd24h": 100000
            }
        
        logger.info(f"Token metadata: {metadata['name']} ({metadata['symbol']})")
        return metadata
    
    async def get_token_liquidity(self, token_address):
        """Simulate getting token liquidity"""
        if token_address == "So11111111111111111111111111111111111111112":
            liquidity = 5000000000
        elif token_address == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v":
            liquidity = 3000000000
        else:
            liquidity = random.uniform(10000, 1000000)
        
        logger.info(f"Token {token_address[:8]}... liquidity: ${liquidity:.2f}")
        return {"liquidity": liquidity, "liquidity_sol": liquidity / 100.0}
    
    async def close(self):
        logger.info("Closed HeliusService")
        return True

class MinimalLocalLLM:
    """Minimal LocalLLM simulation"""
    
    async def initialize(self):
        logger.info("Initialized LocalLLM")
        return True
    
    async def analyze_market(self, token_data):
        """Simulate market analysis"""
        token_symbol = token_data.get("symbol", "UNKNOWN")
        
        # Generate random recommendation
        actions = ["buy", "sell", "hold"]
        action = random.choice(actions)
        confidence = random.uniform(0.5, 0.9)
        
        logger.info(f"Analysis for {token_symbol}: {action.upper()} (confidence: {confidence:.2f})")
        
        return {
            "action": action,
            "confidence": confidence,
            "position_size": random.uniform(0.01, 0.1),
            "reasoning": f"Simple analysis for {token_symbol}"
        }
    
    async def get_market_sentiment(self, token_address):
        """Simulate sentiment analysis"""
        sentiment = random.uniform(-0.7, 0.7)
        logger.info(f"Sentiment for {token_address[:8]}...: {sentiment:.2f}")
        
        return {
            "score": sentiment,
            "magnitude": random.uniform(0.3, 0.9),
            "sources": ["twitter", "telegram", "discord"]
        }
    
    async def get_risk_assessment(self, token_data):
        """Simulate risk assessment"""
        risk = random.uniform(0.2, 0.8)
        categories = ["low", "medium", "high", "extreme"]
        category = categories[min(int(risk * len(categories)), len(categories) - 1)]
        
        logger.info(f"Risk assessment: {category.upper()} ({risk:.2f})")
        
        return {
            "overall_risk": risk,
            "risk_category": category,
            "max_recommended_position": 0.05 * (1 - risk) + 0.01
        }
    
    async def close(self):
        logger.info("Closed LocalLLM")
        return True

class MinimalPortfolio:
    """Minimal portfolio simulation"""
    
    def __init__(self):
        self.balance = 0
        self.holdings = {}
        self.trades = []
    
    async def initialize(self, initial_balance):
        self.balance = initial_balance
        logger.info(f"Portfolio initialized with {initial_balance} SOL")
        return True
    
    def add_trade(self, trade_data):
        """Add a trade to the portfolio"""
        action = trade_data.get("action", "unknown")
        token = trade_data.get("token", "unknown")
        amount = trade_data.get("amount_sol", 0)
        price = trade_data.get("price_usd", 0)
        
        logger.info(f"Added trade: {action.upper()} {amount} SOL of {token} at ${price}")
        self.trades.append(trade_data)
    
    def get_portfolio_summary(self):
        """Get portfolio summary"""
        return {
            "current_value": self.balance,
            "total_return": 0,
            "percent_return": 0,
            "total_trades": len(self.trades),
            "successful_trades": len([t for t in self.trades if t.get("status") == "success"]),
            "win_rate": 50.0
        }

async def test_minimal_bot():
    """Run a minimal test of the bot components"""
    try:
        # Create the components
        helius = MinimalHeliusService()
        llm = MinimalLocalLLM()
        portfolio = MinimalPortfolio()
        
        # Initialize components
        await helius.initialize()
        await llm.initialize()
        await portfolio.initialize(1.0)
        
        # Test a few token operations
        tokens = [
            "So11111111111111111111111111111111111111112",  # SOL
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            f"RANDOM{random.randint(10000, 99999)}111111111111111111111111111"  # Random token
        ]
        
        for token in tokens:
            # Get token data
            price_info = await helius.get_token_price(token)
            metadata = await helius.get_token_metadata(token)
            liquidity_info = await helius.get_token_liquidity(token)
            
            # Create combined token data
            token_data = {
                "address": token,
                "name": metadata.get("name", "Unknown"),
                "symbol": metadata.get("symbol", "UNKNOWN"),
                "price_usd": price_info.get("price", 0),
                "price_sol": price_info.get("pricePerSol", 0),
                "liquidity_usd": liquidity_info.get("liquidity", 0),
                "volumeUsd24h": metadata.get("volumeUsd24h", 0),
                "holders": metadata.get("holders", 0)
            }
            
            # Analyze market
            analysis = await llm.analyze_market(token_data)
            
            # Get sentiment if considering a buy
            if analysis.get("action") == "buy":
                sentiment = await llm.get_market_sentiment(token)
            
            # Get risk assessment
            risk = await llm.get_risk_assessment(token_data)
            
            # Simulate trade
            if random.random() < 0.5:  # 50% chance to trade
                action = analysis.get("action")
                position_size = analysis.get("position_size", 0.01)
                
                trade_data = {
                    "action": action,
                    "token": token_data.get("symbol"),
                    "token_address": token,
                    "amount_sol": position_size,
                    "price_usd": token_data.get("price_usd"),
                    "timestamp": time.time(),
                    "status": "success"
                }
                
                portfolio.add_trade(trade_data)
            
            # Sleep between tokens
            await asyncio.sleep(1)
            
        # Show portfolio summary
        summary = portfolio.get_portfolio_summary()
        logger.info(f"Portfolio summary:")
        logger.info(f"  Value: {summary['current_value']} SOL")
        logger.info(f"  Trades: {summary['total_trades']}")
        logger.info(f"  Win rate: {summary['win_rate']}%")
        
        # Close components
        await helius.close()
        await llm.close()
        
        logger.info("Test completed successfully")
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")

async def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()
    
    # Run the test
    await test_minimal_bot()

if __name__ == "__main__":
    asyncio.run(main()) 